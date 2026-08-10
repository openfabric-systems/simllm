"""Run the frozen PLAY-2 arrival-join study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from simllm._local_config import path_from_env
from simllm.core import CreatedObjectRecord, RequestBookkeeper
from simllm.preplay import (
    ForwardPhase,
    ForwardTokenTrace,
    LayerRouting,
    PreplayTrace,
    PromptFormat,
    RequestArrival,
    RequestTrace,
    SamplingConfig,
    StopReason,
    TraceProvenance,
    join_preplay_arrivals,
    read_preplay_replay_run,
    write_preplay_replay_run,
    write_preplay_trace,
)

EXPECTED_GRANITE_SHA256 = (
    "36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341"
)


def _routing(token_id: int, phase: ForwardPhase, token_index: int) -> ForwardTokenTrace:
    return ForwardTokenTrace(
        phase=phase,
        token_index=token_index,
        token_id=token_id,
        routing=(LayerRouting(layer_index=0, expert_ids=(0,), gate_weights=(1.0,)),),
    )


def _request(
    request_id: str,
    output_token_ids: tuple[int, ...],
    stop_reason: StopReason,
) -> RequestTrace:
    return RequestTrace(
        request_id=request_id,
        prompt_sha256=hashlib.sha256(request_id.encode()).hexdigest(),
        prompt_format=PromptFormat.TEXT,
        input_token_ids=(10,),
        max_new_tokens=len(output_token_ids),
        stop_strings=(),
        output_text=request_id,
        output_token_ids=output_token_ids,
        stop_reason=stop_reason,
        matched_stop_string=None,
        prefill_tokens=(_routing(10, ForwardPhase.PREFILL, 0),),
        decode_tokens=tuple(
            _routing(token_id, ForwardPhase.DECODE, index)
            for index, token_id in enumerate(output_token_ids[:-1])
        ),
    )


def _synthetic_trace() -> PreplayTrace:
    return PreplayTrace(
        provenance=TraceProvenance(
            model_id="test/model",
            model_revision="test-revision",
            model_class="TestMoeForCausalLM",
            dtype="float32",
            tokenizer_sha256="a" * 64,
            sampling=SamplingConfig.greedy(),
            capture_host="study-host",
            runner="study-fixture",
            transformers_version="5.14.1",
            torch_version="2.11.0",
            device="cpu",
            torch_num_threads=1,
            eos_token_id=0,
            top_k=1,
            expert_count=1,
            moe_layer_indices=(0,),
        ),
        requests=(
            _request("alpha", (101, 102, 0), StopReason.EOS),
            _request("beta", (201,), StopReason.LENGTH_CAP),
        ),
    )


def _objects(bookkeeper: RequestBookkeeper) -> list[CreatedObjectRecord]:
    return [
        entry.fact
        for entry in bookkeeper.snapshot().entries
        if isinstance(entry.fact, CreatedObjectRecord)
    ]


def _cell(name: str, trace_path: Path, arrivals: tuple[RequestArrival, ...]) -> dict:
    bookkeeper = RequestBookkeeper()
    run = join_preplay_arrivals(arrivals, trace_path, bookkeeper)
    objects = _objects(bookkeeper)
    projection_ok = len(objects) == len(run.requests) == len(arrivals)
    for arrival, request, record in zip(arrivals, run.requests, objects, strict=True):
        metadata = dict(record.metadata)
        projection_ok = projection_ok and all(
            (
                arrival.request_id == request.request_id == record.native_id,
                arrival.arrived_at_ps == request.arrived_at_ps == record.created_at_ps,
                request.output_length == len(request.output_token_ids),
                metadata["preplay_output_length"] == request.output_length,
                metadata["preplay_stop_reason"] == request.stop_reason.value,
                json.loads(metadata["preplay_output_token_ids"])
                == list(request.output_token_ids),
            )
        )
    authority_ok = all(
        request.routing_reference.trace_sha256 == run.trace.sha256
        and request.routing_reference.request_id == request.request_id
        for request in run.requests
    )
    return {
        "name": name,
        "request_count": len(run.requests),
        "arrival_values": [request.arrived_at_ps for request in run.requests],
        "trace_path": run.trace.path,
        "trace_sha256": run.trace.sha256,
        "request_ids": [request.request_id for request in run.requests],
        "output_lengths": [request.output_length for request in run.requests],
        "output_token_ids": [list(request.output_token_ids) for request in run.requests],
        "projection_ok": projection_ok,
        "authority_ok": authority_ok,
        "object_count": len(objects),
        "run": run,
        "objects": objects,
    }


def _arrival_only_shift(base: dict, shifted: dict) -> bool:
    if len(base["run"].requests) != len(shifted["run"].requests):
        return False
    for before, after in zip(base["run"].requests, shifted["run"].requests, strict=True):
        if after.arrived_at_ps - before.arrived_at_ps != 7_000:
            return False
        if (
            before.request_id,
            before.output_length,
            before.stop_reason,
            before.output_token_ids,
            before.routing_reference,
            before.bookkeeping_object_id,
        ) != (
            after.request_id,
            after.output_length,
            after.stop_reason,
            after.output_token_ids,
            after.routing_reference,
            after.bookkeeping_object_id,
        ):
            return False
    for before, after in zip(base["objects"], shifted["objects"], strict=True):
        before_metadata = dict(before.metadata)
        after_metadata = dict(after.metadata)
        if after.created_at_ps - before.created_at_ps != 7_000:
            return False
        if (
            after_metadata.pop("preplay_arrived_at_ps")
            - before_metadata.pop("preplay_arrived_at_ps")
            != 7_000
        ):
            return False
        if before_metadata != after_metadata:
            return False
        if (before.ref, before.scope, before.native_id) != (
            after.ref,
            after.scope,
            after.native_id,
        ):
            return False
    return True


def _fatal_rejections(trace_path: Path) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    invalid = {
        "empty": (),
        "duplicate": (
            RequestArrival(request_id="alpha", arrived_at_ps=1),
            RequestArrival(request_id="alpha", arrived_at_ps=2),
        ),
        "negative": (RequestArrival(request_id="alpha", arrived_at_ps=-1),),
        "boolean": (RequestArrival(request_id="alpha", arrived_at_ps=True),),
        "missing": (RequestArrival(request_id="missing", arrived_at_ps=1),),
    }
    for name, arrivals in invalid.items():
        bookkeeper = RequestBookkeeper()
        before = bookkeeper.snapshot()
        try:
            join_preplay_arrivals(arrivals, trace_path, bookkeeper)
        except (TypeError, ValueError):
            checks[name] = bookkeeper.snapshot() == before
        else:
            checks[name] = False
    return checks


def run_study(run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=False)
    granite = Path(__file__).parents[1] / "preplay_trace_v1/granite_length_cap.jsonl"
    granite_sha256 = hashlib.sha256(granite.read_bytes()).hexdigest()
    if granite_sha256 != EXPECTED_GRANITE_SHA256:
        raise AssertionError(
            f"tracked Granite fixture hash {granite_sha256} != {EXPECTED_GRANITE_SHA256}"
        )
    synthetic_trace = _synthetic_trace()
    synthetic = write_preplay_trace(
        run_dir / "synthetic.jsonl",
        synthetic_trace.provenance,
        synthetic_trace.requests,
    )

    cells = [
        _cell(
            "granite-base",
            granite,
            (RequestArrival(request_id="length-cap", arrived_at_ps=1_000),),
        ),
        _cell(
            "granite-shifted",
            granite,
            (RequestArrival(request_id="length-cap", arrived_at_ps=8_000),),
        ),
        _cell(
            "synthetic-base",
            synthetic,
            (
                RequestArrival(request_id="alpha", arrived_at_ps=1_000),
                RequestArrival(request_id="beta", arrived_at_ps=4_000),
            ),
        ),
        _cell(
            "synthetic-shifted",
            synthetic,
            (
                RequestArrival(request_id="alpha", arrived_at_ps=8_000),
                RequestArrival(request_id="beta", arrived_at_ps=11_000),
            ),
        ),
    ]
    by_name = {cell["name"]: cell for cell in cells}
    shift_ok = _arrival_only_shift(
        by_name["granite-base"], by_name["granite-shifted"]
    ) and _arrival_only_shift(
        by_name["synthetic-base"], by_name["synthetic-shifted"]
    )
    b1 = all(cell["projection_ok"] for cell in cells) and shift_ok
    b2 = all(cell["authority_ok"] for cell in cells) and all(
        cell["trace_sha256"] == EXPECTED_GRANITE_SHA256
        for cell in cells
        if cell["name"].startswith("granite")
    )
    b3 = all(
        (
            by_name["granite-base"]["request_count"] == 1,
            by_name["granite-base"]["object_count"] == 1,
            by_name["synthetic-base"]["request_count"] == 2,
            by_name["synthetic-base"]["object_count"] == 2,
        )
    )

    canonical_path = write_preplay_replay_run(
        by_name["synthetic-base"]["run"], run_dir / "joined.json"
    )
    round_trip_path = write_preplay_replay_run(
        read_preplay_replay_run(canonical_path), run_dir / "joined-round-trip.json"
    )
    e1 = canonical_path.read_bytes() == round_trip_path.read_bytes()
    fatal = _fatal_rejections(synthetic)

    public_cells = []
    for cell in cells:
        public_cells.append(
            {
                key: value
                for key, value in cell.items()
                if key not in {"run", "objects"}
            }
        )
    summary = {
        "scored": {"B1": b1, "B2": b2, "B3": b3},
        "exact_oracle": {"E1": e1},
        "fatal_unscored": fatal,
        "cells": public_cells,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (run_dir / "cells.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "name",
                "request_count",
                "object_count",
                "arrival_values",
                "output_lengths",
                "trace_sha256",
                "projection_ok",
                "authority_ok",
            ),
        )
        writer.writeheader()
        for cell in public_cells:
            writer.writerow({name: cell[name] for name in writer.fieldnames})
    if not all(summary["scored"].values()):
        raise AssertionError(f"scored relation failed: {summary['scored']}")
    if not all(summary["exact_oracle"].values()):
        raise AssertionError(f"exact oracle failed: {summary['exact_oracle']}")
    if not all(summary["fatal_unscored"].values()):
        raise AssertionError(f"fatal guard failed: {summary['fatal_unscored']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    fixture = Path(__file__).parents[1] / "preplay_trace_v1/granite_length_cap.jsonl"
    if not fixture.is_file():
        raise SystemExit(f"missing tracked fixture: {fixture}")
    if args.check_only:
        return
    if args.run_dir is None:
        data_root = path_from_env("SIMLLM_DATA_ROOT")
        if data_root is None:
            parser.error("--run-dir is required when SIMLLM_DATA_ROOT is not set")
        args.run_dir = data_root / "preplay_arrival_join_v1"
    summary = run_study(args.run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
