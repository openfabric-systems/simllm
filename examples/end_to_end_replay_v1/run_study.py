"""Run the frozen end-to-end per-request replay study.

The chain is: a pinned CPU oracle capture, a real vLLM scheduler driving the
simulated executor with arrival gating and token pinning, captured MoE routing
driving the fabric, ``htsim_rnic`` timing every dispatch and combine, and the
per-request TTFT and TPOT coming back out with a conserved seven-component
partition.

Every acceptance clause, bound and relation this script evaluates is frozen in
``expectations.md`` and was committed before any of the code below existed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"
MODEL_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
MODEL_RELATIVE_PATH = (
    Path("hub")
    / "models--ibm-granite--granite-3.0-1b-a400m-instruct"
    / "snapshots"
    / MODEL_REVISION
)

#: geometry of the captured model, frozen in expectations.md
NUM_LAYERS = 24
HIDDEN_SIZE = 1024
DTYPE_BYTES = 2
NUM_EXPERTS = 32
TOP_K = 8
MOE_INTERMEDIATE_SIZE = 512
VECTOR_BYTES = HIDDEN_SIZE * DTYPE_BYTES

#: the one rank that dispatches scheduled tokens
ENGINE_RANK = 0
#: declared arrival spacing, one millisecond between successive requests
ARRIVAL_SPACING_PS = 1_000_000_000
#: frozen backend profile and compute derate
PROFILE = "rnic-nn-fluid"
ROOFLINE_EFFICIENCY = 0.7

BANDWIDTHS_BPS = (100_000_000_000, 200_000_000_000, 400_000_000_000)

#: cell name, case, expert-parallel world, link rate
CELLS = (
    ("a-ep8-400g", "a", 8, 400_000_000_000),
    ("a-ep8-200g", "a", 8, 200_000_000_000),
    ("a-ep8-100g", "a", 8, 100_000_000_000),
    ("a-ep4-400g", "a", 4, 400_000_000_000),
    ("b-ep8-400g", "b", 8, 400_000_000_000),
)

#: physical sanity bounds from expectations.md, all in picoseconds or bytes
CASE_A_PROMPT_TOKENS = 54
S1_MOE_BYTES_FLOOR = CASE_A_PROMPT_TOKENS * NUM_LAYERS * 2 * 1 * VECTOR_BYTES
S1_MOE_BYTES_CEILING = CASE_A_PROMPT_TOKENS * NUM_LAYERS * 2 * 7 * VECTOR_BYTES
S3_COMPUTE_FLOOR_PS = 60_000_000
S3_COMPUTE_CEILING_PS = 200_000_000
S4_PREFILL_FLOOR_PS = 270_000_000
S4_PREFILL_CEILING_PS = 3_000_000_000
S5_DECODE_FLOOR_PS = 150_000_000
S5_DECODE_CEILING_PS = 600_000_000
S6_COMPUTE_SPREAD = 0.10
S7_DECODE_RATE_FLOOR = 1_000.0
S7_DECODE_RATE_CEILING = 20_000.0
C5_4_COMPUTE_RATIO_FLOOR = 1.35
C5_4_COMPUTE_RATIO_CEILING = 1.75
C5_1_ABSOLUTE_TOLERANCE_PS = 1_000
C5_1_RELATIVE_TOLERANCE = 1e-6

#: scored relation counts, split by evidence class and never summed
EXPECTED_EXACT_ORACLE_RELATIONS = 13
EXPECTED_BEHAVIORAL_RELATIONS = 4

PS_PER_SECOND = 1_000_000_000_000

#: case A is the first three; case B is all twelve
REQUEST_SPECS = (
    ("r00", "How does mixture of experts routing change network traffic?", 24),
    ("r01", "Reply with exactly one word: OK", 8),
    ("r02", "List three properties of a datacenter network fabric.", 32),
    ("r03", "What is the difference between prefill and decode?", 20),
    ("r04", "Explain expert parallelism in two sentences.", 24),
    ("r05", "Name one cause of incast in a Clos network.", 16),
    ("r06", "Why does a decode step read the whole weight set?", 20),
    ("r07", "Give a one line definition of time to first token.", 16),
    ("r08", "What does an all-to-all collective move?", 20),
    ("r09", "Summarize continuous batching in one sentence.", 20),
    ("r10", "State one reason a KV cache grows with context.", 16),
    ("r11", "What limits throughput on a single accelerator?", 24),
)
CASE_REQUEST_IDS = {
    "a": tuple(spec[0] for spec in REQUEST_SPECS[:3]),
    "b": tuple(spec[0] for spec in REQUEST_SPECS),
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_only(args: argparse.Namespace) -> None:
    """Validate every frozen input without producing an artifact."""

    model = args.cache_dir / MODEL_RELATIVE_PATH
    if not model.is_dir() or not (model / "config.json").is_file():
        raise SystemExit(f"pinned model snapshot is missing: {model}")
    if importlib.metadata.version("vllm") != "0.26.0":
        raise SystemExit("this study requires vLLM 0.26.0")
    if not args.htsim_rnic.is_file() or not args.htsim_rnic.stat().st_mode & 0o111:
        raise SystemExit(f"htsim_rnic is missing or not executable: {args.htsim_rnic}")
    txt2bin = os.environ.get("SIMLLM_TXT2BIN")
    if txt2bin is not None:
        converter = Path(txt2bin)
        if not converter.is_file() or not converter.stat().st_mode & 0o111:
            raise SystemExit(f"SIMLLM_TXT2BIN is not an executable: {converter}")
    if VECTOR_BYTES != 2048:
        raise AssertionError("routed hidden-vector size changed")
    if NUM_EXPERTS % 8 or NUM_EXPERTS % 4:
        raise AssertionError("expert count no longer divides both frozen EP worlds")
    if S1_MOE_BYTES_FLOOR != 5_308_416 or S1_MOE_BYTES_CEILING != 37_158_912:
        raise AssertionError("case A step-0 routed-byte bounds changed")
    if 8_000_000_000_000 // BANDWIDTHS_BPS[2] != 20:
        raise AssertionError("400 Gbit/s byte serialization literal changed")
    if 8_000_000_000_000 // BANDWIDTHS_BPS[1] != 40:
        raise AssertionError("200 Gbit/s byte serialization literal changed")
    if 8_000_000_000_000 // BANDWIDTHS_BPS[0] != 80:
        raise AssertionError("100 Gbit/s byte serialization literal changed")
    if BANDWIDTHS_BPS[1] != 2 * BANDWIDTHS_BPS[0]:
        raise AssertionError("bandwidth ladder must retain its 2x spacing")
    if BANDWIDTHS_BPS[2] != 2 * BANDWIDTHS_BPS[1]:
        raise AssertionError("bandwidth ladder must retain its 2x spacing")
    if len({name for name, *_ in CELLS}) != len(CELLS):
        raise AssertionError("cell names must be distinct")
    if {bandwidth for name, case, ep, bandwidth in CELLS if case == "a" and ep == 8} != set(
        BANDWIDTHS_BPS
    ):
        raise AssertionError("the three-point linearity test lost a bandwidth cell")
    if S4_PREFILL_FLOOR_PS >= S4_PREFILL_CEILING_PS:
        raise AssertionError("prefill makespan bounds are inverted")
    if S5_DECODE_FLOOR_PS >= S5_DECODE_CEILING_PS:
        raise AssertionError("decode makespan bounds are inverted")
    if S3_COMPUTE_FLOOR_PS >= S3_COMPUTE_CEILING_PS:
        raise AssertionError("compute service bounds are inverted")
    if not 0.0 < ROOFLINE_EFFICIENCY <= 1.0:
        raise AssertionError("roofline derate left its valid range")
    if EXPECTED_EXACT_ORACLE_RELATIONS != 3 + 4 + 5 + 1:
        raise AssertionError("exact-oracle relation denominator changed")
    if EXPECTED_BEHAVIORAL_RELATIONS != 4:
        raise AssertionError("behavioral relation denominator changed")
    if CASE_REQUEST_IDS["a"] != CASE_REQUEST_IDS["b"][:3]:
        raise AssertionError("case A must be the leading prefix of case B")
    if len(CASE_REQUEST_IDS["b"]) != 4 * len(CASE_REQUEST_IDS["a"]):
        raise AssertionError("the registered expansion changed its factor")
    print(
        f"check-only run-dir={args.run_dir}; validated the frozen end-to-end "
        "inputs and produced no artifacts"
    )


@contextmanager
def offline_environment() -> Iterator[None]:
    values = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_V1": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "SIMLLM_VLLM_WORKER_MODE": "skeleton",
        "SIMLLM_VLLM_MODE": "virtual",
    }
    previous = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# ---------------------------------------------------------------- capture ---


def _capture(args: argparse.Namespace) -> None:
    from simllm.preplay import (
        PreplayRequest,
        SamplingConfig,
        read_preplay_trace,
        run_transformers_preplay,
    )

    capture_dir = args.run_dir / "capture"
    capture_dir.mkdir(parents=True, exist_ok=False)
    trace_path = capture_dir / "greedy.jsonl"
    with offline_environment():
        run_transformers_preplay(
            (
                PreplayRequest(
                    request_id=request_id,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                )
                for request_id, prompt, max_new_tokens in REQUEST_SPECS
            ),
            trace_path,
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=args.cache_dir,
            sampling=SamplingConfig.greedy(),
        )
    trace = read_preplay_trace(trace_path)
    summary = {
        "sha256": file_sha256(trace_path),
        "requests": [
            {
                "request_id": request.request_id,
                "prompt_tokens": len(request.input_token_ids),
                "output_tokens": len(request.output_token_ids),
                "stop_reason": request.stop_reason.value,
            }
            for request in trace.requests
        ],
    }
    (capture_dir / "capture_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _raw_capture(trace_path: Path) -> dict[str, Any]:
    """Re-read the capture with the standard library only.

    Nothing from ``simllm`` participates. This is the independent side of every
    routing-fidelity relation.
    """

    requests: dict[str, dict[str, Any]] = {}
    forwards: dict[tuple[str, str], list[dict[str, Any]]] = {}
    header: dict[str, Any] = {}
    footer: dict[str, Any] = {}
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            kind = row["row_type"]
            if kind == "header":
                header = row["provenance"]
            elif kind == "footer":
                footer = row
            elif kind == "request":
                requests[row["request_id"]] = {
                    "input_token_ids": list(row["input_token_ids"]),
                    "output_token_ids": list(row["output_token_ids"]),
                    "stop_reason": row["stop_reason"],
                    "stop_strings": list(row["stop_strings"]),
                    "max_new_tokens": row["max_new_tokens"],
                }
            elif kind == "forward-token":
                key = (row["request_id"], row["phase"])
                routing = {
                    entry["layer_index"]: tuple(entry["expert_ids"])
                    for entry in row["routing"]
                }
                forwards.setdefault(key, []).append(
                    {
                        "token_index": row["token_index"],
                        "token_id": row["token_id"],
                        "routing": routing,
                    }
                )
            else:
                raise AssertionError(f"unexpected capture row type {kind!r}")
    for key, rows in forwards.items():
        if [row["token_index"] for row in rows] != list(range(len(rows))):
            raise AssertionError(f"capture forward rows for {key} are not contiguous")
    return {"header": header, "footer": footer, "requests": requests, "forwards": forwards}


# ------------------------------------------------------------------- cells ---


def _cell_spec(name: str) -> tuple[str, int, int]:
    for cell_name, case, ep_world, bandwidth in CELLS:
        if cell_name == name:
            return case, ep_world, bandwidth
    raise SystemExit(f"unknown cell {name!r}")


def _dims(ep_world: int) -> Any:
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=512,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=49_152,
        dtype_bytes=DTYPE_BYTES,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        moe_intermediate_size=MOE_INTERMEDIATE_SIZE,
        local_num_experts=NUM_EXPERTS // ep_world,
    )


def _goal_sends(path: Path) -> dict[int, dict[tuple[int, int], int]]:
    """Parse one executed GOAL artifact into ``tag -> {(source, dest): bytes}``."""

    result: dict[int, dict[tuple[int, int], int]] = {}
    rank = -1
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("rank "):
            rank = int(line.split()[1])
        if ": send " not in line:
            continue
        words = line.split()
        pair = (rank, int(words[4]))
        tag = int(words[6])
        bucket = result.setdefault(tag, {})
        if pair in bucket:
            raise AssertionError(f"{path.name}: duplicate GOAL send for tag {tag}, pair {pair}")
        bucket[pair] = int(words[2].removesuffix("b"))
    return result


def _independent_request_pairs(
    record: Any,
    raw: dict[str, Any],
    ep_world: int,
) -> dict[str, dict[tuple[str, int, int], int]]:
    """Recompute the per-request directed byte table straight from the capture.

    Returns ``operation_id -> {(request_id, source, destination): bytes}`` with
    the operation identity spelled exactly as ``simllm.traffic`` spells it, so a
    disagreement about which layer or phase a byte belongs to is visible rather
    than absorbed.
    """

    tables: dict[str, dict[tuple[str, int, int], int]] = {}
    for scheduled in record.scheduled:
        if scheduled.num_new_tokens == 0:
            continue
        request_id = scheduled.request_id
        entry = raw["requests"][request_id]
        prompt_token_count = len(entry["input_token_ids"])
        if scheduled.phase.value == "prefill":
            end = scheduled.context_length
            start = end - scheduled.num_new_tokens
            rows = raw["forwards"][(request_id, "prefill")]
        else:
            start = scheduled.context_length - scheduled.num_new_tokens - prompt_token_count
            end = start + scheduled.num_new_tokens
            rows = raw["forwards"][(request_id, "decode")]
        if start < 0 or end > len(rows) or start >= end:
            raise AssertionError(
                f"independent slice [{start}, {end}) for {request_id} is outside the capture"
            )
        for row in rows[start:end]:
            for layer in range(NUM_LAYERS):
                experts = row["routing"][layer]
                if len(experts) != TOP_K:
                    raise AssertionError("capture row does not carry the frozen top-k")
                destinations = {expert % ep_world for expert in experts}
                destinations.discard(ENGINE_RANK)
                for destination in destinations:
                    dispatch_id = f"step-{record.step_index}:layer-{layer}:ep-dispatch"
                    combine_id = f"step-{record.step_index}:layer-{layer}:ep-combine"
                    dispatch = tables.setdefault(dispatch_id, {})
                    combine = tables.setdefault(combine_id, {})
                    dispatch_key = (request_id, ENGINE_RANK, destination)
                    combine_key = (request_id, destination, ENGINE_RANK)
                    dispatch[dispatch_key] = dispatch.get(dispatch_key, 0) + VECTOR_BYTES
                    combine[combine_key] = combine.get(combine_key, 0) + VECTOR_BYTES
    return tables


def _library_request_pairs(
    record: Any,
    dims: Any,
    ep_ranks: tuple[int, ...],
    supply: Any,
) -> dict[str, dict[tuple[str, int, int], int]]:
    from simllm.traffic import step_moe_alltoalls

    operations = step_moe_alltoalls(record, dims, ep_ranks, routed_supply=supply)
    tables: dict[str, dict[tuple[str, int, int], int]] = {}
    for operation in operations:
        operation_id = f"step-{record.step_index}:layer-{operation.layer}:ep-{operation.phase}"
        tables[operation_id] = {
            (request_id, source, destination): size
            for request_id, source, destination, size in operation.request_pair_payload_bytes
        }
    return tables


def _attribution_json(attribution: Any) -> dict[str, int]:
    return {
        "queue_ps": attribution.queue_ps,
        "kv_ps": attribution.kv_ps,
        "kernel_ps": attribution.kernel_ps,
        "dma_ps": attribution.dma_ps,
        "collective_ps": attribution.collective_ps,
        "nic_ps": attribution.nic_ps,
        "control_ps": attribution.control_ps,
        "total_ps": attribution.total_ps,
    }


def _observe_outputs(
    outputs: Sequence[Any],
    completed_at_ps: int,
    served: dict[str, list[int]],
    token_times: dict[str, list[int]],
    finishes: dict[str, dict[str, Any]],
) -> None:
    for output in outputs:
        request_id = output.request_id
        if request_id not in served or len(output.outputs) != 1:
            raise AssertionError("engine returned an unknown or multi-choice request")
        choice = output.outputs[0]
        cumulative = list(choice.token_ids)
        previous = served[request_id]
        if cumulative[: len(previous)] != previous:
            raise AssertionError(f"request {request_id} changed an emitted prefix")
        new_tokens = cumulative[len(previous) :]
        served[request_id].extend(new_tokens)
        token_times[request_id].extend([completed_at_ps] * len(new_tokens))
        if output.finished:
            if request_id in finishes:
                raise AssertionError(f"request {request_id} finished twice")
            finishes[request_id] = {
                "finish_reason": choice.finish_reason,
                "stop_reason": choice.stop_reason,
            }


def _cell(args: argparse.Namespace, name: str) -> None:
    from vllm import LLM, SamplingParams

    from simllm.adapters.vllm import (
        SimExecutorConfig,
        configure,
        latest_worker,
        reset_configuration,
    )
    from simllm.backends import (
        HtsimRequestMetricReducer,
        HtsimStepSink,
        HtsimStepSinkConfig,
        attribute_step,
    )
    from simllm.compute import RooflineProvider
    from simllm.core import RequestBookkeeper, framework_request_arrivals
    from simllm.preplay import (
        RequestArrival,
        join_preplay_arrivals,
        project_preplay_routing,
        read_preplay_trace,
        write_preplay_replay_run,
        write_routed_experts,
    )
    from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply
    from simllm.workload import AdmissionMode, RequestAdmissionGate

    case, ep_world, bandwidth = _cell_spec(name)
    request_ids = CASE_REQUEST_IDS[case]
    ep_ranks = tuple(range(ep_world))
    cell_dir = args.run_dir / "cells" / name
    cell_dir.mkdir(parents=True, exist_ok=False)
    trace_path = args.run_dir / "capture" / "greedy.jsonl"
    trace = read_preplay_trace(trace_path)
    raw = _raw_capture(trace_path)
    requests = {
        request.request_id: request
        for request in trace.requests
        if request.request_id in request_ids
    }
    if set(requests) != set(request_ids):
        raise AssertionError("capture does not carry every request of this case")

    bookkeeper = RequestBookkeeper()
    replay_run = join_preplay_arrivals(
        (
            RequestArrival(
                request_id=request_id,
                arrived_at_ps=index * ARRIVAL_SPACING_PS,
            )
            for index, request_id in enumerate(request_ids)
        ),
        trace_path,
        bookkeeper,
    )
    replay_path = write_preplay_replay_run(replay_run, cell_dir / "replay-run.json")
    routed = project_preplay_routing(replay_run)
    write_routed_experts(routed, cell_dir / "routed-experts.json")
    arrivals = {
        arrival.request_id: arrival.arrived_at_ps
        for arrival in framework_request_arrivals(bookkeeper.snapshot())
    }

    placement = ExpertPlacementSnapshot(
        placement_epoch=0,
        expert_owners=tuple(
            (layer, expert, expert % ep_world)
            for layer in range(NUM_LAYERS)
            for expert in range(NUM_EXPERTS)
        ),
    )
    supply = RoutedMoeSupply(
        engine_rank=ENGINE_RANK,
        routed_experts=routed,
        placements=(placement,),
        step_placement_epochs=tuple((step, 0) for step in range(4096)),
    )
    dims = _dims(ep_world)
    workdir = cell_dir / "htsim"
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile=PROFILE,
            tp_ranks=(0,),
            dims=dims,
            workdir=workdir,
            ep_ranks=ep_ranks,
            linkspeed_bps=bandwidth,
            provider=RooflineProvider(ROOFLINE_EFFICIENCY),
            routed_moe_supply=supply,
        )
    )

    served: dict[str, list[int]] = {request_id: [] for request_id in request_ids}
    token_times: dict[str, list[int]] = {request_id: [] for request_id in request_ids}
    finishes: dict[str, dict[str, Any]] = {}
    identity_ok = True
    llm: Any | None = None
    started = time.time()
    reset_configuration()
    configure(
        step_sink=sink,
        config=SimExecutorConfig(
            mode="virtual",
            token_id=512,
            step_records_path=str(cell_dir / "steps.jsonl"),
            replay_run_path=str(replay_path),
        ),
    )
    try:
        with offline_environment():
            llm = LLM(
                model=str(args.cache_dir / MODEL_RELATIVE_PATH),
                worker_cls="simllm.adapters.vllm.SimWorker",
                enforce_eager=True,
                max_model_len=128,
                num_gpu_blocks_override=512,
                disable_log_stats=True,
                enable_chunked_prefill=False,
                enable_prefix_caching=False,
                async_scheduling=False,
            )
            worker = latest_worker()
            if worker is None or worker.replay is None:
                raise AssertionError("replay SimWorker was not constructed")

            def submit(arrival: Any) -> None:
                nonlocal identity_ok
                request = requests[arrival.request_id]
                assigned = llm.llm_engine.add_request(
                    arrival.request_id,
                    {"prompt_token_ids": list(request.input_token_ids)},
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=len(request.output_token_ids),
                        min_tokens=0,
                        stop=list(request.stop_strings),
                        detokenize=True,
                    ),
                    arrival_time=arrival.arrived_at_ps / PS_PER_SECOND,
                )
                identity_ok = identity_ok and assigned == arrival.request_id

            gate = RequestAdmissionGate(
                worker.clock,
                bookkeeper,
                mode=AdmissionMode.ARRIVAL_GATED,
            )
            while gate.has_pending or llm.llm_engine.has_unfinished_requests():
                gate.admit_ready(submit)
                if llm.llm_engine.has_unfinished_requests():
                    before = len(worker.step_records)
                    outputs = llm.llm_engine.step()
                    if len(worker.step_records) != before + 1:
                        raise AssertionError("one engine step did not emit one record")
                    _observe_outputs(
                        outputs,
                        worker.clock.now_ps,
                        served,
                        token_times,
                        finishes,
                    )
                elif gate.has_pending:
                    gate.advance_to_next_arrival()
            records = tuple(worker.step_records)
            results = tuple(worker.step_results)
            replay_snapshot = worker.replay.snapshot()
            admitted = gate.admitted_request_ids
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        reset_configuration()
    wall_seconds = time.time() - started

    if len(records) != len(results):
        raise AssertionError("step record and result cardinality differ")
    locality_by_step = {
        outcome.step_index: outcome for outcome in sink.locality_outcomes
    }
    network_by_step = {outcome.step_index: outcome for outcome in sink.outcomes}
    if len(locality_by_step) != len(sink.locality_outcomes):
        raise AssertionError("locality outcomes repeat a step index")

    reducer = HtsimRequestMetricReducer(arrivals)
    step_rows: list[dict[str, Any]] = []
    routing = {
        "token_cells_compared": 0,
        "token_cells_equal": 0,
        "request_pair_rows": 0,
        "request_pair_mismatches": [],
        "goal_rows": 0,
        "goal_mismatches": [],
        "ownership_violations": [],
        "layer_bytes": {str(layer): 0 for layer in range(NUM_LAYERS)},
        "request_bytes": {request_id: 0 for request_id in request_ids},
        "rank_egress_bytes": {str(rank): 0 for rank in ep_ranks},
    }
    conservation = {
        "intervals": 0,
        "interval_conservation_failures": 0,
        "inactive_component_violations": 0,
        "artifact_partition_failures": 0,
        "kernel_disagrees_with_compute_service": 0,
        "makespan_conservation_failures": 0,
    }
    htsim_invocations = 0
    total_routed_bytes = 0

    for record, result in zip(records, results, strict=True):
        locality = locality_by_step.get(record.step_index)
        network = network_by_step.get(record.step_index)
        if locality is not None and (
            locality.compute_service_ps
            + sum(
                composed
                for composed, fabric in zip(
                    locality.composed_phase_service_ps,
                    locality.fabric_phase_service_ps,
                    strict=True,
                )
                if fabric
            )
            != result.step_latency_ps
        ):
            conservation["makespan_conservation_failures"] += 1
        step_attribution = attribute_step(result, locality)
        metrics = reducer.consume(record, result, locality)
        conservation["intervals"] += len(metrics)
        for metric in metrics:
            if metric.attribution.total_ps != metric.latency_ps:
                conservation["interval_conservation_failures"] += 1
            if (
                metric.attribution.kv_ps
                or metric.attribution.dma_ps
                or metric.attribution.nic_ps
                or metric.attribution.control_ps
            ):
                conservation["inactive_component_violations"] += 1

        row: dict[str, Any] = {
            "step_index": record.step_index,
            "virtual_time_ps": record.virtual_time_ps,
            "step_latency_ps": result.step_latency_ps,
            "completed_at_ps": result.completed_at_ps,
            "total_new_tokens": record.total_new_tokens,
            "num_sampled": record.num_sampled,
            "scheduled": [
                {
                    "request_id": scheduled.request_id,
                    "phase": scheduled.phase.value,
                    "num_new_tokens": scheduled.num_new_tokens,
                    "context_length": scheduled.context_length,
                }
                for scheduled in record.scheduled
            ],
            "attribution": _attribution_json(step_attribution),
            "simulated": locality is not None,
        }
        if locality is not None and network is not None:
            if step_attribution.kernel_ps != locality.compute_service_ps:
                conservation["kernel_disagrees_with_compute_service"] += 1
            if len(locality.composed_phase_service_ps) != locality.artifact_count:
                conservation["artifact_partition_failures"] += 1
            row.update(
                {
                    "compute_service_ps": locality.compute_service_ps,
                    "artifact_count": locality.artifact_count,
                    "total_directed_bytes": locality.total_directed_bytes,
                    "nvlink_directed_bytes": locality.nvlink_directed_bytes,
                    "fabric_phase_service_ps": list(locality.fabric_phase_service_ps),
                    "composed_phase_service_ps": list(locality.composed_phase_service_ps),
                    "artifact_operation_ids": [
                        list(ids) for ids in locality.artifact_operation_ids
                    ],
                    "routing_mode": network.routing_mode,
                    "placement_epoch": network.placement_epoch,
                    "quiescent": network.quiescent,
                    "num_flows": network.num_flows,
                    "backend_runs": locality.backend_runs,
                }
            )
            htsim_invocations += locality.backend_runs
            total_routed_bytes += locality.total_directed_bytes
            _check_step_routing(
                record,
                locality,
                raw,
                dims,
                ep_ranks,
                supply,
                workdir,
                routing,
            )
        step_rows.append(row)

    if raw is not None:
        for (request_id, phase), rows in raw["forwards"].items():
            if request_id not in requests:
                continue
            projected = routed.by_request_id(request_id)
            tokens = projected.prefill_tokens if phase == "prefill" else projected.decode_tokens
            if len(tokens) != len(rows):
                raise AssertionError(
                    f"routed projection lost {phase} tokens for {request_id}"
                )
            for token, row in zip(tokens, rows, strict=True):
                for layer in range(NUM_LAYERS):
                    routing["token_cells_compared"] += 1
                    if tuple(token.layers[layer].expert_ids) == row["routing"][layer]:
                        routing["token_cells_equal"] += 1

    totals = {row.request_id: row for row in reducer.totals()}
    request_rows = []
    for index, request_id in enumerate(request_ids):
        oracle = requests[request_id]
        total = totals.get(request_id)
        finish = finishes.get(request_id, {})
        tpot = None if total is None or total.tpot_ps is None else total.tpot_ps
        request_rows.append(
            {
                "request_id": request_id,
                "arrival_ps": index * ARRIVAL_SPACING_PS,
                "prompt_tokens": len(oracle.input_token_ids),
                "oracle_output_tokens": len(oracle.output_token_ids),
                "oracle_stop_reason": oracle.stop_reason.value,
                "served_matches_oracle": served[request_id] == list(oracle.output_token_ids),
                "served_tokens": len(served[request_id]),
                "finish_reason": finish.get("finish_reason"),
                "stop_reason": finish.get("stop_reason"),
                "token_count": None if total is None else total.token_count,
                "first_token_at_ps": None if total is None else total.first_token_at_ps,
                "last_token_at_ps": None if total is None else total.last_token_at_ps,
                "ttft_ps": None if total is None else total.ttft_ps,
                "tpot_numerator": None if tpot is None else tpot.numerator,
                "tpot_denominator": None if tpot is None else tpot.denominator,
                "ttft_attribution": (
                    None if total is None else _attribution_json(total.ttft_attribution)
                ),
                "decode_attribution": (
                    None if total is None else _attribution_json(total.decode_attribution)
                ),
            }
        )

    payload = {
        "name": name,
        "case": case,
        "ep_world": ep_world,
        "bandwidth_bps": bandwidth,
        "wall_seconds": round(wall_seconds, 2),
        "scheduler_steps": len(records),
        "simulated_steps": len(sink.locality_outcomes),
        "htsim_invocations": htsim_invocations,
        "total_routed_bytes": total_routed_bytes,
        "identity": {
            "add_request_returned_oracle_id": identity_ok,
            "admitted": list(admitted),
            "scheduler_ids_subset": all(
                scheduled.request_id in requests
                for record in records
                for scheduled in record.scheduled
            ),
            "replay_completed": sorted(replay_snapshot.completed_request_ids),
            "replay_drained": sorted(replay_snapshot.drained_request_ids),
        },
        "requests": request_rows,
        "steps": step_rows,
        "routing": routing,
        "conservation": conservation,
    }
    (cell_dir / "cell.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _check_step_routing(
    record: Any,
    locality: Any,
    raw: dict[str, Any],
    dims: Any,
    ep_ranks: tuple[int, ...],
    supply: Any,
    workdir: Path,
    routing: dict[str, Any],
) -> None:
    """Score C2.2, C2.3 and C2.4 for one simulated step."""

    independent = _independent_request_pairs(record, raw, len(ep_ranks))
    library = _library_request_pairs(record, dims, ep_ranks, supply)
    for operation_id in sorted(set(independent) | set(library)):
        expected = independent.get(operation_id, {})
        actual = library.get(operation_id, {})
        routing["request_pair_rows"] += len(expected)
        if expected != actual:
            routing["request_pair_mismatches"].append(
                {
                    "operation_id": operation_id,
                    "expected_rows": len(expected),
                    "actual_rows": len(actual),
                }
            )
        layer = int(operation_id.split(":")[1].split("-")[1])
        for (request_id, source, _destination), size in expected.items():
            routing["layer_bytes"][str(layer)] += size
            routing["request_bytes"][request_id] += size
            routing["rank_egress_bytes"][str(source)] += size

    for index, operation_ids in enumerate(locality.artifact_operation_ids):
        moe_ids = [
            operation_id for operation_id in operation_ids if ":ep-" in operation_id
        ]
        if not moe_ids:
            continue
        if len(moe_ids) != 1:
            raise AssertionError("one executed artifact carries several MoE operations")
        operation_id = moe_ids[0]
        goal_path = workdir / f"step-{record.step_index:06d}.artifact-{index:04d}.goal"
        if not goal_path.is_file():
            routing["goal_mismatches"].append(
                {"operation_id": operation_id, "reason": "missing artifact"}
            )
            continue
        sends = _goal_sends(goal_path)
        observed: dict[tuple[int, int], int] = {}
        for pairs in sends.values():
            for pair, size in pairs.items():
                observed[pair] = observed.get(pair, 0) + size
        expected_pairs: dict[tuple[int, int], int] = {}
        for (_request_id, source, destination), size in independent.get(
            operation_id, {}
        ).items():
            pair = (source, destination)
            expected_pairs[pair] = expected_pairs.get(pair, 0) + size
        routing["goal_rows"] += len(expected_pairs)
        if observed != expected_pairs:
            routing["goal_mismatches"].append(
                {
                    "operation_id": operation_id,
                    "artifact": goal_path.name,
                    "expected_pairs": len(expected_pairs),
                    "observed_pairs": len(observed),
                }
            )
        dispatch = operation_id.endswith(":ep-dispatch")
        for source, destination in observed:
            owner = source if dispatch else destination
            if owner != ENGINE_RANK:
                routing["ownership_violations"].append(
                    {
                        "operation_id": operation_id,
                        "source": source,
                        "destination": destination,
                    }
                )


# --------------------------------------------------------------- analysis ---


def _read_cells(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    cells = {}
    for name, *_ in CELLS:
        path = args.run_dir / "cells" / name / "cell.json"
        cells[name] = json.loads(path.read_text(encoding="utf-8"))
    return cells


def _stop_reason_agrees(row: dict[str, Any]) -> bool:
    oracle = row["oracle_stop_reason"]
    finish = row["finish_reason"]
    stop = row["stop_reason"]
    if oracle == "length-cap":
        return finish == "length"
    if oracle == "eos":
        return finish == "stop" and stop is None
    if oracle == "stop-string":
        return finish == "stop" and stop is not None
    raise AssertionError(f"unknown oracle stop reason {oracle!r}")


def _decode_steps(cell: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for step in cell["steps"]
        if step.get("simulated")
        and all(scheduled["phase"] == "decode" for scheduled in step["scheduled"])
        and step["scheduled"]
    ]


def _prefill_steps(cell: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for step in cell["steps"]
        if step.get("simulated")
        and any(scheduled["phase"] == "prefill" for scheduled in step["scheduled"])
    ]


def _composition_key(step: dict[str, Any]) -> tuple[tuple[str, str, int, int], ...]:
    """Identify a step by what it schedules rather than by its ordinal.

    A faster fabric reaches the next declared arrival after a different number
    of steps, so the same step index carries different work in different cells.
    Every cross-cell relation is therefore evaluated over steps that schedule
    exactly the same requests, phases, token counts and context lengths.
    """

    return tuple(
        (
            scheduled["request_id"],
            scheduled["phase"],
            scheduled["num_new_tokens"],
            scheduled["context_length"],
        )
        for scheduled in step["scheduled"]
    )


def _steps_by_composition(cell: dict[str, Any]) -> dict[Any, dict[str, Any]]:
    rows: dict[Any, dict[str, Any]] = {}
    for step in cell["steps"]:
        if not step.get("simulated"):
            continue
        key = _composition_key(step)
        if key in rows:
            raise AssertionError("two simulated steps share a scheduling composition")
        rows[key] = step
    return rows


def _fluid_fit(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Extract the ideal-network fixed term exactly from two link rates.

    Post-specified diagnostic. If artifact service is a fixed term plus bytes
    over the link rate, then ``2 * S(400G) - S(200G)`` cancels the byte term and
    leaves the fixed term alone, per artifact and with no averaging. Reporting
    it makes the network half of the error budget a measured constant rather
    than an assertion.
    """

    fast = _steps_by_composition(cells["a-ep8-400g"])
    half = _steps_by_composition(cells["a-ep8-200g"])
    fixed_terms = []
    implied_bytes = []
    for key in set(fast) & set(half):
        pairs = zip(
            fast[key]["fabric_phase_service_ps"],
            half[key]["fabric_phase_service_ps"],
            strict=True,
        )
        for quick, slow in pairs:
            if not quick and not slow:
                continue
            fixed_terms.append(2 * quick - slow)
            implied_bytes.append((slow - quick) / 20)
    if not fixed_terms:
        return {"artifacts": 0}
    return {
        "artifacts": len(fixed_terms),
        "fixed_term_ps": {
            "min": min(fixed_terms),
            "max": max(fixed_terms),
            "mean": sum(fixed_terms) / len(fixed_terms),
        },
        "implied_bottleneck_bytes": {
            "min": min(implied_bytes),
            "max": max(implied_bytes),
        },
    }


def _halving_response(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Score C5.2 over every halving the frozen bandwidth ladder contains.

    The clause names no particular pair and the ladder has three rates, so both
    400 to 200 and 200 to 100 are in scope. Step makespans are compared over
    identical scheduling compositions; TTFT and TPOT are compared per request.
    """

    pairs = (("a-ep8-400g", "a-ep8-200g"), ("a-ep8-200g", "a-ep8-100g"))
    rounds = []
    for fast_name, slow_name in pairs:
        fast, slow = cells[fast_name], cells[slow_name]
        fast_steps = _steps_by_composition(fast)
        slow_steps = _steps_by_composition(slow)
        shared = set(fast_steps) & set(slow_steps)
        step_ratios = [
            slow_steps[key]["step_latency_ps"] / fast_steps[key]["step_latency_ps"]
            for key in shared
        ]
        fast_by_id = {row["request_id"]: row for row in fast["requests"]}
        slow_by_id = {row["request_id"]: row for row in slow["requests"]}
        latency_ratios = []
        for request_id, fast_row in fast_by_id.items():
            slow_row = slow_by_id[request_id]
            latency_ratios.append(
                {
                    "request_id": request_id,
                    "metric": "ttft",
                    "ratio": slow_row["ttft_ps"] / fast_row["ttft_ps"],
                }
            )
            if fast_row["tpot_numerator"] is None:
                continue
            quick = Fraction(fast_row["tpot_numerator"], fast_row["tpot_denominator"])
            drawn = Fraction(slow_row["tpot_numerator"], slow_row["tpot_denominator"])
            latency_ratios.append(
                {
                    "request_id": request_id,
                    "metric": "tpot",
                    "ratio": float(drawn / quick),
                    "tokens": fast_row["token_count"],
                }
            )
        violations = [
            row for row in latency_ratios if not 1.0 < row["ratio"] < 2.0
        ] + [
            {"metric": "step_makespan", "ratio": ratio}
            for ratio in step_ratios
            if not 1.0 < ratio < 2.0
        ]
        rounds.append(
            {
                "from": fast_name,
                "to": slow_name,
                "matched_steps": len(shared),
                "step_makespan_ratio_min": min(step_ratios, default=None),
                "step_makespan_ratio_max": max(step_ratios, default=None),
                "latency_comparisons": len(latency_ratios),
                "violations": violations,
                "passed": bool(step_ratios) and not violations,
            }
        )
    return {
        "passed": all(entry["passed"] for entry in rounds),
        "halvings": rounds,
    }


def _linearity(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_bandwidth = {
        cells[name]["bandwidth_bps"]: _steps_by_composition(cells[name])
        for name in ("a-ep8-100g", "a-ep8-200g", "a-ep8-400g")
    }
    shared = set.intersection(*(set(rows) for rows in by_bandwidth.values()))
    compared = 0
    failures = []
    worst = 0.0
    for key in sorted(shared, key=lambda value: (len(value), value)):
        services = {
            bandwidth: rows[key]["fabric_phase_service_ps"]
            for bandwidth, rows in by_bandwidth.items()
        }
        lengths = {len(value) for value in services.values()}
        if len(lengths) != 1:
            failures.append({"composition": list(key), "reason": "artifact count differs"})
            continue
        for index in range(next(iter(lengths))):
            slow = services[100_000_000_000][index]
            middle = services[200_000_000_000][index]
            fast = services[400_000_000_000][index]
            if not slow and not middle and not fast:
                continue
            compared += 1
            residual = abs(slow - 3 * middle + 2 * fast)
            tolerance = max(C5_1_ABSOLUTE_TOLERANCE_PS, C5_1_RELATIVE_TOLERANCE * slow)
            worst = max(worst, residual / slow if slow else 0.0)
            if residual > tolerance:
                failures.append(
                    {
                        "composition": list(key),
                        "artifact": index,
                        "s100": slow,
                        "s200": middle,
                        "s400": fast,
                        "residual_ps": residual,
                    }
                )
    return {
        "matched_steps": len(shared),
        "compared": compared,
        "failures": failures[:20],
        "failure_count": len(failures),
        "worst_relative_residual": worst,
        "passed": compared > 0 and not failures,
    }


def _summarize(args: argparse.Namespace) -> dict[str, Any]:
    cells = _read_cells(args)
    exact: dict[str, dict[str, Any]] = {}
    fatal: dict[str, dict[str, Any]] = {}

    def score(name: str, ok: bool, detail: dict[str, Any]) -> None:
        exact[name] = {"passed": bool(ok), **detail}

    def guard(name: str, ok: bool, detail: dict[str, Any]) -> None:
        fatal[name] = {"held": bool(ok), **detail}

    rows = [(name, row) for name, cell in cells.items() for row in cell["requests"]]
    score(
        "C1.1",
        all(row["served_matches_oracle"] for _, row in rows),
        {
            "requests": len(rows),
            "divergent": [name for name, row in rows if not row["served_matches_oracle"]],
        },
    )
    score(
        "C1.2",
        all(_stop_reason_agrees(row) for _, row in rows),
        {
            "requests": len(rows),
            "disagreeing": [
                {"cell": name, "request_id": row["request_id"]}
                for name, row in rows
                if not _stop_reason_agrees(row)
            ],
            "oracle_stop_reasons": sorted({row["oracle_stop_reason"] for _, row in rows}),
        },
    )
    score(
        "C1.3",
        all(
            cell["identity"]["add_request_returned_oracle_id"]
            and cell["identity"]["scheduler_ids_subset"]
            and cell["identity"]["replay_completed"]
            == sorted(row["request_id"] for row in cell["requests"])
            for cell in cells.values()
        ),
        {"cells": len(cells)},
    )

    token_cells = sum(cell["routing"]["token_cells_compared"] for cell in cells.values())
    token_equal = sum(cell["routing"]["token_cells_equal"] for cell in cells.values())
    score(
        "C2.1",
        token_cells > 0 and token_cells == token_equal,
        {"compared": token_cells, "equal": token_equal},
    )
    pair_rows = sum(cell["routing"]["request_pair_rows"] for cell in cells.values())
    pair_bad = sum(len(cell["routing"]["request_pair_mismatches"]) for cell in cells.values())
    score(
        "C2.2",
        pair_rows > 0 and pair_bad == 0,
        {
            "rows": pair_rows,
            "mismatched_operations": pair_bad,
            "layer_bytes": {
                name: cell["routing"]["layer_bytes"] for name, cell in cells.items()
            },
            "request_bytes": {
                name: cell["routing"]["request_bytes"] for name, cell in cells.items()
            },
        },
    )
    goal_rows = sum(cell["routing"]["goal_rows"] for cell in cells.values())
    goal_bad = sum(len(cell["routing"]["goal_mismatches"]) for cell in cells.values())
    score("C2.3", goal_rows > 0 and goal_bad == 0, {"rows": goal_rows, "mismatches": goal_bad})
    ownership_bad = sum(
        len(cell["routing"]["ownership_violations"]) for cell in cells.values()
    )
    score("C2.4", ownership_bad == 0, {"violations": ownership_bad})
    guard(
        "C2.5",
        all(
            step.get("routing_mode") == "captured"
            and step.get("placement_epoch") == 0
            and step.get("quiescent") is True
            for cell in cells.values()
            for step in cell["steps"]
            if step.get("simulated")
        ),
        {
            "simulated_steps": sum(cell["simulated_steps"] for cell in cells.values()),
        },
    )

    guard(
        "C3.1",
        all(
            cell["conservation"]["makespan_conservation_failures"] == 0
            and step["completed_at_ps"] == step["virtual_time_ps"] + step["step_latency_ps"]
            for cell in cells.values()
            for step in cell["steps"]
        ),
        {
            "failures": sum(
                cell["conservation"]["makespan_conservation_failures"]
                for cell in cells.values()
            )
        },
    )
    intervals = sum(cell["conservation"]["intervals"] for cell in cells.values())
    score(
        "C3.2",
        intervals > 0
        and all(
            cell["conservation"]["interval_conservation_failures"] == 0
            for cell in cells.values()
        ),
        {"intervals": intervals},
    )
    guard(
        "C3.3",
        all(
            cell["conservation"]["inactive_component_violations"] == 0
            for cell in cells.values()
        ),
        {"intervals": intervals},
    )
    guard(
        "C3.4",
        all(
            cell["conservation"]["artifact_partition_failures"] == 0
            for cell in cells.values()
        ),
        {"artifacts": sum(len(step.get("composed_phase_service_ps", [])) for cell in
                          cells.values() for step in cell["steps"])},
    )
    score(
        "C3.5",
        all(
            row["ttft_attribution"] is not None
            and row["ttft_ps"] == row["first_token_at_ps"] - row["arrival_ps"]
            and row["ttft_attribution"]["total_ps"] == row["ttft_ps"]
            for _, row in rows
        ),
        {
            "requests": len(rows),
            "without_first_token": [
                {"cell": name, "request_id": row["request_id"]}
                for name, row in rows
                if row["ttft_attribution"] is None
            ],
        },
    )
    tpot_rows = [row for _, row in rows if row["token_count"] and row["token_count"] > 1]
    score(
        "C3.6",
        bool(tpot_rows)
        and all(
            Fraction(row["tpot_numerator"], row["tpot_denominator"]) * (row["token_count"] - 1)
            == row["decode_attribution"]["total_ps"]
            == row["last_token_at_ps"] - row["first_token_at_ps"]
            for row in tpot_rows
        ),
        {"requests": len(tpot_rows)},
    )
    score(
        "C3.7",
        all(row["ttft_ps"] is not None and row["ttft_ps"] > 0 for _, row in rows),
        {
            "requests": len(rows),
            "nonpositive": [
                {"cell": name, "request_id": row["request_id"], "ttft_ps": row["ttft_ps"]}
                for name, row in rows
                if row["ttft_ps"] is None or row["ttft_ps"] <= 0
            ],
        },
    )
    score(
        "C3.8",
        all(
            cell["conservation"]["kernel_disagrees_with_compute_service"] == 0
            for cell in cells.values()
        ),
        {
            "disagreements": sum(
                cell["conservation"]["kernel_disagrees_with_compute_service"]
                for cell in cells.values()
            )
        },
    )

    case_b = cells["b-ep8-400g"]
    score(
        "C4.1",
        len(case_b["requests"]) == 12
        and all(row["served_matches_oracle"] for row in case_b["requests"])
        and case_b["conservation"]["interval_conservation_failures"] == 0
        and not case_b["routing"]["request_pair_mismatches"]
        and not case_b["routing"]["goal_mismatches"]
        and not case_b["routing"]["ownership_violations"],
        {"requests": len(case_b["requests"])},
    )

    behavioral: dict[str, dict[str, Any]] = {}
    behavioral["C5.1"] = _linearity(cells)
    fast = cells["a-ep8-400g"]
    behavioral["C5.2"] = _halving_response(cells)
    ep4 = cells["a-ep4-400g"]
    ep8_bytes = sum(int(value) for value in fast["routing"]["layer_bytes"].values())
    ep4_bytes = sum(int(value) for value in ep4["routing"]["layer_bytes"].values())
    behavioral["C5.3"] = {
        "passed": 0 < ep4_bytes < ep8_bytes,
        "ep8_bytes": ep8_bytes,
        "ep4_bytes": ep4_bytes,
        "ratio": ep4_bytes / ep8_bytes if ep8_bytes else None,
    }
    ep8_compute = [
        step["compute_service_ps"] for step in fast["steps"] if step.get("simulated")
    ]
    ep4_compute = [
        step["compute_service_ps"] for step in ep4["steps"] if step.get("simulated")
    ]
    compute_ratio = (
        (sum(ep4_compute) / len(ep4_compute)) / (sum(ep8_compute) / len(ep8_compute))
        if ep8_compute and ep4_compute
        else None
    )
    behavioral["C5.4"] = {
        "passed": compute_ratio is not None
        and C5_4_COMPUTE_RATIO_FLOOR <= compute_ratio <= C5_4_COMPUTE_RATIO_CEILING,
        "ratio": compute_ratio,
        "ep8_mean_ps": sum(ep8_compute) / len(ep8_compute) if ep8_compute else None,
        "ep4_mean_ps": sum(ep4_compute) / len(ep4_compute) if ep4_compute else None,
    }

    prefill = _prefill_steps(fast)
    decode = _decode_steps(fast)
    step0_bytes = prefill[0]["total_directed_bytes"] if prefill else 0
    compute_values = [step["compute_service_ps"] for step in decode]
    spread = (
        (max(compute_values) - min(compute_values)) / min(compute_values)
        if compute_values and min(compute_values)
        else 0.0
    )
    decode_rates = [
        PS_PER_SECOND / (row["tpot_numerator"] / row["tpot_denominator"])
        for _, row in rows
        if row["tpot_numerator"] is not None
    ]
    bounds = {
        "S1": {
            "value": step0_bytes,
            "floor": S1_MOE_BYTES_FLOOR,
            "ceiling": S1_MOE_BYTES_CEILING,
            "held": S1_MOE_BYTES_FLOOR <= step0_bytes <= S1_MOE_BYTES_CEILING,
        },
        "S3": {
            "values": [min(compute_values, default=0), max(compute_values, default=0)],
            "floor": S3_COMPUTE_FLOOR_PS,
            "ceiling": S3_COMPUTE_CEILING_PS,
            "held": bool(compute_values)
            and all(
                S3_COMPUTE_FLOOR_PS <= value <= S3_COMPUTE_CEILING_PS
                for value in compute_values
            ),
        },
        "S4": {
            "values": [step["step_latency_ps"] for step in prefill],
            "floor": S4_PREFILL_FLOOR_PS,
            "ceiling": S4_PREFILL_CEILING_PS,
            "held": bool(prefill)
            and all(
                S4_PREFILL_FLOOR_PS <= step["step_latency_ps"] <= S4_PREFILL_CEILING_PS
                for step in prefill
            ),
        },
        "S5": {
            "min": min((step["step_latency_ps"] for step in decode), default=0),
            "max": max((step["step_latency_ps"] for step in decode), default=0),
            "floor": S5_DECODE_FLOOR_PS,
            "ceiling": S5_DECODE_CEILING_PS,
            "held": bool(decode)
            and all(
                S5_DECODE_FLOOR_PS <= step["step_latency_ps"] <= S5_DECODE_CEILING_PS
                for step in decode
            ),
        },
        "S6": {
            "relative_spread": spread,
            "ceiling": S6_COMPUTE_SPREAD,
            "held": spread <= S6_COMPUTE_SPREAD,
        },
        "S7": {
            "min_tokens_per_second": min(decode_rates, default=0.0),
            "max_tokens_per_second": max(decode_rates, default=0.0),
            "floor": S7_DECODE_RATE_FLOOR,
            "ceiling": S7_DECODE_RATE_CEILING,
            "held": bool(decode_rates)
            and all(
                S7_DECODE_RATE_FLOOR <= rate <= S7_DECODE_RATE_CEILING
                for rate in decode_rates
            ),
        },
    }
    for name, detail in bounds.items():
        guard(name, detail["held"], detail)

    # Post-specified diagnostics. These were not frozen and are reported, never
    # scored. The S1 literal was derived from a 54-token first prefill step;
    # arrival gating admits one request at a time, so the realized first prefill
    # step carries only the leading request's prompt. The recomputed bound below
    # is the same first-principles rule applied to the token count the run
    # actually scheduled.
    first_prefill = prefill[0] if prefill else None
    observed_tokens = first_prefill["total_new_tokens"] if first_prefill else 0
    per_token_floor = observed_tokens * NUM_LAYERS * 2 * 1 * VECTOR_BYTES
    per_token_ceiling = observed_tokens * NUM_LAYERS * 2 * 7 * VECTOR_BYTES
    diagnostics = {
        "s1_recomputed_for_observed_tokens": {
            "first_prefill_tokens": observed_tokens,
            "frozen_token_assumption": CASE_A_PROMPT_TOKENS,
            "value": step0_bytes,
            "floor": per_token_floor,
            "ceiling": per_token_ceiling,
            "inside": per_token_floor <= step0_bytes <= per_token_ceiling,
            "mean_remote_destinations_per_token_layer": (
                step0_bytes / (observed_tokens * NUM_LAYERS * 2 * VECTOR_BYTES)
                if observed_tokens
                else None
            ),
        },
        "schedule_divergence": {
            name: {
                "simulated_steps": cells[name]["simulated_steps"],
                "compositions_shared_with_400g": len(
                    set(_steps_by_composition(cells[name]))
                    & set(_steps_by_composition(cells["a-ep8-400g"]))
                ),
            }
            for name in ("a-ep8-100g", "a-ep8-200g", "a-ep4-400g")
        },
        "fluid_service_fit": _fluid_fit(cells),
    }

    scale = {
        name: {
            "requests": len(cells[name]["requests"]),
            "wall_seconds": cells[name]["wall_seconds"],
            "scheduler_steps": cells[name]["scheduler_steps"],
            "simulated_steps": cells[name]["simulated_steps"],
            "htsim_invocations": cells[name]["htsim_invocations"],
            "total_routed_bytes": cells[name]["total_routed_bytes"],
            "peak_rank_egress_bytes": max(
                int(value) for value in cells[name]["routing"]["rank_egress_bytes"].values()
            ),
        }
        for name in ("a-ep8-400g", "b-ep8-400g")
    }

    void = [name for name, detail in fatal.items() if not detail["held"]]
    summary = {
        "freeze_commit": FREEZE_COMMIT,
        "fatal_guards": fatal,
        "void": bool(void),
        "violated_fatal_guards": void,
        "exact_oracle_relations": exact,
        "exact_oracle_passed": sum(1 for detail in exact.values() if detail["passed"]),
        "exact_oracle_total": len(exact),
        "behavioral_relations": behavioral,
        "behavioral_passed": sum(1 for detail in behavioral.values() if detail["passed"]),
        "behavioral_total": len(behavioral),
        "diagnostics": diagnostics,
        "scale": scale,
        "cells": {
            name: {
                "wall_seconds": cell["wall_seconds"],
                "scheduler_steps": cell["scheduler_steps"],
                "htsim_invocations": cell["htsim_invocations"],
                "requests": [
                    {
                        "request_id": row["request_id"],
                        "arrival_ps": row["arrival_ps"],
                        "ttft_ps": row["ttft_ps"],
                        "tpot_ps": (
                            None
                            if row["tpot_numerator"] is None
                            else row["tpot_numerator"] / row["tpot_denominator"]
                        ),
                        "token_count": row["token_count"],
                        "ttft_attribution": row["ttft_attribution"],
                        "decode_attribution": row["decode_attribution"],
                    }
                    for row in cell["requests"]
                ],
            }
            for name, cell in cells.items()
        },
    }
    if len(exact) != EXPECTED_EXACT_ORACLE_RELATIONS:
        raise AssertionError(
            f"expected {EXPECTED_EXACT_ORACLE_RELATIONS} exact-oracle relations, "
            f"evaluated {len(exact)}"
        )
    if len(behavioral) != EXPECTED_BEHAVIORAL_RELATIONS:
        raise AssertionError("behavioral relation count changed")
    return summary


FREEZE_COMMIT = "0e3d38b"


# ------------------------------------------------------------------ driver ---


def _run_child(args: argparse.Namespace, mode: str) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cache-dir",
        str(args.cache_dir),
        "--htsim-rnic",
        str(args.htsim_rnic),
        "--run-dir",
        str(args.run_dir),
        "--internal",
        mode,
    ]
    completed = subprocess.run(command, check=False, env=os.environ.copy())
    if completed.returncode != 0:
        raise SystemExit(f"child stage {mode!r} failed with code {completed.returncode}")


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    args.run_dir.mkdir(parents=True, exist_ok=False)
    os.environ["SIMLLM_HTSIM_RNIC"] = str(args.htsim_rnic)
    _run_child(args, "capture")
    for name, *_ in CELLS:
        _run_child(args, f"cell:{name}")
    summary = _summarize(args)
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--htsim-rnic", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--internal", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_only(args)
    if args.check_only:
        return
    if args.internal == "capture":
        _capture(args)
        return
    if args.internal and args.internal.startswith("cell:"):
        _cell(args, args.internal.removeprefix("cell:"))
        return
    if args.internal == "summarize":
        summary = _summarize(args)
        (args.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary["exact_oracle_relations"], indent=2, sort_keys=True))
        return
    summary = run_study(args)
    print(
        json.dumps(
            {
                "void": summary["void"],
                "violated_fatal_guards": summary["violated_fatal_guards"],
                "exact_oracle": [
                    summary["exact_oracle_passed"],
                    summary["exact_oracle_total"],
                ],
                "behavioral": [summary["behavioral_passed"], summary["behavioral_total"]],
                "scale": summary["scale"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
