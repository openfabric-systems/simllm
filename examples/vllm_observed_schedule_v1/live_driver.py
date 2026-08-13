"""Drive the frozen Granite replay through real vLLM DP scheduler processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import socket
import traceback
from pathlib import Path
from typing import Any

DP_SIZE = 8
NUM_LAYERS = 24
NUM_EXPERTS = 32
MODEL_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--replay-run", type=Path, required=True)
    parser.add_argument("--routed-experts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--producer-mode",
        choices=("enabled", "disabled"),
        default="enabled",
    )
    return parser.parse_args()


def _model_dims() -> Any:
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=NUM_LAYERS,
        hidden_size=1_024,
        intermediate_size=512,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=49_155,
        dtype_bytes=2,
        num_experts=NUM_EXPERTS,
        top_k=8,
        moe_intermediate_size=512,
        local_num_experts=NUM_EXPERTS // DP_SIZE,
    )


def _routed_supply(path: Path) -> Any:
    from simllm.preplay import read_routed_experts
    from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply

    placement = ExpertPlacementSnapshot(
        placement_epoch=0,
        expert_owners=tuple(
            (layer, expert, expert % DP_SIZE)
            for layer in range(NUM_LAYERS)
            for expert in range(NUM_EXPERTS)
        ),
    )
    return RoutedMoeSupply(
        placements=(placement,),
        step_placement_epochs=tuple((step, 0) for step in range(4_096)),
        routed_experts=read_routed_experts(path),
        engine_rank=0,
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_identity(value: object) -> dict[str, object]:
    wire = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return {
        "bytes": len(wire),
        "sha256": hashlib.sha256(wire).hexdigest(),
    }


def _byte_identity(value: bytes) -> dict[str, object]:
    return {
        "bytes": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
    }


class _LiveObservationSink:
    """Persist the producer tuple and execute its cross-node live metric path."""

    def __init__(self, output: Path, routed_experts: Path) -> None:
        from simllm.backends import DeviceRuntimeStepSink, SerialStepLowererConfig
        from simllm.compute import RooflineProvider
        from simllm.core import CoarseDeviceProfile, CoarseDeviceRuntime

        class _OneRankPerNodeProfile(CoarseDeviceProfile):
            def node_gpu(self, rank: int) -> tuple[int, int]:
                super().node_gpu(rank)
                return rank, 0

        config = SerialStepLowererConfig(
            _model_dims(),
            (0,),
            ep_ranks=tuple(range(DP_SIZE)),
            provider=RooflineProvider(efficiency=0.7),
            routed_moe_supply=_routed_supply(routed_experts),
        )
        profile = _OneRankPerNodeProfile(
            rnic_rate_bps=400_000_000_000,
        )
        self._delegate = DeviceRuntimeStepSink(
            config,
            runtime=CoarseDeviceRuntime(profile),
        )
        self._output = output
        self._started = False

    def bind_clock(self, clock: Any) -> None:
        self._delegate.bind_clock(clock)

    def __call__(self, record: Any, observations: Any) -> Any:
        from simllm.core import (
            execution_graph_from_observations,
            execution_graph_to_json,
            execution_result_to_json,
            step_record_to_json,
            step_result_to_json,
        )

        if record.total_new_tokens > 0 and observations is None:
            raise RuntimeError("active Granite step emitted no ExecutionObservations")
        source_graph = (
            None
            if observations is None
            else execution_graph_to_json(
                execution_graph_from_observations(
                    record,
                    observations,
                    execution_id=f"source-step-{record.step_index}",
                )
            )
        )
        result = self._delegate(record, observations)
        outcome = self._delegate.outcomes[-1]
        lowered_graph = execution_graph_to_json(outcome.graph)
        execution_result = execution_result_to_json(outcome.execution_result)
        row = {
            "record": step_record_to_json(record),
            "source_graph": source_graph,
            "lowered_graph_identity": _canonical_identity(lowered_graph),
            "execution_result_identity": _canonical_identity(execution_result),
            "step_result": step_result_to_json(result),
        }
        if not self._started:
            self._output.write_text("", encoding="utf-8")
            self._started = True
        with self._output.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        return result


class _LiveLegacySink:
    """Exercise the producer-off one-argument sink and exact serial path."""

    def __init__(self, output: Path, routed_experts: Path) -> None:
        from simllm.backends import (
            DeviceRuntimeStepSink,
            SerialStepLowerer,
            SerialStepLowererConfig,
        )
        from simllm.compute import RooflineProvider
        from simllm.core import CoarseDeviceProfile, CoarseDeviceRuntime

        class _OneRankPerNodeProfile(CoarseDeviceProfile):
            def node_gpu(self, rank: int) -> tuple[int, int]:
                super().node_gpu(rank)
                return rank, 0

        self._config = SerialStepLowererConfig(
            _model_dims(),
            (0,),
            ep_ranks=tuple(range(DP_SIZE)),
            provider=RooflineProvider(efficiency=0.7),
            routed_moe_supply=_routed_supply(routed_experts),
        )
        self._lowerer = SerialStepLowerer(self._config)
        self._delegate = DeviceRuntimeStepSink(
            self._config,
            runtime=CoarseDeviceRuntime(
                _OneRankPerNodeProfile(rnic_rate_bps=400_000_000_000)
            ),
        )
        self._output = output
        self._clock = None
        self._calls = 0

    def __call__(self, record: Any) -> Any:
        from simllm.core import (
            VirtualClock,
            execution_graph_to_json,
            execution_result_to_json,
            step_record_to_json,
            step_result_to_json,
        )
        from simllm.traffic import project_execution_graph_goal, render_step_goal

        if self._clock is None:
            self._clock = VirtualClock(start_ps=record.virtual_time_ps)
            self._delegate.bind_clock(self._clock)
        if self._clock.now_ps != record.virtual_time_ps:
            raise RuntimeError("legacy serial sink clock drifted from the adapter")
        result = self._delegate(record, None)
        outcome = self._delegate.outcomes[-1]
        graph_json = execution_graph_to_json(outcome.graph)
        execution_json = execution_result_to_json(outcome.execution_result)
        goal = render_step_goal(
            record,
            self._config.dims,
            self._config.tp_ranks,
            per_layer_calc_ns=self._lowerer.timing(record).layer_calc_ns,
            ep_ranks=self._config.ep_ranks,
            routed_supply=self._config.routed_moe_supply,
        ).render().encode()
        projection = project_execution_graph_goal(outcome.graph)
        projection_json = {
            "artifacts": [
                artifact.trace.render() for artifact in projection.artifacts
            ],
            "boundaries": len(projection.boundaries),
            "serialized_edges": len(projection.serialized_edges),
            "step_index": record.step_index,
        }
        projection_identity = {
            "artifacts": [
                _byte_identity(artifact.trace.render().encode())
                for artifact in projection.artifacts
            ],
            "boundaries": len(projection.boundaries),
            "serialized_edges": len(projection.serialized_edges),
            "step_index": record.step_index,
        }
        row = {
            "execution_result": execution_json,
            "execution_result_identity": _canonical_identity(execution_json),
            "legacy_call_arity": 1,
            "legacy_call_index": self._calls,
            "legacy_goal": goal.decode(),
            "legacy_goal_identity": _byte_identity(goal),
            "lowered_graph": graph_json,
            "lowered_graph_identity": _canonical_identity(graph_json),
            "observations_absent": True,
            "projected_goal": projection_json,
            "projected_goal_identity": projection_identity,
            "record": step_record_to_json(record),
            "record_identity": _canonical_identity(step_record_to_json(record)),
            "source_graph": None,
            "step_result": step_result_to_json(result),
        }
        if self._calls == 0:
            self._output.write_text("", encoding="utf-8")
        with self._output.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        self._calls += 1
        return result


def _configure_rank_environment(rank: int, port: int, producer_mode: str) -> None:
    values = {
        "OMP_NUM_THREADS": "1",
        "SIMLLM_VLLM_MODE": "virtual",
        "SIMLLM_VLLM_OBSERVED_SCHEDULE": (
            "granite-dbo" if producer_mode == "enabled" else "off"
        ),
        "SIMLLM_VLLM_WORKER_MODE": "skeleton",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
        "VLLM_DP_MASTER_IP": "127.0.0.1",
        "VLLM_DP_MASTER_PORT": str(port),
        "VLLM_DP_RANK": str(rank),
        "VLLM_DP_RANK_LOCAL": str(rank),
        "VLLM_DP_SIZE": str(DP_SIZE),
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_LOGGING_LEVEL": "ERROR",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "fork",
    }
    os.environ.update(values)


def _run_rank(
    rank: int,
    port: int,
    capture: str,
    replay_run: str,
    routed_experts: str,
    output_dir: str,
    producer_mode: str,
) -> None:
    output = Path(output_dir)
    try:
        _configure_rank_environment(rank, port, producer_mode)
        from vllm import LLM, SamplingParams

        from simllm.adapters.vllm import SimExecutorConfig, configure, reset_configuration
        from simllm.preplay import read_preplay_trace

        trace = read_preplay_trace(capture)
        if trace.provenance.model_revision != MODEL_REVISION:
            raise RuntimeError("capture model revision disagrees with the frozen replay")
        sink = (
            _LiveObservationSink(
                output / "live_observations.jsonl",
                Path(routed_experts),
            )
            if producer_mode == "enabled"
            else _LiveLegacySink(
                output / "live_observations.jsonl",
                Path(routed_experts),
            )
        )
        reset_configuration()
        configure(
            step_sink=sink,
            config=SimExecutorConfig(
                mode="virtual",
                token_id=512,
                step_records_path=(
                    str(output / "live_steps.jsonl") if rank == 0 else None
                ),
                replay_run_path=replay_run if rank == 0 else None,
                observed_schedule=(
                    "granite-dbo" if producer_mode == "enabled" else "off"
                ),
            ),
        )
        llm = LLM(
            model=trace.provenance.model_id,
            revision=trace.provenance.model_revision,
            worker_cls="simllm.adapters.vllm.SimWorker",
            dtype="bfloat16",
            enforce_eager=True,
            max_model_len=128,
            num_gpu_blocks_override=128,
            disable_log_stats=True,
            enable_chunked_prefill=False,
            async_scheduling=False,
            enable_expert_parallel=True,
            all2all_backend="deepep_high_throughput",
            enable_dbo=True,
            dbo_decode_token_threshold=2,
            dbo_prefill_token_threshold=512,
        )

        if rank == 0:
            requests = trace.requests
        else:
            requests = (trace.requests[2], trace.requests[2])
        expected: dict[str, tuple[int, ...]] = {}
        for index, request in enumerate(requests):
            request_id = request.request_id if rank == 0 else f"dummy-{rank}-{index}"
            max_tokens = len(request.output_token_ids) if rank == 0 else 32
            assigned = llm.llm_engine.add_request(
                request_id,
                {"prompt_token_ids": list(request.input_token_ids)},
                SamplingParams(
                    temperature=0.0,
                    max_tokens=max_tokens,
                    min_tokens=0,
                    stop=list(request.stop_strings) if rank == 0 else [],
                    detokenize=rank == 0,
                ),
            )
            if assigned != request_id:
                raise RuntimeError("vLLM changed a frozen request identity")
            if rank == 0:
                expected[request_id] = request.output_token_ids

        served: dict[str, list[int]] = {request_id: [] for request_id in expected}
        while llm.llm_engine.has_unfinished_requests():
            for response in llm.llm_engine.step():
                if rank != 0:
                    continue
                token_ids = list(response.outputs[0].token_ids)
                served[response.request_id] = token_ids
        llm.llm_engine.engine_core.shutdown()
        reset_configuration()
        if rank == 0:
            mismatches = {
                request_id: {
                    "expected": list(expected[request_id]),
                    "observed": served[request_id],
                }
                for request_id in expected
                if tuple(served[request_id]) != expected[request_id]
            }
            if mismatches:
                raise RuntimeError(f"live replay token mismatch: {mismatches!r}")
        _write_json(
            output / f"live_rank_{rank}.json",
            {
                "rank": rank,
                "request_count": len(requests),
                "producer_mode": producer_mode,
                "status": "PASS",
            },
        )
    except BaseException as exc:
        _write_json(
            output / f"live_rank_{rank}.json",
            {
                "error": repr(exc),
                "rank": rank,
                "status": "FAIL",
                "traceback": traceback.format_exc(),
            },
        )
        raise


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    port = _open_port()
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_run_rank,
            args=(
                rank,
                port,
                str(args.capture),
                str(args.replay_run),
                str(args.routed_experts),
                str(args.output_dir),
                args.producer_mode,
            ),
            name=f"vllm-observed-rank-{rank}",
        )
        for rank in range(DP_SIZE)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=900)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
    exit_codes = [process.exitcode for process in processes]
    if any(code != 0 for code in exit_codes):
        raise SystemExit(f"live vLLM rank exit codes: {exit_codes}")
    _write_json(
        args.output_dir / "live_driver.json",
        {
            "data_parallel_size": DP_SIZE,
            "exit_codes": exit_codes,
            "model_revision": MODEL_REVISION,
            "producer_mode": args.producer_mode,
            "status": "PASS",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
