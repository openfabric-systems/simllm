"""Qualify and measure the vLLM observed-schedule path."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

CAPTURE_SHA256 = "5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6"
CAPTURE_ROWS = 120
NUM_LAYERS = 24
EXPECTED_MOE_SITES = 48
EXPECTED_GENUINE_RISK_INSTANCES = 4
SERIAL_GRAPH_BYTES = 4_127
SERIAL_GRAPH_SHA256 = "aa3c836fe559973a7bf0940384c2e8a84e6af84e0fbd2c02d3b89774ee0c8e2d"
SERIAL_GOAL_BYTES = 1_880
SERIAL_GOAL_SHA256 = "7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6"

REDUCTION_BANDS_PS = {
    "single-node": (1_000_000, 5_000_000),
    "cross-node": (20_000_000, 130_000_000),
}
PERTURBATION_MIN_PS = {
    "single-node": 100_000,
    "cross-node": 5_000_000,
}

SOURCE_HASHES = {
    "vllm/model_executor/models/granitemoe.py": (
        "b60e452c3f28b25aa104c88869daa25c06a7fb6ed45bd34e908fa6a8395efda1"
    ),
    "vllm/v1/worker/gpu_model_runner.py": (
        "81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0"
    ),
    "vllm/v1/worker/ubatching.py": (
        "40391241c564feb5f16c77898ae6ae152ed6e71a4682e2a406387785d8de02d7"
    ),
    "vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py": (
        "465cdf1d6cee91b2ee8c2e43abbea6e8408976e3048c10f44c089f34b415bc60"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def check_expectation_registry(capture: Path, vllm_source: Path) -> None:
    """Validate frozen inputs and literals without importing target code."""

    if not capture.is_file():
        raise SystemExit(f"capture is not a file: {capture}")
    if _sha256(capture) != CAPTURE_SHA256:
        raise SystemExit("capture SHA-256 does not match the frozen input")
    if _line_count(capture) != CAPTURE_ROWS:
        raise SystemExit("capture row count does not match the frozen input")

    if not vllm_source.is_dir():
        raise SystemExit(f"vLLM source is not a directory: {vllm_source}")
    for relative, expected in SOURCE_HASHES.items():
        source = vllm_source / relative
        if not source.is_file():
            raise SystemExit(f"audited vLLM source is missing: {relative}")
        if _sha256(source) != expected:
            raise SystemExit(f"audited vLLM source hash changed: {relative}")

    assert NUM_LAYERS == 24
    assert EXPECTED_MOE_SITES == 2 * NUM_LAYERS
    assert EXPECTED_GENUINE_RISK_INSTANCES == 2 * len(REDUCTION_BANDS_PS)
    assert set(REDUCTION_BANDS_PS) == set(PERTURBATION_MIN_PS)
    for low, high in REDUCTION_BANDS_PS.values():
        assert 0 < low <= high
    for placement, minimum in PERTURBATION_MIN_PS.items():
        assert 0 < minimum <= REDUCTION_BANDS_PS[placement][1]
    assert SERIAL_GRAPH_BYTES > SERIAL_GOAL_BYTES > 0
    assert len(SERIAL_GRAPH_SHA256) == len(SERIAL_GOAL_SHA256) == 64


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


class _ProbeObservationSink:
    def __init__(self) -> None:
        self.clock = None
        self.calls: list[tuple[object, object]] = []

    def bind_clock(self, clock: object) -> None:
        self.clock = clock

    def __call__(self, record: object, observations: object) -> object:
        from simllm.core import StepResult

        self.calls.append((record, observations))
        return StepResult(
            step_index=record.step_index,
            step_latency_ps=0,
            completed_at_ps=record.virtual_time_ps,
        )


def _adapter_probe() -> dict[str, object]:
    """Run the import-free skeleton boundary without claiming a live framework."""

    from simllm.adapters.vllm import (
        GroupCoordinatorEvent,
        SimExecutorConfig,
        SimWorker,
        configure,
        reset_configuration,
    )
    from simllm.core import VirtualClock

    class _FakeDType:
        itemsize = 2

    class _FakeIrOpPriority:
        @staticmethod
        def set_default() -> None:
            return None

    class _FakeModelConfig:
        runner_type = "generate"
        dtype = _FakeDType()
        hf_text_config = SimpleNamespace(intermediate_size=256)
        max_model_len = 4096

        @staticmethod
        def get_hidden_size() -> int:
            return 128

        @staticmethod
        def get_num_layers(parallel_config: object) -> int:
            return 4

        @staticmethod
        def get_num_attention_heads(parallel_config: object) -> int:
            return 8

        @staticmethod
        def get_num_kv_heads(parallel_config: object) -> int:
            return 2

        @staticmethod
        def get_head_size() -> int:
            return 16

        @staticmethod
        def get_vocab_size() -> int:
            return 1024

        @staticmethod
        def get_total_num_hidden_layers() -> int:
            return 4

    @dataclasses.dataclass
    class _NewRequest:
        req_id: str
        prompt_token_ids: list[int]
        num_computed_tokens: int = 0
        sampling_params: object | None = None

    @dataclasses.dataclass
    class _CachedRequests:
        req_ids: list[str] = dataclasses.field(default_factory=list)
        num_computed_tokens: list[int] = dataclasses.field(default_factory=list)
        num_output_tokens: list[int] = dataclasses.field(default_factory=list)

    @dataclasses.dataclass
    class _SchedulerOutput:
        scheduled_new_reqs: list[_NewRequest] = dataclasses.field(default_factory=list)
        scheduled_cached_reqs: _CachedRequests = dataclasses.field(
            default_factory=_CachedRequests
        )
        num_scheduled_tokens: dict[str, int] = dataclasses.field(default_factory=dict)
        finished_req_ids: set[str] = dataclasses.field(default_factory=set)
        preempted_req_ids: set[str] | None = None
        has_structured_output_requests: bool = False

        @property
        def total_num_scheduled_tokens(self) -> int:
            return sum(self.num_scheduled_tokens.values())

    vllm_config = SimpleNamespace(
        model_config=_FakeModelConfig(),
        cache_config=SimpleNamespace(
            block_size=16,
            cache_dtype="auto",
            num_gpu_blocks=None,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            world_size=1,
            rank=0,
        ),
        scheduler_config=SimpleNamespace(),
        device_config=SimpleNamespace(device="cuda"),
        speculative_config=None,
        lora_config=None,
        load_config=None,
        observability_config=None,
        kv_transfer_config=None,
        compilation_config=SimpleNamespace(ir_enable_torch_wrap=True),
        kernel_config=SimpleNamespace(ir_op_priority=_FakeIrOpPriority()),
        profiler_config=SimpleNamespace(profiler=None),
        quant_config=None,
        use_v2_model_runner=False,
    )
    sink = _ProbeObservationSink()
    previous_mode = os.environ.get("SIMLLM_VLLM_WORKER_MODE")
    os.environ["SIMLLM_VLLM_WORKER_MODE"] = "skeleton"
    reset_configuration()
    try:
        configure(step_sink=sink)
        worker = SimWorker(
            vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:1",
            is_driver_worker=True,
            _simllm_clock=VirtualClock(),
            _simllm_config=SimExecutorConfig(),
        )
        worker.init_device()
        worker.execute_model(
            _SchedulerOutput(
                scheduled_new_reqs=[_NewRequest("probe", [1, 2, 3, 4])],
                num_scheduled_tokens={"probe": 4},
            )
        )
        events = worker.coordinator_events
        received = sink.calls[0][1] if sink.calls else "missing-call"
        fields = tuple(field.name for field in dataclasses.fields(GroupCoordinatorEvent))
        missing_schedule_fields = tuple(
            name
            for name in (
                "layer",
                "logical_stream",
                "depends_on",
                "request_ids",
                "completion_operation_ids",
            )
            if name not in fields
        )
        return {
            "probe_kind": "import-free skeleton component",
            "fixture_layers": worker.dims.num_layers,
            "coordinator_event_count": len(events),
            "coordinator_operations": [event.operation for event in events],
            "coordinator_payload_bytes": [event.payload_bytes for event in events],
            "coordinator_event_fields": list(fields),
            "missing_required_schedule_fields": list(missing_schedule_fields),
            "sink_call_count": len(sink.calls),
            "sink_received_observations": received is not None,
        }
    finally:
        reset_configuration()
        if previous_mode is None:
            os.environ.pop("SIMLLM_VLLM_WORKER_MODE", None)
        else:
            os.environ["SIMLLM_VLLM_WORKER_MODE"] = previous_mode


def _serial_and_sink_guards() -> dict[str, object]:
    """Exercise exact serial identity and the new live reducer sink component."""

    from simllm.backends import (
        DeviceRuntimeStepSink,
        ObservedStepLowerer,
        SerialStepLowerer,
        SerialStepLowererConfig,
    )
    from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
    from simllm.core import (
        ExecutionObservations,
        RequestPhase,
        ScheduledRequest,
        StepRecord,
        VirtualClock,
        execution_graph_to_json,
    )
    from simllm.traffic import render_serial_execution_graph_goal

    class _FlopProvider(ComputeProvider):
        def estimate(self, kernel: object, gpu: object) -> object:
            return DurationEstimate(duration_ps=int(kernel.flops), bound="compute")

    dims = ModelDims(2, 64, 128, 4, 4, 16, 256, 2)
    record = StepRecord(
        0,
        0,
        [
            ScheduledRequest(
                "p",
                RequestPhase.PREFILL,
                4,
                num_cached_tokens=4,
                context_length=8,
            ),
            ScheduledRequest("d", RequestPhase.DECODE, 1, context_length=32),
        ],
        num_sampled=None,
    )
    config = SerialStepLowererConfig(dims, (0, 1), provider=_FlopProvider())
    serial_graph = ObservedStepLowerer(config).lower(record, None)
    direct_serial = SerialStepLowerer(config).lower(record)
    wire = (
        json.dumps(
            execution_graph_to_json(serial_graph),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    goal = render_serial_execution_graph_goal(serial_graph).render().encode()

    observations = ExecutionObservations(
        operations=direct_serial.operations,
        completion_operation_ids=direct_serial.completion_operation_ids,
    )
    clock = VirtualClock()
    sink = DeviceRuntimeStepSink(config)
    sink.bind_clock(clock)
    step_result = sink(record, observations)
    metric_ids = sorted(metric.request_id for metric in step_result.request_metrics)
    return {
        "serial_graph_bytes": len(wire),
        "serial_graph_sha256": hashlib.sha256(wire).hexdigest(),
        "serial_goal_bytes": len(goal),
        "serial_goal_sha256": hashlib.sha256(goal).hexdigest(),
        "serial_graph_matches_direct_delegate": (
            execution_graph_to_json(serial_graph)
            == execution_graph_to_json(direct_serial)
        ),
        "component_observation_routed": sink.outcomes[0].observations is observations,
        "component_completion_event_count": len(
            sink.outcomes[0].execution_result.events
        ),
        "component_request_metric_ids": metric_ids,
        "component_step_latency_ps": step_result.step_latency_ps,
        "component_clock_ps": clock.now_ps,
    }


def _git_head() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _run_study(args: argparse.Namespace) -> dict[str, object]:
    adapter = _adapter_probe()
    guards = _serial_and_sink_guards()
    serial_identity_passed = (
        guards["serial_graph_bytes"] == SERIAL_GRAPH_BYTES
        and guards["serial_graph_sha256"] == SERIAL_GRAPH_SHA256
        and guards["serial_goal_bytes"] == SERIAL_GOAL_BYTES
        and guards["serial_goal_sha256"] == SERIAL_GOAL_SHA256
        and guards["serial_graph_matches_direct_delegate"] is True
    )
    producer_qualified = (
        adapter["sink_received_observations"] is True
        and adapter["missing_required_schedule_fields"] == []
        and adapter["coordinator_event_count"] >= EXPECTED_MOE_SITES
    )
    return {
        "schema": "simllm-observed-schedule-study-v1",
        "expectation_commit": "409b4ade250fcc22ccb36cb4927399694e0cd318",
        "repository_commit_observed": _git_head(),
        "vllm_source": {
            "authored_against_commit": "568afb3a13806beb53bb2e6bd518269357b237c0",
            "observed_commit": None,
            "observed_file_sha256": SOURCE_HASHES,
        },
        "capture": {
            "sha256": _sha256(args.capture),
            "rows": _line_count(args.capture),
        },
        "adapter_probe": adapter,
        "producer_qualification": {
            "passed": producer_qualified,
            "required_moe_sites": EXPECTED_MOE_SITES,
            "observed_semantic_moe_sites": 0,
            "reasons": [
                "the skeleton sink received no ExecutionObservations",
                "the sole coordinator event has no layer or semantic site",
                "the worker exposes no EP group or dispatch/combine event",
                "the active source proves no legal next-layer overlap",
            ],
        },
        "behavioral_evidence": {
            "status": "blocked before behavioral execution",
            "genuine_risk_passed": 0,
            "genuine_risk_executed": 0,
            "genuine_risk_registered": EXPECTED_GENUINE_RISK_INSTANCES,
            "placements_executed": [],
            "ttft_tpot_rows": [],
            "dependency_perturbation_rows": [],
            "evaluation_order": "qualification before behavioral relations",
        },
        "fatal_unscored_guards": {
            "source_and_capture_identity": "PASS",
            "serial_identity": "PASS" if serial_identity_passed else "FAIL",
            "component_observation_routing": (
                "PASS" if guards["component_observation_routed"] else "FAIL"
            ),
            "per_request_component_metrics": (
                "PASS"
                if guards["component_request_metric_ids"] == ["d", "p"]
                else "FAIL"
            ),
        },
        "component_evidence": guards,
    }


def main() -> int:
    args = _parse_args()
    check_expectation_registry(args.capture, args.vllm_source)
    if args.check_only:
        print("observed-schedule expectation registry: PASS")
        return 0
    report = _run_study(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    qualification = report["producer_qualification"]
    behavior = report["behavioral_evidence"]
    print(f"producer qualification: {qualification['passed']}")
    print(
        "genuine-risk instances: "
        f"{behavior['genuine_risk_passed']}/{behavior['genuine_risk_executed']} "
        f"({behavior['status']})"
    )
    print(f"wrote {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
