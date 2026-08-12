"""Run the frozen virtual-time arrival admission study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from simllm.adapters.vllm import (
    SimExecutorConfig,
    configure,
    latest_worker,
    reset_configuration,
)
from simllm.core import (
    BookkeepingLedger,
    CoarseDeviceProfile,
    CoarseDeviceRuntime,
    CompletionReducer,
    ComputeWork,
    ExecutionGraph,
    ExecutionOperation,
    FrameworkRequestArrival,
    OperationCorrelation,
    ProcessingStage,
    RequestBookkeeper,
    RequestMetric,
    StagePhase,
    StageRecord,
    StepRecord,
    StepResult,
    VirtualClock,
    bookkeeping_ledger_to_json,
    framework_request_arrivals,
)
from simllm.preplay import (
    ForwardPhase,
    PreplayTrace,
    RequestArrival,
    StopReason,
    join_preplay_arrivals,
    read_preplay_trace,
    write_preplay_replay_run,
    write_preplay_trace,
)
from simllm.workload import AdmissionMode, RequestAdmissionGate

PS_PER_SECOND = 1_000_000_000_000
SERVICE_PS = 1_000_000
ARRIVAL_OFFSETS_PS = (750_000, 1_250_000)
LOAD_REQUEST_IDS = {
    "one": ("r0", "r1"),
    "three": ("r0", "r1", "r2", "r3"),
}
TOKENS = {
    "r0": (38, 39, 40, 41),
    "r1": (61,),
    "r2": (62,),
    "r3": (63,),
}
EXPECTED_QUEUE_PS = {
    ("one", 750_000, "r0"): 0,
    ("one", 750_000, "r1"): 250_000,
    ("one", 1_250_000, "r0"): 0,
    ("one", 1_250_000, "r1"): 750_000,
    ("three", 750_000, "r0"): 0,
    ("three", 750_000, "r1"): 250_000,
    ("three", 750_000, "r2"): 1_250_000,
    ("three", 750_000, "r3"): 2_250_000,
    ("three", 1_250_000, "r0"): 0,
    ("three", 1_250_000, "r1"): 750_000,
    ("three", 1_250_000, "r2"): 1_750_000,
    ("three", 1_250_000, "r3"): 2_750_000,
}
MODEL_RELATIVE_PATH = (
    Path("hub")
    / "models--ibm-granite--granite-3.0-1b-a400m-instruct"
    / "snapshots"
    / "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
)
GRANITE_FIXTURE = (
    Path(__file__).parents[1] / "preplay_trace_v1/granite_length_cap.jsonl"
)
GRANITE_SHA256 = "36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341"
NO_REPLAY_FIXTURE = (
    Path(__file__).parents[2]
    / "tests/fixtures/vllm/no_replay_r1_p4_steps.jsonl"
)
NO_REPLAY_SHA256 = "71862c9a49814bef3fc830f647f1b439d9c4d6ad0ef9707be6597528adb1808a"
SOURCE_HASHES = {
    "v1/engine/llm_engine.py": "17e5edfc625c77e9663368c7d69136e5e5935ee81608a65be3996411d502225e",
    "v1/engine/core.py": "3ae1381a6af841e21058c825702382dc66faae45c950ac5acb8495d2d3d05aad",
    "v1/core/sched/scheduler.py": "2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941",
    "v1/engine/input_processor.py": "c5673988c0f7cfec268220e3f044e718702c015a4f236c020937cfd40a793f15",
    "v1/request.py": "92124fbad28cda49bd06fa12c2c4fd5f53fc9381ddb4dc35f275c5ccfbd27378",
}


@dataclass(frozen=True, kw_only=True)
class CellResult:
    admission_kind: str
    load: str
    arrival_offset_ps: int
    arrivals_ps: dict[str, int]
    tokens: dict[str, tuple[int, ...]]
    finish_reasons: dict[str, str]
    token_times_ps: dict[str, tuple[int, ...]]
    first_releases_ps: dict[str, int]
    first_metrics: dict[str, RequestMetric]
    step_composition: tuple[tuple[str, ...], ...]
    step_stream_bytes: bytes
    output_bytes: bytes
    replay_snapshot_bytes: bytes
    replay_run_bytes: bytes
    bookkeeping_bytes: bytes
    final_clock_ps: int
    ledger: BookkeepingLedger


class CoreMetricSink:
    """Reduce every live adapter step through one shared core clock."""

    def __init__(self, bookkeeper: RequestBookkeeper) -> None:
        self.bookkeeper = bookkeeper
        self.runtime = CoarseDeviceRuntime(CoarseDeviceProfile())
        self.reducer: CompletionReducer | None = None

    def bind_clock(self, clock: VirtualClock) -> None:
        if self.reducer is not None:
            raise RuntimeError("metric sink clock is already bound")
        self.reducer = CompletionReducer(
            clock,
            bookkeeping=self.bookkeeper.snapshot(),
        )

    def __call__(self, record: StepRecord) -> StepResult | None:
        if not record.scheduled:
            return None
        if self.reducer is None:
            raise RuntimeError("metric sink clock was not bound before execution")
        request_ids = tuple(request.request_id for request in record.scheduled)
        operation = ExecutionOperation(
            operation_id=f"compute-{record.step_index}",
            rank=0,
            logical_queue="arrival-study-compute",
            work=ComputeWork(
                "arrival-study-first-token",
                nominal_duration_ps=SERVICE_PS,
            ),
            correlation=OperationCorrelation(request_ids=request_ids),
        )
        graph = ExecutionGraph(
            execution_id=f"arrival-admission-step-{record.step_index}",
            step_index=record.step_index,
            released_at_ps=record.virtual_time_ps,
            operations=(operation,),
            completion_operation_ids=(operation.operation_id,),
        )
        execution = self.runtime.execute(graph, bookkeeping=self.bookkeeper)
        report = self.runtime.last_report
        if report is None:
            raise AssertionError("core runtime omitted its report")
        return self.reducer.reduce(record, graph, execution, report)


def _required_directory(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must name an existing directory")
    path = Path(value)
    if not path.is_dir():
        raise RuntimeError(f"{name} must name an existing directory")
    return path


def _model_path() -> Path:
    return _required_directory("HF_HOME") / MODEL_RELATIVE_PATH


def _vllm_package_root() -> Path:
    return _required_directory("SIMLLM_VLLM_PACKAGE_ROOT")


@contextmanager
def _engine_environment() -> Iterator[None]:
    values = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
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


def _fixed_trace() -> PreplayTrace:
    granite = read_preplay_trace(GRANITE_FIXTURE)
    source = granite.requests[0]
    prefill = tuple(source.prefill_tokens[:2])
    prompt = tuple(token.token_id for token in prefill)
    route_template = source.prefill_tokens[0]
    requests = []
    for index, (request_id, output_tokens) in enumerate(TOKENS.items()):
        decode = tuple(
            replace(
                route_template,
                phase=ForwardPhase.DECODE,
                token_index=token_index,
                token_id=token_id,
            )
            for token_index, token_id in enumerate(output_tokens[:-1])
        )
        requests.append(
            replace(
                source,
                request_id=request_id,
                prompt_sha256=f"{index + 1:x}" * 64,
                input_token_ids=prompt,
                max_new_tokens=len(output_tokens),
                stop_strings=(),
                output_text=f"synthetic {request_id}",
                output_token_ids=output_tokens,
                stop_reason=StopReason.LENGTH_CAP,
                matched_stop_string=None,
                prefill_tokens=prefill,
                decode_tokens=decode,
            )
        )
    return PreplayTrace(provenance=granite.provenance, requests=tuple(requests))


def _write_trace(run_dir: Path) -> Path:
    trace = _fixed_trace()
    return write_preplay_trace(
        run_dir / "fixed-trace.jsonl",
        trace.provenance,
        trace.requests,
    )


def _arrivals(load: str, offset_ps: int) -> tuple[RequestArrival, ...]:
    return tuple(
        RequestArrival(
            request_id=request_id,
            arrived_at_ps=0 if request_id == "r0" else offset_ps,
        )
        for request_id in LOAD_REQUEST_IDS[load]
    )


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _observe_outputs(
    outputs: list[Any],
    completed_at_ps: int,
    served: dict[str, list[int]],
    token_times: dict[str, list[int]],
    finish_reasons: dict[str, str],
) -> None:
    for output in outputs:
        request_id = output.request_id
        if request_id not in served:
            raise AssertionError(f"engine returned unknown request {request_id!r}")
        if len(output.outputs) != 1:
            raise AssertionError(f"request {request_id!r} returned multiple choices")
        choice = output.outputs[0]
        cumulative = tuple(choice.token_ids)
        previous = tuple(served[request_id])
        if cumulative[: len(previous)] != previous:
            raise AssertionError(f"request {request_id!r} changed an emitted prefix")
        new_tokens = cumulative[len(previous) :]
        served[request_id].extend(new_tokens)
        token_times[request_id].extend([completed_at_ps] * len(new_tokens))
        if output.finished:
            if request_id in finish_reasons:
                raise AssertionError(f"request {request_id!r} finished twice")
            if choice.finish_reason is None:
                raise AssertionError(f"request {request_id!r} has no finish reason")
            finish_reasons[request_id] = choice.finish_reason


def _metric_json(metric: RequestMetric) -> dict[str, object]:
    return {
        "request_id": metric.request_id,
        "token_index": metric.token_index,
        "completed_at_ps": metric.completed_at_ps,
        "latency_ps": metric.latency_ps,
        "ttft_ps": metric.ttft_ps,
        "attribution": asdict(metric.attribution),
        "additive_visit_totals": asdict(metric.additive_visit_totals),
    }


def _drive_cell(
    cell_dir: Path,
    trace_path: Path,
    *,
    admission_kind: str,
    load: str,
    arrival_offset_ps: int,
) -> CellResult:
    from vllm import LLM, SamplingParams

    cell_dir.mkdir(parents=True, exist_ok=False)
    bookkeeper = RequestBookkeeper()
    replay_run = join_preplay_arrivals(
        _arrivals(load, arrival_offset_ps),
        trace_path,
        bookkeeper,
    )
    replay_path = write_preplay_replay_run(
        replay_run,
        cell_dir / "replay-run.json",
    )
    trace = read_preplay_trace(trace_path)
    requests = {
        request.request_id: request
        for request in trace.requests
        if request.request_id in LOAD_REQUEST_IDS[load]
    }
    stream_path = cell_dir / "steps.jsonl"
    sink = CoreMetricSink(bookkeeper)
    config = SimExecutorConfig(
        mode="virtual",
        token_id=512,
        step_records_path=str(stream_path),
        replay_run_path=str(replay_path),
    )
    served = {request_id: [] for request_id in LOAD_REQUEST_IDS[load]}
    token_times = {request_id: [] for request_id in LOAD_REQUEST_IDS[load]}
    finish_reasons: dict[str, str] = {}
    llm: Any | None = None
    worker: Any | None = None
    gate: RequestAdmissionGate | None = None
    reset_configuration()
    configure(step_sink=sink, config=config)
    try:
        with _engine_environment():
            llm = LLM(
                model=str(_model_path()),
                worker_cls="simllm.adapters.vllm.SimWorker",
                enforce_eager=True,
                max_model_len=64,
                max_num_seqs=2,
                num_gpu_blocks_override=64,
                disable_log_stats=True,
                enable_chunked_prefill=False,
                enable_prefix_caching=False,
                async_scheduling=False,
            )
            worker = latest_worker()
            if worker is None or worker.replay is None:
                raise AssertionError("replay SimWorker was not constructed")
            sink.bind_clock(worker.clock)

            def submit(arrival: FrameworkRequestArrival) -> None:
                request = requests[arrival.request_id]
                keyword = (
                    {"arrival_time": arrival.arrived_at_ps / PS_PER_SECOND}
                    if admission_kind == "gated"
                    else {}
                )
                request_id = llm.llm_engine.add_request(
                    arrival.request_id,
                    {"prompt_token_ids": list(request.input_token_ids)},
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=len(request.output_token_ids),
                        min_tokens=0,
                        stop=[],
                        detokenize=False,
                    ),
                    **keyword,
                )
                if request_id != arrival.request_id:
                    raise AssertionError(
                        f"scheduler changed {arrival.request_id!r} to {request_id!r}"
                    )

            if admission_kind == "direct":
                for arrival in framework_request_arrivals(bookkeeper.snapshot()):
                    submit(arrival)
            else:
                gate = RequestAdmissionGate(
                    worker.clock,
                    bookkeeper,
                    mode=(
                        AdmissionMode.ARRIVAL_GATED
                        if admission_kind == "gated"
                        else AdmissionMode.ALL_AT_ONCE
                    ),
                )

            while (
                (gate is not None and gate.has_pending)
                or llm.llm_engine.has_unfinished_requests()
            ):
                if gate is not None:
                    gate.admit_ready(submit)
                if llm.llm_engine.has_unfinished_requests():
                    before = len(worker.step_records)
                    outputs = llm.llm_engine.step()
                    if len(worker.step_records) != before + 1:
                        raise AssertionError(
                            "one engine step did not emit one adapter record"
                        )
                    _observe_outputs(
                        outputs,
                        worker.clock.now_ps,
                        served,
                        token_times,
                        finish_reasons,
                    )
                elif gate is not None and gate.has_pending:
                    gate.advance_to_next_arrival()

            records = tuple(worker.step_records)
            results = tuple(worker.step_results)
            replay_snapshot = worker.replay.snapshot()
            final_clock_ps = worker.clock.now_ps
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        reset_configuration()

    if worker is None:
        raise AssertionError("cell ended without constructing a worker")
    if len(records) != len(results):
        raise AssertionError("step record and result cardinality differ")
    first_releases: dict[str, int] = {}
    first_metrics: dict[str, RequestMetric] = {}
    for record, result in zip(records, results, strict=True):
        for scheduled in record.scheduled:
            first_releases.setdefault(
                scheduled.request_id,
                record.virtual_time_ps,
            )
        for metric in result.request_metrics:
            if metric.token_index == 1:
                if metric.request_id in first_metrics:
                    raise AssertionError("request emitted two first-token metrics")
                first_metrics[metric.request_id] = metric
    request_ids = set(LOAD_REQUEST_IDS[load])
    if set(first_releases) != request_ids or set(first_metrics) != request_ids:
        raise AssertionError("cell lost a first-service or first-token metric")
    if set(finish_reasons) != request_ids:
        raise AssertionError("cell lost a framework finish reason")
    arrivals_ps = {
        arrival.request_id: arrival.arrived_at_ps
        for arrival in framework_request_arrivals(bookkeeper.snapshot())
    }
    ledger = bookkeeper.snapshot()
    output_value = {
        "tokens": {
            request_id: served[request_id]
            for request_id in LOAD_REQUEST_IDS[load]
        },
        "finish_reasons": finish_reasons,
        "final_clock_ps": final_clock_ps,
        "replay_completed": replay_snapshot.completed_request_ids,
        "replay_drained": replay_snapshot.drained_request_ids,
    }
    snapshot_value = {
        "served_token_ids": replay_snapshot.served_token_ids,
        "completed_request_ids": replay_snapshot.completed_request_ids,
        "drained_request_ids": replay_snapshot.drained_request_ids,
    }
    return CellResult(
        admission_kind=admission_kind,
        load=load,
        arrival_offset_ps=arrival_offset_ps,
        arrivals_ps=arrivals_ps,
        tokens={request_id: tuple(values) for request_id, values in served.items()},
        finish_reasons=finish_reasons,
        token_times_ps={
            request_id: tuple(values)
            for request_id, values in token_times.items()
        },
        first_releases_ps=first_releases,
        first_metrics=first_metrics,
        step_composition=tuple(
            tuple(request.request_id for request in record.scheduled)
            for record in records
        ),
        step_stream_bytes=stream_path.read_bytes(),
        output_bytes=_canonical_bytes(output_value),
        replay_snapshot_bytes=_canonical_bytes(snapshot_value),
        replay_run_bytes=replay_path.read_bytes(),
        bookkeeping_bytes=_canonical_bytes(bookkeeping_ledger_to_json(ledger)),
        final_clock_ps=final_clock_ps,
        ledger=ledger,
    )


def _raw_queue(cell: CellResult, request_id: str) -> int:
    return cell.first_releases_ps[request_id] - cell.arrivals_ps[request_id]


def _raw_service(cell: CellResult, request_id: str) -> int:
    return (
        cell.token_times_ps[request_id][0]
        - cell.first_releases_ps[request_id]
    )


def _raw_ttft(cell: CellResult, request_id: str) -> int:
    return cell.token_times_ps[request_id][0] - cell.arrivals_ps[request_id]


def _metric_service(metric: RequestMetric) -> int:
    return metric.attribution.total_ps - metric.attribution.queue_ps


def _scored_check(
    check_id: str,
    observed: object,
    expected: object,
) -> dict[str, object]:
    return {
        "check_id": check_id,
        "observed": observed,
        "expected": expected,
        "passed": observed == expected,
        "genuine_risk": True,
    }


def _evaluate_scored(
    gated: dict[tuple[str, int], CellResult],
) -> dict[str, object]:
    spacing_checks: list[dict[str, object]] = []
    small_offset, large_offset = ARRIVAL_OFFSETS_PS
    for load, request_ids in LOAD_REQUEST_IDS.items():
        small = gated[(load, small_offset)]
        large = gated[(load, large_offset)]
        for request_id in request_ids[1:]:
            small_metric = small.first_metrics[request_id]
            large_metric = large.first_metrics[request_id]
            observed = {
                "raw_queue_delta_ps": (
                    _raw_queue(large, request_id)
                    - _raw_queue(small, request_id)
                ),
                "reported_queue_delta_ps": (
                    large_metric.attribution.queue_ps
                    - small_metric.attribution.queue_ps
                ),
                "raw_ttft_delta_ps": (
                    _raw_ttft(large, request_id)
                    - _raw_ttft(small, request_id)
                ),
                "reported_ttft_delta_ps": (
                    large_metric.ttft_ps - small_metric.ttft_ps
                ),
                "raw_service_delta_ps": (
                    _raw_service(large, request_id)
                    - _raw_service(small, request_id)
                ),
                "reported_service_delta_ps": (
                    _metric_service(large_metric)
                    - _metric_service(small_metric)
                ),
                "tokens_unchanged": (
                    large.tokens[request_id] == small.tokens[request_id]
                ),
                "finish_reason_unchanged": (
                    large.finish_reasons[request_id]
                    == small.finish_reasons[request_id]
                ),
            }
            spacing_checks.append(
                _scored_check(
                    f"spacing:{load}:{request_id}",
                    observed,
                    {
                        "raw_queue_delta_ps": 500_000,
                        "reported_queue_delta_ps": 500_000,
                        "raw_ttft_delta_ps": 500_000,
                        "reported_ttft_delta_ps": 500_000,
                        "raw_service_delta_ps": 0,
                        "reported_service_delta_ps": 0,
                        "tokens_unchanged": True,
                        "finish_reason_unchanged": True,
                    },
                )
            )

    load_checks: list[dict[str, object]] = []
    for offset_ps in ARRIVAL_OFFSETS_PS:
        cell = gated[("three", offset_ps)]
        for earlier, later in (("r1", "r2"), ("r2", "r3")):
            observed = {
                "raw_queue_delta_ps": (
                    _raw_queue(cell, later) - _raw_queue(cell, earlier)
                ),
                "reported_queue_delta_ps": (
                    cell.first_metrics[later].attribution.queue_ps
                    - cell.first_metrics[earlier].attribution.queue_ps
                ),
                "strictly_positive": (
                    _raw_queue(cell, later) > _raw_queue(cell, earlier)
                    and cell.first_metrics[later].attribution.queue_ps
                    > cell.first_metrics[earlier].attribution.queue_ps
                ),
            }
            load_checks.append(
                _scored_check(
                    f"offered-load:{offset_ps}:{earlier}-to-{later}",
                    observed,
                    {
                        "raw_queue_delta_ps": SERVICE_PS,
                        "reported_queue_delta_ps": SERVICE_PS,
                        "strictly_positive": True,
                    },
                )
            )

    def family(checks: list[dict[str, object]]) -> dict[str, object]:
        return {
            "passed": sum(check["passed"] is True for check in checks),
            "total": len(checks),
            "genuine_risk_instances": sum(
                check["genuine_risk"] is True for check in checks
            ),
            "checks": checks,
        }

    return {
        "arrival_stagger_movement": family(spacing_checks),
        "offered_load_queue_slope": family(load_checks),
    }


def _unscored_check(
    check_id: str,
    observed: object,
    expected: object,
) -> dict[str, object]:
    return {
        "check_id": check_id,
        "observed": observed,
        "expected": expected,
        "passed": observed == expected,
        "scored": False,
    }


def _evaluate_exact_decomposition(
    gated: dict[tuple[str, int], CellResult],
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for (load, offset_ps), cell in gated.items():
        for request_id in LOAD_REQUEST_IDS[load]:
            metric = cell.first_metrics[request_id]
            queue_ps = EXPECTED_QUEUE_PS[(load, offset_ps, request_id)]
            expected = {
                "raw_queue_ps": queue_ps,
                "reported_queue_ps": queue_ps,
                "raw_service_ps": SERVICE_PS,
                "reported_service_ps": SERVICE_PS,
                "raw_ttft_ps": queue_ps + SERVICE_PS,
                "reported_ttft_ps": queue_ps + SERVICE_PS,
                "completion_agrees": True,
                "tokens": list(TOKENS[request_id]),
                "finish_reason": "length",
            }
            observed = {
                "raw_queue_ps": _raw_queue(cell, request_id),
                "reported_queue_ps": metric.attribution.queue_ps,
                "raw_service_ps": _raw_service(cell, request_id),
                "reported_service_ps": _metric_service(metric),
                "raw_ttft_ps": _raw_ttft(cell, request_id),
                "reported_ttft_ps": metric.ttft_ps,
                "completion_agrees": (
                    metric.completed_at_ps
                    == cell.token_times_ps[request_id][0]
                ),
                "tokens": list(cell.tokens[request_id]),
                "finish_reason": cell.finish_reasons[request_id],
            }
            checks.append(
                _unscored_check(
                    f"decomposition:{load}:{offset_ps}:{request_id}",
                    observed,
                    expected,
                )
            )
    return checks


def _evaluate_admission_facts(
    gated: dict[tuple[str, int], CellResult],
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for (load, offset_ps), cell in gated.items():
        facts = [
            entry.fact
            for entry in cell.ledger.entries
            if isinstance(entry.fact, StageRecord)
            and entry.fact.stage is ProcessingStage.SCHEDULER
            and entry.fact.phase is StagePhase.ENTERED
        ]
        observed = {
            fact.scope.correlation.request_ids[0]: fact.timestamp_ps
            for fact in facts
        }
        expected_boundary = SERVICE_PS if offset_ps < SERVICE_PS else 2 * SERVICE_PS
        expected = {
            request_id: 0 if request_id == "r0" else expected_boundary
            for request_id in LOAD_REQUEST_IDS[load]
        }
        checks.append(
            _unscored_check(
                f"admission-facts:{load}:{offset_ps}",
                observed,
                expected,
            )
        )
    return checks


def _artifact_hashes(cell: CellResult) -> dict[str, str]:
    return {
        "step_stream": hashlib.sha256(cell.step_stream_bytes).hexdigest(),
        "outputs": hashlib.sha256(cell.output_bytes).hexdigest(),
        "replay_snapshot": hashlib.sha256(
            cell.replay_snapshot_bytes
        ).hexdigest(),
        "replay_run": hashlib.sha256(cell.replay_run_bytes).hexdigest(),
        "bookkeeping": hashlib.sha256(cell.bookkeeping_bytes).hexdigest(),
    }


def _evaluate_identity(
    direct: dict[str, CellResult],
    disabled: dict[str, CellResult],
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for load in LOAD_REQUEST_IDS:
        baseline = direct[load]
        bypass = disabled[load]
        observed = {
            "step_stream_equal": (
                baseline.step_stream_bytes == bypass.step_stream_bytes
            ),
            "outputs_equal": baseline.output_bytes == bypass.output_bytes,
            "replay_snapshot_equal": (
                baseline.replay_snapshot_bytes == bypass.replay_snapshot_bytes
            ),
            "replay_run_equal": (
                baseline.replay_run_bytes == bypass.replay_run_bytes
            ),
            "bookkeeping_equal": (
                baseline.bookkeeping_bytes == bypass.bookkeeping_bytes
            ),
            "clock_equal": baseline.final_clock_ps == bypass.final_clock_ps,
            "direct_hashes": _artifact_hashes(baseline),
            "disabled_hashes": _artifact_hashes(bypass),
        }
        expected = {
            "step_stream_equal": True,
            "outputs_equal": True,
            "replay_snapshot_equal": True,
            "replay_run_equal": True,
            "bookkeeping_equal": True,
            "clock_equal": True,
            "direct_hashes": _artifact_hashes(baseline),
            "disabled_hashes": _artifact_hashes(baseline),
        }
        checks.append(
            _unscored_check(f"all-at-once-identity:{load}", observed, expected)
        )
    return checks


def _cell_json(cell: CellResult) -> dict[str, object]:
    return {
        "admission_kind": cell.admission_kind,
        "load": cell.load,
        "arrival_offset_ps": cell.arrival_offset_ps,
        "arrivals_ps": cell.arrivals_ps,
        "tokens": {
            request_id: list(tokens)
            for request_id, tokens in cell.tokens.items()
        },
        "finish_reasons": cell.finish_reasons,
        "token_times_ps": {
            request_id: list(times)
            for request_id, times in cell.token_times_ps.items()
        },
        "first_releases_ps": cell.first_releases_ps,
        "first_metrics": {
            request_id: _metric_json(metric)
            for request_id, metric in cell.first_metrics.items()
        },
        "step_composition": [list(row) for row in cell.step_composition],
        "artifact_hashes": _artifact_hashes(cell),
        "final_clock_ps": cell.final_clock_ps,
    }


def _all_pass(records: list[dict[str, object]]) -> bool:
    return all(record["passed"] is True for record in records)


def run_study(run_dir: Path) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=False)
    trace_path = _write_trace(run_dir)
    gated = {
        (load, offset_ps): _drive_cell(
            run_dir / f"gated-{load}-{offset_ps}",
            trace_path,
            admission_kind="gated",
            load=load,
            arrival_offset_ps=offset_ps,
        )
        for load in LOAD_REQUEST_IDS
        for offset_ps in ARRIVAL_OFFSETS_PS
    }

    direct = {
        load: _drive_cell(
            run_dir / f"direct-{load}",
            trace_path,
            admission_kind="direct",
            load=load,
            arrival_offset_ps=ARRIVAL_OFFSETS_PS[0],
        )
        for load in LOAD_REQUEST_IDS
    }
    disabled = {
        load: _drive_cell(
            run_dir / f"disabled-{load}",
            trace_path,
            admission_kind="disabled",
            load=load,
            arrival_offset_ps=ARRIVAL_OFFSETS_PS[0],
        )
        for load in LOAD_REQUEST_IDS
    }

    # Entailment discipline: scored raw relations are evaluated before the
    # later exact decomposition and identity oracles.
    scored = _evaluate_scored(gated)
    decomposition = _evaluate_exact_decomposition(gated)
    admission_facts = _evaluate_admission_facts(gated)
    identity = _evaluate_identity(direct, disabled)
    fixture_guard = _unscored_check(
        "existing-no-replay-byte-lock",
        hashlib.sha256(NO_REPLAY_FIXTURE.read_bytes()).hexdigest(),
        NO_REPLAY_SHA256,
    )
    summary = {
        "schema": "simllm-arrival-admission-study-v1",
        "run_configurations": {
            "gated_cells": len(gated),
            "identity_cells": len(direct) + len(disabled),
            "service_ps": SERVICE_PS,
            "arrival_offsets_ps": list(ARRIVAL_OFFSETS_PS),
            "loads": {
                load: list(request_ids)
                for load, request_ids in LOAD_REQUEST_IDS.items()
            },
            "vllm_version": importlib.metadata.version("vllm"),
            "source_hashes": SOURCE_HASHES,
            "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        },
        "scored_families": scored,
        "fatal_unscored": {
            "exact_decomposition": decomposition,
            "admission_facts": admission_facts,
            "identity": identity,
            "fixture_guard": fixture_guard,
        },
        "raw_cells": [
            _cell_json(gated[key])
            for key in sorted(gated)
        ],
        "identity_cells": [
            _cell_json(direct[load])
            for load in LOAD_REQUEST_IDS
        ]
        + [
            _cell_json(disabled[load])
            for load in LOAD_REQUEST_IDS
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (run_dir / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "load",
                "arrival_offset_ps",
                "request_id",
                "queue_ps",
                "service_ps",
                "ttft_ps",
                "token_count",
                "finish_reason",
            )
        )
        for key in sorted(gated):
            cell = gated[key]
            for request_id in LOAD_REQUEST_IDS[cell.load]:
                writer.writerow(
                    (
                        cell.load,
                        cell.arrival_offset_ps,
                        request_id,
                        cell.first_metrics[request_id].attribution.queue_ps,
                        _metric_service(cell.first_metrics[request_id]),
                        cell.first_metrics[request_id].ttft_ps,
                        len(cell.tokens[request_id]),
                        cell.finish_reasons[request_id],
                    )
                )

    scored_pass = all(
        family["passed"] == family["total"]
        for family in scored.values()
    )
    fatal_pass = (
        _all_pass(decomposition)
        and _all_pass(admission_facts)
        and _all_pass(identity)
        and fixture_guard["passed"] is True
    )
    if not scored_pass:
        raise AssertionError("one or more scored arrival relations failed")
    if not fatal_pass:
        raise AssertionError("one or more fatal unscored guards failed")
    return summary


def check_inputs() -> None:
    if importlib.metadata.version("vllm") != "0.26.0":
        raise SystemExit("arrival study requires vLLM 0.26.0")
    if hashlib.sha256(GRANITE_FIXTURE.read_bytes()).hexdigest() != GRANITE_SHA256:
        raise SystemExit("tracked Granite fixture changed")
    if hashlib.sha256(NO_REPLAY_FIXTURE.read_bytes()).hexdigest() != NO_REPLAY_SHA256:
        raise SystemExit("tracked no-replay byte fixture changed")
    package = _vllm_package_root()
    for relative, expected in SOURCE_HASHES.items():
        observed = hashlib.sha256((package / relative).read_bytes()).hexdigest()
        if observed != expected:
            raise SystemExit(f"audited vLLM source changed: {relative}")
    if not _model_path().is_dir():
        raise SystemExit("the configured Hugging Face cache lacks the frozen model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    check_inputs()
    if args.check_only:
        return
    summary = run_study(args.run_dir)
    scored = summary["scored_families"]
    print(
        "arrival admission study passed "
        f"{scored['arrival_stagger_movement']['passed']}/"
        f"{scored['arrival_stagger_movement']['total']} spacing and "
        f"{scored['offered_load_queue_slope']['passed']}/"
        f"{scored['offered_load_queue_slope']['total']} load relations"
    )


if __name__ == "__main__":
    main()
