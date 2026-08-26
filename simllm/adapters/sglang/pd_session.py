"""Process-isolated SGLang prefill/decode session over simulated GPUs.

Pinned SGLang keeps model-parallel groups in process-global state. One child
process therefore retains each stock scheduler. The parent owns the session
clock, request lifecycle and CORE-51 handoff. A child receives the parent's
current timestamp, executes exactly one scheduler-selected batch and returns
immutable observations plus the finishing timestamp.

SGLang's native disaggregation pair is not reachable with the bufferless
simulated worker at the pinned commit. Prefill rejects the fake transfer
backend, while decode requires registered KV tensor metadata. This module uses
the driver-level fallback explicitly allowed by SGL-33 and never reports a
native connector success.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from simllm.compute import (
    GPU_ENVELOPES,
    ComputeProvider,
    GpuSpec,
    HostInitiationModel,
    ModelDims,
    RooflineProvider,
)
from simllm.core import (
    DisaggregatedRequestTimeline,
    KvHandoffEvent,
    KvHandoffGeometry,
    KvHandoffPolicy,
    ServingPoolRole,
    StepRecord,
    StepResult,
    VirtualClock,
)
from simllm.placement import (
    PlacementManifest,
    SglangPoolArrangement,
    sglang_disaggregated_manifests,
)

SGLANG_PD_JOIN_SCHEMA = "simllm-sglang-driver-pd-join-v1"
SGLANG_PD_JOIN_MODE = "driver-level"
SGLANG_NATIVE_SEAM_STATUS = "unreachable-bufferless-worker-bfeae4e"
SGLANG_SESSION_AUTHORITY = "simllm-parent-virtual-clock-v1"
SGLANG_VERSION = "0.5.19.dev345+gbfeae4e79"
DEPLOYMENT_CURVE_SCHEMA = "simllm-deployment-curve-v1"
DEPLOYMENT_CURVE_POINT_SCHEMA = "simllm-deployment-curve-point-v1"
PS_PER_SECOND = 1_000_000_000_000


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _fraction_json(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


@dataclass(frozen=True)
class SglangPdSessionConfig:
    """Construction, structural placement and pricing inputs for one session."""

    model_path: Path
    workdir: Path
    dims: ModelDims
    handoff_geometry: KvHandoffGeometry
    handoff_policy: KvHandoffPolicy
    prefill_arrangement: SglangPoolArrangement
    decode_arrangement: SglangPoolArrangement
    prefill_engines: int = 1
    decode_engines: int = 1
    simulated_gpus_per_engine: int = 8
    context_length: int = 64
    max_total_tokens: int = 512
    max_running_requests: int = 8
    token_id: int = 512
    random_seed: int = 0
    provider: ComputeProvider = field(
        default_factory=lambda: RooflineProvider(efficiency=0.7)
    )
    gpu: GpuSpec = GPU_ENVELOPES["b100"]
    host_model: HostInitiationModel = field(default_factory=HostInitiationModel.ideal)
    framework_version: str = SGLANG_VERSION
    rpc_timeout_s: int = 300

    def __post_init__(self) -> None:
        for name in ("model_path", "workdir"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a Path")
        if not isinstance(self.dims, ModelDims):
            raise TypeError("dims must be ModelDims")
        if not isinstance(self.handoff_geometry, KvHandoffGeometry):
            raise TypeError("handoff_geometry must be KvHandoffGeometry")
        if not isinstance(self.handoff_policy, KvHandoffPolicy):
            raise TypeError("handoff_policy must implement KvHandoffPolicy")
        if not isinstance(self.prefill_arrangement, SglangPoolArrangement):
            raise TypeError("prefill_arrangement must be SglangPoolArrangement")
        if not isinstance(self.decode_arrangement, SglangPoolArrangement):
            raise TypeError("decode_arrangement must be SglangPoolArrangement")
        for name in (
            "prefill_engines",
            "decode_engines",
            "simulated_gpus_per_engine",
            "context_length",
            "max_total_tokens",
            "max_running_requests",
            "token_id",
            "rpc_timeout_s",
        ):
            _positive_int(name, getattr(self, name))
        _nonnegative_int("random_seed", self.random_seed)
        if not isinstance(self.provider, ComputeProvider):
            raise TypeError("provider must implement ComputeProvider")
        if not isinstance(self.gpu, GpuSpec):
            raise TypeError("gpu must be GpuSpec")
        if not isinstance(self.host_model, HostInitiationModel):
            raise TypeError("host_model must be HostInitiationModel")
        if not isinstance(self.framework_version, str) or not self.framework_version:
            raise ValueError("framework_version must be a nonblank string")
        self.host_model.validate_device(self.gpu)
        role_widths = {
            "prefill": self.prefill_engines * self.simulated_gpus_per_engine,
            "decode": self.decode_engines * self.simulated_gpus_per_engine,
        }
        for role, arrangement in (
            ("prefill", self.prefill_arrangement),
            ("decode", self.decode_arrangement),
        ):
            width = role_widths[role]
            for field_name in (
                "attention_data_parallel_size",
                "dense_data_parallel_size",
                "expert_parallel_size",
            ):
                size = getattr(arrangement, field_name)
                if width % size:
                    raise ValueError(
                        f"{role} {field_name} {size} does not divide simulated "
                        f"role width {width}"
                    )


@dataclass(frozen=True)
class SglangPdRequest:
    """One stable request admitted to the concurrent session."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    decode_output_tokens: int
    admitted_at_ps: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a nonblank string")
        prompt = tuple(self.prompt_token_ids)
        if not prompt:
            raise ValueError("prompt_token_ids must not be empty")
        if any(
            isinstance(token, bool) or type(token) is not int or token < 0
            for token in prompt
        ):
            raise ValueError("prompt_token_ids must be nonnegative integers")
        object.__setattr__(self, "prompt_token_ids", prompt)
        _positive_int("decode_output_tokens", self.decode_output_tokens)
        _nonnegative_int("admitted_at_ps", self.admitted_at_ps)


@dataclass(frozen=True)
class SglangPdRequestResult:
    """Stock-scheduler output joined to one framework-neutral timeline."""

    timeline: DisaggregatedRequestTimeline
    prefill_engine_id: str
    decode_engine_id: str
    prefill_internal_request_id: str
    decode_internal_request_id: str
    bootstrap_token_id: int
    decode_token_ids: tuple[int, ...]
    join_metadata: dict[str, Any]
    prefill_records: tuple[StepRecord, ...]
    decode_records: tuple[StepRecord, ...]
    prefill_results: tuple[StepResult, ...]
    decode_results: tuple[StepResult, ...]

    def to_json(self) -> dict[str, object]:
        value = self.timeline.to_json()
        value.update(
            {
                "prefill_engine_id": self.prefill_engine_id,
                "decode_engine_id": self.decode_engine_id,
                "prefill_internal_request_id": self.prefill_internal_request_id,
                "decode_internal_request_id": self.decode_internal_request_id,
                "bootstrap_token_id": self.bootstrap_token_id,
                "decode_token_ids": list(self.decode_token_ids),
                "join_metadata": dict(self.join_metadata),
                "prefill_step_count": len(self.prefill_records),
                "decode_step_count": len(self.decode_records),
            }
        )
        return value


@dataclass(frozen=True)
class SglangPdCurvePoint:
    """One exact throughput-versus-delay point from terminal requests."""

    offered_load_requests_per_second: Fraction
    aggregated_output_throughput_tokens_per_second: Fraction
    per_token_request_delay_ps: Fraction
    request_count: int
    output_token_count: int
    first_admitted_at_ps: int
    last_completed_at_ps: int

    def __post_init__(self) -> None:
        for name in (
            "offered_load_requests_per_second",
            "aggregated_output_throughput_tokens_per_second",
            "per_token_request_delay_ps",
        ):
            value = getattr(self, name)
            if not isinstance(value, Fraction):
                raise TypeError(f"{name} must be a Fraction")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        _positive_int("request_count", self.request_count)
        _positive_int("output_token_count", self.output_token_count)
        _nonnegative_int("first_admitted_at_ps", self.first_admitted_at_ps)
        _positive_int("last_completed_at_ps", self.last_completed_at_ps)
        if self.last_completed_at_ps <= self.first_admitted_at_ps:
            raise ValueError("curve observation interval must be positive")

    @classmethod
    def from_requests(
        cls,
        offered_load_requests_per_second: Fraction,
        requests: Sequence[SglangPdRequestResult],
    ) -> SglangPdCurvePoint:
        rows = tuple(requests)
        if not rows:
            raise ValueError("requests must not be empty")
        if len({row.timeline.request_id for row in rows}) != len(rows):
            raise ValueError("curve requests must have unique stable identities")
        first = min(row.timeline.admitted_at_ps for row in rows)
        last = max(row.timeline.decode_token_completed_at_ps[-1] for row in rows)
        output_tokens = sum(len(row.decode_token_ids) for row in rows)
        delay = sum(
            (
                Fraction(
                    row.timeline.decode_token_completed_at_ps[-1]
                    - row.timeline.admitted_at_ps,
                    len(row.decode_token_ids),
                )
                for row in rows
            ),
            start=Fraction(),
        ) / len(rows)
        return cls(
            offered_load_requests_per_second=offered_load_requests_per_second,
            aggregated_output_throughput_tokens_per_second=Fraction(
                output_tokens * PS_PER_SECOND,
                last - first,
            ),
            per_token_request_delay_ps=delay,
            request_count=len(rows),
            output_token_count=output_tokens,
            first_admitted_at_ps=first,
            last_completed_at_ps=last,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema": DEPLOYMENT_CURVE_POINT_SCHEMA,
            "offered_load_requests_per_second": _fraction_json(
                self.offered_load_requests_per_second
            ),
            "aggregated_output_throughput_tokens_per_second": _fraction_json(
                self.aggregated_output_throughput_tokens_per_second
            ),
            "per_token_request_delay_ps": _fraction_json(
                self.per_token_request_delay_ps
            ),
            "request_count": self.request_count,
            "output_token_count": self.output_token_count,
            "first_admitted_at_ps": self.first_admitted_at_ps,
            "last_completed_at_ps": self.last_completed_at_ps,
        }


@dataclass(frozen=True)
class SglangPdCurveRecord:
    """Machine-readable curve for one SGLang deployment configuration."""

    configuration_id: str
    prefill_engines: int
    decode_engines: int
    prompt_tokens: int
    points: tuple[SglangPdCurvePoint, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.configuration_id, str) or not self.configuration_id:
            raise ValueError("configuration_id must be a nonblank string")
        for name in ("prefill_engines", "decode_engines", "prompt_tokens"):
            _positive_int(name, getattr(self, name))
        points = tuple(self.points)
        if not points or any(not isinstance(row, SglangPdCurvePoint) for row in points):
            raise TypeError("points must contain SglangPdCurvePoint values")
        loads = [point.offered_load_requests_per_second for point in points]
        if loads != sorted(loads) or len(loads) != len(set(loads)):
            raise ValueError("curve offered loads must be unique and increasing")
        object.__setattr__(self, "points", points)

    def to_json(self) -> dict[str, object]:
        return {
            "schema": DEPLOYMENT_CURVE_SCHEMA,
            "configuration_id": self.configuration_id,
            "prefill_engines": self.prefill_engines,
            "decode_engines": self.decode_engines,
            "prompt_tokens": self.prompt_tokens,
            "orientation": {
                "x": "aggregated-output-throughput-rightward",
                "y": "inverse-per-token-request-delay-upward",
            },
            "points": [point.to_json() for point in self.points],
        }


@dataclass(frozen=True)
class SglangPdConcurrentResult:
    """Completed requests plus batches authored by the stock schedulers."""

    requests: tuple[SglangPdRequestResult, ...]
    prefill_batches: tuple[tuple[str, ...], ...]
    decode_batches: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        requests = tuple(self.requests)
        if not requests:
            raise ValueError("requests must not be empty")
        request_ids = tuple(row.timeline.request_id for row in requests)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("concurrent results must have unique request identities")
        for name in ("prefill_batches", "decode_batches"):
            batches = tuple(tuple(batch) for batch in getattr(self, name))
            if not batches or any(not batch for batch in batches):
                raise ValueError(f"{name} must contain nonempty scheduler batches")
            unknown = {item for batch in batches for item in batch} - set(request_ids)
            if unknown:
                raise ValueError(f"{name} contains unknown requests: {sorted(unknown)}")
            object.__setattr__(self, name, batches)
        object.__setattr__(self, "requests", requests)

    @property
    def maximum_prefill_batch_size(self) -> int:
        return max(map(len, self.prefill_batches))

    @property
    def maximum_decode_batch_size(self) -> int:
        return max(map(len, self.decode_batches))

    def curve_point(
        self, offered_load_requests_per_second: Fraction
    ) -> SglangPdCurvePoint:
        return SglangPdCurvePoint.from_requests(
            offered_load_requests_per_second,
            self.requests,
        )


@dataclass(frozen=True)
class _EngineLaunchConfig:
    role: ServingPoolRole
    ordinal: int
    engine_id: str
    model_path: Path
    engine_workdir: Path
    ranks: tuple[int, ...]
    tensor_parallel_ranks: tuple[int, ...]
    attention_data_parallel_ranks: tuple[int, ...]
    dense_data_parallel_ranks: tuple[int, ...]
    expert_parallel_ranks: tuple[int, ...]
    placement: PlacementManifest
    dims: ModelDims
    provider: ComputeProvider
    gpu: GpuSpec
    host_model: HostInitiationModel
    context_length: int
    max_total_tokens: int
    max_running_requests: int
    token_id: int
    random_seed: int


def _engine_process_main(connection: Any, config: _EngineLaunchConfig) -> None:
    """Construct and serve one SGLang scheduler in an isolated process."""

    try:
        from simllm.adapters.sglang import (
            SglangSchedulerPump,
            SimWorkerConfig,
            build_in_process_scheduler,
            configure,
            install,
            latest_worker,
            reset_configuration,
            tokenized_generate_request,
        )
        from simllm.backends import HtsimStepSink, HtsimStepSinkConfig

        install()
        reset_configuration()
        sink = HtsimStepSink(
            HtsimStepSinkConfig(
                profile="rnic-nn-fluid",
                tp_ranks=config.tensor_parallel_ranks,
                ep_ranks=config.expert_parallel_ranks,
                dims=config.dims,
                workdir=config.engine_workdir / "steps",
                provider=config.provider,
                gpu=config.gpu,
                host_model=config.host_model,
                placement_manifest=config.placement,
            )
        )
        configure(
            step_sink=sink,
            compute_provider=config.provider,
            gpu=config.gpu,
            host_model=config.host_model,
            config=SimWorkerConfig(
                mode="virtual",
                efficiency=getattr(config.provider, "efficiency", 0.7),
                token_id=config.token_id,
                step_records_path=(
                    config.engine_workdir / "step-records.jsonl"
                ).as_posix(),
            ),
        )
        started = time.perf_counter()
        scheduler = build_in_process_scheduler(
            model_path=config.model_path.as_posix(),
            device="cpu",
            dtype="float32",
            tp_size=1,
            page_size=1,
            context_length=config.context_length,
            max_total_tokens=config.max_total_tokens,
            max_running_requests=config.max_running_requests,
            chunked_prefill_size=-1,
            random_seed=config.random_seed,
        )
        construction_seconds = time.perf_counter() - started
        worker = latest_worker()
        if worker is None:
            raise RuntimeError("SGLang did not construct SimTpModelWorker")
        if worker.step_records or worker.step_results:
            raise RuntimeError("new pool worker already contains simulated steps")
        worker.clock = VirtualClock()
        pump = SglangSchedulerPump(scheduler)
        connection.send(
            {
                "ok": True,
                "construction_seconds": construction_seconds,
                "worker_type": type(worker).__name__,
                "scheduler_type": type(scheduler).__name__,
            }
        )
        while True:
            command = connection.recv()
            operation = command.get("operation")
            if operation == "close":
                connection.send({"ok": True})
                break
            if operation == "submit":
                request = tokenized_generate_request(
                    request_id=command["request_id"],
                    input_token_ids=command["input_token_ids"],
                    max_new_tokens=command["max_new_tokens"],
                    tokenizer=getattr(scheduler, "tokenizer", None),
                )
                pump.submit([request])
                connection.send({"ok": True})
                continue
            if operation != "step":
                raise ValueError(f"unknown engine operation {operation!r}")
            worker.clock.advance_to(command["now_ps"])
            before_records = len(worker.step_records)
            before_results = len(worker.step_results)
            outcome = pump.step()
            record = None
            result = None
            if outcome.ran_batch:
                if len(worker.step_records) != before_records + 1:
                    raise RuntimeError("one SGLang batch did not emit one StepRecord")
                if len(worker.step_results) != before_results + 1:
                    raise RuntimeError("one SGLang batch did not emit one StepResult")
                record = worker.step_records[-1]
                result = worker.step_results[-1]
                if result.step_index != record.step_index:
                    raise RuntimeError("SGLang StepRecord and StepResult disagree")
                if result.completed_at_ps != worker.clock.now_ps:
                    raise RuntimeError("SGLang StepResult disagrees with its child clock")
            elif len(worker.step_records) != before_records:
                raise RuntimeError("an idle SGLang step emitted a StepRecord")
            elif len(worker.step_results) != before_results:
                raise RuntimeError("an idle SGLang step emitted a StepResult")
            connection.send(
                {
                    "ok": True,
                    "completed_at_ps": worker.clock.now_ps,
                    "record": record,
                    "result": result,
                    "completions": outcome.completions,
                    "token_id": worker.token_id,
                    "record_count": len(worker.step_records),
                }
            )
    except Exception as exc:  # noqa: BLE001 - propagate child failure over RPC
        import traceback

        with contextlib.suppress(Exception):
            connection.send(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    finally:
        with contextlib.suppress(Exception):
            from simllm.adapters.sglang import reset_configuration

            reset_configuration()
        with contextlib.suppress(Exception):
            from sglang.srt.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        connection.close()


class _ProcessPoolEngine:
    """Parent-side controller for one process-isolated stock scheduler."""

    def __init__(self, config: _EngineLaunchConfig, *, timeout_s: int) -> None:
        self.config = config
        self.role = config.role
        self.ordinal = config.ordinal
        self.engine_id = config.engine_id
        self.ranks = config.ranks
        self.tensor_parallel_ranks = config.tensor_parallel_ranks
        self.attention_data_parallel_ranks = (
            config.attention_data_parallel_ranks
        )
        self.dense_data_parallel_ranks = config.dense_data_parallel_ranks
        self.expert_parallel_ranks = config.expert_parallel_ranks
        self.records: list[StepRecord] = []
        self.results: list[StepResult] = []
        self._unfinished: set[str] = set()
        self._timeout_s = timeout_s
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(
            target=_engine_process_main,
            args=(child, config),
            name=config.engine_id,
        )
        self._process.start()
        child.close()
        process_id = self._process.pid
        if type(process_id) is not int or process_id < 1:
            self.close()
            raise RuntimeError(f"{self.engine_id} did not acquire a process identity")
        self.process_id = process_id
        try:
            ready = self._receive()
        except Exception:
            self.close()
            raise
        self.construction_seconds = float(ready["construction_seconds"])
        self.worker_type = str(ready["worker_type"])
        self.scheduler_type = str(ready["scheduler_type"])

    @property
    def simulated_worker_count(self) -> int:
        return len(self.ranks)

    @property
    def has_unfinished_requests(self) -> bool:
        return bool(self._unfinished)

    def _receive(self) -> dict[str, Any]:
        if not self._connection.poll(self._timeout_s):
            raise TimeoutError(f"{self.engine_id} did not answer within the RPC timeout")
        try:
            response = self._connection.recv()
        except EOFError as exc:
            raise RuntimeError(f"{self.engine_id} exited before replying") from exc
        if response.get("ok") is not True:
            detail = response.get("traceback") or response.get("message")
            raise RuntimeError(f"{self.engine_id} failed:\n{detail}")
        return response

    def _rpc(self, command: dict[str, Any]) -> dict[str, Any]:
        if not self._process.is_alive():
            raise RuntimeError(f"{self.engine_id} process is not alive")
        self._connection.send(command)
        return self._receive()

    def submit(
        self,
        *,
        request_id: str,
        input_token_ids: Sequence[int],
        max_new_tokens: int,
    ) -> None:
        if request_id in self._unfinished:
            raise ValueError(f"request {request_id!r} is already active")
        self._rpc(
            {
                "operation": "submit",
                "request_id": request_id,
                "input_token_ids": tuple(input_token_ids),
                "max_new_tokens": max_new_tokens,
            }
        )
        self._unfinished.add(request_id)

    def step(self, now_ps: int) -> dict[str, Any]:
        response = self._rpc({"operation": "step", "now_ps": now_ps})
        record = response["record"]
        result = response["result"]
        if record is not None:
            if result is None:
                raise RuntimeError(f"{self.engine_id} omitted its StepResult")
            self.records.append(record)
            self.results.append(result)
        elif result is not None:
            raise RuntimeError(f"{self.engine_id} returned an orphan StepResult")
        for completion in response["completions"]:
            if completion.request_id not in self._unfinished:
                raise RuntimeError(
                    f"{self.engine_id} completed unknown request "
                    f"{completion.request_id!r}"
                )
            self._unfinished.remove(completion.request_id)
        return response

    def close(self) -> None:
        if not hasattr(self, "_process"):
            return
        try:
            if self._process.is_alive():
                self._connection.send({"operation": "close"})
                self._receive()
        except (BrokenPipeError, EOFError, RuntimeError, TimeoutError):
            pass
        finally:
            self._connection.close()
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=10)


@dataclass
class _RequestState:
    request: SglangPdRequest
    prefill: _ProcessPoolEngine
    decode: _ProcessPoolEngine
    prefill_internal_id: str | None = None
    decode_internal_id: str | None = None
    prefill_record_start: int | None = None
    prefill_record_stop: int | None = None
    decode_record_start: int | None = None
    decode_record_stop: int | None = None
    prefill_eligible_at_ps: int | None = None
    prefill_completed_at_ps: int | None = None
    handoff: KvHandoffEvent | None = None
    bootstrap_token_id: int | None = None
    decode_token_ids: list[int] = field(default_factory=list)
    decode_token_completed_at_ps: list[int] = field(default_factory=list)
    join_metadata: dict[str, Any] | None = None


class SglangDisaggregatedSession:
    """Drive separate stock SGLang scheduler processes under one parent clock."""

    def __init__(
        self,
        config: SglangPdSessionConfig,
        *,
        clock: VirtualClock | None = None,
    ) -> None:
        if not isinstance(config, SglangPdSessionConfig):
            raise TypeError("config must be SglangPdSessionConfig")
        if clock is not None and not isinstance(clock, VirtualClock):
            raise TypeError("clock must be a VirtualClock")
        self.config = config
        self.clock = VirtualClock() if clock is None else clock
        self.manifests = sglang_disaggregated_manifests(
            prefill_nodes=config.prefill_engines,
            decode_nodes=config.decode_engines,
            gpus_per_node=config.simulated_gpus_per_engine,
            prefill_arrangement=config.prefill_arrangement,
            decode_arrangement=config.decode_arrangement,
            framework_version=config.framework_version,
        )
        self.prefill_engines: list[_ProcessPoolEngine] = []
        self.decode_engines: list[_ProcessPoolEngine] = []
        self.handoffs: list[KvHandoffEvent] = []
        self._request_ids: set[str] = set()
        self._next_prefill = 0
        self._next_decode = 0
        self._closed = False
        self._build_pools()

    def _engine_ranks(
        self, role: ServingPoolRole, ordinal: int
    ) -> tuple[int, ...]:
        node_id = f"{role.value}-node-{ordinal}"
        ranks = tuple(
            rank.global_rank
            for rank in self.manifests.placement.ranks
            if rank.hostname == node_id and rank.pool_role == role.value
        )
        if len(ranks) != self.config.simulated_gpus_per_engine:
            raise RuntimeError(f"placement did not produce {node_id}'s rank set")
        return ranks

    def _engine_group_ranks(
        self,
        role: ServingPoolRole,
        ordinal: int,
        group: str,
    ) -> tuple[int, ...]:
        engine_ranks = self._engine_ranks(role, ordinal)
        members = self.manifests.placement.group_ranks(engine_ranks[0], group)
        if not members:
            raise RuntimeError(f"placement did not produce a {group} group")
        return tuple(members)

    def _build_pools(self) -> None:
        try:
            for role, count in (
                (ServingPoolRole.PREFILL, self.config.prefill_engines),
                (ServingPoolRole.DECODE, self.config.decode_engines),
            ):
                for ordinal in range(count):
                    engine_id = f"simllm-sglang-{role.value}-{ordinal}"
                    launch = _EngineLaunchConfig(
                        role=role,
                        ordinal=ordinal,
                        engine_id=engine_id,
                        model_path=self.config.model_path,
                        engine_workdir=self.config.workdir / engine_id,
                        ranks=self._engine_ranks(role, ordinal),
                        tensor_parallel_ranks=self._engine_group_ranks(
                            role,
                            ordinal,
                            "tp",
                        ),
                        attention_data_parallel_ranks=self._engine_group_ranks(
                            role,
                            ordinal,
                            "attn_dp",
                        ),
                        dense_data_parallel_ranks=self._engine_group_ranks(
                            role,
                            ordinal,
                            "dense_dp",
                        ),
                        expert_parallel_ranks=self._engine_group_ranks(
                            role,
                            ordinal,
                            "ep",
                        ),
                        placement=self.manifests.placement,
                        dims=self.config.dims,
                        provider=self.config.provider,
                        gpu=self.config.gpu,
                        host_model=self.config.host_model,
                        context_length=self.config.context_length,
                        max_total_tokens=self.config.max_total_tokens,
                        max_running_requests=self.config.max_running_requests,
                        token_id=self.config.token_id,
                        random_seed=self.config.random_seed + ordinal,
                    )
                    engine = _ProcessPoolEngine(
                        launch,
                        timeout_s=self.config.rpc_timeout_s,
                    )
                    target = (
                        self.prefill_engines
                        if role is ServingPoolRole.PREFILL
                        else self.decode_engines
                    )
                    target.append(engine)
        except Exception:
            self.shutdown()
            raise

    @staticmethod
    def _first_release(records: Sequence[StepRecord], request_id: str) -> int:
        for record in records:
            if any(row.request_id == request_id for row in record.scheduled):
                return record.virtual_time_ps
        raise RuntimeError(f"scheduler never released request {request_id!r}")

    @staticmethod
    def _stable_batch(
        engine: _ProcessPoolEngine,
        states: Sequence[_RequestState],
        record: StepRecord,
    ) -> tuple[str, ...]:
        if engine.role is ServingPoolRole.PREFILL:
            by_internal = {
                state.prefill_internal_id: state.request.request_id
                for state in states
                if state.prefill is engine and state.prefill_internal_id is not None
            }
        else:
            by_internal = {
                state.decode_internal_id: state.request.request_id
                for state in states
                if state.decode is engine and state.decode_internal_id is not None
            }
        batch = []
        for scheduled in record.scheduled:
            stable = by_internal.get(scheduled.request_id)
            if stable is None:
                raise RuntimeError(
                    f"{engine.engine_id} scheduled an unknown pool-local identity"
                )
            batch.append(stable)
        if not batch or len(batch) != len(set(batch)):
            raise RuntimeError("stock scheduler emitted an empty or duplicated batch")
        return tuple(batch)

    def _admit_prefill(self, state: _RequestState) -> None:
        request = state.request
        internal_id = f"{state.prefill.engine_id}:prefill:{request.request_id}"
        state.prefill_record_start = len(state.prefill.records)
        state.prefill.submit(
            request_id=internal_id,
            input_token_ids=request.prompt_token_ids,
            max_new_tokens=1,
        )
        state.prefill_internal_id = internal_id

    def _finish_prefill(
        self,
        state: _RequestState,
        *,
        output_token_count: int,
        token_id: int,
        policy: KvHandoffPolicy,
    ) -> None:
        if output_token_count != 1:
            raise RuntimeError("prefill scheduler did not emit one bootstrap token")
        request = state.request
        state.prefill_completed_at_ps = self.clock.now_ps
        state.prefill_record_stop = len(state.prefill.records)
        assert state.prefill_record_start is not None
        assert state.prefill_internal_id is not None
        state.prefill_eligible_at_ps = self._first_release(
            state.prefill.records[
                state.prefill_record_start : state.prefill_record_stop
            ],
            state.prefill_internal_id,
        )
        state.bootstrap_token_id = token_id
        state.handoff = policy.schedule(
            submitted_at_ps=self.clock.now_ps,
            request_id=request.request_id,
            kv_bytes=self.config.handoff_geometry.bytes_for_prompt(
                len(request.prompt_token_ids)
            ),
        )
        self.handoffs.append(state.handoff)
        if sum(
            event.request_id == request.request_id for event in self.handoffs
        ) != 1:
            raise RuntimeError("request emitted other than one KV handoff")

    def _admit_decode(self, state: _RequestState) -> None:
        if state.handoff is None or state.bootstrap_token_id is None:
            raise RuntimeError("decode admission requires a completed handoff")
        request = state.request
        internal_id = f"{state.decode.engine_id}:decode:{request.request_id}"
        state.decode_record_start = len(state.decode.records)
        state.decode.submit(
            request_id=internal_id,
            input_token_ids=(state.bootstrap_token_id,),
            max_new_tokens=request.decode_output_tokens,
        )
        state.decode_internal_id = internal_id
        state.join_metadata = {
            "schema": SGLANG_PD_JOIN_SCHEMA,
            "join_mode": SGLANG_PD_JOIN_MODE,
            "native_seam_status": SGLANG_NATIVE_SEAM_STATUS,
            "session_request_id": request.request_id,
            "remote_num_tokens": len(request.prompt_token_ids),
            "bootstrap_token_id": state.bootstrap_token_id,
            "worker_tensor_transfer": False,
            "timing_authority": state.handoff.authority,
            "pricing_arm": state.handoff.pricing_arm,
            "session_clock_authority": SGLANG_SESSION_AUTHORITY,
            "prefill_ranks": list(state.prefill.ranks),
            "decode_ranks": list(state.decode.ranks),
            "prefill_tensor_parallel_ranks": list(
                state.prefill.tensor_parallel_ranks
            ),
            "decode_tensor_parallel_ranks": list(
                state.decode.tensor_parallel_ranks
            ),
            "prefill_process_id": state.prefill.process_id,
            "decode_process_id": state.decode.process_id,
            "prefill_scheduler_type": state.prefill.scheduler_type,
            "decode_scheduler_type": state.decode.scheduler_type,
            "prefill_worker_type": state.prefill.worker_type,
            "decode_worker_type": state.decode.worker_type,
            "prefill_attention_data_parallel_ranks": list(
                state.prefill.attention_data_parallel_ranks
            ),
            "decode_attention_data_parallel_ranks": list(
                state.decode.attention_data_parallel_ranks
            ),
            "prefill_dense_data_parallel_ranks": list(
                state.prefill.dense_data_parallel_ranks
            ),
            "decode_dense_data_parallel_ranks": list(
                state.decode.dense_data_parallel_ranks
            ),
            "prefill_expert_parallel_ranks": list(
                state.prefill.expert_parallel_ranks
            ),
            "decode_expert_parallel_ranks": list(
                state.decode.expert_parallel_ranks
            ),
            "prefill_arrangement": self.config.prefill_arrangement.to_json(),
            "decode_arrangement": self.config.decode_arrangement.to_json(),
        }

    def _step_engine(
        self,
        engine: _ProcessPoolEngine,
        states: Sequence[_RequestState],
        policy: KvHandoffPolicy,
    ) -> tuple[str, ...] | None:
        response = engine.step(self.clock.now_ps)
        completed_at_ps = response["completed_at_ps"]
        if completed_at_ps < self.clock.now_ps:
            raise RuntimeError("child engine clock moved behind the parent authority")
        self.clock.advance_to(completed_at_ps)
        record = response["record"]
        batch = None if record is None else self._stable_batch(engine, states, record)
        state_by_internal: dict[str, _RequestState] = {}
        for state in states:
            internal = (
                state.prefill_internal_id
                if engine.role is ServingPoolRole.PREFILL
                else state.decode_internal_id
            )
            if internal is not None:
                state_by_internal[internal] = state
        if record is not None:
            if record.sampled_request_ids is None:
                raise RuntimeError("SGLang session requires sampled request identity")
            if engine.role is ServingPoolRole.DECODE:
                for internal_id in record.sampled_request_ids:
                    state = state_by_internal.get(internal_id)
                    if state is None:
                        raise RuntimeError("decode sampled an unknown request identity")
                    state.decode_token_ids.append(response["token_id"])
                    state.decode_token_completed_at_ps.append(self.clock.now_ps)
        for completion in response["completions"]:
            state = state_by_internal.get(completion.request_id)
            if state is None:
                raise RuntimeError("pool completed a foreign request identity")
            if engine.role is ServingPoolRole.PREFILL:
                self._finish_prefill(
                    state,
                    output_token_count=completion.output_token_count,
                    token_id=response["token_id"],
                    policy=policy,
                )
            else:
                if completion.output_token_count != state.request.decode_output_tokens:
                    raise RuntimeError("decode scheduler emitted the wrong token count")
                state.decode_record_stop = len(engine.records)
        return batch

    @staticmethod
    def _future_ready_times(
        states: Sequence[_RequestState], now_ps: int
    ) -> tuple[int, ...]:
        values = []
        for state in states:
            if state.prefill_internal_id is None:
                values.append(state.request.admitted_at_ps)
            elif state.handoff is not None and state.decode_internal_id is None:
                values.append(state.handoff.completed_at_ps)
        return tuple(value for value in values if value > now_ps)

    def run_requests(
        self,
        requests: Sequence[SglangPdRequest],
        *,
        handoff_policy: KvHandoffPolicy | None = None,
    ) -> SglangPdConcurrentResult:
        """Run concurrent requests while stock schedulers alone form batches."""

        if self._closed:
            raise RuntimeError("session is closed")
        rows = tuple(requests)
        if not rows:
            raise ValueError("requests must not be empty")
        if any(not isinstance(row, SglangPdRequest) for row in rows):
            raise TypeError("requests must contain SglangPdRequest values")
        request_ids = tuple(row.request_id for row in rows)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("requests must have unique stable identities")
        duplicate = sorted(set(request_ids) & self._request_ids)
        if duplicate:
            raise ValueError(f"duplicate session request IDs: {duplicate}")
        if min(row.admitted_at_ps for row in rows) < self.clock.now_ps:
            raise ValueError("request admission cannot precede the session clock")
        policy = self.config.handoff_policy if handoff_policy is None else handoff_policy
        if not isinstance(policy, KvHandoffPolicy):
            raise TypeError("handoff_policy must implement KvHandoffPolicy")
        states = tuple(
            _RequestState(
                request=request,
                prefill=self.prefill_engines[
                    (self._next_prefill + index) % len(self.prefill_engines)
                ],
                decode=self.decode_engines[
                    (self._next_decode + index) % len(self.decode_engines)
                ],
            )
            for index, request in enumerate(rows)
        )
        self._next_prefill = (self._next_prefill + len(rows)) % len(
            self.prefill_engines
        )
        self._next_decode = (self._next_decode + len(rows)) % len(
            self.decode_engines
        )
        self._request_ids.update(request_ids)
        driver_order = (*self.prefill_engines, *self.decode_engines)
        driver_cursor = 0
        idle_iterations = 0
        prefill_batches: list[tuple[str, ...]] = []
        decode_batches: list[tuple[str, ...]] = []
        while any(state.decode_record_stop is None for state in states):
            for state in states:
                if (
                    state.prefill_internal_id is None
                    and state.request.admitted_at_ps <= self.clock.now_ps
                ):
                    self._admit_prefill(state)
            for state in states:
                if (
                    state.handoff is not None
                    and state.decode_internal_id is None
                    and state.handoff.completed_at_ps <= self.clock.now_ps
                ):
                    self._admit_decode(state)
            active = None
            for offset in range(len(driver_order)):
                index = (driver_cursor + offset) % len(driver_order)
                candidate = driver_order[index]
                if candidate.has_unfinished_requests:
                    active = candidate
                    driver_cursor = (index + 1) % len(driver_order)
                    break
            if active is not None:
                batch = self._step_engine(active, states, policy)
                if batch is None:
                    idle_iterations += 1
                    if idle_iterations > 1_000:
                        raise RuntimeError("SGLang session made no scheduler progress")
                else:
                    idle_iterations = 0
                    target = (
                        prefill_batches
                        if active.role is ServingPoolRole.PREFILL
                        else decode_batches
                    )
                    target.append(batch)
                continue
            candidates = self._future_ready_times(states, self.clock.now_ps)
            if not candidates:
                unfinished = [
                    state.request.request_id
                    for state in states
                    if state.decode_record_stop is None
                ]
                raise RuntimeError(f"concurrent session made no progress: {unfinished}")
            self.clock.advance_to(min(candidates))

        results = []
        for state in states:
            required = (
                state.prefill_internal_id,
                state.decode_internal_id,
                state.prefill_record_start,
                state.prefill_record_stop,
                state.decode_record_start,
                state.decode_record_stop,
                state.prefill_eligible_at_ps,
                state.prefill_completed_at_ps,
                state.handoff,
                state.bootstrap_token_id,
                state.join_metadata,
            )
            if any(value is None for value in required):
                raise RuntimeError("completed request is missing session state")
            if len(state.decode_token_ids) != state.request.decode_output_tokens:
                raise RuntimeError("decode token observation does not conserve")
            assert state.prefill_record_start is not None
            assert state.prefill_record_stop is not None
            assert state.decode_record_start is not None
            assert state.decode_record_stop is not None
            assert state.prefill_internal_id is not None
            assert state.decode_internal_id is not None
            assert state.prefill_eligible_at_ps is not None
            assert state.prefill_completed_at_ps is not None
            assert state.handoff is not None
            assert state.bootstrap_token_id is not None
            assert state.join_metadata is not None
            prefill_records = tuple(
                state.prefill.records[
                    state.prefill_record_start : state.prefill_record_stop
                ]
            )
            decode_records = tuple(
                state.decode.records[
                    state.decode_record_start : state.decode_record_stop
                ]
            )
            prefill_results = tuple(
                state.prefill.results[
                    state.prefill_record_start : state.prefill_record_stop
                ]
            )
            decode_results = tuple(
                state.decode.results[
                    state.decode_record_start : state.decode_record_stop
                ]
            )
            if len(prefill_results) != len(prefill_records):
                raise RuntimeError("prefill records and results do not align")
            if len(decode_results) != len(decode_records):
                raise RuntimeError("decode records and results do not align")
            decode_eligible = self._first_release(
                decode_records,
                state.decode_internal_id,
            )
            timeline = DisaggregatedRequestTimeline(
                request_id=state.request.request_id,
                admitted_at_ps=state.request.admitted_at_ps,
                prefill_eligible_at_ps=state.prefill_eligible_at_ps,
                prefill_completed_at_ps=state.prefill_completed_at_ps,
                handoff=state.handoff,
                decode_eligible_at_ps=decode_eligible,
                decode_token_completed_at_ps=tuple(
                    state.decode_token_completed_at_ps
                ),
            )
            results.append(
                SglangPdRequestResult(
                    timeline=timeline,
                    prefill_engine_id=state.prefill.engine_id,
                    decode_engine_id=state.decode.engine_id,
                    prefill_internal_request_id=state.prefill_internal_id,
                    decode_internal_request_id=state.decode_internal_id,
                    bootstrap_token_id=state.bootstrap_token_id,
                    decode_token_ids=tuple(state.decode_token_ids),
                    join_metadata=dict(state.join_metadata),
                    prefill_records=prefill_records,
                    decode_records=decode_records,
                    prefill_results=prefill_results,
                    decode_results=decode_results,
                )
            )
        return SglangPdConcurrentResult(
            requests=tuple(results),
            prefill_batches=tuple(prefill_batches),
            decode_batches=tuple(decode_batches),
        )

    def packet_rank_sets(
        self, *, prefill_engine: int = 0, decode_engine: int = 0
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return packet endpoints from the session's placement authority."""

        return (
            self.prefill_engines[prefill_engine].ranks,
            self.decode_engines[decode_engine].ranks,
        )

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        for engine in (*self.prefill_engines, *self.decode_engines):
            engine.close()

    def __enter__(self) -> SglangDisaggregatedSession:  # noqa: PYI034 (Self needs Python 3.11)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.shutdown()


__all__ = [
    "DEPLOYMENT_CURVE_POINT_SCHEMA",
    "DEPLOYMENT_CURVE_SCHEMA",
    "SGLANG_NATIVE_SEAM_STATUS",
    "SGLANG_PD_JOIN_MODE",
    "SGLANG_PD_JOIN_SCHEMA",
    "SGLANG_SESSION_AUTHORITY",
    "SglangDisaggregatedSession",
    "SglangPdConcurrentResult",
    "SglangPdCurvePoint",
    "SglangPdCurveRecord",
    "SglangPdRequest",
    "SglangPdRequestResult",
    "SglangPdSessionConfig",
]
