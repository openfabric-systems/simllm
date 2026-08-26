"""In-process vLLM prefill/decode session over simulated GPUs.

One driver constructs role-declared engine pools, resetting process-wide
adapter hooks before each engine. All executors receive the same core virtual
clock. vLLM owns scheduling inside each pool; this driver only joins a
finished producer request to a consumer request through one core KV-handoff
event.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from simllm.adapters.vllm.executor import (
    SimExecutor,
    SimExecutorConfig,
    configure,
    latest_executor,
    reset_configuration,
)
from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
from simllm.compute import (
    GPU_ENVELOPES,
    ComputeProvider,
    GpuSpec,
    HostInitiationModel,
    ModelDims,
    RooflineProvider,
)
from simllm.core import (
    KV_HANDOFF_AUTHORITY,
    DisaggregatedRequestTimeline,
    KvHandoffEvent,
    KvHandoffGeometry,
    KvHandoffPolicy,
    ServingPoolRole,
    StepRecord,
    VirtualClock,
)
from simllm.placement import PlacementManifest, declared_manifest

PD_CONNECTOR_NAME = "SimPdConnector"
PD_CONNECTOR_MODULE = "simllm.adapters.vllm.pd_connector"
PD_KV_PARAMS_SCHEMA = "simllm-pd-kv-params-v1"
PS_PER_SECOND = 1_000_000_000_000
DEPLOYMENT_CURVE_SCHEMA = "simllm-deployment-curve-v1"
DEPLOYMENT_CURVE_POINT_SCHEMA = "simllm-deployment-curve-point-v1"


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


def _engine_local_placement(role: ServingPoolRole, width: int) -> PlacementManifest:
    placement = declared_manifest(tp=width, nodes=1, gpus_per_node=width)
    for rank in placement.ranks:
        rank.pool_role = role.value
        rank.gpu_uuid = f"sim-{role.value}-gpu-{rank.global_rank}"
        rank.pci_bus_id = f"0000:00:{rank.local_rank:02x}.0"
    return placement


@dataclass(frozen=True)
class VllmPdSessionConfig:
    """Construction and pricing inputs for one vLLM P/D session."""

    model: str
    workdir: Path
    dims: ModelDims
    handoff_geometry: KvHandoffGeometry
    handoff_policy: KvHandoffPolicy
    model_revision: str | None = None
    prefill_engines: int = 1
    decode_engines: int = 1
    tensor_parallel_size: int = 8
    max_model_len: int = 64
    num_gpu_blocks_override: int = 64
    max_num_seqs: int = 8
    token_id: int = 512
    provider: ComputeProvider = field(
        default_factory=lambda: RooflineProvider(efficiency=0.7)
    )
    gpu: GpuSpec = GPU_ENVELOPES["b100"]
    host_model: HostInitiationModel = field(default_factory=HostInitiationModel.ideal)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonblank string")
        if not isinstance(self.workdir, Path):
            raise TypeError("workdir must be a Path")
        if not isinstance(self.dims, ModelDims):
            raise TypeError("dims must be ModelDims")
        if not isinstance(self.handoff_geometry, KvHandoffGeometry):
            raise TypeError("handoff_geometry must be KvHandoffGeometry")
        if not isinstance(self.handoff_policy, KvHandoffPolicy):
            raise TypeError("handoff_policy must implement KvHandoffPolicy")
        for name in (
            "prefill_engines",
            "decode_engines",
            "tensor_parallel_size",
            "max_model_len",
            "num_gpu_blocks_override",
            "max_num_seqs",
            "token_id",
        ):
            _positive_int(name, getattr(self, name))
        if not isinstance(self.provider, ComputeProvider):
            raise TypeError("provider must implement ComputeProvider")
        if not isinstance(self.gpu, GpuSpec):
            raise TypeError("gpu must be GpuSpec")
        if not isinstance(self.host_model, HostInitiationModel):
            raise TypeError("host_model must be HostInitiationModel")
        self.host_model.validate_device(self.gpu)


@dataclass(frozen=True)
class VllmPoolEngine:
    """One role-declared frontend instance and its SimLLM projections."""

    role: ServingPoolRole
    ordinal: int
    engine_id: str
    llm: Any
    executor: SimExecutor
    step_sink: HtsimStepSink
    placement: PlacementManifest
    construction_seconds: float

    @property
    def simulated_worker_count(self) -> int:
        return self.executor.world_size


@dataclass(frozen=True)
class VllmPdRequestResult:
    """Live frontend output plus the framework-neutral session timeline."""

    timeline: DisaggregatedRequestTimeline
    prefill_engine_id: str
    decode_engine_id: str
    prefill_internal_request_id: str
    decode_internal_request_id: str
    bootstrap_token_id: int
    decode_token_ids: tuple[int, ...]
    kv_transfer_params: dict[str, Any]
    prefill_records: tuple[StepRecord, ...]
    decode_records: tuple[StepRecord, ...]

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
                "kv_transfer_params": dict(self.kv_transfer_params),
                "prefill_step_count": len(self.prefill_records),
                "decode_step_count": len(self.decode_records),
            }
        )
        return value


@dataclass(frozen=True)
class VllmPdRequest:
    """One stable session request admitted to the concurrent driver."""

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
class VllmPdCurvePoint:
    """One exact throughput-versus-delay point from completed requests."""

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
        requests: Sequence[VllmPdRequestResult],
    ) -> VllmPdCurvePoint:
        """Reduce terminal request timelines to the frozen exact curve axes."""

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
class VllmPdCurveRecord:
    """Machine-readable offered-load curve for one deployment configuration."""

    configuration_id: str
    prefill_engines: int
    decode_engines: int
    prompt_tokens: int
    points: tuple[VllmPdCurvePoint, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.configuration_id, str)
            or not self.configuration_id.strip()
        ):
            raise ValueError("configuration_id must be a nonblank string")
        _positive_int("prefill_engines", self.prefill_engines)
        _positive_int("decode_engines", self.decode_engines)
        _positive_int("prompt_tokens", self.prompt_tokens)
        points = tuple(self.points)
        if not points:
            raise ValueError("points must not be empty")
        if any(not isinstance(point, VllmPdCurvePoint) for point in points):
            raise TypeError("points must contain VllmPdCurvePoint values")
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
class VllmPdConcurrentResult:
    """Completed requests and scheduler-authored batch observations."""

    requests: tuple[VllmPdRequestResult, ...]
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
            unknown = sorted(
                {request_id for batch in batches for request_id in batch}
                - set(request_ids)
            )
            if unknown:
                raise ValueError(f"{name} contains unknown requests: {unknown}")
            object.__setattr__(self, name, batches)
        object.__setattr__(self, "requests", requests)

    @property
    def maximum_prefill_batch_size(self) -> int:
        return max(map(len, self.prefill_batches))

    @property
    def maximum_decode_batch_size(self) -> int:
        return max(map(len, self.decode_batches))

    def curve_point(self, offered_load_requests_per_second: Fraction) -> VllmPdCurvePoint:
        return VllmPdCurvePoint.from_requests(
            offered_load_requests_per_second,
            self.requests,
        )


@dataclass
class _ConcurrentRequestState:
    request: VllmPdRequest
    prefill: VllmPoolEngine
    decode: VllmPoolEngine
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
    kv_transfer_params: dict[str, Any] | None = None
    decode_token_ids: tuple[int, ...] = ()
    decode_token_completed_at_ps: list[int] = field(default_factory=list)


class VllmDisaggregatedSession:
    """Construct and drive separate vLLM prefill and decode engine pools."""

    def __init__(
        self,
        config: VllmPdSessionConfig,
        *,
        clock: VirtualClock | None = None,
        construction_observer: Callable[[VllmPoolEngine], None] | None = None,
    ) -> None:
        if not isinstance(config, VllmPdSessionConfig):
            raise TypeError("config must be VllmPdSessionConfig")
        if clock is not None and not isinstance(clock, VirtualClock):
            raise TypeError("clock must be a VirtualClock")
        self.config = config
        self.clock = VirtualClock() if clock is None else clock
        self.construction_observer = construction_observer
        self.prefill_engines: list[VllmPoolEngine] = []
        self.decode_engines: list[VllmPoolEngine] = []
        self.handoffs: list[KvHandoffEvent] = []
        self._request_ids: set[str] = set()
        self._next_prefill = 0
        self._next_decode = 0
        self._closed = False
        self._build_pools()

    def _build_pools(self) -> None:
        try:
            for role, count in (
                (ServingPoolRole.PREFILL, self.config.prefill_engines),
                (ServingPoolRole.DECODE, self.config.decode_engines),
            ):
                for ordinal in range(count):
                    engine = self._construct_engine(role, ordinal)
                    target = (
                        self.prefill_engines
                        if role is ServingPoolRole.PREFILL
                        else self.decode_engines
                    )
                    target.append(engine)
                    if self.construction_observer is not None:
                        self.construction_observer(engine)
        except BaseException:
            self.shutdown()
            raise
        finally:
            reset_configuration()

    def _construct_engine(
        self,
        role: ServingPoolRole,
        ordinal: int,
    ) -> VllmPoolEngine:
        from vllm import LLM
        from vllm.config import KVTransferConfig

        reset_configuration()
        engine_id = f"simllm-{role.value}-{ordinal}"
        placement = _engine_local_placement(
            role, self.config.tensor_parallel_size
        )
        engine_workdir = self.config.workdir / engine_id
        sink = HtsimStepSink(
            HtsimStepSinkConfig(
                profile="rnic-nn-fluid",
                tp_ranks=placement.group_ranks(0, "tp"),
                dims=self.config.dims,
                workdir=engine_workdir / "steps",
                provider=self.config.provider,
                gpu=self.config.gpu,
                host_model=self.config.host_model,
                placement_manifest=placement,
                collective_fixed_cost_envelope="intra-node-fixed-cost-v1",
                collective_fixed_cost_arm="lower",
            )
        )
        executor_config = SimExecutorConfig(
            mode="virtual",
            gpu=self.config.gpu.name,
            token_id=self.config.token_id,
            step_records_path=(engine_workdir / "step-records.jsonl").as_posix(),
            pool_role=role.value,
        )
        configure(
            step_sink=sink,
            compute_provider=self.config.provider,
            gpu=self.config.gpu,
            host_model=self.config.host_model,
            config=executor_config,
            clock=self.clock,
        )
        kv_role = (
            "kv_producer"
            if role is ServingPoolRole.PREFILL
            else "kv_consumer"
        )
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "skip_tokenizer_init": True,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "dtype": "bfloat16",
            "enforce_eager": True,
            "distributed_executor_backend": (
                "simllm.adapters.vllm.SimExecutor"
            ),
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "num_gpu_blocks_override": self.config.num_gpu_blocks_override,
            "disable_log_stats": True,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "async_scheduling": False,
            "disable_hybrid_kv_cache_manager": True,
            "kv_transfer_config": KVTransferConfig(
                kv_connector=PD_CONNECTOR_NAME,
                kv_connector_module_path=PD_CONNECTOR_MODULE,
                kv_role=kv_role,
                kv_buffer_device="cpu",
                engine_id=engine_id,
            ),
        }
        if self.config.model_revision is not None:
            kwargs["revision"] = self.config.model_revision
        started = time.perf_counter()
        llm = LLM(**kwargs)
        construction_seconds = time.perf_counter() - started
        executor = latest_executor()
        if executor is None:
            raise RuntimeError("vLLM did not construct SimExecutor")
        if executor.clock is not self.clock:
            raise RuntimeError("pool executor did not receive the session clock")
        if executor.config.pool_role != role.value:
            raise RuntimeError("pool executor role disagrees with the session")
        if executor.world_size != self.config.tensor_parallel_size:
            raise RuntimeError("pool engine constructed the wrong simulated width")
        live_kv = llm.llm_engine.vllm_config.kv_transfer_config
        if live_kv is None or live_kv.kv_role != kv_role:
            raise RuntimeError("pool connector role disagrees with the session")
        return VllmPoolEngine(
            role=role,
            ordinal=ordinal,
            engine_id=engine_id,
            llm=llm,
            executor=executor,
            step_sink=sink,
            placement=placement,
            construction_seconds=construction_seconds,
        )

    @staticmethod
    def _first_release(records: Sequence[StepRecord], request_id: str) -> int:
        for record in records:
            if any(row.request_id == request_id for row in record.scheduled):
                return record.virtual_time_ps
        raise RuntimeError(f"scheduler never released request {request_id!r}")

    def _drive_request(
        self,
        engine: VllmPoolEngine,
        request_id: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...], Any]:
        token_ids: tuple[int, ...] = ()
        token_times: list[int] = []
        final_output: Any | None = None
        while engine.llm.llm_engine.has_unfinished_requests():
            before_records = len(engine.executor.step_records)
            outputs = engine.llm.llm_engine.step()
            if len(engine.executor.step_records) != before_records + 1:
                raise RuntimeError("one vLLM step did not emit one SimLLM record")
            for output in outputs:
                if output.request_id != request_id:
                    raise RuntimeError("pool emitted a foreign request identity")
                current = tuple(output.outputs[0].token_ids)
                if current[: len(token_ids)] != token_ids:
                    raise RuntimeError("vLLM changed an already emitted token")
                token_times.extend(
                    self.clock.now_ps for _ in range(len(current) - len(token_ids))
                )
                token_ids = current
                if output.finished:
                    final_output = output
        if final_output is None:
            raise RuntimeError("vLLM request ended without a finished output")
        return token_ids, tuple(token_times), final_output

    @staticmethod
    def _prefill_sampling_params(request_id: str) -> Any:
        from vllm import SamplingParams

        return SamplingParams(
            temperature=0.0,
            max_tokens=1,
            min_tokens=1,
            ignore_eos=True,
            detokenize=False,
            extra_args={
                "kv_transfer_params": {
                    "schema": PD_KV_PARAMS_SCHEMA,
                    "session_request_id": request_id,
                }
            },
        )

    @staticmethod
    def _decode_sampling_params(
        output_tokens: int,
        kv_transfer_params: dict[str, Any],
    ) -> Any:
        from vllm import SamplingParams

        return SamplingParams(
            temperature=0.0,
            max_tokens=output_tokens,
            min_tokens=output_tokens,
            ignore_eos=True,
            detokenize=False,
            extra_args={"kv_transfer_params": dict(kv_transfer_params)},
        )

    @staticmethod
    def _stable_batch(
        engine: VllmPoolEngine,
        states: Sequence[_ConcurrentRequestState],
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
        if not batch:
            raise RuntimeError("a driven vLLM step emitted an empty scheduler batch")
        if len(batch) != len(set(batch)):
            raise RuntimeError("a scheduler batch duplicated a stable request identity")
        return tuple(batch)

    def _admit_prefill(self, state: _ConcurrentRequestState) -> None:
        request = state.request
        state.prefill_record_start = len(state.prefill.executor.step_records)
        internal_id = state.prefill.llm.llm_engine.add_request(
            request.request_id,
            list(request.prompt_token_ids),
            self._prefill_sampling_params(request.request_id),
            arrival_time=request.admitted_at_ps / PS_PER_SECOND,
        )
        if not isinstance(internal_id, str) or not internal_id:
            raise RuntimeError("prefill scheduler did not assign an internal identity")
        state.prefill_internal_id = internal_id

    def _finish_prefill(
        self,
        state: _ConcurrentRequestState,
        output: Any,
        policy: KvHandoffPolicy,
    ) -> None:
        tokens = tuple(output.outputs[0].token_ids)
        if len(tokens) != 1 or not output.finished:
            raise RuntimeError("prefill pool did not finish with one bootstrap token")
        kv_params = output.kv_transfer_params
        if not isinstance(kv_params, dict):
            raise TypeError("prefill connector did not publish KV parameters")
        request = state.request
        expected = {
            "schema": PD_KV_PARAMS_SCHEMA,
            "remote_request_id": request.request_id,
            "session_request_id": request.request_id,
            "remote_num_tokens": len(request.prompt_token_ids),
            "bootstrap_token_id": tokens[0],
        }
        disagreements = {
            key: (kv_params.get(key), value)
            for key, value in expected.items()
            if kv_params.get(key) != value
        }
        if disagreements:
            raise RuntimeError(
                f"prefill connector parameters disagree: {disagreements}"
            )
        state.prefill_completed_at_ps = self.clock.now_ps
        state.prefill_record_stop = len(state.prefill.executor.step_records)
        state.prefill_eligible_at_ps = self._first_release(
            state.prefill.executor.step_records[
                state.prefill_record_start : state.prefill_record_stop
            ],
            state.prefill_internal_id,
        )
        state.bootstrap_token_id = tokens[0]
        state.handoff = policy.schedule(
            submitted_at_ps=self.clock.now_ps,
            request_id=request.request_id,
            kv_bytes=self.config.handoff_geometry.bytes_for_prompt(
                len(request.prompt_token_ids)
            ),
        )
        state.kv_transfer_params = dict(kv_params)
        if state.handoff.authority != KV_HANDOFF_AUTHORITY:
            state.kv_transfer_params.update(
                {
                    "timing_authority": state.handoff.authority,
                    "pricing_arm": state.handoff.pricing_arm,
                    "handoff_completed_at_ps": state.handoff.completed_at_ps,
                }
            )
        self.handoffs.append(state.handoff)
        if sum(
            event.request_id == request.request_id for event in self.handoffs
        ) != 1:
            raise RuntimeError("request emitted other than one KV handoff")

    def _admit_decode(self, state: _ConcurrentRequestState) -> None:
        request = state.request
        if state.handoff is None or state.bootstrap_token_id is None:
            raise RuntimeError("decode admission requires a completed producer handoff")
        if state.kv_transfer_params is None:
            raise RuntimeError("decode admission requires connector parameters")
        state.decode_record_start = len(state.decode.executor.step_records)
        decode_prompt = [*request.prompt_token_ids, state.bootstrap_token_id]
        internal_id = state.decode.llm.llm_engine.add_request(
            request.request_id,
            decode_prompt,
            self._decode_sampling_params(
                request.decode_output_tokens,
                state.kv_transfer_params,
            ),
            arrival_time=state.handoff.completed_at_ps / PS_PER_SECOND,
        )
        if not isinstance(internal_id, str) or not internal_id:
            raise RuntimeError("decode scheduler did not assign an internal identity")
        if internal_id == state.prefill_internal_id:
            raise RuntimeError("separate pools reused one internal request identity")
        state.decode_internal_id = internal_id

    def _step_concurrent_engine(
        self,
        engine: VllmPoolEngine,
        states: Sequence[_ConcurrentRequestState],
        policy: KvHandoffPolicy,
    ) -> tuple[str, ...]:
        before_records = len(engine.executor.step_records)
        outputs = engine.llm.llm_engine.step()
        if len(engine.executor.step_records) != before_records + 1:
            raise RuntimeError("one vLLM step did not emit one SimLLM record")
        record = engine.executor.step_records[-1]
        batch = self._stable_batch(engine, states, record)
        state_by_id = {state.request.request_id: state for state in states}
        for output in outputs:
            state = state_by_id.get(output.request_id)
            if state is None:
                raise RuntimeError("pool emitted a foreign request identity")
            if engine.role is ServingPoolRole.PREFILL:
                if state.prefill is not engine:
                    raise RuntimeError("prefill output came from the wrong pool engine")
                if output.finished:
                    self._finish_prefill(state, output, policy)
                continue
            if state.decode is not engine:
                raise RuntimeError("decode output came from the wrong pool engine")
            current = tuple(output.outputs[0].token_ids)
            previous = state.decode_token_ids
            if current[: len(previous)] != previous:
                raise RuntimeError("vLLM changed an already emitted decode token")
            state.decode_token_completed_at_ps.extend(
                self.clock.now_ps for _ in range(len(current) - len(previous))
            )
            state.decode_token_ids = current
            if output.finished:
                state.decode_record_stop = len(engine.executor.step_records)
        return batch

    @staticmethod
    def _ready_time_candidates(
        states: Sequence[_ConcurrentRequestState],
        now_ps: int,
    ) -> tuple[int, ...]:
        candidates = []
        for state in states:
            if state.prefill_internal_id is None:
                candidates.append(state.request.admitted_at_ps)
            elif state.handoff is not None and state.decode_internal_id is None:
                candidates.append(state.handoff.completed_at_ps)
        return tuple(value for value in candidates if value > now_ps)

    def run_requests(
        self,
        requests: Sequence[VllmPdRequest],
        *,
        handoff_policy: KvHandoffPolicy | None = None,
    ) -> VllmPdConcurrentResult:
        """Drive several requests while the two stock schedulers own batching."""

        if self._closed:
            raise RuntimeError("session is closed")
        rows = tuple(requests)
        if not rows:
            raise ValueError("requests must not be empty")
        if any(not isinstance(request, VllmPdRequest) for request in rows):
            raise TypeError("requests must contain VllmPdRequest values")
        request_ids = tuple(request.request_id for request in rows)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("requests must have unique stable identities")
        duplicate = sorted(set(request_ids) & self._request_ids)
        if duplicate:
            raise ValueError(f"duplicate session request IDs: {duplicate}")
        if min(request.admitted_at_ps for request in rows) < self.clock.now_ps:
            raise ValueError("request admission cannot precede the session clock")
        policy = self.config.handoff_policy if handoff_policy is None else handoff_policy
        if not isinstance(policy, KvHandoffPolicy):
            raise TypeError("handoff_policy must implement KvHandoffPolicy")

        states = tuple(
            _ConcurrentRequestState(
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

            active: VllmPoolEngine | None = None
            for offset in range(len(driver_order)):
                index = (driver_cursor + offset) % len(driver_order)
                candidate = driver_order[index]
                if candidate.llm.llm_engine.has_unfinished_requests():
                    active = candidate
                    driver_cursor = (index + 1) % len(driver_order)
                    break
            if active is not None:
                batch = self._step_concurrent_engine(active, states, policy)
                target = (
                    prefill_batches
                    if active.role is ServingPoolRole.PREFILL
                    else decode_batches
                )
                target.append(batch)
                continue

            candidates = self._ready_time_candidates(states, self.clock.now_ps)
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
            if None in (
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
                state.kv_transfer_params,
            ):
                raise RuntimeError("completed concurrent request is missing state")
            request = state.request
            if len(state.decode_token_ids) != request.decode_output_tokens:
                raise RuntimeError("decode pool emitted the wrong token count")
            prefill_records = tuple(
                state.prefill.executor.step_records[
                    state.prefill_record_start : state.prefill_record_stop
                ]
            )
            decode_records = tuple(
                state.decode.executor.step_records[
                    state.decode_record_start : state.decode_record_stop
                ]
            )
            decode_eligible = self._first_release(
                decode_records,
                state.decode_internal_id,
            )
            timeline = DisaggregatedRequestTimeline(
                request_id=request.request_id,
                admitted_at_ps=request.admitted_at_ps,
                prefill_eligible_at_ps=state.prefill_eligible_at_ps,
                prefill_completed_at_ps=state.prefill_completed_at_ps,
                handoff=state.handoff,
                decode_eligible_at_ps=decode_eligible,
                decode_token_completed_at_ps=tuple(
                    state.decode_token_completed_at_ps
                ),
            )
            results.append(
                VllmPdRequestResult(
                    timeline=timeline,
                    prefill_engine_id=state.prefill.engine_id,
                    decode_engine_id=state.decode.engine_id,
                    prefill_internal_request_id=state.prefill_internal_id,
                    decode_internal_request_id=state.decode_internal_id,
                    bootstrap_token_id=state.bootstrap_token_id,
                    decode_token_ids=state.decode_token_ids,
                    kv_transfer_params=dict(state.kv_transfer_params),
                    prefill_records=prefill_records,
                    decode_records=decode_records,
                )
            )
        return VllmPdConcurrentResult(
            requests=tuple(results),
            prefill_batches=tuple(prefill_batches),
            decode_batches=tuple(decode_batches),
        )

    def run_request(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        *,
        decode_output_tokens: int,
        admitted_at_ps: int | None = None,
        handoff_policy: KvHandoffPolicy | None = None,
    ) -> VllmPdRequestResult:
        """Run one request through prefill, handoff, and decode in order."""

        from vllm import SamplingParams

        if self._closed:
            raise RuntimeError("session is closed")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a nonblank string")
        if request_id in self._request_ids:
            raise ValueError(f"duplicate session request ID {request_id!r}")
        prompt = tuple(prompt_token_ids)
        if not prompt:
            raise ValueError("prompt_token_ids must not be empty")
        if any(
            isinstance(token, bool) or type(token) is not int or token < 0
            for token in prompt
        ):
            raise ValueError("prompt_token_ids must be nonnegative integers")
        decode_output_tokens = _positive_int(
            "decode_output_tokens", decode_output_tokens
        )
        admitted = self.clock.now_ps if admitted_at_ps is None else admitted_at_ps
        if isinstance(admitted, bool) or type(admitted) is not int:
            raise TypeError("admitted_at_ps must be an integer")
        if admitted < self.clock.now_ps:
            raise ValueError("admitted_at_ps cannot precede the session clock")
        self.clock.advance_to(admitted)
        policy = self.config.handoff_policy if handoff_policy is None else handoff_policy
        if not isinstance(policy, KvHandoffPolicy):
            raise TypeError("handoff_policy must implement KvHandoffPolicy")
        self._request_ids.add(request_id)

        prefill = self.prefill_engines[self._next_prefill]
        self._next_prefill = (self._next_prefill + 1) % len(self.prefill_engines)
        prefill_record_start = len(prefill.executor.step_records)
        prefill_internal_id = prefill.llm.llm_engine.add_request(
            request_id,
            list(prompt),
            SamplingParams(
                temperature=0.0,
                max_tokens=1,
                min_tokens=1,
                ignore_eos=True,
                detokenize=False,
                extra_args={
                    "kv_transfer_params": {
                        "schema": PD_KV_PARAMS_SCHEMA,
                        "session_request_id": request_id,
                    }
                },
            ),
            arrival_time=admitted / PS_PER_SECOND,
        )
        if not isinstance(prefill_internal_id, str) or not prefill_internal_id:
            raise RuntimeError("prefill scheduler did not assign an internal identity")
        prefill_tokens, _, prefill_output = self._drive_request(prefill, request_id)
        prefill_records = tuple(
            prefill.executor.step_records[prefill_record_start:]
        )
        if len(prefill_tokens) != 1:
            raise RuntimeError("prefill pool did not emit one bootstrap token")
        kv_params = prefill_output.kv_transfer_params
        if not isinstance(kv_params, dict):
            raise TypeError("prefill connector did not publish KV parameters")
        if kv_params.get("schema") != PD_KV_PARAMS_SCHEMA:
            raise RuntimeError("prefill connector published the wrong KV schema")
        if kv_params.get("remote_request_id") != request_id:
            raise RuntimeError("prefill connector changed the request identity")
        if kv_params.get("session_request_id") != request_id:
            raise RuntimeError("prefill connector lost the session request identity")
        if kv_params.get("remote_num_tokens") != len(prompt):
            raise RuntimeError("prefill connector published the wrong KV coverage")
        bootstrap = int(kv_params["bootstrap_token_id"])
        if bootstrap != prefill_tokens[0]:
            raise RuntimeError("bootstrap token disagrees with connector parameters")
        prefill_eligible = self._first_release(prefill_records, prefill_internal_id)
        prefill_completed = self.clock.now_ps

        handoff = policy.apply(
            self.clock,
            request_id=request_id,
            kv_bytes=self.config.handoff_geometry.bytes_for_prompt(len(prompt)),
        )
        self.handoffs.append(handoff)
        if sum(event.request_id == request_id for event in self.handoffs) != 1:
            raise RuntimeError("request emitted other than one KV handoff")
        if handoff.authority != KV_HANDOFF_AUTHORITY:
            kv_params = dict(kv_params)
            kv_params.update(
                {
                    "timing_authority": handoff.authority,
                    "pricing_arm": handoff.pricing_arm,
                    "handoff_completed_at_ps": handoff.completed_at_ps,
                }
            )

        decode = self.decode_engines[self._next_decode]
        self._next_decode = (self._next_decode + 1) % len(self.decode_engines)
        decode_record_start = len(decode.executor.step_records)
        decode_prompt = [*prompt, bootstrap]
        decode_internal_id = decode.llm.llm_engine.add_request(
            request_id,
            decode_prompt,
            SamplingParams(
                temperature=0.0,
                max_tokens=decode_output_tokens,
                min_tokens=decode_output_tokens,
                ignore_eos=True,
                detokenize=False,
                extra_args={"kv_transfer_params": dict(kv_params)},
            ),
            arrival_time=handoff.completed_at_ps / PS_PER_SECOND,
        )
        if not isinstance(decode_internal_id, str) or not decode_internal_id:
            raise RuntimeError("decode scheduler did not assign an internal identity")
        if decode_internal_id == prefill_internal_id:
            raise RuntimeError("separate pools reused one internal request identity")
        decode_tokens, decode_token_times, _ = self._drive_request(
            decode, request_id
        )
        decode_records = tuple(decode.executor.step_records[decode_record_start:])
        if len(decode_tokens) != decode_output_tokens:
            raise RuntimeError("decode pool emitted the wrong token count")
        decode_eligible = self._first_release(decode_records, decode_internal_id)
        timeline = DisaggregatedRequestTimeline(
            request_id=request_id,
            admitted_at_ps=admitted,
            prefill_eligible_at_ps=prefill_eligible,
            prefill_completed_at_ps=prefill_completed,
            handoff=handoff,
            decode_eligible_at_ps=decode_eligible,
            decode_token_completed_at_ps=decode_token_times,
        )
        return VllmPdRequestResult(
            timeline=timeline,
            prefill_engine_id=prefill.engine_id,
            decode_engine_id=decode.engine_id,
            prefill_internal_request_id=prefill_internal_id,
            decode_internal_request_id=decode_internal_id,
            bootstrap_token_id=bootstrap,
            decode_token_ids=decode_tokens,
            kv_transfer_params=dict(kv_params),
            prefill_records=prefill_records,
            decode_records=decode_records,
        )

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        for engine in (*self.prefill_engines, *self.decode_engines):
            engine.llm.llm_engine.engine_core.shutdown()
        reset_configuration()

    def __enter__(self) -> VllmDisaggregatedSession:  # noqa: PYI034 (Self needs Python 3.11)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.shutdown()


__all__ = [
    "DEPLOYMENT_CURVE_POINT_SCHEMA",
    "DEPLOYMENT_CURVE_SCHEMA",
    "PD_CONNECTOR_MODULE",
    "PD_CONNECTOR_NAME",
    "PD_KV_PARAMS_SCHEMA",
    "VllmDisaggregatedSession",
    "VllmPdConcurrentResult",
    "VllmPdCurvePoint",
    "VllmPdCurveRecord",
    "VllmPdRequest",
    "VllmPdRequestResult",
    "VllmPdSessionConfig",
    "VllmPoolEngine",
]
