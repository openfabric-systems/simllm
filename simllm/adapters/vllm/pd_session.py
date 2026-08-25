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
from pathlib import Path
from typing import Any

from typing_extensions import Self

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
    DeclaredKvHandoffPolicy,
    DisaggregatedRequestTimeline,
    KvHandoffEvent,
    KvHandoffGeometry,
    ServingPoolRole,
    StepRecord,
    VirtualClock,
)
from simllm.placement import PlacementManifest, declared_manifest

PD_CONNECTOR_NAME = "SimPdConnector"
PD_CONNECTOR_MODULE = "simllm.adapters.vllm.pd_connector"
PD_KV_PARAMS_SCHEMA = "simllm-pd-kv-params-v1"
PS_PER_SECOND = 1_000_000_000_000


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


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
    handoff_policy: DeclaredKvHandoffPolicy
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
        if not isinstance(self.handoff_policy, DeclaredKvHandoffPolicy):
            raise TypeError("handoff_policy must be DeclaredKvHandoffPolicy")
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

    def run_request(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        *,
        decode_output_tokens: int,
        admitted_at_ps: int | None = None,
        handoff_policy: DeclaredKvHandoffPolicy | None = None,
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
        if not isinstance(policy, DeclaredKvHandoffPolicy):
            raise TypeError("handoff_policy must be DeclaredKvHandoffPolicy")
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

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.shutdown()


__all__ = [
    "PD_CONNECTOR_MODULE",
    "PD_CONNECTOR_NAME",
    "PD_KV_PARAMS_SCHEMA",
    "VllmDisaggregatedSession",
    "VllmPdRequestResult",
    "VllmPdSessionConfig",
    "VllmPoolEngine",
]
