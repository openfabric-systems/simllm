"""Flagged vLLM v1 worker skeleton with no physical GPU state.

Select this class through vLLM's dotted worker seam and enable the deliberate
empty-compute path explicitly::

    SIMLLM_VLLM_WORKER_MODE=skeleton \\
    SIMLLM_VLLM_MODE=virtual \\
    VLLM_USE_V2_MODEL_RUNNER=0 \\
    vllm serve <model> \\
        --no-async-scheduling \\
        --worker-cls simllm.adapters.vllm.SimWorker

``SIMLLM_VLLM_WORKER_MODE`` has no permissive default. Constructing
:class:`SimWorker` unless its value is exactly ``skeleton`` raises before the
stock worker can initialize a device. The remaining ``SIMLLM_VLLM_*``
variables are shared with :class:`simllm.adapters.vllm.SimExecutor` and are
documented in :mod:`simllm.adapters.vllm.executor`.

The class subclasses vLLM v0.27.1's GPU ``Worker`` when vLLM is present, but
overrides every ordinary initialization and step entry that would reach CUDA.
Its model runner mirrors the selected v1 algorithm names and call order while
leaving physical ``_model_forward`` computation empty. The optional audited
Granite DBO mode emits source-backed semantic work at that boundary. One
adapter-local lifecycle owns the
:class:`simllm.core.VirtualClock`, step records, results, sink, and streaming
writer. The runner settles a step exactly once during ``execute_model`` and
returns the fabricated output from ``sample_tokens`` without advancing time a
second time.

This module stays importable without vLLM. Import-free tests use transcribed
configuration and scheduler stand-ins, matching the executor adapter's
precedent.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

from simllm.adapters.vllm.communicator import (
    FLOAT32,
    INT32,
    GroupCoordinatorEvent,
    GroupCoordinatorObserver,
    ShapeTensor,
    SimGroupCoordinator,
)
from simllm.adapters.vllm.executor import (
    _HOOKS,
    SimExecutorConfig,
    _ModelAnswers,
    _resolve_token_id,
    _SimStepRuntime,
    fabricate_sampled_tokens,
    model_dims_from_vllm_config,
)
from simllm.adapters.vllm.replay import ReplayTokenSource, sample_adapter_tokens
from simllm.adapters.vllm.schedule import (
    OBSERVED_SCHEDULE_GRANITE_DBO,
    OBSERVED_SCHEDULE_OFF,
    GraniteScheduleTiming,
    observations_from_vllm_step,
)
from simllm.compute import (
    ComputeProvider,
    GpuSpec,
    HostInitiationModel,
    ModelDims,
    RooflineProvider,
)
from simllm.compute.nccl_stack import NcclStackConfig
from simllm.core import ExecutionObservations, VirtualClock

logger = logging.getLogger(__name__)

WORKER_MODE_ENV = "SIMLLM_VLLM_WORKER_MODE"
SKELETON_WORKER_MODE = "skeleton"
SKELETON_TP_PAYLOAD_BYTES = 4_096


def _skeleton_fallback_latency(host_model: HostInitiationModel) -> int:
    """Keep the deliberately empty fallback exclusive to the ideal profile."""

    if not host_model.is_ideal:
        raise RuntimeError(
            "a nonideal SimWorker host model requires a timing sink result"
        )
    return 0


@dataclass(frozen=True)
class _DpCoordinationTensor(ShapeTensor):
    """Shape tensor carrying the skeleton's projected padded-token row."""

    num_tokens_across_dp: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.shape != (4, len(self.num_tokens_across_dp)):
            raise ValueError(
                "DP coordination shape must be (4, len(num_tokens_across_dp))"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.num_tokens_across_dp
        ):
            raise ValueError("DP padded-token values must be nonnegative integers")

    def new_empty(self, size: Any) -> _DpCoordinationTensor:
        return _DpCoordinationTensor(
            tuple(size),
            dtype=self.dtype,
            element_size_bytes=self.element_size_bytes,
            device=self.device,
            num_tokens_across_dp=self.num_tokens_across_dp,
        )


SKELETON_INIT_CALL_SEQUENCE = (
    "worker.init_device",
    "worker.load_model",
    "worker.get_kv_cache_spec",
    "worker.determine_available_memory",
    "worker.initialize_from_config",
    "worker.compile_or_warm_up_model",
    "worker.reset_mm_cache",
    "worker.get_supported_tasks",
)

SKELETON_STEP_CALL_SEQUENCE = (
    "worker.execute_model",
    "runner.execute_model",
    "runner._update_states",
    "runner._prepare_inputs",
    "runner._determine_batch_execution_and_padding",
    "runner._build_attention_metadata",
    "runner._preprocess",
    "runner._model_forward",
    "worker.sample_tokens",
    "runner.sample_tokens",
    "runner._sample",
    "runner._update_states_after_model_execute",
    "runner._bookkeeping_sync",
    "runner.eplb_step",
)

SKELETON_EMPTY_STEP_CALL_SEQUENCE = (
    "worker.execute_model",
    "runner.execute_model",
    "runner._update_states",
)

__all__ = [
    "SKELETON_EMPTY_STEP_CALL_SEQUENCE",
    "SKELETON_INIT_CALL_SEQUENCE",
    "SKELETON_STEP_CALL_SEQUENCE",
    "SKELETON_WORKER_MODE",
    "WORKER_MODE_ENV",
    "MirroredCall",
    "SimModelRunner",
    "SimModelRunnerOutput",
    "SimWorker",
    "latest_worker",
    "skeleton_mode_enabled",
]


def skeleton_mode_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether the explicit worker entry gate selects the skeleton."""
    env = os.environ if env is None else env
    return env.get(WORKER_MODE_ENV) == SKELETON_WORKER_MODE


def _require_skeleton_mode(env: Mapping[str, str] | None = None) -> None:
    if skeleton_mode_enabled(env):
        return
    env = os.environ if env is None else env
    actual = env.get(WORKER_MODE_ENV)
    raise RuntimeError(
        "SimWorker is the flagged empty-compute path and requires "
        f"{WORKER_MODE_ENV}={SKELETON_WORKER_MODE}; got {actual!r}. "
        "Without that exact flag it will not fall through to a partial real worker."
    )


def _v1_multiprocessing_enabled(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    value = env.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    return value.strip().lower() not in {"0", "false", "no", "off"}


try:
    from vllm.v1.worker.gpu_worker import Worker as _GpuWorkerBase

    _VLLM_WORKER_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    _VLLM_WORKER_IMPORT_ERROR = exc

    class _GpuWorkerBase:  # type: ignore[no-redef]
        """Transcribed construction surface used only when vLLM is absent."""

        def __init__(
            self,
            vllm_config: Any,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            is_driver_worker: bool = False,
        ) -> None:
            self.vllm_config = vllm_config
            self.model_config = vllm_config.model_config
            self.cache_config = vllm_config.cache_config
            self.lora_config = vllm_config.lora_config
            self.load_config = vllm_config.load_config
            self.parallel_config = vllm_config.parallel_config
            self.scheduler_config = vllm_config.scheduler_config
            self.device_config = vllm_config.device_config
            self.speculative_config = vllm_config.speculative_config
            self.observability_config = vllm_config.observability_config
            self.kv_transfer_config = vllm_config.kv_transfer_config
            self.compilation_config = vllm_config.compilation_config
            self.parallel_config.rank = rank
            self.local_rank = local_rank
            self.rank = rank
            self.distributed_init_method = distributed_init_method
            self.is_driver_worker = is_driver_worker
            self.device = None
            self.model_runner = None
            vllm_config.kernel_config.ir_op_priority.set_default()

            self.elastic_ep_executor = None
            self.worker_sentinel = None
            self._sleep_saved_buffers: dict[str, Any] = {}
            self._sleep_saved_draft_buffers: dict[str, Any] = {}
            self.weight_transfer_engine = None
            self._weight_update_active = False
            self.profiler = None
            self.profiler_config = vllm_config.profiler_config
            if self.profiler_config.profiler not in ("torch", "cuda", None):
                raise ValueError(
                    f"Unknown profiler type: {self.profiler_config.profiler}"
                )
            self.use_v2_model_runner = vllm_config.use_v2_model_runner
            self._pp_send_work: list[Any] = []
            self._sleep_mode_backend = None


@dataclass
class MirroredCall:
    """One name-mirrored call timestamped by the shared virtual clock."""

    name: str
    started_at_ps: int
    completed_at_ps: int


class _CallLedger:
    def __init__(self, clock: VirtualClock) -> None:
        self.clock = clock
        self.records: list[MirroredCall] = []

    @contextmanager
    def record(self, name: str) -> Iterator[None]:
        started_at_ps = self.clock.now_ps
        call = MirroredCall(
            name=name,
            started_at_ps=started_at_ps,
            completed_at_ps=started_at_ps,
        )
        self.records.append(call)
        try:
            yield
        finally:
            call.completed_at_ps = self.clock.now_ps


@dataclass
class SimModelRunnerOutput:
    """Import-free stand-in for vLLM's required output fields."""

    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]] | None = None


def _model_runner_output(
    req_ids: list[str],
    req_id_to_index: dict[str, int],
    sampled_token_ids: list[list[int]] | None = None,
) -> Any:
    try:
        from vllm.v1.outputs import ModelRunnerOutput
    except ImportError:
        ModelRunnerOutput = SimModelRunnerOutput
    kwargs: dict[str, Any] = {
        "req_ids": req_ids,
        "req_id_to_index": req_id_to_index,
    }
    if sampled_token_ids is not None:
        kwargs["sampled_token_ids"] = sampled_token_ids
    return ModelRunnerOutput(**kwargs)


@dataclass
class _ExecuteModelState:
    translated: Any
    scheduler_output: Any


class SimModelRunner:
    """GPUModelRunner-shaped copy whose model computation is empty."""

    def __init__(
        self,
        vllm_config: Any,
        runtime: _SimStepRuntime,
        answers: _ModelAnswers,
        token_id: int,
        replay: ReplayTokenSource | None,
        call_ledger: _CallLedger,
        tp_group: SimGroupCoordinator,
        dp_group: SimGroupCoordinator,
        compute_provider: ComputeProvider,
        gpu: GpuSpec,
        host_model: HostInitiationModel,
        dims: ModelDims,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.runtime = runtime
        self.answers = answers
        self.token_id = token_id
        self.replay = replay
        self.call_ledger = call_ledger
        self.tp_group = tp_group
        self.dp_group = dp_group
        self.compute_provider = compute_provider
        self.gpu = gpu
        self.host_model = host_model
        self.dims = dims
        if self.tp_group.observer is not self.dp_group.observer:
            raise ValueError("TP and DP groups must share one coordinator observer")
        self.coordinator_observer = self.tp_group.observer
        self.execute_model_state: _ExecuteModelState | None = None
        self.kv_cache_config: Any | None = None
        self.loaded = False
        self._loras: set[int] = set()
        self.is_pooling_model = False
        self.uniform_decode_query_len = 1
        self.latest_observations: ExecutionObservations | None = None
        self.latest_schedule_timing: GraniteScheduleTiming | None = None
        with self.call_ledger.record("runner.__init__"):
            pass

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        with self.call_ledger.record("runner.load_model"):
            self.loaded = True

    def profile_run(self) -> None:
        with self.call_ledger.record("runner.profile_run"):
            pass

    def get_kv_cache_spec(self) -> dict[str, Any]:
        with self.call_ledger.record("runner.get_kv_cache_spec"):
            return self.answers.get_kv_cache_spec(self.answers.vllm_config.parallel_config.rank)

    def initialize_kv_cache(self, kv_cache_config: Any) -> None:
        with self.call_ledger.record("runner.initialize_kv_cache"):
            self.kv_cache_config = kv_cache_config

    def update_max_model_len(self, max_model_len: int) -> None:
        with self.call_ledger.record("runner.update_max_model_len"):
            self.model_config.max_model_len = max_model_len

    def get_supported_tasks(self) -> tuple[str, ...]:
        with self.call_ledger.record("runner.get_supported_tasks"):
            return self.answers.supported_tasks()

    def reset_mm_cache(self) -> None:
        with self.call_ledger.record("runner.reset_mm_cache"):
            pass

    def reset_encoder_cache(self) -> None:
        with self.call_ledger.record("runner.reset_encoder_cache"):
            pass

    def _update_states(self, scheduler_output: Any) -> Any:
        with self.call_ledger.record("runner._update_states"):
            return scheduler_output

    def _prepare_inputs(self, translated: Any) -> Any:
        with self.call_ledger.record("runner._prepare_inputs"):
            return translated

    def _determine_batch_execution_and_padding(self, translated: Any) -> Any:
        with self.call_ledger.record("runner._determine_batch_execution_and_padding"):
            if self.dp_group.world_size > 1:
                local_num_tokens = translated.record.total_new_tokens
                coordination = _DpCoordinationTensor(
                    (4, self.dp_group.world_size),
                    dtype=INT32,
                    element_size_bytes=INT32.itemsize,
                    num_tokens_across_dp=(local_num_tokens,) * self.dp_group.world_size,
                )
                reduced = self.dp_group.all_reduce(coordination)
                if not isinstance(reduced, _DpCoordinationTensor):
                    raise TypeError(
                        "DP coordinator must preserve the padded-token projection"
                    )
                num_tokens_after_padding = reduced.num_tokens_across_dp[
                    self.dp_group.rank_in_group
                ]
                translated.record = replace(
                    translated.record,
                    num_tokens_after_padding=num_tokens_after_padding,
                )
            return translated

    def _build_attention_metadata(self, translated: Any) -> Any:
        with self.call_ledger.record("runner._build_attention_metadata"):
            return translated

    def _preprocess(self, translated: Any) -> Any:
        with self.call_ledger.record("runner._preprocess"):
            return translated

    def _model_forward(self, translated: Any) -> ExecutionObservations | None:
        with self.call_ledger.record("runner._model_forward"):
            mode = self.runtime.config.observed_schedule
            if mode == OBSERVED_SCHEDULE_GRANITE_DBO:
                if not self.runtime.is_authority:
                    self.latest_observations = None
                    self.latest_schedule_timing = None
                    return None
                observations, timing = observations_from_vllm_step(
                    translated.record,
                    self.vllm_config,
                    self.dims,
                    self.dp_group.ranks,
                    self.compute_provider,
                    self.gpu,
                    self.host_model,
                )
                self.latest_observations = observations
                self.latest_schedule_timing = timing
                return observations
            if mode != OBSERVED_SCHEDULE_OFF:
                raise AssertionError(f"unhandled observed schedule mode {mode!r}")
            tensor = ShapeTensor(
                (SKELETON_TP_PAYLOAD_BYTES // FLOAT32.itemsize,),
                dtype=FLOAT32,
                element_size_bytes=FLOAT32.itemsize,
            )
            self.tp_group.all_reduce(tensor)
            self.latest_observations = None
            self.latest_schedule_timing = None
            return None

    @property
    def coordinator_events(self) -> tuple[GroupCoordinatorEvent, ...]:
        return self.coordinator_observer.events

    def bind_simulated_groups(
        self,
        *,
        tp_group: SimGroupCoordinator,
        dp_group: SimGroupCoordinator,
    ) -> None:
        """Bind explicit logical groups before the first model step."""

        if self.execute_model_state is not None:
            raise RuntimeError("cannot replace simulated groups with a pending step")
        if tp_group.clock is not self.runtime.clock or dp_group.clock is not self.runtime.clock:
            raise ValueError("simulated groups must share the runner's virtual clock")
        if tp_group.observer is not dp_group.observer:
            raise ValueError("TP and DP groups must share one coordinator observer")
        self.tp_group = tp_group
        self.dp_group = dp_group
        self.coordinator_observer = tp_group.observer

    def execute_model(self, scheduler_output: Any, intermediate_tensors: Any = None) -> Any:
        with self.call_ledger.record("runner.execute_model"):
            if self.execute_model_state is not None:
                raise RuntimeError(
                    "State error: sample_tokens() must be called after "
                    "execute_model() returns None."
                )
            if getattr(scheduler_output, "has_structured_output_requests", False):
                raise RuntimeError(
                    "SimWorker does not support structured output: the grammar "
                    "would reject its fabricated token id (VLLM-8)."
                )
            updated_scheduler_output = self._update_states(scheduler_output)
            if not (
                getattr(updated_scheduler_output, "num_scheduled_tokens", None) or {}
            ):
                if self.replay is not None:
                    self.replay.validate_completions(updated_scheduler_output)
                self.runtime.drain(updated_scheduler_output)
                if self.replay is not None:
                    self.replay.observe_completions(updated_scheduler_output)
                return _model_runner_output([], {})

            translated = self.runtime.translate(updated_scheduler_output)
            if self.replay is not None:
                self.replay.validate_step(
                    translated.req_ids,
                    translated.produces_token,
                    updated_scheduler_output,
                )
            prepared = self._prepare_inputs(translated)
            padded = self._determine_batch_execution_and_padding(prepared)
            metadata = self._build_attention_metadata(padded)
            preprocessed = self._preprocess(metadata)
            observations = self._model_forward(preprocessed)
            self.runtime.settle(translated, observations)
            self.execute_model_state = _ExecuteModelState(
                translated=translated,
                scheduler_output=scheduler_output,
            )
            return None

    def _sample(
        self,
        translated: Any,
        scheduler_output: Any,
    ) -> tuple[list[str], dict[str, int], list[list[int]]]:
        with self.call_ledger.record("runner._sample"):
            return sample_adapter_tokens(
                self.replay,
                translated.req_ids,
                translated.produces_token,
                self.token_id,
                scheduler_output,
                fabricate=fabricate_sampled_tokens,
            )

    def _update_states_after_model_execute(self, sampled_token_ids: Any, scheduler_output: Any) -> None:
        with self.call_ledger.record("runner._update_states_after_model_execute"):
            pass

    def _bookkeeping_sync(
        self, sampled: tuple[list[str], dict[str, int], list[list[int]]]
    ) -> Any:
        with self.call_ledger.record("runner._bookkeeping_sync"):
            req_ids, req_id_to_index, sampled_token_ids = sampled
            return _model_runner_output(req_ids, req_id_to_index, sampled_token_ids)

    def eplb_step(self) -> None:
        with self.call_ledger.record("runner.eplb_step"):
            pass

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        with self.call_ledger.record("runner.sample_tokens"):
            state = self.execute_model_state
            if state is None:
                raise RuntimeError(
                    "sample_tokens called with no pending model output: the "
                    "execute/sample call order changed"
                )
            self.execute_model_state = None
            sampled = self._sample(state.translated, state.scheduler_output)
            self._update_states_after_model_execute(
                sampled[2],
                state.scheduler_output,
            )
            output = self._bookkeeping_sync(sampled)
            self.eplb_step()
            return output

    def take_draft_token_ids(self) -> None:
        with self.call_ledger.record("runner.take_draft_token_ids"):
            return

    def _dummy_run(self, *args: Any, **kwargs: Any) -> None:
        with self.call_ledger.record("runner._dummy_run"):
            return

    def add_lora(self, lora_request: Any) -> bool:
        with self.call_ledger.record("runner.add_lora"):
            lora_id = int(getattr(lora_request, "lora_int_id", lora_request))
            self._loras.add(lora_id)
            return True

    def remove_lora(self, lora_id: int) -> bool:
        with self.call_ledger.record("runner.remove_lora"):
            self._loras.discard(int(lora_id))
            return True

    def pin_lora(self, lora_id: int) -> bool:
        with self.call_ledger.record("runner.pin_lora"):
            return int(lora_id) in self._loras

    def list_loras(self) -> set[int]:
        with self.call_ledger.record("runner.list_loras"):
            return set(self._loras)

    def shutdown(self) -> None:
        with self.call_ledger.record("runner.shutdown"):
            self.execute_model_state = None


_LATEST_WORKERS: list[SimWorker] = []


def latest_worker() -> SimWorker | None:
    """The most recently constructed skeleton worker in this process."""
    return _LATEST_WORKERS[-1] if _LATEST_WORKERS else None


class SimWorker(_GpuWorkerBase):
    """vLLM GPU Worker subclass whose selected path creates no GPU state."""

    def __init__(
        self,
        vllm_config: Any,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        *,
        _simllm_clock: VirtualClock | None = None,
        _simllm_config: SimExecutorConfig | None = None,
    ) -> None:
        _require_skeleton_mode()
        if bool(getattr(vllm_config, "use_v2_model_runner", False)):
            raise RuntimeError(
                "SimWorker skeleton mirrors the V1 runner and requires "
                "VLLM_USE_V2_MODEL_RUNNER=0; V2 rebinding remains in VLLM-13."
            )
        parallel_config = vllm_config.parallel_config
        if bool(getattr(parallel_config, "enable_fault_tolerance", False)):
            raise RuntimeError(
                "SimWorker skeleton does not support vLLM fault tolerance: "
                "the worker sentinel requires a device-backed worker lifecycle "
                "(VLLM-13)."
            )
        backend = getattr(parallel_config, "distributed_executor_backend", None)
        if backend in {"ray", "external_launcher"}:
            raise RuntimeError(
                "SimWorker skeleton does not support the "
                f"{backend!r} executor backend in this slice; use the in-process "
                "or non-async multiprocessing executor (VLLM-13)."
            )
        async_scheduling = bool(
            getattr(vllm_config.scheduler_config, "async_scheduling", False)
        )
        if async_scheduling and _v1_multiprocessing_enabled():
            raise RuntimeError(
                "SimWorker has no device for vLLM's multiprocessing async-output "
                "thread. Pass --no-async-scheduling or set "
                "VLLM_ENABLE_V1_MULTIPROCESSING=0 (VLLM-13)."
            )
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        if getattr(vllm_config, "speculative_config", None) is not None:
            raise RuntimeError(
                "SimWorker does not support speculative decoding because its "
                "fabricated token would model zero draft acceptance (VLLM-8)."
            )
        parallel_config = self.parallel_config
        pp_size = int(getattr(parallel_config, "pipeline_parallel_size", 1) or 1)
        dp_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)
        if pp_size != 1:
            raise RuntimeError("SimWorker skeleton requires pipeline_parallel_size=1 (VLLM-10).")
        self.sim_config = (
            _simllm_config
            if _simllm_config is not None
            else (_HOOKS.config or self._config_from_env())
        )
        self.dims = model_dims_from_vllm_config(vllm_config)
        self.gpu = _HOOKS.gpu or self.sim_config.gpu_spec()
        self.compute_provider = _HOOKS.compute_provider or RooflineProvider(
            efficiency=self.sim_config.efficiency
        )
        self.host_model = _HOOKS.host_model or (
            HostInitiationModel(
                initiation_delay_ps=self.sim_config.host_initiation_ps
            )
            if self.sim_config.host_initiation_ps
            else HostInitiationModel.ideal()
        )
        if not self.host_model.is_ideal and _HOOKS.step_sink is None:
            raise RuntimeError(
                "a nonideal SimWorker host model requires a host-model-aware "
                "timing sink"
            )
        tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
        worker_world_size = int(
            getattr(parallel_config, "world_size", tp_size * pp_size) or tp_size * pp_size
        )
        dp_rank = int(getattr(parallel_config, "data_parallel_rank", 0) or 0)
        global_rank = dp_rank * worker_world_size + rank
        self._answers = _ModelAnswers(
            vllm_config=vllm_config,
            dims=self.dims,
            config=self.sim_config,
            tp_size=tp_size,
            pp_size=pp_size,
        )
        self.token_id = _resolve_token_id(self.dims.vocab_size, self.sim_config.token_id)
        self.replay = (
            ReplayTokenSource.from_path(
                self.sim_config.replay_run_path,
                max_model_len=int(self.model_config.max_model_len),
            )
            if self.sim_config.replay_run_path
            else None
        )
        # External-launch executors mark the local worker as a driver in every
        # process. Global rank zero is the only safe clock, sink, and stream
        # authority across those processes.
        is_authority = global_rank == 0
        self._runtime = _SimStepRuntime(
            config=self.sim_config,
            step_sink=_HOOKS.step_sink,
            fallback_latency=lambda translated: _skeleton_fallback_latency(
                self.host_model
            ),
            clock=_simllm_clock,
            is_authority=is_authority,
            host_model=self.host_model,
            gpu=self.gpu,
        )
        self.clock = self._runtime.clock
        self.step_records = self._runtime.step_records
        self.step_results = self._runtime.step_results
        self._call_ledger = _CallLedger(self.clock)
        self._coordinator_observer = GroupCoordinatorObserver(self.clock)
        tp_base = global_rank - (rank % tp_size)
        tp_ranks = tuple(range(tp_base, tp_base + tp_size))
        dp_ranks = tuple(rank + index * worker_world_size for index in range(dp_size))
        tp_chunk_bytes = (
            SKELETON_TP_PAYLOAD_BYTES // tp_size
            if SKELETON_TP_PAYLOAD_BYTES % tp_size == 0
            else 1
        )
        self.tp_group = SimGroupCoordinator(
            group_name="tp",
            ranks=tp_ranks,
            rank=global_rank,
            local_rank=local_rank,
            clock=self.clock,
            observer=self._coordinator_observer,
            stack_config=NcclStackConfig(
                channel_count=1,
                chunk_bytes=tp_chunk_bytes,
                fifo_slots_per_channel=2,
            ),
        )
        self.dp_group = SimGroupCoordinator(
            group_name="dp",
            ranks=dp_ranks,
            rank=global_rank,
            local_rank=local_rank,
            clock=self.clock,
            observer=self._coordinator_observer,
            stack_config=NcclStackConfig(
                channel_count=1,
                chunk_bytes=INT32.itemsize,
                fifo_slots_per_channel=2,
            ),
        )
        self.model_runner = None
        self.is_time_authority = is_authority
        _LATEST_WORKERS[:] = [self]

    @staticmethod
    def _config_from_env() -> Any:
        from simllm.adapters.vllm.executor import SimExecutorConfig

        return SimExecutorConfig.from_env()

    @property
    def mirrored_calls(self) -> list[MirroredCall]:
        return list(self._call_ledger.records)

    @property
    def mirrored_call_names(self) -> list[str]:
        return [call.name for call in self._call_ledger.records]

    @property
    def coordinator_events(self) -> tuple[GroupCoordinatorEvent, ...]:
        runner = self.model_runner
        if isinstance(runner, SimModelRunner):
            return runner.coordinator_events
        return self._coordinator_observer.events

    def _runner(self) -> SimModelRunner:
        runner = self.model_runner
        if runner is None:
            raise RuntimeError("SimWorker.init_device must run before worker RPCs")
        return runner

    def init_device(self) -> None:
        with self._call_ledger.record("worker.init_device"):
            self.device = None
            self.model_runner = SimModelRunner(
                self.vllm_config,
                self._runtime,
                self._answers,
                self.token_id,
                self.replay,
                self._call_ledger,
                self.tp_group,
                self.dp_group,
                self.compute_provider,
                self.gpu,
                self.host_model,
                self.dims,
            )

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        with self._call_ledger.record("worker.load_model"):
            self._runner().load_model(load_dummy_weights=load_dummy_weights)

    def get_kv_cache_spec(self) -> dict[str, Any]:
        with self._call_ledger.record("worker.get_kv_cache_spec"):
            return self._runner().get_kv_cache_spec()

    def determine_available_memory(self) -> int:
        with self._call_ledger.record("worker.determine_available_memory"):
            self._runner().profile_run()
            return self._answers.determine_available_memory()

    def update_max_model_len(self, max_model_len: int) -> None:
        with self._call_ledger.record("worker.update_max_model_len"):
            if self.replay is not None:
                self.replay.update_max_model_len(max_model_len)
            self.model_config.max_model_len = max_model_len
            self._runner().update_max_model_len(max_model_len)

    def initialize_from_config(self, kv_cache_config: Any) -> None:
        with self._call_ledger.record("worker.initialize_from_config"):
            num_blocks = getattr(kv_cache_config, "num_blocks", None)
            if num_blocks is not None:
                self.cache_config.num_gpu_blocks = num_blocks
            self._runner().initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> Any:
        with self._call_ledger.record("worker.compile_or_warm_up_model"):
            return self._answers.compilation_times()

    def get_kv_connector_handshake_metadata(self) -> None:
        with self._call_ledger.record("worker.get_kv_connector_handshake_metadata"):
            return

    def reset_mm_cache(self) -> None:
        with self._call_ledger.record("worker.reset_mm_cache"):
            self._runner().reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        with self._call_ledger.record("worker.reset_encoder_cache"):
            self._runner().reset_encoder_cache()

    def get_supported_tasks(self) -> tuple[str, ...]:
        with self._call_ledger.record("worker.get_supported_tasks"):
            return self._runner().get_supported_tasks()

    def execute_model(self, scheduler_output: Any) -> Any:
        with self._call_ledger.record("worker.execute_model"):
            return self._runner().execute_model(scheduler_output)

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        with self._call_ledger.record("worker.sample_tokens"):
            return self._runner().sample_tokens(grammar_output)

    def take_draft_token_ids(self) -> None:
        with self._call_ledger.record("worker.take_draft_token_ids"):
            return self._runner().take_draft_token_ids()

    def execute_dummy_batch(self) -> None:
        with self._call_ledger.record("worker.execute_dummy_batch"):
            self._runner()._dummy_run(1)

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        with self._call_ledger.record("worker.profile"):
            return

    def add_lora(self, lora_request: Any) -> bool:
        with self._call_ledger.record("worker.add_lora"):
            return self._runner().add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        with self._call_ledger.record("worker.remove_lora"):
            return self._runner().remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        with self._call_ledger.record("worker.pin_lora"):
            return self._runner().pin_lora(lora_id)

    def list_loras(self) -> set[int]:
        with self._call_ledger.record("worker.list_loras"):
            return self._runner().list_loras()

    def sleep(self, level: int = 1) -> None:
        with self._call_ledger.record("worker.sleep"):
            return

    def wake_up(self, tags: list[str] | None = None) -> None:
        with self._call_ledger.record("worker.wake_up"):
            return

    def check_health(self) -> None:
        with self._call_ledger.record("worker.check_health"):
            return

    def get_model(self) -> Any:
        raise RuntimeError("SimWorker skeleton has no physical model object (VLLM-13)")

    def get_draft_model(self) -> None:
        return None

    def shutdown(self) -> None:
        with self._call_ledger.record("worker.shutdown"):
            runner = self.model_runner
            if isinstance(runner, SimModelRunner):
                runner.shutdown()
