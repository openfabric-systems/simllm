"""vLLM v1 executor that replaces model execution with a simulated GPU.

Pinned to vLLM **v0.27.1**. The v1 engine resolves its executor class from a
dotted import path, so SimLLM plugs in without a fork::

    SIMLLM_VLLM_MODE=virtual SIMLLM_VLLM_GPU=b100 \\
    vllm serve meta-llama/Llama-3.1-8B \\
        --distributed-executor-backend simllm.adapters.vllm.SimExecutor \\
        --num-gpu-blocks-override 8192

The scheduler, KV-cache manager, block pool, prefix hashing and preemption
logic all run for real; only model execution is fabricated. Every engine step
becomes one :class:`simllm.core.StepRecord`, handed to an injected sink and
accumulated on the executor, which is what the offline (open-loop) mode later
renders as a GOAL trace.

This module must stay importable without vLLM: the ``vllm`` imports live in a
guarded block and inside methods, so importing it raises nothing when vLLM is
absent and constructing :class:`SimExecutor` raises a clear ``ImportError``.

Configuration that no vLLM CLI flag can carry comes from environment
variables; everything vLLM already exposes (parallel sizes, block size,
``--num-gpu-blocks-override``) keeps its own flag.

==============================  ==============================================
Environment variable            Meaning (default)
==============================  ==============================================
``SIMLLM_VLLM_MODE``            ``virtual`` returns immediately and reports
                                sim-native timings; ``paced`` sleeps the
                                estimated step latency so vLLM's own
                                wall-clock TTFT/TPOT stay meaningful
                                (``virtual``).
``SIMLLM_VLLM_KV_MEMORY_BYTES`` bytes reported by
                                ``determine_available_memory`` per worker
                                (``68719476736``, i.e. 64 GiB). Combined with
                                ``--num-gpu-blocks-override`` the resulting
                                block count is exact.
``SIMLLM_VLLM_GPU``             GPU envelope for the default roofline
                                estimate: ``a100``, ``h100``, ``h200``,
                                ``b100`` or ``b200`` (``b100``).
``SIMLLM_VLLM_PEAK_FLOPS``      override the envelope's dense FLOP/s (unset).
``SIMLLM_VLLM_MEM_BANDWIDTH``   override the envelope's HBM bytes/s (unset).
``SIMLLM_VLLM_EFFICIENCY``      roofline derate in (0, 1] (``0.7``).
``SIMLLM_VLLM_HOST_INIT_PS``    per-step host initiation delay in ps (``0``,
                                see :mod:`simllm.compute.host`).
``SIMLLM_VLLM_TOKEN_ID``        token id fabricated for every generated token
                                (``vocab_size // 2``).
``SIMLLM_VLLM_STEP_RECORDS``    JSONL path for the step records; each record
                                is appended the moment its step completes
                                (unset).
``SIMLLM_VLLM_SAMPLED_REQUEST_IDS``
                                ``1`` adds exact sampled identities to each
                                step projection; ``0`` retains accepted
                                count-only record bytes (``0``).
``SIMLLM_VLLM_NATIVE_STEPS``    JSONL path pairing the native scheduler fields
                                with their step projection (unset).
``SIMLLM_VLLM_REPLAY_RUN``      joined ``simllm-preplay-replay-run-v1`` JSON
                                whose exact tokens replace fabrication; every
                                request must enter with its oracle length
                                pinned as ``max_tokens`` (unset).
``SIMLLM_VLLM_OBSERVED_SCHEDULE``
                                ``granite-dbo`` enables the audited Granite
                                MoE observation producer; ``off`` preserves
                                the absent-observation path (``off``).
``SIMLLM_VLLM_POOL_ROLE``       declares this engine instance as ``single``,
                                ``prefill``, or ``decode`` (``single``).
``SIMLLM_VLLM_WORKER_MODE``     ``skeleton`` enables the flagged
                                :class:`simllm.adapters.vllm.SimWorker` path.
                                Any other value is rejected when that worker
                                class is selected (unset).
==============================  ==============================================

The fabricated token id is a fixed mid-vocabulary id, so generation never
stops on an EOS token: request lengths come from the workload generator (or
from ``max_tokens`` / ``ignore_eos``), not from the model's stop tokens.

Configurations the fabricated token would silently corrupt are refused
loudly instead: speculative decoding (every draft would be rejected, i.e. a
0% acceptance rate) raises at construction, structured output (the grammar
rejects the fabricated id, killing every request at its first token) raises
at the first step that schedules one, and pipeline parallelism is rejected
via ``supports_pp = False`` until the batch-queue output FIFO exists
(VLLM-8, VLLM-10 in docs/modules/adapters-vllm.md).

Objects passed in code (a :class:`~simllm.compute.ComputeProvider`, a
:class:`~simllm.compute.HostInitiationModel`, a step sink) go through
:func:`configure`, which only reaches the executor when the engine core runs
in this process (``LLM(...)``, or ``VLLM_ENABLE_V1_MULTIPROCESSING=0``). Under
``vllm serve`` the engine core is a separate process, so use the environment
variables and the JSONL dump.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

from simllm.adapters.vllm._version import PINNED_VLLM_VERSION
from simllm.adapters.vllm.capture import (
    VllmNativeStepCaptureStream,
    capture_vllm_native_step,
)
from simllm.adapters.vllm.replay import ReplayTokenSource, sample_adapter_tokens
from simllm.adapters.vllm.schedule import (
    OBSERVED_SCHEDULE_MODES,
    OBSERVED_SCHEDULE_OFF,
)
from simllm.compute import (
    GPU_ENVELOPES,
    PS_PER_SECOND,
    ComputeProvider,
    GpuSpec,
    HostInitiationModel,
    ModelDims,
    RooflineProvider,
    estimate_step_latency_ps,
    step_kernel,
)
from simllm.core import (
    DependencyLevel,
    ExecutionObservations,
    PrecisionConfig,
    RequestOutcomeLevel,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    StepRecordStream,
    StepResult,
    VirtualClock,
    check_precision_selection,
    step_records_to_json,
    write_step_records,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)

__all__ = [
    "GPU_ENVELOPES",
    "PINNED_VLLM_VERSION",
    "PS_PER_SECOND",
    "ExpertGroupStepSink",
    "ExpertParallelGeometry",
    "HostModelStepSink",
    "ModelDims",
    "ObservationStepSink",
    "ReplayTokenSource",
    "SimExecutor",
    "SimExecutorConfig",
    "SimExecutorHooks",
    "StepTranslator",
    "TranslatedStep",
    "configure",
    "estimate_step_latency_ps",
    "expert_group_ranks",
    "expert_parallel_geometry",
    "fabricate_sampled_tokens",
    "latest_executor",
    "model_dims_from_vllm_config",
    "observe_scheduler_output",
    "reset_configuration",
    "step_kernel",
    "step_records_to_json",
    "translate_scheduler_output",
    "vllm_is_available",
    "write_step_records",
]

#: A legacy sink consumes one record and may return the simulated outcome; a
#: sink that returns ``None`` leaves the estimate to the compute provider.
LegacyStepSink: TypeAlias = Callable[[StepRecord], StepResult | None]


@runtime_checkable
class ObservationStepSink(Protocol):
    """Consume a step plus its optional framework execution observations.

    The adapter binds its sole :class:`VirtualClock` before the first call.
    Omitting observations is meaningful: the sink must select its exact serial
    compatibility path rather than infer a framework schedule.
    """

    def bind_clock(self, clock: VirtualClock) -> None:
        """Bind the adapter's timing authority before any step executes."""
        ...

    def __call__(
        self,
        record: StepRecord,
        observations: ExecutionObservations | None,
    ) -> StepResult:
        """Return the authoritative result for one translated step."""
        ...


@runtime_checkable
class ExpertGroupStepSink(Protocol):
    """A sink that accepts the adapter's derived expert-parallel group.

    The adapter binds the group once, before any step, and only when the
    active parallel configuration actually uses expert parallelism. A sink
    that does not implement this stays on whatever group its own
    configuration declared, which is the explicit no-EP path.
    """

    def bind_expert_group(self, ep_ranks: Sequence[int]) -> None:
        """Bind the adapter-derived expert-parallel group before any step."""
        ...


@runtime_checkable
class HostModelStepSink(Protocol):
    """A sink that exposes the host model owned by its timing authority."""

    @property
    def host_model(self) -> HostInitiationModel:
        """Return the exact host model that the sink applies."""
        ...


StepSink: TypeAlias = LegacyStepSink | ObservationStepSink


def _missing_vllm_error() -> ImportError:
    detail = f" ({_VLLM_IMPORT_ERROR})" if _VLLM_IMPORT_ERROR is not None else ""
    return ImportError(
        f"simllm.adapters.vllm requires vLLM v{PINNED_VLLM_VERSION} installed in "
        f"the same environment{detail}. The adapter is imported by vLLM itself "
        "through --distributed-executor-backend, so it is only constructible "
        "inside a vLLM process."
    )


try:
    from vllm.v1.executor.abstract import Executor as _ExecutorBase

    _VLLM_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # vLLM absent: keep the module importable
    _VLLM_IMPORT_ERROR = exc

    class _ExecutorBase:  # type: ignore[no-redef]
        """Stand-in for ``vllm.v1.executor.abstract.Executor`` without vLLM.

        Only reachable when vLLM is missing, and its sole job is to turn a
        construction attempt into an actionable error instead of a NameError
        at class-definition time.
        """

        def __init__(self, vllm_config: Any) -> None:
            raise _missing_vllm_error()


def vllm_is_available() -> bool:
    """Whether the vLLM executor base class imported successfully."""
    return _VLLM_IMPORT_ERROR is None


# Configuration

def _env_str(env: Mapping[str, str], name: str, default: str | None) -> str | None:
    value = env.get(name)
    return value if value else default


def _env_int(env: Mapping[str, str], name: str, default: int | None) -> int | None:
    value = env.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_float(env: Mapping[str, str], name: str, default: float | None) -> float | None:
    value = env.get(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if not value:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


@dataclass(frozen=True)
class SimExecutorConfig:
    """Executor settings that have no vLLM CLI flag.

    Constructed from ``SIMLLM_VLLM_*`` environment variables; see the module
    docstring for the full table.
    """

    mode: str = "virtual"
    kv_memory_bytes: int = 64 * 1024**3
    gpu: str = "b100"
    peak_flops: float | None = None
    mem_bandwidth: float | None = None
    efficiency: float = 0.7
    host_initiation_ps: int = 0
    token_id: int | None = None
    step_records_path: str | None = None
    emit_sampled_request_ids: bool = False
    native_step_capture_path: str | None = None
    replay_run_path: str | None = None
    observed_schedule: str = OBSERVED_SCHEDULE_OFF
    pool_role: str = "single"

    def __post_init__(self) -> None:
        if self.mode not in ("virtual", "paced"):
            raise ValueError(f"SIMLLM_VLLM_MODE must be virtual or paced, got {self.mode!r}")
        if self.observed_schedule not in OBSERVED_SCHEDULE_MODES:
            known = ", ".join(OBSERVED_SCHEDULE_MODES)
            raise ValueError(
                "SIMLLM_VLLM_OBSERVED_SCHEDULE must be one of "
                f"{known}; got {self.observed_schedule!r}"
            )
        if self.pool_role not in ("single", "prefill", "decode"):
            raise ValueError(
                "SIMLLM_VLLM_POOL_ROLE must be single, prefill, or decode, "
                f"got {self.pool_role!r}"
            )
        if self.kv_memory_bytes <= 0:
            raise ValueError("SIMLLM_VLLM_KV_MEMORY_BYTES must be positive")
        if not isinstance(self.emit_sampled_request_ids, bool):
            raise TypeError("emit_sampled_request_ids must be a boolean")

    def selected_precision_levels(
        self,
        precision: PrecisionConfig | None = None,
    ) -> dict[str, Any]:
        """Report the seams these environment spellings select.

        ``replay_run_path`` selects the request-outcome level and
        ``observed_schedule`` selects the dependency level. The framework and
        workload seams are chosen by which entry point a deployment starts,
        not by this record, so they are absent here rather than guessed. An
        explicit ``precision`` that contradicts either observed level is
        refused before the executor runs anything.
        """

        return check_precision_selection(
            precision,
            request_outcome=(
                RequestOutcomeLevel.FABRICATED
                if self.replay_run_path is None
                else RequestOutcomeLevel.PREPLAY_ORACLE
            ),
            dependency=(
                DependencyLevel.SERIAL
                if self.observed_schedule == OBSERVED_SCHEDULE_OFF
                else DependencyLevel.OBSERVED_FRAMEWORK_SCHEDULE
            ),
            selection_source="SimExecutorConfig",
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> SimExecutorConfig:
        env = os.environ if env is None else env
        return cls(
            mode=_env_str(env, "SIMLLM_VLLM_MODE", "virtual") or "virtual",
            kv_memory_bytes=_env_int(env, "SIMLLM_VLLM_KV_MEMORY_BYTES", 64 * 1024**3),
            gpu=(_env_str(env, "SIMLLM_VLLM_GPU", "b100") or "b100").lower(),
            peak_flops=_env_float(env, "SIMLLM_VLLM_PEAK_FLOPS", None),
            mem_bandwidth=_env_float(env, "SIMLLM_VLLM_MEM_BANDWIDTH", None),
            efficiency=_env_float(env, "SIMLLM_VLLM_EFFICIENCY", 0.7) or 0.7,
            host_initiation_ps=_env_int(env, "SIMLLM_VLLM_HOST_INIT_PS", 0) or 0,
            token_id=_env_int(env, "SIMLLM_VLLM_TOKEN_ID", None),
            step_records_path=_env_str(env, "SIMLLM_VLLM_STEP_RECORDS", None),
            emit_sampled_request_ids=_env_bool(
                env, "SIMLLM_VLLM_SAMPLED_REQUEST_IDS", False
            ),
            native_step_capture_path=_env_str(
                env, "SIMLLM_VLLM_NATIVE_STEPS", None
            ),
            replay_run_path=_env_str(env, "SIMLLM_VLLM_REPLAY_RUN", None),
            observed_schedule=(
                _env_str(
                    env,
                    "SIMLLM_VLLM_OBSERVED_SCHEDULE",
                    OBSERVED_SCHEDULE_OFF,
                )
                or OBSERVED_SCHEDULE_OFF
            ).lower(),
            pool_role=(
                _env_str(env, "SIMLLM_VLLM_POOL_ROLE", "single") or "single"
            ).lower(),
        )

    def gpu_spec(self) -> GpuSpec:
        """GPU envelope with the per-field environment overrides applied."""
        base = GPU_ENVELOPES.get(self.gpu)
        if base is None:
            known = ", ".join(sorted(GPU_ENVELOPES))
            raise ValueError(f"unknown SIMLLM_VLLM_GPU {self.gpu!r}; known: {known}")
        return GpuSpec(
            name=base.name,
            peak_flops=self.peak_flops or base.peak_flops,
            mem_bandwidth=self.mem_bandwidth or base.mem_bandwidth,
        )


@dataclass
class SimExecutorHooks:
    """Process-wide injection point for objects vLLM's CLI cannot carry.

    vLLM constructs the executor itself (``executor_class(vllm_config)``), so
    a caller that drives the engine in this process sets the hooks first with
    :func:`configure`.
    """

    step_sink: StepSink | None = None
    compute_provider: ComputeProvider | None = None
    gpu: GpuSpec | None = None
    host_model: HostInitiationModel | None = None
    config: SimExecutorConfig | None = None
    clock: VirtualClock | None = None


_HOOKS = SimExecutorHooks()
#: at most one entry, the executor most recently built in this process
_LATEST: list[SimExecutor] = []


def configure(
    *,
    step_sink: StepSink | None = None,
    compute_provider: ComputeProvider | None = None,
    gpu: GpuSpec | None = None,
    host_model: HostInitiationModel | None = None,
    config: SimExecutorConfig | None = None,
    clock: VirtualClock | None = None,
) -> SimExecutorHooks:
    """Set the hooks the next :class:`SimExecutor` in this process picks up.

    Only arguments that are not ``None`` are applied, so repeated calls
    accumulate. Returns the live hook object.
    """
    if step_sink is not None:
        _HOOKS.step_sink = step_sink
    if compute_provider is not None:
        _HOOKS.compute_provider = compute_provider
    if gpu is not None:
        _HOOKS.gpu = gpu
    if host_model is not None:
        _HOOKS.host_model = host_model
    if config is not None:
        _HOOKS.config = config
    if clock is not None:
        if not isinstance(clock, VirtualClock):
            raise TypeError("clock must be a VirtualClock")
        _HOOKS.clock = clock
    return _HOOKS


def reset_configuration() -> SimExecutorHooks:
    """Clear every process-wide executor injection hook.

    This is the explicit boundary between independent in-process engine runs.
    It prevents a replay configuration or sink from one run becoming an
    implicit input to the next one.
    """

    _HOOKS.step_sink = None
    _HOOKS.compute_provider = None
    _HOOKS.gpu = None
    _HOOKS.host_model = None
    _HOOKS.config = None
    _HOOKS.clock = None
    return _HOOKS


def latest_executor() -> SimExecutor | None:
    """The most recently constructed executor in this process, if any."""
    return _LATEST[-1] if _LATEST else None


# Step translation (pure: no vLLM types cross this boundary)

@dataclass
class _RequestState:
    """What the executor remembers about one request between steps."""

    prompt_len: int
    #: prompt tokens the scheduler already had cached when it admitted the
    #: request, reported once (the step record is per step, not cumulative)
    num_cached_tokens: int = 0
    cached_reported: bool = False


@dataclass
class TranslatedStep:
    """One engine step in SimLLM terms plus what the fabricated output needs."""

    record: StepRecord
    #: request ids in ``ModelRunnerOutput`` order (scheduler iteration order)
    req_ids: list[str]
    #: parallel to :attr:`req_ids`: the prompt is complete after this step, so
    #: the request samples a token
    produces_token: list[bool]

    @property
    def num_sampled(self) -> int:
        return sum(self.produces_token)


class StepTranslator:
    """vLLM scheduler decisions to :class:`~simllm.core.StepRecord`.

    Fed plain data (request id, prompt length, computed and output token
    counts) rather than vLLM objects, so the translation is exercised without
    a vLLM installation. The scheduler reports ``num_computed_tokens`` afresh
    every step, so the translator trusts that value instead of accumulating
    its own: preemption resets it.
    """

    def __init__(self, *, emit_sampled_request_ids: bool = False) -> None:
        if not isinstance(emit_sampled_request_ids, bool):
            raise TypeError("emit_sampled_request_ids must be a boolean")
        self._emit_sampled_request_ids = emit_sampled_request_ids
        self._requests: dict[str, _RequestState] = {}

    def __len__(self) -> int:
        return len(self._requests)

    def add_new_request(self, req_id: str, prompt_len: int, num_computed_tokens: int = 0) -> None:
        """Register a request the scheduler admitted this step.

        ``num_computed_tokens`` at admission is exactly the prefix-cache hit,
        which is what :attr:`ScheduledRequest.num_cached_tokens` means.
        """
        self._requests[req_id] = _RequestState(
            prompt_len=max(int(prompt_len), 0),
            num_cached_tokens=max(int(num_computed_tokens), 0),
        )

    def update_cached_request(
        self, req_id: str, num_computed_tokens: int, num_output_tokens: int = 0
    ) -> None:
        """Refresh a request the workers already know about.

        A request that was never seen as new (e.g. the executor was attached
        mid-flight) gets its prompt length reconstructed from
        ``num_computed_tokens - num_output_tokens``, exact once the prompt is
        done and an underestimate while it is still being chunked.
        """
        state = self._requests.get(req_id)
        if state is None:
            state = _RequestState(prompt_len=max(num_computed_tokens - num_output_tokens, 0))
            state.cached_reported = True
            self._requests[req_id] = state

    def forget(self, req_ids: Iterable[str]) -> None:
        for req_id in req_ids:
            self._requests.pop(req_id, None)

    def translate(
        self,
        *,
        step_index: int,
        virtual_time_ps: int,
        num_scheduled_tokens: Mapping[str, int],
        num_computed_tokens: Mapping[str, int],
        finished_req_ids: Iterable[str] = (),
        preempted_req_ids: Iterable[str] = (),
    ) -> TranslatedStep:
        """Build the record for one step.

        ``num_computed_tokens`` holds the pre-step computed count per request,
        as reported by the scheduler in this step's new/cached request data.
        A request is in PREFILL while its computed count is below its prompt
        length, and samples a token on the step that completes the prompt.

        Attribution asymmetry, kept as vLLM reports it: ``finished_req_ids``
        on step N are the requests that completed during step N-1 (the
        scheduler rebinds its finished set after constructing the step, so
        the ids arrive one step later), while ``preempted_req_ids`` is
        same-step. Records are streamed to disk as each step completes, so
        the ids are recorded where they arrive and the shift is documented
        (here and in :func:`simllm.core.step_record_to_json`) instead of
        rewriting an already-durable record.
        """
        scheduled: list[ScheduledRequest] = []
        req_ids: list[str] = []
        produces_token: list[bool] = []
        for req_id, num_new_tokens in num_scheduled_tokens.items():
            computed_before = int(num_computed_tokens.get(req_id, 0))
            state = self._requests.get(req_id)
            if state is None:
                logger.warning("request %s scheduled without data; assuming prefill", req_id)
                state = _RequestState(prompt_len=computed_before + int(num_new_tokens))
                self._requests[req_id] = state
            context_length = computed_before + int(num_new_tokens)
            phase = (
                RequestPhase.PREFILL if computed_before < state.prompt_len else RequestPhase.DECODE
            )
            num_cached = 0 if state.cached_reported else state.num_cached_tokens
            state.cached_reported = True
            scheduled.append(
                ScheduledRequest(
                    request_id=req_id,
                    phase=phase,
                    num_new_tokens=int(num_new_tokens),
                    num_cached_tokens=num_cached,
                    context_length=context_length,
                )
            )
            req_ids.append(req_id)
            produces_token.append(context_length >= state.prompt_len)

        finished = sorted(finished_req_ids)
        record = StepRecord(
            step_index=step_index,
            virtual_time_ps=virtual_time_ps,
            scheduled=scheduled,
            preempted_request_ids=sorted(preempted_req_ids),
            finished_request_ids=finished,
            num_sampled=sum(produces_token),
            sampled_request_ids=(
                [
                    request_id
                    for request_id, sampled in zip(
                        req_ids, produces_token, strict=True
                    )
                    if sampled
                ]
                if self._emit_sampled_request_ids
                else None
            ),
        )
        self.forget(finished)
        return TranslatedStep(record=record, req_ids=req_ids, produces_token=produces_token)


def _prompt_length(new_req: Any) -> int:
    """Prompt length of a ``NewRequestData``, tolerating embeddings inputs."""
    token_ids = getattr(new_req, "prompt_token_ids", None)
    if token_ids is not None:
        return len(token_ids)
    shape = getattr(getattr(new_req, "prompt_embeds", None), "shape", None)
    if shape:
        return int(shape[0])
    return int(getattr(new_req, "num_computed_tokens", 0) or 0)


def observe_scheduler_output(translator: StepTranslator, scheduler_output: Any) -> dict[str, int]:
    """Feed one ``SchedulerOutput`` to the translator.

    Returns the pre-step computed-token count per request id, the only
    per-step quantity the translator cannot remember for itself. Every field
    is read with getattr so a stub with the same attribute names works, and so
    an additive vLLM change cannot break the read path.
    """
    computed: dict[str, int] = {}
    for new_req in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
        req_id = getattr(new_req, "req_id", None)
        if req_id is None:
            continue
        num_computed = int(getattr(new_req, "num_computed_tokens", 0) or 0)
        translator.add_new_request(req_id, _prompt_length(new_req), num_computed)
        computed[req_id] = num_computed
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    cached_ids = list(getattr(cached, "req_ids", ()) or ())
    cached_computed = list(getattr(cached, "num_computed_tokens", ()) or ())
    cached_outputs = list(getattr(cached, "num_output_tokens", ()) or ())
    for index, req_id in enumerate(cached_ids):
        num_computed = int(cached_computed[index]) if index < len(cached_computed) else 0
        num_output = int(cached_outputs[index]) if index < len(cached_outputs) else 0
        translator.update_cached_request(req_id, num_computed, num_output)
        computed[req_id] = num_computed
    return computed


def translate_scheduler_output(
    translator: StepTranslator,
    scheduler_output: Any,
    *,
    step_index: int,
    virtual_time_ps: int,
) -> TranslatedStep:
    """One ``SchedulerOutput`` to one :class:`TranslatedStep`."""
    computed = observe_scheduler_output(translator, scheduler_output)
    return translator.translate(
        step_index=step_index,
        virtual_time_ps=virtual_time_ps,
        num_scheduled_tokens=getattr(scheduler_output, "num_scheduled_tokens", None) or {},
        num_computed_tokens=computed,
        finished_req_ids=getattr(scheduler_output, "finished_req_ids", ()) or (),
        preempted_req_ids=getattr(scheduler_output, "preempted_req_ids", None) or (),
    )


def fabricate_sampled_tokens(
    req_ids: Sequence[str], produces_token: Sequence[bool], token_id: int
) -> tuple[list[str], dict[str, int], list[list[int]]]:
    """Build the three required ``ModelRunnerOutput`` fields.

    One inner list per request, in ``req_ids`` order: a single fabricated
    token for a request whose prompt is complete this step, and an empty list
    for a request still mid-prefill (a chunked-prefill step that produces
    nothing). ``req_id_to_index`` must cover every scheduled request or
    ``Scheduler.update_from_output`` raises ``KeyError``.
    """
    if len(req_ids) != len(produces_token):
        raise ValueError("req_ids and produces_token must have the same length")
    ordered = list(req_ids)
    req_id_to_index = {req_id: index for index, req_id in enumerate(ordered)}
    if len(req_id_to_index) != len(ordered):
        raise ValueError("duplicate request id in a single step")
    sampled = [[token_id] if produced else [] for produced in produces_token]
    return ordered, req_id_to_index, sampled


# Step cost model

def _safe(getter: Callable[[], Any], default: Any = None) -> Any:
    """Call a vLLM accessor, falling back when the internal API moved.

    Discovery of vLLM internals is deliberately forgiving: a renamed accessor
    degrades the estimate, it does not take the run down.
    """
    try:
        value = getter()
    except Exception as exc:  # noqa: BLE001 - vLLM internals differ across releases
        logger.debug("vLLM accessor failed (%s), using default %r", exc, default)
        return default
    return default if value is None else value


def _kv_element_bytes_from_cache_config(cache_config: Any, default: float) -> float:
    """Bytes per KV-cache element, honoring ``--kv-cache-dtype``.

    The executor's KV spec already reports the quantized torch dtype, so the
    roofline must charge the same width or every fp8-cache decode step comes
    out 2x too slow (decode is memory-bound on exactly this term).
    """
    cache_dtype = str(getattr(cache_config, "cache_dtype", None) or "auto")
    if cache_dtype.startswith("fp8"):
        return 1.0
    return default


def _weight_element_bytes_from_quant(quant_config: Any, default: float) -> float:
    """Bytes per weight element under weight quantization, heuristically.

    ``ModelConfig.dtype`` stays the activation dtype (bf16) on quantized
    checkpoints, so sizing the weight read off it overstates the bytes 2-4x.
    The quantization method name is mapped to an element width; an
    unrecognized method keeps the activation width and warns.
    """
    if quant_config is None:
        return default
    name = ""
    get_name = getattr(quant_config, "get_name", None)
    if callable(get_name):
        try:
            name = str(get_name() or "")
        except Exception as exc:  # noqa: BLE001 - vLLM internals differ across releases
            logger.debug("quant_config.get_name() failed: %s", exc)
    if not name:
        name = type(quant_config).__name__
    lowered = name.lower()
    if "fp8" in lowered or "int8" in lowered or "w8" in lowered:
        return 1.0
    if "awq" in lowered or "gptq" in lowered or "int4" in lowered or "w4" in lowered:
        return 0.5
    logger.warning(
        "ModelDims: unrecognized quantization method %r; sizing weights at the "
        "activation dtype width, which overstates the weight read",
        name,
    )
    return default


@dataclass(frozen=True)
class ExpertParallelGeometry:
    """vLLM's resolved MoE parallel shape, as seen by one global rank.

    ``flatten_tp_size`` is vLLM's ``dp * pcp * tp`` flattened device set
    (``fused_moe/config.py:1113-1121``). Expert parallelism is only in use when
    that set holds more than one device AND the flag is set
    (``fused_moe/config.py:1204-1207``); enabling the flag on a single device
    does nothing. In use, a device owns whole experts and the expert weights
    are not tensor-sharded (``moe_tp_size`` 1, ``ep_size`` the flattened size);
    out of use, the experts are tensor-sharded over the whole flattened set
    (``fused_moe/config.py:1217-1252``).

    ``ep_ranks`` is the expert-parallel group of this rank in vLLM's
    ``ExternalDP x DP x PP x PCP x TP`` layout order
    (``distributed/parallel_state.py:1789-1801``), i.e. every rank sharing this
    rank's ExternalDP index and pipeline stage, ordered by ``(dp, pcp, tp)``
    (``distributed/parallel_state.py:1893-1919``). ``ep_rank`` is this rank's
    index inside it, which equals vLLM's flattened TP rank.

    ``renders_expert_combine`` answers a narrower question than ``use_ep``:
    does this configuration execute an all-to-all whose combine returns an
    already reduced layer output? Two conditions must both hold.
    ``use_all2all_kernels`` needs expert parallelism AND one of ``dp_size > 1``,
    ``pcp_size > 1`` or sequence parallelism
    (``fused_moe/config.py:1052-1055``), so a ``tp=8, ep=8, dp=1`` shape runs
    naive expert parallelism with no all-to-all. And the selected backend must
    reduce: ``config/parallel.py:186`` defaults ``all2all_backend`` to
    ``allgather_reducescatter``, whose prepare-finalize returns
    ``output_is_reduced()`` False
    (``fused_moe/prepare_finalize/naive_dp_ep.py:109`` and ``:242``), while the
    deepep, mori, nixl and flashinfer families return True. When either fails,
    ``fused_moe/runner/moe_runner.py:436-465`` all-reduces the fused output
    over the tensor-parallel group, which is the two-allreduce dense-shaped
    inventory rather than the one-allreduce routed one.

    Sequence parallelism is not read, and the omission is exact rather than
    merely scoped. ``ParallelConfig.use_sequence_parallel_moe`` is a real
    config property (``config/parallel.py:653-668``, read by the model
    definitions), and it itself requires ``data_parallel_size > 1``. The
    ``is_sequence_parallel`` clause of ``use_all2all_kernels`` can therefore
    never fire in a case the ``dp_size > 1`` clause does not already cover, so
    reading it would change no answer. Rendering sequence parallelism at all
    remains TRAF-6.
    """

    flatten_tp_size: int
    use_ep: bool
    ep_size: int
    moe_tp_size: int
    ep_ranks: tuple[int, ...]
    ep_rank: int
    all2all_backend: str = "allgather_reducescatter"
    use_all2all_kernels: bool = False
    combine_is_reducing: bool = False

    @property
    def renders_expert_combine(self) -> bool:
        """Whether the combine returns an already reduced layer output."""

        return self.use_all2all_kernels and self.combine_is_reducing


#: pinned vLLM 0.26.0 backends whose prepare-finalize reduces the fused output
REDUCING_ALL2ALL_BACKENDS = frozenset(
    {
        "deepep_high_throughput",
        "deepep_low_latency",
        "deepep_v2",
        "mori_high_throughput",
        "mori_low_latency",
        "nixl_ep",
        "flashinfer_all2allv",
        "flashinfer_nvlink_two_sided",
        "flashinfer_nvlink_one_sided",
    }
)

#: pinned backends whose combine leaves the fused output unreduced
NON_REDUCING_ALL2ALL_BACKENDS = frozenset({"allgather_reducescatter"})


def _combine_is_reducing(backend: str) -> bool:
    """Classify one all2all backend, refusing any name we cannot place.

    True names are the deepep, mori, nixl and flashinfer prepare-finalize
    classes whose ``output_is_reduced()`` returns True
    (``fused_moe/prepare_finalize/deepep_ht.py:83``, ``deepep_ll.py:142``,
    ``deepep_v2.py:86``, ``mori.py:37``, ``nixl_ep.py:134``,
    ``flashinfer_nvlink_two_sided.py:52``,
    ``flashinfer_nvlink_one_sided.py:69``). The one False name is
    ``allgather_reducescatter`` (``naive_dp_ep.py:109`` and ``:242``). The
    remaining literals of ``config/parallel.py:40-53``, ``naive`` and ``pplx``,
    are removed backends that ``config/parallel.py:448-454`` rewrites to
    ``allgather_reducescatter`` during validation, so a real ``ParallelConfig``
    never carries them and this refusal is reachable only from a hand-built
    config. It is kept so such a config fails loudly instead of silently
    inheriting whichever answer the rewrite would have produced.
    """

    if backend in REDUCING_ALL2ALL_BACKENDS:
        return True
    if backend in NON_REDUCING_ALL2ALL_BACKENDS:
        return False
    raise NotImplementedError(
        f"vLLM all2all_backend {backend!r} cannot be classified as reducing or "
        "non-reducing, so the layer's allreduce inventory would be a guess; "
        "tracked by TRAF-40"
    )


def _parallel_size(parallel_config: Any, name: str) -> int:
    return max(int(getattr(parallel_config, name, 1) or 1), 1)


def _num_redundant_experts(parallel_config: Any) -> int:
    eplb_config = getattr(parallel_config, "eplb_config", None)
    if eplb_config is None:
        return 0
    return max(int(getattr(eplb_config, "num_redundant_experts", 0) or 0), 0)


#: fields whose positive value declares a mechanism ``ModelDims`` cannot carry
UNSUPPORTED_POSITIVE_MOE_FIELDS = (
    "n_shared_experts",
    "num_shared_experts",
    "shared_expert_intermediate_size",
    "moe_shared_expert_intermediate_size",
    "shared_intermediate_size",
    "first_k_dense_replace",
    "num_dense_layers",
)

#: routed-layer strides whose only fully routed value is 1
UNSUPPORTED_STRIDE_MOE_FIELDS = ("moe_layer_freq", "decoder_sparse_step")

#: per-layer exception lists whose non-empty value declares a mixed schedule
UNSUPPORTED_LAYER_LIST_MOE_FIELDS = ("mlp_only_layers",)

#: every field the reader refuses, in one tuple for tests and documentation
UNSUPPORTED_VLLM_MOE_FIELDS = (
    UNSUPPORTED_POSITIVE_MOE_FIELDS
    + UNSUPPORTED_STRIDE_MOE_FIELDS
    + UNSUPPORTED_LAYER_LIST_MOE_FIELDS
)


def _moe_field_values(*configs: Any, name: str) -> list[Any]:
    return [
        value
        for config in configs
        if config is not None
        for value in (getattr(config, name, None),)
        if value is not None
    ]


def _require_integer_moe_field(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"vLLM MoE mechanism field {name} must be an integer, got {value!r}"
        )
    return int(value)


def _reject_unsupported_moe_mechanisms(*configs: Any) -> None:
    """Refuse MoE geometries whose reduction inventory would be wrong.

    This reaches parity with the SGLang reader
    (``simllm/adapters/sglang/worker.py:775-834``) on the shared-expert and
    mixed dense-and-routed families, with the same per-field predicates,
    because both readers feed the same whole-model ``ModelDims`` and the same
    allreduce site rule. The two lists are not identical in either direction.
    This one adds ``num_shared_experts`` and ``shared_intermediate_size``,
    which the vLLM model definitions spell and the SGLang reader does not
    carry. The SGLang reader additionally refuses MLA (``kv_lora_rank``,
    ``q_lora_rank``, ``qk_nope_head_dim``, ``qk_rope_head_dim``), speculative
    (``num_nextn_predict_layers``) and quantization fields, which are compute
    and sampling concerns outside this guard's reduction-inventory scope.

    A shared expert's output is all-reduced over the tensor-parallel group even
    when the combine kernel already reduced the routed output
    (``model_executor/layers/fused_moe/runner/moe_runner.py:416-433``), so a
    shared-expert layer keeps an mlp-site allreduce that
    ``simllm.traffic.layer_tp_allreduce_sites`` drops for a routed all-to-all
    layer. The shared MLP itself rides a row-parallel projection with
    ``reduce_results=True`` (``model_executor/models/granitemoeshared.py:48``
    and ``:108``), which is the shared sibling of the family this repository
    drives live, and other families spell the same thing
    ``num_shared_experts`` (``model_executor/models/cohere2_moe.py:286-289``,
    ``exaone_moe.py:116-117``).

    A mixed dense and routed schedule leaves some layers with two allreduce
    sites and no all-to-all, which one whole-model ``ModelDims`` cannot
    express. It is spelled as a dense prefix (``first_k_dense_replace``,
    ``num_dense_layers``), as a routed stride whose fully routed value is 1
    (``moe_layer_freq``, and ``decoder_sparse_step`` at
    ``model_executor/models/qwen2_moe.py:310-316`` and
    ``qwen3_moe.py:385-391``), or as an explicit per-layer exception list
    (``mlp_only_layers`` in the same two models).

    All are refused rather than priced as fully routed. VLLM-25 owns
    supporting them here and TRAF-34 owns the traffic-side per-layer schedule.
    """

    for name in UNSUPPORTED_POSITIVE_MOE_FIELDS:
        for value in _moe_field_values(*configs, name=name):
            if _require_integer_moe_field(name, value) > 0:
                raise NotImplementedError(
                    f"vLLM MoE field {name}={int(value)} needs shared-expert or "
                    "mixed dense and routed geometry that ModelDims cannot "
                    "represent; tracked by VLLM-25"
                )
    for name in UNSUPPORTED_STRIDE_MOE_FIELDS:
        for value in _moe_field_values(*configs, name=name):
            if _require_integer_moe_field(name, value) != 1:
                raise NotImplementedError(
                    f"vLLM MoE field {name}={int(value)} declares a mixed dense "
                    "and routed layer schedule that ModelDims cannot represent; "
                    "tracked by VLLM-25"
                )
    for name in UNSUPPORTED_LAYER_LIST_MOE_FIELDS:
        for value in _moe_field_values(*configs, name=name):
            if value:
                raise NotImplementedError(
                    f"vLLM MoE field {name}={value!r} declares a mixed dense and "
                    "routed layer schedule that ModelDims cannot represent; "
                    "tracked by VLLM-25"
                )


def expert_parallel_geometry(vllm_config: VllmConfig) -> ExpertParallelGeometry:
    """Resolve the MoE parallel shape of the config's own global rank."""

    parallel_config = vllm_config.parallel_config
    data_parallel = _parallel_size(parallel_config, "data_parallel_size")
    context_parallel = _parallel_size(parallel_config, "prefill_context_parallel_size")
    tensor_parallel = _parallel_size(parallel_config, "tensor_parallel_size")
    pipeline_parallel = _parallel_size(parallel_config, "pipeline_parallel_size")
    rank = max(int(getattr(parallel_config, "rank", 0) or 0), 0)

    flatten_tp_size = data_parallel * context_parallel * tensor_parallel
    use_ep = (
        bool(getattr(parallel_config, "enable_expert_parallel", False))
        and flatten_tp_size > 1
    )
    stride = context_parallel * tensor_parallel
    block = data_parallel * pipeline_parallel * stride
    base = (rank // block) * block
    stage = (rank % block) // stride % pipeline_parallel
    ep_ranks = tuple(
        base
        + ((replica * pipeline_parallel + stage) * context_parallel + context)
        * tensor_parallel
        + tensor
        for replica in range(data_parallel)
        for context in range(context_parallel)
        for tensor in range(tensor_parallel)
    )
    backend = str(
        getattr(parallel_config, "all2all_backend", "") or "allgather_reducescatter"
    )
    use_all2all_kernels = use_ep and (data_parallel > 1 or context_parallel > 1)
    return ExpertParallelGeometry(
        flatten_tp_size=flatten_tp_size,
        use_ep=use_ep,
        ep_size=flatten_tp_size if use_ep else 1,
        moe_tp_size=1 if use_ep else flatten_tp_size,
        ep_ranks=ep_ranks,
        ep_rank=ep_ranks.index(rank),
        all2all_backend=backend,
        use_all2all_kernels=use_all2all_kernels,
        combine_is_reducing=(
            _combine_is_reducing(backend) if use_all2all_kernels else False
        ),
    )


def expert_group_ranks(vllm_config: VllmConfig) -> tuple[int, ...] | None:
    """The rank's expert-parallel group, or ``None`` for a dense model.

    vLLM creates no expert-parallel group at all for a non-MoE model
    (``distributed/parallel_state.py:1893-1896``), so a dense config has no
    group to report rather than a group of one.
    """

    model_config = vllm_config.model_config
    text_config = _safe(lambda: model_config.hf_text_config)
    is_moe = getattr(model_config, "is_moe", None)
    if is_moe is None:
        is_moe = bool(getattr(text_config, "num_local_experts", None))
    if not is_moe:
        return None
    return expert_parallel_geometry(vllm_config).ep_ranks


def model_dims_from_vllm_config(vllm_config: VllmConfig) -> ModelDims:
    """Read the per-rank geometry off a ``VllmConfig``.

    The geometry class itself is framework-neutral
    (:class:`simllm.compute.ModelDims`); this reader knows vLLM's accessor
    names. Every accessor has a Llama-8B-shaped fallback so a renamed vLLM
    internal degrades the estimate instead of taking the run down, but a
    fallback is never silent: the defaulted field names are warned once and
    stamped on ``ModelDims.defaulted_fields``.
    """
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    defaulted: list[str] = []

    def geom(name: str, getter: Callable[[], Any], default: Any) -> Any:
        try:
            value = getter()
        except Exception as exc:  # noqa: BLE001 - vLLM internals differ across releases
            logger.debug("vLLM accessor for %s failed: %s", name, exc)
            value = None
        if value is None:
            defaulted.append(name)
            return default
        return value

    tp_size = max(
        int(geom("tensor_parallel_size", lambda: parallel_config.tensor_parallel_size, 1)),
        1,
    )
    hidden_size = int(geom("hidden_size", model_config.get_hidden_size, 4096))
    text_config = _safe(lambda: model_config.hf_text_config)
    intermediate = int(
        geom(
            "intermediate_size",
            lambda: getattr(text_config, "intermediate_size", None),
            4 * hidden_size,
        )
    )
    raw_num_experts = getattr(text_config, "num_local_experts", None)
    # The Hugging Face field named num_local_experts is the GLOBAL expert count
    # of one Granite MoE layer; vLLM passes it straight through as num_experts
    # (model_executor/models/granitemoe.py:247-254). vLLM then adds the EPLB
    # redundant copies to obtain its global count
    # (model_executor/layers/fused_moe/layer.py:73-96), which defaults to zero
    # (config/parallel.py:70).
    routed_experts = (
        int(geom("num_experts", lambda: raw_num_experts, 0))
        if raw_num_experts is not None
        else 0
    )
    num_experts = (
        routed_experts + _num_redundant_experts(parallel_config)
        if routed_experts > 0
        else 0
    )
    top_k = (
        int(
            geom(
                "top_k",
                lambda: getattr(text_config, "num_experts_per_tok", None),
                0,
            )
        )
        if num_experts > 0
        else 0
    )
    if num_experts > 0:
        _reject_unsupported_moe_mechanisms(
            text_config, _safe(lambda: model_config.hf_config)
        )
    geometry = expert_parallel_geometry(vllm_config)
    if num_experts > 0 and geometry.ep_size > num_experts:
        raise ValueError(
            f"vLLM expert-parallel world {geometry.ep_size} exceeds the "
            f"{num_experts} experts of an MoE layer, so a rank would own none "
            "and receive no dispatched token"
        )
    if num_experts > 0:
        # model_executor/layers/fused_moe/expert_map_manager.py:62-69 spreads
        # the global experts as evenly as possible and gives the low EP ranks
        # the remainder; an uneven division is legal for the expert map.
        base_experts, remainder = divmod(num_experts, geometry.ep_size)
        local_num_experts = (
            base_experts + 1 if geometry.ep_rank < remainder else base_experts
        )
    else:
        local_num_experts = 0
    dtype_bytes = int(geom("dtype_bytes", lambda: model_config.dtype.itemsize, 2))
    dims = ModelDims(
        num_layers=int(
            geom("num_layers", lambda: model_config.get_num_layers(parallel_config), 32)
        ),
        hidden_size=hidden_size,
        intermediate_size=max(intermediate // tp_size, 1),
        num_heads=int(
            geom("num_heads", lambda: model_config.get_num_attention_heads(parallel_config), 32)
        ),
        num_kv_heads=int(
            geom("num_kv_heads", lambda: model_config.get_num_kv_heads(parallel_config), 8)
        ),
        head_size=int(geom("head_size", model_config.get_head_size, 128)),
        vocab_size=int(geom("vocab_size", model_config.get_vocab_size, 128256)),
        dtype_bytes=dtype_bytes,
        weight_dtype_bytes=_weight_element_bytes_from_quant(
            getattr(vllm_config, "quant_config", None), float(dtype_bytes)
        ),
        kv_dtype_bytes=_kv_element_bytes_from_cache_config(
            getattr(vllm_config, "cache_config", None), float(dtype_bytes)
        ),
        defaulted_fields=tuple(defaulted),
        num_experts=num_experts,
        top_k=top_k,
        # Under expert parallelism a device owns whole experts and the expert
        # weights are not tensor-sharded at all (moe tp_size becomes 1); without
        # it they shard across the FLATTENED dp * pcp * tp device set, not
        # across tp alone (fused_moe/config.py:1217-1252, 1324-1329).
        moe_intermediate_size=(
            max(intermediate // geometry.moe_tp_size, 1) if num_experts > 0 else None
        ),
        local_num_experts=local_num_experts,
    )
    if defaulted:
        logger.warning(
            "ModelDims: geometry fields defaulted to Llama-8B-shaped values: %s. "
            "Latency terms driven by them describe a phantom geometry.",
            ", ".join(defaulted),
        )
    return dims


# Record export lives in simllm.core.step (step_records_to_json,
# write_step_records): one schema-tagged JSON form for the offline dump and
# the closed-loop manifests, re-exported here for the adapter's public API.


# The executor

@dataclass
class _SimWorker:
    """Stand-in worker handed to callable ``collective_rpc`` payloads.

    There is no real worker process under this executor, so a callable RPC
    sees the configuration and its rank identity and nothing else.
    """

    vllm_config: Any
    rank: int
    local_rank: int
    device: Any = None
    model_runner: Any = None

    @property
    def model_config(self) -> Any:
        return self.vllm_config.model_config

    @property
    def parallel_config(self) -> Any:
        return self.vllm_config.parallel_config

    @property
    def cache_config(self) -> Any:
        return self.vllm_config.cache_config


@dataclass(frozen=True)
class _CompilationTimesFallback:
    """Duck type of ``vllm.v1.worker.worker_base.CompilationTimes``.

    Used only if that import moves; the engine only reads these two float
    fields in its ``max`` reduction over the per-worker times.
    """

    language_model: float = 0.0
    encoder: float = 0.0


@dataclass(frozen=True)
class _FullAttentionSpecFallback:
    """Import-free stand-in for the fields the engine consumes in tests."""

    block_size: int
    num_kv_heads: int
    head_size: int
    dtype: Any


def _make_compilation_times() -> Any:
    """One zero-valued vLLM ``CompilationTimes`` answer.

    Both adapter modes use this constructor so the engine-facing value stays
    identical. The fallback keeps import-free tests independent of vLLM.
    """
    try:
        from vllm.v1.worker.worker_base import CompilationTimes
    except ImportError:
        CompilationTimes = _CompilationTimesFallback
    return CompilationTimes(language_model=0.0, encoder=0.0)


def _resolve_token_id(vocab_size: int, configured_token_id: int | None) -> int:
    """Resolve the existing fixed-token policy without a framework type."""
    vocab_size = max(int(vocab_size), 2)
    if configured_token_id is not None:
        return max(0, min(int(configured_token_id), vocab_size - 1))
    return vocab_size // 2


@dataclass
class _ModelAnswers:
    """Model-derived answers shared by executor and worker simulation modes."""

    vllm_config: Any
    dims: ModelDims
    config: SimExecutorConfig
    tp_size: int
    pp_size: int

    def pp_layer_range(self, rank: int) -> tuple[int, int]:
        """Layers owned by ``rank``, using vLLM's uneven PP partitioning."""
        model_config = self.vllm_config.model_config
        total = int(_safe(model_config.get_total_num_hidden_layers, self.dims.num_layers))
        pp_rank = (rank // max(self.tp_size, 1)) % max(self.pp_size, 1)

        def _indices() -> tuple[int, int]:
            from vllm.distributed.utils import get_pp_indices

            return get_pp_indices(total, pp_rank, self.pp_size)

        even = total // max(self.pp_size, 1)
        fallback = (
            pp_rank * even,
            total if pp_rank == self.pp_size - 1 else (pp_rank + 1) * even,
        )
        return _safe(_indices, fallback)

    def kv_cache_dtype(self) -> Any:
        """Resolve the cache dtype exactly as the accepted executor path did."""
        try:
            import torch
        except ImportError:
            return getattr(self.vllm_config.model_config, "dtype", "bfloat16")

        cache_config = self.vllm_config.cache_config
        model_config = self.vllm_config.model_config
        cache_dtype = getattr(cache_config, "cache_dtype", "auto")
        quantized = {
            "fp8": "float8_e4m3fn",
            "fp8_e4m3": "float8_e4m3fn",
            "fp8_e5m2": "float8_e5m2",
        }.get(str(cache_dtype))
        if quantized is not None:
            resolved = getattr(torch, quantized, None)
            if resolved is not None:
                return resolved
        model_dtype = getattr(model_config, "dtype", None)
        return model_dtype if isinstance(model_dtype, torch.dtype) else torch.bfloat16

    def get_kv_cache_spec(self, rank: int) -> dict[str, Any]:
        """One full-attention spec per layer owned by ``rank``."""
        try:
            from vllm.v1.kv_cache_interface import FullAttentionSpec
        except ImportError:
            FullAttentionSpec = _FullAttentionSpecFallback

        cache_config = self.vllm_config.cache_config
        block_size = int(getattr(cache_config, "block_size", None) or 16)
        spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.dims.num_kv_heads,
            head_size=self.dims.head_size,
            dtype=self.kv_cache_dtype(),
        )
        start, end = self.pp_layer_range(rank)
        return {f"model.layers.{layer}.self_attn.attn": spec for layer in range(start, end)}

    def determine_available_memory(self) -> int:
        return self.config.kv_memory_bytes

    def compilation_times(self) -> Any:
        return _make_compilation_times()

    def supported_tasks(self) -> tuple[str, ...]:
        runner_type = getattr(self.vllm_config.model_config, "runner_type", "generate")
        return ("embed",) if runner_type == "pooling" else ("generate",)


def _completed_future(value: Any) -> Future:
    future: Future = Future()
    future.set_result(value)
    return future


class _SimStepRuntime:
    """Shared step-record lifecycle with one central virtual clock.

    The executor supplies its accepted roofline fallback. The worker skeleton
    supplies zero, since its copied model computation is deliberately empty.
    A sink remains authoritative when it returns a ``StepResult``.
    """

    def __init__(
        self,
        *,
        config: SimExecutorConfig,
        step_sink: StepSink | None,
        fallback_latency: Callable[[TranslatedStep], int],
        clock: VirtualClock | None = None,
        is_authority: bool = True,
        host_model: HostInitiationModel | None = None,
        gpu: GpuSpec | None = None,
    ) -> None:
        self.config = config
        self.step_sink = step_sink if is_authority else None
        self.fallback_latency = fallback_latency
        self.is_authority = is_authority
        self.host_model = host_model
        self.gpu = gpu
        self._validate_host_model_selection()
        self.clock = VirtualClock() if clock is None else clock
        if isinstance(self.step_sink, ObservationStepSink):
            self.step_sink.bind_clock(self.clock)
        self.translator = StepTranslator(
            emit_sampled_request_ids=config.emit_sampled_request_ids
        )
        self.step_index = 0
        self.step_records: list[StepRecord] = []
        self.step_results: list[StepResult] = []
        self.record_stream = (
            StepRecordStream(config.step_records_path)
            if is_authority and config.step_records_path
            else None
        )
        self.native_step_stream = (
            VllmNativeStepCaptureStream(config.native_step_capture_path)
            if is_authority and config.native_step_capture_path
            else None
        )

    def _validate_host_model_selection(self) -> None:
        """Validate one shared model without adding its duration here."""

        model = self.host_model
        sink = self.step_sink
        if model is None:
            sink_model = getattr(sink, "host_model", None)
            if sink_model is None:
                return
            if not isinstance(sink_model, HostInitiationModel):
                raise TypeError("step sink host_model must be a HostInitiationModel")
            self.host_model = sink_model
            model = sink_model
        if not isinstance(model, HostInitiationModel):
            raise TypeError("host_model must be a HostInitiationModel")
        if self.gpu is not None:
            model.validate_device(self.gpu)
        if sink is None:
            return
        sink_model = getattr(sink, "host_model", None)
        if sink_model is not None:
            if sink_model != model:
                raise RuntimeError(
                    "adapter and step sink must select the same host model"
                )
            return
        if not model.is_ideal:
            raise RuntimeError(
                "a nonideal adapter host model requires a host-model-aware step sink"
            )

    def translate(self, scheduler_output: Any) -> TranslatedStep:
        translated = translate_scheduler_output(
            self.translator,
            scheduler_output,
            step_index=self.step_index,
            virtual_time_ps=self.clock.now_ps,
        )
        if self.native_step_stream is not None:
            self.native_step_stream.append(
                capture_vllm_native_step(scheduler_output, translated.record)
            )
        self.step_index += 1
        return translated

    def settle(
        self,
        translated: TranslatedStep,
        observations: ExecutionObservations | None = None,
    ) -> StepResult:
        """Apply the accepted sink, fallback, append, advance, pace order."""
        self._validate_host_model_selection()
        record = translated.record
        if isinstance(self.step_sink, ObservationStepSink):
            result = self.step_sink(record, observations)
        elif self.step_sink is not None:
            if observations is not None:
                raise TypeError(
                    "framework observations require an ObservationStepSink; "
                    "the configured legacy sink accepts StepRecord only"
                )
            result = self.step_sink(record)
        else:
            if observations is not None:
                raise RuntimeError(
                    "framework observations require a configured ObservationStepSink"
                )
            result = None
        if result is None:
            latency_ps = self.fallback_latency(translated)
            result = StepResult(
                step_index=record.step_index,
                step_latency_ps=latency_ps,
                completed_at_ps=self.clock.now_ps + latency_ps,
            )
        if self.is_authority:
            self.step_records.append(record)
            self.step_results.append(result)
            if self.record_stream is not None:
                self.record_stream.append(record)
            self.clock.advance_to(max(result.completed_at_ps, self.clock.now_ps))
            if self.config.mode == "paced" and result.step_latency_ps > 0:
                time.sleep(result.step_latency_ps / PS_PER_SECOND)
        return result

    def drain(self, scheduler_output: Any) -> bool:
        """Record an empty completion-bearing step with zero elapsed time."""
        self._validate_host_model_selection()
        finished = getattr(scheduler_output, "finished_req_ids", ()) or ()
        preempted = getattr(scheduler_output, "preempted_req_ids", None) or ()
        if not (finished or preempted):
            return False
        translated = self.translate(scheduler_output)
        record = translated.record
        result = None
        if isinstance(self.step_sink, ObservationStepSink):
            result = self.step_sink(record, None)
        elif self.step_sink is not None:
            self.step_sink(record)
        if self.is_authority:
            self.step_records.append(record)
            self.step_results.append(
                result
                if result is not None
                else StepResult(
                    step_index=record.step_index,
                    step_latency_ps=0,
                    completed_at_ps=self.clock.now_ps,
                )
            )
            if self.record_stream is not None:
                self.record_stream.append(record)
        return True


class SimExecutor(_ExecutorBase):
    """vLLM v1 executor whose workers are simulated, not real GPUs.

    Selected with ``--distributed-executor-backend
    simllm.adapters.vllm.SimExecutor``. It services every init-time RPC with
    model-derived values, fabricates one ``ModelRunnerOutput`` per step, and
    records each step as a :class:`~simllm.core.StepRecord`.
    """

    uses_ray = False
    #: Pipeline parallelism is off until the batch-queue output FIFO exists
    #: (VLLM-10): the PP > 1 engine loop interleaves execute_model and
    #: sample_tokens across in-flight steps, which a single pending-output
    #: slot serves wrongly and silently. Note the CLI dotted-path spelling
    #: could never reach PP anyway: vLLM reads supports_pp off the *string*
    #: before resolving it, so --pipeline-parallel-size > 1 fails in
    #: EngineArgs regardless of this attribute.
    supports_pp = False

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        step_sink: StepSink | None = None,
        compute_provider: ComputeProvider | None = None,
        gpu: GpuSpec | None = None,
        host_model: HostInitiationModel | None = None,
        config: SimExecutorConfig | None = None,
        clock: VirtualClock | None = None,
    ) -> None:
        if _VLLM_IMPORT_ERROR is not None:
            raise _missing_vllm_error()
        self.config = (
            config
            if config is not None
            else (_HOOKS.config or SimExecutorConfig.from_env())
        )
        self.gpu = gpu or _HOOKS.gpu or self.config.gpu_spec()
        self.compute_provider: ComputeProvider = (
            compute_provider
            or _HOOKS.compute_provider
            or RooflineProvider(efficiency=self.config.efficiency)
        )
        self.host_model = (
            host_model
            or _HOOKS.host_model
            or (
                HostInitiationModel(
                    initiation_delay_ps=self.config.host_initiation_ps
                )
                if self.config.host_initiation_ps
                else HostInitiationModel.ideal()
            )
        )
        self.step_sink: StepSink | None = step_sink or _HOOKS.step_sink
        self.replay = (
            ReplayTokenSource.from_path(
                self.config.replay_run_path,
                max_model_len=int(vllm_config.model_config.max_model_len),
            )
            if self.config.replay_run_path
            else None
        )
        self._runtime = _SimStepRuntime(
            config=self.config,
            step_sink=self.step_sink,
            fallback_latency=self._estimate_latency,
            host_model=self.host_model,
            gpu=self.gpu,
            clock=clock if clock is not None else _HOOKS.clock,
        )
        #: every translated step, in order; the offline mode renders these
        self.step_records = self._runtime.step_records
        #: simulated outcome per recorded step, parallel to step_records
        self.step_results = self._runtime.step_results
        self.clock = self._runtime.clock
        self.translator = self._runtime.translator
        self._record_stream = self._runtime.record_stream
        super().__init__(vllm_config)
        _LATEST[:] = [self]

    # Lifecycle

    def _init_executor(self) -> None:
        if getattr(self.vllm_config, "speculative_config", None) is not None:
            # A fabricated token means every draft is rejected: the scheduler
            # walks num_computed_tokens back each step and the trace models a
            # 0% acceptance rate without saying so (VLLM-8).
            raise RuntimeError(
                "SimExecutor does not support speculative decoding: the "
                "fabricated token id would silently model a 0% draft "
                "acceptance rate. Remove --speculative-config (VLLM-8)."
            )
        parallel_config = self.parallel_config
        self.tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
        self.pp_size = int(getattr(parallel_config, "pipeline_parallel_size", 1) or 1)
        self.world_size = int(
            getattr(parallel_config, "world_size", 0) or self.tp_size * self.pp_size
        )
        self.dims = model_dims_from_vllm_config(self.vllm_config)
        self.expert_parallel = expert_parallel_geometry(self.vllm_config)
        #: the rank's expert-parallel group, or None for a dense model
        self.ep_ranks = expert_group_ranks(self.vllm_config)
        self._bind_expert_group()
        self.step_index = 0
        self._model_answers = _ModelAnswers(
            vllm_config=self.vllm_config,
            dims=self.dims,
            config=self.config,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
        )
        self.token_id = self._resolve_token_id()
        self.unhandled_rpcs: dict[str, int] = {}
        # Simulated workers all live on one nominal node, so the local index
        # is the global rank; a multi-node placement is a manifest concern,
        # not something this executor has to invent.
        self._workers = [
            _SimWorker(vllm_config=self.vllm_config, rank=rank, local_rank=rank)
            for rank in range(self.world_size)
        ]
        self._pending_output: Any | None = None
        self._rank_handlers = self._build_rank_handlers()
        #: RPCs answered once for the whole world rather than per rank
        self._list_handlers = {
            "compile_or_warm_up_model": self._rpc_compile_or_warm_up_model,
            "execute_model": self._rpc_execute_model,
            "sample_tokens": self._rpc_sample_tokens,
            "shutdown": self._rpc_shutdown,
        }
        logger.info(
            "SimExecutor: %d simulated workers (tp=%d, pp=%d), mode=%s, gpu=%s, "
            "pinned vLLM v%s",
            self.world_size,
            self.tp_size,
            self.pp_size,
            self.config.mode,
            self.gpu.name,
            PINNED_VLLM_VERSION,
        )

    @property
    def step_index(self) -> int:
        return self._runtime.step_index

    @step_index.setter
    def step_index(self, value: int) -> None:
        self._runtime.step_index = int(value)

    def _estimate_latency(self, translated: TranslatedStep) -> int:
        return estimate_step_latency_ps(
            self.dims,
            translated.record,
            translated.num_sampled,
            self.compute_provider,
            self.gpu,
            self.host_model,
        )

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        """Async scheduling stays off.

        vLLM v0.26.0 moved batch-queue depth onto ``VllmConfig`` (breaking
        item 4 of the M2 API review): an executor can no longer force
        ``batch_queue=None``, and returning False here is what makes the
        config post-init auto-disable async scheduling. Together with
        ``supports_pp = False`` this keeps the step loop on the simple
        ``EngineCore.step()`` path, where every ``execute_model`` output is
        consumed inline.
        """
        return False

    def check_health(self) -> None:
        return

    # RPC plumbing

    def _build_rank_handlers(self) -> dict[str, Callable[..., Any]]:
        return {
            "get_kv_cache_spec": self._rpc_get_kv_cache_spec,
            "determine_available_memory": self._rpc_determine_available_memory,
            "update_max_model_len": self._rpc_update_max_model_len,
            "initialize_from_config": self._rpc_initialize_from_config,
            "get_kv_connector_handshake_metadata": self._rpc_none,
            "get_supported_tasks": self._rpc_get_supported_tasks,
            "take_draft_token_ids": self._rpc_none,
            "execute_dummy_batch": self._rpc_none,
            "profile": self._rpc_profile,
            "reset_mm_cache": self._rpc_none,
            "reset_encoder_cache": self._rpc_none,
            "save_sharded_state": self._rpc_none,
            "sleep": self._rpc_none,
            "wake_up": self._rpc_none,
            "check_health": self._rpc_none,
            "add_lora": self._rpc_true,
            "remove_lora": self._rpc_true,
            "pin_lora": self._rpc_true,
            "list_loras": self._rpc_list_loras,
        }

    def collective_rpc(
        self,
        method: str | Callable[[Any], Any],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
    ) -> Any:
        """Service an RPC for every simulated worker.

        A callable is applied to each :class:`_SimWorker`, exactly as vLLM
        applies it to each real worker. An unknown method name yields ``None``
        per worker and is counted in :attr:`unhandled_rpcs` rather than
        raising: vLLM calls optional RPCs that a simulated executor has
        nothing to say about.
        """
        kwargs = kwargs or {}
        if callable(method):
            results: Any = [method(worker, *args, **kwargs) for worker in self._workers]
        elif method in self._list_handlers:
            results = self._list_handlers[method](*args, **kwargs)
        elif method in self._rank_handlers:
            handler = self._rank_handlers[method]
            results = [handler(rank, *args, **kwargs) for rank in range(self.world_size)]
        else:
            seen = self.unhandled_rpcs.get(method, 0)
            self.unhandled_rpcs[method] = seen + 1
            if seen == 0:
                logger.warning("SimExecutor has no handler for RPC %r, returning None", method)
            results = [None] * self.world_size
        return _completed_future(results) if non_block else results

    # Init-path RPCs

    def _rpc_none(self, rank: int, *args: Any, **kwargs: Any) -> None:
        return None

    def _rpc_true(self, rank: int, *args: Any, **kwargs: Any) -> bool:
        return True

    def _rpc_list_loras(self, rank: int, *args: Any, **kwargs: Any) -> set[int]:
        return set()

    def _pp_layer_range(self, rank: int) -> tuple[int, int]:
        """Layers owned by ``rank``, using vLLM's own uneven PP partitioning."""
        return self._model_answers.pp_layer_range(rank)

    def _kv_cache_dtype(self) -> Any:
        return self._model_answers.kv_cache_dtype()

    def _rpc_get_kv_cache_spec(self, rank: int) -> dict[str, Any]:
        """One full-attention spec per layer this rank owns.

        Layer names are fabricated but must be distinct per pipeline stage:
        ``get_kv_cache_configs`` merges the workers' specs by name and then
        projects the groups back onto each worker.
        """
        return self._model_answers.get_kv_cache_spec(rank)

    def _rpc_determine_available_memory(self, rank: int) -> int:
        return self._model_answers.determine_available_memory()

    def _rpc_update_max_model_len(self, rank: int, max_model_len: int) -> None:
        """Apply an auto-fit context shrink to replay admission guards."""
        if rank == 0:
            if self.replay is not None:
                self.replay.update_max_model_len(max_model_len)
            logger.info("SimExecutor: max_model_len updated to %d", max_model_len)

    def _rpc_initialize_from_config(self, rank: int, kv_cache_configs: Any = None) -> None:
        if rank == 0:
            # v0.26.0 dropped the initialize_cache RPC: the engine sets
            # cache_config.num_gpu_blocks itself before this call.
            logger.info(
                "SimExecutor: KV pool pinned to %s blocks of %s tokens",
                getattr(self.cache_config, "num_gpu_blocks", None),
                getattr(self.cache_config, "block_size", None),
            )

    def _rpc_compile_or_warm_up_model(self) -> list[Any]:
        """Zero compilation time per worker.

        The engine reduces with ``max(t.language_model for t in times)``, so
        both a list of ``None`` (AttributeError) and an empty list
        (``max()`` on an empty sequence) would crash it. If the import moves,
        a duck-typed stand-in with the same two fields serves the reduction.
        """
        return [self._model_answers.compilation_times() for _ in range(self.world_size)]

    def _rpc_get_supported_tasks(self, rank: int) -> tuple[str, ...]:
        return self._model_answers.supported_tasks()

    def _rpc_profile(self, rank: int, is_start: bool = True, profile_prefix: str | None = None):
        if rank == 0:
            logger.info(
                "SimExecutor: profile(is_start=%s, prefix=%s) ignored", is_start, profile_prefix
            )

    def _rpc_shutdown(self, *args: Any, **kwargs: Any) -> list[None]:
        self.shutdown()
        return [None] * self.world_size

    # Step path

    def _rpc_execute_model(self, scheduler_output: Any) -> list[Any]:
        """Step once for the whole world, as a driver rank would.

        Answered as a list handler on purpose: a per-rank handler would
        simulate (and record) the same step once per rank.
        """
        return [self._run_step(scheduler_output)] + [None] * (self.world_size - 1)

    def _rpc_sample_tokens(self, grammar_output: Any = None) -> list[Any]:
        return [self._take_pending_output()] + [None] * (self.world_size - 1)

    def execute_model(self, scheduler_output: Any, non_block: bool = False) -> Any:
        """Simulate one step and fabricate its ``ModelRunnerOutput``.

        ``EngineCore.step()`` always calls this with ``non_block=True`` and
        then immediately calls ``.result()``, so an already-completed future
        is both sufficient and honest: the simulated work is done inline.
        """
        output = self._run_step(scheduler_output)
        return _completed_future(output) if non_block else output

    def sample_tokens(self, grammar_output: Any = None, non_block: bool = False) -> Any:
        """Return the output stashed by :meth:`execute_model`, exactly once.

        With ``supports_pp = False`` and structured output refused, the
        engine never takes this path today (it reads the non-None return of
        ``execute_model`` directly); it is served defensively so a changed
        engine loop fails loudly here instead of silently dropping a step.
        """
        output = self._take_pending_output()
        return _completed_future(output) if non_block else output

    def _take_pending_output(self) -> Any:
        if self._pending_output is None:
            raise RuntimeError(
                "sample_tokens called with no pending model output: the engine "
                "step loop changed shape (every execute_model output is "
                "consumed at most once)"
            )
        output = self._pending_output
        self._pending_output = None
        return output

    def _empty_output(self) -> Any:
        from vllm.v1.outputs import ModelRunnerOutput

        return ModelRunnerOutput(req_ids=[], req_id_to_index={})

    def _resolve_token_id(self) -> int:
        """Fabricated token id: mid-vocabulary, never a stop token."""
        return _resolve_token_id(self.dims.vocab_size, self.config.token_id)

    def _bind_expert_group(self) -> None:
        """Hand the derived expert group to a sink that accepts one.

        Binding happens exactly once, before any step, and only when the active
        parallel configuration executes an all-to-all whose combine returns an
        already reduced layer output, which is what declaring the group to the
        traffic renderers asserts. Expert parallelism alone is not enough: a
        naive expert-parallel configuration, or one whose backend does not
        reduce, executes two tensor-parallel allreduces per layer and no
        all-to-all, and that is exactly what the renderers produce when no
        group is bound. Every other configuration performs no binding at all,
        so a sink keeps whatever ``ep_ranks`` its own configuration declared.
        """

        geometry = self.expert_parallel
        if (
            geometry.use_all2all_kernels
            and not geometry.combine_is_reducing
            and self.ep_ranks is not None
        ):
            raise NotImplementedError(
                "vLLM all2all_backend "
                f"{geometry.all2all_backend!r} moves expert activations through "
                "an allgather and a reduce-scatter, a traffic shape this "
                "repository renders nothing for, so declaring the expert group "
                "would price a pairwise all-to-allv this deployment never "
                "executes; tracked by TRAF-40"
            )
        if not geometry.renders_expert_combine or self.ep_ranks is None:
            return
        sink = self.step_sink
        if not isinstance(sink, ExpertGroupStepSink):
            return
        sink.bind_expert_group(self.ep_ranks)

    def _sample_output_fields(
        self,
        translated: TranslatedStep,
        scheduler_output: Any,
    ) -> tuple[list[str], dict[str, int], list[list[int]]]:
        """Select joined replay or the exact fabricated-token off path."""

        return sample_adapter_tokens(
            self.replay,
            translated.req_ids,
            translated.produces_token,
            self.token_id,
            scheduler_output,
            fabricate=fabricate_sampled_tokens,
        )

    def _run_step(self, scheduler_output: Any) -> Any:
        if getattr(scheduler_output, "has_structured_output_requests", False):
            raise RuntimeError(
                "SimExecutor does not support structured output: the grammar "
                "would reject the fabricated token id, so every such request "
                "would be terminated as FINISHED_ERROR at its first token. "
                "Remove structured-output sampling params (VLLM-8)."
            )
        if not (getattr(scheduler_output, "num_scheduled_tokens", None) or {}):
            return self._drain_step(scheduler_output)

        translated = self._runtime.translate(scheduler_output)
        if self.replay is not None:
            self.replay.validate_step(
                translated.req_ids,
                translated.produces_token,
                scheduler_output,
            )
        self._settle(translated)

        from vllm.v1.outputs import ModelRunnerOutput

        req_ids, req_id_to_index, sampled = self._sample_output_fields(
            translated, scheduler_output
        )
        self._pending_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled,
        )
        return self._pending_output

    def _drain_step(self, scheduler_output: Any) -> Any:
        """An empty batch: nothing runs, but completions still arrive here.

        The scheduler stays live while its finished set is non-empty, so
        after the last request stops the ``EngineCore`` busy loop issues one
        more step that schedules nothing and carries the final
        ``finished_req_ids`` (and a mid-run step can do the same). Dropping
        it would lose those completions, so it is recorded as a zero-cost
        step: no simulated time passes, and the sink is informed (a closed
        loop must learn of the completions) but cannot override the
        nonexistent duration. The in-process ``LLM.generate`` loop stops
        stepping before this drain step (confirmed on v0.26.0), so on that
        path the final completions never reach the executor at all and a
        record consumer infers completion from a request's last scheduled
        record.
        """
        if self.replay is not None:
            self.replay.validate_completions(scheduler_output)
        self._runtime.drain(scheduler_output)
        if self.replay is not None:
            self.replay.observe_completions(scheduler_output)
        self._pending_output = self._empty_output()
        return self._pending_output

    def _settle(self, translated: TranslatedStep) -> StepResult:
        """Record the step, ask the sink or the provider for its duration."""
        return self._runtime.settle(translated)

    # Export

    def _append_step_record(self, record: StepRecord) -> None:
        """Compatibility helper for appending one record immediately.

        vLLM does not reliably route in-process engine teardown through the
        shutdown RPC (observed on v0.26.0), so the dump must never depend on
        a teardown callback. The shared runtime calls the same stream writer
        directly so each step is durable the moment it completes.
        """
        if self._record_stream is not None:
            self._record_stream.append(record)

    def shutdown(self) -> None:
        """Nothing to flush: every record is durable when its step completes."""
        path = self.config.step_records_path
        if path and self.step_records:
            logger.info(
                "SimExecutor: %d step records streamed to %s", len(self.step_records), path
            )

    def dump_step_records(self, path: str | Path | None = None) -> Path | None:
        """Rewrite the full accumulated record set, for a caller-chosen path.

        The configured ``SIMLLM_VLLM_STEP_RECORDS`` path is already written
        incrementally by the shared step runtime; this helper exports the same
        records to a different location from in-process drivers.
        """
        target = path or self.config.step_records_path
        if not target or not self.step_records:
            return None
        written = write_step_records(self.step_records, target)
        logger.info("SimExecutor: wrote %d step records to %s", len(self.step_records), written)
        return written
