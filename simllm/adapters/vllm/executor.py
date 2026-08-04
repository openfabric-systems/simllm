"""vLLM v1 executor that replaces model execution with a simulated GPU.

Pinned to vLLM **v0.26.0**. The v1 engine resolves its executor class from a
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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from simllm.adapters.vllm._version import PINNED_VLLM_VERSION
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
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    StepRecordStream,
    StepResult,
    VirtualClock,
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
    "ModelDims",
    "SimExecutor",
    "SimExecutorConfig",
    "SimExecutorHooks",
    "StepTranslator",
    "TranslatedStep",
    "configure",
    "estimate_step_latency_ps",
    "fabricate_sampled_tokens",
    "latest_executor",
    "model_dims_from_vllm_config",
    "observe_scheduler_output",
    "step_kernel",
    "step_records_to_json",
    "translate_scheduler_output",
    "vllm_is_available",
    "write_step_records",
]

#: A step sink consumes one record and may return the simulated outcome; a
#: sink that returns ``None`` leaves the estimate to the compute provider.
StepSink = Callable[[StepRecord], "StepResult | None"]


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

    def __post_init__(self) -> None:
        if self.mode not in ("virtual", "paced"):
            raise ValueError(f"SIMLLM_VLLM_MODE must be virtual or paced, got {self.mode!r}")
        if self.kv_memory_bytes <= 0:
            raise ValueError("SIMLLM_VLLM_KV_MEMORY_BYTES must be positive")

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

    def __init__(self) -> None:
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


def _completed_future(value: Any) -> Future:
    future: Future = Future()
    future.set_result(value)
    return future


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
    ) -> None:
        if _VLLM_IMPORT_ERROR is not None:
            raise _missing_vllm_error()
        self.config = config or _HOOKS.config or SimExecutorConfig.from_env()
        self.gpu = gpu or _HOOKS.gpu or self.config.gpu_spec()
        self.compute_provider: ComputeProvider = (
            compute_provider
            or _HOOKS.compute_provider
            or RooflineProvider(efficiency=self.config.efficiency)
        )
        self.host_model = (
            host_model
            or _HOOKS.host_model
            or HostInitiationModel(initiation_delay_ps=self.config.host_initiation_ps)
        )
        self.step_sink: StepSink | None = step_sink or _HOOKS.step_sink
        #: every translated step, in order; the offline mode renders these
        self.step_records: list[StepRecord] = []
        self._record_stream = (
            StepRecordStream(self.config.step_records_path)
            if self.config.step_records_path
            else None
        )
        #: simulated outcome per recorded step, parallel to step_records
        self.step_results: list[StepResult] = []
        self.clock = VirtualClock()
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
        self.translator = StepTranslator()
        self.step_index = 0
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
            "reset_prefix_cache": self._rpc_none,
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
        total = int(_safe(self.model_config.get_total_num_hidden_layers, self.dims.num_layers))
        pp_rank = (rank // max(self.tp_size, 1)) % max(self.pp_size, 1)

        def _indices() -> tuple[int, int]:
            from vllm.distributed.utils import get_pp_indices

            return get_pp_indices(total, pp_rank, self.pp_size)

        even = total // max(self.pp_size, 1)
        fallback = (pp_rank * even, total if pp_rank == self.pp_size - 1 else (pp_rank + 1) * even)
        return _safe(_indices, fallback)

    def _kv_cache_dtype(self) -> Any:
        import torch

        cache_dtype = getattr(self.cache_config, "cache_dtype", "auto")
        quantized = {
            "fp8": "float8_e4m3fn",
            "fp8_e4m3": "float8_e4m3fn",
            "fp8_e5m2": "float8_e5m2",
        }.get(str(cache_dtype))
        if quantized is not None:
            resolved = getattr(torch, quantized, None)
            if resolved is not None:
                return resolved
        model_dtype = getattr(self.model_config, "dtype", None)
        return model_dtype if isinstance(model_dtype, torch.dtype) else torch.bfloat16

    def _rpc_get_kv_cache_spec(self, rank: int) -> dict[str, Any]:
        """One full-attention spec per layer this rank owns.

        Layer names are fabricated but must be distinct per pipeline stage:
        ``get_kv_cache_configs`` merges the workers' specs by name and then
        projects the groups back onto each worker.
        """
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        block_size = int(getattr(self.cache_config, "block_size", None) or 16)
        spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.dims.num_kv_heads,
            head_size=self.dims.head_size,
            dtype=self._kv_cache_dtype(),
        )
        start, end = self._pp_layer_range(rank)
        return {f"model.layers.{layer}.self_attn.attn": spec for layer in range(start, end)}

    def _rpc_determine_available_memory(self, rank: int) -> int:
        return self.config.kv_memory_bytes

    def _rpc_update_max_model_len(self, rank: int, max_model_len: int) -> None:
        """Auto-fit shrank the context; nothing to resize on a fake worker."""
        if rank == 0:
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
        try:
            from vllm.v1.worker.worker_base import CompilationTimes
        except ImportError:
            CompilationTimes = _CompilationTimesFallback
        return [
            CompilationTimes(language_model=0.0, encoder=0.0) for _ in range(self.world_size)
        ]

    def _rpc_get_supported_tasks(self, rank: int) -> tuple[str, ...]:
        runner_type = getattr(self.model_config, "runner_type", "generate")
        return ("embed",) if runner_type == "pooling" else ("generate",)

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
        vocab_size = max(self.dims.vocab_size, 2)
        if self.config.token_id is not None:
            return max(0, min(self.config.token_id, vocab_size - 1))
        return vocab_size // 2

    def _run_step(self, scheduler_output: Any) -> Any:
        structured = getattr(scheduler_output, "structured_output_request_ids", None)
        if structured:
            raise RuntimeError(
                "SimExecutor does not support structured output: the grammar "
                "would reject the fabricated token id, so every such request "
                "would be terminated as FINISHED_ERROR at its first token. "
                "Remove structured-output sampling params (VLLM-8)."
            )
        if not (getattr(scheduler_output, "num_scheduled_tokens", None) or {}):
            return self._drain_step(scheduler_output)

        translated = translate_scheduler_output(
            self.translator,
            scheduler_output,
            step_index=self.step_index,
            virtual_time_ps=self.clock.now_ps,
        )
        self.step_index += 1
        self._settle(translated)

        from vllm.v1.outputs import ModelRunnerOutput

        req_ids, req_id_to_index, sampled = fabricate_sampled_tokens(
            translated.req_ids, translated.produces_token, self.token_id
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
        finished = getattr(scheduler_output, "finished_req_ids", ()) or ()
        preempted = getattr(scheduler_output, "preempted_req_ids", None) or ()
        if finished or preempted:
            translated = translate_scheduler_output(
                self.translator,
                scheduler_output,
                step_index=self.step_index,
                virtual_time_ps=self.clock.now_ps,
            )
            self.step_index += 1
            record = translated.record
            if self.step_sink is not None:
                self.step_sink(record)
            self.step_records.append(record)
            self.step_results.append(
                StepResult(
                    step_index=record.step_index,
                    step_latency_ps=0,
                    completed_at_ps=self.clock.now_ps,
                )
            )
            self._append_step_record(record)
        self._pending_output = self._empty_output()
        return self._pending_output

    def _settle(self, translated: TranslatedStep) -> StepResult:
        """Record the step, ask the sink or the provider for its duration."""
        record = translated.record
        result = self.step_sink(record) if self.step_sink is not None else None
        if result is None:
            latency_ps = estimate_step_latency_ps(
                self.dims,
                record,
                translated.num_sampled,
                self.compute_provider,
                self.gpu,
                self.host_model,
            )
            result = StepResult(
                step_index=record.step_index,
                step_latency_ps=latency_ps,
                completed_at_ps=self.clock.now_ps + latency_ps,
            )
        self.step_records.append(record)
        self.step_results.append(result)
        self._append_step_record(record)
        self.clock.advance_to(max(result.completed_at_ps, self.clock.now_ps))
        if self.config.mode == "paced" and result.step_latency_ps > 0:
            time.sleep(result.step_latency_ps / PS_PER_SECOND)
        return result

    # Export

    def _append_step_record(self, record: StepRecord) -> None:
        """Append one record to the configured JSONL path as it happens.

        vLLM does not reliably route in-process engine teardown through the
        shutdown RPC (observed on v0.26.0), so the dump must never depend on
        a teardown callback: each step is durable the moment it completes.
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
        incrementally by :meth:`_append_step_record` (the sole producer for
        it); this helper exports the same records to a different location
        from in-process drivers.
        """
        target = path or self.config.step_records_path
        if not target or not self.step_records:
            return None
        written = write_step_records(self.step_records, target)
        logger.info("SimExecutor: wrote %d step records to %s", len(self.step_records), written)
        return written
