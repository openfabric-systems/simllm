"""SGLang TP model worker that replaces model execution with a simulated GPU.

Pinned to SGLang main at commit ``8f2a3ad`` (2026-08-04). The seam is the TP
worker: the scheduler constructs it in ``Scheduler.init_tp_model_worker``,
the same point where SGLang swaps in its MLX worker on Apple Silicon, and
SGLang's plugin framework can replace that method without a fork (see
:mod:`simllm.adapters.sglang.plugin`)::

    SIMLLM_SGLANG_ENABLE=1 SIMLLM_SGLANG_MODE=virtual SIMLLM_SGLANG_GPU=b100 \\
    SIMLLM_SGLANG_STEP_RECORDS=... \\
    python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B \\
        --disable-overlap-schedule --max-total-tokens 32768

The scheduler, RadixCache prefix matching, token/request pool accounting and
retraction logic all run for real; only model execution is fabricated. Every
``forward_batch_generation`` call becomes one
:class:`simllm.core.StepRecord`, handed to an injected sink, accumulated on
the worker, and streamed to the configured JSONL.

This module must stay importable without SGLang: the ``sglang`` imports live
in a guarded block and inside methods.

==============================  ==============================================
Environment variable            Meaning (default)
==============================  ==============================================
``SIMLLM_SGLANG_ENABLE``        ``1`` activates the plugin hook that installs
                                this worker; any other value leaves SGLang
                                untouched (unset). Read by the plugin, listed
                                here for completeness.
``SIMLLM_SGLANG_MODE``          ``virtual`` returns immediately and reports
                                sim-native timings; ``paced`` sleeps the
                                estimated step latency so SGLang's own
                                wall-clock metrics stay meaningful
                                (``virtual``).
``SIMLLM_SGLANG_GPU``           GPU envelope for the default roofline
                                estimate: ``a100``, ``h100``, ``h200``,
                                ``b100`` or ``b200`` (``b100``).
``SIMLLM_SGLANG_PEAK_FLOPS``    override the envelope's dense FLOP/s (unset).
``SIMLLM_SGLANG_MEM_BANDWIDTH`` override the envelope's HBM bytes/s (unset).
``SIMLLM_SGLANG_EFFICIENCY``    roofline derate in (0, 1] (``0.7``).
``SIMLLM_SGLANG_HOST_INIT_PS``  per-step host initiation delay in ps (``0``).
``SIMLLM_SGLANG_TOKEN_ID``      token id fabricated for every generated token
                                (``vocab_size // 2``).
``SIMLLM_SGLANG_STEP_RECORDS``  JSONL path for the step records; each record
                                is appended the moment its step completes
                                (unset).
``SIMLLM_SGLANG_COMMUNICATOR_TP_SIZE``
                                logical TP group size for the optional
                                zero-time communicator observation; unset
                                preserves the existing worker exactly.
``SIMLLM_SGLANG_COMMUNICATOR_EVENTS``
                                JSONL path for communicator events from the
                                scheduler process; requires the logical TP
                                size (unset).
==============================  ==============================================

The simulated KV pool size comes from SGLang's own ``--max-total-tokens``
flag (falling back to the model context length), so KV pressure, radix
eviction and retraction respond to it exactly as in production.

Configurations the fabricated token would silently corrupt are refused
loudly instead: a batch with ``return_logprob`` raises (the fabricated
``LogitsProcessorOutput`` has no logprobs and SGLang dereferences them
unguarded), and speculative decoding never reaches this worker because the
scheduler only routes ``spec_algorithm.is_none()`` runs through the plain TP
worker (SGL-5 in docs/modules/adapters-sglang.md).

Completion visibility: the worker never sees finish decisions (SGLang
applies EOS / ``max_new_tokens`` in ``process_batch_result`` after the
forward returns), so ``finished_request_ids`` is always empty and a record
consumer infers a request's completion from the last record that schedules
it, the same convention as the vLLM adapter's in-process path.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from simllm.adapters.sglang._version import PINNED_SGLANG_COMMIT
from simllm.adapters.sglang.communicator import (
    FLOAT32,
    SGLANG_TP_PAYLOAD_BYTES,
    GroupCoordinatorEvent,
    GroupCoordinatorEventStream,
    GroupCoordinatorObserver,
    ShapeTensor,
    SimGroupCoordinator,
)
from simllm.compute import (
    GPU_ENVELOPES,
    PS_PER_SECOND,
    ComputeProvider,
    GpuSpec,
    HostInitiationModel,
    ModelDims,
    NcclStackConfig,
    RooflineProvider,
    estimate_step_latency_ps,
)
from simllm.core import (
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    StepRecordStream,
    StepResult,
    VirtualClock,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PINNED_SGLANG_COMMIT",
    "BatchRow",
    "SglStepTranslator",
    "SimTpModelWorker",
    "SimWorkerConfig",
    "SimWorkerHooks",
    "configure",
    "latest_worker",
    "model_dims_from_sglang",
    "observe_schedule_batch",
    "sglang_is_available",
]

#: A step sink consumes one record and may return the simulated outcome; a
#: sink that returns ``None`` leaves the estimate to the compute provider.
StepSink = Callable[[StepRecord], "StepResult | None"]


def _missing_sglang_error() -> ImportError:
    detail = f" ({_SGLANG_IMPORT_ERROR})" if _SGLANG_IMPORT_ERROR is not None else ""
    return ImportError(
        f"simllm.adapters.sglang requires SGLang (pinned commit "
        f"{PINNED_SGLANG_COMMIT}) installed in the same environment{detail}. "
        "The worker is constructed inside SGLang's scheduler process via the "
        "simllm plugin hook, so it is only constructible there."
    )


try:
    from sglang.srt.managers.tp_worker import TpModelWorker as _TpWorkerBase
    from sglang.srt.model_executor.model_runner import ModelRunner as _ModelRunnerBase

    _SGLANG_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # SGLang absent: keep the module importable
    _SGLANG_IMPORT_ERROR = exc

    class _TpWorkerBase:  # type: ignore[no-redef]
        """Stand-in for SGLang's ``TpModelWorker`` without SGLang installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise _missing_sglang_error()

    class _ModelRunnerBase:  # type: ignore[no-redef]
        """Stand-in for SGLang's ``ModelRunner`` without SGLang installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise _missing_sglang_error()


def sglang_is_available() -> bool:
    """Whether the SGLang base classes imported successfully."""
    return _SGLANG_IMPORT_ERROR is None


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
class SimWorkerConfig:
    """Worker settings that have no SGLang CLI flag.

    Constructed from ``SIMLLM_SGLANG_*`` environment variables; see the
    module docstring for the full table.
    """

    mode: str = "virtual"
    gpu: str = "b100"
    peak_flops: float | None = None
    mem_bandwidth: float | None = None
    efficiency: float = 0.7
    host_initiation_ps: int = 0
    token_id: int | None = None
    step_records_path: str | None = None
    communicator_tp_size: int | None = None
    communicator_events_path: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("virtual", "paced"):
            raise ValueError(
                f"SIMLLM_SGLANG_MODE must be virtual or paced, got {self.mode!r}"
            )
        if self.communicator_tp_size is not None:
            if self.communicator_tp_size <= 0:
                raise ValueError("SIMLLM_SGLANG_COMMUNICATOR_TP_SIZE must be positive")
            if SGLANG_TP_PAYLOAD_BYTES % self.communicator_tp_size:
                raise ValueError(
                    "SIMLLM_SGLANG_COMMUNICATOR_TP_SIZE must divide the fixed "
                    f"{SGLANG_TP_PAYLOAD_BYTES}-byte TP observation"
                )
        if self.communicator_events_path and self.communicator_tp_size is None:
            raise ValueError(
                "SIMLLM_SGLANG_COMMUNICATOR_EVENTS requires "
                "SIMLLM_SGLANG_COMMUNICATOR_TP_SIZE"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> SimWorkerConfig:
        env = os.environ if env is None else env
        return cls(
            mode=_env_str(env, "SIMLLM_SGLANG_MODE", "virtual") or "virtual",
            gpu=(_env_str(env, "SIMLLM_SGLANG_GPU", "b100") or "b100").lower(),
            peak_flops=_env_float(env, "SIMLLM_SGLANG_PEAK_FLOPS", None),
            mem_bandwidth=_env_float(env, "SIMLLM_SGLANG_MEM_BANDWIDTH", None),
            efficiency=_env_float(env, "SIMLLM_SGLANG_EFFICIENCY", 0.7) or 0.7,
            host_initiation_ps=_env_int(env, "SIMLLM_SGLANG_HOST_INIT_PS", 0) or 0,
            token_id=_env_int(env, "SIMLLM_SGLANG_TOKEN_ID", None),
            step_records_path=_env_str(env, "SIMLLM_SGLANG_STEP_RECORDS", None),
            communicator_tp_size=_env_int(
                env, "SIMLLM_SGLANG_COMMUNICATOR_TP_SIZE", None
            ),
            communicator_events_path=_env_str(
                env, "SIMLLM_SGLANG_COMMUNICATOR_EVENTS", None
            ),
        )

    def gpu_spec(self) -> GpuSpec:
        """GPU envelope with the per-field environment overrides applied."""
        base = GPU_ENVELOPES.get(self.gpu)
        if base is None:
            known = ", ".join(sorted(GPU_ENVELOPES))
            raise ValueError(f"unknown SIMLLM_SGLANG_GPU {self.gpu!r}; known: {known}")
        return GpuSpec(
            name=base.name,
            peak_flops=self.peak_flops or base.peak_flops,
            mem_bandwidth=self.mem_bandwidth or base.mem_bandwidth,
        )


@dataclass
class SimWorkerHooks:
    """Process-wide injection point for objects no CLI flag can carry.

    SGLang constructs the worker inside its scheduler process, so a caller
    that drives the scheduler in this process sets the hooks first with
    :func:`configure`.
    """

    step_sink: StepSink | None = None
    compute_provider: ComputeProvider | None = None
    gpu: GpuSpec | None = None
    host_model: HostInitiationModel | None = None
    config: SimWorkerConfig | None = None


_HOOKS = SimWorkerHooks()
#: at most one entry, the worker most recently built in this process
_LATEST: list[SimTpModelWorker] = []


def configure(
    *,
    step_sink: StepSink | None = None,
    compute_provider: ComputeProvider | None = None,
    gpu: GpuSpec | None = None,
    host_model: HostInitiationModel | None = None,
    config: SimWorkerConfig | None = None,
) -> SimWorkerHooks:
    """Set the hooks the next :class:`SimTpModelWorker` in this process uses.

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


def latest_worker() -> SimTpModelWorker | None:
    """The most recently constructed worker in this process, if any."""
    return _LATEST[-1] if _LATEST else None


# Batch translation (pure: no SGLang types cross this boundary)

@dataclass(frozen=True)
class BatchRow:
    """One request's share of a batch, in framework-neutral terms."""

    rid: str
    is_decode: bool
    num_new_tokens: int
    #: total KV context length after this step
    context_length: int
    #: cumulative radix-cache hit of this request (``Req.cached_tokens``);
    #: the translator reports it once, on the request's first record
    cached_tokens: int = 0


def _context_length(batch: Any, index: int, req: Any) -> int:
    """Post-step context of one request, preferring the batch tensor.

    ``seq_lens_cpu`` is authoritative on both extend and decode batches
    (``prepare_for_decode`` has already added this step's token); the
    per-request ``kv_committed_len`` is the fallback for stub batches.
    """
    seq_lens_cpu = getattr(batch, "seq_lens_cpu", None)
    if seq_lens_cpu is not None:
        try:
            return int(seq_lens_cpu[index])
        except (IndexError, TypeError, ValueError):
            pass
    committed = getattr(req, "kv_committed_len", None)
    if isinstance(committed, int) and committed > 0:
        return committed
    origin = getattr(req, "origin_input_ids", ()) or ()
    output = getattr(req, "output_ids", ()) or ()
    return len(origin) + len(output)


def observe_schedule_batch(batch: Any) -> list[BatchRow]:
    """Read one ``ScheduleBatch`` into plain rows, in ``reqs`` order.

    Every field is read with getattr so a stub with the same attribute
    names works. Facts this encodes (verified at the pinned commit):

    - ``forward_mode.is_extend()`` covers EXTEND and MIXED; a MIXED batch
      appends its running decode requests *after* the prefill requests and
      lists them in ``decoding_reqs``, with an ``extend_lens`` entry of 1.
      Those rows are logically decode and their ``prefix_lens`` entry is
      synthetic, never a radix hit.
    - ``extend_lens`` and ``prefix_lens`` are host-side ``List[int]`` on the
      ScheduleBatch (the tensor twins with other names live on
      ``ForwardBatch``, which this adapter never builds).
    - On a decode batch ``extend_lens`` holds stale values and must not be
      read; every row computes exactly one token.
    - ``Req.cached_tokens`` is final by forward time (``prepare_for_extend``
      accounts the admission-time radix hit before ``run_batch``).
    """
    reqs = list(getattr(batch, "reqs", ()) or ())
    if not reqs:
        return []
    mode = getattr(batch, "forward_mode", None)

    def _mode_is(name: str) -> bool:
        probe = getattr(mode, name, None)
        return bool(probe()) if callable(probe) else False

    rows: list[BatchRow] = []
    if _mode_is("is_decode"):
        for index, req in enumerate(reqs):
            rows.append(
                BatchRow(
                    rid=str(getattr(req, "rid", index)),
                    is_decode=True,
                    num_new_tokens=1,
                    context_length=_context_length(batch, index, req),
                )
            )
        return rows
    if not _mode_is("is_extend"):
        # IDLE and the speculative/dLLM modes never reach this worker.
        return []

    decoding_rids = {
        str(getattr(req, "rid", id(req)))
        for req in (getattr(batch, "decoding_reqs", None) or ())
    }
    extend_lens = list(getattr(batch, "extend_lens", ()) or ())
    for index, req in enumerate(reqs):
        rid = str(getattr(req, "rid", index))
        num_new = int(extend_lens[index]) if index < len(extend_lens) else 1
        is_decode_row = rid in decoding_rids
        rows.append(
            BatchRow(
                rid=rid,
                is_decode=is_decode_row,
                num_new_tokens=max(num_new, 1),
                context_length=_context_length(batch, index, req),
                cached_tokens=(
                    0 if is_decode_row else int(getattr(req, "cached_tokens", 0) or 0)
                ),
            )
        )
    return rows


class SglStepTranslator:
    """Batch rows to :class:`~simllm.core.StepRecord`.

    Tracks which request ids already reported their radix hit so
    ``num_cached_tokens`` appears exactly once per request (on its first
    record), matching the vLLM adapter's convention. The id set grows with
    the total number of requests seen; there is no finish signal at this
    seam to prune on (see the module docstring).
    """

    def __init__(self) -> None:
        self._cached_reported: set[str] = set()

    def __len__(self) -> int:
        return len(self._cached_reported)

    def translate(
        self, *, step_index: int, virtual_time_ps: int, rows: list[BatchRow]
    ) -> StepRecord:
        scheduled: list[ScheduledRequest] = []
        for row in rows:
            report_cache = (
                not row.is_decode
                and row.cached_tokens > 0
                and row.rid not in self._cached_reported
            )
            if not row.is_decode:
                # Only a prefill appearance consumes the one-time report: a
                # mixed-batch decode row must not eat the hit of a request
                # that later resumes as a prefill after retraction.
                self._cached_reported.add(row.rid)
            scheduled.append(
                ScheduledRequest(
                    request_id=row.rid,
                    phase=RequestPhase.DECODE if row.is_decode else RequestPhase.PREFILL,
                    num_new_tokens=row.num_new_tokens,
                    num_cached_tokens=row.cached_tokens if report_cache else 0,
                    context_length=row.context_length,
                )
            )
        return StepRecord(
            step_index=step_index,
            virtual_time_ps=virtual_time_ps,
            scheduled=scheduled,
        )


# Geometry

def model_dims_from_sglang(model_config: Any, tp_size: int = 1) -> ModelDims:
    """Read the per-rank geometry off an SGLang ``ModelConfig``.

    The geometry class itself is framework-neutral
    (:class:`simllm.compute.ModelDims`); this reader knows SGLang's accessor
    names. Every accessor has a Llama-8B-shaped fallback so a renamed
    internal degrades the estimate instead of taking the run down, but a
    fallback is never silent: the defaulted field names are warned once and
    stamped on ``ModelDims.defaulted_fields``.
    """
    tp_size = max(int(tp_size or 1), 1)
    defaulted: list[str] = []

    def geom(name: str, getter: Callable[[], Any], default: Any) -> Any:
        try:
            value = getter()
        except Exception as exc:  # noqa: BLE001 - SGLang internals move quickly
            logger.debug("SGLang accessor for %s failed: %s", name, exc)
            value = None
        if value is None:
            defaulted.append(name)
            return default
        return value

    hf = getattr(model_config, "hf_text_config", None) or getattr(
        model_config, "hf_config", None
    )
    hidden_size = int(geom("hidden_size", lambda: getattr(hf, "hidden_size", None), 4096))
    num_heads_total = int(
        geom(
            "num_heads",
            lambda: getattr(model_config, "num_attention_heads", None)
            or getattr(hf, "num_attention_heads", None),
            32,
        )
    )

    def _kv_heads() -> Any:
        getter = getattr(model_config, "get_num_kv_heads", None)
        if callable(getter):
            return getter(tp_size)
        total = getattr(hf, "num_key_value_heads", None)
        return None if total is None else max(int(total) // tp_size, 1)

    dtype = getattr(model_config, "dtype", None)
    dims = ModelDims(
        num_layers=int(
            geom(
                "num_layers",
                lambda: getattr(model_config, "num_hidden_layers", None)
                or getattr(hf, "num_hidden_layers", None),
                32,
            )
        ),
        hidden_size=hidden_size,
        intermediate_size=max(
            int(
                geom(
                    "intermediate_size",
                    lambda: getattr(hf, "intermediate_size", None),
                    4 * hidden_size,
                )
            )
            // tp_size,
            1,
        ),
        num_heads=max(num_heads_total // tp_size, 1),
        num_kv_heads=int(geom("num_kv_heads", _kv_heads, 8)),
        head_size=int(
            geom(
                "head_size",
                lambda: getattr(model_config, "head_dim", None)
                or getattr(hf, "head_dim", None)
                or (hidden_size // num_heads_total if num_heads_total else None),
                128,
            )
        ),
        vocab_size=int(
            geom(
                "vocab_size",
                lambda: getattr(model_config, "vocab_size", None)
                or getattr(hf, "vocab_size", None),
                128256,
            )
        ),
        dtype_bytes=int(geom("dtype_bytes", lambda: getattr(dtype, "itemsize", None), 2)),
        defaulted_fields=tuple(defaulted),
    )
    if defaulted:
        logger.warning(
            "ModelDims: geometry fields defaulted to Llama-8B-shaped values: %s. "
            "Latency terms driven by them describe a phantom geometry.",
            ", ".join(defaulted),
        )
    return dims


# The runner stub and the worker (SGLang required beyond this point)

class SimModelRunnerStub(_ModelRunnerBase):
    """``ModelRunner`` that loads no weights and allocates no KV tensors.

    Modeled on SGLang's own ``MlxModelRunnerStub`` (the in-tree proof that
    the scheduler runs against fabricated pools): ``__init__`` is inherited,
    so device selection, ``init_torch_distributed`` (gloo on CPU) and the
    ``forward_stream`` the scheduler adopts all happen for real; only
    ``initialize`` and ``load_model`` are replaced with metadata-only
    versions, and ``alloc_memory_pool`` becomes a no-op because
    ``initialize`` already built the pools.
    """

    #: The base runner installs these in its full ``initialize``; the
    #: scheduler reads both (``canary_manager`` guarded, ``prefill_aware_swa``
    #: unconditionally on every prefill admission), so pin them here.
    canary_manager = None
    prefill_aware_swa = False

    def bind_simulated_tp_group(
        self,
        group: SimGroupCoordinator,
        *,
        event_stream: GroupCoordinatorEventStream | None = None,
    ) -> None:
        """Bind the optional logical TP group after base runner construction."""

        if not isinstance(group, SimGroupCoordinator):
            raise TypeError("group must be an SGLang SimGroupCoordinator")
        self.simulated_tp_group = group
        self._coordinator_event_stream = event_stream

    @property
    def coordinator_events(self) -> tuple[GroupCoordinatorEvent, ...]:
        group = getattr(self, "simulated_tp_group", None)
        return () if group is None else group.events

    def observe_tp_step(self) -> GroupCoordinatorEvent | None:
        """Emit the fixed shape-only TP boundary for one model step."""

        group = getattr(self, "simulated_tp_group", None)
        if group is None:
            return None
        tensor = ShapeTensor(
            (1, SGLANG_TP_PAYLOAD_BYTES // FLOAT32.itemsize),
            dtype=FLOAT32,
            element_size_bytes=FLOAT32.itemsize,
        )
        group.all_reduce(tensor)
        event = group.events[-1]
        stream = getattr(self, "_coordinator_event_stream", None)
        if stream is not None:
            stream.append(event)
        return event

    def load_model(self) -> None:
        """Metadata only; no weights. ``support_pp`` probes the forward
        signature, so a stand-in model with a no-arg ``forward`` resolves it
        to False, which matches the adapter's PP stance."""

        class _SimModel:
            @staticmethod
            def forward() -> None:
                return None

        self.model = _SimModel()
        self.sliding_window_size = None
        chunk = getattr(self.model_config, "attention_chunk_size", None)
        if getattr(self.model_config, "is_hybrid_swa", False) and getattr(
            self.model_config, "sliding_window_size", None
        ):
            self.sliding_window_size = self.model_config.sliding_window_size
        elif chunk is not None:
            self.sliding_window_size = chunk
        self.dtype = self.model_config.dtype
        self.weight_load_mem_usage = 0

    def _schedule_ctx(self) -> Any:
        try:
            from sglang.srt.runtime_context import get_schedule

            return get_schedule()
        except Exception:  # noqa: BLE001 - context bag is newer than some trees
            return self.server_args

    def initialize(self) -> None:
        from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
        from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
        from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

        class _SimKVCache(KVCache):
            """KV cache that owns no buffers; every accessor raises.

            Bypasses ``KVCache.__init__`` (which may touch CUDA memory
            pools) and hand-sets the attributes the allocator and scheduler
            read, exactly as SGLang's ``_DummyKVCache`` does.
            """

            def __init__(self, size: int, dtype: Any, device: str) -> None:
                self.size = size
                self.page_size = 1
                self.dtype = dtype
                self.store_dtype = dtype
                self.device = device
                self.layer_num = 0
                self.start_layer = 0
                self.end_layer = 0
                self.mem_usage = 0
                self.cpu_offloading_chunk_size = 8192
                self.layer_transfer_counter = None
                self.enable_custom_mem_pool = False
                self.custom_mem_pool = None

            def get_key_buffer(self, layer_id: int) -> Any:
                raise RuntimeError("SimModelRunnerStub owns no KV buffers")

            def get_value_buffer(self, layer_id: int) -> Any:
                raise RuntimeError("SimModelRunnerStub owns no KV buffers")

            def get_kv_buffer(self, layer_id: int) -> Any:
                raise RuntimeError("SimModelRunnerStub owns no KV buffers")

            def set_kv_buffer(self, layer: Any, loc: Any, cache_k: Any, cache_v: Any) -> None:
                raise RuntimeError("SimModelRunnerStub owns no KV buffers")

            def get_kv_size_bytes(self) -> tuple[int, int]:
                return 0, 0

        try:
            from sglang.srt.runtime_context import get_exec

            memory_saver = bool(get_exec().features.enable_memory_saver)
        except Exception:  # noqa: BLE001 - context bag is newer than some trees
            memory_saver = bool(getattr(self.server_args, "enable_memory_saver", False))
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(enable=memory_saver)

        self.sampler = None
        self.load_model()

        num_layers = int(
            getattr(self.model_config, "num_hidden_layers", 0)
            or getattr(self.model_config, "num_attention_layers", 0)
            or 1
        )
        try:
            from sglang.srt.model_executor.model_runner_components.layer_setup import (
                ModelLayerInfo,
            )

            self.layer_info = ModelLayerInfo(
                start_layer=0, end_layer=num_layers, num_effective_layers=num_layers
            )
        except ImportError:
            from types import SimpleNamespace

            self.layer_info = SimpleNamespace(
                start_layer=0, end_layer=num_layers, num_effective_layers=num_layers
            )

        self.kv_cache_dtype = self.dtype
        schedule = self._schedule_ctx()
        pool_size = getattr(schedule, "max_total_tokens", None)
        self.max_total_num_tokens = int(pool_size or self.model_config.context_len)
        # get_tokens_per_layer_info reads these unconditionally; SGLang's own
        # MLX stub forgets them (latent AttributeError upstream).
        self.full_max_total_num_tokens = self.max_total_num_tokens
        self.swa_max_total_num_tokens = self.max_total_num_tokens
        self.is_hybrid_swa = False

        capacity_cap = max(self.max_total_num_tokens // 2, 1)
        requested = getattr(schedule, "max_running_requests", None)
        if requested is None:
            self.max_running_requests = min(capacity_cap, 4096)
        else:
            per_worker = max(
                int(requested) // max(int(getattr(self.ps, "attn_dp_size", 1) or 1), 1), 1
            )
            self.max_running_requests = min(per_worker, capacity_cap)

        self.req_to_token_pool = ReqToTokenPool(
            size=self.max_running_requests,
            max_context_len=self.model_config.context_len,
            device="cpu",
            enable_memory_saver=False,
        )
        sim_kv = _SimKVCache(
            size=self.max_total_num_tokens, dtype=self.kv_cache_dtype, device="cpu"
        )
        self.token_to_kv_pool = sim_kv
        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            size=self.max_total_num_tokens,
            dtype=self.kv_cache_dtype,
            device="cpu",
            kvcache=sim_kv,
            need_sort=False,
        )

        self.decode_cuda_graph_runner = None
        self.graph_mem_usage = 0
        self.attn_backend = None
        self.init_ngram_embedding_manager()
        logger.info(
            "SimModelRunnerStub: pools fabricated (max_total_num_tokens=%d, "
            "max_running_requests=%d, zero KV allocation)",
            self.max_total_num_tokens,
            self.max_running_requests,
        )

    def alloc_memory_pool(self, *args: Any, **kwargs: Any) -> None:
        """No-op: ``initialize`` already built the fabricated pools."""

    def init_attention_backends(self, *args: Any, **kwargs: Any) -> None:
        """No-op: there is no attention to back."""

    def init_cuda_graphs(self, *args: Any, **kwargs: Any) -> None:
        """No-op: nothing to capture."""


class SimTpModelWorker(_TpWorkerBase):
    """SGLang TP worker whose forward passes are simulated, not run.

    Installed by :mod:`simllm.adapters.sglang.plugin` (a ``REPLACE`` hook on
    ``Scheduler.init_tp_model_worker``). Inherits ``TpModelWorker`` for all
    scheduler bookkeeping, the same pattern as SGLang's MLX worker, and
    overrides model-runner construction and the forward itself.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _SGLANG_IMPORT_ERROR is not None:
            raise _missing_sglang_error()
        self.sim_config = _HOOKS.config or SimWorkerConfig.from_env()
        self.sim_gpu = _HOOKS.gpu or self.sim_config.gpu_spec()
        self.compute_provider: ComputeProvider = (
            _HOOKS.compute_provider or RooflineProvider(efficiency=self.sim_config.efficiency)
        )
        self.host_model = _HOOKS.host_model or HostInitiationModel(
            initiation_delay_ps=self.sim_config.host_initiation_ps
        )
        self.step_sink: StepSink | None = _HOOKS.step_sink
        self.step_records: list[StepRecord] = []
        self.step_results: list[StepResult] = []
        self._record_stream = (
            StepRecordStream(self.sim_config.step_records_path)
            if self.sim_config.step_records_path
            else None
        )
        self.clock = VirtualClock()
        self.translator = SglStepTranslator()
        self.step_index = 0
        self.simulated_tp_group: SimGroupCoordinator | None = None
        self.coordinator_observer: GroupCoordinatorObserver | None = None
        self._coordinator_event_stream: GroupCoordinatorEventStream | None = None
        super().__init__(*args, **kwargs)
        tp_size = int(getattr(self.server_args, "tp_size", 1) or 1)
        self.dims = model_dims_from_sglang(self.model_config, tp_size)
        self.token_id = self._resolve_token_id()
        self._configure_simulated_communicator()
        _LATEST[:] = [self]
        logger.info(
            "SimTpModelWorker: mode=%s, gpu=%s, pinned SGLang commit %s",
            self.sim_config.mode,
            self.sim_gpu.name,
            PINNED_SGLANG_COMMIT,
        )

    def _configure_simulated_communicator(self) -> None:
        logical_size = self.sim_config.communicator_tp_size
        if logical_size is None:
            return
        logical_rank = int(getattr(self.ps, "tp_rank", 0) or 0)
        if logical_rank >= logical_size:
            raise ValueError(
                f"SGLang TP rank {logical_rank} is outside simulated TP size "
                f"{logical_size}"
            )
        self.coordinator_observer = GroupCoordinatorObserver(self.clock)
        self.simulated_tp_group = SimGroupCoordinator(
            group_name="tp",
            ranks=tuple(range(logical_size)),
            rank=logical_rank,
            local_rank=int(getattr(self.ps, "gpu_id", self.gpu_id) or 0),
            clock=self.clock,
            observer=self.coordinator_observer,
            stack_config=NcclStackConfig(
                channel_count=1,
                chunk_bytes=SGLANG_TP_PAYLOAD_BYTES // logical_size,
                fifo_slots_per_channel=2,
            ),
        )
        if self.sim_config.communicator_events_path:
            self._coordinator_event_stream = GroupCoordinatorEventStream(
                self.sim_config.communicator_events_path
            )
        self._model_runner.bind_simulated_tp_group(
            self.simulated_tp_group,
            event_stream=self._coordinator_event_stream,
        )

    @property
    def coordinator_events(self) -> tuple[GroupCoordinatorEvent, ...]:
        return self._model_runner.coordinator_events

    def _init_model_runner(self) -> None:
        try:
            from sglang.srt.runtime_context import get_schedule

            mem_fraction = get_schedule().mem_fraction_static
        except Exception:  # noqa: BLE001 - context bag is newer than some trees
            mem_fraction = getattr(self.server_args, "mem_fraction_static", 0.9)
        self._model_runner = SimModelRunnerStub(
            model_config=self.model_config,
            mem_fraction_static=mem_fraction,
            gpu_id=self.gpu_id,
            ps=self.ps,
            nccl_port=self.nccl_port,
            server_args=self.server_args,
            is_draft_worker=self.is_draft_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
        )

    def get_pad_input_ids_func(self) -> None:
        """The stub runner has no real model to take a pad function from."""
        return

    def init_attention_backends(self, *args: Any, **kwargs: Any) -> None:
        """No-op: the base would rebuild real backends over the stub."""

    def init_cuda_graphs(self, *args: Any, **kwargs: Any) -> None:
        """No-op: nothing to capture."""

    def _resolve_token_id(self) -> int:
        """Fabricated token id: mid-vocabulary, never a stop token."""
        vocab_size = max(self.dims.vocab_size, 2)
        if self.sim_config.token_id is not None:
            return max(0, min(self.sim_config.token_id, vocab_size - 1))
        return vocab_size // 2

    def forward_batch_generation(
        self,
        batch: Any,
        forward_batch: Any = None,
        pp_proxy_tensors: Any = None,
        is_verify: bool = False,
        skip_attn_backend_init: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Simulate one step and fabricate its ``GenerationBatchResult``.

        ``next_token_ids`` must be an int64 tensor on the batch device: the
        scheduler's FutureMap relay checks ``isinstance(..., torch.Tensor)``
        and writes into a scheduler-device buffer without a device cast, so
        a list (or a wrong-device tensor) silently corrupts the next decode
        step's inputs.
        """
        import torch
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.managers.utils import GenerationBatchResult

        if batch is None:
            raise RuntimeError(
                "SimTpModelWorker only serves the ScheduleBatch path; a bare "
                "ForwardBatch call means the engine loop changed shape."
            )
        if getattr(batch, "return_logprob", False):
            raise NotImplementedError(
                "SimTpModelWorker fabricates no logprobs and SGLang "
                "dereferences them unguarded; remove logprob sampling "
                "params (SGL-5)."
            )
        rows = observe_schedule_batch(batch)
        if not rows:
            return GenerationBatchResult(
                logits_output=LogitsProcessorOutput(next_token_logits=None),
                can_run_cuda_graph=False,
            )

        record = self.translator.translate(
            step_index=self.step_index,
            virtual_time_ps=self.clock.now_ps,
            rows=rows,
        )
        self.step_index += 1
        self._model_runner.observe_tp_step()
        self._settle(record, num_sampled=len(rows))

        device = getattr(batch, "device", None) or "cpu"
        next_token_ids = torch.full(
            (len(rows),), self.token_id, dtype=torch.long, device=device
        )
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=next_token_ids,
            can_run_cuda_graph=False,
        )

    def _settle(self, record: StepRecord, num_sampled: int) -> StepResult:
        """Record the step, ask the sink or the provider for its duration."""
        result = self.step_sink(record) if self.step_sink is not None else None
        if result is None:
            latency_ps = estimate_step_latency_ps(
                self.dims,
                record,
                num_sampled,
                self.compute_provider,
                self.sim_gpu,
                self.host_model,
            )
            result = StepResult(
                step_index=record.step_index,
                step_latency_ps=latency_ps,
                completed_at_ps=self.clock.now_ps + latency_ps,
            )
        self.step_records.append(record)
        self.step_results.append(result)
        if self._record_stream is not None:
            self._record_stream.append(record)
        self.clock.advance_to(max(result.completed_at_ps, self.clock.now_ps))
        if self.sim_config.mode == "paced" and result.step_latency_ps > 0:
            time.sleep(result.step_latency_ps / PS_PER_SECOND)
        return result
