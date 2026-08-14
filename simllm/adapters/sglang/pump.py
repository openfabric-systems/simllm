"""Drive one SGLang ``Scheduler`` synchronously, in the calling process.

SGLang runs its scheduler in an ``mp.Process`` and loads plugins inside
``run_scheduler_process``, so a driver that calls
:func:`simllm.adapters.sglang.worker.configure` in a parent process never
reaches the worker: ``configure`` is process local and says so in its own
docstring. That is why no live step sink has ever been installed on this
adapter, and why the M4 closed-loop slice covered SGLang by JSONL replay only.

This module removes the process boundary rather than trying to reach across
it. :func:`build_in_process_scheduler` constructs
``sglang.srt.managers.scheduler.Scheduler`` in the current process, after the
caller has installed the simulated worker and configured its hooks, and
:class:`SglangSchedulerPump` unrolls the body of ``Scheduler.event_loop_normal``
into a synchronous :meth:`SglangSchedulerPump.step`. A driver can then
interleave scheduler steps with an arrival gate exactly as the vLLM study
interleaves ``llm.llm_engine.step()``.

SGLang ships in-tree precedent for constructing the scheduler directly:
``srt/ray/scheduler_actor.py`` does it with no ``mp.Process`` at all.

Three facts about the pinned commit ``8f2a3ad`` are load bearing here and are
recorded so a future reader can re-verify them:

- ``event_loop_normal`` reads requests, calls ``process_input_requests``, plans
  with ``get_next_batch_to_run``, runs ``run_batch`` and ``process_batch_result``
  when the plan carries a batch and ``on_idle`` when it does not, then assigns
  ``last_batch``. ``process_input_requests`` takes a plain list of already
  constructed request objects, so no ZMQ is involved in ingress.
- the loop is decorated ``@DynamicGradMode()``. A hand-rolled step must
  therefore run under ``torch.no_grad()`` or it builds autograd graphs.
- the watchdog thread treats ``scheduler.cur_batch_for_debug is not None`` as
  "busy" and uses ``forward_ct`` as its progress counter, so the pump assigns
  ``cur_batch_for_debug`` on every step exactly as the real loop does.

Generation results leave the scheduler through
``scheduler.output_streamer.send_to_detokenizer``, a ``SenderWrapper`` around a
``zmq.PUSH`` socket with nobody on the other end in this configuration. The
pump replaces that one object with :class:`SchedulerOutputCollector`, which
keeps the payloads in memory instead of encoding them, and reads finished
requests out of them. That is the pump's only mutation of scheduler state.

This module must stay importable without SGLang and without torch: every
framework import lives inside a function, and the pump itself only calls
duck-typed methods, so it can be driven against a stub scheduler in a test.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "PumpCompletion",
    "PumpStepOutcome",
    "SchedulerOutputCollector",
    "SglangSchedulerPump",
    "build_in_process_scheduler",
    "chunked_prefill_refusal",
    "read_output_batch",
    "tokenized_generate_request",
]


def chunked_prefill_refusal(
    chunked_prefill_size: Any,
    *,
    sample_identity: bool,
) -> str | None:
    """Why chunked prefill is refused for this configuration, or ``None``.

    The refusal covers the one state in which a mid-prompt row is known to be
    mis-scored: ``SIMLLM_SGLANG_SAMPLE_IDENTITY=0`` selects the pre-SGL-12
    compatibility stream, where ``num_sampled`` and ``sampled_request_ids``
    stay absent and every consumer reads the whole scheduled batch as sampled,
    so a mid-prompt chunk is counted as a generated token. That state, not
    chunked prefill itself, is the refusal condition.

    ``None`` is not a safety certificate. On the default path every record does
    carry the sampled count and identity, transcribed from the pinned
    ``process_batch_result_prefill`` rule, so a mid-prompt extend row is
    excluded from the sampled set. That rule is source transcription checked
    against stub batches carrying the pinned attribute names; no running
    scheduler has been observed agreeing with it and this gate does not claim
    one has, which is SGL-22. Chunked-prefill hazards outside the sampled-row
    rule are outside what this gate inspects at all.
    """

    if chunked_prefill_size is None:
        return None
    if sample_identity:
        return None
    return (
        "chunked prefill is enabled while SIMLLM_SGLANG_SAMPLE_IDENTITY=0 "
        "selects the compatibility record stream, in which "
        "StepRecord.num_sampled and sampled_request_ids stay absent and every "
        "scheduled row is read as having produced a token, so a mid-prompt "
        "extend row would silently be scored as a generated token. Keep the "
        "sampled-identity path, or disable chunked prefill. The sampled-row "
        "rule itself is source transcription and its live-scheduler "
        "confirmation is SGL-22."
    )


@dataclass(frozen=True)
class PumpCompletion:
    """One finished request, as the scheduler published it."""

    request_id: str
    #: SGLang's own finish-reason type string, e.g. ``length`` or ``stop``
    finish_reason: str | None
    #: generated tokens the scheduler counted for this request
    output_token_count: int
    #: prompt tokens served from the radix cache
    cached_token_count: int
    #: times the scheduler retracted this request under KV pressure
    retraction_count: int


@dataclass(frozen=True)
class PumpStepOutcome:
    """What one unrolled ``event_loop_normal`` body did."""

    step_ordinal: int
    #: whether the plan carried a batch, i.e. whether a forward ran
    ran_batch: bool
    #: number of requests in that batch, zero when no batch ran
    batch_size: int
    #: the batch's forward mode as a string, ``None`` when no batch ran
    forward_mode: str | None
    completions: tuple[PumpCompletion, ...] = ()


class SchedulerOutputCollector:
    """Stand-in for the scheduler's detokenizer socket.

    ``SenderWrapper.send_output`` is the only method the scheduler calls on
    that object, and it is called with the fully built output payload. Keeping
    the payloads here costs no serialization and gives an in-process view of
    every completion, which is the one fact the worker seam cannot report
    (finish decisions are applied in ``process_batch_result``, after the
    forward returns).
    """

    def __init__(self) -> None:
        self.batches: list[Any] = []

    def send_output(self, output: Any, recv_obj: Any = None) -> None:
        del recv_obj
        self.batches.append(output)

    def drain(self) -> tuple[Any, ...]:
        """Return and clear the payloads accumulated since the last drain."""

        payloads = tuple(self.batches)
        self.batches.clear()
        return payloads


def _row(values: Any, index: int, default: Any) -> Any:
    """Read one parallel-array entry, tolerating an absent or short array."""

    if values is None:
        return default
    try:
        return values[index]
    except (IndexError, KeyError, TypeError):
        return default


def read_output_batch(payload: Any) -> tuple[PumpCompletion, ...]:
    """Project one ``BatchTokenIDOutput`` into finished-request rows.

    Every field is read with ``getattr`` so a stub with the same attribute
    names works. The payload carries parallel arrays keyed by ``rids``; a row
    whose ``finished_reasons`` entry is ``None`` is a streaming update rather
    than a completion and is skipped. SGLang spells the reason as a mapping
    with a ``type`` key at the pinned commit.
    """

    rids = getattr(payload, "rids", None)
    if not rids:
        return ()
    reasons = getattr(payload, "finished_reasons", None)
    completion_tokens = getattr(payload, "completion_tokens", None)
    cached_tokens = getattr(payload, "cached_tokens", None)
    retraction_counts = getattr(payload, "retraction_counts", None)
    rows: list[PumpCompletion] = []
    for index, request_id in enumerate(rids):
        reason = _row(reasons, index, None)
        if reason is None:
            continue
        if isinstance(reason, dict):
            reason_type = reason.get("type")
        else:
            reason_type = getattr(reason, "type", None) or str(reason)
        rows.append(
            PumpCompletion(
                request_id=str(request_id),
                finish_reason=None if reason_type is None else str(reason_type),
                output_token_count=int(_row(completion_tokens, index, 0) or 0),
                cached_token_count=int(_row(cached_tokens, index, 0) or 0),
                retraction_count=int(_row(retraction_counts, index, 0) or 0),
            )
        )
    return tuple(rows)


def _no_grad() -> Any:
    """``torch.no_grad()`` when torch is importable, else a null context.

    The real loop is decorated ``@DynamicGradMode()``. Without torch there is
    no autograd to disable and the pump is being driven against a stub.
    """

    try:
        import torch
    except ImportError:
        return contextlib.nullcontext()
    return torch.no_grad()


class SglangSchedulerPump:
    """One SGLang scheduler, stepped by hand.

    The pump owns no timing authority and no batching policy. It calls the
    scheduler's own methods in the order ``event_loop_normal`` calls them and
    reports what happened. Request admission stays with the caller, so an
    arrival gate on the worker's virtual clock decides when a request enters
    the framework and the framework alone decides what to run.
    """

    def __init__(self, scheduler: Any, *, attach_output_collector: bool = True) -> None:
        self.scheduler = scheduler
        self.step_ordinal = 0
        self.completions: list[PumpCompletion] = []
        self._submitted: set[str] = set()
        self._finished: set[str] = set()
        self.collector = SchedulerOutputCollector()
        if attach_output_collector:
            self._attach_output_collector()

    def _attach_output_collector(self) -> None:
        streamer = getattr(self.scheduler, "output_streamer", None)
        if streamer is None:
            raise RuntimeError(
                "scheduler has no output_streamer; the pump cannot observe "
                "completions on this build"
            )
        streamer.send_to_detokenizer = self.collector
        # The control-response path is a second socket with no listener. It is
        # never read here, so silence it rather than letting it queue payloads.
        channels = getattr(self.scheduler, "ipc_channels", None)
        sender = getattr(channels, "send_to_tokenizer", None)
        if sender is not None and hasattr(sender, "socket"):
            sender.socket = None

    @property
    def submitted_request_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._submitted))

    @property
    def finished_request_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._finished))

    @property
    def has_unfinished_requests(self) -> bool:
        """Whether a submitted request has not yet been reported finished."""

        return bool(self._submitted - self._finished)

    def submit(self, requests: Sequence[Any]) -> None:
        """Hand already tokenized requests to the scheduler.

        This is the ingress half of one event-loop body, with the ZMQ receive
        replaced by the caller's own list. The scheduler's queueing and
        admission decisions are untouched.
        """

        rows = list(requests)
        if not rows:
            return
        for request in rows:
            request_id = getattr(request, "rid", None)
            if not isinstance(request_id, str) or not request_id.strip():
                raise ValueError("every submitted request needs a nonblank rid")
            if request_id in self._submitted:
                raise ValueError(f"request {request_id!r} was submitted twice")
            self._submitted.add(request_id)
        with _no_grad():
            self.scheduler.process_input_requests(rows)

    def step(self) -> PumpStepOutcome:
        """Run one unrolled ``event_loop_normal`` body.

        Mirrors the pinned loop exactly: plan, run and settle a batch when the
        plan carries one, otherwise call the scheduler's idle handler, then
        publish ``last_batch``.
        """

        scheduler = self.scheduler
        ordinal = self.step_ordinal
        self.step_ordinal += 1
        with _no_grad():
            plan = scheduler.get_next_batch_to_run(
                running_batch=scheduler.running_batch,
                last_batch=scheduler.last_batch,
            )
            scheduler.running_batch = plan.running_batch
            batch = plan.batch_to_run
            scheduler.cur_batch_for_debug = batch
            if batch:
                result = scheduler.run_batch(batch)
                scheduler.process_batch_result(batch, result)
            else:
                scheduler.on_idle()
            scheduler.last_batch = batch
        completions = self._collect()
        return PumpStepOutcome(
            step_ordinal=ordinal,
            ran_batch=bool(batch),
            batch_size=len(getattr(batch, "reqs", ()) or ()) if batch else 0,
            forward_mode=None if not batch else str(getattr(batch, "forward_mode", None)),
            completions=completions,
        )

    def _collect(self) -> tuple[PumpCompletion, ...]:
        rows: list[PumpCompletion] = []
        for payload in self.collector.drain():
            for completion in read_output_batch(payload):
                if completion.request_id in self._finished:
                    raise RuntimeError(
                        f"request {completion.request_id!r} finished twice"
                    )
                self._finished.add(completion.request_id)
                rows.append(completion)
        self.completions.extend(rows)
        return tuple(rows)


def tokenized_generate_request(
    *,
    request_id: str,
    input_token_ids: Sequence[int],
    max_new_tokens: int,
    tokenizer: Any = None,
) -> Any:
    """Build one ``TokenizedGenerateReqInput`` the scheduler will accept.

    Mirrors ``TokenizerManager._create_tokenized_object`` at the pinned commit:
    the token ids travel as ``array("q", ...)`` and the sampling parameters must
    be normalized before the scheduler reads ``stop_strs``.
    """

    from array import array

    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    from sglang.srt.sampling.sampling_params import SamplingParams

    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a nonblank string")
    tokens = [int(token) for token in input_token_ids]
    if not tokens:
        raise ValueError("input_token_ids must not be empty")
    if int(max_new_tokens) <= 0:
        raise ValueError("max_new_tokens must be positive")
    sampling_params = SamplingParams(
        temperature=0.0,
        max_new_tokens=int(max_new_tokens),
    )
    sampling_params.normalize(tokenizer)
    return TokenizedGenerateReqInput(
        rid=request_id,
        input_text=None,
        input_ids=array("q", tokens),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=sampling_params,
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )


@contextlib.contextmanager
def _offline_environment() -> Iterator[None]:
    values = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
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


def build_in_process_scheduler(
    *,
    model_path: str,
    device: str = "cpu",
    dtype: str = "float32",
    tp_size: int = 1,
    page_size: int = 1,
    context_length: int = 512,
    max_total_tokens: int = 4096,
    max_running_requests: int = 8,
    chunked_prefill_size: int = -1,
    random_seed: int = 0,
) -> Any:
    """Construct an SGLang ``Scheduler`` in this process and return it.

    The caller must have installed the simulated worker
    (``simllm.adapters.sglang.install()``) and set its hooks
    (``configure(step_sink=...)``) first: the scheduler builds its TP worker
    during construction and the hooks are read there.

    The prelude mirrors ``run_scheduler_process`` minus the parts that only
    make sense in a child process: ``publish(server_args, role="scheduler")``
    is required, because ``Scheduler.__init__`` reads the process-global config
    bags rather than ``server_args`` for several fields, including the chunked
    prefill size. ``configure_scheduler_process`` is deliberately not called:
    it installs ``kill_itself_when_parent_died`` and rewrites the process
    title, neither of which belongs in an in-process driver.

    ``disable_overlap_schedule`` is forced on, because the pump unrolls
    ``event_loop_normal`` and the overlap loop has different result semantics.
    """

    from sglang.srt.managers.scheduler import Scheduler, publish
    from sglang.srt.server_args import PortArgs, ServerArgs

    with _offline_environment():
        server_args = ServerArgs(
            model_path=str(model_path),
            device=device,
            dtype=dtype,
            tp_size=int(tp_size),
            page_size=int(page_size),
            context_length=int(context_length),
            max_total_tokens=int(max_total_tokens),
            max_running_requests=int(max_running_requests),
            chunked_prefill_size=int(chunked_prefill_size),
            random_seed=int(random_seed),
            disable_overlap_schedule=True,
            disable_cuda_graph=True,
        )
        port_args = PortArgs.init_new(server_args)
        publish(server_args, role="scheduler")
        scheduler = Scheduler(
            server_args=server_args,
            port_args=port_args,
            gpu_id=0,
            tp_rank=0,
            moe_ep_rank=0,
            pp_rank=0,
            attn_cp_rank=0,
            moe_dp_rank=0,
            dp_rank=None,
        )
    from simllm.adapters.sglang.worker import active_sample_identity

    refusal = chunked_prefill_refusal(
        getattr(scheduler, "chunked_prefill_size", None),
        sample_identity=active_sample_identity(),
    )
    if refusal is not None:
        raise RuntimeError(refusal)
    return scheduler
