"""In-process SGLang scheduler pump, exercised without SGLang or torch.

The pump only calls duck-typed scheduler methods, so a stub scheduler proves
the ordering contract it inherits from ``event_loop_normal``. Nothing here
imports SGLang: the gate environment does not have it, and CI never does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from simllm.adapters.sglang.pump import (
    PumpCompletion,
    SchedulerOutputCollector,
    SglangSchedulerPump,
    read_output_batch,
)


class _Payload:
    """Stand-in for SGLang's ``BatchTokenIDOutput``."""

    def __init__(
        self,
        rids: list[str],
        finished_reasons: list[Any] | None = None,
        completion_tokens: list[int] | None = None,
        cached_tokens: list[int] | None = None,
        retraction_counts: list[int] | None = None,
    ) -> None:
        self.rids = rids
        self.finished_reasons = finished_reasons
        self.completion_tokens = completion_tokens
        self.cached_tokens = cached_tokens
        self.retraction_counts = retraction_counts


@dataclass
class _Request:
    rid: str


class _Batch:
    def __init__(self, request_ids: list[str], forward_mode: str = "EXTEND") -> None:
        self.reqs = [_Request(rid) for rid in request_ids]
        self.forward_mode = forward_mode

    def __repr__(self) -> str:
        return f"_Batch({[req.rid for req in self.reqs]!r}, {self.forward_mode})"


class _Plan:
    def __init__(self, batch_to_run: Any, running_batch: Any) -> None:
        self.batch_to_run = batch_to_run
        self.running_batch = running_batch


class _Sender:
    def __init__(self) -> None:
        self.socket = object()


class _Channels:
    def __init__(self) -> None:
        self.send_to_tokenizer = _Sender()


class _Streamer:
    def __init__(self) -> None:
        self.send_to_detokenizer = _Sender()


class _Scheduler:
    """Records the exact call order the pinned event loop performs."""

    def __init__(self, plans: list[_Plan], payloads: dict[int, _Payload] | None = None):
        self.calls: list[tuple[Any, ...]] = []
        self.running_batch = "running-0"
        self.last_batch: Any = "sentinel-last"
        self.cur_batch_for_debug: Any = "sentinel-debug"
        self.output_streamer = _Streamer()
        self.ipc_channels = _Channels()
        self._plans = list(plans)
        self._payloads = payloads or {}
        self._batch_ordinal = 0

    def process_input_requests(self, recv_reqs: list[Any]) -> None:
        self.calls.append(("process_input_requests", tuple(req.rid for req in recv_reqs)))

    def get_next_batch_to_run(self, running_batch: Any, last_batch: Any) -> _Plan:
        self.calls.append(("get_next_batch_to_run", running_batch, last_batch))
        return self._plans.pop(0)

    def run_batch(self, batch: Any) -> Any:
        self.calls.append(("run_batch", batch))
        return ("result", batch)

    def process_batch_result(self, batch: Any, result: Any) -> None:
        self.calls.append(("process_batch_result", batch, result))
        payload = self._payloads.get(self._batch_ordinal)
        self._batch_ordinal += 1
        if payload is not None:
            self.output_streamer.send_to_detokenizer.send_output(payload)

    def on_idle(self) -> None:
        self.calls.append(("on_idle",))


def test_read_output_batch_projects_finished_rows_only():
    payload = _Payload(
        rids=["a", "b", "c"],
        finished_reasons=[{"type": "length", "length": 4}, None, {"type": "stop"}],
        completion_tokens=[4, 2, 7],
        cached_tokens=[3, 0, 1],
        retraction_counts=[0, 0, 2],
    )

    rows = read_output_batch(payload)

    assert rows == (
        PumpCompletion(
            request_id="a",
            finish_reason="length",
            output_token_count=4,
            cached_token_count=3,
            retraction_count=0,
        ),
        PumpCompletion(
            request_id="c",
            finish_reason="stop",
            output_token_count=7,
            cached_token_count=1,
            retraction_count=2,
        ),
    )


def test_read_output_batch_tolerates_absent_parallel_arrays():
    payload = _Payload(rids=["a"], finished_reasons=[{"type": "length"}])

    (row,) = read_output_batch(payload)

    assert row.output_token_count == 0
    assert row.cached_token_count == 0
    assert row.retraction_count == 0


def test_read_output_batch_ignores_an_empty_payload():
    assert read_output_batch(_Payload(rids=[])) == ()
    assert read_output_batch(_Payload(rids=["a"], finished_reasons=[None])) == ()


def test_step_mirrors_the_pinned_event_loop_body():
    batch = _Batch(["r0"], forward_mode="EXTEND")
    scheduler = _Scheduler([_Plan(batch, "running-1")])
    pump = SglangSchedulerPump(scheduler)

    outcome = pump.step()

    assert scheduler.calls == [
        ("get_next_batch_to_run", "running-0", "sentinel-last"),
        ("run_batch", batch),
        ("process_batch_result", batch, ("result", batch)),
    ]
    assert scheduler.running_batch == "running-1"
    assert scheduler.last_batch is batch
    assert scheduler.cur_batch_for_debug is batch
    assert outcome.step_ordinal == 0
    assert outcome.ran_batch is True
    assert outcome.batch_size == 1
    assert outcome.forward_mode == "EXTEND"
    assert outcome.completions == ()


def test_an_empty_plan_calls_the_idle_handler_and_clears_the_last_batch():
    scheduler = _Scheduler([_Plan(None, "running-1")])
    pump = SglangSchedulerPump(scheduler)

    outcome = pump.step()

    assert scheduler.calls == [
        ("get_next_batch_to_run", "running-0", "sentinel-last"),
        ("on_idle",),
    ]
    assert scheduler.last_batch is None
    assert scheduler.cur_batch_for_debug is None
    assert outcome.ran_batch is False
    assert outcome.batch_size == 0
    assert outcome.forward_mode is None


def test_submit_tracks_inflight_requests_until_the_scheduler_reports_them():
    batch = _Batch(["r0"])
    payload = _Payload(
        rids=["r0"],
        finished_reasons=[{"type": "length"}],
        completion_tokens=[12],
    )
    scheduler = _Scheduler([_Plan(batch, "running-1")], payloads={0: payload})
    pump = SglangSchedulerPump(scheduler)
    assert pump.has_unfinished_requests is False

    pump.submit([_Request("r0")])
    assert pump.has_unfinished_requests is True
    assert pump.submitted_request_ids == ("r0",)

    outcome = pump.step()

    assert outcome.completions == (
        PumpCompletion(
            request_id="r0",
            finish_reason="length",
            output_token_count=12,
            cached_token_count=0,
            retraction_count=0,
        ),
    )
    assert pump.has_unfinished_requests is False
    assert pump.finished_request_ids == ("r0",)
    assert pump.completions == list(outcome.completions)


def test_submit_refuses_a_blank_or_repeated_request_identity():
    scheduler = _Scheduler([])
    pump = SglangSchedulerPump(scheduler)

    with pytest.raises(ValueError, match="nonblank rid"):
        pump.submit([_Request("  ")])
    pump.submit([_Request("r0")])
    with pytest.raises(ValueError, match="submitted twice"):
        pump.submit([_Request("r0")])
    assert pump.submitted_request_ids == ("r0",)


def test_submitting_nothing_does_not_reach_the_scheduler():
    scheduler = _Scheduler([])
    pump = SglangSchedulerPump(scheduler)

    pump.submit([])

    assert scheduler.calls == []


def test_a_request_reported_finished_twice_is_fatal():
    payload = _Payload(rids=["r0"], finished_reasons=[{"type": "length"}])
    scheduler = _Scheduler(
        [_Plan(_Batch(["r0"]), "r"), _Plan(_Batch(["r0"]), "r")],
        payloads={0: payload, 1: payload},
    )
    pump = SglangSchedulerPump(scheduler)
    pump.submit([_Request("r0")])
    pump.step()

    with pytest.raises(RuntimeError, match="finished twice"):
        pump.step()


def test_attaching_the_collector_replaces_the_sender_and_silences_the_control_path():
    scheduler = _Scheduler([])
    original_socket = scheduler.ipc_channels.send_to_tokenizer.socket

    pump = SglangSchedulerPump(scheduler)

    assert scheduler.output_streamer.send_to_detokenizer is pump.collector
    assert scheduler.ipc_channels.send_to_tokenizer.socket is None
    assert original_socket is not None


def test_the_off_path_leaves_every_scheduler_object_untouched():
    """The disabled collector must not mutate the scheduler at all."""

    scheduler = _Scheduler([_Plan(_Batch(["r0"]), "running-1")])
    sender = scheduler.output_streamer.send_to_detokenizer
    socket = scheduler.ipc_channels.send_to_tokenizer.socket

    pump = SglangSchedulerPump(scheduler, attach_output_collector=False)
    outcome = pump.step()

    assert scheduler.output_streamer.send_to_detokenizer is sender
    assert scheduler.ipc_channels.send_to_tokenizer.socket is socket
    assert outcome.ran_batch is True
    assert outcome.completions == ()
    assert pump.completions == []


def test_a_scheduler_without_an_output_streamer_is_refused():
    class _Bare:
        pass

    with pytest.raises(RuntimeError, match="output_streamer"):
        SglangSchedulerPump(_Bare())


def test_the_output_collector_drains_exactly_once():
    collector = SchedulerOutputCollector()
    collector.send_output("first")
    collector.send_output("second", recv_obj="ignored")

    assert collector.drain() == ("first", "second")
    assert collector.drain() == ()


def test_the_pump_module_imports_without_sglang_or_torch():
    import simllm.adapters.sglang.pump as pump_module

    assert "sglang" not in dir(pump_module)
    with pytest.raises(ImportError):
        pump_module.tokenized_generate_request(
            request_id="r0", input_token_ids=(1, 2), max_new_tokens=1
        )


# The chunked-prefill gate (SGL-12 closed; the refusal condition moved)


def test_chunked_prefill_is_admitted_on_the_sampled_identity_path():
    from simllm.adapters.sglang.pump import chunked_prefill_refusal

    assert chunked_prefill_refusal(2048, sample_identity=True) is None


def test_chunked_prefill_is_refused_on_the_compatibility_record_stream():
    from simllm.adapters.sglang.pump import chunked_prefill_refusal

    message = chunked_prefill_refusal(2048, sample_identity=False)

    assert message is not None
    assert "SIMLLM_SGLANG_SAMPLE_IDENTITY=0" in message
    assert "SGL-22" in message
    # The refusal no longer claims the seam cannot populate the fields.
    assert "SGL-12" not in message


def test_the_gate_is_inert_when_chunked_prefill_is_disabled():
    from simllm.adapters.sglang.pump import chunked_prefill_refusal

    assert chunked_prefill_refusal(None, sample_identity=True) is None
    assert chunked_prefill_refusal(None, sample_identity=False) is None


def test_the_gate_falls_back_to_the_next_worker_s_value_when_none_exists(
    monkeypatch,
):
    from simllm.adapters.sglang import worker as worker_module

    monkeypatch.delenv("SIMLLM_SGLANG_SAMPLE_IDENTITY", raising=False)
    monkeypatch.setattr(worker_module, "_LATEST", [])
    worker_module.reset_configuration()
    try:
        assert worker_module.active_sample_identity() is True

        monkeypatch.setenv("SIMLLM_SGLANG_SAMPLE_IDENTITY", "0")
        assert worker_module.active_sample_identity() is False

        worker_module.configure(
            config=worker_module.SimWorkerConfig(sample_identity=True)
        )
        assert worker_module.active_sample_identity() is True
    finally:
        worker_module.reset_configuration()


def test_the_constructed_worker_outranks_the_hooks_and_the_environment(
    monkeypatch,
):
    """One authority: the worker latched the value, so the worker answers.

    ``Scheduler`` construction builds the worker through the plugin hook, and
    the worker's constructor latches ``sample_identity`` into its translator.
    A ``configure`` call or an environment change after that point never
    reaches the worker that already exists, so re-deriving from the hooks
    would report a value the emitted records do not have.
    """

    from simllm.adapters.sglang import worker as worker_module

    class _BuiltWorker:
        def __init__(self, sample_identity: bool) -> None:
            self.sim_config = worker_module.SimWorkerConfig(
                sample_identity=sample_identity
            )

    monkeypatch.delenv("SIMLLM_SGLANG_SAMPLE_IDENTITY", raising=False)
    worker_module.reset_configuration()
    monkeypatch.setattr(worker_module, "_LATEST", [_BuiltWorker(False)])
    try:
        # The hooks and the environment both say the identity is emitted; the
        # worker that exists latched the opposite, and it wins.
        worker_module.configure(
            config=worker_module.SimWorkerConfig(sample_identity=True)
        )
        monkeypatch.setenv("SIMLLM_SGLANG_SAMPLE_IDENTITY", "1")
        assert worker_module.active_sample_identity() is False
        assert worker_module.latest_worker().sim_config.sample_identity is False

        monkeypatch.setattr(worker_module, "_LATEST", [_BuiltWorker(True)])
        assert worker_module.active_sample_identity() is True
    finally:
        worker_module.reset_configuration()


# Mid-prompt chunk rows through the seam replay path


class _ChunkReq:
    """Shaped like ``Req`` at the pinned commit, for the fields the seam reads."""

    def __init__(
        self,
        rid: str,
        *,
        context_length: int,
        inflight_middle_chunks: int = 0,
        output_ids: int = 0,
    ) -> None:
        self.rid = rid
        self.kv_committed_len = context_length
        self.cached_tokens = 0
        self.origin_input_ids: list[int] = []
        self.output_ids = [0] * output_ids
        self.inflight_middle_chunks = inflight_middle_chunks
        self.is_retracted = False

    def finished(self) -> bool:
        return False


class _Mode:
    def __init__(self, name: str) -> None:
        self._name = name

    def is_extend(self) -> bool:
        return self._name in ("extend", "mixed")

    def is_decode(self) -> bool:
        return self._name == "decode"

    def is_idle(self) -> bool:
        return self._name == "idle"


class _ChunkBatch:
    def __init__(
        self,
        reqs: list[_ChunkReq],
        mode: str,
        *,
        seq_lens_cpu: list[int],
        extend_lens: list[int] | None = None,
        decoding_reqs: list[_ChunkReq] | None = None,
    ) -> None:
        self.reqs = reqs
        self.forward_mode = _Mode(mode)
        self.seq_lens_cpu = seq_lens_cpu
        self.extend_lens = extend_lens or []
        self.prefix_lens: list[int] = []
        self.decoding_reqs = decoding_reqs
        self.return_logprob = False


def _chunked_stream():
    """A two-chunk prefill running beside a decode, as the scheduler builds it.

    ``inflight_middle_chunks`` is positive on the first chunk and zero on the
    last, which is the pinned rule the seam transcribes: SGLang discards the
    mid-prompt row's token and appends the final chunk's.
    """

    running = _ChunkReq("running", context_length=13, output_ids=1)
    chunked_first = _ChunkReq(
        "chunked", context_length=4, inflight_middle_chunks=1
    )
    chunked_last = _ChunkReq("chunked", context_length=8)
    return [
        _ChunkBatch(
            [chunked_first, running],
            "mixed",
            seq_lens_cpu=[4, 14],
            extend_lens=[4, 1],
            decoding_reqs=[running],
        ),
        _ChunkBatch(
            [chunked_last, running],
            "mixed",
            seq_lens_cpu=[8, 15],
            extend_lens=[4, 1],
            decoding_reqs=[running],
        ),
        _ChunkBatch(
            [_ChunkReq("chunked", context_length=9, output_ids=1), running],
            "decode",
            seq_lens_cpu=[9, 16],
        ),
    ]


def test_mid_prompt_chunk_rows_score_correctly_through_the_seam_replay_path(tmp_path):
    """The gate's premise, checked as far as the sink's compute estimate.

    The refusal was rewritten to admit chunked prefill whenever the sampled
    identity is emitted. This is the evidence for that: the same three batches
    are translated on both record paths and priced by the sink's own serial
    lowerer through ``compute_estimate_ps``. The identity path charges the LM
    head for one token on the mid-prompt step and the compatibility path
    charges two, and on the identity path the chunked request's first sampled
    step is the one that drains its last chunk.

    This stops at the compute estimate. It runs no backend, computes no
    makespan and drives no scheduler, so it is component evidence for the
    gate's premise and not an end-to-end metric claim.
    """

    from simllm.adapters.sglang.worker import SglStepTranslator, observe_schedule_batch
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
    from simllm.compute import ComputeProvider, DurationEstimate, ModelDims

    class _FlopProvider(ComputeProvider):
        """One picosecond per FLOP, so the sampled count is visible exactly."""

        def estimate(self, kernel, gpu):
            del gpu
            return DurationEstimate(duration_ps=int(kernel.flops), bound="compute")

    dims = ModelDims(
        num_layers=2,
        hidden_size=256,
        intermediate_size=512,
        num_heads=4,
        num_kv_heads=4,
        head_size=64,
        vocab_size=32000,
        dtype_bytes=2,
    )

    streams = {}
    for label, sample_identity in (("identity", True), ("compatibility", False)):
        translator = SglStepTranslator(sample_identity=sample_identity)
        streams[label] = [
            translator.translate(
                step_index=index,
                virtual_time_ps=index * 1_000_000,
                rows=observe_schedule_batch(batch),
            )
            for index, batch in enumerate(_chunked_stream())
        ]

    identity, compatibility = streams["identity"], streams["compatibility"]

    # The mid-prompt row is scheduled but does not produce a token.
    assert [request.request_id for request in identity[0].scheduled] == [
        "chunked",
        "running",
    ]
    assert identity[0].num_sampled == 1
    assert identity[0].sampled_request_ids == ["running"]
    assert identity[1].num_sampled == 2
    assert identity[2].num_sampled == 2
    assert compatibility[0].num_sampled is None

    # The first step that generates a token for the chunked request is the one
    # that drains its last chunk, not the one that started its prompt.
    first_token_step = next(
        record.step_index
        for record in identity
        if "chunked" in (record.sampled_request_ids or ())
    )
    assert first_token_step == 1

    sinks = {
        label: HtsimStepSink(
            HtsimStepSinkConfig(
                profile="rnic-nn-fluid",
                tp_ranks=(0, 1),
                dims=dims,
                workdir=tmp_path / label,
                provider=_FlopProvider(),
            )
        )
        for label in streams
    }
    estimates = {
        label: [sinks[label].compute_estimate_ps(record) for record in streams[label]]
        for label in streams
    }

    # Only the mid-prompt step differs, and it differs by exactly the LM head
    # work of the one row the compatibility path wrongly counts as sampled.
    head_flops = 2 * dims.hidden_size * dims.vocab_size
    assert (
        estimates["compatibility"][0] - estimates["identity"][0] == head_flops
    )
    assert estimates["identity"][1:] == estimates["compatibility"][1:]
