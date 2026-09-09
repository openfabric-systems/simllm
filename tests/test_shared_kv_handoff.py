"""Scripted native frames isolate handoff ownership and exact visibility."""

import sys
from dataclasses import replace
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
from test_flow_session import Peer, Stream, counters

from simllm.backends import flow_session
from simllm.backends.flow_session import FlowSessionConfig, FlowSessionError
from simllm.core import (
    DeclaredKvHandoffPolicy,
    DisaggregatedRequestTimeline,
    KvHandoffJoin,
    PendingKvHandoffPolicy,
    VirtualClock,
)
from simllm.core.engine_steps import EngineStepRuntime
from simllm.core.step import RequestPhase, ScheduledRequest, StepRecord, StepResult
from simllm.placement import disaggregated_manifests
from simllm.traffic import SharedKvHandoffRuntime

PREFILL = ("simllm-prefill-0", "simllm-prefill-1")
DECODE = ("simllm-decode-0",)


class HandoffPeer(Peer):
    """Use the common full framed client fixture with a six-endpoint inventory."""

    def reply(self, request):
        result = super().reply(request)
        if request["verb"] == "drain":
            result["sq_high_watermarks"] = [
                int(any(flow["source"] == endpoint for flow in self.flows.values()))
                for endpoint in range(6)
            ]
        return result


@pytest.fixture
def shared(monkeypatch):
    peer, calls = HandoffPeer(), []
    stream = Stream(peer)

    def start(command, **kwargs):
        calls.append((command, kwargs))
        return stream

    monkeypatch.setattr(flow_session, "OwnedBinaryProcess", start)
    deployment = disaggregated_manifests(prefill_nodes=2, decode_nodes=1, gpus_per_node=2,
                                        render_physical_topology=False)
    bindings = {PREFILL[0]: "prefill-node-0", PREFILL[1]: "prefill-node-1", DECODE[0]: "decode-node-0"}
    config = FlowSessionConfig("rnic-nn", 6, 400_000_000_000, "a" * 64, 9001)
    runtime = SharedKvHandoffRuntime(config, ("scripted-native", "--flow-session"),
                                     session_id="shared", deployment=deployment, engine_nodes=bindings,
                                     pcie_submission_ps=10)
    runtime.validate_engines(PREFILL, DECODE, 2)
    clock = VirtualClock()
    runtime.bind(clock, {engine: (0, 1) for engine in bindings})
    return NS(runtime=runtime, peer=peer, stream=stream, calls=calls, clock=clock,
              deployment=deployment, bindings=bindings, config=config)


def submit(shared, request="request-a", producer=0, size=8192):
    return shared.runtime.submit(request_id=request, source_engine_id=PREFILL[producer],
                                 destination_engine_id=DECODE[0], kv_bytes=size,
                                 submitted_at_ps=shared.clock.now_ps)


def complete(shared, receipts, at, sequences):
    shared.peer.updates.append(("completion", at, list(sequences), True, 4 * len(sequences)))
    assert shared.runtime.progress(receipts, through_ps=None) == at
    shared.clock.advance_to(at)
    return shared.runtime.complete_due(receipts)


def test_future_registration_preserves_clock_and_global_endpoints(shared):
    first, second = submit(shared), submit(shared, "request-b", producer=1, size=8193)
    assert isinstance(shared.runtime, PendingKvHandoffPolicy)
    assert shared.clock.now_ps == 0
    assert first.sequences == (1, 2) and second.sequences == (3, 4)
    assert not shared.runtime.joins and not shared.runtime.native_rows
    assert [(row["source"], row["destination"], row["payload_bytes"], row["eligible_at_ps"])
            for row in shared.runtime.accepted_shards] == [
                (0, 4, 4096, 10), (1, 5, 4096, 10), (2, 4, 4097, 10), (3, 5, 4096, 10)]
    assert [row["verb"] for row in shared.peer.requests] == ["open", "inject", "inject", "inject", "inject"]


def test_all_shards_join_without_early_visibility_or_fabricated_queue_fields(shared):
    receipt = submit(shared)
    shared.peer.updates = [("completion", 100, [1], False, 4), ("horizon", 100, [], False, 0),
                           ("completion", 200, [2], True, 1)]
    assert shared.runtime.progress((receipt,), through_ps=None) == 100
    assert shared.runtime.complete_due((receipt,)) == ()
    shared.clock.advance_to(100)
    assert shared.runtime.complete_due((receipt,)) == ()
    assert not shared.runtime.joins
    assert shared.runtime.progress((receipt,), through_ps=None) == 200
    shared.clock.advance_to(199)
    assert shared.runtime.complete_due((receipt,)) == ()
    shared.clock.advance_to(200)
    (join,) = shared.runtime.complete_due((receipt,))
    assert isinstance(join, KvHandoffJoin)
    assert join.critical_shard_sequences == (2,)
    assert join.completed_at_ps == 200 and join.total_ps == 200
    assert len(join.shards) == 2 and sum(len(shard.events) for shard in join.shards) == 8
    assert not {"started_at_ps", "finished_at_ps", "queue_wait_ps", "service_ps"} & join.to_json().keys()
    timeline = DisaggregatedRequestTimeline("request-a", 0, 0, 0, join, 200, (230, 260))
    assert timeline.ttft_ps == 230 and timeline.tpot_ps == 30
    assert timeline.to_json()["handoff"] == join.to_json()
    assert shared.runtime.complete_due(()) == ()


def test_distinct_same_time_callbacks_are_all_drained_before_join_publication(shared):
    receipts = (submit(shared), submit(shared, "request-b", producer=1))
    shared.peer.updates = [("completion", 100, [1], False, 4),
                           ("completion", 100, [3], False, 1),
                           ("completion", 100, [2], False, 1),
                           ("completion", 100, [4], True, 1)]
    assert shared.runtime.progress(receipts, through_ps=100) == 100
    shared.clock.advance_to(100)
    joins = shared.runtime.complete_due(receipts)
    assert [join.request_id for join in joins] == ["request-a", "request-b"]
    assert all(join.completed_at_ps == 100 for join in joins)
    waits = [row for row in shared.peer.requests if row["verb"] == "await_completion"]
    assert [row["through_ps"] for row in waits] == [100, 100, 100, 100]
    assert [row["completion_sequences"] for row in waits] == [[1, 2, 3, 4], [2, 3, 4], [2, 4], [4]]


def test_lookahead_yields_before_local_action_and_keeps_eligibility_boundary_open(shared):
    first = submit(shared)
    shared.peer.updates = [("horizon", 99, [], False, 3)]
    assert shared.runtime.progress((first,), through_ps=99) is None
    shared.clock.advance_to(100)
    second = submit(shared, "request-b", producer=1)
    assert second.eligible_at_ps == 110
    assert len(shared.calls) == 1
    assert not any(row["verb"] == "inject_at_boundary" for row in shared.peer.requests)


def test_repeated_batches_preserve_native_owner_history_and_prior_evidence(shared):
    first = submit(shared)
    (join,) = complete(shared, (first,), 100, (1, 2))
    first_json = join.to_json()
    shared.clock.advance_to(1000)
    second = submit(shared, "request-b")
    assert second.sequences == (3, 4)
    (later,) = complete(shared, (second,), 1100, (3, 4))
    assert later.total_ps == join.total_ps == 100
    assert first_json == shared.runtime.joins[0].to_json()
    assert len(shared.calls) == 1
    assert not any(row["verb"] in {"close", "drain"} for row in shared.peer.requests)
    shared.runtime.close()
    assert shared.clock.now_ps == 1100
    assert shared.stream.finished
    assert len(shared.runtime.native_rows) == 4 and len(shared.runtime.native_events) == 16
    assert [row["verb"] for row in shared.peer.requests][-3:] == ["await_completion", "drain", "close"]


@pytest.mark.parametrize("kind", ["missing", "duplicate", "forged", "bytes", "sequence", "foreign-request"])
def test_bad_private_receipt_inventory_poison_and_reap(shared, kind):
    receipt = submit(shared)
    pending = (receipt,)
    if kind == "missing":
        pending = ()
    elif kind == "duplicate":
        pending = (receipt, receipt)
    elif kind == "forged":
        pending = (replace(receipt),)
    elif kind == "foreign-request":
        pending = (replace(receipt, request_id="foreign"),)
    else:
        object.__setattr__(receipt, "kv_bytes" if kind == "bytes" else "sequences",
                           8193 if kind == "bytes" else (3, 4))
    before = list(shared.peer.requests)
    with pytest.raises(RuntimeError, match="receipt|inventory"):
        shared.runtime.complete_due(pending)
    assert shared.runtime.poisoned and shared.stream.aborted
    assert shared.peer.requests == before


@pytest.mark.parametrize("entry", ["submit", "validate", "close"])
def test_damaged_accepted_receipt_cannot_be_repaired_and_retried(shared, entry):
    receipt = submit(shared)
    object.__setattr__(receipt, "kv_bytes", 8193)
    before = list(shared.peer.requests)
    with pytest.raises(RuntimeError, match="accepted handoff receipt was changed"):
        if entry == "submit":
            submit(shared, "request-b")
        elif entry == "validate":
            shared.runtime.validate_engines(PREFILL, DECODE, 2)
        else:
            shared.runtime.close()
    assert shared.runtime.poisoned and shared.stream.aborted
    object.__setattr__(receipt, "kv_bytes", 8192)
    with pytest.raises(RuntimeError, match="poisoned"):
        submit(shared, "request-b")
    assert shared.peer.requests == before


def bare_serving_session(shared):
    from simllm.adapters.vllm.pd_session import VllmDisaggregatedSession

    session = VllmDisaggregatedSession.__new__(VllmDisaggregatedSession)
    session.clock, session._shared_handoff = shared.clock, shared.runtime
    session.engine_runtime = EngineStepRuntime(shared.clock)
    session._closed, session._request_ids = False, set()
    session.prefill_engines, session.decode_engines = [], []
    session._completion_bridges = {}
    session.config = NS(handoff_policy=shared.runtime)
    session.handoffs = []
    return session


@pytest.mark.parametrize("failure", ["network-close", "poisoned-engine", "network-abort"])
def test_terminal_cleanup_releases_empty_clock_owner_after_independent_failures(shared, monkeypatch, failure):
    session = bare_serving_session(shared)
    calls = []

    def fail_close():
        calls.append("network-close")
        raise RuntimeError("network close failed first")

    def fail_abort():
        calls.append("network-abort")
        raise RuntimeError("network abort failed second")

    monkeypatch.setattr(shared.runtime, "close", fail_close)
    if failure == "poisoned-engine":
        session.engine_runtime.invalidate("engine failed first")
    if failure == "network-abort":
        monkeypatch.setattr(shared.runtime, "abort", fail_abort)
    session._completion_bridges = {"bridge": NS(close=lambda: calls.append("bridge-close"))}
    session.prefill_engines = [NS(engine_id="engine", llm=NS(llm_engine=NS(
        engine_core=NS(shutdown=lambda: calls.append("engine-close")))))]
    with pytest.raises(RuntimeError, match="shared session cleanup failed") as error:
        session.shutdown()
    assert "failed first" in str(error.value)
    assert session._closed and calls[-2:] == ["bridge-close", "engine-close"]
    replacement = EngineStepRuntime(shared.clock)
    replacement.close()
    shared.runtime.abort = SharedKvHandoffRuntime.abort.__get__(shared.runtime)
    shared.runtime.abort()


def test_cleanup_abort_keeps_the_original_exception(shared, monkeypatch):
    session = bare_serving_session(shared)

    def fail_abort():
        raise RuntimeError("secondary abort failure")

    monkeypatch.setattr(shared.runtime, "abort", fail_abort)
    original = ValueError("original body failure")
    with pytest.raises(ValueError) as error, session:
        raise original
    assert error.value is original and "secondary abort failure" in session._shared_cleanup_error
    replacement = EngineStepRuntime(shared.clock)
    replacement.close()


def test_serialized_pending_policy_override_rejects_before_clock_or_identity_changes(shared, monkeypatch):
    session = bare_serving_session(shared)
    session.engine_runtime.close()
    session.engine_runtime, session._shared_handoff = None, None
    session.config = NS(handoff_policy=DeclaredKvHandoffPolicy.off())
    module = ModuleType("vllm")
    module.SamplingParams = object
    monkeypatch.setitem(sys.modules, "vllm", module)
    with pytest.raises(TypeError, match="original independent session policy owner"):
        session.run_request("rejected", (1,), decode_output_tokens=1,
                            admitted_at_ps=100, handoff_policy=shared.runtime)
    assert shared.clock.now_ps == 0 and not session._request_ids
    assert not shared.runtime.accepted_shards and len(shared.peer.requests) == 1
    shared.runtime.close()


def test_driver_abort_keeps_the_original_exception(shared, monkeypatch):
    session = bare_serving_session(shared)
    original = ValueError("original driver failure")

    def fail_abort():
        raise RuntimeError("secondary abort failure")

    def fail_handoff(states):
        raise original

    monkeypatch.setattr(shared.runtime, "abort", fail_abort)
    session._complete_handoffs = fail_handoff
    with pytest.raises(ValueError) as error:
        session._drive_independent((), shared.runtime, [], [])
    assert error.value is original and session.engine_runtime.failure == str(original)
    assert session._shared_cleanup_error == "secondary abort failure"
    session._shutdown_after_failure()
    assert session._closed
    EngineStepRuntime(shared.clock).close()


def test_guard_abort_failure_cannot_replace_receipt_corruption(shared, monkeypatch):
    receipt = submit(shared)
    object.__setattr__(receipt, "kv_bytes", 8193)

    def fail_abort():
        raise OSError("child cleanup failed")

    monkeypatch.setattr(shared.runtime._session, "abort", fail_abort)
    with pytest.raises(RuntimeError, match="accepted handoff receipt was changed"):
        submit(shared, "request-b")
    assert shared.runtime.poisoned and shared.runtime._abort_error == "child cleanup failed"
    object.__setattr__(receipt, "kv_bytes", 8192)
    with pytest.raises(RuntimeError, match="poisoned"):
        submit(shared, "request-b")


def test_shared_session_pending_engine_work_remains_uncancellable(shared):
    session = bare_serving_session(shared)
    record = StepRecord(0, 0, [ScheduledRequest("busy", RequestPhase.DECODE, 1)])
    receipt = session.engine_runtime.submit("busy", record, StepResult(0, 100, 100),
                                           guard=lambda: None, publish=lambda result: None)
    with pytest.raises(RuntimeError, match="pending engine work"):
        session.shutdown()
    assert session._closed and shared.stream.aborted
    assert session.engine_runtime.pending_for("busy") is receipt
    with pytest.raises(ValueError, match="already has an engine completion authority"):
        EngineStepRuntime(shared.clock)


@pytest.mark.parametrize("entry", ["direct", "body-failure"])
def test_interrupted_shared_teardown_finishes_every_owner_and_preserves_first_failure(shared, monkeypatch, entry):
    session = bare_serving_session(shared)
    interrupt, original, calls = KeyboardInterrupt("cleanup interrupted"), ValueError("serving failed"), []

    def stop_close():
        raise interrupt

    monkeypatch.setattr(shared.runtime, "close", stop_close)
    session._completion_bridges = {"bridge": NS(close=lambda: calls.append("bridge"))}
    session.prefill_engines = [NS(engine_id="native", llm=NS(llm_engine=NS(
        engine_core=NS(shutdown=lambda: calls.append("native")))))]
    if entry == "direct":
        with pytest.raises(KeyboardInterrupt) as error:
            session.shutdown()
        assert error.value is interrupt
    else:
        with pytest.raises(ValueError) as error, session:
            raise original
        assert error.value is original
    assert calls == ["bridge", "native"] and session._closed and shared.stream.aborted
    EngineStepRuntime(shared.clock).close()


def test_composed_engine_packet_arrival_and_prior_eligibility_tie(shared):
    """Actual driver and owners, with finite frontend admissions and framed packets."""
    session = bare_serving_session(shared)
    authority, order = session.engine_runtime, []
    engines = {identity: NS(engine_id=identity, executor=NS(step_records=[]))
               for identity in (*PREFILL, *DECODE)}
    session.prefill_engines = [engines[key] for key in PREFILL]
    session.decode_engines = [engines[key] for key in DECODE]
    first = submit(shared)
    record = StepRecord(0, 0, [ScheduledRequest("busy", RequestPhase.DECODE, 1)])
    authority.submit(PREFILL[0], record, StepResult(0, 100, 100), guard=lambda: None,
                     publish=lambda result: order.append(("engine-completed", shared.clock.now_ps)))
    authority.advance_to(90)
    second = submit(shared, "request-b", producer=1)
    assert second.eligible_at_ps == 100

    def state(identity, receipt, admitted, prefill):
        return NS(request=NS(request_id=identity, admitted_at_ps=admitted), pending_handoff=receipt,
                  prefill=engines[prefill], decode=engines[DECODE[0]], handoff=None,
                  prefill_internal_id=None if receipt is None else "native-" + identity,
                  decode_internal_id=None, decode_record_stop=None, kv_transfer_params={})

    states = [state("request-a", first, 0, PREFILL[0]), state("request-b", second, 0, PREFILL[1]),
              state("arrival", None, 100, PREFILL[0])]
    shared.peer.updates = [("horizon", 90, [], False, 3), ("horizon", 99, [], False, 0),
                           ("completion", 100, [1], False, 4), ("completion", 100, [3], False, 1),
                           ("completion", 100, [2], False, 1), ("completion", 100, [4], True, 1)]
    draining = NS(pending=None, has_work=False)

    def admit_prefill(value):
        assert [item.request_id for item in shared.runtime.joins] == ["request-a", "request-b"]
        order.append(("arrival", shared.clock.now_ps))
        value.prefill_internal_id = "native-arrival"

    def admit_decode(value):
        order.append(("decode-" + value.request.request_id, shared.clock.now_ps))
        value.decode_internal_id = "decode-" + value.request.request_id
        draining.has_work = True

    def drain(consume):
        now = shared.clock.now_ps
        record = StepRecord(0, now, finished_request_ids=[item.request.request_id for item in states])
        engines[DECODE[0]].executor.step_records.append(record)

        def finish(result):
            for value in states:
                value.decode_record_stop = 0
            draining.pending, draining.has_work = None, False
            order.append(("drain", shared.clock.now_ps))

        draining.pending = authority.submit(DECODE[0], record, StepResult(0, 0, now),
                                             guard=lambda: None, publish=finish)

    draining.submit = drain
    session._completion_bridges = {key: NS(pending=None, has_work=False) for key in PREFILL}
    session._completion_bridges[DECODE[0]] = draining
    session._admit_prefill, session._admit_decode = admit_prefill, admit_decode
    publish = session._publish_handoff

    def publish_join(value, join):
        assert order and order[0] == ("engine-completed", 100)
        order.append(("join-" + join.request_id, shared.clock.now_ps))
        publish(value, join)

    session._publish_handoff = publish_join
    session._drive_independent(states, shared.runtime, [], [])
    assert order == [("engine-completed", 100), ("join-request-a", 100), ("join-request-b", 100),
                     ("arrival", 100), ("decode-request-a", 100),
                     ("decode-request-b", 100), ("drain", 100)]
    assert [join.completed_at_ps for join in session.handoffs] == [100, 100]
    assert len(shared.runtime.native_events) == 16 and len(shared.runtime.native_rows) == 4
    assert len(shared.calls) == 1 and not shared.peer.updates
    assert not any(row["verb"] == "inject_at_boundary" for row in shared.peer.requests)
    shared.runtime.close()
    authority.close()


def test_reused_publication_and_late_publication_are_terminal(shared):
    receipt = submit(shared)
    shared.peer.updates = [("completion", 100, [1, 2], True, 8)]
    assert shared.runtime.progress((receipt,), through_ps=None) == 100
    shared.clock.advance_to(101)
    with pytest.raises(RuntimeError, match="missed its exact"):
        shared.runtime.complete_due((receipt,))
    assert shared.runtime.poisoned and shared.stream.aborted and not shared.runtime.joins


def test_reused_receipt_cannot_publish_twice(shared):
    receipt = submit(shared)
    complete(shared, (receipt,), 100, (1, 2))
    with pytest.raises(RuntimeError, match="already published"):
        shared.runtime.complete_due((receipt,))
    assert len(shared.runtime.joins) == 1


def test_partial_injection_retains_first_accepted_shard_and_original_failure(shared):
    def fail_second(request, response):
        if request["verb"] == "inject" and request["sequence"] == 2:
            return {"schema": "simllm-htsim-flow-session-v1", "status": "error", "verb": "error",
                    "terminal": True, "authority_counters": counters(0),
                    "error": {"code": "fixture_failure", "message": "second shard failed"}}
        return response

    shared.peer.mutate = fail_second
    with pytest.raises(FlowSessionError, match="second shard failed"):
        submit(shared)
    assert [row["sequence"] for row in shared.runtime.accepted_shards] == [1]
    assert shared.runtime.poisoned and shared.stream.aborted
    assert shared.runtime.transcript and not shared.runtime.joins
    with pytest.raises(RuntimeError, match="poisoned"):
        submit(shared, "retry")


def test_pending_close_does_not_drain_or_advance(shared):
    submit(shared)
    before = list(shared.peer.requests)
    with pytest.raises(RuntimeError, match="unpublished work"):
        shared.runtime.close()
    assert shared.peer.requests == before and shared.clock.now_ps == 0
    shared.runtime.abort()
    assert shared.stream.aborted


def test_mutable_input_manifests_cannot_reroute_bound_engines(shared):
    shared.deployment.placement.ranks[0].global_rank = 999
    shared.deployment.fabric.nodes.clear()
    shared.bindings[PREFILL[0]] = "foreign"
    receipt = submit(shared)
    assert receipt.sequences == (1, 2)
    assert [row["source"] for row in shared.runtime.accepted_shards] == [0, 1]


@pytest.mark.parametrize("changes", [{"request_id": ""}, {"source_engine_id": "foreign"},
    {"destination_engine_id": PREFILL[0]}, {"kv_bytes": True}, {"kv_bytes": 1}, {"submitted_at_ps": 1}])
def test_invalid_submission_rejects_before_native_mutation(shared, changes):
    values = {"request_id": "a", "source_engine_id": PREFILL[0], "destination_engine_id": DECODE[0],
              "kv_bytes": 8192, "submitted_at_ps": 0}
    values.update(changes)
    before = list(shared.peer.requests)
    with pytest.raises(ValueError):
        shared.runtime.submit(**values)
    assert shared.peer.requests == before


def test_native_foreign_row_is_rejected_by_complete_framed_admission(shared):
    receipt = submit(shared)
    shared.peer.updates = [("completion", 100, [1, 2], True, 8)]

    def corrupt(request, response):
        if request["verb"] == "await_completion":
            response["completion_rows"][0]["destination"] = 3
        return response

    shared.peer.mutate = corrupt
    with pytest.raises(FlowSessionError, match="changed destination"):
        shared.runtime.progress((receipt,), through_ps=None)
    assert shared.runtime.poisoned and shared.stream.aborted and not shared.runtime.joins


def test_zero_submission_delay_rejects_before_starting_another_child(shared):
    with pytest.raises(ValueError, match="positive submission"):
        SharedKvHandoffRuntime(shared.config, ("scripted-native",), session_id="invalid",
            deployment=shared.deployment, engine_nodes=shared.bindings, pcie_submission_ps=0)
    assert len(shared.calls) == 1 and shared.clock.now_ps == 0


def test_duplicate_request_rejects_before_more_shards_are_injected(shared):
    receipt = submit(shared)
    before = list(shared.peer.requests)
    with pytest.raises(ValueError, match="duplicate"):
        submit(shared)
    assert shared.peer.requests == before and shared.runtime.accepted_shards
    assert shared.runtime._pending == {receipt.request_id: receipt}


@pytest.mark.parametrize("kind", ["missing-node", "duplicate-node", "endpoint-count"])
def test_invalid_endpoint_bijection_rejects_before_native_launch(shared, kind):
    config, bindings = shared.config, dict(shared.bindings)
    if kind == "missing-node":
        bindings.pop(PREFILL[0])
    elif kind == "duplicate-node":
        bindings[PREFILL[1]] = bindings[PREFILL[0]]
    else:
        config = replace(config, node_count=5)
    with pytest.raises(ValueError, match="endpoint|exactly once"):
        SharedKvHandoffRuntime(config, ("scripted-native",), session_id="invalid",
            deployment=shared.deployment, engine_nodes=bindings, pcie_submission_ps=10)
    assert len(shared.calls) == 1


def test_shared_policy_replacement_rejects_before_admission_and_clock_change(shared):
    from simllm.adapters.vllm.pd_session import VllmPdRequest

    session = bare_serving_session(shared)
    before = list(shared.peer.requests)
    request = VllmPdRequest("new", (1,), 4, 100)
    with pytest.raises(ValueError, match="policy replacement"):
        session.run_requests((request,), handoff_policy=DeclaredKvHandoffPolicy.off())
    assert session._request_ids == set() and shared.peer.requests == before and shared.clock.now_ps == 0
    session.__exit__(None, None, None)
