"""Scripted native frames isolate handoff ownership and exact visibility."""

from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
from test_flow_session import Peer, Stream, counters

from simllm.backends import flow_session
from simllm.backends.flow_session import FlowSessionConfig, FlowSessionError
from simllm.core import (
    DisaggregatedRequestTimeline,
    KvHandoffJoin,
    PendingKvHandoffPolicy,
    VirtualClock,
)
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


@pytest.mark.parametrize("kind", ["missing", "duplicate", "forged", "bytes", "sequence"])
def test_bad_private_receipt_inventory_poison_and_reap(shared, kind):
    receipt = submit(shared)
    pending = (receipt,)
    if kind == "missing":
        pending = ()
    elif kind == "duplicate":
        pending = (receipt, receipt)
    elif kind == "forged":
        pending = (replace(receipt),)
    else:
        object.__setattr__(receipt, "kv_bytes" if kind == "bytes" else "sequences",
                           8193 if kind == "bytes" else (3, 4))
    before = list(shared.peer.requests)
    with pytest.raises(RuntimeError, match="receipt|inventory"):
        shared.runtime.complete_due(pending)
    assert shared.runtime.poisoned and shared.stream.aborted
    assert shared.peer.requests == before


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
