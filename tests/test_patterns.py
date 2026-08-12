import re

import pytest

from simllm.goal import GoalTrace
from simllm.traffic import (
    binomial_broadcast,
    gather,
    ordered_pairwise_messages,
    pairwise_all_to_allv,
    ring_allreduce,
    scatter,
)


def test_scatter_gather_chain():
    trace = GoalTrace(4)
    start = {0: trace.rank(0).calc(1000)}
    sdone = scatter(trace, root=0, workers=[1, 2, 3], size_bytes=4096, tag=1, after=start)
    cdone = {w: trace.rank(w).calc(500) for w in [1, 2, 3]}
    for w in [1, 2, 3]:
        trace.rank(w).requires(cdone[w], sdone[w])
    gather(trace, root=0, workers=[1, 2, 3], size_bytes=4096, tag=2, after=cdone)

    text = trace.render()
    assert text.count("send 4096b") == 6
    assert text.count("recv 4096b") == 6
    # every scatter send on rank 0 waits for the arrival calc
    assert len(re.findall(r"r0op\d+ requires r0op0", text)) == 3


def test_ring_allreduce_round_structure():
    trace = GoalTrace(4)
    done = ring_allreduce(trace, ranks=[0, 1, 2, 3], size_bytes=4096, base_tag=10)
    text = trace.render()
    # 2*(W-1) rounds, one send+recv per rank per round, chunk = size/W
    assert text.count("send 1024b") == 4 * 6
    assert text.count("recv 1024b") == 4 * 6
    assert set(done) == {0, 1, 2, 3}
    # rounds are chained: each round's ops require the previous round's recv
    assert "requires" in text


def test_pairwise_all_to_allv_skips_zero_and_self():
    trace = GoalTrace(3)
    done = pairwise_all_to_allv(
        trace, ranks=[0, 1, 2],
        send_bytes={(0, 1): 100, (1, 0): 200, (0, 0): 999, (2, 1): 0},
        tag=5,
    )
    text = trace.render()
    assert text.count("send") == 2
    assert "send 999b" not in text
    assert 2 not in done


def test_pairwise_all_to_allv_records_request_partition_without_splitting_send():
    trace = GoalTrace(2)
    pairwise_all_to_allv(
        trace,
        ranks=[0, 1],
        send_bytes={(0, 1): 12},
        tag=5,
        operation_id="a2av",
        request_send_bytes={(0, 1): (("alpha", 4), ("beta", 8))},
    )

    assert trace.render().count(": send ") == 1
    assert trace.messages[0].payload_bytes == 12
    assert trace.messages[0].request_payload_bytes == (("alpha", 4), ("beta", 8))


def test_pairwise_all_to_allv_rejects_incomplete_request_partition():
    trace = GoalTrace(2)
    with pytest.raises(ValueError, match="cover exactly"):
        pairwise_all_to_allv(
            trace,
            ranks=[0, 1],
            send_bytes={(0, 1): 12},
            tag=5,
            operation_id="a2av",
            request_send_bytes={},
        )


def test_ordered_pairwise_messages_preserves_rows_and_source_issue_order():
    trace = GoalTrace(3)
    start = {rank: trace.rank(rank).calc(1) for rank in range(3)}
    done = ordered_pairwise_messages(
        trace,
        ranks=[0, 1, 2],
        messages=(
            ("alpha", 0, 2, 4),
            ("beta", 1, 2, 5),
            ("alpha", 0, 1, 6),
        ),
        tag=7,
        after=start,
        operation_id="sequenced-a2av",
    )

    assert [
        (
            message.request_payload_bytes[0][0],
            message.source_rank,
            message.destination_rank,
            message.payload_bytes,
        )
        for message in trace.messages
    ] == [
        ("alpha", 0, 2, 4),
        ("beta", 1, 2, 5),
        ("alpha", 0, 1, 6),
    ]
    first, _, third = trace.messages
    source_order = [
        dependency
        for dependency in trace.dependencies
        if dependency.operation_label == third.send_label
    ]
    assert len(source_order) == 1
    assert source_order[0].relation == "irequires"
    assert source_order[0].predecessor_label == first.send_label
    assert set(done) == {0, 1, 2}
    assert all(
        any(
            dependency.operation_label == frontier
            and dependency.relation == "requires"
            for dependency in trace.dependencies
        )
        for frontier in done.values()
    )


def test_ordered_pairwise_messages_rejects_before_mutating_without_identity():
    trace = GoalTrace(2)

    with pytest.raises(ValueError, match="require operation_id"):
        ordered_pairwise_messages(
            trace,
            ranks=[0, 1],
            messages=(("alpha", 0, 1, 4),),
            tag=7,
        )

    assert trace.operations == ()
    assert trace.messages == ()


def test_binomial_broadcast_rounds():
    trace = GoalTrace(8)
    done = binomial_broadcast(trace, root=0, ranks=list(range(8)), size_bytes=2048, tag=3)
    text = trace.render()
    # 7 transmissions total for 8 ranks, in log2(8)=3 chained rounds
    assert text.count("send 2048b") == 7
    assert set(done) == set(range(8))
