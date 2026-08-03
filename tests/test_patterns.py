import re

from simllm.goal import GoalTrace
from simllm.traffic import (
    binomial_broadcast,
    gather,
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


def test_binomial_broadcast_rounds():
    trace = GoalTrace(8)
    done = binomial_broadcast(trace, root=0, ranks=list(range(8)), size_bytes=2048, tag=3)
    text = trace.render()
    # 7 transmissions total for 8 ranks, in log2(8)=3 chained rounds
    assert text.count("send 2048b") == 7
    assert set(done) == set(range(8))
