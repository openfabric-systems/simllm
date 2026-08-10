"""Tests for the per-step collective mapping (simllm.traffic.step_comm)."""

import pytest

from simllm.compute import ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.traffic import (
    render_step_goal,
    step_moe_alltoalls,
    step_tp_allreduces,
)

TINY_DIMS = ModelDims(
    num_layers=2,
    hidden_size=4,
    intermediate_size=8,
    num_heads=2,
    num_kv_heads=2,
    head_size=2,
    vocab_size=16,
    dtype_bytes=2,
)

TINY_MOE_DIMS = ModelDims(
    num_layers=2,
    hidden_size=4,
    intermediate_size=8,
    num_heads=2,
    num_kv_heads=2,
    head_size=2,
    vocab_size=16,
    dtype_bytes=2,
    num_experts=8,
    top_k=2,
    moe_intermediate_size=4,
    local_num_experts=2,
)


def decode_record(num_new_tokens: int = 3) -> StepRecord:
    return StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest("a", RequestPhase.DECODE,
                             num_new_tokens=num_new_tokens, context_length=8)
        ],
    )


def test_step_tp_allreduces_counts_and_payload():
    ops = step_tp_allreduces(decode_record(), TINY_DIMS, [0, 1, 2, 3])
    # 2 allreduces per layer, in layer-major (attention, mlp) order
    assert len(ops) == 2 * TINY_DIMS.num_layers
    assert [(op.layer, op.site) for op in ops] == [
        (0, "attention"), (0, "mlp"), (1, "attention"), (1, "mlp")]
    # payload = total_new_tokens * hidden * dtype_bytes, same for every op
    assert {op.payload_bytes for op in ops} == {3 * 4 * 2}
    assert {op.ranks for op in ops} == {(0, 1, 2, 3)}


def test_step_tp_allreduces_sums_all_scheduled_tokens():
    record = StepRecord(step_index=0, virtual_time_ps=0, scheduled=[
        ScheduledRequest("p", RequestPhase.PREFILL, num_new_tokens=5, context_length=5),
        ScheduledRequest("d", RequestPhase.DECODE, num_new_tokens=1, context_length=9),
    ])
    ops = step_tp_allreduces(record, TINY_DIMS, [0, 1])
    assert ops[0].payload_bytes == 6 * 4 * 2


def test_step_tp_allreduces_empty_cases():
    # TP world of 1: nothing to reduce across
    assert step_tp_allreduces(decode_record(), TINY_DIMS, [0]) == []
    # drain record: zero new tokens
    drain = StepRecord(step_index=9, virtual_time_ps=100, finished_request_ids=["a"])
    assert step_tp_allreduces(drain, TINY_DIMS, [0, 1]) == []


def test_render_step_goal_refuses_empty_step():
    with pytest.raises(ValueError, match="no tensor-parallel collectives"):
        render_step_goal(decode_record(), TINY_DIMS, [0], per_layer_calc_ns=1)


def test_render_step_goal_rejects_too_few_goal_ranks():
    with pytest.raises(ValueError, match="cannot contain rank 3"):
        render_step_goal(
            decode_record(),
            TINY_DIMS,
            [2, 3],
            per_layer_calc_ns=1,
            num_goal_ranks=3,
        )


def test_render_step_goal_accepts_unequal_layer_calcs():
    trace = render_step_goal(
        decode_record(),
        TINY_DIMS,
        [0, 1],
        per_layer_calc_ns=(3, 7),
    )
    text = trace.render()
    assert text.count("calc 3") == 2
    assert text.count("calc 7") == 2


@pytest.mark.parametrize(
    ("layer_calc_ns", "message"),
    [((1,), "received 1"), ((1, -1), "nonnegative")],
)
def test_render_step_goal_rejects_invalid_layer_calcs(layer_calc_ns, message):
    with pytest.raises(ValueError, match=message):
        render_step_goal(
            decode_record(),
            TINY_DIMS,
            [0, 1],
            per_layer_calc_ns=layer_calc_ns,
        )


def test_render_step_goal_structure():
    trace = render_step_goal(decode_record(), TINY_DIMS, [0, 1, 2, 3],
                             per_layer_calc_ns=42)
    text = trace.render()
    # chunk = 24 // 4 = 6; per allreduce 2*(W-1) = 6 rounds of W sends each,
    # 4 allreduces total
    assert text.count("send 6b") == 4 * 6 * 4
    assert text.count("recv 6b") == 4 * 6 * 4
    # one calc per layer per rank
    assert text.count("calc 42") == 2 * 4
    # disjoint tag blocks: op k uses tags base + k*6 .. base + k*6 + 5
    for op_index in range(4):
        for round_index in range(6):
            assert f"tag {1000 + op_index * 6 + round_index}" in text
    assert f"tag {1000 + 4 * 6}" not in text


GOLDEN_TINY_STEP = """num_ranks 3
rank 0 {
r0op0: calc 7
r0op1: send 12b to 1 tag 1000
r0op2: recv 12b from 1 tag 1000
r0op3: send 12b to 1 tag 1001
r0op4: recv 12b from 1 tag 1001
r0op5: send 12b to 1 tag 1002
r0op6: recv 12b from 1 tag 1002
r0op7: send 12b to 1 tag 1003
r0op8: recv 12b from 1 tag 1003
r0op9: calc 7
r0op10: send 12b to 1 tag 1004
r0op11: recv 12b from 1 tag 1004
r0op12: send 12b to 1 tag 1005
r0op13: recv 12b from 1 tag 1005
r0op14: send 12b to 1 tag 1006
r0op15: recv 12b from 1 tag 1006
r0op16: send 12b to 1 tag 1007
r0op17: recv 12b from 1 tag 1007
r0op1 requires r0op0
r0op2 requires r0op0
r0op3 requires r0op2
r0op4 requires r0op2
r0op5 requires r0op4
r0op6 requires r0op4
r0op7 requires r0op6
r0op8 requires r0op6
r0op9 requires r0op8
r0op10 requires r0op9
r0op11 requires r0op9
r0op12 requires r0op11
r0op13 requires r0op11
r0op14 requires r0op13
r0op15 requires r0op13
r0op16 requires r0op15
r0op17 requires r0op15
}
rank 1 {
r1op0: calc 7
r1op1: send 12b to 0 tag 1000
r1op2: recv 12b from 0 tag 1000
r1op3: send 12b to 0 tag 1001
r1op4: recv 12b from 0 tag 1001
r1op5: send 12b to 0 tag 1002
r1op6: recv 12b from 0 tag 1002
r1op7: send 12b to 0 tag 1003
r1op8: recv 12b from 0 tag 1003
r1op9: calc 7
r1op10: send 12b to 0 tag 1004
r1op11: recv 12b from 0 tag 1004
r1op12: send 12b to 0 tag 1005
r1op13: recv 12b from 0 tag 1005
r1op14: send 12b to 0 tag 1006
r1op15: recv 12b from 0 tag 1006
r1op16: send 12b to 0 tag 1007
r1op17: recv 12b from 0 tag 1007
r1op1 requires r1op0
r1op2 requires r1op0
r1op3 requires r1op2
r1op4 requires r1op2
r1op5 requires r1op4
r1op6 requires r1op4
r1op7 requires r1op6
r1op8 requires r1op6
r1op9 requires r1op8
r1op10 requires r1op9
r1op11 requires r1op9
r1op12 requires r1op11
r1op13 requires r1op11
r1op14 requires r1op13
r1op15 requires r1op13
r1op16 requires r1op15
r1op17 requires r1op15
}
rank 2 {
r2op0: calc 0
}
"""


def test_render_step_goal_golden_text():
    """Golden GOAL for a 2-layer, W=2 step: hand-verified serial chain.

    Chunk = 3 tokens * 4 hidden * 2 bytes / 2 ranks = 12 B; each allreduce
    is 2(W-1) = 2 rounds; tags block per allreduce (1000, 1002, 1004, 1006);
    layer 1's calc requires layer 0's last MLP-allreduce recv; rank 2 is
    idle-filled.
    """
    trace = render_step_goal(decode_record(), TINY_DIMS, [0, 1],
                             per_layer_calc_ns=7, num_goal_ranks=3)
    assert trace.render() == GOLDEN_TINY_STEP


def test_render_step_goal_dense_ignores_ep_ranks():
    """Dense dims with an EP group configured: byte-identical to no EP."""
    trace = render_step_goal(decode_record(), TINY_DIMS, [0, 1],
                             per_layer_calc_ns=7, ep_ranks=[0, 1, 2],
                             num_goal_ranks=3)
    assert trace.render() == GOLDEN_TINY_STEP


# ---- MoE all-to-alls (TRAF-2 first half) ----

def test_step_moe_alltoalls_counts_and_payload():
    ops = step_moe_alltoalls(decode_record(), TINY_MOE_DIMS, [0, 1, 2, 3])
    # 2 phases per layer, layer-major (dispatch, combine) order
    assert [(op.layer, op.phase) for op in ops] == [
        (0, "dispatch"), (0, "combine"), (1, "dispatch"), (1, "combine")]
    # per pair: total_new_tokens * top_k * hidden * dtype / W
    assert {op.per_pair_bytes for op in ops} == {3 * 2 * 4 * 2 // 4}
    assert {op.ranks for op in ops} == {(0, 1, 2, 3)}


def test_step_moe_alltoalls_empty_cases():
    # dense dims: no experts, no all-to-alls
    assert step_moe_alltoalls(decode_record(), TINY_DIMS, [0, 1]) == []
    # EP world of 1: everything stays local
    assert step_moe_alltoalls(decode_record(), TINY_MOE_DIMS, [0]) == []
    # drain record: zero new tokens
    drain = StepRecord(step_index=9, virtual_time_ps=100, finished_request_ids=["a"])
    assert step_moe_alltoalls(drain, TINY_MOE_DIMS, [0, 1]) == []


def test_render_step_goal_moe_only_structure():
    """EP-only step (TP world of 1): calc, dispatch, combine per layer."""
    trace = render_step_goal(decode_record(), TINY_MOE_DIMS, [0],
                             per_layer_calc_ns=5, ep_ranks=[0, 1, 2, 3])
    text = trace.render()
    # per pair 12 B; per a2av W(W-1) = 12 sends, 2 a2avs per layer, 2 layers
    assert text.count("send 12b") == 2 * 2 * 12
    assert text.count("recv 12b") == 2 * 2 * 12
    # one calc per layer on each of the 4 EP ranks, no idle fillers
    assert text.count("calc 5") == 2 * 4
    assert text.count("calc 0") == 0
    # no TP ops: a2av j takes tag base_tag + j
    for tag in (1000, 1001, 1002, 1003):
        assert f"tag {tag}" in text
    assert "tag 1004" not in text


def test_render_step_goal_moe_with_tp_structure():
    """TP=2 inside EP=[0..3]: allreduces then a2avs, disjoint tag spaces."""
    trace = render_step_goal(decode_record(), TINY_MOE_DIMS, [0, 1],
                             per_layer_calc_ns=5, ep_ranks=[0, 1, 2, 3])
    text = trace.render()
    # TP payload = 24 B over W=2: chunk 12 B, 2 rounds per allreduce, 2
    # sends per round; 4 allreduces total; a2avs also move 12 B pairs
    assert text.count("send 12b") == 4 * 2 * 2 + 2 * 2 * 12
    # allreduce tag blocks 1000..1007 (stride 2), then a2avs at 1008..1011
    for tag in (1000, 1002, 1004, 1006, 1008, 1009, 1010, 1011):
        assert f"tag {tag}" in text
    assert "tag 1012" not in text
    # every participant calcs every layer
    assert text.count("calc 5") == 2 * 4


def test_render_step_goal_moe_chains_phases():
    """Layer 1's calc on an EP-only rank requires layer 0's combine recv."""
    trace = render_step_goal(decode_record(), TINY_MOE_DIMS, [0],
                             per_layer_calc_ns=5, ep_ranks=[0, 1])
    text = trace.render()
    # rank 0, W=2: per layer 1 calc + dispatch (send+recv) + combine
    # (send+recv): ops 0..4 in layer 0, layer 1 calc is op 5 and must
    # require the combine recv (op 4)
    assert "r0op5 requires r0op4" in text
