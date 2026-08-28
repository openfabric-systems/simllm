"""Tests for the per-step collective mapping (simllm.traffic.step_comm)."""

import hashlib
from dataclasses import replace

import pytest

from simllm.compute import ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.traffic import (
    EXPERT_PARALLEL_TP_ALLREDUCE_SITES,
    TP_ALLREDUCE_SITES,
    layer_tp_allreduce_sites,
    render_step_goal,
    renders_expert_combine,
    step_communication_phases,
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

#: the same mixture with no expert parallelism, i.e. every expert resident and
#: tensor-sharded across the TP group, which keeps both allreduce sites
TINY_SHARDED_MOE_DIMS = replace(TINY_MOE_DIMS, local_num_experts=8)


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


# ---- which allreduce sites a layer has (TRAF-33) ----

def test_layer_tp_allreduce_sites_follow_the_declared_all_to_all():
    """A layer reduces its MLP output once, in exactly one mechanism."""
    record = decode_record()
    # dense: no combine exists, so both row-parallel sites reduce
    assert layer_tp_allreduce_sites(record, TINY_DIMS) == TP_ALLREDUCE_SITES
    assert (
        layer_tp_allreduce_sites(record, TINY_DIMS, ep_ranks=[0, 1, 2, 3])
        == TP_ALLREDUCE_SITES
    )
    # routed with a declared reducing all-to-all group: the combine returns it
    assert (
        layer_tp_allreduce_sites(record, TINY_MOE_DIMS, ep_ranks=[0, 1])
        == EXPERT_PARALLEL_TP_ALLREDUCE_SITES
        == ("attention",)
    )
    # routed with no declared group is naive expert parallelism: vLLM
    # 0.26.0 config.py:1052-1055 needs dp, pcp or sequence parallelism for
    # all-to-all kernels, and runner/moe_runner.py:436-465 then all-reduces
    # the unreduced fused output over the TP group
    assert layer_tp_allreduce_sites(record, TINY_MOE_DIMS) == TP_ALLREDUCE_SITES
    # a one-rank group renders no combine either, so both sites survive
    assert (
        layer_tp_allreduce_sites(record, TINY_MOE_DIMS, ep_ranks=[0])
        == TP_ALLREDUCE_SITES
    )
    # expert-tensor-sharded dims behave the same way
    assert (
        layer_tp_allreduce_sites(record, TINY_SHARDED_MOE_DIMS) == TP_ALLREDUCE_SITES
    )


def test_every_layer_reduces_its_mlp_output_exactly_once():
    """The degenerate uniform share must not delete the only reduction.

    With a per-pair share that floors to zero bytes, step_moe_alltoalls
    renders no combine at all, so suppressing the mlp site would leave that
    layer's MLP output reduced zero times.
    """
    degenerate = ModelDims(
        num_layers=2,
        hidden_size=4,
        intermediate_size=8,
        num_heads=2,
        num_kv_heads=2,
        head_size=2,
        vocab_size=16,
        dtype_bytes=1,
        num_experts=8,
        top_k=1,
        moe_intermediate_size=4,
        local_num_experts=1,
    )
    record = decode_record(num_new_tokens=1)
    ep_ranks = tuple(range(8))
    # one token times top_k 1 times 4 activation bytes over 8 ranks floors to 0
    assert step_moe_alltoalls(record, degenerate, ep_ranks) == []
    assert not renders_expert_combine(record, degenerate, ep_ranks)
    operations = step_tp_allreduces(record, degenerate, [0, 1], ep_ranks=ep_ranks)
    assert [op.site for op in operations] == ["attention", "mlp"] * 2

    # and the invariant holds across every representable shape nearby
    for tokens in (0, 1, 3, 12):
        for ep_width in (0, 1, 2, 8):
            for dims in (TINY_DIMS, TINY_MOE_DIMS, degenerate):
                group = tuple(range(ep_width))
                step = decode_record(num_new_tokens=tokens)
                combines = step_moe_alltoalls(step, dims, group)
                sites = step_tp_allreduces(step, dims, [0, 1], ep_ranks=group)
                for layer in range(dims.num_layers):
                    reductions = sum(
                        1
                        for op in combines
                        if op.layer == layer and op.phase == "combine"
                    ) + sum(
                        1 for op in sites if op.layer == layer and op.site == "mlp"
                    )
                    # zero only when the step renders no traffic at all
                    assert reductions == (0 if not sites and not combines else 1)


def test_step_tp_allreduces_naive_expert_parallelism_keeps_both_sites():
    """Undeclared expert parallelism renders 2 allreduces and no all-to-all."""
    record = decode_record()
    operations = step_tp_allreduces(record, TINY_MOE_DIMS, [0, 1, 2, 3])
    assert [op.site for op in operations] == ["attention", "mlp"] * 2
    assert step_moe_alltoalls(record, TINY_MOE_DIMS, ()) == []


def test_step_tp_allreduces_sites_do_not_depend_on_group_widths():
    """The site tuple is fixed by the declaration, not by any width."""
    record = decode_record()
    for tp_ranks in ([0, 1], [0, 1, 2, 3], [8, 9, 10, 11], list(range(8))):
        undeclared = step_tp_allreduces(record, TINY_MOE_DIMS, tp_ranks)
        assert [op.site for op in undeclared] == ["attention", "mlp"] * 2
        for ep_ranks in ([0, 1], [0, 1, 2, 3], [16, 17], list(range(8))):
            declared = step_tp_allreduces(
                record, TINY_MOE_DIMS, tp_ranks, ep_ranks=ep_ranks
            )
            assert [op.site for op in declared] == ["attention"] * 2
            assert {op.payload_bytes for op in declared + undeclared} == {24}


def test_expert_parallel_moe_renders_one_allreduce_and_two_a2avs_per_layer():
    """24 layers, TP 8 and EP 8: 24 + 48 = 72 collectives, not 96."""
    dims = replace(TINY_MOE_DIMS, num_layers=24, num_experts=32,
                   local_num_experts=4)
    record = decode_record()
    tp_ranks = tuple(range(8))
    ep_ranks = tuple(range(8))

    tp_ops = step_tp_allreduces(record, dims, tp_ranks, ep_ranks=ep_ranks)
    moe_ops = step_moe_alltoalls(record, dims, ep_ranks)
    assert len(tp_ops) == 24
    assert {op.site for op in tp_ops} == {"attention"}
    assert len(moe_ops) == 48
    assert len(tp_ops) + len(moe_ops) == 72

    trace = render_step_goal(record, dims, tp_ranks, per_layer_calc_ns=5,
                             ep_ranks=ep_ranks)
    operation_ids = {message.operation_id for message in trace.messages}
    assert len(operation_ids) == 72
    assert sum(1 for name in operation_ids if ":tp-" in name) == 24
    assert sum(1 for name in operation_ids if ":ep-" in name) == 48
    # every ring round of every site owns a tag no other operation uses
    tags_by_operation: dict[str, set[int]] = {}
    for message in trace.messages:
        tags_by_operation.setdefault(message.operation_id, set()).add(message.tag)
    seen: set[int] = set()
    for tags in tags_by_operation.values():
        assert not (tags & seen)
        seen |= tags
    # the all-to-all block starts right after the 24 shortened ring blocks
    moe_tags = {
        tag
        for name, tags in tags_by_operation.items()
        if ":ep-" in name
        for tag in tags
    }
    assert min(moe_tags) == 1000 + 24 * 2 * (len(tp_ranks) - 1)


def test_uniform_full_population_emits_every_directed_ep_pair():
    dims = replace(
        TINY_MOE_DIMS,
        num_layers=1,
        hidden_size=3072,
        dtype_bytes=1,
        num_experts=256,
        top_k=8,
        local_num_experts=1,
    )
    record = decode_record(num_new_tokens=4)
    ep_ranks = tuple(range(256))

    single_engine = step_moe_alltoalls(record, dims, ep_ranks)
    full_population = step_moe_alltoalls(
        record,
        dims,
        ep_ranks,
        uniform_tokens_per_rank=4,
    )
    assert sum(len(operation.pair_payload_bytes) for operation in single_engine) == 510
    assert sum(len(operation.pair_payload_bytes) for operation in full_population) == 130560
    assert {
        payload_bytes
        for operation in full_population
        for _, _, payload_bytes in operation.pair_payload_bytes
    } == {384}
    assert sum(
        payload_bytes
        for operation in full_population
        for _, _, payload_bytes in operation.pair_payload_bytes
    ) == 50135040
    assert 130560 * 65 == 8486400
    assert 50135040 * 65 == 3258777600

    phases = step_communication_phases(
        record,
        dims,
        [0],
        ep_ranks=ep_ranks,
        uniform_tokens_per_rank=4,
    )
    assert len(phases) == 2
    assert all(len(phase.segments) == 65280 for phase in phases)


@pytest.mark.parametrize("tokens", (0, 3, 5))
def test_uniform_full_population_rejects_invalid_per_rank_count(tokens):
    with pytest.raises(ValueError, match="uniform_tokens_per_rank"):
        step_moe_alltoalls(
            decode_record(num_new_tokens=4),
            TINY_MOE_DIMS,
            [0, 1],
            uniform_tokens_per_rank=tokens,
        )


def test_dense_and_expert_sharded_goal_text_is_unchanged():
    """Neither arm's rendered GOAL moved a byte across TRAF-33.

    The digest was taken from the renderer at the pre-change revision
    e18b9b0102808e9b8e0f276c2b82c51ed8c5b51d.
    """
    for dims in (TINY_DIMS, TINY_SHARDED_MOE_DIMS):
        text = render_step_goal(
            decode_record(), dims, [0, 1, 2, 3], per_layer_calc_ns=(7, 11)
        ).render()
        assert hashlib.sha256(text.encode()).hexdigest() == (
            "c53782b27c241a85b37f9d81342ed8618e4402a8d2c6c3c5dbe4e59a1a587301"
        )


def test_expert_parallel_moe_communication_phases_match_the_renderer():
    """The phase planner emits the same shortened ring inventory."""
    record = decode_record()
    phases = step_communication_phases(
        record, TINY_MOE_DIMS, [0, 1], ep_ranks=[0, 1, 2, 3]
    )
    ring_phases = [phase for phase in phases if ":tp-" in phase.phase_id]
    assert {phase.phase_id.split(":")[1] for phase in ring_phases} == {
        "tp-attention"
    }
    # 2 layers x 1 site x 2(W-1) rounds, each round W directed segments
    assert len(ring_phases) == 2 * 1 * 2
    assert sum(
        segment.payload_bytes
        for phase in ring_phases
        for segment in phase.segments
    ) == 2 * 1 * 2 * 2 * 12


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
    # One engine's uniform share is a sparse dispatch star and its transpose.
    per_pair = 3 * 2 * 4 * 2 // 4
    assert {op.per_pair_bytes for op in ops} == {0}
    assert all(
        op.pair_payload_bytes
        == (
            tuple((0, destination, per_pair) for destination in (1, 2, 3))
            if op.phase == "dispatch"
            else tuple((source, 0, per_pair) for source in (1, 2, 3))
        )
        for op in ops
    )
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
    # Per pair is 12 B. One source star has W-1 sends per phase.
    assert text.count("send 12b") == 2 * 2 * 3
    assert text.count("recv 12b") == 2 * 2 * 3
    # one calc per layer on each of the 4 EP ranks, no idle fillers
    assert text.count("calc 5") == 2 * 4
    assert text.count("calc 0") == 0
    # no TP ops: a2av j takes tag base_tag + j
    for tag in (1000, 1001, 1002, 1003):
        assert f"tag {tag}" in text
    assert "tag 1004" not in text


def test_render_step_goal_moe_with_tp_structure():
    """TP=2 inside EP=[0..3]: allreduces then a2avs, disjoint tag spaces.

    The experts are expert-parallel here (2 of 8 resident), so each layer
    reduces once, after attention, and the a2av tag base moves down with the
    shortened allreduce list instead of leaving a hole (TRAF-33).
    """
    trace = render_step_goal(decode_record(), TINY_MOE_DIMS, [0, 1],
                             per_layer_calc_ns=5, ep_ranks=[0, 1, 2, 3])
    text = trace.render()
    # TP payload = 24 B over W=2: chunk 12 B, 2 rounds per allreduce, 2
    # sends per round; 2 allreduces total; MoE uses one three-peer star.
    assert text.count("send 12b") == 2 * 2 * 2 + 2 * 2 * 3
    # allreduce tag blocks 1000..1003 (stride 2), then a2avs at 1004..1007
    for tag in (1000, 1002, 1004, 1005, 1006, 1007):
        assert f"tag {tag}" in text
    assert "tag 1008" not in text
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
