"""Run the frozen VLLM-6 MoE geometry and expert group qualification."""

from __future__ import annotations

import argparse

EVIDENCE_AUTHORED_AGAINST = "aeb40ac95cdd8163942297335948c94df0376e04"
VLLM_AUTHORED_AGAINST = "568afb3a13806beb53bb2e6bd518269357b237c0"

GRANITE_INTERMEDIATE_SIZE = 512
GRANITE_TOP_K = 8
GRANITE_EXPERTS = 32

#: cell -> (dp, pcp, tp, enable_expert_parallel, rank, num_local_experts,
#:          num_redundant_experts, pp)
GEOMETRY_INPUTS = {
    "g-dense-world1": (1, 1, 1, False, 0, 32, 0, 1),
    "g-ep-off-tp2": (1, 1, 2, False, 1, 32, 0, 1),
    "g-ep-off-dp2-tp2": (2, 1, 2, False, 3, 32, 0, 1),
    "g-ep-flag-world1": (1, 1, 1, True, 0, 32, 0, 1),
    "g-ep-dp8": (8, 1, 1, True, 5, 32, 0, 1),
    "g-ep-dp2-tp2": (2, 1, 2, True, 3, 32, 0, 1),
    "g-ep-dp2-pcp2-tp2": (2, 2, 2, True, 7, 32, 0, 1),
    "g-ep-dp8-uneven": (8, 1, 1, True, 0, 30, 0, 1),
    "g-ep-dp8-uneven-hi": (8, 1, 1, True, 7, 30, 0, 1),
    "g-ep-dp2-eplb": (2, 1, 1, True, 1, 32, 2, 1),
    "llama-dense": (1, 1, 1, False, 0, None, 0, 1),
    "llama-dense-tp2": (1, 1, 2, False, 1, None, 0, 1),
}

#: cell -> (num_experts, top_k, moe_intermediate_size, local_num_experts)
EXPECTED_GEOMETRY = {
    "g-dense-world1": (32, 8, 512, 32),
    "g-ep-off-tp2": (32, 8, 256, 32),
    "g-ep-off-dp2-tp2": (32, 8, 128, 32),
    "g-ep-flag-world1": (32, 8, 512, 32),
    "g-ep-dp8": (32, 8, 512, 4),
    "g-ep-dp2-tp2": (32, 8, 512, 8),
    "g-ep-dp2-pcp2-tp2": (32, 8, 512, 4),
    "g-ep-dp8-uneven": (30, 8, 512, 4),
    "g-ep-dp8-uneven-hi": (30, 8, 512, 3),
    "g-ep-dp2-eplb": (34, 8, 512, 17),
    "llama-dense": (0, 0, None, 0),
    "llama-dense-tp2": (0, 0, None, 0),
}

#: cell -> expected EP group of the cell's rank, None when the model is dense
EXPECTED_GEOMETRY_EP_RANKS = {
    "g-dense-world1": (0,),
    "g-ep-off-tp2": (0, 1),
    "g-ep-off-dp2-tp2": (0, 1, 2, 3),
    "g-ep-flag-world1": (0,),
    "g-ep-dp8": tuple(range(8)),
    "g-ep-dp2-tp2": (0, 1, 2, 3),
    "g-ep-dp2-pcp2-tp2": tuple(range(8)),
    "g-ep-dp8-uneven": tuple(range(8)),
    "g-ep-dp8-uneven-hi": tuple(range(8)),
    "g-ep-dp2-eplb": (0, 1),
    "llama-dense": None,
    "llama-dense-tp2": None,
}

#: cell -> (external_dp, dp, pp, pcp, tp, rank)
LAYOUT_INPUTS = {
    "layout-dp2-tp2": (1, 2, 1, 1, 2, 3),
    "layout-dp2-pp2-tp2": (1, 2, 2, 1, 2, 6),
    "layout-dp2-pp2-tp2-lo": (1, 2, 2, 1, 2, 1),
    "layout-dp2-pcp2-tp2": (1, 2, 1, 2, 2, 5),
    "layout-extdp2-dp2-tp2": (2, 2, 1, 1, 2, 5),
}

#: cell -> (ep_ranks, ep_rank)
EXPECTED_LAYOUT = {
    "layout-dp2-tp2": ((0, 1, 2, 3), 3),
    "layout-dp2-pp2-tp2": ((2, 3, 6, 7), 2),
    "layout-dp2-pp2-tp2-lo": ((0, 1, 4, 5), 1),
    "layout-dp2-pcp2-tp2": (tuple(range(8)), 5),
    "layout-extdp2-dp2-tp2": ((4, 5, 6, 7), 1),
}

#: cell -> (field, pre-change value, post-change value)
EXPECTED_DIRECTIONS = (
    ("g-ep-dp2-tp2", "moe_intermediate_size", 256, 512),
    ("g-ep-dp2-tp2", "local_num_experts", 16, 8),
    ("g-ep-off-dp2-tp2", "moe_intermediate_size", 256, 128),
)

EXPECTED_SCORED_INSTANCES = 22
EXPECTED_FATAL_GUARDS = ("default-and-dense-behavior", "vllm22-schedule-identity")


def _reference_ep_group(
    external_dp: int, dp: int, pp: int, pcp: int, tp: int, rank: int
) -> tuple[tuple[int, ...], int]:
    """The frozen closed form for vLLM's ExternalDP x DP x PP x PCP x TP layout."""

    stride = pcp * tp
    block = dp * pp * stride
    base = (rank // block) * block
    within = rank % block
    pipeline_stage = (within // stride) % pp
    members = tuple(
        base + ((index * pp + pipeline_stage) * pcp + context) * tp + tensor
        for index in range(dp)
        for context in range(pcp)
        for tensor in range(tp)
    )
    return members, members.index(rank)


def _reference_geometry(
    dp: int,
    pcp: int,
    tp: int,
    enable_expert_parallel: bool,
    rank: int,
    num_local_experts: int | None,
    num_redundant_experts: int,
    pp: int,
) -> tuple[int, int, int | None, int]:
    """The frozen closed form for the pinned vLLM v0.26.0 MoE geometry."""

    if not num_local_experts:
        return 0, 0, None, 0
    num_experts = num_local_experts + num_redundant_experts
    flatten_tp = dp * pcp * tp
    use_ep = enable_expert_parallel and flatten_tp > 1
    ep_size = flatten_tp if use_ep else 1
    moe_tp_size = 1 if use_ep else flatten_tp
    moe_intermediate = GRANITE_INTERMEDIATE_SIZE // moe_tp_size
    _, ep_rank = _reference_ep_group(1, dp, pp, pcp, tp, rank)
    base, remainder = divmod(num_experts, ep_size)
    local = base + 1 if ep_rank < remainder else base
    return num_experts, GRANITE_TOP_K, moe_intermediate, local


def _check_frozen_registry() -> None:
    if set(GEOMETRY_INPUTS) != set(EXPECTED_GEOMETRY):
        raise AssertionError("geometry cell registry is incomplete")
    if set(GEOMETRY_INPUTS) != set(EXPECTED_GEOMETRY_EP_RANKS):
        raise AssertionError("geometry EP group registry is incomplete")
    if set(LAYOUT_INPUTS) != set(EXPECTED_LAYOUT):
        raise AssertionError("layout cell registry is incomplete")

    for cell, inputs in GEOMETRY_INPUTS.items():
        derived = _reference_geometry(*inputs)
        if derived != EXPECTED_GEOMETRY[cell]:
            raise AssertionError(
                f"frozen geometry row {cell} disagrees with its own closed form: "
                f"{EXPECTED_GEOMETRY[cell]} against {derived}"
            )
        num_experts, _, _, local = derived
        expected_group = EXPECTED_GEOMETRY_EP_RANKS[cell]
        if num_experts == 0:
            if expected_group is not None:
                raise AssertionError(f"dense cell {cell} must have no EP group")
            continue
        if expected_group is None:
            raise AssertionError(f"MoE cell {cell} must have an EP group")
        dp, pcp, tp = inputs[0], inputs[1], inputs[2]
        pp, rank = inputs[7], inputs[4]
        members, _ = _reference_ep_group(1, dp, pp, pcp, tp, rank)
        if members != expected_group:
            raise AssertionError(f"frozen EP group row {cell} drifted")
        if not 1 <= local <= num_experts:
            raise AssertionError(f"cell {cell} violates the local expert floor")

    # sum over EP ranks of local_num_experts == num_experts, wherever expert
    # parallelism is actually in use; without it every rank holds every expert
    # and the experts are tensor-sharded instead
    for cell, inputs in GEOMETRY_INPUTS.items():
        num_experts = EXPECTED_GEOMETRY[cell][0]
        dp, pcp, tp = inputs[0], inputs[1], inputs[2]
        pp = inputs[7]
        if num_experts == 0 or not (inputs[3] and dp * pcp * tp > 1):
            continue
        members, _ = _reference_ep_group(1, dp, pp, pcp, tp, inputs[4])
        total = sum(
            _reference_geometry(
                dp, pcp, tp, inputs[3], member, inputs[5], inputs[6], pp
            )[3]
            for member in members
        )
        if total != num_experts:
            raise AssertionError(
                f"cell {cell} loses experts across its EP group: {total} of "
                f"{num_experts}"
            )

    for cell, inputs in LAYOUT_INPUTS.items():
        members, index = _reference_ep_group(*inputs)
        if (members, index) != EXPECTED_LAYOUT[cell]:
            raise AssertionError(
                f"frozen layout row {cell} disagrees with its own closed form: "
                f"{EXPECTED_LAYOUT[cell]} against {(members, index)}"
            )
        if len(members) != inputs[1] * inputs[3] * inputs[4]:
            raise AssertionError(f"layout row {cell} has the wrong EP world size")
        if len(set(members)) != len(members):
            raise AssertionError(f"layout row {cell} repeats a rank")

    for cell, field, before, after in EXPECTED_DIRECTIONS:
        index = {"moe_intermediate_size": 2, "local_num_experts": 3}[field]
        if EXPECTED_GEOMETRY[cell][index] != after:
            raise AssertionError(f"direction row {cell}/{field} disagrees with F1")
        if before == after:
            raise AssertionError(f"direction row {cell}/{field} claims no change")

    if EXPECTED_SCORED_INSTANCES != (
        len(GEOMETRY_INPUTS) + len(LAYOUT_INPUTS) + len(EXPECTED_DIRECTIONS) + 2
    ):
        raise AssertionError("scored instance arithmetic drifted")
    if len(EXPECTED_FATAL_GUARDS) != 2:
        raise AssertionError("fatal guard registry drifted")
    # the weight-byte companion stated in the freeze
    if 512 * 8 != 256 * 16:
        raise AssertionError("weight-byte companion arithmetic drifted")


def check_only(args: argparse.Namespace) -> None:
    _check_frozen_registry()
    print(
        f"check-only out={args.out}; validated {len(GEOMETRY_INPUTS)} geometry "
        f"cells, {len(LAYOUT_INPUTS)} layout cells and "
        f"{EXPECTED_SCORED_INSTANCES} scored instances, and produced no artifacts"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="vllm_moe_geometry_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        check_only(args)
        return
    raise SystemExit(
        "the production half of this study is not implemented yet; this commit "
        "freezes expectations only"
    )


if __name__ == "__main__":
    main()
