"""Run the frozen VLLM-24 independent routed-MoE byte conservation study."""

from __future__ import annotations

import argparse

EVIDENCE_AUTHORED_AGAINST = "aeb40ac95cdd8163942297335948c94df0376e04"
VLLM_AUTHORED_AGAINST = "568afb3a13806beb53bb2e6bd518269357b237c0"

#: the nine frozen conservation rules, in evaluation order
ALWAYS_RULES = (
    "source-attribution",
    "destination-legality",
    "owner-egress",
    "transpose-symmetry",
    "step-hop-bound",
)
CAPTURED_ONLY_RULES = (
    "vector-granularity",
    "request-identity",
    "per-request-hop-bound",
    "per-layer-hop-bound",
)

#: captured Granite geometry, read off the tracked preplay trace header
EXPERT_COUNT = 32
TOP_K = 8
NUM_LAYERS = 24
HIDDEN_SIZE = 1_024
DTYPE_BYTES = 2
VECTOR_BYTES = HIDDEN_SIZE * DTYPE_BYTES

#: frozen steps: name -> total_new_tokens
STEP_TOKENS = {"prefill": 22, "decode": 1}
EP_WORLDS = (2, 8)
ARMS = ("owner-attributed", "source-replicated")

#: exact closed-form oracles, hops
EXPECTED_STEP_HOP_BOUND = {"prefill": 8_448, "decode": 384}
EXPECTED_PER_LAYER_HOP_BOUND = {
    ("prefill", 2): 22,
    ("prefill", 8): 154,
    ("decode", 2): 1,
    ("decode", 8): 7,
}
#: hops_A ceilings, i.e. T * min(top_k, W - 1) * num_layers * 2
EXPECTED_ARM_A_HOP_CEILING = {
    ("prefill", 2): 1_056,
    ("prefill", 8): 7_392,
    ("decode", 2): 48,
    ("decode", 8): 336,
}
#: hops_A floors from the block structure: at W = 8 every token touches at
#: least ceil(top_k / (32 // 8)) = 2 owner blocks, so at least one is remote
EXPECTED_ARM_A_HOP_FLOOR = {
    ("prefill", 2): 0,
    ("prefill", 8): 1_056,
    ("decode", 2): 0,
    ("decode", 8): 48,
}

#: which rule detects the replicated arm in which cell
EXPECTED_DETECTION = {
    ("prefill", 2, "source-attribution"): True,
    ("prefill", 8, "source-attribution"): True,
    ("decode", 2, "source-attribution"): True,
    ("decode", 8, "source-attribution"): True,
    ("prefill", 2, "step-hop-bound"): False,
    ("prefill", 8, "step-hop-bound"): True,
    ("decode", 2, "step-hop-bound"): False,
    ("decode", 8, "step-hop-bound"): True,
}

EXPECTED_SCORED_INSTANCES = 9
EXPECTED_FATAL_GUARDS = ("arm-a-conserves", "replication-multiplier")


def _check_frozen_registry() -> None:
    if len(ALWAYS_RULES) + len(CAPTURED_ONLY_RULES) != 9:
        raise AssertionError("the frozen rule registry is not the nine named rules")
    if set(ALWAYS_RULES) & set(CAPTURED_ONLY_RULES):
        raise AssertionError("a rule appears in both applicability classes")
    if VECTOR_BYTES != 2_048:
        raise AssertionError("hidden vector byte arithmetic drifted")

    for step, tokens in STEP_TOKENS.items():
        bound = tokens * TOP_K * NUM_LAYERS * 2
        if EXPECTED_STEP_HOP_BOUND[step] != bound:
            raise AssertionError(f"step hop bound arithmetic drifted at {step}")
        for world in EP_WORLDS:
            per_layer = tokens * min(TOP_K, world - 1)
            if EXPECTED_PER_LAYER_HOP_BOUND[(step, world)] != per_layer:
                raise AssertionError(
                    f"per-layer hop bound arithmetic drifted at {step} W={world}"
                )
            ceiling = per_layer * NUM_LAYERS * 2
            if EXPECTED_ARM_A_HOP_CEILING[(step, world)] != ceiling:
                raise AssertionError(
                    f"arm A hop ceiling arithmetic drifted at {step} W={world}"
                )
            floor = EXPECTED_ARM_A_HOP_FLOOR[(step, world)]
            if floor < 0 or floor > ceiling:
                raise AssertionError(
                    f"arm A physical interval is empty at {step} W={world}"
                )

    # The bound cannot detect an unbiased W-fold replication when even the
    # ceiling of the correct arm, multiplied by W, stays inside the bound.
    for step, tokens in STEP_TOKENS.items():
        for world in EP_WORLDS:
            ceiling = EXPECTED_ARM_A_HOP_CEILING[(step, world)]
            floor = EXPECTED_ARM_A_HOP_FLOOR[(step, world)]
            bound = EXPECTED_STEP_HOP_BOUND[step]
            certain_miss = world * ceiling <= bound
            certain_hit = world * floor > bound
            expected = EXPECTED_DETECTION[(step, world, "step-hop-bound")]
            if certain_miss and expected:
                raise AssertionError(
                    f"frozen detection claims a hit the bound cannot produce at "
                    f"{step} W={world}"
                )
            if certain_hit and not expected:
                raise AssertionError(
                    f"frozen detection claims a miss the bound cannot produce at "
                    f"{step} W={world}"
                )
    if any(
        not detected
        for (_, _, rule), detected in EXPECTED_DETECTION.items()
        if rule == "source-attribution"
    ):
        raise AssertionError("structural source replication must be detected everywhere")
    if EXPECTED_SCORED_INSTANCES != 2 + 2 + 4 + 1:
        raise AssertionError("scored instance arithmetic drifted")
    if len(EXPECTED_FATAL_GUARDS) != 2:
        raise AssertionError("fatal guard registry drifted")
    if set(ARMS) != {"owner-attributed", "source-replicated"}:
        raise AssertionError("arm registry drifted")
    if EXPERT_COUNT % max(EP_WORLDS) or EXPERT_COUNT % min(EP_WORLDS):
        raise AssertionError("contiguous owner blocks do not divide the expert count")


def check_only(args: argparse.Namespace) -> None:
    _check_frozen_registry()
    print(
        f"check-only out={args.out}; validated the frozen VLLM-24 rule registry, "
        f"{len(STEP_TOKENS) * len(EP_WORLDS) * len(ARMS)} cells and "
        f"{EXPECTED_SCORED_INSTANCES} scored instances, and produced no artifacts"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="routed_byte_conservation_v1")
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
