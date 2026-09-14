# Declared expert placement amendment, 2026-09-13

This expectations amendment is post-specified to a review finding on the
implemented slice and is frozen before the corrected implementation and its
rerun. The original freeze of 2026-09-12 stays in place unchanged; this file
states what it got wrong, what replaces it, and what is not affected.

## Refuted assumption

The freeze stated that the pinned vLLM 0.27.1 fused MoE layer refuses an
expert count that the expert-parallel size does not divide, and the builder
therefore refused such layouts (cell C5, first control). Reading the pinned
source again shows the refusal exists only when expert-parallel load
balancing (EPLB) is enabled (`model_executor/layers/fused_moe/layer.py`, the
`enable_eplb` branch). Without EPLB, `determine_expert_map` distributes the
remainder: with `base = num_experts // ep_size` and
`remainder = num_experts % ep_size`, EP rank `r` owns `base + 1` experts when
`r < remainder` and `base` otherwise; `linear` starts rank `r` at
`r * base + min(r, remainder)`, and `round_robin` gives rank `r` every
`ep_size`-th expert from `r` upward, which yields the same per-rank counts.
The builder must reproduce that rule rather than refuse it. A declared layout
does not model EPLB, so the EPLB-only refusal is not a builder rule.

## Replacement rules

- Ownership for any positive `num_experts` and any `ep_size = DP * TP`
  follows the remainder rule above. The divisibility refusal is withdrawn
  from the builder, from `declared_local_expert_ids` and from cell C5.
- The `round_robin` strategy is the caller's explicit declaration. The
  framework's own fallback to `linear` (when the model has at most one
  expert group, or has redundant experts, or EPLB is on, or the all-to-all
  backend lacks round-robin routing tables) is not modeled by the declared
  builder; the module doc and the placement doc say so in plain words, and an
  extracted manifest records the map that really ran.
- The expert-parallel-disabled variant registered as PLACE-7 shards every
  expert across the flattened DP x TP group, not across the tensor group
  alone; its registry text is corrected.

## New frozen cell

Cell C8, remainder layout: `tp=1, pp=1, dp=8`, 24 MoE layers, 30 experts.
Per-rank counts are `4, 4, 4, 4, 4, 4, 3, 3`. Under `linear` the owners are
rank 0 `[0, 1, 2, 3]`, rank 1 `[4, 5, 6, 7]`, rank 2 `[8, 9, 10, 11]`, rank 3
`[12, 13, 14, 15]`, rank 4 `[16, 17, 18, 19]`, rank 5 `[20, 21, 22, 23]`,
rank 6 `[24, 25, 26]`, rank 7 `[27, 28, 29]`. Under `round_robin` rank 0 owns
`[0, 8, 16, 24]`, rank 5 owns `[5, 13, 21, 29]`, rank 6 owns `[6, 14, 22]`
and rank 7 owns `[7, 15, 23]`. In both strategies every expert of every layer
is owned exactly once and the snapshot built from the manifest carries
`24 * 30 = 720` owner entries. These values were computed from the pinned
`determine_expert_map` before the corrected builder existed.

## Assertions the review found missing

The refusals for `num_experts` below one and a negative MoE layer index gain
tests. The claim that a manifest built with the option present differs from
its expert-free twin only in the `ep` group, the layer range, the expert
ownership and the epoch is asserted by a field-level comparison. The results
text says a refusal is a `ValueError` raised by construction, not "before any
rank is built", unless the test observes that ordering.

## What does not change

Every other frozen cell (C1, C2, C3, C4, C6 and the five compatibility
digests) keeps its literals; the m5 identity relation is unaffected because
its 32 experts divide every world. The step-sink consumers that require
uniform per-rank expert geometry (`local_num_experts * len(ep_ranks) ==
num_experts`) are unchanged: a remainder layout is representable in the
manifest today but cannot yet drive those consumers, and that consumer gap is
registered as a new task in the same change. PLACE-3 closes only when the
corrected builder passes the original cells plus C8.
