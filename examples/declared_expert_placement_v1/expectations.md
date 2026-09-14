# Declared expert placement expectations

Date: 2026-09-12

This is the expectations-only freeze for PLACE-3. It precedes the builder
extension, its tests, the study harness, every generated placement or Group
Operation Assembly Language (GOAL) artifact, and every result-producing run.
The declared layout is a what-if placement computed from the pinned
framework's layout rules; it is not an extracted record of a live run.

## Question

Can `declared_manifest` emit the expert-parallel (EP) group memberships, the
pipeline layer range and the per-MoE-layer expert ownership that the pinned
vLLM 0.27.1 creates for the same DP x PP x TP layout, so that a study reads
its EP group and its expert owners from the manifest instead of a hand-typed
rank list, without changing one byte of any manifest that does not request
experts?

## Frozen source and compatibility identity

The implementation starts from commit
`26704c1c18e3c5503cb717cdd31391b82d3983a0`. The JSON registry records the
pre-change SHA-256 identities of the declared builder, the manifest module,
the placement package entry, the focused test, the routed-expert projection
and the m5 run script.

The compatibility authority is the UTF-8 JSON emitted by
`PlacementManifest.save`, including its terminal line feed. Without the new
option, five reference manifests must keep their pre-change byte lengths and
digests:

| Record | Builder call | Bytes | SHA-256 |
|---|---|---:|---|
| worked example | `declared_manifest(tp=4, pp=2, dp=2)` | 10,832 | `2e46eadcaccd83de1778ea98c368585b452b83067489b9a2df064e25a14d857e` |
| m4 tensor group | `declared_manifest(tp=8)` | 5,698 | `3812aef241d93f2af8d86da0f6bbb606dae9c43c768ceb3efb8d649dc3bacfd6` |
| rail pipeline | `declared_pipeline_placement(8)` | 52,310 | `8d38cf4b6990bfd75fc90dcfe994180c3c1150adce297b6f71983fd8c9c877db` |
| m5 expert world | `declared_manifest(tp=1, dp=8)` | 5,698 | `0894fae1687217d88466b5692034a8d1863fb8c658afdfde2c317d1c412f60a5` |
| width tail | `declared_manifest(tp=64, nodes=8, gpus_per_node=8)` | 102,050 | `bed8d26c72feaf117f48b651ec18b4fb0caf39df2e47866a550f74252972aecd` |

This identity is a fatal, unscored guard. A manifest built with the option
absent must be byte identical to the pre-change output, and a manifest built
with the option present must differ only in the fields this freeze names.

## Pinned framework semantics

The rules below are read from the installed vLLM 0.27.1 package and are the
specification of the declared layout. Prefill context parallelism is one
throughout.

- Rank tensor. vLLM shapes `arange(world)` as `(external DP, DP, PP, PCP,
  TP)`, so `global_rank = (dp * PP + pp) * TP + tp`, which is the accepted
  declared formula.
- EP group. For every pipeline index `pp`, the EP group is the `DP * TP`
  ranks `(dp * PP + pp) * TP + tp` ordered data-parallel major and
  tensor-parallel minor, and `rank_in_group = dp * TP + tp`. vLLM creates this
  group for mixture-of-experts (MoE) models only.
- EP size. With expert parallelism enabled, the fused MoE layer flattens the
  tensor group across data parallel: `ep_size = DP * TP` and
  `ep_rank = dp * TP + tp`; the experts are then not tensor sharded. The
  layer refuses `num_experts % ep_size != 0`.
- Expert map. With `base = num_experts // ep_size`, the `linear` strategy
  gives rank `r` the contiguous experts `[r * base, (r + 1) * base)`, and the
  `round_robin` strategy gives it `{r + k * ep_size : 0 <= k < base}`. The
  remainder branch of the expert map is unreachable behind the layer's
  divisibility gate, so the builder refuses a non-divisible expert count
  rather than modeling a remainder.
- Pipeline partition. Without the environment override, `base = L // PP`
  layers per stage, and a remainder `L mod PP` adds one layer to the stages
  indexed `-2, -3, ...` in that order, so the last stage never gains a layer
  and the first stage gains one only when the remainder exceeds `PP - 2`.
  Stage `p` owns `[sum(partitions[:p]), sum(partitions[:p + 1]))`.

## Declared interface

```text
DeclaredExpertLayout(
    num_layers: int,                 # hidden layers, for the pipeline partition
    moe_layers: tuple[int, ...],     # ascending layer indices carrying a routed MoE block
    num_experts: int,                # global experts per MoE layer
    placement_strategy: str = "linear",   # or "round_robin"
    placement_epoch: int = 0,
)
declared_manifest(..., experts: DeclaredExpertLayout | None = None)
```

With the option present, every rank gains an `ep` membership placed after
`tp`, `pp` and `dp`; its `pipeline_layer_range` is the stage interval above;
its `local_expert_ids` maps each MoE layer inside that interval, in ascending
layer order, to the experts the strategy assigns to its EP rank; and its
`placement_epoch` is the declared epoch. With the option absent, nothing is
added.

Construction refuses, with a `ValueError` naming the field: `num_layers`
below `pp` (a zero-layer stage), a MoE layer outside `[0, num_layers)`,
duplicate or unsorted MoE layers, an empty MoE layer list, `num_experts`
below one or not divisible by `DP * TP`, an unknown strategy, and a negative
epoch. Boolean values are not integers.

## Frozen cells

Cell C1, the worked example: `tp=4, pp=2, dp=2` with 48 layers, all 48
carrying MoE blocks, 32 experts, both strategies. The EP size is 8. Rank 9
is `(dp=1, pp=0, tp=1)`: its EP group is `[0, 1, 2, 3, 8, 9, 10, 11]` with
`rank_in_group` 5, its layer range is `[0, 24)`, its `local_expert_ids`
keys are exactly the layers 0 through 23, and each maps to `[20, 21, 22, 23]`
under `linear` and to `[5, 13, 21, 29]` under `round_robin`. The round-robin
row reproduces the hand-written worked example already in
`tests/test_placement.py`. Rank 15 is `(dp=1, pp=1, tp=3)`: EP group
`[4, 5, 6, 7, 12, 13, 14, 15]`, `rank_in_group` 7, layer range `[24, 48)`,
`[28, 29, 30, 31]` under `linear` and `[7, 15, 23, 31]` under `round_robin`.
For every stage and every MoE layer, the eight owners partition the 32
experts exactly.

Cell C2, the m5 identity: `tp=1, pp=1, dp=W` for `W` in 2, 4, 8, with the
m5 granite geometry (24 layers, all MoE, 32 experts, `linear`). The EP group
of rank 0 is `[0, ..., W-1]` and rank `r` owns `[r * 32 / W, (r + 1) * 32 / W)`.
The live-reachable relation: `HtsimStepSink` on `rnic-nn-fluid` at
400 Gbit/s, with `ep_ranks = manifest.group_ranks(0, "ep")` and the m5 step
records, reproduces the six frozen m5 check-B makespans exactly, and its GOAL
text is byte identical to the run with `ep_ranks = tuple(range(W))`:

| Shape | W | Makespan (ps) |
|---|---:|---:|
| decode8x2048 | 2 | 563,362,560 |
| decode8x2048 | 4 | 486,963,888 |
| decode8x2048 | 8 | 448,764,528 |
| prefill2048 | 2 | 17,592,951,360 |
| prefill2048 | 4 | 25,646,015,088 |
| prefill2048 | 8 | 29,672,546,928 |

The expected direction, as m5 recorded, is that the decode makespan falls
with `W` (the per-rank expert compute shrinks faster than the all-to-all
grows) while the prefill makespan rises with `W`. In the same cell,
`ExpertPlacementSnapshot.from_manifest(manifest, ep_ranks)` carries exactly
`24 * 32 = 768` owner entries, every `(layer, expert)` pair exactly once,
with owner `expert // (32 / W)`.

Cell C3, a DeepSeek-class pipeline: `tp=8, pp=4, dp=2` on eight nodes of
eight (64 ranks), 61 layers with MoE blocks on layers 3 through 60 (58
layers), 256 experts, `linear`. The EP size is 16 and there are four EP
groups, one per stage, each `[(dp * 4 + pp) * 8 + tp]` for `dp` in 0, 1 and
`tp` in 0 through 7, with `rank_in_group = dp * 8 + tp`. The stage ranges are
`[0, 15)`, `[15, 30)`, `[30, 46)` and `[46, 61)`, since the single remainder
layer goes to the stage indexed `-2`, which is stage 2. The MoE layers per
stage are 12, 15, 16 and 15, and
rank `r` of a group owns `[16 r, 16 r + 16)` in each of its stage's MoE
layers. Summed over all 64 ranks the ownership entries number exactly
`58 * 256 = 14,848`, and every `(layer, expert)` pair is owned exactly once.

Cell C4, partition arithmetic without experts in the loop: the stage
intervals for `(L, PP)` equal to `(48, 2)`, `(61, 4)`, `(61, 8)`, `(30, 4)`
and `(5, 2)` are `[0, 24), [24, 48)`; `[0, 15), [15, 30), [30, 46), [46, 61)`;
`[0, 7), [7, 14), [14, 22), [22, 30), [30, 38), [38, 46), [46, 54), [54, 61)`;
`[0, 7), [7, 15), [15, 23), [23, 30)`; and `[0, 3), [3, 5)`.

Cell C5, refusals: 30 experts at EP size 8, MoE layer 48 with 48 layers,
one layer with `pp=2`, an unknown strategy, duplicate MoE layers, an empty
MoE layer list, epoch minus one, and a boolean expert count are each refused
before any rank is built.

Cell C6, wire identity: the C1 and C3 manifests round-trip through
`PlacementManifest.save` and `PlacementManifest.load` to equal objects, the
`ep` key follows `dp` in every rank's `groups`, and `local_expert_ids` keys
serialize in ascending layer order.

## Physical sanity before observation

The only timed quantity is C2, and its oracle is the accepted m5 record, so
the check is exact rather than bounded. As a floor, one decode step at `W=8`
cannot complete faster than the 24 per-layer compute gates the m5 record
froze, `24 * 10,111 ns = 242,664,000 ps`, and the measured 448,764,528 ps sits
above it. A manifest-driven run that lands on any other value than the m5
literal is a defect in the builder, the sink or the harness, never a
calibration finding.

## Evidence accounting and closure

The six C2 makespan rows are exact-oracle rows and form the only scored
denominator. C1, C3, C4 and C6 are structural exact guards, C5 is a rejection
control family, and the five compatibility digests are fatal, unscored
by-construction identities. Counts from these classes are never added. A
violated fatal guard voids the run and PLACE-3 stays open.

PLACE-3 closes only if every structural cell is exact, all six makespans
match the m5 literals, the GOAL text is byte identical between the manifest
and hand-typed EP lists, the snapshot conservation holds, and the five
digests remain exact. Closure makes the placement doc's "EP layout from the
manifest" statement literal. The implementation registers a new task for the
one deliberately unmodeled variant: a MoE deployment with expert parallelism
disabled, where vLLM still creates the EP group but tensor-shards every
expert across the tensor group and each rank owns every expert. That variant
is representable today by omitting the option, which is exactly the accepted
all-experts-local geometry, so it is a completeness gap, not a defect.
PLACE-1, PLACE-2 and PLACE-6 are untouched.
