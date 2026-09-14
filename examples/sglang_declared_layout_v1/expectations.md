# SGLang declared layout expectations

Date: 2026-09-14

This is the expectations-only freeze for PLACE-13. It precedes the second
declared builder, its tests, the study harness, every generated placement or
Group Operation Assembly Language (GOAL) artifact, and every result-producing
run. The declared layout is a what-if placement computed from the pinned
framework's layout rules; it is not an extracted record of a live run.

## Question

Can a second declared builder emit the rank layout, the pipeline layer range,
the expert-parallel (EP) group and the two mixture-of-experts (MoE) side
groups that the pinned SGLang commit creates for one `tp x pp` world with
`--ep-size` and `--moe-dp-size`, so that a study reads its EP group and its
expert owners from an SGLang-declared manifest exactly as it reads them from
a vLLM-declared one, without changing one byte of any manifest the vLLM
builder emits today?

## Why a second builder

The accepted declared builder follows vLLM 0.27.1. SGLang's layout differs in
four places that no option of that builder can express. Its world is
`tp x pp` with no data-parallel term in the rank space: attention data
parallelism subdivides the tensor group and router-style replicas are separate
worlds. Its EP group is carved out of one tensor group per pipeline stage,
with the MoE tensor index innermost, the EP index next and the MoE
data-parallel index outermost, instead of spanning the DP x TP ranks of a
stage. Its expert map is the contiguous block only, with a divisibility
assertion and no round-robin placement. And its pipeline partition hands the
remainder layers to the last stages, where vLLM hands them to the stages
indexed `-2, -3, ...`. A consumer that read a vLLM-declared manifest for an
SGLang deployment would therefore place the wrong layers on two of four
stages of a 61-layer model, and the wrong ranks in every EP group whenever
`ep_size < tp`.

## Frozen source and compatibility identity

The implementation starts from commit
`2a4abea67b9b1ce793781b76f330574059ffe817`. The JSON registry records the
pre-change SHA-256 identities of the declared builder, the placement package
entry, the manifest module, the two placement test files, the PLACE-3 harness
and the placement registry.

The compatibility authority is the UTF-8 JSON emitted by
`PlacementManifest.save`, including its terminal line feed. The five reference
manifests of the PLACE-3 freeze must keep their byte lengths and digests, and
the PLACE-3 study's `--check` must reproduce its tracked results:

| Record | Builder call | Bytes | SHA-256 |
|---|---|---:|---|
| worked example | `declared_manifest(tp=4, pp=2, dp=2)` | 10,832 | `2e46eadcaccd83de1778ea98c368585b452b83067489b9a2df064e25a14d857e` |
| m4 tensor group | `declared_manifest(tp=8)` | 5,698 | `3812aef241d93f2af8d86da0f6bbb606dae9c43c768ceb3efb8d649dc3bacfd6` |
| rail pipeline | `declared_pipeline_placement(8)` | 52,310 | `8d38cf4b6990bfd75fc90dcfe994180c3c1150adce297b6f71983fd8c9c877db` |
| m5 expert world | `declared_manifest(tp=1, dp=8)` | 5,698 | `0894fae1687217d88466b5692034a8d1863fb8c658afdfde2c317d1c412f60a5` |
| width tail | `declared_manifest(tp=64, nodes=8, gpus_per_node=8)` | 102,050 | `bed8d26c72feaf117f48b651ec18b4fb0caf39df2e47866a550f74252972aecd` |

This identity is a fatal, unscored guard. The vLLM builder is not touched by
this slice; a manifest it builds must be byte identical to the pre-change
output, and the new builder adds a second entry point rather than an option.

## Pinned framework semantics

The rules below are read from the installed SGLang package at the pinned
commit `bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3` (the installed distribution
reports `0.5.6.post3.dev9406+gbfeae4e79`; the adapter registry cites the same
commit under a different tag distance, and the freeze binds to the commit).
File names are given relative to the installed `sglang/srt` package.
Attention data parallelism, attention and decode context parallelism, and
the elastic EP joiner offset are all one or zero throughout.

- Rank formula. `distributed/bootstrap.py` computes
  `world_size = tp_size * pp_size` and `rank = tp_size * pp_rank + tp_rank`,
  and `distributed/parallel_state.py` (`initialize_model_parallel`) refuses
  any other world size. Tensor groups are contiguous blocks of `tp` ranks
  and pipeline groups stride by `tp`. There is no data-parallel axis in the
  rank space: `layers/dp_attention.py` places attention data parallelism
  inside the tensor group, and `managers/data_parallel_controller.py`
  launches router-style replicas as separate worlds with their own GPU
  offset. The declared manifest describes one replica and records a
  singleton `dp` membership for every rank.
- MoE sizes. `initialize_model_parallel` sets `moe_ep_size` from `--ep-size`,
  `moe_dp_size` from `--moe-dp-size`, and
  `moe_tp_size = tp // moe_ep_size // moe_dp_size`. `server_args.py` asserts,
  only when `moe_dp_size > 1`, that `tp % moe_dp_size == 0`,
  `ep_size * moe_dp_size <= tp`, `pp == 1`, and, when also `ep_size > 1`,
  `ep_size * moe_dp_size == tp`. The general divisibility of `tp` by
  `ep_size * moe_dp_size` is asserted by the framework only for quantized
  models (`model_executor/model_runner_components/moe_ep_setup.py`); the
  declared builder refuses it for every layout, because the group loops
  below cover the tensor group only when it holds.
- EP group. For each tensor group with base `b = pp_rank * tp`, each
  `moe_dp_idx` and each `moe_tp_idx`, the EP group is
  `range(s, s + ep * moe_tp, moe_tp)` with
  `s = b + moe_dp_idx * ep * moe_tp + moe_tp_idx`. A rank's index inside it
  is `moe_ep_rank = tp_rank % (tp // moe_dp) // moe_tp`, the formula the
  scheduler launcher prints. When `ep_size == tp` the group is the tensor
  group itself and `moe_ep_rank == tp_rank`.
- MoE tensor group. For each combined `ep_dp_idx` in
  `range(ep * moe_dp)`, the contiguous block
  `range(b + ep_dp_idx * moe_tp, b + (ep_dp_idx + 1) * moe_tp)`; the rank's
  index is `tp_rank % moe_tp`.
- MoE data-parallel group. For each `idx` in `range(moe_tp * ep)`, the
  strided set `range(b + idx, b + tp + idx, moe_tp * ep)`; the rank's index
  is `moe_dp_rank = tp_rank // (tp // moe_dp)`.
- Expert map. With the default `--init-expert-location trivial`,
  `eplb/expert_location.py` maps physical expert `i` to logical expert `i`
  and asserts `num_experts % ep_size == 0`;
  `layers/moe/fused_moe_triton/layer.py` asserts the same and gives EP rank
  `r` the contiguous experts `[r * L, (r + 1) * L)` with
  `L = num_experts // ep_size`. There is no round-robin placement; the
  declared builder refuses `round_robin` and a non-divisible expert count.
  Under `moe_tp > 1` every rank of one MoE tensor group owns the same expert
  ids and holds one shard of each; the shard is not represented, which is the
  gap PLACE-7 already records for vLLM.
- Pipeline partition. `distributed/utils.py` (`get_pp_indices`) gives every
  stage `base = L // PP` layers and one extra layer to each of the last
  `L mod PP` stages: stage `p` owns `base + 1` layers when
  `p >= PP - remainder`, starting at `p * (base + 1) - (PP - remainder)`,
  and `base` layers starting at `p * base` otherwise. The environment
  override `SGLANG_PP_LAYER_PARTITION` is deliberately not modeled.
- GPU and node placement. `managers/data_parallel_controller.py` hosts
  `pp // nnodes` pipeline stages per node when `nnodes <= pp`, otherwise
  one stage per `nnodes // pp` nodes with `tp // (nnodes // pp)` tensor
  ranks each, and numbers GPUs
  `(pp_rank mod pp_per_node) * tp_per_node + tp_rank mod tp_per_node`. In
  both cases ranks fill nodes in global-rank order, `world // nnodes` per
  node, so the declared builder uses that fill and refuses a node count that
  neither divides nor is divided by `pp`, or that leaves `tp` indivisible by
  `nnodes // pp`.

## Declared interface

```text
declared_sglang_manifest(
    *,
    tp: int = 1,
    pp: int = 1,
    ep_size: int = 1,                # SGLang --ep-size
    moe_dp_size: int = 1,            # SGLang --moe-dp-size
    nodes: int | None = None,        # SGLang --nnodes; default: the fewest that fit
    gpus_per_node: int = 8,
    hostname_pattern: str = "node-{}",
    framework_version: str | None = None,
    experts: DeclaredExpertLayout | None = None,
) -> PlacementManifest             # source="declared", framework="sglang"
declared_sglang_pipeline_partition(num_layers: int, pp: int) -> tuple[tuple[int, int], ...]
```

Every rank carries `tp`, `pp` and a singleton `dp` membership, in that order,
its hostname `hostname_pattern.format(global_rank // ranks_per_node)` and
`local_rank = global_rank % ranks_per_node` with
`ranks_per_node = tp * pp // nodes`. With `experts` present every rank gains,
after `dp` and in this order, the `ep`, `moe_tp` and `moe_dp` memberships
above; its `pipeline_layer_range` is the stage interval of the SGLang
partition; its `local_expert_ids` maps each MoE layer inside that interval,
in ascending layer order, to the contiguous block of its EP rank; and its
`placement_epoch` is the declared epoch. With `experts` absent, the manifest
carries the three base memberships and nothing else. `DeclaredExpertLayout`
is reused unchanged; the builder reuses `declared_local_expert_ids` with the
`linear` strategy, which equals SGLang's block under divisibility.

Construction refuses, with a `ValueError` naming the field: `tp`, `pp`,
`ep_size`, `moe_dp_size` or `gpus_per_node` below one; `tp` not divisible by
`ep_size * moe_dp_size`; `moe_dp_size > 1` with `pp > 1`; `moe_dp_size > 1`
and `ep_size > 1` with `ep_size * moe_dp_size != tp`; a node count that does
not divide the world, exceeds `gpus_per_node` ranks per node, neither divides
nor is divided by `pp`, or leaves `tp` indivisible by `nodes // pp`; the
`round_robin` strategy; `num_experts` not divisible by `ep_size`; and
`num_layers` below `pp`. The field-local `DeclaredExpertLayout` refusals of
PLACE-3 apply unchanged. Boolean values are not integers.

## Frozen cells

Cell S1, the coincidence: `tp=8, pp=1, ep_size=8` with 48 layers, all
carrying MoE blocks, 32 experts. The EP group of every rank is
`[0, 1, 2, 3, 4, 5, 6, 7]` with `rank_in_group = tp_rank`, rank 5 owns
`[20, 21, 22, 23]` in every layer 0 through 47, and its `moe_tp` and `moe_dp`
memberships are the singletons `[5]`. For every rank, the `global_rank`,
`hostname`, `local_rank`, `tp`, `pp`, `dp` and `ep` memberships,
`pipeline_layer_range`, `local_expert_ids` and `placement_epoch` equal the
same fields of `declared_manifest(tp=8, experts=...)` with the same layout;
the SGLang manifest differs only in `framework` and in the two extra
memberships. This is the one geometry where both frameworks agree, and it is
frozen so that the divergence cells below cannot be blamed on the builder.

Cell S2, expert sharding: `tp=8, pp=1, ep_size=2` with the S1 layout, so
`moe_tp = 4`. Rank 5 has EP group `[1, 5]` with `rank_in_group` 1, `moe_tp`
group `[4, 5, 6, 7]` with index 1, `moe_dp` group `[5]`, and owns
`[16, ..., 31]` in every layer. Rank 0 has EP group `[0, 4]` with index 0 and
owns `[0, ..., 15]`. The four EP groups are `[0, 4]`, `[1, 5]`, `[2, 6]` and
`[3, 7]`; in each, the two owners partition the 32 experts exactly, and the
four ranks of each `moe_tp` group own identical expert sets.

Cell S3, MoE data parallelism: `tp=8, pp=1, ep_size=2, moe_dp_size=4` with
the S1 layout, so `moe_tp = 1`. Rank 5 has EP group `[4, 5]` with index 1,
`moe_tp` group `[5]`, `moe_dp` group `[1, 3, 5, 7]` with index 2, and owns
`[16, ..., 31]`. Rank 6 has EP group `[6, 7]` with index 0, `moe_dp` group
`[0, 2, 4, 6]` with index 3, and owns `[0, ..., 15]`. The EP groups are
`[0, 1]`, `[2, 3]`, `[4, 5]` and `[6, 7]`.

Cell S4, a pipeline: `tp=4, pp=4, ep_size=4` on 16 ranks, 61 layers with MoE
blocks on layers 3 through 60, 64 experts. The stage ranges are `[0, 15)`,
`[15, 30)`, `[30, 45)` and `[45, 61)`, the last stage taking the remainder
layer; the MoE layers per stage are 12, 15, 15 and 16. Rank 13 is
`(pp=3, tp=1)`: EP group `[12, 13, 14, 15]` with index 1, layer range
`[45, 61)`, and `[16, ..., 31]` in each of layers 45 through 60. Summed over
all 16 ranks the ownership entries number exactly `58 * 64 = 3,712`, and
every `(layer, expert)` pair is owned exactly once.

Cell S5, partition arithmetic with the framework as oracle: the stage
intervals for `(L, PP)` equal to `(48, 2)`, `(61, 4)`, `(61, 8)`, `(30, 4)`,
`(5, 2)` and `(24, 3)` are `[0, 24), [24, 48)`;
`[0, 15), [15, 30), [30, 45), [45, 61)`;
`[0, 7), [7, 14), [14, 21), [21, 29), [29, 37), [37, 45), [45, 53), [53, 61)`;
`[0, 7), [7, 14), [14, 22), [22, 30)`; `[0, 2), [2, 5)`; and
`[0, 8), [8, 16), [16, 24)`. The harness also runs `get_pp_indices` of the
installed SGLang package in the interpreter named by `SIMLLM_SGLANG_ENV`
(or `--sglang-python`) for every row and requires equality, recording that
interpreter's reported SGLang version, which must end in `gbfeae4e79`. The
oracle arm is required for closure; a run without it is incomplete, not
failed.

Cell S6, refusals: `round_robin` under the SGLang builder; 30 experts at
`ep_size=8`; `tp=8, ep_size=3`; `tp=8, ep_size=16`; `tp=8, ep_size=2,
moe_dp_size=2`; `tp=8, moe_dp_size=2, pp=2`; `ep_size=0`; `moe_dp_size=0`;
one layer with `pp=2`; `tp=4, pp=3, nodes=2`; `tp=4, pp=1, nodes=3`; and
`tp=2, pp=1, nodes=1, gpus_per_node=1` are each refused before any rank is
built.

Cell S7, wire identity: the S2 and S4 manifests round-trip through
`PlacementManifest.save` and `PlacementManifest.load` to equal objects; the
`groups` keys of every rank are exactly `tp`, `pp`, `dp`, `ep`, `moe_tp`,
`moe_dp` in that order; `framework` is `sglang`; `framework_version` is
`None` when omitted and the pinned commit string when passed;
`local_expert_ids` keys serialize in ascending layer order.

Cell S8, node placement: `tp=8, pp=2, nodes=2` places ranks 0 through 7 on
`node-0` and 8 through 15 on `node-1` with `local_rank = rank mod 8`;
`tp=16, pp=1, nodes=2` places the same ranks the same way; `tp=4, pp=1,
nodes=2` places ranks 0 and 1 on `node-0` and 2 and 3 on `node-1` with local
ranks 0 and 1 on each. Without `nodes`, `tp=8, pp=2` takes two nodes of
eight.

Cell S9, divergence from the vLLM builder: for `tp=4, pp=4` with the S4
layout, the vLLM builder's stage ranges are `[0, 15)`, `[15, 30)`,
`[30, 46)`, `[46, 61)` and the SGLang builder's are the S4 rows, so the two
differ at stages 2 and 3 and agree at 0 and 1; and for `tp=8, pp=1` with
`ep_size=2` the SGLang EP group of rank 5 is `[1, 5]` where the vLLM builder
gives `[0, ..., 7]`. Both rows are frozen literals; the cell is a structural
guard that the second builder is not the first one relabeled.

Cell S10, the m5 identity: `tp=W, pp=1, ep_size=W` for `W` in 2, 4, 8, with
the m5 granite geometry (24 layers, all MoE, 32 experts). The EP group of
rank 0 is `[0, ..., W-1]` and rank `r` owns `[r * 32 / W, (r + 1) * 32 / W)`.
The live-reachable relation: `HtsimStepSink` on `rnic-nn-fluid` at
400 Gbit/s, with `ep_ranks = manifest.group_ranks(0, "ep")` from the SGLang
manifest and the m5 step records, reproduces the six frozen m5 check-B
makespans exactly, and its GOAL text is byte identical to the run with
`ep_ranks = tuple(range(W))`:

| Shape | W | Makespan (ps) |
|---|---:|---:|
| decode8x2048 | 2 | 563,362,560 |
| decode8x2048 | 4 | 486,963,888 |
| decode8x2048 | 8 | 448,764,528 |
| prefill2048 | 2 | 17,592,951,360 |
| prefill2048 | 4 | 25,646,015,088 |
| prefill2048 | 8 | 29,672,546,928 |

In the same cell, `ExpertPlacementSnapshot.from_manifest(manifest, ep_ranks)`
carries exactly `24 * 32 = 768` owner entries, every `(layer, expert)` pair
exactly once, with owner `expert // (32 / W)`.

## Physical sanity before observation

The only timed quantity is S10, and its oracle is the accepted m5 record, so
the check is exact rather than bounded. As a floor, one decode step at `W=8`
cannot complete faster than the 24 per-layer compute gates the m5 record
froze, `24 * 10,111 ns = 242,664,000 ps`, and the frozen 448,764,528 ps sits
above it. A manifest-driven run that lands on any other value than the m5
literal is a defect in the builder, the sink or the harness, never a
calibration finding.

## Evidence accounting and closure

The six S10 makespan rows are exact-oracle rows and form the only scored
denominator. S1, S2, S3, S4, S5, S7, S8 and S9 are structural exact guards,
S6 is a rejection control family, and the five compatibility digests
together with the PLACE-3 study check are fatal, unscored by-construction
identities. Counts from these classes are never added. A violated fatal
guard voids the run and PLACE-13 stays open.

PLACE-13 closes only if every structural cell is exact, the S5 oracle arm
ran and agreed, all six makespans match the m5 literals, the GOAL text is
byte identical between the manifest and hand-typed EP lists, the snapshot
conservation holds, and the five digests and the PLACE-3 check remain
exact. Closure makes the placement doc's statement that a declared manifest
can follow either pinned framework literal, and adds to the SGL-18 entry the
cross-link that the declared what-if counterpart of its extracted ownership
exists. The implementation registers one task for the SGLang variants this
slice leaves out: attention data and context parallelism inside the tensor
group, decode context parallelism, the partition override, fused shared and
redundant physical experts, EPLB maps, and the elastic EP joiner offset.
PLACE-7, PLACE-8 and PLACE-11 are untouched.
