# SGLang declared layout result

## Outcome

What ran: `examples/sglang_declared_layout_v1`, the frozen PLACE-13
qualification, built declared placements for the pinned SGLang commit
`bfeae4e7` from implementation commit `2b3787ff`, checked them against that
package's layout rules with its own `get_pp_indices` as an executable oracle,
and drove `HtsimStepSink` on `rnic-nn-fluid` with the expert-parallel (EP)
group read from an SGLang-declared manifest. The expectations-only commit is
`61fc8878` and the harness commit is `536ee6a4`.

What came out: the result is `PASS` with no finding. The deciding number is
6 of 6 frozen m5 check-B makespans reproduced exactly with the EP group taken
from the SGLang manifest, each with Group Operation Assembly Language (GOAL)
text byte identical to the run with the hand-typed rank list. All eight
structural cells were exact, the S5 oracle arm ran in an interpreter reporting
SGLang `0.5.6.post3.dev9406+gbfeae4e79` and agreed on all six partition rows,
all twelve rejection controls refused, the five vLLM reference manifests kept
their bytes and digests, and the PLACE-3 study reproduced its tracked results.

What it changes for the project: PLACE-13 closes. The placement doc's
statement that a declared manifest can follow either pinned framework becomes
literal: `declared_sglang_manifest` emits `framework="sglang"` with the `ep`,
`moe_tp` and `moe_dp` memberships, the SGLang pipeline partition and
contiguous block ownership, so a consumer reads
`manifest.group_ranks(rank, "ep")` and builds
`ExpertPlacementSnapshot.from_manifest` from an SGLang-declared manifest
exactly as it does from a vLLM-declared one. SGL-18 gains the cross-link that
the declared what-if counterpart of its extracted ownership exists. The SGLang
variants this slice leaves out are registered as PLACE-14.

What it does not change: no packet backend other than the fluid closed form
ran, no time to first token (TTFT) or time per output token (TPOT) moved, no
extracted manifest changed, the m5 study and its record are untouched, and the
vLLM builder is byte identical in all five reference manifests. Nothing here
calibrates a device or a fabric, and no live SGLang deployment was observed:
the layout rules are read from the installed package, and the only thing the
framework itself executed is the partition oracle. PLACE-7, PLACE-8 and
PLACE-11 stay open exactly as they were.

## Pinned semantics that the builder reproduces

The freeze read eight rules from the installed SGLang package at the pinned
commit and the implementation reproduces them, with the one gating caveat
recorded below: the rank formula
`tp * pp_rank + tp_rank` over a world of `tp * pp` with no data-parallel term
in the rank space; `moe_tp_size = tp // ep_size // moe_dp_size` with the
launcher's own assertions; the EP group carved out of one tensor group per
pipeline stage as `range(s, s + ep * moe_tp, moe_tp)` with the MoE tensor
index innermost and the MoE data-parallel index outermost; the contiguous MoE
tensor group; the strided MoE data-parallel group; the trivial expert map
giving EP rank `r` the block `[r * L, (r + 1) * L)` with
`L = num_experts // ep_size` and no round-robin alternative; the pipeline
partition that hands the layer remainder to the *last* stages; and the node
fill in global-rank order with `world // nodes` ranks per node. Attention data
parallelism, attention and decode context parallelism, and the elastic EP
joiner offset are one or zero throughout, and the `SGLANG_PP_LAYER_PARTITION`
override is deliberately not modeled, so the harness removes that variable
from the oracle interpreter's environment before asking the framework for its
rows.

The caveat is when the three MoE memberships appear.
`initialize_model_parallel` builds the expert-parallel group and both MoE side
groups for every world it initializes, MoE model or not, while this builder
emits `ep`, `moe_tp` and `moe_dp` only when an expert layout is declared. The
memberships the builder does emit follow the framework's formulas exactly;
what is gated is whether they are emitted at all, and that gate is a
declared-manifest convention (it keeps the expert-free manifest to the
memberships a consumer can act on) rather than a claim about the framework's
process groups.

## Evidence

| Evidence class | Result |
|---|---|
| Scored exact-oracle rows (S10 makespans) | 6 of 6 exact, GOAL text byte identical in every row |
| Structural exact guards (S1, S2, S3, S4, S5, S7, S8, S9) | 8 of 8 cells exact |
| Executable framework oracle (S5) | ran, agreed on 6 of 6 partition rows, interpreter version suffix `gbfeae4e79` |
| Rejection controls (S6) | 12 of 12 refused, each a `ValueError`, and each with no rank built: every control carries a hostname pattern whose `format` raises, and the same pattern fires on an accepted layout |
| Fatal compatibility digests | 5 of 5 vLLM reference manifests byte identical to the pre-change record |
| Fatal PLACE-3 study check | reproduced its tracked results |

Counts in different evidence classes are not added. The tracked
[results](results.json) hold every cell; bulk manifests, GOAL programs and
completion tables stay under the configured data root.

S1, the coincidence (`tp=8, pp=1, ep_size=8`, 48 MoE layers, 32 experts): every
rank's EP group is `[0, ..., 7]` with `rank_in_group` equal to its tensor rank,
rank 5 owns `[20, 21, 22, 23]` in every layer 0 through 47 and holds the
singleton `moe_tp` and `moe_dp` groups `[5]`. Field by field over all eight
ranks, the global rank, hostname, local rank, `tp`, `pp`, `dp` and `ep`
memberships, layer range, expert ownership and placement epoch equal those of
`declared_manifest(tp=8, experts=...)`; the two manifests differ only in
`framework` and in the two extra memberships. This is the one geometry where
both frameworks agree, which is what makes the divergence cells readable.

S2, expert sharding (`ep_size=2`, so `moe_tp = 4`): the four EP groups are
`[0, 4]`, `[1, 5]`, `[2, 6]` and `[3, 7]`. Rank 5 sits in `[1, 5]` at index 1
with `moe_tp` group `[4, 5, 6, 7]` at index 1 and the singleton `moe_dp` group
`[5]`, and owns experts 16 through 31; rank 0 sits in `[0, 4]` at index 0 and
owns 0 through 15. In each EP group the two owners partition the 32 experts
exactly, and the four ranks of each MoE tensor group own identical sets.

S3, MoE data parallelism (`ep_size=2, moe_dp_size=4`, so `moe_tp = 1`): the EP
groups are `[0, 1]`, `[2, 3]`, `[4, 5]` and `[6, 7]`. Rank 5 sits in `[4, 5]`
at index 1 with `moe_dp` group `[1, 3, 5, 7]` at index 2 and owns 16 through
31; rank 6 sits in `[6, 7]` at index 0 with `moe_dp` group `[0, 2, 4, 6]` at
index 3 and owns 0 through 15.

S4, the pipeline (`tp=4, pp=4, ep_size=4`, 61 layers, MoE on layers 3 through
60, 64 experts): stage ranges `[0, 15)`, `[15, 30)`, `[30, 45)` and `[45, 61)`
with the last stage taking the remainder layer, MoE layers per stage 12, 15,
15 and 16, and rank 13 at `(pp=3, tp=1)` in EP group `[12, 13, 14, 15]` at
index 1 owning 16 through 31 in each of layers 45 through 60. Summed over all
16 ranks the ownership entries number exactly 3,712, with every
`(layer, expert)` pair owned exactly once.

S5, partition arithmetic with the framework as oracle:

| Layers / PP | Stage intervals |
|---|---|
| 48 / 2 | `[0, 24)`, `[24, 48)` |
| 61 / 4 | `[0, 15)`, `[15, 30)`, `[30, 45)`, `[45, 61)` |
| 61 / 8 | `[0, 7)`, `[7, 14)`, `[14, 21)`, `[21, 29)`, `[29, 37)`, `[37, 45)`, `[45, 53)`, `[53, 61)` |
| 30 / 4 | `[0, 7)`, `[7, 14)`, `[14, 22)`, `[22, 30)` |
| 5 / 2 | `[0, 2)`, `[2, 5)` |
| 24 / 3 | `[0, 8)`, `[8, 16)`, `[16, 24)` |

The installed package's own `get_pp_indices`, called in the interpreter named
by `SIMLLM_SGLANG_ENV`, returned the same six rows.

S10, the m5 identity (`tp=W, pp=1, ep_size=W`, granite geometry):

| Shape | W | Makespan (ps) | Matches m5 | GOAL identical |
|---|---:|---:|---|---|
| decode8x2048 | 2 | 563,362,560 | yes | yes |
| decode8x2048 | 4 | 486,963,888 | yes | yes |
| decode8x2048 | 8 | 448,764,528 | yes | yes |
| prefill2048 | 2 | 17,592,951,360 | yes | yes |
| prefill2048 | 4 | 25,646,015,088 | yes | yes |
| prefill2048 | 8 | 29,672,546,928 | yes | yes |

In every world the expert placement snapshot built from the SGLang manifest
carries exactly 768 owner entries with owner `expert // (32 / W)`.

S6 refuses all twelve malformed layouts, S7 round-trips the S2 and S4
manifests through save and load with the six group keys in the frozen order
and ascending expert-layer keys, S8 fills nodes in global-rank order with the
default node count equal to the explicit one, and S9 shows the two builders
disagreeing where the freeze said they must: for 61 layers on four stages the
vLLM partition is `[0, 15)`, `[15, 30)`, `[30, 46)`, `[46, 61)` against the
SGLang rows above, differing at stages 2 and 3, and at `tp=8, ep_size=2` the
SGLang EP group of rank 5 is `[1, 5]` where the vLLM builder gives
`[0, ..., 7]`.

## Physical sanity

The only timed quantity is S10 and its oracle is the accepted m5 record, so
the check is exact rather than bounded. The floor stated before the run still
holds: one decode step at `W=8` cannot complete faster than the 24 per-layer
compute gates the m5 record froze, `24 * 10,111 ns = 242,664,000 ps`, and the
measured 448,764,528 ps sits a factor of 1.85 above it. The harness compares
that bound against the value this run measured, not against the frozen
literal, so it is a guard a defect can trip; the observed makespan and the
floor are both written into the tracked results.

The directions are the ones m5 recorded, and the reason is the one m5 gave.
Its check C-q1 measured the all-to-allv network component growing with `W` at
fixed total payload: 158,914,560 then 190,371,888 then 206,100,528 ps for
decode and 16,202,127,360 then 24,255,191,088 then 28,281,722,928 ps for
prefill at `W` of 2, 4 and 8. Prefill is compute-bound with its per-rank
expert work invariant in `W`, so nothing offsets that growth and the makespan
rises. Decode is memory-bound with the resident expert weights per rank
shrinking as `W` grows, and that saving outruns the same network growth, so
the makespan falls. A manifest-driven run landing on any other value than the
m5 literal would be a defect in the builder, the sink or the harness, never a
calibration finding, and none did.

## Choices the freeze left open

These points were not settled by the freeze, or are narrower than the freeze
reads, and are recorded here rather than left implicit in the code.

- **The GPU number a rank lands on.** The launcher numbers a rank's GPU
  `base_gpu_id + (pp_rank % pp_per_node) * tp_per_node +
  (tp_rank % tp_per_node) * gpu_id_step`
  (`managers/data_parallel_controller.py`), where the base term is the
  `--base-gpu-id` option plus the replica's own offset. The builder states the
  default deployment of one replica with `--base-gpu-id 0` and
  `--gpu-id-step 1`, under which that expression collapses to the
  global-rank-order fill the freeze describes. The emitted `local_rank`
  therefore equals the launcher's `gpu_id` only under those defaults; a
  deployment that offsets or strides its GPU ids is placed differently on the
  device axis while its rank space is unchanged. Both options are added to
  PLACE-14.
- **When the three MoE memberships appear.** `initialize_model_parallel`
  creates the expert-parallel group and both MoE side groups unconditionally,
  while the builder emits them only when `experts` is given. The gate is a
  declared-manifest convention, not a framework rule, and it is what keeps the
  expert-free SGLang manifest to the three base memberships the freeze's S7
  and the `group_key_order_without_experts` row name.
- **What the singleton `dp` membership means.** SGLang builds no
  data-parallel process group over these ranks: attention data parallelism
  lives inside the tensor group and router-style replicas are separate worlds.
  The singleton entry is a manifest-schema convention recording that this
  manifest describes one replica, in the vocabulary every other declared
  manifest already uses. It is not a projection of a framework group, and a
  consumer must not read it as one.
- **The node refusal that needs no branch.** The freeze lists four node
  refusals; the interface refuses all four with three tests. Above `pp` the
  only node count surviving the "divides or is divided by `pp`" refusal is
  `nodes = pp * k`, and a world of `tp * pp` divisible by `pp * k` forces `k`
  to divide `tp`, so "tp indivisible by `nodes // pp`" is implied by the
  world-divisibility refusal rather than tested separately. A brute force over
  `tp` to 64, `pp` to 32 and `nodes` to 128 finds no input reaching it. The
  earlier draft carried the branch anyway, which was code no caller could
  execute; it is removed and the implication is stated in the builder.
- **The default node count.** The freeze says `nodes` defaults to "the fewest
  that fit", and separately refuses a node count that neither divides nor is
  divided by `pp`. The implementation reads "fit" literally as the world at
  `gpus_per_node`, then applies the launcher's constraints to whatever count
  is in force, so a defaulted and an explicit count are refused on the same
  terms. The alternative, silently inflating the default to the smallest
  conforming count, would have made `declared_sglang_manifest(tp=4, pp=3)`
  quietly take three nodes where two fit. Every frozen cell is unaffected:
  `tp=8, pp=2` takes two nodes of eight either way.
- **Where the refusals sit.** The freeze names the refusals but not their
  order. Width and divisibility are checked first, then the node fill, then
  the expert layout, all before the first rank is built. Every frozen S6 row
  raises a `ValueError` naming its own field under this order, and the run
  observes the "before any rank" half rather than asserting it: each control
  is built with a hostname pattern whose `format` raises, so a refusal that
  fired late would surface as a failed control, and the same pattern is run
  against an accepted layout to prove it is not inert.
- **How the harness reaches its own tree.** The study inserts the repository
  root at the front of `sys.path` and passes it to the PLACE-3 subprocess,
  because an installed editable distribution otherwise resolves `simllm` to a
  different checkout than the freeze the study is checking against. The
  PLACE-3 harness does not do this; it is the one place this harness does not
  mirror it.

## Reproduction

```bash
source .env.local.sh
python examples/sglang_declared_layout_v1/run_study.py
python examples/sglang_declared_layout_v1/run_study.py --check --output-root <fresh>
```

`SIMLLM_HTSIM_RNIC` and `SIMLLM_TXT2BIN` select the pinned binaries,
`SIMLLM_SGLANG_ENV` (or `--sglang-python`) selects the interpreter that
answers the S5 oracle arm, and `SIMLLM_DATA_ROOT` or `--output-root` owns the
bulk artifacts. Without an SGLang interpreter the oracle cell records "not
run" and the status is `INCOMPLETE` rather than `PASS`, because the freeze
makes that arm required for closure. The check mode rebuilds every cell and
compares the summary to the tracked results, ignoring only the implementation
commit stamp.

The tracked `implementation_commit` field is that stamp: it records whichever
commit was checked out when the run happened, it is the one field `--check`
excludes, and it is therefore not repointed when history is rewritten. It
still reads `7c02b28e`, the pre-rebase hash of the run; the rebased twin of
that commit is `79c055e8`. The expectations and harness commits above, which
`--check` does enforce, were repointed at their merged hashes.
