# Declared expert placement result

## Outcome

What ran: `examples/declared_expert_placement_v1`, the frozen PLACE-3
qualification, built the declared expert layouts of the worked example, the
m5 expert worlds and a DeepSeek-class 64-rank pipeline from implementation
commit `db59e2a8`, checked them against the pinned vLLM 0.27.1 layout rules,
and drove `HtsimStepSink` on `rnic-nn-fluid` with the expert-parallel (EP)
group read from the manifest. The expectations-only commit is `821969ae`.

What came out: the result is `PASS` with no finding. The deciding number is
6 of 6 frozen m5 check-B makespans reproduced exactly with the EP group taken
from the manifest, each with Group Operation Assembly Language (GOAL) text
byte identical to the run with the hand-typed rank list. All four structural
cells, all eight rejection controls and all five compatibility digests held.

What it changes for the project: PLACE-3 closes. The placement doc's
statement that a declared deployment carries its EP layout becomes literal:
a study can read `manifest.group_ranks(rank, "ep")` instead of typing the
group, and `ExpertPlacementSnapshot.from_manifest` builds expert ownership
from a declared manifest. The one deliberately unmodeled variant, a MoE
deployment with expert parallelism disabled, is registered as PLACE-7.

What it does not change: no packet backend other than the fluid closed form
ran, no time to first token or time per output token moved, no extracted
manifest changed, the m5 study and its record are untouched, and PLACE-1,
PLACE-2 and PLACE-6 stay open. The layout rules are the pinned framework's
own; nothing here calibrates a device or a fabric.

## Pinned semantics that the builder reproduces

The freeze read four rules from the installed vLLM 0.27.1 package and the
implementation reproduces them: the rank tensor `(dp * PP + pp) * TP + tp`,
the EP group of one pipeline stage as its `DP * TP` ranks ordered
data-parallel major with `rank_in_group = dp * TP + tp`, the `linear` and
`round_robin` expert maps over `num_experts // (DP * TP)` experts per rank
behind the framework's divisibility gate, and the pipeline partition that
hands the layer remainder to the stages indexed `-2, -3, ...` in order. The
freeze was checked against that source before the implementation started;
the check corrected one wrong stage interval in the draft before the freeze
commit `821969ae` (61 layers on four stages give `[15, 30)` then `[30, 46)`, not `[15, 31)`),
which is why the frozen C3 and C4 rows carry those values.

## Evidence

| Evidence class | Result |
|---|---|
| Scored exact-oracle rows (C2 makespans) | 6 of 6 exact, GOAL text byte identical in every row |
| Structural exact guards (C1, C3, C4, C6) | 4 of 4 cells exact |
| Rejection controls (C5) | 8 of 8 refused, each a `ValueError` raised by construction |
| Fatal compatibility digests | 5 of 5 reference manifests byte identical to the pre-change record |

Counts in different evidence classes are not added. The tracked
[results](results.json) hold every cell; bulk manifests, GOAL programs and
completion tables stay under the configured data root.

C1, the worked example (`tp=4, pp=2, dp=2`, 48 MoE layers, 32 experts): rank
9 sits in EP group `[0, 1, 2, 3, 8, 9, 10, 11]` at index 5 with layer range
`[0, 24)` and owns `[20, 21, 22, 23]` under `linear` and `[5, 13, 21, 29]`
under `round_robin`; rank 15 sits in `[4, 5, 6, 7, 12, 13, 14, 15]` at index
7 with `[24, 48)` and owns `[28, 29, 30, 31]` and `[7, 15, 23, 31]`. The
round-robin row is the hand-written worked example the placement tests have
carried since the manifest schema landed.

C2, the m5 identity (`tp=1, dp=W`, granite geometry):

| Shape | W | Makespan (ps) | Matches m5 | GOAL identical |
|---|---:|---:|---|---|
| decode8x2048 | 2 | 563,362,560 | yes | yes |
| decode8x2048 | 4 | 486,963,888 | yes | yes |
| decode8x2048 | 8 | 448,764,528 | yes | yes |
| prefill2048 | 2 | 17,592,951,360 | yes | yes |
| prefill2048 | 4 | 25,646,015,088 | yes | yes |
| prefill2048 | 8 | 29,672,546,928 | yes | yes |

The decode makespan falls with `W` and the prefill makespan rises with `W`,
the directions m5 recorded. The `W=8` decode value sits above the
242,664,000 ps compute floor stated in the freeze. In every world the expert
placement snapshot built from the manifest carries exactly 768 owner entries
with owner `expert // (32 / W)`.

C3, the DeepSeek-class pipeline (`tp=8, pp=4, dp=2`, 61 layers, MoE on
layers 3 through 60, 256 experts): four EP groups of 16, stage ranges
`[0, 15)`, `[15, 30)`, `[30, 46)`, `[46, 61)`, MoE layers per stage 12, 15,
16 and 15, 16 experts per rank, and 14,848 ownership entries with every
`(layer, expert)` pair owned exactly once.

C4 reproduces the five frozen partitions, C5 refuses all eight malformed
layouts, and C6 round-trips the C1 and C3 manifests through save and load
with `ep` following `dp` in every rank's groups.

## Amendment

What ran: the [2026-09-13 amendment](expectations-amendment-2026-09-13.md),
frozen at `121098c3` after a review finding and before the correction, and
the whole qualification rerun from implementation commit `bd93569c`.

What was refuted: the freeze's statement that the pinned vLLM 0.27.1 fused
MoE layer refuses an expert count that `DP * TP` does not divide. That
refusal exists only when expert-parallel load balancing is enabled; without
it `determine_expert_map` gives each EP rank below the remainder one extra
expert. The builder now follows that rule, the refusal is withdrawn from the
builder and from `declared_local_expert_ids`, and the C5 control
`num_experts 30 at ep_size 8` is withdrawn with it. The statement above that
places the expert maps behind the framework's divisibility gate is
superseded.

What came out: the result is `PASS` with no finding. Cell C8 (`tp=1, dp=8`,
24 MoE layers, 30 experts) gives per-rank counts `4, 4, 4, 4, 4, 4, 3, 3`
under both strategies, reproduces every owner list the amendment names,
owns every `(layer, expert)` pair exactly once, and projects 720 snapshot
owner entries in each strategy. C1 through C6 and the five compatibility
digests keep their frozen literals, and all six C2 makespans still match
exactly.

| Evidence class | Amended result |
|---|---|
| Scored exact-oracle rows (C2 makespans) | 6 of 6 exact, GOAL text byte identical in every row |
| Structural exact guards (C1, C3, C4, C6, C8) | 5 of 5 cells exact |
| Rejection controls (C5) | 7 of 7 refused, each a `ValueError` raised by construction |
| Fatal compatibility digests | 5 of 5 reference manifests byte identical to the pre-change record |

What it changes: PLACE-3 closes on the corrected builder, which passes the
original cells plus C8, and a remainder layout is representable in a
declared manifest. The refusals for an expert count below one and a negative
MoE layer index gain tests, and a field-level test shows that a manifest
built with experts differs from its expert-free twin only in the `ep` group,
the layer range, the expert ownership and the epoch. The PLACE-7 text now
shards experts across the flattened DP x TP group.

What it does not change: no step metric, GOAL text or reference manifest
moved, and the m5 record is untouched because its 32 experts divide every
world. The step-sink consumers still assume uniform per-rank expert
geometry, so a remainder layout cannot yet drive a timed run; that gap is
registered as PLACE-8.

## Reproduction

```bash
source .env.local.sh
python examples/declared_expert_placement_v1/run_study.py
python examples/declared_expert_placement_v1/run_study.py --check
```

`SIMLLM_HTSIM_RNIC` and `SIMLLM_TXT2BIN` select the pinned binaries;
`SIMLLM_DATA_ROOT` or `--output-root` owns the bulk artifacts. The check mode
rebuilds every cell and compares the summary to the tracked results, ignoring
only the implementation commit stamp.
