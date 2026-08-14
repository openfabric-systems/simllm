# MoE tensor-parallel all-reduce site inventory, v1 results

The tensor-parallel lowering used to emit two ring all-reduces for every layer
of every model. For a routed MoE layer whose output arrives through a combine
all-to-all, the mlp-site all-reduce is a collective the deployment never
executes: the combine returns finished expert vectors and the token's home rank
forms the layer output by a local weighted sum, so no partial sum spans the
tensor-parallel group. TRAF-33 removed that site, and this run qualifies the
result.

All 120 pre-registered scored instances in 4 families passed, no fatal guard
was violated, and a post-specified 36 further instances covering naive expert
parallelism also passed. The reference cell, 24 layers with an 8-rank
tensor-parallel group and a declared 8-rank all-to-all expert-parallel group,
renders 24 all-reduces plus 48 all-to-alls, that is 72 collectives rather than
96, and 8,257,536 rather than 16,515,072 tensor-parallel bytes.

## Chronology, and what was corrected after the review

The expectations were frozen in commit
`75a0a94c9035019230e65ec3475cc63d341bab10`, which contains no implementation,
no harness and no measured value. The behavior landed in
`d539feed62ad67f539f071e3a25dadea7c6be38b`, and the harness and the first run
followed. The freeze precedes both the implementation and the first run, so the
four families below are pre-registered.

An integrator review on 2026-08-14 then found the merged rule wrong for one
configuration and this report wrong in two statements. Everything in this
section is **post-specified**: it was written after results were observed and
carries no pre-registration credit.

- **The rule was corrected.** The first implementation keyed the suppression on
  the dims alone, `resident_experts < num_experts`. That is true for two
  physically different deployments. In the pinned vLLM 0.26.0,
  `model_executor/layers/fused_moe/config.py:1052-1055` enables all-to-all
  kernels only when expert parallelism is combined with `dp_size > 1`,
  `pcp_size > 1` or sequence parallelism, and
  `model_executor/layers/fused_moe/runner/moe_runner.py:436-465` all-reduces
  the fused output over the tensor-parallel group whenever the combine did not
  reduce it. A `tp=8, ep=8, dp=1` deployment, which is exactly what the
  reference cell and the collective plan default study declare, therefore runs
  naive expert parallelism with two all-reduces and no all-to-all per layer,
  and the merged rule rendered one and two. The rule now keys on the
  declaration that separates them: the mlp site is suppressed exactly when the
  layer renders a combine, i.e. routed dims plus a declared expert-parallel
  group of at least two ranks, which is the same gate `step_moe_alltoalls`
  uses. Naive expert parallelism is rendered by not declaring the group.
  TRAF-40 owns making that mode explicit instead of implicit.
- **A shared-expert hole was closed.** The runner all-reduces the shared-expert
  output over the tensor-parallel group even on the all-to-all path
  (`model_executor/layers/fused_moe/runner/moe_runner.py:416-433`), so a
  shared-expert model keeps an mlp-site reduction this rule drops. The vLLM
  reader accepted such models; it now refuses them, as the SGLang reader
  already did. VLLM-25 owns supporting them.
- **The magnitude claim was corrected**, see the physical sanity section.
- **The non-portability disclosure was extended**, see the stale surfaces
  section, and TRAF-41 now owns requalifying both.
- **Two registered fatal clauses were executed late** and one was strengthened,
  see the fatal guards section.

Every frozen number in this report was reproduced exactly by the rerun after
those changes. The frozen body of `expectations.md` was not edited.

## What ran

`run_study.py` renders one decode step over 72 cells. The 54 frozen cells cross
three model kinds (`dense`, `routed-tp` with all 32 experts resident,
`routed-ep` with 4 of 32 resident and a declared 8-rank expert-parallel group),
tensor-parallel width 1, 2 and 8, layer counts 1, 4 and 24, and 3 or 12 new
tokens. The 18 post-specified cells add a fourth kind, `routed-naive-ep`: the
same expert-parallel dims with no declared group, i.e. the naive shape the
review identified.

Every cell passes through three independent consumers of `step_tp_allreduces`:
the GOAL renderer `render_step_goal`, the phase planner
`step_communication_phases`, and the graph path `SerialStepLowerer` with its
attached `CollectivePlan`. No backend binary, no network model and no timing is
involved. Raw rows land outside Git under the run root, 123,439 bytes of JSON.

```bash
python examples/moe_tp_sites_v1/run_study.py --out <run directory>
```

## Fatal guards, held and never scored

| Guard | Checks | Result |
|---|---:|---|
| F1 site rule returns the expected tuple | 72 | held |
| F2 site-tuple invariance, four clauses | 48 | held |
| F2 the W=1 substitution | 24 | held |
| F3 routed all-to-all inventory untouched, operations and bytes | 72 | held |
| F4 every arm that renders no all-to-all keeps one GOAL identity | 12 | held |
| F5 a step with no collective at all is still refused | 18 | held |

Three of these are stronger than the first run's, and the strengthening is
post-specified.

F2's registered clauses are now executed rather than approximated. The freeze
registered invariance under tensor-parallel width, under a disjoint rank
relabeling, and under a change of expert-parallel group width; the first run
executed only the width clause. Each of the 48 checks now compares the site
tuple across four variations: the declared groups, tensor-parallel ranks
relabeled by plus 64, the expert-parallel group narrowed from 8 ranks to 2, and
the resident-expert count swept to all-resident, which the corrected rule no
longer reads. All four agree at every cell. The 24 checks at tensor-parallel
width 1 are reported under their own name because they are a substitution
rather than the registered predicate: a one-rank world emits no all-reduce, so
there is no site tuple to compare and the guard falls back to "emitted
nothing".

F3 now asserts bytes as well as operation counts, which the freeze registered
("may not move one expert byte") and the first harness computed but never
checked. The oracle is the uniform closed form
`2 * layers * (W_ep - 1) * ((tokens * top_k * hidden * dtype) // W_ep)`, which
never passes through the changed code. Two independent confirmations that this
is the merge-base value: `step_moe_alltoalls` and
`_routed_moe_alltoalls_from_layers` are textually identical to their
`e18b9b0102808e9b8e0f276c2b82c51ed8c5b51d` versions, and executing the
merge-base module over all 72 cells reproduces every operation count and byte
total with 0 mismatches.

F4 now covers three arms rather than two. `dense`, `routed-tp` and the new
`routed-naive-ep` all render no all-to-all and produce one identical GOAL
digest per shape, which is the pre-change renderer's output. On the two-layer
four-rank fixture pinned in `tests/test_step_comm.py` that digest is
`c53782b27c241a85b37f9d81342ed8618e4402a8d2c6c3c5dbe4e59a1a587301`, compared
byte for byte against the pre-change renderer executed from the freeze
revision.

## Scored families

| Family | Instances | Passed | On the changed path |
|---|---:|---:|---:|
| S1 GOAL renderer against the frozen closed form | 36 | 36 | 12 |
| S2 phase planner against the frozen closed form | 36 | 36 | 12 |
| S3 graph lowerer and collective plan against the closed form | 36 | 36 | 12 |
| S4 tag disjointness and the moved all-to-all tag base | 12 | 12 | 12 |

Post-specified, on the naive expert-parallel arm the freeze did not enumerate:

| Family | Instances | Passed |
|---|---:|---:|
| S1 renderer | 12 | 12 |
| S2 phases | 12 | 12 |
| S3 graph | 12 | 12 |

The two denominators are separate evidence classes and are not summed, and
neither is summed with the guard counts.

What the 120 does and does not mean. Only 48 of the 120 pre-registered
instances exercise a layer whose inventory the change alters; the other 72 are
`dense` and `routed-tp` regression arms whose value is that they did not move.
That is legitimate evidence, since a refactor of three positional indexes could
easily have moved them, but it is regression evidence rather than 120
independent tests of the new behavior. S3's consumer,
`simllm/backends/step_lowerer.py`, already looked its sites up by key and was
not edited in the first round; the correction round edits it to thread the
expert-parallel group through, so it is now an edited consumer rather than an
untouched one.

The three path families remain independent implementations rather than
restatements: the renderer and the phase planner each carried their own
positional index over the all-reduce list, and the plan's rounds, tags and
extents are built in another module. S4 is the relation the refactor was most
likely to break. Had the positional index `layer * 2 + site_index` been kept
while the list shortened, the reference cell's all-to-all block would still
start at tag 1,336 (`1000 + 24 * 14`) while layer 12's attention site would
occupy tags 1,336 to 1,349, the first 14 tags of that block. The measured run
has zero tags shared between distinct operations at every one of the 12 cells
that render both families.

## Exact rows at 24 layers

`W` is the tensor-parallel width, `T` the new-token count. Sites are rendered
all-reduce operations; bytes are the directed payload the renderer records for
them.

| Kind | W | T | Sites | Messages | TP bytes | Phase rounds | Graph ops | a2av tag base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2 | 3 | 48 | 192 | 2,359,296 | 96 | 48 | none |
| dense | 2 | 12 | 48 | 192 | 9,437,184 | 96 | 48 | none |
| dense | 8 | 3 | 48 | 5,376 | 16,515,072 | 672 | 48 | none |
| dense | 8 | 12 | 48 | 5,376 | 66,060,288 | 672 | 48 | none |
| routed-tp | 2 | 3 | 48 | 192 | 2,359,296 | 96 | 48 | none |
| routed-tp | 8 | 3 | 48 | 5,376 | 16,515,072 | 672 | 48 | none |
| routed-naive-ep | 2 | 3 | 48 | 192 | 2,359,296 | 96 | 48 | none |
| routed-naive-ep | 8 | 3 | 48 | 5,376 | 16,515,072 | 672 | 48 | none |
| routed-ep | 2 | 3 | 24 | 96 | 1,179,648 | 48 | 24 | 1,048 |
| routed-ep | 2 | 12 | 24 | 96 | 4,718,592 | 48 | 24 | 1,048 |
| routed-ep | 8 | 3 | 24 | 2,688 | 8,257,536 | 336 | 24 | 1,336 |
| routed-ep | 8 | 12 | 24 | 2,688 | 33,030,144 | 336 | 24 | 1,336 |

The three no-all-to-all kinds match each other exactly in every column, and
that is the substantive point of the corrected rule. `routed-tp` and
`routed-naive-ep` carry different `local_num_experts` (32 of 32 against 4 of
32) and render identically, which directly demonstrates that the rule no longer
reads the resident-expert count. `routed-naive-ep` and `routed-ep` carry
identical dims and differ only in whether the expert-parallel group is
declared, which is the whole of the corrected mechanism.

## Physical sanity, and two corrections

Bounds were written before any digit was read. A bandwidth-optimal all-reduce
over `W` ranks moves at least `2(W-1) * P` bytes in aggregate per call site;
the naive all-gather-then-reduce-locally alternative moves `W(W-1) * P`. At
`W = 8` and `T = 3`, `P` is 24,576 bytes, so the floor is 344,064 and the
ceiling 1,376,256 bytes per site. The measured per-site figure is exactly
344,064, on the floor and a factor of four below the ceiling. At `W = 2` the
floor and the ceiling coincide at 49,152 bytes and the measured value is
49,152, so that width does not discriminate between the two schedules and is
reported as such.

The covariates scale as the model claims: 4.000000 for `T` from 3 to 12 and
7.000000 for `W` from 2 to 8, at every kind and layer count.

**Correction one, the freeze's own napkin line.** It charged the whole
8,257,536-byte aggregate to a single 400 Gbit/s link and reported 165.2
microseconds. Under this repository's default reference configuration, one
400 Gbit/s NIC per GPU, the ring's rounds proceed concurrently on eight NICs
and the honest figure is per-rank egress: 43,008 bytes per site, 1,032,192
bytes over 24 sites, that is 20.64 microseconds at 400 Gbit/s. The freeze
overstated the serialization surplus by a factor of eight. No scored relation
or fatal guard used that line.

**Correction two, the denominator.** The first version of this report said the
defect was worth about 38 percent of a composed decode step. The 1.916754 ms
composed decode step it divided by was measured at `tp_ranks=(0,)`, a
configuration with zero all-reduces, so the denominator excluded the very term
being weighed. With a tensor-parallel group of 8 declared beside the
expert-parallel group, and charging the calibrated 30,128,029 ps base latency
per semantic collective additively as the composed step budget study measured
it, the step becomes `1,916.754 + 24 * 30.128029 = 2,639.827` microseconds
under the corrected inventory and `1,916.754 + 48 * 30.128029 = 3,362.899`
microseconds under the defect. The 24 phantom collectives are therefore
**21.50 percent of the defective step and 27.39 percent of the corrected
step**, not 38 percent. That is still roughly 35 times the 20.64 microsecond
byte-serialization term, so the conclusion that the base latency and not the
bytes carries the weight is unchanged. Both figures are arithmetic on other
studies' published constants, not a measured composed run.

Sanity against the system being imitated: one all-reduce plus two all-to-alls
per layer is the shape a deployed routed-MoE engine executes when the combine
kernels are active, and two all-reduces plus no all-to-all is the shape it
executes when they are not. The previous inventory claimed two plus two, which
is neither.

## Entailed corollaries, reported and not scored

These follow arithmetically from S1 together with F1, so scoring them would pad
a denominator with restatements.

- The token ratio is exactly 4.000000 at every cell.
- The width ratio from `W = 2` to `W = 8` is exactly 7.000000 at every cell.
- The `routed-ep` arm carries exactly 0.500000 of the `dense` arm's
  tensor-parallel bytes at matched width, layer count and token count.
- The headline inventory under declared all-to-all expert parallelism is 24
  all-reduces plus 48 all-to-alls, 72 collectives and not 96. The same dims
  under naive expert parallelism render 48 all-reduces and no all-to-all.

## Published surfaces this change makes stale

TRAF-41 owns requalifying both, and neither is rerun here.

- The Granite live cells of
  [the plan default study](../collective_plan_default_v1/RESULTS.md) declare
  one 8-rank group as both the tensor-parallel and the expert-parallel group
  over expert-parallel dims, so their 709,803,840 ps TTFT, 132,794,880 ps TPOT
  and transport rows were measured with 48 rather than 24 all-reduces per step.
  That study's dense coverage, perturbation and bypass cells, including the
  196,608-byte and 4,730,040 ps rank-order row, use dense dims and are
  unaffected.
- [The composed step budget study](../composed_step_budget_v1/RESULTS.md)
  publishes `48 * 30,128,029 = 1,446,145,392` ps as the tensor-parallel
  collective-floor addition for the mission `a-ep8` dims and a 74.73 to 75.45
  percent share of the composed step at the graph profile. Those dims are
  expert-parallel (`examples/end_to_end_replay_v1/run_study.py:415` sets
  `local_num_experts` to 4 of 32), so under a declared all-to-all group the
  corrected addition is `24 * 30,128,029 = 723,072,696` ps and the band roughly
  halves. Under naive expert parallelism the 48 figure stands, which is
  precisely why TRAF-40 wants the mode declared rather than inferred.
- The MoE studies that render an expert-parallel group with a tensor-parallel
  world of one are unaffected, because they emit no all-reduce either way.

## What this run does not establish

- No timing, no backend and no placement. The study is lowering arithmetic; the
  end-to-end figures above are arithmetic on other studies' published
  constants, not a composed run.
- Nothing detects a caller that declares an expert-parallel group for a `dp=1`
  deployment, because neither `ModelDims` nor the group inputs carry
  `dp_size`. The declaration is trusted. TRAF-40 owns that.
- Mixed dense and routed layer schedules are not represented, because
  `ModelDims` carries one whole-model mixture geometry. TRAF-34 owns the
  traffic half, VLLM-25 and SGL-18 the reader halves, and both readers now
  refuse the sentinel fields rather than pricing them as fully routed.
- An expert-parallel group that still tensor-shards the expert weights
  (`moe_tp` above 1) is not represented, and neither adapter can produce one
  today. TRAF-35 owns that.
- Shared-expert models are refused rather than rendered, because their
  shared-expert output keeps a tensor-parallel all-reduce this rule would drop.
  VLLM-25 owns support.
- Failed attempts and defects found, all disclosed: the merged rule was wrong
  for naive expert parallelism and was corrected after review; two harness
  defects were fixed before any relation was evaluated in the first run, a
  lookup that assumed every cell renders something and two lint findings; and
  the two corrections above were made after observing results and are labeled
  post-specified. No closed form was edited after a measurement.

## Storage

Raw rows live outside Git under the configured external run root, in
`moe_tp_sites_v1/cells.json`, 123,439 bytes. Nothing in this study writes a
large artifact.
