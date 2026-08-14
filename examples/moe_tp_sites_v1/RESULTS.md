# MoE tensor-parallel all-reduce site inventory, v1 results

The tensor-parallel lowering used to emit two ring all-reduces for every layer
of every model. For a routed MoE model whose experts are expert-parallel, the
mlp-site all-reduce is a collective the deployment never executes: with
`moe_tp` equal to 1 each expert's down projection is computed whole on its
owner rank, the combine all-to-all returns finished expert vectors, and the
token's home rank forms the layer output by a local weighted sum, so no
partial sum spans the tensor-parallel group. TRAF-33 removed that site, keyed
on the model geometry alone, and this run qualifies the result.

All 120 scored instances in 4 families passed and no fatal guard was violated.
The reference cell, 24 layers with an 8-rank tensor-parallel group and an
8-rank expert-parallel group, now renders 24 all-reduces plus 48 all-to-alls,
that is 72 collectives rather than 96, and 8,257,536 rather than 16,515,072
tensor-parallel bytes.

## Chronology

The expectations were frozen in commit
`75a0a94c9035019230e65ec3475cc63d341bab10`, which contains no implementation,
no harness and no measured value. The behavior landed afterwards in
`d539feed62ad67f539f071e3a25dadea7c6be38b`, and the harness and the run
followed that. The freeze precedes both the implementation and the first run,
so the relations below are pre-registered rather than post-specified. One
disclosure belongs here rather than lower down: the freeze's own napkin
arithmetic was wrong, and the correction is in the physical sanity section.

## What ran

`run_study.py` renders one decode step over 54 cells: three model kinds
(`dense`, `routed-tp` with all 32 experts resident, `routed-ep` with 4 of 32
resident and an 8-rank expert-parallel group), tensor-parallel width 1, 2 and
8, layer counts 1, 4 and 24, and 3 or 12 new tokens. Every cell passes through
three independent consumers of `step_tp_allreduces`: the GOAL renderer
`render_step_goal`, the phase planner `step_communication_phases`, and the
graph path `SerialStepLowerer` with its attached `CollectivePlan`. No backend
binary, no network model and no timing is involved. Raw rows land outside Git
under the run root, 52,586 bytes of JSON.

```bash
python examples/moe_tp_sites_v1/run_study.py --out <run directory>
```

## Fatal guards, held and never scored

| Guard | Checks | Result |
|---|---:|---|
| F1 site rule returns the frozen tuple | 54 | held |
| F2 no tensor-parallel or expert-parallel width dependence | 54 | held |
| F3 the routed all-to-all inventory is untouched | 54 | held |
| F4 the dense and expert-tensor-sharded arms are unchanged | 12 | held |
| F5 a step with no collective at all is still refused | 12 | held |

F4 is stronger than the table row suggests. Besides the 12 in-study identity
checks, the dense and expert-tensor-sharded GOAL renders were compared byte for
byte against the pre-change renderer executed from the freeze revision, and
both are identical, digest
`c53782b27c241a85b37f9d81342ed8618e4402a8d2c6c3c5dbe4e59a1a587301` on the
two-layer four-rank fixture now pinned in `tests/test_step_comm.py`. The whole
suite is green apart from the developer-guide task-progress block, which the
integrator regenerates.

## Scored families, genuine risk only

| Family | Instances | Passed |
|---|---:|---:|
| S1 GOAL renderer against the frozen closed form | 36 | 36 |
| S2 phase planner against the frozen closed form | 36 | 36 |
| S3 graph lowerer and collective plan against the closed form | 36 | 36 |
| S4 tag disjointness and the moved all-to-all tag base | 12 | 12 |

The three path families are independent implementations rather than
restatements: the renderer and the phase planner each carried their own
positional index over the all-reduce list, and the plan's rounds, tags and
extents are built in another module. S4 is the relation the refactor was most
likely to break, and it is not entailed by the byte and count families.

Concretely, had the positional index `layer * 2 + site_index` been kept while
the list shortened, the reference cell's all-to-all block would still start at
tag 1,336 (`1000 + 24 * 14`) while layer 12's attention site would occupy tags
1,336 to 1,349, the first 14 tags of that block. The measured run has zero
tags shared between distinct operations at every one of the 12 cells that
render both families, and the all-to-all base sits exactly where the shortened
list puts it.

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
| routed-ep | 2 | 3 | 24 | 96 | 1,179,648 | 48 | 24 | 1,048 |
| routed-ep | 2 | 12 | 24 | 96 | 4,718,592 | 48 | 24 | 1,048 |
| routed-ep | 8 | 3 | 24 | 2,688 | 8,257,536 | 336 | 24 | 1,336 |
| routed-ep | 8 | 12 | 24 | 2,688 | 33,030,144 | 336 | 24 | 1,336 |

`routed-tp` matches `dense` exactly in every column at every cell, which is
the point of keeping it in the matrix: a routed model with no expert
parallelism tensor-shards its experts, its expert down projection really is a
partial sum over the tensor-parallel group, and dropping its mlp site would
have replaced one defect with another.

## Physical sanity, and a correction to the freeze

Bounds were written before any digit was read. A bandwidth-optimal all-reduce
over `W` ranks moves at least `2(W-1) * P` bytes in aggregate per call site;
the naive all-gather-then-reduce alternative moves `W(W-1) * P`. At `W = 8`
and `T = 3`, `P` is 24,576 bytes, so the floor is 344,064 and the ceiling
1,376,256 bytes per site. The measured per-site figure is exactly 344,064, on
the floor and a factor of four below the ceiling. At `W = 2` the floor and the
ceiling coincide at 49,152 bytes and the measured value is 49,152, so that
width does not discriminate between the two schedules and is reported as such.

The covariates scale as the model claims: 4.000000 for `T` from 3 to 12 and
7.000000 for `W` from 2 to 8, at every kind and layer count.

The freeze's end-to-end plausibility line is wrong and is corrected here. It
charged the whole 8,257,536-byte aggregate to a single 400 Gbit/s link and
reported 165.2 microseconds. Under this repository's default reference
configuration, one 400 Gbit/s NIC per GPU, the ring's rounds proceed
concurrently on eight NICs and the honest figure is per-rank egress:
43,008 bytes per site, 1,032,192 bytes over 24 sites, that is 20.64
microseconds at 400 Gbit/s. The freeze overstated the serialization surplus by
a factor of eight. No scored relation or fatal guard used that line, and the
direction of the finding is unchanged, but the magnitude claim was wrong and
is retracted.

The magnitude that actually matters is not bytes. Each suppressed site is one
semantic collective, and the calibrated collective base latency this repository
now charges is 30,128,029 ps per semantic collective (see the collective
latency floor and composed step budget studies). Removing 24 phantom
collectives from a 24-layer step removes `24 * 30.128029 = 723.073`
microseconds of additive base latency, against the 1.916754 millisecond
measured composed decode step of the composed step budget study. The defect
was therefore worth about 38 percent of a composed decode step once a real
tensor-parallel group is declared beside expert parallelism, roughly 35 times
the byte-serialization term. That is the number wave 15 should carry, and it
is why this was a correctness fix rather than a rounding matter.

Sanity against the system being imitated: one all-reduce plus two all-to-alls
per layer is the shape a deployed routed-MoE engine under expert parallelism
executes. Two plus two is a shape no such deployment runs, and the previous
inventory claimed it for every configuration.

## Entailed corollaries, reported and not scored

These follow arithmetically from S1 together with F1, so scoring them would
pad a denominator with restatements.

- The token ratio is exactly 4.000000 at every cell.
- The width ratio from `W = 2` to `W = 8` is exactly 7.000000 at every cell.
- The `routed-ep` arm carries exactly 0.500000 of the `dense` arm's
  tensor-parallel bytes at matched width, layer count and token count.
- The headline inventory is 24 all-reduces plus 48 all-to-alls, 72 collectives
  and not 96.

## What this run does not establish

- No timing, no backend and no placement. The study is lowering arithmetic;
  the end-to-end figure above is arithmetic on other studies' published
  constants, not a composed run.
- Mixed dense and routed layer schedules are not represented at all, because
  `ModelDims` carries one whole-model mixture geometry. TRAF-34 owns that.
- An expert-parallel group that still tensor-shards the expert weights
  (`moe_tp` above 1) is not represented, and neither adapter can produce one
  today. TRAF-35 owns that.
- `local_num_experts` defaulting to 0 means "all experts resident", so an
  expert-parallel group with more ranks than experts, where some rank owns
  none, is indistinguishable from no expert parallelism at all. That ambiguity
  predates this change and is unchanged by it; it is recorded under TRAF-35
  rather than silently narrowed here.
- Two harness defects were found and fixed before any relation was evaluated:
  a lookup that assumed every cell renders something, and two lint findings in
  the report formatting. Neither touched a modeled value, and no closed form
  was edited after a measurement.

## Storage

Raw rows live outside Git under the configured external run root, in
`moe_tp_sites_v1/cells.json`, 52,586 bytes. Nothing in this study writes a
large artifact.
