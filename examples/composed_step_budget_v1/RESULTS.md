# Composed step budget v1 results

The two dominant terms of the mission error budget landed on main in one
integration wave, and the two branches published arithmetic projections of the
same merged code that disagree by a factor of 1.7521233202. This study ran the
mission chain with both features enabled and with both disabled, and measured
the composed step instead of projecting it.

Expectations were frozen in commit `f1371e8`, before the harness that produces a
measured number existed and before any run. Every clause quoted here is quoted
from that commit, except the attempt-two refreeze of one fatal predicate, which
is labelled post-specified everywhere it appears.

## Outcome in one paragraph

The merged code composes **additively**: a step costs
`max(provider compute, launch count times per-launch point)`, plus one
calibrated base latency per semantic collective, plus the raw packet-level
fabric service. The overlapped reading, `max(compute + network, launch
demand)`, is not what the repository computes. Production attempt one is
**void** because a fatal predicate of the freeze compared a raw provider value
against a quantized literal; it is retained with findings and closes nothing.
Production attempt two is **not void**: all ten fatal guards held, and all
**3 of 3** scored behavioral families passed, for a genuine-risk denominator of
3. A case A decode step at 400 Gbit/s moves from the accepted 0.204527 ms to
**1.916754 ms** under the CUDA-graph host profile and **2.901192 ms** under the
eager profile, so the mission's composed optimism against its own 1.1 to 4.5 ms
comparable-deployment band moves from 5.38x to 22.00x down to **0.379x to
2.348x**. The composition is defensible rather than defective, so **no ID is
registered**. The headline finding is not the millisecond: it is that
**94.03 to 96.05 percent of the composed decode step is two transferred
constants**, one measured on a consumer GPU and one on an intra-node NVLink
all-reduce, and only 3.9 to 6.0 percent of it is the packet-level fabric this
repository actually simulates.

## Chronology and two-sided freeze integrity

The first measured run happened while `33b4b12` was checked out. Every commit
after it is classified here.

| Commit | Classification | Modeled behavior and measured before/after |
|---|---|---|
| `f1371e8` | Expectations only, pre-run | The freeze. No implementation of the composed timing path, no measured value. Its check-only entry point re-derives every literal and creates nothing. |
| `33b4b12` | Harness and default-preserving library seam | Added the mission runner's three optional selectors, the composition record and the new study. The accepted configuration produces byte-identical artifacts, proven by G1 below and by a pytest. |
| `0b00ad0` | Expectations only, post-specified refreeze after a void run | No implementation behavior changed. Fatal G10 changed from "the literal 99,024,000 ps appears among the raw provider values" to "every raw provider value lies inside the already frozen `CQ` interval". No interval of F1, F2 or F3 moved, no constant moved. |
| `9fc5d77` | Evidence harness only | Implemented the refrozen predicate and locked both directions with a regression. No modeled behavior and no measured value changed: attempt two reproduced every attempt-one raw value exactly. |
| this evidence commit | Reporting fix plus evidence | Corrected one diagnostic denominator that divided a non-median step's components by the median latency, so the three reported shares now sum to exactly 1. Every fatal guard and every scored relation is byte-identical between the run-time summary and the re-analysed one; only `fabric_share` moved, by at most 0.0011. |

No commit after the first measured run changed the execution graph, the
lowerer, the collective plan, the backend call, a calibrated constant or the
service equation. The only modelled-behavior change in the branch, the mission
runner's optional selectors, landed before the first run and its disabled path
is byte-locked.

**Attempt two reproduces attempt one exactly.** All four cells' `steps.jsonl`
streams, all four `cell.json` payloads with `wall_seconds` removed, all three
`composition.json` records and the capture are byte-identical across the two
attempts. The run is deterministic, and the refreeze demonstrably changed
nothing but the guard predicate.

## What ran

| cell | host profile | launches | collective profile | rate | steps | backend runs | wall |
|---|---|---:|---|---|---:|---:|---:|
| `off-400g` | `ideal` | 0 | off | 400G | 41 | 1,968 | 641 s |
| `on-graph440-400g` | `turing-cuda-graph` | 440 | `b200-nccl-2.27-local-v1` | 400G | 33 | 1,584 | 559 s |
| `on-eager567-400g` | `turing-eager-host` | 567 | `b200-nccl-2.27-local-v1` | 400G | 33 | 1,584 | 559 s |
| `on-graph440-200g` | `turing-cuda-graph` | 440 | `b200-nccl-2.27-local-v1` | 200G | 33 | 1,584 | 552 s |

140 simulated steps, 6,720 `htsim_rnic` invocations, 236 request-token
intervals, 99 steps carrying calibrated base charges, 50,802,688 routed bytes
in every cell. The enabled cells run 33 steps rather than 41 because a slower
step reaches each declared arrival after fewer steps, which is the closed loop
behaving correctly and the reason every cross-cell relation here is evaluated
over matched scheduling compositions.

## Physical sanity, positions stated against bounds written before the run

| quantity | frozen floor | frozen ceiling | measured | position |
|---|---:|---:|---:|---|
| composed decode step, graph, 400G | 1.802240 ms | 1.939528 ms | 1.907252 to 1.926667 ms | inside, in the upper half |
| composed decode step, eager, 400G | 2.786678 ms | 2.923966 ms | 2.891690 to 2.911105 ms | inside, in the upper half |
| raw fabric sum per decode step, 400G | 96.000 us | 140.000 us | 105.011 to 124.426 us | inside |
| raw fabric sum per decode step, 200G | 96.000 us | 200.000 us | 114.022 to 152.853 us | inside |
| critical endpoint load per collective | 14 B | 458,752 B | 6,144 to 378,880 B | inside, at 83 percent of the ceiling |

The endpoint-load position is worth stating plainly because it was a real
hazard rather than a formality. With the enabled features a step takes about
2 ms, so both later requests are admitted during step 0 and vLLM batches their
two prefills into one step of 34 tokens. That step's largest per-rank load is
378,880 bytes against a calibrated envelope whose width-8 ceiling is 458,752
bytes. The guard held with 17 percent of margin. Case B, at four times the
requests, would very likely exceed it, which is why the freeze declined to run
case B and why an envelope violation was registered in advance as a TRAF-32
defect rather than as a surprise.

Two further positions, from the same frozen bounds:

- **Weight-read floor.** Per-rank resident weights are 554,631,168 bytes, i.e.
  69.329 us at the B100 envelope's 8.0e12 bytes/s. The composed step is 27.6x
  (graph) to 41.9x (eager) that floor, so the modeled step is no longer compute
  bound in any sense.
- **Serialization share.** A decode step moves at most 2,064,384 bytes, i.e.
  41.288 us at 400 Gbit/s, so at most 2.2 percent of the composed step can be
  byte serialization. Everything else is fixed cost. That is why F3's ratio had
  to be near one, and it is.

## The measured composition

One decode step, `on-graph440-400g` step 2, with three requests decoding, in
picoseconds:

```text
1,926,666,680  =  356,095,000  +  1,446,145,392  +  124,426,288
step latency       host term       48 base           raw fabric
                                   charges           transport
```

The same step composition in `on-eager567-400g`:

```text
2,911,104,680  =  1,340,533,000  +  1,446,145,392  +  124,426,288
```

The two host profiles produce **the identical raw fabric term**, 124,426,288
ps, and differ only by the host term. That is the composition, read directly
off the measurement: the host launch demand overlaps provider compute and
nothing else, and the collective-bearing service is charged outside that
overlap.

Two supporting observations from the same data:

- **The provider estimate is fully masked.** The raw B100 provider compute over
  all enabled steps runs 99,032,502 to 99,585,462 ps, always below the graph
  profile's 356,094,640 ps launch demand and far below the eager profile's
  1,340,532,585 ps. The published compute service is exactly 356,095,000 or
  1,340,533,000 ps for every simulated step of the respective cell, so the
  modeled compute contributes exactly zero exposed picoseconds. With a
  calibrated host profile selected, the compute model stops affecting the step
  at all.
- **The composition is additive in both new terms at once.** Over the 18 decode
  compositions shared between the disabled cell and an enabled cell, the enabled
  latency minus the disabled latency equals
  `Q - disabled compute service + 48 * 30,128,029` exactly, to the picosecond,
  with zero violations. Nothing else about the step changed.

### Which projection was right

| reading | projected | measured |
|---|---:|---:|
| additive, graph 440 | 1.907743 ms | 1.907252 to 1.926667 ms |
| additive, eager 567 | 2.892181 ms | 2.891690 to 2.911105 ms |
| overlapped, both profiles | 1.650672 ms | not observed in any of 93 decode steps |

The published additive projections land 491,524 ps below the minimum measured
decode step of their cell, a 0.026 percent difference, because the projection
used the accepted step-1 fabric service while the enabled schedule reaches
r00's decode tokens in a different order. The overlapped projection is
2.11e8 ps outside the nearest measured value.

## Fatal guards

Fatal means void, not a lost point, so these are never reported as a fraction.
In attempt two all ten held.

| guard | asserts | observed |
|---|---|---|
| G1 | the disabled cell reproduces every accepted mission literal | `steps.jsonl` SHA-256 `f7c3b858...`, step 0 372,217,008 ps, step 1 204,526,734 ps, three TTFTs and their queue, kernel and collective components, 41 steps, 1,968 backend runs, 50,802,688 routed bytes, 25,401,344 peak egress, three per-request byte totals: all exact |
| G2 | oracle identity | capture SHA-256 `ef570a67...` |
| G3 | conservation | 0 failures over 140 steps and 236 intervals |
| G4 | inactive components | 0 violations |
| G5 | replay identity | 12 request instances, tokens, stop reasons and positive TTFT all exact |
| G6 | backend health | 140 of 140 steps `captured`, epoch 0, quiescent |
| G7 | floor reach | 99 steps, each exactly 48 charges of 30,128,029 ps summing to 1,446,145,392 ps, one per semantic collective; the disabled cell charged none |
| G8 | endpoint envelope | 6,144 to 378,880 B inside `[14, 458,752]` |
| G9 | device disclosure | enabled cells report `gtx1660-ti-sm75` with the `b100` provider envelope |
| G10 | host term reach, post-specified predicate | raw provider compute 99,032,502 to 99,585,462 ps inside `[95,000,000, 105,000,000]` |

G1 is the first evidence that the two disabled paths hold **together**. Each
wave-12 branch proved only its own, and both edits land in the same functions.

## Scored behavioral relations

Three families, all evaluated, all passed. The frozen evaluability rule for F2
and F3 required at least two matched decode compositions; 31 were available for
each, so neither family was reduced.

### F1, the composition rule: **passes**

| cell | decode steps | measured range (ps) | additive interval | in additive | in overlapped | compute service |
|---|---:|---|---|---:|---:|---:|
| `on-graph440-400g` | 31 | 1,907,251,602 to 1,926,666,680 | `[1,898,240,392, 1,942,240,392]` | 31 | 0 | 356,095,000 |
| `on-eager567-400g` | 31 | 2,891,689,602 to 2,911,104,680 | `[2,882,678,392, 2,926,678,392]` | 31 | 0 | 1,340,533,000 |
| `on-graph440-200g` | 31 | 1,916,262,802 to 1,955,092,920 | `[1,898,240,392, 2,002,240,392]` | 31 | 0 | 356,095,000 |

93 decode-step observations, every one inside its cell's additive interval and
none inside the overlapped interval, and the published compute service is the
frozen quantized host term exactly in every simulated step of every enabled
cell.

**Entailment, as frozen.** No registered guard pins `compute_service_ps`. G3
and G7 do force `step latency = compute service + 1,446,145,392 + fabric sum`,
so F1 reduces to a statement about the compute service, which is exactly the
quantity the two merged branches disagree about. The candidate values differ by
3.60x and 13.54x, so no rounding or scheduling accident could move a
measurement between the intervals. The relation could have failed and did not.

### F2, host-profile separation: **passes**

31 matched decode compositions between the two 400 Gbit/s enabled cells. The
eager latency minus the graph latency is **984,438,000 ps in every one of the
31 pairs**, with no other value observed. That is exactly
`Q_eager - Q_graph`. Under the overlapped composition the difference would have
been exactly zero, because the largest launch demand, 1,340,532,585 ps, is
smaller than the collective-bearing step service.

F2 is not entailed by F1: F1's intervals admit any difference in an 88,000,000
ps window, while F2 demands one picosecond-exact value inside it and got it 31
times with zero spread.

### F3, bandwidth compression under the floor: **passes**

31 matched decode compositions between 200 and 400 Gbit/s. Ratios run
**1.0047247 to 1.0147541**, inside the frozen `(1.000, 1.030]` band. The
accepted mission study's published step-makespan ratios for the identical
halving on the disabled path are 1.0441 to 1.4760, so the whole enabled range
sits below the whole disabled range. The floor behaves as an additive,
rate-independent term, and it compresses the model's bandwidth sensitivity by
roughly an order of magnitude.

F3 does not discriminate the two compositions and was never claimed to. It is
the scaling companion the physical-sanity rule requires, and it can fail: a
rate-dependent base latency, a per-byte floor or a 3 percent shift in matched
fabric loads would each break it.

## Exact-unscored relations

All three pass and none carries a numerator or denominator.

- **E1** step latency equals compute service plus base charges plus raw fabric,
  for all 99 enabled steps. Entailed by the artifact equation and G3.
- **E2** the disabled cell charges no base latency and its compute service stays
  in the ideal interval, 99,024,000 to 99,504,000 ps over 19 distinct values.
  Entailed by G1 and G7.
- **E3** per-request TTFT equals the sum of its first interval's seven
  components, and `tpot * (tokens - 1)` equals the sum of the post-first-token
  components exactly in rational arithmetic, in all four cells including the
  enabled ones. Entailed by G3, retained because the task asked for it.

## Post-specified diagnostics

Not frozen, reported and never scored.

1. **Rate cancellation survives the floor.** Because the raw fabric term is
   affine in the reciprocal of the link rate, `2 * fabric(400G) - fabric(200G)`
   cancels the byte term. Over the 31 matched decode compositions it returns
   96,000,006 to 96,000,048 ps, i.e. the mission study's own
   `48 * 2.000 us` propagation constant plus 6 to 48 ps of backend completion
   quantization. The calibrated 30.128029 us intercept is entirely outside the
   raw transport, which is what a separately reported floor is supposed to be.
2. **Cross-cell exactness.** 18 rows, zero violations, described above.
3. **Implied decode rate.** Per-request `1 / TPOT` falls from 4,448 to 4,786
   tokens per second on the disabled path to 343 to 345 (eager) and 518 to 522
   (graph) tokens per second. The lower figures are far more plausible for a
   cross-node expert-parallel deployment of a 400M-active MoE model. That
   plausibility is inherited from two transferred constants, not earned by
   calibration, and the accepted mission study's own S7 band `[1000, 20000]`
   was explicitly a defect detector rather than a plausibility claim, so
   falling below it is not a violation of anything.

## Mission error budget, recomputed from the measurement

| quantity | before this study | measured now |
|---|---:|---:|
| case A decode step at 400 Gbit/s, disabled | 0.204527 ms | 0.204527 ms, reproduced exactly |
| case A decode step, graph 440 | not measured | **1.916754 ms** median, 1.907252 to 1.926667 ms |
| case A decode step, eager 567 | not measured | **2.901192 ms** median, 2.891690 to 2.911105 ms |
| optimism against the 1.1 to 4.5 ms comparable band | 5.3783x to 22.0020x | **0.5739x to 2.3477x** (graph), **0.3792x to 1.5511x** (eager) |

Taking the union across the two host profiles, the composed optimism range is
**0.379x to 2.348x**. A ratio below one means the model is now pessimistic
against the low end of the comparable band. That is not an accuracy win: it is
what happens when two independently transferred constants are added to a model
that previously omitted both, and the direction of the residual is not known.

### What the composed step is made of

The median decode step, decomposed into what actually produced each
picosecond:

| term | graph 440, 400G | share | eager 567, 400G | share |
|---|---:|---:|---:|---:|
| host launch demand, GTX 1660 Ti transfer | 356,095,000 ps | 18.58% | 1,340,533,000 ps | 46.21% |
| collective floor, DGX B200 NVLink all-reduce transfer | 1,446,145,392 ps | 75.45% | 1,446,145,392 ps | 49.85% |
| packet-level fabric service, actually simulated | 114,513,968 ps | 5.97% | 114,513,968 ps | 3.95% |
| modeled B100 compute, exposed | 0 ps | 0.00% | 0 ps | 0.00% |

**94.03 percent (graph) and 96.05 percent (eager) of the composed decode step
is two transferred constants**, and the modeled compute contributes nothing at
all because the launch floor masks it. This is the most useful thing this study
can report. The provenance of both transfers is stated in the wave-12 records
and is unchanged by anything here: the host constant is a CUDA-graph node
replay time on a GTX 1660 Ti with an AMD Ryzen 9 3950X host, multiplied by a
launch count enumerated statically from vLLM 0.26.0; the collective intercept
is a public eight-B200 intra-node NVLink **all-reduce** capture applied
unchanged to cross-node pairwise **all-to-allv**, sitting 0.4 percent above the
upper endpoint of the band it was fitted to land in. The composed number is
therefore a three-device chimera and is not presented as a prediction for the
reference deployment.

## The traffic-coverage claim: verified, and then qualified

The claim under test, stated in the freeze before it was checked: the largest
remaining error is traffic coverage, adding a 24-layer model's tensor-parallel
all-reduces would add roughly another 1.446 ms, and that is 50 to 76 percent of
the whole composed step.

**The arithmetic is verified.** Megatron-style tensor parallelism performs two
all-reduces per transformer layer, one after the attention output projection
and one after the MLP output projection. Twenty-four layers give 48
all-reduces, the identical collective count MoE dispatch and combine already
produce, so at the calibrated width-8 intercept the addition is exactly
`48 * 30,128,029 = 1,446,145,392` ps. Measured against the composed decode step
this is **74.73 to 75.45 percent** at the graph profile and **49.85 percent**
at the eager profile, i.e. **49.85 to 75.45 percent**, which is the quoted 50 to
76 percent to within rounding.

Note what that agreement implies: the quoted denominator only works under the
additive composition. Under the overlapped reading the same addition would be
87.6 percent for every profile. The independent analysis had already assumed
the composition this study measured.

**One pre-registered qualification is dissolved by the measurement.** The
freeze said adding tensor parallelism at width 8 is not a pure addition,
because it shards attention and MLP weights and lowers per-rank compute. In
this model that correction is exactly zero. The compute term is
`max(provider compute, launch demand)`, the launch demand is 356.09 us (graph)
or 1,340.53 us (eager), and the provider compute is at most 99.59 us, so any
reduction in provider compute changes the step by nothing. The launch count
itself does not move either, because the enumeration that produced 440 to 567
explicitly excludes collective launches. The addition really is pure.

**One pre-registered qualification stands.** Weight traffic is a load-time
cost, not a per-step cost, so naming it in a per-step budget overstates the
inventory. For a per-step budget the missing traffic reduces to the
tensor-parallel all-reduces, plus KV movement under disaggregation, which this
configuration does not model.

### Is traffic coverage the largest remaining error? Partly.

It is the largest **missing** term, and its magnitude is arithmetically forced
once the collective count is 48 and the width is 8. It is not the largest
remaining **error**, because two others are of the same size or larger:

1. **The composition charges every artifact serially.** `SerialStepLowerer`
   sums artifact services in graph order, so the modeled step has exactly zero
   compute and communication overlap. A real MoE engine overlaps at least a
   layer's combine with the next layer's pre-dispatch compute. Hiding half of
   the 48 collective floors would remove 723,072,696 ps, i.e. **37.72 percent**
   of the graph composed step and **24.92 percent** of the eager one, in the
   opposite direction to the traffic-coverage addition. Hiding all of them
   would remove 1.446145 ms, exactly cancelling the tensor-parallel addition.
2. **The dominant term is a transfer.** The collective floor is 75.45 percent of
   the graph composed step and it comes from an intra-node NVLink all-reduce
   applied to a cross-node all-to-allv. A factor of two either way in that
   transfer moves the composed step by 0.72 to 1.45 ms, which is at least as
   large as adding every missing tensor-parallel all-reduce.

The honest summary is that the composed step now carries three roughly equal
1.4 ms uncertainties: one missing term, one modeling choice about overlap, and
one transfer whose target operation differs from its source. Ranking them
without new evidence is not supported, and this record does not rank them.

## Defect decision: no ID registered

The frozen decision rule is applied literally.

The measurement shows the additive composition. Under the rule that is **not**
a defect, and the reason is the registered provenance of the launch count:
`examples/compute_fidelity_v1` enumerated 440 to 567 launches per decode step
for this exact geometry and states in its own words that tensor-parallel
all-reduces are excluded "because the reference configuration puts that
collective on the fabric backend, which the network model already prices". A
launch demand built only from compute launches may overlap only compute, and
charging the collective service outside that overlap is consistent with the
count's own scope. No host term is charged more than once per step, no base
latency is charged more than once per semantic collective, and both disabled
paths reproduce their accepted baselines. **COMP-32, TRAF-32 and CORE-50 remain
unused.**

What remains open is a modelling question rather than a defect, and it is
recorded here as prose because no acceptance clause of any open task covers it:
nothing in the repository states whether the host may run ahead across a
collective boundary. The additive form assumes it may not, which is a
defensible assumption for MoE dispatch (the token histogram forces a
device-to-host synchronization before the all-to-all) and a weaker one for
tensor-parallel all-reduces. The choice is worth 0.257 ms at the graph profile
and 1.242 ms at the eager profile, i.e. up to 43 percent of the composed step.

## Evidence classes, kept separate

Counts from different classes are not summed.

- **Fatal-unscored guards:** 10 declared, 10 held in attempt two, run not void.
  Attempt one violated G10 and is void with findings; no pass fraction from it
  is interpretable and it closes nothing.
- **Scored behavioral relations:** **3 of 3** families, F1, F2 and F3. Genuine
  risk denominator 3, no family reduced by the evaluability rule. Instance
  counts are reported per family and never added to the family count: 93 decode
  steps for F1, 31 matched pairs for F2, 31 matched pairs for F3.
- **Exact-unscored relations:** E1, E2 and E3 pass; their predicates are
  entailed by registered guards.
- **Post-specified diagnostics:** rate cancellation, cross-cell exactness and
  the implied decode rate. Reported, never scored.
- **Native regression executables:** the branch's pytest additions, reported
  under validation and never added to the behavioral count.

Correlation disclosure: F1 and F2 both test the composition and are not
independent. F2 remains a separate family because it is exact where F1 is
interval valued, and because it cancels the network term that F1's intervals
have to accommodate. A reader who prefers to count the composition once should
read the denominator as 2 of 2, F1 and F2 merged plus F3, which is also a clean
pass.

## Closure scope

This study closes nothing. COMP-2, TRAF-11 and COMP-11 are already closed on
main, and no open task registers a clause about the composition of the two.
Zero new IDs are registered, which the freeze named as the expected outcome.

## Contradiction sweep

Reported rather than edited, except in the two owning module docs, which this
change reconciles.

1. `examples/host_step_cost_v1/RESULTS.md` says "Whether the collective-bearing
   service overlaps launch demand or composes additively remains unresolved, so
   no combined host-plus-collective magnitude is claimed here." That was true
   when written and is now superseded by measurement. It is the accepted record
   of what that study knew and is left standing.
2. `examples/collective_latency_floor_v1/RESULTS.md` says "Neither branch
   resolved or measured whether these terms compose additively or by whole-step
   overlap, so this report leaves that choice unresolved", and gives a
   1.651145392 ms whole-step figure computed by adding the floor to main's
   rounded 0.205 ms literal. The measured composed step is 1.916754 ms, so that
   arithmetic point understates the graph-profile composition by 13.9 percent
   and does not describe any profile.
3. `docs/README_PRO.md` carries the same "unresolved" wording in its
   `collective_latency_floor_v1` and `host_step_cost_v1` study rows. This change
   adds the new study's row and leaves the historical rows as the record of
   what those studies established.
4. `README.md:17` still says SimLLM "predicts the serving performance ... before
   you buy or reserve the hardware". This study strengthens the mission
   study's existing objection rather than removing it: the composed step is now
   inside a plausible band, but 94 to 96 percent of it is transferred constants
   whose source hardware is neither the modeled B100 nor the modeled fabric.

## Validation and storage

| gate | outcome |
|---|---|
| `.venv/bin/ruff check .` | all checks passed |
| `.venv/bin/pytest -q` | 1,571 passed, 7 skipped |
| `python3 scripts/check_docs_format.py` | 10 module docs match; 27 untagged legacy entries noted, unrelated |
| `python3 scripts/task_progress.py --check` | generated block and module-status open counts current |

Both retained run directories live outside Git under the branch-local run
root and occupy 85 MB each, 170 MB in total, with the largest single file well
under a megabyte. No deletion command was used at any point.

## Reproduction

```
python examples/composed_step_budget_v1/check_only.py \
    --cache-dir <hugging-face-cache-root> \
    --htsim-rnic <path-to-htsim_rnic>

python examples/composed_step_budget_v1/run_study.py \
    --cache-dir <hugging-face-cache-root> \
    --htsim-rnic <path-to-htsim_rnic> \
    --run-dir <writable-output-directory>
```

`--check-only` validates every frozen input and every frozen arithmetic fact and
creates nothing. `--internal analyze` re-evaluates the frozen guards and
relations over an existing run directory without touching a cell artifact.
Capture and replay need the environment carrying torch, transformers and vLLM
0.26.0, with `PYTHONPATH` at this worktree and `SIMLLM_TXT2BIN` selecting the
GOAL text-to-binary converter.
