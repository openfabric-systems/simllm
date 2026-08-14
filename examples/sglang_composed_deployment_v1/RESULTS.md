# Results: the composed realistic-deployment SGLang study

Frozen by expectations-only commit `dd026c0`, which landed before the driver
existed and before any cell ran. Every band, relation, guard and entailment
answer below is quoted from that commit. One relation failed and is reported as
failed.

## Outcome in one paragraph

Eighteen live cells, the same four requests through the same real SGLang
`Scheduler` and `RadixCache` every time, priced as two declared deployments
under three collective arms and two host arms. All eleven fatal guards held, so
the run is not void. Both scored exact relations passed, and five of the six
scored behavioral relations passed; the sixth failed on exactly the half the
freeze had already named as its risky half. The headline is a bracket, not a
number: in the ten cells whose arm charges a nonzero surcharge, **70.6 to 95.3
percent of the median step is one per-collective constant that was never
measured on this chain**, and per individual step that share runs from 57.6 to
95.3 percent. The other eight cells charge nothing for it, and two of those
eight are enabled arms rather than off arms: the intra-node `lower` arm
selects a profile whose surcharge is zero, so it moves the NVLink endpoint
rate and nothing else. Moving from
the `off` arm to the `upper` arm multiplies summed TTFT by 4.95x to 37.1x
depending on the cell, and nothing in this repository narrows that bracket.
The second finding is an ordering that flips. Under arm-name
matching the intra-node deployment looks 1.2x to 15x cheaper than the
cross-node one, but the two envelopes do not have the same arms; matched on the
per-collective constant instead, the ordering **brackets one** and inverts at
30,128,029 ps, because the corrected inventory makes the intra-node deployment
pay 72 surcharges per step against the cross-node deployment's 48. Which
deployment is cheaper is therefore undetermined by the evidence that exists.

## What ran

| stage | selection |
|---|---|
| framework | SGLang at `8f2a3ad6d7d68c58ae65b61a75bb2115449addca`, verified against the source tree's git HEAD |
| driver | `SglangSchedulerPump`, one unrolled `event_loop_normal` body per step, in the process that called `install()` and `configure(step_sink=...)` |
| model | `ibm-granite/granite-3.0-1b-a400m-instruct` at `ffec3c35...`, CPU, float32, `tp_size=1` inside SGLang, chunked prefill disabled |
| admission | `RequestAdmissionGate` in `ARRIVAL_GATED` mode on the worker's `VirtualClock`, four requests one millisecond apart, twelve new tokens each |
| routing | the SGL-16 strict v2 SGLang framework trace, provenance `sglang` / `observed-dispatch`, projected into `RoutedMoeSupply` |
| fabric | `HtsimStepSink` on `rnic-nn-fluid`; the cross-node cells executed `htsim_rnic`, the intra-node cells executed no backend at all |
| collectives | `intra-node-fixed-cost-v1` and `cross-node-fixed-cost-provisional-v1`, one arm per cell |
| host | `select_sglang_host_model`, `ideal` and `turing-cuda-graph` at 440 launches |
| metrics | `HtsimRequestMetricReducer` with the medium partition of `MediumAttribution` |

Two declared deployments, both eight ranks. **Intra-node**: one host,
`tp_ranks == ep_ranks == (0..7)`, per-rank geometry sharded for width eight,
every segment NVLink, 24 attention allreduces and 48 MoE all-to-alls per step,
408 executed artifacts. **Cross-node**: eight hosts, `tp_ranks = (0,)`, 48 MoE
all-to-alls per step, 72 executed artifacts, at 400 and 100 Gbit/s.

Scale: 357 scheduler steps, 864 reduced token intervals, 9,312 `htsim_rnic`
invocations, 112 MB of retained artifacts outside the repository, 4,120 seconds
of summed process time. Disclosure: the cells were executed six at a time as
independent child processes rather than one at a time. Each owns its work
directory, its own scheduler and a deterministic backend, so only the `wall`
column is affected by that concurrency and no simulated quantity is.

| cell | steps | max batch | upper median step, us | htsim runs | routed bytes | wall, s |
|---|---|---|---|---|---|---|
| `intra-off-ideal` | 48 | 1 | 76.686 | 0 | 87,994,368 | 110 |
| `intra-lower-ideal` | 46 | 2 | 83.382 | 0 | 87,994,368 | 107 |
| `intra-upper-ideal` | 14 | 4 | 2276.460 | 0 | 87,994,368 | 36 |
| `intra-off-turing` | 21 | 4 | 359.937 | 0 | 87,994,368 | 53 |
| `intra-lower-turing` | 20 | 4 | 387.359 | 0 | 87,994,368 | 49 |
| `intra-upper-turing` | 14 | 4 | 2557.219 | 0 | 87,994,368 | 36 |
| `cross400-off-ideal` | 26 | 3 | 214.371 | 1248 | 35,696,640 | 426 |
| `cross400-lower-ideal` | 14 | 4 | 1679.131 | 672 | 35,696,640 | 276 |
| `cross400-upper-ideal` | 14 | 4 | 2608.399 | 672 | 35,696,640 | 268 |
| `cross400-off-turing` | 19 | 4 | 489.041 | 912 | 35,696,640 | 349 |
| `cross400-lower-turing` | 14 | 4 | 1935.842 | 672 | 35,696,640 | 275 |
| `cross400-upper-turing` | 14 | 4 | 2865.110 | 672 | 35,696,640 | 268 |
| `cross100-off-ideal` | 21 | 4 | 308.969 | 1008 | 35,696,640 | 425 |
| `cross100-lower-ideal` | 14 | 4 | 1791.935 | 672 | 35,696,640 | 279 |
| `cross100-upper-ideal` | 14 | 4 | 2721.203 | 672 | 35,696,640 | 279 |
| `cross100-off-turing` | 17 | 4 | 602.172 | 816 | 35,696,640 | 341 |
| `cross100-lower-turing` | 14 | 4 | 2048.646 | 672 | 35,696,640 | 280 |
| `cross100-upper-turing` | 13 | 4 | 2977.586 | 624 | 35,696,640 | 261 |

## Fatal guards: 11 declared, 11 held, run not void

Guards are never reported as a fraction. None was declared survivable and none
was violated.

| guard | evidence |
|---|---|
| **G1** provenance | source HEAD `8f2a3ad6...`, trace `framework=sglang`, `routing_source=observed-dispatch`, model revision `ffec3c35...`, observed source equal to the source tree HEAD |
| **G2** identity | in all 18 cells the worker's sink is this driver's object, worker `SimTpModelWorker`, tree cache `RadixCache`, and batch runs equal step records equal locality outcomes equal network outcomes. No step took the adapter fallback |
| **G3** shape | no retraction, no prefill row below the whole prompt, every scheduled row sampled |
| **G4** conservation | 864 intervals, zero makespan, interval, artifact-partition, medium-projection or compute-service failures, every TTFT strictly positive |
| **G5** locality | intra-node cells: 0 fabric bytes, 0 backend runs, positive NVLink bytes. Cross-node cells: 0 NVLink bytes, positive fabric bytes, `captured` routing, epoch 0, quiescent. 9,312 backend runs, all cross-node |
| **G6** completion | every request in every cell finished for reason `length` with exactly 12 output tokens and 12 reduced intervals |
| **G7** inventory | intra-node: site set `("attention",)` alone, 24 allreduces, 48 all-to-alls, 408 artifacts, on every step. Cross-node: 0 allreduces, 48 all-to-alls, 72 artifacts, on every step |
| **G8** envelope admissibility | what the guard evaluates: every summed base equals its collective count times its arm constant exactly, and every nonzero per-artifact base equals that constant. The payload-envelope half is enforced upstream instead, by `validate_endpoint_bytes` aborting a cell, so it can void a run but never appears here as a checked clause. Largest observed step: 24 new tokens |
| **G9** host agreement | worker and sink selected the same host model in all 18 cells; every `ideal` cell reports launch count 0 and zero exposed host time; every `turing` cell reports 440 launches, a 356,094,640 ps floor, device key `gtx1660-ti-sm75`, compute pinned to `b100`, `356,095,000` ps of compute service on every step, and carries the transfer disclosure verbatim |
| **G10** BACK-44 control | executed, not recalled. `tp_ranks=(0, 1)` with `ep_ranks=(0, 1, 2, 3)` was refused with `graph cannot be represented by ordered GOAL artifacts: 'step-0:layer-0:tp-attention' does not depend on 'step-0:layer-0:rank-2:compute'` |
| **G11** byte conservation | 87,994,368 bytes in every one of the six intra-node cells and 35,696,640 in every one of the twelve cross-node cells, both equal to the values predicted in the freeze |

G8 deserves one sentence of honesty. The freeze predicted that the largest
representable step, four co-scheduled eight-token prefills, would put the
critical endpoint load at exactly 458,752 bytes, the profile envelope's own
maximum, so the matrix was admissible with zero margin. The scheduler never
formed that batch: the largest observed step carried 24 new tokens, so the
largest endpoint load actually reached was 344,064 bytes and the zero-margin
edge was never tested. The prediction stands as written and untested rather
than as confirmed.

## Scored exact relations: 2 of 2

> **E1.** In all 18 cells an independent standard-library recomputation of
> per-request TTFT, TPOT and the medium components, from the per-step rows and
> the declared arrivals alone, agrees exactly with `HtsimRequestMetricReducer`.

**Passes.** 72 request rows compared, zero mismatches. The independent side
re-derives the ownership rule from the per-artifact `composed`, `fabric`,
`local`, `base` and medium arrays rather than calling it, so a service assigned
to the wrong medium inside a conserving total would be visible. The
intra-node cells are the first live chain in this repository to drive
`nvlink_ps` and `collective_base_ps` through the reducer. As the freeze
declared in advance, `co_critical_ps` and `control_ps` are
configuration-forced zeros under G2, G4 and G5, so E1 scores five reachable
components and not seven.

> **E2.** The `cross400-off-ideal` cell reproduces the accepted
> `sglang_end_to_end_v1` `ep8-400g` cell to the published precision.

**Passes exactly.** 26 scheduler steps, TTFT 270.04869, 358.38381, 483.59556
and 421.37623 us, TPOT 262.0387, 268.70013, 244.40511 and 214.9525 us. The
accepted study published 270.05, 358.38, 483.60, 421.38 and 262.04, 268.70,
244.41, 214.95. Every digit agrees. This matters because four things differ
from the accepted run: a different trace file carrying the same routing rows, a
declared eight-host placement manifest where the accepted run declared none,
the host model built through the w14d selection seam instead of by hand, and
this study's own driver loop. None of them moved a timestamp, which is what
makes every other cell interpretable as a change from a known baseline.

## Scored behavioral relations: 5 of 6

> **B1.** Within each of the six (topology, link, host) families the scheduler
> step count is non-increasing across `off`, `lower`, `upper`, and is strictly
> smaller at `upper` than at `off` in all six.

**Passes.** 48, 46, 14 for `intra-ideal`; 21, 20, 14 for `intra-turing`;
26, 14, 14 for `cross400-ideal`; 19, 14, 14 for `cross400-turing`; 21, 14, 14
for `cross100-ideal`; 17, 14, 13 for `cross100-turing`. This is the relation
that separates a live closed loop from a replay: adding a per-collective
constant that neither the scheduler nor the workload knows about changed the
number of batches SGLang itself chose to run, by up to 3.4x.

> **B2.** The largest co-scheduled batch is non-decreasing across the arms in
> all six families and equals 4 at the `upper` arm in all six.

**Passes.** The `intra-ideal` family is the clean demonstration: 1, 2, 4. At
the `off` arm an intra-node step costs 76.7 us, so a request finishes its
twelve tokens in under a millisecond and is served entirely alone before the
next arrival; at the `upper` arm a step costs 2.28 ms, so every request has
arrived before the second batch is formed and all four are co-scheduled. The
same constant that inflates the step also changes what the framework batches.

> **B3.** Within each of the nine (topology, link, collective arm) families the
> `turing` cell takes no more steps than the `ideal` cell, strictly fewer in at
> least one.

**Passes**, strictly in five of the nine families: `intra-off` 21 against 48,
`intra-lower` 20 against 46, `cross400-off` 19 against 26, `cross100-off` 17
against 21, and `cross100-upper` 13 against 14. The four families where it is
equal are the ones where the step was already far longer than the arrival
spacing, so the extra 257 us of host cost had nothing left to change.

> **B4.** For the cross-node cell at each arm, summed TTFT at 100 Gbit/s over
> summed TTFT at 400 Gbit/s exceeds one, the three-arm envelope does not
> bracket one, and the ratio is non-increasing as the arm constant grows.

**Fails**, on the third clause, under the `turing` host only.

| host | arm | live TTFT ratio | per-step median ratio | steps, 100G against 400G |
|---|---|---|---|---|
| ideal | off | 1.5922 | 1.4413 | 21 against 26 |
| ideal | lower | 1.2596 | 1.0672 | 14 against 14 |
| ideal | upper | 1.1622 | 1.0432 | 14 against 14 |
| turing | off | 1.0322 | 1.2313 | 17 against 19 |
| turing | lower | 1.0505 | 1.0583 | 14 against 14 |
| turing | upper | 1.0150 | 1.0393 | 13 against 14 |

Both clauses that survive are clean: every ratio exceeds one, and neither
envelope brackets one, so 100 Gbit/s is slower under every arm. The
non-increasing clause holds under `ideal` (1.5922, 1.2596, 1.1622) and breaks
under `turing`, where `off` sits at 1.0322 below `lower` at 1.0505.

The cause is the mechanism the freeze named as this relation's risky half, and
it is worth stating precisely rather than absorbing. **Root cause:** the two
link rates do not produce the same step sequence, and the realized TTFT ratio
is therefore taken over different batchings. At the `turing` off arm the
100 Gbit/s cell runs 17 steps against the 400 Gbit/s cell's 19, so the slower
fabric batches more and pays less queueing, and that saving cancels most of
the per-step penalty without reversing it. In summed TTFT the fabric-owned
term grows by 905.4 us, from 685.8 to 1591.2, while the queue term falls by
793.8 us, from 1356.2 to 562.4, and the kernel term is identical at 1424.4
because the launch floor binds at both rates; the net is plus 111.6 us on a
3466.4 us base. **Effect:** the live ratio drops to 1.0322 while the
per-step median ratio, which the closed form governs, is 1.2313 and is
non-increasing in the arm constant under both hosts (1.4413, 1.0672, 1.0432 and
1.2313, 1.0583, 1.0393). **Reading:** the closed-form prediction is correct
about the per-step quantity and wrong about the live one, and the relation was
written on the live one. It is scored as a failure. Nothing was changed to make
it pass, and the per-step diagnostic is reported as a post-specified
diagnostic, not as a substitute for the relation.

> **B5.** The intra-node over cross400 envelope of summed TTFT, at matched host
> arm, does not bracket one under arm-name matching and does bracket one under
> constant matching, with the 30,128,029 ps pairing above one.

**Passes, on both hosts.** This is the study's second headline.

| host | pairing | arm ratios | envelope | brackets one |
|---|---|---|---|---|
| ideal | arm name | off 0.2226, lower 0.0673, upper 0.8223 | [0.0673, 0.8223] | no |
| ideal | constant | 0 ps 0.2226, 30,128,029 ps 1.3187 | [0.2226, 1.3187] | yes |
| turing | arm name | off 0.5660, lower 0.1891, upper 0.8541 | [0.1891, 0.8541] | no |
| turing | constant | 0 ps 0.5660, 30,128,029 ps 1.2511 | [0.5660, 1.2511] | yes |

Read by arm name, the intra-node deployment is cheaper at every arm and the
answer looks settled. It is not, and the arm-name reading is an artifact: the
two envelopes do not have the same arms. `intra lower` charges no surcharge at
all while `cross lower` charges the full 30,128,029 ps intercept, so the 0.0673
cell compares a zero constant against a large one. Matched on the constant
instead, the intra-node deployment is 4.5x cheaper on the ideal host and 1.8x cheaper
on the turing host at a zero constant, and 1.32x and 1.25x **more expensive**
respectively at 30,128,029 ps, because the corrected inventory
makes it pay 72 surcharges per step, 24 allreduces plus 48 all-to-alls, against
the cross-node deployment's 48. **The ordering of the two deployments is
undetermined by the evidence that exists**: it flips inside the range of
per-collective constants this repository can defend, and nothing here fixes
that constant for a pairwise ALL-TO-ALLV.

One confound in the constant-matched pairing has to be disclosed rather than
left implicit. Its zero-constant point runs the intra-node cell at the
declared 450 GB/s NVLink rate while its 30,128,029 ps point runs it at the
profile's fitted 70.027 GB/s, so the rate moves with the constant. The rate
effect is small beside the constant it travels with: it adds 53,694,000 ps to
the intra-node step, 2.48 percent of the 2,169,218,088 ps surcharge. Repeating
the pairing bandwidth-matched, using `intra-lower` at 70.027 GB/s for the zero
constant, gives 0.4213 on the ideal host and 0.6502 on turing against the same
1.3187 and 1.2511, so both brackets still contain one and the conclusion is
robust to the confound. TRAF-36 hardware, a real cross-node
per-collective fixed-cost capture at widths 2, 4 and 8 for both ring
ALL-REDUCE and pairwise ALL-TO-ALLV, is what would settle it.

> **B6.** For `p1`, `p2` and `p3`, enabling both the `upper` collective arm and
> the `turing` host arm multiplies TTFT by at least 2 against the `off`/`ideal`
> cell of the same topology and link, and the multiplier is larger intra-node
> than at cross400.

**Passes.** Intra-node multipliers 49.63, 37.98 and 56.72; cross400 13.62, 8.03
and 13.73; cross100 8.46, 6.68 and 7.36. `p0`'s multipliers, 30.35, 10.75 and
6.31, are reported as anchored rather than scored because `p0`'s TTFT is
exactly the first step's latency and follows from the closed form.

## Physical sanity, checked against first principles before the digits

Three independent framings, per the physical-sanity rule.

**Compute and memory physics.** Weight bytes over memory bandwidth is a floor
no decode step can beat. The cross-node deployment's per-rank resident bytes
are 554,047,488, which is 69.256 us at the b100 envelope's 8.0e12 B/s and
98.937 us after the provider's 0.7 derate; measured 98.928 us, which is 9,051 ps
**below** the derated estimate rather than equal to it, for the reason set out
in the deviation paragraph below. The intra-node deployment's width-eight shard
leaves 421,582,848 bytes, so 52.698 us and 75.283 us; measured 75.264 us, again
low by 18,651 ps. Neither shortfall touches the floor being checked, since both
measured values still exceed the peak-bandwidth floors of 69.256 and 52.698 us.
The difference between the two, 132,464,640 bytes, is exactly the
132,120,576 bytes of tensor-parallel attention weight the
shard removes plus the 344,064 bytes of KV a narrower head count removes, with
no residual, and the measured compute ratio 98,928,000 over 75,264,000 is
1.31442 against the byte ratio 1.31421, agreeing to 0.02 percent. Nothing in
the model contributed a dense-MLP term, which is correct for a model whose 24
layers are all MoE.

**Network and serialization physics.** Bytes over link rate is a floor no flow
can beat, and propagation is a floor no message can beat. The first prefill
step moves 3,756,032 directed bytes over 48 collectives. At 400 Gbit/s that is
75,120,640 ps of bottleneck serialization plus 48 propagation delays of
2,000,000 ps each, so 171,120,640 ps; measured 171,120,688 ps, 48 ps high over
48 artifacts, one picosecond per artifact of ceiling. At 100 Gbit/s the
serialization term must scale by exactly four and the propagation term must not
move: measured 396,482,608 ps, whose serialization part 300,482,608 ps is
4.0000 times the 400 Gbit/s part and whose propagation part is unchanged at
96,000,000 ps. On NVLink the same step's local service is 10,052,000 ps at the
declared 450 GB/s and 63,746,000 ps at the profile's 70.027 GB/s, a factor of
6.342 against a rate ratio of 6.426, the gap being the whole-nanosecond
enclosure of 384 phase services. Every composed step reproduces the additive
form `compute + local + fabric + surcharge` to the picosecond once its compute
term is taken as measured: `intra-upper-ideal` step zero is
`75,264,000 + 63,746,000 + 2,169,218,088 = 2,308,228,088` and
`cross400-upper-turing` step zero is
`356,095,000 + 171,120,688 + 2,375,413,872 = 2,902,629,560`, both exact. Only
the second of those two derives its compute term from the frozen closed form;
the first substitutes the measured value, and why is the subject of the next
paragraph.

**Post-specified deviation: the frozen host closed form is falsified on the
ideal cells.** The freeze wrote the represented compute term as
`C = 1000 * ceil(max(C_provider, N * g) / 1000)` picoseconds
(`expectations.md`, "Host term"). That holds exactly on the six calibrated
cells and is falsified on the twelve ideal ones. The whole-nanosecond top-up
that the form describes lives inside the `if cfg.host_model.is_calibrated:`
branch at `simllm/backends/step_lowerer.py:247-262`; with an ideal host model
that branch never runs, so the represented term is instead the sum of 24
per-layer cumulative-boundary floors from `_to_goal_layer_calc_ns`, which
truncates rather than rounds up. Every ideal cell therefore lands **below** its
own provider estimate: 98,928,000 ps against 98,937,051 ps cross-node, short by
9,051 ps, and 75,264,000 ps against 75,282,651 ps intra-node, short by
18,651 ps. Both shortfalls sit inside the 23,976 ps that 24 whole-nanosecond
truncations can lose, which is what identifies the mechanism. The frozen form
would have predicted 98,938,000 and 75,283,000 instead. The calibrated identity
is untouched and is the demonstration that the form is right where it applies:
`cross400-upper-turing` step zero composes as
`1000 * ceil(356,094,640 / 1000) + 171,120,688 + 2,375,413,872 = 2,902,629,560`,
exact. This is a post-specified finding, not a pre-registered one, and it is
recorded as a falsification of the freeze rather than corrected in it. The
behavior is inherited from `main` and not introduced here: the accepted
`sglang_end_to_end_v1` run reports the same 98.9 us for the same geometry under
the same ideal host, so no number in that study or this one moves, and what was
wrong was the freeze's description of the mechanism rather than the mechanism.

**End-to-end plausibility against a real serve.** A real deployment of a
400M-active-parameter MoE on one accelerator decodes a single request at
roughly `10^2` tokens per second, because the fixed host cost per decode step
dominates at this model size. The implied `1 / TPOT` here runs:

| cell | tokens per second | roughly, against the `10^2` anchor |
|---|---|---|
| `intra-off-ideal` | 13,035 to 13,045 | 130x optimistic |
| `cross400-off-ideal` | 3,722 to 4,652 | 37x to 47x optimistic |
| `cross100-off-ideal` | 2,314 to 3,208 | 23x to 32x optimistic |
| `intra-off-turing` | 2,175 to 2,778 | 22x to 28x optimistic |
| `cross400-lower-ideal` | 501 to 596 | 5.0x to 6.0x optimistic |
| `intra-upper-ideal` | 370 to 439 | 3.7x to 4.4x optimistic |
| `intra-upper-turing` | 330 to 391 | 3.3x to 3.9x optimistic |
| `cross400-upper-turing` | 294 to 349 | 2.9x to 3.5x optimistic |
| `cross100-upper-turing` | 302 to 336 | 3.0x to 3.4x optimistic |

Enabling both mechanisms moves the chain from two orders of magnitude
optimistic to within half an order. That is worth saying and it is **not**
evidence of accuracy. The constants that closed the gap are a consumer Turing
GPU's launch cost, a launch count enumerated from vLLM sources, and an NVLink
ALL-REDUCE intercept charged to a cross-node ALL-TO-ALLV. A simulator that
reaches a plausible number through three transferred constants has not been
validated; it has been made plausible. The `off`/`ideal` corner, which is what
every earlier SGLang study in this repository reported, remains 40x to 130x
optimistic and that is the correct reading of those earlier numbers too.

## Per-request metrics

`cross400`, ideal host, times in microseconds:

| cell | TTFT p0 | p1 | p2 | p3 | TPOT p0 | p1 | p2 | p3 |
|---|---|---|---|---|---|---|---|---|
| `cross400-off-ideal` | 270.05 | 358.38 | 483.60 | 421.38 | 262.04 | 268.70 | 244.41 | 214.95 |
| `cross400-lower-ideal` | 1716.19 | 2434.11 | 3225.09 | 2225.09 | 1997.95 | 1841.78 | 1678.96 | 1678.96 |
| `cross400-upper-ideal` | 2645.46 | 4368.58 | 3368.58 | 5012.89 | 3096.18 | 2848.62 | 2848.62 | 2608.23 |

`intra`, ideal host:

| cell | TTFT p0 | p1 | p2 | p3 | TPOT p0 | p1 | p2 | p3 |
|---|---|---|---|---|---|---|---|---|
| `intra-off-ideal` | 85.32 | 85.50 | 85.40 | 85.19 | 76.67 | 76.66 | 76.68 | 76.72 |
| `intra-lower-ideal` | 139.01 | 194.98 | 166.28 | 145.75 | 83.25 | 96.61 | 97.34 | 84.30 |
| `intra-upper-ideal` | 2308.23 | 3681.61 | 2681.61 | 3989.02 | 2701.88 | 2486.12 | 2486.12 | 2276.35 |

Every published band held: all 18 cells reported every step latency inside its
frozen `[floor, ceiling]`, every TTFT inside `[floor, 5 x ceiling]` and every
TPOT inside `[floor, 1.5 x ceiling]`.

The medium partition is what the composition was for. Summed over the four
requests, in microseconds, decode intervals:

| cell | queue | kernel | nvlink | fabric | collective base |
|---|---|---|---|---|---|
| `intra-off-ideal` | 0.0 | 3312.3 | 61.6 | 0.0 | 0.0 |
| `intra-lower-ideal` | 277.7 | 3312.3 | 386.4 | 0.0 | 0.0 |
| `intra-upper-ideal` | 9295.6 | 3314.2 | 1399.8 | 0.0 | 95445.6 |
| `cross400-off-ideal` | 1351.1 | 4362.1 | 0.0 | 5177.8 | 0.0 |
| `cross400-lower-ideal` | 5299.9 | 4371.3 | 0.0 | 5872.6 | 63630.4 |
| `cross400-upper-ideal` | 10656.1 | 4371.3 | 0.0 | 5872.6 | 104518.2 |
| `cross400-upper-turing` | 11684.7 | 15668.2 | 0.0 | 5872.6 | 104518.2 |

The intra-node rows carry `nvlink` and no `fabric`, the cross-node rows carry
`fabric` and no `nvlink`, and every row sums to its own decode span with the
queue column. `co_critical_ps` is zero everywhere, which is forced: an
intra-node artifact has zero fabric service and an all-remote artifact has zero
NVLink service, so the two media never tie above zero in this matrix.

## What is measured, what is captured, what is transferred

| term | class | where it came from |
|---|---|---|
| batching, radix behavior, expert routing, sampled counts | executed | the real SGLang scheduler at the pinned commit, in process |
| fabric propagation, serialization and contention | simulated | 9,312 `htsim_rnic` packet-level runs on `rnic-nn-fluid` |
| per-step compute service | modeled | b100 envelope memory bandwidth against the pinned model's resident bytes; no silicon measurement of this model on any device |
| NVLink endpoint serialization | modeled, uncalibrated | a declared flat per-endpoint rate, or the profile's fitted slope; TRAF-31 owns the missing point-to-point capture |
| 24 intra-node ALL-REDUCE surcharges per step, `upper` arm | captured at its own operation, interconnect and width | 723,072,696 ps per step from the nccl-tests issue 333 DGX B200 capture, at the same operation, interconnect and participant width |
| 48 intra-node ALL-TO-ALLV surcharges per step, `upper` arm | transferred at use | 1,446,145,392 ps per step, the same ALL-REDUCE intercept charged to a different collective |
| 48 cross-node surcharges, `lower` arm | transferred at use | 1,446,145,392 ps per step, an intra-node NVLink capture charged across a fabric |
| 48 cross-node surcharges, `upper` arm | provisional transferred | 2,375,413,872 ps per step, that capture plus RDMA anchors; no cross-node measurement exists, TRAF-36 owns it |
| per-step host cost, `turing` arm | transferred, three-source hybrid | 356,094,640 ps: a GTX 1660 Ti launch point, a vLLM 0.26.0 launch count, and b100 compute. SGL-24 owns the SGLang count |
| arrivals | declared | this study's own one-millisecond spacing |

The surcharge share of the upper median step, which is the number the headline
rests on:

| cell | surcharge per step, ps | share of upper median step | published evidence class |
|---|---|---|---|
| `intra-upper-ideal` | 2,169,218,088 | 95.29 percent | transferred-at-use |
| `cross400-upper-ideal` | 2,375,413,872 | 91.07 percent | provisional-transferred |
| `cross100-upper-ideal` | 2,375,413,872 | 87.29 percent | provisional-transferred |
| `cross400-lower-ideal` | 1,446,145,392 | 86.12 percent | transferred-at-use |
| `intra-upper-turing` | 2,169,218,088 | 84.83 percent | transferred-at-use |
| `cross400-upper-turing` | 2,375,413,872 | 82.91 percent | provisional-transferred |
| `cross100-lower-ideal` | 1,446,145,392 | 80.70 percent | transferred-at-use |
| `cross100-upper-turing` | 2,375,413,872 | 79.78 percent | provisional-transferred |
| `cross400-lower-turing` | 1,446,145,392 | 74.70 percent | transferred-at-use |
| `cross100-lower-turing` | 1,446,145,392 | 70.59 percent | transferred-at-use |
| `intra-lower-ideal` | 0 | 0 percent | structural-floor |
| `intra-lower-turing` | 0 | 0 percent | structural-floor |
| the six `off` cells | 0 | 0 percent | none |

All 18 cells are in that table. Two of them, the intra-node `lower` cells, are
**enabled arms that charge nothing**: the arm resolves to
`collective-fixed-cost-floor-v1`, whose surcharge is zero at every width, so
selecting it moves the NVLink endpoint rate and leaves the fixed cost alone.
That is why the headline is stated over the ten nonzero-surcharge cells rather
than over the twelve enabled ones. The share column is taken on the upper
median step of each cell; per individual step the same ten cells run from 57.64
percent, at `cross100-lower-turing`'s cheapest step, to 95.31 percent, at
`intra-upper-ideal`'s most expensive.

One row of that table is finer than the class it publishes.
`intra-upper-ideal` is 95.29 percent surcharge, and one third of that surcharge,
723,072,696 ps, is charged at the capture's own operation, interconnect and
participant width, with the payload inside the fitted envelope. The remaining
1,446,145,392 ps is transferred to a different collective. The envelope
publishes one evidence class per arm, so the record says `transferred-at-use`
for the whole cell and the split had to be derived by hand from the inventory.
That is registered as TRAF-42 rather than left in prose, and it is the record
surface rather than the measurement: TRAF-39 owns capturing an ALL-TO-ALLV
intercept of its own. The at-capture third still carries one device transfer
that no field records either: the capture is a DGX B200 while compute is
priced against the b100 envelope. That third is nonetheless the first
surcharge in this repository charged to the operation, interconnect and
participant width it was measured on, with its payload inside the fitted
envelope: the earlier envelope study reported zero NVLink bytes in every
cell and priced pairwise ALL-TO-ALLV only.

## Refuted premises

Items 1 to 3 were found before the run and are recorded in the freeze. Item 4
is a post-run finding and is labeled as one: the probe data that shows it
existed before the freeze, but the defect was recognized only while reading
the run record, so it is not pre-registered and is not scored.

1. **The intra-node cell drives `htsim_rnic` per step.** It does not. With all
   eight ranks on one host every directed segment is local and the sink
   executes no backend process at all: 0 of the 9,312 runs came from an
   intra-node cell. The cells' `linkspeed_bps` is inert, which is why the
   intra-node deployment carries no link-rate axis.
2. **The arms are a pure fixed-cost bracket.** They are on the cross-node cell,
   which has no local segments. They are not on the intra-node cell: selecting
   any profile also replaces the declared 450,000,000,000 B/s NVLink endpoint
   rate with the profile's fitted 70,027,079,100 B/s
   (`simllm/backends/step_sink.py:809-813`), so the intra-node `lower` arm
   charges no surcharge and still slows every endpoint. Measured: the same
   step's NVLink service is 10,052,000 ps at `off` and 63,746,000 ps at
   `lower`. This study ran all three intra-node arms precisely so the bandwidth
   change and the surcharge change are separated rather than confounded.
3. **BACK-44 is real and both cells avoid it.** The refusal was executed as
   G10, not recalled. It fires when the tensor-parallel group is a strict
   subset of the expert-parallel group; the intra-node cell sets the two equal
   and the cross-node cell has no tensor-parallel collective at all. The
   canonical realistic composition, a tensor-parallel group inside a node with
   an expert-parallel group across nodes, remains unrepresentable and remains
   BACK-44's. Nothing here closes or weakens it.
4. **The intra-node envelope's own claim string is false on the topology it is
   named for.** `INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE.claim` says its arms
   "run from the modeled propagation delay, which is a floor no collective can
   beat", `COLLECTIVE_FIXED_COST_FLOOR_PROFILE`'s transfer text says "the
   claimed per-collective fixed cost is exactly the 2.000 us propagation the
   backend already charges", and `docs/modules/traffic.md:147-149` repeats it.
   On an all-intra-node step no backend runs, so the backend charges no
   propagation and the lower arm's realized fixed cost is zero, not 2.000 us.
   The cells confirm it: on its first prefill step `intra-lower-ideal`
   realizes 885 ns per collective, all of it endpoint serialization, while
   the envelope publishes `realized_bracket_ps = [2,000,000, 32,128,029]`.
   Both edges of that published bracket are overstated by exactly one
   propagation reference on the intra-node path, and the `off` arm on that
   path has no physical floor at all. This is registered as TRAF-42.

## Evidence classes, kept separate

Counts from different classes are never summed.

- **Fatal-unscored guards:** 11 declared, 11 held, run not void.
- **Scored exact relations:** **2 of 2**. E1, E2.
- **Scored behavioral relations:** **5 of 6**. B1, B2, B3, B5, B6 pass; B4
  fails.
- **Reported, not scored:** the band table, the scale table, the surcharge
  shares, the per-step B4 diagnostic, the bandwidth-matched B5 pairing, `p0`'s
  B6 multipliers, the decode-rate comparison, the post-specified falsification
  of the frozen host closed form on the ideal cells, and the six relations the
  freeze removed as entailed.

### Genuine-risk analysis

**8 of the 8 scored relations are genuine risk**, with one declared narrowing.
E1's component coverage was narrowed to five of the seven medium components in
the freeze, before the run, because `co_critical_ps` and `control_ps` are
configuration-forced zeros here. Nothing else was narrowed, and the one failure
landed on a relation whose risky half the freeze had named in advance, which is
the outcome that shows the entailment questions were asked honestly rather than
answered to make the score look good.

The six relations the freeze removed **before** the run because they cannot
fail, each kept as a guard or a diagnostic:

1. TTFT and TPOT conserve against their attributions. `RequestLatencyTotals`
   and `HtsimRequestMetricReducer.consume` raise instead of disagreeing. Kept
   as G4, where it held over 864 intervals.
2. Every `turing` cell reports 356,095,000 ps of compute service on every step.
   Entailed once the resident-byte count keeps provider service under the
   launch floor. Kept as G9 and reported.
3. Each cell's summed base equals its collective count times its arm constant.
   Entailed by the sink charging the base only at ring phase zero. Kept as G8.
4. Intra-node requests carry no fabric time and cross-node requests carry no
   NVLink time. Forced by the declared manifests. Kept as G5.
5. Total directed bytes are identical across arms within a topology. Forced by
   every token being forwarded exactly once. Kept as G11 and reported with the
   exact predicted values, both of which matched.
6. The intra-node inventory is 24 plus 48 rather than 96. Forced by
   `layer_tp_allreduce_sites` under a captured supply. Kept as G7.

## What this study supports, and what it does not

**Supported.** That the four wave-14 mechanisms compose on one live chain: the
corrected allreduce inventory, the arm-selected fixed-cost envelope, the
medium-aware per-request attribution and the host-model seam all ran together
in 18 cells with exact conservation and an independent recomputation. That the
composed cost is honestly bracketed rather than asserted, with the transfer of
every constant stated at the point of use. That the fixed cost feeds back into
the framework's own batching, changing step counts by up to 3.4x and maximum
batch size from 1 to 4. That the arm-name reading of a cross-envelope
comparison is unsafe and the constant-matched reading leaves the intra-node
versus cross-node ordering undetermined.

**Not supported, explicitly.**

- **No calibration of anything.** The upper arms are provenance-transferred,
  the host term is a three-source device hybrid, and the launch count is
  vLLM's. Nothing here was measured on SGLang, on this model, or on this
  interconnect except the framework's own batching decisions.
- **No absolute TTFT or TPOT claim.** SGL-4 owns the silicon comparison; its
  offered-load, paced-mode and `launch_server` arms are untouched by this
  study, which ran four requests through the offline in-process path.
- **No closure of BACK-44, TRAF-31, TRAF-36, SGL-24 or SGL-4.**
- **SGL-26 is not closed.** This study is the first live in-process SGLang run
  to select a nonideal host profile and carry it to TTFT and TPOT, which is
  SGL-26's identifying observation, and G2 shows every step was sink-settled so
  the adapter fallback was never taken. But SGL-26 also requires settling which
  of the sink's whole-nanosecond enclosure and the adapter fallback's raw
  picoseconds is authoritative when a step carries no collective, and this
  matrix makes that path unreachable rather than resolving it. A study that
  cannot reach a clause cannot close it.
- **The declared tensor-parallel shard is not the width SGLang executed.**
  SGLang ran `tp_size=1`. The intra-node cell declares width eight and prices
  it, which is the tensor-parallel analogue of SGL-25's expert-residency gap
  and is registered as SGL-27. It is also why the two topologies are reported
  as two deployments rather than as one controlled locality experiment.
- **No NVLink contention or propagation.** The local path is a flat analytic
  per-endpoint serializer with no contention model and, at the `off` arm, no
  fixed cost at all.
- **The zero-margin endpoint prediction is untested.** The scheduler never
  formed the four-prefill batch that would have reached the profile envelope's
  exact maximum.

## Registry

- **SGL-27** registered (Precision; P1; M): the declared tensor-parallel shard
  against the width the framework executed.
- **TRAF-42** registered (Completeness; P1; M): the fixed-cost surface
  describes itself as if every collective crossed the fabric, which makes the
  intra-node envelope's published realized bracket and floor claim wrong by one
  propagation reference and makes the per-arm evidence class unable to carry
  the at-capture and transferred split of a single cell.
- **SGL-28 and SGL-29 were allocated and are unused.** No further deferral was
  discovered that an existing task does not already own.
- **Nothing is closed.** SGL-4, SGL-24, SGL-25, SGL-26, TRAF-31, TRAF-36 and
  BACK-44 all stay open, and this study's evidence for each of them is stated
  above rather than converted into a closure.

## Reproduction

```
python examples/sglang_composed_deployment_v1/run_study.py \
    --cache-dir <hugging-face-cache-root> \
    --sglang-python <interpreter-that-owns-sglang-and-torch> \
    --sglang-source <sglang-source-checkout-at-the-pinned-commit> \
    --htsim-rnic <path-to-htsim_rnic> \
    --run-dir <writable-output-directory> \
    --jobs <concurrent-child-stages>
```

The routing authority is tracked beside this script at
`examples/sglang_composed_deployment_v1/fixtures/routing-trace.jsonl` and is the
default for `--routing-trace`, so the study no longer depends on a run
directory from another wave surviving a disk cleanup. It is a byte copy of the
SGL-16 framework capture, 369,843 bytes, SHA-256
`da0096b696564f365003d11565f4cbc5bed33ab47f7236c52b3ca7c989fd4982`, which is the
digest the freeze already pins. `.gitattributes` marks that path `text eol=lf`
so a Windows checkout cannot rewrite the bytes, the driver refuses a tracked
fixture whose digest has moved, and `tests/test_sglang_composed_deployment_study.py`
asserts it. The flag still accepts any qualifying capture, whose digest is
reported in guard G1 rather than required.

`--check-only` validates every frozen input and writes nothing. Backends are
deterministic and the scheduler is seeded, so every number above reproduces
exactly. `--jobs` affects the reported wall column and nothing else. The run
reported here used `--jobs 6` on a 32-core host and took about 17 minutes of
wall clock for 4,120 seconds of summed process time.
