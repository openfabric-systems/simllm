# Composed step budget v1 expectations

This document freezes the acceptance contract before the harness that produces
a measured number exists and before any result-producing run. It is the
pre-run record required by the validation discipline.

## The question, and why arithmetic cannot answer it

Wave 12 landed two dominant terms of the mission error budget on main:

- the calibrated fixed per-step host cost, COMP-2, in `simllm/compute/host.py`;
- the calibrated collective latency floor, TRAF-11 and COMP-11, in
  `simllm/traffic/collective_latency.py`.

Neither branch produced a measured composed number, and their two published
arithmetic projections of the same merged code disagree:

| reading | rule | value for a case A decode step at 400 Gbit/s |
|---|---|---:|
| additive | `max(C, N * g) + network` | 1.907743126 ms (graph, 440) to 2.892181126 ms (eager, 567) |
| overlapped | `max(C + network, N * g)` | 1.650672126 ms for every host profile |

with `C = 99,024,000` ps of modeled B100 compute, `network = 105,502,734` ps of
raw fabric service for the accepted `a-ep8-400g` decode step 1, and
`48 * 30,128,029 = 1,446,145,392` ps of calibrated collective floor. The two
readings differ by a factor of 1.7521233202 at the eager endpoint. Both are
statements about the same merged code, so exactly one of them describes what
the repository computes. Arithmetic cannot settle that; only running the code
can.

`examples/end_to_end_replay_v1` is the only harness in the repository that
produces a composed number rather than a projection: real requests through a
live vLLM scheduler at declared arrivals, the simulated executor, captured MoE
routing on a packet-level fabric, and per-request TTFT and TPOT back out with
an exactly conserving seven-component partition. This study runs that chain
with both features enabled and with both disabled.

## What this study claims, and what it refuses to claim

It claims to establish, by measurement, which composition the merged code
computes, and to recompute the mission error budget from that measured number.

It does **not** claim that the composed number predicts the reference
deployment. The two installed constants are transfers, not measurements of the
reference system:

- the host term is a GTX 1660 Ti launch-throughput constant measured on a
  desktop CPU, with the reference B100 host cost explicitly unknown and the
  implementation refusing to transfer the constant to any other device key;
- the collective intercept is a DGX B200 intra-node NVLink **all-reduce**
  applied unchanged to cross-node pairwise **all-to-allv**, and it sits 0.4
  percent above the upper endpoint of the 0.72 to 1.44 ms band it was meant to
  land in.

The composed number therefore inherits both uncertainties, and it is a
three-device chimera: a B100 memory-bandwidth envelope for compute, a Turing
launch constant for the host term, and a B200 NVLink intercept for the
collective floor. This is stated here, before the run, so that no reader can
take the measured milliseconds for a deployment prediction.

## Frozen configuration

The oracle capture, model geometry, placement, ownership, dedup rule, arrival
gating, replay pinning and network profile are exactly those frozen in
`examples/end_to_end_replay_v1/expectations.md`. Nothing in that contract is
changed. This study varies only the two new seams and the link rate.

### The device hybrid, disclosed before the run

A calibrated host profile refuses every GPU key except `gtx1660-ti-sm75`, so a
cell that selects one must present that device key. The mission chain prices
compute against the B100 envelope. To compose the two installed constants at
all, the enabled cells therefore present the Turing device key to the host
model while a pinned provider keeps the accepted B100 roofline service. This is
the same hybrid the host-cost study used and labelled sensitivity evidence
rather than a device-consistent prediction, and it is labelled the same way
here. The disabled cell uses the accepted B100 configuration unchanged.

### Cells

Four cells. Two parameters vary, as the validation discipline requires: the
host profile and the link rate. The feature selection varies as well.

| label | source cell | host profile | launches | collective profile | link rate |
|---|---|---|---:|---|---|
| `off-400g` | `a-ep8-400g` | `ideal` | 0 | default off | 400 Gbit/s |
| `on-graph440-400g` | `a-ep8-400g` | `turing-cuda-graph` | 440 | `b200-nccl-2.27-local-v1` | 400 Gbit/s |
| `on-eager567-400g` | `a-ep8-400g` | `turing-eager-host` | 567 | `b200-nccl-2.27-local-v1` | 400 Gbit/s |
| `on-graph440-200g` | `a-ep8-200g` | `turing-cuda-graph` | 440 | `b200-nccl-2.27-local-v1` | 200 Gbit/s |

The launch counts 440 and 567 are the frozen bracket of
`examples/compute_fidelity_v1`, enumerated statically from vLLM 0.26.0 for this
exact 24-layer top-8 Granite MoE geometry. That enumeration states in its own
words that tensor-parallel all-reduces are excluded from the count "because the
reference configuration puts that collective on the fabric backend, which the
network model already prices". That exclusion clause is load bearing for the
defect decision rule below.

Case B is deliberately not run. Its four times larger token set raises the
per-artifact endpoint load toward the calibrated profile's supported envelope,
and the study needs the smallest cell count that can discriminate the two
compositions.

### Frozen literals

| symbol | value (ps) | source |
|---|---:|---|
| `FLOOR_W8` | 30,128,029 | width-8 base latency of `b200-nccl-2.27-local-v1` |
| `FLOOR_TOTAL` | 1,446,145,392 | `48 * FLOOR_W8`, 48 collectives per simulated step |
| `G_GRAPH` | 809,306 | `turing-cuda-graph` point per launch |
| `G_EAGER` | 2,364,255 | `turing-eager-host` point per launch |
| `Q_GRAPH` | 356,095,000 | `ceil(440 * G_GRAPH / 1000) * 1000` |
| `Q_EAGER` | 1,340,533,000 | `ceil(567 * G_EAGER / 1000) * 1000` |
| `Q_DELTA` | 984,438,000 | `Q_EAGER - Q_GRAPH` |
| `C_DECODE` | 99,024,000 | accepted `a-ep8-400g` decode step 1 compute service |
| `NET_DECODE_400` | 105,502,734 | accepted `a-ep8-400g` decode step 1 fabric service |

### Frozen intervals

A case A decode step schedules at most three decode tokens, so its raw fabric
sum is bounded by the 48 propagation charges below and by three tokens reaching
at most seven remote destinations above:

| interval | endpoints (ps) | meaning |
|---|---|---|
| `NET_400` | `[96,000,000, 140,000,000]` | raw fabric sum, decode step, 400 Gbit/s |
| `NET_200` | `[96,000,000, 200,000,000]` | raw fabric sum, decode step, 200 Gbit/s |
| `CQ` | `[95,000,000, 105,000,000]` | quantized ideal compute service, decode step |

From those, the two compositions predict disjoint intervals for the measured
composed decode step:

| cell | additive interval `Q + FLOOR_TOTAL + NET` (ps) | overlapped interval `CQ + FLOOR_TOTAL + NET` (ps) |
|---|---|---|
| `on-graph440-400g` | `[1,898,240,392, 1,942,240,392]` | `[1,637,145,392, 1,691,145,392]` |
| `on-eager567-400g` | `[2,882,678,392, 2,926,678,392]` | `[1,637,145,392, 1,691,145,392]` |
| `on-graph440-200g` | `[1,898,240,392, 2,002,240,392]` | `[1,637,145,392, 1,751,145,392]` |

The additive and overlapped intervals of every cell are disjoint, and the two
400 Gbit/s additive intervals are disjoint from each other. The check-only
command verifies all three disjointness facts before any run, so the flagship
relation cannot be satisfied by both hypotheses.

## Registered fatal guards, unscored

Fatal means void, not a lost point. One violated guard voids the run for the
purpose of closing or establishing anything, and no behavioral pass fraction is
then interpretable. These are never reported as a fraction.

- **G1 disabled-path identity.** Cell `off-400g` reproduces every accepted
  literal of the mission cell `a-ep8-400g`: `steps.jsonl` SHA-256
  `f7c3b85866ce0fdb6d87c1f706ad6fd21153210c79d3c21d874b7642267eb11a`; step 0
  latency 372,217,008 ps; step 1 latency 204,526,734 ps; `r00` TTFT
  372,217,008 ps with components queue 0, kernel 99,024,000 and collective
  273,193,008 ps; `r01` TTFT 536,566,546 ps and `r02` TTFT 544,628,780 ps;
  41 scheduler steps; 1,968 `htsim_rnic` invocations; 50,802,688 total routed
  bytes; 25,401,344 peak per-rank egress bytes; per-request routed bytes
  19,607,552, 8,028,160 and 23,166,976. Both new seams land in the same
  functions, and each branch proved only its own off path; this guard is the
  first evidence that the two off paths hold together.
- **G2 oracle identity.** The capture SHA-256 is
  `ef570a67fd8bbbb6a8d73b8ad9f73171d3eaaf9a51efc04f676bd9f25c8988fe`.
- **G3 conservation.** In every cell, zero makespan conservation failures, zero
  interval conservation failures, zero artifact partition failures, zero
  disagreements between attributed kernel time and the published compute
  service, and `completed_at_ps == virtual_time_ps + step_latency_ps` for every
  step.
- **G4 inactive components.** `kv_ps`, `dma_ps`, `nic_ps` and `control_ps` are
  exactly zero in every interval of every cell. Configuration forced.
- **G5 replay identity.** Every request in every cell serves the oracle token
  sequence exactly, agrees with the oracle stop reason under the mission
  study's frozen normalization, and reports `ttft_ps > 0`.
- **G6 backend health.** Every simulated step reports routing mode `captured`,
  placement epoch 0 and backend quiescence.
- **G7 floor reach.** In each enabled cell every simulated step charges exactly
  48 collective base latencies of 30,128,029 ps each, summing to 1,446,145,392
  ps, with at most one base charge per semantic collective. In the disabled
  cell no base latency is charged at all.
- **G8 endpoint envelope.** Every collective's critical endpoint load lies
  inside the profile's width-8 envelope `[14, 458,752]` bytes. The
  implementation raises at planning time when it does not, so a violation
  appears as a failed cell rather than a wrong number. If it occurs, the
  finding is that the calibrated envelope is too narrow for the mission
  workload, the composed measurement is void, and the defect takes TRAF-32.
- **G9 device disclosure.** Enabled cells report GPU key `gtx1660-ti-sm75` with
  a pinned `b100` provider envelope; the disabled cell reports `b100` for both.
- **G10 host term reach.** Each enabled cell reports the frozen host profile
  and launch count, and its raw `provider_compute_ps` stays the accepted B100
  value, so the pinned provider did not change the compute input.

## Registered scored behavioral relations

Three families. The pre-freeze entailment question is answered for each: given
the fatal guards and identity checks already registered above, and given how
the fixture is constructed, can this relation fail?

### F1, the composition rule

- **F1.a** Every decode step of `on-graph440-400g` has `step_latency_ps` inside
  `[1,898,240,392, 1,942,240,392]` and outside `[1,637,145,392, 1,691,145,392]`;
  every decode step of `on-eager567-400g` inside
  `[2,882,678,392, 2,926,678,392]` and outside the same overlapped interval;
  every decode step of `on-graph440-200g` inside
  `[1,898,240,392, 2,002,240,392]` and outside `[1,637,145,392, 1,751,145,392]`.
  A decode step is a simulated step in which every scheduled request is in the
  decode phase.
- **F1.b** Every simulated step of `on-graph440-400g` and `on-graph440-200g`
  publishes `compute_service_ps == 356,095,000`, and every simulated step of
  `on-eager567-400g` publishes `compute_service_ps == 1,340,533,000`.

**Can F1 fail?** Yes. No registered guard pins `compute_service_ps`. G3 and G7
together force `step_latency_ps == compute_service_ps + 1,446,145,392 + fabric
sum`, so F1 does reduce to a statement about `compute_service_ps`, and that is
disclosed here rather than hidden: `compute_service_ps` is exactly the quantity
the two merged branches disagree about. If the sink composed the host launch
demand against the collective-bearing step service, `compute_service_ps` would
be the ideal quantized compute near 99,024,000 ps, every decode step would land
in the overlapped interval, and both halves of F1 would fail. The two candidate
values differ by a factor of 3.60 for the graph profile and 13.54 for the eager
profile, so no rounding, quantization or scheduling accident can move a
measurement from one interval to the other. The alternative is not
hypothetical: it is written down in a report merged to main.

### F2, host-profile separation

- **F2** For every decode-step scheduling composition present in both
  `on-graph440-400g` and `on-eager567-400g`, the eager latency minus the graph
  latency equals exactly 984,438,000 ps.

**Can F2 fail?** Yes. Under the overlapped composition the difference is
exactly zero for every pair, because the largest launch demand, 1,340,532,585
ps, is smaller than the collective-bearing step service. It also fails under
any composition that charges the host term per artifact rather than per step,
and under any defect that lets the host term leak into the fabric services. It
is not entailed by F1: F1's intervals admit any difference in
`[940,438,000, 1,028,438,000]` ps, an 88,000,000 ps window, while F2 demands a
single picosecond-exact value inside it.

Evaluability, frozen now: F2 requires at least two matched decode-step
compositions. The two cells run a closed loop, so a different composed step
time changes which requests share a step. If fewer than two matched
compositions exist, F2 is reported as **not evaluated** with the observed
count, it is neither passed nor failed, and the genuine-risk denominator drops
to the number of evaluated families. That reduction is disclosed in the
headline.

### F3, bandwidth compression under the floor

- **F3** For every decode-step scheduling composition present in both
  `on-graph440-200g` and `on-graph440-400g`, the ratio of the 200 Gbit/s
  latency to the 400 Gbit/s latency is strictly greater than 1.0 and at most
  1.030.

**Can F3 fail?** Yes. It fails if the calibrated base latency were link-rate
dependent, if the floor were charged per byte rather than per collective, or if
the halving changed the matched steps' fabric loads by more than 3 percent of
the composed step. The comparator is the accepted mission study's own published
400 to 200 Gbit/s step-makespan ratios on the disabled path, 1.0441 to 1.4760:
if the composed ratio does not fall below that whole published range, the floor
is not behaving as an additive, rate-independent term. F3 is not entailed by F1
or F2, neither of which says anything about the 200 Gbit/s cell. F3 does not
discriminate between the two compositions and is not claimed to; it is the
scaling companion the physical-sanity rule requires.

Evaluability: the same two-matched-composition rule as F2.

## Registered exact-unscored relations

These pass or fail but carry no numerator and no denominator, because
registered guards or fixture construction determine them.

- **E1** `step_latency_ps == compute_service_ps + sum(collective base charges) +
  sum(raw fabric services)` for every simulated step of every cell. Entailed by
  the artifact equation and G3.
- **E2** In the disabled cell, no step publishes a nonzero collective base
  charge and `compute_service_ps` equals the ideal quantized compute. Entailed
  by G1 and G7.
- **E3** Per-request TTFT equals the sum of the seven attribution components of
  the first interval, and `tpot_ps * (token_count - 1)` equals the sum of the
  components of the post-first-token intervals exactly in rational arithmetic,
  in every cell including the enabled ones. Entailed by G3, and retained
  because the task prompt asks for it explicitly.

## Physical sanity, stated before any digit is read

One floor and one ceiling per headline number, from first principles.

1. **Composed decode step, graph profile, 400 Gbit/s.** Floor: the step cannot
   be shorter than the host launch demand alone, `440 * 809,306 = 356.095` us,
   and it cannot be shorter than 48 serial collectives each carrying at least
   the calibrated 30.128029 us intercept, so at least
   `0.356095 + 1.446145 = 1.802240` ms. Ceiling: at most three decode tokens
   reach at most seven remote destinations each, so an artifact carries at most
   `3 * 7 * 2048 = 43,008` bytes, i.e. 0.860160 us at 20 ps per byte, giving a
   per-artifact composed ceiling of `30.128029 + 2.000 + 0.860160 = 32.988189`
   us and a step ceiling of `0.356095 + 48 * 0.032988189 = 1.939528` ms.
2. **Composed decode step, eager profile, 400 Gbit/s.** The same bounds with
   `Q_EAGER`: `[2.786678, 2.923966]` ms.
3. **Weight-read floor.** Per-rank resident weights at `ep_world = 8` are about
   554,631,168 bytes, which at the B100 envelope's 8.0e12 bytes/s takes 69.329
   us. If the composed step is measured near 1.9 ms then compute is about 5.2
   percent of it and the modeled step is no longer compute bound at all; near
   2.9 ms it is about 3.4 percent.
4. **Serialization floor and share.** A decode step moves at most
   `48 * 43,008 = 2,064,384` bytes, i.e. 41.288 us at 400 Gbit/s. No more than
   2.2 percent of a 1.9 ms composed step can be byte serialization; everything
   else is fixed cost. This is the reason F3's ratio must be near one.
5. **Real-system plausibility.** The mission study's own comparable-deployment
   band for a real decode step of this model at this parallelism is 1.1 to 4.5
   ms. A composed step inside that band is not evidence of accuracy: it would
   be produced by two transferred constants whose provenance is a consumer GPU
   and an intra-node NVLink all-reduce. The implied per-request decode rate,
   `1 / TPOT`, is expected to fall from the accepted 3,292 to 4,786 tokens per
   second into the low hundreds, which is closer to a plausible cross-node
   expert-parallel deployment; the report must say that the plausibility is
   inherited from transfers, not earned by calibration.

## The traffic-coverage claim, stated before it is checked

An independent analysis says the largest remaining error after this wave is
traffic coverage: the captured traffic is MoE dispatch and combine only, with
tensor-parallel all-reduces, KV movement and weight traffic absent, and adding
a 24-layer model's tensor-parallel all-reduces would add roughly another 1.446
ms, which is 50 to 76 percent of the whole composed step.

The arithmetic that claim must satisfy, written down here first:

- Megatron-style tensor parallelism performs two all-reduces per transformer
  layer, one after the attention output projection and one after the MLP output
  projection. Twenty-four layers give 48 all-reduces, the identical collective
  count the MoE dispatch and combine pair already produces.
- At the calibrated width-8 intercept the added floor is therefore exactly
  `48 * 30,128,029 = 1,446,145,392` ps, i.e. 1.446145 ms. The claim's magnitude
  is arithmetically forced once the collective count is 48 and the width is 8.
- Its share of the composed step is 1.446145 divided by the measured composed
  step. Under the additive composition that is 75.8 percent at graph-440 and
  50.0 percent at eager-567; under the overlapped composition it would be 87.6
  percent for both. The quoted "50 to 76 percent" therefore already presupposes
  the additive composition, so measuring the composition also decides whether
  the claim's own denominator is right.
- Registered check, reported and not scored because F1 entails it: the measured
  share lies inside `[0.48, 0.78]` across the two 400 Gbit/s enabled cells.

Two qualifications the report must make rather than assume:

- Adding tensor-parallel all-reduces at width 8 is not a pure addition. It
  shards attention and MLP weights eight ways, so the per-rank compute term
  falls. On a step where compute is already about 5 percent of the total, that
  correction is second order, and the report must say so with numbers.
- Weight traffic is a load-time cost, not a per-step cost. Naming it in a
  per-step budget overstates the per-step inventory, and the report must
  separate it from KV movement, which is genuinely per-step under
  disaggregation.

The competing candidate for largest remaining error, registered here so that
the answer is not chosen after the fact: the serial lowering charges every
artifact strictly serially, so the modeled step has exactly zero compute and
communication overlap. A real MoE engine overlaps at least a layer's combine
with the next layer's pre-dispatch compute. If half of the 48 collective floors
were hidden behind compute, the composed step would fall by 0.723073 ms, which
is 37.9 percent of the graph composed step and 25.0 percent of the eager one.
That is the same order as the traffic-coverage addition and in the opposite
direction. The report will state whether traffic coverage is the largest
remaining error or one of two comparable ones, with the arithmetic for both.

## Defect decision rule, frozen before the measurement

The boundary is explicit: no calibrated constant is edited to make a number
look right, and no modeled behavior is changed after a failing measurement.

- If the measurement shows the additive composition, that is **not** a defect.
  The registered provenance of the 440 to 567 launch count excludes the
  collective launches by its own words, so a launch demand built only from
  compute launches may overlap only compute, and charging the collective
  service outside that overlap is consistent. In that case no ID is registered
  for the composition, and the open modelling question, whether the host may
  run ahead across a collective boundary, is reported as prose.
- If the measurement shows a host term charged more than once per step, or a
  base latency charged more than once per semantic collective, or a disabled
  path that does not reproduce its accepted baseline, that is a defect. A host
  term defect takes **COMP-32**, a collective floor defect takes **TRAF-32**,
  and a step-result or attribution contract defect takes **CORE-50**.
- No ID is registered for an adjacent improvement, a nicer representation or
  future work. Zero new IDs is the expected outcome.

## Closure scope

This study closes nothing. COMP-2, TRAF-11 and COMP-11 are already closed on
main, and no open task carries a clause about the composition of the two. The
study produces a measurement, an error-budget recomputation and, if warranted
by the rule above, exactly one defect registration.

## Evidence classes

Counts from different classes are never summed into one headline.

- **Fatal-unscored guards:** G1 through G10. A single violation voids the run.
- **Scored behavioral relations:** F1, F2, F3. Three families, genuine risk,
  reduced by any family reported as not evaluated under its frozen
  evaluability rule.
- **Exact-unscored relations:** E1, E2, E3.
- **Reported, not scored:** the error-budget recomputation, the
  traffic-coverage arithmetic, the overlap counter-candidate, the implied
  decode rate and every physical-sanity position statement.

## Reproduction

The frozen check-only command validates every input and every arithmetic fact
above and creates nothing:

```
python examples/composed_step_budget_v1/check_only.py \
    --cache-dir <hugging-face-cache-root> \
    --htsim-rnic <path-to-htsim_rnic>
```

The run command, once the harness exists, is:

```
python examples/composed_step_budget_v1/run_study.py \
    --cache-dir <hugging-face-cache-root> \
    --htsim-rnic <path-to-htsim_rnic> \
    --run-dir <writable-output-directory>
```

Capture and replay need the environment carrying torch, transformers and vLLM
0.26.0, with `PYTHONPATH` at this worktree and `SIMLLM_TXT2BIN` selecting the
GOAL text-to-binary converter. Bulk outputs stay outside Git.
