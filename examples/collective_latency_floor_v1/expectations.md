# Collective latency floor v1 expectations

The original expectations-only record for TRAF-11 and the calibrated-floor
slice of COMP-11 froze the external-source audit, calibration split, model
form, physical bounds, live metric relations, compatibility guards and
production command before the behavior was implemented or any
result-producing run occurred. The attempt-two refreeze below is explicitly
post-observation and does not rewrite that chronology.

## Attempt-two refreeze disclosure

Production attempt one at SimLLM revision
`cec7109adeb9656de92f9ef5ea54572accdc3208` was void. Its sensitivity fixture
returned the upper endpoint of the previously published fluid-backend
completion quantization range, while the original fatal oracle admitted only
the lower endpoint. Both `sensitivity_fixture_identity` and
`propagation_rate_cancellation` therefore failed, no behavioral score was
interpretable, and no owning task closed.

The earlier mission result reports rate-cancelled propagation from 2,000,000
to 2,000,001 ps over 912 artifacts. Attempt two refreezes this fixture at its
executed integer completion values: 6,014,081 ps at 400 Gbit/s and 10,028,161
ps at 200 Gbit/s, whose difference remains exactly 4,014,080 ps and whose
rate-cancelled value is 2,000,001 ps. The separately reported physical-model
reference remains 2,000,000 ps. This amendment changes only fatal, unscored
harness oracles. It changes no calibrated parameter, modeled behavior, scored
relation, execution graph or closure clause. Any later production result is
reported as attempt two under this refreeze.

## Claim boundary

The study makes three claims:

1. the flat 450 GB/s, zero-latency intra-node surrogate can be replaced by a
   smaller model identified from same-generation public B200 evidence;
2. the same explicitly selected model can add a non-propagation collective
   floor to the fluid fabric path while retaining the backend's 2.000 us
   propagation as a separate run-record field; and
3. that floor reaches `StepResult`, TTFT and TPOT, while the compatibility-off
   and all-remote identity paths reproduce the accepted behavior exactly.

The study does not claim that an all-reduce capture is an all-to-all capture,
that a public issue attachment is a controlled local measurement, or that one
effective affine form describes payloads outside the fitted 8 B to 256 KiB
range. It also does not claim COMP-11's peer topology, per-link routing,
receiving-GPU HBM interaction, reduction lanes or proxy operations. Those are
quoted under closure scope so they cannot disappear behind the calibrated
floor result.

## Source audit

The evidence was authored against SimLLM commit
`6b7200bd68f86208060a825bc6bf18b2ea1bd4ca`. The observed htsim gitlink during
the audit was `fc4400e4ca619223481536632074045cb6af2756`. These are provenance
facts, not equality requirements on a later run or live gitlink.

### Physical point-to-point capacity bound

NVIDIA's [DGX B200 system guide](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html)
states that one DGX B200 contains eight B200 GPUs and two fifth-generation
NVLink switches providing 14.4 TB/s aggregate bandwidth. NVIDIA's
[DGX GB200 system guide](https://docs.nvidia.com/dgx/dgxgb200-user-guide/dgxgb200-user-guide.pdf),
section 1.3, describes the fifth-generation NVLink switch bandwidth as full
duplex. The resulting physical envelope is 1.8 TB/s bidirectional per GPU, or
900 GB/s in one direction. This is a point-to-point capacity ceiling, not an
achieved NCCL rate and not the fitted model parameter.

### Collective capture

The collective source is the public attachment to NVIDIA `nccl-tests`
[issue 333](https://github.com/NVIDIA/nccl-tests/issues/333):
[`nccl-test-result.zip`](https://github.com/user-attachments/files/21326711/nccl-test-result.zip).
The zip SHA-256 observed before freeze is
`91629a3b4a6eff4ac2e8bbc2261a928dbcca42f07c02d7f1fe15f9d981d0713f`;
the contained `nccl-b200.local.tsv` SHA-256 is
`639348d43e625d8b7199c45db29ec7a848165974142016fab7b85ca34564a8f3`.

This is a user-supplied public capture hosted with an NVIDIA repository issue,
not an NVIDIA product guarantee and not a local SimLLM measurement. The issue
records an eight-B200 system, driver 570.158.01, CUDA 12.9, NCCL
`2.27.0a0+cuda12.9` from commit `dec8621`, and `nccl-tests` commit `59072b7`.
The selected `local.tsv` rows use local buffer registration (`-R 1`) and
participant widths 2, 4 and 8. The source command sweeps 8 B through 128 MiB.
This study fits only 8 B through 256 KiB, before later protocol transitions
make one affine rate visibly inadequate.

No same-generation point-to-point payload curve or B200 all-to-all latency
capture was available locally. The model therefore uses the vendor capacity
only as an independent physical bound and uses the public all-reduce capture
as its timing anchor. The resulting parameters are inferred from published
data, not presented as values measured by this worker.

## Frozen calibration

For one collective with participant width `W`, let `E` be the critical GPU's
total directional endpoint bytes over that collective. The selected model is

```text
collective_service_ps(E, W)
    = participant_latency_ps[W]
    + ceil(E * 1e12 / effective_bandwidth_bytes_per_second)
```

The participant-latency table is the concurrency form. The available capture
identifies widths 2, 4 and 8 directly, but it does not identify a trustworthy
interpolation or saturation law between them. A calibrated run must reject any
other width with an actionable error. This direct table is smaller and more
honest than adding a concurrency breakpoint that the three widths cannot
identify.

For an all-reduce source row with per-rank input payload `S`, the independently
computed critical endpoint bytes are `E = 2 * (W - 1) * S / W`. Fit one
latency intercept for each width and one shared byte slope by ordinary least
squares over every `out_time` row from 8 B through 256 KiB except 4 KiB. The
4 KiB row is held out at every participant width. The frozen integer model is:

| Parameter | Value |
|---|---:|
| profile | `b200-nccl-2.27-local-v1` |
| effective bandwidth | 70,027,079,100 B/s |
| width-2 latency | 10,722,112 ps |
| width-4 latency | 15,745,167 ps |
| width-8 latency | 30,128,029 ps |
| supported source payload | 8 B to 262,144 B |

The unrounded fit was 70,027,079,100.09 B/s and latency intercepts 10.72211157,
15.74516736 and 30.12802859 us. The effective bandwidth is only 7.8 percent of
the 900 GB/s one-direction physical envelope. That gap is expected for small
collectives whose control, synchronization, kernel and protocol costs dominate;
the fitted rate must not be described as an NVLink wire-rate measurement.

### C1: held-out phase completion

The registered TRAF-11 bar is quoted exactly:

> "held-out phase completion error is at most 10 percent or 1 microsecond,
> whichever is larger."

The held-out payload is 4,096 B at every captured width. These expectations
use the integer production form above and whole-picosecond upward rounding:

| Width | Endpoint bytes | Public capture ps | Frozen prediction ps | Absolute error ps | Allowed ps |
|---:|---:|---:|---:|---:|---:|
| 2 | 4,096 | 10,520,000 | 10,780,604 | 260,604 | 1,052,000 |
| 4 | 6,144 | 16,180,000 | 15,832,905 | 347,095 | 1,618,000 |
| 8 | 7,168 | 30,310,000 | 30,230,390 | 79,610 | 3,031,000 |

The result run must obtain the predictions from the public production model,
not restate this table. All three raw errors are evaluated before any exact
parameter or report-conservation guard. C1 is one genuine-risk family over
three held-out instances.

### C2: local live-path response

At 4 KiB, width 2, 4 and 8 all-local collectives must execute through
`StepRecord`, the checked graph projection, `HtsimStepSink` and `StepResult`.
For each width, enabling the calibrated profile must add exactly that width's
participant latency once per semantic collective and must replace the local
450 GB/s serialization rate with 70,027,079,100 B/s. It must not add the
latency once per ring round or once per directed peer.

Before exact report guards run, raw live completion must increase strictly
with participant width, and the enabled-minus-off delta must increase strictly
from width 2 to 4 to 8. This is one genuine-risk family over one three-width
grid. Exact byte conservation, operation count and the configured parameter
values are fatal-unscored.

## Fabric floor and run-record decomposition

The selected floor is additive to backend transport service:

```text
composed_collective_ps
    = participant_latency_ps[W]
    + max(local_serialization_ps, fabric_transport_ps)
```

The latency is charged once by the semantic collective authority. It is never
charged independently by both local and fabric projections. The backend stays
the sole authority for propagation, wire serialization and fabric contention.
For `rnic-nn-fluid`, the calibrated profile carries the separately named
2,000,000 ps propagation reference identified by `end_to_end_replay_v1`; the
run record must expose that reference, raw fabric transport, collective base
latency and composed service in distinct fields. The timing path must validate
the reference against the backend's published one-picosecond completion
quantization envelope rather than assuming it changes backend behavior.

The floor is inferred from the B200 collective intercept after removing the
fitted byte term. It is a reduced-form non-serialization NCCL floor. The public
capture does not separately resolve kernel launch, rendezvous, protocol setup,
reduction arithmetic and sub-microsecond intra-node propagation, so the report
must call it inferred and carry the held-out uncertainty. It must not call the
30.128 us value a direct software-only measurement.

### C3: rate sensitivity distinguishes a floor from bandwidth

Use one eight-participant all-remote collective whose bottleneck endpoint moves
200,704 B. On the current fluid transport, the independently frozen services
are:

| Rate | Transport ps | With calibrated floor ps |
|---:|---:|---:|
| 400 Gbit/s | 6,014,081 | 36,142,110 |
| 200 Gbit/s | 10,028,161 | 40,156,190 |

The transport values are the 2,000,000 ps propagation model, 20 or 40 ps per
byte, and the backend's one-picosecond completion quantization for this
fixture. The slow-rate to fast-rate ratio is 1.667447 before the floor. With
the floor, the raw ratio must lie in `[1.10, 1.12]` and be strictly closer to
one than the off-path ratio. This relation is evaluated from raw executed
services before the propagation-reference and decomposition guards. It is one
genuine-risk family over one rate pair. A rescaled bandwidth with no additive
floor fails.

The same two rates must recover exactly 2,000,001 ps by cancelling the byte
term: `2 * transport_400g_ps - transport_200g_ps`. This is the separately
reported 2,000,000 ps propagation reference plus the fixture's registered
one-picosecond backend quantization. That exact check is fatal-unscored. A
violation voids the run.

## Flagship TTFT and TPOT relation

The live fixture uses the Granite geometry from `end_to_end_replay_v1`: 24 MoE
layers, one dispatch and one combine at every layer, EP width 8, TP width 1,
and 400 Gbit/s `rnic-nn-fluid`. It executes one larger prefill-shaped step and
two equal smaller decode-shaped steps. There are exactly 48 semantic
collectives in every nonempty step.

The configured eight-participant floor therefore adds exactly
`48 * 30,128,029 = 1,446,145,392 ps`, or 1.446145392 ms, to each step. Applying
the registered 10 percent calibration uncertainty gives an addition band of
`[1.301530853, 1.590759931] ms`. The mission study's network term to beat is
0.106 ms, so the enabled representative decode network term must land in
`[1.4075, 1.6968] ms`; the exact configured point is about 1.5521 ms. Its
0.205 ms whole-step reference becomes about 1.6511 ms before any host-cost or
compute recalibration.

### C4: end-to-end metric response

Define TTFT as the first prefill-shaped `StepResult` latency and TPOT as the
mean latency of the two equal decode-shaped `StepResult` values. Both metrics
must rise when the floor is enabled. The absolute enabled-minus-off change in
each must equal 1,446,145,392 ps. The relative TTFT increase must be strictly
smaller than the relative TPOT increase because the larger prefill byte term
amortizes the same 48 fixed collective charges over a larger baseline.

C4 reads raw `StepResult` values before exact artifact-count, field-sum or
compatibility guards. A model that reports the new term without placing it on
the scheduler-visible path fails. C4 is one genuine-risk family over the TTFT
and TPOT pair. Their exact equal absolute changes are a wiring oracle and are
not counted as a second family.

## Physical sanity frozen before the result run

Every result table must show these bounds before its measured column.

1. **Link physics.** No local transfer can beat `endpoint_bytes / 900 GB/s`,
   the one-direction vendor capacity bound. At the 4 KiB holdouts those floors
   are 4.551 ns, 6.827 ns and 7.964 ns for widths 2, 4 and 8. The modeled
   values must be above them.
2. **Calibration envelope.** The public B200 capture spans about 9.56 to
   38.83 us over 8 B through 256 KiB for the selected widths. Every held-out
   prediction must remain inside a conservative 5 to 50 us real-system
   envelope before its exact error is read.
3. **Decode collective budget.** Forty-eight small collectives at the mission
   study's 15 to 30 us plausible real band cost 0.72 to 1.44 ms. The configured
   1.446 ms addition is at the upper edge, and the frozen uncertainty band is
   1.302 to 1.591 ms. A result below 0.72 ms has lost a floor; a result above
   1.60 ms has charged it more than once or left the frozen uncertainty.
4. **Bandwidth scaling.** At fixed 200,704 B, halving 400 to 200 Gbit/s doubles
   only serialization. The transport increment must be 4,014,080 ps at either
   profile selection, while the total ratio moves from about 1.667 toward one.
5. **End-to-end plausibility.** The old composed error budget was 1.1 to
   4.5 ms against a 0.205 ms simulated decode step, or roughly 5x to 22x
   optimistic. Adding the exact configured floor makes that simulated point
   about 1.651 ms. Relative to the unchanged 1.1 to 4.5 ms plausibility band,
   the residual ratio becomes about 0.67x to 2.73x. A lower endpoint below one
   is not an accuracy victory; it warns that the all-reduce-derived upper-edge
   floor may overprice the mission's all-to-all and that host and compute terms
   still need independent calibration.

These are independent views: a source-capacity floor, a held-out real capture,
a collective-count budget, a bandwidth relation and an end-to-end deployment
comparison. Agreement with the fitted equation alone is not physical evidence.

## Fatal and unscored guards

A single violation voids the run and leaves both owning tasks open. Fatal
guards are never reported as a fraction.

- The default `None` model and an explicit legacy model reproduce every
  accepted artifact byte, flow row, timestamp, locality field, `StepResult`,
  TTFT and TPOT exactly.
- Omitted placement and an explicit all-remote `gpu-rank` placement remain
  identical with the floor off and with it enabled. The calibrated floor may
  change time, never GOAL bytes, tags, flow bytes or completion order.
- Every semantic collective receives zero or one base-latency charge. No ring
  round, directed peer or local/fabric split duplicates it.
- For every artifact, `composed == base_latency + max(local, fabric)`, and all
  artifact services sum exactly to `StepResult.step_latency_ps`.
- The run record exposes profile identity, effective bandwidth, supported
  participant table, per-artifact base latency, raw fabric transport, the
  2,000,000 ps propagation reference and composed service as distinct fields.
- Rate cancellation recovers exactly 2,000,001 ps on both the off and enabled
  paths, decomposed as the 2,000,000 ps propagation reference plus the
  registered one-picosecond backend completion quantization.
- Unsupported participant widths fail before an output file or published
  result is mutated.
- Backend quiescence, operation inventory, byte conservation and positive
  timestamps hold in every executed cell.
- The public-source URLs, source digests and frozen integer parameters are
  change-set guards. They protect provenance but add no behavioral score.

## Entailment analysis

C1 evaluates raw predictions from the production model before checking its
reported parameter fields. A wrong byte equation, width lookup or rounding can
fail C1 even when later provenance fields are correct. C2 evaluates live sink
completion before exact field and artifact guards, so a correct component can
still fail to reach `StepResult`. C3 evaluates two raw executed rates before
the later exact propagation cancellation; a rescaled bandwidth can conserve
perfectly and still miss the ratio. C4 reads raw TTFT and TPOT before exact
charge-count and sum guards; a report-only field fails it.

The held-out error directions implied by exact predictions, the 48-collective
absolute delta, propagation cancellation, all-remote identity, field sums and
unsupported-width rejection are exact, configuration-forced or
by-construction evidence. They are fatal-unscored and do not enter the
genuine-risk denominator.

The scored headline has four relation families: C1 over three held-out cells,
C2 over one participant grid, C3 over one rate pair and C4 over one metric
pair. Evidence-class counts remain separate.

## Closure scope

TRAF-11's registered clause is:

> "calibrate the current flat 450 GB/s, zero-propagation, per-endpoint NVLink
> surrogate against same-generation point-to-point and collective captures.
> Sweep payload and participant count on the reference eight-GPU node, hold out
> at least one payload per participant width, and replace the constant with the
> smallest identifiable bandwidth, latency and concurrency form whose held-out
> phase completion error is at most 10 percent or 1 microsecond, whichever is
> larger. Report the before/after TTFT and TPOT effect and retain the exact
> all-remote identity path."

C1 owns the quantitative held-out bar, C2 owns local live reachability and the
participant sweep, C4 owns TTFT/TPOT, and the fatal identity guard owns the
all-remote path. The public collective capture and vendor capacity ceiling do
not demonstrate the quoted same-generation point-to-point payload capture.
Even if every registered relation and guard holds in a non-void run, TRAF-11
may be retired only while registering TRAF-31 for that exact undemonstrated
source clause. A failed scored relation or fatal guard instead leaves TRAF-11
open; it does not manufacture another residual task. TRAF-32 remains unused.

COMP-11's registered clause is:

> "Add peer topology and per-link routing, ingress service and its interaction
> with the receiving GPU's HBM, reduction lanes so a collective's arithmetic is
> priced, and proxy operations. Calibrate the egress latency and bandwidth from
> real captures rather than the current synthetic profiles, and reconcile the
> intra-node split with TRAF-10 so one collective is never counted both here and
> on the fabric backend."

C1 and C2 own calibrated latency/bandwidth. C3 plus the one-charge fatal guard
own reconciliation with the TRAF-10 split. This study does not demonstrate the
quoted peer topology, per-link routing, receiving-HBM, reduction-lane or proxy
clauses. Even if the registered floor evidence is non-void, COMP-11 may be
retired only while registering COMP-31 for those exact undemonstrated clauses.
A failed scored relation or fatal guard instead leaves COMP-11 open. Adjacent
ideas do not receive IDs.

## Registered command and pre-freeze dry run

The result-producing command is:

```text
.venv/bin/python examples/collective_latency_floor_v1/run_study.py --run-dir "$SIMLLM_COLLECTIVE_FLOOR_RUN_ROOT" --htsim-rnic "$SIMLLM_HTSIM_RNIC"
```

Before this expectations commit, the exact command is run with
`--check-only`. Check-only parses the complete production CLI, validates only
the frozen source shapes, integer arithmetic, physical-bound ordering and
native executable inputs, imports no SimLLM target module, invokes no native
tool and creates no output directory or artifact.
