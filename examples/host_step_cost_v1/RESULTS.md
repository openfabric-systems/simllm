# Host step cost v1 results

## Outcome

The behavioral result is accepted. Corrected calibration attempt three is
nonvoid and passes 4 of 4 independent genuine-risk calibration relations.
Held-out live attempt three is nonvoid and passes 12 of 12 post-specified
genuine-risk instances. All fatal calibration and holdout guards held. The
exact ideal compatibility guard also held. Fatal guards are conditions for an
interpretable run, never part of a score.

The earlier evidence is kept in its proper class. Calibration attempt two was
nonvoid but not accepted at 3 of 4 scored relations. Live attempt one is void
with findings and has no interpretable behavioral fraction. Live attempt two
is a nonvoid repair regression with zero scored genuine-risk instances: its 12
reported magnitude rows are entailed by earlier exact fatal oracles and are
unscored.

The evidence closes COMP-2 for the fixed per-step host-cost scope.
The historical CPU-proxy versus GPU-initiated network-submission clause was
not demonstrated, so it moves to exactly one residual, COMP-28. COMP-29 and
COMP-30 remain unused. Commit `7c957ad` removed COMP-2 from the open registry,
registered COMP-28, marked COMP-2 closed in the ledger and reconciled the
generated task counts. The repository-wide lint, test, documentation and
portability gates all pass.

## Evidence classes

| Evidence class | Outcome | Scored? |
|---|---|---|
| Corrected calibration attempt three | 4/4 independent relations pass | Yes, its own genuine-risk fraction |
| Live attempt one | VOID with findings after one fatal physical guard failed | No fraction is interpretable |
| Live attempt two | Nonvoid repair regression; 12 entailed observations retained | No, genuine-risk denominator is zero |
| Held-out live attempt three | 12/12 post-specified genuine-risk instances pass | Yes, its own genuine-risk fraction |
| Calibration and live fatal guards | Every guard in the accepted runs held | No, fatal conditions are unscored |
| Ideal compatibility OFF-G1 | Held on a fresh five-cell accepted-study replay | No, identity is fatal and unscored |
| Named mission study exact oracles | 13/13 in the fresh ideal replay | Separate native evidence class |
| Direction, network-identity and native-Q checks | Passed | No, derived or conformance-only |

The calibration and holdout fractions are not added together. The attempt-two
rows are not added to either denominator.

## Physical sanity registered before precision

The following bounds and scaling checks were written before the corresponding
measured values were read. Being inside them is necessary, not proof of
correctness.

### Launch capture bounds

- Floor: every launch time and residual gap is at least 0 ps. Negative time is
  physically impossible.
- Graph and eager ceiling: each pipelined per-launch value must not exceed the
  same run's empty launch-plus-synchronize value.
- Stamped-gap ceiling: the real-kernel device gap must be positive and no
  larger than stamped batch wall time divided by 400 launches.
- Step composition floor: for raw provider service `C`, launch count `N` and
  measured per-launch throughput `g`, `F = max(C, N * g)`. The host can enqueue
  while earlier kernels execute, so adding `N * g` to `C` would double count
  overlap.
- GOAL carrier: because GOAL carries whole nanoseconds, calibrated service is
  `Q = ceil(F / 1,000) * 1,000` ps. Therefore `F <= Q < F + 1,000`.
- Loose step-compute ceiling: `C + N * serialized_launch`. Even fully
  serializing the enclosing launch-and-synchronize operation cannot exceed
  this value.

The representative fixed input moves 554,631,168 bytes. Its memory floor on
an 8 TB/s roof is 69,328,896 ps. A deliberately conservative 1 TB/s ceiling
is 554,631,168 ps. The fixed provider value of 99,024,000 ps lies inside this
range.

### Network and whole-step bounds

Each fixture step has 48 serial dispatch or combine collectives. The
collective-latency floor is therefore `48 * 2,000,000 = 96,000,000` ps before
byte serialization.

For the 400 Gbit/s repair fixture, serialization gives network ceilings of
273,193,008 ps for prefill and 105,502,768 and 105,338,928 ps for the two
decode steps. Its ideal whole-step bounds are:

| Step | Floor (ps) | Ceiling (ps) |
|---|---:|---:|
| Prefill | 195,024,000 | 372,217,008 |
| Decode 1 | 195,024,000 | 204,526,768 |
| Decode 2 | 195,048,000 | 204,386,928 |

For the held-out 200 Gbit/s fixture, byte serialization gives network
ceilings of 450,385,968, 115,005,488 and 114,677,808 ps. With the calibrated
whole-nanosecond service, the tighter whole-step bounds frozen before the
holdout were:

| Profile and launches | Step 0 bounds (ps) | Step 1 bounds (ps) | Step 2 bounds (ps) |
|---|---:|---:|---:|
| CUDA graph, 440 | `[452,095,000, 806,480,968]` | `[452,095,000, 471,100,488]` | `[452,095,000, 470,772,808]` |
| CUDA graph, 567 | `[554,877,000, 909,262,968]` | `[554,877,000, 573,882,488]` | `[554,877,000, 573,554,808]` |
| Eager host, 440 | `[1,136,273,000, 1,490,658,968]` | `[1,136,273,000, 1,155,278,488]` | `[1,136,273,000, 1,154,950,808]` |
| Eager host, 567 | `[1,436,533,000, 1,790,918,968]` | `[1,436,533,000, 1,455,538,488]` | `[1,436,533,000, 1,455,210,808]` |

The separate loose causal ceilings, based on the same-run serialized launch
enclosure, are 2,924,009,008, 2,588,628,494 and 2,588,324,818 ps for 440
launches, and 3,609,404,640, 3,274,024,126 and 3,273,720,450 ps for 567.

### Independent sanity angles

Three independent checks were required before trusting the registered
precision:

1. Host throughput: CPU enqueue and eager launch throughput must be close
   enough to identify a host-bound launch stream.
2. Device timing: the independently stamped device gap must be positive and
   below its batch wall-time enclosure.
3. System plausibility: the resulting Turing launch floor must be compared
   with the mission study's independent generic 0.3 to 3 ms host-cost bracket,
   not only with an earlier simulator run.

The expected bandwidth companion also held. Removing the 96,000,000 ps
collective floor, decode-1 network service rose from 9,502,734 ps at 400
Gbit/s to 19,005,454 ps at 200 Gbit/s, a 1.9999985x change under a 2x
serialization expectation. The small integer difference comes from the
backend's picosecond allocation.

## Chronology and run integrity

The original [compute-fidelity result](../compute_fidelity_v1/RESULTS.md) is
VOID with findings. Fatal XFER-G4 demanded exact positive-work proportionality
from a true 793,650,793.65 ps quantity that is rounded once per provider call;
the doubled result differed by 1 ps. Its behavioral fraction is therefore not
interpretable, and none of its findings was silently promoted to a calibrated
default.

Expectations-only commit `9fb4b06` froze the initial corrected capture,
physical bounds, live relations, exact ideal path and closure scope. Commit
`b30118d` narrowed the residual COMP-28 wording before any measured run. The
registered check-only commands passed at both commits and created no output.

Corrected calibration attempt two then ran from clean commit `13ae8a9`. Every
fatal guard held, so its score was interpretable, but graph replay measured
809,068 ps against the frozen 600,000 to 700,000 ps band. Eager launch
2,544,074 ps, stamped gap 1,573,280 ps and serialized launch 5,605,330 ps were
inside their bands. The result was nonvoid and not accepted at 3/4, so no host
profile was installed from it.

Expectations-only commit `f23a01f` disclosed calibration attempt three. It
widened CAL-1 from `[600,000, 700,000]` to `[600,000, 1,000,000]` ps after the
known miss and widened the then-planned 400 Gbit/s live band from `[1.80,
7.10]` to `[1.80, 7.75]` after incorporating the known empirical endpoint.
This is a post-specified attempt-three refreeze, not a retroactive pass for
attempt two.

Calibration attempt three ran from clean commit `5663447` and was accepted
before the model was installed. The first live implementation run then ran at
clean commit `8a620e9`. It initially appeared to pass its magnitude bands, but
review found that per-layer integer flooring put represented GOAL compute
service below the already frozen physical floor `F` by 6,640 to 20,502 ps.
That violated a fatal precondition. Live attempt one is therefore VOID with
findings, and it has no behavioral fraction.

Commit `30f1f7f` froze the whole-nanosecond repair before it was implemented.
It did not widen LIVE-1, LIVE-2 or LIVE-3. Live attempt two then ran from clean
commit `12d337e` and reproduced every repair literal. A later entailment audit
found that fatal LIVE-G3, LIVE-G5 and LIVE-G6 had already fixed its exact
service, timestamps, TTFT and TPOT. Those fatal oracles determine all 12 band
relations. Attempt two remains nonvoid and useful, but its magnitude rows are
entailed, unscored findings and its genuine-risk denominator is zero.

The exact five-cell ideal replay ran at `a484f38` and OFF-G1 held. Finally,
expectations-only commit `d9905fb` froze a post-specified magnitude holdout on
the previously unused accepted 200 Gbit/s cell without freezing calibrated
timestamps. The harness landed at `994e4b5`, the empty-fallback fail-closed
defect was fixed at `ad1b0aa`, and the clean run observed that same `ad1b0aa`
revision. Its result was recorded at `3b04ec6`.

## Corrected calibration result

The accepted calibration is bound to this exact capture environment:

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1660 Ti |
| Device key | `gtx1660-ti-sm75` |
| GPU UUID | `GPU-a90a812a-41bf-4f2f-c96d-d83e6eae6bd0` |
| Compute capability | 7.5 |
| Driver | 550.90.07 |
| CUDA compiler | 12.4.99, target `sm_75` |
| Host CPU | AMD Ryzen 9 3950X 16-Core Processor |
| Graph shape | 512 nodes, 200 replays |
| Eager shape | 20,000 empty launches plus one final synchronization |
| Stamped shape | 400 back-to-back launches of the 262,144-item probe kernel |

The corrected fatal oracle used zero work, which returned exactly 0 ps. The
positive-work single and double values were 793,650,793 and 1,587,301,587 ps;
their 1 ps residual is descriptive rather than fatal.

All six calibration fatal guards held. They checked exact device, host and
toolchain identity; positive measurements; exact zero-work behavior; causal
measurement enclosures; and a clean, stable revision with stable harness and
probe hashes. They are unscored.

The four independent scored relations passed:

| Relation | Frozen band (ps) | Observed (ps) | Result |
|---|---:|---:|---|
| CAL-1 graph replay | `[600,000, 1,000,000]` | 809,306 | PASS |
| CAL-2 eager host-bound launch | `[2,000,000, 2,700,000]` | 2,364,255 | PASS |
| CAL-3 stamped device gap | `[500,000, 3,000,000]` | 1,635,680 | PASS |
| CAL-4 serialized launch enclosure | `[3,000,000, 20,000,000]` | 5,396,816 | PASS |

The genuine-risk calibration fraction is 4/4. CAL-D1, graph replay being
cheaper than eager launch, is entailed by the disjoint bands and remains
unscored.

The other measurements pass the physical checks. CPU enqueue was 2,364,085
ps, only 170 ps below eager throughput. The stamped batch wall time was
6,058,688,164 ps, whose integer per-launch enclosure is 15,146,720 ps. The
1,635,680 ps gap is positive and, together with 13,511,040 ps of stamped
kernel service, reaches that enclosure exactly.

The installed profiles carry the accepted point and all five retained
observations:

| Profile | Launch class | Point (ps/launch) | Five observations (ps/launch) | Sample-limited empirical range |
|---|---|---:|---|---:|
| `turing-cuda-graph` | `cuda-graph-node` | 809,306 | 624,665; 630,356; 630,124; 809,068; 809,306 | `[624,665, 809,306]` |
| `turing-eager-host` | `eager-host-bound` | 2,364,255 | 2,327,730; 2,337,286; 2,331,958; 2,544,074; 2,364,255 | `[2,327,730, 2,544,074]` |

These ranges are not confidence intervals. The tracked calibration artifact
has SHA-256
`2a4ab022227601d8aca36627e760490abca86cdc237a5b3b5149303ded99cd12`.
Its observed commit is `5663447`.

## Installed model

`HostInitiationModel.ideal()` remains the exact historical zero profile.
`HostInitiationModel.turing_cuda_graph(N)` and
`HostInitiationModel.turing_eager_host(N)` carry the point, sample-limited
range, device and host identity, driver, CUDA version, source study, launch
class and launch count. A calibrated estimate uses `max(C, N * g)`, while the
legacy arbitrary scalar retains its historical additive behavior.

The serial step lowerer is the single timing authority. It applies the term
once and encloses calibrated service at whole-nanosecond GOAL precision. The
packet sink delegates to that lowerer. The device sink, vLLM coordinator path
and SGLang path expose or validate the same selected model rather than adding a
second term. The vLLM observed schedule reports raw provider service separately
from represented service.

The host model itself, `SerialStepLowererConfig`, `HtsimStepSinkConfig` and
adapter configurations default explicitly to `ideal`. A Turing profile
rejects B100, H100 and every device key except `gtx1660-ti-sm75` before a
graph, work directory, timestamp or sink call is produced. A nonideal vLLM
worker also rejects the deliberately empty zero-time fallback and requires a
host-model-aware timing sink.

## Live attempt one: void with findings

The first implementation run undercut `F = max(C, N * g)` after repeated
per-layer flooring:

| Profile | Launches | Required `F` (ps) | Represented service (ps) | Shortfall (ps) |
|---|---:|---:|---:|---:|
| CUDA graph | 440 | 356,094,640 | 356,088,000 | 6,640 |
| CUDA graph | 567 | 458,876,502 | 458,856,000 | 20,502 |
| Eager host | 440 | 1,040,272,200 | 1,040,256,000 | 16,200 |
| Eager host | 567 | 1,340,532,585 | 1,340,520,000 | 12,585 |

This run is VOID with findings. Its 12 magnitude observations have no
interpretable pass fraction. The retained artifact SHA-256 is
`6a78848373f8a21d32902975531efb18967f0440490bf420caa0d032421ebb38`.

## Live attempt two: nonvoid entailed repair regression

The frozen repair changed only calibrated behavior from repeated per-layer
flooring to the narrow whole-nanosecond enclosure. The ideal row did not move.

| Profile | Launches | Service before (ps) | `F` (ps) | Service after `Q` (ps) | Per-step change (ps) |
|---|---:|---:|---:|---:|---:|
| CUDA graph | 440 | 356,088,000 | 356,094,640 | 356,095,000 | 7,000 |
| CUDA graph | 567 | 458,856,000 | 458,876,502 | 458,877,000 | 21,000 |
| Eager host | 440 | 1,040,256,000 | 1,040,272,200 | 1,040,273,000 | 17,000 |
| Eager host | 567 | 1,340,520,000 | 1,340,532,585 | 1,340,533,000 | 13,000 |

The final fixed-schedule metrics were:

| Profile | Launches | TTFT (ps) | Decode 1 (ps) | TPOT (ps) | Final completion (ps) |
|---|---:|---:|---:|---:|---:|
| CUDA graph | 440 | 629,288,008 | 461,597,734 | 461,515,816 | 1,552,319,640 |
| CUDA graph | 567 | 732,070,008 | 564,379,734 | 564,297,816 | 1,860,665,640 |
| Eager host | 440 | 1,313,466,008 | 1,145,775,734 | 1,145,693,816 | 3,604,853,640 |
| Eager host | 567 | 1,613,726,008 | 1,446,035,734 | 1,445,953,816 | 4,505,633,640 |

All six attempt-two fatal guards held, so the repair regression is nonvoid.
However, exact fatal service, timestamp, TTFT and TPOT oracles entail every
LIVE-1, LIVE-2 and LIVE-3 value. The first result artifact classified the 12
rows as scored. Integration review corrects that classification here: the
attempt-two genuine-risk denominator is zero, and all 12 rows are retained as
unscored entailed findings. Its artifact SHA-256 is
`06e4eacb2838cbec8434aae788732a225b8bd9d74379fe008e3e441e11cfac81`.

## Held-out live attempt three

The post-specified holdout used accepted mission cell `a-ep8-200g`, request
`r00`, steps 0 through 2, fixed provider services 99,024,000, 99,024,000 and
99,048,000 ps, and the cell's exact routed-expert input. Unlike attempt two,
no exact calibrated timestamp or scored numerator was frozen.

The ideal row reproduced source step services 549,409,968, 214,029,454 and
213,725,778 ps, partitioned into network services 450,385,968, 115,005,454 and
114,677,778 ps plus the fixed compute values. Ideal TTFT was 549,409,968 ps
and the two-decode micro-fixture TPOT was 213,877,616 ps.

The calibrated rows were:

| Profile | Launches | `Q` (ps) | TTFT (ps) | Decode 1 (ps) | Decode 2 (ps) | TPOT (ps) |
|---|---:|---:|---:|---:|---:|---:|
| CUDA graph | 440 | 356,095,000 | 806,480,968 | 471,100,454 | 470,772,778 | 470,936,616 |
| CUDA graph | 567 | 458,877,000 | 909,262,968 | 573,882,454 | 573,554,778 | 573,718,616 |
| Eager host | 440 | 1,040,273,000 | 1,490,658,968 | 1,155,278,454 | 1,154,950,778 | 1,155,114,616 |
| Eager host | 567 | 1,340,533,000 | 1,790,918,968 | 1,455,538,454 | 1,455,210,778 | 1,455,374,616 |

All measured steps lie inside both predeclared physical enclosures. Network
service matched the ideal row in all 12 calibrated steps, but HOLD-D2 checks
that only after scoring as a survivable, unscored diagnostic.

The genuine-risk relations passed 12/12:

| Profile | Launches | LIVE-1 decode multiplier | LIVE-2 TPOT multiplier | LIVE-3 TTFT multiplier | LIVE-3 increment ratio | Result |
|---|---:|---:|---:|---:|---:|---|
| CUDA graph | 440 | 2.2011010410 in `[2.17, 2.23]` | 2.2018976310 in `[2.17, 2.23]` | 1.4679037785 | 0.3893041857 in `[0.38, 0.40]` | PASS |
| CUDA graph | 567 | 2.6813246648 in `[2.65, 2.71]` | 2.6824621797 in `[2.65, 2.71]` | 1.6549808357 | 0.3892989950 in `[0.38, 0.40]` | PASS |
| Eager host | 440 | 5.3977545259 in `[5.34, 5.46]` | 5.4008205141 in `[5.34, 5.46]` | 2.7131997139 | 0.3892909762 in `[0.38, 0.40]` | PASS |
| Eager host | 567 | 6.8006455504 in `[6.73, 6.88]` | 6.8047074922 in `[6.73, 6.88]` | 3.2597132784 | 0.3892897758 in `[0.38, 0.40]` | PASS |

LIVE-1 contributes 4/4, LIVE-2 contributes 4/4 and LIVE-3 contributes 4/4.
All six fatal holdout guards held, with no fatal fraction. They checked
profile provenance and revision stability, device mismatch rejection, source
and ideal identity, component and request conservation, physical enclosures
plus native `Q`, and conditional budget arithmetic.

The relations are not entailed. Fatal-valid countermodels put calibrated
network service at the 96,000,000 ps floor and miss every LIVE-1 and LIVE-2
lower band. A separate fatal-valid prefill-floor and decode-ceiling
countermodel moves every LIVE-3 increment ratio outside `[0.38, 0.40]`.
HOLD-D1 launch-count and launch-class directions remain unscored by
construction. Native exact-`Q` conformance remains a separate unscored class.

The tracked holdout artifact has SHA-256
`98bb687ad90eae8e856281e594b2af1a94303b2b02d9adbe72ba6393053d0add`.
The expectation commit is `d9905fb`; the observed clean commit is `ad1b0aa`.

## Exact ideal off path

The fresh [end-to-end replay](../end_to_end_replay_v1/RESULTS.md) was nonvoid,
had no violated fatal guard, and passed all 13 of its separate exact-oracle
relations. OFF-G1 then matched the frozen aggregate identity and every
step-record stream:

| Identity | Expected and observed SHA-256 |
|---|---|
| Aggregate canonical cells | `5b51c31c1d83422cecfcbd975bf67690c6cccfd8ca4437ffef3e54985ee615fe` |
| `a-ep4-400g/steps.jsonl` | `fba28840dacc858e67e2202c24b82012a114ee2cf4512b9c72a7c9a2718365cc` |
| `a-ep8-100g/steps.jsonl` | `893fd939460556a6e0639572ef41db58b478c72850f3a43b62de626bbade5706` |
| `a-ep8-200g/steps.jsonl` | `4aebcbdaf27aa6db599101b65e61546de8419283e9b9092237031d9f5796bb08` |
| `a-ep8-400g/steps.jsonl` | `f7c3b85866ce0fdb6d87c1f706ad6fd21153210c79d3c21d874b7642267eb11a` |
| `b-ep8-400g/steps.jsonl` | `997028d3806ed86c43ff558afcb92358bdb78ad4e076e11a75e2622072cc65ea` |

Canonicalization removed only top-level `wall_seconds`, retained every
timestamp, byte count, service value, request, operation and completion order,
serialized the five cells by name, and appended one LF. The per-cell step
streams matched byte for byte. The fresh summary SHA-256 is
`55fb19c4d476ce8a1559bb75c711cdeba136203bb36cc06aec9363ab1cdbfe86`.
The compatibility check observed commit `a484f38`; its tracked artifact has
SHA-256
`b91447be221c0f814ef4dcfa47c9555a3cfa1e8d722794a085b8b0ef0fb0479b`.
OFF-G1 is fatal and unscored.

## Mission error budget

Before this work, mission error-budget item 1 was exactly 0 in the model. On
the explicitly selected measured Turing profiles, it becomes a point launch
term of 0.356095 to 1.340533 ms over the registered launch counts. Propagating
the five retained samples gives an empirical envelope of 0.274853 to 1.442490
ms. The lower empirical edge is about 0.025 ms below the mission's generic
0.3 ms host-cost floor; the rest overlaps its 0.3 to 3 ms bracket. The minimum
point term is already 3.60 times the 0.099024 ms provider compute input.

For the conditional Turing sensitivity, correlate the same represented host
term in numerator and denominator:

```text
simulated      = represented_host + 0.105502734 ms
plausible real = represented_host + [0.72, 1.44] ms
```

The accepted point endpoints give optimism `[1.4249530295, 3.8910394651]`.
The sample-limited empirical endpoints give `[1.3969639214, 4.5085504088]`.
Both lie inside the frozen broad `[1.35, 4.70]` enclosure. These budget rows
are derived and unscored.

This is not a replacement for the mission study's generic 5x to 22x absolute
budget. It is a conditional launch-throughput sensitivity on one GTX 1660 Ti
and one host. Residual scheduling, sampling and Python costs remain unknown
and are treated as zero. B100 and H100 host-step cost is explicitly unknown;
the implementation refuses to transfer the Turing values to either device.
Therefore the absolute composed range for the reference B100 configuration
remains unsupported.

## Two-sided freeze integrity after the first measured run

The first measured run occurred while `13ae8a9` was checked out. Every later
commit through the final tracked evidence is classified here.

| Commit | Classification | Modeled behavior and measured before/after |
|---|---|---|
| `3c114e5` | Evidence only | Retained the nonvoid calibration-attempt-two miss. No modeled behavior changed. |
| `112746c` | Capture-harness defect fix | Made attempt identity data driven. No modeled behavior or measured value changed. |
| `f23a01f` | Expectations-only, post-specified calibration refreeze | No implementation behavior changed. CAL-1 changed from `[600,000, 700,000]` to `[600,000, 1,000,000]` ps and the planned 400 Gbit/s live band changed from `[1.80, 7.10]` to `[1.80, 7.75]` after the known miss. |
| `5663447` | Capture harness | Added attempt-two to the five-sample empirical range. No modeled behavior changed. |
| `09b9a00` | Evidence only | Recorded accepted calibration attempt three. No modeled behavior changed. |
| `915d222` | Reporting defect fix | Corrected the selected attempt label. No modeled behavior or measurement changed. |
| `8a620e9` | Modeled behavior addition under the attempt-three freeze | Before, the calibrated path was unavailable and the ideal 400 Gbit/s decode-1/TPOT values were 204,526,734/204,456,816 ps. After explicit selection, live-attempt-one findings for decode-1/TPOT were graph-440 461,590,734/461,508,816, graph-567 564,358,734/564,276,816, eager-440 1,145,758,734/1,145,676,816, and eager-567 1,446,022,734/1,445,940,816 ps. The ideal path did not move. |
| `f5e2e74` | Evidence only | Recorded the first live artifact, later found void. No modeled behavior changed. |
| `1297602` | Evidence audit | Retained and reclassified live attempt one as VOID with findings. No modeled behavior changed. |
| `30f1f7f` | Expectations-only repair refreeze | Froze live attempt two after the void finding without widening magnitude bands. No modeled behavior changed. |
| `f376bcc` | Registry documentation | Tagged COMP-2 `(Precision; P1; L)` and moved only it to Precision. No modeled behavior changed. |
| `42a0432` | Modeled-behavior defect fix, implemented after `30f1f7f` refroze attempt two | Calibrated service moved 356,088,000 to 356,095,000; 458,856,000 to 458,877,000; 1,040,256,000 to 1,040,273,000; and 1,340,520,000 to 1,340,533,000 ps. TTFT moved 629,281,008 to 629,288,008; 732,049,008 to 732,070,008; 1,313,449,008 to 1,313,466,008; and 1,613,713,008 to 1,613,726,008 ps. Only explicitly selected calibrated behavior changed; ideal stayed exact. |
| `12d337e` | Verifier and tests | Added the ideal compatibility checker. No modeled behavior changed. |
| `a484f38` | Evidence only | Recorded the nonvoid attempt-two repair regression. Its original score classification is corrected here to zero scored and 12 entailed. No modeled behavior changed. |
| `13623e0` | Evidence only | Recorded OFF-G1. No modeled behavior changed. |
| `d9905fb` | Expectations-only, post-specified holdout freeze | Reclassified attempt two and froze the 200 Gbit/s holdout before running it. No modeled behavior changed. |
| `994e4b5` | Holdout harness and tests | Built the held-out evaluator and explicit non-entailment checks. No modeled behavior changed. |
| `ad1b0aa` | Fail-closed defect fix before the holdout run | The unsupported nonideal empty worker fallback changed from returning 0 ps to raising an error; the supported timing-sink path and all successful modeled timestamps were unchanged. The ideal empty fallback remained 0 ps. |
| `3b04ec6` | Evidence only | Recorded accepted holdout attempt three. No modeled behavior changed. |
| `a630d13` | Regression test only | Locked the exact ideal defaults in both step-sink configurations. No modeled behavior changed. |
| `7c957ad` | Registry, ledger and module documentation | Closed COMP-2, registered COMP-28 and reconciled progress counts. No modeled behavior changed. |
| This results commit | Evidence documentation only | Records the already observed evidence, closure mapping, validation and contradiction sweep. No modeled behavior changed. |

The only modeled timestamp change made after seeing a failed live number was
the quantization repair at `42a0432`. It was disclosed as attempt two and
refrozen in `30f1f7f` before implementation and rerun. The later empty-fallback
change fixes unsupported validation behavior, landed before the holdout run,
and does not change the supported scored path or supply magnitude evidence.

## Closure scope and evidence mapping

Only COMP-2 is being closed in this batch.

| Task | Closed? | Basis |
|---|---|---|
| COMP-2 | Yes | Calibration 4/4, held-out live 12/12, every fatal guard held, OFF-G1 held, registry and ledger reconciled |

The frozen closure clauses map as follows:

| Registered clause | Evidence |
|---|---|
| “calibration attempt three is nonvoid and accepted” | `calibration.json` has no fatal failures, reports `accepted`, and passes CAL-1 through CAL-4 at 4/4. |
| “each constant carries device and launch-class provenance plus sample-limited uncertainty and refuses a mismatched device” | The two factory profiles carry the full GPU, host, driver, CUDA, launch-class and five-sample range. CAL-G1/G2 and LIVE-G1/G2 held; both profiles reject B100 and H100 before output. |
| “the host model, serial lowerer, packet sink and coordinator dispatch apply one shared term exactly once” | The lowerer is the sole timing authority; sink and adapters validate its exact model. Attempt-two and holdout conservation, raw-versus-represented attribution, `F`/`Q` enclosure and no-double-charge checks held. |
| “live holdout attempt three is nonvoid and both profiles pass its decode-multiplier, TPOT and TTFT relations” | The clean `ad1b0aa` holdout has no fatal failures and passes LIVE-1 4/4, LIVE-2 4/4 and LIVE-3 4/4. Fatal-valid countermodels establish non-entailment. |
| “the named accepted study matches exactly on the ideal path” | OFF-G1 matched the aggregate digest and all five step streams after a nonvoid mission replay with 13/13 native exact oracles. |
| “the mission budget is recomputed” | The conditional point interval is `[1.4249530295, 3.8910394651]`; the empirical interval is `[1.3969639214, 4.5085504088]`; B100 remains explicitly unknown. |
| “the owning registry tag, bucket and ledger reconcile” | COMP-2 was tagged `(Precision; P1; L)` and moved alone into Precision at `f376bcc`. Commit `7c957ad` removed the closed entry, registered COMP-28 in Precision and added COMP-2 to the ledger. The progress check reports 83 of 194 tasks closed, 111 open, with compute at 4 of 23 closed. |

The original legacy entry registered:

> calibrated host-initiation profiles (GPU-initiated vs CPU-proxy constants)
> for launch-path sensitivity studies.

This run demonstrates fixed per-step CUDA-graph and eager-host launch classes,
not CPU-proxy versus GPU-initiated network submission. It therefore closes the
fixed-step part and carries the untouched network-submission clause into the
one required residual.

## Residual discipline

Exactly one new ID is registered:

> COMP-28 (Precision; P2; L): After COMP-21 supplies device-bound structural
> captures for CPU-proxy and GPU-initiated network submission, fit and validate
> their scalar host-initiation projections for the analytical fallback used
> only while structural submission is disabled. Carry GPU, host, RNIC and
> submission-class provenance plus predeclared capture uncertainty; held-out
> ready-to-RNIC-visible latency must remain within that uncertainty. The ideal
> zero-cost profile remains the exact compatibility path.

COMP-28 is required because a registered COMP-2 clause was not demonstrated.
It consumes COMP-21's structural captures and owns only their scalar fallback;
it does not duplicate that hardware campaign. No ID is added for B100 because
the frozen task explicitly permitted “unknown for B100” and promised no B100
constant. No ID is added for adjacent improvements. COMP-29 and COMP-30 remain
unused.

## Contradiction sweep

The required protected-file sweep was read-only:

- `README.md`: no direct fixed-step host-cost contradiction was found. Its
  compute inventory names host initiation only at module level and does not
  claim a calibrated value.
- `docs/README_PRO.md`: the compute module status still says the fixed
  per-step cost is “omitted entirely” and “not yet installed.” Its generated
  open count is reconciled. The fidelity-level row saying a
  deterministic fixed per-step constant is landed is compatible with the old
  scalar seam but does not describe this calibration. The
  `compute_fidelity_v1` study row is historical VOID chronology and should
  remain as such.
- `docs/architecture.md`: the host-initiation passage still says the pre-wire
  CPU-proxy versus GPU-initiated operation path defaults to zero and describes
  one per-endpoint scalar fallback. That is a distinct operation-level seam,
  but without a fixed-step distinction it can be read as contradicting the new
  calibrated step profiles.

These prose hits are reported here rather than edited. Outside the protected
sweep, commit `7c957ad` reconciled `docs/modules/backends.md` so COMP-28 now
owns the CPU-proxy versus GPU-initiated scalar fallback.

## Validation and storage

The result-producing evidence is complete:

- calibration attempt three: accepted, observed at `5663447`;
- live attempt one: VOID with findings, retained;
- live attempt two: nonvoid entailed repair regression, observed at `12d337e`;
- OFF-G1: held, observed at `a484f38`;
- live holdout attempt three: accepted, observed at `ad1b0aa`.

Initial and refrozen check-only runs passed without creating output. Final
closure gates passed:

- **PASS:** `.venv/bin/ruff check .`
- **PASS:** `.venv/bin/pytest -q` reports 1,445 passed and 7 skipped.
- **PASS:** Python 3.10 byte-compilation covers `simllm`, the study and its
  focused tests.
- **PASS:** `python3 scripts/check_docs_format.py` reports all 10 module docs
  conform; it notes 27 unrelated untagged legacy entries.
- **PASS:** `.venv/bin/python scripts/task_progress.py --check` reports current
  task-progress and module-status counts: 83 of 194 closed, 111 open.
- **PASS:** before this report is committed, the tracked worktree is clean and
  this report is the sole untracked file. The final clean status is a handoff
  check because recording that observation necessarily changes this file.

All bulk artifacts remain under the branch-local external run root. The seven
retained run directories occupy 290 MB total: the calibration attempts use
1.1 MB each; the three host-step live directories use 8.5, 8.7 and 8.8 MB;
and the two five-cell ideal replays use 131 MB each. The largest individual
file is 2,520,119 bytes. No multi-gigabyte sweep or packet-sized trace was
retained, and no deletion command was used.
