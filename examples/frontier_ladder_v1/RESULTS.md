# Frontier ladder result

## Verdict

**PASS.** The ideal LogGOPSim rung is a valid fast substitute for the frozen
serialized point-to-point fabric legs, where the batch-32 packet observation is
only **1.56 percent** above ideal. It is not a
contention model: the batch-32 eight-into-one packet observation is
**8.11x** the ideal leg because eight flows share one packet
receiver ingress while the ideal receiver charges no per-byte gap.

The result is non-void. Exact-oracle families are L-A 6 of 6
and L-B 6 of 6. Behavioral-relation
families are M-1 6 of 6, M-2
6 of 6, M-3 6 of
6 and S 18 of 18. The plot
contract P is 4 of 4. Wall-time family W is
1 of 1, with a median of
0.012780 seconds for all twelve native legs. These evidence
classes are not summed.

The strict ladder record now contains 24 ESTIMATE
points and 30 SIMULATED points. The six B100 ideal-rung
points are closed-form ESTIMATE points with no execution provenance. The twelve
H100 ideal-rung points remain executed SIMULATED points.

## Mechanism envelope

Family L-A executes one flow of the frozen maximum payload. Family L-B
deliberately executes eight equal concurrent flows into rank zero. Its six
receiver fan-in stamps are detected and acknowledged; L-A's six stamps are
clean and unacknowledged. Every row was executed seven times through the pinned binary with the exact argument spelling
`-G 0.02`; the expected column is the frozen literal, not a closed form
evaluated by the runner.

| Family | Batch | Flows | Bytes per flow | Expected ns | Observed ns | Envelope | Verdict |
|---|---:|---:|---:|---:|---:|---|---|
| L-A | 1 | 1 | 6,651,904 | 135,038 | 135,038 | CLEAN | PASS |
| L-A | 2 | 1 | 13,303,808 | 268,076 | 268,076 | CLEAN | PASS |
| L-A | 4 | 1 | 26,607,616 | 534,152 | 534,152 | CLEAN | PASS |
| L-A | 8 | 1 | 53,215,232 | 1,066,304 | 1,066,304 | CLEAN | PASS |
| L-A | 16 | 1 | 106,430,464 | 2,130,609 | 2,130,609 | CLEAN | PASS |
| L-A | 32 | 1 | 212,860,928 | 4,259,218 | 4,259,218 | CLEAN | PASS |
| L-B | 1 | 8 | 1,478,201 | 31,564 | 31,564 | ACKNOWLEDGED | PASS |
| L-B | 2 | 8 | 2,956,402 | 61,128 | 61,128 | ACKNOWLEDGED | PASS |
| L-B | 4 | 8 | 5,912,804 | 120,256 | 120,256 | ACKNOWLEDGED | PASS |
| L-B | 8 | 8 | 11,825,608 | 238,512 | 238,512 | ACKNOWLEDGED | PASS |
| L-B | 16 | 8 | 23,651,215 | 475,024 | 475,024 | ACKNOWLEDGED | PASS |
| L-B | 32 | 8 | 47,302,429 | 948,048 | 948,048 | ACKNOWLEDGED | PASS |

The L-B comparison is not schedule-identical. At batch 32, the pinned packet
record carries payloads [47302429, 47302429, 47302429, 47302429, 47302428, 47302428, 47302428, 47302428], four of
N and four of N-1, while the ideal execution carries
[47302429, 47302429, 47302429, 47302429, 47302429, 47302429, 47302429, 47302429], eight uniform max-size flows. The
comparison remains valid for this audited ideal rule because the receiver
charges zero per-message overhead and no receiver per-byte gap. Completion is
set by the largest incoming flow, and both flow sets retain that same maximum.

The quotient table keeps packet and ideal picoseconds as the unreduced source
integers. M-1 is serialized concurrent packet over ideal, M-2 is incast
concurrent packet over ideal, and M-3 is the isolated packet leg over ideal.

| Family | Batch | Ideal ps | Packet ps | Exact quotient | Decimal | Verdict |
|---|---:|---:|---:|---:|---:|---|
| M-1 | 1 | 135,038,000 | 137,201,000 | 137,201,000 / 135,038,000 | 1.016018 | PASS |
| M-1 | 2 | 268,076,000 | 272,317,000 | 272,317,000 / 268,076,000 | 1.015820 | PASS |
| M-1 | 4 | 534,152,000 | 542,551,000 | 542,551,000 / 534,152,000 | 1.015724 | PASS |
| M-1 | 8 | 1,066,304,000 | 1,083,018,000 | 1,083,018,000 / 1,066,304,000 | 1.015675 | PASS |
| M-1 | 16 | 2,130,609,000 | 2,163,953,000 | 2,163,953,000 / 2,130,609,000 | 1.015650 | PASS |
| M-1 | 32 | 4,259,218,000 | 4,325,821,000 | 4,325,821,000 / 4,259,218,000 | 1.015637 | PASS |
| M-2 | 1 | 31,564,000 | 242,356,000 | 242,356,000 / 31,564,000 | 7.678241 | PASS |
| M-2 | 2 | 61,128,000 | 482,629,000 | 482,629,000 / 61,128,000 | 7.895383 | PASS |
| M-2 | 4 | 120,256,000 | 963,174,000 | 963,174,000 / 120,256,000 | 8.009363 | PASS |
| M-2 | 8 | 238,512,000 | 1,924,264,000 | 1,924,264,000 / 238,512,000 | 8.067787 | PASS |
| M-2 | 16 | 475,024,000 | 3,845,860,000 | 3,845,860,000 / 475,024,000 | 8.096138 | PASS |
| M-2 | 32 | 948,048,000 | 7,689,053,000 | 7,689,053,000 / 948,048,000 | 8.110405 | PASS |
| M-3 | 1 | 31,564,000 | 32,110,000 | 32,110,000 / 31,564,000 | 1.017298 | PASS |
| M-3 | 2 | 61,128,000 | 62,136,000 | 62,136,000 / 61,128,000 | 1.016490 | PASS |
| M-3 | 4 | 120,256,000 | 122,188,000 | 122,188,000 / 120,256,000 | 1.016066 | PASS |
| M-3 | 8 | 238,512,000 | 242,293,000 | 242,293,000 / 238,512,000 | 1.015852 | PASS |
| M-3 | 16 | 475,024,000 | 482,500,000 | 482,500,000 / 475,024,000 | 1.015738 | PASS |
| M-3 | 32 | 948,048,000 | 962,915,000 | 962,915,000 / 948,048,000 | 1.015682 | PASS |

At batch 32, M-1 is exactly 4,325,821,000 / 4,259,218,000 =
1.015637. M-2 is exactly 7,689,053,000 /
948,048,000 = 8.110405. The isolated M-3 control is
962,915,000 / 948,048,000 = 1.015682. The
single-flow physics agrees across levels; the eight-fold gap appears only when
the packet receiver must serialize shared ingress.

## Step-level ladder

The TRAF-68 masking finding stands. On the twelve H100 points, the kernel is
slower than every fabric leg, so all three rungs produce the same step time.
The same is true for five B100 points. Only B100 batch 32 differs: the packet
rung includes the pinned intra-node candidate and reaches 4,523,298,348 ps,
while ESTIMATE and loggopsim-ideal remain at 4,257,218,560 ps.

| Configuration | Batch | ESTIMATE ps | Ideal class | Ideal rung ps | Packet SIMULATED ps | Verdict |
|---|---:|---:|---|---:|---:|---|
| b100-one-node-intra | 1 | 3,448,398,380 | ESTIMATE | 3,448,398,380 | 3,448,398,380 | PASS |
| b100-one-node-intra | 2 | 3,465,966,380 | ESTIMATE | 3,465,966,380 | 3,465,966,380 | PASS |
| b100-one-node-intra | 4 | 3,501,102,380 | ESTIMATE | 3,501,102,380 | 3,501,102,380 | PASS |
| b100-one-node-intra | 8 | 3,571,374,380 | ESTIMATE | 3,571,374,380 | 3,571,374,380 | PASS |
| b100-one-node-intra | 16 | 3,711,918,380 | ESTIMATE | 3,711,918,380 | 3,711,918,380 | PASS |
| b100-one-node-intra | 32 | 4,257,218,560 | ESTIMATE | 4,257,218,560 | 4,523,298,348 | PASS |
| h100-two-node-serialized | 1 | 8,234,981,205 | SIMULATED | 8,234,981,205 | 8,234,981,205 | PASS |
| h100-two-node-serialized | 2 | 8,276,934,638 | SIMULATED | 8,276,934,638 | 8,276,934,638 | PASS |
| h100-two-node-serialized | 4 | 8,360,841,504 | SIMULATED | 8,360,841,504 | 8,360,841,504 | PASS |
| h100-two-node-serialized | 8 | 8,528,655,235 | SIMULATED | 8,528,655,235 | 8,528,655,235 | PASS |
| h100-two-node-serialized | 16 | 8,864,282,698 | SIMULATED | 8,864,282,698 | 8,864,282,698 | PASS |
| h100-two-node-serialized | 32 | 9,535,537,623 | SIMULATED | 9,535,537,623 | 9,535,537,623 | PASS |
| h100-nine-node-incast | 1 | 8,234,981,205 | SIMULATED | 8,234,981,205 | 8,234,981,205 | PASS |
| h100-nine-node-incast | 2 | 8,276,934,638 | SIMULATED | 8,276,934,638 | 8,276,934,638 | PASS |
| h100-nine-node-incast | 4 | 8,360,841,504 | SIMULATED | 8,360,841,504 | 8,360,841,504 | PASS |
| h100-nine-node-incast | 8 | 8,528,655,235 | SIMULATED | 8,528,655,235 | 8,528,655,235 | PASS |
| h100-nine-node-incast | 16 | 8,864,282,698 | SIMULATED | 8,864,282,698 | 8,864,282,698 | PASS |
| h100-nine-node-incast | 32 | 9,535,537,623 | SIMULATED | 9,535,537,623 | 9,535,537,623 | PASS |

## Figure

[PDF](figures/frontier-ladder.pdf) and [PNG](figures/frontier-ladder.png)
render one NV-style two-panel figure through the Agg backend. The left panel
keeps the frozen logarithmic axes, uses a distinct marker for each rung and
emphasizes the exact six-point B100 packet Pareto front. The right panel shows
the three mechanism quotients and labels M-2 at 8.11x. The H100
2N and 9N configurations coincide at step level in the pinned record, so the
figure draws dashed 2N over a wider 9N line to make both configurations visible.

## Physical sanity

Before reading the batch-32 serialized observation, payload bytes over 400
Gbit/s set a floor of 4,257,218,560 ps and
one 2,000 ns propagation delay set a ceiling of
4,259,218,560 ps. The measured ideal leg is
4,259,218,000 ps, inside those bounds.

For batch-32 incast, all remote bytes over one 400 Gbit/s ingress set a floor
of 7,568,388,560 ps, while eight isolated
packet completions set a ceiling of
7,703,320,000 ps. The packet observation is
7,689,053,000 ps, inside the range. From
batch 16 to 32, the ideal and packet incast legs scale by
1.995790x and
1.999307x. At the step
boundary, the
9.536
ms H100 kernel remains above the
7.689
ms packet fabric leg, which independently explains the unchanged step.

## Fatal guards

| Guard | Outcome | Predicate | Mutation control |
|---|---|---|---|
| FG-1 | PASS | pinned record, contract and LogGOPSim binary hashes match before execution | rejected |
| FG-2 | PASS | all 18 ladder points carry strict, distinct authorities with 24 ESTIMATE and 30 SIMULATED points | rejected |
| FG-3 | PASS | all 12 nonzero ideal fabric legs carry executed native provenance, portable argv and exact G | rejected |
| FG-4 | PASS | the frozen expectations commit is an ancestor of the implementation run commit | rejected |

Every mutation control exercised the real predicate and was rejected. Native
stdout and stderr bytes, portable argument vectors and rendered GOALs remain in
the append-only external attempt directory.

## Post-specified corrections

These corrections follow adversarial review of the original publication. They
are post-specified regression checks and record repairs, not amendments to the
immutable ladder expectations.

- The TRAF-20 closure is withdrawn. The ladder measured modeled error through
  M-1, M-2 and M-3, but it read packet observations from a pinned record and
  executed no packet reference, so it measured no packet wall clock. TRAF-68's
  closure remains earned.
- The ideal level now refuses overlapping multi-source receiver fan-in by
  default because the receiver per-byte gap is unmodeled and the frozen cell is
  about 8x optimistic. An explicit acknowledgment permits deliberate runs and
  is stamped in provenance. This rerun exercises that acknowledgment for L-B;
  it is not the separately frozen enforcement acceptance study.
- Registry and backend prose state the same narrowed open scope for TRAF-20.
- The original published ideal wall-time median is
  0.034612 seconds. The
  superseded unpublished-attempt value is no longer used. This correction
  rerun's nondeterministic median is 0.012780 seconds.
- L-B discloses the mixed pinned packet payloads and uniform ideal payloads,
  plus the maximum-flow invariance that makes the ideal makespan unchanged.
- FG-1 mutates only the observed digest. Its predicate derives the comparison
  from observed and expected digests and rejects the mutant without editing a
  cached `matched` flag.
- The publication figure makes the coincident H100 2N and 9N curves visible and
  states why they coincide.
- Six nonexecuted B100 ideal-rung points are corrected from SIMULATED to
  ESTIMATE. The record now tallies 24 ESTIMATE and
  30 SIMULATED points.

The modeled-quantity reproduction check is
PASS:
all L-A and L-B native observations and repetitions, M-1 through M-3 operands and quotients, S step times, and physical-sanity values match the original
published artifact exactly. Wall-clock samples and corrected metadata are
excluded from that deterministic comparison.

## Provenance

- Frozen expectations commit: `228f3c77b98af1f0f60985405a8db67ebb67c0a6`
- Frozen expectations SHA-256: `e3e83264df6e72e83736a06dddcba11a501c75a25c8c1fb0a9c7b1e9c0caeea3`
- Implementation run commit: `b66e621a395af0e614825d2b1268e7efc13d6956`
- Pinned deployment record SHA-256: `f2f216068bf5ba914853c62a2ee965ede0ebfc0a6f29e3d11cfa5f45eac359ad`
- Pinned LogGOPSim binary SHA-256: `7e0f13ee3c87a20e9d2e94dbbd74c46075fd03df2f1b04d1ed9739c43ee0a2bf`
- txt2bin SHA-256: `f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b`

## Project effect

What ran: `frontier_ladder_v1` executed all twelve frozen ideal fabric legs
through the pinned LogGOPSim binary, joined them with the pinned analytical and
packet points, and rendered the three-rung frontier plus mechanism envelope.

What came out: the ideal level stays within about 1.6 percent of packet timing
for serialized point-to-point traffic but is about 8.11x optimistic for
eight-into-one incast, with the shared receiver ingress identified as the
missing mechanism. All deterministic modeled quantities reproduce exactly.

What it changes: TRAF-20 reopens because no packet wall clock was measured.
Its remaining acceptance is narrowed to measured packet-reference speed on
identical flow sets and the separately frozen enforcement study. TRAF-68 stays
closed because the fabric-leg view still exposes the contention its step-level
map masks. The ladder and its six-point packet Pareto front remain the
deployment planning comparison surface.

What it does not change: no rung gains an absolute-accuracy claim against
silicon, no packet execution or enforcement acceptance study occurred, the
TRAF-68 step-masking result remains literal, and statistical transport tails
remain owned by TRAF-19. TRAF-20 does not close.
