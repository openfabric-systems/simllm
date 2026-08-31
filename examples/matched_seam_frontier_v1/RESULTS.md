# Matched-seam frontier result

The first published run is void against its own FG-1. FG-1 required that no
roofline term, declared efficiency or fitted constant appear anywhere in the
scored arm. That requirement cannot hold for this study for two independent
reasons. First, the external resolver is speed-of-light normalized by
construction: it evaluates analytical rooflines inside scored cells, including
240 roofline evaluations across 15 general matrix multiplication (GEMM) shapes
in the TP4 batch-64 cell. Second, the external composition applies five serving
factors of its own: the 1.1 prefill-latency, 1.08 decode-latency, 0.9 prefill
rate-matching, 0.92 decode rate-matching and 1.8 autoscale factors. The guard
was not widened after seeing the result. The first run and every number it
published remain below as void evidence. The corrected freeze at `4ed8d1a`
defines the new run, and DEPLOY-12 is reopened because its prior closure rested
on the void run.

## Corrected contract and run

What ran: append-only `attempt-0001` evaluated the corrected freeze from
`4ed8d1a` through two complete scored evaluations in fresh processes, a live
source audit of the pinned aiconfigurator 0.11.0 installation, all three packet
cells, the original family bands and the record-only figure projection. The
implementation and tracked adjustment table landed in `998a690` before this
attempt ran. The first freeze, figure addendum and corrected freeze remained
byte-identical.

The corrected FG-1 is split into three guards. FG-1a permits the external
resolver's own speed-of-light normalization and empirical terms, but forbids
any SimLLM-authored roofline, declared efficiency, fitted constant or fitted
curve from reaching a scored root. Each scored number carries a dependency
graph, the graph validator walks every reachable value, and the evaluation
process replaces SimLLM's `RooflineProvider` with an aborting sentinel. Thus a
label alone cannot satisfy the guard. FG-1b requires the tracked adjustment
table to match the pinned source by content hash and to contain exactly the
adjustments reached by the composition. FG-1c requires a remove-one Family R
range for every table row.

FG-6 compares the complete canonical scored-evaluation JSON byte for byte
across two fresh processes. Only `elapsed_seconds` and `W-1` are excluded, by
name. Both records hash to
`85a37550456d753efa95d8260d291328626ef8da07b2938cd35e57deb4152f74`.

### Declared external adjustments

The tracked [external adjustment table](external_adjustments.json) has SHA-256
`c6778a81cdc6078ce74f06733e4bce9d99a92b4ab3eccba4a83d14e7d063a09e`.
The live audit found all eight declared rows and no undeclared applied factor.

| Adjustment | Value | Exact pinned source | External description |
|---|---:|---|---|
| Prefill latency correction | 1.1 | `aiconfigurator/sdk/task_v2.py:441` | `aiconfigurator/cli/example_v1_deprecated.yaml:118` says it corrects predictions that are too optimistic. |
| Decode latency correction | 1.08 | `aiconfigurator/sdk/task_v2.py:442` | `aiconfigurator/cli/example_v1_deprecated.yaml:119` says it corrects predictions that are too optimistic. |
| Prefill rate-matching degradation | 0.9 | `aiconfigurator/sdk/task_v2.py:443-447` | `aiconfigurator/sdk/inference_session.py:227-239` attributes it to rate-matching pipeline bubbles. |
| Decode rate-matching degradation | 0.92 | `aiconfigurator/sdk/task_v2.py:443-447` | `aiconfigurator/sdk/inference_session.py:227-239` attributes it to batch-slot under-saturation. |
| Autoscale time-to-first-token heuristic | 1.8 | `aiconfigurator/sdk/task_v2.py:448-452` | `aiconfigurator/sdk/inference_session.py:684-690` approximates concurrent prefill queueing before the time-to-first-token filter. |
| H200 memory-bandwidth empirical scale | 0.8 | `aiconfigurator_core/systems/h200_sxm.yaml:7` | The same source line calls it a nonofficial observation-based bandwidth correction. |
| Memory-operation empirical constant | 0.000003 seconds | `aiconfigurator_core/systems/h200_sxm.yaml:8` | The same source line adds a 3 microsecond observation-based memory-operation latency. |
| Context-attention extra-latency correction | 1.1 | `aiconfigurator_core/sdk/operations/attention.py:549` | Lines 539-549 inflate QK normalization, rotary-position and key/value-write latency. |

The pinned source assigns 0.9 to prefill rate matching and 0.92 to decode rate
matching. The corrected-freeze prose transposes those phase names. The tracked
configuration already carried the source assignments before the corrected
freeze, and the new run follows the pinned source rather than silently
rewriting it.

## New scored outcomes

What came out: the corrected run is nonvoid and MIXED. All eight fatal guards
hold. Family S passes 13 of 13, Family R passes 10 of 10, Family F passes 12 of
13, Family M passes 2 of 2 under its corrected packet-priced versus
unpriced-network meaning, and Family W passes 1 of 1. The deciding scored miss
is still F-2-09 at 0.607495219355 against the unchanged 0.75 lower bound.

| Register | Frozen acceptance | Observed | Outcome |
|---|---|---:|---|
| FG-1a | No SimLLM-authored roofline, efficiency or fit reaches a scored value | Dependency trace clean; aborting `RooflineProvider` never constructed | PASS |
| FG-1b | Applied external adjustments equal the tracked table | Eight of eight declared; no missing or extra applied factor | PASS |
| FG-1c | Remove-one Family R range for every adjustment | Eight of eight published below | PASS |
| FG-2 | Frozen service identity and evidence stamps | Exact | PASS |
| FG-3 | Protected inputs and prior publications unchanged | Byte-identical | PASS |
| FG-4 | Corrected freeze precedes implementation and run | `4ed8d1a` precedes `998a690` and `attempt-0001` | PASS |
| FG-5 | External TTFT stays outside isolated service scoring | No misuse found | PASS |
| FG-6 | Two full fresh-process scored records byte-identical, named wall exclusions only | Equal hashes; two exclusions | PASS |

No fatal guard voids the corrected run.

### Every frozen family band

- Family S requires exact service identity. All ten decode and three prefill
  cells are bit-equal, so S passes 13 of 13.

- Family R requires every local/published decode-step quotient in
  `[0.98, 1.02]`. The ten quotients span 0.999946608534 to 1.000076344974, so R
  passes 10 of 10.

- Family F1 requires the exact coordinate formulas
  `x = 1e12 / decode_step_ps` and request capacity times 500 output tokens over
  used GPUs. It passes. F2 requires each step-frontier quotient in
  `[0.75, 1.35]`; it passes 9 of 10, with F-2-09 at 0.607495219355. F3 requires
  a strictly monotone ideal frontier with at least eight points; all ten points
  pass. F4 permits one local point to answer at most three external rows; the
  observed maximum is two. Family F therefore passes 12 of 13.

- Family M1 requires every packet-priced/unpriced-network capacity quotient to
  be at least 1.000000. All ten pass. M2 requires the maximum to be at least
  1.02; the maximum is 1.042715399805. The two scored M relations pass, but the
  mechanism scope is limited as stated below.

- Family W requires the complete corrected run to finish within 600 seconds.
  It took 289.941444 seconds, so W passes 1 of 1.

Family D remains an unscored decomposition. It retains the 99.203805 ms raw
prefill pass, 9.920380 ms prefill correction, 87.299348 ms autoscale correction,
-0.000533 ms table-precision reconciliation and 196.423000 ms published
time-to-first-token value. Evidence classes and denominators remain separate.

What it changes for the project: the first run no longer closes DEPLOY-12. The
corrected run supplies its FG-1a, FG-1b, FG-1c and FG-6 evidence, but DEPLOY-12
stays open for the LogGOPSim-priced third arm required before any isolated
network-mechanism claim. DEPLOY-13 stays open on the rounded-axis boundary, and
COMP-88 stays open on promotion of the study-local adjustment table into the
installed content-addressed composition surface. The corrected run exposes no
additional residual requiring a new stable task ID.

What it does not change: this result does not validate either planner against
hardware, does not close a calibration task, does not repair F-2-09, does not
price the external planner's omitted network service, and does not establish
receiver-side fan-in serialization as the cause of the Family M gap.

## Family R remove-one sensitivity

Each range is recomputed with exactly one applied factor removed. Factors that
cannot reach Family R retain the baseline range after the complete dependency
trace proves them unreachable. The decode latency factor is decisive, and the
3 microsecond memory constant has the largest range change.

| Removed external factor | Family R quotient range | Family R reachable |
|---|---:|---|
| Prefill latency correction 1.1 | 0.999946608534 to 1.000076344974 | No |
| Decode latency correction 1.08 | 0.925876489383 to 0.925996615717 | Yes |
| Prefill rate matching 0.9 | 0.999946608534 to 1.000076344974 | No |
| Decode rate matching 0.92 | 0.999946608534 to 1.000076344974 | No |
| Autoscale heuristic 1.8 | 0.999946608534 to 1.000076344974 | No |
| Memory-bandwidth scale 0.8 | 0.995899044931 to 0.999412885209 | Yes |
| Memory constant 3 microseconds | 0.873646907044 to 0.964963420120 | Yes |
| Context-attention correction 1.1 | 0.999946608534 to 1.000076344974 | No |

Rate matching changes Family F and M capacity composition but not the Family R
service quotient, which is why the decode rate factor is unreachable here even
though it is applied elsewhere in the scored composition.

## Family F rounding-boundary disclosure

One published rounding unit on the x axis is 0.001 tokens/s/user. The complete
set of local coordinates on the negative side of a published boundary and
within that unit is rows 1, 2, 6, 7 and 9.

| External row | Exact local x | Published x | Local minus published | Selected step-frontier row |
|---:|---:|---:|---:|---:|
| 1 | 56.031756507665577 | 56.032 | -0.000243492334423 | 2 |
| 2 | 59.323892575183358 | 59.324 | -0.000107424816642 | 3 |
| 6 | 119.637922512881872 | 119.638 | -0.000077487118127 | 7 |
| 7 | 127.063674362471744 | 127.064 | -0.000325637528251 | 8 |
| 9 | 168.130792452674825 | 168.131 | -0.000207547325164 | 10 |

Only row 9 fails the unchanged quotient band, but all five boundary crossings
are published rather than selecting the failed row after the fact.

## Corrected Family M scope

The Family M quantity is the capacity-step quotient between a packet-priced
network and an unpriced network. The unpriced arm charges exactly zero network
service. It is therefore not a physical ideal-network reference, and the
1.042715399805 maximum prices the complete packet-network term against a
planner class that prices no network term at all. The maximum occurs on rows 1
and 3; the figure selects row 3 for the arrow.

The optional third arm that would charge the LogGOPSim latency, overhead, gap
and per-byte gap terms did not run. No isolated receiver-side serialization
claim is made. The separate 7.678 to 8.110 eight-into-one fan-in envelope from
`frontier_ladder_v1` and `loggopsim_acceptance_v1` remains a different schedule
regime and is not plotted on these curves.

Physical bounds still constrain the packet observations. The 458,752,000-byte
key/value payload has a 2.293760 ms sender floor at TP4, a 4.587520 ms receive
floor at TP2 and a 9.175040 ms one-link serialization reference. The packet
services are 4.661283 ms at TP2, 2.331683 ms at TP4 and 2.331766 ms at TP8.
Those values sit about 1.6 percent above their endpoint floors and below the
one-link reference. The unpriced arm's zero service lies below every physical
serialization floor by construction, which is why its quotient cannot isolate
one packet mechanism.

## Corrected figure and records

The figure arrow label is exactly:

```text
Their planner class prices no network cost.
Our unpriced-network arm charges zero network service.
This workload: packet-priced / unpriced-network
= 1.042715399805.
Unpriced: MEASURED-EXTERNAL.
Packet-priced: MEASURED-EXTERNAL + SIM-DERIVED.
```

Panel (c)'s two-line quotient label is fully inside the left margin. The PNG
and a 240 dpi rasterization of the PDF were inspected at full resolution after
the final render; the panel titles, annotation, legend, axes and caption are
unclipped.

- [Matched-seam frontier PDF](figures/matched-seam-frontier.pdf), the primary
  vector figure.
- [Matched-seam frontier PNG](figures/matched-seam-frontier.png), the
  full-resolution raster rendering.
- [record.json](record.json), the portable corrected record and complete
  scored-value trace.
- [results.csv](results.csv), the LF-only fatal-guard and family ledger.

The two-panel [publication PDF](figures/matched-seam-frontier-publication.pdf)
and [PNG](figures/matched-seam-frontier-publication.png) label the external
series as AIConfigurator 0.11.0.

## First published run, retained void evidence

The statements and numbers in this section reproduce the first publication for
auditability. Its former nonvoid verdict, guard passes and task closure are
invalidated by FG-1 and are not evidence for closing any task.

What ran: `matched_seam_frontier_v1` evaluated the 25 aggregate and 10
disaggregated topologies in the tracked external tables, composed all service
from the imported Qwen3-32B-FP8 operation database, ran the TP4-to-TP2, TP4 and
TP8 key/value redistribution cells twice through `rnic-nn`, repeated the live
external SDK oracle process twice, scored the frozen families and rendered the
overlay.

What came out: the run is nonvoid and MIXED. All six fatal guards hold. Family
S passes 13 of 13 exact service identities, Family R passes 10 of 10 published
decode-step bands, Family F passes 12 of 13 frontier rows, Family M passes 2 of
2 mechanism checks, and Family W passes 1 of 1 wall-time checks. The deciding
miss is F-2-09: its frontier quotient is 0.607495 against the frozen lower bound
of 0.75. The exact row-9 point is at 168.130792 tokens/s/user, microscopically
left of the external table's three-decimal 168.131 value, so the established
step-frontier lookup advances to row 10. The band fires and remains unchanged.

What it changes for the project: Family R reproduces the external decode
composition on every row, and Family D decomposes the 196.423 ms TTFT into a
99.203805 ms raw prefill pass, a 9.920380 ms prefill correction, an 87.299348 ms
autoscale correction and a -0.000533 ms table-precision reconciliation.
DEPLOY-12 therefore closes because the service-versus-TTFT premise is replaced
by a source-identified decomposition. DEPLOY-13 owns the rounded-axis
step-frontier residual exposed by F-2-09. COMP-88 owns moving the serving
correction constants from this study-local frozen configuration into the
content-addressed imported composition provenance.

What it does not change: the result does not validate either planner against
hardware, does not close a calibration task, does not import another system,
backend, model or database version, and does not turn the external database
into a runtime timing authority. The F-2-09 refutation remains in the published
record. The aggregate external curve remains display evidence only.

The expectations were frozen at `4c7ec88`, the live-SDK Family S oracles at
`5760301`, the binding at `9e6782d`, and the scan implementation at `3e752d5`.
The publication is append-only attempt `attempt-0002`, named portably in the
record through `SIMLLM_P3T_T1_BULK_ROOT`.

## Physical sanity before the headline values

Each check states the independent bound first, then reads the study value.
Agreement with a bound is necessary but is not treated as proof.

### Decode service and Family R

Memory floor first: one FP8 pass over 32 billion parameter bytes costs at least
3.333 ms at TP2, 1.667 ms at TP4 and 0.833 ms at TP8 on 4.8 TB/s HBM. The
measured-external decode services are 11.904 to 17.847 ms at TP2, 6.849 to
9.179 ms at TP4 and 4.946 to 8.359 ms at TP8, all above the corresponding
weight-stream floor.

Table-precision ceiling next: a value rounded to 0.001 ms can differ from its
unrounded source by at most 0.0005 ms. At the shortest published step, 4.946
ms, that permits a quotient only in [0.999898908, 1.000101092] if both sides
use the same composition. The observed quotients span 0.999946609 to
1.000076345. The maximum departure is 76.345 parts per million, inside both
that napkin range and the frozen [0.98, 1.02] band.

### Prefill service and Family D

Compute floor first: the usual dense-transformer lower estimate is two FLOPs
per parameter per token. For 32 billion parameters, 3,500 uncached prompt
tokens, TP4 and 1.979 PFLOP/s per H200, the pass cannot beat
`2 * 32e9 * 3500 / 4 / 1.979e15 = 28.297 ms`, even before attention work and
nonideal utilization. The causal ceiling for an isolated prefill term in this
decomposition is the published 196.423 ms first-token boundary. The imported
raw pass is 99.203805 ms, between those limits.

The independent HBM floor is only 1.667 ms for one TP4 shard of FP8 weights,
so compute and operation composition, not a single weight read, set this
prefill result. Applying the source tool's 1.1 prefill correction gives
109.124185 ms. Applying its 1.8 autoscale factor gives 196.423533 ms, only
0.000533 ms above the three-decimal table value. That closes the previously
conflated semantic premise without calling the table residual queueing.

### Packet seam and Family M

Byte floor first: the uncached FP8 key/value state is
`64 layers * 2 tensors * 8 KV heads * 128 elements * 3500 tokens * 1 byte =
458,752,000 bytes`. Four TP4 send links need at least 2.293760 ms at 400 Gb/s.
TP2 receive fan-in raises the endpoint floor to 4.587520 ms; TP4 and TP8 remain
sender-limited at 2.293760 ms. A single 400 Gb/s link serializing the complete
payload would take 9.175040 ms, which is the conservative wire-only upper
reference for these split transfers.

The packet simulator returns 4.661283 ms for TP2, 2.331683 ms for TP4 and
2.331766 ms for TP8. Each is about 1.6 percent above its endpoint floor and
well below the one-link reference. The largest capacity-controlling packet to
ideal step quotient is 1.042715400 on disaggregated rows 1 and 3. No packet
point is faster than ideal, and the maximum exceeds the frozen 1.02 mechanism
threshold.

### Frontier axes and end-to-end plausibility

For the reference TP4 decode row, the raw decode-capacity ceiling before the
source tool's 0.92 degradation is 653.666868 tokens/s/GPU. Applying that
declared factor gives 601.373518 tokens/s/GPU. The independently composed
prefill ceiling is 644.334708 tokens/s/GPU, so decode is the limiting pool.
The recorded ideal point is therefore 601.373518 tokens/s/GPU at
108.944478 tokens/s/user. The external table gives 602.586 and 108.944,
respectively, which is physically plausible and exposes the small serving
composition difference without exceeding raw decode capacity.

## Fatal guards

| Guard | Outcome | Meaning |
|---|---|---|
| FG-1 | PASS | Every positive duration in the ideal scored arm is `MEASURED-EXTERNAL`; packet additions are isolated to Family M. |
| FG-2 | PASS | Service values and positive ideal stamps retain the frozen slice identity and never become `MEASURED`. |
| FG-3 | PASS | The external tables and parity publication artifacts are byte-identical before and after. |
| FG-4 | PASS | The expectations and service-oracle commits precede the binding and run. |
| FG-5 | PASS | The external TTFT appears only in Family D and is never equated with isolated prefill service. |
| FG-6 | PASS | Local scored quantities and live SDK service oracles repeat bit-equal in fresh processes. |

No guard voids the run.

## Family S: service identity

All ten decode cells and all three prefill cells are bit-equal across the
frozen live-SDK oracle, the imported pass model and the deployment service
consumed by the frontier. Family S passes 13 of 13.

## Family R: published decode composition

| Row | Decode configuration | Ours (ms) | Published (ms) | Quotient | Outcome |
|---:|---|---:|---:|---:|---|
| 1 | TP2, batch 128 | 17.847022159 | 17.847 | 1.000001242 | PASS |
| 2 | TP2, batch 112 | 16.856614706 | 16.857 | 0.999977143 | PASS |
| 3 | TP2, batch 48 | 11.903905722 | 11.904 | 0.999992080 | PASS |
| 4 | TP4, batch 64 | 9.178987488 | 9.179 | 0.999998637 | PASS |
| 5 | TP4, batch 56 | 8.936046839 | 8.936 | 1.000005242 | PASS |
| 6 | TP8, batch 64 | 8.358553701 | 8.359 | 0.999946609 | PASS |
| 7 | TP4, batch 26 | 7.870069908 | 7.870 | 1.000008883 | PASS |
| 8 | TP4, batch 16 | 6.849218381 | 6.849 | 1.000031885 | PASS |
| 9 | TP8, batch 20 | 5.947750471 | 5.948 | 0.999958048 | PASS |
| 10 | TP8, batch 9 | 4.946377602 | 4.946 | 1.000076345 | PASS |

Family R passes 10 of 10. The result identifies the external 9.179 ms row as
the source composition over the imported database, not as the earlier
11.102129 ms parity oracle whose batch-64 configuration used different
composition parameters.

## Family F: frontier overlay

F1 passes: the worked row uses `1e12 / decode_step_ps` on x and exact request
capacity times 500 output tokens divided by used GPUs on y. F3 passes with ten
strictly monotone ideal-frontier points. F4 passes because no point answers
more than two external rows.

F2 passes 9 of 10:

| External row | Step-frontier answer | Quotient | Frozen [0.75, 1.35] |
|---:|---|---:|---|
| 1 | row 2 | 0.988204705 | PASS |
| 2 | row 3 | 0.841593217 | PASS |
| 3 | row 3 | 0.999985579 | PASS |
| 4 | row 4 | 0.997987869 | PASS |
| 5 | row 5 | 0.997969861 | PASS |
| 6 | row 7 | 0.876367649 | PASS |
| 7 | row 8 | 0.888876990 | PASS |
| 8 | row 8 | 0.999985967 | PASS |
| 9 | row 10 | 0.607495219 | **REFUTATION** |
| 10 | row 10 | 0.998085815 | PASS |

The row-9 miss is not softened. The published x coordinate is 168.131 while
the exact local coordinate is 168.130792. Treating the rounded published value
as an exact threshold excludes the matching row and selects row 10. DEPLOY-13
owns a separately frozen interval-aware or unrounded comparison; this record
continues to report the original band as failed.

## Family M: mechanism isolation

M1 passes because every packet-to-ideal capacity-step quotient is at least
1.000000. M2 passes because the maximum is 1.042715400, above 1.02. The
packet curve contains nine Pareto points; the ideal curve contains ten. Both
use the same measured-external compute services, candidate grid and rate
composition. Only the `SIM-DERIVED` packet redistribution term changes.

## Family D: TTFT decomposition

| Term | Value (ms) | Evidence |
|---|---:|---|
| Raw imported TP4 batch-1 prefill pass | 99.203804745 | `MEASURED-EXTERNAL` |
| Source 1.1 prefill correction | 9.920380474 | frozen external SDK composition |
| Source 1.8 TTFT autoscale correction | 87.299348175 | frozen external SDK composition |
| Three-decimal table reconciliation | -0.000533395 | tracked external table precision |
| Published TTFT | 196.423000000 | `MEASURED-EXTERNAL` display row |

The 97.219195255 ms residual from the raw prefill pass is fully attributed
within the table's published precision. It is not labeled queueing.

## Family W and evidence separation

The complete two-process local scan, two-process live-SDK check, packet cells,
scoring and figure took 76.285396 seconds against the 600 second ceiling.
Family W passes 1 of 1.

Fatal guards, S, R, F, M, D and W remain separate registers. D is unscored.
No denominator is summed across families.

## Figure and records

- [Matched-seam frontier PDF](figures/matched-seam-frontier.pdf), the primary
  vector figure.
- [Matched-seam frontier PNG](figures/matched-seam-frontier.png), the print-size
  raster rendering.
- [record.json](record.json), the portable strict result and complete curve
  traceability record.
- [results.csv](results.csv), the LF-only guard and family ledger.

The revised figure makes the matched seam and its residual mechanism visible
in three linked treatments. Panel (a) overlays the external aggregate and
disaggregated curves with the SimLLM contention-off and contention-on curves;
the legend names every evidence class. Panel (b) zooms external row 3, where
the external disaggregated and SimLLM contention-off points coincide, and the
arrow points from contention off to contention on. It identifies the unpriced
term as receiver-side serialization under fan-in, where several senders can
otherwise deliver into one receiver at full rate at once and exceed its ingress
bandwidth. Its 1.042715399805 quotient is this workload's measured separation.
It is not the 7.678 to 8.110 eight-into-one incast envelope from the separate
`frontier_ladder_v1` and `loggopsim_acceptance_v1` schedule regime, which is not
drawn on these curves. Panel (c) shows the Family R row quotients from
0.999946608534 to 1.000076344974 against the frozen [0.98, 1.02] band. The
caption retains the F-2-09 rounded-axis refutation at 0.607495219355.
