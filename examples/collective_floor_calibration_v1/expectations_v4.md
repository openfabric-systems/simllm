# Collective floor completion freeze

This fourth freeze governs the TRAF-76 completion wave. It follows the
published [miss map](MISS_MAP.md), commit `eb889b9`, and precedes every new fit,
behavioral implementation, and result-producing run in this wave. The first
three expectation files and `study_config.json` remain immutable.

The original 63 Family H training cells and 63 holdout cells remain exactly
the same. No held-out latency may influence an anchor value, interpolation
coefficient, transition magnitude, fit decision, or retry. Family H keeps the
larger of 10 percent or two H200 GPU cycles. Family D8 keeps `[0.90, 1.10]`.

## Pre-fix location finding

Attempt 0004 misses 12 cells: seven all-gather and five reduce-scatter, across
ranks 2, 4 and 8, from 8 KiB through 32 MiB. They do not form one payload
cluster. A single universal dip correction is therefore forbidden.

The rank-8 middle of the table contains local reversals, but the misses do not
cluster in that region. The prior GH200 width-2 all-reduce finding of a 22
percent serialization-bandwidth dip therefore does not justify an H200
coordinate-specific regime or correction. This wave does not use it.

## Leg A: training-only anchor authority

The replacement is an opaque aggregate-completion curve for exact-domain,
fully intra-node use. It does not rename completion-table latency as packet,
credit, switch, launch, synchronization, or registration service.

The model is frozen in `study_config_v2.json`:

- An exact same-operation, same-rank and same-byte training anchor is returned
  unchanged when one exists.
- Otherwise, when the complementary ring operation has an exact same-rank and
  same-byte training anchor, that observed anchor carries the local regime.
  Its value is multiplied by the ratio between the requested operation's and
  complementary operation's affine training-only trends at that byte count.
  Each trend uses the adjacent same-operation, same-rank training anchors.
- Otherwise, including the D8 off-grid coordinate, the value is affine
  interpolation of latency on the physical byte axis between adjacent
  same-operation, same-rank training anchors. Below and above the training
  ladder, the nearest two anchors define affine extrapolation.
- The paired-operation rule is one model form at every eligible coordinate,
  not a list of exceptions selected from Family H. It follows the physical
  symmetry of ring all-gather and reduce-scatter: both exchange the same
  endpoint bytes, while the ratio of their local training trends retains the
  operation-dependent completion cost.
- Every numeric latency entering the exact anchor, paired-operation correction
  or fallback interpolation is one of the original 63 training cells. No H200
  Family H holdout enters the model.
- The exact-domain curve is a complete opaque completion. It is charged once
  and has no separately named byte-serialization component. It is legal only
  when the represented collective is fully intra-node. A fabric-bearing use
  must fail closed or deliberately select the existing decomposable transfer
  surrogate with its existing acknowledgement.
- The existing positive floor-plus-byte-slope surrogate remains the exact
  compatibility authority for explicit donor transfers. Its fitted values,
  evidence downgrade, and accepted mixed-locality consumer behavior do not
  change in this wave.

Family H scores the same 63 holdouts once. All 63 must meet the unchanged bar.
The report publishes the prior 51 of 63 tally, the new tally, median and p95,
and every per-cell before and after error. A miss is a refutation. No second
model may be selected from those results without another expectations-only
commit.

Physical sanity precedes the score. The floor for every cell is ring endpoint
bytes divided by 450 GB/s. The source provides no finite algorithm-progress
ceiling, so the ceiling is unbounded. Every completion must remain positive,
and no reported serialization term may be negative because the exact-domain
model exposes no such term.

## Leg B: D8 mechanism test

The scored coordinate remains 196,608 operation-buffer bytes per phase. The
external 1.922050 ms over 65 layers remains unchanged. The 172,032-byte
physical endpoint reading remains diagnostic only.

The named mechanism under test is model-form bias. Attempt 0004 forced the
rank-8 all-gather and reduce-scatter neighborhoods into two broad positive
floor-plus-slope regimes. The source table changes direction inside those
neighborhoods, so their separate biases add at the D8 coordinate. The new
anchor authority removes that broad-regime approximation for a fully local
exact-domain query. It does not add or tune a D8-specific constant.

D8 passes only when the new matched-coordinate quotient lies inside the same
`[0.90, 1.10]` band. A value outside the band is a sustained refutation. The
report names the all-gather and reduce-scatter contributions, the physical
floor, and the change from 1.109143050. Neither the band nor the external arm
moves.

## Leg C: packet-mechanism feasibility and frozen cells

The packet leg is attempted only if the existing evidence identifies the
parameters required to instantiate H200 credits, H200 product geometry,
NVSwitch queue behavior, and product arbitration without borrowing A100
candidates or fitting several mechanisms to one opaque completion number.

The cell matrix is frozen before that decision:

| Family | Participants | Receiver fan-in | Payloads per sender |
|---|---:|---:|---:|
| PZ | 2, 8 | 0 | 65,536 and 1,048,576 bytes |
| PN | 4, 8 | 3, 7 | 65,536 and 1,048,576 bytes |

For every cell, publish aggregate phase completion before and after, raw
application and wire bytes, completion order, credit waits, switch waits,
arbitration grants, and the one timing authority. The error band is the same
larger of 10 percent or two H200 GPU cycles against an independently matched
H200 observation. PZ and PN remain separate evidence families.

Packet fatal guards are:

- PC-FG-1, identifiability: every numeric credit, queue, link, port, switch and
  arbitration value names evidence that independently identifies it. One
  aggregate completion table cannot identify several internal terms.
- PC-FG-2, prerequisite structure: the model uses generation-scoped flits,
  receiver-owned credit release, explicit traffic class and virtual channel,
  replay identity, receive ordering, input ports, virtual output queues and a
  two-sided crossbar policy seam. A fixed 272-byte slot, sender timer, implicit
  virtual channel or flat FIFO voids the run.
- PC-FG-3, one authority: the packet path alone advances each packet and the
  aggregate row is a read-only comparison. The aggregate and packet paths
  never both advance or charge the same collective.
- PC-FG-4, no double count: packet bytes, link service, switch service, receive
  service, aggregate completion, registration and host launch each have one
  owner. An enabled packet cell carries zero aggregate charge.
- PC-FG-5, identity off path: disabling the packet mechanism preserves every
  accepted pre-wave timestamp, application and wire byte count, completion
  order, backend invocation order and random draw exactly.
- PC-FG-6, exact packet conservation and deterministic fresh-process replay.

If PC-FG-1 or PC-FG-2 cannot be made decidable from the current evidence and
tree, the packet family does not run and no before/after denominator is
reported. That is a scope result, not a skipped pass. The final publication
must register the residual packet work in the traffic module with its evidence
and prerequisite dependencies.

## Shared fatal guards and chronology

The prior FG-1 through FG-7 remain binding for the old surrogate and bypass.
For the new exact-domain authority:

- A-H-FG-1: every served exact-domain value records its training anchors and
  whether it used an exact anchor, paired-operation correction or fallback
  interpolation.
- A-H-FG-2: the original training and holdout identities are disjoint and
  byte-identical to `study_config.json`.
- A-H-FG-3: a holdout value is loaded only after the authority is fully built
  and serialized. Hash the serialized authority before Family H is read.
- A-H-FG-4: the old positive floor-plus-slope donor transfer remains
  byte-identical at every fitted regime and published MiniMax query.
- A-H-FG-5: the pre-wave default-off golden remains byte-identical.
- A-H-FG-6: two complete fresh-process evaluations match after excluding only
  the named wall-time field.
- A-H-FG-7: this expectations-only commit precedes implementation and the first
  result-producing run.

Any failed fatal guard makes the complete run void. Fatal guards are never
added to a scored denominator.

## Disposition rule

TRAF-76 closes only if Legs A, B and C all complete under their literal bars.
If A and B pass but the packet leg stops at PC-FG-1 or PC-FG-2, the final
publication narrows TRAF-76 honestly and registers the packet residual under
one of the IDs reserved by the task brief. No other ID may be consumed.
