# TRAF-66 finite two-batch boundary expectations

Status: expectations only and protocol void. No visible calibration comparison
has been performed.

## Frozen boundary form

Let `C` be the existing full-step component compute service and `P` the
existing full-step packet service. The pinned schedule creates exactly two
children. Each child therefore conserves `C / 2` compute service and `P / 2`
packet service. The finite schedule contains an exposed prologue half-service,
one steady interleave of the two child halves, and an exposed epilogue
half-service:

`T2(C, P) = C / 2 + max(C / 2, P / 2) + P / 2`

so, equivalently,

`T2(C, P) = max(C, P) + min(C, P) / 2`.

The factor one half is fixed by the source-backed child count. It is not a
free overlap fraction. No dispatch phase is subtracted. For the frozen 1K
component envelope, `P > C` at every edge, so the form reduces to
`P + C / 2`. The exposed boundary is exactly `C / 2`, and the total service
envelope is 2,967,804,740,360 to 2,967,804,742,680 ps.

## Event conservation

The pinned prefill strategy has one dispatch launch, one dispatch completion,
one combine launch, one combine completion and two yield boundaries in each
sparse layer. Across 58 sparse layers this is, per child, 58 launches and 58
completions for each phase, 116 yields and 117 stages. Across both children it
is 116 launches and 116 completions for each phase, 232 yields and 234 stage
advances.

The pinned `tbo_delta_stages` is zero. The executor therefore performs no
stage-index prologue advances, runs all 117 stage indices in the central loop
as child A followed by child B, performs no stage-index epilogue advances, and
asserts that both children are complete before their outputs merge. The
service prologue and epilogue above are resource boundaries, not extra source
events: their child halves plus the central maximum conserve exactly `C`
compute service and `P` packet service.

## Frozen signed movement

The finite form adds a positive `C / 2` boundary in the selected
packet-dominant regime. Before reading the visible 1K target, it therefore
predicts higher service and lower throughput than COMP-75. For any positive
target, the signed calibration residual must move in the negative direction.
The known proximity between the existing service surplus and one dispatch
phase is not used in this derivation and gives no authority to subtract that
phase.

For the forbidden 2K and 4K compute rows, the structural prediction is plain:
prompt-length dependence re-enters through `compute_service_ps / 2`. While the
packet-dominant branch holds, total service is
`packet_service_ps + compute_service_ps / 2`; increasing compute service lowers
predicted throughput even though packet service remains the larger resource.
No held-out value is computed or compared here.

## Protocol disposition

The source allowlist extension was committed as `dcf6be1` before the exact
pinned ranges were read. No framework evaluation table, web page, model
weight, scored comparison or flagship rerun was accessed or executed.

During later local component inventory, however, the complete
`candidate-record.json` was opened instead of a calibration-only projection.
That file contains a held-out component row. The value is excluded from the
derivation and is not compared, but the exposure makes the held-out ledger
nonempty and prevents literal TRAF-66 closure. The result must register a clean
repetition remainder.
