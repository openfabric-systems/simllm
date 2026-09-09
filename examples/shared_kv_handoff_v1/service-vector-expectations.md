# Compatibility service-vector correction expectations

This expectations-only amendment precedes the checker correction and a fresh
complete campaign. The [first campaign](VOID_RESULTS.md) remains VOID. Its
observations exposed the erroneous contiguous-service assumption, so this
correction is not described as having been specified before that observation.
The old execution is never rescored or substituted for a fresh process.

## Service, waiting and token visibility

The existing frozen native reference schedules serialized engines in their
deterministic ready order. A request can wait between consecutive decode
steps while another engine runs. Its token-to-token interval includes that
waiting; an individual service interval does not.

For each compatibility request and pool role, select the existing reference's
individual service steps by request index, role and engine identity. Add the
batch admission time to each reference start and finish. Require every native
owned step to equal the resulting ordered tuple of start, finish and
finish-minus-start. Request endpoints and the complete source-pair comparison
retain their existing exact checks. Do not infer a later service start from
the previous token's completion, and do not infer service from a token gap.

The packet-backed independent path retains its original contiguous service
check and native all-shard release schedule. No production scheduler, compute
price, clock, network law, native source, physical bound, resource limit,
workload or comparison projection changes. All eighteen fresh processes,
forty requests, twenty oracle vectors and twelve behavioral instances in
three families remain mandatory.

## Regression controls and chronology

Software controls cross serialized/independent timing, zero/positive handoff,
two declared decode service constants and zero/nonzero admission time. Two
ready serialized decode engines each retain the declared single-step service
while alternating token intervals include the other engine's turn. Independent
engines retain their isolated per-step and token intervals. Offsetting admission
translates both endpoints equally and leaves service unchanged.

Missing, duplicated, wrongly owned, early-start, changed-finish and
wait-folded-into-service step vectors remain fatal. These controls are
post-specified regression evidence for the exposed reader defect and prospective
checks for the fresh campaign. They are unscored and do not enlarge any native
behavioral denominator.

Keep the original expectation files, prior VOID report and receipt byte-locked.
Record this amendment's commit with the original, deadline and identity freezes
in the fresh result. Only a complete valid fresh campaign can close CORE-71.
