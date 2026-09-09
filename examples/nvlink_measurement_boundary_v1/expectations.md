# NVLink measurement-boundary audit expectations

TRAF-91 audits whether the historical incast records identify a packet-service
error. The review of those already observed records is post-specified. This
commit prospectively freezes only the new synthetic counterexamples and their
checks, before implementing or executing the audit runner. It does not register
a new physical capture or fit a model parameter.

## Packet-free counterexample

Let a transfer contain B bytes, travel at constant rate r bytes per second,
and pay fixed overhead a seconds. It has no packet headers, retries or credits:
T(B) = a + B/r. Compare it with the same rate without that overhead. The signed
goodput error is E(B) = r*a/B. Therefore E(B)-E(2B) = r*a/(2B).

Floor: transfer time cannot be less than B/r. Ceiling: with this sole fixed
service and no other work, transfer time is exactly a+B/r. These statements
precede every generated value. Sweep base sizes 4 and 8 MiB, paired with their
doubled sizes, rates 50 and 100 GB/s, and overheads 0, 5 and 10 microseconds.
Use rational arithmetic for the independent relations. Feed synthetic rows to
the unchanged historical attribution function, with all three required degree
labels supplied as repeated classifier inputs, not physical multi-source runs.

Expected relation: doubling bytes halves the overhead-induced error, and
doubling positive overhead doubles it. The historical five-percentage-point
size rule labels packet-free cells as packetization whenever r*a/(2B)>0.05.
That is a counterexample to identification, not a proposed explanation or fit
for the captured low rate. The zero-overhead control is unscored.

## Timing-boundary counterexample

For k equal-duration source visits, let duration be d and source starts be
0, s, ..., (k-1)*s. The maximum local duration is d. The actual common phase
length is d+(k-1)*s. Sweep k at 2 and 3, d at 100 and 200 microseconds, and s
at 0 and 20 microseconds. No transport model or real device participates.

Floor: the common phase cannot be shorter than any source visit. Ceiling: in
this fully specified construction it is exactly d+(k-1)*s. At positive s,
dividing total bytes by max(local durations) overstates common-phase goodput
by exactly 1+(k-1)*s/d. The zero-stagger identity is an unscored guard.
Increasing k increases the missing span; doubling d halves the relative error.

## Historical and source audit

Keep the locked historical JSON, expectations, scorer, producer and campaign
source byte-identical. Reconstruct every published launch-skew fraction from
the declared per-additional-sender budget and recorded local durations. Check
whether those records contain an observed common-clock source-start interval.
A budget-derived fraction cannot establish observed alignment. Missing timing
observations make that hardware validity precondition undecidable, which voids
its use for model qualification. Retain original verdicts as historical records.

Inspect the producer's pacing coordinate, the source/destination memory access,
the per-device event interval, the host batch interval and the final-value
ordering ledger separately. Reconstruct the first warp's nominal pacing
deadline at both captured sizes and all three degree labels using the source
hash function. This is derived code behavior, never a measured wait or packet
count. No expected hardware duration is inferred from it.

## Evidence and acceptance

Report synthetic configurations, exact-oracle rows, nonzero relation families,
historical records and unscored guards separately. A violated preservation,
identity, finite-time or independent arithmetic guard voids the synthetic audit
and removes its behavioral score. A confirmed counterexample refutes the old
identification claim while the synthetic audit itself passes.

Close TRAF-91 only after the unsupported current claims are superseded and
TRAF-86 owns the unresolved producer, memory, clock and transport discrimination.
Publish a four-card measurement request with explicit clock domains, producer
controls, topology qualification, raw observations and rejection conditions.
Its reservation status is awaiting-reservation and it contains no measured
timing values. A separate pre-capture freeze must pin the runnable producer,
instrumentation uncertainty and physical acceptance bands before hardware use.
