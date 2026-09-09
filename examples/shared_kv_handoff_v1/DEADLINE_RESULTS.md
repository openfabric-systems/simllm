# Complete exchange deadlines

Eight fresh child processes returned all sixteen complete seven-byte replies
unchanged across the legacy and optional deadline paths. The
`shared_kv_handoff_v1.deadlines` component study varies the exchange allowance
between one and two seconds and the client idle interval between three and
six seconds. Its four source-bound configuration pairs pass. This qualifies
the host communication prerequisite for CORE-71; the forty-request shared
packet and native serving campaign remains required.

The child must remain alive while the serving process does unrelated work.
Once a protocol exchange starts, writing the request and reading its complete
response share one fixed deadline. Reading a new part of the response cannot
refresh that deadline. A separate lifetime deadline still caps the child.
The optional exchange scope changes neither wire bytes nor simulated time.

The legacy session keeps its sixty-second lifetime and no exchange scope.
The shared study selects a 1,800-second lifetime with a sixty-second complete
exchange bound. This is a prospective correction to the original study's
description of the existing timeout, which covered the whole child lifetime.
It leaves the serving worker's 1,800-second and sixteen-gibibyte limits intact.

| Exchange allowance | Client idle interval | Fresh children | Complete replies |
|---|---|---|---|
| 1 second | 3 seconds | 2 | 4 identical |
| 1 second | 6 seconds | 2 | 4 identical |
| 2 seconds | 3 seconds | 2 | 4 identical |
| 2 seconds | 6 seconds | 2 | 4 identical |

Every pair uses a thirty-second lifetime. Its floor is the imposed idle
interval; its ceiling is that unchanged lifetime. Every captured process
finishes inside those bounds, retains the same live child for both replies,
and reaps it after completion. These are component conformance guards, not
behavioral points. The behavioral instance count is zero and its score is
null. The sixteen replies and four pairs are distinct inventories and are
not added together.

The [deadline freeze](deadline-expectations.md) is commit
`0e0e71aa586b2ccd2d786534d6b686d888e300b0`, preceding implementation and both
component grids. The retained second grid executes
`ec976fe51f18dcefc8ce1a6bc28283e2cb4670c3`. Its first-exit and final receipts
match for all 48 child files, totaling 8,662 bytes. The
[portable record](deadline-results.json) carries the source, interpreter,
freeze, raw summary and child-file hashes. The earlier positive grid at
`10d2c497d22da1f26b59544c9920b6225ce683e0` remains a separate retained result.
Neither grid is relabeled as having executed a later source.

Separate failure-path checks cover lifetime expiry, cumulative frame reads,
nested or overlapping scope rejection, interruptions, partial header and body
receipt retention, and cleanup failures. Commit
`1741465a6dcb51150219216772cec66d1337e97d` additionally preserves the first
partial timeout receipt across a rejected retry. The focused software gate
passes 147 tests with three skips; that count is not a study denominator.
Independent review reconciles the raw receipts and confirms that cleanup
preserves the first failure and reaps the child.

CORE-71 remains open for live packet-to-token evidence. CORE-70 retains actual
tensor-buffer ownership and broader shared-resource composition. This study
runs local echo children, with no GPU allocation, native serving request,
network calibration or large-model deployment claim.
