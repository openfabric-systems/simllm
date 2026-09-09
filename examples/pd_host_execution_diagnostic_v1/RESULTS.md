# Native host execution diagnostic

The fresh paired diagnostic is **COMPLETE**: all 28 native requests retain
identical simulated timing and output across the uninstrumented and
instrumented arms. In the two four-request cells, dependency checking consumes
35.4 to 36.1 percent of the enclosing process CPU time as exclusive measured
main-thread work. This identifies a host optimization target. CORE-52 remains
open because this diagnostic does not execute its 56-engine deployment.

## What ran and what remained exact

Two fresh processes each execute four historical controls, then prompt lengths
8 and 16 crossed with serial request counts 1 and 4. Each process retains two
native engines with eight simulated workers each. The instrumented arm adds
passive phase timers and a complete function profile for its two single-request
cells. Neither arm executes a GPU kernel or a packet backend.

All nine protocol stages complete. The evidence consists separately of 114,208
unscored fatal guards, 37 exact oracle rows and 10 rejected semantic corruption
controls. The exact rows include 18 per arm and one complete pair comparison
covering all 14 requests. There are zero scored behavioral instances and the
behavioral score is null. Each arm completes 70 scheduler steps and 98 clock
advances, including 14 unchanged equal-time request admissions.

The frozen per-request job completion times are 507,232,000 ps for prompt 8 and
526,840,000 ps for prompt 16. Four serial requests take exactly four times
those durations. Every request, step, sink outcome and handoff retains its
expected value; opaque native request identities use only the frozen
engine-qualified normalization.

Before execution, the compute floor under the selected all-resident-weight
surrogate is 40,108,032 ps per decode step. The observed declared services,
77,952,000 and 77,976,000 ps, exceed it. This is a conditional surrogate bound,
not a routed-expert hardware bound. For 393,216 and 786,432 transferred cache
bytes, serialization at 400 Gb/s gives floors of 7,864,320 and 15,728,640 ps;
the frozen 10 Gb/s plus 50 microsecond comparison ceilings are 364,572,800 and
679,145,600 ps. The declared 100,000,000 ps handoff lies inside both envelopes.
These checks constrain the simulated timeline without turning host execution
time into GPU service or proving a production deployment rate.

## Host attribution

| Prompt tokens | Requests | Uninstrumented wall, seconds | Instrumented wall, seconds | Detailed profile |
|---|---|---|---|---|
| 8 | 1 | 16.284 | 32.448 | yes |
| 8 | 4 | 64.427 | 64.359 | no |
| 16 | 1 | 16.362 | 32.242 | yes |
| 16 | 4 | 67.492 | 72.209 | no |

These are enclosing cell call times. The complete processes take 253.914 and
294.193 seconds, including construction and historical controls. Sampled
maximum current resident memory is 1,085,304 and 1,087,864 KiB respectively.
The detailed profiler roughly doubles the single-request call time. No
profiler correction, repeatability interval or predicted speedup is fitted.

In the passive four-request cells, the GOAL dependency checker spends 23.220
and 25.516 seconds of exclusive main-thread CPU time. Dividing by the enclosing
process CPU time yields 36.0966 and 35.3843 percent. The numerator and denominator
have different thread scopes; these descriptive shares are not an additive
wall-time decomposition. Other substantial selected phases are graph
projection, locality validation, locality planning and graph validation.

The complete profiles each contain 1,513,185 calls to `_goal_edge` across five
model steps. The source repeatedly scans the effective edge list inside the
serialized-edge verification loop. A separately frozen, call-local edge index
can test whether those repeated conversions can be removed while preserving
all rejection rules and exact simulated outcomes. No such optimization runs
in this diagnostic. The profiles also contain 5,520 calls to
`_validate_collective_plan` per single-request cell. Their cumulative times
include other functions and must not be added to dependency-check times.

For comparison, the selected native schedule, update and output calls total
6.767 and 7.130 milliseconds of wall time in the four-request cells. Progress
persistence, including the unchanged durable write, totals 10.858 and 11.166
milliseconds. These selected subtotals explain why graph work deserves the
next controlled experiment; they do not represent all frontend work.

## Frozen protocol, retained failure and source identity

The original finite protocol was frozen in expectations-only commit
`8f06648d26bbfb2bfff4751d29a2e2bf232b440d` before implementation and execution.
Its first attempt, source `51ed0d1ef5fb86a2fc643dfba3949f67627c3659`, remains
**VOID** with zero admitted arms. The uninstrumented process finished 14
requests in 260.017 seconds, but the checker incorrectly required compact
sorted JSON from the native step writer, which uses standard spaces and
schema insertion order. The instrumented arm was not started.

The source-writer encoding amendment is
`11d7bb0c28a166939ebab0ec0367430afa57f2a8`. It follows that finding and precedes
the parser repair and this fresh run. Its encoding assertions are
post-specified regression checks, not publicly preregistered predictions. The
finite grid, timing oracles, limits and profiling boundary are unchanged.
The fresh executed source is
`3abec1cb91679f8582130fd067b70ad4412d94fa`.

The fresh summary has SHA-256
`b0705dfe482c0cc83e0e8a99079c03dcfa05d1e199a054d059f1790685ab5be2`.
The retained first summary has SHA-256
`ae25e0fdf6f75cf14a66bffc50ce080e15f276b2766a9538ee74e5cf2019779b`.
The [compact publication](results.json) records both attempts, including all
33 fresh raw files totaling 52,299,757 bytes and the 16 first-attempt files
totaling 24,977,530 bytes. Full function identities, caller tables, raw
profiles, source receipts and native records remain in external evidence.
The failed run is never rescored.

## Project effect and limits

CORE-52 gains a qualified host attribution diagnostic and can proceed to a
separately frozen optimization and successor target campaign. Its original
1,200-second target run remains VOID. CORE-68 retains independent-engine
completion; CORE-51 and CORE-54 retain deployment and frontier obligations.
This result closes no deployment task, calibrates no GPU service and supports
no extrapolated 56-engine throughput claim.

Before the fresh execution, 87 focused software checks passed in 4.34 seconds
and Ruff passed. Complete software and continuous-integration gates remain
pending for publication.
