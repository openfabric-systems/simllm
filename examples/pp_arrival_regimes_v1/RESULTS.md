# Early-burst pipeline arrival qualification

The 72-execution arrival study is **VOID with findings**. Its ideal-reference
assumption fails: delaying the first pipeline release by 16 or 80 nanoseconds
leaves its first completion at **4,496,000 ps** under expert background load.
The declared shift is absorbed by the ideal scheduler's existing packet slot
calendar. This is a failed frozen precondition, so the run has no behavioral
score and cannot qualify TRAF-88.

TRAF-88 stays open for a new prospective qualification with a calendar-aware
ideal oracle and a receiver-admission hypothesis that covers the observed late
original packet. The earlier [queue study](../pp_rail_contention_v2/RESULTS.md)
remains a valid refutation of its all-depth penalty claim; its two-stage
absorption and deeper recovery observations reproduce exactly here. This
result does not close TRAF-8's captured serving integration, BACK-38's
persistent physical execution or any hardware-calibration milestone. It
measures one declared forward prefill request, with no time per output token.

## Scope and bounds

The pipeline sends 65,536-byte activations between four or eight stages.
Each stage computes for one microsecond; the first adds 16 or 80 nanoseconds.
Expert-parallel (EP) traffic uses separate network interfaces and width zero
or 32. Both rail and node-local attachments use eight leaves and two spines,
400 Gbit/s links, seed one and one microsecond of propagation per link.
The receiver window stays at 10.5664 microseconds, with 16-nanosecond ticks;
control headroom and exponential DATA recovery retain their explicit prior
settings. The added first-stage work changes arrival phase through the
existing graph. It is not external request-arrival support or GPU calibration.

Floor, frozen before execution: payload serialization costs 1.31072
microseconds, so each pipeline hop needs at least 3.31072 microseconds on rail
or 5.31072 microseconds on node-local attachment. The request floor is
`P*1 us + A + (P-1)*hop_floor`.

Ceiling, frozen before execution: the complete-flow engineering budget is
11.45762816 milliseconds for loaded node-local attachment and 16.7424512
milliseconds for loaded rail attachment. Payload service supplies no
unconditional finite completion ceiling in a model with loss and retries.
These operational budgets include the declared recovery allowance.

Every physical cell's byte, causal, receiver-prefix and engineering-budget
checks hold locally. Loaded node-local complete phases lie between 3.6602432
and 3.7356608 milliseconds; loaded rail phases lie between 6.5184832 and
6.5950944 milliseconds. Both stay below their budgets. Their background
phases also exceed the unchanged 0.5984608-millisecond ideal background phase.
These checks cannot rescue the aggregate after its ideal guard fails.

## Why the ideal control fails

The freeze requires every pipeline start and completion to shift by A, with
unchanged flow completion time (FCT), while all EP rows remain exact. Eight
loaded ideal controls violate this requirement. All 896 EP rows in each of
those controls remain unchanged; pipeline endpoint, tag and payload identities
also match their corresponding predecessor inventories.

| First pipeline flow under EP width 32 | Start (ps) | Completion (ps) | FCT (ps) |
|---|---:|---:|---:|
| Retained A=0 reference | 1,001,000 | 4,496,000 | 3,495,000 |
| A=16,000 ps | 1,017,000 | 4,496,000 | 3,479,000 |
| A=80,000 ps | 1,081,000 | 4,496,000 | 3,415,000 |

The following explanation is **post-specified**, not a replacement acceptance
oracle. The packetized ideal runtime shares a slot calendar across active
flows. EP traffic keeps it active even when the pipeline endpoints are
disjoint. All three first releases select the next slot at 1,081,600 ps.
Their slot waits are 80,600, 64,600 and 600 ps respectively. For these unloaded
pipeline endpoints inside the loaded run, the observed completion follows
`ceil(R/83,200)*83,200 + 17*83,200 + 2,000,000` ps. This accounts for all 40
pipeline rows across the eight loaded ideal controls with zero residual;
later pipeline starts and completions retain their A=0 timestamps.

This calendar is the existing ideal-profile surrogate, not a new physical
backend defect. An idle calendar can rebase to a later release, which explains
why the eight controls without EP preserve the expected shift. A future
freeze must model the active calendar explicitly. Changing the ideal scheduler
would instead change the model and requires a separate prospective decision.

## Other findings in the retained result

The original aggregate contains twelve fatal entries: eight ideal control
mismatches and four rail identity reports. The four rail reports come from a
checker defect. It compares complete hop dictionaries, including renderer tags;
adding EP allocates an earlier tag and shifts every pipeline tag by one.
All 20 corresponding rail hop start, completion and FCT rows are exact, as
are all 320 original packet timelines, all four request times and zero EP
service ahead. The reported `pp_flow_times_exact=false` therefore does not
establish a physical timing change.

The [post-run checks](post_run_checks.md) freeze a correction for future use:
join unique source, destination and payload identities, then compare the three
frozen timing fields. Each cell still validates tags against its own input.
Tests reject missing or duplicate hops and one-picosecond timing changes. This
correction neither reruns nor rescores the measured population. The public
[result projection](results.json) retains all twelve original fatal entries.

One local retry condition also contradicts its hypothesis. At four stages
and A=80 nanoseconds, logical packet 14's original reaches the endpoint at
37,017,600 ps. Its expected arrival plus receiver window is 34,728,080 ps,
so it arrives 2,289,520 ps too late and receives no delivery. Its physical
terminal is `endpoint_consumed` with `admission=late`, not `fabric_drop`.
Retries one through six are fabric-dropped and retry seven succeeds. The
other three inspected cases have the seven fabric drops predicted by B3.
Thus even a corrected ideal oracle would not establish the frozen claim that
every critical original was lost in the fabric.

## Physical request diagnostics, unscored

Time to first token (TTFT) is the final pipeline stage's completion projected
through the existing step result. These raw physical observations remain
useful diagnostics. They are not a qualified behavioral result of this void
run. All table times are microseconds.

| Node-local stages | Added first-stage work | Unloaded TTFT | Loaded TTFT | TTFT increase | EP service-ahead work |
|---|---:|---:|---:|---:|---:|
| 4 | 0.016 | 53.420 | 2593.317 | 2539.897 | 337.25388 |
| 4 | 0.080 | 53.484 | 2593.596 | 2540.112 | 307.19600 |
| 8 | 0.016 | 122.985 | 2663.113 | 2540.128 | 272.94588 |
| 8 | 0.080 | 123.049 | 2663.257 | 2540.208 | 248.65464 |

The increases and largest-hop differences lie inside the frozen 2.4 to 2.8
millisecond band. Every case has positive independently observed earlier EP
DATA service ahead. The work column sums packet visits that can overlap; it
is not an additive contribution to TTFT. The independently buffered earlier
EP subset equals this work in these four cells.

All four critical retry chains retain the six probe intervals of 40, 80,
160, 320, 640 and 1280 microseconds, totaling exactly 2.52 milliseconds.
Every observed fabric drop in those selected chains has positive EP DATA
occupancy, and removing those bytes at the same instant would admit the frame
at both the shared-switch and target-egress limits. This is a local admission
counterfactual, not a replay without EP. The successful retry, receiver release,
serialized service and ordered delivery join to the request boundary exactly.
Source eligibility remains unobserved, so authorization-to-transmission offsets
stay named dispatch offsets. No complete marginal-delay decomposition follows.

The stored relation fields retain their original local `holds` or `refuted`
labels for auditability. The aggregate `verdict=void` governs their use.
Likewise, normalized fields derived from the rejected transformed ideal rows
are retained diagnostics and must not be cited as identical-input normalized
pipeline evidence. Raw FCTs and the unchanged ideal EP phase remain traceable.

## Evidence and chronology

- Expectations-only freeze: `540bf8ddc89b3c0fe23991fcccf100e3dbd7b49b`.
- Final pre-run source: `1d7fb456b9400f43b17ce7bfab5c191a78d33418`.
  The preceding preflight stopped before any native execution because the
  published accepted-digest list was compared as a scalar. The fix accepts
  that declared representation and leaves the frozen population unchanged.
- Native source: `cc1c80f434600ec977fdb5916fc5ff74be63231d`.
  Native binary SHA-256:
  `a0722e32822da8b02461dc057d94a4fe4a4a4cbcb8b9d15354f7a487f06bb491`.
- Execution inventory: 32 physical executions in 16 trace on/off pairs;
  16 ideal controls; 12 fixed-window A=0 regressions; 12 historical
  full-bisection controls. Every native execution completes.
- Exact controls, kept separate from behavioral evidence: all 16 trace pairs
  preserve completion CSVs and metrics, all 12 A=0 regressions reproduce v2,
  and all 12 historical controls reproduce their accepted completion CSVs.
- Full result SHA-256:
  `5cc7d86b9ee471a757a8f0adc5ef155b91d571994c9bccfc7a32c3fe4a72e1de`.
  The compact projection retains its original interpretation and hashes every
  omitted complete trace audit. Bulk packets, visits, commands and logs remain
  external to Git.

The study runner returns status 2 for this void result. Reproduction uses the
recorded source and binary identities, the retained predecessor bulk roots
and the command below; source the environment from local configuration. Use
pre-run source `1d7fb456b9400f43b17ce7bfab5c191a78d33418` to reproduce the
original twelve-entry checker output, since the later rail correction
deliberately changes future comparisons.

```bash
python -m examples.pp_arrival_regimes_v1.run_study \
  --reference "$SIMLLM_PP_V1_RESULTS" --prior "$SIMLLM_PP_V2_RESULTS" \
  --out "$SIMLLM_ARRIVAL_RESULTS" --workers 4 --timeout-s 3600
```
