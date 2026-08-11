# BRIDGE-3 child-lifetime results

## Chronology and evidence

The expectations were frozen at commit `78ef408` before implementation or any
result-producing kill trial. The managed lifecycle implementation used for
this study is commit `d98bf86`. The frozen SimLLM base was `90ada43`; the real
corroboration binaries were freshly built from pinned HTSIM commit `4885c64`.

The registered outputs are external to Git:

- `$SIMLLM_WAVE5_RUN_ROOT/bridge3/summary.json` contains raw owner and target
  observations, the scored relation and real-binary corroboration.
- `$SIMLLM_WAVE5_RUN_ROOT/bridge1-identity/summary.json` contains the unchanged
  BRIDGE-1 diagnostic-versus-prepared identity run.

## Live killed-owner relation

Every owner received a real `SIGTERM` only after its exact marker count had
appeared and procfs showed that each launcher PID had replaced itself with the
registered executable. The controller then waited for owner exit and polled
each `(pid, start-time)` identity every 20 ms for at most 5 seconds. A live or
zombie target counted as remaining.

| Cell | Invocation | Targeted | Unsafe remaining U | Managed remaining M | M minus U | Result |
|---|---|---:|---:|---:|---:|---|
| D1 | diagnostic | 1 | 1 | 0 | -1 | pass |
| P2 | prepared, 2 workers | 2 | 2 | 0 | -2 | pass |
| P4 | prepared, 4 workers | 4 | 4 | 0 | -4 | pass |

All unsafe children were still sleeping after the full five-second poll and
had been reparented to PID 1. Their exact identities were revalidated before
cleanup; no target remained after cleanup. Managed disappearance was observed
within 0.20 ms after owner exit in all three cells. Thus the scored family
passed 3/3, with a genuine-risk fraction of 3/3.

The additional managed diagnostic run reached the actual pinned
`htsim_rnic` executable before the owner was killed. Its targeted count was
one and its remaining count was zero. This is fatal corroborating evidence,
not a fourth scored instance.

### Entailment analysis

The relation is not entailed by an earlier oracle. The harness first records
owner exit and raw post-kill state for every exact target. Marker validity,
executable readiness and later negative cleanup are fatal unscored checks, but
none constrains the post-kill remaining count. Only after all six stand-in raw
rows exist does the harness compare U and M with the frozen exact relation.
Either U or M can therefore violate a scored instance in a run that reaches
it.

## Normal-run and cleanup controls

The unchanged BRIDGE-1 checker ran the vLLM and SGLang captures at worker
widths four and eight. It reported:

| Evidence class | Passed | Total |
|---|---:|---:|
| `StepResult` | 34 | 34 |
| `StepNetworkOutcome` | 34 | 34 |
| GOAL text | 34 | 34 |
| GOAL binary | 34 | 34 |
| completion CSV | 34 | 34 |
| latency streams | 4 | 4 |
| physical-quiescence cells | 6 | 6 |

The focused Python controls compare stdout, stderr and return status with a
direct invocation; terminate and reap a timed-out owned child; call cleanup
and persistent-sink `close` repeatedly; and kill an actual owner while an
unrelated sentinel remains alive. The Windows-only control injects Job Object
creation failure and verifies that the handshake-blocked simulator command is
never executed.

The launch handshake itself and marker/source-pin validation are
change-set guards. Identity, quiescence, targeting, timeout, shutdown,
unrelated-process exclusion and real-binary corroboration are fatal unscored
evidence. None is included in the 3/3 behavioral denominator.

## Supported-platform strategy

| Platform | Ownership guarantee | Failure behavior |
|---|---|---|
| Linux | A new session and process group per invocation, `PR_SET_PDEATHSIG` armed before a one-byte handshake, plus parent signal-and-reap cleanup | Parent-death setup, identity or launch failure rejects before simulator execution |
| Other POSIX | A new session and process group plus parent handling of `SIGTERM`, `SIGINT` and `SIGHUP` | A worker launch rejects unless the main thread installed the handler; uncatchable `SIGKILL` and host failure are not claimed |
| Windows | A per-invocation Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, assigned while the launcher is handshake-blocked | Job creation, configuration or assignment failure rejects; there is no unowned fallback |
| Other platforms | No managed launch | The invocation raises an actionable unsupported-platform error |

Every POSIX cleanup signal is limited to a registered child whose live process
group still equals its registered child PID. Every Windows cleanup is limited
to the per-invocation Job Object. The regression-only unsafe switch exists
solely to measure the frozen negative control and defaults to false.

## BRIDGE-3 acceptance closure

The registered acceptance clauses map to evidence as follows:

| Registered clause | Evidence |
|---|---|
| "no orphan htsim processes after a killed run" | Managed D1, P2 and P4 all had zero live or zombie targets after bounded polling; the real pinned D1 corroboration also had zero |
| Kill diagnostic and prepared runs while native children are in flight | All three hermetic cells reached the executable stand-in through the real sink invocation path; the separate D1 reached pinned `htsim_rnic` |
| Preserve normal diagnostic output byte for byte | Direct-versus-owned stdout, stderr and status control passed |
| Preserve the complete prepared identity family | The existing checker passed 34/34 in each of five per-step evidence classes, 4/4 latency streams and 6/6 quiescence cells |
| Make timeout and normal-shutdown cleanup idempotent | Timeout reap, repeated registry cleanup and repeated persistent close controls passed |
| Avoid signaling unrelated processes | The killed-owner integration control left its independent sentinel running; production cleanup verifies registered process-group identity |
| Document the supported-platform strategy | The table above states Linux, other POSIX, Windows and unsupported-platform behavior without claiming unavailable protection |

Every registered clause is demonstrated or documented within its stated
scope. BRIDGE-3 therefore closes without a residual task.
