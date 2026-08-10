# Live RNIC composition v1 Tier B expectations

## Freeze status and boundary

This new expectations-only file freezes Tier B without changing the frozen
Tier A files. It precedes the Tier B acceptance implementation, the composed
htsim binary, and every Tier B result-producing run in this worktree. Results
must cite this file's final pre-run expectations commit together with the Tier
A chronology required by `expectations.md`.

Tier B is the live metric chain:

```text
ExecutionGraph -> DeviceRuntime -> CompletionEvent -> ExecutionResult
               -> StepResult -> per-request TTFT/TPOT.
```

The native session and htsim transport remain the sole composed physical
authorities selected by structural mode. Completion events, the execution
result, scheduler result and request metrics are immutable projections. The
reducer adds no probe latency and advances no WQE.

## External-source audit before freeze

This file introduces no new hardware or external-runtime timing form. It
inherits the already-frozen Tier A source audit and exact serializer fixture:

- `examples/rnic_live_v1/tier_a_harness_expectations.md:17-69` pins and audits
  the htsim request, drainage, packet-byte and integer serializer sources;
- `examples/rnic_live_v1/tier_a_harness_expectations.md:112-145` freezes
  `L(P,R) = P * 8 * 1000 / R ps`, its four exact values and structural
  `JCT = D + L(P,R)`;
- `examples/rnic_live_v1/expectations.md:32-44` freezes payload, rate,
  doorbell, authority mode and same-GOAL replay inputs;
- `examples/rnic_live_v1/expectations.md:64-80` freezes D additivity,
  inverse-rate serialization and the one composed result boundary;
- `examples/rnic_live_v1/expectations.md:82-100` freezes bypass behavioral
  artifact identity and excludes only the new audit record;
- `simllm/core/runtime.py:1813-1911` is the landed event projection, and
  `simllm/core/runtime.py:1913-2055` is the landed corrected critical-path
  reduction consumed by CORE-5.

The numeric Tier B relations are deductions from those frozen forms plus the
CORE-5 conservation identity. Exact call and event sequences remain fatal and
unscored.

## Frozen Tier B replay

Retain the Tier A single-WQE grid:

- payload P in `{4096, 1048576}` bytes;
- link rate R in `{200, 400}` Gbit/s;
- structural native doorbell service D in `{0, 1000}` ps;
- structural and bypass hardware modes;
- the frozen zero-header, zero-propagation, no-control-frame and
  no-congestion network fixture.

For every cell, replay one request through three graphs with byte-identical
semantic GOAL input: one prefill step followed by two decode steps. The first
graph is released at `T0 = 7,000 ps`; each later graph is released at the
preceding `StepResult.completed_at_ps`. Every step explicitly identifies the
request as sampled.

For structural mode:

```text
L(P, R) = P * 8 * 1000 / R ps
J(P, R, D) = D + L(P, R)
StepResult(step i) = (step_index=i,
                      step_latency_ps=J,
                      completed_at_ps=T0 + (i + 1) * J)
TTFT = J
each inter-token interval = J
TPOT = J exactly
final VirtualClock.now_ps = T0 + 3 * J.
```

The exact structural one-step values are:

| Payload | R (Gbit/s) | D (ps) | J (ps) |
|---:|---:|---:|---:|
| 4 KiB | 200 | 0 | 163,840 |
| 4 KiB | 200 | 1,000 | 164,840 |
| 4 KiB | 400 | 0 | 81,920 |
| 4 KiB | 400 | 1,000 | 82,920 |
| 1 MiB | 200 | 0 | 41,943,040 |
| 1 MiB | 200 | 1,000 | 41,944,040 |
| 1 MiB | 400 | 0 | 20,971,520 |
| 1 MiB | 400 | 1,000 | 20,972,520 |

For every payload and rate, increasing D has exact signed effect `+1,000 ps`
on step latency, every absolute completion, TTFT and TPOT. For D fixed at
zero, doubling R halves J exactly. For D fixed at 1,000 ps, doubling R halves
only L and leaves D unchanged:

```text
J(P, 200, 1000) - 2 * J(P, 400, 1000) = -1000 ps.
```

The component identity in structural mode is

```text
J = queue_ps + kv_ps + kernel_ps + dma_ps
  + collective_ps + nic_ps + control_ps
  = D + 0 + 0 + 0 + 0 + L + 0.
```

If the composed native projection exposes D as selected NIC-local service
rather than wait, an expectations amendment must freeze that resource mapping
before the first result-producing Tier B run. The total J, signed D relation
and one-boundary rule may not change. No measured outcome may choose the
mapping after the fact.

The single-WQE additive totals are reported separately. Their queue and
service sum may equal J in this isolated fixture, but the additive record must
remain a distinct type and must not be used to satisfy request conservation.

## One completion boundary

The authoritative composed completion produces the WQE subject completed
event, the required operation completed event, and
`ExecutionResult.completed_at_ps = T0 + J`. `StepResult` consumes that boundary
once. It must not add the standalone RNIC probe JCT, the direct binary JCT, a
second WQE-start constant, or physical quiescence.

For each step, the streamed callback tuple and `ExecutionResult.events` are
object-identical and ordered by nondecreasing timestamp. `QUEUED` is resource
eligibility and `STARTED` is the grant. The required operation event and WQE
subject event agree with the native projection on identity, byte count and
completion timestamp. These are fatal unscored guards.

## Bypass identity

The bypass rows retain the Tier A reference and do not activate D. Native-only
doorbell input is omitted or rejected. For each retained profile, compare the
same four behavioral artifacts byte for byte:

1. completion CSV;
2. canonical completion rows and final JCT;
3. `StepResult` tuple sequence;
4. replay TTFT and TPOT summary.

GOAL text and binary remain input-identity guards. The Tier B run record must
name `hardware_mode=bypass` and `authority=AtlahsWqeLedger`, but its bytes are
excluded exactly as Tier A specifies. Any changed behavioral byte fails the
bypass cell. This is scored artifact identity, not a zero-field invariant.

## Decision consequences and evidence classes

Score these behavioral families separately:

- structural D additivity, one instance per payload and rate;
- structural inverse-rate serialization, one instance per payload and D;
- StepResult, TTFT and TPOT closed forms, one instance per structural cell;
- seven-component request rows, one instance per structural cell;
- four-class bypass byte identity, one instance per retained profile.

The request conservation equality, authority exclusivity, event projection,
clock monotonicity, schema compatibility, inactive fields and configuration
echoes are fatal unscored invariants. Native executables and Tier A results
remain separate component evidence.

If D or R changes the native timeline but not StepResult and TTFT or TPOT, the
composed mechanism is not live-reachable and Tier B fails. If StepResult moves
by more than J, the reduction double-counted a component boundary. If bypass
bytes move, structural composition changed the explicit off path. Any of
these outcomes keeps the owning live-reachability tasks open.

The Tier B result report must give genuine-risk fractions per scored family.
D additivity can fail if native doorbell time is dropped at the runtime seam;
rate scaling can fail if probe and composed completions are added; the metric
forms can fail if the clock advances to quiescence; component rows can fail
with the correct total but wrong tail owner; bypass identity can fail through
an unintended default-mode change.

## Registered command and pre-freeze dry run

The future result-producing command is:

```bash
.venv/bin/python examples/core5_reduction/run_study.py \
  --out "$SIMLLM_CORE5_RUN_ROOT/tier_b" \
  --tier-b-only \
  --tier-b-producer "$SIMLLM_RNIC_TIER_B_PRODUCER"
```

Before this freeze, the same command must be run with `--check-only` appended.
That mode validates the complete Tier B matrix, closed forms, inherited source
anchors, CLI and output-root shape. It accepts the not-yet-landed producer path
as an opaque value, prints a registry confirmation by design, and creates no
directory, result or measured artifact. The untracked pre-freeze harness may
encode only frozen literals and check-only validation; it must not inspect or
invoke the producer.

## Post-specified filesystem portability note

This note postdates the freeze and does not change its matrix, relations,
producer contract, or chronology. The one-off environment-variable spellings
above remain frozen text. After loading `.env.local.sh`, the current portable
rendering is:

```bash
.venv/bin/python examples/core5_reduction/run_study.py \
  --out "${SIMLLM_DATA_ROOT:?configure SIMLLM_DATA_ROOT}/core5_reduction/tier_b" \
  --tier-b-only \
  --tier-b-producer "${SIMLLM_DATA_ROOT}/core5_reduction/tier_b/build/htsim_rnic_tier_b"
```

The resolved historical machine-local paths are intentionally omitted.
