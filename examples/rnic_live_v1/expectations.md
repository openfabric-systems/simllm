# Live RNIC composition v1 expectations

## Freeze status

This file is the expectations-only record for the first composed native RNIC
and htsim run. It was first written before implementation, before the SimLLM
RNIC library was linked into an htsim driver, and before any run of this study.
This amendment incorporates an independent pre-implementation review and still
contains no measured values. The results must cite the original freeze commit
and the final pre-run expectations commit.

This freeze has two acceptance tiers. Tier A covers the BACK-8 and HTSIM-9
composed binary plus the existing `HtsimStepSink` replay. Tier B is the
repository live-reachability gate owned by CORE-4 and CORE-5. This study does
not create, synthesize or score `CompletionEvent` or `ExecutionResult`. Tier B
requires a separately frozen `ExecutionGraph -> DeviceRuntime ->
CompletionEvent -> ExecutionResult -> StepResult -> TTFT/TPOT` run. Until Tier
B passes, the native composition is component and step-sink evidence and
BACK-8 remains open.

## Authority and modes

Structural hardware mode has exactly one mutable WQE lifecycle. The native
SimLLM RNIC session owns WR, WQE, SQ, RQ, SRQ, CQ, CQE and stage timestamps.
htsim owns transport-policy and fabric state and returns network events through
opaque tokens. Public bookkeeping and completion CSV rows are read-only
projections of native records.

Hardware-bypass mode retains the timing-neutral htsim ledger. The two modes
are mutually exclusive. A run that updates both lifecycles must fail.

## Sweep

Use one signaled SEND on an otherwise idle two-endpoint fixture and sweep:

- payload: 4 KiB and 1 MiB;
- link rate: 200 and 400 Gbit/s;
- native doorbell service: 0 and 1,000 ps;
- hardware mode: structural and bypass.

All other native service times, propagation and congestion are zero in the
exact fixture. Packet wire serialization remains active. The same frozen GOAL
and request replay is used for the direct binary, `HtsimStepSink`, `StepResult`,
TTFT and TPOT checks.

The FIFO fixture posts two signaled 4 KiB SEND WQEs, W0 then W1, to one SQ at
`submitted_at_ps = 0`, publishes both in one doorbell and uses one capacity-one
egress service lane. All other service and propagation delays are zero. Let D
be doorbell service and L(R) be the independently calculated one-WQE network
service at link rate R. The exact timeline is:

```text
first_packet(W0) = D
completed(W0)    = D + L(R)
first_packet(W1) = D + L(R)
completed(W1)    = D + 2 * L(R)
```

W1 queue wait is exactly L(R), completion order is W0 then W1, and JCT is
`D + 2 * L(R)`. Increasing D adds exactly 1,000 ps to all four absolute
boundaries. `L(200) = 2 * L(400)` for this exact rational fixture. No later WQE
may bypass W0.

## Expected behavioral relations

1. In structural mode, increasing doorbell service by 1,000 ps shifts WQE
   fetch eligibility, first-packet issue, CQE visibility, CQ poll, flow
   completion, direct-run JCT, `StepResult.completed_at_ps` and the dependent
   replay request boundary by exactly 1,000 ps in the isolated fixture. It does
   not change serialized payload service.
2. Doubling link rate halves only the independently calculated serialized wire
   term. The 1,000 ps native shift remains additive and unchanged at both
   payload sizes.
3. The composed htsim completion is the one result boundary. `StepResult` must
   not add standalone RNIC probe JCT or any second WQE-start constant.
4. The native FIFO fixture follows the four equations above at both rates and
   both D values.
5. Full `rnic-nn`, `rnic-cn` and DCQCN structural rows for one hardware fixture
   carry the same hardware-configuration hash. Only transport-policy identity
   and its consequences may differ.

## Bypass reference and artifact identity

The frozen bypass reference is HTSim commit
`8c3f8b231a6a9311ffc1e7969a003dcba724b50d`, invoked with the same GOAL bytes,
topology bytes, profile, seed and baseline argv. For every retained bypass
profile, compare these behavioral artifacts byte for byte:

- completion CSV;
- canonical parsed completion rows and final JCT record;
- `StepResult` tuple sequence; and
- replay TTFT and TPOT summary.

GOAL text and binary are input-identity guards. The new configuration and run
audit record is excluded from byte comparison because it must declare
`hardware_mode=bypass` and `authority=AtlahsWqeLedger`; those fields are checked
structurally. Paths, build IDs, elapsed wall time and command spelling are
diagnostic and excluded. Default bypass emits no new legacy stdout line. Native
only knobs are rejected in bypass mode or omitted from the command, never
silently accepted as active behavior.

## Fatal unscored invariants

1. Structural mode reports `native_session_constructed=1`, `native_posts=N`,
   `legacy_ledger_constructed=0` and `legacy_mutations=0`.
2. Bypass mode reports `native_session_constructed=0`, `native_posts=0`,
   `legacy_ledger_constructed=1` and `legacy_posts=N`.
3. Every accepted native post maps to one stable WQE key. Every extent key is
   unique within that WQE. Every attempt token is issued once and terminates in
   exactly one delivery or drop. Unknown, duplicate and cross-WQE terminals are
   rejected before state mutation. Quiescence leaves zero live tokens. A retry
   changes only its attempt index and token.
4. Native terminal records, WQ and CQ producer/consumer sequences, projected
   rows and timestamps reconcile exactly at quiescence.
5. A send WQE names its local SQ and send CQ, never a fabricated remote RQ. A
   receive WQE names exactly one RQ or SRQ and its receive CQ. RX matching is a
   later relation. One-sided operations consume no receive WQE.
6. A signaled WQE completes at CQ poll. An unsignaled successful WQE produces
   no fabricated CQE; reclamation follows a later signaled completion or an
   explicit modeled drain or teardown rule.
7. `first_packet_at_ps` is an explicit packet-issue event. Network acceptance
   and whole-flow delivery are not substitutes for first or last packet issue.

The following negative controls must fail before lifecycle, counter or time
mutation: structural mode with the native library unlinked; structural mode
with the legacy authority also enabled; duplicate, unknown or cross-WQE token
terminal; live token at quiescence; mismatched hardware hash across
`rnic-nn`, `rnic-cn` and DCQCN; and a wrapper-bypass control presented to the
D-additivity checker. A controlled htsim drop must produce the modeled error
completion and must never produce a success CQE.

## Evidence and reported metrics

Report run configurations without scoring them. Score these behavioral
relation families separately: D-additivity over payload by rate, inverse-rate
serialization over payload by D, and two-WQE FIFO over rate by D. Report exact
oracle rows separately. Authority, token conservation, reconciliation,
quiescence, inactive-stage and by-construction zero checks are fatal but
unscored. Native executable tests are component evidence and are not added to
a behavioral pass count.

Tier A reports the complete native WQE timeline, raw per-flow FCT, phase JCT,
`StepResult` boundaries and fixed-replay TTFT and TPOT. Probe-only JCT remains a
component metric. No `CompletionEvent`, `ExecutionResult` or Tier B claim may
appear until CORE-4 and CORE-5 provide and validate that path.
