# Live RNIC composition v1 expectations

## Freeze status

This file is the expectations-only record for the first composed native RNIC
and htsim run. It is written before implementation, before the SimLLM RNIC
library is linked into an htsim driver, and before any run of this study. It
contains no measured values. The results must cite the commit that first adds
this file.

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
exact fixture. A second two-WQE case uses equal release times to exercise one
SQ's FIFO serialization. The same frozen GOAL and request replay is used for
the `ExecutionGraph`, `StepResult`, TTFT and TPOT checks.

## Expected relations

1. In structural mode, increasing doorbell service by 1,000 ps shifts WQE
   fetch eligibility, first-packet issue, CQE visibility, CQ poll, flow
   completion, `ExecutionResult.completed_at_ps`, `StepResult.completed_at_ps`
   and the dependent live request boundary by exactly 1,000 ps in the isolated
   fixture. It does not change serialized payload service.
2. Doubling link rate halves only the serialized wire term. The 1,000 ps
   native shift remains additive and unchanged at both payload sizes.
3. The composed htsim completion is the one result boundary. `StepResult` must
   not add standalone RNIC probe JCT or any second WQE-start constant.
4. In bypass mode, changing native service parameters changes no command,
   completion row, FCT, JCT, `ExecutionResult`, `StepResult`, TTFT or TPOT.
   Accepted bypass artifacts remain byte-identical.
5. Full `rnic-nn`, `rnic-cn` and DCQCN rows for one hardware fixture carry the
   same hardware-configuration hash. Only transport-policy identity and its
   consequences may differ.
6. Every accepted native post maps to exactly one session-stable WQE key. Each
   logical network extent maps to one WQE key and extent index. Every wire
   attempt has a distinct attempt index and opaque token and terminates in one
   delivery or drop event. A dropped attempt may create a later retry while
   preserving the WQE and logical-extent keys. Native records, all attempt
   terminals, projected rows and public object references reconcile exactly at
   quiescence.
7. A send WQE names its local SQ and send CQ, never a fabricated remote RQ. A
   receive WQE names exactly one RQ or SRQ and its receive CQ. RX matching is a
   later relation. One-sided operations consume no receive WQE.
8. For a signaled WQE, application-visible completion occurs at CQ poll. An
   unsignaled successful WQE produces no fabricated CQE or CQ completion; its
   SQ reclamation follows the later signaled completion or explicit drain rule.
9. `first_packet_at_ps` is the native NIC-start timestamp. Network acceptance
   and whole-flow delivery are not substitutes for first or last packet issue.

## Reported metrics

Report the complete native WQE timeline, raw per-flow FCT, phase JCT,
`ExecutionResult` and `StepResult` boundaries, and the fixed replay's TTFT and
TPOT. Keep exact rows, behavioral relations, reconciliation invariants and
native executable tests as separate evidence classes. Probe-only JCT is a
component metric and cannot satisfy the live-reachability gate.
