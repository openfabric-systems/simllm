# RNIC packet-event ABI v2 results

## Claim scope

This report closes BACK-25 and BACK-26. NetworkPort ABI v2 now carries the
packet-attempt and transport-control vocabulary, while ABI v1 remains the
default compatibility path. The htsim packetized manifold is the timing
authority for real TX and RX observations. The SimLLM WQE timeline populates
`first_packet_at_ps` and `last_packet_at_ps` only from explicit data or
retransmission TX-start events.

This is Tier A component and native-composition evidence. It does not claim
the Tier B `ExecutionGraph -> DeviceRuntime -> CompletionEvent -> StepResult
-> TTFT/TPOT` chain. HTSIM-9 remains open only for that live-metric run through
CORE-15.

## Chronology and provenance

The SimLLM expectations were frozen before implementation at
`506f87af93687ccf0df85f6b5307b71a20ed3762`. The paired htsim choices were
frozen at `6ece8bdd908496dadfd4df809e3a4eb660d6cc26`. Both commits cite the audited
source revisions and record the precise pre-commit working-tree state. No
untracked dry-run harness existed in either repository. The registered
check-only command printed its registry confirmation by design and produced
no artifacts.

Implementation is SimLLM commit
`fad1dcf277bab950035e35cd76c83fe1ec3db4f2` and htsim commit
`63e2eb6437ef15b4bb039ce94fe647b7b488dbde`. A nonfinal smoke then exposed two
study-runner defects. The FIFO fatal checker used the cell doorbell time for
both WQEs instead of each WQE's frozen serializer grant, and the inherited
checker required its producer inside the ABI-v1 run directory. SimLLM commit
`c54d556133b411c59ff5094b591d912d4e19006e` repaired only that machinery before
the formal run. The sweep, signed relations, exact bands and denominators did
not change. These repairs are post-specified and are documented in
[`CHECKER_CORRECTION.md`](CHECKER_CORRECTION.md); they are not claimed as
pre-registered assertions.

The formal registered run used SimLLM revision `c54d556` and htsim revision
`63e2eb6`. This is a local pre-run expectation freeze, not a claim of public
pre-registration. Bulk build and raw output stayed outside Git under the
configured `SIMLLM_WAVE3_RUN_ROOT`.

The validated machine-readable summary is [`results.json`](results.json),
with SHA-256
`11a37d089a66ccf36e3a1242f5a2c6d10a69e913ab7076db1e84e6f76443feda`.
The external runner result had SHA-256
`1cf05dd11a2a3bb0abfbcf783db098bed36c17bc4f9d490b6415deb920037064`.
The checked-in copy replaces its two prescribed expectation-commit
placeholders with the actual hashes above; every observation, count, revision
and artifact digest is otherwise unchanged. The same results-only substitution
is now literal in the runner for future reproductions; it changes no input,
oracle or simulation behavior.

## Scored behavioral evidence

Evidence families retain separate denominators.

| Family | Passed | Frozen relation |
|---|---:|---|
| Inherited Tier A D additivity | 4 of 4 | Raising native doorbell service from 0 to 1,000 ps moves all absolute boundaries by exactly +1,000 ps. |
| Inherited inverse-rate serialization | 4 of 4 | Service at 200 Gbit/s is exactly 2 times service at 400 Gbit/s. |
| Inherited two-WQE FIFO | 4 of 4 | Both WQEs match the exact grant, wait, terminal and JCT equations. |
| ABI-v2 TX issue additivity | 4 of 4 | First and last TX issue move by exactly +1,000 ps for every payload-rate pair. |
| ABI-v2 RX additivity | 4 of 4 | First and last RX boundaries move by exactly +1,000 ps for every payload-rate pair. |
| ABI-v2 multi-packet span | 2 of 2 | TX and RX spans at 200 Gbit/s are exactly 2 times the 400 Gbit/s spans at both doorbell values. |
| ABI-v1 artifact identity | 2 of 2 | Raw observations and summary have exactly zero changed bytes. |

The new packet family therefore passes 10 of 10 relations. It makes the
formerly unscorable relation 1 of `rnic_live_v1` observable: changing D by
1,000 ps changes both first and last packet issue by exactly +1,000 ps. For
the 1 MiB rows, the last-minus-first packet span is 41,779,200 ps at
200 Gbit/s and 20,889,600 ps at 400 Gbit/s, exactly the frozen factor of 2.

All eight single-WQE exact-oracle rows also pass. They are exact-row evidence
and do not enter a behavioral denominator. The missing-TX-event mutant was
rejected by the same checker that accepted the real observations.

## Vocabulary and authority evidence

ABI v2 uses separate session-unique extent and packet-attempt tokens. A packet
attempt carries extent and packet indices, transmission attempt, payload
offset, payload and wire bytes, packet kind, and stable drop evidence. TX
start, TX finish and RX arrival are intermediate observations; Delivered and
Dropped are the only packet-attempt terminals. An extent cannot terminate
while one of its attempts remains live.

The htsim packetized manifold emits TX start and finish from committed source
serializer boundaries and RX arrival and delivery from committed destination
boundaries. A directed composed test for an 8,192-byte WQE at 400 Gbit/s and
D equal to 1,000 ps observes first TX at 1,000 ps, last TX at 82,920 ps and
the extent terminal at 246,760 ps. This test crosses the actual packetized
runtime, the htsim wrapper and the native WQE timeline; no acceptance-time
surrogate supplies those packet fields.

The discriminated control vocabulary covers packet-keyed ECN and CNP,
policy-context-keyed eligibility and rate updates, PFC submit, pause and
resume, and stable link-state transitions. The wrapper relay test covers every
form and rejects any event that its bound runtime did not advertise. Current
physical runtimes do not advertise a timestamped dynamic-link producer, so
requesting that optional path rejects before mutation. HTSIM-15 owns that
deliberately disabled enabled path.

Fatal unscored checks all hold: authority exclusivity, token conservation,
packet lifecycle ordering, exact payload closure, controlled drop, FIFO
ordering, terminal atomicity, quiescence, capability validation and wrapper
bypass sensitivity. These invariants and the 368 native test executables do
not increase a behavioral denominator.

## ABI-v1 and validation gates

ABI v1 remains constructible without a new virtual override. Its raw Tier A
artifact SHA-256 is
`37a4e9cf88a1b60094409150dfad25599eb77cbf268b3d08bfacf527e493a26a`, and its
summary SHA-256 is
`00ef7e4f5bdbd38f4eabe9ba42dc75f56de528c8751b93e6eef4a3089fa61004`.
Both exactly match the accepted legacy reference. The v1 path emits no packet
timeline fields, packet events or control events.

Final gates:

- Registered Release study and complete htsim CTest suite: 368 of 368 passed.
- Standalone SimLLM native CTest suite: 6 of 6 passed with warnings as errors.
- `.venv/bin/ruff check .`: all checks passed.
- `.venv/bin/pytest -q`: 646 passed, 4 skipped.

## Genuine-risk fraction

Fractions are reported per scored evidence family and are not combined with
exact rows, fatal invariants or native executables.

- Inherited Tier A: 12 of 12 relations were plausible failures. Admission
  could have replaced serializer issue, changed the doorbell ownership seam or
  disturbed FIFO grant order.
- New packet timeline: 10 of 10 relations were plausible failures. A competent
  relay could timestamp acceptance, choose TX finish instead of TX issue,
  mishandle a right-aligned tail or lose the rate scaling at one packet
  boundary.
- ABI-v1 identity: 2 of 2 relations were plausible failures. Version
  negotiation, event scheduling or serialization could have changed either
  accepted artifact even when the new mode was disabled.

## Registry result and deliberate omissions

BACK-25 and BACK-26 are removed from the open registry. HTSIM-9
`(Completeness; P1; L)` now states precisely that only the Tier B live-metric
run through CORE-15 remains. HTSIM-15 `(Completeness; P2; L)` owns a future
timestamped dynamic-link producer; the explicit rejection path preserves the
accepted baselines.

This slice deliberately does not claim Tier B, TTFT or TPOT evidence. It does
not implement dynamic link transitions, change the default ABI, edit
`README.md` or `docs/README_PRO.md`, or commit bulk output. The paired htsim
commit is intentionally not installed as the SimLLM submodule pin in this
worker worktree; pin integration remains a separate maintainer operation after
the two commit series are reviewed together.
