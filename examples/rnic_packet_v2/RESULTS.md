# RNIC packet-event ABI v2 results

## Claim scope

This report closes BACK-25 and BACK-26. NetworkPort ABI v2 now carries the
packet-attempt and transport-control vocabulary, while ABI v1 remains the
default compatibility path. The scored study grid uses the unbound
compatibility port's internal serializer as its timing authority. The htsim
packetized manifold is the independent timing authority only in the directed
composed 8,192-byte test. In both paths, the SimLLM WQE timeline populates
`first_packet_at_ps` and `last_packet_at_ps` only from explicit data or
retransmission TX-start events.

This is Tier A component and native-composition evidence. It does not claim
the Tier B `ExecutionGraph -> DeviceRuntime -> CompletionEvent -> StepResult
-> TTFT/TPOT` chain. HTSIM-9 remains open until a Tier B-class run passes
under ABI v2 with explicit packet-issue evidence populating the native
timeline.

## Chronology and provenance

The SimLLM expectations were frozen before implementation at
`506f87af93687ccf0df85f6b5307b71a20ed3762`. The paired htsim choices were
frozen at `6ece8bdd908496dadfd4df809e3a4eb660d6cc26`. Both commits cite the audited
source revisions and record the precise pre-commit working-tree state. No
untracked dry-run harness existed in either repository. The registered
check-only command printed its registry confirmation by design and produced
no artifacts.

The first review regressions were frozen separately at
`07521786020e41f56196d13718c62169d47ad70d`. That freeze leaves the original
expectations untouched and labels every fix-round assertion as post-specified.
Its dry-run command also printed its registry confirmation by design and
produced no artifacts.

Implementation is SimLLM commit
`fad1dcf277bab950035e35cd76c83fe1ec3db4f2` and htsim commit
`63e2eb6437ef15b4bb039ce94fe647b7b488dbde`. After implementation started but
before `fad1dcf` was committed, a nonfinal smoke exposed that the frozen FIFO
fatal checker used the cell doorbell time for both WQEs. The post-specified
mutation to each WQE's `port_tx_at_ps` landed inside `fad1dcf`. A later
read-through found that the inherited checker required its producer inside
the ABI-v1 run directory; only that path repair landed in
`c54d556133b411c59ff5094b591d912d4e19006e`. The original c54d556 commit message
and the first version of this report attributed both repairs to c54d556; that
attribution was incorrect. The frozen sweep, signed relations, exact bands
and registered family sizes did not change. Both repairs are post-specified
and are documented in [`CHECKER_CORRECTION.md`](CHECKER_CORRECTION.md);
neither is claimed as a pre-registered assertion.

The first formal registered run used SimLLM revision `c54d556` and htsim
revision `63e2eb6`. This is a local pre-run expectation freeze, not a claim of
public pre-registration. Bulk build and raw output stayed outside Git under
the configured `SIMLLM_WAVE3_RUN_ROOT`.

Fix-round implementation is SimLLM commit
`b7116739961b7d6b9d413cb020d43112b4d58692` and htsim commit
`5445b81fd89c2e8c00bdf74e48d453da2a73eb30`. The first fix-round invocation
built the committed tips and passed all 370 native tests, then stopped before
either ABI run because the local `SIMLLM_TIER_A_RUN_ROOT` setting was absent.
It produced build artifacts but no ABI observation or result summary. That
incomplete directory was preserved under the external run root. The formal
rerun set the missing root explicitly and used the same clean commits.

The validated machine-readable summary is [`results.json`](results.json),
with SHA-256
`660295101a98bb40bb49714cc98e2f2b2dc4da989acd32964898493d7b4e5efd`.
It is byte-identical to the external runner result. The ABI-v2 raw observation
SHA-256 is
`39059d56663f73869224613c9c7a0de3bee5733a6654469cd2a54c22354cc692`.

## Post-specified fix-round 1 corrections

The first report's 10 of 10 packet-family score is withdrawn as a
genuine-risk score. In that run, the inherited Tier A checker and then the
fatal per-packet exact oracle ran before the packet relation loops. Once the
exact oracle pinned every first and last TX and RX timestamp, none of those
ten relations could fail in an execution that reached the scored loops. The
observations still support exact-row and fatal-invariant evidence, but the
original execution independently scored 0 of the reported 10 packet
relations.

The fix-round runner evaluates the raw ABI-v2 projections before either
entailing exact oracle. Its scored packet surface is exactly four TX
D-additivity instances, four RX D-additivity instances and two multi-packet
inverse-rate span instances. Each instance can now fail before an exact
timestamp is pinned. The later exact rows, payload and ordering checks, and
missing-TX-event mutant remain fatal unscored evidence. This ordering and
accounting correction is post-specified; it does not alter the frozen matrix,
relations or quantitative bands.

This correction supersedes the earlier statement in
[`CHECKER_CORRECTION.md`](CHECKER_CORRECTION.md) that the genuine-risk
denominators were unchanged. The frozen family sizes remain untouched, but
the original oracle-first execution no longer counts as an independent
packet-family score.

The FIFO fatal-checker mutation is also post-specified. The frozen checker
used the cell doorbell value as the TX base for both WQEs. A smoke after
implementation started exposed that W1 instead starts at its capacity-one
serializer grant. The corrected checker uses each WQE's `port_tx_at_ps`; that
mutation landed in `fad1dcf`, not `c54d556`. The later `c54d556` commit changed
only where the already built producer was placed for the inherited checker.

## Scored behavioral evidence

Evidence families from the corrected fix-round run retain separate
denominators.

| Family | Passed | Frozen relation |
|---|---:|---|
| Inherited Tier A D additivity | 4 of 4 | Raising native doorbell service from 0 to 1,000 ps moves all absolute boundaries by exactly +1,000 ps. |
| Inherited inverse-rate serialization | 4 of 4 | Service at 200 Gbit/s is exactly 2 times service at 400 Gbit/s. |
| Inherited two-WQE FIFO | 4 of 4 | Both WQEs match the exact grant, wait, terminal and JCT equations. |
| ABI-v2 TX issue additivity | 4 of 4 | First and last TX issue move by exactly +1,000 ps for every payload-rate pair. |
| ABI-v2 RX additivity | 4 of 4 | First and last RX boundaries move by exactly +1,000 ps for every payload-rate pair. |
| ABI-v2 multi-packet span | 2 of 2 | TX and RX spans at 200 Gbit/s are exactly 2 times the 400 Gbit/s spans at both doorbell values. |
| ABI-v1 artifact identity | 2 of 2 | Raw observations and summary have exactly zero changed bytes. |

The corrected packet family therefore passes 10 of 10 independently
evaluated relations. Taken together with inherited Tier A D-additivity, this
study scores six of the eight conjunctive boundaries in frozen relation 1 of
`rnic_live_v1`: WQE fetch eligibility, explicit first-packet issue, CQE
visibility, CQ poll, flow completion and direct-run JCT. It does not score
`StepResult.completed_at_ps` or the dependent replay request boundary. The
last TX issue and first and last RX checks are packet-study refinements, not
additional conjuncts of the original relation. For the 1 MiB rows, the
last-minus-first packet span is 41,779,200 ps at 200 Gbit/s and 20,889,600 ps
at 400 Gbit/s, exactly the frozen factor of 2.

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

Completed packet-attempt correlation is retained only until the parent extent
terminal, so a CNP that follows packet delivery remains valid without becoming
unbounded state. A failed runtime submission now rolls back every synchronously
scheduled event and defers ready notifications until commit. The directed
rollback test repeats the same failing submission and receives the injected
runtime error both times, with no leaked event or reused event-key collision.
The capacity-two v1 regression also confirms that `drop_first` selects the
first due terminal, not the first submitted extent.

The htsim packetized manifold emits TX start and finish from committed source
serializer boundaries and RX arrival and delivery from committed destination
boundaries. A directed composed test for an 8,192-byte WQE at 400 Gbit/s and
D equal to 1,000 ps observes first TX at 1,000 ps, last TX at 82,920 ps and
the extent terminal at 246,760 ps. This test crosses the actual packetized
runtime, the htsim wrapper and the native WQE timeline; no acceptance-time
surrogate supplies those packet fields.

The discriminated control vocabulary covers packet-keyed ECN and CNP,
policy-context-keyed eligibility and rate updates, PFC submit, pause and
resume, and stable link-state transitions. Only the test fake emits these
control forms. The physical packetized manifold advertises packet-attempt
events alone, and the wrapper rejects events that a bound runtime did not
advertise. HTSIM-16 owns physical ECN/CNP and rate-update producers from the
DCQCN policy plus PFC and link-state producers from the fabric. HTSIM-15 owns
the separately disabled timestamped dynamic-link transition source.

The frozen 4,096-byte and 1 MiB payloads and the directed 8,192-byte composed
test are all exact multiples of the 4,096-byte production quantum. The
oracle's final-packet branch is therefore not exercised with a partial final
packet at that quantum. BACK-34 owns the missing registered matrix and
directed-composition cell.

Fatal unscored checks all hold: authority exclusivity, token conservation,
packet lifecycle ordering, exact payload closure, controlled drop, FIFO
ordering, terminal atomicity, quiescence, capability validation and wrapper
bypass sensitivity. These invariants and the 370 native test executables do
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

- Registered Release study and complete htsim CTest suite: 370 of 370 passed.
- Standalone SimLLM native CTest suite: 6 of 6 passed with warnings as errors.
- `.venv/bin/ruff check .`: all checks passed.
- `.venv/bin/pytest -q`: 646 passed, 4 skipped.

## Genuine-risk fraction

Fractions are reported per scored evidence family and are not combined with
exact rows, fatal invariants or native executables. The original oracle-first
packet execution independently scored 0 of its reported 10 relations, so its
10 of 10 label is withdrawn rather than included in the corrected fractions.

- Inherited Tier A: 12 of 12 relations were plausible failures. Admission
  could have replaced serializer issue, changed the doorbell ownership seam or
  disturbed FIFO grant order.
- Fix-round packet timeline: 10 of 10 relations were plausible failures at
  their evaluation point because they ran against raw observations before any
  entailing exact oracle. A competent relay could timestamp acceptance,
  choose TX finish instead of TX issue or lose doorbell or rate scaling at one
  packet boundary.
- ABI-v1 identity: 2 of 2 relations were plausible failures. Version
  negotiation, event scheduling or serialization could have changed either
  accepted artifact even when the new mode was disabled.

## Registry result and deliberate omissions

BACK-25 and BACK-26 are removed from the open registry with a dated closure
narrative and this evidence link. HTSIM-9 `(Completeness; P1; L)` remains open
until Tier B passes under ABI v2 with packet-issue evidence in the native
timeline. BACK-34 `(Precision; P1; M)` owns production-quantum partial-tail
evidence. HTSIM-15 `(Completeness; P2; L)` owns a future timestamped
dynamic-link producer, and HTSIM-16 `(Completeness; P2; L)` owns every
physical control-event producer. Their explicit rejection and disabled paths
preserve the accepted baselines.

The final conditional fetch found `origin/main` at
`a620180c6bc980f1c695ade95b95cea7b407f199`, without the pending Tier B merge;
CORE-15 and BACK-8 were still open there. This branch therefore did not merge
that main or claim the pending ABI-v1 Tier B passage. Once the Tier B merge is
present, the registry union must retain its ABI-v1 passage, retain this landed
vocabulary, and close HTSIM-9 only after a Tier B-class run passes with ABI-v2
packet-issue evidence populating the native timeline.

This slice deliberately does not claim Tier B, TTFT or TPOT evidence. It does
not exercise a production-quantum partial final packet or implement physical
ECN/CNP, rate-update, PFC or link-state producers. It does not change the
default ABI, edit `README.md` or `docs/README_PRO.md`, or commit bulk output.
The paired htsim commit is intentionally not installed as the SimLLM submodule
pin in this worker worktree; pin integration remains a separate maintainer
operation after the two commit series are reviewed together.
