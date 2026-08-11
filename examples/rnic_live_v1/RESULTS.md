# Live RNIC composition: Tier A results

Current disposition, 2026-08-11: this file is chronological. Statements in
the Tier A and Tier B sections that HTSIM-9 remained open describe those
earlier checkpoints. The Tier C section below records the later ABI-v2 packet
chain evidence and the HTSIM-19 off-path blocker. HTSIM-9 is still open because
no single qualifying current-pin outer run passed every frozen gate.

## Claim scope

This report claims TIER A of the frozen two-tier gate and nothing more.
Per the frozen expectations, until Tier B passes, the native composition
is component and step-sink evidence, BACK-8 remains open, and no
`CompletionEvent`, `ExecutionResult`, TTFT or TPOT claim is made here.
Packet-level first and last packet issue timestamps remain
`not_applicable` at the ABI-v1 flow-extent port; the packetized event
vocabulary is BACK-25 and BACK-26 scope, and HTSIM-9 therefore remains
open as an ABI-v1 checkpoint.

## Freeze provenance

The expectations were frozen before any implementation at commit
`65b5609`, amended pre-run at `facb26d` and `947399c`, with the final
lineage recorded at `d5d98a2`. The frozen file was never edited after
its freeze; the Tier A harness and its port-factory seam were landed and
separately frozen by the preparation work
([tier_a_harness_expectations.md](tier_a_harness_expectations.md),
fake-port precedent in
[tier_a_harness_results.md](tier_a_harness_results.md)). The Tier B run
expectations are frozen in [tier_b_expectations.md](tier_b_expectations.md)
with the review supplement, awaiting execution.

## The composed binary

The htsim side landed on the backend main through its PR 11: the
combined session implements `AtlahsFlowRuntime`, the SimLLM native RNIC
library is compiled from the pinned SimLLM checkout behind the
`HTSIM_ENABLE_SIMLLM_RNIC` build option, opaque tokens only cross the
boundary, structural mode constructs no legacy ledger (exclusivity
counters wired to observations), backpressure is relayed from htsim
rather than fabricated (a review round removed a fabricated-drop path
and added runtime-bound multi-flow tests plus a composed eight-rank GOAL
smoke with three overlapping flows), and completion projections carry
stable directed-pair transport identities. The submodule pin in this
change points at that backend main merge commit, which also carries the
full former addon-branch lineage, so the frozen bypass reference
`8c3f8b2` is reachable from the pinned history.

## Tier A evidence

Scored and fatal families, from the frozen acceptance harness driven
against the composed producer (`--factory htsim`), independently rebuilt
and rerun twice by the integrator from fresh build trees:

- Exact oracle rows: 8 of 8.
- D-additivity over payload by rate: 4 of 4 (the 1,000 ps native
  doorbell shift is additive at both link rates).
- Inverse-rate serialization: 4 of 4 (the wire term halves exactly when
  the rate doubles).
- Two-WQE FIFO: 4 of 4 (the frozen four-equation timeline, W1 queue wait
  exactly L(R), no bypass of W0).
- Fatal unscored invariant families, all holding: authority exclusivity,
  token conservation, controlled drop producing the modeled error
  completion and never a success CQE, FIFO ordering, terminal atomicity,
  quiescence, and wrapper-bypass sensitivity (the bypass mutant is
  rejected by the same D predicate that accepts the composition).
- Full backend test suite: 363 of 363 native tests on the composed
  build; the legacy `htsim_uec` one-flow scenario is byte-identical to
  the protected reference `8c3f8b2` (stdout, stderr, `logout.dat`,
  `idmap.txt`, `flowsInfo.csv`).

Step-sink replay half of Tier A: the complete pre-registered m4 study
(sections A through E, 36 rows: fluid closed forms, packetized point
forms, the tp=8 replay, and both recorded frontend replays) reran with
`SIMLLM_HTSIM_RNIC` pointing at the composed binary. Every row
reproduced with zero FAIL verdicts and every residual exactly 0 ps, so
the composed binary is a drop-in producer for the accepted step-sink
path.

## What Tier A passage does not claim

The DeviceRuntime-to-TTFT/TPOT chain (Tier B) has not been executed
against the composed binary; its expectations are frozen and its
producer invocation contract is pinned, so the run requires no further
agreement. BACK-8 remains open for live metric reachability. The
per-WQE packet-issue timeline remains unclaimed at ABI-v1.

Postscript, 2026-08-11: BACK-25 and BACK-26 later closed at the vocabulary and
relay boundary; see [the ABI-v2 packet study](../rnic_packet_v2/RESULTS.md) and
its registered producer residuals.

## Tier B live reachability

Tier B passed on 2026-08-11 and makes the first network-affected TTFT and TPOT
claim in this repository. The claim is limited to the frozen isolated
`rnic_live_v1` fixture: one request, one prefill graph, two decode graphs,
single-WQE payloads of 4 KiB and 1 MiB, the two-WQE 4 KiB FIFO, link rates of
200 and 400 Gbit/s, and native doorbell service of 0 and 1,000 ps. It does not
generalize these exact relations to congestion, headers, propagation, control
frames, packetized issue timing, or arbitrary application graphs.

The original Tier B expectations were frozen at `fc3836d`, and the review
supplement and machine-readable schema were frozen at `067cbfb`. Their final
pre-run SHA-256 values remained:

- `0058fbe9a2fe3892f739af736ce2523250e1274d4731caaef64d00497482460c`
  for the two-tier expectations;
- `be540c2b30aa300a6f92e31ca3f8bd724ca0c9e8aba6ed94240444c33693603d`
  for the original Tier B relations;
- `99c8206e9f434f671807cdcf5f060ce596794213e75da64647a93624f3f31ab2`
  for the review supplement; and
- `ef27bf3ff455fe6144337573d752bd671e8b7b6730e4915b7fdcf4f57ac7fe3d`
  for the machine-readable schema.

Implementation `8a95e3b` connected immutable composed observations to the
runtime's structural transaction and completion projection. The first
result-producing attempt stopped before publishing raw observations or a
result ledger because the protected DCQCN reference rejected a 400 Gbit/s
override against a tracked 100 Gbit/s topology. Review commit `42222d7`
corrected only that producer fixture by deriving the same 32-node topology at
400 Gbit/s for both reference and candidate runs. The successful rerun used
the unchanged registered command and frozen producer argument vector.

The composed build source resolved to the pinned htsim gitlink
`edb28c3015c173b4251abc5858c587df325e1ebc` and source tree
`238f6deab7a98efffe51ec619dd1352cbe4bb2e0`. The SimLLM native sources came
from frozen base `fc282efc91573638de7dcfae2befee1cf022011b`, tree
`ad5ed557a541c26ad061f47ac7d3a12deb223da6`. Bypass reference binaries were
rebuilt from the frozen `8c3f8b231a6a9311ffc1e7969a003dcba724b50d`
reference, while candidate bypass binaries came from the pinned main commit
with structural RNIC composition disabled.

The independent checker reported these genuine-risk fractions:

- single-WQE D additivity: 4/4;
- single-WQE inverse-rate serialization: 4/4;
- live StepResult, TTFT and TPOT forms: 8/8;
- single-WQE seven-component rows: 8/8;
- two-WQE FIFO contention: 4/4; and
- bypass artifact identity: 4/4.

All 8 single-WQE and 4 FIFO exact rows passed. Increasing D added exactly
1,000 ps to each single-WQE step latency, TTFT and decode TPOT. Doubling R
halved the wire term while retaining D as an additive constant. The FIFO W1
queue wait was exactly 163,840 ps at 200 Gbit/s and 81,920 ps at 400 Gbit/s in
every step. The objectively selected doorbell projection was `nic_owner`:
doorbell wait was zero, doorbell service was D, single-WQE request attribution
was `queue_ps = 0` and `nic_ps = D + L`, and FIFO attribution was
`queue_ps = L` and `nic_ps = D + L`. Every request's seven components summed
exactly to its live latency. Additive visit totals remained separate.

The four retained bypass profiles were `rnic-nn-fluid`, `rnic-nn`, `rnic-cn`
and `dcqcn`. Completion CSV bytes, canonical completion rows and JCT, replayed
StepResult tuples, and TTFT/TPOT summaries were byte-identical between the
protected reference and pinned candidate for every profile. All twelve fatal
unscored invariant families and all five checker-sensitivity controls held.
The raw observations contained no expected values or producer verdicts.

The published raw-observation and result SHA-256 values are
`acaca5c57134848a314a92d223c283a7dc63f1c3ef964f65f7dea75487d6dfa1` and
`3755bf5c2b37e9c30f90f97e3d6920841c70d052ff9164434d26c4f56773f0ed`.
CORE-15 and BACK-8 close on this live metric evidence.

Packet-level first-packet and last-packet issue timestamps remain outside this
claim. ABI-v1 exposes network acceptance and whole-flow terminal timestamps,
which are not substitutes for packet issue. BACK-25 owns the versioned packet
attempt and TX/RX observation vocabulary, BACK-26 owns transport-control
events, and HTSIM-9 remains open until their packet-issue evidence populates
the native timeline.

## Post-specified Tier B review correction and round-1 rerun

This section was added after the four-lens review. It corrects the published
claim language without changing any frozen expectation, relation, matrix or
observation field. Checker and documentation corrections are commit `5769447`.
The registered Tier B command then reran against that commit on 2026-08-11.
The composed build reported htsim describe `edb28c3`; its full source commit
was `edb28c3015c173b4251abc5858c587df325e1ebc`, exactly equal to the pinned
submodule gitlink.

The phrase "the first network-affected TTFT and TPOT claim in this repository"
above is overbroad. The supported claim is the first TTFT and TPOT evidence
through the composed native RNIC chain. Earlier repository studies changed
TTFT through other network paths. Even within the composed chain, the claim is
limited to the frozen isolated fixture. Congestion, headers, propagation,
control frames, packetized issue timing, arbitrary application graphs,
multi-request and scheduler contention, GPU compute service,
compute-communication overlap, and host-side software service remain outside
the claim.

The Tier B gate demonstrated these CORE-15 clauses: structural native timing
reached a changed graph completion, `ExecutionResult`, `StepResult`, TTFT and
TPOT; the native session remained the sole structural WQE authority; and the
separate bypass rows retained their protected artifacts. The gate did not run
one fixed contended graph through both bypass and composed native authority,
and therefore did not measure the registered signed JCT difference between
those two modes. CORE-21 now owns that comparison and requires real
`StepResult` replay on both sides. Failed adapter transaction atomicity is
unit-test evidence from `tests/test_composed_rnic.py`, not Tier B run evidence.
That test shows that a rejected adapter transaction consumes neither native
observations nor runtime state before a later valid transaction commits.

BACK-8 likewise closes only for clauses supported across its component, Tier A
and Tier B evidence. The session-record study supports policy-invariant
hardware hashes, versioned records, projections and authority controls. Tier A
supports direct composed WQE, FCT and JCT movement plus step-sink replay. Tier
B supports the live core metric projection and four retained bypass profiles.
BACK-31 retains the executable-level unlinked-native negative that the gates
did not run. HTSIM-1 retains explicit rejection of the unsupported `rnic-ss`
legacy profile. Packet-attempt and transport-control vocabulary remains with
BACK-25 and BACK-26, and HTSIM-9 retains the composed first-packet and
last-packet issue evidence.

The earlier description of bypass artifact classes 3 and 4 as "replayed
StepResult tuples" is incorrect. The producer synthesized the StepResult tuple
array and request TTFT/TPOT summary from the scalar JCT by formula; neither
artifact traversed StepResult machinery. Consequently the bypass identity
family's discriminating power rests on class 1, completion CSV bytes, and
class 2, canonical completion rows and JCT. The review correction now routes
all four comparisons through the repository-standard `BypassArtifacts`
comparator in `simllm/backends/rnic_records.py`, while CORE-21 retains real
same-graph StepResult replay.

The checker now derives every published fatal-unscored boolean from the
predicate that enforces it. FIFO W0 then W1 SQ, CQ and completion order is a
fatal rejection path rather than part of the scored FIFO family. The scored
FIFO family retains the frozen wait and live metric magnitudes. Bypass guards
now reject empty behavioral artifacts and require different SHA-256 values for
the reference and candidate executables. The executable hashes used by the
rerun were:

- reference RNIC: `b156414b758fa54eb74251ce5aa02adf4c5d80ef5555cf3945b2c5e40322beeb`;
- candidate RNIC: `aeb2ce155ed69d8cd697a31eb28e8eed6455ce3f69c5d024e64e546ebc579c9e`;
- reference DCQCN: `e1f215575d30ddd6df8f8bf5525d5462bd2ae6588f8de4b67f91cc6de83e06b4`;
  and
- candidate DCQCN: `c62f751fd2ad5109cd5238cf02cba0e284f0949d14cfb9008d03423f9446b649`.

The count of five checker-sensitivity controls was not registered in any
frozen file. It is post-specified diagnostic evidence, not another scored
denominator. After the correction, all five controls mutate a complete raw
observation and route it through the deployed Tier B checker. This includes
the event-object-reuse control, which now removes one callback index from a
real observation rather than comparing two locally constructed lists.

The six published family fractions are not six independent risks.
`single_wqe_d_additivity` and `single_wqe_inverse_rate` overlap with
`single_wqe_metric_forms`: they use the same structural cells and the same
step-latency, completion, TTFT and TPOT values, so a defect can produce paired
misses. The separate fractions remain the frozen reporting form and must not
be summed into an independent-risk total.

The frozen expectations permitted correction of a genuine producer-fixture
defect after a failed attempt without changing the registered relations or
producer contract. Commit `42222d7` made that permitted correction by deriving
the same 400 Gbit/s 32-node DCQCN topology for reference and candidate runs.
The minimizing word "only" in the earlier chronology paragraph is withdrawn.

The round-1 rerun reproduced every family fraction: 4/4 D additivity, 4/4
inverse-rate serialization, 8/8 live metric forms, 8/8 component rows, 4/4
FIFO contention and 4/4 bypass identity. `raw_observations.json` reproduced
byte for byte at
`acaca5c57134848a314a92d223c283a7dc63f1c3ef964f65f7dea75487d6dfa1`.
`results.json` also reproduced byte for byte at
`3755bf5c2b37e9c30f90f97e3d6920841c70d052ff9164434d26c4f56773f0ed`.
The stricter fatal-boolean provenance changed how the booleans are derived but
did not change their serialized shape or values, so no result hash change
occurred.

## Tier C ABI-v2 packet-chain run and HTSIM-19 blocker

### Frozen scope and chronology

The additive Tier C expectations were frozen at
`2bd61cdfe7b6d545c05ea17db6894bb50eb14735`, before the producer, checker or
result-producing run. The machine-readable expectations SHA-256 is
`c2d8ffcf36c54c9ac5ddf2b89e1cf57317ede5031baf3afd05f9ef56b5fb1358`.
The freeze commit records a clean tracked tree and one untracked dry-run
harness containing only frozen literals and check-only validation. The
registered check-only command had passed without producing artifacts. The
audited backend remained
`4885c647eecdfdf81479d1df052223c016ad086b`.

Implementation `42a4a12e541cab666d996087779e302e278c6c0e` added ABI-v2
packet parsing and live projection. Three machinery corrections then passed
the frozen check-only registry without changing its relation set:

- `11f555c919e64a60abb14419db347994bd8c3e75` forwarded the inherited Tier A
  run-root guard;
- `51d85e4ff67b307a19872b5af04cbbd85dd5116e` supplied the byte-identical
  Tier B topology from the audited checkout when private submodules are absent;
  and
- `73ffb43ee192126631f0ac80d461a70c0149d8cf` made the path-launched Tier C
  runner importable.

The first two full attempts stopped before Tier C observations existed, first
on the missing Tier A run-root handoff and then on the uninitialized private
submodule topology path. The next current-pin attempt passed every native and
structural gate but found the ABI-v1 identity defect described below. A later
attempt reproduced accepted ABI-v1 bytes by substituting the older accepted
Tier B candidate executables, then reached the Tier C checker after the import
fix. That binary substitution was chosen after observing the current-pin
failure. Commit `ee4b85a87da7a197e3d06a11eb5e9b0ceac3b2e8` therefore removed
it and restored the frozen requirement that Tier B use the RNIC and DCQCN
executables built from the audited pin. History was not rewritten.

The final registered command at `ee4b85a` used output
`$SIMLLM_WAVE5_RUN_ROOT/codex/htsim9_packet_closure`. It passed ruff, 685
pytest tests with 5 skips, all 370 htsim CTest cases and all 6 standalone
native CTest cases. Tier A reproduced both accepted digests. Tier B then
failed its fatal off-path gate at bypass identity 2/4, so the outer runner
published no Tier C observation and no top-level accepted result.

### Current-pin ABI-v1 failure

The current-pin Tier B raw and result digests are respectively
`d04ff7e6fddb5c35f487b50b5bd0ea61a8265a3a8fe732d5ce9620f85cf6b850`
and
`d25cd2876a211a6b4aadd9cc192c5b2e2f9799c4775f481adca87f0db0b1ff36`.
All structural families retained their full fractions: 4/4 doorbell
additivity, 4/4 inverse rate, 8/8 live forms, 8/8 component rows and 4/4 FIFO.
The `rnic-nn-fluid` and DCQCN bypass rows also remained exact. Only `rnic-nn`
and `rnic-cn` changed their completion CSV and canonical completion rows:

- `rnic-nn` retained FCT 165,120 ps, but WQE/RQ/CQ identity changed from
  `97/34/65` to `1/0/1`;
- `rnic-cn` retained FCT 6,161,920 ps and transport object 97, but WQE/RQ/CQ
  identity changed from `1089/34/65` to `1/0/1`.

The inputs, FCT, scalar-derived StepResult tuples and request summaries were
unchanged. The repository-standard `BypassArtifacts` comparator rejected the
two changed behavioral artifact classes. HTSIM-19 owns restoration of this
current-backend compatibility path.

### Frozen-relation packet diagnostic

The packet mechanism itself reached the live chain in two nonqualifying forms:
a direct Tier C invocation after the fourth outer attempt had already stopped,
and a post-specified hybrid outer invocation that used the older accepted Tier
B candidates. Both used the current `4885c64` composed build for ABI-v2
production. The hybrid output is retained at
`$SIMLLM_WAVE5_RUN_ROOT/codex/htsim9_packet_closure-postspecified-hybrid`.
Its overall, Tier C raw and Tier C result SHA-256 values are respectively
`ea9125c76855ea8dc1fcf92fc2541689a8089d29e7c1247cc5c24bb2c18c336b`,
`41138345b6aa306db91dcf929b5bbf9cdbf3a649a36cee2142c4ad755e8eef84`
and
`143303bef066172a0964afac0e81b90b49f06f39db87ab82ce2dca2c99efedf8`.
These are diagnostic evidence against the frozen relations, not an accepted
closure result.

The diagnostic genuine-risk fractions were:

- doorbell packet-to-live chain: 4/4. In every payload-rate instance and all
  three request steps, changing native doorbell service from 0 to 1,000 ps
  moved first packet, last packet, `CompletionEvent.STARTED`, step latency and
  TTFT by exactly `+1000 ps`; both TPOT values moved by `+1000 ps`, and the
  absolute step completions moved by `+1000`, `+2000` and `+3000 ps`;
- link-rate packet-to-live chain: 4/4. Slow-minus-fast first-packet movement
  was 0 in every instance. At 4 KiB, last-packet movement was 0 and each live
  metric moved by `+81920 ps`. At 1 MiB, last-packet movement was
  `+20889600 ps` and each live metric moved by `+20971520 ps`.

The two fractions are overlapping relation families and are not summed into
an 8/8 independent-risk headline. All 8 single-WQE and 4 FIFO exact rows held.
The acceptance-surrogate, producer-constant and missing-TX-start mutants were
all rejected. Packet closed forms, TX-start origin, event projection,
acceptance and terminal separation, inherited live-chain invariants and the
ABI-v1 digest checks remained fatal and unscored. In particular, the ABI-v1
digest check is a run gate, not a behavioral family.

For the 1 MiB, 400 Gbit/s, zero-doorbell cell, network acceptance and first
packet were both at release offset 0, last packet was at 20,889,600 ps and the
whole-flow terminal was at 20,971,520 ps. The acceptance-surrogate mutant
copied acceptance into both packet fields and failed. The explicit TX-start
origin and closed-form checks also prevent a whole-flow terminal from serving
as packet issue.

### Entailment analysis

The deployed Tier C checker evaluates both scored families directly against
raw observations before either the per-cell packet exact oracle or the
inherited Tier B checker. The explicit-origin guard only requires each cell's
packet fields to equal the minimum and maximum data or retransmission TX-start
events for that WQE. It does not fix the cross-cell doorbell or rate movement,
the `CompletionEvent` projection, or any live request metric. A valid TX stream
can therefore reach and fail either scored family before an exact oracle pins
the same quantity. The checker result records
`scored_evaluation=raw_observations_before_exact_oracles`, with both later
oracle orders explicit. This is genuine-risk diagnostic evidence. It is not
promoted to closure because the qualifying outer run never reached it.

### HTSIM-9 acceptance-clause mapping

The registered closure clauses and their dispositions are:

1. "one composed run of the Tier B class passes": not demonstrated. The
   current-pin outer run stopped at the fatal ABI-v1 Tier B identity gate;
   the hybrid run used a different Tier B candidate provenance.
2. "ABI-v2 packet-issue evidence populating the native timeline through
   `ExecutionGraph` to `CompletionEvent`, `StepResult`, TTFT and TPOT":
   demonstrated by the frozen-relation diagnostic, but not inside the single
   qualifying run required by clause 1.
3. "Network acceptance and whole-flow terminal events do not satisfy that
   evidence": demonstrated diagnostically by the 1 MiB separation cells, the
   explicit TX-start origin guard and the rejected acceptance surrogate.

Because clause 1 is unmet and clause 2 is not demonstrated within its required
run scope, HTSIM-9 does not close. HTSIM-19 `(Precision; P0; M)` is the exact
residual blocker. No broader packet-chain claim is made.
