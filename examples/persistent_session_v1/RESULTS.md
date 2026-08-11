# HTSIM-18 persistent session and CORE-24 result codec results

Run on 2026-08-11. The functional foundation passes: the opt-in backend
process retains native simulator state across flow steps, the accepted
one-GOAL path remains byte-identical, and the strict full `StepResult` codec
round-trips every frozen shape without float conversion. The retained-state
family passes `2/2` genuine-risk instances. The originally frozen wall family
is `0/2` because its BRIDGE-1-derived bands describe a different workload;
both signed speedup directions nevertheless pass. HTSIM-24 owns the clean
held-out wall-band residual.

## Chronology and provenance

The original expectations were frozen at
`e77f3854f544052af9b9c695391540ab88727bc8`, before either implementation and
before any result-producing run. The full result codec landed at
`92be65a11c3dd1fc87b88810b089dfc68b1a22ba`. The paired backend session landed
at `f8e1ee923a9c108cd698786c1824b9722d22d0e1`.

A wall-only expectations amendment was committed at
`71afffe602a527a5fde72e1e47a7987d85ebf479`. It correctly identified that the
historical 7.252-second BRIDGE-1 observation used complete TP-8 step programs,
whereas this study uses lightweight flow frames. Five exact base-binary CLI
calibrations informed its corrected bands. Final chronology review also found
that a precommit session smoke had already exposed session timings before the
amendment. The corrected bands therefore cannot be treated as independent
preimplementation expectations. Their outcomes are reported below as
post-specified diagnostics, not scored evidence.

The first registered-path run used a literal reset SQ watermark. Its summary
at
`${SIMLLM_DATA_ROOT}/persistent_session_v1_by_construction_withdrawn/summary.json`
has SHA-256
`35e4af390e8446cfc2d98bf841e70a5fc756e32fe82073a7897ae904615e33c7`.
That `2/2` retained-state headline is withdrawn because one subcheck was by
construction. Commit `c18bfffebe72d2c13033fc46db033ad9bba03700` replaced the
literal with a live fresh one-flow session observation without changing the
frozen relation or threshold. The exact registered path was then rerun.

The official external summary is
`${SIMLLM_DATA_ROOT}/persistent_session_v1/summary.json`, SHA-256
`1d9992f7d0a07f5458523e077316f7046037eeb34839a037a1d25a85140ef036`.
It records Linux x86-64 and Python 3.12.12. Raw GOAL, binary, CSV and framed
session artifacts remain outside Git. The exact binaries were:

| binary | SHA-256 |
|---|---|
| `htsim_rnic` | `7ee8b0d8ef522dfe7cdeceeb21f6bf077b1640548899d2b99bd2d8a94f927e9f` |
| `txt2bin` | `5e142f9761a7845676697f1fe959e39c2887c6f4e386cddc2813ababd38e0245` |

## Scored retained-state relation

Each row compares the second overlapping same-source flow with a fresh reset
execution. Persistent and reset SQ high-water marks are both live native
session observations. The reset FCT is independently observed through the
one-GOAL CLI and tied to the fresh one-flow session by a fatal exact check.

| payload, bytes | reset FCT, ps | persistent second FCT, ps | reset SQ high | persistent SQ high | result |
|---:|---:|---:|---:|---:|---|
| 4096 | 2166400 | 2249600 | 1 | 2 | PASS |
| 8192 | 2249600 | 2416000 | 1 | 2 | PASS |

Both instances pass the registered signed direction: the persistent second
flow has greater latency and greater queue occupancy than the reset model.
The genuine-risk fraction is `2/2` (100 percent).

### Entailment analysis

The fatal stateless-equivalent identity family does not entail this relation.
It uses nonoverlapping flows separated by 10,000,000 ps. The scored family
uses two flows eligible at 0 ps and reads the second FCT plus both native SQ
high-water marks before applying any explanatory packet arithmetic. No earlier
fatal check fixes the persistent second FCT or either high-water predicate.
A server that recreates state per injection, loses FIFO state, drains early or
projects a parallel ledger can reach and fail either row.

The exact reset session FCT equality is fatal-unscored. It pins only the reset
observation and does not constrain the persistent second flow. The withdrawn
literal watermark is not counted anywhere in the scored denominator.

## Wall-clock evidence and residual

The official raw observations were:

| replay | isolated, s | session, s | speedup | signed direction |
|---|---:|---:|---:|---|
| two-node | 0.009332766 | 0.002623037 | 3.5580x | PASS |
| four-node | 0.016693106 | 0.002864300 | 5.8280x | PASS |

The original preimplementation bands were `[10, 25]` and `[0.01, 8]` seconds
for two nodes, and `[20, 45]` and `[0.01, 10]` seconds for four nodes. Both
instances miss both lower band bounds because those values came from a much
larger BRIDGE-1 workload. The originally frozen wall family is therefore
`0/2` genuine-risk instances, even though both original speedup thresholds
and both signed directions pass.

The amended lightweight bands pass `2/2`, but that figure is diagnostic only
for the chronology reason above. It is excluded from the scored headline and
genuine-risk numerator. HTSIM-24 `(Precision; P1; S)` requires a held-out
replay and bands selected without observing its session result.

The byte-identity family does not entail either wall observation because
simulated picoseconds place no bound on host process, framing, scheduling or
parsing time. Both original wall instances reached their checks and could
fail. Their scored genuine-risk fraction is `0/2` (0 percent). Across the two
valid scored families, the aggregate is `2/4` (50 percent).

## Fatal and unscored evidence

Fatal invariants and structural guards do not increase a scored denominator.

### Stateless-equivalent latency identity

| replay | CLI FCT stream, ps | session FCT stream, ps | result |
|---|---|---|---|
| two-node | `[2166400,2249600]` | `[2166400,2249600]` | byte-identical |
| four-node | `[2166400,2249600,2166400,2249600]` | `[2166400,2249600,2166400,2249600]` | byte-identical |

Both runs reported physical quiescence and exact completion cardinality. The
session lifecycle rows came from the existing native WQE and API completion
records, not a parallel comparator.

### One-GOAL off path

The base and new binaries were invoked with identical argv and artifact paths.
Each pair compared equal with `cmp`:

| artifact | shared SHA-256 |
|---|---|
| help stdout | `314816a5056f1b84313e97fce9f620784bc23c8af1823f642e0ca1d72f062e4d` |
| one-GOAL stdout | `5404ab442bde40b4e8d6db0b924c147505e5979b46d26ad4929b4357e0dadbb0` |
| stderr | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| completion CSV | `39049963fb162de74e388c7642909ecff2163ea27d000ab8a2fb8a8f30a57dfc` |

Both invocations exited zero. This is an exact compatibility invariant, not a
behavioral score.

### Protocol and authority guards

Seven focused C++ tests pass independently: retained queue state and conserved
drain rows; duplicate sequence rejection; skipped sequence rejection; stale
horizon rejection; explicit post-terminal rejection; partial-body atomicity;
and noncanonical JSON rejection. Rejection responses show unchanged native
post counters at each validation boundary. These checks are fatal-unscored
because they are protocol safety requirements.

A separate generated `rnic-cn` binary smoke opened a two-node session,
injected and completed one flow at 4835200 ps, drained a live native post count
of one and closed with empty stderr. This verified that legacy constructor
diagnostics cannot corrupt framed stdout. It is a support check, not a scored
study cell.

### Full StepResult codec

All four frozen in-memory to real JSON bytes to in-memory cases pass:

| case | canonical wire SHA-256 |
|---|---|
| empty | `03d9fba22e68cb84da67fb4a4d773ee317d73634f43f10cd96f7b5671983b26c` |
| prefill | `834b95de788fd42d29b2ab16a6756323f803bafb4736a90c3486dea9bcc397d4` |
| decode | `a05cbdfdf7e686f44f4bdc0e9d7b83d3ef2e5aaa16c5f985b9a12f0fe55b0231` |
| mixed | `fe1207c3574cb7b3863d2654c8f74d5d84ee098645f0ce97ea2f755650331b88` |

The decode and mixed cases retain TPOT as numerator 1 and denominator 3.
Strict negative tests cover unknown and missing fields, booleans in integer
fields, attribution nonconservation, invalid or unreduced rational values,
duplicate request identities and the unsupported payload-less legacy name.
These exact round trips are contract invariants and remain fatal-unscored.

## Closure scope and acceptance mapping

HTSIM-18 closes for its functional protocol and state-retention clauses:

| registered acceptance clause | evidence |
|---|---|
| "genuinely persistent, opt-in stdin/stdout session" | The `--flow-session` branch owns one `FlowSession` from open through close; the scored second-flow FCT and SQ watermark differ from fresh reset. |
| "one-GOAL CLI remains the exact off path" | Help, stdout, stderr, completion CSV, exit status and both stateless FCT streams match the base binary. |
| "32-bit big-endian byte length followed by one canonical JSON object per frame" and the five verbs | Binary framing study, canonical reader and writer, seven focused protocol tests, and successful `rnic-nn` plus `rnic-cn` sessions. |
| "advance names the causal boundary that may run" | The frozen boundary is inclusive virtual-time `through_ps`; nonoverlapping identity and overlapping retention cells execute it, and stale horizons reject before another post. |
| "Duplicate, skipped, stale or post-terminal sequences fail before authority mutation" | Focused rejection tests inspect native authority counters at the failure response. |
| "a disconnected client cannot silently commit a partial frame" | The partial-body test returns only the preceding open response, no accepted injection and a truncated-frame diagnostic. |
| "event list, topology, native RNIC session and transport policy state live from open through close" | The two live retention rows, exclusive native counters, SQ high-water marks, generated `rnic-cn` smoke and quiescent drains. |
| "two steps to prove state retention and exact completion/bookkeeping conservation" | `2/2` live state rows plus fatal completion cardinality, native alias, lifecycle ordering, authority-counter and quiescence checks. |

The wall-band clause added by the wave task is not included in this closure.
Its clean independent proof moves to HTSIM-24.

CORE-24 closes completely:

| registered acceptance clause | evidence |
|---|---|
| "strict, versioned full StepResult wire codec" | `simllm-step-result-v2`, shared strict wire helpers and focused negative tests. |
| "carry step identity and boundaries, every RequestMetric, exact rational TPOT, the conserved LatencyAttribution partition and separately typed AdditiveVisitTotals" | The four wire digests and field-shape tests cover every field and type. |
| "Preserve a strict reader for any accepted legacy result form" | Repository audit found no accepted legacy payload. The reader explicitly rejects the old name rather than fabricating CORE-5 fields. |
| "prove in-memory to wire to in-memory identity for empty, prefill, decode and mixed-request results" | Four of four real-JSON round trips pass, including exact `Fraction(1, 3)`. |

## Residuals and deliberate omissions

- HTSIM-24 `(Precision; P1; S)` owns only the held-out wall-band evidence.
- BRIDGE-2 `(Completeness; P1; L)` remains open. It must implement the
  graph-level process client, dependency-to-horizon lowering, lifecycle to
  `CompletionEvent` translation, bookkeeping append cursors,
  `ExecutionResult` construction, full-result reduction and transactional
  publication.
- No CORE residual is needed. CORE-28, CORE-29, HTSIM-25, HTSIM-26 and
  BRIDGE-6 remain unused because no distinct deferred scope was found.
- The session deliberately rejects `rnic-nn-fluid`; it has no structural
  native WQE authority to retain. The generated structural `rnic-nn` and
  `rnic-cn` paths are supported.
- No reader was invented for `atlahs-closed-loop-result-v1` because no accepted
  bytes exist. No bulk run artifact is tracked.
- The graph-level BRIDGE-2 client, portable killed-child cleanup owned by
  BRIDGE-3, physical control producers and dynamic links are not implemented
  in this change.

## Integrator-owned stale statements

The wave contract forbids this worker from editing these files. They still
describe CORE-24 or HTSIM-18 as open and require integrator reconciliation:

- `docs/README_PRO.md:372`
- `docs/README_PRO.md:465`
- `docs/architecture.md:432-435`

No stale hit was found in `README.md`.

## Repository gates

- `.venv/bin/ruff check .`: `All checks passed!`
- `.venv/bin/pytest -q`: `696 passed, 5 skipped in 15.56s`
- full release all-target CMake build: completed at `100%`
- full CTest suite: `100% tests passed, 0 tests failed out of 377`

The five Python skips are the repository's explicit optional-runtime and
absent-private-submodule cases. No test added by this change requires an
initialized `third_party/` checkout.
