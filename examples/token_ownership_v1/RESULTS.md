# Token ownership v1 results

TRAF-25 is complete. One `StepRecord` now represents one engine scheduler's
tokens in both TP and MoE traffic. Captured routing requires one declared
`engine_rank`; only that rank dispatches the scheduled tokens. The other EP
ranks own experts and return combine traffic, but carry no scheduled tokens in
this isolated projection. The no-supply approximation uses the first EP rank
under the same convention.

The old source-multiplied behavior was wrong. It is not retained as a
compatibility mode. A full DP times EP population is a different model that
needs explicit peer workloads and independent peer routing; TRAF-26 owns that
work. Copying the same captured routing table onto every peer remains forbidden
because it would manufacture correlated hot-expert incast.

The primary result is 3/3 genuine-risk families and 9/9 instances. All fatal
exact, conservation, source, request, profile and quiescence checks passed.

## Chronology and provenance

The expectations-only commit is `cdf03d2`. Its registered `--check-only`
command validated the frozen arithmetic without importing SimLLM, reading the
capture, invoking a native executable or creating an artifact. It preceded the
implementation and every result-producing run.

The implementation sequence was:

1. `b67e44f` added the declared engine owner, single-source captured and
   uniform rendering, per-request attribution and the EP-width-8 conservation
   test.
2. The first corrected study execution exposed a harness error that scored a
   direct diagnostic instead of the registered live `StepResult`. That failed
   run is preserved and contributes no pass.
3. All-local routed layers exposed a second projection edge. `c221735`
   retained their zero-byte semantic collective frontier while assigning zero
   backend service. The authoritative ownership run observed this revision.
4. A blast-radius execution then found that the per-request study still used
   the deliberately restricted monolithic serial graph renderer. `937de03`
   moved that check to the ordered execution-graph projection. Its failed
   predecessor is preserved and contributes no pass.

The primary `summary.json` has SHA-256
`cd0b35d8df927716dcfa768b6a5b03bcf24f7bdcd719fbf6a517d8ae52b92275`.
It records SimLLM revision
`c2217356e5c456256557072ba9723945cb69f8a2`, observed htsim gitlink
`fc4400e4ca619223481536632074045cb6af2756`, and the separately observed
native executable hashes. No frozen literal is compared with the live
submodule pin.

## Declared population

The measured Granite cell is prefill step 0: 54 scheduled tokens, 24 MoE
layers, top-k eight, 2,048 bytes per routed hidden vector and expert owner
`expert_id % W`. Rank 0 is the concrete engine owner at every EP width. The TP
payload remains `54 * 2,048 = 110,592` bytes per TP collective. MoE uses those
same 54 tokens and does not infer seven peer scheduler populations.

The corrected EP-width sweep separated population work from critical-rank
work:

| EP width | Old total bytes | Corrected total bytes | Old/corrected | Old peak egress | Corrected peak egress | Old/corrected |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 10,612,736 | 5,304,320 | 2.000772 | 5,306,368 | 2,652,160 | 2.000772 |
| 4 | 58,773,504 | 14,594,048 | 4.027224 | 14,792,704 | 7,297,024 | 2.027224 |
| 8 | 207,499,264 | 25,563,136 | 8.117129 | 27,060,224 | 12,781,568 | 2.117129 |

At width 8, the often-quoted division by eight gives 25,937,408 bytes, but
that is the mean over eight possible home ranks. The concrete rank-0 home
omits its own local destinations and emits 25,563,136 bytes. Total group work
therefore fell by 8.117 times, while peak per-rank egress fell by only 2.117
times. This is the expected distinction between token population and the
bandwidth-critical rank, and it rules out a blind byte division.

Per-request ownership also changed by the same source correction:

| Request | Tokens | Old rows | New rows | Old bytes | New bytes | Old canonical bytes | New canonical bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| `r0` | 22 | 2,688 | 336 | 84,439,040 | 10,403,840 | 80,824 | 10,104 |
| `r1` | 12 | 2,688 | 336 | 46,190,592 | 5,701,632 | 80,516 | 10,064 |
| `r2` | 20 | 2,688 | 336 | 76,869,632 | 9,457,664 | 80,810 | 10,102 |
| all | 54 | 8,064 | 1,008 | 207,499,264 | 25,563,136 | 242,146 | 30,266 |

Every dispatch attribution row now names rank 0 as source. Every combine row
is the exact transpose. The corrected aggregate GOAL changed from 334,432
bytes, 2,688 sends and 207,499,264 send bytes to 47,399 bytes, 336 sends and
25,563,136 send bytes. Its SHA-256 changed from
`08a0403af66ff8a9d6b18f93afd15ae0bc925cc85555acf8a0593438a3d7bc92`
to `6bb83366a3936bcf1e435cab008bb55c2966777528a9e8d885dd44d47f5a4943`.

## Live makespan and physical sanity

The live graph-projected `StepResult` is the metric authority. The direct GOAL
run remains a diagnostic because it has different participant-local frontier
semantics.

| Profile | Rate | Old defective ps | Corrected live ps | Corrected/old | Change ps |
|---|---:|---:|---:|---:|---:|
| `rnic-nn-fluid` | 400 Gbit/s | 974,838,253 | 706,622,768 | 0.724862 | -268,215,485 |
| `rnic-nn` | 400 Gbit/s | 991,051,680 | 724,527,360 | 0.731069 | -266,524,320 |

The corrected 200 Gbit/s fluid result was 1,217,885,488 ps. Its increase over
the 400 Gbit/s result was 511,262,720 ps, exactly the additional full-rate
serialization of 25,563,136 bytes when bandwidth is halved, before the shared
48 ps aggregate rounding offset.

Physical sanity was checked before accepting the measured values:

- Network floor: `12,781,568 * 8 / 400e9 = 255,631,360 ps`. The fluid result
  is 2.764 times this floor and the packet result is 2.834 times it.
- Compute and propagation: the fixed roofline compute term is 99,360,000 ps,
  and 48 directed phases at 2,000,000 ps contribute 96,000,000 ps. Both live
  results sit above each independent lower term.
- Scaling and ceiling: halving bandwidth moved only serialization by the exact
  511,262,720 ps relation. Both 400 Gbit/s results are far below the
  conservative 25,574,622,720 ps ceiling that serializes every remote vector
  and charges propagation separately.

These bounds and scaling relations reject an impossible result, but they do
not constitute absolute hardware calibration. The locality bandwidth and
same-generation calibration residual remains TRAF-11.

For reference, the direct diagnostic reported 970,339,371 ps at 200 Gbit/s
fluid, 582,235,435 ps at 400 Gbit/s fluid and 594,423,040 ps at 400 Gbit/s
packet. These values are not substituted for the graph-owned live result and
do not enter the score.

## Evidence accounting and entailment

The scored observations were evaluated before any exact oracle that could pin
them:

| Family | Instances | Result | Raw observation |
|---|---:|---:|---|
| TRAF-B1 population response | 3 | 3/3 | Raw legacy-to-corrected total-byte ratios over EP width. |
| TRAF-B2 critical-rank response | 3 | 3/3 | Raw peak-rank egress ratios, including the non-eightfold width-8 response. |
| TRAF-B3 live makespan response | 3 | 3/3 | Raw 200 and 400 Gbit/s fluid `StepResult` values and the 400 Gbit/s packet value. |

The genuine-risk fraction is 3/3 families and 9/9 instances, or 100 percent.
A competent but wrong implementation could keep the source loop, divide bytes
without changing source identity, choose a different home rank, lose request
attribution or fail to move the live bottleneck.

Exact byte and peak rows, request sums, source identity, transpose, TP payload,
input hashes, backend quiescence, dense and drain bypasses and native identity
are fatal-unscored. The EP-width-8 unit fixture independently projected 42
dispatch-plus-combine hops, observed the same 42 renderer hops and enforced an
upper bound of 48. The Granite study independently projected and observed
12,482 hops against the 20,736 bound. These conservation and by-construction
guards add no scored pass.

## Blast-radius corrections

The original source-multiplied literals remain historical records, not
acceptance oracles. The tables below list every published numeric surface that
changed in the six named consumers.

### framework_oracle_v1

No published number changed. This study compares each request's two routing
decisions from an already declared `source_rank`; it does not call the traffic
renderer. Its three changed-byte rows remain 0 bytes, and its behavioral
headline remains 7/8. The expensive CPU build and framework run were not
repeated because there is no affected traffic projection to observe.

### nvlink_locality_v1

The rerun passed 3/3 genuine-risk families and 8/8 instances. The historical
2/3 and 6/8 outcome remains valid for its old source-multiplied workload; this
post-specified refreeze does not rewrite that chronology.

| Vector bytes | Placement | Old total | New total | Old fabric | New fabric | Old NVLink | New NVLink | Old service ps | New service ps | Old JCT ps | New JCT ps |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | `AAAA` | 11,870,208 | 2,983,936 | 0 | 0 | 11,870,208 | 2,983,936 | 7,097,000 | 4,538,000 | 7,121,000 | 4,562,000 |
| 1,024 | `AABB` | 11,870,208 | 2,983,936 | 7,913,472 | 2,011,136 | 3,956,736 | 972,800 | 2,442,000 | 2,194,000 | 139,195,840 | 136,246,720 |
| 1,024 | `ABCD` | 11,870,208 | 2,983,936 | 11,870,208 | 2,983,936 | 0 | 0 | 0 | 0 | 156,569,755 | 155,702,768 |
| 2,048 | `AAAA` | 23,740,416 | 5,967,872 | 0 | 0 | 23,740,416 | 5,967,872 | 14,156,000 | 9,047,000 | 14,180,000 | 9,071,000 |
| 2,048 | `AABB` | 23,740,416 | 5,967,872 | 15,826,944 | 4,022,272 | 7,913,472 | 1,945,600 | 4,838,000 | 4,358,000 | 182,367,680 | 176,469,440 |
| 2,048 | `ABCD` | 23,740,416 | 5,967,872 | 23,740,416 | 5,967,872 | 0 | 0 | 0 | 0 | 217,222,486 | 215,381,488 |

Positive ordered pairs changed from 576 to 144; the phase count stayed 48.
The 1,024-byte direct GOAL changed from 72,819 bytes and SHA-256
`0417832c8788a0477d48b414cf2d8456b87215abd1d0193ba46fb8db46185d8a`
to 20,392 bytes and
`917961edf996753223857d64010fc61e4f6b08672f18dcadf42c70d60ee36c4a`.
The 2,048-byte GOAL changed from 72,819 bytes and
`bcd72e63546d03efaddd48c16e160457d1e28f19795036d1f871788d78cf5a02`
to 20,392 bytes and
`16ee686eda4634886b117788b3893c893f5e12ea819736e0afdbdf63bab0e826`.

### per_request_fidelity_v1

The rerun retained the scored negative-control result at 2/2 families and 5/5
instances. All 12 native sanity cells now match their corrected literals.

| Epoch | Requests | Old GOAL bytes | New GOAL bytes | Old sends | New sends | Old send bytes | New send bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 744 | 336 | 6 | 2 | 48 | 16 |
| 0 | 2 | 952 | 540 | 8 | 4 | 80 | 32 |
| 0 | 3 | 964 | 548 | 8 | 4 | 128 | 64 |
| 1 | 1 | 744 | 336 | 6 | 2 | 48 | 16 |
| 1 | 2 | 960 | 544 | 8 | 4 | 112 | 48 |
| 1 | 3 | 964 | 548 | 8 | 4 | 176 | 80 |

| Epoch | Requests | Old 200G JCT ps | New 200G JCT ps | Old 400G JCT ps | New 400G JCT ps |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 8,003,280 | 4,002,640 | 8,002,640 | 4,002,320 |
| 0 | 2 | 8,003,920 | 8,003,280 | 8,002,640 | 8,002,640 |
| 0 | 3 | 8,004,560 | 8,004,560 | 8,003,280 | 8,003,280 |
| 1 | 1 | 8,003,280 | 4,002,640 | 8,002,640 | 4,002,320 |
| 1 | 2 | 8,004,560 | 8,003,920 | 8,002,960 | 8,002,960 |
| 1 | 3 | 8,005,200 | 8,005,200 | 8,003,600 | 8,003,600 |

For the epoch-0 two- and three-request permutations, request mismatches changed
from 12 to 8, request L1 error from 96 to 64 bytes and signed `alpha` error
from -16 to 0 bytes. Epoch-1 permutation values remained 4 mismatches, 32 L1
bytes and +16 `alpha` bytes.

The Granite request table is the old-to-new table in the declared-population
section. Its permutation changed from 5,348 to 668 request mismatches,
76,496,896 to 9,404,416 L1 bytes, and signed request errors
`(-38,248,448, +38,248,448, 0)` to
`(-4,702,208, +4,702,208, 0)` for `(r0, r1, r2)`.

### preplay_validation_v1

The preserved vLLM 0.26 replay inputs were rerun at both bandwidths. Each of
the five steps changed from 96 positive flows and 96 sends to 48 flows and 48
sends. Every old published JCT remained numerically unchanged:

| Step | 200 Gbit/s old and new ps | 400 Gbit/s old and new ps |
|---:|---:|---:|
| 0 | 320,157,120 | 208,090,560 |
| 1 | 103,888,320 | 99,956,160 |
| 2 | 103,888,320 | 99,956,160 |
| 3 | 99,956,160 | 97,990,080 |
| 4 | 99,956,160 | 97,990,080 |

The removed reverse flow did not own the two-rank critical path. The replay
half again passed 13/13 scored relations: all three TTFT changes remained
-112,066,560 ps, while TPOT changes remained -3,932,160 ps and -2,949,120 ps.
The already separate CPU oracle half was not rerun.

### routed_supply_v1

The captured source-0 direction remained critical, so all published JCTs and
bandwidth deltas stayed unchanged. The physical population did change:

| Surface | Old | New |
|---|---:|---:|
| Epoch-0 dispatch 0 to 1 bytes | 1,081,344 | 1,081,344 |
| Epoch-0 dispatch 1 to 0 bytes | 1,079,296 | 0 |
| Epoch-1 dispatch 0 to 1 bytes | 1,077,248 | 1,077,248 |
| Epoch-1 dispatch 1 to 0 bytes | 1,073,152 | 0 |
| Captured positive flows per step | 96 | 48 |
| Uniform positive flows per step | 96 | 48 |
| Uniform total directed bytes | 17,301,504 | 8,650,752 |
| Uniform GOAL bytes | 13,200 | 7,418 |

The uniform GOAL SHA-256 changed from
`d708e998685b617478e891b316728d14b8ac6185a62b73817f80af1c5adff518`
to `94f0a1f3a17f59db1a1a88c1885a5eae2f71c0e0703dbba1d4b055c0e567b21c`.
The four routed JCTs remained 182,531,520, 139,277,760, 182,203,840 and
139,113,920 ps for epoch 0 at 200G and 400G, then epoch 1 at 200G and 400G.

### routing_lifetime_v1

All 32 compatibility-object versus packed-arena traffic rows remained exactly
equal under the corrected source convention. The published step-0 GOAL changed
from 334,432 bytes and SHA-256
`08a0403af66ff8a9d6b18f93afd15ae0bc925cc85555acf8a0593438a3d7bc92`
to 47,399 bytes and
`6bb83366a3936bcf1e435cab008bb55c2966777528a9e8d885dd44d47f5a4943`.
The old archive equality is now expected to be false. The memory and lifetime
score remains 3/3 families and 6/6 instances because those relations do not
depend on source multiplication.

## Blast-radius artifacts

| Study | Result | Summary SHA-256 |
|---|---|---|
| token ownership | 3/3 families, 9/9 instances | `cd0b35d8df927716dcfa768b6a5b03bcf24f7bdcd719fbf6a517d8ae52b92275` |
| NVLink locality | 3/3 families, 8/8 instances | `1cf112a6b71209a17bc5b3e8524704af7bff6665225cf4e683e882a58c0ee84b` |
| per-request fidelity | 2/2 families, 5/5 instances | `f443b3e2a49705e8d0b25c9cf2325b1f1d9c77b8b8a23ad28f040697e6e705dc` |
| routed supply traffic | all traffic relations and exact oracles pass | `b24f234bf937d6c9ec1670bd92cf480781497872aaa65aedc06431edc9076a63` |
| routing lifetime | 3/3 families, 6/6 instances | `2488cbe14004fe02a9c1ccda2ba539a67f05e68d26cc96c244318705a501fdb1` |
| preplay replay half | 13/13 scored relations | `4e37d2339cb1d681b367a6667df482c041d662a671e759b3afc5b51caabbc522` |

Bulk artifacts are retained below the configured
`SIMLLM_TOKEN_OWNERSHIP_RUN_ROOT`. The legacy converter fault, wrong direct
scorer, pre-frontier corrected run, stale NVLink run, stale routing-lifetime
serial projection, per-request serial-projection failure and preplay run with a
missing native environment variable are preserved under explicit failed or
pre-fix labels. None contributes a pass.

## Closure mapping

TRAF-25 registered this acceptance:

> Make one declared EP engine rank the sole source of every token in one
> captured `StepRecord`; peer ranks own experts but hold zero scheduled tokens
> in this isolated projection. Remove source multiplication from both captured
> and uniform paths rather than preserving it as compatibility. Acceptance
> requires: (1) TP and MoE consume the same `record.total_new_tokens`
> population; (2) each request's dispatch bytes come only from its engine rank
> and combine is the exact transpose; (3) EP-width-8 tests conserve per-layer
> token sources, satisfy the independent
> `total_new_tokens * top_k * num_layers * 2` hop bound and agree with the
> routed-token projection; (4) the Granite EP-width sweep reports total bytes
> and peak-rank egress separately, then moves fluid and packet-level makespan
> in the preregistered direction above the physical serialization floor; and
> (5) every affected routed study is refrozen or explicitly reported as an
> unavailable rerun, with old and corrected published numbers listed.

The declared `engine_rank`, source-0 uniform approximation and unchanged
110,592-byte TP payload cover clause 1. The per-request table and transpose
oracles cover clause 2. The 42-of-48 unit fixture and 12,482-of-20,736 Granite
projection cover clause 3. The EP-width and live-makespan tables plus physical
bounds cover clause 4. The six consumer sections and retained result hashes
cover clause 5 only in part.

Clause 5 is not fully met, and this is the one deferred piece. A seventh
routed consumer was missed: `examples/dependency_authority_v1/run_study.py`
renders captured MoE traffic through `_routed_supply(_routed_projection(...))`
imported from `nvlink_locality_v1`, at lines 207, 213, 268, 271 and 406, and
this branch leaves it untouched. Its frozen registry at lines 22 to 57 still
carries pre-correction literals: `GRAPH_ARTIFACT_COUNT = 72`,
`LEGACY_GOAL_ORACLES` at 72,819 bytes, and
`LEGACY_JCT_PS = {1024: 156_569_755, 2048: 217_222_486}`. That study would
fail if rerun today, so it is reported here as an unavailable rerun rather
than as a refrozen consumer. It must be rerun and refrozen before any further
branch merges through this renderer.

## Contradiction sweep

The post-closure sweep found no contradictory token-population statement in
`docs/architecture.md`. It found two integrator-owned routed-supply index rows
that still say the absent-capture uniform path "stays byte-locked":
`README.md:167` and `docs/README_PRO.md:568`. That was true across the original
TRAF-2 captured-routing seam, but is no longer a current identity claim because
TRAF-25 intentionally corrected the uniform source population from 96 to 48
flows in the two-rank study. The rows are reported here and intentionally not
edited.

`docs/README_PRO.md:566` also points readers to the historical PLAY-5 result
without mentioning that its 96-send per-step traffic table has been superseded
by the corrected 48-send replay. The 13/13 scheduler, TTFT and TPOT result is
still correct, so this is a stale numeric-summary omission rather than a
contradictory behavioral claim. It is likewise left for integration.

## Repository gates

The first full native-enabled suite passed 1,039 tests and failed one stale M5
flow-count assertion. Its timing closed form already matched the single-source
star, but its comment and count still expected `W(W-1)` rather than `W-1`
flows per phase. Correcting that test oracle changed no implementation or
measured value.

The final gates are:

```text
.venv/bin/ruff check .
All checks passed!
```

```text
SIMLLM_HTSIM_RNIC=... SIMLLM_TXT2BIN=... .venv/bin/pytest -q
1040 passed, 4 skipped in 29.38s
```

`python3 scripts/task_progress.py --check` passed, and the path-portability
plus task-progress subset passed 12/12. Every changed Python file also compiled
under Python 3.10. No native C++ source or submodule pin changed, so no CMake or
CTest gate applies.

## Deliberate omissions and residual work

- No compute provider or compute model changed.
- No peer scheduler load was invented. TRAF-26 remains the explicit full-group
  workload and independent-routing task.
- The framework CPU oracle and preplay CPU-oracle half were not rerun because
  neither supplies additional affected traffic evidence. The live preplay
  replay half was rerun.
- This study does not calibrate NVLink, model expert compute, attribute
  per-request latency, or claim a full DP throughput result.
