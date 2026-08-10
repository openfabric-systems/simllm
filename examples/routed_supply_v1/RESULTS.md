# Routed supply v1 results

Date: 2026-08-10

CORE-6, PLAY-4 and the captured-routing half of TRAF-2 pass their combined
study. The graph contract carries exact sparse all-to-allv sizes, the joined
pre-play trace projects executed input-token assignments without terminal
forwards, and the traffic path maps those assignments through explicit expert
placement epochs into both the graph and the live fluid step sink. All four
routed JCT cells match the frozen closed forms with 0 ps residual. The
uniform path retains its frozen bytes.

## Expectations and chronology

The three dependency-ordered expectations-only commits are:

1. CORE-6: `91bed6fd201e4fa1d810e3322905632bb54714c6`, before
   implementation commit `6b3e46f49c83067c8d30517efde860a3c9eb433e`.
2. PLAY-4: `a778cd9fea3e0ed1d2ec5250c148c068396aa497`, before
   implementation commit `d8784641df2ff02bbffa9950da47c9fdda6a5b56`.
3. TRAF-2: `365efe54241a894e83dd1065fb1a93502e6a7a9f`, before
   implementation commit `8ccb44806f90b8548755e0949fd7d1934191493f`.

Every registered check-only command ran before its corresponding freeze. The
commands printed confirmation by design and produced no artifacts. The freeze
commit bodies record the precise staged, unstaged and untracked state,
including whether an untracked harness existed.

The first result-producing combined run occurred only after all three freezes
and implementations. It passed every frozen numeric value. Review of its
summary found that the tracked PLAY source's terminal-token relation was
enforced inside PLAY-E1 but was not also listed under the scored PLAY-B1
block. Commit `b4d364405423ca4171b7bfb7d1c9e39426f8586e` moved that already
frozen assertion into PLAY-B1. It added no expectation and changed no measured
value. The final evidence below is the clean rerun after that correction.

## Reproduction and raw artifacts

**Post-specified portability note.** Integration review rendered the output
location through `SIMLLM_DATA_ROOT`. This presentation change does not alter
the frozen command options, archived artifacts or canonical hashes.

Set these local variables in the gitignored `.env.local.sh`: the external data
root, the full Granite decode trace and the two provided executables:

```text
SIMLLM_DATA_ROOT
SIMLLM_GRANITE_DECODE_TRACE
SIMLLM_HTSIM_RNIC
SIMLLM_TXT2BIN
```

Then run:

```bash
source .env.local.sh
.venv/bin/python examples/routed_supply_v1/run_study.py \
  --sections core,play,traffic \
  --out "${SIMLLM_DATA_ROOT:?configure SIMLLM_DATA_ROOT}/wave3-runs/codex/routing_supply_set" \
  --decode-trace "$SIMLLM_GRANITE_DECODE_TRACE"
```

Bulk GOAL, binary, completion-ledger, placement and projection artifacts stay
outside Git under `${SIMLLM_DATA_ROOT}/wave3-runs/codex/routing_supply_set`.
The final summaries have these canonical hashes:

| Summary | Bytes | SHA-256 |
|---|---:|---|
| `core_summary.json` | 1,229 | `39b140caf6b9b5a27b05cd7a6c0cea81c3b91ad7c6ec5c0e04bbe16ac1566e18` |
| `play_summary.json` | 2,816 | `ddfcf354d9127e7502d022534073688a5aeee9e4073eccc525c007e2012cca5d` |
| `traffic_summary.json` | 4,061 | `a82f3bff5f8a7e57df04ad55fd07977ee5c52131170ed9fc6cc45016efdecbf7` |

## Evidence classes

Evidence classes remain separate. Configuration-forced identities and
structural guards do not increase the behavioral denominator.

| Evidence class | Result | Accounting |
|---|---:|---|
| Combined study invocation | 1 | Unscored run record |
| Fluid backend configurations | 6 | Four routed plus two uniform, unscored configurations |
| CORE-B1 variable-pair instances | 2/2 pass | Scored component relation |
| PLAY-B1 projection sources | 2/2 pass | Scored tracked and full source relation |
| PLAY-B2 reordered join | 1/1 pass | Scored association relation |
| TRAF-B1 placement epochs | 2/2 pass | Scored graph and GOAL relation |
| TRAF-B2 fluid cells | 4/4 pass | Scored live JCT relation |
| CORE-E1, PLAY-E1 and TRAF-E1 | all pass | Fatal exact or identity oracles, unscored |
| Path portability scanner | 6/6 pass | Fatal repository-policy checks, unscored |
| Structural validation tests | all pass | Fatal and unscored |
| Repository pytest | 619 passed, 1 skipped | Separate executable evidence; the skipped communicator test requires Torch |

## CORE-B1: variable all-to-allv graph contract

Both sparse tables survived strict v1 JSON round trip, produced the same
ordered-pair GOAL sends and reached `CoarseDeviceRuntime` with the same service
bytes:

| Instance | Pair 0 to 1 bytes | Pair 1 to 0 bytes | Runtime completed bytes | Result |
|---:|---:|---:|---:|---|
| 1 | 2,048 | 4,096 | 6,144 | pass |
| 2 | 2,048 | 6,144 | 8,192 | pass |

Changing only pair 1 to 0 increased only that transfer by 2,048 bytes. The
uniform v1 fixture remained 559 bytes with SHA-256
`f4a5a70f5bd4a0c2fed874baa88f3035266a54f386a59927e115872c2bcff0a3`,
and its GOAL remained 166 bytes with SHA-256
`46ca1ea42952c5e0c66ea9eebb8947e770f7090f6cbdea6c711b4e764b412f5b`.
The writer omitted the optional table when empty.

## PLAY-B1 and PLAY-B2: captured routing projection

The tracked Granite source projected 22 prefill forwards, no decode forwards,
528 token-layer rows and 4,224 expert assignments. Its canonical 30,874 bytes
matched SHA-256
`e3af45f896ff0a7005c4da0d6b4d3cfba7a00c868653e9aea581f49c37392e7a`.

The full source projected three requests, 57 prefill forwards, six decode
forwards, 1,512 token-layer rows and 12,096 assignments. Each request had
exactly `output_token_count - 1` decode forwards:

| Request | Prompt forwards | Output tokens | Decode forwards | Result |
|---|---:|---:|---:|---|
| `eos-brief` | 15 | 3 | 2 | pass |
| `length-cap` | 22 | 1 | 0 | pass |
| `stop-string` | 20 | 5 | 4 | pass |

The trace-order full projection was 87,845 bytes with SHA-256
`7d1875ac46de07f7ed2ed814dc8596ecc500a74f51c626a9b98b2ecb38d949d5`.
Joining in reverse order produced exactly `stop-string`, `length-cap`,
`eos-brief`; each request payload stayed identical, and the resulting
87,845-byte projection matched SHA-256
`18a5f737d1680aac22df3ca4a095d2f4ef5205c2433379de86ed96afc77687c1`.

## TRAF-B1: pair distribution and placement epoch

For epoch 0, every layer's dispatch table was 45,056 bytes in each direction
except layer 2, whose rank-1-to-rank-0 pair was 43,008 bytes. Epoch 1 retained
those values except for the frozen layer-13 EPLB snapshot, whose pairs became
40,960 and 38,912 bytes. Every combine table was the exact dispatch transpose.

| Epoch | Dispatch 0 to 1 total bytes | Dispatch 1 to 0 total bytes | Sum of per-layer maxima |
|---:|---:|---:|---:|
| 0 | 1,081,344 | 1,079,296 | 1,081,344 |
| 1 | 1,077,248 | 1,073,152 | 1,077,248 |

Both lowered graphs carried 48 sparse collective operations with the selected
`placement_epoch`. At both link rates, the step sink emitted all 24 dispatch
and 24 combine tables exactly, with 96 positive flows, no pair mismatch and
verified quiescence. Destination deduplication therefore reached the same
live path that produced JCT rather than stopping at a component probe.

## TRAF-B2: exact live fluid JCT

All six fluid runs completed with verified quiescence. Routed outcomes named
the captured authority and selected epoch; uniform outcomes named the uniform
authority and no placement epoch.

| Epoch | Link rate | Routed JCT ps | Uniform JCT ps | Delta ps | Residual ps |
|---:|---:|---:|---:|---:|---:|
| 0 | 200 Gbit/s | 182,531,520 | 442,054,080 | -259,522,560 | 0 |
| 0 | 400 Gbit/s | 139,277,760 | 269,039,040 | -129,761,280 | 0 |
| 1 | 200 Gbit/s | 182,203,840 | 442,054,080 | -259,850,240 | 0 |
| 1 | 400 Gbit/s | 139,113,920 | 269,039,040 | -129,925,120 | 0 |

The required direction is negative in all four cells. Doubling bandwidth
reduced routed JCT by exactly 43,253,760 ps at epoch 0 and 43,089,920 ps at
epoch 1. Selecting the layer-13 EPLB snapshot reduced JCT by exactly 327,680
ps at 200G and 163,840 ps at 400G. These match the frozen serialization-only
changes with 0 ps residual.

## Identity-off and fatal guards

The absent-supply path emitted the frozen 13,200-byte GOAL with SHA-256
`d708e998685b617478e891b316728d14b8ac6185a62b73817f80af1c5adff518`
at both link rates. It retained scalar `payload_bytes = 180224`, an empty pair
table and 96 uniform flows. This is fatal identity evidence and is not scored.

Tests reject duplicate or inconsistent placement epochs, missing and multiply
owned experts, owner ranks outside the EP group, model and routing shape
disagreement, unknown or duplicate scheduled requests, invalid prefill and
decode slices, malformed sparse tables and changed trace authority. Dense,
single-rank and drain bypasses stay empty. Canonical order and the fixed
author-defined placement sequence are structural and unscored.

## Genuine-risk fraction

All five scored relation families, 5/5 or 100 percent, were genuinely at risk
in a competent implementation:

| Family | At-risk families | Scored families | Fraction | Plausible failure |
|---|---:|---:|---:|---|
| CORE-B1 | 1 | 1 | 100% | A reader could preserve the table while a renderer or runtime silently expanded the scalar. |
| PLAY-B1 | 1 | 1 | 100% | Decode attribution could include a terminal token or shift phase-local indices. |
| PLAY-B2 | 1 | 1 | 100% | Reordered joins could associate routing by row position instead of request identity. |
| TRAF-B1 | 1 | 1 | 100% | Expansion could count experts instead of destinations, use the wrong epoch or lose the table before GOAL. |
| TRAF-B2 | 1 | 1 | 100% | Correct-looking tables could fail to reach the live sink or produce a different bottleneck and JCT. |

The exact byte hashes, ownership completeness, flow counts, quiescence and
identity-off equality are necessary fatal checks, but they do not increase
this denominator. The two bandwidth copies of TRAF-B1 are repeated renderings
of the same two epoch relations and are not counted as extra families.

## Repository gates

The backend executable variables were set, so the live htsim tests ran rather
than self-skipping.

```text
$ .venv/bin/ruff check .
All checks passed!
```

```text
$ .venv/bin/pytest -q
........................................................................ [ 92%]
......................................s.....                             [100%]
SKIPPED [1] tests/test_vllm_communicator.py:226: torch is not installed
619 passed, 1 skipped
```

The portability scanner is part of that suite and also passes directly:

```text
$ .venv/bin/pytest -q tests/test_path_portability.py
......                                                                   [100%]
6 passed
```

No native C++ source changed, so no CMake or CTest gate applies.

## Deliberate omissions and residual work

No residual task was created from the allocated CORE-20, TRAF-11 through
TRAF-13, PLAY-10 through PLAY-12 or COMP-20 ranges. The requested routing
supply set is complete, so CORE-6, PLAY-4 and TRAF-2 close rather than leaving
partial entries.

Gate weights remain deliberately absent from the routing projection because
traffic uses destination identity and the later compute consumer counts
expert load. Routed compute imbalance remains owned by the pre-existing
COMP-7 registry entry. The independent framework-oracle comparison remains
PLAY-5, and MoE compute/communication splitting and overlap remain TRAF-9 and
TRAF-7. This study uses the exact fluid profile requested for the acceptance
relation; it makes no physical-topology or congestion-control calibration
claim.
