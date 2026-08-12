# Per-request replay fidelity v1 results

PLAY-11 and CORE-28 are complete for captured MoE traffic byte attribution.
The routed supply remains the routing authority. Its scheduled request identity
now projects into a loss-checked request partition of each aggregate physical
pair, survives execution-graph JSON and GOAL rendering, and is checked before a
backend run begins.

The genuine-risk result is 2/2 behavioral families and 5/5 instances. In every
negative control the aggregate-only comparison still passed, while the
per-request comparison detected the attribution error and the fail-closed gate
rejected it. Exact tables, hashes, conservation checks, native reachability and
structural validation are reported separately and do not increase that score.

The physical byte, hash and JCT values below are the historical pre-TRAF-25
source-multiplied observations. The corrected single-engine tables are in
[the token ownership results](../token_ownership_v1/RESULTS.md#per_request_fidelity_v1).

## Provenance and chronology

The expectations were frozen at commit
`eaa8b23860c7a5e357dc509fcf0897176a40df66`. The first literal-only check run
before that commit found an incorrect tuple index in the dry-run registry and
produced no artifacts. After correcting only that literal lookup, the complete
registered command with `--check-only` passed and again produced no artifacts.
The freeze preceded implementation and every result-producing run.

The implementation landed at `a152ff6`. The result chronology was retained
rather than rewritten:

1. `run-1` completed the synthetic and Granite byte checks, then the supplied
   46,360-byte GOAL converter exited with signal 11 on the frozen 744-byte
   synthetic GOAL. The GOAL already had the frozen SHA-256
   `1eb2bbff8a981523b5f6733420aa9d5d3509aa473ed991409b8d455e619e5864`.
   This run produced no native result and is not evidence.
2. A converter preflight using the htsim source-tree binary compiled that same
   GOAL and the paired backend reached `physical_quiescence=verified`.
   `run-2` then completed every byte check and all 12 native cells. Its final
   assertion exposed four unexpected failures of the frozen unscored JCT
   formula.
3. Commit `6facf48` preserved every frozen JCT value, kept native structural
   guards fatal, and changed only result reporting so an unexpected unscored
   deviation is recorded rather than converted into a pass. `run-3` produced
   the final summary.

The final `$SIMLLM_PER_REQUEST_RUN_ROOT/summary.json` has SHA-256
`4a01e90c797053a689d0672221273adf236a44f517bc312b706ab29d7c27e4e2`.
It records run head `6facf48dfddb2754a990d9f195d793a6e6fd45b6`, htsim binary SHA-256
`cfb5014a663791f7619fe33309114a74e82878de860c14fc8a723713501f027d`
and converter SHA-256
`f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b`.
The expectations were authored against htsim gitlink
`fc4400e4ca619223481536632074045cb6af2756`, and the final run observed the
same gitlink. This is reported as an observation only. The study sets
`gitlink_equality_required=false` and makes no pin-equality assumption.

The four external inputs retained their frozen identities:

| Input | SHA-256 |
|---|---|
| Granite capture | `5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6` |
| Step records | `824cd9557293328bb42b593ac893b6a067302e545b087c9219195ccb8031d755` |
| Routed-experts projection | `24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f` |
| Archived aggregate GOAL | `08a0403af66ff8a9d6b18f93afd15ae0bc925cc85555acf8a0593438a3d7bc92` |

These input identities are fatal configuration guards and are unscored.

## Configurations

The synthetic sweep used two MoE layers, four experts, top-k two, two EP
ranks, eight bytes per hidden vector, request-count prefixes of one, two and
three, and both frozen expert-placement epochs. The real cell used Granite
prefill step 0, all three scheduled requests, 24 layers, eight EP ranks and
expert owner `expert_id % 8`.

The native sanity sweep crossed both placements, all three request counts and
200 or 400 Gbit/s, for 12 cells. It used `rnic-nn-fluid` and the same frozen
physical GOALs as the byte study.

## Scored behavioral relations

### PLAY-B2: synthetic aggregate-preserving permutation

After each valid trace was rendered, the control swapped `alpha` and `beta`
only in the structured request partition. It did not change a physical size,
peer, tag, dependency or GOAL byte. The raw comparison ran before exact-table
and hash checks.

| Placement | Requests | Aggregate mismatches | Request mismatches | Request L1 bytes | Signed `alpha` bytes | Aggregate-only | Gate |
|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 2 | 0 | 12 | 96 | -16 | passed | rejected |
| 0 | 3 | 0 | 12 | 96 | -16 | passed | rejected |
| 1 | 2 | 0 | 4 | 32 | +16 | passed | rejected |
| 1 | 3 | 0 | 4 | 32 | +16 | passed | rejected |

PLAY-B2 passed 4/4 instances. It is genuine-risk evidence because an
implementation that retained only aggregate pairs, dropped ownership, or
associated by request position would reach these observations and accept the
permutation.

### PLAY-B3: Granite aggregate-preserving permutation

Swapping `r0` and `r1` left all 2,688 physical sends and 207,499,264 bytes
unchanged. The aggregate comparison reported zero mismatched rows. The
per-request comparison reported exactly 5,348 mismatched rows and 76,496,896
bytes of L1 attribution error, with signed errors -38,248,448 bytes for `r0`,
+38,248,448 bytes for `r1` and zero for `r2`. The gate rejected the trace.

PLAY-B3 passed 1/1 instance. Together the retained genuine-risk denominator is
2/2 families and 5/5 instances.

## Entailment analysis

The freeze expected PLAY-B1 and CORE-B1 to be scored from positive direct and
graph-rendered observations. Execution showed that those observations are not
reachable without first passing the same identity relation:

- `render_step_goal` compares its routed operations with the structured
  messages and calls `require_match()` before returning.
- `render_serial_execution_graph_goal` compares the graph partition with its
  structured messages and calls `require_match()` before returning.

Consequently the six PLAY-B1 and six CORE-B1 positive instances are withdrawn
from the behavioral denominator and classified as fatal-unscored entailments.
They still provide change-set and exact-oracle evidence, but they are not
genuine-risk passes. The negative controls remain scored because they mutate a
valid returned trace afterward; no earlier positive gate entails whether the
permuted ownership will be detected.

## Fatal exact and structural evidence

All six synthetic configurations matched the independent frozen per-request
tables. Direct and strict graph-round-trip renderers produced identical request
rows and identical physical GOAL text. Their exact GOAL byte counts, hashes,
send counts and send bytes matched all six frozen rows.

The Granite canonical request rows matched exactly:

| Request | Positive rows | Total bytes | Canonical bytes | SHA-256 |
|---|---:|---:|---:|---|
| `r0` | 2,688 | 84,439,040 | 80,824 | `d2d5564c0507ae8e9946e377dfd9df0fca3eab20910d150faba03b1576e5e75a` |
| `r1` | 2,688 | 46,190,592 | 80,516 | `5f7603ec085e76e86b022b688404c428c90344115ac675ef40b59e609b90f568` |
| `r2` | 2,688 | 76,869,632 | 80,810 | `c441be8e81936ef0d32d32d59dfaf20f08bf496d588836edfee84058dbe0c89f` |
| all | 8,064 | 207,499,264 | 242,146 | `bcb21232c6f433e64ca0efb9bbfdaab4c008b087249f5d4b849dfb9bc646c077` |

The newly rendered Granite GOAL also retained the archived 334,432 bytes,
2,688 sends, 207,499,264 send bytes and SHA-256
`08a0403af66ff8a9d6b18f93afd15ae0bc925cc85555acf8a0593438a3d7bc92`.
This proves that attribution is a read-only partition and does not split or
resize physical messages.

Unit evidence separately rejects blank and unknown request identities,
duplicate or noncanonical request pairs, nonpositive sizes, self pairs, ranks
outside the collective, an unsupported collective kind, a per-pair sum that
differs from the aggregate table, malformed message partitions, and missing or
extra attributed messages. Empty attribution remains absent from the JSON wire
form, and existing physical hashes remain unchanged. These are fatal guards,
not scored instances.

## Native whole-step sanity and unexpected timing relation

All 12 native runs completed with captured routing, the selected placement,
the frozen physical GOAL hash and verified physical quiescence. Eight matched
the frozen JCT value exactly. Four were lower than the frozen formula:

| Placement | Requests | Gbit/s | Frozen JCT, ps | Observed JCT, ps | Residual, ps |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 200 | 8,003,280 | 8,003,280 | 0 |
| 0 | 1 | 400 | 8,002,640 | 8,002,640 | 0 |
| 0 | 2 | 200 | 8,003,920 | 8,003,920 | 0 |
| 0 | 2 | 400 | 8,002,960 | 8,002,640 | -320 |
| 0 | 3 | 200 | 8,004,560 | 8,004,560 | 0 |
| 0 | 3 | 400 | 8,003,280 | 8,003,280 | 0 |
| 1 | 1 | 200 | 8,003,280 | 8,003,280 | 0 |
| 1 | 1 | 400 | 8,002,640 | 8,002,640 | 0 |
| 1 | 2 | 200 | 8,004,560 | 8,004,560 | 0 |
| 1 | 2 | 400 | 8,003,280 | 8,002,960 | -320 |
| 1 | 3 | 200 | 8,005,840 | 8,005,200 | -640 |
| 1 | 3 | 400 | 8,003,920 | 8,003,600 | -320 |

The frozen formula summed the larger directional payload as if every
pairwise phase ended at a global barrier. The physical GOAL instead advances
each rank from its participant-local completion frontier. With asymmetric
directions, one rank may enter the adjacent combine or next-layer phase while
the other direction is still completing. Summing the four global phase maxima
therefore overestimates cells where the critical direction changes. Exact
physical GOAL identity rules out the attribution change as the cause.

This relation is reported as an unexpected post-freeze deviation, not a pass
and not a revised expectation. It is also unscored: whole-step JCT proves that
the checked renderer reaches the live metric chain, but it provides no rule
for dividing latency among co-scheduled requests.

## Closure scope

PLAY-11 registered this acceptance:

> Preserve each scheduled request's captured MoE routing identity through the
> traffic expansion and rendered aggregate GOAL messages; fail closed by
> request, layer, phase and directed pair; sweep one, two and three requests
> across two placements; match exact per-request byte tables; reject a
> two-request attribution permutation while aggregate pairs and physical GOAL
> stay byte-identical.

The routed expansion and message partition demonstrate the first clause. The
four synthetic and one Granite controls demonstrate fail-closed discrimination.
The six synthetic cells demonstrate the request-count and placement sweep. The
synthetic and Granite canonical rows demonstrate the exact tables. Frozen GOAL
hashes and zero aggregate mismatches demonstrate physical identity. Every
PLAY-11 clause is covered.

CORE-28 registered this acceptance:

> Add an optional, loss-checked per-request partition whose sums exactly retain
> the aggregate physical demand; preserve it through strict graph JSON and
> graph-only GOAL rendering; reject unknown, duplicate and
> aggregate-inconsistent entries; keep legacy graph bytes and physical GOAL
> operations byte-identical when attribution is absent or added.

The `CollectiveWork` validator and aggregate-from-partition construction cover
the loss check. Six strict JSON round trips and graph renders cover preservation.
The malformed-input tests cover rejection. Optional-field omission, existing
wire fixtures and all physical GOAL hashes cover both identity paths. Every
CORE-28 clause is covered.

No residual task is needed. The work does not implement or claim per-request
latency attribution, KV-cache behavior, expert compute fidelity, gate weights,
TP collective attribution or packet-level calibration. KV-cache behavior
remains owned by the parallel framework-oracle work, and this change neither
depends on that branch nor edits the preplay runner.
