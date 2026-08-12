# Dependency authority TRAF-27 refreeze expectations

This expectations-only supplement freezes the TRAF-27 rerun after TRAF-25
gave every routed token one home rank. It was written before the dependency
study was executed against the corrected renderer. The original TRAF-12 and
cross-check expectation records remain historical chronology and are not
rewritten.

## Scope and input identity

The dependency sweep continues to use the tracked 22-token Granite
`length-cap` fixture with SHA-256
`36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341`.
That is the workload used by `nvlink_locality_v1` and by every historical
dependency-authority cell. Substituting the separate 54-token, three-request
full capture would mix a workload change with the ownership correction.

The full-capture provenance is nevertheless required through
`SIMLLM_MOE_E2E_ROOT`, never through a tracked absolute path. Before the run,
the harness will validate these observed inputs without using them as the
dependency sweep workload:

- `capture/granite-greedy.jsonl`, SHA-256
  `5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6`;
- `replay-400g/steps.jsonl`, SHA-256
  `824cd9557293328bb42b593ac893b6a067302e545b087c9219195ccb8031d755`;
- `replay-400g/routed-experts.json`, SHA-256
  `24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f`.

The evidence was authored against SimLLM revision
`14d8447b838e651f8321ffb0588ea02219e26e9a`. That revision and the revision
later observed by the run are separate provenance fields. The record makes no
equality assumption between an authored-against revision and a live submodule
pin.

## Ownership arithmetic and structural predictions

The tracked fixture has 48 routed phases. Under contiguous EP-width-four
ownership with home rank 0, every phase has exactly three positive star flows.
The corrected totals are 2,983,936 bytes at vector size 1,024 and 5,967,872
bytes at vector size 2,048, with 144 positive flows in either cell. Before the
correction, the totals were 11,870,208 and 23,740,416 bytes with 576 flows.

Ownership changes sparse pair payloads, not declared operation participation
or logical dependencies. The structural bands are therefore exact singleton
bands at both vector sizes:

| Quantity | Pre-correction | Corrected prediction band |
|---|---:|---:|
| Operations | 144 | [144, 144] |
| Effective edges | 423 | [423, 423] |
| Causal graph artifacts | 72 | [72, 72] |
| Required artifact boundaries | 47 | [47, 47] |
| Other serialized edges | 376 | [376, 376] |
| Backend GOAL artifacts | 48 | [48, 48] |
| Positive physical flows | 576 | [144, 144] |

These graph and conservation inventories are by-construction or prerequisite
guards. They are fatal-unscored and cannot enter the behavioral denominator.

## Physical sanity and JCT predictions

At vector size 1,024, corrected rank egress is 1,491,968 bytes on rank 0 and
486,400, 507,904 and 497,664 bytes on ranks 1 through 3. The 2,048-byte cell
doubles each value. At 400 Gbit/s, peak egress alone gives serialization
floors of 29,839,360 ps and 59,678,720 ps.

The graph-authoritative path is stricter. Its 48 serial phases each pay
2,000,000 ps propagation, and its one star bottleneck per phase serializes all
directed bytes. Including the fixed 24,000 ps compute term gives phase-chain
floors of 155,702,720 ps and 215,381,440 ps. Allowing at most one ps of integer
quantization per positive flow gives these preregistered JCT bands:

| Vector bytes | Direct-GOAL JCT prediction ps | Graph-authority JCT band ps | Graph minus direct band ps |
|---:|---:|---:|---:|
| 1,024 | 150,838,767 | [155,702,720, 155,702,864] | [+4,863,953, +4,864,097] |
| 2,048 | 205,653,487 | [215,381,440, 215,381,584] | [+9,727,953, +9,728,097] |

Conservative model ceilings serialize all bytes and charge propagation to all
144 flows separately: 347,702,720 ps and 407,381,440 ps. Every direct and
graph prediction lies between its applicable floor and ceiling. The corrected
graph JCT should fall only about 3.2 percent and 4.5 percent from the
pre-correction values. It must not scale down by the 3.978 total-byte ratio or
the 2.007 peak-egress ratio because fixed propagation and the phase star's
critical port remain on the serial path. Scaling with either wrong ratio is a
defect, not an acceptable refreeze.

The signed graph-minus-direct relation is evaluated from the raw graph
`StepResult` and direct completion before any exact timing, hash or comparator
guard. Its two payload instances retain the original TRAF-B1 genuine-risk
family. The missing-edge mutation retains TRAF-B3. The headline remains two
families and three instances. Prior corrected graph observations reduce the
novelty of the absolute graph values, but they do not entail the corrected
direct-versus-graph gap inside this consumer.

## Locality precision boundary

The corrected exact locality rows are expected to match the refrozen
`nvlink_locality_v1` registry. The single-node `AAAA` cells are baseline
observations only, pending CORE-41. The current analytic service charges
maximum source egress but omits destination ingress, so the corrected combine
star is undercharged. The current services are 4,538,000 ps and 9,047,000 ps;
an ingress-aware calculation predicts 6,652,000 ps and 13,286,000 ps. These
affected values are not accepted as precision evidence and do not close that
defect. The `AABB` cells have one local pair in each direction and the `ABCD`
cells are all remote, so the omission does not affect them.

## Dependency reconciliation prediction

`ExecutionGraph` remains the predicted sole ordering and completion authority.
The ATLAHS direct GOAL remains an opt-in independent cross-check whose result
cannot change `StepResult`.

The comparator will still audit all 423 effective edges. All 47 distributed
whole-operation FIFO mismatches remain. Participant-local syntactic mismatches
are predicted to fall from 188 to 47 because only the rank-0 star hub retains
multiple independent syntactic terminals at each phase boundary. The total
disagreement prediction is therefore 94, split exactly 47 participant-local
and 47 whole-operation FIFO. Of the 47 direct phase frontiers, 32 are predicted
unequal and negative. The first gaps are -81,920 ps and -163,840 ps; the
minimum gaps are -716,800 ps and -1,433,600 ps.

The direct artifacts are predicted to be 20,392 bytes with SHA-256 values
`917961edf996753223857d64010fc61e4f6b08672f18dcadf42c70d60ee36c4a`
and
`16ee686eda4634886b117788b3893c893f5e12ea819736e0afdbdf63bab0e826`.
These exact artifact and comparator checks are fatal-unscored. The raw signed
completion relation is evaluated first, so no fatal oracle pins a scored
instance before scoring.

If operation authority changes, if the opt-in cross-check changes the graph
result, or if either inventory is incomplete, the run is void. A numerical
cross-check disagreement is a finding and does not itself invalidate the
authority choice.

## Registered acceptance clauses

1. The tracked dependency workload and all three external provenance inputs
   match their frozen hashes, and no external absolute path is stored.
2. Corrected ownership yields 144 positive flows and the frozen byte rows,
   while graph operations, effective edges, artifacts, boundaries and
   serialized edges remain in their exact singleton bands.
3. Both graph JCTs and both raw graph-minus-direct relations land in the
   registered bands and within the physical floors and ceilings.
4. `ExecutionGraph` remains authoritative, selecting ATLAHS changes no graph
   result or authoritative artifact, and the comparator reports the registered
   423-edge inventory and 94 disagreements.
5. Exact cells, causal projection, negative-control acceptance, quiescence,
   identity and provenance guards all pass. Any failure makes the run void;
   fatal guards are not reported as fractions.
6. Single-node analytic values are explicitly pending CORE-41, all stale
   published current-value surfaces are corrected, historical consumers point
   to the corrected table, and every remaining contradiction is reported.

## Registered command and check-only dry run

The production command is:

```bash
.venv/bin/python examples/dependency_authority_v1/run_study.py \
  --source-root "$SIMLLM_MOE_E2E_ROOT" \
  --out "$SIMLLM_DEPENDENCY_AUTHORITY_RUN_ROOT"
```

Before this expectations-only commit, the same command is run with
`--check-only`. That path parses both paths and validates only frozen literal
shape and arithmetic. It imports no SimLLM implementation, reads neither
input path, invokes no native binary and creates no output artifact.
