# CORE-54 fourth scored DeepSeek-V3 deployment curve

## MTP per-layer arithmetic and verdict

The disclosed MTP shape is batch 128 per node on eight GPUs, or batch 16 per
GPU. Simulated MTP emits one ordinary token plus one simulated speculative
token for each request. One node therefore emits

`128 requests * 2 tokens/request = 256 tokens/step`.

The measured four-layer service is 2,033,951,000 ps. The frozen linear depth
treatment gives

`2,033,951,000 ps * 61 / 4 = 31,017,752,750 ps`,

so the prediction in every layer is

`256 * 10^12 / 31,017,752,750`

`= 1,024,000,000,000 / 124,071,011`

`= 8,253.338082 tokens/s/node`.

| MTP layer | Published tokens/s/node | simLLM point | Propagated band | Signed relative error | Verdict |
|---|---:|---:|---:|---:|---|
| Physics only | 17,373.000000 | 8,253.338082 | [8,253.338082, 8,253.338082] | -52.493305% | REFUTED |
| Physics plus boundary | 17,373.000000 | 8,253.338082 | [8,253.338082, 8,253.338082] | -52.493305% | REFUTED |
| Physics plus boundary plus attenuation | 17,373.000000 | 8,253.338082 | [8,253.338082, 8,253.338082] | -52.493305% | REFUTED |

The published 5 percent interval is [16,504.35, 18,241.65] tokens/s/node.
Every frozen layer lies below it. The scored layer is therefore REFUTED, and
the combined literal verdict over all three scorable held-out anchors is
`ALL_SCORABLE_HELD_OUT_REFUTED`.

## Figure and record paths

- [Publication PDF](figures/deepseek-deployment-curve-run4.pdf)
- [Publication PNG](figures/deepseek-deployment-curve-run4.png)
- [Nine-row layer table](flagship_run4_score_table.csv)
- [Compact scored record](flagship_run4_result.json)
- [Content address](flagship_run4_result.sha256)
- External full result:
  `$SIMLLM_CORE54RUN4_RUN_ROOT/attempt-1/result.json`
- External frozen prediction:
  `$SIMLLM_CORE54RUN4_RUN_ROOT/attempt-1/frozen-prediction.json`
- External one-shot score:
  `$SIMLLM_CORE54RUN4_RUN_ROOT/attempt-1/held-out-score.json`
- External access ledger: `$SIMLLM_CORE54RUN4_RUN_ROOT/access.jsonl`

The flagship figure adds the MTP held-out panel, its three zero-width bands,
the absence of decode attenuation and the honest miss. Its prefill panel
carries the run-3 rows unchanged. Its deployment-curve panel retains the
standard-decode curve, published decode disclosure, H800 production context
and PLACE-5 16-prefill plus 40-decode what-if. A visual inspection at the
published PNG resolution found no clipping, label overlap or unreadable
legend.

The deployment-frontier figure remains byte-identical. Its frozen v2 contract
has a paired standard-decode marker and a y-only H800 anchor, but no MTP marker
slot. Adding one would violate that study's contract and the run-4 preservation
lock.

## Combined held-out scope

The two run-3 rows retain their exact authority digest
`255a73b120e2ad6e3a7b202475419d30174298590d6c9d3c22f9cfb6063489fe`.
The authority itself stayed byte-for-byte unchanged; its two rows were carried
without semantic changes and were not rescored.

| Held-out anchor | Scored layer | Published tokens/s/node | simLLM point | Signed relative error | Verdict |
|---|---|---:|---:|---:|---|
| SGLang EP32 prefill, 2K input | Physics plus boundary plus attenuation | 54,543.000000 | 52,077.816412 | -4.519707% | PASS, carried from run 3 |
| SGLang EP32 prefill, 4K input | Physics plus boundary plus attenuation | 50,302.000000 | 52,077.816412 | +3.530310% | PASS, carried from run 3 |
| SGLang EP72 simulated-MTP decode, batch 128 and KV 4,000 | Physics plus boundary plus attenuation, identical to all layers | 17,373.000000 | 8,253.338082 | -52.493305% | REFUTED, scored in run 4 |

The full layer table preserves the run-3 unattenuated findings as well: the 2K
row is 5.113992 percent high and the 4K row is 13.976233 percent high in both
the physics-only and physics-plus-boundary layers. No run-3 band, point,
factor, fit or verdict changed.

## MTP accounting, depth and attenuation

The speculative accounting is pinned to SGLang commit
`bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3`:
`eagle_worker_common.py` lines 584-650 and `eagle_worker_v2.py` lines 899-920.
The source makes `accept_lens` include the bonus token and subtracts one for
the count of correct drafts. The disclosed simulated configuration is the
one-plus-one case priced above. It is not a claim about production rejection,
replay or acceptance distributions. The disclosure also says MTP integration
with data-parallel attention is incomplete.

The 61-over-4 depth treatment is the same declared linear extrapolation used
by the standard-decode cell. CORE-61's depth-linearity question remains open.
The extrapolation is the dominant contributor to the low prediction and is
reported as a modeling limitation, not adjusted after the score.

No attenuation factor is admissible for this EP72 decode anchor. The run-3
factor corrects EP32 prefill routing incidence and does not transfer. EP72
destination incidence is derivable from architecture arithmetic without an
anchor, but the successor measures total step service and provides only a
disclosed component overlay. No independent coefficient maps destination
incidence to total elapsed service. Applying that ratio would promote a
disclosed attribution to measured latency. Policy rule five also forbids
attenuating incomplete integration or depth linearity. All three MTP layers
are consequently identical.

The two independent MTP observations partially unlock COMP-74, but no
validated distribution rule exists yet. The run therefore keeps the MTP bands
at their preregistered zero width rather than inventing an interval.

## Frozen fit, one-shot chronology and preservation

The run inherits the run-3 calibration-only fit at
`78a798178234932325381aa7328ebd0dc816400e5a9caa3d6e5577edd0724883`.
Its collective surcharge and overlap-exposed fraction both remain zero. There
was no refit, no MTP parameter, no held-out fit input and no envelope change.

Attempt 1 serialized the frozen prediction, copied the run-3 authority,
realized exactly 128 requests as 16 requests on each of eight GPUs, then read
the MTP held-out row once and scored it once. The scorer returned exit status
1 for the expected honest refutation. No second score was run. The access
ledger contains one measured-evidence projection, one run-3 projection, one
MTP held-out access and five later run-4 publication projections. The latter
include a manual audit, two pre-figure publication stops, a preliminary render
and the final render after the visual layout correction. None reread the
anchor. Every post-boundary record access was field-addressed and reported
`whole_record_loaded: false`.

The initial broad-search chronology breach disclosed by the sizing note and
freeze remains part of the record. It occurred before the run-specific reader,
did not compute a prediction, fit or score, and exposed no held-out value
beyond the task brief.

All 57 prior artifacts in the preservation class pass byte-for-byte. This
includes every first, second and third scored-run result and figure, the
partial Hopper successor evidence and the frozen deployment-frontier study.
No model weights were loaded or downloaded, no web page was fetched, and the
scored run used Python 3.10.18.

## Registry movement and remaining work

All three numeric held-out anchors have now been scored, so the former MTP
pricing blocker is removed. CORE-54 does not close because its literal 5
percent acceptance is refuted by the MTP row. The remaining work is:

- reproduce the standard-decode calibration basis independently under
  COMP-76;
- propagate COMP-74 distributions, now partially unlocked by the retained
  two-observation repeats;
- finish the registered Granite campaign arm through COMP-78;
- resolve the CORE-61 depth-linearity question.

SGL-36 and TRAF-64 retain their separate physical load-surface and
topology-qualified handoff scopes. Reserved identifiers CORE-63 and COMP-79
remain unallocated.

## Artifact identities

| Artifact | SHA-256 |
|---|---|
| fourth expectations freeze | `bf874fb6a1afa63f03766caa8c8043682ddcace23c6ce2b33df1a43d196e2f0c` |
| fourth configuration | `794a41e6564f50469e49ac37b16c8de9bfa9bf3b7932c038eafcc55ec70e1eb8` |
| frozen prediction | `56b37ac4b36eff16d5f2be527b7b1a234147d2dfd1c031dec67dc84a81b7d652` |
| one-shot held-out score | `da4458dfc097a7990805528a3ce824101c927398670c726f8428442d64a1f3bf` |
| run-3 carry-forward payload | `0badb89adc0c95f0f98104cab174042341f06cd80e61360a5920ef24be5dae97` |
| full external scored result | `89443a2e9b98828b2f9350b8411eb79ca78566d7e88650281105cd1e974fc26d` |
| final access ledger | `550b7c82ab3de8a95b4263734b79cda760acd0eb57433bc77a4b9c9856645719` |
| nine-row layer table | `7fe5c27d0a5350f79d39fb20bdfcaba13324a82ee50048c29a4216697dababa4` |
| compact content-addressed record | `e2fd0811638af02ea4389f456e0e796d9a2b24e550da3217dddd2ecc6872a6cd` |
| publication PDF | `6a5249b04826a3ce1176daa152014cf236a695dff542927799925aac421c4be2` |
| publication PNG | `6041fd4e94b1d7ae7ac2cae513892cd11ddbbdc089a89a393ecba5b81def4a94` |
