# CORE-54 scored DeepSeek-V3 deployment curve

## Held-out per-anchor score

| Held-out anchor | Published tokens/s/node | simLLM point | Propagated band | Absolute relative error | Verdict |
|---|---:|---:|---:|---:|---|
| SGLang EP32 prefill, 2K input | 54,543.000 | 92,284.945 | [92,058.421, 92,284.945] | 69.20% | REFUTED |
| SGLang EP32 prefill, 4K input | 50,302.000 | 82,169.930 | [81,990.294, 82,169.930] | 63.35% | REFUTED |
| SGLang simulated-MTP decode, batch 128 and KV 4000 | not read | not priced | none | not scored | BLOCKED on COMP-72 |

The frozen decision statistic is the maximum point absolute relative error over
priced held-out anchors. It is 69.20%, above the 5% acceptance bar, so the
literal verdict is `SCORABLE_HELD_OUT_REFUTED_MTP_BLOCKED`. CORE-54 remains
open. The MTP row was neither read nor imputed because candidate record
`ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52`
has no EP72 MTP batch-16 KV-4000 cell. COMP-72's resumable Merlin execution is
the exact dependency for that cell.

## Fitted constant and envelopes

| Constant | Initial | Frozen envelope | Applications/step | Fitted value | Disposition |
|---|---:|---:|---:|---:|---|
| `intra_node_collective_surcharge_ps` | 15,064,014 ps | [0, 30,128,029] ps | 116 | 0 ps | fitted parameter, not a measurement |

Tuning minimized the sum of squared relative point errors over only
`sglang_prefill_1k` and `sglang_decode_standard`. The fit artifact was written
and hashed as
`d26aa95857766dabd46ea80cf89b8c7ab480f301a3c205ca67e544bbde3751c1`
before the held-out scoring function was called.

| Calibration anchor | Published | Prediction at fitted value | Absolute relative error |
|---|---:|---:|---:|
| SGLang EP32 prefill, 1K input | 57,674.000 | 96,146.711 | 66.71% |
| SGLang EP72 standard decode | 22,282.000 | 8,949.760 | 59.83% |

The shared term reaches its physical floor. Increasing it would reduce the
already-high prefill projection while making the already-low decode projection
worse, so one shared additive collective term cannot reconcile the opposing
role errors. No envelope was widened after this result.

The propagated bands are deterministic component intervals, not confidence
intervals. They contain the candidate-record point service, the frozen
constant envelope and a zero-width distribution term marked as lacking a
stability claim because the candidate retains only one seed. The missing
repeat-derived distribution is registered separately; a zero width is not a
claim of zero physical variance.

## Publication figure and result

- [Publication PDF](figures/deepseek-deployment-curve.pdf)
- [Publication PNG](figures/deepseek-deployment-curve.png)
- [Compact scored result](flagship_result.json)
- External scored bulk:
  `$SIMLLM_CORE54_RUN_ROOT/attempt-5/result.json`
- External post-score binding qualification:
  `$SIMLLM_CORE54_RUN_ROOT/attempt-6-binding-qualification/binding-qualification.json`

The figure has the frozen output-throughput-rightward and inverse-delay-upward
orientation, so the upper-right corner is optimal. Panel a shows the
candidate-priced, role-scaled SGLang standard-decode projection with its band,
both published decode disclosures, and a second legend containing the
DeepSeek H800 production throughput context and the declared PLACE-5
16-prefill plus 40-decode what-if. Panel b keeps the published prefill anchors
on their honest input-length axis instead of placing input throughput on the
output-throughput axis. The figure has no dry-run watermark.

The output curves are analytic capacity projections emitted in the shared
`simllm-deployment-curve-v1` schema. They are not presented as measured
steady-state SGLang load curves: SGL-36 still owns that physical surface. The
standard curve covers 2,000 to 32,000 offered requests/s and saturates at
80,547.837 aggregate output tokens/s. The declared 16P+40D what-if covers
10,000 to 160,000 offered requests/s and saturates at 357,990.387 aggregate
output tokens/s.

## Allocation and conserved scale mapping

The disclosure's prefill and decode tables are separate experiments on the
12-node cluster:

- prefill: 4 nodes, 32 ranks, EP32, at 1K, 2K and 4K input;
- decode: 9 nodes, 72 ranks, EP72, at standard batch 256 KV 2000;
- the simultaneous structural render: 13 nodes and 104 ranks, retained only
  as a comparator and never called the 96-GPU system;
- joint prefill-plus-decode deployments: second-legend what-if context only.

The largest faithful live scheduler scale is one eight-rank prefill engine plus
one eight-rank decode engine. Per-node prefill observables map to the separate
four-node experiment with an exact factor of 4 and per-node decode observables
map to the separate nine-node experiment with an exact factor of 9. The full
EP32 and EP72 role subsets and the 448-rank 16P+40D context come from
`disaggregated_target_topology_v1`; the one-plus-one live scale is never
relabeled as a full engine deployment.

## Candidate binding qualification

The scored attempt surfaces candidate status and the exact record content
address in run provenance. Its stock SGLang scheduler divided the equality-edge
prefill token budget and the provider therefore delegated every scored-attempt
live lookup to the comparator. This did not change the already-frozen
record-derived point projections or the held-out arithmetic, but it prevented
that attempt from qualifying the exact live keys.

A post-score qualification raised only the scheduler prefill token ceiling and
performed no fit, read no anchor numeric value and performed no held-out score.
It selected the three exact EP32 entries through the child-owned compute
provider chain:

| Shape | Batch conservation | Selected key SHA-256 | Result |
|---|---:|---|---|
| 16 requests x 1K | 16,384 tokens | `ecbe7732eeaec8de700bf82b1db34c3a220aeeb81012944f0c16c9ab5096bc40` | exact hit |
| 8 requests x 2K | 16,384 tokens | `3a219ae5e79084e9d319eb704f5ccf63991f93a8e8e41b4ac027b0b0801df204` | exact hit |
| 4 requests x 4K | 16,384 tokens | `7275952ede07e81dd3d327735ab962157b9c628529117e6149b7edea10fe2404` | exact hit |
| 32 requests, remote KV 2000 | 32 terminals | none | comparator miss |

The standard-decode candidate key is
`05d1c33cdef9c12e25eb9159adc9dc80f1cd57b6333778f9efb5fb24cd6a74aa`.
The driver-level join currently submits a bootstrap token to the decode
scheduler without projecting the remote KV-2000 prefix into its
`KernelRequestShape`, so SGL-38 owns that exact live-key gap. CORE-56 remains
open because three of the four full-depth rows, not all four, were qualified.

## Session, identity and packet evidence

The scored live observations conserved admissions and terminals at 16, 8, 4
and 32 requests for 1K prefill, 2K prefill, 4K prefill and standard decode.
No weights were loaded. The run used Python 3.10.18, SGLang
`0.5.19.dev345+gbfeae4e79` at `bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3`,
and the frozen DeepSeek configuration digest
`cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9`.

The duplicate standard-decode guard compared only the preregistered stable
field projection. Both repetitions hashed to
`8ecb678a016fa0615e155b76d3f085acf267b60c40a3d6e284a254bbc389a4aa`.
Complete serialized request bytes, SGLang internal request IDs and process IDs
were not compared.

The DeepSeek MLA handoff carried 281,088,000 bytes at prompt length 2000,
split into eight exact 35,136,000-byte messages from ranks 0 through 7 to ranks
32 through 39. Both arms conserved bytes and endpoints and reached quiescence:

| Link arm | Packet service | PCIe submission | Total handoff |
|---|---:|---:|---:|
| 400 Gbit/s point | 715,784,320 ps | 20,000,000 ps | 735,784,320 ps |
| 200 Gbit/s sensitivity | 1,429,568,640 ps | 20,000,000 ps | 1,449,568,640 ps |

These rank sets are derived from the PLACE-5 role allocation. The bounded
packet driver still does not consume and qualify every physical path in the
PLACE-5 manifest, so this result does not close TRAF-64.

## Chronology and verdict scope

The expectations-only freeze is commit
`c390618327de6950c542ef22698a337bf821e012`. Attempts 1 through 4 ended before
held-out scoring on, respectively, provider serialization, unused tokenizer
initialization, strict generic shared-expert parsing and a packet-event
projection assumption. Their directories remain under the external run root.
Attempt 5 at run head `d8aab9d0fe937bd211e240491399ff34178a07e6`
wrote the fit first, completed the live and packet guards, then scored the two
priced held-out anchors once. Attempt 6 at
`9e7f55ff0d619c7b6690f882119c163335e84f87` was a no-fit, no-anchor-value,
no-score binding qualification.

The verdict covers only the two priced held-out prefill anchors. It does not
cover MTP, validate the candidate as silicon truth, qualify a physical
steady-state load surface, or turn the joint what-if into a disclosure
experiment. CORE-54 remains literal and open because the priced score is a
refutation and MTP is blocked.

## Residuals

- CORE-59 owns the role-specific mechanistic residual exposed by the opposing
  prefill and decode calibration errors.
- COMP-74 owns repeat-derived distribution and component uncertainty for the
  DeepSeek candidate rows; the current zero-width distribution term makes no
  stability claim.
- SGL-38 owns remote-KV projection into the decode request shape so the exact
  EP72 standard row can select live.
- COMP-72 remains the exact dependency for the absent MTP cell.
- SGL-36 remains the owner of a physical steady-state load-throughput and
  load-delay surface.
- TRAF-64 remains the owner of full PLACE-5 path qualification.

## Artifact identities

| Artifact | SHA-256 |
|---|---|
| anchor freeze | `b1a918ed02329a242d033943fb18b93fd9be8fdaa18093477e6abb8298540df5` |
| candidate record | `ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52` |
| scored full result | `3acbeeeee5efeb81416bbf472c4f35d95f2412031d152e5e95f2b8c9ad0344b9` |
| frozen fit | `d26aa95857766dabd46ea80cf89b8c7ab480f301a3c205ca67e544bbde3751c1` |
| held-out score | `d0fb2d472f2387cb3c32c3cc79216a0148852a5343ce7039fc3ba74a5adc01fb` |
| post-score binding qualification | `2ad58f4c680a4c9dbc632172f104d96cec7bf92de090af3cfa27d1465daf89ff` |
| compact published result | `ed06a2e8bce838f96c8365151a450ddf9a80ebde7520c3e37c40960b4a633e08` |
| publication PDF | `0105947ed8eb524c178f3ee7025c4e035aa93e2362d7842815947d50527e0063` |
| publication PNG | `b51ff4726ba23ac65ea23be34b9394654517b66a14360dfd91478e9767e42553` |
| accepted `htsim_rnic` | `388415f92d6ef54c84bb5d2b7f7dabcaad27574ec235d62260f08175f3958bd9` |
| accepted `txt2bin` | `f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b` |
