# CORE-54 third scored DeepSeek-V3 deployment curve

## Held-out per-anchor score in all three layers

| Held-out anchor | Layer | Published tokens/s/node | simLLM point | Propagated band | Signed relative error | Verdict |
|---|---|---:|---:|---:|---:|---|
| SGLang EP32 prefill, 2K input | Physics only | 54,543.000000 | 57,332.324550 | [57,332.324492, 57,332.324550] | +5.113992% | REFUTED |
| SGLang EP32 prefill, 2K input | Physics plus boundary | 54,543.000000 | 57,332.324550 | [43,744.208139, 57,332.324550] | +5.113992% | REFUTED |
| SGLang EP32 prefill, 2K input | Physics plus boundary plus attenuation | 54,543.000000 | 52,077.816412 | [39,671.190380, 52,161.513856] | -4.519707% | PASS |
| SGLang EP32 prefill, 4K input | Physics only | 50,302.000000 | 57,332.324550 | [57,332.324492, 57,332.324550] | +13.976233% | REFUTED |
| SGLang EP32 prefill, 4K input | Physics plus boundary | 50,302.000000 | 57,332.324550 | [42,504.142847, 57,332.324550] | +13.976233% | REFUTED |
| SGLang EP32 prefill, 4K input | Physics plus boundary plus attenuation | 50,302.000000 | 52,077.816412 | [38,546.587414, 52,161.513856] | +3.530310% | PASS |
| SGLang simulated-MTP decode, batch 128 and KV 4000 | All layers | not read | not priced | none | not scored | BLOCKED on COMP-72 |

The frozen decision statistic is the maximum point absolute relative error in
the attenuated layer over the two priced held-out anchors. It is 4.519707
percent, below the 5 percent bar, so the literal verdict is
`SCORABLE_HELD_OUT_PASS_MTP_BLOCKED`. The unattenuated maximum is 13.976233
percent and is published alongside the scored layer. The claim is limited to
prefill under the declared benchmark-bias model. The MTP number was neither
read nor imputed, standard decode remains a registered calibration miss, and
CORE-54 does not close.

## Attenuation factors and derivations

| Candidate | Benchmark simplification corrected | Zero-anchor derivation and uncertainty | Anchors touched | Decision |
|---|---|---|---:|---|
| In-distribution expert balance versus uniform destination incidence | The disclosure uses in-distribution expert-balance data, while simLLM prices uniform top-k routing with same-destination rank deduplication. | Exact hypergeometric occupancy gives `E[unique destination ranks] / top_k = 939691952959 / 1034504281000 = 0.908349989669110`. The frozen interval `[0.906890124855826, 0.909809854482394]` is two standard errors from exact destination-indicator covariance over 16,384 routed tokens. | 3 prefill anchors | ADMITTED, one factor is fewer than the three anchors it touches |
| Exact-length packing versus per-request overhead | The benchmark and configured observations both pack exact lengths to 16,384 tokens per rank. | No residual correction remains. No independent per-request overhead magnitude is published or mechanically derivable without anchor input. | 0 | NOT ADMITTED |
| Decode depth attenuation | None. The decode miss is the registered 61-over-4 modeling residual, not benchmark bias. | Policy rule five forbids attenuating a registered modeling residual. | 0 | FORBIDDEN |

The admitted factor used zero anchor numeric input and was not fitted. It was
frozen before the scored run and multiplies the physics-plus-boundary prefill
throughput. No candidate factor was admitted by inspecting an anchor.

## Figure and result paths

- [Publication PDF](figures/deepseek-deployment-curve-run3.pdf)
- [Publication PNG](figures/deepseek-deployment-curve-run3.png)
- [Compact scored result](flagship_run3_result.json)
- External full scored result:
  `$SIMLLM_CORE54RUN3_RUN_ROOT/attempt-1/result.json`
- External frozen fit:
  `$SIMLLM_CORE54RUN3_RUN_ROOT/attempt-1/frozen-fit.json`
- External one-shot score:
  `$SIMLLM_CORE54RUN3_RUN_ROOT/attempt-1/held-out-score.json`
- External anchor-access ledger:
  `$SIMLLM_CORE54RUN3_RUN_ROOT/attempt-1/anchor-access-ledger.jsonl`

The figure has no watermark. Output throughput increases rightward and inverse
per-token request delay increases upward, so the upper-right corner is
optimal. Panel a shows the standard-decode curve with propagated record,
constant, boundary and attenuation uncertainties. Its second legend carries
the DeepSeek H800 production context and the declared PLACE-5 16-prefill plus
40-decode what-if. Panel b shows attenuated and unattenuated prefill
projections against the published anchors. The claim scope, attenuation
disclosure, MTP blocker and decode residual appear on the figure.

## Frozen fit and clean boundary

| Constant | Frozen envelope | Frozen value | Fit behavior |
|---|---:|---:|---|
| `intra_node_collective_surcharge_ps` | [0, 30,128,029] ps | 0 ps | Zero applications per step, inherited tie rule selected the lower bound |
| `overlap_exposed_fraction` | [0, 1/2] | 0 | Unconstrained solution was negative, so the clean physical floor was selected |

The fit used only `sglang_prefill_1k` and `sglang_decode_standard`. Its
objective minimized summed squared relative point error in the fully declared
attenuation layer, inside both inherited closed envelopes. The fit was
serialized with SHA-256
`78a798178234932325381aa7328ebd0dc816400e5a9caa3d6e5577edd0724883`
before either held-out row was accessed. The attenuation factor was not a fit
constant. No envelope was widened and no in-run adjustment occurred.

The boundary service is
`max(C, P) + f * min(C, P)`. Its `f = 0` endpoint is the clean COMP-75
perfect-overlap floor. Its `f = 1/2` endpoint is the clean TRAF-67 two-child
ceiling, whose 1K calibration residual is -23.423673 percent compared with
-0.592425 percent at the floor. The point fit selects the floor, while the
entire clean bracket propagates into the boundary and attenuated bands.

The existing topology-derived locality refinement remains frozen: an EP32
source rank has seven same-node destination peers and 24 fabric peers. COMP-75
prices the local bytes through its analytic NVLink endpoint serializer and the
remote bytes through the fabric arm. The A100 three-module NVLink candidate is
not substituted into this H100 target because that would add unsupported
cross-architecture physics.

## Decode calibration reality

The standard-decode observation selects candidate key
`05d1c33cdef9c12e25eb9159adc9dc80f1cd57b6333778f9efb5fb24cd6a74aa`
with one hit and no miss. The other three prefill candidate keys also select
with one hit and no miss. SGL-38's remote-KV projection is enabled explicitly
for the run and remains default-off generally.

The declared 61-over-4 depth extrapolation prices the 1,875,680,000 ps
four-layer basis as a 28.604120 ms step. It predicts 8,949.759685 tokens per
second per node against the 22,282 published calibration value, or 59.834128
percent low. Policy rule five leaves this registered modeling residual
unattenuated. COMP-76 owns its independent clean repetition and CORE-61 owns
depth-extrapolation validity.

## Allocation and conserved evidence

The disclosure configurations remain separate experiments:

- prefill: 4 nodes, 32 ranks and EP32 at 1K, 2K and 4K input;
- decode: 9 nodes, 72 ranks and EP72 for standard batch 256 and KV 2000;
- simultaneous rendering: 13 nodes and 104 ranks, comparator only and never
  called the 96-GPU system;
- joint prefill plus decode: second-legend what-if context only.

The live scheduler runs one eight-rank prefill engine and one eight-rank
decode engine. Admissions and terminals conserve at 16, 8, 4 and 32 requests
for 1K prefill, 2K prefill, 4K prefill and standard decode. The duplicate
standard-decode stable projection hashes to
`98a196d1a5ba2705218896afa4b2e7bf65cc89bc7f739f5a60a87b25597423f5`
in both repetitions under the unchanged CORE-58 field set.

The packet handoff carries 281,088,000 bytes as eight exact 35,136,000-byte
rank-paired messages. Both arms conserve bytes and endpoints and reach
quiescence:

| Link arm | Packet service | PCIe submission | Total handoff |
|---|---:|---:|---:|
| 400 Gbit/s point | 715,784,320 ps | 20,000,000 ps | 735,784,320 ps |
| 200 Gbit/s sensitivity | 1,429,568,640 ps | 20,000,000 ps | 1,449,568,640 ps |

The inherited offered-load sweeps are byte-identical to the second run and all
interarrivals are integer picoseconds. The analytic standard-decode and
PLACE-5 what-if curves are capacity projections in the shared curve schema,
not measured steady-state SGLang load surfaces. SGL-36 retains that physical
surface.

## Chronology, preservation and scope

The expectations-only freeze is commit
`45251494fa7c9dc0b872bf5324619380cf516a7b`. The scored runner commit is
`3d13cde15b19d105d91df4986751c03bebdb56b7`. Attempt 1 wrote the calibration
fit first and then accessed the 2K and 4K held-out anchors exactly once each.
The field-addressed chronology is 1K calibration, standard-decode calibration,
2K held-out and 4K held-out. No whole anchor record was loaded and the MTP
numeric access count is zero.

All 33 preservation locks pass. Every first-run, second-run, CORE-59, CORE-60,
COMP-75, TRAF-66 and TRAF-67 artifact in the lock class remains byte-identical.
The void CORE-60 record stays void. Only the clean COMP-75 and TRAF-67
reproductions are used as physical authorities.

No model weights were loaded or downloaded and no web page was fetched. The
run used Python 3.10.18, SGLang `0.5.19.dev345+gbfeae4e79` at
`bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3`, the frozen DeepSeek configuration
digest `cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9`
and the accepted htsim binaries.

The pass scope is only the two priced held-out prefill anchors under the
declared benchmark-bias model. It does not score MTP, turn decode calibration
into held-out evidence, validate candidate prices as silicon truth, qualify
the physical steady-state load surface, or promote the 104-rank comparator.
CORE-54 remains open on COMP-72, COMP-74, COMP-76, CORE-61, SGL-36 and TRAF-64.

## Artifact identities

| Artifact | SHA-256 |
|---|---|
| third expectations freeze | `9764f4c910c2ac7410c8ac447936b5ca48964096cf6240521c3f7888754fe637` |
| inherited second-run configuration | `3e5cca6693be05d9bd93870158ee24f7bee9092c2ce981c287fd94765d2d1970` |
| candidate record | `ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52` |
| clean composition record | `702b12259973d35072e20cd34c9fbb9e319fefd07cb1cafcb9f5e0856fbceecb` |
| clean boundary record | `37d56e5ac0793a5fe8071be03021ba3c4a66a36d90c1a984b4b531f4877c3cd1` |
| full scored result | `e9b7e5a2094a97aeb7e0b10b3d43e170d1c074b2e20804027701e43f2cd8bbfd` |
| frozen fit | `78a798178234932325381aa7328ebd0dc816400e5a9caa3d6e5577edd0724883` |
| held-out score | `877181ddc6441d07f38be00a79843efa603f3ccdd0f978744e2adbc28d4d5aa8` |
| anchor-access ledger | `94b1b6cf5d402510bce5430d6bd39649473a79428121a9ed7c22a8bfc2cad631` |
| compact published result | `255a73b120e2ad6e3a7b202475419d30174298590d6c9d3c22f9cfb6063489fe` |
| publication PDF | `a28fcc56399dd980e0a9758e720916186206360934431a635f944a3ae359a536` |
| publication PNG | `03bdfc7fab4cc71e4991ff8c3683336a5625f7390b8a2e6fe402efc0dd3ca811` |
| accepted `htsim_rnic` | `388415f92d6ef54c84bb5d2b7f7dabcaad27574ec235d62260f08175f3958bd9` |
| accepted `txt2bin` | `f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b` |
