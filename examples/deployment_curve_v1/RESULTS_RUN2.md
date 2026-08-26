# CORE-54 second scored DeepSeek-V3 deployment curve

## Held-out per-anchor score

| Held-out anchor | Published tokens/s/node | simLLM point | Propagated band | Signed relative error | Verdict |
|---|---:|---:|---:|---:|---|
| SGLang EP32 prefill, 2K input | 54,543.000000 | 57,332.324550 | [57,332.324492, 57,332.324550] | +5.113992% | REFUTED |
| SGLang EP32 prefill, 4K input | 50,302.000000 | 57,332.324550 | [57,332.324492, 57,332.324550] | +13.976233% | REFUTED |
| SGLang simulated-MTP decode, batch 128 and KV 4000 | not read | not priced | none | not scored | BLOCKED on COMP-72 |

The frozen decision statistic is the maximum point absolute relative error over
the priced held-out anchors. It is 13.976233 percent, above the 5 percent bar,
so the literal verdict is `SCORABLE_HELD_OUT_REFUTED_MTP_BLOCKED`. CORE-54
remains open. The MTP value was neither read nor imputed because the candidate
record has no EP72 MTP batch-16 KV-4000 cell. COMP-72 remains the exact
dependency for that physical cell.

## Fitted constant

| Constant | Initial | Frozen envelope | Applications/step | Fitted value | New tunables from composition |
|---|---:|---:|---:|---:|---:|
| `intra_node_collective_surcharge_ps` | 15,064,014 ps | [0, 30,128,029] ps | 0 | 0 ps | 0 |

The inherited objective minimizes summed squared relative point error over
only `sglang_prefill_1k` and `sglang_decode_standard`. Because CORE-59 found no
successor-path application of the inherited surcharge, every value inside its
envelope gives the same predictions; the frozen tie rule selects the lower
bound. The fit was serialized and hashed as
`be96c1de5b6a9eff3b8529ee1947482453faeed30d08c4d3132624dfbef72fde`
before any held-out value was accessed.

| Calibration anchor | Published tokens/s/node | Prediction | Signed relative error |
|---|---:|---:|---:|
| SGLang EP32 prefill, 1K input | 57,674.000000 | 57,332.324550 | -0.592425% |
| SGLang EP72 standard decode | 22,282.000000 | 8,949.759685 | -59.834128% |

No envelope was widened and no in-run adjustment followed either calibration
comparison.

## Figure and result paths

- [Publication PDF](figures/deepseek-deployment-curve-run2.pdf)
- [Publication PNG](figures/deepseek-deployment-curve-run2.png)
- [Compact scored result](flagship_run2_result.json)
- External full scored result:
  `$SIMLLM_CORE54RUN2_RUN_ROOT/attempt-2/result.json`
- External frozen fit:
  `$SIMLLM_CORE54RUN2_RUN_ROOT/attempt-2/frozen-fit.json`
- External one-shot score:
  `$SIMLLM_CORE54RUN2_RUN_ROOT/attempt-2/held-out-score.json`

The figure uses output throughput increasing rightward and inverse per-token
request delay increasing upward, so the upper-right corner is optimal. Its
first panel shows the simulated standard-decode curve with propagated bands
and both published decode disclosures. Its second legend contains the
DeepSeek H800 production throughput context and the declared PLACE-5
16-prefill plus 40-decode what-if. Its second panel places the three published
prefill anchors on the disclosed prompt-length axis. The held-out refutation,
MTP blocker and standard-decode calibration miss are visible on the figure.
There is no watermark.

The analytic standard-decode curve covers 2,000 through 32,000 offered
requests per second and saturates at 80,547.837165 aggregate output tokens per
second. The declared 16P+40D what-if covers 10,000 through 160,000 offered
requests per second and saturates at 357,990.387399 aggregate output tokens per
second. These are analytic capacity projections in the shared curve schema,
not measured steady-state SGLang load surfaces. SGL-36 retains that physical
surface.

## Composed prefill mechanism and intervals

Prefill is priced through CORE-59's dispatch and combine mechanism using the
clean COMP-75 authority:

- FP8 dispatch wire bytes plus one float32 scale per 128 hidden elements;
- BF16 combine wire bytes;
- same-destination expert deduplication;
- max-like compute and communication overlap with the measured compute row as
  the overlap budget;
- 400 Gbit/s as the point arm and 200 Gbit/s as the named sensitivity arm;
- zero new free or fitted constants.

Per-rank token totals are constant across the disclosed 1K, 2K and 4K prompt
lengths. The communication term is therefore shared, and only the measured
compute rows differ. The point communication service is
2,286,179,760,360 ps, above each compute row, so max-like composition produces
the same 57,332.324550 tokens per second per node point at all three lengths.
That flattening is the dominant held-out contributor, and the longest prompt
has the largest residual.

Every curve band propagates three named contributions: the COMP-75 record
interval, the unchanged inherited constant envelope, and the COMP-74
distribution contribution. The last is the preregistered zero-width
placeholder because the retained rows lack enough independent repetitions.
It explicitly makes no stability or zero-variance claim.

## Decode calibration reality

SGL-38's remote-KV projection is default-off generally and enabled explicitly
for this run. The live standard-decode observation selects exact candidate key
`05d1c33cdef9c12e25eb9159adc9dc80f1cd57b6333778f9efb5fb24cd6a74aa`
with one hit and no miss. The other three exact candidate keys also select
live, completing CORE-56's bounded binding acceptance.

CORE-59 identified zero decode-side mechanisms. The declared 61-over-4 depth
extrapolation turns the measured 1,875,680,000 ps four-layer basis into a
28.604120 ms standard-decode step. The published throughput implies an
11.489094 ms step. The resulting throughput prediction is 59.834128 percent
low, exactly the signed direction frozen before the run. It is disclosed as a
calibration miss, not scored as a held-out anchor and not adjusted in-run.
COMP-76 owns an independent clean repetition of the four-layer basis, CORE-61
owns depth-extrapolation validity, TRAF-66 owns the finite-overlap residual,
and COMP-72 owns the missing measured cells including MTP.

## Allocation and conserved evidence

The disclosure configurations remain separate experiments:

- prefill: 4 nodes, 32 ranks and EP32 at 1K, 2K and 4K input;
- decode: 9 nodes, 72 ranks and EP72 for standard batch 256 and KV 2000;
- simultaneous rendering: 13 nodes and 104 ranks, comparator only and never
  called the 96-GPU system;
- joint prefill plus decode: second-legend what-if context only.

The live scheduler runs one eight-rank prefill engine and one eight-rank
decode engine. Per-node observables map to the separate disclosure roles with
the same conserved factors as the first scored run. Admissions and terminals
conserve at 16, 8, 4 and 32 requests for 1K prefill, 2K prefill, 4K prefill and
standard decode. The duplicate standard-decode stable projection hashes to
`98a196d1a5ba2705218896afa4b2e7bf65cc89bc7f739f5a60a87b25597423f5`
in both repetitions under the unchanged CORE-58 field set.

The packet handoff carries 281,088,000 bytes at prompt length 2000 as eight
exact 35,136,000-byte rank-paired messages. Point and sensitivity arms both
conserve bytes and endpoints and reach quiescence:

| Link arm | Packet service | PCIe submission | Total handoff |
|---|---:|---:|---:|
| 400 Gbit/s point | 715,784,320 ps | 20,000,000 ps | 735,784,320 ps |
| 200 Gbit/s sensitivity | 1,429,568,640 ps | 20,000,000 ps | 1,449,568,640 ps |

No model weights were loaded or downloaded and no web page was fetched. The
run used Python 3.10.18, SGLang `0.5.19.dev345+gbfeae4e79` at
`bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3`, the frozen DeepSeek configuration
digest `cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9`
and the accepted htsim binaries.

## Chronology, preservation and scope

The expectations-only freeze is commit
`cfe894814bfef15a3d3101b73d0203f4c05735bd`. Attempt 1 terminated during
runtime import before a score or void record was written; its frozen fit
remains preserved. Attempt 2 at run head
`8158c56fb4e4e54a6911082a50666339fbed41c5` wrote the fit first, completed the
live binding, identity and packet guards, and then accessed and scored the two
priced held-out anchors exactly once.

The preservation-lock check passes for all 24 frozen first-run and lineage
artifacts. The first scored result, both first-run figures, all CORE-59 and
CORE-60 records, and every COMP-75 record remain byte-identical. The void
CORE-60 record stays void; only the clean COMP-75 reproduction is used as
composition authority.

The verdict scope is the two priced held-out prefill anchors. It does not
score MTP, turn decode calibration into held-out evidence, validate candidate
prices as silicon truth, qualify the physical steady-state load surface, or
promote the 104-rank comparator. CORE-54 stays open because the scorable
prefill anchors miss and because MTP, decode reproduction and distributions
remain unresolved.

## Artifact identities

| Artifact | SHA-256 |
|---|---|
| second expectations freeze | `abd29ea7c18667a7b0068cba1ee2b6690452034b2d8b677c4988241f83d9c95c` |
| run configuration | `3e5cca6693be05d9bd93870158ee24f7bee9092c2ce981c287fd94765d2d1970` |
| candidate record | `ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52` |
| clean composition record | `702b12259973d35072e20cd34c9fbb9e319fefd07cb1cafcb9f5e0856fbceecb` |
| full scored result | `e6927a4450c36af3f151920074d84f73cf2a208495793d9cc70be5cb37a6cdd9` |
| frozen fit | `be96c1de5b6a9eff3b8529ee1947482453faeed30d08c4d3132624dfbef72fde` |
| held-out score | `52767e4e8c928f0956ad8e99ff8099e590643bd9a464015108fbb3200345d52e` |
| compact published result | `0e3db1a8d8ecc79d54618bbef7d2d2801862d1ec3188e3cc2a209f225a3919dd` |
| publication PDF | `3ec23203cbb5d154b635be8b5d05b230b41d777235028af931e3ca33bcca81ae` |
| publication PNG | `7b389006d91b6928fa16e24dca5f154432a3dbd887867ca2da1ee3224107ea9a` |
| accepted `htsim_rnic` | `388415f92d6ef54c84bb5d2b7f7dabcaad27574ec235d62260f08175f3958bd9` |
| accepted `txt2bin` | `f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b` |
