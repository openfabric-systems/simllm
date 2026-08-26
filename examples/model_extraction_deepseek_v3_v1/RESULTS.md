# DeepSeek-V3 family inventory and deployment projection results

## Outcome

What ran: the frozen `model-extraction-deepseek-v3-v1` study drove the pinned
vLLM and SGLang CPU-only configuration surfaces twice each over 20 text cases,
then checked every logical family, integer work value, layer position,
canonical byte and disclosed expert-parallel rank class without reading model
weights or executing a GPU.

What came out: the run is nonvoid. It produced two complete 14-family
inventories and one framework-neutral four-unit deployment projection. The
vLLM inventory is
`2209f1bdb2055007d935d5e64e79e9cc89d36585415eb220dc90be9f333f53ff`,
the SGLang inventory is
`5f3f92884fd028532aef0eaa884218a865060780dac92dc02e910beb260967a3`,
and the deployment projection is
`ee154ed5f07c104269df9cf60d8730b8c6dced0ccf619fb7ff7146ec2ddfd5a2`.
Every ordinary case conserves at **666 logical family visits** and the one
enabled multi-token-prediction case conserves at **667**.

What it changes: COMP-67 closes. The DeepSeek-V3 coverage cell is now a
published text-only denominator for both pinned frameworks, and the SGLang
EP32 prefill, SGLang EP72 decode, DeepSeek production EP32 prefill and
DeepSeek production EP144 decode layouts are available by exact lookup.
COMP-69 and COMP-70 register the two compute residuals that the projection
keeps explicit. VLLM-38 and SGL-34 register the framework-owned physical
launch joins.

What it does not change: CORE-54, CORE-52, CORE-53, SGL-33 and TRAF-61 stay
open. No throughput point, time to first token, time per output token, kernel
duration, code object, observed GPU launch, redundant-slot traffic split,
multi-token acceptance schedule or model-weight byte was measured. DeepSeek's
production disclosure supplies no batch or context shape, so its two units
publish exact static rank shapes and no invented dynamic workload. Granite
and Qwen suites and inventories remain byte-identical.

## Chronology and identities

The expectations-only freeze began at commit
`724666a90ac643647c8f304ee052b92850be1656`. Commit
`e1e26634bd68c54b6073d2ec663da0a121d909e0` corrected the authored router
scale from a noncanonical decimal to the exact rational string `5/2` before
implementation. After implementation, but before any scored extraction,
commit `5dc2877292fe40a74a49c1e2270e6a39d08613db` corrected only the frozen
SGLang source symbol from the vLLM spelling to SGLang's actual
`DeepseekV2AttentionMLA`; commit `8871daa` then bound that source exactly.
The result record names `5dc2877` as the final freeze. This chronology is
reported as it occurred and was not reordered after observing a score.

The chosen checkpoint is `deepseek-ai/DeepSeek-V3` at revision
`e815299b0bcbac849fa540c768ef21845365c9eb`. The public deployment disclosure
names the original V3 system, and the pinned framework registries both resolve
that architecture natively. V3-0324 has the same frozen structural fields,
but it is not substituted for the disclosed checkpoint. DeepSeek-R1 is out of
scope.

The config is identified by SHA-256
`cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9`.
The Hugging Face application programming interface reports 163 weight shards,
688,586,727,753 payload bytes and 684,531,386,000 parameters. Their canonical
per-shard digest manifest has SHA-256
`ec8b878368c5fdb9f3288bd3a36a723a1637ec76464135a3f5b2e9aeff4072b4`.
The study used only that metadata and the config snapshot.

The vLLM source is release 0.27.1 at commit
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`. The SGLang source is
`0.5.19.dev345+gbfeae4e79` at commit
`bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3`, tree
`9ffe149f40e1cd5bff7dadc6806ad1927d312e69`. Each source independently
projects the exact 61-layer schedule, with 3 dense layers followed by 58
mixture-of-experts layers.

## Published artifacts

| Artifact | Cases or units | Bytes |
|---|---:|---:|
| [vLLM inventory](../../offline/calibration/model-inventories/2209f1bdb2055007d935d5e64e79e9cc89d36585415eb220dc90be9f333f53ff.json) | 20 cases | 77,397 |
| [SGLang inventory](../../offline/calibration/model-inventories/5f3f92884fd028532aef0eaa884218a865060780dac92dc02e910beb260967a3.json) | 20 cases | 77,448 |
| [Deployment projection](../../offline/calibration/deployment-projections/ee154ed5f07c104269df9cf60d8730b8c6dced0ccf619fb7ff7146ec2ddfd5a2.json) | 4 units | 27,611 |

Each filename is the SHA-256 of its exact canonical bytes. Both repeat
extractions reproduced their inventory bytes and the common StepRecord stream
with SHA-256
`fdabc79b3d7025ec06cd80d5fd482efb49b8ae7a9c214d912746fea80d1c973f`.
After framework provenance and the framework-owned physical join task are
removed, the two inventories agree exactly. The external summary record is
1,241 bytes with SHA-256
`daef040982bf277c89b4dda7fc093358cb3cdff947e70c58c6238a9b9b4f87dd`.
The suite SHA-256 is
`0a8297a7990c42ee6b2277c7507d90ee875ab07ba3c91324b5098d1a928dabea`.

## What the inventory counts

The attention path first compresses each query to width 1,536 and expands it
into separate 128-wide nonrotary and 64-wide rotary components for each of
128 heads. The key-value path compresses each token to a 512-wide latent plus
the 64-wide rotary component. Attention reads that 576-value BF16 compressed
cache vector instead of a full key and value vector for every head, then an
output projection returns the 128-wide values to model width 7,168.

That mechanism becomes eight multi-head latent attention families in each of
61 layers. The first 3 layers add one dense multilayer-perceptron family. The
remaining 58 layers add one router, one shared expert and the selected work of
8 routed experts out of 256. One language-model head completes every case.
The optional checkpoint multi-token-prediction block is one declared aggregate
family whose case flag is zero for ordinary cases and one for the disclosed
simulated-MTP arm. The visit derivation is therefore

`61 * 8 + 3 + 58 * 3 + 1 = 666`,

with one additional enabled aggregate family giving 667.

For a base case with `T` new tokens, `P` attention query-key pairs, `Z`
sampled tokens and `KVT` cached context tokens, the independently conserved
totals are

`F = 71,397,377,782*T + A*P + 1,853,358,080*Z`

FLOPs, where `A` is 4,997,120 for prefill and 16,990,208 for decode, and

`B = 671,295,240,544 + 70,272*KVT`

HBM bytes. Enabling the multi-token-prediction family adds
1,376,150,231 FLOPs per new token, 81,920 prefill or 278,528 decode FLOPs
per attention pair, 1,853,358,080 FLOPs per sampled token,
13,570,793,696 static HBM bytes and 1,152 HBM bytes per context token.
Every value is an integer.

## Per-rank conservation

The checkpoint owns 256 logical routed experts. The disclosed balancer adds
32 redundant physical slots, so physical residency has 288 slots while
logical work still counts every expert identity exactly once. One unique
expert occupies 2,554,954,752 static bytes across the 58 base
mixture-of-experts layers. The exact rank classes are:

| Expert parallelism | Rank classes `(rank count, unique, redundant, physical)` | Logical sum | Physical sum |
|---|---|---:|---:|
| EP32 | `(32, 8, 1, 9)` | 256 | 288 |
| EP72 | `(40, 4, 0, 4)`, `(32, 3, 1, 4)` | 256 | 288 |
| EP144 | `(112, 2, 0, 2)`, `(32, 1, 1, 2)` | 256 | 288 |

The logical base routed bytes conserve exactly as

`256 * 2,554,954,752 = 654,068,416,512`.

Physical residency is exactly 9/8 of that routed term because 288 slots are
resident. The redundant slots do not create logical expert visits. Which
physical copy serves a routed token remains undisclosed and is registered as
COMP-69 rather than invented.

The three disclosed SGLang prefill cases each use 16,384 new tokens per EP32
rank. Their global routed visits per mixture-of-experts layer are
`32 * 16,384 * 8 = 4,194,304`, or 16,384 visits per unique expert. Each rank
therefore owns `8 * 16,384 = 131,072` logical visits per layer, and all 32
ranks return the global total exactly.

Standard EP72 decode uses 32 new tokens per rank. It produces
`72 * 32 * 8 = 18,432` global visits per layer, or 72 per expert. Its two rank
classes own 288 and 216 visits per rank respectively; their weighted sum is
`40 * 288 + 32 * 216 = 18,432`.

The simulated-MTP EP72 case uses 16 new tokens per rank. It produces
`72 * 16 * 8 = 9,216` global base visits per layer, or 36 per expert. The two
rank classes own 144 and 108 visits per rank; their weighted sum is
`40 * 144 + 32 * 108 = 9,216`. Base FLOPs conserve at
162,656,153,457,408 on both the unsharded and rank-class sides. The enabled
MTP family conserves separately at 5,003,529,734,016 FLOPs on both sides.

## Evidence and physical sanity

No fatal guard was violated. Both configuration surfaces matched the frozen
geometry, source symbols, layer schedule, quantization and exclusions. Each
framework emitted 280 family rows, all 20 cases matched their exact oracle,
both repeat pairs were byte-identical, the framework-neutral structures
matched, and every rank class conserved logical work and physical residency.
The historical Granite and Qwen byte locks held.

The base inventory owns 671,295,240,544 static HBM bytes. Enabling the MTP
family raises that to 684,866,034,240 bytes, still below the
688,586,727,753-byte checkpoint payload. This is required but is not proof by
itself because embeddings, normalization tensors and checkpoint metadata are
outside the declared families.

The base physical static bytes per rank are 40,221,416,800 at EP32,
27,446,643,040 at EP72 and 22,336,733,536 at EP144. At 3.35 TB/s of HBM
bandwidth, one complete weight pass cannot beat 12.006, 8.193 or 6.668
milliseconds respectively. Real service must be slower because compute,
compressed-cache reads, communication and scheduling remain. The published
decode rates are below those ideal weight-read ceilings, so they pass this
coarse check. No timing constant is fitted from it.

The publication audit found and corrected one pre-result defect. The first
projection attempt labeled a rank field static while including the first
suite case's 128 compressed KV tokens, adding 8,994,816 bytes per rank. It was
not published. Commit `67f65c0` excluded the dynamic cache-read family from
that static field and added exact floor assertions. A fresh study produced
the deployment digest reported above.

## Reproduction

Machine-specific values belong in the gitignored local environment file. With
`VLLM_PYTHON`, `SGLANG_PYTHON`, `DEEPSEEK_CHECKPOINT_ROOT` and
`DEEPSEEK_RUN_ROOT` bound to the pinned CPU runtimes, exact config snapshot and
a fresh external output directory, run:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python \
  examples/model_extraction_deepseek_v3_v1/run_study.py \
  --vllm-python "$VLLM_PYTHON" \
  --sglang-python "$SGLANG_PYTHON" \
  --suite-root offline/calibration \
  --checkpoint-root "$DEEPSEEK_CHECKPOINT_ROOT" \
  --output-root "$DEEPSEEK_RUN_ROOT"
```

The runner requires a fresh output directory. It writes repeated StepRecord
streams, logs, inventories, the projection and the summary outside Git. It
performs no network request, GPU work or model-weight read.
