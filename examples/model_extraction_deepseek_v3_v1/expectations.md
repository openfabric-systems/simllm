# DeepSeek-V3 family inventory expectations

## Freeze scope and chronology

This expectations-only freeze follows commit
`dc350b6996215adf69384c23335b496440042fe7` and precedes the COMP-67
implementation and every scored run. Immediately before the first tracked
freeze edit, the tracked working tree was clean. The required sizing note was
present only in the ignored local notes layer.

The suite is `deepseek-v3-text-v1-frameworks-2026-08-25`, with file SHA-256
`0a8297a7990c42ee6b2277c7507d90ee875ab07ba3c91324b5098d1a928dabea`.
It contains 20 authored cases and four deployment projections. This freeze
contains only identities, formulas, exact oracles and expected relations. It
contains no implementation, inventory, run log, observed digest, measured
value or outcome-dependent threshold.

An expectations-only canonical-format amendment follows freeze commit
`724666a90ac643647c8f304ee052b92850be1656`. It spells the router scale as
the exact rational string `5/2`, because repository canonical JSON forbids
floating-point tokens. The value, formulas, cases and acceptance relations do
not change, and no scored extraction preceded the amendment.

A second expectations-only source-name amendment follows implementation
commit `15c7bb8fc64b4e34aef7af1a19f74b3a679d750e`. Direct inspection of the
pinned SGLang source identified its class as `DeepseekV2AttentionMLA`; vLLM
uses `DeepseekV2Attention`. The suite now records each framework's literal
class name. No formula, case, acceptance relation or observed inventory
changed, and no scored extraction preceded this amendment.

## Checkpoint and framework boundary

The selected checkpoint is `deepseek-ai/DeepSeek-V3` at revision
`e815299b0bcbac849fa540c768ef21845365c9eb`. This is the original V3 model
named by the DeepSeek production disclosure. V3-0324 at revision
`e9b33add76883f293d6bf61f6bd89b497e80e335` has the same architecture,
geometry, mixture-of-experts and quantization fields. Only its recorded
Transformers version differs, so the structural projection also covers the
V3-0324 class used by the SGLang reproduction. DeepSeek-R1 remains excluded.

No model weight is downloaded, mapped or read. The exact local `config.json`
and the API-served 163-shard manifest are fatal identity guards. vLLM 0.27.1
must construct its configuration with tokenizer initialization skipped.
SGLang at `bfeae4e7` must construct its CPU device and model configuration
with multimodal execution disabled.

Both paths independently project the same 61 base layers: layers 0 through 2
have dense feed-forward blocks, and layers 3 through 60 have DeepSeek
mixture-of-experts blocks. Every base layer has multi-head latent attention.
The optional multi-token-prediction path has one additional
mixture-of-experts decoder block, its two-hidden-state projection and its
language-model head.

The vLLM derivation is pinned to `deepseek_v2.py` and `deepseek_mtp.py` at
source commit `6e448d0e`. The SGLang derivation is pinned to
`deepseek_v2.py` and `deepseek_nextn.py` at source commit `bfeae4e7`. Each
source independently constructs query compression and decompression, shared
key-value compression and decompression, the rotary split, compressed latent
cache, shared and routed experts, router, dense prefix and optional
multi-token-prediction block.

## Integer accounting convention

One multiply-accumulate is two floating-point operations. One scalar
arithmetic or nonlinear operation is one logical floating-point operation.
Layout, metadata and top-k selection are zero. Each serialized FP8 matrix
uses one byte per element plus one four-byte inverse scale per 128 by 128
block. Router, correction bias, multi-token-prediction input projection and
language-model heads use their checkpoint dtypes. Cache state is BF16.
Activation traffic between logical families is not invented.

The frozen geometry is:

| Symbol | Meaning | Value |
|---|---|---:|
| `H`, `I`, `M` | hidden, dense intermediate and expert intermediate widths | 7,168, 18,432, 2,048 |
| `Nq` | query heads | 128 |
| `Qr`, `KVr` | query and key-value latent ranks | 1,536, 512 |
| `Dn`, `Dr`, `Dv` | non-rotary query-key, rotary and value widths | 128, 64, 128 |
| `L`, `Ld`, `Lm` | base, dense and mixture-of-experts layers | 61, 3, 58 |
| `E`, `K` | unique routed experts and selected experts | 256, 8 |
| `V` | vocabulary size | 129,280 |

For an FP8 matrix with `r` rows and `c` columns, stored bytes are
`r*c + 4*ceil(r/128)*ceil(c/128)`. This block formula applies separately to
the matrices both pinned frameworks construct, even when a runtime fuses
their execution.

## Frozen family forms

The ordered inventory contains 14 families. The first eight are multi-head
latent attention: query compression, query decompression, key-value
compression, key-value decompression, rotary split, attention, compressed
cache read and output projection. The next four are the three dense early
layers and the router, one shared expert and eight selected routed experts in
each of 58 later layers. The final two are the language-model head and the
optional multi-token-prediction head.

Per base layer, the fixed attention FLOPs are:

- query compression: `2*H*Qr = 22,020,096`;
- query decompression: `2*Qr*Nq*(Dn+Dr) = 75,497,472`;
- key-value compression: `2*H*(KVr+Dr) = 8,257,536`;
- key-value decompression: `2*KVr*Nq*(Dn+Dv) = 33,554,432`;
- rotary split: `3*Dr*(Nq+1) = 24,768`;
- output projection: `2*Nq*Dv*H = 234,881,024`.

Prefill attention materializes the logical head representation and costs
`2*Nq*((Dn+Dr)+Dv) = 81,920` FLOPs per query-key pair per layer. Decode uses
the absorbed latent representation and costs
`2*Nq*(2*KVr+Dr) = 278,528` FLOPs per pair per layer. Both read exactly one
compressed `KVr+Dr` BF16 cache vector per context token and layer.

Each dense early layer costs `2*3*H*I = 792,723,456` FLOPs per token. Each
mixture-of-experts layer costs 3,670,551 router FLOPs, 88,080,384 shared
expert FLOPs and `8*88,080,384 = 704,643,072` routed expert FLOPs per token.
The router scalar term is `2E + 3K - 1`: sigmoid, correction addition,
normalization and routed scaling. The language-model head costs
`2*H*V = 1,853,358,080` FLOPs per sampled token.

The base coefficients are therefore:

- 71,397,377,782 fixed FLOPs per new token;
- 4,997,120 prefill or 16,990,208 decode FLOPs per attention pair;
- 1,853,358,080 FLOPs per sampled token;
- 671,295,240,544 static HBM bytes;
- 70,272 compressed-cache HBM bytes per context token.

When enabled, the one multi-token-prediction family adds one
`2H`-to-`H` projection, one mixture-of-experts latent-attention layer and one
language-model head. It adds 1,376,150,231 fixed FLOPs per token,
81,920 prefill or 278,528 decode FLOPs per pair, 1,853,358,080 FLOPs per
sampled token, 13,570,793,696 static bytes and 1,152 cache bytes per context
token. When disabled, all five values and its logical visit are exactly zero.

For case axes `T` new tokens, `P` attention pairs, `Z` sampled tokens, `KVT`
context tokens and flag `m`, every case satisfies exactly:

`F = 71,397,377,782*T + phase_pair*P + 1,853,358,080*Z + m*(1,376,150,231*T + mtp_phase_pair*P + 1,853,358,080*Z)`

and

`B = 671,295,240,544 + 70,272*KVT + m*(13,570,793,696 + 1,152*KVT)`.

## Logical visits and exact relations

Without multi-token prediction, each case has
`61*8 + 3 + 58*3 + 1 = 666` logical family visits. The terms are eight
latent-attention families in 61 layers, the dense family in three layers,
three mixture-of-experts families in 58 layers and one language-model head.
The simulated multi-token-prediction case has one additional aggregate family
visit, for 667.

R1, both current framework configuration surfaces independently emit the
exact frozen geometry, layer schedule, quantization and exclusions.

R2, every case from each framework contains the 14 ordered families with the
exact phase shapes, integer FLOPs, integer HBM bytes and 666 or 667 visits.

R3, family values sum to the independent formulas for all 20 cases through
both framework surfaces. A disabled multi-token-prediction flag contributes
exactly zero work, bytes and visits.

R4, two same-framework extractions emit identical StepRecord bytes and
identical canonical inventory bytes. Each object filename equals the SHA-256
of its bytes.

R5, after removing framework identity and framework-owned physical join
tasks, the vLLM and SGLang structural inventory objects are byte-equivalent.

R6, the historical Granite and Qwen suite and inventory byte locks remain
unchanged.

R7, no inventory contains a weight byte, physical code object or observed
launch. Ordinary `simllm` import loads neither framework runtime.

## Deployment rank conservation

The 288-expert arrangement means 256 unique expert identities plus 32
redundant physical copies. Logical family totals count each unique identity
once. Physical residency counts every copy. The rank classes are exact:

| Unit | Rank classes `(rank count, unique, redundant, physical)` | Unique sum | Physical sum |
|---|---|---:|---:|
| EP32 | `(32, 8, 1, 9)` | 256 | 288 |
| EP72 | `(40, 4, 0, 4)`, `(32, 3, 1, 4)` | 256 | 288 |
| EP144 | `(112, 2, 0, 2)`, `(32, 1, 1, 2)` | 256 | 288 |

One unique routed expert occupies 2,554,954,752 static bytes across the 58
base mixture-of-experts layers. Multiplying this value by each rank's unique
count and summing rank classes equals 654,068,416,512 bytes, the unsharded
routed family exactly. Multiplying by physical counts gives a 9/8 residency
factor and does not alter logical work.

For the disclosed SGLang shapes, ideal balance is integer-exact. EP32 with
16,384 local tokens produces 4,194,304 global expert visits per layer, or
16,384 per unique expert. EP72 standard decode produces 18,432 global visits,
or 72 per expert. Its simulated multi-token-prediction arm produces 9,216,
or 36 per expert. Each rank's unique count times these per-expert visits sums
to the unsharded top-8 total. The redundant copies do not add visits; their
physical scheduling split is not disclosed and is not invented.

DeepSeek production disclosed EP32 prefill and EP144 decode units but no
batch or context shape. Those projections publish exact per-rank family
residency and authored-case shape rules. They make no throughput or physical
load-balancer split claim.

## Fatal guards and project effect

Any suite mismatch, historical byte-lock mismatch, checkpoint or API manifest
mismatch, local safetensors file, framework identity mismatch, geometry or
schedule mismatch, noninteger value, case loss, family loss, conservation
mismatch, rank-class mismatch, noncanonical object or present physical
identity voids the run. Fatal means void with findings, never a fraction.

If every guard and relation passes, the deciding result is two byte-stable
inventories with 20 conserved cases per framework, 666 visits in every
ordinary case, 667 in the simulated multi-token-prediction case, and four
exact sharded projections. COMP-67 closes and the DeepSeek lookup cell can be
published. Residual framework-native capture or calibration work, if found,
uses only COMP-69, COMP-70, VLLM-38 or SGL-34.

## Physical sanity before interpretation

The base inventory has 671,295,240,544 static bytes. Adding the optional
multi-token-prediction family gives 684,866,034,240 bytes. Both remain below
the 688,586,727,753-byte checkpoint payload because input embeddings,
normalization tensors and non-family metadata are not claimed. Exceeding the
checkpoint is a fatal defect; being below it is not proof of correctness.

After expert sharding and redundant residency, the base per-rank static byte
floors are 40,221,416,800 for EP32, 27,446,643,040 for EP72 and
22,336,733,536 for EP144. At 3.35 TB/s of HBM bandwidth, no one-pass weight
read can beat 12.006, 8.193 or 6.668 milliseconds respectively. Real service
must be slower because compute, cache reads, communication and scheduling
remain. The disclosed SGLang decode rates sit below the corresponding ideal
weight-read ceiling, so they pass this coarse bound. This bound cannot
validate the family arithmetic by itself.
