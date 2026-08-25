# Qwen3.8-27B Gated DeltaNet inventory expectations

## Freeze scope and chronology

This expectations-only freeze follows commit
`76d389f1fa3dde5b7935d5cc0b85401849fe3026` and precedes the COMP-62
implementation and every scored run. Immediately before the first tracked
freeze edit, `git status --porcelain=v1` produced no rows. The required sizing
note was present only in the ignored local notes layer.

The current-pin suite is
`qwen3.8-27b-text-v1-frameworks-2026-08-25`, with file SHA-256
`7be24843ffae71de65a1eab243eab9f592ce614097d701d5234eabd0c5980a9c`.
Its reference-model and 15-cell canonical object digests equal the historical
Qwen suite exactly. The historical suite and all five files in
`model_extraction_qwen38_v1` are byte locks, not inputs that this study edits.

This freeze contains authored inputs, formulas, exact oracles and expected
relations only. It contains no implementation, StepRecord stream, inventory,
run log, observed digest, result report or outcome-dependent threshold.

## Framework and text-stack boundary

No model weight is downloaded, mapped or read. Both framework paths stop at
their own current pinned configuration object. vLLM 0.27.1 constructs its
`ModelConfig` with tokenizer initialization skipped. SGLang at `bfeae4e7`
constructs `DeviceConfig(device="cpu")` and its own `ModelConfig` with
multimodal execution disabled. The exact local `config.json`, API-served
weight manifest identity and absence of local safetensors files remain fatal
guards.

Both paths must independently project the outer architecture
`Qwen3_5ForConditionalGeneration`, wrapper type `qwen3_5`, text type
`qwen3_5_text`, dense 64-layer geometry and the exact ordered repetition

`linear_attention, linear_attention, linear_attention, full_attention`

sixteen times. That is 48 linear-attention layers followed at each fourth
position by one of 16 full-attention layers. The text projection must exclude
the multimodal vision encoder and one-layer multi-token-prediction speculative
head.

The derivation is pinned to both native implementations. vLLM's
`model_executor/models/qwen3_5.py` selects
`QwenGatedDeltaNetAttention`, whose construction in
`model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` creates the QKVZ and
BA input projections, width-four depthwise causal convolution, float32
recurrent state, gated root-mean-square normalization and output projection.
SGLang's `python/sglang/srt/models/qwen3_5.py` constructs the same geometry as
`Qwen3_5GatedDeltaNet` behind `RadixLinearAttention`. Both sources construct a
dense gate, up and down MLP in every text layer.

## Integer accounting convention

All arithmetic uses Python integers from configuration fields. One
multiply-accumulate is two floating-point operations. One scalar add,
subtract, multiply, divide, square root, reciprocal square root, exponential,
sigmoid or softplus evaluation is one logical floating-point operation. A
reshape, view, index, metadata lookup or data-layout description is zero. The
inventory counts compulsory HBM traffic for weights and persistent state. It
does not invent activation traffic between logical families.

Symbols:

| Symbol | Meaning | Frozen value |
|---|---|---:|
| `H` | hidden width | 5,120 |
| `I` | dense MLP intermediate width | 17,408 |
| `Nq`, `Nkv`, `D` | full-attention query heads, KV heads and head width | 24, 4, 256 |
| `Nk`, `Nv` | linear key and value heads | 16, 48 |
| `Dk`, `Dv` | linear key and value head widths | 128, 128 |
| `W` | causal convolution width | 4 |
| `Lf`, `Ll`, `L` | full, linear and total layers | 16, 48, 64 |
| `bw`, `ba`, `bs` | BF16 weight, BF16 activation-state and float32 state bytes | 2, 2, 4 |
| `T`, `R`, `KVT`, `P`, `Z` | new tokens, sequences, context tokens, full-attention query-key pairs and sampled tokens in one case | case-derived |

Derived dimensions are `K = Nk * Dk = 2,048`, `V = Nv * Dv = 6,144`,
`C = 2K + V = 10,240`, recurrent-state elements
`S = Nv * Dv * Dk = 786,432`, and convolution-state elements per sequence
`X = C * (W - 1) = 30,720`.

## Frozen family forms

The Qwen inventory contains these 12 families in this order. Every formula is
an aggregate over the named layer population. Both phases use the same family
list and layer counts; only the case axes change.

| Family | Shape axes | FLOPs | HBM bytes | Logical launches |
|---|---|---:|---:|---:|
| `attn_gemm` | `new_tokens` | `2*T*Lf*Pfull` | `bw*Lf*Pfull` | 16 |
| `attn_score` | `new_tokens,kv_tokens` | `4*P*Lf*Nq*D` | 0 | 16 |
| `kv_read` | `kv_tokens` | 0 | `2*KVT*Lf*Nkv*D*ba` | 16 |
| `gdn_input_projection` | `new_tokens` | `2*T*Ll*Pin` | `bw*Ll*Pin` | 48 |
| `gdn_short_convolution` | `new_tokens,sequences` | `2*T*Ll*C*W` | `bw*Ll*C*W + 2*ba*R*Ll*X` | 48 |
| `gdn_state_read` | `sequences` | 0 | `bs*R*Ll*S` | 48 |
| `gdn_state_update` | `new_tokens` | `T*Ll*U` | `bs*Ll*2*Nv` | 48 |
| `gdn_state_write` | `sequences` | 0 | `bs*R*Ll*S` | 48 |
| `gdn_gated_norm` | `new_tokens` | `T*Ll*Nv*(7*Dv+2)` | `bw*Ll*Dv` | 48 |
| `gdn_output_projection` | `new_tokens` | `2*T*Ll*Pout` | `bw*Ll*Pout` | 48 |
| `mlp_gemm` | `new_tokens` | `2*T*L*Pmlp` | `bw*L*Pmlp` | 64 |
| `lm_head` | `sampled` | `2*Z*Phead` | `bw*Phead` | 1 |

The accepted full-attention family definitions remain unchanged. Their
per-layer parameter count is
`Pfull = H*(Nq*D + 2*Nkv*D) + Nq*D*H = 73,400,320`. Gated DeltaNet uses
`Pin = H*(2*K + 2*V + 2*Nv) = 84,377,600`: Q, K, V and Z occupy
`K, K, V, V`, while B and A each occupy `Nv`. The convolution has
`C*W = 40,960` weights. The output projection has
`Pout = V*H = 31,457,280`. Every layer's dense gate, up and down projections
have `Pmlp = 3*H*I = 267,386,880` parameters. The language-model head has
`Phead = H*248,320 = 1,271,398,400` parameters.

The recurrent update follows the two pinned sources' same algebra. Q and K
normalization costs `Nk*(7*Dk+2)` per token. For each value head, gate setup
counts seven scalar operations: A exponential, A and time-step combination,
softplus, sign, recurrent decay exponential and B sigmoid. The state then
decays, multiplies K to predict V, subtracts and gates the residual, adds its
outer product with K, and multiplies Q for the output. That costs
`7*Dv*Dk + 2*Dv`. Therefore

`U = Nk*(7*Dk+2) + Nv*(7*Dv*Dk + 2*Dv + 7) = 5,532,016`

FLOPs per token per linear layer. Gated RMS normalization squares and reduces
each value-head row, forms its reciprocal root mean square, applies its shared
learned weight, evaluates the swish gate and multiplies the gate into the
normalized output. That is exactly `7*Dv+2 = 898` operations per value head.

The short-convolution family owns both the width-four weights and one read and
write of the `W-1` persistent BF16 samples per sequence. The recurrent read
and write families own one float32 matrix movement per sequence. The update
family owns the two float32 vectors `A_log` and `dt_bias`. The normalization
family owns its shared `Dv` BF16 weights. No persistent state byte is counted
twice.

The fixed all-family coefficients are:

- 47,966,017,280 FLOPs per new token;
- 393,216 FLOPs per full-attention query-key pair;
- 2,542,796,800 FLOPs per sampled token;
- 50,241,239,040 static HBM bytes per step;
- 307,888,128 persistent-state HBM bytes per sequence;
- 65,536 KV-cache HBM bytes per context token.

Thus every case must satisfy exactly

`F = 47,966,017,280*T + 393,216*P + 2,542,796,800*Z`

and

`B = 50,241,239,040 + 307,888,128*R + 65,536*KVT`.

## Schedule and shape relations

R1, exact framework projection: both current framework surfaces independently
produce the frozen wrapper, text geometry, ordered layer schedule, Gated
DeltaNet dimensions and exclusions.

R2, exact family projection: every one of the 15 cases from each framework
contains the 12 ordered families, the exact shape vector, exact integer FLOPs,
exact integer HBM bytes and exact phase launch count above.

R3, family conservation: family FLOPs and HBM bytes sum to the independent
case formulas above for all 15 cases through both frameworks. Family launches
sum to `48*8 + 16*4 + 1 = 449` in every case. The 48 linear layers each own
seven Gated DeltaNet families plus the dense MLP. The 16 full layers each own
three full-attention families plus the dense MLP. The one head launch is not a
layer.

R4, ordered hybrid schedule: each framework preserves all 64 positions, with
linear layers at positions 0, 1 and 2 of each four-layer block and full
attention at position 3. Totals alone are insufficient.

R5, shape-axis sensitivity: the five prefill cases vary `T` while holding
`R=4`; the five memory-decode cases vary `KVT` while holding `R=T=Z=4`; the
five dense-decode cases vary `R=T=Z` while holding context per request at
2,048. Gated DeltaNet state bytes do not change with decode context at fixed
batch. Full-attention score work and KV bytes do.

R6, byte determinism: two same-framework extractions emit identical
StepRecord bytes and identical canonical inventory bytes. Each object filename
equals the SHA-256 of its bytes.

R7, cross-framework structural agreement: after removing framework identity
and framework-owned physical join tasks, the vLLM and SGLang inventory objects
are byte-equivalent.

R8, scope exclusion: neither inventory contains a multimodal encoder,
speculative head, physical code object or observed launch. Ordinary `simllm`
import loads neither framework runtime.

R1 and R4 are partially producer-checker shared because inspection and
extraction reuse each framework's projection helper. Frozen expected values,
an independent position walk and cross-framework comparison constrain that
sharing. R2 and R3 use an independent oracle in the study runner, not the
production family builder. R5 derives its axes from authored cells. R6 checks
process outputs by bytes. R7 compares independent framework processes. R8
checks inventory fields and module imports outside the producer.

## Fatal guards and interpretation

Any current-suite byte mismatch, historical-byte-lock mismatch, checkpoint or
API-manifest identity mismatch, local safetensors file, framework identity or
binding mismatch, text-stack or exclusion mismatch, missing or reordered
case, unknown family, noninteger work value, conservation mismatch,
StepRecord loss, noncanonical object or present physical identity voids the
run. Fatal means void with findings, never a fraction.

If no fatal guard is violated and every relation holds, the deciding outcome
is two published inventories, 15 conserved cases per framework and 449
logical launches per case. COMP-62 closes. The Qwen lookup column becomes
published, and COMP-54 remains open only for the Kimi K3 structure half. The
result enables that half to reuse the recurrent-state family discipline, but
does not publish a Kimi inventory or close COMP-54 or COMP-59.

## Physical sanity before interpretation

The logical model moves a `48*128*128` float32 recurrent matrix per sequence
per linear layer. One read plus one write is 6,291,456 bytes. At the declared
A100 HBM roof of 2.039 TB/s, no batch-one recurrent decode kernel can beat
3.086 microseconds and no batch-16 kernel can beat 49.369 microseconds before
fixed overhead. The retained read-only captures report roughly 7.7 to 57
microsecond recurrent decode medians across their batch sweep. They lie on the
physical side of those logical floors and are used only as a sanity bound,
never as fitted constants or scored observations.

The 48 linear layers' input and output projections plus all 64 dense MLPs
dominate per-token work, while recurrent state traffic scales with sequence
count and not context. Doubling only decode context must therefore move the
full-attention score and KV terms exactly while leaving every Gated DeltaNet
family unchanged. Doubling only batch doubles every dynamic term exactly and
leaves static weights unchanged.

The inventory's 50,241,239,040 static family bytes are below the checkpoint's
55,562,855,904 BF16 parameter payload. That is required because the text-only
family projection excludes the vision encoder and does not claim every small
non-family tensor. Being below that ceiling is not proof of correctness;
exceeding the whole checkpoint payload would refute the accounting.

The prefill capture contains a gated-delta inverse-merge specialization. It is
a physical realization of the recurrent update, not an extra logical family,
and no captured duration is transferred into this analytic record. Physical
implementation identity remains absent by design for COMP-6, VLLM-12 and
SGL-10 to join later.
