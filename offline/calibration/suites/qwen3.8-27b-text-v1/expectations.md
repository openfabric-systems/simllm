# Qwen3.8-27B text extraction suite expectations

## Scope and identity source

This authored suite defines the text-only structure slice for
`Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. It includes prefill and decode
configuration surfaces. The multimodal vision encoder and the opt-in
one-layer multi-token prediction speculative head are excluded. Neither may
contribute a kernel family to this grid.

Model weights are never downloaded or read for this suite. The pinned Hugging
Face API revision response supplies the BF16 parameter count. Its recursive
tree response supplies each safetensors shard's Git Large File Storage object
digest and physical byte count. Local weight-byte verification is
intentionally not performed.

The suite sorts shard records by `name`, serializes the exact array of
`name`, `sha256` and `bytes` objects as canonical JSON with sorted object keys,
no insignificant whitespace and UTF-8 encoding, then hashes those bytes with
SHA-256. That manifest digest is the suite's `weight_sha256`. `weight_bytes`
is the exact sum of the 18 API-served physical shard sizes. This rule is
deterministic and does not claim that any local shard was present.

The allowed local substrate contains only configuration and tokenizer files.
The config file is checked locally against the frozen config digest. Each
framework must construct its own configuration object from that substrate and
must not construct an engine, model module or weight loader.

## Text-stack geometry

The outer `Qwen3_5ForConditionalGeneration` multimodal wrapper exposes a
`qwen3_5_text` stack with 64 dense MLP layers. Hidden size is 5120,
intermediate size is 17408, query-head count is 24, key/value-head count is 4,
head size is 256 and vocabulary size is 248320.

The ordered attention schedule repeats three `linear_attention` layers and
one `full_attention` layer 16 times. The result is exactly 48 linear-attention
layers and 16 full-attention layers. The linear layers implement Qwen3.5 Gated
DeltaNet, including a width-four short convolution, recurrent state, gated
normalization and state read, update and write behavior. The current five
inventory families cannot represent those operations or their bytes and
logical launches. A framework extraction that sees this exact schedule must
therefore reject the total inventory under COMP-62 before writing any
StepRecord stream or inventory object.

The framework-specific bindings are frozen independently. vLLM 0.26.0 maps
`Qwen3_5ForConditionalGeneration` to its `qwen3_5` implementation and selects
`QwenGatedDeltaNetAttention` for linear layers. The pinned SGLang tree exports
the same wrapper and selects `Qwen3_5GatedDeltaNet` backed by
`RadixLinearAttention`. Geometry must come from each framework's own text
configuration object. One framework's values and the raw Hugging Face JSON are
not substitutes for the other framework surface.

## Shape grid

The 15 cases preserve the Granite study's three five-cell families while
replacing the mixture-of-experts batch family with a dense batch sweep.

| Family | Varied axis | Frozen values | Fixed input |
|---|---|---|---|
| compute prefill | prompt tokens per request | 32, 128, 192, 256, 512 | four requests |
| memory decode | context tokens | 128, 512, 1024, 2048, 8192 | batch four |
| dense batch decode | batch | 1, 4, 8, 16, 64 | context 2048 |

All decode cells advance one token per request. The third family tests dense
batch scaling without inventing expert communication. Case order is the
`graph_cells` array order. A missing, duplicate or reordered case is fatal.

## Expected result

Both framework configuration surfaces must agree on the wrapper identity,
text geometry, layer order and Gated DeltaNet parameters. They must then
reject total extraction with COMP-62 named. The expected published denominator
is zero complete inventories out of two requested framework inventories. This
is a supported reject-and-register result, not a fractional score and not a
partial inventory.
