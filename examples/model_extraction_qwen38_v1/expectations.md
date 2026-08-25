# Qwen3.8-27B offline extraction expectations

## Freeze scope and chronology

This expectations-only freeze follows commit
`59e27d0ed2c105a0aebc8bffd7ccf647cdf0527f` and precedes every Qwen-specific
extraction implementation and study run. Immediately before the first tracked
freeze file was authored, `git status --porcelain=v1` produced no rows. The
worktree was clean.

The suite bytes hash to
`560aab048f7c9db463f53614178faded06a7d3b62b7e775f6943e1b52fbfe6e2`.
This freeze contains authored inputs and expected relations only. It contains
no StepRecord stream, inventory object, run log, observed digest, result report
or outcome-dependent threshold.

## Weight-free extraction boundary

Model weights are never downloaded, mapped or read. The pinned Hugging Face
API provides the revision identity, BF16 parameter total, and every shard's
digest and physical byte count. The suite records all 18 shard rows, their
55,563,006,776-byte sum and the deterministic manifest digest. Local
weight-byte verification is intentionally not performed.

The prior plan's fatal guard for local shard bytes and hashes is dropped by
maintainer rule. It is replaced by fatal checks over the frozen API metadata,
canonical shard-manifest arithmetic, exact local config digest, exact revision
directory, and absence of local safetensors files. This redesign cannot detect
a future discrepancy between API metadata and bytes that are deliberately
absent. It does ensure the structure study never implies that those bytes were
verified.

Both structure paths stop at framework configuration objects. vLLM constructs
`ModelConfig` with tokenizer initialization skipped. SGLang constructs
`DeviceConfig(device="cpu")` and `ModelConfig` with multimodal execution
disabled. Neither path creates an engine, model module or weight loader. The
SGLang seam is intentionally its configuration surface, not the Granite
study's broader CPU-engine label, because the engine seam may require weights
and adds no structural evidence needed here. The evidence class remains
framework-native configuration projection: each framework parses and unwraps
the checkpoint with its own pinned configuration classes and verifies its own
architecture binding. Neither reads geometry from raw Hugging Face JSON or
from the other framework.

## Frozen framework projection

Both frameworks must project the exact outer architecture
`Qwen3_5ForConditionalGeneration`, wrapper type `qwen3_5`, and text type
`qwen3_5_text`. The text stack is dense with 64 layers, hidden size 5120,
intermediate size 17408, 24 query heads, 4 key/value heads, head size 256 and
vocabulary size 248320.

The framework-specific ordered layer arrays use different spellings for full
attention. Normalization maps only SGLang's `attention` to `full_attention`.
No other spelling is accepted. The normalized array must equal 16 repetitions
of:

`linear_attention, linear_attention, linear_attention, full_attention`.

The 48 linear layers name Qwen3.5 Gated DeltaNet in both pinned sources. Their
frozen geometry includes a width-four short convolution, key head dimension
128, value head dimension 128, 16 key heads, 48 value heads, a swish output
gate and float32 recurrent state. Those native sources include state reads,
updates and writes. The existing five inventory families do not price this
mechanism. Mapping it to ordinary attention would silently replace 48 of 64
layers and is forbidden.

## Parameter sweep

The 15 cases vary three independent parameters:

| Family | Varied parameter | Frozen values | Fixed input |
|---|---|---|---|
| compute prefill | prompt tokens per request | 32, 128, 192, 256, 512 | four requests |
| memory decode | context tokens | 128, 512, 1024, 2048, 8192 | batch four |
| dense batch decode | batch | 1, 4, 8, 16, 64 | context 2048 |

Every decode request adds exactly one token. The checker derives expected
StepRecord axes from the authored cell fields. A constant axis, swapped axis,
wrong product or case reorder can fail R3.

## Expected relations

R1, exact framework text projection: each framework must independently expose
the frozen wrapper identity, dense text geometry and linear-attention
parameters. Any default, inherited mixture-of-experts interpretation, wrapper
leak or geometry mismatch fails the relation.

R2, hybrid layer scaling: the pattern repeats exactly 16 times, producing 48
linear-attention layers, 16 full-attention layers and 64 total layers. The
existing full-attention formula `4L + 1` would produce 257 logical launches
and is explicitly invalid for this stack. No replacement launch denominator
is scored until COMP-62 defines the missing families.

R3, shape-axis sensitivity: prefill new-token count equals total authored
prompt tokens. Decode new-token and sampled counts equal batch, while context
is preserved per request. All three sweeps must change the declared axes
exactly.

R4, total rejection: after framework identity and structure checks, each
driver must name both COMP-62 and Qwen3.5 Gated DeltaNet, return the rejected
status, and write neither a StepRecord stream nor an inventory object. A
partial five-family result fails this invariant.

R5, rejection byte determinism: two same-framework runs must emit identical
canonical rejection records. The record contains status and structural
findings only. It is not a model inventory.

R6, cross-framework structural agreement: after the single allowed SGLang
full-attention spelling normalization, the two framework projections must be
byte-equivalent. This includes ordered layers and every geometry field.

R1 and R2 are partially producer-checker shared because configuration
inspection and extraction use the same framework projection helpers. Their
expected values and independent count checks are frozen here. R3 shares the
production case parser but derives expected values from authored cells. R4
checks subprocess status and filesystem absence outside the producer. R5 is
an independent byte comparison. R6 compares independent framework producers
through one normalized checker. This disclosure is part of the freeze and is
not upgraded after results are known.

Family projection and its conservation half, a replacement logical-launch
formula, execution-graph template equivalence classes, and inventory-object
byte determinism are unscored until COMP-62 makes total extraction possible.
Their absence does not count as a passing relation.

## Fatal guards and interpretation

Any suite-byte, API-metadata, local-config, framework-binding, geometry,
schedule or case-totality mismatch voids the run. A local safetensors file or
any StepRecord or inventory written after the required reject also voids it.
Fatal means void with findings, never a scored fraction.

If every fatal guard holds, the expected deciding number is zero complete
inventories from the two requested framework rows. That expected rejection
keeps COMP-54 open and registers COMP-62. It does not publish a partial
inventory, close a physical-capture task, or change Granite bytes.

## Physical sanity before interpretation

No duration is measured, so timing floors do not apply. Three independent
checks bound the structural result before any exact relation is read:

- Weight identity: 27,781,427,952 BF16 parameters require exactly
  55,562,855,904 payload bytes. The API's 55,563,006,776 physical shard bytes
  exceed that floor by 150,872 bytes of safetensors headers and metadata.
- Geometry: 64 normalized layers must decompose into 48 linear plus 16 full
  layers. Any other total is impossible under the frozen schedule.
- Framework plausibility: both pinned native implementations select a
  stateful Gated DeltaNet path for linear layers. A stateless ordinary
  attention inventory for all 64 layers contradicts both framework sources.

These checks can refute the study substrate. Passing them is not proof that a
future Gated DeltaNet family model is correct.
