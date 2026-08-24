# Qwen3.8-27B offline extraction results

## Outcome

What ran: the frozen `model-extraction-qwen38-v1` study drove the pinned vLLM
and SGLang configuration-only seams, inspected each framework's independently
unwrapped text stack, and attempted total extraction twice per framework over
the 15-case text-only suite. No GPU, model engine or weight loader ran.

What came out: the run is nonvoid and the expected total rejection held. The
one deciding number is **zero complete inventories** from two requested
framework rows. Both frameworks found the same 48 Qwen3.5 Gated DeltaNet
linear-attention layers and 16 full-attention layers, rejected with status 2
under COMP-62 before writing a StepRecord or inventory, and reproduced their
framework-specific rejection bytes exactly.

What it changes: the Qwen3.8-27B coverage-column state is now a verified
blocked result rather than a plan. COMP-62 owns the missing Gated DeltaNet
inventory families. COMP-54 stays open because this nominated column has no
total inventory, but its pinned-framework support and exact structural blocker
are now established.

What it does not change: no inventory is published, no model weight is
downloaded or locally verified, and no GPU, physical capture, code object,
observed launch, Accel-Sim, timing, time to first token (TTFT), time per output
token (TPOT), multimodal encoder or speculative-head result is claimed. The
Granite suite, inventories, expectations, runner and result bytes remain
unchanged. No task closes.

## Chronology and frozen input

Immediately before the first tracked freeze edit, `git status --porcelain=v1`
produced no rows. The worktree was clean. The expectations-only freeze is
`f95d05a9bc0defa7171e371bcd2b2ad03db46954`, which precedes the requested
calibration-base merge, implementation commit
`dfa9d789f4d9862dbcdd2aaae8233e2d5a7b71e2`, and the first scored run. The
base merge did not alter the frozen suite or expectation bytes.

The suite SHA-256 is
`560aab048f7c9db463f53614178faded06a7d3b62b7e775f6943e1b52fbfe6e2`.
The study-expectations SHA-256 is
`40317759c47ada8b8215b97c1495c7b513e9b2593e76fec25b2e02b669073ff7`.
The canonical external result record has SHA-256
`b759767a5ce403c6c6d18e00952ee16eccb270be8d1efef9189f78dcf5b527d3`.
The complete external run directory is 60 KiB.

## Model and weight identity

The model is `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The local config SHA-256 is
`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`.
The minimal cache contained configuration and tokenizer files only and no
safetensors file before or after the run.

Weight identity comes from Hugging Face API metadata at the pinned revision,
not local checkpoint bytes. The revision response reports 27,781,427,952 BF16
parameters. The recursive tree response reports 18 safetensors shards totaling
55,563,006,776 physical bytes. The suite records every API-served shard digest
and size. Sorting those rows by name and hashing their canonical JSON produces
manifest SHA-256
`72b5a8b6db0ad258d743ddbf3de4efda86b1ee894f08564f31044d921c17074c`.

Local weight-byte and weight-hash verification is intentionally not performed.
The freeze dropped the former local-byte fatal guard under maintainer policy
and disclosed the loss of that evidence. The replacement guards check exact
API-metadata arithmetic, canonical manifest identity, local config identity,
revision identity and the absence of local safetensors.

## Framework configuration surfaces

vLLM 0.26.0 at source commit
`568afb3a13806beb53bb2e6bd518269357b237c0` used `ModelConfig` with tokenizer
initialization skipped. Its pinned registry maps
`Qwen3_5ForConditionalGeneration` to
`vllm.model_executor.models.qwen3_5.Qwen3_5ForConditionalGeneration`, and the
text implementation selects `QwenGatedDeltaNetAttention`.

SGLang `0.0.0.dev1+g8f2a3ad6d` at source commit
`8f2a3ad6d7d68c58ae65b61a75bb2115449addca` and tree
`5be26db1f559064c0f9e724e78c1a8f619754867` used
`DeviceConfig(device="cpu")` plus `ModelConfig` with multimodal execution
disabled. Its pinned `qwen3_5.py` exports the exact wrapper through
`EntryClass` and selects `Qwen3_5GatedDeltaNet` with
`RadixLinearAttention`.

The SGLang path deliberately stops at configuration rather than constructing a
CPU engine. An engine may insist on loading the 55.6 GB checkpoint, while the
configuration object provides the same structure evidence used by the Granite
driver: the framework's pinned config class, model registry, text-stack
unwrap, dtype and geometry. It does not read raw Hugging Face JSON directly
and does not borrow vLLM geometry. The inherited Qwen3.5 MoE defaults are
ignored only when the exact text model type is `qwen3_5_text`; unknown, routed
and conflicting MoE identities still reject.

## Evidence by class

### Run configuration

The runner used the two pinned framework Python environments, Hugging Face
offline mode, the exact minimal config snapshot, tensor, pipeline, data and
expert widths of one, and two extraction attempts per framework. The phase
scope was text-only prefill and decode. The shape grid varied prompt tokens per
request, decode context and dense decode batch across five authored values
each.

### Behavioral relations

- R1 passed for both frameworks. Each independently projected the exact
  multimodal wrapper, dense text model, 64-layer geometry, ordered attention
  schedule and Gated DeltaNet parameters.
- R2 passed exactly. Sixteen repetitions of three linear-attention layers and
  one full-attention layer produced 48 linear, 16 full and 64 total layers.
  The ordinary full-attention `4L + 1` launch formula was not scored.
- R3 passed exactly. The five prefill prompt values, five decode-context values
  and five dense-batch values all changed the production StepRecord axes in
  authored suite order.
- R6 passed exactly. After SGLang's sole `attention` to `full_attention`
  spelling normalization, the vLLM and SGLang text-stack projections were
  byte-equivalent.

All four behavioral relation families passed. Exact oracles, fatal guards,
structural invariants and native tests are separate evidence classes and are
not added to that denominator.

### Exact oracle

R5 passed. The two vLLM rejection records both have SHA-256
`7ce5b750672e9c90a1212079e8a55098d1f0c8126460e2ff99ce08893869cb6d`.
The two SGLang rejection records both have SHA-256
`f06893cb8df040f6adbe5d511f8cd9f99429861ef5e18e194c6398cd9d699993`.
Framework records differ because the framework identifier and rejection text
are part of the evidence.

### Structural invariant

R4 passed. All four extraction attempts returned status 2, named both COMP-62
and Qwen3.5 Gated DeltaNet, and left their StepRecord and inventory-object paths
absent. A partial five-family inventory would have failed this invariant.

### Fatal guards

No fatal guard was violated. Suite bytes, API metadata, local config identity,
weight-file absence, framework versions and source bindings, text geometry,
layer order, case totality, output absence and lazy ordinary imports all held.
Had any guard failed, the run would have been void and no behavioral relation
would be interpreted.

### Native tests

The final gates passed Ruff and the full repository suite with 2,774 tests
passed and 8 skipped. Focused tests cover metadata-manifest
mutation, forbidden local weight presence, all three shape axes, framework
projection mismatch, exact dense Qwen3.5 sentinel handling, and preserved
unknown or routed MoE rejection. Native tests are not added to the behavioral
denominator.

## Sanity bounds

The weight floor is exact for unquantized BF16: 27,781,427,952 parameters need
55,562,855,904 payload bytes. The 55,563,006,776 API-served physical shard
bytes sit 150,872 bytes above that floor, consistent with safetensors headers
and metadata. This is metadata arithmetic, not local byte verification.

The geometry floor and ceiling are both 64 layers under the exact frozen
schedule. The observed configuration projections sit at that bound and
conserve as 48 linear plus 16 full layers.

The independent mechanism check is the pinned native source in both
frameworks. Each selects stateful Gated DeltaNet with short convolution,
recurrent-state handling and gated output normalization for 48 layers. Pricing
all 64 layers as stateless full attention would contradict both sources. These
three checks can refute the substrate, but they do not validate a future
COMP-62 service model.

## Published artifacts and task effect

No file was added under `offline/calibration/model-inventories/`. Publishing a
record with the existing five families would violate COMP-54's totality rule.
The coverage table therefore links this study as a blocked column state and
does not link an inventory.

COMP-62 remains open for exact Gated DeltaNet family work. COMP-54 remains open
for the complete Qwen column and the other nominated columns. COMP-6, VLLM-12
and SGL-10 remain unchanged because no physical identity was captured. No
milestone or projected physical-calibration claim moves.

## Reproduction

Machine-specific paths belong in the gitignored local environment file. With
`VLLM_PYTHON`, `SGLANG_PYTHON`, `QWEN38_CHECKPOINT_ROOT`, `QWEN38_RUN_ROOT` and
`HF_HOME` bound to the pinned local runtimes, minimal exact-revision snapshot,
fresh external output directory and minimal cache, run:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python \
  examples/model_extraction_qwen38_v1/run_study.py \
  --vllm-python "$VLLM_PYTHON" \
  --sglang-python "$SGLANG_PYTHON" \
  --suite-root offline/calibration \
  --checkpoint-root "$QWEN38_CHECKPOINT_ROOT" \
  --output-root "$QWEN38_RUN_ROOT"
```

The runner keeps raw logs, repeated rejection records and its canonical result
record outside Git. A rerun requires a fresh output directory. It performs no
network request and rejects a local safetensors file.
