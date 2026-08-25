# Qwen3.8-27B Gated DeltaNet inventory results

## Outcome

What ran: the frozen `model-extraction-qwen38-v2` study drove the current
pinned vLLM and SGLang CPU-only configuration seams twice each over all 15
Qwen3.8-27B text cases, then checked every family shape, integer work value,
logical visit, schedule position and canonical byte.

What came out: the run is nonvoid and produced **two complete inventories**.
The vLLM record is
`77ea0abb4803d2cab5689f6893563e9a973f0e29160ec1975f1c76b3046e30d1`;
the SGLang record is
`9d1c6164d149b98a4d019bee31d8d3a7ce3ee38cb96b7e0bab03393fde3d4747`.
Every one of the 15 cases per framework conserved exactly, and every case has
the deciding structural count of **449 logical family visits**.

What it changes: COMP-62 closes. The Qwen3.8-27B coverage cell is a published
text-only denominator for both current framework pins rather than a verified
rejection. This completes the Qwen part of COMP-54 and leaves that task open
for the Kimi K3 structure half, which can reuse the same recurrent-state
family discipline.

What it does not change: COMP-54 and COMP-59 stay open. No Kimi inventory,
GPU execution, model-weight read, physical code object, observed launch,
calibrated duration, multimodal encoder or speculative-head denominator is
claimed. The historical rejection study remains unchanged as the record of
why COMP-62 was registered. The Granite suite and all four previously
published Granite inventory bytes remain unchanged.

## Chronology and run record

The expectations-only freeze is
`bf14d7563bdb52a0c8052309f477a022f1951cc4`. It follows base commit
`76d389f1fa3dde5b7935d5cc0b85401849fe3026` and precedes implementation
commit `9470230`. The freeze contains the current framework identities, exact
integer family forms, 15 case oracles, fatal guards and expected directions,
but no implementation or result.

The first post-implementation attempt completed all four framework
extractions, then its independent scorer stopped before producing a verdict
because it addressed `ShapeAxis.name` rather than the canonical
`ShapeAxis.axis_id`. No relation result or `results.json` existed for that
attempt. Commit `d335740` fixes only that checker field. The reported run used
a fresh output directory after that commit.

The successful external result record has SHA-256
`a0345119455e65ba9d59e26d6b78f1e5d43f25f919a7438c24a114b6b193bf02`.
It is 8,821 bytes; the complete external run directory is 368 KiB. Its suite
SHA-256 is
`7be24843ffae71de65a1eab243eab9f592ce614097d701d5234eabd0c5980a9c`,
and its expectations SHA-256 is
`e058c9e29056ffa2141b7c34349f038d98df1f7423eec81df577978e6887c962`.

## Published inventories

| Framework | Canonical inventory | Cases | Bytes |
|---|---|---:|---:|
| vLLM 0.27.1 | [`77ea0abb...46e30d1`](../../offline/calibration/model-inventories/77ea0abb4803d2cab5689f6893563e9a973f0e29160ec1975f1c76b3046e30d1.json) | 15 | 50,315 |
| SGLang `0.5.19.dev345+gbfeae4e79` | [`9d1c6164...e3d4747`](../../offline/calibration/model-inventories/9d1c6164d149b98a4d019bee31d8d3a7ce3ee38cb96b7e0bab03393fde3d4747.json) | 15 | 50,366 |

Both records pass the strict `simllm-model-kernel-inventory-v1` reader. Each
filename is the SHA-256 of its exact canonical bytes. The two repeated
extractions for each framework reproduced both the inventory and StepRecord
stream byte for byte. The common StepRecord stream has SHA-256
`7aa340751933a27135792099de97b1fde1a8f58cee64783b8ee7b82b12f174fd`.

After removing framework identity and the framework-owned physical join-task
list, the two inventory objects are byte-equivalent. Their full record hashes
differ because framework provenance is part of record identity.

## What the inventory counts

The GPU first projects each token into Q, K, V and gate vectors. A width-four
causal convolution reads the three retained samples that precede the current
sample. The recurrent kernel then reads a float32 state matrix, decays it,
forms a rank-one update from the current key and value residual, writes the
matrix back, and multiplies the current query through it. A gated
root-mean-square normalization and output projection return the value heads to
the model width. Every layer also runs the dense gate, up and down MLP
projections.

That mechanism becomes seven Gated DeltaNet families plus the dense MLP in
each of 48 linear-attention layers. The 16 full-attention layers retain the
accepted attention GEMM, attention-score and KV-read families plus the same
dense MLP. One language-model head completes the step. The exact visit count
is therefore

`48 * 8 + 16 * 4 + 1 = 449`.

The conserved case totals are

`F = 47,966,017,280*T + 393,216*P + 2,542,796,800*Z`

FLOPs and

`B = 50,241,239,040 + 307,888,128*R + 65,536*KVT`

HBM bytes, where `T` is new tokens, `P` is the full-attention query-key pair
count, `Z` is sampled tokens, `R` is sequences and `KVT` is full-attention KV
tokens. All coefficients and all case results are integers.

## Evidence by class

### Behavioral relations

- R1 passed independently through both framework configuration surfaces.
  Each reported the exact wrapper and text types, dense geometry, Gated
  DeltaNet dimensions, float32 recurrent state and exclusions.
- R4 passed with all 16 exact four-layer blocks. Every block places linear
  attention at positions 0, 1 and 2 and full attention at position 3, for 48
  linear and 16 full layers. Totals alone were not accepted.
- R5 passed all three authored five-point shape sweeps. Prefill changes new
  tokens at four sequences; memory decode changes context at four sequences;
  dense decode changes batch at a fixed 2,048-token context per request.
  Recurrent-state bytes stay constant with decode context at fixed batch and
  scale exactly with batch.
- R7 passed: the framework-neutral inventory projections are byte-equivalent.

These four relations form the behavioral class. Exact oracles, structural
invariants, fatal guards and native tests are reported separately.

### Exact oracles

R2 checked 180 family projections per framework, 12 families across 15 cases.
Every shape vector, aggregate FLOP count, aggregate HBM-byte count and phase
visit count equals the frozen independent oracle.

R6 compared two process outputs per framework. Both inventory repeat pairs
and both StepRecord repeat pairs are byte-identical and content-addressed.

### Structural invariants

R3 conserved every case through each framework. Family FLOPs and HBM bytes
sum to the independent coefficient forms above, and family visits sum to 449.

R8 held. The inventory family set contains no vision, multimodal,
multi-token-prediction or speculative family. Code-object and observed-launch
fields remain `absent-by-design`, and an ordinary `simllm` import loads neither
framework runtime.

### Fatal guards

No fatal guard was violated. The current suite and all historical byte locks,
checkpoint revision and config identity, API weight-manifest identity, local
weight-file absence, framework versions and source bindings, text geometry,
schedule, cases, family order, integer work, canonical bytes and absent
physical identities all held.

Had one guard failed, the run would have been void and none of the relations
above would be interpreted.

## Physical sanity before interpretation

One linear layer keeps a `48 * 128 * 128` float32 recurrent matrix for each
sequence. Reading and writing that matrix moves 6,291,456 bytes per sequence
per layer. Across 48 linear layers, an A100 at its declared 2.039 TB/s HBM
roof cannot move the batch-one state in less than 3.086 microseconds or the
batch-16 state in less than 49.369 microseconds. The retained decode-kernel
medians span roughly 7.7 to 57 microseconds, so these analytic floors sit
below or within the observed range. The measurements are a sanity bound only;
no constant is fitted from them.

A second independent check compares static bytes. The text inventory owns
50,241,239,040 static bytes, below the 55,562,855,904-byte BF16 payload of the
whole checkpoint. That direction is required because the text-only inventory
excludes the vision encoder and speculative head.

A third check is mechanism-level. Both current pinned sources independently
construct the same stateful recurrence, width-four convolution and gated
normalization for the 48 linear layers. Pricing those layers as ordinary
stateless full attention would contradict both implementations even if the
final arithmetic happened to conserve.

These checks bound physical plausibility. They do not turn logical HBM work
into a calibrated duration or claim that a kernel reaches peak bandwidth.

## Reproduction

Machine-specific values belong in the gitignored local environment file. With
`VLLM_PYTHON`, `SGLANG_PYTHON`, `QWEN38_CHECKPOINT_ROOT` and
`QWEN38_RUN_ROOT` bound to the pinned local runtimes, exact weight-free config
snapshot and a fresh external output directory, run:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python \
  examples/model_extraction_qwen38_v2/run_study.py \
  --vllm-python "$VLLM_PYTHON" \
  --sglang-python "$SGLANG_PYTHON" \
  --suite-root offline/calibration \
  --checkpoint-root "$QWEN38_CHECKPOINT_ROOT" \
  --output-root "$QWEN38_RUN_ROOT"
```

The runner requires a fresh output directory and writes all logs, repeated
StepRecord streams and the result record outside Git. It performs no network
request, GPU work or model-weight read. Only the two small canonical inventory
objects are promoted into the repository.
