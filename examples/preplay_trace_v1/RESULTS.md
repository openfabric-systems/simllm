# Pre-play trace v1 results

All frozen checks passed. The expectations were committed before the runner
implementation and before inference as commit
`1fee0891dc127da91c2e75a10da1151164ae3d7f`.

## Run chronology and configuration

The first study invocation stopped during offline cache resolution, before
the model loaded and before any inference result existed. The runner had
treated the supplied HF home as the `hub/` directory itself. Commit
`7597b2b` changed lookup to resolve the exact local snapshot directory and
fail locally when that revision is absent. The first inference-capable run
then used:

- model `ibm-granite/granite-3.0-1b-a400m-instruct` at revision
  `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`;
- Transformers 5.14.1 and Torch 2.11.0+cu130;
- CPU, float32, and eight Torch threads;
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and an exact local snapshot;
- tokenizer-file manifest SHA-256
  `f62a105df4c8b4dff8b605b58d2bdfbbdec57485659c75ea5e6052c103abcc7c`;
- greedy decoding, plus seeded sampling with seed 173, temperature 0.8, and
  top-p 0.9.

The model was loaded once. Capture-only wall times, excluding model load,
were 4.043 seconds for greedy, 7.047 seconds for sampled A, and 6.760 seconds
for sampled B.

## Source-verified routing capture

The installed `GraniteMoeTopKRouter.forward` computes float32 router logits,
selects `top_k` with `Tensor.topk`, normalizes the selected logits with
softmax, and returns the selected IDs, weights, and full logits. The runner
discovers each `layers.<index>.block_sparse_moe.router`, hooks that return,
and derives the stored IDs and normalized weights from the returned logits.
This mechanism found layers 0 through 23 with top-k 8 and 32 experts, exactly
matching the pinned model configuration.

Generation is incremental with the model's Transformers cache. After each
token is selected, the runner forwards that token through every router and
records the resulting assignments. It also forwards the terminal token, so
EOS, length-capped, and stop-string outputs all have a complete route rather
than an uncaptured final token.

## Behavioral relations

The three frozen behavioral relation families passed. Structural checks are
reported separately below and do not contribute to this count.

### B1: seeded determinism

Sampled A and sampled B are byte-identical. Both files are 126,740 bytes and
have SHA-256
`cec2ff6db219497ae5edfd357a869fe2585569b6c843ecaacac07acc3967188c`.
An exact `cmp` returned zero, so the difference is 0 bytes.

### B2: sampling provenance

The greedy header contains mode `greedy` with null seed, temperature, and
top-p. Both sampled headers contain mode `seeded-sampling`, seed 173,
temperature 0.8, and top-p 0.9. Replacing only the sampled header's sampling
object with the greedy object makes its remaining provenance exactly equal
to the greedy provenance.

Token differences were permitted, not required. They were observed for
`eos-brief`: greedy emitted `(2950, 32, 0)`, while sampled decoding emitted
16 tokens `(59, 7648, 688, 844, 4484, 9696, 436, 312, 4281, 1789, 32, 313,
2950, 20, 438, 312)`.

### B3: stop semantics

All three engineered greedy requests matched the frozen terminal condition:

| Request | Observed text | Token IDs | Token count | Stop reason |
|---|---|---|---:|---|
| `eos-brief` | `OK.` | `(2950, 32, 0)` | 3 | `eos` |
| `length-cap` | `4` | `(38,)` | 1 | `length-cap` |
| `stop-string` | `SIMLLM_STOP` | `(2123, 1679, 21062, 81, 15707)` | 5 | `stop-string` |

The EOS request ends in the configured EOS token ID 0. The length-cap
request emitted exactly its one-token budget. The stop-string request stored
`SIMLLM_STOP` as the exact matched string.

## Exact-oracle relation

E1 passed for all three instances. Strict read followed by canonical
streamed write produced byte-identical files for greedy, sampled A, and
sampled B. The greedy original and round trip share SHA-256
`a707829ffaedd2c99f95e756a5fb793882522124ffeef49103cd8a609829d7c2`.
The sampled originals and round trips share the sampled hash reported under
B1. Exact-oracle results are not added to the behavioral relation count.

## Structural invariants, unscored

All structural checks passed and were treated as fatal, unscored gates:

| Trace | Requests | Tokens | Layer routes | Largest weight-sum error | Expert ID range |
|---|---:|---:|---:|---:|---:|
| greedy | 3 | 9 | 216 | `1.22934579849e-07` | 0 to 31 |
| sampled A | 3 | 22 | 528 | `1.34110450745e-07` | 0 to 31 |
| sampled B | 3 | 22 | 528 | `1.34110450745e-07` | 0 to 31 |

Every token had exactly 24 ordered layer records. Every layer record had
eight distinct expert IDs below 32 and eight finite, nonnegative weights.
Every weight sum stayed inside the frozen `1e-5` absolute bound. Strict
parsing also verified unique request identities, contiguous token indices,
row schemas, declared counts, and the completeness footer.

## Artifacts

Detailed captures and their round trips remain outside the repository under
`/data3/yifeng/simllm-dev/wave1-runs/play1_preplay_runner/`. The run summary
is `summary.json`, SHA-256
`05ce6479263a147455efbe9a29778af90b2a12e3cc5abc8aeed1d620a7c99414`.

The repository fixture `granite_length_cap.jsonl` is one complete real
request with one generated token and all 24 routing records. It is 6,913
bytes with SHA-256
`97c745bd967999b92c48d6aaee6a3bcd0a485d1e1c78cd3bebcad436c377a3c0`.
It exercises strict parsing without requiring Torch or Transformers.

These results establish a deterministic CPU realization and strict trace
artifact for the pinned environment. They do not claim token parity with a
GPU serving run.
