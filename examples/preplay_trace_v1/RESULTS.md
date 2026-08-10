# Pre-play trace v1 results

All frozen checks passed. The original expectations were committed before
the runner implementation and before inference as
`1fee0891dc127da91c2e75a10da1151164ae3d7f`. Integration review then exposed
a consumer-facing routing-attribution gap. The review-triggered amendment
`24116f1aedafb11ad9dc6698d8d70eeefde85cfb` froze corrected prefill coverage,
terminal-token attribution, writer bytes and reader rejections before the
corrective implementation and recapture. The amendment does not rewrite the
original expectations or claim knowledge of the corrective results.

## Run chronology and configuration

The initial implementation landed as `c8b7f2`. The first study invocation
stopped during offline cache resolution, before the model loaded and before
any inference result existed. Commit `7597b2b` changed lookup to resolve the
exact local snapshot directory and fail locally when that revision is absent.
Commit `cb69b02` then made the tokenizer-file manifest part of provenance.
The successful run reported in the first version of this document used the
`cb69b02` code state, including that tokenizer-file manifest.

After integration review, the amended expectations landed as `24116f1` and
the corrected implementation landed as
`a2ec18082a120399e7e79612226f76a33173308b`. The corrective study ran only
after that implementation commit. It used:

- model `ibm-granite/granite-3.0-1b-a400m-instruct` at revision
  `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`;
- Transformers 5.14.1 and Torch 2.11.0+cu130;
- CPU, float32, eight Torch threads, and capture host `teferi.ethz.ch`;
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and the exact local snapshot;
- tokenizer-file manifest SHA-256
  `f62a105df4c8b4dff8b605b58d2bdfbbdec57485659c75ea5e6052c103abcc7c`;
- greedy decoding, plus seeded sampling with seed 173, temperature 0.8, and
  top-p 0.9.

The model was loaded once. Capture-only wall times, excluding model load,
were 3.117 seconds for greedy, 5.880 seconds for sampled A, and 5.799 seconds
for sampled B.

## Routing capture and attribution

The installed `GraniteMoeTopKRouter.forward` returns selected expert IDs,
selected weights and full router logits. The runner hooks each
`layers.<index>.block_sparse_moe.router` return but deliberately recomputes
top-k IDs and normalized weights from the full logits. The trace therefore
records a top-k plus softmax reconstruction, not an observation of the
model's expert dispatch. Discovery depends on Transformers-internal module
names and attributes. Both assumptions are version-sensitive and are
source-verified only for Transformers 5.14.1 and this pinned Granite
snapshot.

The initial prompt forward now contributes one routing record for every
prompt token. During decode, routing is attributed to the forward that takes
generated token `i` as input and produces token `i+1`. Generation stops before
forwarding the terminal EOS, length-cap, or stop-string token. Thus each
request has exactly `len(input_token_ids)` prefill records and
`len(output_token_ids) - 1` decode records.

## Behavioral relations

All three frozen behavioral relation families passed. Structural checks are
reported separately and do not contribute to this count.

### B1: seeded determinism

Sampled A and sampled B are byte-identical. Both files are 434,864 bytes and
have SHA-256
`bf2b338e50a028261a4735a08ae708ef5753ad541573befd60597e618e1fe156`.
The difference is exactly 0 bytes.

### B2: sampling provenance

The greedy header contains mode `greedy` with null seed, temperature and
top-p. Both sampled headers contain mode `seeded-sampling`, seed 173,
temperature 0.8 and top-p 0.9. Replacing only the sampled header's sampling
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

The EOS request ends in token ID 0. The length-cap request emitted exactly
its one-token budget. The stop-string request stored `SIMLLM_STOP` as the
exact matched string.

## Exact-oracle relations

E1 passed for all three live instances. Strict read followed by canonical
streamed write produced byte-identical files. The greedy original and round
trip are 360,758 bytes with SHA-256
`5d0ee3a1af045c404f9aa9baa7d063dc446584da60282f4492a1e72f08e081b5`.
The sampled originals and round trips share the sampled hash reported under
B1.

E2 passed in the dependency-free native suite. Writing the in-memory
synthetic trace reproduced the frozen 1,368-byte `writer_golden.jsonl`
fixture exactly, with SHA-256
`cabd00c77372e859a1996aee262e18512eda6aec5b257c531f2329b150882f9e`.
The same test confirmed that default creation leaves an existing file
byte-identical and raises `FileExistsError`, while explicit
`overwrite=True` reproduces the frozen bytes.

Exact-oracle results are not added to the behavioral relation count.

## Structural invariants, unscored

All structural checks passed and were treated as fatal, unscored gates. The
largest weight-sum errors below come directly from the committed
`run_study.py` field `max_weight_sum_error` in the generated `summary.json`.

| Trace | Requests | Output tokens | Prefill forwards | Decode forwards | Layer routes | Largest weight-sum error | Expert ID range |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy | 3 | 9 | 57 | 6 | 1,512 | `1.4901161193847656e-07` | 0 to 31 |
| sampled A | 3 | 22 | 57 | 19 | 1,824 | `1.4901161193847656e-07` | 0 to 31 |
| sampled B | 3 | 22 | 57 | 19 | 1,824 | `1.4901161193847656e-07` | 0 to 31 |

Every forwarded input token had exactly 24 ordered layer records. Every
layer record had eight distinct expert IDs below 32 and eight finite,
nonnegative weights. Every weight sum stayed inside the frozen `1e-5`
absolute bound. The 57 prefill records equal the total prompt-token count.
The greedy and sampled decode counts equal their output counts minus one
terminal token per request, proving that no phantom terminal forward remains.

Strict parsing also verified unique request identities, exact phase-specific
token indices and IDs, row schemas, declared counts and the completeness
footer. Native negative tests rejected mid-line truncated JSON, a header
missing `model_revision`, unknown fields, missing or extra phase rows and an
invented terminal-token forward. These are fatal negative checks, not scored
relations.

## Artifacts

Detailed captures and their round trips remain outside Git in the
machine-local directory used for the historical run; its resolved historical path is
intentionally omitted. The run summary is `summary.json`, SHA-256
`78d38d67d06e415c9915ea5dc5bca10c77fc8cfd26107b9a823e2112a72dc0f3`.
New runs default to `${SIMLLM_DATA_ROOT}/preplay_trace_v1/`.

The tracked Granite fixture is one complete `length-cap` request with 22
prefill records, one terminal generated token and zero decode records. It is
126,563 bytes with SHA-256
`36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341`.
It exercises real 24-layer routing and strict parsing without requiring
Torch or Transformers. The much smaller frozen writer fixture supplies the
independent canonical-byte oracle.

These results establish a deterministic CPU realization and strict trace
artifact for the pinned environment. They do not claim token parity with a
GPU serving run.
