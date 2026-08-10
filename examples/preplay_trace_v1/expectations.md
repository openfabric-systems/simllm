# Pre-play trace v1 expectations

This document freezes the PLAY-1 validation contract before the runner or
trace implementation exists and before the first inference run. The live
study uses only the already cached model files and runs on CPU.

## Frozen model and environment

- Model: `ibm-granite/granite-3.0-1b-a400m-instruct`
- Revision: `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`
- Historical cache root: resolved machine-local path intentionally omitted
- Historical runtime: machine-local pinned environment, resolved historical path
  intentionally omitted
- Device: CPU only
- Effective dtype: `float32`
- Sampling seed: `173`
- Seeded-sampling configuration: temperature `0.8`, top-p `0.9`
- Granite routing configuration: 24 MoE layers, top-k 8, 32 local experts

The run sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Network access
is not an admissible fallback.

For a current reproduction, `HF_HOME` selects the cache root and
`SIMLLM_VLLM_PYTHON` selects the compatible interpreter.

## Frozen requests

Each prompt is formatted with the cached tokenizer's chat template and an
assistant generation prompt.

1. `eos-brief`: prompt `Reply with exactly one word: OK`, no stop string,
   `max_new_tokens=16`. Greedy decoding must terminate on token ID 0 and the
   artifact stop reason must be `eos`.
2. `length-cap`: prompt `Continue this sequence with ten more integers: 1 2 3`,
   no stop string, `max_new_tokens=1`. Greedy decoding must emit exactly one
   token and the artifact stop reason must be `length-cap`.
3. `stop-string`: prompt
   `Reply with exactly SIMLLM_STOP and no other text`, stop string
   `SIMLLM_STOP`, `max_new_tokens=16`. Greedy decoding must include the token
   that completes that text and report `stop-string` with the exact matched
   string.

The three requests run once in greedy mode. The same ordered request set also
runs twice in seeded-sampling mode with seed 173. Sampling mode and the
request-level length or stopping condition are the two varied parameter
families.

## Scored behavioral relations

### B1: seeded determinism

The two seeded-sampling captures use identical requests, model provenance,
sampling configuration, seed, and capture host. Their complete JSONL bytes
must be identical. This is an exact relation, with zero differing bytes and
the same request and token row order.

### B2: sampling provenance

The greedy trace records mode `greedy`, a null seed, and null sampling
parameters. Each sampled trace records mode `seeded-sampling`, seed 173,
temperature 0.8, and top-p 0.9. The mode records must differ exactly in those
sampling fields. Output token IDs are allowed to differ and any token
difference is diagnostic, not required for a pass.

### B3: stop semantics

The greedy run must report all three engineered terminal conditions exactly:
`eos-brief` as `eos`, `length-cap` as `length-cap`, and `stop-string` as
`stop-string`. The EOS output ends in token ID 0. The length-capped output has
exactly one token. The stop-string record names `SIMLLM_STOP` as its matched
string. No request may have a terminal condition inconsistent with its
output.

## Exact-oracle relation

### E1: strict schema round trip

Reading each complete trace through the strict reader and writing the parsed
provenance and requests again must reproduce the original bytes exactly.
Unknown fields, malformed row ordering, duplicate request identities,
missing token rows, non-contiguous token indices, and trailing incomplete
requests must be rejected. E1 is an exact oracle and is reported separately
from B1 through B3.

## Structural invariants

These checks are fatal if violated but do not increase the behavioral pass
count.

- Every row carries schema tag `simllm-preplay-trace-v1`, and the header's
  provenance repeats that schema version.
- Provenance records model identity, exact revision, effective dtype,
  tokenizer SHA-256, sampling mode and seed, capture host, Transformers and
  Torch versions, top-k 8, expert count 32, and MoE layer indices 0 through
  23.
- Request identities are unique and every declared output token has exactly
  one token row.
- Every generated token carries exactly 24 layer-routing records. Each layer
  record carries exactly 8 distinct expert IDs in `[0, 32)` and 8 finite,
  nonnegative gate weights.
- Each layer's gate weights sum to 1 within absolute error `1e-5`.
- The reader reaches a complete footer whose request and token counts equal
  the rows observed before it, with no nonblank trailing content.

## Interpretation

A pass demonstrates a deterministic CPU realization and a strict,
stream-written trace artifact for this cached model. It does not claim token
parity with GPU serving. CPU and GPU numeric order may change sampled tokens
and subsequent routing.
