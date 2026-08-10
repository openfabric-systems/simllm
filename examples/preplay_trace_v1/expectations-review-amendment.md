# Pre-play trace v1 integration-review amendment

This expectations amendment is triggered by integration review of the
unmerged PLAY-1 implementation. It is committed before the corrective code
changes and before the corrective Granite capture. It does not alter or
rewrite the original expectations commit
`1fee0891dc127da91c2e75a10da1151164ae3d7f`, and it does not retroactively
claim that facts observed in the first study were pre-registered.

The frozen model, environment, requests, sampling parameters, behavioral
relations B1 through B3, and exact round-trip relation E1 remain as specified
in `expectations.md`. This amendment corrects routing coverage and
attribution, and adds exact writer and reader oracles.

## Corrected routing coverage

Each request separates output tokens from tokens actually executed by a
model forward pass:

- `output_token_ids` contains every generated token, including the terminal
  EOS, length-cap, or stop-string token.
- `prefill` forward-token records contain exactly one record for every
  `input_token_ids` entry, in the same order, with matching token index and
  token ID.
- `decode` forward-token records contain exactly one record for every
  generated token except the terminal token. Their token IDs and indices
  equal `output_token_ids[:-1]`.

For an output of length `N`, the request therefore has exactly
`len(input_token_ids)` prefill records and `max(N - 1, 0)` decode records.
Every forward-token record carries the complete per-layer routing shape
already frozen in the original expectations: 24 layers, top-k 8, expert IDs
in `[0, 32)`, and normalized gate weights for the Granite study.

These coverage checks are structural invariants. They are fatal if violated
and remain unscored.

## Routing attribution convention

Routing belongs to the forward pass that takes token `t` as input and
produces logits for token `t+1`.

- The prefill forward takes all prompt tokens as input and produces the first
  generated token from the final prompt position.
- A decode forward that takes generated token index `i` as input produces
  generated token index `i + 1`.
- The terminal generated token is never forwarded by the deployment, so the
  artifact must not invent routing for it.

This convention is exact and consumer-facing. The writer and reader must
reject request objects whose forward-token indices, token IDs, phases, or
counts disagree with it.

## Capture-mechanism scope

The pinned Transformers runner recomputes routing assignments from hooked
router logits with top-k selection and softmax. It does not observe the
model's eventual expert dispatch. Router discovery depends on internal
Transformers module names matching `layers.<index>...router`. Both
assumptions are version-sensitive and are verified only for the pinned
Transformers 5.14.1 Granite implementation in this study.

## Exact writer oracle E2

`writer_golden.jsonl` is the frozen canonical byte representation of a
synthetic one-request trace. It has one prefill input token, one terminal
output token, no decode forward-token record, two MoE layers, top-k 2, and
four experts. Writing the corresponding in-memory trace must reproduce that
file byte-for-byte.

The writer must use exclusive creation by default. If the target exists, the
default write must fail without changing one byte. An explicit
`overwrite=True` selection may replace it and must reproduce the same frozen
bytes. These are exact-oracle checks and are reported separately from B1
through B3.

## Strict reader negative cases

In addition to E1, native reader tests must reject:

- JSON truncated in the middle of a row;
- a header provenance object missing `model_revision`;
- missing or extra prefill forward-token rows;
- missing or extra decode forward-token rows;
- a terminal output token represented as a decode forward-token row.

These are fatal negative tests, not scored behavioral relations.

## Study tooling evidence

The corrective `run_study.py` must calculate and emit the largest absolute
gate-weight sum error for each trace in `summary.json`. The existing frozen
bound remains `1e-5`. The result report may quote a measured error only from
that committed summary field.

The corrective study must cite both the original expectations commit and
this amendment commit. Counts from behavioral, exact-oracle, structural, and
native-test evidence remain separate.
