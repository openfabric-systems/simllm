# Routed supply v1 expectations

This study is frozen in three dependency-ordered sections. Each section is
committed before its implementation and before any result-producing run of
that section. The CORE-6 section is the first freeze. PLAY-4 and the second
half of TRAF-2 will be appended in later expectations-only commits. Results
must cite all three freeze commits and preserve their chronology.

## CORE-6 graph contract freeze

### Decision and source audit

The audited source state is repository commit `fc282ef`. Before this freeze:

- `simllm/core/execution.py:149-163` gives `CollectiveWork` one scalar
  `payload_bytes` and explicitly says that captured variable per-pair sizes
  are not representable.
- `simllm/core/execution_io.py:148-160` validates only that scalar, while
  `simllm/core/execution_io.py:353-361` and
  `simllm/core/execution_io.py:500-520` define its strict JSON write and read
  paths.
- `simllm/traffic/patterns.py:103-127` already consumes a sparse mapping from
  ordered rank pairs to bytes. Missing and zero entries emit no send. This is
  the downstream semantic source for the graph representation.
- `simllm/traffic/execution_goal.py:254-273` currently expands the scalar to
  every off-diagonal ordered pair. The implementation must change only this
  decoding when the new table is present.
- `simllm/core/runtime.py:1603-1643` likewise schedules one scalar-sized send
  for every off-diagonal pair. It is the second consumer that must agree with
  the serial renderer.

The chosen representation is an optional sparse table on `CollectiveWork`,
not a schema bump. The in-memory and JSON field is `pair_payload_bytes`.
Every entry is `(source_rank, destination_rank, payload_bytes)`, entries are
strictly source-major sorted, zero-byte pairs are omitted, and a nonempty
table is valid only for `("all-to-allv", "pairwise")`. Table and scalar are
mutually exclusive: a table requires `payload_bytes == 0`; an empty table
leaves the scalar authoritative. This keeps one byte authority and retains
`simllm-execution-graph-v1` because the extension is optional and old v1
objects keep exactly their prior meaning and encoding.

### CORE-B1: decision-relevant rendered-size relation

Build one two-rank pairwise all-to-allv graph with sparse table
`((0, 1, 2048), (1, 0, 4096))`. After strict JSON round trip, the serial GOAL
renderer and `CoarseDeviceRuntime` must each create exactly two semantic
sends with ordered-pair sizes `{(0, 1): 2048, (1, 0): 4096}`. In the second
instance, change only `(1, 0)` to 6144. Exactly that rendered send and runtime
transfer must grow by 2048 bytes; `(0, 1)` must remain 2048 bytes. Both live
instances are scored. Ordering of the two entries is structural and is not a
scored behavior.

The successor is the TRAF-2 section: its captured routing must populate the
same table and reach the step sink's JCT. CORE-B1 alone is component evidence
and does not close the live-chain claim.

### CORE-E1: strict v1 compatibility oracle

The uniform graph fixture has execution ID `core6-uniform`, step index 7,
release time 11 ps, ranks `(0, 1)`, scalar payload 2048 bytes, pairwise
algorithm, operation ID `a2av`, and completion boundary `a2av`. Canonical
sorted compact JSON plus one LF is 559 bytes with SHA-256
`f4a5a70f5bd4a0c2fed874baa88f3035266a54f386a59927e115872c2bcff0a3`.
Its rendered GOAL is 166 bytes with SHA-256
`46ca1ea42952c5e0c66ea9eebb8947e770f7090f6cbdea6c711b4e764b412f5b`.

Reading the frozen old-v1 JSON and writing it again must reproduce those 559
bytes exactly. Its decoded table is empty. Re-rendering must reproduce the
166 GOAL bytes exactly. The new writer must omit `pair_payload_bytes` when the
table is empty. This identity-off oracle is fatal and unscored because it is
forced by the absent optional field.

### CORE structural invariants, fatal and unscored

Validation must reject duplicate pairs, self-pairs, ranks outside the
collective, nonpositive pair sizes, noncanonical pair order, a nonempty table
with a nonzero scalar, and a table on any collective or algorithm other than
pairwise all-to-allv. A pairwise operation with neither a positive scalar nor
a nonempty table must still fail before runtime state changes. Unknown JSON
fields remain rejected.

### Registered CORE command and pre-freeze dry run

The registered result-producing invocation for this section is:

```text
.venv/bin/python examples/routed_supply_v1/run_study.py --sections core --out "$SIMLLM_ROUTING_RUN_ROOT"
```

Its pre-freeze dry run is the same command with `--check-only`. Check-only
parses the complete CLI and validates only frozen literal shape. It must print
its confirmation by design, import no target implementation, execute no graph
behavior, and create no result path or artifact.

## PLAY-4 captured-routing projection freeze

### Schema decision and source audit

The audited source state is repository commit `6b3e46f`. Before this freeze:

- `simllm/preplay/schema.py:132-177` defines the forwarded-token and request
  authority. Prefill tokens mirror prompt inputs; decode tokens mirror
  `output_token_ids[:-1]` because no forward consumes the terminal token.
- `simllm/preplay/schema.py:340-368` requires contiguous phase-local token
  indices and exact ordered MoE layers, while
  `simllm/preplay/schema.py:413-425` checks the prompt and nonterminal decode
  token identities against the request.
- `simllm/preplay/join.py:422-484` hashes and strictly parses the trace, joins
  only named arrivals, preserves arrival order in `run.requests`, and pins
  each request to that hash before atomically extending bookkeeping.
- `examples/preplay_trace_v1/RESULTS.md:41-58` records the Granite router
  capture mechanism and the input-token attribution convention.
  `examples/preplay_trace_v1/RESULTS.md:130-154` records the exact routed row
  counts and both the full greedy artifact and tracked fixture hashes. The
  tracked fixture has real prefill routing and no decode forward; the full
  greedy artifact supplies six real decode forwards through the local
  `SIMLLM_GRANITE_DECODE_TRACE` setting.

PLAY-4 adds the strict object schema `simllm-routed-experts-v1`. One projection
records the source trace schema and SHA-256, expert count, top-k, ordered MoE
layers, and joined requests in `PreplayReplayRun.requests` order. Each request
records prompt and output token counts plus one ordered token list. Each token
retains phase, phase-local index, input token ID, and ordered
`(layer_index, expert_ids)` assignments. Gate weights are deliberately absent:
TRAF-2 routes bytes by destination identity and COMP-7 later counts expert
load; neither consumer uses the normalized weights.

The public names are `RoutedExperts`, `RoutedRequest`, `RoutedToken`, and
`RoutedLayer`, with strict JSON read, write and validation functions plus
`project_preplay_routing(run)`. The projection must re-read and hash the trace
named by the joined run. A stale path, byte mismatch, request mismatch or
routing-reference mismatch fails before a projection is returned.

### PLAY-B1: executed-input attribution relation

Project two source instances after joining their requests:

1. The tracked Granite `length-cap` fixture has prompt count 22, output count
   1, 22 prefill forwards and 0 decode forwards. The projected forwarded-token
   count is exactly `22 + (1 - 1) = 22`, one below the naive prompt-plus-output
   count of 23.
2. The full greedy Granite artifact has three requests with prompt counts
   15, 22 and 20 and output counts 3, 1 and 5. It has 57 prefill forwards and
   6 decode forwards. The projected count is exactly
   `57 + ((3 - 1) + (1 - 1) + (5 - 1)) = 63`, three below the naive count of
   66. Per request the decode counts are exactly 2, 0 and 4.

Both live projection instances are scored. The signed relation is fewer
forwarded tokens than prompt plus outputs, with the exact difference equal to
one terminal token per joined request. Every forwarded token retains 24 layers
and eight expert IDs per layer, so the assignment totals are exactly 4,224 and
12,096.

### PLAY-B2: per-request join relation

Join the full Granite requests in order `stop-string`, `length-cap`,
`eos-brief`. The projected request order must match exactly, while each
request's token rows remain byte-identical to that request in the trace-order
projection. There must be no unjoined row, loss, duplication or cross-request
token movement. This live reordered join is one scored instance. Array order
by itself is structural; the scored content is stable request association
under the changed join order.

The successor is the TRAF-2 section, which consumes these same assignments.
PLAY-B1 and PLAY-B2 alone are component evidence and do not close the
step-sink metric claim.

### PLAY-E1: exact canonical projection oracle

Canonical sorted compact JSON plus one LF must match these frozen values:

| Source and join order | Bytes | SHA-256 |
|---|---:|---|
| tracked `length-cap` | 30,874 | `e3af45f896ff0a7005c4da0d6b4d3cfba7a00c868653e9aea581f49c37392e7a` |
| full greedy, trace order | 87,845 | `7d1875ac46de07f7ed2ed814dc8596ecc500a74f51c626a9b98b2ecb38d949d5` |
| full greedy, reversed request order | 87,845 | `18a5f737d1680aac22df3ca4a095d2f4ef5205c2433379de86ed96afc77687c1` |

The source trace hashes are
`36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341`
for the tracked fixture and
`5d0ee3a1af045c404f9aa9baa7d063dc446584da60282f4492a1e72f08e081b5`
for the full greedy artifact. These oracles were derived before implementation
by parsing the already-landed JSONL rows with the Python standard library,
constructing the schema object above in memory, and hashing it. The audit
printed counts and hashes by design and produced no artifact. Exact wire
oracles are reported separately from PLAY-B1 and PLAY-B2.

### PLAY structural invariants, fatal and unscored

Strict validation rejects an unknown schema or field; an invalid trace hash;
duplicate requests; a missing, duplicate or noncontiguous phase token; a
decode terminal forward; wrong prompt or output counts; missing, duplicate or
out-of-order layers; duplicate, negative or out-of-range expert IDs; and an
expert count, top-k or layer set that disagrees with the source trace. Reading
and writing never imports Torch, Transformers, vLLM or SGLang.

### Registered PLAY command and pre-freeze dry run

The registered result-producing invocation for this section is:

```text
.venv/bin/python examples/routed_supply_v1/run_study.py --sections play --out "$SIMLLM_ROUTING_RUN_ROOT" --decode-trace "$SIMLLM_GRANITE_DECODE_TRACE"
```

Its pre-freeze dry run is the same command with `--check-only`. Check-only
parses the complete CLI, requires a decode-trace argument for this section,
and validates only the frozen hashes, byte counts and row counts. It prints its
confirmation by design, does not read either trace, executes no projection,
and produces no artifacts.
