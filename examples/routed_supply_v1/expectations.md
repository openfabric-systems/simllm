# Routed supply v1 expectations

This study is frozen in three dependency-ordered sections. Each section is
committed before its implementation and before any result-producing run of
that section. CORE-6 is first, PLAY-4 is second, and the second half of TRAF-2
is third. Results must cite all three freeze commits and preserve their
chronology.

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

## TRAF-2 captured-routing expansion freeze

### Expansion decision and source audit

The audited source state is repository commit `d878464`. Before this freeze:

- `examples/preplay_trace_v1/granite_length_cap.jsonl:1-2` declares 32 experts,
  top-k 8, 24 MoE layers and the joined `length-cap` request.
  `examples/preplay_trace_v1/granite_length_cap.jsonl:3-24` is the external
  Granite capture source: exactly 22 prefill input tokens, each with eight
  expert IDs at every layer. Line 25 closes the source with the exact row
  counts. The frozen trace hash is the PLAY-E1 tracked hash above.
- `simllm/preplay/routing.py:40-95` is the versioned request, token and layer
  projection that TRAF-2 consumes. `simllm/preplay/routing.py:482-544` binds it
  to the joined source trace hash and request identity.
- `simllm/placement/manifest.py:17-19` requires every EPLB replacement to bump
  `placement_epoch`; `simllm/placement/manifest.py:48-65` carries per-rank,
  per-layer global expert IDs at that epoch.
- `simllm/traffic/step_comm.py:91-142` currently emits one uniform scalar
  payload on every ordered pair. It does not consume captured assignments,
  does not deduplicate experts that share a destination rank, and does not
  select an ownership epoch.
- `simllm/core/execution.py:148-169` is now the sole graph representation for
  sparse ordered-pair payloads. `simllm/traffic/patterns.py:103-127` is the
  downstream expansion authority: it emits exactly the positive ordered pairs
  supplied and nothing for a missing or zero pair.
- `simllm/backends/step_sink.py:195-235` is the live JCT path. It expands the
  step, renders GOAL, invokes the selected `htsim_rnic` profile and returns its
  JCT as `StepResult.step_latency_ps`. `simllm/backends/step_lowerer.py:133-159`
  and `simllm/backends/step_lowerer.py:224-249` are the graph projection that
  must carry the same pair table and placement epoch.
- The upstream fluid-source audit is already frozen at
  `examples/m5/expectations.md:24-46`: a full-rate flow of S bytes has service
  debt `S * 8 * 10^12` bit-picoseconds, whole-bps rate, whole-ps ceiling and
  2,000,000 ps propagation. For the two-rank study there is one flow per
  source and destination port, so its rate is exactly B and serialization is
  exactly 40 ps/byte at 200 Gbit/s or 20 ps/byte at 400 Gbit/s. The phase
  structure is independently fixed by
  `simllm/traffic/patterns.py:103-127`: both ordered pairs release together and
  the phase completes at the larger pair's completion.

The new traffic-side input is one captured `RoutedExperts` projection plus a
strict sequence of placement-manifest snapshots and an explicit step-index to
placement-epoch map. A step without this optional supply follows the existing
uniform scalar path. A supplied step must find exactly one named request row,
exactly one selected snapshot, complete ownership of every expert at every MoE
layer, and a scheduled token slice within the captured prefill or decode
phase. It then emits a sparse table and records the selected epoch on the
lowered collective operation.

Real dispatch sends at most one hidden vector per `(token, destination rank)`
even when several of that token's selected experts live there. The combine is
pre-reduced at the expert owner and returns one vector along the reverse pair.
For vector width `V = hidden_size * dtype_bytes = 1024 * 2 = 2,048` bytes,
source rank s, destination rank d and layer l, the frozen expansion is

```text
dispatch_bytes(s,d,l) = V * sum over scheduled tokens t of
    indicator(s != d and any owner(epoch,l,e) == d for e in experts(t,l))
combine_bytes(s,d,l) = dispatch_bytes(d,s,l)
```

This definition removes both registered approximations: no uniform routing
and no one-copy-per-expert inflation. Gate weights do not affect bytes.

### Fixed record and placement snapshots

The study record schedules the joined `length-cap` request as one 22-token
prefill at context length 22, with no cached tokens. The EP group is global
ranks `(0, 1)`. Model geometry is 24 layers, hidden size 1024, dtype width 2,
32 experts, top-k 8 and 16 resident experts per rank. The same request record
is evaluated at two placement epochs:

- Epoch 0 assigns experts 0 through 15 to rank 0 and 16 through 31 to rank 1
  at every layer.
- Epoch 1 retains epoch 0 except at layer 13. Rank 0 then owns experts
  `(2, 7, 8, 10, 11, 13, 14, 15, 16, 18, 20, 21, 22, 24, 25, 27)` and rank 1
  owns the complement. This is a fixed EPLB snapshot chosen before execution
  to make epoch selection observable in both pair sizes and JCT.

The ownership lists and the step-to-epoch sequence are author-defined run
configuration. Their validity is fatal and unscored; they do not add to a
behavioral denominator.

### TRAF-B1: exact captured pair distribution

For each layer below, a row `(x, y)` means dispatch pair sizes
`(0, 1, x)` and `(1, 0, y)` in canonical source-major order. Combine must be
the exact transpose `(0, 1, y)`, `(1, 0, x)`. These bytes are a closed-form
count over the external fixture rows, evaluated with the standard library
before implementation and before any backend run:

- Epoch 0: every layer is `(45,056, 45,056)` except layer 2, which is
  `(45,056, 43,008)`. Dispatch totals are 1,081,344 bytes from rank 0 to rank
  1 and 1,079,296 bytes from rank 1 to rank 0.
- Epoch 1: epoch 0 remains exact except layer 13 becomes
  `(40,960, 38,912)`. Dispatch totals are 1,077,248 bytes from rank 0 to rank
  1 and 1,073,152 bytes from rank 1 to rank 0.

The graph lowerer and the GOAL text written by the step sink must contain
exactly these 24 dispatch and 24 transposed combine tables for the selected
epoch. Every `ExecutionOperation` for those tables must carry that epoch.
These are two scored live expansion instances, one per epoch. Canonical table
ordering, snapshot cardinality and the 96-flow count are structural and
unscored.

### TRAF-B2: exact fluid JCT and signed baseline delta

The study uses a fixed provider estimate of 24,000 ps and zero host delay.
The sink therefore emits one 1 ns calc per layer. The existing uniform path
uses

```text
U = 22 tokens * 8 experts * 1,024 hidden * 2 bytes / 2 ranks
  = 180,224 bytes per ordered pair per phase
```

For epoch e, let `M(e,l)` be the larger of its two frozen dispatch sizes at
layer l. The exact two-rank fluid relations are

```text
q(B) = 8 * 10^12 / B = 40 ps/byte at 200G, 20 ps/byte at 400G
JCT_routed(e,B) = 24 * 1,000
                + 2 * sum_l (M(e,l) * q(B) + 2,000,000)
JCT_uniform(B) = 24 * 1,000
               + 48 * (180,224 * q(B) + 2,000,000)
```

Here `sum_l M(0,l) = 1,081,344` bytes and
`sum_l M(1,l) = 1,077,248` bytes. The frozen grid is:

| Epoch | Link rate | Routed JCT ps | Uniform JCT ps | Routed minus uniform ps |
|---:|---:|---:|---:|---:|
| 0 | 200 Gbit/s | 182,531,520 | 442,054,080 | -259,522,560 |
| 0 | 400 Gbit/s | 139,277,760 | 269,039,040 | -129,761,280 |
| 1 | 200 Gbit/s | 182,203,840 | 442,054,080 | -259,850,240 |
| 1 | 400 Gbit/s | 139,113,920 | 269,039,040 | -129,925,120 |

Every cell must match to 0 ps. The required signed direction is strictly
negative in all four cells: destination deduplication makes routed JCT lower
than the old uniform approximation. Doubling bandwidth halves only the wire
term, so routed JCT decreases by exactly 43,253,760 ps at epoch 0 and
43,089,920 ps at epoch 1. The layer-13 EPLB change decreases JCT by exactly
327,680 ps at 200G and 163,840 ps at 400G. These four live sink cells are
scored. The exact arithmetic is pre-specified; no measured value may update
it.

### TRAF-E1: uniform identity-off oracle

With the same record, geometry, EP group and 1 ns layer calcs but no captured
supply, the current uniform GOAL is exactly 13,200 bytes with SHA-256
`d708e998685b617478e891b316728d14b8ac6185a62b73817f80af1c5adff518`.
It has 96 sends, every one 180,224 bytes. The optional supply's absent path
must reproduce those bytes exactly and the lowered graph must retain scalar
`payload_bytes = 180224` with no pair table. This identity-off oracle is fatal
and unscored because configuration forces the legacy path.

### TRAF structural invariants, fatal and unscored

Reject duplicate placement epochs; an unsorted or duplicate step map; a
missing selected epoch; manifest ranks outside the EP group; inconsistent
rank epochs; missing, duplicate or multiply owned experts; routing expert
count, top-k or layer indices that disagree with model geometry; a missing or
duplicate scheduled request; a prefill or decode slice outside the captured
phase; and a projection whose source trace changed. Fail before GOAL or
backend artifacts are created. A zero-token drain stays empty. Dense geometry
and EP width one retain their existing bypass behavior.

### Registered TRAF and combined commands

The registered result-producing invocation for this section is:

```text
.venv/bin/python examples/routed_supply_v1/run_study.py --sections traffic --out "$SIMLLM_ROUTING_RUN_ROOT"
```

After all three freezes, the canonical single-study invocation is:

```text
.venv/bin/python examples/routed_supply_v1/run_study.py --sections core,play,traffic --out "$SIMLLM_ROUTING_RUN_ROOT" --decode-trace "$SIMLLM_GRANITE_DECODE_TRACE"
```

Both commands require `SIMLLM_HTSIM_RNIC` and `SIMLLM_TXT2BIN` to resolve the
provided backend executables for result production. Their pre-freeze dry runs
are the same commands with `--check-only`. Check-only parses the complete CLI
and validates only frozen literal shape and exact signed-delta consistency. It
prints its confirmation by design, imports no traffic target implementation,
does not invoke either executable, and produces no artifacts.
