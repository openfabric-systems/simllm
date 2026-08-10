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
