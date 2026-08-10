# Pre-play arrival join v1 results

All frozen PLAY-2 checks passed. The expectations-only commit
`c4c17cff81e550053e090af430e3041e9efde057` preceded implementation and the
first study run. The implementation and study harness landed as
`017a7219a22b24f56d44bbfac60df8b35a25be5e`. The scored run was executed only
after that implementation commit.

## Run configuration

The run used the repository `.venv`, the tracked Granite length-cap fixture,
and a dependency-free synthetic two-request trace. It exercised one and two
requests, plus zero-origin and exactly 7,000 ps shifted arrivals. Bulk output
remains outside Git in the machine-local directory used for the historical
run; its resolved historical path is intentionally omitted. New runs default to
`${SIMLLM_DATA_ROOT}/preplay_arrival_join_v1/`.

The generated `summary.json` is 2,779 bytes with SHA-256
`fb4fad6dcac4730c0b124f46accc48d24a306232968d623a73b4afb13a1a3463`.
The generated `cells.csv` is 556 bytes with SHA-256
`004c68e713ddc53c8eb7f204768cada11b757c7c9147f7b51dc411243334991a`.

## Scored behavioral relations

All three scored families passed.

### B1: exact request projection

All four cells preserved exact identity, arrival timestamp, output length,
stop reason, token IDs and routing reference between each returned joined
request and its framework-request bookkeeping object. Object cardinality was
one for each Granite cell and two for each synthetic cell. Shifting arrivals
by 7,000 ps changed every `created_at_ps` and arrival metadata value by exactly
7,000 ps and changed no non-arrival projection field.

### B2: trace authority survives the join

Every run record and request routing reference carried the hash of the trace
bytes used for that cell. Both Granite cells retained the frozen hash
`36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341`
exactly. The path recorded for the tracked fixture resolved to the tracked
file in this worktree, and every request reference named its own trace row.

### B3: cardinality scaling

The tracked one-request cell produced exactly one joined projection, one
framework-request object and one routing reference. The synthetic two-request
cell produced exactly two of each. Each returned projection and bookkeeping
object shared one stable object identity; no second mutable lifecycle was
created.

## Exact oracle and fatal guards

E1 passed. Canonical write, strict read and canonical rewrite produced
byte-identical 1,160-byte run records. The first record has SHA-256
`6d8a667cbf806dd7170ed2c6596560b1feacfdd023a2088c699b3cbc145a64b7`.
Native negative tests separately rejected unknown fields, unsupported schema,
duplicate identities, an inconsistent output length and a mismatched routing
hash.

All fatal unscored study guards passed. Empty arrivals, duplicate identities,
negative timestamps, boolean timestamps and a request missing from the trace
all failed before the bookkeeper changed. Unit tests also confirmed that a
pre-existing object collision rolls back the complete second join. These
checks remain outside the scored denominator.

## Genuine-risk fraction

Two of the three scored families, 67 percent, carried genuine implementation
risk for a competent implementation:

- B1 could plausibly fail through timestamp-unit drift, a token serialization
  mismatch, or a non-atomic partial append.
- B2 could plausibly fail if the run named only a path, hashed different bytes
  than it parsed, or omitted the per-request versioned routing reference.
- B3 was low-risk rather than genuinely risky. Its one-to-two scaling follows
  the same request loop and mainly detects accidental loss or duplication.

The decision-relevant B2 relation passed, so the adapter-side projection
boundary remains accepted. PLAY-3 is now responsible for proving that this
component changes the live scheduler and metric chain while the bypass stays
byte-identical.
