# Single-pass pending publication snapshot result

The source-paired reader study passes. In its largest frozen input, one
pending-publication check visits **1,523 values instead of 7,030**, saving
exactly 5,507 recursive visits. All six local job configurations preserve
byte-identical records across sources and exact ordinary-versus-deferred
publication values.

This qualifies the reader refinement needed for the next CORE-68 native
attempt. CORE-68 remains open until its complete native engine grid passes
under the original process limits. Neither retained native VOID is rescored.
CORE-69 cancellation and CORE-70 shared-resource composition remain excluded.
This study makes no native-runtime speedup, GPU latency, distributed-scale or
whole-runtime complexity claim.

## What ran

Two fresh local worker processes run the same committed capture code against
before source `ce8bf357e7477291d2a3cfc55b8a6875dd789c06` and after source
`1549b6be0afd55f76de64d6049b23f9cc9ab8874`. The expectations-only commit
`5efe20b9df1c8156cf33409a9f5db6086a209fba` precedes implementation and the first run.
Only `simllm/backends/step_sink.py` differs among 216 installed package files.
The typed snapshot visitor is unchanged. All 89 loaded package origins per
worker and 365 tracked study source files are locked to their selected sources.

Ten reader inputs vary prior row count and row width. Synthetic rows populate
three of the ten publication lists only for these reader mechanism inputs;
they are not simulated request outcomes. Six supported local job
configurations per source vary one or two model layers and zero, one or four
prior completed steps. Each compares ordinary execution with deferred
publication at the core-owned completion time. No serving framework, packet
backend or GPU process runs.

## Exact work relation

Let `N(x)` count recursive visits to an input and `V(x)` be its typed snapshot.
For raw selected state `S`, publication state `P`, history `h` and width `q`,
the freeze requires:

```text
N(P) = 21 + 3h(2 + q)
N(V(P)) = 94 + 3h(10 + 4q)
before - after = N(V(S)) + N(V(P)) - 8
```

The final eight visits are the two added source-binding tuples, each carrying
three implementation identities. The selected state has `N(S) = 377` and
`N(V(S)) = 1,869`. The zero-history saving is 1,955 visits; the largest saving
is `1,869 + 3,646 - 8 = 5,507`. Successful top-level snapshot calls fall from
three to one in every case. The core still reads all prior rows at every
check, and the independent price validator retains its full snapshots.

| Prior rows per populated list | Values per synthetic row | Before visits | After visits | Saved visits |
|---|---|---|---|---|
| 0 | 1 | 2,614 | 659 | 1,955 |
| 1 | 1 | 2,665 | 668 | 1,997 |
| 1 | 4 | 2,710 | 677 | 2,033 |
| 1 | 16 | 2,890 | 713 | 2,177 |
| 4 | 1 | 2,818 | 695 | 2,123 |
| 4 | 4 | 2,998 | 731 | 2,267 |
| 4 | 16 | 3,718 | 875 | 2,843 |
| 16 | 1 | 3,430 | 803 | 2,627 |
| 16 | 4 | 4,150 | 947 | 3,203 |
| 16 | 16 | 7,030 | 1,523 | 5,507 |

All ten paired instances satisfy that exact relation. Full binary profiles,
complete exported function/caller rows and pre-profile input captures remain
in the raw receipts. Elapsed worker time is an unscored process observation:
2.668 seconds before and 2.418 seconds after. It is not a native performance
comparison.

## Physical bounds and completion compatibility

Floor: each declared rank must read at least one 16,384-byte matrix per layer;
at the selected 8 TB/s memory rate this requires at least 2,048 ps per layer.

Ceiling: the frozen toy bound is 1,000,000,000 ps per layer for less than one
million arithmetic operations and one MiB of represented work per layer.

The one-layer step is 25,000 ps and the two-layer step is 44,000 ps, both
inside those bounds. These are tiny ideal-host model inputs, not measured
GPU kernel or launch latencies. Their purpose is to preserve the represented
model while changing host-side bookkeeping.

For each layer count the step service `C` is constant, so job completion time
(JCT) is exactly `(h + 1)C`. The layer relation is
`25,000 < 44,000 <= 2 * 25,000 + 1,000` ps. Complete local and collective
artifact rows sum to each step result; identity and graph ownership remain
exact. No publication appears one picosecond before its due time, and exactly
one appears at retirement.

| Layers | Prior completed steps | Step service (ps) | JCT (ps) |
|---|---|---|---|
| 1 | 0 | 25,000 | 25,000 |
| 1 | 1 | 25,000 | 50,000 |
| 1 | 4 | 25,000 | 125,000 |
| 2 | 0 | 44,000 | 44,000 |
| 2 | 1 | 44,000 | 88,000 |
| 2 | 4 | 44,000 | 220,000 |

All six complete job files are byte-identical between sources. Typed records,
results, selected configuration, all ten publication collections, queue
visits, completion events and the final clock agree. These compatibility and
physical-bound checks are fatal and unscored.

## Integrity and evidence classes

The 14 common corruption controls per source preserve their exact exception
classes and messages. They include a changed old row, type-preserving and
type-changing configuration edits, coherent prepared-payload forgery,
nonfinite values, cycles, unsupported objects and early or duplicate
publication. The six new helper controls reject class/instance shadows and
forged reader/binding reports. Full fixture, pre-trigger and post-trigger
values are checked against the independently derived mutation, with explicit
markers only at the declared nonserializable locations. Restoring the fields
does not clear the runtime's poisoned state.

The duplicate-publication control retains its first authorized publication
only: three lists grow from one row to two. It produces no completed core
visit or result. The other controls produce no additional publication beyond
the deliberately inserted corrupt data.

The completed evidence classes are separate:

- Eight required stages complete with 3,555 unscored guards and no violation.
- Twenty exact oracle rows check top-level call counts.
- Ten paired visitor-reduction instances form one behavioral family.
- Six evidence corruption controls reject altered source, count, result,
  inventory and disk receipts. They are unscored integrity guards.
- The software gate passes 157 tests; repository lint passes separately.

All 146 worker raw files, totaling 27,165,848 bytes, match their first receipts
at final reread. Each completed job/control is persisted immediately, and
each mutation input is persisted before its trigger. Full process records,
source manifests and binary profiles remain outside Git. The portable
[publication](results.json) carries every raw hash and byte count.

Raw summary SHA-256:
`ce8f925630fd37185d111e5f024cf6f5a6b32a165a8d30f0cf5550ed9de0514b`.

The first software gate exposed JSON-only assertion input handling. Independent
review then strengthened full-row/fixture admission, origin locking, immediate
retention and typed scalar decoding. Those are post-specified software
regression checks. No frozen behavioral expectation changed, and no paired
campaign was run before the reviewed source was committed.

To reproduce, use a checkout of each recorded source and a fresh external
output directory:

```bash
python -m examples.publication_snapshot_v1.run_study \
  --before-repository "$SIMLLM_PUBLICATION_BEFORE_ROOT" \
  --output "$SIMLLM_EVIDENCE_ROOT/publication-snapshot-fresh"
```
