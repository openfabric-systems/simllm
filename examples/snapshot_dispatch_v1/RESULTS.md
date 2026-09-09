# Primitive snapshot dispatch result

**PASS: the largest frozen primitive input removes exactly 1,536 generic type-check calls while retaining all 1,553 input visits.** `snapshot_dispatch_v1` runs a source-paired typed-reader study with actual local model jobs and complete mutation fixtures. The nine nonempty cases satisfy the frozen count relation. Every complete model job and every package function-call count in its profile is identical across the two sources.

The expectations-only commit is `f08da959e9aca1be75e56a21e4cc5db0fa8b15f9`. It precedes the implementation and the first campaign. The qualified implementation source is `9556d845f21e9f142131b1affb85b2b3adf3a38f`; the before source is `f904b29a3d25c2ae0a955cb1d771f6148f3ceb2b`. See [the frozen contract](expectations.md), [machine-readable results](results.json) and [the owning task](../../docs/modules/core.md#open-tasks).

## Work and compatibility

Every primitive value still creates its typed output tuple. The removed work is the generic Enum type test and the temporary type-tag construction for exact builtins. Containers, custom metadata, virtual subclasses, dataclass fields, projections, cycles and ordered failures retain the original fallback. No cache or historical-row shortcut is introduced.

Each input contains h tuple rows, each repeating the six-value primitive vector q times. Generic calls are direct calls from the snapshot visitor to `isinstance`.

| Rows (h) | Repeats (q) | Visits, each source | Generic calls before | Generic calls after | Removed calls |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 3 | 3 | 0 |
| 1 | 1 | 8 | 12 | 6 | 6 |
| 1 | 4 | 26 | 30 | 6 | 24 |
| 1 | 16 | 98 | 102 | 6 | 96 |
| 4 | 1 | 29 | 39 | 15 | 24 |
| 4 | 4 | 101 | 111 | 15 | 96 |
| 4 | 16 | 389 | 399 | 15 | 384 |
| 16 | 1 | 113 | 147 | 51 | 96 |
| 16 | 4 | 401 | 435 | 51 | 384 |
| 16 | 16 | 1553 | 1587 | 51 | 1536 |

The empty-list control remains exactly three generic calls. It is an unscored guard. The ten actual publication-reader cases retain their complete raw snapshots and object-identity slots. Cross-process comparison substitutes fixed labels only for the declared process-local IDs. All other values and every package function-call count are identical.

The six local model jobs cross layer count 1 or 2 with prior-step history 0, 1 or 4. Their full records, prepared simulations, ten publication collections, completion events, queue visits and results are equal across sources. One-layer service is 25,000 ps and two-layer service is 44,000 ps, above the 2,048-ps per-layer byte-read floor and below the conservative 1,000,000,000-ps per-layer ceiling. Whole-job completion adds exactly one unchanged service per step. These tiny ideal-host inputs establish compatibility and do not estimate a real full-model GPU step.

The finite semantic matrix retains complete results or ordered exception type/message and hook trace, including opaque-object identity witnesses. Two isolated virtual-registration processes per source preserve the original missing-numerator rejection. Fourteen deferred-mutation fixtures and six reader-binding fixtures per source retain their full pre-trigger and post-trigger observations, poison state and rejected retry.

## Evidence classes

- Nine completed stages and 5,248 unscored fatal guards, with no violation.
- Sixteen exact-oracle vectors: ten primitive snapshot/count pairs and six complete model-job pairs.
- Nine behavioral instances in one generic-call reduction family.
- Six deliberately corrupted evidence cases, all rejected, unscored.
- Six fresh processes, with matching first and final raw receipts.

Binary profiles, full function/caller rows, complete raw captures and first-exit receipts remain in external bulk storage. Their hashes are in the tracked result. The source, helper and raw inventories are checked again after admission. The raw summary SHA-256 is `93ed0dbc25eee178a12b9ed6e275ffb64a303b8736c8a8755e473390ef2886d0`.

## Project effect and limits

The [first source-pair attempt](void-frozen-v1.json) remains VOID because process and data admission reused a check name. Its 138 raw files and first/final receipts remain unchanged. The naming repair and a composed-admission regression precede this fresh campaign; the frozen inputs and acceptance relations are unchanged.

This qualifies fewer generic `isinstance` calls while preserving complete validation on the declared finite domain. The exact-type prefix adds identity comparisons on fallback visits, and the profiles do not count those bytecode comparisons. This result supplies no total instruction-count or wall-speedup claim. CORE-68 stays open until a fresh native independent-engine campaign completes the original workload within the original limits. Its three earlier VOID attempts remain unchanged. CORE-52, CORE-70, shared-resource composition and the large-model deployment frontier do not close here. No GPU allocation, native framework run, weight download or packet-backend run is part of this qualification.

## Windows runtime identity

The later Windows integration gate exposes a reader assumption: its built-in
`math` module has no separate source file. The
[portability amendment](portability-expectations.md), frozen at
`5fceb445`, responds to that software failure before the correction and new
fixture runs. File-backed identities remain exactly equal to the accepted
Linux helper's complete runtime record. Built-in identities additionally name
their origin and hash the actual loaded Python library obtained from its
operating-system handle. Unrecognized provenance, missing files, invalid
handles, lookup failure and truncated paths reject.

The two library kinds crossed with two backing-file contents preserve the
complete expected receipts and detect changed bytes. The local focused gate
passes 50 checks with one Windows-only check skipped; these are unscored
software checks, not new behavioral points. The actual Windows lookup is also
part of the hosted gate. This changes runtime provenance capture only. All
reported model jobs, profile relations and prior VOID receipts above retain
their original executed source and verdict.
