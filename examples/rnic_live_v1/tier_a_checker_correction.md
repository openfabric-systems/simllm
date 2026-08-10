# Tier A checker correction chronology

This note is post-specified. It is not an expectation amendment and makes no
pre-registration claim.

The behavior and acceptance relations were frozen in commit `35c2ee4` before
implementation began. During the first nonfinal native producer smoke, before
the registered acceptance run, the checker rejected the first structurally
valid row at the one-token-per-WQE cardinality check. Python's
`Counter(mapping)` constructor interprets mapping values as counts. The
checker incorrectly compared the observed WQE counts with
`Counter(cell_wqes)`, whose counts were the raw WQE dictionaries, rather than
with one count for each WQE key.

The correction changes only the issued and terminal comparisons to use
`Counter(cell_wqes.keys())`. It does not change the grid, oracle, behavioral
relations, quantitative bands, fatal invariant inventory, raw schema,
negative control or evidence-class counts. The nonfinal smoke output remains
external and is not study evidence. The registered fake and htsim commands
are dry-run again with `--check-only` before this correction is committed.
