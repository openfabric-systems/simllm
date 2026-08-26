# COMP-75 clean repetition expectations

Status: expectations only. No visible calibration value has been read and no
comparison has been performed.

The source allowlist was committed first as `a219ac5`. Its SHA-256 is
`e5bc633175a1636615c9867a2caa10e591cace4ab40bcef13c18108d77c4190b`.
The held-out access ledger is empty. No framework evaluation table, forbidden
anchor payload, CORE-60 external-source value, scored result, model weight, or
web page is an input.

## Frozen physical expectations

The pinned implementation selects FP8 dispatch, groups 7,168 hidden elements
in groups of 128, and carries one four-byte scale for each group. The expected
dispatch vector is therefore `7,168 + 56 * 4 = 7,392` bytes. Combine returns
the BF16 hidden vector and remains `7,168 * 2 = 14,336` bytes.

For eight uniformly selected experts without replacement from 256 logical
experts, one destination rank is incident with probability
`1 - C(248, 8) / C(256, 8)`. Multiple experts on the same destination are
deduplicated to one vector per token. The expected EP32 destination count is
`32p`; the remote count is `31p`, split over seven local peers and 24 fabric
peers.

The accepted local packet machinery reproduced all eight integer-rounding and
rate endpoints before this freeze. At 400 Gbit/s the dispatch service is
13,410,556,120 to 13,410,556,140 ps and combine is 26,006,336,300 to
26,006,336,320 ps. The 200 Gbit/s sensitivity services are exactly twice those
values. Bulk output is retained under `<COMP75_RUN_ROOT>/`, configured to the
required external COMP-75 run root.

Two-batch overlap executes two copies of the same operation sequence with
stage offsets zero and `tbo_delta_stages`, using distinct dispatchers, before
merging the outputs. The step composition expectation is therefore max-like:
`max(candidate compute service, communication service)`, not an additive sum
and not a fitted overlap fraction.

## Frozen comparison expectations

COMP-75 expects to reproduce CORE-60 destination arithmetic, packet services,
and max-like composition. This does not promote or amend the void CORE-60
record. The clean repetition owns a separate record.

For the visible 1K row, the positive exposed service is expected to decrease
throughput relative to candidate-only pricing. Because max-like composition
hides more compute than CORE-59's additive mechanism, it is expected to
increase throughput relative to CORE-59. The sign and magnitude of remaining
error are deliberately not preregistered without reading the visible target.
Decode pricing remains unchanged under SGL-38 ownership.
