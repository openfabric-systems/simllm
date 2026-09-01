# Collective completion second-candidate freeze

This fifth freeze follows the interpreted attempt 0005 report and precedes
the second candidate's implementation in the repository. The original 63
training cells, 63 Family H holdouts, bands, D8 coordinate, packet cells and
all earlier immutable files remain unchanged.

## Finding carried from attempt 0005

The paired-operation trend ratio is refuted at 46 of 63. Its 17 misses span
both signs: multiplying by a ratio between two broad trends sometimes
amplifies operation differences and sometimes erases the complementary
operation's local regime. The ratio is the mispriced term. It is removed.

Attempt 0005 also resolves D8 at quotient 0.946736591. The D8 off-grid rule
therefore stays byte-identical: same-operation affine interpolation on the
physical byte axis, with no D8-specific constant.

## Frozen second candidate

The exact same-operation training-anchor rule and off-grid fallback from the
fourth freeze do not change. At an interleaved coordinate where only the
complementary operation has an exact training observation, the second
candidate uses these rules in order:

1. Compute the same-operation affine value from its adjacent training
   anchors. Take its arithmetic mean with the complementary operation's exact
   same-rank, same-byte training anchor. This treats common ring traffic and
   operation-specific work symmetrically.
2. If the complementary exact anchor is lower than both of its own adjacent
   same-operation training anchors, keep that trough unchanged instead of
   averaging it away. The already published A100 and GH200 evidence shows
   reproducible protocol-transition bandwidth troughs, not isolated
   bandwidth spikes.
3. For reduce-scatter at no more than 8 KiB, take the larger of the symmetric
   value, the same-operation affine value and the complementary all-gather
   anchor. At floor-dominated sizes, reduction work cannot make
   reduce-scatter complete faster than the same-byte gather transfer floor.
4. Apply the independent Hopper transition prior only to all-gather at the
   declared transition coordinates. At rank 8 and 256 KiB, divide the complete
   symmetric value by 0.78 because the observed independent width-2 transition
   is a complete local-regime disturbance at this floor-dominated size. At
   4 MiB for ranks 2, 4 and 8, divide only the value above that curve's minimum
   training latency by 0.78. The physical ring endpoint coordinate is 2 MiB
   through 3.5 MiB across those ranks, matching the independently observed
   2 MiB through 4 MiB transition neighborhood. The minimum training latency
   remains the opaque floor and is not scaled.

Every value and decision except the frozen 0.78 factor comes from the original
training cells. The factor is exactly the already published 22 percent GH200
width-2 bandwidth trough. No Family H latency sets a coefficient, branch
threshold or floor. The model remains an opaque fully local completion charged
once, with no separately reported serialization service.

## Acceptance and guards

Family H is evaluated once over the unchanged 63 cells. All 63 must meet the
larger of 10 percent or two H200 GPU cycles. The report publishes every
attempt-0004, attempt-0005 and second-candidate error, plus median, p95 and
maximum. Any miss refutes the candidate.

The serialized authority is hashed before Family H is loaded. Every served
estimate records the rule and all training anchors. The compatibility
floor-plus-slope authority, MiniMax donor queries and default-off golden remain
byte-identical. The opaque authority is legal only for a fully local
collective and is charged exactly once. All fatal guards from the fourth
freeze remain binding.

Physical sanity remains ring endpoint bytes divided by 450 GB/s as the floor,
with no finite algorithm-progress ceiling supplied by the source.
