# LL128 shared-memory staging correction: expectations

Source review after the first component run found that `storeRegs` in pinned
`src/device/prims_ll128.h` writes every vector that is not a complete aligned
output vector into shared memory. This includes inactive vectors beyond the
useful tail, not only one partial vector. That earlier accounting is refuted.
Retain the first run as void for source-work qualification. This amendment
precedes the correction and its new run; it is not prior registration of the
already observed defect. The original source and hardware freezes remain
unchanged.

For an aligned float32 warp stripe with positive useful bytes `u <= 1920`,
expect exactly `1920 - 16 * floor(u / 16)` shared-memory store bytes and
`u % 16` shared-memory tail-read bytes. Global output bytes remain exactly
`u`; unused staged vectors do not become output or network payload. The
source warp synchronization occurs even when the shared-memory byte count
is zero. Test `u = {4, 12, 16, 20, 120, 1024, 1916, 1920}` independently.

Give shared memory an explicit positive per-SM service rate, separate from
the per-GPU global-memory rate. The declared fixture uses 1 TB/s per SM,
without claiming that this is a hardware calibration. For isolated shared
requests vary byte counts `{16, 128, 1920}` and service rates `{0.5, 1, 2}`
times that baseline. Duration is `ceil(bytes * 10^12 / rate)` picoseconds;
two resident blocks on the same SM share capacity, and blocks on separate
SMs use independent capacity. Global-memory byte counts do not change when
only the shared-memory rate changes. Guards and byte identities are unscored.

The fixture also records exact original-graph token metrics for two/four
ranks, all three protocols, two available-SM counts and three link rates.
Fixed compute remains independent of either intervention. Time to first
token and time per output token each equal fixed serial compute plus the two
executed collective durations. A separate isolated-link fixture checks the
original serialization relation. These are model checks, not hardware fits.

Per-channel physical peer permutations and calibrated instruction, cache,
barrier and polling costs remain explicit TRAF-54/TRAF-43 obligations. No
hardware control contrast or accuracy threshold changes in this amendment.
