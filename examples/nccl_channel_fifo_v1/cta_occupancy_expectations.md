# Work warps and resident block warps: source correction

The pinned host launcher sets `threadPerBlock` to at least
`NCCL_MIN_NTHREADS`, defined as four warps in `src/include/device.h`.
An individual work descriptor can use three warps after tuning. Its three
working warps do not reduce the allocated block below four warps.

Before changing this path, freeze the following relation: for an isolated
collective, residency uses `max(4, work_warps)`, while primitive barriers and
source work keep their selected work-warp count. A three-warp LL descriptor
must reject a declared SM that has capacity for only three resident warps,
before admission; the same descriptor fits a four-warp SM and its residency
trace records four allocated warps. Larger work descriptors retain their
existing timestamps and resource footprint exactly.

This is a source-derived guard for the executable isolated-collective scope.
Grouped launches can allocate more threads to accommodate other work in the
same kernel and remain outside this first source slice. Hardware register
and shared-memory footprints still need independent observation under
TRAF-54. The existing study fixtures use at least four work warps and are
unaffected by this correction.
