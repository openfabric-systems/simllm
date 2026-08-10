# RNIC device v1 expectations

This expectations-only change precedes the BACK-18 implementation and every
run of this study. The results report must cite the commit that first contains
this file. No measured value, generated row or implementation is part of this
freeze.

## Scope and evidence boundary

The study validates one versioned native RNIC composition entry point. The
entry point owns one work queue and composes it with optional QPC, DMA and
network modules. It may heap-own a PCIe fabric or retain an explicitly shared
external fabric. It may use an injected `NetworkPort` or an owned inert port.
The caller remains the only clock authority.

The behavioral evidence is exact direct-versus-composed equivalence. A direct
construction uses the accepted `WorkQueue`, `PcieFabric` and a reference inert
port without the new composer. The paired composed construction uses the same
sub-configurations, requests and caller timestamps through `RnicDevice`.
Equality means equality of every public field, optional state, timestamp,
completion entry, WQE record, counter, evidence event and relevant PCIe
accounting field. Pointer values and object addresses are outside the
comparison.

Run rows, exact-equivalence predicates, structural invariants and native test
executables are separate evidence classes. Configuration rejection,
version rejection, module applicability, conservation, quiescence and
non-collision checks are fatal structural invariants, but they do not increase
the behavioral pass denominator. Native executable counts are component
evidence and are never added to the behavioral row count.

## Scalar sweep

Post `N = 32` signaled WQEs at time zero into an SQ and CQ of depth 64. The
reference inert network accepts every WQE with a fresh nonzero token and
delivers it from the caller-driven progress pump without extra latency. Use a
WQE-fetch service time `F = 10 ps`; QPC, scheduler and CQE-write scalar service
times are zero. Sweep doorbell batch size `B` in `{1, 4, 16}` and doorbell
service time `D` in `{0, 1000} ps`. All doorbells are called at time zero, and
the event loop delivers same-timestamp network outcomes before device
progress, then polls the CQ after progress.

For every one of the six cells, direct and composed results must be identical
field by field. Both paths must also match these frozen scalar relations:

```text
doorbells = N / B
CQEs = N
JCT(D = 0) = N * F
JCT(D = 1000) = (N / B) * D + B * F
```

| B | Expected JCT at D = 0 ps | Expected JCT at D = 1000 ps |
|---:|---:|---:|
| 1 | 320 | 32010 |
| 4 | 320 | 8040 |
| 16 | 320 | 2160 |

All six cells must have 32 accepted, delivered and reclaimed WQEs; zero
network-busy attempts, drops, rejections, SQ-full rejections and CQ overruns;
and an empty controlled-evidence ledger. These zero and conservation checks
are structural invariants, not extra behavioral predicates.

The six exact-equivalence cells are the only parameterized behavioral cohort.
No total may combine them with the directed scenarios or native executables.

## PCIe-bound equivalence

Drive the same finite signaled-WQE scenario once through a direct
`WorkQueue` bound to a default `PcieFabric` and once through an otherwise
identical composed device with DMA enabled. Doorbell, WQE-fetch and CQE-write
scalar service times are zero in both paths. Compare every queue result and
the complete per-class and aggregate fabric accounting exactly. The composed
timeline's doorbell observation, fetch begin/end and CQE visibility must equal
the corresponding fabric result exactly. No independent scalar cursor may add
delay to a fabric-charged stage.

The direct and composed fabrics must finish at the same generation and pass
their invariant checks. A failed device construction or failed operation must
leave a shared external fabric's generation, transaction IDs and accounting
unchanged, preserving the existing two-phase plan/commit guarantee.

This is one directed exact-equivalence scenario. It is reported separately
from the six scalar sweep cells.

## Inert-network equivalence

With no external network module, the composed device owns an inert port. Each
accepted descriptor receives a fresh, nonzero token. Its delivery is returned
only by the device progress pump, using the caller's timestamp and the
documented same-timestamp ordering. Compare this path exactly with a direct
`WorkQueue` driven by the reference inert port and the same pump order.

The descriptor must retain WQE, WR, flow, tag, source, destination, QP number
and policy-context identity. The device-level QP number and policy-context
token remain unchanged when the QPC module is disabled. No packet-issue
timestamp is fabricated by either inert port.

This is one directed exact-equivalence scenario. It is reported separately
from the scalar cohort and the PCIe-bound scenario.

## Disabled modules and version closure

Directed construction checks freeze these outcomes:

1. DMA disabled uses the scalar work-queue services and produces the accepted
   scalar result exactly. DMA-only parameters do not alter that result. Passing
   an external fabric while DMA is disabled is rejected.
2. DMA enabled rejects any nonzero scalar doorbell, WQE-fetch or CQE-write
   service before queue or fabric state can mutate.
3. QPC disabled rejects a nonzero QPC service rather than charging a hidden
   compatibility stage. Device QP and policy-context identity remain valid and
   visible.
4. Network disabled rejects an injected external port. Network enabled
   requires one injected port. The disabled path uses only the owned inert
   stub.
5. The device config, every module config, `WorkQueueConfig`,
   `WorkQueuePcieBinding`, `PcieFabricConfig`, network descriptors and network
   events retain their own version fields. Every supported version reaches the
   owning constructor unchanged. Any mismatched version hard-fails, including
   a mismatched sub-config belonging to a disabled module.
6. A disabled module reports its module-specific stages as not applicable. It
   never converts the module's configured delay into another scalar stage or
   silently transfers scheduling authority.

These are fatal structural checks and do not count as behavioral sweep cells.

## Shared-fabric ordering domains and lifetime

Construct two devices with identical default SQ and CQ IDs on one explicitly
shared external fabric. Give the devices distinct nonzero device namespaces.
The composer must derive distinct nonzero submission domains and distinct
nonzero completion domains, while each device remains deterministic. Repeating
either device alone against a fresh fabric must reproduce its derived domains.

A single device with an owned fabric and no explicit namespace must retain the
accepted direct-construction domains derived from its SQ and CQ IDs. Therefore
namespacing changes ordering-domain identity only in the shared-fabric case;
it must not change request fields, byte counts, service classes, transaction
order or any timestamp when the two devices do not contend. Explicit caller
domains pass through unchanged and a colliding shared-fabric domain assignment
is rejected rather than silently shared.

The external fabric is retained by shared ownership for the full lifetime of
every bound queue. An owned fabric has a stable heap address and is destroyed
after its queue. No composer move may invalidate the address observed by the
bound `WorkQueue`.

These domain, address-lifetime and rejection checks are structural invariants,
not behavioral sweep cells.
