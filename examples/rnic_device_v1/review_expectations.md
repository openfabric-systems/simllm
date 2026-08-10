# RNIC device v1 review correction expectations

This expectations-only change freezes the adversarial-review corrections
before their implementation and before any corrective run. It contains no
measured value, generated row or implementation. The original study
expectations remain byte-identical to their first committed form.

## Shared-fabric configuration authority

An `RnicDevice` attached to an external shared `PcieFabric` must require its
embedded `RnicDmaConfig.fabric` to equal the attached fabric configuration
field by field. Equality includes every version, scalar, credit limit,
latency sample, analytical-delay profile and ordered path entry. Any mismatch
must reject construction before an ordering-domain claim, work-queue
construction or fabric mutation. A matching non-default configuration must
construct successfully, and `device.config().dma.fabric` must describe the
effective attached fabric exactly.

## Ordering-domain claim enforcement

Each shared-fabric device owns the submission and completion domains it
claims during construction. A request submitted through that device may use
either of its own claimed domains or an unclaimed domain. A request using a
domain claimed by another live device must be rejected before caller-clock or
fabric mutation. The rejection must leave generation, transaction IDs,
accounting and visibility horizons unchanged. An otherwise identical request
using the submitting device's own claimed domain must be accepted.

Claim release remains tied to the owning device lifetime. A device must not
release another device's claim.

## Validation and clock atomicity

With DMA enabled, each scalar service field is checked independently.
Nonzero `doorbell_service_ps`, `wqe_fetch_service_ps` and
`cqe_write_service_ps` must each reject construction before queue or fabric
state changes.

A failed `submitPcie` validation must not advance the caller-driven device
clock. In particular, an invalid request at a later timestamp followed by a
valid request at an earlier timestamp must be accepted when no earlier
successful device call established the later time. The fabric's existing
two-phase operation atomicity remains unchanged.

## Namespace field applicability

`shared_ordering_domain_namespace` has an explicit three-way contract. It is
inert when DMA is disabled. DMA with an owned fabric requires zero. DMA with a
shared fabric uses a nonzero value only to derive missing submission and
completion domains; an explicit domain pair requires zero. Invalid enabled
combinations reject rather than being silently reinterpreted.

## Regression and evidence expectations

The accepted scalar, PCIe-bound and inert-network behavioral artifacts must
remain byte-for-byte identical. The scalar sweep still contains the same six
exact-equivalence cells and the directed scenarios remain separate evidence
classes.

The study runner must discover native test executables from the generated
CTest metadata and list their names in `native_tests.csv`. No source constant
may state the executable count. CTest entry totals and native executable
totals remain component evidence and never enter a behavioral pass count.
