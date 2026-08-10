# RNIC device v1 results

All six frozen scalar cells match direct construction exactly, field by field,
and match their closed forms. The directed PCIe-bound and inert-network
comparisons also match exactly. The accepted predecessor artifacts remain
byte identical: all 11 tracked `rnic_wq_v1` rows and all 35 tracked
`rnic_pcie_v1` rows pass their existing `--check` runs through probes that now
construct `RnicDevice`.

The expectations were frozen in commit
`2c7926040d6dd2a3b46af8b0bf41c841dfca8174` before the implementation and
before the first study run.

## Method

The study builds the dependency-free C++17 library with warnings as errors in
an external wave-1 build directory, runs CTest, then invokes the composition
test executable in its behavioral CSV mode:

```bash
.venv/bin/python examples/rnic_device_v1/run_rnic_device_v1.py
```

The paired scalar paths receive the same `WorkQueueConfig`, requests and
caller timestamps. The direct path uses `WorkQueue` plus a reference inert
port. The composed path uses `RnicDevice` plus its owned inert port. The
comparison canonicalizes every public work-queue config field, resolved PCIe
binding field, post result, doorbell result, completion field, WQE request and
timeline field, state, token, completion status, counter, evidence event and
terminal queue state. A mismatch in any field fails the row.

The PCIe-directed comparison additionally checks every field of every
per-class and aggregate `PcieClassAccounting` record plus the fabric
generation. Both fabrics run their invariant checks.

Raw behavioral rows are in [results.csv](results.csv). Native executable
counts are separate in [native_tests.csv](native_tests.csv).

## Parameterized behavioral cohort

All six `B x D` cells pass exact direct-versus-composed equality and the frozen
closed forms.

| Doorbell batch B | Doorbell service D (ps) | Direct JCT (ps) | Composed JCT (ps) | Doorbells | Exact surface equality |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 320 | 320 | 32 | PASS |
| 4 | 0 | 320 | 320 | 8 | PASS |
| 16 | 0 | 320 | 320 | 2 | PASS |
| 1 | 1000 | 32010 | 32010 | 32 | PASS |
| 4 | 1000 | 8040 | 8040 | 8 | PASS |
| 16 | 1000 | 2160 | 2160 | 2 | PASS |

For `D = 0`, every batch size gives `JCT = N * F = 320 ps`. For
`D = 1000 ps`, every row gives
`JCT = (N / B) * D + B * F` exactly. Doorbell count is `N / B`, and all rows
produce 32 CQEs.

The behavioral headline is 6 of 6 exact-equivalence cells. Structural checks
are not added to this denominator.

## Directed exact-equivalence scenarios

The PCIe-bound scenario uses a default fabric modified to MPS and MRRS 128,
fixed host-store, posted-visibility and read-response service of 10, 20 and
30 ps, path delays of 5 and 7 ps, four signaled WQEs, scalar QPC lookup of
11 ps and scheduler service of 13 ps. Direct and composed queue surfaces,
resolved binding, fabric generation, every service-class ledger and aggregate
accounting are exactly equal. Scalar doorbell, WQE-fetch and CQE-write stages
report `not_applicable`; the four corresponding fabric stages report
`applicable`.

The separate inert-network scenario uses three signaled WQEs, batches of two,
doorbell, fetch, QPC, scheduler and CQE scalar services of 37, 11, 5, 7 and
13 ps. The owned inert port matches the direct reference port exactly. It
returns fresh tokens `{1, 2, 3}` on the device progress pump, preserves
device identity, records acceptance and delivery at the caller's timestamp,
and leaves first-packet and last-packet timestamps unset. The absent external
network stage reports `not_applicable`.

These two directed comparisons are reported separately from the six-row
parameterized cohort.

## Structural invariants

The following checks are fatal and unscored:

- DMA-off parameters are inert, and an attached fabric is rejected while DMA
  is off.
- DMA-on rejects scalar doorbell, WQE-fetch and CQE-write service before the
  shared fabric can mutate.
- QPC-off rejects scalar QPC service, reports its stage `not_applicable`, and
  retains the device-level QP number and policy-context token.
- Network-off rejects an external pointer and uses only the owned inert stub;
  network-on requires an injected port.
- Device, identity, QPC, DMA, network, work-queue, binding, fabric and
  analytical-profile version mismatches all hard-fail, including versions in
  disabled modules.
- Two devices with identical SQ and CQ IDs on one shared fabric derive the
  distinct domain pairs `{21, 20}` and `{23, 22}` from namespaces 10 and 11.
  Explicit collision is rejected. An owned fabric retains the accepted
  SQ/CQ-derived domain pair. An isolated shared run equals the owned run in
  every queue and accounting field after masking only the resolved domains.
- A shared fabric remains alive while any device retains it. The composer,
  fabric and bound queue are non-movable, or heap-stable where ownership is
  shared.
- A failed shared-device construction releases its domain claim and leaves
  fabric generation and accounting unchanged, so a corrected construction can
  reuse the namespace.
- Every scalar row has 32 posts, deliveries and reclaims; zero busy attempts,
  rejections, drops, SQ-full events and CQ overruns; and an empty evidence
  ledger.

## Baseline gates

The existing studies were rerun with their tracked byte comparison after both
probes moved behind `RnicDevice`:

```text
rnic_wq_v1: tracked results match 11 measured rows
rnic_wq_v1: checks 11/11 PASS

rnic_pcie_v1: tracked results match 35 measured rows
rnic_pcie_v1: exact-oracle rows 35/35 PASS
rnic_pcie_v1: behavioral relation families 10/10 PASS
rnic_pcie_v1: behavioral predicate instances 18/18 PASS
rnic_pcie_v1: structural invariants PASS, unscored
```

Thus DMA off preserves the accepted scalar artifact, and the fabric probe
preserves the accepted PCIe artifact exactly. These predecessor regression
counts are not added to the six BACK-18 behavioral cells.

## Native component evidence

The clean study build reports all 3 native test executables passing. The full
CTest suite reports 4 of 4 entries because it separately wraps the negative
probe parser check:

```text
1/4 simllm_rnic_pcie_fabric_test passed
2/4 simllm_rnic_work_queue_test passed
3/4 simllm_rnic_device_test passed
4/4 simllm_rnic_wq_probe_rejects_negative_service passed
100% tests passed, 0 tests failed out of 4
```

This is native component evidence. It is not added to the behavioral or
directed-equivalence counts.

## Scope after BACK-18

The composition surface, inert port, external port injection seam, stable
fabric lifetime and cross-module validation are complete. BACK-8 retains run
records, configuration hash, sole-authority projection and bypass equivalence.
HTSIM-9 retains the concrete htsim `NetworkPort` adapter. QPC lifecycle and
host-memory backing remain BACK-11 and BACK-19; submission-source and CQ-owner
selection remain BACK-20.
