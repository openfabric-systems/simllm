# RNIC device v1 results

All six frozen scalar cells match direct construction exactly, field by field,
and match their closed forms. The directed PCIe-bound and inert-network
comparisons also match exactly. The accepted predecessor artifacts remain
byte identical: all 11 tracked `rnic_wq_v1` rows and all 35 tracked
`rnic_pcie_v1` rows pass their existing `--check` runs through probes that now
construct `RnicDevice`.

The initial expectation freeze is commit-granular. Commit
`2c7926040d6dd2a3b46af8b0bf41c841dfca8174` precedes implementation commit
`d1ed7dba23f3cd0b94b9157bd071f11be1213d91` by 21 minutes and 33 seconds in
Git history, but that interval does not establish that the 2,470-line landing
had not already been written in the working tree. The initial BACK-18
assertions are therefore post-specified regression checks, not a public
pre-registration claim.

The mitigating facts are that the frozen scalar values are analytic closed
forms, [expectations.md](expectations.md) is byte-identical to its first
committed form, the expectation commit contains no measured results, and the
integrator independently reproduced both exact baselines and the native
suite. The adversarial-review corrections were separately frozen in
expectations-only commit
`110e503491f0aee19b13b9b5893bf1ac4099d026` before their corrective
implementation and runs; see
[review_expectations.md](review_expectations.md).

## Method

The study builds the dependency-free C++17 library with warnings as errors in
an external build directory, runs CTest, then invokes the composition test
executable in its behavioral CSV mode. The build directory comes from
`SIMLLM_RNIC_DEVICE_BUILD_DIR` when set. Otherwise it is a portable external
temporary path keyed by the resolved study output location:

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
names and counts are separate in [native_tests.csv](native_tests.csv). The
runner obtains those names from generated CTest JSON metadata and derives the
count from the list, so no source constant can drift from the build.

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
- DMA-on separately rejects nonzero scalar doorbell, WQE-fetch and CQE-write
  service before the shared fabric can mutate. All three constructor branches
  are exercised independently.
- QPC-off rejects scalar QPC service, reports its stage `not_applicable`, and
  retains the device-level QP number and policy-context token.
- Network-off rejects an external pointer and uses only the owned inert stub;
  network-on requires an injected port.
- Device, identity, QPC, DMA, network, work-queue, binding, fabric and
  analytical-profile version mismatches all hard-fail, including versions in
  disabled modules.
- Two devices with identical SQ and CQ IDs on one shared fabric derive the
  distinct domain pairs `{21, 20}` and `{23, 22}` from namespaces 10 and 11.
  Explicit collision is rejected. Device submissions using either of the
  other device's claimed domains are also rejected without fabric or clock
  mutation, while both of the submitting device's own claimed domains are
  accepted. An owned fabric retains the accepted SQ/CQ-derived domain pair.
  An isolated shared run equals the owned run in every queue and accounting
  field after masking only the resolved domains.
- A shared attachment requires exact equality between its effective fabric
  config and the embedded device config. The adversarial `lane_count = 0`
  plus empty-path config is rejected without mutation, while a matching
  non-default config constructs and remains visible through `device.config()`.
- A fabric-invalid request at 100 ps does not ratchet the device clock: a
  valid request at 50 ps immediately afterward is accepted and committed.
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

The clean study build reports all 3 native test executables passing. Their
names are discovered from CTest metadata and recorded individually, not
represented by a hardcoded count. The full CTest suite reports 4 of 4 entries
because it separately wraps the negative probe parser check:

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
fabric lifetime and cross-module validation are complete. The BACK-8 component
record remainder is reported separately in
[`rnic_session_records_v1`](../rnic_session_records_v1/RESULTS.md). BACK-8 now
retains only the frozen live-reachability gate. HTSIM-9 retains the concrete
htsim `NetworkPort` adapter. QPC lifecycle and host-memory backing remain
BACK-11 and BACK-19; submission-source and CQ-owner selection remain BACK-20.

## BACK-24 transactional rejection correction

Expectations-only commit
`b81ceed2ecd9e6dd6acb0f44c5c4040dde9b46ec` precedes the corrective source
change and every result-producing run. Its registered `--check-only` command
passed without creating the output directory. At freeze time the working tree
contained no BACK-24 implementation or portability-fix files; the pre-existing
untracked reviewer build directory was disclosed and left untouched. See
[`back24_expectations.md`](back24_expectations.md).

The pre-fix diagnostic produced all six frozen rows. In every row the exact
exception identity, WQE records, counters, evidence, complete fake-port ledger
and public physical state remained equal, but the future terminal ratcheted
the device caller clock. The `progress(10)` probe and valid `20 ps`
continuation then failed in all six rows. This isolates the defect from the
WorkQueue retirement transaction.

The corrected `RnicDevice` first validates monotonic caller time, delegates
the complete WorkQueue terminal transaction, and commits its caller time only
after that transaction returns. The historical registered study used the same
executable basename, script, options and pinned inputs; resolved machine-local
paths are intentionally omitted. The
following is a portable post-run rendering, not a verbatim transcript. Source
the local configuration first:

```bash
.venv/bin/python examples/rnic_device_v1/run_back24_study.py \
  --out "${SIMLLM_DATA_ROOT:?configure SIMLLM_DATA_ROOT}/rnic_session_records_v1/back24"
```

It passed 4 of 4 Release CTest entries and wrote external result record
`back24_results.json`, SHA-256
`9ae8dfbd0184fc86fde6632c0385dad68e5bd2522a80a7b8fa8839540891f7e0`.
All 6 of 6 scored paired-control continuations passed. Unknown, duplicate and
cross-WQE terminals were each exercised at `110 ps` and `1010 ps` after an
accepted `10 ps` boundary. The valid continuation at `20 ps` matched its
control exactly, with native terminal and CQE-visible timestamp deltas both
`0 ps` in the frozen `[0 ps, 0 ps]` band.

The fatal unscored guards passed in every cell: exact dynamic exception type
and message, inert pre-probe, immediate WQE-record equality, counter equality,
evidence equality, complete port-ledger equality, physical-state equality,
PCIe-state applicability, inert post-probe and invariant validation. These
guards do not increase the six-relation scored denominator.

The portability correction was exercised with the working directory set to
the operating system temporary directory, whose resolved historical path is intentionally
omitted, with no `--build-dir` and with
`SIMLLM_RNIC_DEVICE_BUILD_DIR` explicitly unset. The temporary root was
placed under the historical wave-2 output directory, whose resolved historical path is
also intentionally omitted; the runner derived its own location-keyed cache
below that root. All 4 CTest entries passed, and the
tracked files retained exact SHA-256 values
`7a0b8423d0a99de9538047f307bb7fd2f20c8d19bd408ef90fe02199da868934`
for `results.csv` and
`969963477314bfb723770556a02e4f038c7220820d522ae60dfa8c80744a202d`
for `native_tests.csv`. Focused unit checks separately prove exact environment
override use, stable fallback derivation and distinct cache keys for distinct
resolved study locations.

### Genuine-risk fraction

The post-rejection clock-continuity family has 6 of 6 plausible-failure
relations, or 100 percent. A competent implementation can preserve every
WorkQueue mutation guard while still committing the enclosing device clock
before delegation, which is the reproduced defect. Exception, snapshot,
artifact and portability guards are fatal and unscored, so they are excluded
from this fraction.

This remains component evidence. HTSIM-9 provides live external-port delivery,
while CORE-4 and CORE-5 join completion into `CompletionEvent`, `StepResult`,
TTFT and TPOT. No htsim submodule, live-runtime path, acceptance harness,
`README.md`, `docs/README_PRO.md` or frozen `rnic_live_v1` expectation was
changed. BACK-24 is closed; no new task ID was needed.
