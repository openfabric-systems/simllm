# RNIC host-memory v1 expectations

## Freeze status and scope

This is the expectations-only record for BACK-19. It precedes the virtual
host-memory implementation, every new native test, and every result-producing
run in this study. The companion command registry contains only frozen
literals and check logic. It does not include or import the not-yet-written
host-memory API in `--check-only` mode.

This is component scope. It does not claim a composed htsim run, a
`CompletionEvent`, a `StepResult`, TTFT or TPOT reachability result. HTSIM-9 is
the successor that links the native device and its memory-backed transactions
to the packet simulator. The wave-4 compute coupling is the successor that
submits GPU-side producer work through the concurrent compute service. Those
successors must carry the selected path into the live metric chain before this
mechanism can support a final-metric claim.

## External-source audit before freeze

The audit was completed before this freeze against SimLLM base commit
`fc282efc91573638de7dcfae2befee1cf022011b`, rdma-core commit
`c1c5bf1f480312c07ed4d23f0feecf8b5fd73289`, and Linux commit
`db2ddb87143519e20a95aa36c60b36107b736a58`. No implementation or
result-producing command was run before this record.

- rdma-core `providers/mlx5/mlx5.h:665-701` keeps the QP buffer, SQ, RQ and
  doorbell-record pointer as distinct provider objects. Its CQ structure keeps
  the active CQ buffer and doorbell-record pointer separately at
  `providers/mlx5/mlx5.h:496-509`.
- rdma-core `providers/mlx5/qp.c:65-68,167-170` locates receive and send WQEs
  by queue-relative indexed offsets in their allocated buffers. The posting
  path orders descriptor writes before the located doorbell record and then
  rings the UAR or BlueFlame register at
  `providers/mlx5/qp.c:754-786`.
- Linux `include/linux/mlx5/mlx5_ifc.h:3652-3793` defines the QPC separately,
  including queue geometry, page geometry, CQ identities and the doorbell
  record address. QP creation supplies the QPC, WQ UMEM identity and WQ PAS
  page list as separate fields at
  `include/linux/mlx5/mlx5_ifc.h:9471-9496`.
- Linux defines the memory-key context with access, protection-domain, start,
  length, translation size and page geometry at
  `include/linux/mlx5/mlx5_ifc.h:4521-4577`. MKey creation carries that MKC
  plus the KLM, PAS or MTT translation array at
  `include/linux/mlx5/mlx5_ifc.h:9534-9568`.
- The accepted SimLLM transaction ABI already has separate `QpcIcm` and
  `MttMpt` service classes at
  `simllm/backends/rnic/include/simllm/rnic/pcie_fabric.h:23-37`.
  The current work-queue path uses neither class and validates all SQ, CQ and
  doorbell-record paths as host-pinned at
  `simllm/backends/rnic/src/work_queue.cpp:161-191`.

These sources establish separate QPC, queue-page-list, doorbell-location and
MKey translation surfaces. They do not establish cache sizes, service
latencies or a ConnectX-7 timing profile. Exact transaction sizes and the
chosen combined `MttMpt` service representation below are SimLLM construction
choices, so their checks are fatal structural evidence rather than scored
behavioral evidence.

## Tracked allocation contract

An enabled host-memory model owns one live allocation registry and an ordered
lifecycle-event ledger. Every allocation has a stable nonzero identity, a
typed owner, an object kind, a host-pinned or GPU-memory endpoint, a located
virtual address, a byte extent, a PCIe path and explicit page geometry. Page
geometry contains the page size and the ordered physical page list used to
cover the extent. Overlap is checked within an endpoint address space.

The composed one-QP fixture registers exactly these six objects at device
construction:

1. one host-pinned QPC/ICM region owned by the QP;
2. one SQ ring owned by the send queue;
3. one RQ ring owned by the receive queue;
4. one CQ ring owned by the completion queue;
5. one located doorbell record owned by the send queue; and
6. one registered data region owned by its memory key.

Registration is all-or-nothing. Explicit device teardown records exactly one
teardown event for every live object owned by that device, also all-or-nothing.
Duplicate identities, owner mismatch, address overlap, bad page coverage,
wrong endpoint or path kind, missing MKey, duplicate MKey, use after teardown,
and foreign-owner teardown must reject without changing allocations, events,
fabric generation, accounting or caller time. These are fatal unscored
invariants.

The QPC/ICM allocation is always host-pinned and uses device-managed ICM
pages. Queue rings use their creation-time page lists. A data region names one
MKey and resolves through MKey, MPT and MTT semantic stages. The doorbell
record is a located object whose registered address and page offset must agree
with the queue binding.

## Frozen two-axis matrix

Sweep page size `P` in `{4096, 2097152}` bytes and signaled WQE batch size `B`
in `{1, 4}`. This gives four unique run configurations. Each fixture uses two
data pages and accesses the second page, while all queue entries fit in their
registered rings. Every nonzero-payload WQE names the data allocation, MKey,
offset and length explicitly.

For each WQE, the enabled path performs these semantic accesses:

- SQ ring page-list resolution, then `WqeRead`;
- QPC context read through `QpcIcm`, with no MKey, MPT or MTT event;
- data MKey lookup, MPT lookup and MTT resolution, then `PayloadRead`; and
- CQ ring page-list resolution, then `CqeWrite` for the signaled completion.

The two data metadata reads and each queue page-list read use the existing
combined `MttMpt` PCIe service class. Consequently the constructed fixture has
`4 * B` `MttMpt` PCIe transactions, `B` each of `QpcIcm`, `WqeRead`,
`PayloadRead` and `CqeWrite`, plus one doorbell-record store and one UAR write.
These exact counts and the author-defined ordering are fatal structural checks.
They are not scored relations.

## Decision-relevant translation asymmetry

For every `(P, B)` cell, each of the `B` active QPC fetches must return a
`QpcIcm` transaction and exactly zero MKey, MPT or MTT translation events.
The quantitative band is exactly zero QPC-attributed MTT events across all
four cells, with `1 + 4 + 1 + 4 = 10` active QPC fetches observed. A QPC fetch
that consumes even one MTT event fails the family.

This relation decides whether QPC locality may be calibrated independently
from memory-translation locality. Failure collapses the two mechanisms into
one surrogate and blocks the BACK-19 design. It cannot be waived as a timing
calibration difference.

The positive controls require each SQ and CQ ring access to carry its
queue-page-list event and each data access to carry MKey, MPT and MTT events.
They prove that zero QPC translation is not produced by disabling translation
globally. Positive-control counts are fatal and unscored.

## Default byte-identity relation

The host-memory model has an explicit disabled compatibility mode. With that
mode selected, default `RnicDevice` construction and every accepted predecessor
study must preserve exact bytes. The frozen artifact inventory is:

| Artifact | Frozen SHA-256 |
|---|---|
| `examples/rnic_wq_v1/results.csv` | `598f0e10ca4e5a83a9dfb8ed8289e25cdc4c80fc24f92f2f70db967724be5682` |
| `examples/rnic_pcie_v1/results.csv` | `464b92fd5327287db6b5e71a5449add5b893285bc3c4bcdf6a4950355339a5e2` |
| `examples/rnic_device_v1/results.csv` | `7a0b8423d0a99de9538047f307bb7fd2f20c8d19bd408ef90fe02199da868934` |
| `examples/rnic_device_v1/native_tests.csv` | `969963477314bfb723770556a02e4f038c7220820d522ae60dfa8c80744a202d` |
| `examples/rnic_session_records_v1/results.json` | `d83575d1c873d3375bc24819c4d6eca0b85ea3a414fe8578f30262268a39fdf6` |

All five artifacts must remain byte-identical. This is a scored compatibility
family with five instances and an exact zero-byte-difference band. A mismatch
changes the design decision: the new model cannot become an optional composed
module until the identity path is repaired.

Default construction must also create no memory allocation or lifecycle event
and consume no `QpcIcm` or `MttMpt` transaction. Those disabled-path zeros are
fatal unscored guards and do not increase the compatibility denominator.

## Evidence classes and acceptance

The two scored component families remain separate:

- active QPC translation asymmetry, four parameterized cells and ten active
  QPC accesses; and
- default artifact byte identity, five independently hashed artifacts.

Run configurations, exact transaction rows and native executables are reported
separately. Allocation conservation, lifecycle order, positive translation
controls, page selection, path compatibility, rejection atomicity, disabled
paths and invariant validation are fatal unscored evidence. Counts from these
classes are never added into one headline total.

The results must report the genuine-risk fraction separately for both scored
families. QPC accesses are genuine risk because a conventional generic memory
model can route every device read through one MKey/MTT helper. Artifact
identity is genuine risk because adding default fields, transactions or output
columns can perturb each accepted producer independently.

## Registered command and pre-freeze dry run

The local machine configuration must set `SIMLLM_WAVE3_RUN_ROOT` to the
external wave-3 run root. The result-producing command is:

```bash
.venv/bin/python examples/rnic_hostmem_v1/run_study.py \
  --out "$SIMLLM_WAVE3_RUN_ROOT/codex/rnic1920_hostmem_submission/back19"
```

Before this freeze, the same command was executed with `--check-only` appended.
That mode parses the complete CLI, validates the four-cell matrix, source pins,
artifact inventory, exact frozen digests and external-output rule. It prints a
registry confirmation by design. It does not import the future host-memory API,
configure CMake, create the output directory or produce an artifact.
