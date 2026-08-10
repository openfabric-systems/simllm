# RNIC submission-source v1 expectations

## Freeze status and scope

This is the expectations-only record for BACK-20. It precedes the submission-
source implementation, every new native test, and every result-producing run
in this study. The companion command registry contains only frozen literals,
build orchestration and validation logic. Its `--check-only` path does not
include or import the not-yet-written submission API.

This is component scope. It does not claim a composed htsim run, a
`CompletionEvent`, a `StepResult`, TTFT or TPOT reachability result. HTSIM-9
is the successor that connects the selected producer and CQ owner to packet
transport. The wave-4 compute coupling is the successor that submits the
GPU-side producer through the concurrent compute service. Those successors
must carry the selected shape into the live metric chain before this mechanism
can support a signed final-metric relation.

## External-source audit before freeze

The audit was completed before this freeze against SimLLM commit
`dba467984b9d82ba374dce5d64d687ca59074135` and the official NVIDIA NCCL
repository at commit `5067397c2676d5aed50042fc39e5c8ee96eb0027`. No
submission-source implementation or result-producing command was run before
this record.

- NCCL `src/transport/net.cc:1304-1363` implements CPU send-proxy progress:
  the proxy posts GPU-visible slots, publishes the head and waits for the GPU
  to advance data readiness. It calls the network send interface and later
  tests completion at `src/transport/net.cc:1406-1449`.
- NCCL `src/device/prims_simple.h:101-171` shows GPU threads waiting on the
  connection step, selecting the data buffer, publishing FIFO size and
  advancing the system-visible step after a system-scope fence.
- NCCL `src/transport/net.cc:505-510,939-1007` maps head and tail control
  records between host and GPU views, allocates host-mapped control memory,
  and registers either CUDA or host data buffers with the network provider.
- NCCL GDAKI constructs a GPU verbs QP with an explicit SQ depth, GPU device
  and doorbell-record mode at
  `src/transport/net_ib/gdaki/gin_host_gdaki.cc:579-654`. It exports QPs,
  counters and signals into a GPU context, copies that context to device
  memory, and marks the GDAKI backend at
  `src/transport/net_ib/gdaki/gin_host_gdaki.cc:735-794`.
- NCCL's bundled device-verbs layer locates and reserves GPU WQE slots at
  `src/transport/net_ib/gdaki/doca-gpunetio/include/device/`
  `doca_gpunetio_dev_verbs_qp.cuh:49-103`, then updates the producer index,
  doorbell record and GPU-visible doorbell at the same file's lines 370-470.
  Device-side CQ polling and consumer-index advancement are at
  `src/transport/net_ib/gdaki/doca-gpunetio/include/device/`
  `doca_gpunetio_dev_verbs_cq.cuh:150-249,340-380`.
- The accepted SimLLM work queue currently assigns `config_.qpn` to every
  PCIe transaction requester and requires SQ, CQ and doorbell paths to be
  host-pinned at `simllm/backends/rnic/src/work_queue.cpp:179-208,318-500,`
  `1148-1178`.

These sources establish the three ownership and placement shapes. They do not
establish CPU-proxy service time, GPU-kernel duration, callback cost, cache
behavior or a calibrated ConnectX-7 timing profile. This component therefore
records who owns each action and where each object resides, without inventing
timing constants. Timing and compute-task coupling remain successor work.

## Producer, requester and consumer contract

Each composed queue selects exactly one producer shape. A producer names its
kind and stable nonzero agent identity independently from the nonzero QP
number. Device-initiated PCIe reads and writes name a separate RNIC requester
identity. A CQ names exactly one owning consumer with a stable nonzero agent
identity. These identities are attribution dimensions; none becomes a second
WQE or CQ lifecycle authority.

Posting, doorbell publication and CQ consumption append read-only records to
the lifecycle owned by the existing `WorkQueue`. One submission record joins
one WQE, queue, QP, producer and doorbell batch. One CQ-consumption record
joins one consumed CQE, queue, owner and poll timestamp. Failed posting,
doorbell, progress or polling must append no partial record and must not
change queue, host-memory or fabric state. Record sequences are contiguous
within their own ledgers. These conservation and ordering rules are fatal
unscored invariants.

QPC is never producer memory. It remains a host-pinned, device-managed ICM
allocation and every QPC fetch uses the direct `QpcIcm` class in all shapes.
SQ, CQ and doorbell allocation endpoints must agree with the selected shape.
The UAR remains an MMIO BAR; its mapping owner changes from host to GPU for
GPU-initiated submission. Data placement is explicit and may differ from ring
placement.

## Frozen two-axis matrix

Sweep producer shape `S` in `{host_cpu_driver, cpu_proxy, gpu_initiated}` and
signaled WQE batch size `B` in `{1, 4}`. This gives six unique run
configurations and fifteen active QPC fetches. Every WQE has a registered data
region, one submission record and one CQ-consumption record.

The exact fixture shapes are:

| Shape | WQE producer | Upstream descriptor queue | SQ/CQ/DB | Data | UAR owner | CQ consumer |
|---|---|---|---|---|---|---|
| `host_cpu_driver` | host CPU agent 7101 | none | host-pinned | host-pinned | host CPU | host CPU agent 8101 |
| `cpu_proxy` | CPU proxy agent 7102 | GPU agent 7202 writing host-visible memory | host-pinned | GPU memory | host CPU | CPU proxy agent 8102 |
| `gpu_initiated` | GPU agent 7103 | none, WQEs are written directly | GPU memory | GPU memory | GPU | GPU agent 8103 |

The RNIC requester is agent 9100 and the QP number is 19 in every cell. Thus
producer, consumer, requester and QP identity cannot be inferred from one
another. Each cell must complete exactly `B` WQEs, emit exactly `B`
submission records and consume exactly `B` CQEs through its sole named
consumer. The CPU-proxy upstream descriptor fact is present only in the
CPU-proxy cells. These exact values and all author-defined sequence checks are
fatal structural evidence, not scored relations.

## Decision-relevant translation asymmetry

For every `(S, B)` cell, each of the `B` QPC fetches must return one `QpcIcm`
transaction and exactly zero QPC-attributed MKey, MPT or MTT translation
events. The quantitative band is exactly zero QPC-attributed MTT events over
all six cells, with `3 * (1 + 4) = 15` active QPC fetches observed. A QPC
fetch that consumes one translation event fails the family.

This is a live-runtime scored relation. It decides whether moving queue and
data objects to GPU memory accidentally moves the QPC onto the generic memory
translation path. Failure blocks the producer-shape design. The positive
control requires every data access to carry MKey, MPT and MTT events, proving
that zero QPC translation was not obtained by disabling translation globally.
Positive-control and exact transaction counts are fatal and unscored.

## Default-shape byte-identity relation

`host_cpu_driver` is the default producer shape. Leaving the new composition
fields at their defaults must preserve every accepted predecessor artifact
byte for byte, including the BACK-19 host-memory study. The frozen inventory
is:

| Artifact | Frozen SHA-256 |
|---|---|
| `examples/rnic_wq_v1/results.csv` | `598f0e10ca4e5a83a9dfb8ed8289e25cdc4c80fc24f92f2f70db967724be5682` |
| `examples/rnic_pcie_v1/results.csv` | `464b92fd5327287db6b5e71a5449add5b893285bc3c4bcdf6a4950355339a5e2` |
| `examples/rnic_device_v1/results.csv` | `7a0b8423d0a99de9538047f307bb7fd2f20c8d19bd408ef90fe02199da868934` |
| `examples/rnic_device_v1/native_tests.csv` | `969963477314bfb723770556a02e4f038c7220820d522ae60dfa8c80744a202d` |
| `examples/rnic_session_records_v1/results.json` | `d83575d1c873d3375bc24819c4d6eca0b85ea3a414fe8578f30262268a39fdf6` |
| `examples/rnic_hostmem_v1/results.csv` | `1bc7bcc8e72b7aef9fda1ed7e6ca2078d60c48a00377cbf8dfded75ff4d2fa53` |

All six artifacts must remain byte-identical. This is a scored compatibility
family with six instances and an exact zero-byte-difference band. A mismatch
changes the design decision: the default producer cannot be accepted until
the identity path is repaired.

DMA-off construction keeps the producer fields inert and retains effective-
hardware v1 bytes. Host-memory configurations use a new strict schema only
when a submission configuration must be represented. The disabled and
default-path guards are fatal unscored evidence and do not increase the
compatibility denominator.

## Evidence classes and acceptance

The two scored component families remain separate:

- active QPC translation asymmetry, six parameterized cells and fifteen
  active QPC accesses; and
- default producer artifact byte identity, six independently hashed
  artifacts.

Run configurations, exact structural rows and native executables are reported
separately. Endpoint agreement, sole CQ ownership, attribution identity,
ledger conservation, sequence order, positive translation controls, failure
atomicity, strict-schema rejection and disabled paths are fatal unscored
evidence. Counts from these classes are never added into one headline total.

The results must report the genuine-risk fraction separately for both scored
families. QPC cells are genuine risk because the GPU-memory relaxation can
accidentally route every object through MKey/MTT. Artifact identities are
genuine risk because new default fields, records or schema bytes can perturb
each accepted producer independently.

## Registered command and pre-freeze dry run

The local machine configuration must set `SIMLLM_WAVE3_RUN_ROOT` to the
external wave-3 run root. The result-producing command is:

```bash
.venv/bin/python examples/rnic_submission_v1/run_study.py \
  --out "$SIMLLM_WAVE3_RUN_ROOT/codex/rnic1920_hostmem_submission/back20"
```

Before this freeze, the same command was executed with `--check-only`
appended. That mode parses the complete CLI, validates the six-cell matrix,
source pins, artifact inventory, exact frozen digests and external-output
rule. It prints a registry confirmation by design. It does not import the
future submission API, configure CMake, create the output directory or
produce an artifact.
