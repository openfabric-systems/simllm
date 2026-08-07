# RNIC PCIe v1 results

All 35 pre-registered sweep rows match the independent byte, rational-time and
integer analytical-sampler oracles exactly. All 18 cross-checks pass,
and the native CTest suite passes the PCIe fabric, WorkQueue integration and
negative-input checks.

## Method

The study builds the dependency-free C++17 RNIC library in Release mode, runs
CTest, then drives `simllm_rnic_pcie_probe` over the parameter grid frozen in
[expectations.md](expectations.md). The Python runner computes its expected
TLP counts, directional bytes and serializer times independently. It does not
reuse C++ model results as its oracle.

Reproduce and compare every byte of the tracked CSV from the repository root:

```bash
python3 examples/rnic_pcie_v1/run_rnic_pcie_v1.py --check
```

Raw probe knobs, effective configuration, requested and accounted transaction
counts, analytical-profile parameters, measurements and expected values are in
[results.csv](results.csv).

## MPS and MRRS byte conservation

All nine 512-byte DMA-read cells preserve exactly 512 useful and transferred
bytes. The measured total modeled-link bytes are:

| MRRS \ MPS | 128 | 256 | 512 |
|---|---:|---:|---:|
| 128 | 688 | 688 | 688 |
| 256 | 640 | 600 | 600 |
| 512 | 616 | 576 | 556 |

The request direction contains MRd headers and no payload. The opposite
direction contains all 512 payload bytes plus CplD overhead. Every directional
row satisfies `modeled-link = payload + overhead` exactly, and directional TLP
counts sum to the independent request-plus-completion count.

## Generation, width and MPS serialization

The 4096-byte posted-write cells match continuous rational serialization with
zero residual. Larger MPS reduces headers from 32 to 8 and lowers modeled-link
traffic from 4864 to 4288 bytes.

| MPS (B) | Modeled-link bytes | Gen5 x16 (ps) | Gen5 x8 / Gen4 x16 (ps) | Gen4 x8 (ps) |
|---:|---:|---:|---:|---:|
| 128 | 4864 | 77188 | 154375 | 308750 |
| 256 | 4480 | 71094 | 142188 | 284375 |
| 512 | 4288 | 68047 | 136094 | 272188 |

Gen5 x8 equals Gen4 x16 in every row. The x8 values are exactly twice the x16
continuous result, subject only to the final integer-picosecond ceiling.

## Outstanding-read queue

Sixteen 512-byte DMA reads were submitted at time zero with a fixed 1,000,000
ps response delay. Increasing the read window from one to four reduces both
JCT and accumulated outstanding-slot wait exactly as predicted.

| Read slots | MPS (B) | JCT (ps) | Outstanding wait (ps) |
|---:|---:|---:|---:|
| 1 | 128 | 16156416 | 15140925 |
| 1 | 512 | 16141184 | 15126645 |
| 4 | 128 | 4067289 | 3051798 |
| 4 | 512 | 4060623 | 3046084 |

At each window size, MPS 512 is slightly faster because it emits fewer CplD
headers. The four-slot case is about one quarter of the one-slot JCT, with the
remaining difference explained exactly by request and completion
serialization.

## Path attribution

The local 512-byte read completes in 8824 ps. The labeled path penalties are
additive, remain in their own accounting fields and do not change modeled-link
bytes.

| Profile | NUMA attributed (ps) | IOMMU attributed (ps) | JCT (ps) |
|---|---:|---:|---:|
| local | 0 | 0 | 8824 |
| remote NUMA | 100000 | 0 | 108824 |
| IOMMU | 0 | 200000 | 208824 |
| remote NUMA plus IOMMU | 100000 | 200000 | 308824 |

These are explicit fixed analytical profiles, not a derived PCIe topology or
a measured ConnectX-7 profile.

## Analytical penalty profiles

Each row below contains 4,096 independent 8-byte host stores at time zero.
Only the NUMA profile is active, so completion time is the realized profile
sample and no PCIe link bytes are emitted. Every aggregate, incidence decision
and tail selection matches the independent integer-only Python replay exactly.

| Profile | Realized delay sum (ps) | Occurrences | Tail draws | Min (ps) | Max/JCT (ps) |
|---|---:|---:|---:|---:|---:|
| fixed | 409600000 | 4096 | 0 | 100000 | 100000 |
| Gaussian, sigma 10000 ps | 408738343 | 4096 | 0 | 75824 | 124176 |
| Gaussian, sigma 40000 ps | 406153332 | 4096 | 0 | 3298 | 196702 |
| mixture, 1% tail | 423145553 | 4096 | 36 | 75824 | 588084 |
| mixture, 10% tail | 569124409 | 4096 | 395 | 75824 | 620878 |
| Gaussian, 25% incidence | 100506113 | 1003 | 0 | 0 | 124176 |

The wider Gaussian expands the observed range from 48,352 to 193,404 ps,
exactly fourfold for the frozen quantile stream. Raising tail probability
increases selections from 36 to 395 and increases aggregate delay. The 25
percent incidence profile applies 1,003 times and returns zero for every
absent event. These are deterministic finite discrete distributions. The
two-Gaussian mixture is a rare-tail surrogate, not a mathematically
long-tailed law. Incidence is analytical and independent; it does not claim
to detect an IOTLB, ATC or DDIO-cache transition.

## Directed boundaries and atomicity

The native harness additionally verifies DWORD and 4 KiB splitting, eager RCB
splitting, posted/non-posted/completion credit stalls, outstanding tags,
completion-buffer capacity and release, directional full-duplex accounting,
stale and discarded plans, and visibility-dependency domains. It checks all
semantic service-class ledgers against the aggregate ledger field by field.

Timestamp, transaction-ID and accounting overflow tests run through the real
PCIe fabric. Cross-class byte and path-delay overflow is detected while the
candidate plan is still private, leaving shared counters, resources and time
unchanged. The WorkQueue tests separately prove that failed doorbell planning
leaves its WQE in the unpublished SQ owner and commits no fabric transaction.
A failed CQE-write plan likewise keeps the WQE and network token in flight;
retry then publishes CQE sequence one and commits the fabric exactly once.

The regular WorkQueue path accounts one 4-byte send doorbell-record host
store and one 8-byte UAR posted write per batch, one MRd/CplD WQE fetch per
WQE, and one posted CQE write per required completion. QPC lookup and scheduler
service remain local RNIC stages.

## What this validates and what remains

This closes BACK-10 at its accepted transaction-level queueing boundary. The
model now includes deterministic shared serialization and resources,
transactional WorkQueue integration, explicit class and path ledgers, and
fixed, Gaussian and rare-tail mixture profiles for every analytical path
penalty. BACK-16 owns higher-precision occurrence mechanisms, chronological
and class-specific arbitration, measured replay, PCIe ordering, 10-bit tag
scaling and CX-7 calibration. BACK-17 owns optional BlueFlame, ATS/ATC, MSI-X,
missing traffic producers and lower-layer PCIe events. Defaults remain
synthetic and no analytical incidence draw is presented as detected hardware
state.
