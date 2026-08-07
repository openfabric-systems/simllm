# RNIC PCIe v1 results

The study reproduces 35 configurations: 29 baseline rows and 6 analytical-
profile rows. All 35 match their deterministic row oracles. Ten behavioral
relation families are instantiated 18 times, and all 18 instances pass.
Structural conservation, inactive-field and configuration-forced zero
invariants also pass, but are unscored. Fifteen directed requirement families
run through three native test executables and are reported separately. Counts
from these evidence classes are not added together.

## Method

The study builds the dependency-free C++17 RNIC library in Release mode, runs
CTest, then drives `simllm_rnic_pcie_probe` over the parameter grid specified in
[expectations.md](expectations.md). The Python runner computes expected TLP
counts, directional bytes, serializer times and link-queue attribution
independently. It does not reuse C++ model results as its oracle.

The original sweeps first entered public history together with their results,
so they are post-specified regression oracles, not publicly auditable
preregistration. The main review-correction expectations were committed alone
in `91cfe65` before the corrected implementation or run. Additional credit and
split-horizon edges were frozen in `7f2e961` and `acbef82`; `f269bc8` froze the
legal posted-after-completion queue ledger. Commit `7f592d3`
records an explicit pre-landing retraction: an intermediate expectation
generalized the mandatory posted-over-non-posted rule to completions, but the
first native regression exposed that as an overconstraint on a legal
posted-after-completion order. These commits are the audit trail for the
correction reported here.

Reproduce and compare every byte of the tracked CSV from the repository root:

```bash
python3 examples/rnic_pcie_v1/run_rnic_pcie_v1.py --check
```

Raw probe knobs, effective configuration, requested and accounted transaction
counts, analytical-profile parameters, measurements and expected values are in
[results.csv](results.csv).

The correction changes no JCT, transaction timestamp, byte, TLP, credit,
finite-resource or analytical-profile result in the 35-row study. Among fields
present in both CSV versions, only `link_queue_wait_ps` changes. The CSV now
also carries its independent expected value and a separate fatal, unscored
structural-invariant status.

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

## Link-queue accounting correction

The old ledger measured every fragment from transaction eligibility. It
therefore charged a transaction for its own preceding fragments, producing the
triangular `N * (N - 1) / 2` overcount. The corrected metric chains accounting
eligibility across every same-direction TLP in one public transaction while
leaving the rational serializer recurrence unchanged.

All 25 single-transaction link rows in sweeps A, B and D now report zero
link-queue wait. Representative corrections are:

| Transaction | Segmentation | Old link queue (ps) | Corrected (ps) | Unchanged JCT (ps) |
|---|---:|---:|---:|---:|
| 4096-byte MWr, Gen5 x16 | MPS 128, 32 MWr | 1196422 | 0 | 77188 |
| 4096-byte MWr, Gen5 x16 | MPS 256, 16 MWr | 533211 | 0 | 71094 |
| 512-byte read | MRRS 128, MPS 128 | 14093 | 0 | 9776 |
| 512-byte read | MRRS 512, MPS 512 | one MRd, one CplD | 0 | 0 | 8824 |

The 16-transaction read-window rows retain real external contention. Their
independent oracle and measured ledger agree exactly:

| Read slots | MPS (B) | Old link queue (ps) | Corrected link queue (ps) |
|---:|---:|---:|---:|
| 1 | 128 | 106257683 | 106032195 |
| 1 | 512 | 105932235 | 105932235 |
| 4 | 128 | 21796041 | 21408305 |
| 4 | 512 | 21368316 | 21368316 |

These are sums of per-transaction waits, so they may exceed the final JCT. MPS
128 changes because each read has four CplDs; MPS 512 already had one and was
not affected.

## Outstanding-read queue

Sixteen 512-byte DMA reads were submitted at time zero with a fixed 1,000,000
ps response delay. Increasing the read window from one to four reduces both
JCT and accumulated outstanding-slot wait exactly as predicted.

| Read slots | MPS (B) | JCT (ps) | Outstanding wait (ps) | Link queue (ps) |
|---:|---:|---:|---:|---:|
| 1 | 128 | 16156416 | 15140925 | 106032195 |
| 1 | 512 | 16141184 | 15126645 | 105932235 |
| 4 | 128 | 4067289 | 3051798 | 21408305 |
| 4 | 512 | 4060623 | 3046084 | 21368316 |

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

Posted visibility and non-posted completion now have separate domain horizons.
In the directed long-read case, a same-domain 4-byte MWr has zero ordering wait,
starts at 381 ps and completes at 826 ps instead of waiting for the MRd
completion at 1,002,730 ps. A second directed case exhausts the one-entry read
window: read B is blocked until 1,002,730 ps, while the ready MWr fills the idle
request-link gap from 381 to 826 ps. A posted transaction too large for the
pre-reserved gap fails inside its private plan; the following 4-byte retry then
uses transaction ID 3 and the exact same gap. The eager API never silently
reports the forbidden posted-behind-blocked-read order.

The complementary dependency test fixes both read-side horizons numerically.
A 4-byte MWr completes at 445 ps; the following same-domain MRd attributes 445
ps of ordering wait and completes at 3,175 ps; the next MRd attributes 3,175 ps
and completes at 5,905 ps. Independent work in the same numeric domain has zero
ordering wait and completes at time zero.

Credit-aware arbitration is also pinned on both sides of a future MRd
reservation. When a posted credit returns during the MRd, the later MWr starts
at 1,003,556 ps and attributes 937 ps of link queue: 826 ps before credit
waiting and 111 ps after credit availability. Its separate credit wait is
1,002,619 ps. In the companion short-gap case, the MWr is not credit-ready
before the MRd, so it is not falsely rejected for failing to fit there. It
starts when its credit returns at 3,945 ps, completes at 6,358 ps and attributes
826 ps of link queue plus 3,119 ps of credit wait.

Completion ordering is deliberately not folded into the mandatory
posted-over-non-posted exception. In its directed case, a D2H completion owns
the link from 10,381 to 12,730 ps. A later eight-TLP D2H MWr legally emits four
TLPs before that interval and four after it, completes at 22,379 ps and charges
the intervening 3,081 ps exactly once as external link queueing.

This boundary follows the PCI-SIG ordering rationale that independent posted,
non-posted and completion flow-control classes preserve forward progress, plus
the AMD requester guidance that tag- or non-posted-credit-starved reads must
not create posted head-of-line blocking. See the
[PCI-SIG ordering webinar](https://pcisig.com/sites/default/files/files/PCI-SIG%20Unordered%20IO%20Webinar_Rev4_FINAL.pdf)
and
[AMD requester guidance](https://docs.amd.com/r/en-US/pg346-cpm-pcie/Avoiding-Head-of-Line-Blocking-for-Posted-Requests?contentId=M~o53XyovPwgj5x38Vm~_g).

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
and class-specific arbitration, completion arbitration, deferred displacement
of already returned reservations, measured replay, the remaining PCIe
RO/IDO/TC/VC ordering matrix, 10-bit tag scaling and CX-7 calibration. BACK-17
owns optional BlueFlame,
ATS/ATC, MSI-X, missing traffic producers and lower-layer PCIe events. Defaults
remain synthetic and no analytical incidence draw is presented as detected
hardware state.
