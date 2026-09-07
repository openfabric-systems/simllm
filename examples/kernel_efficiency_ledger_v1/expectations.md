# Kernel efficiency ledger v1 expectations

This freeze precedes this study's implementation and first analysis run. It
selects already published evidence, including a void study; it is not a blind
hardware prediction or a new hardware pre-registration. No new measurement or
ledger outcome is written into this file. COMP-45 and COMP-46 own the relevant
measurement and kernel replacement work; this analysis closes neither task.

## Physical interpretation before arithmetic

A kernel moves operands through memory and performs arithmetic on them. Its
arithmetic intensity is declared floating-point operations (FLOPs) divided by
declared bytes. The ridge point is the device's arithmetic ceiling divided by
its memory bandwidth. Below that ridge, memory supplies work too slowly to
fill the arithmetic units; above it, arithmetic is the ideal limiting term.
This ideal comparison identifies a candidate bottleneck, not the cause of a
slow implementation. Small work can instead sit below a launch-cost scale.

Floor: elapsed time is positive and at least the greater of arithmetic work
at the clock-derived ceiling and unavoidable bytes at the memory nameplate.
For repeated warm buffers, unavoidable bytes are
`max(distinct_bytes - device_L2_bytes, 0)`; nominal traffic is not a measured
HBM transaction count. A two-pass normalization's activation rereads need not
all reach high-bandwidth memory (HBM).

Ceiling: achieved nominal byte rate divided by the selected measured HBM
envelope must be at most 1 for a cell to support an efficiency verdict here;
so must its fraction of the binding roof. A breach voids that cell's verdict,
including cache-resident cells. It is a declaration or measurement finding,
not evidence that silicon exceeded physics. Elapsed time has no finite upper
bound from hardware ceilings alone; no fabricated latency ceiling is used.

## Exact retained population

The adjacent `expectations.json` fixes input SHA-256 digests and ordered cell
identities. Source files are read only. No reruns, fitting, filtering by time,
GPU, framework, remote service or additional evidence are allowed.

- Every run-2 `cells.base` and `cells.boosted` entry of
  `a100_kernel_constants_v1/measurements/results.json`: 18 base and 246
  boosted cells. Keep warm, rotated, held-out, paired-clock and host-issue
  bound rows. Identity is source plus arm plus cell ID, never ID alone.
  `previous_run` has relation summaries rather than a per-cell population;
  it is not a second set of kernel times. The host-issue-bound array repeats
  a selected cell, so it supplies annotations rather than a duplicate row.
- Both hardware-envelope lane-A records: every read, write and copy at every
  one of the 13 retained sizes, plus every one of the 19 GEMMs (general
  matrix multiplications). This is 58 rows per device, 116 rows total.
  Include small cached cells even when their HBM interpretation fails.
- All five Granite kernels selected by the retained lookup fixture manifest:
  `flash_combine`, `flash_split_kv`, `fused_moe`, `gemvx`, `topk_gating`.
  Reconstruct the existing lookup record with its existing analyzer and
  require the published record digest. Carry both elapsed instruments and
  observed clocks. None has a complete per-kernel operand shape or operation
  count: rates, fractions, class and rank must be unavailable, not zero or
  inferred from utilization counters. The step shape cannot identify each
  routed kernel's work. Retain these five rows with a missing-work reason.

Total inventory is 385 source cells, of which 380 have declared work. These
are evidence rows, not behavioral test instances or a score denominator.
NVLink peer and collective records supply link context only: they are not
local compute kernels and their bytes are not HBM demand for these cells.

## Work accounting

Use exact integer work recomputed from declared shapes, then cross-check the
source byte/FLOP fields with relative tolerance `5e-9` for the source's printed
floating-point precision. Do not reuse the source efficiency fields.

| Family | Declared bytes D | Declared FLOPs F |
|---|---|---|
| BF16 GEMM, including expert GEMMs | `2*(M*K + K*N + M*N)` | `2*M*N*K` |
| Prefill score and value, H heads of width W, length S | `4*H*(2*S*W + S*S)` | `4*H*S*S*W` |
| Decode, batch B, H query heads, H/2 key/value heads, width W, length L | `4*B*(H/2)*L*W` | `4*B*H*L*W` |
| HBM read or write, buffer Q bytes | `Q` | source-declared 0 |
| HBM copy or elementwise scale | `2*Q` | source-declared 0 |
| HBM triad, elementwise add or normalization | `3*Q` | source-declared 0 |

BF16 is the two-byte brain floating-point operand format. GEMM work counts a
multiply and an add as two operations. Decode counts dot products and value
accumulation, not softmax instructions. A source-declared zero for streaming
kernels means no arithmetic work was inventoried, not zero executed scalar
instructions. Report that qualifier beside their FLOP rates. In particular,
normalization omits weight traffic in the retained cell's byte declaration,
although the original prose mentions it. Preserve that declaration visibly;
do not silently repair the original measurement or claim hardware counters.

## Denominators and classification

For each device, primary HBM bandwidth R is the maximum of `D/time` across
read, write and copy lane-A cells at sizes at least 256 MiB. Select against
that hardware-envelope study, never the void constants study's own maximum.
The secondary HBM ceiling is `2*mem_clock_khz*1000*mem_bus_bits/8`.
These exact selectors, device records and their digests freeze the denominators
without copying measured values into an expectations-only commit.

The arithmetic ceiling P is clock-conditioned: A100 uses
`108*2048*SM_MHz*1e6`; GH200 uses `132*4096*SM_MHz*1e6`. Constants-study cells
use `scored_state`, not the arm's name. Envelope GEMMs use the maximum of
before and after clocks because an endpoint clock drop does not identify the
clock throughout the timed interval. The interval endpoints are both retained.
Both denominator variants use this same P: the primary roof combines measured
HBM with theoretical arithmetic, not a measured compute ceiling. Show
`GPU_ENVELOPES` alongside these denominators: A100's 312 TFLOP/s and
2.039 TB/s are rounded model values. GH200 has no exact entry; report the
H100 entry as a comparison only and never substitute it for GH200.

Report measured NVLink per-pair rates and fan-out per-GPU egress from each
lane B, beside the corrected payload nameplates (A100 100/300 GB/s, GH200
150/450 GB/s). They are context, not a denominator for local kernel work.

For each cell and envelope source:
`AI=F/D`, `ridge=P/R`, `t_memory=D/R`, `t_compute=F/P`,
`binding_floor=max(t_memory,t_compute)`, `fraction=binding_floor/time`.
Report both achieved byte rate and achieved FLOP rate, as well as the two
individual ceiling fractions. Below the ridge classify HBM-bound; equality
and above classify compute-bound, matching `RooflineProvider`.

Launch-bound overrides that class when the primary measured binding floor is
at most the device's lane-A pipelined launch period, times two for a prefill
cell's two GEMMs and times one otherwise, or the source says host-issue-bound.
Keep the underlying ridge class separately. This is a declared conservative
size screen using an independently retained launch scale, not measured proof
of launch causality and not a subtraction from elapsed time.

Declare an inefficient-kernel diagnostic when `fraction < threshold` and the
cell is not launch-bound. The headline threshold is 0.5, a declared screen,
not a calibration. A void source or cell blocks an accepted verdict even if
the diagnostic predicate is true. The measured physical guard is applied in
both variants, so the datasheet variant cannot rehabilitate a breached cell.

## Frozen sweep and relations

Run the full cross product of envelope source `{measured, datasheet}` and
threshold `{0.25, 0.5, 0.75}`: six configurations on the identical population.
Rank all work-declared cells by increasing fraction, using cell identity to
break exact ties. Void rows remain visible and ranked for diagnosis only.
Missing-work rows have no rank. Compare complete orders, Spearman rank
correlation, maximum rank displacement and pairwise inversion count.

- R1, decode: the six boosted decode cells with at least 160 MiB declared KV
  traffic remain HBM-bound and between 0.04 and 0.16 of the primary roof,
  all below 0.25. Direction is informed by the published low-bandwidth finding.
- R2, large GEMMs: A100 `gemm_G4_m8192` boosted and the 8192 and 16384 square
  envelope GEMMs on both devices are compute-bound and reach [0.8, 1.0] of
  their clock-conditioned compute ceiling. This is a high-roof expectation,
  not a claim that every smaller or oddly shaped GEMM has high efficiency.
- R3, HBM copy: all selected copy cells at 256 MiB and above reach [0.8, 1.0]
  of their device's measured HBM envelope and remain HBM-bound.
- R4, rank stability: the two full work-declared orders have Spearman
  correlation at least 0.98. Exact global ordering is not predicted: changing
  only the memory ceiling can reorder different regimes and different devices.
  Report every changed rank and the inversions whether this bound holds or not.

Structural guards, unscored: threshold changes leave fractions and ranks
exactly identical and nest the diagnostic candidate sets. Within one device,
non-launch cells remaining memory-bound in both variants preserve their
relative order and their fractions change by exactly `R_measured/R_datasheet`.
Holding R fixed and sweeping P over `{0.5, 1, 2}` times its clock-derived
value leaves the fraction exactly unchanged for cells that remain HBM-bound
at all three settings. The last check conditions on staying below the ridge;
no invariance is claimed across a regime change.

## Evidence validity and reporting

Source digest, inventory, unique identity, finite positive time and clocks,
work-declaration agreement, record reconstruction, deterministic output and
sweep algebra are fatal study guards. A failure voids the analysis itself and
prevents a completed report. Never count them as behavioral passes.

The inherited constants-study VOID state and source host-issue exclusion are
preserved on every applicable row. A per-cell measured-envelope breach is
survivable only for the descriptive ledger: it voids that cell's accepted
verdict and leaves other source cells interpretable in their own evidence
class. The mixed-source study is VOID for calibration or task closure, with
findings; no aggregate behavioral score is published. R1 to R4 retain their
observed ranges and descriptive relation outcomes. No guard fraction or
combined total of source rows, configurations and test cases is permitted.

The report cites the expectations-only commit, places the ranked ledger first,
explains physical floors and ceilings, and distinguishes findings from accepted
verdicts. One plain matplotlib figure plus PNG and PDF shows the ranked
fractions and denominator sensitivity. No serving defaults, runtime service,
time to first token (TTFT), time per output token (TPOT), COMP-45 acceptance or
COMP-46 replacement is changed. Missing Granite work attribution and any
refuted relation are residuals for the orchestrator, who owns registration.
