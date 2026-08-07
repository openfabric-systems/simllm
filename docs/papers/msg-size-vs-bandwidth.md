# Message size vs bandwidth: hardware and CC calibration parameters

Working note for calibrating the common RNIC hardware initiation envelope,
the persistent DCQCN rate state and the `rnic-cn` WQ lookahead policy. Source
PDFs live in the gitignored `papers/` directory at the repo root; this summary
and the extracted numbers are the committed record. The 2026-08-07
architecture split assigns WQ/PCIe/DMA behavior to the SimLLM hardware model
and CC state to htsim. The complete queueing, mlx5 capture and CX-7 evidence
plan is [rnic-hardware-calibration.md](rnic-hardware-calibration.md).

## Sources

| Paper | Local file | What it contributes |
|---|---|---|
| UCCL, "An Extensible Software Transport Layer for GPU Networking", arXiv 2504.17307v2 | `papers/uccl-2504.17307.pdf` | goodput vs message size at 400G (Fig. 14), chunk-size saturation (Fig. 15a), host control-path delays (Table 2), production context (Meta and DeepSeek disable NIC CC) |
| Zhu et al., "Congestion Control for Large-Scale RDMA Deployments" (DCQCN), SIGCOMM'15 | `papers/dcqcn-sigcomm15.pdf` | the DCQCN state machine and official parameters; the no-slow-start rule; CNP and buffer thresholds |
| Kalia et al., "Design Guidelines for High Performance RDMA Systems", ATC'16 | `papers/kalia-atc16-rdma-guidelines.pdf` | WQE and doorbell mechanics, PCIe overheads, the ~2 us commodity RTT anchor |
| Li et al., "HPCC: High Precision Congestion Control", SIGCOMM'19 | `papers/hpcc-sigcomm19.pdf` | vendor-default DCQCN timers, ECN-threshold sweeps at 100G, small-flow slowdown magnitudes |

## Extracted data

### UCCL (400G ConnectX-7 / Thor-2 class)

- Fig. 14 (single QP, CC disabled, 16 in-flight messages, no loss):
  goodput vs message size runs about 27 to 30 GB/s at 32 KB, about 45 at
  64 KB, and reaches the ~50 GB/s line rate around 128 to 256 KB. Under
  induced loss ratios 1/4096 to 1/256 the small-message end degrades
  further (down to ~17 GB/s at 32 KB at 1/256).
- Fig. 15a (all-to-all, per-chunk control): 8 KB chunks plateau at about
  30 to 35 GB/s (60 to 70 percent of line rate), 16 KB at about 45,
  32 KB saturates. Stated conclusion: saturating 400G needs at least
  16 KB (4 MTU) per posted unit.
- Table 2 (host control path): CC decision delay 1.7 us p50 / 3.6 to
  10.8 us p99; ACK turnaround 2 to 3 us p50 / 3 to 36 us p99.
- Production context (section 1): Meta disabled NIC CC and schedules at
  the application layer; DeepSeek disabled CC for large-scale all-to-all
  serving. CNP-driven recovery is considered too slow at 400G.

### DCQCN (SIGCOMM'15, 40G ConnectX-3 era)

- No slow start: a new flow sends at full line rate if no other flow is
  active from the host (otherwise the local QoS policy rate). The
  starting cost of DCQCN is therefore not a ramp-up from zero; it is the
  recovery ramp after any rate cut.
- Rate cut on CNP: Rc = Rc (1 - alpha/2), target Rt remembered.
- Recovery: FastRecovery, F = 5 iterations of Rc = (Rt + Rc)/2 per timer
  T = 55 us (or per B = 10 MB byte counter); then AdditiveIncrease with
  fixed R_AI = 40 Mbps per step; alpha decays with g = 1/256 every 55 us
  without CNPs. CNP generation at most one per 50 us per flow.
- Marking and buffers (40G testbed): Kmin = Kmax = 40 KB with Pmax = 1
  in the experiments (DCTCP-like cutoff); PFC thresholds computed from
  the 12 MB Trident II shared buffer, t_flight = 22.4 KB headroom per
  port per priority.
- The 400G arithmetic that matters for us: recovering 200 Gbps of rate
  through AdditiveIncrease at 40 Mbps per 55 us takes 5000 steps, about
  275 ms. Any small WQE issued on a QP whose rate state was cut minutes
  of RTTs ago still sees a fraction of line rate. This is the "slow rate
  ramping" mechanism and it is why production operators disable CC.

### HPCC (100G testbeds)

- DCQCN timer configurations observed in the wild: the paper's original
  Ti = 55 us / Td = 50 us; a vendor default of Ti = 300 us / Td = 4 us;
  and a conservative Ti = 900 us. Aggressive timers reduce FCT slowdown
  but multiply PFC pauses; more than 10 percent PFC pause duration
  suppresses more than 3 percent of total capacity.
- ECN threshold sweeps at 100G: (Kmin, Kmax) in {(12, 50), (100, 400),
  (400, 1600)} KB. High thresholds push p95 small-flow slowdown to about
  30x over the ~5 us baseline RTT (~150 us); low thresholds protect
  small flows and starve large ones. No single DCQCN configuration
  achieves both.

### Kalia et al. (per-WQE mechanics)

- WQE headers: 36 B (RC/UC), 68 B (UD); recv WQE 16 B. Doorbell method:
  one 8 B MMIO ring plus NIC DMA of the WQE; WQE-by-MMIO writes the
  whole (cache-line-aligned) WQE via write-combining MMIO.
- PCIe 3.0 request/completion headers 26/22 B; a DMA read costs less
  host-to-device bandwidth than an equal-sized MMIO.
- Commodity RDMA RTT anchor: about 2 us round trip, so a strictly
  serial post-completion-post loop caps a single-outstanding-WQE flow at
  S / (2 us + S/C).

## Candidate parameter sets

Two separately owned mechanisms are composed in a full-RNIC run. Model A is a
reduced-form check on common hardware initiation and queue service; Model B is
the CC policy state inherited by later WQEs on the same QP. Model A must not be
charged only to DCQCN.

### Model A: common RNIC hardware initiation envelope

Effective goodput of a stream of S-byte WQEs with Q outstanding:
`B(S, Q) = S / (T0/Q + S/C)`; the half-rate message size is
`S_half = T0 C / Q`. If the structural posting, MMIO, WQE/context fetch,
admission and completion queues are collapsed into one diagnostic number,
fitting T0 gives:

| Set | T0 | Anchor |
|---|---|---|
| A1 optimistic | 2.0 us | Kalia's commodity RTT with a fully pipelined doorbell path |
| A2 calibrated | 5.2 us | the maintainer's datum, half of 400G at S = 256 KB with one outstanding WQE: T0 = 262,144 B / 50 GB/s |
| A3 pessimistic | 8.2 us | UCCL Fig. 14 no-loss point (32 KB at ~28 GB/s with Q = 16): T0 = 16 (S/B - S/C) = 16 x 515 ns |

The UCCL Fig. 15a 8 KB plateau under contended all-to-all fits T0 of
about 12 us at Q = 128, consistent with A3 once contention is added; A2
sits between the clean anchors, and A2 at Q = 16 predicts 32 KB at
~34 GB/s, inside the Fig. 14 band.

T0 is not an implementation sleep on every WQE. BACK-9/BACK-10 must reproduce
the envelope through explicit doorbell batching, finite queues, shared PCIe
service, context locality and overlapped WQE issue, while retaining each stage
timestamp. The fitted values remain regression summaries and initialization
bounds for measurements that are not yet available.

### Model B: DCQCN rate ramp (per-QP state the WQE inherits)

| Set | Ti (rate increase) | Td / CNP interval | F | R_AI | g | Kmin/Kmax | Anchor |
|---|---|---|---|---|---|---|---|
| D1 paper | 55 us | 50 us | 5 | 40 Mbps | 1/256 | 40/40 KB (at 40G) | DCQCN SIGCOMM'15 defaults |
| D2 vendor | 300 us | 4 us | 5 | 40 Mbps | 1/256 | 100/400 KB (at 100G) | HPCC's vendor-default timers and mid ECN sweep |
| D3 400G-scaled | 55 us | 50 us | 5 | 400 Mbps | 1/256 | 400/1600 KB (at 400G) | R_AI and thresholds scaled with C so recovery takes the same time in rate fraction as D1 at 40G |

Starting behavior under model B: a WQE's flow starts at min(line rate,
QP current rate). With D1 at 400G a single cut to half rate costs 5
FastRecovery steps (275 us) plus 200 Gbps / 40 Mbps x 55 us of additive
increase, about 275 ms, to return to line rate; under D3 that is 27.5 ms.
Small WQEs issued in that window run at the depressed rate, which is the
measured "DCQCN slows down" effect for small-message streams after any
congestion event.

## Where the current comparator stands (probe of 2026-08-05, report only)

Single cross-leaf flow on the 64-node Clos at 400G, one WQE and a
serialized 16-WQE queue (each send requires the previous, i.e. one
outstanding), no contention, `/data3/yifeng/simllm-dev/wqe-ladder`:

| S | fluid GB/s | rnic-cn GB/s | dcqcn GB/s |
|---|---|---|---|
| 4 KB | 1.97 | 0.37 | 0.49 |
| 16 KB | 7.04 | 1.46 | 1.91 |
| 64 KB | 19.8 | 5.24 | 6.84 |
| 256 KB | 36.2 | 15.5 | 19.3 |
| 1 MB | 45.6 | 30.1 | 35.5 |
| 4 MB | 48.8 | 39.7 | 44.9 |

Readings:

- Every engine's single-WQE curve is latency-bounded with an effective
  fixed offset: fluid T = 2.08 us (pure propagation), dcqcn about
  8.3 us (propagation plus per-hop store-and-forward), rnic-cn about
  11.0 us (the same plus declare/control). The dcqcn curve's implied
  half-rate size is about 415 KB, coincidentally near the A2 anchor,
  but for the wrong reason: it is all topology serialization, with no
  WQE initiation cost and no QP rate state at all.
- The 16-deep queue shows zero amortization in any engine (16x the
  single-WQE time exactly): the probe serializes WQEs, and the models
  have no pipelining or state reuse across WQEs of the same
  destination. Real NICs pipeline the queue (UCCL's Q = 16 curve), and
  the maintainer's cn design amortizes the control cost (below).
- The current DCQCN runtime recreates source state per WQE, so policy state
  does not persist into later WQEs on the same QP. An uncontended WQE starts
  at line rate, as it should; post-CNP persistence and recovery across WQEs in
  Model B remain missing.

## Calibration and design tasks

- BACK-9/BACK-10 (SimLLM): implement Model A as structural WQ/CQ and
  MMIO/PCIe/DMA service shared by every full-RNIC policy. Sweep the A1 to A3
  envelope as a diagnostic, not as three hard-coded sleeps. Acceptance:
  reproduce UCCL Fig. 14 no-loss goodput within 15 percent at Q = 16 and the
  maintainer's 256 KB half-rate datum at Q = 1 under the A2 summary, while
  accounting for every stage and PCIe byte.
- HTSIM-5 (backend): implement Model B as persistent DCQCN policy state shared
  by WQEs of one hardware QP. Sweep D1 to D3, start a new QP at line/local-QoS
  rate, then verify that a controlled CNP affects later WQEs until timer/byte
  recovery restores the rate. The policy must not add doorbell, DMA or CQ
  cost.
- HTSIM-6 (backend): implement `rnic-cn` policy lookahead. A WQE toward a
  destination whose link table is already established must not wait when the
  granted bandwidth suffices; the policy receives bounded lookahead from
  BACK-9 and pre-declares one RTT ahead. Expected effect on the probe: the
  16-deep same-destination queue collapses from 16 x 11.0 us of serial control
  cost toward fluid-plus-one-setup, and the sub-BDP corners (the 1.68x
  buffer-absorbed incast, the 14.3x a2a16 tail vs `rnic-nn`) shrink toward the
  large-flow 1.13 to 1.17x band.
- BACK-8/HTSIM-9 own the common C++ boundary. A comparison is invalid if the
  hardware configuration hash differs between `rnic-nn`, `rnic-cn` and
  DCQCN.

The probe above is a gap measurement, not a registered study; the
registered validation of the calibrated behaviors happens with the relevant
BACK-9/BACK-10/HTSIM-5/HTSIM-6 landings, with expectations frozen against the
anchor tables in this note.
