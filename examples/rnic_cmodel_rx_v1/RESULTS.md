# RNIC golden C model slice C results

Run on 2026-09-02 against the expectations frozen in
[expectations.md](expectations.md), committed as
`c25251319522be585b6d8b50be737a1de9987b00` before the ingress meter, the
receive processor and the requester transport existed and before any number
was produced by them.

**Verdict: 35 of 40 registered checks pass. One miss is the one the
expectations registered in advance with its mechanism (`simplex_dirty`). One
is a defect in the frozen sweep itself, not in the model: the registered
4-queue-pair unreliable offer is four times the port's byte capacity, so the
ingress meter binds before the packet-rate ceiling can. The remaining three
are the incast, and they are one finding: a go-back-N requester with no
congestion-control reaction point does not reach a stable equilibrium at the
utilization the campaign measured, and the campaign's own counters show that
run was congestion-controlled. Every fatal guard held, so the run is scored
rather than voided.**

## Method

The study drives the `extern "C"` facade, not the C++ classes. Behind it the
requester packetizes, paces and keeps go-back-N state; the responder runs the
ingress meter and the receive processor; the probe owns the wire, serializing
each packet on a per-direction link with exact rational arithmetic, adding the
measured one-way latency floor, optionally losing it in the fabric, handing it
to the responder's receive entry point and carrying the responder's ACK or NAK
back. Every wire event is delivered in one merged time order, because a facade
whose clock has been pushed past a timestamp refuses the event that carries it.

Reproduce from the repository root:

```bash
python examples/rnic_cmodel_rx_v1/run_rx_study.py
```

Per-cell rows are in [curves.csv](curves.csv) and one row per registered check
is in [summary.csv](summary.csv). Raw output and the replay traces are written
under `${SIMLLM_DATA_ROOT}/rnic_cmodel_rx_v1/` and are not tracked.

## Fitted latent parameters

| parameter | fitted | band the checks leave around it | how |
|---|---:|---|---|
| `rx_drain_bps` | 96.6e9 wire bit/s | 96.2e9 to 96.8e9 | least squares over a declared candidate grid from 95.0e9 to 98.0e9 in 0.2e9 steps, against the measured saturated goodput at 8 KiB and 64 KiB |
| `rx_ingress_bytes` | 262016 | at least 170 KiB | unchanged from the slice-A profile; it only has to hold one burst's backlog at the widest clean gap, and it does |
| `rx_pps_per_qp_ud` | 3.07e6 | 2.79e6 to 3.38e6 | unchanged; it is the directly measured cap and the model reproduces it exactly |
| `rx_pps_per_nic` | 9.65e6 | 8.69e6 to 10.6e6 | set from the measured multi-queue aggregate; four queue pairs each offered above their own ceiling deliver exactly this |
| `rx_pps_per_qp_rc` | unset | not established | no reliable cell reached it, so the study leaves it at zero rather than inventing one |

The drain-rate fit is over-determined, which is what makes it worth having.
The same single value has to place two drain-window thresholds and one
equilibrium window:

| candidate (Gb/s) | 95.0 | 95.8 | 96.2 | 96.4 | **96.6** | 96.8 | 97.2 | 98.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| squared relative residual | 0.0666 | 0.0179 | 0.0055 | 0.0024 | **0.0012** | 0.0020 | 0.0097 | 0.0576 |

## Checks

| check | rows | verdict |
|---|---:|---|
| `gap_discards`, categorical against the measured threshold | 9 | PASS |
| `paced_goodput`, within 15 percent of 92.2 and 97.3 | 2 | PASS at 92.32 and 96.48 Gb/s |
| `saturated_goodput`, inside the 78 to 92 Gb/s window | 2 | PASS at 79.25 Gb/s |
| `depth_ratio_measured`, within 20 percent of 5.9 and 1.57 | 2 | PASS at 6.23 and 1.49 |
| `depth1_unchanged`, within 1 percent of the slice-B law | 2 | PASS at 12.725 and 53.094 Gb/s |
| `ud_cap`, within 10 percent of 3.07 Mpps | 2 | PASS at 3.070 Mpps, twice |
| `ud_passthrough`, delivered equals offered below the cap | 2 | PASS, exactly |
| `ud_silent`, discards exact and no transport signal | 4 | PASS |
| `ud_aggregate`, within 10 percent of 9.65 Mpps | 1 | FAIL at 5.72 Mpps |
| `incast_wire`, at least 97 percent utilization | 1 | FAIL at 55.8 percent |
| `incast_tax`, within 25 percent of 26.9 percent | 1 | FAIL at 98.8 percent |
| `incast_fair`, 50 plus or minus 2 points | 1 | PASS at 50.00 |
| `incast_direction`, tax rising in loss and in message size | 2 | 1 PASS, 1 FAIL |
| `duplex_clean`, every counter zero at 91.8 Gb/s | 1 | PASS |
| `simplex_dirty`, discards nonzero at 93.4 Gb/s | 1 | FAIL, registered in advance |
| `ledger_identity`, the two sequence counters agree within 10 percent | 6 | PASS, exactly equal in every cell |
| `inert_marking`, `np_ecn_marked_roce_packets` exactly zero | 1 | PASS |

### The drain window falls out of the meter

The categorical pattern the campaign measured is reproduced exactly, and it is
reproduced by one drain rate rather than by a rule per size:

| size | gap 0 | gap 4 us | gap 100 us | gap 368 us |
|---|---|---|---|---|
| 8 KiB measured | dirty | clean | clean | clean |
| 8 KiB model | dirty, 335 discards | clean | clean | clean |
| 64 KiB measured | dirty | dirty | clean | clean |
| 64 KiB model | dirty, 335 | dirty, 120 | clean | clean |
| 1 MiB model | dirty, 2389 | dirty, 2329 | dirty, 2238 | dirty, 2238 |

The mechanism is the one the closed form in the expectations names: a burst of
128 messages leaves `p` times its wire bits of backlog behind it, and the gap
that clears it scales with the burst's byte count. At 8 KiB that gap is under
4 us, at 64 KiB it is between 4 and 100 us, and at 1 MiB it is beyond the
widest gap swept, which is the registered monotone claim.

The in-burst goodput at the smallest clean gap is 92.32 Gb/s at 8 KiB against
a measured 92.20, a residual of 0.13 percent, and 96.48 Gb/s at 64 KiB against
a measured 97.31, a residual of 0.86 percent.

### The depth-ratio residual slice B registered is closed

Slice B reported the depth-1024 over depth-1 ratio at 8 KiB as 7.62 against a
measured 5.9 and named the ingress meter as the missing mechanism. With the
meter enabled the ratio is 6.23, inside the registered 20 percent band, and
the 64 KiB ratio is 1.49 against a measured 1.57. The depth-1 goodput is
unchanged to within 0.1 percent, so the meter costs nothing where the arrival
rate is far below it, which is the other half of that claim.

The model's saturated goodput is 79.25 Gb/s at every message size, because at
gap 0 it depends only on the total bytes offered. The silicon's did not: it
measured 77.52 at 8 KiB and 81.44 at 64 KiB. Both sit inside the registered
window and the model sits between them, but the size dependence itself is not
reproduced and no mechanism in this slice would produce it.

### The unreliable cap is exact

| offered (Mpps) | 2.00 | 3.00 | 4.00 | 5.85 |
|---|---:|---:|---:|---:|
| delivered, one QP (Mpps) | 2.000 | 3.000 | 3.070 | 3.070 |
| discarded | 0 | 0 | 46500 | 95043 |

Discarded equals offered minus delivered exactly, every discard lands on
`rx_discards_phy`, and `out_of_sequence`, `packet_seq_err` and
`roce_adp_retrans` stay at zero throughout: the loss is invisible to the
sender, which is what the campaign measured.

### The counter ledger

The requester's `packet_seq_err` and the responder's `out_of_sequence` are
equal in every cell, not merely within 10 percent, and both equal the number
of loss events. That follows from the mechanism rather than from a rule: a
discarded packet signals nothing itself, and the next in-order packet that
arrives out of sequence is what raises one count at each end.

| cell | losses | `out_of_sequence` | `packet_seq_err` |
|---|---:|---:|---:|
| 8 KiB gap 0 | 335 | 335 | 335 |
| 64 KiB gap 4 us | 120 | 120 | 120 |
| 1 MiB gap 0 | 2389 | 2389 | 2389 |

## The misses

**`simplex_dirty`, registered in advance.** The expectations predicted this one
and named its mechanism: the measured 93.4 against 91.8 pair implies a
responder discard threshold between 93.23 and 94.86 Gb/s of wire, while the
gap sweep and the equilibrium window together require a drain rate between
95.7e9 and 97.9e9. One meter with one drain rate cannot satisfy both, and the
fitted 96.6e9 leaves 93.4 Gb/s clean. The model reports zero discards at
93.4 Gb/s where the silicon reported 43040. The duplex half of the pair passes,
so the disagreement is localized to the threshold's position, not to the meter.

**`ud_aggregate`, a defect in the frozen sweep.** The expectations registered
four unreliable queue pairs each offered 5.85 Mpps of 2 KiB messages. That is
395 Gb/s of wire into a 100 Gb/s port, so the ingress meter binds long before
the per-NIC packet-rate ceiling does and the model delivers 5.72 Mpps, the
meter's limit. The campaign's own aggregate point used 1 KiB messages, which
fits the port. Run at that size the model delivers **9.650 Mpps** against a
measured 9.65, so the ceiling itself is right and the frozen offer was not.
That supplementary row is in `curves.csv` as `ud_supplement`; the registered
check is reported as the failure it is rather than rebanded.

**The incast, one finding in three checks.** At the frozen configuration, two
senders paced at their measured share into one responder, the model does not
reach an equilibrium: goodput falls to 0.66 Gb/s, wire utilization to 55.8
percent and the tax to 98.8 percent, and the tax stops rising with the loss
rate because it is already saturated at 0.5 percent. The mechanism is a
congestive collapse. A go-back-N requester answers a loss by replaying
everything behind it, which raises the offered load, which raises the loss,
with nothing in the loop to reduce the sending rate.

Two pieces of evidence say this is a missing block rather than a broken one.
First, the same model with headroom is well behaved: two senders at 24 Gb/s
each and 0.5 percent injected loss produce 62 recovery episodes for 45 losses
and a 7.04 percent tax, and one sender at 97.1 Gb/s with 0.5 percent loss runs
at 93.0 Gb/s with 22 episodes for 20 losses. The transport recovers correctly;
what it cannot do is share a saturated bottleneck. Second, the measured run was
congestion-controlled: the campaign's P5a counters record 78058 CNPs sent by
the receiver and 179746 handled by the senders during that incast. The
reaction point that consumed them is BACK-58's block, not this one.

The fair-share check passes at exactly 50.00, but it passes by construction:
the probe paces each sender at its share, because without a reaction point the
model has no mechanism that would arbitrate one. It is reported as a
construction, not as a result.

## Fatal guards

All held, so the run is scored:

- deterministic replay identity: the designated replay cell (64 KiB, gap 4 us,
  depth 1024, deterministic 1-in-100 injected loss) ran twice in one process
  and produced byte-identical rows and byte-identical facade traces;
- byte conservation: in every cell the delivered plus discarded plus
  sequence-rejected packet counts never exceed what was offered, and every
  completed message carries exactly its offered byte count;
- no completion before delivery: every cell completed exactly its offered
  message count with zero errors, and no message completed twice;
- counter monotonicity and inertness: `np_ecn_marked_roce_packets`,
  `rx_pause_ctrl_phy`, `rx_global_pause`, `rx_out_of_buffer` and both
  `outbound_pci_stalled_*` counters are exactly zero in every cell, which the
  receive pipeline also asserts on itself;
- pacing integrity: zero late releases in every cell;
- identity off: with the receive block disabled the slice-B study reproduces
  its tracked `curves.csv` and `summary.csv` byte for byte, including its own
  registered miss, and all eight native tests pass unchanged.

## What this does not show

No congestion response, no ECN marking, no CNP generation or handling, no
in-NIC arbiter, no loopback path and no PFC: all of those are BACK-58, and the
incast miss above is the clearest statement of why they are needed. The drain
rate and the buffer size are fitted, not read: they are opaque parameters of
the silicon and the study only shows that one value of each reproduces four
measured facts at once. The incast's fair share is imposed rather than
emergent. And the model's saturated goodput has no message-size dependence,
so the 77.5 against 81.4 spread the campaign measured across sizes is
reproduced only as a single value between them.
