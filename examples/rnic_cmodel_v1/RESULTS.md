# RNIC golden C model slice B results

Run on 2026-09-01 against the expectations frozen in
[expectations.md](expectations.md), committed as
`2a36ee48f645c43b7276ecfcc11e240404b76459` before the transmit pipeline
existed and before any number was produced by it.

**Verdict: 41 of 42 registered checks pass. The one miss is the registered
one: the depth-1024 over depth-1 ratio at 8 KiB is 7.62 against the measured
5.9, because a lossless transmit pipeline saturates at the goodput ceiling
where the silicon sat in a loss equilibrium. Every fatal guard held, so the
run is scored rather than voided.**

## Method

The study drives the `extern "C"` facade, not the C++ classes, so the entry
points under test are the ones an RTL testbench uses. Behind the facade the
transmit pipeline segments each work request at the MTU, bounds the
outstanding work per QP and paces packet issue; the probe implements the wire,
serializing each packet at the link rate, adding the measured one-way latency
floor in each direction and acknowledging every packet. Nothing is dropped.

Reproduce from the repository root:

```bash
python examples/rnic_cmodel_v1/run_cmodel.py
```

Per-cell rows are in [curves.csv](curves.csv) and one row per registered check
is in [summary.csv](summary.csv). Raw per-cell probe output and the replay
traces are written under `${SIMLLM_DATA_ROOT}/rnic_cmodel_v1/` and are not
tracked.

## Fitted law

Fitting `t = T_eff + S / C` to the six depth-1 cells of each profile
reproduces the campaign constants the profile was built from:

| profile | fitted T_eff | fitted C | frozen T_eff | frozen C |
|---|---:|---:|---:|---:|
| cx5_100g | 4.475 us | 97.100 Gb/s | 4.48 us | 97.1 Gb/s |
| cx7_400g | 4.479 us | 388.400 Gb/s | 4.48 us | 388.4 Gb/s |

This is a closure check, not a calibration: T_eff and C are inputs to the
profile, and the fit only shows that segmentation, header accounting, the
window and the pacer compose into the law rather than distorting it. The
5 ns of T_eff the fit loses is the one residual the design has: the wire
serializes the last packet of a message at the raw link rate while the pacer
issues at the effective wire rate, which is 4.7 ns per message at MTU 4096.

## Checks

| check | cells | verdict |
|---|---:|---|
| `depth1_law`, within 15 percent of `B = S / (T_eff + S / C)` | 12 | PASS, worst residual 0.10 percent |
| `depth_ratio_law`, within 2 percent of the ceiling ratio | 4 | PASS, worst residual 0.67 percent |
| `depth_ratio_measured`, within 20 percent of 5.9 and 1.57 | 2 | 1 PASS, 1 FAIL |
| `mtu_tax`, 5.6 plus or minus 2 percentage points | 2 | PASS at 4.43 and 4.47 |
| `cx7_scaling`, within 1 percent of the law at 4C | 6 | PASS, worst residual 0.03 percent |
| `pps_ceiling`, within 5 percent of the per-QP message rate | 2 | PASS at 3.868 and 15.448 Mpps |
| `depth_monotone`, non-decreasing in queue depth | 12 | PASS |
| `ceiling_bound`, no cell above its profile goodput | 2 | PASS at 97.095 and 388.321 Gb/s |

The depth-1 curve, per profile and size, in Gb/s:

| size | cx5 measured | cx5 law | cx7 measured | cx7 law |
|---|---:|---:|---:|---:|
| 4 KiB | 6.809 | 6.802 | 7.181 | 7.179 |
| 8 KiB | 12.725 | 12.713 | 14.101 | 14.098 |
| 16 KiB | 22.501 | 22.483 | 27.214 | 27.208 |
| 64 KiB | 53.094 | 53.069 | 89.949 | 89.931 |
| 256 KiB | 80.433 | 80.419 | 212.299 | 212.274 |
| 1 MiB | 92.318 | 92.313 | 321.690 | 321.676 |

## The registered miss

`depth_ratio_measured` at 8 KiB on `cx5_100g` reports 7.618 against a band of
4.72 to 7.08 around the measured 5.9. The expectations registered this before
the run, including its direction and its mechanism: at depth 1024 and 8 KiB
the silicon sat in the saturated loss equilibrium the campaign measured at 78
to 92 Gb/s, and this slice has no ingress meter, no loss and no
retransmission, so it saturates at the 97.1 Gb/s ceiling instead. The
model-internal `depth_ratio_law` check at the same cell passes at 0.25 percent,
which localizes the disagreement to the missing mechanism rather than to the
packetizer, the window or the pacer. BACK-56 owns it.

The same reading explains why 64 KiB passes both checks: at 64 KiB the
measured depth-1024 point was already close to the lossless ceiling, so the
missing equilibrium costs little.

## Fatal guards

All held, so the run is scored:

- deterministic replay identity: the designated replay cell (`cx5_100g`,
  64 KiB, depth 16) ran twice in one process and produced byte-identical
  rows and byte-identical facade traces;
- one-authority conservation: in all 40 cells posted equals delivered equals
  reclaimed equals the offered message count, with zero errors and zero CQ
  overruns;
- packet conservation: every cell emitted exactly `ceil(size / mtu)` packets
  per message carrying exactly the offered payload, with exactly 64 wire
  header bytes per packet;
- pacing integrity: zero late releases in every cell, so the caller stepped to
  every paced issue instant the model announced.

## What this does not show

No loss, no retransmission, no congestion response, no receive-side behavior,
no multi-QP arbitration and nothing the fabric owns. The depth-1 law is not
independently confirmed here: T_eff and C are profile inputs. The wire is a
deterministic serializer with a fixed latency, not a packet simulator, so the
incast, drain-window, equilibrium and pause rows of the anomaly table remain
untouched by this study.
