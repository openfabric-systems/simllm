# Single-switch TM1 simulation results

Frozen expectations: 33a44d6d; pre-run clarification: 3d75af59.
Packet-level synthetic UDP component evidence; no hardware calibration,
distributed control deployment, RC/DCQCN performance or GPU metrics.

Executed 920 native configurations. Fatal guards are separate
from behavioral comparisons. Raw CSV and manifests stay outside Git.

## Candidate configurations

Four senders each provide 1 MiB to one receiver. Balanced explicit lanes,
95 percent wire allocation, W=32 us, tick=8 ns, seed=1, static DATA cap.

| Ports | IP MTU | Policy | Makespan us | Payload Gb/s | Max DATA cells | Drops |
| :-- | --: | :-- | --: | --: | --: | --: |
| 4x25 | 1500 | quota_dd | 380.056160 | 88.2881 | 18 | 0 |
| 4x25 | 9000 | quota_dd | 371.040160 | 90.4334 | 88 | 0 |
| 1x100 | 1500 | quota_dd | 379.014400 | 88.5308 | 42 | 0 |
| 1x100 | 9000 | quota_dd | 360.212160 | 93.1519 | 108 | 0 |
| 4x25 | 1500 | quota_prbs | 380.056160 | 88.2881 | 18 | 0 |
| 4x25 | 9000 | quota_prbs | 371.040160 | 90.4334 | 88 | 0 |
| 1x100 | 1500 | quota_prbs | 379.014720 | 88.5307 | 42 | 0 |
| 1x100 | 9000 | quota_prbs | 360.220320 | 93.1497 | 104 | 0 |
| 4x25 | 1500 | prbs | 400.920160 | 83.6936 | 232 | 0 |
| 4x25 | 9000 | prbs | 403.192160 | 83.2219 | 528 | 0 |
| 1x100 | 1500 | prbs | 398.374880 | 84.2283 | 168 | 0 |
| 1x100 | 9000 | prbs | 392.097280 | 85.5768 | 836 | 0 |
| 4x25 | 1500 | unpaced | 168.117600 | 92.7161 | 1258 | 1564 |
| 4x25 | 9000 | unpaced | 166.522080 | 96.7521 | 1248 | 244 |
| 1x100 | 1500 | unpaced | 112.933120 | 92.7124 | 1260 | 2011 |
| 1x100 | 9000 | unpaced | 105.751040 | 97.5762 | 1248 | 327 |

Dropped-run makespans describe the last delivered packet and are not
completion times for the original workload. Their goodput alone cannot
rank a policy against a loss-free calendar.

## Window and tick tradeoff

All three seeds are included in these ranges, using four 25G lanes and
95 percent wire allocation. Lower completion is preferable only if all
bytes arrive and the proposed queue cap remains respected.

| MTU | Policy | W us | Tick ns | Makespan range us | Max cells | Max drops |
| --: | :-- | --: | --: | :-- | --: | --: |
| 1500 | quota_dd | 8 | 8 | 380.248160 to 380.248160 | 18 | 0 |
| 1500 | quota_dd | 8 | 64 | 380.944160 to 380.944160 | 18 | 0 |
| 1500 | quota_dd | 16 | 8 | 380.184160 to 380.184160 | 18 | 0 |
| 1500 | quota_dd | 16 | 64 | 380.944160 to 380.944160 | 18 | 0 |
| 1500 | quota_dd | 32 | 8 | 380.056160 to 380.056160 | 18 | 0 |
| 1500 | quota_dd | 32 | 64 | 380.944160 to 380.944160 | 18 | 0 |
| 1500 | quota_dd | 64 | 8 | 377.816160 to 377.816160 | 18 | 0 |
| 1500 | quota_dd | 64 | 64 | 379.664160 to 379.664160 | 18 | 0 |
| 1500 | quota_prbs | 8 | 8 | 380.248160 to 380.248160 | 18 | 0 |
| 1500 | quota_prbs | 8 | 64 | 380.944160 to 380.944160 | 18 | 0 |
| 1500 | quota_prbs | 16 | 8 | 380.184160 to 380.184160 | 18 | 0 |
| 1500 | quota_prbs | 16 | 64 | 380.944160 to 380.944160 | 18 | 0 |
| 1500 | quota_prbs | 32 | 8 | 380.056160 to 380.056160 | 18 | 0 |
| 1500 | quota_prbs | 32 | 64 | 380.944160 to 380.944160 | 18 | 0 |
| 1500 | quota_prbs | 64 | 8 | 377.816160 to 377.816160 | 18 | 0 |
| 1500 | quota_prbs | 64 | 64 | 379.664160 to 379.664160 | 18 | 0 |
| 1500 | prbs | 8 | 8 | 395.464160 to 454.488160 | 344 | 0 |
| 1500 | prbs | 8 | 64 | 408.208160 to 469.136160 | 296 | 0 |
| 1500 | prbs | 16 | 8 | 395.464160 to 454.488160 | 344 | 0 |
| 1500 | prbs | 16 | 64 | 408.208160 to 469.136160 | 296 | 0 |
| 1500 | prbs | 32 | 8 | 395.464160 to 454.488160 | 344 | 0 |
| 1500 | prbs | 32 | 64 | 408.208160 to 469.136160 | 296 | 0 |
| 1500 | prbs | 64 | 8 | 395.464160 to 454.488160 | 344 | 0 |
| 1500 | prbs | 64 | 64 | 408.208160 to 469.136160 | 296 | 0 |
| 9000 | quota_dd | 8 | 8 | 371.040160 to 371.040160 | 88 | 0 |
| 9000 | quota_dd | 8 | 64 | 371.248160 to 371.248160 | 88 | 0 |
| 9000 | quota_dd | 16 | 8 | 371.040160 to 371.040160 | 88 | 0 |
| 9000 | quota_dd | 16 | 64 | 371.248160 to 371.248160 | 88 | 0 |
| 9000 | quota_dd | 32 | 8 | 371.040160 to 371.040160 | 88 | 0 |
| 9000 | quota_dd | 32 | 64 | 371.248160 to 371.248160 | 88 | 0 |
| 9000 | quota_dd | 64 | 8 | 362.208160 to 362.208160 | 88 | 0 |
| 9000 | quota_dd | 64 | 64 | 364.336160 to 364.336160 | 88 | 0 |
| 9000 | quota_prbs | 8 | 8 | 371.040160 to 371.040160 | 88 | 0 |
| 9000 | quota_prbs | 8 | 64 | 371.248160 to 371.248160 | 88 | 0 |
| 9000 | quota_prbs | 16 | 8 | 371.040160 to 371.040160 | 88 | 0 |
| 9000 | quota_prbs | 16 | 64 | 371.248160 to 371.248160 | 88 | 0 |
| 9000 | quota_prbs | 32 | 8 | 371.040160 to 371.040160 | 88 | 0 |
| 9000 | quota_prbs | 32 | 64 | 371.248160 to 371.248160 | 88 | 0 |
| 9000 | quota_prbs | 64 | 8 | 362.208160 to 362.208160 | 88 | 0 |
| 9000 | quota_prbs | 64 | 64 | 364.325280 to 364.336160 | 88 | 0 |
| 9000 | prbs | 8 | 8 | 403.192160 to 466.041120 | 880 | 0 |
| 9000 | prbs | 8 | 64 | 409.776160 to 473.753120 | 808 | 0 |
| 9000 | prbs | 16 | 8 | 403.192160 to 466.041120 | 880 | 0 |
| 9000 | prbs | 16 | 64 | 409.776160 to 473.753120 | 808 | 0 |
| 9000 | prbs | 32 | 8 | 403.192160 to 466.041120 | 880 | 0 |
| 9000 | prbs | 32 | 64 | 409.776160 to 473.753120 | 808 | 0 |
| 9000 | prbs | 64 | 8 | 403.192160 to 466.041120 | 880 | 0 |
| 9000 | prbs | 64 | 64 | 409.776160 to 473.753120 | 808 | 0 |

## Routing and accounting

| Family | Configurations | Dropped configurations | Largest DATA cells |
| :-- | --: | --: | --: |
| core | 768 | 192 | 1260 |
| routing | 48 | 12 | 1260 |
| accounting | 96 | 32 | 3360 |
| rate | 8 | 0 | 880 |

XPE masks and internal overhead are uncalibrated sensitivities.
Dynamic alpha admission has no 256 KiB static cap in this profile.
Buffered residence includes processing eligibility, while queue wait
starts at processing-ready eligibility. Packet
latency starts at granted/lottery emission and excludes earlier
application waiting, declaration setup and receiver resequencing.

## Interpretation and selected proposal

All 920 configurations pass the fatal ledger, capacity and byte-floor guards.
The independent full-packet wire oracle also agrees for every configuration.
Three unscored analytical relation families agree in 22 instances:
half-rate serialization floors scale exactly by two, jumbo MTU reduces wire
cost for the same requested payload, and a forced single-lane collision
allocates only 23.75 Gb/s. Loaded half-rate makespan ratios range from
1.9603 to 2.0280; fixed delays, ticks, discrete packets and window carry explain
why loaded makespan is not exactly twice. The geometry checks derive from
packet extents/allocation constants and are not scored scheduling behavior.
Keep them separate from native tests, fatal invariants and measured timing.

At 95 percent allocation and 8 ns ticks, W=64 us is the fastest evaluated
bulk calendar: 377.81616 us (1500) and 362.20816 us (9000) for four 25G lanes.
W=32 us costs only 2.24 us more with small frames but 8.832 us more with jumbo
frames. Select W=32 us for the initial interactive prototype, because it
halves the receiver allocation horizon while still permitting a one-window
lead under the candidate 19 us control setup. Select W=64 us and jumbo MTU
for long, advance-announced bulk phases when that extra response granularity
is acceptable. No dynamic-demand experiment or global optimality is claimed.

Bounded-quota PRBS and deterministic round-robin have the same completion in
this balanced, equal-weight workload. Independent lotteries have longer and
seed-dependent tails; there is no observed need for additional randomization
inside the already coordinated calendar. Four ports do not create interior
path diversity. One 100G port is faster in the selected cases, especially for
jumbo frames; four 25G ports remain useful for independent physical queues
and parallel lane service, not a universal throughput or randomness advantage.

Unpaced configurations lose application bytes. Their last-delivery times and
high apparent delivered-byte goodput cannot be presented as faster completion.
Static DATA caps remain at or below 1260 cells. Dynamic admission can reach
3360 cells in the accounting sweep because that deliberately separate profile
has no 256 KiB hard DATA cap. Unknown XPE masks/reservations and SDK header
charges are sensitivities, not calibrated NX-OS settings.

The final C++ suite passes 549 tests, including 14 TM1 focused tests, frozen
legacy behavior and exception cleanup. Earlier runs and their binary hashes
remain historical; this report refers to the independently audited binary
SHA256 885789d2f53f5e8bcc9c432b5a5226d38b9a7b1ba0ebc629dfcab96a72040cc5.
The full source hashes and every exact command are recorded in the external
manifest. Hardware calibration is HTSIM-43; matched RNIC/ECN/RC integration
is HTSIM-44. Neither is closed by this component study.

## Reproducibility and repository validation

The published backend is `534723c` on `codex/asu_tm1_single_switch`; its
source hashes and the binary hash above identify the executable study.
The paired full Python suite produced 6993 passes, 33 skips and one failure:
the generated developer-guide index still counted 51 backend follow-ups
after HTSIM-43 and HTSIM-44 raised that count to 53. Regenerating the index
and updating its module-status count resolved the failure. All 20 affected
task-index/document-format tests then passed. Ruff, the task-index check and
the module-format check also passed. The full Python suite was not repeated
after this documentation-only correction.
