# Frontier ladder expectations

These expectations freeze the three-rung frontier: the CORE-62 compatibility
grid priced side by side at the ESTIMATE rung (closed forms), the
SIMULATED loggopsim-ideal rung (the TRAF-20 level on the same declared
inputs), and the SIMULATED packet rung (the pinned CORE-62 htsim
observations), rendered as one NV-style figure with per-rung point classes
and the Pareto front. They are committed before any implementation of the
ladder study exists. The load-bearing product of this study is the
mechanism envelope: where the ideal rung is valid and where only the packet
rung prices the truth.

## Frozen inputs

- `examples/deployment_frontier_v1/result.json`, SHA-256
  `f2f216068bf5ba914853c62a2ee965ede0ebfc0a6f29e3d11cfa5f45eac359ad`
  (byte partitions, fabric observations, analytic and simulated steps).
- The frozen LogGOPSim study contract
  `examples/loggopsim_ideal_v1/expectations.md` (binary SHA-256
  `7e0f13ee3c87a20e9d2e94dbbd74c46075fd03df2f1b04d1ed9739c43ee0a2bf`).
- Declared ideal-level parameters for every cell here: L = 2000 ns (the
  repository's 2.000 microsecond propagation reference, DECLARED), o = 0,
  g = 0, O = 0, S large enough that every payload is eager, and G derived
  from the declared 400e9 bits per second as the exact string `0.02`.

## Fatal guards

- FG-1 pinned-record hash and the loggopsim binary hash verify first.
- FG-2 every ladder point carries its estimator or level stamp with the
  correct point class (ESTIMATE for closed forms; SIMULATED with
  loggopsim-ideal provenance; SIMULATED with the pinned packet-record
  provenance) and the two SIMULATED families are never merged.
- FG-3 the ideal-rung fabric legs are produced by executing the pinned
  binary on rendered GOALs (argv recorded, G as the exact string), never
  by evaluating the closed form that predicts them.
- FG-4 chronology.

## Family L-A: ideal-rung fabric legs, serialized shape (exact, scored)

The two-node serialized configuration renders one flow of the pinned
max_flow bytes per batch; the ideal-rung fabric leg must equal
`L + floor((max_flow - 1) * 0.02)` nanoseconds exactly, executed not
computed:

| batch | max_flow bytes | expected ns |
|---:|---:|---:|
| 1 | 6,651,904 | 135,038 |
| 2 | 13,303,808 | 268,076 |
| 4 | 26,607,616 | 534,152 |
| 8 | 53,215,232 | 1,066,304 |
| 16 | 106,430,464 | 2,130,609 |
| 32 | 212,860,928 | 4,259,218 |

## Family L-B: ideal-rung fabric legs, incast shape (exact, scored)

The nine-node incast configuration renders eight concurrent flows of the
pinned per-flow bytes into one destination. Under the declared parameters
the audited receiver rule charges only per-message overheads, which are
zero here, so the ideal-rung leg equals the single-flow form exactly:

| batch | per-flow bytes | expected ns |
|---:|---:|---:|
| 1 | 1,478,201 | 31,564 |
| 2 | 2,956,402 | 61,128 |
| 4 | 5,912,804 | 120,256 |
| 8 | 11,825,608 | 238,512 |
| 16 | 23,651,215 | 475,024 |
| 32 | 47,302,429 | 948,048 |

## Family M: the mechanism envelope (exact quotients, scored)

Against the pinned packet-rung fabric observations:

- M-1 serialized validity: for every two-node batch, the pinned
  `concurrent_service_ps` divided by the ideal leg (in ps) lies in the
  frozen band [1.000, 1.020]; the batch-32 quotient is exactly
  4,325,821,000 / 4,259,218,000 (about 1.0156). The ideal rung is a valid
  fast substitute for contention-free point-to-point at these shapes, and
  the residual is packet-protocol overhead by the pinned record's own
  attribution.
- M-2 incast blindness: for every nine-node batch, the pinned concurrent
  observation divided by the ideal leg lies in the frozen band
  [7.5, 8.5]; the batch-32 quotient is exactly
  7,689,053,000 / 948,048,000 (about 8.11). The audited cause is stated
  with the number: the ideal rung's receiver charges no per-byte gap, so
  eight-into-one fan-in completes at single-flow time, while the packet
  rung serializes the shared ingress. This mechanism class is exactly the
  no-shared-link-contention assumption of the external planning stacks the
  repository compares against; only the packet rung prices it.
- M-3 isolated agreement: for every nine-node batch, the pinned
  `isolated_service_ps` divided by the ideal leg lies in [1.000, 1.035]
  (single-flow physics agrees across rungs; batch-32:
  962,915,000 / 948,048,000, about 1.0157).

## Family S: step-level ladder (exact, scored)

Ladder step times per point: the ESTIMATE rung equals the pinned
`analytical_step_ps`; the packet rung equals the pinned
`simulated_step_ps`; the ideal rung's step applies the frozen telescoping
with its own fabric leg in place of the packet observation. Frozen
consequences, stated before any run: on this grid the kernel roofline
masks every fabric difference below it, so all three rungs agree at step
level on all 18 points except b100 batch 32 (where the packet rung's
intra-node excess raises the step to the pinned 4,523,298,348 ps while
the ideal rung, which does not model the intra-node candidate, stays at
the analytic 4,257,218,560 ps). The step-level agreement is itself the
honest TRAF-68 finding carried forward: fabric mechanisms are invisible
at these operating points, and the fabric-leg families above are where
the rungs genuinely differ.

## Family P: figure and frontier (scored)

The NV-style figure renders all three rungs on the frozen axes with
per-class markers, the Pareto front computed over the packet-rung points
equals the six b100 points (the wave P0-1 literal), and the plot data
round trips strictly. One figure; the fabric-leg envelope appears as an
inset or companion panel with the M-2 quotient labeled.

## Family W: wall time (scored, generous)

Pricing all 12 fabric legs through the ideal rung completes in at most
5 s total (the audited per-invocation medians are 2 to 21 ms); the
packet-rung numbers are read from the pinned record with zero packet
executions in this study.

## Closure

This study installs the ladder view and freezes the ideal rung's validity
envelope with exact quotients: within about 1.6 percent where the fabric
is a contention-free pipe, about 8x optimistic where fan-in contention
rules, with the mechanism named and owned. It makes no absolute-accuracy
claim about any rung against silicon, does not alter TRAF-68's step-level
finding, and does not touch the frozen deployment scan or loggopsim
study bytes.
