# Declared receive serialization results

The prospective relations in [expectations.json](expectations.json) were frozen
at `b022b3f3e1ec0e1ca186c4dc5cf276110fe8dff5` against upstream
`2b5c3f8621d952e009b990c93e8bf613befdb813`, before implementation or this study.
The reviewed implementation is `aaba0b0e6e570965d04fba78d1777aedba45a152`.
All 36 registered grid rows match independent deadlines, occupancy and service
conservation exactly. Geometry and rate monotonicity hold, and all three
forbidden-service negative copies are caught. This is continuously ready native
component evidence, with declared geometry and unchanged CX5 fluid defaults.

## Method and metric

The facade receives FIFO packets at the public 2500 ps phase. Each independent
oracle uses only packet lengths, 5000 ps period, configured width and rate.
Initial credit is zero. It computes cumulative useful-byte affordability and
one beat per clock, preserving packet tails. The grid varies width 64/128 bytes,
rate 40/96.6/160 Gb/s, packet length 64/85/4096 bytes and receive capacity
262016/16384 bytes. Token capacity is 4224 bytes. Each row queues
`min(128, floor(capacity/length))` packets.

The completion metric is the last debit time minus the arrival phase, when the
selected receive queue becomes empty. It is neither ACK time nor transport,
CQE or request completion. Every row checks exact accepted byte high-water,
zero unexpected meter drops and exactly `packets*ceil(length/width)` service
events. The probe requires `RNIC_CM_NO_EVENT` on termination, so a service error
cannot masquerade as quiescence. Exact clock rounding is retained.

The raw constant-length geometric ceiling is
`8L*1e12/(T*ceil(L/W))` bits per second. For 85-byte packets at 5 ns it is 68
Gb/s at width 64 and 136 Gb/s at width 128. Token rate and finite capacity are
additional limits. The finite 128-packet 85-byte rows show both effects:

| Width (bytes) | 40 Gb/s interval (us) | 96.6 Gb/s interval (us) | 160 Gb/s interval (us) |
|---|---:|---:|---:|
| 64 | 2.180 | 1.285 | 1.280 |
| 128 | 2.180 | 0.905 | 0.640 |

Both receive capacities produce these intervals because all 10880 accepted
bytes fit. The 64/96.6 row includes one additional initial refill clock; the
geometric ceiling is a steady bound, not a promise of the finite completion
metric. A separate binding-capacity control uses width 128, rate 160 Gb/s and
a 128-byte token cap: 4096 bytes require 64 clocks, compared with 41 with the
larger budget. This distinguishes token capacity from beat geometry.

## Controls and compatibility

Native fixtures cover independent 64/85/4096-byte schedules, phase and
same-clock ordering, finite credit, empty-credit reset, exact capacity edges,
FIFO tails, original sequence/rate processing and invalid configuration.
Receive-only work remains internally scheduled and prevents premature host
memory teardown. Pending work near the timestamp horizon raises an explicit
error; if a legal final 64-byte debit of an 85-byte packet has already occurred,
public occupancy retains the correct 21-byte residual.

Separate negative source copies allow cross-packet tail packing, bank idle
credit, or substitute fluid occupancy for selected admission. All three reach
and fail independent fixtures; none is scored from a compiler refusal.

Three-way checks compare the older downstream vendor pin
`b283af414fe335fd078e6c4e1fe410e88285db68`, the frozen upstream base and the new
checkpoint. Ten original ABI structures and 194 field offsets agree. Default
and explicitly fluid receive results, counters and three replay traces are
byte-identical. A separate normal CX5 packetized transmit control, 32 work
requests of 4096 bytes at depth 16, retains identical result rows and both
replay traces. The option adds a separately versioned construction structure;
original ABI1 layouts remain unchanged.

The native warnings-as-errors build passes all ten tests. Focused existing
Python RNIC checks pass 27 tests with one declared skip; the documentation
format tests pass 14. Ruff passes. The complete upstream Python gate is
recorded in [evidence.json](evidence.json) with its terminal status before
publication. No component result is represented as a request-level result.

## Preserved chronology and scope

An initial arithmetic-horizon fixture was corrected with its failed log
retained. Independent review then exposed receive-only teardown, silent
pending-deadline exhaustion and partial-service counter projection defects;
all three have narrow native/facade hazard fixtures and repaired outcomes.
An initial full Python process terminated with exit 143 and no verdict; its
log remains an incomplete attempt. The owned bounded rerun is separately
recorded. Frozen expectations and historical outcomes were not rewritten.

Generated source pins, grid, traces, compiler logs, mutation verdicts and
compatibility records remain below `${SIMLLM_DATA_ROOT}`. The compact artifact
record contains their relative locations and hashes. Reproduction uses the
[study runner](README.md) and explicit external output root.

The mode supports continuously ready downstream service only. Its public
geometry is a declared implementation contract, not a new silicon calibration.
Framework configuration and selected event propagation into `StepResult`, TTFT
and TPOT remain BACK-76. No RTL matching, physical FPGA, HACC, fabric-wide or
wire interoperability verdict follows from this component study. Downstream
production synchronization requires upstream main landing; original strict
matching bands and failed downstream evidence retain their earlier oracle/pin.
