# GPU device ports v1 results

Run: 2026-08-17. Expectations frozen at commit
`ecd84f8b519eb4ad1f88f5429ee81ec4a1241681`, which contains
[expectations.md](expectations.md) and no implementation, no harness and no
measured value. The implementation, this harness and every number below came
after it. This is preregistration in the repository's sense.

Result: **not void, 15 of 15 scored genuine-risk instances pass, 47 fatal
guards evaluated with none violated.** Evidence classes are reported separately
and are never added into one total. Fatal guards are never reported as a
fraction.

| evidence class | result | scoring |
|---|---|---|
| fatal guards | 47 evaluated, 0 violated | unscored, a single violation voids the run |
| scored behavioral instances | 15 of 15 pass across 4 families | scored |
| derived relations | reported | unscored, entailed by S1 and S3 |
| raw observations and run configurations | reported | assert nothing |

## Reproduction

```bash
uv run --extra dev python examples/gpu_device_ports_v1/run_study.py --out <dir>
```

`--out` is required and must sit outside the repository, because the run
regenerates the accepted artifacts in order to compare them. The harness writes
`rows.csv`, `port_reports.csv`, `raw_observations.json` and `summary.json` into
`<dir>`, plus the regenerated accepted artifacts under `<dir>/byte_identity/`
and `<dir>/mutation_control/`. No artifact of this study is tracked, exactly as
for [mixed_makespan_v1](../mixed_makespan_v1/RESULTS.md); every number is in the
tables here, and `tests/test_gpu_device_ports.py` locks 44 of the same
assertions inside `pytest -q`. The study runs in about one second.

## Physical sanity before precision

Stated in the freeze before any value was read:

| bound | floor | ceiling | measured | where it sits |
|---|---:|---:|---:|---|
| host link, 4,096 bytes at 64 bytes per cycle (64 GB/s), ps | 64,000 | 84,000 | 84,000 | exactly at the ceiling |
| peer link, 32 stores of 64 bytes at 16 bytes per cycle, cycles | 328 | 328 | 328 | floor and ceiling coincide |

The host-link floor is 4096 / 64e9 s and no copy can beat it; the ceiling adds
the engine's declared 20-cycle setup, and with one descriptor on an idle engine
there is nothing else to pay. The peer-link floor is 2,048 bytes over 16 bytes
per cycle plus the 200-cycle NVLink latency term, neither of which any store can
beat. Both measurements sit inside their interval, which is a necessary
condition and not proof of correctness; the scaling checks below are what test
the mechanism.

The scaling companion also holds exactly. Halving a declared ceiling doubles
the serialization term and leaves the constant term untouched: the host-link
term moves 64,000, 128,000, 256,000 ps at 4,096 bytes with the 20,000 ps setup
constant, and the peer-link term moves 128, 256, 512 cycles at a 64-byte chunk
with the 200-cycle latency constant. A term that had moved by 1.05 or by 40
would have refuted the port mechanism regardless of how exactly the primary
number matched.

## S1, a host-link port ceiling reaches the end-to-end metric

One `DmaWork` descriptor through `CoarseDeviceRuntime`, with the composed
device's copy service as the runtime's only copy engine, launch and delivery
service zero, and the reported metric the job completion time.

| bytes | port ceiling, bytes per cycle | frozen JCT, ps | measured JCT, ps |
|---:|---:|---:|---:|
| 4,096 | 64 (read from the engine) | 84,000 | 84,000 |
| 4,096 | 32 | 148,000 | 148,000 |
| 4,096 | 16 | 276,000 | 276,000 |
| 16,384 | 64 (read from the engine) | 276,000 | 276,000 |
| 16,384 | 32 | 532,000 | 532,000 |
| 16,384 | 16 | 1,044,000 | 1,044,000 |

Six of six exact. The 84, 148 and 276 cycle values are also the accepted copy
rows of [gpu_service_model](../gpu_service_model/RESULTS.md), whose fixture uses
the same 20-cycle setup and 64 bytes per cycle, so a port-declared ceiling of 32
bytes per cycle reaches exactly the duration that study measured by configuring
the engine directly. The port is a declaration surface over the copy engine, not
a second timing model, and this row pair is the arithmetic proof of that.

## S2, the ceiling stays inside the direction its port carries

With `pcie-host-ingress` declared at 16 bytes per cycle and `pcie-host-egress`
untouched, the device-to-host descriptor keeps its baseline exactly:

| bytes | frozen device-to-host JCT, ps | measured, ps |
|---:|---:|---:|
| 4,096 | 84,000 | 84,000 |
| 16,384 | 276,000 | 276,000 |

Two of two exact. This is the cell that separates a port from an engine knob: a
naive implementation that rescoped the whole copy engine would pass S1 and fail
here. It is also the shape the GH200 envelope study forces, where Grace C2C
measured 419.93 GB/s inbound against 169.96 GB/s outbound, a factor 2.47 apart.
The port layer refuses to average two disagreeing mechanism ceilings into one
bidirectional port at all: that configuration is rejected, and the asymmetry has
to be declared as two ports.

## S3, a peer-link port ceiling moves the egress term

The accepted task-mix C2 cells, replayed through the composed device with a
declared peer ceiling:

| chunk bytes | port ceiling, bytes per cycle | frozen cycles | measured cycles |
|---:|---:|---:|---:|
| 64 | 16 (read from the profile) | 328 | 328 |
| 64 | 8 | 456 | 456 |
| 64 | 4 | 712 | 712 |
| 128 | 16 (read from the profile) | 456 | 456 |
| 128 | 8 | 712 | 712 |
| 128 | 4 | 1,224 | 1,224 |

Six of six exact. The 328, 456 and 712 values are rows the accepted
[task-mix study](../gpu_task_mix/RESULTS.md) already published for its C2 sweep,
so the port-declared ceiling reaches that study's own reported completion metric
rather than a lookalike of it.

## S4, the accepted ring cell under a halved peer ceiling

| quantity | frozen | measured |
|---|---:|---:|
| accepted C3 ring row at 16 bytes per cycle, cycles | 4,397 | 4,397 |
| the same ring at a declared 8 bytes per cycle, cycles | band [8,392, 8,493] | 8,493 |

One of one inside the band, and the finding is where inside. The band's floor
was the egress bound at the halved ceiling (1,024 stores times 8 cycles plus the
200-cycle latency) and its ceiling was the published baseline plus the 4,096
cycles of serialization the halving adds. The measurement landed exactly on the
upper edge, so the additive argument holds with equality: at eight warps per
channel the ring is already egress-bound to within 101 cycles of its own bound,
and every added egress cycle lands on the critical path with nothing left to
hide it. Recorded as finding F1. The 101-cycle band was narrow, 1.2 percent of
its floor, so this instance was genuinely losable, and it resolved to a point
prediction rather than to an interval.

## Fatal guards, all 47 holding

Reported as a list of what was asserted, never as a fraction.

**Byte identity of the accepted artifacts (13 guards).** Each accepted harness
was driven through the composed device with default ports, i.e. with every port
reading its ceiling out of the mechanism and none declaring one.

- `examples/gpu_task_mix/results.csv`, `nccl_convergence.csv` and
  `diagnostics.csv` reproduce byte for byte, the harness exits zero, and its
  printed evidence counts (38 run configurations, 46 replay invocations, 36 of
  36 exact rows, 6 of 6 relation families, 17 of 17 instances, 21 of 21
  invariants) are unchanged.
- `examples/gpu_service_model/results.csv` reproduces byte for byte.
- The `mixed_makespan_v1` raw observation record, which carries every component
  and live cycle count, admission cycle and step timestamp of that study, is
  byte-identical between a bare run and a composed run in the same session. That
  study writes no tracked artifact by the repository's bulk-output policy, so the
  paired run is the available form of the same guard.
- The accepted C3 ring row reproduces at 4,397 cycles.
- For all four fixtures, a device whose ports declare no ceiling returns the
  input architecture object itself, by object identity. That is what makes the
  clauses above exact rather than approximately equal.
- Every default port's effective ceiling equals the mechanism parameter it wraps
  exactly and carries `calibration_derived` provenance, so the ports are reading
  the mechanism rather than decorating it.

**Mutation sensitivity (1 guard).** With one declared ceiling at half the
peer-link value, the regenerated `gpu_task_mix/results.csv` differs from the
accepted bytes. Without this guard the byte locks above would pass for a device
that was never in the path at all. The same negative control sits in
`tests/test_gpu_device_ports.py`, so the lock is a test, not only a study.

**Configuration-time rejection (21 guards).** Every clause raises during
configuration, before any estimate, replay or transfer call: a disabled port
carrying a declared ceiling; a declared ceiling without the override capability;
a capability requested of a disabled port; a capability the port does not
advertise; each of the three transport-control capabilities, whose diagnostic
names BACK-48 as the owner of making the ABI v2 vocabulary reachable from a
non-wire port; an xGMI port, whose diagnostic names COMP-35 and the port; a
peer-store port on a calibration with no NVLink profile; a copy-engine port
naming an engine the architecture does not declare; a copy-engine port naming a
direction the engine does not declare; an ingress port carrying a device-to-host
copy; a port naming the device-to-device copy direction, which stays inside one
GPU and crosses no port; a duplicate port identifier; two ports claiming one
copy direction of one engine; two ports claiming the peer-store egress cursor;
an unsupported port config version; an unsupported device config version; a
declared ceiling on the wrong clock; a declared ceiling claiming
calibration-derived provenance; NVLink declared on a host-link role; an enabled
port declaring no mechanism capability; and one bidirectional port over two
disagreeing mechanism ceilings.

The last of those was added during implementation and was not in the frozen
list. It is a guard, so it is unscored either way, and it is disclosed here
rather than folded silently into the frozen twenty. The mechanism's own
first-use rejection inside `CopyEngineServiceModel.estimate` is untouched and is
guarded separately.

**Applicability and inertness (3 guards).** A disabled port is still reported,
with `not_applicable` applicability, no ceiling and its declared capabilities
visible; the effective architecture of a device whose only port is disabled is
the input architecture object itself; and the mechanism behind the disabled
declaration keeps its accepted timing exactly (84,000 ps against 84,000 ps).

**Physical bounds (2 guards).** Each measurement sits inside its
first-principles interval, as tabulated above.

## What this does not show

- No packet crosses a port. The ports carry protocol identity, direction,
  ceiling, capabilities and provenance, and they negotiate by rejecting what
  they do not advertise. Emitting a packet attempt, a TX boundary or an arrival
  in the ABI v2 vocabulary needs that vocabulary reachable from a non-wire port,
  which is BACK-48, and the compute-side binding is registered as COMP-40.
- The end-to-end metric reached here is the job completion time of a fixed DMA
  task through `CoarseDeviceRuntime`, not TTFT or TPOT. The chain is the live
  one (input, `ExecutionGraph`, runtime authority, queue visits, completion),
  and it is the same chain the accepted `core4_runtime` study reports against,
  but no request-metric reduction runs here and no step sink selects a composed
  device. A study that wants a port ceiling to move TTFT has to select the
  device from a step path, which is COMP-25's shape of problem, not this one's.
- No shipped architecture profile carries a measured per-port ceiling. Every
  ceiling here is either read out of a synthetic study calibration or declared by
  the study itself with `model_configuration` provenance. Attaching the measured
  A100 and GH200 port ceilings of the design statement's port taxonomy to a
  shipped profile is registered as COMP-41.
- No AMD cell runs. An xGMI port is nameable and is rejected with a diagnostic
  naming COMP-35, which keeps vendor instantiation where it belongs.
- Peer topology, per-link routing, ingress service and reduction lanes are
  untouched. They stay with COMP-31, and this study adds no term to any of them.
- Both fixtures are synthetic 1 GHz mechanism fixtures. Nothing here is a
  silicon claim, and no calibration changed.

## Findings

- **F1.** Halving the peer-link ceiling of the accepted ring cell adds exactly
  the full serialization delta, 4,096 cycles, to the published 4,397-cycle
  duration. The frozen band allowed anything from the pure egress bound to that
  additive value, and the mechanism chose the additive edge, because at eight
  warps per channel the kernel has no slack left to absorb a slower cursor. Any
  later registration that assumes a slower egress cursor is partly hidden by
  overlap is wrong in this regime.
- **F2.** The port layer's most useful rejection was not in the frozen list. One
  port may not read two mechanism ceilings that disagree, which turns the
  measured Grace C2C asymmetry from a modeling hazard into a configuration
  error. The freeze had registered the asymmetry as the reason for two separate
  ports; it had not registered that a single bidirectional port over an
  asymmetric engine must fail closed.
