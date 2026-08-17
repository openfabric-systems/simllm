# GPU device ports v1 results

Run: 2026-08-17. Expectations frozen at commit
`ecd84f8b519eb4ad1f88f5429ee81ec4a1241681`, which contains
[expectations.md](expectations.md) and no implementation, no harness and no
measured value. The implementation, this harness and every number below came
after it. This is preregistration in the repository's sense.

Result: **not void, 11 of 11 scored genuine-risk instances pass, 54 fatal
guards evaluated with none violated.** Evidence classes are reported separately
and are never added into one total. Fatal guards are never reported as a
fraction.

| evidence class | result | scoring |
|---|---|---|
| fatal guards | 54 evaluated, 0 violated | unscored, a single violation voids the run |
| scored behavioral instances | 11 of 11 pass across 4 families | scored |
| identity-path baseline register | 4 instances, 0 missed | unscored, entailed by a fatal guard |
| derived relations | reported | unscored, entailed by S1 and S3 |
| raw observations and run configurations | reported | assert nothing |

## Correction to the first publication

The commit that closed COMP-34, `a3f9baf`, published this study as 15 of 15
scored instances with 47 fatal guards. Both numbers are superseded, by review
rather than by a new measurement, and no measured value changed. History is not
rewritten; this section is the correction of record.

- **The scored denominator was 15 and is 11.** Four of the fifteen cells ran the
  identity path: S1 at a ceiling of 64 bytes per cycle for both transfer sizes,
  and S3 at a ceiling of 16 bytes per cycle for both chunk sizes. Those are the
  mechanism's own values, so the family code declared no ceiling at all, the
  device returned the input architecture object (which a fatal guard asserts),
  and the cell cannot discriminate a single one of the failure modes the freeze
  registered for its family: an inert port, an override reaching the wrong
  direction, a rescoped setup or latency term. The freeze's own rule, that a
  relation which cannot fail once another registered row passes is unscored,
  applies to them. They are retained and still checked as a named unscored
  baseline register, and a miss there still fails the run.
- **One of those four was a duplicate of a guard.** S3 at chunk 64 and ceiling
  16 asserts 328 cycles, which is exactly what the physical-bounds fatal guard
  for that cell asserts, because the freeze registered its interval as
  `(328, 328, 328)`. A floor equal to a ceiling is a degenerate interval: it
  turns a bounds check into an equality, so the same assertion was counted once
  unscored and once scored. This is a defect in the freeze's specification of
  that bound, not in the measurement, and per the freeze-first rule the freeze is
  disclosed rather than edited. The bound held; the accounting did not.
- **The guard headline was 47 and is 54.** The 47 was published with per-section
  subtotals of 13, 1, 21, 3 and 2, which sum to 40 rather than 47. The correct
  subtotals are below, and seven guards were added in the fix round that follows
  this correction. Every guard, frozen or added, is enumerated with its origin in
  the guard section.

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
tables here, and `tests/test_gpu_device_ports.py` locks 53 of the same
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

The peer-link row is where the freeze specified a degenerate interval. Its floor
and its ceiling are both 328, so the guard is an equality rather than a bounds
check, and it therefore also entails the S3 chunk-64 identity-path cell that the
first publication scored. The bound is physically right, the arithmetic is right,
and the specification is still defective: a floor that equals its ceiling leaves
nothing for the measurement to fall inside. The freeze is disclosed rather than
edited, and the affected cell is now an unscored register row.

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
| 4,096 | 64 (no declared ceiling) | 84,000 | 84,000 (register) |
| 4,096 | 32 | 148,000 | 148,000 |
| 4,096 | 16 | 276,000 | 276,000 |
| 16,384 | 64 (no declared ceiling) | 276,000 | 276,000 (register) |
| 16,384 | 32 | 532,000 | 532,000 |
| 16,384 | 16 | 1,044,000 | 1,044,000 |

Four scored of four exact, plus the two identity-path rows marked register, which
are reported and checked but never scored (see the correction section). The 84,
148 and 276 cycle values are also the accepted copy
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
| 64 | 16 (no declared ceiling) | 328 | 328 (register) |
| 64 | 8 | 456 | 456 |
| 64 | 4 | 712 | 712 |
| 128 | 16 (no declared ceiling) | 456 | 456 (register) |
| 128 | 8 | 712 | 712 |
| 128 | 4 | 1,224 | 1,224 |

Four scored of four exact, plus the two identity-path register rows. The 328, 456
and 712 values are rows the accepted
[task-mix study](../gpu_task_mix/RESULTS.md) already published for its C2 sweep,
so the port-declared ceiling reaches that study's own reported completion metric
rather than a lookalike of it. The chunk-64 register row is the one that duplicates
a fatal guard, because the freeze gave that cell a degenerate
`(328, 328, 328)` interval.

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

## Fatal guards, all 54 holding

Reported as a list of what was asserted, never as a fraction. The subtotals add
to the headline exactly, which the first publication's did not.

| guard section | rows | from the frozen clauses | added after the freeze |
|---|---:|---:|---:|
| byte identity of the accepted artifacts | 17 | 11 | 6 |
| mutation sensitivity of the byte lock | 4 | 1 | 3 |
| configuration-time rejection | 28 | 20 | 8 |
| applicability and inertness | 3 | 2 | 1 |
| physical bounds | 2 | 0 | 2 |
| total | 54 | 34 | 20 |

Additions strengthen the run and are unscored either way, but they are not part
of what was frozen, so every one of the twenty is named below. Thirteen were
added while implementing the freeze and were published inside the 47; seven more
came from the fix round, and those are marked.

**Byte identity of the accepted artifacts (17 rows).** Each accepted harness was
driven through the composed device with default ports, i.e. with every port
reading its ceiling out of the mechanism and none declaring one.

- Frozen: `examples/gpu_task_mix/results.csv`, `nccl_convergence.csv` and
  `diagnostics.csv` reproduce byte for byte (3 rows).
- Frozen: `examples/gpu_service_model/results.csv` reproduces byte for byte
  (1 row).
- Frozen: the `mixed_makespan_v1` raw observation record, which carries every
  component and live cycle count, admission cycle and step timestamp of that
  study, is identical between a bare run and a composed run in the same session
  (1 row). That study writes no tracked artifact by the repository's bulk-output
  policy, so the paired run is the available form of the guard. With default
  ports the composed device returns the input architecture object, so this
  comparison is exact by construction rather than a measurement of agreement;
  the measurement that gives it content is its mutation control below.
- Frozen: a device whose ports declare no ceiling returns the input architecture
  object itself, by object identity, for each fixture the freeze named (3 rows).
  That is what makes the clauses above exact rather than approximately equal.
- Frozen: every default port's effective ceiling equals the mechanism parameter
  it wraps exactly and carries `calibration_derived` provenance, for each fixture
  the freeze named (3 rows), so the ports read the mechanism rather than
  decorating it.
- Added: the task-mix harness exits zero through the composed device (1 row).
- Added: its printed evidence counts (38 run configurations, 46 replay
  invocations, 36 of 36 exact rows, 6 of 6 relation families, 17 of 17
  instances, 21 of 21 invariants) are unchanged (1 row). A composed run that
  silently dropped a replay could still write matching CSV bytes.
- Added: the accepted C3 ring row reproduces at 4,397 cycles (1 row).
- Added: the host-link fixture, which the freeze did not name, is carried through
  the same object-identity clause (1 row) and the same ceiling-read clause for
  both of its ports (2 rows).

**Mutation sensitivity of the byte lock (4 rows).** A byte lock that cannot see a
mechanism change is not a lock.

- Frozen: with one declared ceiling at half the peer-link value, the regenerated
  `gpu_task_mix/results.csv` differs from the accepted bytes (1 row). This
  control covers the task-mix lock only. The same control sits in
  `tests/test_gpu_device_ports.py`, so the lock is a test and not only a study.
- Added in the fix round: that mutated run also fails its own accepted rows
  (1 row). Changed bytes alone could mean the harness stopped checking rather
  than that the mutation reached the mechanism.
- Added in the fix round: a declared host ceiling of 32 bytes per cycle makes the
  `gpu_service_model` harness raise on its own copy rows and write no artifact
  (1 row). Until this round that lock had no control at all.
- Added in the fix round: a halved peer ceiling changes the `mixed_makespan_v1`
  record (1 row), which is what turns the exact-by-construction comparison above
  into evidence that the composed device was in the path.

The S1 and S2 families are the mutation evidence for the copy-engine mechanism
on the host fixture, and they are scored. They are not a control for the
`gpu_service_model` artifact lock; that artifact has its own control, listed
above.

**Configuration-time rejection (28 rows).** Every clause raises during
configuration, before any estimate, replay or transfer call.

- Frozen (20 rows): a disabled port carrying a declared ceiling; a declared
  ceiling without the override capability; a capability requested of a disabled
  port; a capability the port does not advertise; each of the three
  transport-control capabilities, whose diagnostic names BACK-48 as the owner of
  making the ABI v2 vocabulary reachable from a non-wire port; an xGMI port,
  whose diagnostic names COMP-35 and the port; a peer-store port on a calibration
  with no NVLink profile; a copy-engine port naming an engine the architecture
  does not declare; a copy-engine port naming a direction the engine does not
  declare; an ingress port carrying a device-to-host copy; a duplicate port
  identifier; two ports claiming one copy direction of one engine; two ports
  claiming the peer-store egress cursor; an unsupported port config version; an
  unsupported device config version; a declared ceiling on the wrong clock; a
  declared ceiling claiming calibration-derived provenance; and NVLink declared
  on a host-link role.
- Added: an enabled port declaring no mechanism capability, so an enabled port
  cannot be silently inert (1 row).
- Added: a port naming the device-to-device copy direction, which stays inside
  one GPU and crosses no port (1 row).
- Added: one bidirectional port over two disagreeing mechanism ceilings (1 row).
  This is the rejection that keeps the measured Grace C2C asymmetry two ports
  instead of one averaged rate.
- Added: the mechanism's own first-use rejection inside
  `CopyEngineServiceModel.estimate` still raises for an absent engine (1 row).
  The freeze required that behavior to stay untouched in prose; this makes it a
  row.
- Added in the fix round: a bidirectional port that names only one direction of
  its own link (1 row). Until this round such a port was granted, and its single
  published ceiling then governed half of what it advertised.
- Added in the fix round: a declared ceiling with no relative uncertainty, a
  declared ceiling with no created date, and an unmeasured rescope that does not
  widen the uncertainty of a calibration claiming measured confidence (3 rows).

**Applicability and inertness (3 rows).**

- Frozen: a disabled port is still reported, with `not_applicable`
  applicability, no ceiling and its declared capabilities visible (1 row), and
  the effective architecture of a device whose only port is disabled is the input
  architecture object itself (1 row).
- Added: the mechanism behind the disabled declaration keeps its accepted timing
  exactly, 84,000 ps against 84,000 ps (1 row).

**Physical bounds (2 rows).** Added: the freeze stated both intervals in prose
outside its numbered guard list, and the harness promotes each into a guard that
the measurement sits inside its first-principles interval. One of the two, the
peer-link bound, is the degenerate `(328, 328, 328)` interval discussed in the
correction section.

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
  device. The registered acceptance that requires a port to move per-request TTFT
  and TPOT belongs to TRAF-45, which packetizes the intra-node leg over these
  ports; reaching those metrics is that task's clause, not this one's.
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
- **F2.** The port layer's most useful rejections were not in the frozen list.
  One port may not read two mechanism ceilings that disagree, and a bidirectional
  port may not name only one direction of its own link. Together they turn the
  measured Grace C2C asymmetry from a modeling hazard into a configuration error.
  The freeze had registered the asymmetry as the reason for two separate ports; it
  had not registered that a single bidirectional port over an asymmetric engine
  must fail closed, nor that a partially declared bidirectional port publishes a
  ceiling for a direction it never named.
- **F3.** A registered relation can be unlosable by accident rather than by
  design. Four of the fifteen originally scored cells ran the identity path
  because their ceiling equalled the mechanism's own value, and one of those was
  byte-identical to a fatal guard whose frozen interval was degenerate. The
  freeze contained the rule that catches this ("a relation which cannot fail once
  another registered row passes is unscored") and the harness still scored them,
  because the entailment check was applied to the relation as written rather than
  to each parameterized instance. Instance-level entailment is the check that was
  missing.
