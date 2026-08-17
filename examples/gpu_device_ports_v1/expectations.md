# GPU device ports v1 expectations

This expectations-only change precedes the COMP-34 implementation and every run
of this study. The results report must cite the commit that first contains this
file. No implementation, generated row or measured value is part of this
freeze. Every number below is either a prediction derived from a closed form
stated here, or a value already published by an accepted study and quoted as
the identity target.

## Scope and evidence boundary

The study validates one versioned GPU device composition entry point with typed
ports over mechanisms that already exist and stay authoritative:

- the per-direction copy-engine profiles (`CopyEngineProfile`,
  `CopyDirectionProfile`, `CopyEngineServiceModel`) for the host link, and
- the flat per-GPU NVLink egress cursor (`NvlinkProfile`) plus the peer copy
  direction for the peer link,

both in `simllm/compute/gpu_model.py`. The port object adds protocol identity,
direction, ceiling, declared capabilities, the provenance of that ceiling and
configuration-time rejection. It adds no second timing mechanism. Peer
topology, per-link routing, ingress service and reduction lanes stay with
COMP-31 and are outside this study. Vendor instantiation stays with COMP-35: an
xGMI port is nameable but must fail closed, and no AMD cell runs here.

Two port semantics are frozen here because both are easy to get wrong:

1. **A disabled port is a declaration that is absent, not a mechanism that is
   off.** Disabling a port never rescopes the underlying copy engine or egress
   cursor: the pre-existing mechanism stays exactly as it is and stays
   authoritative, the port keeps its interface and is still reported, its own
   parameters are inert, and any request made of the port is rejected with a
   diagnostic naming it. A disabled port that silently accepted a declared
   ceiling, or that silently removed a mechanism, is the defect this clause
   exists to forbid.
2. **A ceiling belongs to a port, and reading one is not declaring one.** A
   port with no declared ceiling reads its effective ceiling out of the
   mechanism it wraps, and carrying that value must change nothing. A port with
   a declared ceiling replaces the mechanism parameter for the directions that
   port carries, and only those.

Evidence classes stay separate and are never added into one total:

- **fatal guards** (byte identity, configuration-time rejection, applicability,
  mutation sensitivity): a single violated guard voids the run. Guards are
  never reported as a fraction.
- **scored behavioral instances**: the genuine-risk predictions listed under
  "Scored denominator" below.
- **raw observations and run configurations**: reported, assert nothing.

Relations that cannot fail once another registered row passes are recorded as
derived and are unscored. The entailment answer for every scored row is given
below.

## Fixtures

### Fixture H, the host link

A synthetic 1 GHz architecture with one copy engine `ce0` at 1 GHz declaring
exactly two directions, both with `setup_cycles = 20` and
`bandwidth_bytes_per_cycle = 64`:

```text
HOST_TO_DEVICE   setup 20 cycles, 64 bytes per cycle
DEVICE_TO_HOST   setup 20 cycles, 64 bytes per cycle
```

The calibration carries no NVLink profile, so a peer-store port on this fixture
must fail closed. Two ports are declared on it:

```text
pcie-host-ingress   PCIe, host link, ingress, copy direction HOST_TO_DEVICE
pcie-host-egress    PCIe, host link, egress,  copy direction DEVICE_TO_HOST
```

`ingress` means entering the GPU and `egress` means leaving it. The two ports
exist separately because the measured host link is not symmetric everywhere:
the GH200 envelope study measured Grace C2C at 419.93 GB/s inbound against
169.96 GB/s outbound, so one bidirectional host rate is wrong on that machine
in one direction. This fixture makes no C2C claim; it only keeps the two
directions independently addressable.

The live chain is the accepted CORE-4 one:
`ExecutionGraph` with one `DmaWork` operation on rank 0, executed by
`CoarseDeviceRuntime` with
`CoarseDeviceProfile(launch_service_ps = 0, completion_delivery_ps = 0,
copy_engines = (the device's copy-engine service,))`, and the reported metric is
the job completion time `JCT = result.completed_at_ps - graph.released_at_ps`.
With a released-at time of zero, no dependency and no launch or delivery
service, the runtime's DMA visit starts at zero, so

```text
JCT_ps = (setup_cycles + ceil(bytes / bandwidth_bytes_per_cycle)) * 1000
```

### Fixture P, the peer link

The accepted task-mix fixture, imported from
`examples/gpu_task_mix/run_gpu_task_mix.py` so that the cells are the study's
own and not a lookalike: `architecture(nvlink_bandwidth = bw)` and
`egress_launch(warps = 8, per_warp = 4, chunk_bytes = chunk)`, which issues 32
NVLink stores of `chunk` bytes against one flat egress cursor with a 200-cycle
NVLink latency. One port is declared on it:

```text
nvlink-peer-store   NVLink, peer link, egress, peer-store egress capability
```

The closed form the accepted study already published for these cells is

```text
duration_cycles = 32 * ceil(chunk / bw) + 200
```

The ring cell uses the accepted C3 configuration:
`nccl_ring_allreduce_launch(payload_bytes = 65536, world_size = 2,
channels = 2, chunk_bytes = 64, warps_per_channel = 8)` on
`architecture(sm_count = 2)`, whose published duration is 4,397 cycles against
its own 4,296-cycle egress bound.

## Physical bounds, stated before any measurement

- Host link, 4,096 bytes at 64 bytes per cycle at 1 GHz, i.e. 64 GB/s: the
  serialization floor is 4096 / 64e9 s = 64,000 ps and no copy can beat it. The
  engine is a single server with no contention in this graph, so the ceiling is
  the floor plus the declared 20-cycle setup, 84,000 ps. The measurement must
  sit exactly at that ceiling; anything below the floor is a defect.
- Peer link, 32 stores of 64 bytes = 2,048 bytes at 16 bytes per cycle: the
  serialization floor is 128 cycles, and the 200-cycle link latency is a floor
  no store can beat, so 328 cycles is both floor and ceiling for that cell.
- Halving a ceiling must move a serialization-bound term by exactly two and
  must leave a latency or setup term untouched. A term that moves by 1.05 or by
  40 refutes the mechanism claim regardless of how exactly the primary number
  matched.

## Scored denominator

Fifteen genuine-risk instances, in four families. The measured value is
compared against the frozen expectation with no tolerance unless a band is
named.

### S1, host-link ceiling override reaches the end-to-end metric (6 instances)

The `pcie-host-ingress` port carries a declared ceiling; the JCT of the
host-to-device `DmaWork` is read from the live runtime.

| bytes | port ceiling, bytes per cycle | expected JCT, ps |
|---:|---:|---:|
| 4096 | 64 (no declared ceiling, read from the engine) | 84000 |
| 4096 | 32 | 148000 |
| 4096 | 16 | 276000 |
| 16384 | 64 (no declared ceiling, read from the engine) | 276000 |
| 16384 | 32 | 532000 |
| 16384 | 16 | 1044000 |

Entailment: losable. The relation fails if the port is inert, if the override
reaches the wrong direction or the whole engine, if the 20-cycle setup term is
rescoped by the override, or if the ceiling arithmetic drops the ceiling
division.

### S2, the override stays inside the direction its port carries (2 instances)

With `pcie-host-ingress` overridden to 16 bytes per cycle and
`pcie-host-egress` untouched, the device-to-host `DmaWork` JCT must stay at its
baseline value exactly:

| bytes | expected device-to-host JCT, ps |
|---:|---:|
| 4096 | 84000 |
| 16384 | 276000 |

Entailment: losable. A port implementation that rescopes the engine rather than
the directions the port carries fails this and passes S1.

### S3, peer-link ceiling override moves the egress term (6 instances)

The `nvlink-peer-store` port carries a declared ceiling; the measured value is
the replay duration in cycles.

| chunk bytes | port ceiling, bytes per cycle | expected duration, cycles |
|---:|---:|---:|
| 64 | 16 (no declared ceiling, read from the profile) | 328 |
| 64 | 8 | 456 |
| 64 | 4 | 712 |
| 128 | 16 (no declared ceiling, read from the profile) | 456 |
| 128 | 8 | 712 |
| 128 | 4 | 1224 |

The 328, 456 and 712 cells are the accepted C2 rows of
`examples/gpu_task_mix/RESULTS.md`, so an override that reaches the study's own
reported completion metric lands on a value that study already published.

Entailment: losable. The relation fails if the port is inert, if the override
also moves the 200-cycle NVLink latency term, or if the egress cursor is not
the term the port ceiling governs.

### S4, the ring cell under a halved peer ceiling (1 instance)

With `nvlink-peer-store` overridden from 16 to 8 bytes per cycle, the accepted
C3 ring replay must land inside

```text
[8392, 8493] cycles
```

The floor is the egress bound at the overridden ceiling, 1024 stores times 8
cycles plus the 200-cycle latency. The upper edge is the published 4,397-cycle
baseline plus the 4,096 cycles of serialization the override adds: the loads,
the issue path and the latency terms are unchanged, and every added cycle can
at worst land on the critical path. Exceeding the upper edge means the halved
ceiling compounded through queueing rather than adding, which is a finding
about the mechanism, and this instance is then scored as a miss rather than
treated as a guard.

Entailment: losable, and the band is narrow (101 cycles on 8,392, i.e. 1.2
percent).

### Derived, therefore unscored

Once S1 and S3 pass, these follow by arithmetic and are recorded without being
scored: the host-link serialization term (JCT minus 20,000 ps) scales as
exactly 1 over the ceiling ratio across 64,000, 128,000 and 256,000 ps at 4,096
bytes and across 256,000, 512,000 and 1,024,000 ps at 16,384 bytes; the
peer-link serialization term (duration minus 200 cycles) scales the same way
across 128, 256 and 512 cycles at a 64-byte chunk and across 256, 512 and 1,024
cycles at a 128-byte chunk; and both setup and latency terms stay constant.

## Fatal guards

A violated guard voids the run. The owning task stays open, the evidence is
retained, and no behavioral fraction is reported.

### Byte identity of the accepted artifacts

Each accepted study is driven through the composed device with its default
ports, i.e. with every port reading its ceiling out of the mechanism and no
port declaring one.

1. `examples/gpu_task_mix/results.csv`, `nccl_convergence.csv` and
   `diagnostics.csv` reproduce byte for byte.
2. `examples/gpu_service_model/results.csv` reproduces byte for byte.
3. `examples/mixed_makespan_v1` has no tracked artifact, by the repository's
   bulk-output policy, so its raw observation record is produced twice in the
   same session, once through the bare mechanism and once through the composed
   device, and the two serializations are compared byte for byte. That record
   carries every timestamp, counter and cycle count of the study's component and
   live families.
4. For all three fixtures, the composed device's effective architecture is the
   input architecture object itself, by object identity, when no port declares a
   ceiling. This is what makes clause 1 to 3 exact rather than approximately
   equal.
5. Every default port's effective ceiling equals the mechanism parameter it
   wraps exactly, and carries `calibration-derived` provenance. A port that read
   nothing would satisfy clauses 1 to 4 while being decorative.

### Mutation sensitivity of the lock

6. With one declared ceiling at half the peer-link value, the regenerated
   `examples/gpu_task_mix/results.csv` must differ from the accepted bytes. A
   byte lock that cannot detect a mechanism change is not a lock. This guard
   fails if the bytes are equal.

### Configuration-time rejection

Every clause below must raise during device configuration, before any estimate,
replay or transfer call. This is the acceptance clause that a disabled port and
an unadvertised capability "both reject at configuration time rather than at
first use".

7. A disabled port carrying a declared ceiling is rejected.
8. A declared ceiling on a port that does not advertise the ceiling-override
   capability is rejected.
9. Requesting a capability of a disabled port is rejected.
10. Requesting a capability the port does not advertise is rejected.
11. Declaring a wire-transport control capability (ECN marking, PFC, congestion
    notification) on a GPU port is rejected, with a diagnostic naming BACK-48 as
    the owner of making that vocabulary reachable from a non-wire port. No
    mechanism exists behind those capabilities today, so the port fails closed
    rather than advertising something it cannot do.
12. An xGMI port is rejected, with a diagnostic naming COMP-35 as the owner of
    vendor instantiation and naming the port. Naming the protocol is not
    instantiating it.
13. A peer-store port on a calibration with no NVLink profile is rejected. A
    port with no measured or declared profile fails closed.
14. A copy-engine port naming an engine the architecture does not declare is
    rejected.
15. A copy-engine port naming a direction its engine does not declare is
    rejected at configuration time. The existing first-use rejection inside
    `CopyEngineServiceModel.estimate` stays exactly as it is.
16. A port whose declared copy directions contradict its own direction, for
    example an ingress port carrying a device-to-host copy, is rejected.
17. Duplicate port identifiers are rejected.
18. Two enabled ports claiming the same copy direction on the same engine are
    rejected, and two enabled ports claiming the peer-store egress cursor are
    rejected. One mechanism has one port authority.
19. An unsupported configuration version, on the device config or on any port
    config, is rejected.
20. A declared ceiling whose clock does not match the mechanism's clock is
    rejected, and a declared ceiling claiming `calibration-derived` provenance is
    rejected. Provenance of a declared value is never the mechanism's own.
21. A protocol that does not belong to the port's role, for example NVLink on a
    host link, is rejected.

### Applicability and inertness

22. A disabled port is still reported, with a not-applicable applicability, a
    ceiling of none, and its declared capabilities visible. The effective
    architecture of a device whose only port is disabled is the input
    architecture object itself, so disabling a port rescopes nothing.

## Open decision, to be answered by the implementation

The design statement leaves open whether a dedicated `simllm.device` module is
carved out, and requires the implementing change to record the answer with its
reasoning. The test applied is the one that document states: whether the port
objects can be expressed inside the existing compute and backend surfaces
without duplicating the packet vocabulary and without importing across the
Python and C++ boundary in a new direction. This freeze commits to applying that
test and to recording the answer and its reasoning in
`docs/design/packet-device-model.md`; it does not pre-announce the answer,
because the criterion is only answerable with the first port implementation in
hand.

## Reproduction

```bash
uv run --extra dev python examples/gpu_device_ports_v1/run_study.py --out <dir>
```

The harness exits non-zero when any fatal guard fails or any scored instance
misses. Guard failure prints a void banner and suppresses the behavioral
fraction.
