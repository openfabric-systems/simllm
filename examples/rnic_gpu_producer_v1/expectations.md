# RNIC GPU producer coupling v1 expectations

## Freeze status and metric boundary

This is the expectations-only record for the producer half of BACK-27. It
precedes the producer-task builder, timed `GpuTask` admission, native producer
linkage, every new test and every result-producing run in this study. The
companion runner contains frozen literals, build orchestration and validation
logic. Its `--check-only` path does not import the not-yet-written producer
API, configure CMake, create an output directory or produce an artifact.

This is component evidence. The live path ends at a compute-owned task
completion joined to a native RNIC submission record. It does not yet create
an htsim network event, `CompletionEvent`, `StepResult`, TTFT or TPOT result.
HTSIM-9 is the successor that carries the structural RNIC submission onto the
packet simulator, and CORE-5 is the successor that reduces the resulting
completion stream into TTFT and TPOT. No final-metric claim is made here.

The GPU-owned CQ consumer and runner callback half of the former BACK-27 scope
is outside this producer study. The implementation change must move that work
to a new stable backend task rather than defer it silently.

## External-source audit before freeze

The audit was completed before this freeze against SimLLM base commit
`b74629b4b4da1addda9ff21226cfabf5c09aad87` and the official NVIDIA NCCL
repository at commit `5067397c2676d5aed50042fc39e5c8ee96eb0027`. No producer
coupling implementation or result-producing command was run before this
record.

- NCCL's bundled device-verbs code locates GPU WQE slots and reserves them
  with a GPU-side atomic producer index at
  `src/transport/net_ib/gdaki/doca-gpunetio/include/device/`
  `doca_gpunetio_dev_verbs_qp.cuh:44-103`. It performs GPU-side ordering,
  producer-index update, doorbell-record update and mapped doorbell writes at
  the same file's lines `350-470`.
- NCCL's GPU primitive waits for send readiness, writes the connection FIFO
  size, applies a system-scope fence and publishes the GPU-visible connection
  step at `src/device/prims_simple.h:100-172`.
- The CPU send proxy observes that GPU publication, posts the network send and
  later returns completion credit at `src/transport/net.cc:1352-1452`.
- The accepted SimLLM concurrent service treats `GpuTask.kind` as attribution
  only and prices instructions through shared SM, issue, HBM and NVLink
  resources at base commit
  `simllm/compute/gpu_model.py:553-568,789-845,924-950,975-1049`.
  The existing NCCL builder emits per-channel CTA traces with HBM loads and
  NVLink stores at `simllm/compute/nccl.py:34-142`.
- The accepted native submission record is a read-only WQE projection with
  producer identity and caller-supplied timestamps at
  `simllm/backends/rnic/include/simllm/rnic/submission.h:67-84` and
  `simllm/backends/rnic/src/work_queue.cpp:575-604,1033-1080`.

These sources establish that GPU work produces and publishes descriptors, and
that the CPU proxy consumes a GPU publication. They do not establish a B100
instruction trace, task duration, WQE batching law, SM occupancy, HBM service
time, UAR latency or CPU callback cost. The trace and rates below are synthetic
fixture inputs. They are not a hardware calibration or a claim about NCCL
performance.

## Authority and queue-visit contract

An enabled non-host producer request creates exactly one `GpuTask`. The
compute scheduler is the sole mutable authority for that task's submitted,
eligible, admitted and completion cycles. The native RNIC receives a
versioned, immutable task link only after scheduling. It checks identity and
timestamp agreement and copies the link into each submission record in the
doorbell batch. It never advances or recomputes the task.

For a task link, define:

```text
submitted_cycle <= eligible_cycle <= started_cycle
started_cycle <= finished_cycle <= completed_cycle
task_queue_wait = started_cycle - eligible_cycle
task_service = finished_cycle - started_cycle
```

The v1 producer task has no later compute-side delivery stage, so
`finished_cycle = completed_cycle`. The native record's submission timestamp
must be at least the task completion timestamp. A link with the wrong shape,
wrong GPU owner, empty task identity, nonmonotonic timestamps or completion
after the RNIC submission is rejected before queue, PCIe, memory, record or
caller-clock mutation.

GPU-initiated mode links the task owner to the configured GPU producer.
CPU-proxy mode links it to the configured GPU descriptor writer, not to the
CPU proxy agent. Host-CPU mode accepts no GPU task link and invokes no compute
service. The explicit caller-timestamp path remains valid without a task link
for either non-host shape.

## Synthetic task and occupancy fixture

All fixture time is in integer GPU cycles. The clock is exactly 1 GHz, so one
cycle is exactly 1,000 ps when a link crosses into the native RNIC. The profile
has one SM, one scheduler issue slot, one load/store lane, one control lane,
64 KiB of shared memory, 64 bytes/cycle HBM and NVLink service, and two cycles
of HBM and NVLink return latency. Pipeline instruction latency is one cycle.
There is no random mechanism.

The producer request contains one 64-byte WQE and is eligible at cycle zero.
Its accepted caller deadline is the frozen baseline submission cycle
`B = 16`.

The CPU-proxy task contains one 64-byte GPU descriptor store followed by one
ordered publication instruction. Its isolated completion is exactly
`P_proxy = 4` cycles. The GPU-initiated task contains one 64-byte WQE store,
one 4-byte doorbell-record store and one ordered UAR publication instruction.
Its isolated completion is exactly `P_gpu = 7` cycles. Exact instruction
sequence and byte counters are structural fixture oracles. They are fatal when
wrong but unscored because they are author-defined.

The surrounding kernel has the same CTA and memory shape used by the NCCL
egress model: one HBM load, one dependent NVLink store and 26 dependent ALU
instructions. It completes at exactly `H = 32` cycles. Its shared-memory
demand selects the occupancy cell:

| Occupancy cell | Surrounding shared memory | Producer admission effect |
|---|---:|---|
| `idle` | no surrounding task | producer starts at cycle 0 |
| `half` | 32 KiB | producer co-resides and loses one issue cycle |
| `saturated` | 64 KiB | producer cannot admit before cycle 32 |

Sweep producer shape over `host_cpu_driver`, `cpu_proxy` and `gpu_initiated`,
occupancy over all three cells, and coupling over disabled and enabled. This
gives 18 unique run configurations. Host mode has no producer task in every
cell. Disabled mode has no producer task in every cell. Those configuration-
forced absences are fatal unscored guards.

## Decision-relevant relations

For either non-host shape, let `P` be its isolated producer completion. The
half-occupancy task completion must be exactly:

```text
C_half = P + 1
```

The signed task delay is therefore `+1` cycle for both producer shapes. The
caller slack absorbs it, so the RNIC submission remains at cycle 16. This
relation proves that an admitted producer shares the live issue and HBM
service path with the surrounding HBM-to-NVLink kernel rather than being
priced in isolation.

At saturated occupancy, full-SM residency delays admission until cycle 32:

```text
C_saturated = H + P
submission_cycle = max(B, C_saturated)
submission_delay = max(0, H + P - B)
```

The exact rows are:

| Producer shape | Idle completion | Half completion | Saturated completion | Saturated submission delay |
|---|---:|---:|---:|---:|
| `cpu_proxy` | 4 | 5 | 36 | +20 |
| `gpu_initiated` | 7 | 8 | 39 | +23 |

Both saturated delays must be positive and exact. Failure rejects the design
decision that GPU-side descriptor and WQE production participates in the
concurrent SM service. BACK-27 must then remain open and no runner may claim
that producer cadence reflects surrounding compute occupancy.

In both non-host idle cells, the task completes before `B`, so the complete
canonical submission timeline must be byte-identical to the corresponding
disabled timeline. The band is exactly zero changed bytes. This is distinct
from the exact task-completion controls: it detects accidental double charging
of the accepted caller slack.

The half-occupancy and saturated relations are live scheduler assertions, not
scores for calling `estimate_concurrent` or emitting a chosen instruction
sequence. Task and record identity conservation, exact byte counters, task
kind, zero random draws, queue chronology and native rejection atomicity are
fatal unscored evidence.

## Disabled-path and predecessor identity

Coupling is disabled by default. The disabled path must not call the compute
service, construct a producer task or add an RNIC task link. It returns the
caller-supplied submission timeline byte for byte for all shapes and occupancy
labels. The existing BACK-20 `--study-csv` output must also remain byte-
identical.

The frozen accepted artifact inventory is:

| Artifact | Frozen SHA-256 |
|---|---|
| `examples/rnic_wq_v1/results.csv` | `598f0e10ca4e5a83a9dfb8ed8289e25cdc4c80fc24f92f2f70db967724be5682` |
| `examples/rnic_pcie_v1/results.csv` | `464b92fd5327287db6b5e71a5449add5b893285bc3c4bcdf6a4950355339a5e2` |
| `examples/rnic_device_v1/results.csv` | `7a0b8423d0a99de9538047f307bb7fd2f20c8d19bd408ef90fe02199da868934` |
| `examples/rnic_device_v1/native_tests.csv` | `969963477314bfb723770556a02e4f038c7220820d522ae60dfa8c80744a202d` |
| `examples/rnic_session_records_v1/results.json` | `d83575d1c873d3375bc24819c4d6eca0b85ea3a414fe8578f30262268a39fdf6` |
| `examples/rnic_hostmem_v1/results.csv` | `1bc7bcc8e72b7aef9fda1ed7e6ca2078d60c48a00377cbf8dfded75ff4d2fa53` |
| `examples/rnic_submission_v1/results.csv` | `8f74c6fd92d012f2c70c1c2b09d6f49a4d99bcc35fd418a239f7b577777edbc7` |
| `examples/gpu_service_model/results.csv` | `c6e98d8cdca82d72a0ff82a60f6880246849e327c4de4ff7c59f563d52b03032` |
| `examples/gpu_task_mix/results.csv` | `cc6a6e18d574be9a3fe5f52d1a78b235342d57fbd68595d51883f6840f4c8611` |
| `examples/gpu_task_mix/diagnostics.csv` | `1c3767eef14241cf4e5ccf3bad925c5674101b16631fec36549f850910c3a3b5` |
| `examples/gpu_task_mix/nccl_convergence.csv` | `a45a3dac202f12603fb3aa004db6467f8e194be5451570cdd1244d3f2dea58a2` |

All eleven artifacts must retain exact bytes. This compatibility family is
reported separately from scheduler behavior. A mismatch blocks acceptance
even if every new timing cell passes.

## Evidence accounting

The result report keeps these classes separate:

- 18 run configurations;
- isolated service and background exact-oracle rows;
- two half-occupancy behavioral instances;
- two saturated-cadence behavioral instances;
- two idle non-host timeline-identity instances;
- eleven predecessor artifact identities;
- fatal structural, authority and rejection guards;
- native CTest executables and the full Python regression suite.

`RESULTS.md` must report the genuine-risk fraction separately for every scored
family. Half occupancy is genuine risk because the producer can accidentally
run through an isolated estimate. Saturated cadence is genuine risk because a
caller timestamp can bypass SM admission. Idle identity is genuine risk
because producer service can be charged once in the task and again at the
submission boundary. Artifact identity is genuine risk because new task
timing fields and native record fields can perturb established renderers.

## Registered command and pre-freeze dry run

The local machine configuration must set `SIMLLM_WAVE3_RUN_ROOT` to the
external wave-3 run root. The result-producing command is:

```bash
.venv/bin/python examples/rnic_gpu_producer_v1/run_study.py \
  --out "$SIMLLM_WAVE3_RUN_ROOT/codex/back27_gpu_producer_coupling/back27"
```

Before this freeze, the same command is executed with `--check-only`
appended. That mode parses the complete CLI and validates the 18-cell grid,
closed forms, source pins, artifact inventory, row schemas and external-output
rule. It prints a registry confirmation by design. It does not import the
future producer API, configure CMake, create the output directory or produce
an artifact.
