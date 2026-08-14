# A100 environment qualification v1 results

The reviewed study state is `QUALIFIED`. Slurm job `195283` held every frozen
fatal guard, reported no capability blocker, and completed working Nsight
Systems and Nsight Compute probes on one A100 SXM4 80 GB. This is capability
evidence only. It is not a compute calibration, a production SGLang capture,
dynamic SASS evidence, or an end-to-end TTFT or TPOT result.

This study has no scored behavioral denominator. Its acceptance is one
indivisible state, so structural guards and tool checks are not reported as a
pass fraction.

## Freeze integrity and chronology

The expectations-only commit
`26f8ab4ffd8f78673acdcd0511bfbb5dbab8a25f` preceded the probe
implementation, every Slurm submission, and every profiler invocation. The
[expectations](expectations.md) remained byte-identical through the closure
run, at SHA-256
`dfaca0fafec07be4ce4e400f7363b14bd292aa592403ae4e61395fee9e3e6fa1`.
No observed value was written back into that freeze.

The closure run observed repository commit
`3c829c660ec6d48a627447632ee99bd40f001784`. Its self-contained source bundle
had SHA-256
`210c6a073e242da193587fe9482d3d95cb28e38de44d052623f6fd0c09b1fd47`,
and the submitted Slurm script had SHA-256
`a24381c534d35ea47b04b825302016ea4d021503f251b2935e297174d68774f7`.
The retained wrapper ledger has SHA-256
`f4c64d25281ae2496d857c49947ec58d70e4a1e02e768e600b5fd3373e49b900`.
The retained `qualification.json` has SHA-256
`f18e9025b5bdb67a2b82bfcdb06c8c454689525900a8132fda6a14f19e684dfb`.

Six earlier attempts are retained as failure evidence. They do not count as
near-passes and do not weaken the final state. A serialized state cannot
override a violated frozen fatal guard, which is why job `195271` is reviewed
as `VOID` even though its older runner wrote `QUALIFIED`. A strict post-run
audit also reclassified job `195277` as `VOID`: it queried MIG and allowed
clock policy immediately before profiling but not immediately after it, so
frozen Required Observation 4 was incomplete. Job `195280` added both sides
but took its pre-profile snapshot before compilation and an unprofiled CUDA
launch. That was a campaign baseline rather than the frozen immediate
pre-profiler observation, so strict review also voided that run.

| Job | Observed commit | Result SHA-256 | Reviewed state | Finding |
|---|---|---|---|---|
| `195266` | `cde459f2f207e2efdc5bee232b84c6be16e0b0ce` | none | wrapper failure | The unqualified system Python was 3.6.15, so the runner did not parse and no result directory existed. No probe launched. |
| `195267` | `cde459f2f207e2efdc5bee232b84c6be16e0b0ce` | `324402a279dc62f8313abf21fbacc104b477fda428b941a3d65d485457a93cf7` | `VOID` | The runner treated a scheduler-global GPU token as an in-job selector even though the Slurm task exposed local ordinal `CUDA_VISIBLE_DEVICES=0`. |
| `195269` | `6be0970f0b9411d465e63fca3446aa45ad19715d` | `bcb5ae6cfdf0844e50dcfc09205fd50920e2e833c78ea414a52e83bbdced07a4` | `VOID` | The site `nvidia-smi` rejected the display filter `-d MIG`; no profiler ran. |
| `195271` | `80effe9e7f12832dbe0d23915c3dca7073497124` | `5c3bed7336c3493aa08ed98b23ec1653d242198899fd0369758d8b1a70d6c325` | `VOID` | Nsight Systems wrote its transient stream in a host-global temporary directory, outside the configured result root. This violated a frozen fatal guard after both profilers otherwise completed. |
| `195277` | `5b035be07a3d2153824b422bf53aa345544f33c0` | `ae03bb601f11df60f8745e73d8517986a3abce8088ec1facc18f65a06f6b4ae4` | `VOID` | Both profilers completed, but the runner did not repeat the MIG and allowed-clock policy observations immediately after profiling. This left a frozen required observation incomplete. |
| `195280` | `30d12256d9ff6b1a9579953242805d4a96ab2693` | `4a80296b07e4ddade05a64237abf6b9364a2dbce83b67a8ad32691e334cc8654` | `VOID` | The saved pre-profile state preceded compilation, unprofiled correctness execution and SASS extraction. The intervening CUDA launch made it nonadjacent to the profiler probes. |
| `195283` | `3c829c660ec6d48a627447632ee99bd40f001784` | `f18e9025b5bdb67a2b82bfcdb06c8c454689525900a8132fda6a14f19e684dfb` | `QUALIFIED` | The authoritative pre-state followed unprofiled correctness and SASS and directly preceded Nsight Systems. The post-state followed Nsight Compute. Both profilers completed, exact target metric values were numeric, retained paths stayed confined, and every fatal guard held. |

The corrections after failed attempts are post-specified harness repairs, not
pre-registered assertions. Commit
`6be0970f0b9411d465e63fca3446aa45ad19715d` made
`CUDA_VISIBLE_DEVICES` the fail-closed job-local selector. Commit
`6abce6e76725cd65921101190fb050987ba161e3` replaced the unsupported MIG
display filter with a full UUID-scoped query. Commit
`80effe9e7f12832dbe0d23915c3dca7073497124` required both current and pending
MIG state to be `Disabled`. Commit
`5b035be07a3d2153824b422bf53aa345544f33c0` confined every child home, cache,
configuration, CUDA cache, and temporary directory to the result root. It also
validated the paths declared by Nsight Systems and measured total scratch,
including tool state, after both profilers.

Commit `940e3937b0d6fa37fc544332e8c880364302ee61` made the same
path and permission checks portable to Windows without weakening the POSIX
permission check. Commit `975cfb88faf34c35e633b63e851e8b06ce248ec2`
added the missing post-profile MIG and allowed-clock observations. Commit
`30d12256d9ff6b1a9579953242805d4a96ab2693` made the counter gate parse the
actual Nsight Compute CSV header and require a finite metric value for exactly
one exact target kernel ID. Commit
`3c829c660ec6d48a627447632ee99bd40f001784` retained an initial safety guard
but moved the result-authoritative pre-state after unprofiled correctness and
static SASS, directly before Nsight Systems. It also added a regression that
locks the order through the post-Nsight-Compute state. These are post-specified
harness repairs and do not alter the frozen expectations.

## Allocation and safety outcome

Post-run `sacct` accounting shows that the scheduler granted the frozen
request exactly:

| Resource | Requested and allocated | Observed use |
|---|---:|---:|
| nodes and tasks | 1 node, 1 task | 1 node, 1 task |
| GPU | 1 A100 SXM4 80 GB | 1 A100 SXM4 80 GB |
| CPU | 4 CPUs | 4 CPUs, billing weight 4 |
| host memory | 32 GiB | step maximum RSS 1,211,568 KiB |
| wall limit | 20 minutes | 20 seconds |
| scratch hard ceiling | 10 GiB | 38,098,728 bytes before persistence |
| result hard ceiling | 10 GiB | 3,805,352 bytes before persistence |

The allocation was nonexclusive. Compilation, CUDA execution, and profiling
ran inside the Slurm step. Login-host work was limited to submission,
scheduler queries, compact log reads, and checksum validation. The job
performed no download, installation, model load, clock change, GPU reset, MIG
change, MPS change, privilege escalation, or profiling-policy workaround.

The 20-second runtime was below the conservative 5 to 15 minute planning
estimate. The frozen wall limit was an upper bound, and duration was not a
scored observation. This therefore does not violate the reviewed resource
envelope.

## Capability evidence

The Slurm task exposed one local CUDA device. It was
`NVIDIA A100-SXM4-80GB`, compute capability 8.0, with 81,920 MiB reported
memory. CUDA and `nvidia-smi` agreed on GPU UUID and PCI identity before
execution, and the same UUID remained after profiling. MIG current and pending
states were both `Disabled` immediately before and after profiling. The full
allowed-clock list also remained identical. Persistence mode was `Enabled`,
compute mode was `Default`, and no foreign compute process existed before or
after the probes.

The driver was 565.57.01. The CUDA 12.9.1 environment reported CUDA compiler
12.9.86, `cuobjdump` 12.9.82, Nsight Systems 2025.1.3, and Nsight Compute
2025.2.1. The probe saw CUDA driver API version 12.7 and runtime version 12.9.

All three probe processes, unprofiled, Nsight Systems, and Nsight Compute,
returned status 0 and the same output checksum
`0x92f9c705b0302325`. Each reported zero mismatches for 16,777,216 checked
elements. The Nsight Systems trace contained six target rows matching five
warmups and one measured launch. Every row used grid `65536 x 1 x 1`, block
`256 x 1 x 1`, one device, one context, and one stream. Target durations ranged
from 115.488 to 128.832 microseconds.

Nsight Compute counter access was available. Its selected target launch
reported a numeric 110.752 microsecond duration, 85.83 percent DRAM
throughput, 84.69 percent achieved occupancy, and 108 SMs. The exact target
kernel appeared under one profiler target ID. No counter-policy
denial or empty metric set occurred. Static SASS was nonempty and contained
the exact target kernel in an `sm_80` code object. This is static binary
evidence only and is not a dynamic instruction trace.

The declared Nsight transient stream was
`tool-state/tmp/nsys-report-4a97.qdstrm`, and the final report was
`capture/a100_environment_probe.nsys-rep`. Both paths were validated beneath
the configured result root. Scratch occupied 36,264,473 bytes after Nsight
Systems, 38,017,662 bytes after Nsight Compute, and 38,098,728 bytes at the
wrapper's final pre-persistence check.

An independent ledger audit reconciled all 67 artifact-manifest rows to their
retained regular files with exact size and SHA-256 agreement. It found no
missing manifest entry, hash mismatch, size mismatch, symlink, or special
file. Representative independently rechecked hashes are:

| Artifact | SHA-256 |
|---|---|
| probe source | `47ec8a94f7b29feee34608e0b127eb7ef16ba2a0208205633549acdcb47953af` |
| probe binary | `1ab78cddeed654c47b223a123baa99107b390f1c2d5c52e35c4a0f37c21d0101` |
| static SASS | `d1047b7b2200e17b36e97ed3f1648466ef6537d9528c4efeeda73f298a208aba` |
| Nsight Systems report | `ba0d7df7f17c63528aeea656e20a90fc2689c23ee7459826f40ac0a5557fea2c` |
| Nsight Systems SQLite database | `c0938cfcb60656771dbae1c4e2504af2e770c5ffa059fa29a878b8c360320e2a` |
| exported CUDA trace | `73d1f0461e2d6d608725c4ab79364683bce077ee62228cb95184f51fe09ea112` |
| Nsight Compute log | `2f35ed390938380b093ec06f0565795084e049bc1f6e4c5627d5ab50904a6dad` |

Raw profiler databases, binaries, and bulk traces remain outside Git. This
report retains only the compact result and its integrity ledger.

## Non-scored physical sanity

The probe authors two FP32 reads and one FP32 write for every element, so one
launch has 201,326,592 authored logical bytes, exactly 192 MiB. The built-in
A100 seed uses 2.039 TB/s as peak memory bandwidth. Dividing the logical bytes
by that peak gives 98.738 microseconds, but this is only a conditional all-HBM
serialization reference. It is not a hard physical floor because neither
cache state nor actual DRAM bytes were captured.

The 110.752 microsecond counter duration gives a logical-byte-per-time quotient
of 1.818 TB/s. The 115.488 to 128.832 microsecond activity rows give 1.563 to
1.743 TB/s by the same arithmetic. These are plausibility quotients, not
measured HBM bandwidth. Nsight Compute's 85.83 percent of the 2.039 TB/s seed
is 1.750 TB/s, about 3.7 percent below the logical quotient from its duration.
That is broad cross-tool plausibility, not an exact reconciliation.

The surrounding device state gives a third independent angle. Memory clock
remained at 1,593 MHz, SM clock rose from 1,275 to 1,410 MHz, power rose from
69.64 to 81.43 W, and temperature rose from 26 to 27 degrees Celsius. No
foreign process appeared. Those observations are physically unsurprising for
the small probe, but they do not establish a controlled cell or a transferable
performance parameter. No duration or derived quotient enters a SimLLM
profile.

## Warning and next gate

Nsight Systems emitted one warning that device-side CUDA-event completion
tracing was enabled and could add overhead or false cross-stream dependencies.
The qualification probe used one stream and required only a nonempty CUDA
activity row, so the warning does not invalidate this capability result. The
production SGLang expectations must explicitly set
`--cuda-event-trace=false`, or freeze and defend an alternative, before any
multi-stream dependency evidence is interpreted.

The `QUALIFIED` state opens the next gate but closes neither COMP-5 nor COMP-1.
Pinned NVBit capture, dynamic SASS, Accel-Sim compatibility, controlled-cell
stability, and real production SGLang kernels remain unproven. A separate
expectations-only commit must freeze that production study before its harness
or first run. It should jointly gather the evidence owned by COMP-1, SGL-24,
COMP-6, and SGL-10, starting with the frozen single-A100 decode and prefill
instrumentation pilot before the full matrix.
