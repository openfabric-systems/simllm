# NCCL data, control and timing ownership

The additional time belongs to GPU protocol work above the link. A sender
writes bytes, makes those writes visible, and advertises readiness. The peer
polls readiness before consuming the bytes and eventually returns buffer
credits. A link serializer alone cannot predict these costs or the change
between protocols.

## Source identities

The audit uses NCCL `7b83616df3ae082a1f32bb74c27458bfe8153a13`
(version 2.31.2), nccl-tests `b4d5beebca8a76cf01335f724d154b9b9d394d96`,
and the study's pinned htsim `6efec16c54595e8905d37eb45ed9fc9fd7650be2`.
Raw source snapshots and their hashes accompany the external capture bundle.
No backend submodule source is edited from this repository.

## What the transport and C++ model already do

`simllm/backends/nvlink_runtime.py` owns the causal transport calendar:
port serialization, source supply, receive capacity, credit return,
acknowledgement, replay, ordering and consumer visibility. The historical
`htsim_nvlink.py` module carries compatibility profiles. The native C++ kernel
in `simllm/backends/nvswitch/switch.cpp` arbitrates switch input and output
occupancy; its caller owns queues, bytes, credits and the event calendar.
It does not implement NCCL's collective algorithm. Merlin's measured four-GPU
NV4 and NV6 direct meshes do not pass through that switch kernel.

The exact pinned htsim source tree has no NVLink or NCCL implementation to
amend. The relevant physical model is on the SimLLM side. The new analytic
protocol model therefore plugs into the existing collective latency profile
and StepResult path. The configuration rejects simultaneous physical peer
packet timing. TRAF-54 remains the owner of expanding actual collective
protocol operations into that physical calendar; this analytic projection
does not claim to complete it.

## Software-visible data and control bytes

| Protocol | Application bytes | Inline readiness bytes | Separate control |
|---|---:|---:|---|
| LL, one line | 8 | 8 | Buffer-credit head counter |
| LL128, one line | 120 | 8 | Buffer-credit head counter |
| Simple, one data slice | Slice payload | 0 | Head/tail counters, barriers, data-visibility fence |

LL means low latency. Its two data words each have a readiness flag. LL128
places fifteen data words and one flag in each 128-byte group. Partial groups
consume complete protocol lines. The encoding counts describe NCCL software
buffers, not a newly discovered NVLink packet format. The application endpoint
traffic of a Ring all-reduce is `2(n-1)S/n`, where `n` is the rank count and `S`
is the payload per rank. Source: [protocol line definitions](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/include/device.h)
and [Ring algorithm stages](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/device/all_reduce.h).

Simple's `waitPeer` polls peer counters. `postPeer` publishes a counter after
a system fence when data was stored. Empty slice bodies still perform
synchronization and counter operations. These execute inside the collective's
GPU kernel, so adding a separately launched polling kernel would be incorrect.
Both LL and Simple check reusable-buffer credits. A fresh round trip is not
unconditionally required for every data word. Source: [Simple primitives](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/device/prims_simple.h)
and [LL primitives](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/device/prims_ll.h).

## Source geometry and identifiable costs

The translation preserves thread thresholds, channel maxima, scheduling-cell
rounding, protocol buffer chunk limits and data-work rounds. Logical channels
are work partitions, not physical NVLink ports. The planner's extra weight for
LL is a scheduling heuristic; it must not be counted as four times the physical
payload. Direct callbacks validate channel and warp counts separately from
all timing fits. Sources: [thread and channel selection](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/tuning/tuning_general.cc)
and [work partitioning](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/enqueue/enqueue.cc).

The reference estimate is apparent startup plus the larger of physical byte
service and effective GPU data work, plus synchronization work. GPU supply and
link transfer can overlap; summing them in full failed the retained development
checks. The calibration identifies effective work costs, not an independent
fence time, polling time and propagation time. An optional visibility round
trip partitions the synchronization budget; only an assumed contribution above
that budget adds further time. This prevents charging an acknowledgement twice.
The independent GPU exchange probe illustrates the scale and ordering of a
round trip, rather than measuring NCCL's internal counter latency.

## Why the library switches before the faster measured crossover

NCCL's Ring chooser uses `latency_us + payload_bytes / (1000 * bandwidth_GBps)`
for these single-node Ampere and Hopper all-reduces. The recorded two-GPU
cost tables assign LL and Simple 7.8 and 15.2 microseconds of latency.
Their bandwidth estimates are 40 and 80 GB/s on A100, and 60 and 120 GB/s
on GH200. Those rounded tables cross near 592,000 and 888,000 bytes,
inside the observed selection brackets. These constants are software
estimates, not NVIDIA-published hardware completion measurements.

The fixed-protocol measurements cross later: the first sampled Simple result
no slower than LL is at 1,568 KiB on A100 and 1,120 KiB on GH200. At 1 MiB,
LL and Simple measure 40.29 and 47.84 microseconds on A100, and 27.15 and
29.37 microseconds on GH200. The library has already selected Simple there.
The observed slowdown therefore combines a real protocol cost change with an
overoptimistic selection heuristic. The simulator must reproduce that selected
path rather than silently choosing whichever protocol is fastest after fitting.
Sources: [Ring cost model](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/tuning/ring.cc),
[time formula](https://github.com/NVIDIA/nccl/blob/7b83616df3ae082a1f32bb74c27458bfe8153a13/src/tuning/tuning_general.cc)
and retained communicator cost tables in the earlier diagnostic captures.

## The A100 comparison and a correction

The earlier dense report described both primary measurements as GPU-event
spans. That statement was wrong. The original harness measures CUDA events
around twenty iterations after five warmups, taking the maximum over ranks.
Its host workers are recreated for each payload and reuse fixed buffers.
The pinned NVIDIA benchmark's default `per_iter_timing=0` uses host elapsed
time through collective stream completion, divided by the iteration count.
It uses persistent workers, twenty warmups and one hundred timed iterations
in the retained capture. Its buffer offsets rotate, and its allocation and
data initialization differ. A host span can still be lower than a GPU-event
span from another process when those execution conditions differ.
Source: [benchmark timing and buffer offsets](https://github.com/NVIDIA/nccl-tests/blob/b4d5beebca8a76cf01335f724d154b9b9d394d96/src/common.cu).

NVIDIA's public 600 GB/s A100 figure is aggregate bidirectional NVLink
bandwidth. It is not a one-direction bandwidth to each peer in Merlin's mesh.
Merlin's NV4 pair has four links, or 100 GB/s in one direction; twelve links
across three peers give 300 GB/s aggregate one-direction bandwidth. NVIDIA's
HGX eight-GPU board uses NVSwitch, unlike its four-GPU direct mesh. An A100
label alone therefore does not establish a comparable published latency curve.
Sources: [Ampere tuning guide](https://docs.nvidia.com/cuda/archive/12.1.0/ampere-tuning-guide/index.html)
and [HGX configuration guide](https://docs.nvidia.com/datacenter/tesla/hgx-software-guide/index.html).
The plotted NVIDIA reference is nccl-tests measured on Merlin, not a latency
curve published by NVIDIA. The report keeps the controlled local comparison
separate from this public topology and bandwidth comparison.
