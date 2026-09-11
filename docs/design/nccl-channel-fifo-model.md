# Source-derived NCCL channel FIFO model and identification plan

Implement each communication channel as finite state that produces, consumes,
and reuses protocol slots while competing for GPU and link service. Completion
follows the executed dependency graph. The
[article summary and primary-source review](../papers/nccl-channel-fifo-source-review.md)
supplies the guidance and version corrections. The
[channel execution study](../../examples/nccl_channel_fifo_v1/RESULTS.md)
implements the first buffered-write Ring slice and the shared-resource checks
in steps 1 and 2. Source/hardware qualification, cost identification and a
fresh uncertainty-band validation retain their separate obligations below.

## Required outcome and ownership

| Maintainer requirement | Concrete implementation obligation | Owning task |
|---|---|---|
| Model at FIFO-per-channel level | Persistent connection sequence, finite slots, partial payloads, readiness and reuse events for every active channel | TRAF-54 |
| Decompose composite overhead | Independently identified startup, GPU work, memory service, polling, barrier and publication costs, with explicit SM and protocol dependence | TRAF-54 mechanism; TRAF-43 accuracy; COMP-44 host terms |
| Model the source code | Versioned primitive transitions and branch predicates checked against the selected NCCL source and observer captures | TRAF-54 |
| Shaded curve covers measured points | Propagate frozen parameter uncertainty through the same execution graph and score untouched validation data | TRAF-43 |

The first implementation covers Ring sum all-reduce of 32-bit floats on the
separately identified A100 NV4 and GH200 NV6 direct meshes, with two and four
participants and LL, LL128 and Simple. Keep this slice explicit. TRAF-54 also
retains its wider collective and RCCL scope; RCCL is AMD's collective library.
TRAF-92 owns later switched-board and workload qualification. Neither a GIN
backend nor an NVSwitch is inserted into the Merlin local path.

## Existing evidence defines the defect, not the new calibration

Physical floor: a four-rank Ring moves `2(n-1)S/n` useful bytes per rank, so
at `S = 3497984` bytes and aggregate one-direction capacity 300 GB/s its
application-byte floor is 17.48992 microseconds. Protocol bytes and actual
route bottlenecks can raise that floor.

Physical ceiling: there is no finite latency ceiling without a bound on stalls.

The retained five-repeat medians sit above that floor. The reference method
measures the host launch loop through stream completion; the original harness
uses GPU events. They are separate execution contexts:

| A100, four GPUs, LL128 | Measured median | Current model | Interquartile repeat span |
|---|---:|---:|---:|
| Reference benchmark, 3,497,984 bytes | 60.06712 us | 54.434191 us | 0.10368 us |
| Original GPU-event harness, same payload | 67.32799858 us | 59.086009 us | 0.87039918 us |

Across 3,170,304 to 3,497,984 bytes, the reference prediction remains
54.434191 microseconds while the measured median rises from 56.98235 to
60.06712 microseconds. Source-derived active channels increase from 22 to 24;
the maximum per-channel work stays fixed and partially filled channels change.
The current maximum-work operator suppresses that additional shared work.
This motivates the replacement; it does not identify one unique hardware cause.
The two timer methods also differ systematically, so their difference is not
automatically GPU kernel service. Source:
[retained prediction rows](../../examples/nccl_protocol_model_v1/measurements/independent_predictions.csv)
and [model implementation](../../simllm/traffic/collective_protocol.py).

The [dense protocol capture](../../examples/nccl_transition_v1/RESULTS.md)
identifies LL-to-LL128 on four-GPU A100, whereas the two-GPU transition is
LL-to-Simple. Model both source paths; an extra Simple latency cannot explain
the four-GPU LL128 residual.

All previously inspected captures, including that independent grid, become
retrospective diagnostics for the next revision. Preserve their original
reports and chronology. Freeze new identification cells before implementation
or measurement, and freeze the fitted model before accessing new validation.
This planning document itself is not an expectations-only commit.

## One execution authority and a narrow integration boundary

Extend the existing `NvlinkCausalEngine` event calendar, its physical binding,
and the live peer path in `simllm/backends/step_sink.py`. Its current retained
phase API accepts peer writes; instruction-level readiness requires resumable
protocol admission and completion callbacks on that same calendar. Add that
seam explicitly instead of precomputing whole-channel finish times. Reads and
control transactions required by a selected branch need real request/response
dependencies through the existing transaction vocabulary before qualification.

| Object | Sole mutable owner | Read-only projection |
|---|---|---|
| Communicator, channel, connection, slot sequences and protocol program position | Protocol state owned by the retained runtime | Source-bound operation and FIFO observations |
| Resident blocks, ready warp work and shared GPU service | GPU resource state on the same calendar | Resource visits and issued/completed work |
| Physical packets, link service, hardware credits and receiver visibility | Existing physical transport state | Packet and buffer-claim observations |
| Collective completion and request latency | Existing graph completion path | `CompletionEvent`, `StepResult`, time to first token (TTFT), time per output token (TPOT) |

Logical channel IDs, physical link IDs, and hardware virtual channels remain
different identities. A channel binds to its captured peer connection and
resolved physical path set, which may use several links. The native C++ switch
allocator is relevant only when a switched topology selects it; BACK-74 owns
its extension. The pinned htsim tree has no source-level NCCL channel execution model to copy.
The existing transport is reused rather than supplemented by a second timer.

The old analytic `NcclRingProtocolModel` remains an explicit, exact bypass.
The existing `simllm.compute.nccl_stack` zero-time source-name skeleton stays
an observation surface and does not acquire a competing clock. When structural
execution is enabled, aggregate analytic collective charging is zero. Disabled
execution preserves the accepted timestamps, bytes, ordering and metrics.

## Per-channel FIFO contract

Use stable connection identity `(communicator, channel, source rank,
destination rank, connection index)`. The pinned `ncclConnInfo` has separate
protocol buffer pointers but shared head/tail pointers and a connection step.
Keep protocol-specific buffer/flag state beneath that connection; switching
LL, LL128 and Simple must not reset or fork its progress counters. Preserve
the source's constructor alignment and final step writeback. An operation ID
identifies work within that lifecycle, not a new FIFO. Store absolute sequence
numbers; use modulo eight only for slot addresses.

Each reservation records the source primitive, ring iteration, chunk, slice,
slot sequence, step increment, useful byte range, encoded byte range, active
lane mask and connection mode. Distinguish the following transitions:

1. Submit work when its program-order predecessor permits issue.
2. Make a reservation eligible only after the required buffer space and
   source dependencies exist. Preserve the source's send/receive wait order.
3. Grant GPU execution and memory work to the active lanes. Partial work
   consumes its actual data service and the source-prescribed control work.
4. Issue peer transactions as their producing work permits, with readiness
   information ordered exactly as the selected protocol requires.
5. Let receiving lanes observe the required flags or counters, then perform
   their source load, reduction, output write or forwarding operation.
6. Release the consumed software slot and publish reusable capacity. A remote
   producer can reuse it only after that update becomes visible to its poll.
7. Complete the collective after all required output writes, observations and
   kernel dependencies finish. The last payload packet alone is insufficient.

Track producer-reserved sequence, receiver-consumed sequence, and the
producer's cached view of returned capacity separately. For a reservation of
`q` steps, the buffer-space predicate is `cached_head + 8 >= producer_step + q`.
The allocated interval cannot exceed capacity even before its bytes arrive.
Actual producer and consumer counters remain monotone; an out-of-date cached
head can stall but cannot grant capacity early. Data-ready and buffer-reuse
notifications are distinct causal edges.

| Protocol path | Source-derived work and control |
|---|---|
| LL | Two data/flag pairs per 16-byte software line; receive flag checks; reusable-slot head polling; barriers; step increments and prescribed flag-wrap cleanup |
| LL128 | 120 data bytes plus 8 flag bytes per line; actual warp load/shuffle/reduce/store order; partial-lane masks; head polling and barriers; tail/fence only when the connection and architecture branch requires it |
| Simple | Separate data and progress counters; `waitPeer`, worker barrier, data/reduction work, full barrier and `postPeer`; two steps per Ring slice, two slices per chunk; empty slices still execute their applicable control path |

Use the default allocation arithmetic in the source review, then apply actual
communicator overrides. Do not charge every reserved slot as a full data
transfer. For buffered Simple read placement, direct-read/direct-write or registered-buffer branches,
retain their synchronization while omitting copies the source omits. Reject
an unimplemented selected branch explicitly instead of substituting a generic
FIFO-copy path.

NCCL software capacity, NVLink receive credits and link acknowledgements are
three separate lifecycles. Inline flags count once in encoded data. Separate
counter operations enter the physical transaction layer once. Repeated polling
loads consume GPU issue and memory service, but one poll is not assumed to
produce one full NVLink packet; cache/coalescing behavior needs identification.

## GPU resources and decomposed service

An SM executes resident thread blocks, issuing work from ready warps, which
are groups of 32 threads. A block waiting on a peer can retain its registers
and shared memory. Other ready warps can proceed, and all channels still share
the relevant memory and physical-link resources. Preserve this overlap through
events, rather than applying one maximum to the complete collective.

Record actual block threads, registers and shared-memory footprint. Admission
uses available SMs and per-SM block, warp, register and shared-memory limits.
Channel count `C`, available SM count `M`, resident blocks and active warps are
separate quantities. Do not assign one SM to every channel by definition.
Source-controlled instruction groups and masks determine work; a supported
architecture profile supplies independently identified service parameters.

| Term | Parameter and dependence | Identification contrast |
|---|---|---|
| Host submission and kernel entry | Per-invocation/per-launch cost and launch mode | Null launch and small positive collectives, paired host/device boundaries; COMP-44 owns attribution |
| Block setup | Cycles per participating block and protocol setup path | Small work at fixed threads, changing block count and available SMs |
| Local loads and output stores | Bytes and transactions through shared memory-system service | Same instruction pattern, cache-resident versus larger working sets, read/write traffic recorded |
| Reduction and pack/shuffle work | Executed instruction groups, active lanes, protocol, datatype, measured GPU clock | Copy-only versus reduction-bearing source primitives with matched byte traffic |
| Polling | Poll issue cycles and observation cadence, by address path | Already-ready peer versus controlled receiver delay; count checks and separate waiting from execution |
| Barrier and publication | Participating warps, scope, selected fence/counter branch | Source-faithful empty/nonempty slices and publication microprobes with pre-satisfied dependencies |
| Transport and returned capacity | Existing physical resource service and explicit return dependencies | Peer exchange and finite-window controls; transport parameters retain their own evidence |

The general resource visit uses `eligible = max(required predecessor times)`;
the resource policy grants `started`, releases it at `finished`, and reports
downstream visibility at `completed`. Queue wait is `started - eligible` and
service is `finished - started`. A barrier's synchronization wait is caused by
its arriving participants, not an arbitrary additional latency per byte.
Polling has both executed work and readiness wait; do not charge that same
interval again as a round-trip constant.

For fixed code and instruction groups, a candidate GPU service term can use
`cycles(protocol, operation, active lanes) / observed_clock`. Memory service is
scheduled against shared identified rates. Total time is the realized critical
path through those visits, plus only separately owned exposed host terms.
Do not fit a free offset for every width, channel count or payload: width
changes the algorithm graph, and resource sharing changes the queue service.

The source establishes ordering, not proprietary cache behavior or exact cycle
costs. Fit only parameter combinations distinguishable by the interventions.
Report normalized sensitivity rank, singular values and parameter correlations;
if two terms remain indistinguishable, retain a named joint interval and add
an identifying contrast before claiming they are separate physical costs.

## Implementation and measurement sequence

### 1. Freeze source transitions and conformance stimuli

Before code or a new study, commit expectations containing the exact source
and binary identities, operation/transport scope, generated cell manifest,
fatal guards, numeric relations and rejection rules. Keep known residual rows
in a separately labelled regression set. The first implementation slice is
source-correct FIFO state and event emission on the live calendar.

Cover participants `{2,4}`, protocols `{LL,LL128,Simple}`, one and multiple
channels, empty primitive slices, partial protocol lines, final partial
channels, exactly full slots, more than eight outstanding steps and LL flag
wrap. Include successive calls switching protocols on the same connection,
checking shared sequence, constructor alignment and protocol-buffer selection.
Derive near-boundary payloads using `boundary + {-4,0,4}` bytes for the
float32 scope. Invoke empty-slice primitives in the conformance harness;
a zero-size public NCCL call can skip execution and is not an overhead probe.
Include observed direct/connection branches and unsupported-branch rejection.

Compare transition order, predicates, useful/encoded bytes, masks, slot
addresses and sequence increments with a diagnostic build of the pinned
source. Observer runs are separate from timed production-binary runs and
cannot silently become calibration timings.

### 2. Add shared resources and exact model-side relations

Schedule per-channel instruction groups and memory requests against the shared
GPU and transport resources, retaining partial-channel work. For each protocol,
vary link rate by `{0.5,1,2}` and available SMs by at least two supported values.
Freeze the following relations in the implementation expectations:

- An isolated serializer's byte-service component doubles when rate halves,
  within integer-picosecond rounding. Propagation and GPU instruction counts
  stay fixed; total time need not double when another resource limits it.
- Identical independent block jobs on a declared one-block-per-SM fixture take
  exactly `ceil(C/M)` service waves. This exact fixture is not a universal
  claim about NCCL's real block residency.
- At fixed per-channel work, increasing channels increases aggregate issued
  work exactly. Completion saturates or grows once a shared resource limits
  it; it cannot acquire additional physical capacity from the channel label.
- In a deliberately window-limited fixture, capacity `K` useful bytes and
  reuse round trip `R` bound steady useful throughput by `K/R`, as well as
  the applicable encoded-link bound. Delaying a consumer delays only causal
  dependants until a shared resource connects the paths.
- Tail payload and masks control data bytes exactly. Source-mandated empty
  slice state remains observable, and adding its control work never invents
  payload bytes.

Each relation needs an independent oracle, not a restatement of the runtime
implementation. Fatal invariants and disabled-path equalities are unscored.

### 3. Identify costs with controlled Merlin interventions

Use both architectures and both participant counts. Keep float32 sum, Ring,
connection mode, topology and the tested protocol fixed within each contrast.
The proposed identification matrix below is made executable and frozen before
the first new capture. Read-only inventory can constrain requested controls;
an executable capability pilot gets its own prior expectations freeze and
does not become calibration evidence by default.

| Controlled family | Proposed sweep | What it distinguishes |
|---|---|---|
| Channels at fixed total payload | `C={1,2,4,8,12,16,20,22,23,24,28,32}`; `S={256,512,1024,2048,3072,4096}` KiB | Partition effects, block setup and shared-resource saturation |
| Channels at fixed work per channel | Same C set; useful bytes/channel `{16,64,128,256}` KiB | Additional aggregate work without holding only the largest channel fixed |
| Partial-channel and LL128 residual window | `S=3 MiB + k*16 KiB`, `k=0..32`, with automatic selection and fixed `C={22,23,24}` | Current plateaus, tail padding and channel-count changes |
| GPU resources independent of C | Fixed `C={8,24,32}`; requested `M={8,16,32,full device}`, rounded to supported partitions and recorded as granted | Residency waves versus shared memory/link ceilings |
| Warp work independent of C | Source-valid thread settings spanning 4, 8, 16 warps and LL128's 20-warp path where realizable | Protocol work, barriers and per-block resource footprint |
| Receiver progress and FIFO wrap | Primitive streams of `{1,7,8,9,16,17}` reservations; receiver delays `{0,0.5,2,8}` us | Poll work, capacity exhaustion and reuse return time |
| Memory and arithmetic | Source-faithful copy/reduction primitives, small/large rotating working sets | Shared memory service versus arithmetic and packing |
| Timing context | Paired host and GPU events on the same invocation, warmups `{5,20}`, timed iterations `{20,100}` | Measurement boundary and host exposure without cross-process subtraction |

Requested channel/thread settings are not proof of realized geometry. Capture
selected algorithm, protocol, channels and warps for every cell and fail the
identification contrast when a required setting is not realized. Changing a
channel limit is not a controlled SM-count change.

Use CUDA green contexts only after verifying that the installed driver,
library and peer-memory setup support the required restricted resource path.
Record the granted SM set and actual residency. NVIDIA's
[green-context contract](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html)
does not guarantee general concurrent forward progress, and partition sizes
have hardware constraints. A competing kernel or a Multi-Process Service
percentage is not an equivalent substitute. If the full NCCL path cannot use
the restriction, identify source-faithful primitive service under that control
and leave the full-collective SM-transfer claim unqualified under TRAF-43.

Lock GPU and memory clocks where the allocation permits, and verify them.
Otherwise retain observed clocks and throttle state and withhold claims that
require fixed frequency. Randomize payload/condition order in five independent
process repetitions; hold CPU placement, worker lifecycle, data initialization,
allocation, stream mode and buffer policy fixed within paired contrasts.
Keep per-rank observations and use a declared collective completion reduction.
Unsynchronized clocks on different GPUs are not directly subtractable.

Prefer sparse source instrumentation and existing profiler observations that
measure instruction work, memory traffic and block residency. Measure observer
perturbation against the ordinary binary. Do not infer exact hardware queue
depths, per-poll wire transactions or SM placement solely from all-reduce time.

### 4. Lock parameters, then validate new conditions

Identify startup and ready-peer GPU work before delayed-peer and contention
terms. Then check joint sensitivity and fit the complete resource graph using
the identification set only. Report before/after predictions of the existing
surrogate on exactly the same scored cells.

Before fitting, reserve a separate cell manifest with unseen payload offsets,
channel/resource combinations and process runs. Keep its observations sealed
until the candidate, timer boundaries and interval construction are locked.
Freeze the sampling rule against all known payload lists, including older
power-of-two anchors; do not call another repetition of a known payload an
unseen-size validation. Repeated processes, condition cells, and exact oracles
have separate denominators.

For the 256-KiB to 4-MiB acceptance window, retain the existing candidate bars:
every reference median within 10 percent, every original-method median within
15 percent, every required median covered, and full band width at most 40
percent of the corresponding measured median. Score A100 four-GPU LL128
separately so the other curves cannot hide its failure. The inherited full-range
15-percent and anchor/bypass requirements still govern TRAF-43 closure.
For each independently resolved nonzero intervention effect, also require
the predicted time difference within `max(10% * abs(observed difference), 1 us)`.
Resolve an effect only when its magnitude exceeds twice the combined repeat
interquartile spread; report unresolved contrasts without declaring a cause.

Run lower, central and upper parameter cases through the same event graph.
The shaded area describes a declared parameter/condition envelope, not random
kernel-time noise or a statistical confidence level unless that level has
been established. Keep timer-method envelopes separate where necessary. An
aggregate union can be displayed with its meaning stated, but must also meet
the 40-percent width bar against each matched median and does not replace
each method's acceptance check. Never widen it after inspecting validation.

### 5. Reach request metrics and explain the A100 residual

Exercise an original execution graph with two all-reduces per controlled step
and fixed compute. Vary link rate and parallelism/resource count. For an
explicitly serial fixture, token-time change equals the sum of changed
collective durations exactly; for overlap, use the realized critical path.
The production path must reach `CompletionEvent`, `StepResult`, TTFT and TPOT,
and the analytic bypass must preserve the accepted baseline exactly.

Publish measured/model latency and uncertainty bands, signed residuals,
per-channel useful/encoded work, FIFO occupancy and full-slot waits, resident
blocks/active warps, memory/link utilization, and critical-path attribution.
The deciding diagnostic is whether the A100 LL128 plateau and boundary jump
are explained by independently identified resource work on untouched cells.
Coverage alone does not establish that explanation.

Compare NVIDIA public results only after matching GPU SKU, node topology,
participant count, library/benchmark version, selected protocol/algorithm,
channel geometry, timer, clocks, buffer policy and repetition procedure.
Use the source review's experimental-control assessment. A published H100
network ping-pong or eight-A100 switched result cannot calibrate a four-A100
direct-mesh FIFO cost. TRAF-92 owns any additional product qualification.

## Acceptance gates and task consequences

Any violated fatal guard voids the affected acceptance run: missing source
identity, wrong realized control, data mismatch, slot overwrite, capacity or
byte-conservation failure, visibility before prerequisites, duplicate timing
ownership, or changed disabled-path artifacts. Keep void evidence and its
findings. Accuracy or band-width failure with valid guards is a refutation of
the candidate, not a reason to discard measurements.

The plan turns TRAF-54's next slice into a source-derived execution model and
makes TRAF-43 depend on independent identification rather than a wider fit band.
TRAF-54 stays open until its enabled path and broader registered scope qualify;
TRAF-43 stays open until its accuracy, range and bypass requirements hold.
COMP-44 still owns host composition. TRAF-73/TRAF-86 and TRAF-92 still own
unidentified transport and product parameters. A successful component study
alone does not close an end-to-end inference-accuracy or Pareto-frontier claim.
