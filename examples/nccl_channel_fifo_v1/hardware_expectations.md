# Frozen channel and GPU-resource identification capture

This is an expectations-only hardware freeze. The source execution study has
its separate prior freeze. All observations from earlier campaigns remain
retrospective; no new capture here is an independent validation of a fitted
candidate. These are identification and capability observations only.

The companion JSON gives every condition family and payload-generation rule.
Expand it into a deterministic, hashed manifest before the first invocation.
Use five independent process repetitions, randomized condition and payload
order with the frozen seed. Both architectures use four allocated GPUs on
one direct-mesh node, with two/four selected ranks, float32 sum and Ring.
Keep the NVIDIA NCCL 2.31.2 shared-library hashes in the JSON. Compile the
timing harness with the selected toolkit and retain its exact hash. Save
topology, GPU/driver identities, linked libraries, environment overrides,
process inventory and sampled clocks/power/temperature. Raw data stays outside
Git. Fixed GPU clocks are claimed only if a permitted allocation-local lock
is successful and observed state verifies it; otherwise clock-dependent costs
remain conditional on observed state. Restore any acquired clock lock.

## Controls and timers

The baseline channel interventions fix `NCCL_MIN_NCHANNELS` and
`NCCL_MAX_NCHANNELS` together and set all six `NCCL_THREAD_THRESHOLDS` to zero.
This removes the source's small-work channel/thread reduction in that arm;
the scheduler may still realize fewer active channels when there are too few
work cells. Record actual algorithm, protocol, channel count and warp count
with a separate profiler-plugin run. A requested setting is not a realized
control. Do not silently replace the controlled family with default tuning.
The default chooser/residual family keeps default thresholds and channels.

LL/Simple thread interventions request 128, 256 and 512 threads. Simple adds
its source-prescribed synchronization warp. LL128 requests 256, 512 and 640;
the source minimum of 160 makes a 128-thread request invalid. No annotation
calls a thread count an SM count.

Measure host wall time and GPU events on the same invocation, retain each
rank and the maximum-rank collective reduction. Primary counts are 20 warmup
and 100 timed iterations. Persistent workers, explicit CPU affinity, initialized
out-of-place buffers and a 64-MiB allocation are fixed. Separate count and
rotation contrasts change only their stated field. GPU-event time is not
assumed to equal pure kernel service when host issue gaps are visible.

## Resource capability pilot, excluded from calibration

On each architecture, first request full device and 8/16/32 SMs for a small
Ring LL128 call at fixed channels. Use CUDA green contexts where available.
Record the actual granted SM count, associated stream context, and a diagnostic
SM-ID sampling kernel on that same stream. The diagnostic NCCL launch callback
also checks the green context of NCCL's actual launch stream. No multiprocess
percentage or competing kernel is substituted. A timeout, unsupported API,
failed peer mapping, escaping launch stream, wrong result, or resource mismatch
fails that resource pilot. Resource timings are admitted only for passing
configurations; other families remain interpretable because they do not select
the failed capability. Kernel residency and register/shared-memory instruction
costs still need source-faithful probes; a granted SM partition alone does not
identify them. Keep instrumented timings separate from ordinary timings and
show their perturbation on matched conditions.

## Prior directions, relations and falsifiers

Floor: all-reduce sends `2(n-1)S/n` useful bytes per rank on average. The sum
over ranks cannot exceed the aggregate directed physical capacity times the
interval. Protocol expansion raises wire demand. No finite upper latency
bound exists without bounded stalls.

At fixed total bytes, more channels can improve overlap until shared service
saturates, and may then add overhead. Do not impose monotonic speedup. At
fixed bytes per channel, total source work grows exactly with active channels;
after saturation a maximum-channel-only predictor should miss the added
shared work. A resolved timing rise at fixed maximum channel work refutes that
operator. In a source-conformant prediction, LL128 partial warp stripes carry
2048 encoded bytes per positive 1920-byte useful stripe, even at the tail.

At fixed realized channels, protocol, threads and payload, a smaller verified
SM partition cannot be treated as a smaller link rate. It can create block
waves and more waiting while leaving application bytes unchanged. If changing
SMs does not resolve an effect, report that rather than assign a fitted SM term.
Changing thread count can change instruction overlap, barriers and occupancy;
no universal faster/slower ordering is asserted across protocols.

A paired timing-boundary effect is wall-minus-event for the same invocation.
Warmup/iteration/rotation effects are matched differences within one declared
family. Resolve a difference only when its absolute median exceeds twice the
sum of the two repetition interquartile ranges. Before any fit, plot the
controls and repeat spreads. Report normalized finite-difference sensitivity
rank and parameter correlations; an all-reduce-only fit does not establish
separate barrier, fence, polling, memory and instruction latencies.

Fatal guards are unscored: shared-library identity, float32 sum correctness,
no foreign GPU process, declared local topology, complete rank/repetition
inventory, positive timers, and realized control for each qualified contrast.
A violated global guard voids its allocation. An unavailable resource pilot
or unrealized channel/thread request voids that contrast's causal claim,
with raw observations retained and labeled. None is a lost behavioral point.

TRAF-54 owns source mechanism and resource execution; TRAF-43 owns calibration
and fresh validation; COMP-44 owns host composition. This capture does not
close them. Lock any candidate and interval rule before collecting unseen
validation cells. Never use these data both to choose a band and validate it.
