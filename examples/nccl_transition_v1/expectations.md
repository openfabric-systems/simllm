# Dense NCCL transition study: frozen expectations

## Scope and chronology

TRAF-43 owns the unexplained all-reduce bandwidth dip near 1 MiB on two
GPUs and near 2 MiB on four GPUs. The earlier A100 and GH200 measurements,
and the five-anchor model's approximately 27 percent underestimate, are known
before this freeze. This is a prospective mechanism experiment informed by
those observations, not a new prediction of the existence of the old dip.
No model, anchor, profile or acceptance bar is fitted or changed here.

The accompanying JSON freezes the sweep before implementation and before any
new GPU timing. The result report cites this expectations-only commit and
the implementation commit actually staged on Merlin. Unavailable hardware or
unsupported controls remain explicit missing evidence, not successful checks.

## Measurement contract

Use one Merlin node with four allocated, otherwise idle GPUs, separately on
A100 and GH200. Run float32 sum all-reduce at widths two and four using the
same staged NCCL 2.31.2 libraries as the original hardware envelopes. Preserve
the original CUDA toolkits (12.2.2 on A100, 12.9.1 on GH200), topology,
library hashes, GPU identities and clock observations. Device IDs zero through
width minus one identify the participants. No clock-setting or system change.

The dense payload grid is 256 KiB through 4 MiB, inclusive, every 16 KiB:
241 distinct sizes per curve. Five independent process repetitions retain
every raw observation. The original host-thread/event harness supplies a
comparison on this whole grid, preserving its five warmup and twenty timed
iterations, out-of-place buffers and maximum of per-rank event durations.
Only its size list and selection of all-reduce without unrelated envelope
tests change. This retains the timing boundary of the attached figure; it is
a steady-stream per-call duration, not a globally synchronized phase span.

An independent benchmark uses official NVIDIA nccl-tests commit
`b4d5beebca8a76cf01335f724d154b9b9d394d96`, with one host thread per GPU,
20 warmups and 100 timed iterations, no graph, one aggregate operation, and
correctness checking. Retain both in-place and out-of-place results but use
out-of-place results for comparisons. Repeated process blocks, not individual
CUDA iterations, are the independent repetition unit.

Instrumentation is a separate lane: nccl-tests `-U 1` enables its profiler
plugin to record the actual algorithm, protocol, channel count, warp count
and kernel variant. Its timings never enter latency summaries. Initialization
graphs or configured channel maxima alone do not identify the choice for an
individual collective. The tuning lane has the same shapes, launch mode and
control environment as its matching timing arm, with 5 warmups and 5 timed
iterations. Compare selections across repetitions where available.

## Controlled arms

The control grid is 512 KiB through 2.5 MiB, inclusive, every 32 KiB:
65 sizes at each width on each architecture. Each timing arm has three
independent process repetitions. Run every Ring/Tree algorithm crossed with
LL/LL128/Simple protocol; run Ring/LL128 with requested channel counts
4, 8, 16 and 24; and run automatic selection with a CUDA graph replay.
Both channel bounds are set to the requested count. Report the observed count
because small-message selection may use fewer channels than requested.
Do not interpret an ignored knob as a successful fixed-channel experiment.

Run automatic dense timing first and last through randomized repeated blocks
with a frozen seed. Within a process, sizes ascend, matching the inherited
harness. The exact command, environment overrides, return status and sequence
number are retained. Unsupported algorithm/protocol controls may be retained
as unsupported without voiding other arms. No failed control is silently
replaced with automatic selection.

## Physical bounds, before reading new timings

Floor: using the all-reduce endpoint byte convention `2(n-1)S/n`, a deliberately
loose serialization floor is endpoint bytes divided by 300 GB/s on A100 or
450 GB/s on GH200, the one-direction aggregate NVLink egress per GPU.
The width-two direct-peer ceilings are tighter, 100 GB/s and 150 GB/s; use
these as a diagnostic when the recorded transport is the direct peer path.
Protocol overhead and GPU work can only increase these times.

Ceiling: no finite physical completion-time ceiling follows from byte count
alone when host scheduling, clocks or stalls are unconstrained. A process
watchdog is an operational timeout, not a physical bound. Time must be finite
and positive. Check inverse-throughput arithmetic independently and compare
the new and original hardware curves before attributing a mechanism.

## Hypotheses and quantitative decision rules

These are five relation families, reported separately by architecture and
width. They are not a sum of individual sizes, invariant checks or commands.
Refuting one is an interpretable result, not a fatal guard violation.

- H1, reproduction: at the original 512 KiB, 1 MiB and 2 MiB points,
  the inherited harness median is within 10 percent of its old capture.
  Report every discrepancy. A failure limits transfer to the old figure;
  it does not justify rescaling new measurements to make them agree.
- H2, independent timing: inherited and uninstrumented nccl-tests medians
  agree within the larger of 10 percent or 2 microseconds at each shared
  payload. Report worst error and where it occurs. Disagreement implicates
  timing, launch or buffer context and blocks a shared latency claim.
- H3, discrete selection: the automatic lane changes protocol, algorithm,
  channel count, warp count or kernel variant within 768 KiB to 1.25 MiB
  at width two, or 1.5 to 2.5 MiB at width four. Report the last old and
  first new sampled payload. A visible kink without a recorded choice change
  is not confirmation.
- H4, intervention: for each recorded automatic transition, compute the
  adjacent latency ratio across the boundary in automatic and compatible
  fixed arms. A mechanism is supported only if its choice is observed and
  holding that choice fixed reduces the automatic excess step by at least
  50 percent, with the automatic step larger than both 5 percent and twice
  the relative between-process interquartile spread. Dense auto boundaries
  are bracketed by the nearest control-grid points. Report all tested
  boundaries, including inconclusive and refuted cases. Controls need not be
  faster overall and unsupported combinations are not causal evidence.
- H5, host-launch sensitivity: graph replay and ordinary automatic timing
  agree within the larger of 10 percent or 2 microseconds on the control
  grid if host issue cost is negligible. When they differ, compare their
  recorded choices before attributing the difference to host launch cost.

Median, minimum, maximum and interquartile spread are descriptive summaries
of the retained process repetitions, not confidence intervals. Mark any point
whose interquartile spread exceeds 10 percent of its median as unstable;
do not discard it or make a mechanism claim from that point. No assumption
requires latency or effective bandwidth to be monotone across a threshold.

## Fatal guards and evidence separation

F1: primary arms have exactly the frozen widths, payloads and repetitions,
with unique keys, finite positive durations and no failed process. F2: every
reported correctness check passes. F3: library version/hash, upstream source
pin and GPU identities are recorded and agree within each architecture's
campaign. F4: no foreign GPU compute process is present on the allocation at
entry or exit. F5: aggregate serialization floors hold within 2 percent to
allow printed timing precision. F6: instrumented timings are never pooled
with uninstrumented timings, and original evidence is unchanged.

A violated guard voids the affected architecture/measurement lane and any
claim depending on it. Preserve the evidence and report findings rather than
a pass fraction. An unavailable architecture is incomplete, not a void run.
Unsupported controls are missing interventions and do not invalidate a valid
automatic lane. The compatibility capture's sparse correctness probes remain
explicit; the independent benchmark supplies its own full validation.

## Project consequence

This study identifies or narrows a component-level NCCL mechanism. It closes
no TRAF-43 model-accuracy requirement by itself and changes no time to first
token or time per output token. A replacement curve still needs a separate
freeze, held-out accuracy evidence and its existing identity bypass. Retain
the original five-anchor failure without moving its anchors or relabeling its
post-specified regression as prospective validation.

## Reproduction references

- [NVIDIA nccl-tests pinned source](https://github.com/NVIDIA/nccl-tests/tree/b4d5beebca8a76cf01335f724d154b9b9d394d96)
- [NCCL environment controls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [Original A100 harness](../a100_hardware_envelope_v1/lane_b_multi_card.cu)
- [Original GH200 harness](../gh200_hardware_envelope_v1/lane_b_multi_card.cu)
- [Existing model comparison](../collective_regime_curve_v1/RESULTS.md)
