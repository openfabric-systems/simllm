# Protocol-aware collective model: frozen expectations

## Scope and chronology

This follow-up replaces the smooth, five-anchor interpolation with an explicit
NCCL 2.31.2 Ring protocol and channel-work model. All measurements and protocol
boundaries in nccl_transition_v1 are already known. Calibration and withheld
comparisons drawn from that capture are retrospective model-development
checks, even though this specification precedes the new implementation and
first fit. No public preregistration is claimed. Fresh off-grid hardware
measurements are separate validation: lock the candidate parameters in a
commit before those runs, and never use those measurements to set its band.

TRAF-43 owns the analytic accuracy change. TRAF-54 owns the remaining fully
packetized collective protocol. Inspect the exact pinned htsim commit and the
existing SimLLM NVLink calendar and native NVSwitch allocator before deciding
which layer owns each added cost. Do not install two transport calendars or
count a physical acknowledgement twice. The measured Merlin machines are
four-GPU direct meshes, not switched DGX boards.

## Mechanism model, before fitting

Use float32 sum Ring all-reduce on two or four GPUs in the frozen payload
range. Reject other operations, widths and ranges for this calibration.
Protocol selection uses the previously observed LL/Simple or LL/LL128 boundary;
record its 16-KiB observational resolution. Channel selection and data
partitioning are an independent translation of the pinned public source:
thread thresholds, the 32-KiB scheduling traffic cell, protocol-specific cell
sizes, channel-count quantization, and aligned ring chunks. Check predicted
channel and warp counts against every retained compatible callback. A mismatch
is a finding and blocks claiming an exact source projection for that arm.

Keep three byte ledgers distinct:

- LL: two 32-bit data words and two 32-bit readiness flags per 16-byte line.
  Eight application bytes occupy sixteen protocol-buffer bytes.
- LL128: fifteen 64-bit data words and one 64-bit flag per 128-byte line.
  Final partial groups are rounded to whole lines.
- Simple: application data has no embedded readiness flags. Separate 64-bit
  head/tail counter stores, slice boundaries and system fences are explicit.
  These are software-visible bytes and operations, not a newly claimed
  A100 or GH200 NVLink physical packet format.

The ring has 2(n-1) communicating stages. Preserve source chunk limits and
instruction work quanta. Count nonempty Simple publications separately from
empty synchronization slices. One additional visibility round trip per
selected publication is an explicit sensitivity parameter. Both LL and Simple
have buffer-reuse credit checks; neither unconditionally waits for a fresh
credit round trip on every datum. Polling and fences execute inside the
collective kernel rather than launching an additional polling kernel.

For each architecture, width and protocol, fit only three nonnegative service
coefficients: cost per maximum-channel encoded MiB, cost per source-derived
instruction work round, and cost per nonempty Simple publication. The fixed
terms are the historical tiny-message floor and encoded endpoint bytes over
the one-direction physical rate. The first coefficient is effective GPU
supply/reduction service, not an adjustment to advertised NVLink bandwidth.
The publication coefficient is a composite synchronization cost. Collective
timings alone cannot separate its memory visibility, polling and fence terms.
An independent ping-pong probe reports the scale of an acknowledged GPU
exchange; it cannot turn that composite into a uniquely identified NCCL RTT.

Represent the optional RTT as a partition of the composite publication cost
when within its identified nonnegative budget, or as an explicit added
sensitivity beyond that budget. Report both parts without advancing time twice.
All point estimates remain deterministic. An interval represents parameter
and measurement-method uncertainty, not random kernel service.

## Calibration, envelope and withheld checks

The JSON fixes automatic anchors every 128 KiB, beginning at 256 KiB, and the
three fixed Ring protocol controls. Fit reference timing medians on these
shapes only, with nonnegative least squares and no payload-specific timing
parameters. Use the source formulas at every payload, including withheld ones.
Fit a nonnegative intercept plus physical-service term to the inherited minus
reference timing difference at the same automatic anchors. The plotted center
is the reference prediction plus half that method allowance.

Compute the relative residual radius only from those calibration cells. It is
the largest absolute fractional residual, including the inherited method after
its allowance, plus the 90th percentile interquartile repeat spread. Apply the
same frozen radius rule to both sides of the method interval. Clip the lower
edge only at the physical floor. This is an empirical software and method
envelope, not a confidence interval or a uniquely measured polling cost.

Report center errors and coverage independently for each architecture, width
and method. Target at most 10 percent center error against the reference and
15 percent against the inherited method. The requested visual target is all
median points inside the band; qualifying coverage is at least 99 percent on
each curve with full band width at most 40 percent of the measured median.
Every uncovered point remains visible. Do not enlarge the band using withheld
or fresh measurements. Refuting a target is interpretable and keeps the owning
accuracy task open; it is not a fatal integrity failure.

## Fresh hardware and timing-method controls

After committing fitted parameters, measure the off-grid sizes in the JSON
with five independent process repeats, using the same NCCL libraries, CUDA
toolkits, topology checks and one host thread per GPU as the earlier study.
The original and reference timing methods remain separate. Keep correctness
checks and all raw repetitions. No per-collective profiler enters timing runs.

For the eight specified method-control payloads, vary warmup count (5,20) and
timed count (20,100) independently. Record both CUDA-event and host-wall spans
on the same inherited block. Compare fresh-thread and retained-thread launch
contexts and fixed-buffer versus offset-rotating access where implemented.
Match nccl-tests iteration and warmup counts. A mechanism is attributed only
if the isolated control reduces the absolute method gap by at least half and
the change exceeds twice the pooled interquartile spread. Multiple changed
factors support only a joint explanation. A negative result remains explicit.

Run a two-GPU acknowledgement microbenchmark with 256 warmup exchanges and
4,096 measured exchanges, for embedded-ready data versus data/fence/separate
ready-counter variants. Use one GPU's monotonic hardware timer for the round
trip. Check returned data and sequence numbers, retain process repeats and
label it a protocol illustration rather than an NCCL internal timing capture.

## Physical and live-path checks

Floor: ring endpoint application bytes are 2(n-1)S/n; protocol encoding cannot
reduce them. Two-GPU direct-peer one-direction ceilings are 100 GB/s on A100
and 150 GB/s on GH200; four-GPU aggregate ceilings are 300 and 450 GB/s.
Memory work, synchronization and launch cannot make a transfer beat that bound.
Ceiling: no finite hardware latency bound follows without a stall bound. A
watchdog is an operational limit, not a physical upper bound.

Vary physical rate by 0.5, 1 and 2 and participant width by two and four.
The isolated byte service scales exactly inversely with rate; physical
propagation and declared software costs do not. Vary the visibility RTT over
0, 0.25 and 1 microsecond: its exposed contribution is publication count times
RTT, with no second independent timing authority. These exact accounting and
identity properties are unscored guards.

Connect the opt-in model through CollectiveLatencyProfile and the existing
StepResult path. Run an original-graph prefill and decode example at two
payloads and both widths. The token metrics must change by exactly the
selected collective-service change. The absent model preserves every accepted
baseline timestamp and byte count. Refuse combining this analytic profile
with a separately advancing physical peer-packet authority, and reject mixed
fabric or unsupported collective scope before runtime mutation.

Fatal guards include invalid correctness, missing repetitions, unverified
library/source identity, impossible byte/time arithmetic, non-finite values,
leakage of validation values into calibration, and duplicate timing authority.
A fatal failure voids the affected run. Guards, unit tests, relation families,
point instances and hardware processes remain separate evidence classes.
