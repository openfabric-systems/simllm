# NVLink measurement-boundary audit result

The audit refutes the historical claim that the size trend identifies packet
overhead. The unchanged attribution function labels four packet-free size pairs
as packetization. One pair also contains a cell outside the original 16 percent
acceptance band, so the counterexample applies to an actual miss, not only to
the attribution function called in isolation. The synthetic audit passes all
20 exact checks at zero residual. It runs no hardware and fits no parameter.

TRAF-91 closes the inference correction. The prior TRAF-74 aligned hardware
qualification is **VOID because its alignment precondition is undecidable**
from the retained observations. Its original records remain historical evidence;
TRAF-86 owns fresh producer, memory, clock and transport discrimination. This
does not show that alignment actually failed, explain the captured slowdown,
calibrate a packet conversion or close the end-to-end deployment comparison.

## Chronology and independent evidence

The historical review is post-specified. The new synthetic study is frozen at
`06a88dea7c4647aeb8eeb44f8628f743a8527397` before implementation and first
execution. The first execution at `230ab5f` passes. Independent review corrects
its reporting of 24 evaluations as configurations, since the overlapping size
pairs contain only 18 distinct settings. It also separates the unchanged
historical producer from proposed new launch modes in the measurement request.
The repetition at `a6c29b6a6b0fca45f438b10852985d0fd0aa2648` passes with identical
synthetic values and historical reconstructions. Both executions are retained.

| Evidence class | Result |
|---|---|
| Packet-free evaluations | 12 size pairs, 24 evaluated rows, 18 distinct settings |
| Synthetic clock-boundary cases | 8 configurations |
| Independent exact checks | 20 rows, zero residual |
| Nonzero relation families | 4 families, 16 instances |
| Fatal synthetic-audit verdict | PASS, no findings |
| Historical budget reconstructions | 42 rows, exact reconstruction |
| Historical aligned qualification | VOID, missing observed alignment precondition |

Zero overhead, zero stagger and preserved source identities are unscored
guards. Historical records and source-derived pacing coordinates contribute
no behavioral denominator. [results.json](results.json) preserves the separate
classes, source digests and findings.

## Why the size rule fails

A sender that pays fixed time a and then copies B bytes at rate r takes
T(B)=a+B/r without any packets. Comparing its goodput with r gives relative
error E(B)=r*a/B. Doubling B halves this error; the drop is r*a/(2B).
Before inspecting a value, the floor is B/r and the ceiling is a+B/r in this
fully specified construction. Every synthetic time respects those bounds.

At 100 GB/s and 10 microseconds of fixed overhead, a 4 MiB copy takes
51.94304 microseconds, between its 41.94304 microsecond serialization floor
and its exact ceiling. Its goodput error is 23.841858 percent. At 8 MiB the
error is 11.920929 percent. The 11.920929 percentage-point drop exceeds the
historical five-point threshold, which assigns packetization even though the
construction has zero packet overhead. This proves non-identification; it
does not estimate the fixed overhead of the real producer.

The other three direct attribution-function outputs are below the historical
acceptance band. They are function-level counterexamples and are not reported
as three additional failed full-validation cells. Doubling positive fixed
overhead doubles its error, and doubling payload halves it, exactly across
the frozen rate and payload grid.

## What the historical clocks establish

The reported maximum 1.129 percent launch-skew fraction is reconstructed from
`(degree-1)*5 microseconds / minimum local device-event duration`. Its numerator
is a declared budget, not an observed difference between source starts. The
retained per-source summaries carry elapsed durations without a common start
epoch. Reconstructing the budget calculation exactly does not measure alignment.

Equal local durations also do not identify a common phase. Three 100 microsecond
visits starting at 0, 20 and 40 microseconds have a 140 microsecond phase. Using
the maximum local duration as the denominator overstates phase goodput by
40 percent. The aligned case has identical local durations and a different
phase. The frozen duration and degree variations reproduce this difference
exactly. NVIDIA's multi-device API likewise restricts elapsed-time comparisons
to events on the same device; it does not supply a cross-device epoch by
subtracting those durations. See the [multi-GPU event contract](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#multi-device-stream-event-and-memory-copy-behavior).

The original first capture remains void. The second capture's original six
misses and rates remain descriptive diagnostics, with no interpretable
hardware behavioral score for aligned model qualification. No observed timing,
original JSON verdict, expectation or producer source is rewritten.

## What the producer establishes

The persistent copy kernel waits on a deadline derived from the shuffled
message address rather than the monotone issue ordinal. Consequently, the
degree label changes its nominal pacing even at a nominal 100 percent offered
rate. For the first warp, the source-derived deadlines are:

| Payload | One-source label | Two-source label | Three-source label |
|---|---:|---:|---:|
| 4 MiB | 30.656 us | 32.3968 us | 2.01984 us |
| 8 MiB | 72.59904 us | 74.33984 us | 2.01984 us |

These are code coordinates, not measured waits. Other warps, memory service and
instruction issue determine the realized execution. The inner loop copies
individual bytes; an `access_width=16` argument does not prove a 16-byte machine
instruction or an NVLink packet layout. Memory transactions depend on the
compiled accesses and address grouping, as described in NVIDIA's
[memory coalescing guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-to-global-memory).

The final `order[message]=message+1` values prove coverage and final contents,
not arrival chronology. The host batch timer also includes submission and
synchronization, while the incast comparison uses local device-event durations.
Host batching alone therefore does not explain every low-rate diagnostic.
The device `clock64()` counter belongs to a streaming multiprocessor, not an
established common epoch across cards. See the [device time-function contract](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-c-programming-guide/index.html#time-function).

## Four-card request and remaining boundary

[measurement-request.json](measurement-request.json) defines separate A100 and
GH200 campaigns for topology and clock qualification, producer/launch controls,
memory-path discrimination, sharing and common-phase timing, and library
collective comparison. Each stage states its observable and rejection rule.
The unchanged historical producer retains its original 256 blocks of 256
threads and eager launch. Proposed new scalar/vector kernels and copy-engine
paths receive their own launch matrices.

The request contains synthetic payloads, no measurement rows and no submitted
attempts. It is awaiting reservation and explicitly not capture-ready. The
producer implementation, instruction inspection, clock uncertainty and exact
physical acceptance bands require a separate freeze before execution. Counter
records must retain field identity, scope, units, sample time and update latency;
the [NVML field record](https://docs.nvidia.com/deploy/nvml-api/structnvmlFieldValue__t.html)
provides distinct fields for these timing properties. A raw counter is not
automatically a packet count.

TRAF-86 preserves the existing candidate as its exact off path until a valid,
discriminating capture identifies a named mechanism and the enabled change
reaches time to first token or time per output token. Four cards physically
qualify at most three senders into one receiver. Larger degrees, actual switch
attachments and full model deployments retain their separate acceptance work.

Run the synthetic audit in a new bulk-output directory:

```bash
python examples/nvlink_measurement_boundary_v1/run_study.py --output-root "$SIMLLM_STUDY_OUTPUT"
```
