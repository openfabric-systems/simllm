# Standardized NCCL primitive identification, version 1

This contract separates useful GPU work, memory service, executed polling and
waiting for a peer. Its machine-readable matrix is
[manifest.json](../../examples/nccl_primitive_identification_v1/manifest.json).
TRAF-94 owns the runner, source-conformance pilot and hardware execution.
This document is the experiment design, not a completed measurement campaign.
TRAF-93 supplies the buffered-read execution path that consumes its findings.

## Source and experiment identity

Pin NCCL commit `7b83616df3ae082a1f32bb74c27458bfe8153a13`, compiler options,
CUDA and driver versions, GPU identities/topology, executable/library hashes,
and selected primitive specialization. Use float32 sum and the actual
LL/LL128/Simple primitive code and memory-order instructions. A wrapper that
changes these is a separately labeled component probe, not an NCCL result.
Retain disassembly and compiled registers/shared-memory/block-thread counts.

Record source operations, channel byte partitions, peer permutations, active
channels, work warps, allocated block size, SM grant and buffer placement in a
separate diagnostic run. Requesting a control does not establish that it ran.
The observer never supplies the ordinary timing sample. Measure its intrusion
separately; do not subtract an estimated observer cost from production timings.

A100 and GH200 run separately. Use the same physical node and build within a
paired contrast. Ordinary/full resources and restricted contexts remain
separate conditions. Clock locking is optional only if permission is absent;
record that outcome and do not convert elapsed time using an assumed clock.
An SM availability pilot is not evidence of the primitive's block residency.

## Paired interventions

Use one active channel on two GPUs for identification before multi-channel or
four-GPU confirmation. Match buffers, code specialization, synchronization,
loop count and launch context within each pair. Change only the named factor.

| Family | Matched pair and observable | What it can identify |
|---|---|---|
| Ready peer | Data and matching sequence published before timed consumer, versus the same consumer awaiting a controlled publication | Executed ready checks versus additional waiting and repeated checks |
| Delayed publication | Producer waits for a consumer start signal, delays locally, then publishes; compare zero delay with the declared delay grid | Receiver sensitivity to data readiness, including polling observation cadence |
| Delayed consumption | Producer fills slots; receiver delays after readiness before reading/consuming, then returns head | Buffer reuse and producer backpressure, separate from data publication |
| Empty/nonempty | Preserve the same two-step Simple slice, or the same LL/LL128 primitive, with zero versus positive useful elements | Control-only work versus incremental data work; do not assume all protocols poll empty data |
| Copy/reduction | Same primitive geometry with copy versus float32 sum, preserving input-count accounting | Reduction instructions plus any extra source load; identify memory separately before calling the difference arithmetic |
| Memory working set | Same operations and useful bytes, reused versus rotating initialized addresses | Cache/transaction sensitivity and shared memory service |
| Resource sharing | Same realized channel/warp geometry, changing available SMs; then vary work warps at fixed SM grant | Residency, instruction overlap and synchronization scaling |

For delayed publication, use a device-side start signal and record the
producer's actual local delay. The zero-delay control includes that same
signal. Do not subtract timestamps from different GPUs: they need not share a
clock. Time each consumer interval locally; use same-device CUDA events and
host wall time as separate boundaries. A CPU sleep is not the delay control.
For ready-peer cells, a pre-timing dependency establishes publication first;
the diagnostic must verify zero unsuccessful readiness checks.

Instrument diagnostic counts of checks, successful observations, tail/head
publication, source loads, shared staging, remote reads/writes and consumed
steps. Keep those counts out of the timed binary unless an unchanged-code
measurement method is demonstrated. Count instructions separately from the
wall interval spent stalled. Polling can retain a block while other warps run;
never charge the same wait again as memory service or an extra round trip.

## Matrix, repetitions and data contract

The manifest defines sequential stages, not one blind Cartesian product.
For each stage, expand only its named axes and hold the baseline fixed.
Unsupported source specializations fail the capability pilot and remain
explicitly unqualified. Before ordinary timing, freeze the expanded cell
inventory, capability outcomes and its digest in an expectations-only commit.
Any new axes or changed bounds require a new freeze before affected execution.

Use five independent processes per cell, each with 20 warmups and 100 recorded
iterations. Randomize cell order using the declared seed, pair AB/BA order
across repetitions and retain every observation and process exit. Perform
correctness checks outside the timing interval against known float32 sums.
Warm and initialize every rotated allocation; exclude allocation, communicator
construction and initialization from device-service timing. Measure them in
separate records if host initiation is the question.

Each result row carries schema version, manifest/cell/process/repetition IDs,
source/build/device identity digests, all requested and realized controls,
phase/family, local timer boundary and units, raw duration, observed local
delay, correctness result, qualification status and explicit reason. Diagnostic
rows additionally carry source-operation counts and device-local timestamps.
Artifacts carry content hashes. Retain raw rows externally; publish compact
paired summaries with interquartile spreads and their raw-artifact manifest.
Never pool unqualified cells or partially complete campaigns as a full curve.

## Expectations and acceptance

Physical floor: transferred bytes divided by the slowest required service
rate, plus causally unavoidable propagation; a dependent read includes its
request before the response. Apply the floor to the measured timer boundary.
Physical ceiling: no universal finite ceiling is claimed without bounded
scheduling and peer stalls.

A delayed publication cannot be observed before that publication. At large
controlled delays, the exposed waiting interval should track the added delay,
subject to measured observation cadence and scheduling spread. Small delays
can be hidden by existing work; no universal additive delay is assumed.
A delayed consumer must prevent reuse of its live slot until head return.
Empty payloads retain the source-prescribed control operations and move no
payload bytes. Extra warps have no universal speedup expectation.

A contrast is resolved only when its absolute paired median difference exceeds
twice the sum of the two five-process interquartile ranges. Unresolved terms
remain named joint intervals. Verify realized controls and exact source/byte
conservation as fatal guards, separate from timing relations. A violated guard
voids the affected scope and retains the evidence.

Estimate parameters only from identification cells. Review parameter
separability before interpreting polling, publication or barrier coefficients.
Freeze parameters and the uncertainty propagation rule before opening fresh
confirmation cells. A resolved held-out intervention delta must be predicted
within the greater of 10 percent of its magnitude or the frozen repeat-spread
resolution threshold. Confirmation on a second architecture requires its own
identified architecture inputs. It is not permission to refit the first one.

TRAF-94 closes only when the executable standard, source-qualified pilot,
complete controlled measurements and identifiable parameter results meet this
contract. TRAF-43 retains its collective-accuracy and band-width requirements;
COMP-44 retains host initiation. No primitive result alone closes inference
accuracy or an end-to-end Pareto frontier.
