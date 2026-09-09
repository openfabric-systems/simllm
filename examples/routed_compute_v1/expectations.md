# Routed compute mechanism expectations

This expectations-only commit precedes implementation and every study run.
It defines a mechanism slice of COMP-7 and COMP-43. Kernel costs are declared
model inputs; no value from a VOID hardware campaign becomes a calibration.

## One routed input, one causal execution graph

The enabled path consumes the existing `RoutedMoeSupply`. One validated
projection selects the scheduled prefill/decode token slices, the step's exact
placement epoch and every `(request, token, layer, top-k slot, expert, owner)`
assignment. Communication deduplicates owner destinations as it already does;
compute retains every expert assignment, including assignments staying local.
Two experts on one owner need one transmitted vector and two expert rows.
They cannot be reconstructed from the deduplicated traffic byte count.

Use the repository's existing step lowerer, `ExecutionGraph`, runtime,
completion events and `StepResult` metric chain. Select routed compute
explicitly. Its disabled path retains all existing serialized and observed
graphs, traffic, timestamps and result bytes exactly. Reject conflicting
observed schedules, missing routing, tensor parallelism wider than one,
non-ideal host composition and unsupported model geometry before runtime
mutation. An unsupported provider estimate fails before graph execution.

This first envelope has one token-owning engine and at least two expert-owner
ranks. Only the engine performs attention and output projection. Expert-owner
ranks compute precisely their routed rows. Per layer, the causal order is
attention, dispatch, local expert projections, then combine. The next layer
depends on combine; output projection follows the final combine. Whole-phase
barriers are declared conservative ordering. No captured native concurrency,
cut-through execution or peer-engine workload is inferred.

The compute and traffic operations carry the same selected placement epoch.
The runtime is the sole timing authority. Load rows and compute-price records
are immutable input projections, with no queues or private advancing clocks.
An idle expert rank emits no expert compute operation and pays no kernel floor.

## Work and declared minimum service

For hidden width H, expert intermediate width I and R assigned rows on a
rank, grouped gate/up projections perform `4 * H * I * R` floating-point
operations; down projection performs `2 * H * I * R`. The total is
`6 * H * I * R`. If A distinct experts are active, their weight extents are
`2 * H * I * A` and `H * I * A` elements respectively. Stream each active
weight extent once in this declared model, rather than every resident expert.
Retain the per-expert row histogram, so skew within one rank remains visible.
Inactive residents add no weight-read demand. No cache-hit or tensor-layout
measurement is implied by that streaming assumption.

Each nonempty expert rank executes one modeled grouped gate/up kernel and one
modeled down kernel. This is two invocations regardless of the number of active
experts. Their minimum service applies once to each invocation, not once to
each expert, token or host launch. It is a maximum with that invocation's
provider service, never an additive cost and never a floor on the whole step.
Bind the declared minimum to the complete selected GPU envelope; reject a
different architecture or envelope. Zero minimum preserves the provider's
estimate exactly. The off path supplies no floor or new result metadata.

Retain the existing non-expert arithmetic model, splitting attention work
by layer and placing the output head once. Memory streaming and grouped GEMM
timing remain model assumptions. Router, activation, fusion, physical launch
identities and calibrated constants retain COMP-6, COMP-43 and COMP-45.
The enabled result must name its declared floor and actual routed work in
the graph evidence; it must not label them measured or architecture-calibrated.

## Prospective study

Run twelve CPU model configurations: eight or sixteen concurrent requests,
balanced or hot routing, and declared expert-kernel minimums of zero, five or
twenty microseconds. Each request has one prefill token followed by two decode
tokens. Each configuration runs three successive real step-sink calls and
retains every input, graph, runtime report, completion result and request metric.
The model has two all-MoE layers, H=16, I=32, four experts, top-k one, one
attention head of width sixteen, one key/value head, vocabulary 64 and two-byte
elements. Expert ranks are 0 and 3; the token engine is rank 0. Expert 0 belongs
to rank 0 and expert 2 belongs to rank 3. Balanced routing splits request rows
equally between those experts; hot routing sends every row to expert 2.
The other two experts are present in placement and inactive in this grid.

The declared GPU has 10^9 floating-point operations per second and 10^15 memory
bytes per second. Provider efficiency is one. Both ideal endpoint rates are
8 * 10^12 bits per second; all extra control, channel, launch and completion
costs are zero. These deliberately simple envelopes identify the arithmetic
and ordering mechanism; they describe no physical GPU or interconnect.

Floor: each request cannot finish before its causal compute chain and every
required transfer serialization; each expert kernel cannot beat its own work
over peak rate, active weights over memory rate, or selected minimum.
Ceiling: serializing all nonempty compute visits and all directed transfer
bytes twice, starting from step release, bounds the finite graph. Review these
before reading outcomes. No behavior is scored if a causal, identity, byte,
placement, source or receipt guard fails.

Keep the evidence classes separate:

- Thirty-six exact expert-service vectors, one per actual step, use
  `max(4 * H * I * R / peak, gate_up_bytes / bandwidth, floor)` and
  `max(2 * H * I * R / peak, down_bytes / bandwidth, floor)` for each active
  owner. Compare the graph's demand and actual runtime service, not only a
  final scalar. Picosecond conversion follows the existing provider once.
- Five load-imbalance instances compare hot with balanced routing: both
  request counts at zero/five-microsecond minimum and sixteen requests at
  twenty microseconds. In each, hot expert service strictly increases. Its
  TTFT and TPOT increase by the two layers' exact critical expert-service
  difference plus a nonnegative communication difference bounded by twice
  the additional directed wire serialization. The eight-request twenty-
  microsecond expert-service equality is a forced, unscored guard.
- Five minimum-service instances compare zero with five microseconds for
  eight balanced requests, then five with twenty for all four count/routing
  pairs. TTFT and TPOT increase by exactly two layers' change in the critical
  owner's two expert invocations. Routing and every transferred byte stay
  unchanged. Other zero-to-five pairs do not expose the minimum and are
  unscored identities.
- Six request-count instances compare eight with sixteen requests at every
  minimum and routing choice. TTFT and TPOT strictly increase and stay at or
  below twice the smaller-count value. Fixed minimum costs need not double.

The sixteen behavioral instances form three families. The twelve
configurations, thirty-six exact vectors, conservation identities, zero/off
paths and software test cases never enlarge that denominator. The reported
time to first token (TTFT) and time per output token (TPOT) come from the
existing reducer's actual request history.

## Controls, preservation and effect

Software controls cover local-only routes, two selected experts on one owner,
uneven expert histograms, changed placement epochs, mixed scheduled phases,
prefill chunks and decode offsets. Packed routing-arena and strict projection
inputs must produce equivalent loads and traffic for the same assignments.
Reject missing/duplicate/foreign ownership, mismatched model/route geometry,
unknown request slices, mutated or closed arena views, cross-envelope floors,
conflicting observed schedules and unsupported host/parallel modes. Verify
failures leave runtime clocks, request history and previous outcomes unchanged.

Retain complete old graph and result bytes with routed compute explicitly off,
including an existing captured routing supply. Preserve every existing routed
traffic fixture. Invalid inputs fail before any real step executes. A drain
with no new work retains the existing empty graph and zero service.

The source commit and frozen-input hashes are recorded before the run. Every
run uses a new external output directory, preserves its first receipts and
records PASS or VOID with a null behavioral score on fatal failure. No failed
run is rescored. Unit tests and the standard repository gates supplement the
36 actual model-step calls.

The result can qualify routed demand and declared kernel-floor composition
on this supported model path. COMP-7 and COMP-43 retain any remaining measured
acceptance obligations explicitly. It cannot close COMP-45 hardware capture,
COMP-6 physical invocation fidelity, TRAF-26 full peer workloads, CORE-51 or
CORE-54 large-model deployment frontiers. Real Granite, DeepSeek-V3 and Kimi K3
GPU performance is outside this synthetic mechanism grid.
