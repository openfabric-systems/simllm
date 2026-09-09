# Native host execution diagnostic

This expectations-only contract precedes the first diagnostic execution. It
investigates the host cost that prevented CORE-52's target campaign from
finishing its first scale. The original target run remains VOID, with its
1,200-second timeout and partial evidence unchanged. No production behavior
or model-service value changes in this diagnostic.

## Source-writer encoding amendment

The finite protocol below was frozen before its implementation at
`8f06648d26bbfb2bfff4751d29a2e2bf232b440d`. This encoding amendment is a
post-specified regression contract after the first diagnostic's retained VOID.
It precedes the parser repair and fresh execution; it does not establish public
pre-registration for the corrected checker. The old run stays unrescored.

The unchanged `StepRecordStream.append` in `simllm/core/step.py` emits
`json.dumps(step_record_to_json(record))` followed by exactly one LF. Its ASCII
bytes retain schema insertion order and standard spaces. Bind admission to the
writer's source hash and reconstruct those exact bytes, including whole-file
framing. Positive fixtures use the actual writer. Reject CRLF, missing final
LF, additional blank lines and alternate member order.

Parse every native line with duplicate-member and finite-value checks,
including nested duplicates and exponent overflow. Rehydration through the
step schema cannot replace the original parsed object: compare that original
object to `native.json` using the existing type-aware equality. Unknown fields
and missing fields cannot disappear through rehydration. Diagnostic JSON keeps
its exact compact sorted encoding; progress JSONL adds exactly one LF per
compact row. Raw bytes and their first receipts remain untouched.

No native writer, service value, finite case, timing limit, identity rule or
profiling boundary changes with this amendment.

## Question and physical boundary

The host constructs a model graph, validates and projects its dependencies,
computes deterministic service and advances a native serving scheduler. It
also writes progress to storage. Which of these host operations consumes the
execution time is unknown. Source inspection suggests repeated graph work and
an edge-by-edge linear dependency search, but no hotspot or speedup is assumed.

The simulated device time is a separate quantity. The native frontend uses
simulated workers, no weights and no GPU context. Handoff is a declared
constant and intra-node collective cost uses the accepted analytic arm. A
host profile cannot calibrate GPU service or establish independent-engine
throughput.

## Paired fresh-process protocol

Run two fresh one-prefill, one-decode processes, in this fixed order:
uninstrumented, then instrumented. Both use the exact pinned vLLM 0.27.1
environment and Granite config-only checkpoint from the original session.
Each engine retains eight simulated workers, the same shared clock and the
unchanged serial `run_request` path. Keep the same progress writes, including
file synchronization, in both arms.

Each process first repeats the four original CORE-58 controls under their
original labels and order. It then runs prompt lengths 8 and 16 crossed with
serial request counts 1 and 4, in the order enumerated in `expectations.json`.
Each new request has four output tokens and a 100-microsecond handoff. A
request is admitted exactly when the previous request finishes. This gives
14 requests per process and 28 overall, including eight historical controls.

The instrumented arm adds passive counters at a closed list of native and
model entry points. Counters observe call count, wall nanoseconds and main
thread CPU nanoseconds. Record inclusive time and direct nested instrumented
child time separately; their difference is exclusive time among these selected
entry points. Uninstrumented work inside an entry point remains included in
its exclusive value. Never sum overlapping inclusive timings or interpret a
phase subtotal as total host wall time. Check exact counter arithmetic and
require restoration of every wrapped callable, including on exceptions.
Remove temporary instance attributes for originally inherited methods; leaving
a bound-method shadow is not restoration.

Counters wrap only the main thread, return the identical result object and
propagate the identical exception. They never change the model clock, function
arguments, scheduling decisions or persistence policy. Retain actual wrapper
bindings and their original source identities. Absence or replacement of a
declared entry point is fatal. The uninstrumented arm has no wrappers and no
profile hooks.

Detailed call profiling is bounded to the two one-request cells in the
instrumented arm. It includes request execution and the original progress
write. Keep the raw profile and a complete readable function/caller export,
not only a selected top-function list. Function identity is the full
`(filename, line, function)` tuple, with all caller edges retained. The four-request cells have passive
counters but no detailed call profile. Profiling overhead is descriptive; do
not subtract it to invent an unobserved execution time.

The phase list distinguishes native schedule, scheduler output update,
frontend output processing, step lowering, validation, locality planning and
checking, GOAL projection and checking, sink execution and publication, and
progress persistence. Some functions are imported under aliases: the phase
name identifies the exact wrapped entry point. The detailed profile retains
all original function calls, including unwrapped aliases, so an alias cannot
be silently treated as complete phase coverage.

## Model and evidence invariants

Every request result is retained in full. Compare both arms through the
existing complete CORE-58 comparison boundary: remove only the two named
opaque native root request IDs. Preserve every other member exactly, including
all four output token times, connector metadata and the causal decomposition.
The four historical control hashes must also reproduce in each fresh process.

Retain complete unique executor `StepRecord` and `StepResult` rows, clock
advances and all public sink outcome collections. Bind opaque native IDs to
stable request IDs by their actual per-request mapping. For cross-arm step
comparison, normalize only scheduled, finished and preempted request-ID fields
through the complete engine-qualified bijection
`(engine_id, opaque_id) -> (engine_id, stable_request_id)`. A same-request
opposite-role substitution must reject. The optional `sampled_request_ids`
field remains absent and cannot be silently added to normalization. No arbitrary recursive string substitution
or ignored timing field is allowed. Sink outcomes must match in full. Their
model identities, dependencies, prices, counts and service decomposition
remain intact. The admitted local outcome collections have no path-bearing fields. The
path allowlist is empty: an unexpected path rejects instead of being erased by
a generic output-root replacement.

Require the actual session-owned native engines, executor and worker bindings
before and after execution. Construction advances the shared clock by zero.
Source files, imported source origins, model configuration, interpreter and
selected profile remain exact before and after. Model weights stay absent,
the GPU context remains uninitialized and the packet backend runs zero times.
These are fatal guards, not scored behavioral observations.

Known historical request values are regression oracles. Prompt service is
95,424,000 or 114,936,000 ps. Every decode service is respectively 77,952,000
or 77,976,000 ps. A new serial request has job completion time equal to prompt
service plus 100,000,000 ps handoff plus four decode services. A cell of `N`
serial requests has exactly `N` times that virtual duration. The causal
decomposition and all input token/context/phase fields remain exact.

Before reading new output, cache-transfer floors remain bytes divided by
400 Gbit/s, with 49,152 bytes per prompt token. Declared ceilings use
10 Gbit/s plus 50 microseconds. The charged resident expert inventory divided
by memory bandwidth remains a conditional floor for the existing all-expert
streaming surrogate, not a hardware floor for routed tokens. A request cannot
finish before its prefill, handoff and four decode services. Neither profiling
nor a future host optimization can change these simulated values.

## Result and stopping policy

Each process has the original 1,200-second timeout and 16 GiB sampled resident
memory cap. Sample every 0.25 seconds and retain the exact stopping point.
Missing output, malformed records, changed source, semantic disagreement,
unrestored instrumentation or any other fatal guard makes this diagnostic
VOID with a null behavioral score. Ordinary exceptions retain a traceback.
Never replace the failed target campaign's verdict with this smaller result.

A complete diagnostic has verdict `COMPLETE`, a null behavioral score and
separate exact-oracle, structural-guard and descriptive timing evidence.
There is no scored speedup relation. Report request wall time, process CPU
time, selected inclusive/exclusive phase time and detailed profile function
cost with their denominators and instrumentation scope. Do not claim that
one host run establishes universal performance or infer time for an unrun
56-engine target.

CORE-52 remains open even when this diagnostic completes. Attribution may
justify a separately frozen optimization. Its prospective acceptance must
preserve exact graph/projection and rejection behavior and the complete native
request results. A call-local lookup index is preferable to cross-request
result caching if the evidence identifies that search as material. CORE-68
retains independent-engine timing; CORE-51 and CORE-54 retain deployment and
frontier obligations.
