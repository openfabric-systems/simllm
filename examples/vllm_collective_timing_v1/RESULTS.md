# VLLM-48 live collective timing result

What ran: two fresh live Granite inference processes used the pinned vLLM
0.27.1 CPU source build, tensor parallel size two, gloo, two explicit logical
request IDs and two manual engine steps per process. Each process emitted an
optional collective-service envelope from the stock model-runner path.

What came out: the nonvoid study scored five of seven frozen behavioral
instances. It captured exactly 100 distributed operations in each run and all
six mutation controls fired. The two M1 instances failed because the frozen
source oracle named the final logits operation `gather`, while pinned vLLM
executed native `all_gather` at ordinal 49 in both steps of both runs. Payload,
rank, dtype, element width, tensor shape, group and layer metadata otherwise
matched the source-derived population exactly.

What it changes: the optional live seam, schema compatibility and comparator
contracts are implemented, but VLLM-48 remains open because its own frozen
metadata-conservation family failed. VLLM-49 remains the A100 matched-hardware
campaign and cannot inherit a closure from these CPU results.

What it does not change: the result does not calibrate an A100 or H200 floor,
does not score local gloo service values, does not establish a signed time to
first token or time per output token consequence, and does not change the
VLLM-46 or VLLM-47 KV-bridge lane. The pinned CPU build succeeded, so the
bounded fallback was not taken and VLLM-50 was not consumed.

## Build and live-path evidence

The pinned path was taken. The first serious source-build attempt reached the
native extension and failed because the compiler did not inherit the isolated
NUMA header path. The second serious attempt supplied the include and library
paths directly, built the AVX2, AVX512 and default CPU extensions, installed
`vllm==0.27.1+cpu`, and passed CPU-platform and AVX2 dispatch preflight. The
CPU-compatible `torchcodec==0.14.0+cpu` replaced an incompatible multimedia
wheel before the live run. No protected vLLM environment was modified.

The first live launch was retained unscored after vLLM's automatic CPU-binding
guard rejected two workers on one NUMA node. The accepted launch explicitly
split physical cores between the two workers. Each fresh process then reported
world size two, rank zero and rank one, backend gloo, two completed manual
steps and the original logical request IDs at the output boundary.

Each accepted run contained 98 `all_reduce` calls and two `all_gather` calls.
Every step contained 49 all-reduces and one logits all-gather. The prefill
all-reduces carried 24,576 bytes with shape `[12, 1024]`; decode all-reduces
carried 4,096 bytes with shape `[2, 1024]`; each logits all-gather carried
98,432 bytes with shape `[2, 24608]`. World size was two, dtype was bfloat16,
element width was two bytes and the group tag was `tp:0` throughout. Layered
calls carried all 24 exact `model.layers.N` identities.

## Frozen families

| Family | Result | Evidence |
|---|---:|---|
| M1 call and metadata conservation | 0 of 2 | Both 100-call populations were complete, but the frozen `gather` kind disagreed with native `all_gather` at call 50 of each step. |
| M2 shape determinism | 1 of 1 | Removing service values produced the same ordered collective projection digest in both fresh runs. |
| M3 schema compatibility | 2 of 2 | The old absent-field record retained exact canonical bytes; new envelopes loaded and serialized to exact canonical bytes. |
| M4 comparison refusal | 1 of 1 | Equal all-reduce kind, 24,576 bytes and two ranks with different system/backend identities raised the dedicated refusal by default. |
| M5 acknowledged comparison | 1 of 1 | The same deliberate cross-environment comparison returned a result with `cross_environment_acknowledged=true`. |

All fatal guards passed. Dropping one call, changing one payload byte, changing
one second-run kind, deleting the optional envelope, bypassing the refusal or
removing the acknowledgement stamp made its owning predicate fail. The study
therefore remains interpretable and is reported as a behavioral failure, not
as a void run.

## Local timing interpretation

The seam recorded 200 positive host-monotonic service values from 171,305,000
to 373,712,983,000 picoseconds. A byte-only lower bound is tens of nanoseconds
for 4 KiB at a nominal 100 GB/s memory rate, while the conservative whole-run
ceiling is seconds per call. Every observation lies inside that deliberately
wide physical range. The spread is dominated by CPU gloo, process scheduling
and software synchronization, so these values are environment-labeled
diagnostics only and are explicitly unscored.

The complete bulk evidence remains under the append-only attempt label
`live-study-attempt-3-20260901`. This tracked summary retains only portable
digests and the result needed to reproduce the ruling.
