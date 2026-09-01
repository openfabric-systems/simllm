# VLLM-48 live collective timing result

What ran: qualifying attempt 4 used two fresh live Granite inference processes
with the pinned vLLM 0.27.1 CPU source build, tensor parallel size two, gloo,
two explicit logical request IDs and two manual engine steps per process. It ran
after request-identity amendment `ad98074` and emitted the optional
collective-service envelope from the stock model-runner path.

What came out: attempt 4 was nonvoid and scored five of seven frozen behavioral
instances. It captured exactly 100 distributed operations in each run. All six
mutation controls now demonstrate a baseline-pass to mutant-fail transition.
The two M1 instances still fail because the frozen source oracle named the
final logits operation `gather`, while the standard pinned vLLM path executed
`all_gather` at ordinal 49 in both steps of both runs. Every other payload,
rank, dtype, element width, tensor shape, group and layer cell matched.

What it changes: the live seam, optional schema and comparator contracts have
valid mechanism evidence, but VLLM-48 remains open. The refuted kind family can
close only under successor expectations committed before another pair of
fresh live runs. VLLM-49 now expects `all_gather` on the standard A100 tensor
parallel logits path and owns the matched-hardware floor score.

What it does not change: this result does not calibrate an A100 or H200 floor,
does not score local gloo service values, does not establish a signed time to
first token or time per output token consequence, and does not change the
VLLM-46 or VLLM-47 KV-bridge lane. The pinned CPU build succeeded, so the
bounded fallback was not taken and VLLM-50 was not consumed.

## Pinned build

The pinned path was taken. The first serious source-build attempt reached the
native extension and failed because the compiler did not inherit the isolated
NUMA header path. The second serious attempt supplied the include and library
paths directly, built the AVX2, AVX512 and default CPU extensions, installed
`vllm==0.27.1+cpu`, and passed CPU-platform and AVX2 dispatch preflight. The
CPU-compatible `torchcodec==0.14.0+cpu` replaced an incompatible multimedia
wheel before the live run. No protected vLLM environment was modified.

## Campaign chronology and identity amendment

Nothing from the earlier campaign was removed or rescored as qualifying
evidence.

- Attempt 1 stopped before capture. vLLM's automatic binding required two NUMA
  nodes for two local workers, while the process was allowed one. It is retained
  as an unscored launch failure.
- Attempt 2 completed both fresh processes. Each finished the two logical
  requests in two manual steps and captured 100 calls, comprising 98
  `all_reduce` and two `all_gather` calls. The assigned engine IDs were the
  logical ID plus a random eight-character suffix, while completion and output
  maps used the original logical IDs. The then-frozen equality guard rejected
  both runs, so the recorded result is VOID with
  `run-1-request-identity` and `run-2-request-identity` failed. It has no
  behavioral score. A read-only counterfactual evaluation of its unchanged raw
  evidence under the final scorer passes every current fatal guard and produces
  five of seven, but it remains nonqualifying because that rule postdates the
  attempt.
- Attempt 3 ran after the scorer was loosened but before an amendment was
  committed. It also produced five of seven, but its identity ruling is
  post-specified and the attempt is retained as diagnostic evidence only.
- Amendment `ad98074` then froze the exact stock identity relation. Pinned vLLM
  stores the caller ID in `external_req_id` and assigns the internal value as
  the exact logical ID, one hyphen and eight lowercase hexadecimal characters
  (`vllm/v1/engine/input_processor.py:232-249`; the generator is pinned at
  `vllm/utils/__init__.py:11-12`). The scheduler must carry those exact internal
  IDs in order, while completion and output maps must carry the original IDs.
  The amendment explicitly requires two new processes and leaves the `gather`
  oracle unchanged.
- Attempt 4 is that fresh post-amendment attempt. Every fatal guard passed, so
  its five-of-seven failure is interpretable.

## Captured population and metadata

Each qualifying run contained 98 `all_reduce` calls and two `all_gather` calls.
Every step contained 49 all-reduces and one logits all-gather. The prefill
all-reduces carried 24,576 bytes with shape `[12, 1024]`; decode all-reduces
carried 4,096 bytes with shape `[2, 1024]`; each logits all-gather carried
98,432 bytes with shape `[2, 24608]`. World size was two, dtype was bfloat16,
element width was two bytes and the group tag was `tp:0` throughout. Layered
calls carried all 24 exact `model.layers.N` identities.

The attempt-4 capture hashes are
`3d1e33421e64df2cd40916283c102a5a283f980b1b004615ca8691fb4f28cd9a`
and
`582b746ee8ef2da8a8404311f4a4e140583d33b96e6f6b0543ab40cd49bfc256`.
Removing service values produced byte-identical ordered shape projections
across the two fresh runs.

## Refuted collective kind

The mismatch is a cross-platform vLLM default, not CPU-specific behavior.
`LogitsProcessor` stores `current_platform.use_all_gather()`
(`vllm/model_executor/layers/logits_processor.py:55`), the platform interface
default returns true (`vllm/platforms/interface.py:1102`), and `CpuPlatform`
inherits that interface without overriding the method
(`vllm/platforms/cpu.py:42`). The original freeze was therefore wrong about
the standard vLLM tensor-parallel logits path generally. VLLM-49 must expect
`all_gather` on its standard A100 path unless its pinned platform source
explicitly overrides that default.

## Frozen families

| Family | Result | Evidence |
|---|---:|---|
| M1 call and metadata conservation | 0 of 2 | Both 100-call populations were complete, but frozen `gather` disagreed with native `all_gather` at call 50 of each step. All other cells matched. |
| M2 shape determinism | 1 of 1 | Removing service values produced the same ordered collective projection in both fresh runs. |
| M3 schema compatibility | 2 of 2 | The old absent-field record retained exact canonical bytes; new envelopes loaded and serialized to exact canonical bytes. |
| M4 comparison refusal | 1 of 1 | Equal all-reduce kind, 24,576 bytes and two ranks with different system/backend identities raised the dedicated refusal by default. |
| M5 acknowledged comparison | 1 of 1 | The same deliberate cross-environment comparison returned a result with `cross_environment_acknowledged=true`. |

All attempt-4 fatal guards passed. A refused cross-environment comparison and
an acknowledged one both fired at the same `(all_reduce, 24576 bytes, 2 ranks)`
coordinate. The accepted result carried the acknowledgement stamp on the
comparison itself.

## Mutation controls

The first two controls compare only conserved M1 cells. They mask the one
already-refuted kind cell at zero-based ordinal 49 and no other cell. This
makes the unmutated observation pass before either mutation is applied.

| Control | Baseline | Mutant | Transition |
|---|---:|---:|---:|
| Drop first call | pass | fail | pass to fail |
| Increment first payload by one byte | pass | fail | pass to fail |
| Change second-run first kind | pass | fail | pass to fail |
| Delete optional envelope | pass | fail | pass to fail |
| Bypass cross-environment refusal | pass | fail | pass to fail |
| Remove acknowledgement stamp | pass | fail | pass to fail |

The controls are therefore mutation-sensitive independently of the known kind
refutation.

## Schema compatibility

An old step record without `collective_service` loaded and serialized to its
exact original canonical bytes. Attempt 4's new records loaded and serialized
to exact canonical bytes with the optional envelope present. No pre-existing
byte-locked fixture changed.

## Timing interpretation

Attempt 3's previously reported range is 171,305,000 to 373,712,983,000
picoseconds, or 0.171305 to 373.712983 milliseconds. The maximum is run 1's
first prefill collective, ordinal 0, with no layer. Its position at the first
distributed call after process startup gives it cold-start and rendezvous
character. The minimum is run 1 decode ordinal 33 at `model.layers.16`.

Attempt 4 recorded 200 positive host-monotonic values from 145,565,000 to
15,211,205,000 picoseconds, or 0.145565 to 15.211205 milliseconds. Its minimum
is run 1 decode ordinal 5 at `model.layers.2`; its maximum is run 2 prefill
ordinal 2 at `model.layers.0`.

For scale, moving 4,096 bytes at 100 GB/s takes 40.96 nanoseconds and moving
24,576 bytes takes 245.76 nanoseconds before synchronization or software cost.
The much larger observations include CPU gloo rendezvous, process scheduling
and software synchronization. They are environment-labeled diagnostics and
remain explicitly unscored.

## CUDA resolution

The CUDA path remains unexercised on this host. It records start and end events
on the calling stream. After the wrapped model step returns, it submits the
completed session to one ordered background resolver. The model-step thread
returns without waiting for `cuda_end.synchronize()`; event synchronization,
elapsed-time conversion and record append occur in the resolver. Unit evidence
locks the event order and verifies that synchronization occurs only after the
wrapped-step return marker. VLLM-49 must exercise this path on A100 hardware
and confirm ordered flush at process shutdown.

The complete bulk evidence remains under the append-only labels
`live-study-attempt-1-20260901` through `live-study-attempt-4-20260901`. This
tracked summary retains only portable labels, hashes and the rulings required
to reproduce the result.
