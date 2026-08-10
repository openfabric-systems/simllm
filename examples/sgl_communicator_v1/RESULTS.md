# Simulated SGLang communicator v1 results

The SGL-11 zero-time first slice passes every frozen scored relation. All four
shape-sweep cells and both payload-scaling instances pass. A paired offline
SGLang Engine run at the pinned commit reaches `SimTpModelWorker` and
`SimModelRunnerStub`: two enabled model steps emit the frozen TP event order,
while the flag-off and enabled step records remain byte-identical.

This is a zero-time component result. It does not project communication into a
runtime authority, `CompletionEvent`, or `StepResult`, and it makes no TTFT,
TPOT, or communication-latency claim.

## Expectations and chronology

Commit `b0c5b73` (`Freeze simulated SGLang communicator expectations`) is the
final expectations-only ancestor. Before staging that commit, the worktree had
one tracked registry edit and exactly three untracked files. Two untracked
dry-run harnesses encoded only frozen literals, source audits, and refusal to
execute target behavior; the third file was the expectations contract. At the
commit there were no unstaged or untracked files. Both registered check-only
commands printed their designed confirmations and produced no artifacts.

The freeze audited SGLang commit
`8f2a3ad6d7d68c58ae65b61a75bb2115449addca` before implementation. The audit
records the pinned `GroupCoordinator` definitions and supported call sites by
repository-relative file and line. No communicator implementation, engine
construction, measured value, or result artifact preceded `b0c5b73`.

Implementation and every result-producing run followed the freeze. The first
post-freeze component and paired live runs passed. Final code review then found
that inherited methods had compatible Python types but retained the vLLM
annotation spelling for optional arguments. Thin delegating overrides now
preserve SGLang's exact `Optional[List[torch.Tensor]]` and `Optional[int]`
surface without changing the shared implementation. This was a post-freeze
interface correction, not an outcome-dependent expectations edit. Both final
studies and both full test environments passed after the correction.

## Evidence accounting

Evidence classes remain separate.

| Evidence class | Result | Scored meaning |
|---|---:|---|
| Run configurations | 4 component cells plus 2 paired live configurations | Unscored configuration records |
| Shape relation family | 4/4 pass | Scored component behavior |
| Payload-scaling family | 2/2 pass | Scored component behavior |
| Real SGLang reachability family | 1/1 pass | Scored external-runtime behavior |
| Frozen COMP-15 reference | pass, 14 nested and 17 full events | Fatal unscored component guard |
| Singleton and flag-off bypass | pass | Fatal unscored identity guards |
| VLLM source and behavior parity | pass | Fatal unscored compatibility guard |
| Base-environment affected tests | 47 passed, 1 skipped | Separate executable |
| Pinned-SGLang affected tests | 46 passed, 2 skipped | Separate executable |
| Full base-environment suite | 581 passed, 4 skipped | Separate executable |
| Full pinned-SGLang suite | 580 passed, 5 skipped | Separate executable |

Test counts and structural guards are not added to the seven scored relation
instances.

## Shape and payload sweep

Each cell uses a float32 shape-only input `(4, E)`, so payload is `16 * E`
bytes. `G` is logical group size. All-reduce and receive preserve the input
shape, ordinary all-gather multiplies axis 1 by `G`, the output-list form
preserves `G` caller-owned input-shaped parts, broadcast preserves object
identity, and send returns `None`.

| G | E | Input bytes | All-reduce shape | All-gather shape | Output-list parts | Shape relation |
|---:|---:|---:|---|---|---:|---|
| 2 | 8 | 128 | `(4, 8)` | `(4, 16)` | 2 | pass |
| 2 | 16 | 256 | `(4, 16)` | `(4, 32)` | 2 | pass |
| 4 | 8 | 128 | `(4, 8)` | `(4, 32)` | 4 | pass |
| 4 | 16 | 256 | `(4, 16)` | `(4, 64)` | 4 | pass |

At both group sizes, increasing `E` from 8 to 16 doubles all six input-payload
observations from 128 to 256 bytes exactly. Increasing `G` from 2 to 4 doubles
only the gathered axis and output-list length; all input payloads and
non-gather result shapes remain fixed.

## COMP-15, parity, and bypass guards

The fixed four-rank, 4,096-byte call lowers to
`CollectiveWork("all-reduce", (0, 1, 2, 3), 4096, "ring")`. Its coordinator
event contains the literal frozen 14-event nested projection:

```text
ncclAllReduce
ncclEnqueueCheck
scheduleCollTasksToPlan
calcCollChunking
ncclLaunchKernel
ncclKernelMain
runRing
genericOp
genericOp
genericOp
genericOp
genericOp
genericOp
simllmKernelComplete
```

The full stack prepends `ncclCommInitRank`, `ncclBuildRings`, and
`initChannel`, for 17 events. The singleton case emits four upper observations
but no stack events and leaves the injected clock at `123000 ps`.

The landed vLLM communicator retained frozen SHA-256
`9b7b4bf6e49d6b35979ef8532873a35b4321453ecb78e9d58aa5b97adf85475e`.
Matching four-rank SGLang and vLLM all-reduces produced identical shape,
coordinator-event, and stack-event values. These are fatal compatibility
guards, not scored behavior.

## Scored live SGLang relation

The paired smoke used SGLang `0.0.0.dev1+g8f2a3ad6d`, the cached offline
Granite model, the CPU engine, non-overlap scheduling, the SimLLM plugin entry
point, one request, and two generated tokens with configured token id 512.
Each configuration reached the scheduler subprocess's `SimTpModelWorker` and
wrote two `atlahs-closed-loop-step-v1` records.

The enabled run emitted exactly:

| Event | Operation | Group | Payload bytes | Nested events | Timestamp ps |
|---:|---|---|---:|---:|---:|
| 0 | `all_reduce` | `tp` | 4,096 | 14 | 0 |
| 1 | `all_reduce` | `tp` | 4,096 | 14 | 58,483,200 |

Each event timestamp and every nested timestamp equal the corresponding step
record's starting `virtual_time_ps`. Observation therefore adds no time. The
baseline created no communicator sidecar. Both cases returned `[512, 512]`,
and both 542-byte step files have SHA-256
`6dfface7ab3d55c5344baa01ce9c4f5b797074bf64c9f3ea3e986ba5b726e18a`.
The byte comparison passed exactly.

## Genuine-risk fraction

The estimates below count only scored relations. They are not additional
scores.

| Family | Risk-bearing fraction | Plausible failure mode |
|---|---:|---|
| Shape relation | 4/4 instances, 100% | The SGLang-only output list could have been treated as a concatenated return, the wrong axis could have been multiplied, dtype could have been lost, or caller-owned part validation could have occurred after event emission. |
| Payload scaling | 2/2 instances, 100% | The adapter could have counted elements or gathered-output bytes instead of input bytes, breaking exact doubling or group-size invariance. |
| Real SGLang reachability | 1/1 instance, 100% | Plugin discovery, scheduler subprocess construction, stub binding, event streaming, or call placement could have bypassed the coordinator, reordered observations, changed step bytes, or terminated the pinned Engine. |

All three scored families and all seven scored instances therefore exercised a
genuine failure risk. The live family is the strongest evidence because it
depends on the pinned external runtime rather than only adapter-authored
objects.

## Reproduction and stored evidence

From the repository root, set `SIMLLM_WAVE3_RUN_ROOT`,
`SIMLLM_SGLANG_PYTHON`, `SIMLLM_SGLANG_SOURCE_ROOT`, and
`SIMLLM_SGLANG_MODEL`, then run:

```bash
.venv/bin/python examples/sgl_communicator_v1/run_study.py \
  --check --run-dir "$SIMLLM_WAVE3_RUN_ROOT/sgl_communicator_v1"

"$SIMLLM_SGLANG_PYTHON" examples/sgl_communicator_v1/live_smoke.py \
  --run \
  --source-root "$SIMLLM_SGLANG_SOURCE_ROOT" \
  --model "$SIMLLM_SGLANG_MODEL" \
  --run-dir "$SIMLLM_WAVE3_RUN_ROOT/sgl_communicator_v1/live"
```

Machine-readable outputs remain outside Git under
`$SIMLLM_WAVE3_RUN_ROOT/sgl_communicator_v1`. `component_results.json`
contains the shape cells and component guards. `live/live_evidence.json`
contains the paired live relation; each case directory contains its own step
records and case evidence, and only `live/enabled` contains communicator
events.

## Deliberate exclusions and residual work

No shared vLLM communicator code changed. This slice adds no real torch process
group, NCCL execution, device-communicator fast path, communication timing,
runtime projection, completion delivery, TTFT/TPOT contribution, overlap
scheduler support, or htsim closed-loop call. No C++ changed, so native
cmake/ctest is not applicable.

- SGL-11 (Completeness; P1; L) remains open for call surfaces reached only by
  future accepted DCP-attention and MoE adapter modes, including
  `all_gather_into_tensor`, `reduce_scatter_tensor`, `all_gatherv`, and
  `reduce_scatterv`. The current dense TP mode does not claim those paths.
- SGL-13 (Completeness; P1; L) owns the single-authority runtime projection
  into `CompletionEvent`, `StepResult`, and signed TTFT/TPOT evidence after
  CORE-4 and CORE-5.
- SGL-14 (Precision; P1; M) owns native operation-specific lowerings and
  removal of the COMP-15 ring-layout payload restriction, with an exact
  compatibility bypass.
- SGL-15 (Precision; P1; L) owns the deferred bottleneck study: pinned-SGLang
  Python dispatch, custom-op routing, device-communicator selection, and
  synchronization-stall measurements, followed by calibrated call cost and a
  signed metric effect once SGL-13 is live.
