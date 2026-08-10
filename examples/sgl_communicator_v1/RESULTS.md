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

Integration review after commit `9884001` found one CI blocker and four
evidence-honesty gaps. This fix round is entirely post-specified. The original
expectations file remains unchanged, with SHA-256
`433d5a4ef77dece7927aeaab101cf9631edcde222d06ed1530c594f523854bd7`.
The repairs add a CI-runnable frozen-byte identity test, replace two
self-comparing source-line rows with AST-derived observations, correct the
output-list caller citation below, route the output-list observation through
the shared base, and narrow the genuine-risk claims.
Commit `922cd21` carries these post-specified implementation and regression
repairs.

The first fix-round live attempt completed both engine cases and refreshed
their case artifacts, then failed before rewriting aggregate evidence. The new
parent-side SGLang helper probe resolved an older installed SimLLM copy because
script execution had not put this repository first on the parent's import
path. The probe now binds the repository root before importing the adapter.
The next full invocation passed and rewrote `live_evidence.json`. This failure
and correction followed the freeze and are not represented as pre-registered
evidence.

## Post-specified source-audit correction

The frozen expectations incorrectly cite
`python/sglang/srt/model_executor/model_runner.py:880-916` as the reason to
preserve `output_tensor_list`. The calls at lines 913-916 are ordinary
`all_gather(tensor, dim=0)` calls and do not supply an output list. The two
actual pinned callers identified during integration review are:

- `python/sglang/srt/layers/dp_attention.py:993-994`, where
  `attn_tp_all_gather(output_list, input)` calls
  `get_attn_tp_group().all_gather(input, output_tensor_list=output_list)`;
- `python/sglang/srt/layers/attention/mamba/mixer2_rms_norm_gated.py:93-95`,
  where `forward_native` calls the attention-TP group with caller-owned parts.

The frozen file is not rewritten. The live check now derives both corrected
rows from AST. It also derives the model-runner line 913 and non-overlap
scheduler line 3633 from their actual call nodes rather than assigning the
expected literals to the observed table. A pin bump that moves or removes any
of these calls now fails check-only mode.

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
| Frozen flag identity fixture | both flag states equal 545 tracked LF bytes | Post-specified fatal CI guard |
| AST-derived pinned call-site audit | pass | Post-specified fatal source guard |
| Real SGLang output-list helper | pass | Post-specified unscored external-call guard |
| Event-stream ownership | truncation and duplicate-open guards pass | Post-specified fatal durability guard |
| Base-environment affected tests | 48 passed, 1 skipped | Separate executable |
| Pinned-SGLang affected tests | 47 passed, 2 skipped | Separate executable |
| Full base-environment suite | 582 passed, 4 skipped | Separate executable |
| Full pinned-SGLang suite | 581 passed, 5 skipped | Separate executable |

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

Integration review noted that the frozen doubling relation alone cannot
distinguish bytes from elements or input bytes from a different quantity that
is also linear in `E`. The component study now additionally requires every
event's absolute payload to equal `16 * E`. That absolute check is a
post-specified fatal guard and does not alter the frozen payload score.

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

The SGLang output-list branch now delegates observation to the shared base
all-gather path after validating caller-owned shapes. The event stream
documents that its first append truncates an existing sidecar. A process-wide
resolved-path claim makes a second stream instance fail before truncation, and
the unit test verifies that the first stream's bytes remain intact.

The CI identity test drives two records through `SglStepTranslator` and
`StepRecordStream` in both flag states. `observe_tp_step()` is interposed before
each append on the same `VirtualClock` used by the enabled coordinator. Both
streams equal the tracked 545-byte LF fixture exactly, while only the enabled
state emits events at `123000 ps` and `124000 ps`. This closes the review
blocker without relying on the live smoke.

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

As a post-specified external-call guard, the same pinned-runtime invocation
calls SGLang's real `attn_tp_all_gather` helper with two caller-owned
shape-only outputs. It returns `None` and reaches one
`("all_gather", "attn_tp", 128)` coordinator observation. This verifies the
SGLang-only call form through an upstream helper, but it is not added to the
frozen scored denominator.

## Genuine-risk fraction

The estimates below count only scored relations. They are not additional
scores.

| Family | Risk-bearing fraction | Plausible failure mode |
|---|---:|---|
| Shape relation | 4/4 instances, 100% | The SGLang-only output list could have been treated as a concatenated return, the wrong axis could have been multiplied, dtype could have been lost, or caller-owned part validation could have occurred after event emission. |
| Payload scaling | 0/2 instances, 0% | The frozen relation checks only doubling with `E`. Element counts, input bytes, and several wrong linearly scaled quantities all double, so the previously named failures cannot make this relation fail. The new absolute-byte guard is post-specified and unscored. |
| Real SGLang reachability | 1/1 instance, 100% | The accepted SGL-1 smoke already established plugin discovery, scheduler construction, and fabricated generation. The genuine new risk here is only the communicator seam: logical-group binding, one observation per step, sidecar order, timestamp equality, and unchanged step bytes. |

Two of three scored families and five of seven scored instances therefore
exercise a genuine failure risk, for a total risk-bearing fraction of `5/7`,
or about 71.4%. The payload score remains reported because it was frozen, but
it is not counted as risk-bearing. The live family's incremental evidence is
limited to the communicator seam over the accepted SGL-1 engine baseline.

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
