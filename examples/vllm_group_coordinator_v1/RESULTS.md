# Simulated vLLM GroupCoordinator v1 results

The VLLM-14 first slice passes every frozen scored relation. All four
shape-sweep cells and both payload-scaling instances pass, and the pinned real
vLLM v0.26.0 engine reaches the simulated coordinator without a fork. Two
live model steps emit the frozen `DP, TP, DP, TP` coordinator order and the
expected COMP-15 stack sequences beneath it.

This is a zero-time component result. It does not project communication into
`CompletionEvent` or `StepResult`, and it makes no TTFT, TPOT, or communication
latency claim.

## Expectations and chronology

Commit `29221e4` (`Freeze simulated coordinator expectations`) is the final
expectations-only ancestor. It contains the external-source audit, literal
stack oracles, scored relations, and check-only launchers. Both registered
check-only commands passed before that commit. No engine, implementation file,
measured value, or result artifact existed at the freeze.

All implementation and result-producing runs followed `29221e4`. Both defect
catches in the first implementation round were post-freeze, during
implementation and verification. The first result-producing live launch found
that script execution resolved an older installed SimLLM copy from a parent
checkout rather than this worktree. It stopped before engine
construction, then the launcher was bound explicitly to its repository root.

The next live launch constructed the pinned engine and passed. Subsequent
dual-environment verification found the second defect: when torch was
installed, shape-only `recv` tried to pass `ShapeDType` to `torch.empty`. The
repair keeps `ShapeDType` on the import-free tensor path and accepts a real
`torch.dtype` through the guarded optional import. Neither post-freeze catch is
represented as pre-registered evidence.

After implementation commit `3ed8d0e`, integration review identified the
silent COMP-15 payload domain, operation-ID gap on failed calls, and discarded
DP coordinator return. Commit `4645d2c` records the corrective expectations as
post-specified regressions before the corrective implementation. It does not
change the original freeze or scored denominator.

During final review verification, the first pinned-vLLM full-suite invocation
let a legacy study subprocess resolve the same older installed SimLLM copy.
The affected tests had already passed. Repeating the full gate with this
worktree explicitly on `PYTHONPATH` passed; no unrelated launcher was changed.
The single corrective live-smoke attempt then passed, including the new
serialized DP-padding assertion.

## Evidence accounting

Evidence classes remain separate.

| Evidence class | Result | Scored meaning |
|---|---:|---|
| Run configurations | 4 component cells plus 1 live smoke | Unscored configuration records |
| Shape relation family | 4/4 pass | Scored component behavior |
| Payload-scaling family | 2/2 pass | Scored component behavior |
| Real vLLM reachability family | 1/1 pass | Scored external-runtime behavior |
| Frozen COMP-15 reference | pass, 14 nested and 17 full events | Fatal unscored component guard |
| Singleton identity path | pass, 0 stack events | Fatal unscored bypass guard |
| Payload-domain review guards | pass | Post-specified fatal structural guards |
| Serialized DP padding | `(4, 1)` | Post-specified unscored live regression |
| VLLM-13 accepted baseline | pass | Fatal unscored bypass guard |
| Base-environment affected tests | 75 passed, 1 skipped | Separate executable |
| Pinned-vLLM affected tests | 74 passed, 2 skipped | Separate executable |
| Full base-environment suite | 472 passed, 4 skipped | Separate executable |
| Full pinned-vLLM suite, worktree-bound | 471 passed, 5 skipped | Separate executable |

The affected base skip requires torch. The two affected pinned-vLLM skips are
the existing absence-only adapter checks. Test counts are not added to the
seven scored relation instances.

## Shape and payload sweep

Each cell uses a float32 shape-only input `(4, E)`, so payload is `16 * E`
bytes. `G` is group size. All-reduce and receive preserve the input shape,
all-gather multiplies axis 1 by `G`, broadcast preserves object identity, and
send returns `None`.

| G | E | Input bytes | All-reduce shape | All-gather shape | Receive shape | Shape relation |
|---:|---:|---:|---|---|---|---|
| 2 | 8 | 128 | `(4, 8)` | `(4, 16)` | `(4, 8)` | pass |
| 2 | 16 | 256 | `(4, 16)` | `(4, 32)` | `(4, 16)` | pass |
| 4 | 8 | 128 | `(4, 8)` | `(4, 32)` | `(4, 8)` | pass |
| 4 | 16 | 256 | `(4, 16)` | `(4, 64)` | `(4, 16)` | pass |

At both `G=2` and `G=4`, doubling `E` doubles the observed payload for all
five methods from 128 to 256 bytes exactly. Changing `G` from 2 to 4 doubles
only the gathered axis; input payload and every non-gather result shape stay
fixed.

## COMP-15 and bypass guards

The fixed four-rank, 4,096-byte all-reduce lowers to
`CollectiveWork("all-reduce", (0, 1, 2, 3), 4096, "ring")`. Its coordinator
event contains this literal 14-event nested projection:

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
`initChannel`, for 17 events total. Every event reads the injected
`123000 ps` clock without advancing it.

The singleton path emits three upper observations for all-reduce, all-gather,
and broadcast, returns the exact identity results, emits zero stack events,
and leaves the clock at `123000 ps`. Replaying the existing one-request,
two-step VLLM-13 fixture still yields two sampled tokens, two records, the
frozen mirrored call order, and final clock `123000 ps`.

These facts are fatal structural guards. They do not increase the scored
denominator.

For the post-specified payload-domain guards, a four-rank zero-byte call emits
one upper event with `stack_disposition="zero_payload_bypass"` and zero nested
events. A rejected 10-byte call consumes neither an event sequence nor an
operation ID; the next valid call is `tp:all_reduce:0`. Nonzero multi-rank
payloads remain servable only when they satisfy COMP-15's documented lane and
chunk divisibility rules. VLLM-20 owns removal of that compatibility limit.

## Scored live vLLM relation

The final smoke used the cached Granite snapshot, vLLM v0.26.0,
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, V1 runner selection, the dotted
`simllm.adapters.vllm.SimWorker` class, offline model files, and a 64-block
logical KV pool. After engine initialization, the harness bound explicit
four-rank DP and TP simulated groups to the reached `SimModelRunner`.

Two model steps emitted exactly:

| Event | Operation | Group | Payload bytes | Nested stack events |
|---:|---|---|---:|---:|
| 0 | `all_reduce` | `dp` | 64 | 32 |
| 1 | `all_reduce` | `tp` | 4,096 | 14 |
| 2 | `all_reduce` | `dp` | 64 | 32 |
| 3 | `all_reduce` | `tp` | 4,096 | 14 |

Every DP event matched the literal 32-name oracle and every TP event matched
the literal 14-name oracle. The request returned fabricated token id `24577`
twice and wrote exactly two `atlahs-closed-loop-step-v1` records. Final virtual
time remained zero because this slice has no communication service time.

The copied runner now consumes the DP coordinator return's local-rank padded
token value, matching the pinned upstream projection, and serializes it as
`StepRecord.num_tokens_after_padding`. The final records contain `(4, 1)`.
That exact pair is a clearly labeled post-specified regression from integration
review, not a new scored instance.

The host again exposed an NVIDIA GeForce GTX 1660 Ti despite
`CUDA_VISIBLE_DEVICES=`. This run is scored evidence that the real external
runtime reaches and survives the communicator seam. It is not evidence for a
GPU-invisible platform, which remains VLLM-16.

## Genuine-risk fraction

The estimates below count only scored relations. They are not additional
scores.

| Family | Risk-bearing fraction | Why a competent implementation could fail |
|---|---:|---|
| Shape relation | 4/4 instances, 100% | A mirror could return the input for multi-rank all-reduce, gather the wrong axis, multiply the whole shape, lose dtype, or construct receive output through the wrong optional dependency. The last class of defect occurred before the final rerun. |
| Payload scaling | 2/2 instances, 100% | An implementation could count elements instead of bytes or record all-gather output bytes rather than the caller payload, breaking exact doubling. |
| Real vLLM reachability | 1/1 instance, 100% | Import resolution, worker selection, group binding, DP-before-TP order, or a shape/type mismatch could bypass or terminate the real engine path. The first launch exposed the import-resolution risk before engine construction. |

Thus all three scored families, and all seven scored instances, exercised a
plausible failure mode. The live family is the strongest evidence because it
depends on the pinned external runtime rather than only adapter-authored
objects. The payload-domain and DP-padding review checks are post-specified and
therefore excluded from this risk fraction and the scored denominator.

## Reproduction and stored evidence

The historical run used the same executable basenames, scripts, options and
pinned inputs; resolved machine-local paths are intentionally omitted. The
following is a portable post-run rendering, not a verbatim transcript. Source
the local configuration first, then run it from the repository root:

```bash
.venv/bin/python examples/vllm_group_coordinator_v1/run_study.py --check
"${SIMLLM_VLLM_PYTHON:?configure SIMLLM_VLLM_PYTHON}" \
  examples/vllm_group_coordinator_v1/live_smoke.py --run
```

The final machine-readable outputs remain outside Git in the machine-local
directory used for the historical run; its resolved historical path is intentionally
omitted. `component_results.json` contains the component
cells and guards. `live/live_evidence.json` contains the scored live
projection; `live/live_steps.jsonl` contains the two unchanged step records.
New runs default to
`${SIMLLM_DATA_ROOT}/vllm_group_coordinator_v1/`.

## Deliberate exclusions and residual work

No file under `simllm/compute` changed. The slice does not add runtime
projection, completion delivery, communication service time, real NCCL,
custom-allreduce, symmetric-memory behavior, replay tokens, executor record
construction, or SGLang wiring. Side calls stay omitted or inert.

VLLM-19 owns runtime and TTFT/TPOT reachability after CORE-4/5. VLLM-20 owns
native lower-stack operations for all-gather, broadcast, send, and receive in
place of the current all-reduce-shaped compatibility call. VLLM-21 owns real
dispatch-cost measurement and calibrated timing. SGL-11 remains untouched for
the wave-3 worker that reuses the torch-optional base.
