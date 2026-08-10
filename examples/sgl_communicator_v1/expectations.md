# Simulated SGLang communicator v1 expectations

## Freeze status and scope

This file freezes the SGL-11 zero-time first-slice expectations before the
SGLang communicator, runner wiring, tests, or any result-producing study run
exists. The two registered commands have check-only modes. Those modes must
run before this file is committed, and they must produce no artifacts.

The slice is deliberately component-only. It adds SGLang-name-mirrored
signatures, shape-only returns, the shared VLLM-14 event base,
`CollectiveWork` lowering, and the existing COMP-15
`ncclAllReduce`-shaped compatibility call. It does not project coordinator
work into a runtime authority, `CompletionEvent`, `StepResult`, TTFT, or TPOT.
It does not add communication time. SGL-13 is the specific successor for a
signed end-to-end metric relation after CORE-4 and CORE-5 provide the runtime
and completion contracts. SGL-14 owns native lower-stack operations, and
SGL-15 owns the real-dispatch bottleneck study and calibrated timing. Until
then, timestamp equality is a fatal unscored guard and no performance claim is
admissible.

## External-source audit before freeze

The audited checkout is SGLang main commit
`8f2a3ad6d7d68c58ae65b61a75bb2115449addca`, supplied to the check-only
harness as `SIMLLM_SGLANG_SOURCE_ROOT`. All source paths below are relative to
that checkout and were inspected before this freeze.

SGLang's `GroupCoordinator` is in
`python/sglang/srt/distributed/parallel_state.py:221-1785`. Its public
membership and device-communicator fields are declared at lines 232-260, and
its constructor resolves `ranks`, `world_size`, and `rank_in_group` at lines
262-385 before selecting PyNccl and the other optional device communicators.
The simulated constructor deliberately accepts resolved ranks and a
caller-owned clock instead of constructing torch process groups.

The first-slice callable signatures are exact name, parameter, default, and
return-annotation mirrors of that pinned class:

| Mirrored name | Pinned signature | Source |
|---|---|---|
| `all_reduce` | `all_reduce(self, input_: torch.Tensor) -> torch.Tensor` | `parallel_state.py:622-720` |
| `all_gather` | `all_gather(self, input_: torch.Tensor, dim: int = -1, output_tensor_list: Optional[List[torch.Tensor]] = None) -> torch.Tensor` | `parallel_state.py:1207-1279` |
| `broadcast` | `broadcast(self, input_: torch.Tensor, src: int = 0)` | `parallel_state.py:1387-1400` |
| `send` | `send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None` | `parallel_state.py:1743-1753` |
| `recv` | `recv(self, size: torch.Size, dtype: torch.dtype, src: Optional[int] = None) -> torch.Tensor` | `parallel_state.py:1755-1769` |

The rank-navigation properties `first_rank`, `last_rank`, `is_first_rank`,
`is_last_rank`, `next_rank`, and `prev_rank` are at
`parallel_state.py:525-557`. They use the same membership formulas as the
VLLM-14 base and remain part of the simulated surface.

The standard tensor-parallel model path reaches the mirrored boundary through
`python/sglang/srt/distributed/communication_op.py:18-20`, where
`tensor_model_parallel_all_reduce(input_)` calls
`get_tp_group().all_reduce(input_)`. The row-parallel layer selects that helper
after its local matrix multiply at
`python/sglang/srt/layers/linear.py:1563-1608`. The model runner also calls the
three-argument SGLang `all_gather` directly for optional DCP Q-projection
replication at `python/sglang/srt/model_executor/model_runner.py:880-916`.
These references identify why SGLang's added `output_tensor_list` parameter
must not be replaced by vLLM's two-argument surface.

The non-overlap scheduler calls `model_worker.forward_batch_generation` at
`python/sglang/srt/managers/scheduler.py:3626-3635`; the worker signature is at
`python/sglang/srt/managers/tp_worker.py:537-546`. The existing
`SimTpModelWorker` owns this supported seam. The new call must therefore
originate in its `SimModelRunnerStub` path, not by replacing SGLang globals or
real process groups.

## Decision-relevant live relation

The seam premise is that the pinned offline SGLang Engine on the CPU engine
path can reach `SimTpModelWorker`, and that `SimModelRunnerStub` can invoke a
simulated logical TP group without an SGLang fork. The registered smoke runs
one fixed request twice, once with the communicator flag absent and once with
logical TP size four enabled. Each run requests exactly two fabricated tokens,
so the enabled run must produce two model steps and this exact scored event
projection:

```text
(all_reduce, tp, 4096)
(all_reduce, tp, 4096)
```

Each event has one 14-name nested COMP-15 projection, in the frozen order in
the next section. Its timestamp and every nested timestamp equal the
corresponding step record's starting `virtual_time_ps`; the observation does
not advance the runner clock. The request must still return exactly two token
ids with the configured literal value 512.

This relation changes a design decision if it fails. If the real pinned
Engine cannot reach the communicator through the existing plugin, worker, and
stub runner seam, the current boundary is too shallow and SGL-13 must not bind
runtime projection there. The live reachability assertion is scored because
it executes against the real external runtime. Model existence, source pin,
event schema, configured rank membership, operation ids, timestamp equality,
and output echoes are fatal unscored guards.

## Shape and payload sweep

The import-free study crosses logical group size `G` in `{2, 4}` with
second-axis extent `E` in `{8, 16}`. Each input is a shape-only float32 tensor
with shape `(4, E)`, so its payload is exactly `16 * E` bytes. Every cell
invokes all five mirrored methods, and also exercises SGLang's
`output_tensor_list` all-gather form.

The scored upstream-referenced relations are:

- `all_reduce` returns shape `(4, E)` with the input dtype.
- `all_gather(input_, dim=1)` returns shape `(4, G * E)` with the input dtype.
- `all_gather(input_, dim=1, output_tensor_list=parts)` returns `None`, keeps
  exactly `G` caller-owned parts of shape `(4, E)`, and emits one input-payload
  observation.
- `broadcast` returns the same input object, `send` returns `None`, and
  `recv((4, E), input_.dtype)` returns the requested shape and dtype.
- every boundary reports `16 * E` input bytes. Increasing `E` from 8 to 16
  doubles all six observations exactly.
- increasing `G` from 2 to 4 doubles only the gathered second-axis extent and
  the required output-list length. It leaves non-gather shapes and input
  payloads unchanged.

All four Cartesian cells must pass. This is one scored shape family with four
instances and one scored payload-scaling family with two group-size
instances. Rank navigation, peer validation, event sequence, shared event
schema, singleton identity, and VLLM parity are fatal unscored guards.

## Frozen COMP-15 connection

One fresh four-rank SGLang coordinator uses one channel, 1,024-byte chunks,
two FIFO slots, rank zero, and a 4,096-byte input. Its first call lowers to
`CollectiveWork("all-reduce", (0, 1, 2, 3), 4096, "ring")`. The nested
projection is exactly:

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

Communicator construction prepends `ncclCommInitRank`, `ncclBuildRings`, and
`initChannel`, for 17 events on the first call. The study owns these literal
tuples and must not import an expected sequence from the implementation. The
sequence is a fatal unscored structural guard locally. Reaching the frozen
count and order through the pinned Engine is part of the scored live family,
as required by the wave-3 contract.

Every semantic operation currently enters this all-reduce-shaped surrogate.
SGL-14 owns native all-gather, broadcast, send, and receive entries and removal
of the COMP-15 ring-layout payload restriction. SGL-15 owns measured dispatch
cost. Neither residual is treated as implemented here.

## Bypass and evidence accounting

The optional environment flag is `SIMLLM_SGLANG_COMMUNICATOR_TP_SIZE`. When it
is absent, no simulated group is bound and no communicator event or sidecar is
created. The paired pinned-runtime runs must produce byte-identical step JSONL
and identical output token ids with the flag absent and present. The enabled
events are zero-time observations, so any byte difference is fatal. This is
the explicit identity baseline and is unscored.

The shared base remains the landed VLLM-14 implementation. Its source hash at
the freeze base is
`9b7b4bf6e49d6b35979ef8532873a35b4321453ecb78e9d58aa5b97adf85475e`.
This task does not edit that file. The component study also compares matching
vLLM and SGLang calls for event, work, stack, and shape parity. Existing vLLM
tests must pass unchanged. These are fatal unscored compatibility guards.

Evidence classes remain separate:

| Evidence class | Frozen count | Scored meaning |
|---|---:|---|
| Run configurations | 4 component cells plus 2 paired live configurations | Unscored configuration records |
| Shape relation family | 4 instances | Scored component behavior |
| Payload-scaling family | 2 instances | Scored component behavior |
| Real SGLang reachability family | 1 enabled live instance | Scored external-runtime behavior |
| COMP-15 literal sequence | 1 component reference plus 2 live nested checks | Fatal locally, included in live reachability |
| Flag-off and singleton bypass | 1 live byte comparison plus 1 component case | Fatal unscored |
| VLLM source and behavior parity | 1 source hash plus component and test checks | Fatal unscored |
| Unit tests | Separate executables | Not added to the behavioral denominator |

## Registered commands

The component study is registered as:

```bash
.venv/bin/python examples/sgl_communicator_v1/run_study.py \
  --check --run-dir "$SIMLLM_WAVE3_RUN_ROOT/sgl_communicator_v1"
```

Its pre-freeze dry run replaces `--check` with `--check-only` and omits
`--run-dir`. The pinned-runtime smoke is registered as:

```bash
"$SIMLLM_SGLANG_PYTHON" examples/sgl_communicator_v1/live_smoke.py \
  --run \
  --source-root "$SIMLLM_SGLANG_SOURCE_ROOT" \
  --model "$SIMLLM_SGLANG_MODEL" \
  --run-dir "$SIMLLM_WAVE3_RUN_ROOT/sgl_communicator_v1/live"
```

Its pre-freeze dry run replaces `--run` with `--check-only` and omits
`--run-dir`. The result report must cite the final expectations-only commit,
record exact chronology, keep scored and structural evidence separate, and
estimate the genuine-risk fraction for every scored family.
