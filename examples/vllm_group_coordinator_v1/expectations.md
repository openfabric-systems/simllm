# Simulated vLLM GroupCoordinator v1 expectations

## Freeze status and scope

This file freezes the VLLM-14 first-slice expectations before the simulated
coordinator, worker wiring, tests, or any result-producing study run exists.
The two registered commands have check-only modes. Those modes must be run
before this file is committed, and they must create no result artifact.

This slice is deliberately component-only. It adds name-mirrored signatures,
shape-only returns, zero-time observability events, `CollectiveWork` lowering,
and the existing COMP-15 `ncclAllReduce`-shaped stack call. It does not project
coordinator work into a runtime authority, `CompletionEvent`, `StepResult`,
TTFT, or TPOT. It does not add a communication timing model. VLLM-19 is the
specific successor that may make this component live-reachable after CORE-4
and CORE-5 establish the required runtime and completion contracts. Until
then, any timestamp equality is a fatal unscored structural guard and no
end-to-end performance claim is admissible.

## External-source audit before freeze

The audited install is the vLLM v0.26.0 package from a machine-local pinned
environment whose resolved historical path is intentionally omitted. For a
current reproduction, `SIMLLM_VLLM_PYTHON` selects the compatible
interpreter.
All paths and line numbers below refer to that pinned install and were checked
before this freeze.

The mirrored callable signatures in
`distributed/parallel_state.py` are exact:

| Mirrored name | Pinned signature | Source |
|---|---|---|
| `all_reduce` | `all_reduce(self, input_: torch.Tensor) -> torch.Tensor` | `distributed/parallel_state.py:641-668` |
| `all_gather` | `all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor` | `distributed/parallel_state.py:670-689` |
| `broadcast` | `broadcast(self, input_: torch.Tensor, src: int = 0)` | `distributed/parallel_state.py:745-758` |
| `send` | `send(self, tensor: torch.Tensor, dst: int | None = None) -> None` | `distributed/parallel_state.py:1188-1193` |
| `recv` | `recv(self, size: torch.Size, dtype: torch.dtype, src: int | None = None) -> torch.Tensor` | `distributed/parallel_state.py:1195-1202` |

The public membership fields are `rank`, `ranks`, `world_size`, `local_rank`,
and `rank_in_group` at `distributed/parallel_state.py:368-385`. The real
constructor selects the caller's membership and fills `ranks`, `world_size`,
and `rank_in_group` at `distributed/parallel_state.py:387-450`. The simulated
constructor intentionally accepts an already-resolved rank list and a
caller-supplied clock instead of creating real torch process groups. The
callable methods above remain signature mirrors.

The rank navigation properties are `first_rank`, `last_rank`,
`is_first_rank`, `is_last_rank`, `next_rank`, and `prev_rank`, with their
membership formulas at `distributed/parallel_state.py:563-595`. The tensor
parallel convenience functions read `world_size` and `rank_in_group` at
`distributed/parallel_state.py:2042-2049`. These fields and properties are
therefore part of the required simulated surface, not diagnostic extras.

The empty-computation shape referents are author-independent upstream fake
implementations. `all_reduce_fake` returns `torch.empty_like(tensor)` at
`distributed/parallel_state.py:138-139`. `all_gather_fake` multiplies exactly
the selected dimension by `world_size` and preserves dtype and device at
`distributed/parallel_state.py:170-175`. The single-rank real all-reduce and
all-gather paths return the input unchanged at
`distributed/parallel_state.py:656-658,671-674`. Broadcast returns the input
with unchanged shape after validating a group-local source rank at
`distributed/parallel_state.py:745-758`.

The copied runner's DP referent is also audited. The V1 runner invokes
`coordinate_batch_across_dp` only when `data_parallel_size > 1` at
`vllm/v1/worker/gpu_model_runner.py:3946-3964`. Its helper constructs one
`int32` tensor of shape `(4, dp_size)` and performs one all-reduce at
`vllm/v1/worker/dp_utils.py:36-54`. The skeleton mirror must issue that same
shape-only DP coordination call through its simulated DP coordinator. This
does not claim to reproduce the numerical reduction.

## Decision-relevant live relation

The communicator-layer seam premise is that a stock, in-process vLLM engine
can reach `SimModelRunner`, and that the copied runner can call a simulated
coordinator without a vLLM fork. One live Granite request produces two model
steps. Before generation, the smoke binds four-rank simulated DP and TP groups
to the reached runner. Each step must emit coordinator calls in this exact
order:

1. DP `all_reduce` over the upstream-shaped `(4, 4)` int32 coordination tensor,
   with payload 64 bytes.
2. TP `all_reduce` over the frozen shape-only model tensor, with payload 4,096
   bytes.

The two-step scored live projection is therefore
`((all_reduce, dp, 64), (all_reduce, tp, 4096))` repeated exactly twice. The
four coordinator timestamps and every nested stack timestamp equal the
runner's caller-supplied virtual clock. The DP call must return a shape-correct
tensor so the runner continues to sampling, and the request must still produce
the two configured fabricated tokens.

This relation changes a design decision if it fails. If the real engine cannot
reach this coordinator through the dotted `SimWorker` and copied runner seam
without modifying vLLM, the communicator-layer coupling premise fails. The
next design would need to reconsider the seam, including an upstream hook or a
maintained fork, before SGL-11 shares this base.

The live assertions are scored because they execute against the real pinned
runtime. Model existence, configured group sizes, event-schema identity,
unchanged virtual time, and fixture echoes are fatal unscored guards.

## Shape and payload sweep

The import-free study crosses group size `G` in `{2, 4}` with second-axis
extent `E` in `{8, 16}`. Each input is a shape-only float32 tensor with shape
`(4, E)`, so payload bytes are exactly `16 * E`. Every cell invokes all five
mirrored methods.

The scored upstream-referenced relations are:

- `all_reduce` returns shape `(4, E)` with the input dtype.
- `all_gather(input_, dim=1)` returns shape `(4, G * E)` with the input dtype.
- `broadcast` returns the same input object and therefore shape `(4, E)`.
- `send` returns `None`.
- `recv((4, E), input_.dtype)` returns shape `(4, E)` with the requested
  dtype.
- every boundary event reports payload `16 * E` bytes. Increasing `E` from 8
  to 16 doubles all five payload observations exactly.
- increasing `G` from 2 to 4 doubles only the gathered second-axis extent. It
  leaves all-reduce, broadcast, receive, and input payload shapes unchanged.

All four Cartesian cells must pass. This is one scored shape family with four
instances and one scored payload-scaling family with two group-size instances.
Rank navigation, local-rank validation, source and peer validation, event
sequence monotonicity, and singleton identity behavior are fatal unscored
structural guards.

## Frozen COMP-15 connection

The COMP-15 stack is read-only in this slice. The reference is the landed
implementation at base commit `6aa3a76` and its exact intra-node fixture in
`tests/test_nccl_stack.py`. One fresh four-rank coordinator uses one channel,
1,024-byte chunks, two FIFO slots, rank 0, and a 4,096-byte input. Its first
`all_reduce` must lower to
`CollectiveWork("all-reduce", (0, 1, 2, 3), 4096, "ring")` and emit one
coordinator boundary event. Beneath that event, the literal stack projection
is:

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

Including communicator construction, the full stack contains the three-event
prefix `ncclCommInitRank`, `ncclBuildRings`, `initChannel`, then the fourteen
events above. The study owns literal tuples for both the fixed coordinator
projection and this stack projection. It must not import an expected sequence
from the implementation.

This exact sequence is a fatal unscored structural guard. It is not behavioral
evidence because it is a skeleton call order. In the real-runtime smoke, the
fact that the pinned engine reaches the boundary and produces the frozen
counts and order is scored live evidence under the wave-2 rule.

The first slice uses the existing `ncclAllReduce`-shaped entry beneath every
multi-rank boundary while preserving the semantic operation in
`CollectiveWork`. VLLM-20 owns native COMP stack entries for all-gather,
broadcast, send, and receive. VLLM-21 owns measured real communicator dispatch
cost and any later calibrated timing model. Neither deferred item may be
silently treated as implemented here.

## Bypass and evidence accounting

A singleton simulated group is the explicit identity path. It emits its
coordinator observation and `CollectiveWork`, returns the exact upstream
identity or shape result, emits no COMP-15 ring event, and leaves the virtual
clock unchanged. Binding no multi-rank groups to `SimModelRunner` must preserve
the accepted VLLM-13 skeleton baseline: the same step records, fabricated
tokens, mirrored worker calls, and zero step latency. This is a fatal unscored
bypass check because the component does not yet enter the metric chain.

Evidence classes remain separate:

| Evidence class | Frozen count | Scored meaning |
|---|---:|---|
| Run configurations | 4 sweep cells plus 1 live smoke | Unscored configuration records |
| Shape relation family | 4 instances | Scored component behavior |
| Payload-scaling family | 2 instances | Scored component behavior |
| Real vLLM reachability family | 1 live instance | Scored external-runtime behavior |
| COMP-15 literal sequence | 1 reference plus live nested checks | Fatal structural locally, scored only as part of live reachability |
| Singleton and no-binding bypass | 1 component case plus baseline comparison | Fatal unscored |
| Unit tests | Separate executable | Not added to the behavioral denominator |

## Registered commands

The historical dry runs used the same executable basenames, scripts, options
and pinned inputs; resolved machine-local paths are intentionally omitted. The
following blocks are portable post-freeze renderings, not verbatim
transcripts. Source the local configuration first.

The deterministic component study is registered as:

```bash
.venv/bin/python examples/vllm_group_coordinator_v1/run_study.py --check
```

Its pre-freeze dry run used the historical resolved form of the same command
with `--check-only`. The live-smoke rendering is:

```bash
"${SIMLLM_VLLM_PYTHON:?configure SIMLLM_VLLM_PYTHON}" \
  examples/vllm_group_coordinator_v1/live_smoke.py --run
```

Its pre-freeze dry run used the historical resolved form with `--check-only`.
The result report must cite the final expectations-only commit, record the
exact chronology, keep structural and scored counts separate, and estimate the
genuine-risk fraction for each scored family.
