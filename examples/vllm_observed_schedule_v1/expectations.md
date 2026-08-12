# vLLM observed schedule v1 expectations

This document freezes the VLLM-22 and TRAF-13 expectations before the
source-backed schedule producer, traffic extension, result-producing replay,
or measured result exists. The repository source at this boundary is commit
`cede92930a469bd0be2f2c588866885c9e0e3618`.

The evidence is authored against official vLLM v0.26.0 commit
`568afb3a13806beb53bb2e6bd518269357b237c0`. A run must record the source
files and version it actually observes. The authored-against identity and the
observed identity are independent provenance fields. No check may require a
live checkout, package, or submodule pin to equal the authored-against commit.

## Pinned-source audit

The following vLLM v0.26.0 source was read before this freeze. Paths and line
numbers are relative to the installed `vllm` package.

- `model_executor/models/granitemoe.py:117-136` applies the router and fused
  experts in `GraniteMoeMoE.forward`. Lines 264-282 execute attention before
  the MoE block in one decoder layer. Lines 315-340 call the decoder layers in
  a synchronous Python loop. A single batch therefore preserves each layer's
  data dependency and supplies no legal next-layer overlap by itself.
- `config/parallel.py:186-196` identifies
  `deepep_high_throughput` as an MoE all-to-all backend. Lines 209-223 define
  the DBO enable and decode/prefill token thresholds. Lines 535-540 make DBO a
  two-microbatch mode.
- `v1/worker/gpu_model_runner.py:3860-3989` recognizes uniform decode and asks
  `coordinate_batch_across_dp` whether to microbatch only when DP is greater
  than one. Lines 4210-4244 use that answer to create the microbatch slices.
  Lines 4377-4400 then invoke the model under the resulting forward context.
  Lines 5405-5420 select `UBatchWrapper` when ubatching is configured.
- `v1/worker/dp_utils.py:164-225` makes DP coordination the authority for the
  microbatch decision and explicitly returns no microbatch for DP size one.
  `v1/worker/ubatch_utils.py:38-46` selects the decode or prefill threshold.
  Lines 63-114 split the ordered token span into the configured number of
  slices and derive the overlapping request slices from cumulative scheduled
  token counts.
- `v1/worker/gpu_ubatch_wrapper.py:113-128` creates one communication stream
  shared by the microbatches. Lines 305-341 launch the two model forwards in
  separate cooperative threads, start microbatch zero first, join both, and
  restore result order. Lines 441-490 select the single-batch bypass or bind
  the common compute stream and per-microbatch metadata.
- `v1/worker/ubatching.py:77-92` records and waits on compute and communication
  stream events. Lines 94-147 define the cooperative CPU yield plus the exact
  compute-to-communication and communication-to-compute event waits. These
  waits, the two shared logical streams, and their FIFO order are the legal
  concurrency authority. No overlap fraction occurs in this source.
- `model_executor/layers/fused_moe/modular_kernel.py:1162-1229` runs async
  dispatch preparation, registers a DBO receive hook, yields, and waits for
  the receiver before expert compute. Lines 1312-1378 do the corresponding
  async combine and receive. Lines 1422-1473 fix each layer's semantic order
  as prepare, fused experts, then finalize.
- `model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py:106-192`
  captures the preceding compute event, yields before dispatch, launches the
  DeepEP dispatch on the communication stream, and waits before returning to
  compute. Lines 347-410 perform the analogous combine, communication event,
  and stream switch.
- `v1/worker/gpu_model_runner.py:4394-4429` computes logits only after the
  model forward returns. The completion frontier must therefore follow every
  layer combine and the final logits work for the requests in that
  microbatch.

The audited files have these SHA-256 identities:

| Source file | SHA-256 |
|---|---|
| `vllm/model_executor/models/granitemoe.py` | `b60e452c3f28b25aa104c88869daa25c06a7fb6ed45bd34e908fa6a8395efda1` |
| `vllm/config/parallel.py` | `a6581c267ab265e24905d2f5caa514482c28359f71380c6f894ceab25aa22541` |
| `vllm/v1/worker/gpu_model_runner.py` | `81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0` |
| `vllm/v1/worker/dp_utils.py` | `2ba84bbf92a25e756576918bfb215c1fb387b006899885d811bdb2f774e843a9` |
| `vllm/v1/worker/ubatch_utils.py` | `0b727aaa1c7072152e25f684ddc2fc9790c430eddd862e610c97a8e3e9febdc4` |
| `vllm/v1/worker/gpu_ubatch_wrapper.py` | `4eae50c929f3ba873072c13291c7140be3dd00d4a5b623170dff44754519c021` |
| `vllm/v1/worker/ubatching.py` | `40391241c564feb5f16c77898ae6ae152ed6e71a4682e2a406387785d8de02d7` |
| `vllm/model_executor/layers/fused_moe/modular_kernel.py` | `f78ae626babfd69f3c6ba37eef9c8f5186f28cd9064f566e341ca0c9e0fdb9b9` |
| `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py` | `465cdf1d6cee91b2ee8c2e43abbea6e8408976e3048c10f44c089f34b415bc60` |

## Frozen replay and active source mechanism

The input is the existing three-request Granite greedy replay:

- capture schema `simllm-preplay-trace-v1`, 120 rows, SHA-256
  `5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6`;
- replay-run SHA-256
  `b4d38a09011caf6de159c22133264d62a2727063496953f4337b17d79cfde93e`;
- routed-experts SHA-256
  `24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f`;
- model `ibm-granite/granite-3.0-1b-a400m-instruct`, revision
  `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`;
- 24 MoE layers, hidden width 1,024, 32 experts, top-k 8, BF16
  activations, and four resident experts per EP rank;
- requests `r0`, `r1`, and `r2`, with prompt lengths 22, 12, and 20 and
  output lengths 24, 8, and 32.

The enabled observation cell fixes TP 1, DP/EP 8, PP 1,
`enable_expert_parallel=True`,
`all2all_backend=deepep_high_throughput`, `enable_dbo=True`, two
microbatches, decode threshold 2, and prefill threshold 512. The 54-token
initial prefill remains a single batch. Every decode interval used in `r0`'s
TPOT has at least two active one-token requests and selects DBO. A one-request
decode tail selects the audited single-batch bypass.

The producer may translate only these source facts. It may not start from the
serial compatibility graph and delete edges. It may not contain an overlap
percentage, duration, discount, fitted concurrency constant, or random
choice. An unsupported model, all-to-all backend, DBO shape, or request split
must fail explicitly rather than emit a plausible schedule.

## Frozen schedule relations

For a single-batch Granite step, each layer has pre-dispatch compute,
dispatch, expert compute, and combine in that exact causal order. Consecutive
layers remain causally chained. The final logits operation follows layer 23's
combine. There are 24 unique ordered layers and exactly 48 unique semantic
MoE sites, one dispatch and one combine per layer.

For a qualifying DBO decode step, each semantic site has one invocation per
microbatch, so there are 96 collective invocations but still exactly 48
unique `(layer, site)` identities. Microbatch zero and one carry disjoint
request correlation for the frozen uniform-decode slices. The shared compute
queue is FIFO, the shared communication queue is FIFO, dispatch waits on its
microbatch's pre-dispatch compute, expert compute waits on dispatch, combine
waits on expert compute, and each microbatch's next layer waits on its own
combine. No edge orders the other microbatch's ready compute behind a
communication operation unless the audited event wait requires it.

The operation tuple must preserve source submission order. Every operation
must carry its layer, microbatch, logical queue, request IDs, dependency kind,
and stable step identity. The completion frontier must contain exactly one
request-correlated endpoint for each active microbatch and must follow the
last combine plus logits work. Lowering must preserve every tuple position,
queue, explicit edge, correlation, and completion endpoint.

Traffic rebinding must partition each routed pair table by the correlated
microbatch requests. The microbatch tables must be disjoint for the frozen
uniform-decode steps, their union must equal the full-step request table, and
each aggregate pair table must equal the sum of its request rows. Completion
reduction must return the original request identities.

## Placements and physical bounds

Both cells use the identical semantic EP group, routed bytes, compute
provider, operation graph, requests, and completion policy. Only the physical
rank placement changes:

1. `single-node` maps all eight EP ranks to one eight-GPU node. Remote GPU
   pairs use the accepted 450 GB/s NVLink-class byte rate.
2. `cross-node` maps the same eight EP ranks one per node. Remote pairs use
   one 400 Gbit/s RNIC per GPU.

Before reading results, the physical bounds are:

- Compute floor: 452,984,832 resident layer-weight bytes plus 100,663,296 LM
  head bytes at 8 TB/s require at least 69,206,016 ps. The frozen 0.7 roofline
  efficiency predicts about 98,865,737 ps before communication.
- Single-node communication floor: the supplied 1.5 MB peak per-rank decode
  egress needs about 3.33 microseconds at 450 GB/s. Even serializing the full
  supplied 11.6 MB on one such link adds only about 25.8 microseconds, so a
  roughly 99 microsecond compute term should dominate.
- Cross-node communication floor: 1.5 MB through one 400 Gbit/s RNIC needs
  about 30 microseconds. A deliberately conservative ceiling serializes all
  11.6 MB through one RNIC, about 232 microseconds, then adds 100 microseconds
  of compute. Any decode step below the 69.2 microsecond compute floor or
  above 332 microseconds is a defect in the model, harness, or reading.

The cross-node overlap reduction must exceed the single-node reduction. This
is the independent bandwidth scaling check. A reduction that is equal across
the two placements, or grows by less than 5x, contradicts the claimed
serialization mechanism even if both individual bands pass.

The end-to-end plausibility check compares the roughly 100 microsecond decode
floor with comparable accelerator serving behavior. A result implying tens
of thousands of tokens per second for this 400M-active-parameter model is not
accepted merely because it matches the simulator's own equation.

## Frozen decision relations

The complete chain is:

```text
vLLM scheduler step and active parallel configuration
  -> adapter-emitted ExecutionObservations
  -> ObservedStepLowerer and routed traffic binding
  -> CoarseDeviceRuntime and CompletionEvent
  -> CompletionReducer and StepResult
  -> request r0 TTFT and TPOT
```

All raw observed, serial, and perturbed metrics must be collected before any
exact or fatal check is evaluated.

### Observed versus serial TPOT

For `r0`, `serial TPOT - observed TPOT` must be strictly positive in both
placements and fall in these inclusive bands:

| Placement | Reduction band |
|---|---:|
| `single-node` | 1,000,000 through 5,000,000 ps |
| `cross-node` | 20,000,000 through 130,000,000 ps |

Each placement is one genuine-risk instance. TTFT is reported but is not a
scored instance because the initial prefill stays on the source-serial path.
Observed TTFT must equal serial TTFT exactly in each cell.

### Dependency perturbation

For every qualifying DBO decode step, copy the raw observations and add one
whole-operation dependency from microbatch zero's layer-12 combine to
microbatch one's layer-12 expert compute. Change no tuple position, queue,
work item, correlation, completion endpoint, or other edge. This removes one
source-legal combine/expert overlap without changing bytes or service demand.

The perturbed TPOT must be greater than observed TPOT and no greater than
serial TPOT. The inclusive minimum increases are 100,000 ps for
`single-node` and 5,000,000 ps for `cross-node`. Each placement is one
genuine-risk instance. A lowerer that ignores observations cannot pass this
relation.

### Exact serial off path

With observations absent, the sink must delegate directly to
`SerialStepLowerer`. The accepted two-layer fixture remains:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Canonical execution graph JSON plus LF | 4,127 | `aa3c836fe559973a7bf0940384c2e8a84e6af84e0fbd2c02d3b89774ee0c8e2d` |
| Serial graph-only GOAL | 1,880 | `7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6` |

For every frozen Granite step, absent observations must also preserve the
serial graph JSON, graph-only GOAL, operation timestamps, completion event
order, `StepResult`, and request metrics byte for byte against direct serial
lowering. These are fatal unscored identity guards.

## Evidence accounting and entailment

The behavioral headline contains four genuine-risk instances: two raw TPOT
reductions and two raw dependency-perturbation increases. The expected result
is `4/4 = 100%`.

The runner must evaluate all four scored relations directly from the raw
metric rows before source hashes, configuration echoes, graph preservation,
pair-table conservation, completion identities, serial digests, or exact
metric regression rows. None of those later checks may entail a scored
relation. If the producer qualification fails before a metric row exists, the
headline is `0/0, blocked before behavioral execution`, not `0/4`.

Source and capture identities, schedule counts, queue and edge preservation,
request attribution, byte conservation, completion-frontier identity,
disabled-path identity, physical bounds, and configuration-forced inactive
paths are fatal unscored guards. Unit tests, source audit, the live adapter
probe, and native-tool executions remain separate evidence classes. Counts
from different classes are never added.

## Closure scope frozen from the registries

VLLM-22 currently requires:

> add a real source-backed vLLM `ExecutionObservations` producer for each
> translated step.

The live replay must show one emitted observation object for every nonempty
translated step and `None` only on the explicit producer-disabled run.

> observe all 24 ordered layers and exactly 48 semantic MoE sites, one
> dispatch and one combine per layer, with exact submission order, logical
> streams, program-order and event-wait dependencies, request correlation,
> and completion frontier.

The raw producer inventory, independent graph inventory, and source citations
must demonstrate each named field. DBO invocation duplication does not change
the 48-site semantic inventory.

> Name the active source mechanism that makes every claimed concurrency legal,
> and derive no edge from an overlap percentage or compatibility schedule.

The active mechanism is vLLM DBO's two cooperative model threads, shared
compute and communication streams, DeepEP high-throughput yields, and explicit
CUDA event waits. Static and runtime searches must find no overlap knob.

> Extend the per-request regression to the adapter-emitted schedule, proving
> that traffic rebinding preserves every request-pair byte and that completion
> reduction returns the correct request identities.

The routed-table partition and `StepResult` checks above map directly to this
clause.

> With the producer absent, preserve the legacy sink call, serial graph and
> GOAL bytes, timestamps, and completion order exactly.

The direct serial comparisons above map directly to this clause.

TRAF-13 currently requires:

> connect at least one real framework schedule producer to
> `ObservedStepLowerer`

The vLLM producer and observation-capable sink call demonstrate this.

> Replay a fixed captured step through the traffic binding, `DeviceRuntime`,
> `CompletionEvent`, `StepResult`, TTFT and TPOT; require every captured order
> and dependency fact to survive exactly and show that one observed legal
> overlap changes the live metric in its registered direction.

The two placement rows, graph preservation checks, and dependency
perturbation map directly to this clause.

> Disabling the producer must select the serial lowerer and preserve every
> accepted serial graph, GOAL byte, timestamp and completion order exactly.

The two-layer fixture plus every-step Granite comparisons map directly to this
clause.

Any clause not fully demonstrated stays open under its existing ID or moves
to one of the allocated residual IDs VLLM-23, VLLM-24, TRAF-23, or CORE-38.
A closure removes its registry entry, updates `docs/task-ledger.json`, and
regenerates the progress block. A contradiction sweep reports stale hits in
the two READMEs and architecture document without editing them.

## Registered command and artifact-free dry run

Configure machine-local paths, then run:

```bash
.venv/bin/python examples/vllm_observed_schedule_v1/run_study.py \
  --capture "${SIMLLM_MOE_E2E_ROOT:?configure SIMLLM_MOE_E2E_ROOT}/capture/granite-greedy.jsonl" \
  --replay-run "${SIMLLM_MOE_E2E_ROOT:?configure SIMLLM_MOE_E2E_ROOT}/replay-400g-nvlink/run.json" \
  --routed-experts "${SIMLLM_MOE_E2E_ROOT:?configure SIMLLM_MOE_E2E_ROOT}/replay-400g-nvlink/routed-experts.json" \
  --vllm-source "${SIMLLM_VLLM_SOURCE:?configure SIMLLM_VLLM_SOURCE}" \
  --vllm-python "${SIMLLM_VLLM_PYTHON:?configure SIMLLM_VLLM_PYTHON}" \
  --output-dir "${SIMLLM_VLLM22_RUN_ROOT:?configure SIMLLM_VLLM22_RUN_ROOT}"
```

Before this freeze, the same complete command is run with `--check-only`.
Check-only parses the entire option surface, verifies the three frozen inputs,
the installed vLLM version and audited source hashes, the source-derived
operation counts, all signed bands, physical bounds, perturbation thresholds,
serial digest shapes, and the evidence denominator. It does not import a
SimLLM target module, construct a vLLM engine, execute a schedule or metric,
invoke a native simulator, create the output directory, or write an artifact.
