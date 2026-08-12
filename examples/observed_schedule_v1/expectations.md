# vLLM observed schedule v1 expectations

This document freezes the TRAF-13 qualification and metric expectations before
any adapter implementation, result-producing study run, or Granite replay for
this change. The study may score overlap only after a vLLM producer proves that
its operations and dependencies came from a supported framework execution
path. A graph inferred by deleting edges from the serial compatibility lowerer
does not qualify.

The repository source at the freeze boundary is commit
`dcbef8684f35d5b1cb0138412094186000ac18fa`. The vLLM evidence is authored
against the official v0.26.0 source at commit
`568afb3a13806beb53bb2e6bd518269357b237c0`. A result record reports the
source identity it actually observed without requiring a live source tree or
installed package to equal that authored-against identity.

## Source audit and qualification risk

The following source was read before this freeze:

- vLLM `model_executor/models/granitemoe.py:264-282,315-340` executes attention
  and MoE within one decoder layer, returns the layer output, and calls decoder
  layers in a synchronous Python loop. This proves program order but does not
  prove that a collective may overlap the next layer's dependent compute.
- vLLM `v1/worker/gpu_model_runner.py:4111-4479` prepares one scheduler step and
  calls `_model_forward`. The stock runner can select microbatching, but the
  selected Granite replay did not enable dual-batch overlap.
- vLLM `v1/worker/ubatching.py:77-147` defines distinct compute and
  communication streams plus explicit event waits for dual-batch overlap.
  That mechanism overlaps different microbatches. It is not evidence that the
  single-batch Granite layer loop may drop its data dependency.
- vLLM
  `model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py:113-181,357-411`
  switches streams and yields between dual-batch microbatches. The path is
  conditional on the DeepEP high-throughput and dual-batch configuration. It
  is not active in the frozen replay.
- `simllm/adapters/vllm/worker.py:397-405,429-469` currently mirrors one
  whole-model `_model_forward` call, emits one fixed 4,096-byte TP coordinator
  event, and settles the step. It does not execute or observe individual
  Granite layers or MoE dispatch and combine calls.
- `simllm/adapters/vllm/communicator.py:110-169,366-404` records coordinator
  sequence, time, group, payload, work, and COMP-stack events. It records no
  layer, semantic collective site, logical stream, dependency edge,
  microbatch, request membership, or completion frontier.
- `simllm/traffic/step_comm.py:185-279` requires every traffic-planned TP and
  MoE site exactly once and preserves adapter-authored order and dependencies.
  `simllm/backends/step_lowerer.py:266-294` selects this binding only when
  `ExecutionObservations` are present and delegates absent observations
  directly to the serial lowerer.

The source audit therefore predicts a qualification failure at the current
boundary. That prediction is genuinely at risk only to the live observation:
the worker or its installed vLLM source could expose a schedule not visible in
the audited files. A qualification failure is a blocker, not a behavioral
pass. It must not be repaired by transcribing the synthetic pipeline from
`examples/compute_comm_overlap_v1`.

## Frozen replay and placements

The input is the existing `simllm-preplay-trace-v1` Granite capture:

- model `ibm-granite/granite-3.0-1b-a400m-instruct`;
- model revision `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`;
- capture SHA-256
  `5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6`;
- 24 ordered MoE layers, 32 experts, top-k 8, hidden width 1,024, and
  two-byte traffic activations;
- the existing three-request greedy replay, with request `r0` used for the
  TTFT and TPOT decision row because the historical all-at-once driver assigned
  later arrival times to already-admitted requests `r1` and `r2`.

The two placement cells use the same captured routing and compute provider:

1. `single-node`: EP ranks 0 through 7 map to one eight-GPU node and remote
   pairs use the accepted 450 GB/s NVLink-class rate.
2. `cross-node`: the same eight semantic EP ranks map one per node and remote
   pairs use 400 Gbit/s RNIC service.

The placement mapping is a run configuration. The observed and serial rows
within a cell must use identical model, requests, routing, compute provider,
link constants, rank ownership, and sampled-request identities.

## Producer qualification

A step qualifies only if all of these fatal gates pass from raw adapter
observations:

- the adapter returns one `ExecutionObservations` object for that exact
  translated `StepRecord`, before the timing sink runs;
- operation tuple order is the adapter's observed submission order;
- every compute and collective operation carries the observed logical stream,
  program-order edges, event-wait edges, layer and request correlation, and a
  completion boundary;
- all 24 Granite layers are represented, and the traffic binder can match all
  24 dispatch plus 24 combine sites exactly, with no fabricated, missing,
  duplicate, or unknown site;
- at least one pair consisting of a layer collective and later-layer compute
  is on distinct logical streams without a dependency path, and the adapter
  names the source-level launch and synchronization facts that make that
  concurrency legal in the active configuration;
- the active live configuration reports that source-backed mechanism enabled;
- neither the producer, lowerer, sink, result, nor CLI contains an overlap
  percentage, overlap duration, or discount factor.

The existing single fixed TP event cannot pass these gates. A source-derived
serial schedule can preserve useful order evidence, but it cannot qualify the
decision row because the task specifically requires one observed legal
overlap.

## Frozen decision relations

If and only if producer qualification passes, run the live chain

```text
vLLM step translation
  -> ExecutionObservations
  -> ObservedStepLowerer and traffic binding
  -> DeviceRuntime
  -> CompletionEvent and RuntimeReport
  -> CompletionReducer
  -> StepResult, TTFT, and TPOT
```

for the two placement cells. Collect all raw observed, serial, and perturbed
rows before evaluating exact identities or other fatal gates.

### A. Observed versus serial TPOT

For request `r0`, observed-schedule TPOT must be strictly lower than serial
TPOT in both cells. Freeze these additive reduction bands:

| Placement | `serial TPOT - observed TPOT` |
|---|---:|
| single-node | 1,000,000 through 5,000,000 ps |
| cross-node | 20,000,000 through 130,000,000 ps |

These bands are intentionally wider than the exploratory context. The
single-node band surrounds the approximately 3.4 microsecond communication
term. The cross-node band permits partial rather than ideal overlap while
excluding a silently ignored observation graph. Each placement is one scored
instance.

TTFT is reported for the same rows. Its observed value must not exceed its
serial value, but TTFT is not scored separately because the first replay step
has a different token shape and the task decision is decode TPOT.

### B. Dependency perturbation

For each placement, copy the qualified raw observation tuple and add exactly
one dependency edge from the selected layer collective to the next eligible
layer-compute operation. Do not alter tuple order, work, queues, traffic,
completion IDs, or any other edge. The perturbed TPOT must be strictly greater
than the unperturbed observed TPOT and no greater than serial TPOT. The increase
must be at least 100,000 ps for `single-node` and 5,000,000 ps for
`cross-node`. Each placement is one scored instance.

This perturbation is deliberately evaluated through the normal lowerer and
runtime. It can fail if observations are ignored, if queue identity globally
serializes work, or if the chosen edge is not metric-live.

### C. Serial off-path identity

With the producer absent, the observation-aware sink must pass `None` to
`ObservedStepLowerer`, which must delegate to `SerialStepLowerer`. The accepted
two-layer compatibility fixture remains:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| canonical execution graph JSON plus LF | 4,127 | `aa3c836fe559973a7bf0940384c2e8a84e6af84e0fbd2c02d3b89774ee0c8e2d` |
| serial graph-only GOAL | 1,880 | `7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6` |

For every Granite step, the off path also preserves the accepted serial graph,
GOAL bytes, timestamps, and completion order. These identity checks are fatal
and unscored.

## Per-request attribution

For every observed MoE operation, the traffic planner's rebound
`request_pair_payload_bytes` must conserve the aggregate pair table exactly.
Every scheduled request must reach a unique required completion endpoint and
the resulting `RequestMetric` must carry the correct request ID. This is a
fatal conservation and identity guard. It is not scored because a result that
passes it by construction after strict traffic binding has not supplied a
second independent behavioral risk.

## Evidence accounting and entailment

If both placements qualify, the behavioral headline has four genuine-risk
instances: two raw observed-versus-serial TPOT reductions and two raw
dependency-perturbation increases. The expected genuine-risk fraction is
`4/4 = 100%`.

The runner must evaluate those four relations directly from raw `StepResult`
metrics before serial byte identities, graph-field preservation, attribution
conservation, source hashes, or fixed configuration checks. No earlier fatal
oracle may entail a scored sign or band. Exact metric rows, if retained for
regression, form a separate evidence class and are checked later.

If producer qualification fails, zero scored instances execute. The report
uses `0/0, blocked before behavioral execution`, not `0/4`, and preserves the
exact missing observation fields or unsupported source mechanism. Source
hashes, capture identity, model shape, configuration echoes, producer
qualification, serial digests, operation conservation, request attribution,
event ordering, and check-only cleanliness are fatal unscored guards. Unit
tests and live integration attempts remain separate evidence classes.

## Closure scope

TRAF-13 can close only if every clause in its module registry is demonstrated
by qualifying raw evidence. If the current source boundary cannot observe the
required schedule, the result must leave TRAF-13 open and register the missing
vLLM and SGLang producer work under allocated IDs. A graph-native coarse
runtime result does not establish packet-level htsim overlap or calibrated
GPU-resident collective contention; any such residual uses an allocated
traffic or core ID rather than broadening this study's claim.

## Registered command and dry run

Source machine-local configuration first. The registered command is:

```bash
.venv/bin/python examples/observed_schedule_v1/run_study.py \
  --capture "${SIMLLM_MOE_E2E_ROOT:?configure SIMLLM_MOE_E2E_ROOT}/capture/granite-greedy.jsonl" \
  --vllm-source "${SIMLLM_VLLM_SOURCE:?configure SIMLLM_VLLM_SOURCE}" \
  --output-dir "${SIMLLM_OBSERVED_SCHEDULE_RUN_ROOT:?configure SIMLLM_OBSERVED_SCHEDULE_RUN_ROOT}"
```

Before this freeze, the same command is run with `--check-only`. Check-only
parses the complete option surface, verifies the frozen capture and audited
source-file hashes, validates the operation counts, sign bands, perturbation
bounds, serial digest shapes, and evidence denominator, and creates no output
directory or artifact. It imports no SimLLM target module, constructs no vLLM
engine, executes no schedule or metric relation, and invokes no native tool.

