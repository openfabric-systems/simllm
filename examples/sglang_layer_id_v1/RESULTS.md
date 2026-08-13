# SGL-16 results: SGLang-supplied dispatch layer identity

Frozen by the expectations-only commit `f7865107a44efb782ad930462d4dedcde919ec7d`,
which precedes both the implementation and every run reported here. The
baseline phase ran at that commit; the treatment phase ran after the
implementation landed.

**Status: the run is not void, every frozen fatal guard held. All nine frozen
relation instances passed; 3 of 9 are genuine-risk behavioral evidence.**
SGL-16 remains open under its current `(Precision; P1; M)` tag because this
component evidence does not reach the repository's reported metric chain.

## What changed

The pinned SGLang Granite MoE block builds its router without a layer
identity: `models/granitemoe.py:65-68` constructs `TopK(top_k=...,
renormalize=True)` while `:71-79` hands the same block's explicit `layer_id`
to `FusedMoE`. The `None` travels through `select_experts`
(`layers/moe/topk.py:486`, `:567`) to SGLang's single capture gate
(`:1829-1845`, called at `:1864`) and would land in
`self.buffer[:batch, layer_id, :]` (`state_capturer/base.py:38-40`), where a
`None` is a hard failure rather than a silent one.

The replaced surrogate invented the label by cycling a per-capturer counter
modulo the model's 24 MoE modules. The replacement reads SGLang's own explicit
identity off the constructed model and binds it to the router by module
identity, mirroring the vLLM oracle's `_cpu_layer_ids` map
(`simllm/adapters/vllm/oracle.py:334-352`):

- `RoutedExpertsCapturer.create` receives the model
  (`model_executor/model_runner.py:932-947`); an AROUND hook there walks
  `named_modules()`, takes each router's own `layer_id` when SGLang set one
  and otherwise the unique integer `layer_id` on a sibling of the same parent,
  and refuses any router whose resolved id disagrees with the layer index in
  its own registered module name.
- An AROUND hook on `capture_routed_experts_if_allowed` substitutes the bound
  id, and only when SGLang passes `None`, keyed on the identity of the
  per-router `TopKConfig`.
- The hook on `RoutedExpertsCapturer.capture` infers nothing any more. A
  `None` label there is now an error, which also covers the models that
  bypass the gate and call the capturer directly
  (`models/inkling_common/moe.py:450`).

Provenance moves from `dispatch_layer_mapping="granite-model-order"` to
`"framework-layer-id"`, the value the vLLM runner already writes
(`simllm/preplay/framework_runner.py:1229`).

## Run record

Three frozen cells, two phases, one live SGLang CPU engine per cell at
observed commit `8f2a3ad6d7d68c58ae65b61a75bb2115449addca`, which equals the
commit this evidence was authored against for this run (recorded separately,
with no equality assumed). Model
`ibm-granite/granite-3.0-1b-a400m-instruct` revision
`ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`, float32 on CPU, `tp_size=1`,
`page_size=1`, greedy, `disable_overlap_schedule=True`.

| cell | requests | prompt tokens | new tokens | KV capacity | forwarded tokens observed |
|---|---|---|---|---|---|
| `short` | 1 | 8 | 4 | 256 | 11 |
| `long` | 1 | 96 | 8 | 256 | 103 |
| `preempt` | 8 | 8 | 20 | 96 | 216 |

## Fatal, unscored guards

The run is not void, every frozen fatal guard held. G8 is a post-freeze
consistency guard and is reported as a named addition to the frozen G1 through
G7 set.

| guard | result |
|---|---|
| G1, stock worker, runner, model and CPU parameters, both phases, all cells | held in both phases for every cell |
| G2, unpinned CPU capture storage, both phases, all cells | held in both phases for every cell |
| G3, `granite-model-order` in baseline and `framework-layer-id` in treatment | held in every cell |
| G4, resolved ids are exactly `0`..`23`, one per MoE module, each agreeing with its registered module name | held in every cell |
| G5, real decode retraction and resume in the `preempt` cell, both phases | held in both phases; one retraction and one resume in each |
| G6, zero layer-label disagreements over every audited capture | held in every cell; 1,752 audited captures and zero disagreements |
| G7, frozen, no writes outside the run directory and `--check-only` writes nothing | held on the harness-visible write surface; every explicit output path is rooted under `--run-dir`, and the dry check reported `artifacts_written == 0` |
| G8, post-freeze addition, derived `request` and `observed-dispatch` trace rows identical between phases | held in every cell |

G5 detail: SGLang retracted `p3` under decode pressure at framework step 18,
released its 24 token slots, and `p3` still reached its 20-token length cap
with `framework_preemption_count == 1`. The same retraction occurred in both
phases.

G6 detail: 96 audited captures in `short`, 192 in `long`, 1,464 in `preempt`,
zero disagreements. Under the frozen configuration, G6 and R1 mutually entail
one another: the routed-expert buffer is indexed by the label, and the frozen
reachability proof excludes any other effect of the replacement. G6 is
therefore fatal-unscored rather than independent behavioral evidence.

G7 detail: every explicit write in `run_study.py` targets a path derived from
`--run-dir`. Lines 484-487 partly enforce that boundary by requiring an
explicit absolute run directory and rejecting the repository root. The dry
check reported `artifacts_written == 0`. This is evidence over the harness's
visible write surface, not a filesystem-wide write trace.

## Frozen relation outcomes: 3 of 9 genuine-risk

All nine frozen relation instances passed. After review, only R1's three cells
count as genuine-risk behavioral evidence. R2 is an orthogonality check, and
R3 is a validity control.

| family | cell | result |
|---|---|---|
| R1, genuine-risk raw framework response identity | `short` | passed, 11,700 comparable characters equal |
| R1 | `long` | passed, 105,953 equal |
| R1 | `preempt` | passed, 225,016 equal |
| R2, KV-event orthogonality check | `short` | passed, 5 events |
| R2 | `long` | passed, 9 events |
| R2 | `preempt` | passed, 186 events |
| R3, treatment-trace validity control | `short` | passed, 11 of 11 tokens change under a one-layer rotation |
| R3 | `long` | passed, 103 of 103 |
| R3 | `preempt` | passed, 216 of 216 |

R2 is retained as an orthogonality check. The frozen reachability analysis
shows that `layer_id` cannot reach the allocator in this CPU configuration, so
KV-event identity can detect collateral effects but does not test the label
replacement. R3 is retained as a validity control. It examines only the
treatment trace and confirms that each token's 24 layer tuples are not all
identical; it never compares the two phases. G6 and R1 mutually entail one
another under the frozen configuration and are not independent evidence. The
corrected genuine-risk fraction is 3 of 9.

## Harness correction, reported rather than hidden

The first comparator pass was void on a G5 instrument defect. It read a `kind`
key from trace rows that spell it `event_kind`, so the fatal guard evaluated as
failed even though the underlying retraction and resume were present in both
phases. A void run has no interpretable behavioral fraction, so none is
reported for that pass.

The same comparator initially compared `responses.json` as whole files and
flagged all three R1 cells. Its only observed differences were
`meta_info.e2e_latency` and `meta_info.response_sent_to_client_ts`, which are
wall-clock properties of the capture process and which no two runs of any code
can make equal. The frozen expectation names the response content it compares
rather than the file bytes, so the corrected comparator drops those two fields
and nothing else and reads `event_kind`. The pre-correction `summary.json` was
overwritten rather than retained, which is a limitation of the evidence trail.
After correction, the run was not void and all nine frozen relation instances
passed; 3 of 9 count as genuine-risk behavioral evidence.

No expectation, cell, relation or threshold was changed. The correction is to
the reader, not to the claim.

## Physical sanity

This study introduces no timing, so there is no modeled number to bound. It
reports no latency, no bandwidth, no byte rate and no derived cost. The two
wall-clock fields it excludes from comparison are measurements of the capture
harness, not of anything modeled. The one quantity worth a sanity note is the
capture buffer shape, `[257, 24, 8]` for the 256-token cells: 24 MoE modules
and top-8 routing over 32 experts, which is exactly the pinned Granite
geometry and not a shape the oracle chose.

## Post-specified structural demonstration

Not part of the frozen scoring, run after the results were read, and reported
as such: each treatment trace projects through
`simllm.preplay.project_framework_routing` into a validated `RoutedExperts`,
the same authority `simllm.traffic.RoutedMoeSupply` consumes, with 24 layers
per token, top-8 tuples and 32 experts in all three cells. That establishes
the SGLang v2 trace is shaped for the traffic path. It does not establish that
any SGLang run has driven it: no placement manifest, GOAL emission, backend
run or metric was produced from an SGLang trace in this study or any other.

## Landed scope and remaining closure condition

Each registered SGL-16 clause, quoted, with the component evidence that landed:

- "replace the framework-oracle fallback's Granite model-order layer inference
  with stable layer IDs supplied by SGLang": done. The surrogate is gone from
  the label path; `simllm/adapters/sglang/oracle.py` no longer contains a
  capture counter that produces labels.
- "The current surrogate cycles missing capture labels through the model's 24
  MoE modules in execution order": confirmed as the thing replaced, and its
  24-module assumption is now only an audit comparator.
- "the identifying observable is an explicit framework layer ID at the
  post-selection capturer": the capturer now receives an explicit framework
  layer id on every call and refuses a call without one. The id is read from
  SGLang's own `layer_id`, on the router itself when a model forwards it and
  otherwise on the sibling `FusedMoE` that Granite does give one, then
  cross-checked against the router's registered module name.
- "Freeze at least two prompt shapes and a preemption resume before changing
  it": commit `f786510` freezes the 8-token and 96-token prompt shapes and the
  8-request retraction cell, and precedes both the implementation and both
  phases.
- "Acceptance requires zero layer-label disagreements": 1,752 audited captures
  across three cells, zero disagreements.
- "and byte-identical expert IDs, request outputs and KV events relative to
  the current qualified Granite fallback": R1 covers the base64 routed-expert
  payload, output token ids, finish reasons and the cached-token and
  preemption counters; R2 covers the KV events; the post-freeze G8 consistency
  guard covers the derived per-token dispatch rows.

The source change and component study landed against every clause quoted above.
SGL-16 nevertheless remains open because no supported SGLang path connects
this routing identity through a placement manifest, GOAL emission, backend
run, `CompletionEvent`, `StepResult`, and TTFT or TPOT. Reopening the existing
ID preserves that missing precision condition without inventing a new tag or
registering a new ID.

Two further component-coverage limitations remain under the reopened SGL-16:
only one model exercises the sibling-`layer_id` resolution path, and a model
whose MoE blocks are a strict subset of its decoder layers would break the
audit comparator rather than the binding, since the comparator is the
surrogate this task removed.

## Contradiction sweep

- `README.md` and `docs/architecture.md` contain no statement about the
  SGLang dispatch layer label, so neither contradicts the landed component
  evidence or the reopened status.
- `docs/README_PRO.md` carries the generated task-progress block, which is
  regenerated here with SGL-16 open, and no prose claim about the layer
  surrogate.
- `examples/framework_oracle_v1/RESULTS.md:253` says "SGL-16 owns replacement
  of the fallback's Granite model-order layer-label surrogate". That is a
  chronological record of the earlier study and is left as written rather than
  edited, per the sweep contract.

## Reproduction

```bash
STUDY="${SIMLLM_WAVE11_RUN_ROOT:?configure SIMLLM_WAVE11_RUN_ROOT}/sgl16"

python examples/sglang_layer_id_v1/run_study.py \
  --run-dir "${STUDY}" \
  --sglang-source "${SIMLLM_SGLANG_SOURCE:?configure SIMLLM_SGLANG_SOURCE}" \
  --sglang-python "${SIMLLM_SGLANG_PYTHON:?configure SIMLLM_SGLANG_PYTHON}" \
  --model-path "${SIMLLM_GRANITE_MOE_PATH:?configure SIMLLM_GRANITE_MOE_PATH}" \
  --check-only
```

The dry check emits `"artifacts_written":0` and writes nothing. Replace
`--check-only` with `--phase baseline` at the freeze commit, `--phase
treatment --layer-audit` at this commit, and `--phase compare` to score.
