# Pre-play validation v1 expectations

This document freezes the PLAY-5 validation study before its comparison
implementation, result-producing harness, or first inference or replay run.
The two halves have separate evidence ledgers. An unavailable independent CPU
runtime cannot be replaced with a GPU run, a Transformers-backed imitation, or
the device-free replay worker.

## Audited source state

The repository source is commit `b74629b4b4da1addda9ff21226cfabf5c09aad87`.
The external runtime is vLLM 0.26.0 with Transformers 5.14.1 and Torch
2.11.0+cu130. The following sources were read before this freeze:

- `simllm/preplay/runner.py:321-394` hooks all Transformers Granite routers and
  reconstructs top-k expert IDs and normalized weights from full float32 router
  logits. `simllm/preplay/runner.py:396-416` uses argmax for greedy decoding and
  Torch multinomial after temperature and top-p filtering for seeded sampling.
  `simllm/preplay/runner.py:418-492` defines prompt and nonterminal decode
  attribution and the terminal stop decision.
- `transformers/models/granitemoe/modeling_granitemoe.py:137-156` computes the
  Granite router with one float32 linear projection, top-k, and softmax. Its
  SHA-256 is
  `2490f7cfbd3f362f931bf9b95d488ad7be6d1365b3cc55dd11947b0eed69a809`.
- `vllm/model_executor/models/granitemoe.py:70-136` is an independent Granite
  implementation. Its replicated gate supplies router logits to the fused MoE
  implementation. Its SHA-256 is
  `b60e452c3f28b25aa104c88869daa25c06a7fb6ed45bd34e908fa6a8395efda1`.
- `vllm/v1/worker/cpu_worker.py:33-104,120-183` constructs the CPU worker,
  selects `torch.device("cpu")`, initializes its CPU runtime, seeds the model,
  and constructs `CPUModelRunner`. Its SHA-256 is
  `ccf18240a7605ebda0dbf27bdbda83a39e83e7baece95bff68e0f3e5beb6103e`.
  `vllm/v1/worker/cpu_model_runner.py:22-37,121-149` keeps the model on CPU and
  loads it through vLLM's own model loader. Its SHA-256 is
  `dd3a7686b567c52363454ed6b353bf8a968fa60c6ad3dbcf30c8044f5602d7ed`.
- `vllm/v1/sample/sampler.py:20-58,228-302` converts logits to float32, uses
  argmax for greedy requests, then applies temperature and top-p before random
  sampling. Its SHA-256 is
  `315af950ef4c35fced53dc3a5df49a80af20b47e417e8f12bf315f535769bab2`.
  `vllm/v1/sample/ops/topk_topp_sampler.py:176-204,345-405,431-444` shows that
  the CPU path uses exponential noise and argmax rather than Torch multinomial.
  Its SHA-256 is
  `ad9406a08a9bfcc84f182dab4522920f73605d4999191ee8f0dbb1479d946506`.
- `vllm/v1/core/sched/scheduler.py:1808-1853` emits the scheduler's own finish
  and stop reasons. `vllm/v1/engine/output_processor.py:630-684` performs the
  detokenized stop-string check and publishes request outputs. Their hashes are
  `2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941`
  and `ee10351275d90796c8b901a5f4b23d5a046ef6ee72fd2921aff2ae78ca58bd9b`.
- `simllm/adapters/vllm/replay.py:156-226` requires the scheduler admission
  limit to equal the oracle output length and rejects premature stop tokens.
  `simllm/adapters/vllm/replay.py:274-402` serves the token at the exact
  scheduler-reported output index and validates delayed completions.
- `simllm/traffic/step_comm.py:109-166` maps each scheduler-visible prefill or
  decode slice to captured forwarded tokens. Lines 169-269 deduplicate expert
  destinations, form exact sparse dispatch tables, and transpose them for
  combine. `simllm/backends/step_sink.py:213-288` carries those tables through
  GOAL and the fluid RNIC backend into the live `StepResult` latency.

The installed vLLM environment presents a known pre-run risk. Its package
version is `0.26.0`, not a CPU-tagged build. The CPU platform plugin returns no
candidate and `torch.ops._C.init_cpu_memory_env` is absent. This observation is
an environment audit, not an inference attempt. The registered run still makes
one honest attempt to select the audited CPU platform and stock CPU worker. If
that path cannot initialize, the oracle-consistency half is blocked and must be
reported with its first failing source boundary. It earns no scored pass.

## Frozen model, requests, and parameters

- Model: `ibm-granite/granite-3.0-1b-a400m-instruct`
- Revision: `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`
- Device and dtype for both oracle runners: CPU and float32
- Torch threads: eight
- Offline controls: `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
- Greedy configuration: temperature zero and no random seed
- Seeded configuration: seed 173, temperature 0.8, top-p 0.9
- Router shape: 24 ordered layers, top-k 8, 32 experts

Both runners receive the exact PLAY-1 prompt token IDs from the same cached
tokenizer and the following request policies:

| Request | Prompt | Maximum output | Stop string |
|---|---|---:|---|
| `eos-brief` | `Reply with exactly one word: OK` | 16 | none |
| `length-cap` | `Continue this sequence with ten more integers: 1 2 3` | 1 | none |
| `stop-string` | `Reply with exactly SIMLLM_STOP and no other text` | 16 | `SIMLLM_STOP` |

The already accepted PLAY-1 greedy oracle has prompt lengths `(15, 22, 20)`,
output lengths `(3, 1, 5)`, and normalized stop reasons
`(eos, length-cap, stop-string)` in that request order. Those values are input
oracles for replay. They are not observations from this study.

Sampling mode and request policy are the two oracle parameter families. Replay
varies fluid link bandwidth over exactly 200 and 400 Gbit/s while retaining the
same request set, routing, placement, compute cost, and scheduler configuration.

## Independent framework qualification

The independent observation qualifies only if all of these fatal gates pass:

- vLLM reports its CPU platform before worker construction;
- the reached worker is the stock `CPUWorker` or a validation-only subclass
  whose execution methods delegate to that stock worker;
- the reached model runner is `CPUModelRunner`, its model class is vLLM's
  `GraniteMoeForCausalLM`, every parameter is on CPU, and no CUDA allocation
  increases during the run;
- exactly 24 vLLM Granite gates are observed in layer order, with one routing
  row per scheduled input token;
- the vLLM sampler, scheduler stop state, and output processor remain the
  authorities for token choice and termination;
- no Transformers model class supplies the independent forward pass.

Failure of a qualification gate is an environmental or integration blocker,
not an oracle divergence. The report must retain the attempted configuration,
exception type, message, reached boundary, and absence or presence of any
partial artifact.

## Divergence taxonomy

Every comparison begins at identical prompt token IDs and walks output and
routing decisions in causal order. The only admissible root classifications
are `sampler-difference` and `numerics-near-tie-flip`. Exact agreement is
recorded as `exact`, not as a divergence.

`sampler-difference` applies only to the first output-token disagreement in a
seeded row whose two runners consumed the same preceding token context. The
classification is justified by the audited algorithms: Transformers uses
multinomial, while vLLM CPU uses exponential noise followed by argmax. Later
token, length, stop, or decode-routing differences may retain that same root
classification with `cascade=true`. It cannot classify a prefill-routing
difference or any greedy-token difference.

`numerics-near-tie-flip` applies only when both runners consumed the same input
context and the changed decision straddles a frozen absolute logit margin of
`1e-5` or less in at least one runner. For token selection, the two selected
tokens must occupy the top-two boundary being measured. For routing, every
expert in the symmetric difference must straddle the top-8 versus top-9
boundary. Later causal differences may retain the same root with
`cascade=true`.

Any divergence that satisfies neither definition is `unclassified` and fails
the study. A missing logit margin, a different prompt tokenization, a routing
change before the first sampled-token difference with a margin above the
bound, or a length or stop difference without an earlier classified root also
fails. The implementation must never convert an unclassified row into an
accepted warning.

## PLAY-B1: oracle consistency

The six request by sampling-mode rows are scored live observations. Each row
must report exact agreement or a complete list of divergences rooted in the
taxonomy above. The following fields are compared:

- prompt token IDs;
- output length and normalized stop reason;
- output token IDs, to establish causal ancestry;
- every prefill and nonterminal decode `(phase, token index, input token ID,
  layer, expert IDs)` routing decision;
- the token top-two and routing top-8 versus top-9 margins needed by any
  non-exact classification.

A row passes only when its unclassified count is zero. The report separately
states exact agreement fractions for lengths, stop reasons, output tokens, and
routing decisions. Classification coverage does not relabel a divergence as
agreement.

If the CPU qualification gate blocks the framework run, PLAY-B1 has zero
executed scored rows and remains incomplete. The blocked row count must not be
reported as a pass denominator.

## Replay construction

The accepted greedy Transformers trace is joined at arrival time zero and
projected through `simllm-routed-experts-v1`. The three requests are submitted
together to the real in-process vLLM 0.26.0 scheduler with request-ID
randomization disabled. Replay uses the device-free `SimWorker`, exact oracle
token IDs, each request's exact maximum output length and stop strings, and one
shared `HtsimStepSink` per bandwidth. The scheduler remains the only completion
authority.

The fixed traffic geometry is two EP ranks, hidden size 1,024, dtype width two
bytes, 24 MoE layers, 32 experts, and top-k 8. At epoch zero, rank 0 owns
experts 0 through 15 and rank 1 owns experts 16 through 31 at every layer. The
compute provider contributes exactly 24,000 ps per step, split into 1,000 ps
per layer. The backend profile is `rnic-nn-fluid`.

The scheduler's exact step sequence is not a scored oracle. Whatever legal
sequence it realizes must consume each captured prefill and decode input token
exactly once and must never consume a terminal generated token.

## PLAY-B2: scheduler-visible completion

For each of the two bandwidths and three requests, the scheduler must publish
exactly the greedy oracle token sequence and finish after exactly the oracle
output length. Its normalized stop reason must be `eos`, `length-cap`, or
`stop-string` as listed above. This gives six scored live rows. A completion
before or after the oracle length, a changed token, duplicate completion, or an
unreported request is fatal.

Bandwidth must not change token IDs, lengths, stop reasons, scheduler step
membership, or completion order. This identity relation is fatal and unscored
because bandwidth cannot affect the scheduler's deterministic choices in this
configuration.

## PLAY-B3: captured all-to-all sizes

For vector width `V = 1024 * 2 = 2048` bytes, scheduled captured input-token
set `T`, source rank `s`, destination rank `d`, and layer `l`, the independent
closed form is:

```text
dispatch_bytes(s,d,l) = V * sum over t in T of
    indicator(s != d and any owner(l,e) == d for e in experts(t,l))
combine_bytes(s,d,l) = dispatch_bytes(d,s,l)
```

For each bandwidth, every sparse pair table in every emitted GOAL must equal
this closed form exactly. The comparison is one scored live stream relation per
bandwidth, two instances total. It is computed from the trace rows and fixed
ownership table without calling the implementation's traffic expansion.
Missing, extra, duplicated, uniformly fabricated, or wrong-sized pairs fail.

## PLAY-B4: TTFT and TPOT bandwidth relation

Let `D(k,l)` and `C(k,l)` be the maximum dispatch and combine pair bytes in
live scheduler step `k`, layer `l`. In the two-rank fluid topology, each phase
has one flow per direction, both directions start together, and the larger
pair controls phase completion. Propagation and the fixed compute term do not
change with bandwidth. Therefore the exact step relation is:

```text
JCT_400(k) - JCT_200(k)
    = -20 ps/byte * sum over layers l of (D(k,l) + C(k,l))
```

For each request, TTFT is the completion time of its first returned token minus
arrival. The 400 Gbit/s minus 200 Gbit/s TTFT difference must equal the sum of
the step relation through that first token, exactly to 0 ps. This is three
scored instances and the signed direction is strictly negative.

For each request with more than one oracle output token, TPOT is the mean of
its consecutive token-completion intervals. Its 400 Gbit/s minus 200 Gbit/s
difference must equal the sum of the corresponding post-first-token step
relations divided by `output_length - 1`, exactly as a rational value. This is
two scored instances and the signed direction is strictly negative. The
one-token request has no TPOT and is excluded rather than counted as zero.

These metric relations prove that the captured routing is live-reachable
through framework scheduling, `StepRecord`, the routed traffic supply,
`StepResult`, TTFT, and TPOT. Pair-size equality alone is component evidence.

## Evidence accounting and fatal guards

When both halves execute, the behavioral headline is 19 scored instances:
six oracle-consistency rows, six replay-completion rows, two routed-stream
relations, three TTFT relations, and two TPOT relations. Relation families,
exact field-agreement fractions, run configurations, fatal guards, and test
executables remain separate evidence classes.

Every scored assertion is a live cross-implementation or live-runtime fact
that can genuinely fail. The expected genuine-risk fraction is therefore
`19/19 = 100%`. If the independent CPU half is blocked, the executed replay
headline is 13 scored instances with genuine-risk fraction
`13/13 = 100%`; the six blocked oracle rows are reported outside the executed
denominator and PLAY-5 remains incomplete.

Fatal unscored guards include source hashes, model revision, offline mode,
request and trace identity, trace hash, prompt token equality, CPU
qualification, placement completeness, captured-token conservation, no
terminal forward, GOAL tag and pair uniqueness, backend quiescence, exact
record/result cardinality, and bandwidth identity of scheduler choices.
Author-defined requests, ownership, bandwidths, and fixed compute cost are run
configuration and never add to the behavioral denominator.

## Registered command and pre-freeze dry run

Source local configuration first. The single registered invocation is:

```text
"${SIMLLM_VLLM_PYTHON:?configure SIMLLM_VLLM_PYTHON}" \
  examples/preplay_validation_v1/run_study.py \
  --cache-dir "${HF_HOME:?configure HF_HOME}" \
  --vllm-package-root "${SIMLLM_VLLM_PACKAGE_ROOT:?configure SIMLLM_VLLM_PACKAGE_ROOT}" \
  --htsim-rnic "${SIMLLM_HTSIM_RNIC:?configure SIMLLM_HTSIM_RNIC}" \
  --run-dir "${SIMLLM_PLAY5_RUN_ROOT:?configure SIMLLM_PLAY5_RUN_ROOT}"
```

Before this freeze, the same command with `--check-only` was run against an
untracked parser and literal-audit harness. It checked the complete option
surface, model snapshot, external source hashes, runtime version, executable
inputs, taxonomy constants, evidence counts, and bandwidth relation algebra.
It printed a confirmation line by design. It imported no SimLLM target
implementation, constructed no model, executed no scheduler or backend, and
produced no artifacts. The untracked harness encoded only literals frozen in
this document.

Runtime artifacts go only below the configured external run directory. No
generated trace, GOAL, CSV, log, or summary is tracked.

## Deliberate omissions

This study does not add a selectable production vLLM pre-play runner. That
optional backend remains PLAY-6. It does not claim GPU parity, SGLang replay,
gate-weight traffic scaling, placement changes, or a new routing schema. The
independent worker and diagnostic margins exist only to validate PLAY-5 and do
not widen the public pre-play artifact.
