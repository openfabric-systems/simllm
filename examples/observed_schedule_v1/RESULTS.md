# vLLM observed schedule v1 results

TRAF-13 remains open. Producer qualification failed as predicted by the
expectations-only source audit. The skeleton adapter called the
observation-capable sink once with no `ExecutionObservations`, exposed one
fixed 4,096-byte TP `all_reduce`, and supplied zero of the required 48
semantic Granite MoE sites. Its event schema lacked the required layer,
logical-stream, dependency, request, and completion-frontier fields.

The behavioral result is `0/0, blocked before behavioral execution`. This is
not `0/4`: none of the four registered genuine-risk instances executed. No
single-node or cross-node Granite placement ran, no Granite TTFT or TPOT was
measured, and no dependency perturbation was applied. The observation-aware
sink component and the serial compatibility identities passed separately, but
they do not qualify a framework schedule or close TRAF-13.

## Chronology, provenance, and reproduction

The expectations were frozen in commit
`409b4ade250fcc22ccb36cb4927399694e0cd318` before implementation or a
result-producing run. Before that commit, the complete registered command
passed with `--check-only`. It validated only the frozen capture, audited
source hashes, operation counts, signed bands, perturbation bounds, serial
digest shapes, and evidence denominator. It created no output directory or
artifact, imported no SimLLM target module, constructed no vLLM engine, and
invoked no native tool.

The observation-aware sink component landed in commit
`ea8df515da0efcbb272fca832f03b2a4db95a145`. The qualification record observed
that same repository commit. The external result is at
`$SIMLLM_OBSERVED_SCHEDULE_RUN_ROOT/qualification-2026-08-12/results.json` and
has SHA-256
`d555647c9398ff244ac598463e07fb4676cf27c90ddbfa8a1369d8d050be8f50`.

The evidence was authored against official vLLM v0.26.0 commit
`568afb3a13806beb53bb2e6bd518269357b237c0`. The run did not resolve a live
vLLM source commit, so `observed_commit` is `null`. It independently observed
the four frozen source-file hashes:

| Source file | Observed SHA-256 |
|---|---|
| `vllm/model_executor/models/granitemoe.py` | `b60e452c3f28b25aa104c88869daa25c06a7fb6ed45bd34e908fa6a8395efda1` |
| `vllm/v1/worker/gpu_model_runner.py` | `81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0` |
| `vllm/v1/worker/ubatching.py` | `40391241c564feb5f16c77898ae6ae152ed6e71a4682e2a406387785d8de02d7` |
| `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py` | `465cdf1d6cee91b2ee8c2e43abbea6e8408976e3048c10f44c089f34b415bc60` |

The authored-against commit and the observed file identities are independent
provenance fields. The study makes no equality assumption between a live
source checkout and the authored-against commit. The frozen 120-row capture
was observed at SHA-256
`5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6`.

Configure the three machine-local roots and reproduce from the repository
root. The output directory must not already exist:

```bash
.venv/bin/python examples/observed_schedule_v1/run_study.py \
  --capture "${SIMLLM_MOE_E2E_ROOT:?configure SIMLLM_MOE_E2E_ROOT}/capture/granite-greedy.jsonl" \
  --vllm-source "${SIMLLM_VLLM_SOURCE:?configure SIMLLM_VLLM_SOURCE}" \
  --output-dir "${SIMLLM_OBSERVED_SCHEDULE_RUN_ROOT:?configure SIMLLM_OBSERVED_SCHEDULE_RUN_ROOT}/qualification-reproduction"
```

Add `--check-only` to repeat the artifact-free registry validation.

## Producer qualification

The raw adapter probe failed the frozen producer gates:

| Required observation | Raw result |
|---|---|
| One observation object for the translated step | Sink called once, observations absent |
| Adapter submission order and logical streams | No logical-stream field |
| Program-order and event-wait dependencies | No dependency field |
| Layer and request correlation | No layer or request field |
| Completion boundary | No completion-operation field |
| All 24 Granite layers and 48 semantic MoE sites | Four-layer component fixture, zero semantic MoE sites |
| Source-backed legal next-layer concurrency | No qualifying active mechanism |

The only coordinator event was an `all_reduce` with a 4,096-byte payload. It
was not a per-layer EP dispatch or combine event. The audited Granite model
loop proves synchronous program order, while the audited dual-batch paths were
inactive for the frozen replay. Neither fact establishes legal overlap between
one layer's collective and the next layer's dependent compute.

## Evidence classes

The evidence classes remain separate and no count below is added to another.

| Evidence class | Registered or observed count | Outcome |
|---|---:|---|
| Granite placement configurations | 2 registered, 0 executed | Blocked by producer qualification |
| Scored behavioral families | 2 registered, 0 executed | `0/0, blocked before behavioral execution` |
| Scored genuine-risk instances | 4 registered, 0 executed | `0/0, blocked before behavioral execution` |
| Fatal producer qualification | 1 checklist | Failed |
| Fatal unscored supporting guard categories | 4 | 4 passed |
| Exact serial compatibility artifacts | 2 | 2 matched within the serial-identity guard |
| Import-free skeleton component attempt | 1 | Reached the sink, did not produce a schedule |
| Post-run live vLLM diagnostic | 1 | Component diagnostic only, did not produce a schedule |

The four passed fatal-unscored categories were source and capture identity,
serial identity, component observation routing, and component request-metric
identity. Their success cannot compensate for the failed producer gate.

## No behavioral placement rows

The frozen behavioral relations were not evaluated because no producer
qualified. The empty raw result arrays are reported directly:

| Placement | Observed TPOT | Serial TPOT | Reduction | Perturbation increase | TTFT |
|---|---:|---:|---:|---:|---:|
| `single-node` | Not run | Not run | Not run | Not run | Not run |
| `cross-node` | Not run | Not run | Not run | Not run | Not run |

Consequently there is no evidence for either registered TPOT band, the TTFT
non-increase guard, or the dependency-serialization direction. There are also
no Granite graph, GOAL, timestamp, or completion-order identity rows from this
run.

## Component and serial evidence

The supporting component fixture supplied a synthetic serial observation
tuple directly to `DeviceRuntimeStepSink`. It proved that an already-available
`ExecutionObservations` object can pass through the sink, observed lowerer,
coarse runtime, completion events, reducer, and request-attributed
`StepResult`. The component produced 200 completion events, request metrics for
`d` and `p`, and a 934,760 ps step latency and clock value. Because the tuple
was constructed from the serial lowerer rather than emitted by vLLM, this is
component routing evidence only.

The absent-observation lowerer remained the exact direct serial delegate for
the accepted two-layer compatibility fixture:

| Artifact | Observed bytes | Observed SHA-256 | Outcome |
|---|---:|---|---|
| Canonical execution graph JSON plus LF | 4,127 | `aa3c836fe559973a7bf0940384c2e8a84e6af84e0fbd2c02d3b89774ee0c8e2d` | Passed |
| Serial graph-only GOAL | 1,880 | `7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6` | Passed |

These exact identities, direct-delegate equality, operation conservation, and
request IDs are fatal-unscored guards. They are not behavioral overlap
evidence. The full Granite serial timestamp and completion-order clause remains
unexecuted.

## Entailment and genuine-risk accounting

Producer qualification ran before the behavioral relations, as frozen. Its
failure prevented the runner from constructing any placement result. It did
not establish a failed or passed TPOT sign, band, or perturbation relation;
those raw observations do not exist. The genuine-risk denominator is therefore
zero executed instances, reported as `0/0, blocked before behavioral
execution`, rather than treating four unexecuted registrations as four
failures.

Had qualification passed, the runner was required to collect all observed,
serial, and perturbed `StepResult` rows and evaluate the two TPOT reductions
and two perturbation increases before checking serial digests, fixed source or
capture identities, attribution conservation, or configuration echoes. No
earlier exact oracle could then entail a scored result. Because execution
stopped at qualification, no exact metric oracle or later fatal guard can be
counted as a substitute genuine-risk pass.

The synthetic observation routing, exact serial digests, source hashes,
capture hash, request-metric identities, schema-field absence, and zero
semantic-site count are all fatal-unscored or component evidence. Several are
fixed, by-construction, or identity checks. None increases a behavioral
denominator.

## Post-run live vLLM diagnostic

A separate post-run diagnostic launched real vLLM 0.26.0 through
`vllm.LLM` with the Granite skeleton runner and reached `SimWorker` and
`SimModelRunner`. Across one prefill and one decode step it emitted four
coordinator events: a 64-byte DP `all_reduce` and a 4,096-byte TP `all_reduce`
for each step. It still emitted no `ExecutionObservations` and no per-layer or
EP dispatch and combine schedule.

This diagnostic was not part of the frozen TRAF-13 runner, did not execute a
placement or metric relation, and is not included in the scored denominator.
Its artifacts are:

| Artifact | SHA-256 |
|---|---|
| `$SIMLLM_OBSERVED_SCHEDULE_RUN_ROOT/live-diagnostic-2026-08-12/live_evidence.json` | `0e1eb5f60bca23255282ee44b43b47404be4a3b2dde730bc346baae24c17dc01` |
| `$SIMLLM_OBSERVED_SCHEDULE_RUN_ROOT/live-diagnostic-2026-08-12/live_steps.jsonl` | `c795d24a7c8c7827873f7aef10ecceb12002f827ef1e3b3f268683761dd43243` |

The diagnostic reinforces the producer blocker but does not turn that blocker
into behavioral evidence.

## Verification evidence

Verification remains a separate evidence class. The post-implementation
registered command passed with `--check-only` and created no artifact. Ruff
passed for the full tree. The focused adapter, device-sink, and per-request
tests passed 61/61.

The full Python suite passed 918 tests and skipped 7, with one integration
failure: `test_readme_pro_progress_block_is_current`. Registering VLLM-22 and
SGL-17 increases their module open-task counts, while this worker's contract
forbids editing the integrator-owned `docs/README_PRO.md`. The regenerated
block expects 16 vLLM and 14 SGLang open tasks. This documentation drift is not
a runtime, adapter, lowering, or study failure, but the branch does not claim a
green full-suite gate until the integrator reconciles that generated block.

## TRAF-13 closure-scope map

The current registered TRAF-13 clauses map as follows.

> "connect at least one real framework schedule producer to
> `ObservedStepLowerer` after VLLM-22 or SGL-17 supplies captured operation
> order, streams, events and completion boundaries."

Not demonstrated. The new sink provides the handoff, but both the import-free
skeleton probe and the separate live vLLM diagnostic supplied no
`ExecutionObservations`. VLLM-22 owns the missing source-backed vLLM producer.
SGLang was explicitly outside this implementation; SGL-17 owns its optional
producer and exact disabled path.

> "The `DeviceRuntimeStepSink` component is ready, but its 2026-08-12
> qualification observed no vLLM schedule and matched 0 of 48 required
> semantic MoE sites."

Demonstrated. The component sink, raw adapter qualification, source audit, and
separate live diagnostic supply exactly this limited result. This sentence is
a status boundary, not evidence that any remaining acceptance clause passed.

> "Replay a fixed captured step through the traffic binding,
> `DeviceRuntime`, `CompletionEvent`, `StepResult`, TTFT and TPOT;"

Not demonstrated. The synthetic component fixture reached the coarse runtime,
events, reducer, and `StepResult`, but no captured Granite schedule reached the
chain and no TTFT or TPOT placement row ran. This clause remains in TRAF-13.

> "require every captured order and dependency fact to survive exactly"

Not demonstrated. There were no captured operation, stream, event-wait, layer,
request, or completion-frontier facts to preserve. VLLM-22 supplies the
missing producer facts; their end-to-end preservation remains in TRAF-13.

> "and show that one observed legal overlap changes the live metric in its
> registered direction."

Not demonstrated. Neither observed-versus-serial placement row nor the added
dependency-edge perturbation ran. The registered signs and bands remain
untested under TRAF-13.

> "Disabling the producer must select the serial lowerer and preserve every
> accepted serial graph, GOAL byte, timestamp and completion order exactly."

Partially demonstrated. The accepted two-layer graph and GOAL bytes matched
exactly, and absent observations selected the direct serial lowerer. The
Granite per-step graphs, GOAL bytes, timestamps, and completion order were not
run, so the full clause remains in TRAF-13.

The frozen per-request clause is also incomplete. The synthetic sink result
retained request IDs `d` and `p`, but no adapter-emitted MoE schedule existed
for traffic rebinding to preserve `request_pair_payload_bytes`. VLLM-22 owns
the adapter-emitted regression input, and TRAF-13 retains its metric-live
completion evidence.

TRAF-13 therefore does not close, and no ledger closure is claimed. No new
traffic or core residual is created: the existing TRAF-13 entry retains the
Granite metric, perturbation, attribution, and complete serial-off-path work.
The only newly mapped producer residuals are VLLM-22 and SGL-17.

## Integrator-owned contradiction sweep

`README.md` contains no matching stale overlap statement.
`docs/architecture.md:162-164` remains consistent: it says dependencies define
legal concurrency, the runtime realizes overlap, and no framework supplies an
overlap percentage. `docs/README_PRO.md:219-222` still points the
dependency-driven-overlap roadmap item only to closed TRAF-7. It should reflect
that TRAF-13 remains open behind VLLM-22 or SGL-17. Its generated task-progress
block also needs the two open-count updates described above. Per the worker
contract, those integrator-owned files were reported rather than edited.
