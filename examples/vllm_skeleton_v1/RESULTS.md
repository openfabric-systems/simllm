# vLLM worker skeleton v1 results

The flagged VLLM-13 skeleton passes all four exact-oracle sweep rows and all
four behavioral relation instances. Every ordinary step followed the frozen
14-call V1 sequence, emitted the existing schema-tagged `StepRecord`, and
kept the injected virtual clock at `123000 ps` because model computation was
deliberately empty. Both the initial and review-round live vLLM v0.26.0 smokes
succeeded through the dotted worker-class seam and generated one request with
two output tokens.

## Expectations and chronology

The original final expectations-only ancestor is commit `582d3de` (`docs:
complete vLLM skeleton expectations`). It follows the initial expectations
commit `6ef1910` and precedes the initial implementation, scripted study, and
live smoke. Those original checks are therefore auditable pre-registered
expectations, not post-specified regression checks.

Integration review then requested stronger structural checks and one
correctness repair. Commit `17b7bd1` (`docs: add review-triggered vLLM
expectations`) freezes those expectations after the initial evidence, but
before the fix implementation and every fix-round run. They are explicitly
review-triggered expectations and are not presented as retroactive
pre-registration for the initial implementation.

Reproduce the deterministic study from the repository root:

```bash
.venv/bin/python examples/vllm_skeleton_v1/run_vllm_skeleton_v1.py --check
```

The four small tracked rows are in [results.csv](results.csv). Per-cell JSONL
streams are runtime evidence rather than repository content and are written
under the runner's `--run-dir`, whose default is on `/data3/yifeng/`.

## Evidence accounting

Evidence classes remain separate as required by the repository validation
contract.

| Evidence class | Result | Scored meaning |
|---|---:|---|
| Run configurations | 4 | Unscored parameter records |
| Exact-oracle sweep rows | 4/4 pass | Behavioral headline |
| Behavioral relation families | 2/2 pass | Request fanout and prompt scaling |
| Behavioral relation instances | 4/4 pass | Two instances per family |
| Mirrored call-sequence comparison | pass in all 4 cells | Separate exact structural check |
| Fatal structural cells | 4/4 pass | Unscored invariants |
| Flag-gate negative controls | 3/3 pass | Unscored off-path invariants |
| Mirror tests without vLLM | 37/37 pass | Separate unit-test executable |
| Mirror tests with real vLLM | 35/35 applicable pass | Separate unit-test executable |
| Live vLLM smokes | 2 successes, one per round | Separate integration disposition |

Zero latency, schema identity, clock equality, record/result cardinality, no
physical device, and no stock runner are configuration-forced or structural
facts. A violation fails the study, but these checks never increase the four
row behavioral denominator.

The frozen expectations file has one overbroad summary sentence that says a
row pass includes the zero-latency clock relation. Its detailed evidence rules
and the repository contract instead classify configuration-forced zero
assertions as fatal and unscored. This report applies that required
classification: the row score covers the token and cardinality oracles, while
the clock relation remains a fatal structural check. All checks passed, so
this correction does not depend on an observed failure.

## Four-cell sweep

The sweep varies request count `R` in `{1, 3}` and prompt length `P` in
`{4, 16}`. Each cell has one full-prompt step and one decode step.

| R | P | Steps | Scheduled entries | Sampled tokens | Prefill tokens | Decode tokens | Total new tokens | Exact row |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 4 | 2 | 2 | 2 | 4 | 1 | 5 | pass |
| 1 | 16 | 2 | 2 | 2 | 16 | 1 | 17 | pass |
| 3 | 4 | 2 | 6 | 6 | 12 | 3 | 15 | pass |
| 3 | 16 | 2 | 6 | 6 | 48 | 3 | 51 | pass |

At fixed `P`, changing `R` from 1 to 3 triples scheduled entries, sampled
tokens, and total new-token work exactly in both instances. At fixed `R`,
changing `P` from 4 to 16 multiplies prefill work by four while decode work
and the two-step count remain unchanged exactly in both instances.

The source-frozen construction sequence was:

```text
init_device
load_model
get_kv_cache_spec
determine_available_memory
initialize_from_config
compile_or_warm_up_model
reset_mm_cache
get_supported_tasks
```

Every nonempty step followed:

```text
worker.execute_model
runner.execute_model
runner._update_states
runner._prepare_inputs
runner._determine_batch_execution_and_padding
runner._build_attention_metadata
runner._preprocess
runner._model_forward
worker.sample_tokens
runner.sample_tokens
runner._sample
runner._update_states_after_model_execute
runner._bookkeeping_sync
runner.eplb_step
```

The stock sources that identify this boundary are
`vllm/v1/worker/worker_base.py:245-259,317-320` for dotted worker resolution,
`vllm/v1/executor/uniproc_executor.py:62-69` for device initialization and
model loading, `vllm/v1/engine/core.py:243-324,576-606` for KV setup and the
execute/sample loop, and
`vllm/v1/worker/gpu_model_runner.py:4111-4479,4497-4736` for the selected V1
runner algorithm.

The study harness owns literal copies of both sequences. It does not import
the implementation's sequence constants, so changing implementation and
oracle together cannot make this structural comparison pass circularly.

## Virtual-clock and stream invariants

All four fixtures inject one `VirtualClock(start_ps=123000)`. Every mirrored
call obtains both timestamps from that object. Each of the eight step records
has `virtual_time_ps=123000`; each paired result has zero latency and
`completed_at_ps=123000`; every final clock remains `123000 ps`. The runner
and worker hold the same clock object.

All eight streamed JSON objects match their in-memory record exactly and use
schema `atlahs-closed-loop-step-v1`. These are fatal unscored invariants. In
particular, `completed_at = virtual_time + zero` is true by construction and
is not presented as independent behavioral evidence.

## Initial live in-process smoke

Exactly one live attempt ran on 2026-08-10 with the cached Granite snapshot,
offline Hugging Face mode, in-process V1 execution, the upstream V1 runner
selector, and the dotted worker-class argument:

```bash
env PYTHONPATH=/data3/yifeng/simllm-dev/worktrees/vllm13 \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V2_MODEL_RUNNER=0 \
  SIMLLM_VLLM_WORKER_MODE=skeleton SIMLLM_VLLM_MODE=virtual \
  SIMLLM_VLLM_STEP_RECORDS=/data3/yifeng/simllm-dev/wave1-runs/codex_vllm13_skeleton_mode/vllm_skeleton_v1/live_steps.jsonl \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_HOME=/home/yifeng/packages/vllm-rnic-capture/hf-cache \
  CUDA_VISIBLE_DEVICES= \
  /data3/yifeng/simllm-dev/venv-vllm/bin/python \
  examples/vllm_skeleton_v1/live_smoke.py
```

The environment was not fully GPU-invisible despite `CUDA_VISIBLE_DEVICES=`:
vLLM reported no CUDA runtime during extension setup, then its platform path
identified an NVIDIA GeForce GTX 1660 Ti and selected `device_config=cuda`.
There was therefore no pre-worker platform blocker to file. This discrepancy
is recorded rather than describing the host as CUDA-less.

The engine log confirmed vLLM v0.26.0, the
`simllm.adapters.vllm.SimWorker` argument, a 64-block logical KV pool, and
engine initialization in 0.00 seconds. The harness ended with:

```text
SMOKE_SIMWORKER_REACHED=True
SMOKE_OUTPUT_COUNT=1
```

The live stream contains two records, one three-token prefill and one
one-token decode, both at virtual time zero and both schema-tagged. Two model
steps supplied the requested two output tokens. The stock worker would select
and construct its runner at `vllm/v1/worker/gpu_worker.py:397-416`; this
skeleton's `init_device` override never calls that body. Worker-to-runner
forwarding is source-visible at `gpu_worker.py:701-713,923-927,955-956,
1080-1178`.

## Review-triggered fix round

The mirror tests now exercise the same fake `VllmConfig` against both the
transcribed no-vLLM worker base and the real pinned v0.26.0 `Worker`
constructor. The repository environment passes all 37 tests. The real-vLLM
environment passes all 35 applicable tests, with only the two tests whose
purpose requires vLLM to be absent skipped. The pinned environment does not
bundle pytest, so the local run exposed only pytest, `_pytest`, pluggy,
iniconfig, and `py.py` from a test-runner overlay under `/data3/yifeng/`; the
interpreter, torch, vLLM, and all runtime dependencies remained those of
`venv-vllm`.

The review also identified a silent failure in the documented VLLM-8 guard.
On v0.26.0, the executor-visible signal is
`SchedulerOutput.has_structured_output_requests`; request ids belong to the
later `GrammarOutput`. The executor and worker now refuse the scheduler
boolean before token fabrication, and the dual-environment test suite covers
the executor's public `execute_model` path. The source-inaccurate
`reset_prefix_cache` worker projection was removed; prefix-cache reset is a
scheduler-only operation at `vllm/v1/engine/core.py:779-784`.

The independent-literal deterministic study reproduced without changing the
tracked CSV:

```text
tracked results match 4 measured rows
exact-oracle rows: 4/4 PASS
behavioral relation instances: 4/4 PASS
fatal structural cells: 4/4 PASS
flag-gate negative controls: 3/3 PASS
```

Exactly one strengthened live attempt ran in this review round. It used a
fresh JSONL path and otherwise retained the initial cached-model and offline
configuration:

```bash
env PYTHONPATH=/data3/yifeng/simllm-dev/worktrees/vllm13 \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V2_MODEL_RUNNER=0 \
  SIMLLM_VLLM_WORKER_MODE=skeleton SIMLLM_VLLM_MODE=virtual \
  SIMLLM_VLLM_STEP_RECORDS=/data3/yifeng/simllm-dev/wave1-runs/codex_vllm13_skeleton_mode/vllm_skeleton_v1/live_steps_review_round.jsonl \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_HOME=/home/yifeng/packages/vllm-rnic-capture/hf-cache \
  CUDA_VISIBLE_DEVICES= \
  /data3/yifeng/simllm-dev/venv-vllm/bin/python \
  examples/vllm_skeleton_v1/live_smoke.py
```

All strengthened assertions passed:

```text
SMOKE_SIMWORKER_REACHED=True
SMOKE_SIMRUNNER_MIRROR=True
SMOKE_OUTPUT_COUNT=1
SMOKE_FABRICATED_TOKEN_ID=24577
SMOKE_SAMPLED_TOKEN_IDS=24577,24577
SMOKE_STEP_RECORD_COUNT=2
SMOKE_STEP_SCHEMA=atlahs-closed-loop-step-v1
```

The host again exposed the GTX 1660 Ti despite `CUDA_VISIBLE_DEVICES=`. This
run validates the asserted worker, runner, token, record-count, and schema
path, but it is not evidence for a genuinely GPU-invisible platform.

## Scope and residual work

This result validates the flagged GPU-state-free worker path and its live
reachability only. It does not validate the later GPU-present stock-init and
runner-rebind mode, simulated GPU or NCCL service, DP coordination above one,
device-free async multiprocessing, Ray or external-launch execution, CQ
consumer ownership, completion delivery, or device-schedule capture. Those
remain in VLLM-13 and its linked backend and core tasks. VLLM-16 separately
tracks the equivalent asserted smoke on a genuinely GPU-invisible host.
