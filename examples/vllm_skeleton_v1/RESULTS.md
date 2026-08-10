# vLLM worker skeleton v1 results

The flagged VLLM-13 skeleton passes all four exact-oracle sweep rows and all
four behavioral relation instances. Every ordinary step followed the frozen
14-call V1 sequence, emitted the existing schema-tagged `StepRecord`, and
kept the injected virtual clock at `123000 ps` because model computation was
deliberately empty. The single live vLLM v0.26.0 smoke also succeeded through
the dotted worker-class seam and generated one request with two output tokens.

## Expectations and chronology

The final expectations-only ancestor is commit `582d3de` (`docs: complete
vLLM skeleton expectations`). It follows the initial expectations-only commit
`6ef1910` and precedes the implementation, the scripted study, and the live
smoke. The checks below are therefore auditable pre-registered expectations,
not post-specified regression checks.

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
| Live vLLM smoke | 1 success | Separate integration disposition |

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

## Live in-process smoke

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

## Scope and residual work

This result validates the flagged GPU-state-free worker path and its live
reachability only. It does not validate the later GPU-present stock-init and
runner-rebind mode, simulated GPU or NCCL service, DP coordination above one,
device-free async multiprocessing, Ray or external-launch execution, CQ
consumer ownership, completion delivery, or device-schedule capture. Those
remain in VLLM-13 and its linked backend and core tasks.
