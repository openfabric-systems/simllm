# Pre-play adapter replay v1 expectations

This document freezes the PLAY-3 validation contract before replay serving is
implemented and before the first replay or metric study run. PLAY-2 supplies
the strict joined run consumed here. PLAY-3 is live-reachable through the
existing adapter path from scheduler output to `StepRecord`, `StepResult`,
TTFT and TPOT.

## Frozen external-source audit

The mirrored runtime is the installed vLLM v0.26.0 under
`/data3/yifeng/simllm-dev/venv-vllm/lib/python3.12/site-packages/vllm`.
The source was audited before this freeze. The relevant files and complete
file SHA-256 values are:

- `vllm/v1/outputs.py`,
  `1e87bf44162452c1908d3a5003685937dbdc56f5634e35e11ed7b6a5322a1c15`;
- `vllm/v1/core/sched/output.py`,
  `d5d61ff186bea8deb09edfae3148531eb981f0079879df4e79e65ce5ad516d06`;
- `vllm/v1/core/sched/scheduler.py`,
  `2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941`;
- `vllm/v1/core/sched/utils.py`,
  `85e82eae555a03497ad2ac1540ed562a6c36fc26185aa6233725c914816aa1b3`;
- `vllm/entrypoints/offline_utils.py`,
  `688fbad0af9c2180b83aa77dcd0dbda85ca076a6c72bffa61840896d950cf458`.

The audited contracts are:

1. `ModelRunnerOutput` carries ordered request IDs, an ID-to-index map and a
   possibly different token list per request at `vllm/v1/outputs.py:231-244`.
2. `NewRequestData` carries the scheduler request ID and sampling parameters
   at `vllm/v1/core/sched/output.py:32-67`.
3. Scheduler update maps every scheduled ID through `req_id_to_index` and
   reads that request's sampled row at
   `vllm/v1/core/sched/scheduler.py:1652-1673`.
4. The scheduler appends generated tokens one at a time and invokes
   `check_stop` after each append at
   `vllm/v1/core/sched/scheduler.py:2006-2022`.
5. `check_stop` defers all stopping below `min_tokens`, then checks EOS, stop
   token IDs and the request's admission-time maximum at
   `vllm/v1/core/sched/utils.py:94-117`.
6. Offline `LLM` assigns `str(next(request_counter))` before calling the
   engine at `vllm/entrypoints/offline_utils.py:552-570`. The live smoke uses
   this source-frozen seam to assign the tracked request identity
   `length-cap`; it is not scored as an event-sequence oracle.
7. The stock V1 runner separates nonempty `execute_model` and `sample_tokens`
   at `vllm/v1/worker/gpu_model_runner.py:4111-4179,4497-4537`. The skeleton
   continues to mirror this split, but its author-defined internal call
   sequence remains a fatal unscored compatibility guard.

These citations are the author-independent referent for the adapter contract.
No exact internal call sequence is counted as scored behavioral evidence.

## Frozen replay artifact and request contract

The external oracle input is the tracked
`examples/preplay_trace_v1/granite_length_cap.jsonl` artifact. Its SHA-256 is
`36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341`.
Request `length-cap` has 22 prompt tokens, exactly one output token `(38,)`,
and stop reason `length-cap`.

A replay-enabled vLLM request must enter the scheduler with `max_tokens`
equal to its joined oracle output length. The adapter validates this on the
first scheduled appearance. A mismatch is fatal because a worker process
cannot safely rewrite the scheduler process's admission-time `Request.max_tokens`.
No joined run means no replay validation, token cursor or stop contract: the
accepted fabricated-token path remains the exact off mode.

Speculative decoding and structured output remain refused in both modes. No
replay configuration relaxes either VLLM-8 guard.

## Frozen metric workload and compute model

The deterministic metric workload has two requests and a one-step admission
boundary:

- `r0` arrives at 0 ps with prompt length 2. Its workload-model baseline has
  output length 4 and fixed token 512. Replay uses the tracked Granite oracle
  outcome `(38,)`, so its output length is 1.
- `r1` arrives at 500 ps with prompt length 3. Its registered synthetic oracle
  output is `(61, 62, 63, 64)`, length 4, in both baseline and replay. It is
  admitted at step 1, after the common `r0` prefill step.

The scheduler admits every ready request and executes one prefill or decode
visit per active request in each step. Therefore baseline step composition is
`(r0)`, `(r0,r1)`, `(r0,r1)`, `(r0,r1)`, `(r1)`, while replay composition is
`(r0)`, `(r1)`, `(r1)`, `(r1)`, `(r1)`. This sequence is a workload
definition and is a fatal unscored structural guard, not scored evidence.

Each step record is settled by the frozen linear compute model

```text
L = 1000 + c_token * sum(num_new_tokens) + 10 * sum(context_length) ps
```

with `c_token` swept over 100 and 200 ps. The replay mode and token-cost
coefficient are the two independent parameter families. The fixed overhead
and the `r1` contribution cancel in each baseline-to-replay delta.

TTFT is the first-token completion timestamp minus the request's fixed arrival
timestamp. TPOT for `r1` is the arithmetic mean of its three post-first-token
completion deltas. All metric arithmetic is exact rational arithmetic until
rendering.

## Scored behavioral relations

### B1: exact oracle tokens and completion counts

In both SimExecutor and skeleton-worker replay paths, every sampling visit
must return the joined token at the scheduler-reported output index. For the
metric workload, served sequences must be exactly `r0=(38,)` and
`r1=(61,62,63,64)`. The number of token-producing visits at completion must
equal the oracle lengths exactly, `r0=1` and `r1=4`. An out-of-range index or
a scheduled request absent from the joined run is a fatal error, never a
fabricated fallback.

### B2: end-to-end TTFT and TPOT movement

For `c_token=100 ps`, replay must change `r1` TTFT from 2,180 ps to 2,050 ps,
an exact delta of `-130 ps`. Its TPOT must change from `3740/3 ps` to
`1150 ps`, an exact delta of `-290/3 ps`.

For `c_token=200 ps`, replay must change `r1` TTFT from 2,780 ps to 2,550 ps,
an exact delta of `-230 ps`. Its TPOT must change from `4240/3 ps` to
`1250 ps`, an exact delta of `-490/3 ps`.

The general frozen deltas are

```text
delta_TTFT = -(c_token + 30) ps
delta_TPOT = -(2*c_token + 90) / 3 ps
```

Both directions are strictly negative. Every latency comes from a
`StepRecord` consumed by the registered compute sink and a `StepResult`
completion boundary. Direct arithmetic that bypasses this chain is not an
admissible pass.

This relation is decision-relevant. If joined replay cannot change
scheduler-visible completion and the resulting TTFT or TPOT through the
existing step chain, the pre-play oracle premise fails. PLAY-4 and PLAY-5
must then be redesigned rather than treating trace routing as live-reachable.

### B3: token-cost scaling of the replay delta

Increasing `c_token` from 100 to 200 ps must make the TTFT improvement 100 ps
more negative exactly. It must make the TPOT improvement `200/3 ps` more
negative exactly. This relation distinguishes removal of the scheduled `r0`
decode visits from a constant post-hoc metric adjustment.

### B4: live vLLM skeleton replay

One in-process vLLM v0.26.0 smoke uses the cached Granite snapshot at revision
`ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`, offline Hugging Face mode,
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, the dotted `SimWorker`, and the joined
tracked fixture. The request counter is replaced before submission with a
one-item iterator yielding `length-cap`, matching the audited source seam.

The smoke is scored. It must assert that vLLM returns exactly token ID 38,
the reached worker owns a `SimModelRunner` with replay enabled, the nonempty
step serves the same token, and an explicit empty completion-bearing call
after generation returns no token while recording `finished_request_ids` as
`length-cap` with zero additional latency. The streamed records must retain
schema `atlahs-closed-loop-step-v1`.

## Bypass-preserves-baseline lock

Before this freeze, the accepted four-cell VLLM-13 study was rerun with no
joined run and matched its tracked CSV. Its four emitted JSONL hashes were:

- `r1_p4_steps.jsonl`:
  `e61fbcbda575adb68b7a6cb0b68581eb11ece8341cf413ecbca590d1853b3807`;
- `r1_p16_steps.jsonl`:
  `b77ebcb411a4fb2bdded4dcdb740eb9a5f64771484982508aeb7ef4a538b5b92`;
- `r3_p4_steps.jsonl`:
  `8d33f45905ab16d1445c1f59a30e627b497ef33e19dde3e654bea9761b3be527`;
- `r3_p16_steps.jsonl`:
  `cfd9202cccdc2caa9db8f54b95c96d71965fc971066b4f4ea03fd04320313cc5`.

After implementation, rerunning the same command with replay configuration
absent must reproduce all four hashes exactly, as well as the same fabricated
sampled tokens and call names. This identity-off check is fatal and unscored.

## Other fatal unscored guards

- joined-run schema and trace hash validation complete before any token is
  served;
- request IDs are mapped by identity, never by batch position;
- duplicate request IDs in one output remain rejected;
- a sampling limit different from the oracle length is rejected on first
  admission;
- replay exhaustion, an unknown request, and a scheduler-reported index gap
  are rejected loudly;
- speculative decoding and structured output keep their existing errors;
- the off path constructs no replay state and calls the unchanged fabricated
  token helper;
- exact author-defined skeleton call sequences and fixed schedule shapes are
  structural only and never added to the scored denominator.

## Registered commands and pre-freeze dry runs

The registered commands are:

```text
.venv/bin/python examples/preplay_adapter_replay_v1/run_study.py --check-only
/data3/yifeng/simllm-dev/venv-vllm/bin/python examples/preplay_adapter_replay_v1/live_smoke.py --check-only
```

Before this freeze, both exact commands were executed against argument-parser
skeletons. They exited zero after resolving the tracked trace, cached model,
runtime and external output-root defaults. They produced no result rows,
model construction or output files. The parser skeletons were then removed,
so this expectations-only freeze contains no PLAY-3 implementation or study
harness.

The scored runs replace `--check-only` with fresh paths under
`/data3/yifeng/simllm-dev/wave2-runs/codex_play23_arrival_replay/`.

## Deliberate omission

This slice implements SimExecutor and the VLLM-13 skeleton only. The SGLang
adapter half is deferred explicitly to PLAY-7. Its unchanged fabricated-token
path is the required identity off mode until that task is selected.
