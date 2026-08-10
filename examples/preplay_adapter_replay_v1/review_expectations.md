# Pre-play adapter replay review expectations

This is a review-triggered amendment to the PLAY-3 validation contract. It is
frozen after the original PLAY-3 implementation and evidence, but before any
review-round implementation or review study run. It does not edit or claim
retroactive pre-registration for `expectations.md`.

The review has two objectives. First, move the identity-off byte lock into the
ordinary test suite. Second, replace the decision-relevant schedule fixture
with a scored in-process vLLM run whose real scheduler decides membership and
completion. The remaining sections freeze the requested state-isolation,
stop, identity and pre-settlement regressions.

## External-source audit before freeze

The review study uses the installed vLLM v0.26.0 package from the
machine-local environment audited before this amendment; its resolved
historical path is intentionally omitted. The complete hashes below identify
the audited source. For a current reproduction, `SIMLLM_VLLM_PYTHON`
selects the compatible interpreter.

These complete source-file hashes and line ranges were audited before this
amendment:

- `vllm/v1/engine/input_processor.py`, SHA-256
  `c5673988c0f7cfec268220e3f044e718702c015a4f236c020937cfd40a793f15`.
  Lines 223-240 preserve the external request ID and otherwise add an
  eight-hex internal suffix. Lines 310-327 clone sampling parameters, fill an
  absent request limit from `max_model_len`, and inject the tokenizer and
  generation-config EOS channels before scheduler admission.
- `vllm/sampling_params.py`, SHA-256
  `d2f9789ba2b93819c4918159cfd29818eab3ba4f9098241e2febadcc690aa767`.
  Lines 632-651 set the primary EOS ID and any additional EOS IDs; lines
  709-714 expose the scheduler-visible EOS and combined stop-token sets.
- `vllm/v1/core/sched/utils.py`, SHA-256
  `85e82eae555a03497ad2ac1540ed562a6c36fc26185aa6233725c914816aa1b3`.
  Lines 94-117 apply minimum-token gating, then primary EOS, stop-token,
  model-length and request-length completion after each appended token.
- `vllm/v1/core/sched/scheduler.py`, SHA-256
  `2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941`.
  Lines 465-530 schedule running requests first and compute each request's
  next required work from its live token state. Lines 2006-2022 append the
  model output and call `check_stop` before constructing completion output.
- `vllm/v1/engine/llm_engine.py`, SHA-256
  `17e5edfc625c77e9663368c7d69136e5e5935ee81608a65be3996411d502225e`.
  Lines 290-320 drive one real engine-core step and process its outputs.
- `vllm/v1/engine/output_processor.py`, SHA-256
  `ee10351275d90796c8b901a5f4b23d5a046ef6ee72fd2921aff2ae78ca58bd9b`.
  Lines 610-689 project each engine-core output, its new tokens and its finish
  reason into request-visible outputs, then remove completed requests.

The `ModelRunnerOutput` identity mapping and executor-to-scheduler token path
remain pinned by the source audit in the original expectations. No
author-defined internal call sequence is scored here.

## R1: pytest identity-off byte lock

The tracked fixture will be
`tests/fixtures/vllm/no_replay_r1_p4_steps.jsonl`. It contains the accepted
two-step, one-request, four-prompt-token VLLM-13 skeleton stream at virtual
time 123,000 ps. Its already frozen baseline SHA-256 is
`e61fbcbda575adb68b7a6cb0b68581eb11ece8341cf413ecbca590d1853b3807`.

A normal pytest must construct the skeleton with no replay configuration,
drive the prefill and decode visits end to end, and compare the produced JSONL
bytes directly with that tracked file. The fixture must have an explicit
`text eol=lf` rule in the repository `.gitattributes`. Any byte difference is
fatal and unscored; no semantic JSON comparison may substitute for it.

## R2: real-scheduler B2 relation

The cached Granite snapshot and synthetic joined metric trace remain the same
model and oracle basis as the original study. The in-process vLLM engine runs
with the real v0.26.0 scheduler, the dotted `SimWorker`, no chunked prefill,
no asynchronous scheduling, a 64-token model limit and a 64-block logical KV
pool. Because joined request identities are unique, the study sets
`VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`; the scheduler must then expose the
exact joined IDs. A suffix-shaped lookalike is not accepted as evidence of an
external identity.

Each cell submits the same two requests at the same scheduler boundaries:

- `r0` has prompt length 2 and is the only request admitted for engine step 0.
  The workload baseline limit is 4 with fabricated token 512. Replay uses
  oracle sequence `(38,)`, length 1.
- `r1` has prompt length 3 and is admitted immediately after engine step 0.
  Its baseline limit and replay oracle length are both 4; replay tokens are
  `(61,62,63,64)`.

After `r1` admission, the harness only calls `LLMEngine.step()` until the real
engine reports no unfinished requests. It does not supply scheduler outputs,
active-request lists, finish steps or batch shapes. Request outputs from each
engine step identify token-visible completion. The corresponding
`StepRecord` is priced online by the same deterministic sink in every cell:

```text
L = 1000 + c_token * sum(num_new_tokens) + 10 * sum(context_length) ps
```

The sweep uses `c_token` values 100 and 200 ps. `r1` arrival is its actual
engine-admission virtual timestamp after step 0. TTFT is its first output
timestamp minus that arrival; TPOT is the exact rational mean of its three
later inter-token intervals.

The scored B2 assertions are:

1. Baseline scheduler-visible output counts are `r0=4,r1=4`; the `r0` count
   differs from its oracle. Replay counts are exactly `r0=1,r1=4` and replay
   token IDs equal both oracles.
2. The real scheduler completes baseline `r0` on engine step 3 and replay
   `r0` on step 0. It completes `r1` on step 4 in both modes. These finish
   steps come from vLLM request outputs, not a supplied schedule fixture.
3. The exact metrics are:

| Token cost (ps) | Baseline TTFT (ps) | Replay TTFT (ps) | TTFT delta (ps) | Baseline TPOT (ps) | Replay TPOT (ps) | TPOT delta (ps) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1460 | 1330 | -130 | 3740/3 | 1150 | -290/3 |
| 200 | 1860 | 1630 | -230 | 4240/3 | 1250 | -490/3 |

Both metric directions are strictly negative. The deltas retain the frozen
closed forms `-(c_token + 30)` and `-(2*c_token + 90)/3`, but the pass is
computed from engine-produced records and output timestamps.

This is the decision-relevant relation. If the real scheduler does not remove
`r0` after one replayed token, or that decision does not move `r1` TTFT and
TPOT through the common sink, the pre-play oracle premise fails and PLAY-4
and PLAY-5 require redesign.

The expected running-set shapes and exact coefficient-scaling identities are
fatal unscored consistency checks. The original B3 is removed from the scored
family count because its arithmetic is implied by the exact B2 cells.

## R3: process-wide configuration reset

The adapter must expose an explicit reset that clears every process-wide
injection hook, including replay configuration. An explicitly injected
constructor config has first priority, followed by current hooks, then the
environment. A regression must configure replay, reset, construct another
worker or executor in the same process and prove the second construction has
no replay state. A separate assertion must prove an injected no-replay config
overrides a stale replay hook.

## R4: stop-position closure

Replay admission must reject every scheduler stop channel that can end a
request before its oracle length:

- a primary `sampling_params.eos_token_id` appearing in the oracle prefix at
  or after `min_tokens`;
- any stop-token ID appearing in that same effective prefix;
- `prompt_length + oracle_output_length > max_model_len`.

Equality with `max_model_len` and an EOS token only at the final oracle
position are allowed because they stop on the required position. A focused
regression uses a two-token oracle whose first token is the injected EOS ID;
it must fail before its would-be final engine step can settle or emit a
record. This closes the offline path where no later drain reports the length
disagreement.

## R5: unambiguous request identity

Replay accepts only an exact joined request ID or a previously explicit
one-to-one binding. It must reject an otherwise unjoined string that merely
has the shape `<joined-id>-<eight hex digits>`. Real vLLM replay therefore
uses its audited no-randomization mode for unique IDs. No suffix heuristic may
silently infer external identity.

## R6: validate before settlement

For replay mode, complete identity, admission, stop-channel, cursor and batch
validation must run before the compute sink, durable `StepRecord` append or
virtual-clock advance. A rejected batch leaves record count, result count,
stream bytes, clock and replay cursor unchanged. The accepted path retains
the existing settlement and sample order as observed by vLLM. The no-replay
path remains the byte-locked identity path from R1.

## R7: evidence corrections and documentation

The review result must:

- classify original B3 as fatal and unscored, then restate the PLAY-3 scored
  family count and genuine-risk fraction;
- assert, rather than merely record, that the live smoke returns external
  request identity `length-cap`;
- state plainly that an intermediate PLAY-2 results commit contained two
  fabricated full hashes and that the closing evidence commit corrected them
  to the real commit hashes;
- add a PLAY-7 cross-reference to `docs/modules/adapters-sglang.md` without
  claiming SGLang replay is implemented.

These are evidence and navigation guards, not behavioral scores.

## Registered command and dry run

The historical dry run used the same executable basename, script, options and
pinned inputs; resolved machine-local paths are intentionally omitted. The
following is a portable post-freeze rendering, not a verbatim transcript.
Source the local configuration first:

```text
"${SIMLLM_VLLM_PYTHON:?configure SIMLLM_VLLM_PYTHON}" examples/preplay_adapter_replay_v1/run_engine_scheduler_study.py --check-only
```

Before this freeze, the historical resolved form of that command ran against
a temporary parser skeleton. It verified the tracked trace, cached model and
external output-root default, exited zero, constructed no engine, produced no
result rows and did not create the run directory. The temporary skeleton was
then removed. This amendment contains no review implementation, fixture or
generated result.

The frozen storage requirement is that the scored run replace `--check-only`
with a fresh `--run-dir` outside Git and run only after the review
implementation commit; the resolved historical target is intentionally
omitted. As a post-freeze portability convention, new runs default below
`${SIMLLM_DATA_ROOT}/preplay_adapter_replay_v1/`.

## Genuine-risk estimate before implementation

The final scored PLAY-3 families will be B1, B2 and B4. B2 gains this real
scheduler instance; B3 is unscored. All three families can plausibly fail:
B1 through cursor or adapter divergence, B2 through scheduler completion or
metric-chain divergence, and B4 through the pinned external runtime. The
expected scored genuine-risk fraction is therefore 3/3, 100 percent. R1 and
R3 through R7 are fatal regressions and do not increase the scored
denominator.
