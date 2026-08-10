# Pre-play adapter replay v1 results

PLAY-3 passes its three scored relation families. Both vLLM adapter paths
serve exact joined token IDs, a real vLLM scheduler changes completion and the
existing TTFT and TPOT chain by the registered closed forms, and the live
vLLM v0.26.0 smoke reaches the skeleton runner. The fabricated-token off path
is locked by a tracked pytest byte fixture. The original B3 arithmetic check
passes but is fatal and unscored because it is implied by B2.

## Expectations and chronology

The original expectations-only commit is
`edcb2b9569053845d489a7908c472009b04f0454`. It precedes the original
implementation commit `e803b68acc9dfe9775e350f99c5e9d84d67c77bd` and every
PLAY-3 study or live run. The first live smoke exposed vLLM's randomized
internal request suffix, and the second exposed the offline convenience
wrapper's integer-only final-output sort. Commits
`894b4b0cc0948ec11923fa2a09a0ffdc127a6c92` and
`cc7d1b7a962243f5f27a4eec7e4768ff2938fcc0` made those post-specified
corrections before the first passing live smoke. Commit
`9db343669e2776b717b31031b677aafa09d9f0fc` recorded the original evidence.

The integration review identified that the no-replay bytes were locked only
by a study and that the B2 schedules were supplied by the harness. The
review-triggered expectations amendment is
`2dc6f973284829dd42bda9679e84ee43d9d62bb0`. It contains no implementation,
leaves the original freeze unchanged, and precedes review implementation
commit `0af815c338e05893929f50601ad26043624ed8c0`. Its registered check-only
command ran before that freeze and again before implementation was committed.
The real-scheduler study and amended live smoke ran only on the implementation
commit. Follow-up commit `97b85824c9a9a10d4cf7a7fdfa464d54d23f8b4e`
applied a later executor RPC context shrink to the same replay validator; it
does not alter the worker path or any measured study cell.

The intermediate PLAY-2 evidence commit
`ed02d20e61e17007df8c0460ca46a01c90c2b499` contained fabricated full hashes
for its freeze and implementation commits; closing evidence commit
`9db343669e2776b717b31031b677aafa09d9f0fc` corrected them to
`c4c17cff81e550053e090af430e3041e9efde057` and
`017a7219a22b24f56d44bbfac60df8b35a25be5e`.

## Run configuration and evidence classes

The original component study was rerun after the review fix under the
repository `.venv`; its bulk output remains outside Git in the machine-local
directory used for the historical run. The decision-relevant review study used
a machine-local vLLM v0.26.0 environment whose resolved historical path is
intentionally omitted, the cached Granite
revision `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`, offline Hugging Face
mode, no chunked prefill or asynchronous scheduling, a 64-token model limit
and a 64-block logical KV pool. Its output and the amended live-smoke output
remain in their historical machine-local directories. All three resolved
artifact paths are intentionally omitted. For current reproductions,
`SIMLLM_VLLM_PYTHON` selects the compatible vLLM interpreter and new output
sets default to `${SIMLLM_DATA_ROOT}/preplay_adapter_replay_v1/`, with
`review`, `engine` and `live_review` suffixes.

Evidence classes remain separate.

| Evidence class | Result | Accounting |
|---|---:|---|
| Metric configurations | 4 | Unscored run records |
| B1 exact token and completion family | pass | Scored |
| B2 real-scheduler TTFT and TPOT family | 4/4 cells pass | Scored |
| B3 coefficient scaling | pass | Fatal and unscored |
| B4 final live-vLLM smoke | 1/1 pass | Scored |
| Real-scheduler shape and drain guards | 4/4 pass | Fatal and unscored |
| Tracked no-replay JSONL comparison | pass | Fatal byte lock |
| Stop, identity, reset and atomicity regressions | pass | Fatal and unscored |
| Repository pytest | 469 passed, 3 skipped | Separate executable |
| Real-vLLM pytest with worktree `PYTHONPATH` | 467 passed, 5 skipped | Separate executable |
| Focused adapter pytest without vLLM | 50 passed | Separate executable |
| Focused adapter pytest with real vLLM | 48 passed, 2 skipped | Separate executable |

## B1: exact oracle serving

At both token-cost coefficients, `SimExecutor` and the VLLM-13 skeleton served
exactly `r0=(38,)` and `r1=(61,62,63,64)`. Scheduler-reported output indices
selected every token. The component study's replay streams retain SHA-256
values
`149aaa066b042a1e25873f1cb4e8cd92994297914d4d220298be6e63c0cbdcd6`
at 100 ps and
`b102416b53e0a01f2a8d8eadd800602d68d7950e44b2016bb54f6a6d33f36a59`
at 200 ps.

Replay now accepts only an exact joined scheduler ID. The live harness uses
vLLM's audited `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` mode, while a string
that merely looks like `<joined-id>-<eight hex digits>` fails as unjoined.
Admissions reject an early primary EOS, an early stop token and any
prompt-plus-oracle length beyond `max_model_len`. Equality with the model
limit and EOS only at the final oracle position remain valid.

The complete replay plan validates before the compute sink, durable record
append or virtual-clock advance. A two-token oracle whose first token is the
configured EOS fails with zero records, zero results, no stream bytes, an
unchanged clock and an unchanged replay cursor. `reset_configuration()` clears
every process-wide injection hook, and an explicitly injected no-replay config
overrides a stale replay hook.

## B2: real scheduler and end-to-end metrics

The review harness submitted `r0` alone, called one real `LLMEngine.step()`,
then submitted `r1` and only called the engine while requests remained. It
never supplied scheduler outputs, running sets, finish steps or batch shapes.
Both modes used the same prompts and request IDs. Baseline `r0` used the
workload limit four; replay `r0` used oracle length one. `r1` used length four
in both modes.

The vLLM scheduler produced these results in both pricing cells:

- baseline tokens were four fabricated 512 IDs for each request, with finish
  steps `r0=3` and `r1=4`;
- replay tokens were exactly `r0=(38,)` and `r1=(61,62,63,64)`, with finish
  steps `r0=0` and `r1=4`;
- baseline running sets were `r0`, then three `r0+r1` steps, then `r1`;
- replay running sets were `r0`, then four `r1` steps.

Each engine-produced `StepRecord` was priced online by the common sink
`1000 + c_token * sum(num_new_tokens) + 10 * sum(context_length)` ps.
`r1` arrival was the actual virtual timestamp immediately after engine step
0, namely 1,220 ps for `c_token=100` and 1,420 ps for `c_token=200`.

| Token cost (ps) | Baseline TTFT (ps) | Replay TTFT (ps) | TTFT delta (ps) | Baseline TPOT (ps) | Replay TPOT (ps) | TPOT delta (ps) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1460 | 1330 | -130 | 3740/3 | 1150 | -290/3 |
| 200 | 1860 | 1630 | -230 | 4240/3 | 1250 | -490/3 |

All exact values and both negative directions match the amendment. The deltas
are `-(c_token + 30)` for TTFT and `-(2*c_token + 90)/3` for TPOT. The
4,790-byte engine `summary.json` has SHA-256
`03eaf3fae964e6cf5fb3a72e85307b7b12bff513b47c808e3907154be44a0f7f`;
the 127-byte `metrics.csv` has SHA-256
`08a0f8f8b529e018c70cdc840cc17be5646e54796a28e0d03ae9c3d902e8fb8d`.

The decision-relevant relation passes using scheduler-owned completion and
engine-produced records. The pre-play oracle premise therefore remains viable
for PLAY-4 and PLAY-5.

The coefficient-scaling identity also passes: increasing token cost from 100
to 200 ps makes the TTFT improvement 100 ps more negative and the TPOT
improvement `200/3 ps` more negative. This is B3, retained as a fatal
consistency guard but removed from the scored denominator because the exact B2
cells already imply it.

## B4: live vLLM skeleton replay

The amended live smoke asserts that both the engine's internal request ID and
the externally visible completed request ID are exactly `length-cap`.
`SimWorker` loaded the joined run, constructed `SimModelRunner` and returned
oracle token ID 38. An explicit empty completion-bearing call then returned
no token, recorded `finished_request_ids=["length-cap"]`, emitted a second
`atlahs-closed-loop-step-v1` record and added zero picoseconds.

The 379-byte live `summary.json` has SHA-256
`817cfd381ee912f854498bd3c6f706fab0bd2324d886e2dfc217b9f423a4909c`.
The 436-byte `steps.jsonl` has SHA-256
`f777cb041f513cd3d9e45eced3fa745d096b4d0110d94343187b7b065a4e09a1`.
The host exposed an NVIDIA GeForce GTX 1660 Ti and vLLM selected its CUDA
platform, but the skeleton loaded no GPU model state and initialized the
engine in 0.00 seconds. This is real vLLM control-path evidence, not a claim
of a GPU-invisible host.

## Bypass and fatal guards

The normal pytest suite now drives the no-replay skeleton through one prefill
and one decode step and directly compares the emitted bytes with
`tests/fixtures/vllm/no_replay_r1_p4_steps.jsonl`. The tracked 529-byte
fixture has SHA-256
`e61fbcbda575adb68b7a6cb0b68581eb11ece8341cf413ecbca590d1853b3807`
and an explicit `text eol=lf` attribute. This turns off-path byte identity
into an ordinary suite failure instead of relying on a manual study rerun.

The broader no-replay study still reproduces all four frozen VLLM-13 hashes:

- `r1_p4_steps.jsonl`:
  `e61fbcbda575adb68b7a6cb0b68581eb11ece8341cf413ecbca590d1853b3807`;
- `r1_p16_steps.jsonl`:
  `b77ebcb411a4fb2bdded4dcdb740eb9a5f64771484982508aeb7ef4a538b5b92`;
- `r3_p4_steps.jsonl`:
  `8d33f45905ab16d1445c1f59a30e627b497ef33e19dde3e654bea9761b3be527`;
- `r3_p16_steps.jsonl`:
  `cfd9202cccdc2caa9db8f54b95c96d71965fc971066b4f4ea03fd04320313cc5`.

The off path constructs no replay state and calls the unchanged token
fabricator. Speculative decoding and structured output keep their explicit
refusal paths. Config echoes, expected batch shapes, delayed-drain state,
B3 arithmetic and rejection tests are fatal and unscored.

## Genuine-risk fraction

All three scored PLAY-3 families, 3/3 or 100 percent, could plausibly fail in
a competent implementation:

- B1 could fail through cursor drift, ambiguous identity, early scheduler
  stops, or divergence between the executor and skeleton token paths.
- B2 could serve correct tokens without changing scheduler completion, or
  change completion without propagating through `StepRecord`, `StepResult`,
  TTFT and TPOT. The first implementation's supplied schedules left this
  risk unresolved; the real-engine cells now exercise it.
- B4 could fail at the pinned external runtime through request-ID rewriting,
  runner-boundary mismatch or drain loss. The earlier smoke failures show
  that this risk is genuine.

B3 is not counted. Its coefficient difference is mechanically implied once
both exact B2 pricing cells pass, so treating it as a fourth scored family
would inflate the genuine-risk denominator.

## Deliberate omission and residual work

This slice does not implement SGLang replay. PLAY-7
`(Completeness; P2; M)` owns that adapter half, its fabricated-token identity
off path and a real in-process smoke. PLAY-4 and PLAY-5 remain responsible for
routing projection and the broader oracle-consistency study. No PLAY-8 or
PLAY-9 task was needed.
