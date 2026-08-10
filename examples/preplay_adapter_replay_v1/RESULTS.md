# Pre-play adapter replay v1 results

The final PLAY-3 implementation passes all four frozen scored relation
families. Both vLLM adapter paths serve exact joined token IDs, replay changes
scheduler-visible completion and the existing TTFT and TPOT chain by the
registered closed forms, the fabricated-token off path remains byte-identical,
and the real in-process vLLM v0.26.0 smoke reaches the skeleton runner.

## Expectations and chronology

The expectations-only commit is
`edcb2b9569053845d489a7908c472009b04f0454`. It precedes implementation and
every replay study or live run. The initial adapter implementation and both
study harnesses landed in
`e803b68acc9dfe9775e350f99c5e9d84d67c77bd` before the first scored run.

The metric study and off-path byte lock passed on that commit. The live smoke
then had three chronological attempts:

1. The first attempt failed before a token was served. vLLM transformed the
   external request ID `length-cap` into an internal ID of the form
   `length-cap-<eight hex digits>`, which the first implementation rejected as
   unjoined.
2. Commit `894b4b0cc0948ec11923fa2a09a0ffdc127a6c92` added a strict one-to-one
   mapping from that pinned internal form to the joined external identity.
   The second attempt served the request, then the offline `LLM.generate()`
   convenience loop failed while sorting the nonnumeric external ID with
   `int(request_id)`.
3. Commit `cc7d1b7a962243f5f27a4eec7e4768ff2938fcc0` retained counter-based
   submission and drove the same in-process `LLMEngine` loop directly below
   that unrelated final-output sort. The third attempt passed every frozen
   token, runner and drain assertion.

The identity randomization is source-visible at installed vLLM v0.26.0
`vllm/v1/engine/input_processor.py:223-240`; that complete file has SHA-256
`c5673988c0f7cfec268220e3f044e718702c015a4f236c020937cfd40a793f15`.
The integer-only convenience sort is at
`vllm/entrypoints/offline_utils.py:594-626`, in the file whose pre-freeze hash
is recorded in the expectations. These two details were found after the
freeze. Their unit regressions and harness correction are therefore
post-specified checks, not retroactive external-source pre-registration. The
frozen scored outcomes, exact oracle tokens and live drain semantics, were not
changed after either failure.

After the final fix commit, the complete metric study and byte-lock study were
rerun from fresh directories. The evidence reported below comes from those
final-commit reruns.

## Run configuration and evidence classes

The metric run used the repository `.venv`, two replay modes and token-cost
coefficients 100 and 200 ps. Bulk output is retained at
`/data3/yifeng/simllm-dev/wave2-runs/codex_play23_arrival_replay/preplay_adapter_replay_v1_final/`.
The live run used `/data3/yifeng/simllm-dev/venv-vllm`, the cached Granite
snapshot at revision `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`, offline
Hugging Face mode and a 64-block logical KV pool. Its passing artifacts are at
`/data3/yifeng/simllm-dev/wave2-runs/codex_play23_arrival_replay/preplay_adapter_replay_live_retry2/`.

Evidence classes remain separate.

| Evidence class | Result | Accounting |
|---|---:|---|
| Metric configurations | 4 | Unscored run records |
| B1 exact token and completion family | pass | Scored |
| B2 TTFT and TPOT cells | 2/2 pass | Scored |
| B3 coefficient-scaling family | pass | Scored |
| B4 final live-vLLM smoke | 1/1 pass | Scored |
| Metric structural guards | 4/4 pass | Fatal and unscored |
| No-replay VLLM-13 rows | 4/4 pass | Fatal byte lock |
| Full repository tests | 464 pass, 3 skipped | Separate executable |
| Focused tests without vLLM | 45/45 pass | Separate executable |
| Focused tests with real vLLM | 43/43 applicable pass | Separate executable, 2 skipped |

## B1: exact oracle serving and completion

At both token-cost coefficients, `SimExecutor` and the VLLM-13 skeleton served
exactly `r0=(38,)` and `r1=(61,62,63,64)`. The replayed token-producing visit
counts were exactly one for `r0` and four for `r1`. Scheduler-reported output
indices selected the tokens; no batch-position fallback participated.

The final replay step streams have SHA-256 values:

- `replay_c100_steps.jsonl`,
  `149aaa066b042a1e25873f1cb4e8cd92994297914d4d220298be6e63c0cbdcd6`;
- `replay_c200_steps.jsonl`,
  `b102416b53e0a01f2a8d8eadd800602d68d7950e44b2016bb54f6a6d33f36a59`.

Unit regressions reject unknown requests, invalid eight-hex aliases, two
runtime IDs bound to one joined request, admission limits that differ from the
oracle length, exhausted cursors and scheduler index gaps. A late invalid
request cannot partially advance an earlier request in the same batch.

## B2 and B3: end-to-end metrics

Every latency below came from a translated `StepRecord`, the frozen linear
compute sink and its `StepResult` completion timestamp.

| Token cost (ps) | Baseline TTFT (ps) | Replay TTFT (ps) | TTFT delta (ps) | Baseline TPOT (ps) | Replay TPOT (ps) | TPOT delta (ps) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 2180 | 2050 | -130 | 3740/3 | 1150 | -290/3 |
| 200 | 2780 | 2550 | -230 | 4240/3 | 1250 | -490/3 |

Both signed directions and all exact values match the frozen closed forms.
Increasing token cost from 100 to 200 ps made the TTFT improvement exactly
100 ps more negative and the TPOT improvement exactly `200/3 ps` more
negative. This rules out a constant post-hoc metric adjustment.

The 4,426-byte `summary.json` has SHA-256
`a36784750fcc19aa854fc72a40c891468d66bf91f08b9a18a70321f7d15a120e`.
The 127-byte `metrics.csv` has SHA-256
`9fd222edf81dbed09e46c377fb358246ce298b957b47d1e5e4f709de653c6e04`.

The decision-relevant B2 relation passed. Joined replay changes
scheduler-visible completion and reaches TTFT and TPOT through the accepted
step chain, so the pre-play oracle premise remains viable for PLAY-4 and
PLAY-5.

## B4: live vLLM skeleton replay

The passing smoke assigned external identity `length-cap`, which vLLM mapped
to internal identity `length-cap-ae7e3b36` in that run. `SimWorker` loaded the
joined run, constructed `SimModelRunner`, resolved the internal ID to exactly
one joined request and returned token ID 38. The externally visible completed
request retained identity `length-cap`.

An explicit empty completion-bearing call then returned no token, recorded
`finished_request_ids=["length-cap"]`, emitted a second
`atlahs-closed-loop-step-v1` record and added exactly zero picoseconds. The
live artifacts are:

- `summary.json`, 388 bytes, SHA-256
  `74b9a58ff0a5d672fc045fd625f96b418c2d08f5a2e906ab0290bc67b7e4fb22`;
- `steps.jsonl`, 445 bytes, SHA-256
  `add71dfbd00b2339b14d906eb849a29e7ce61fd73c2695ffc0bb14f6ac8e132b`;
- `joined-replay.json`, 721 bytes, SHA-256
  `bd32a975522092461fe840837a773aaf9ac67301b6b308cfcab0f1972b2ae96d`.

The host exposed an NVIDIA GeForce GTX 1660 Ti and vLLM selected its CUDA
platform, as in the earlier VLLM-13 smoke. The skeleton still touched no GPU
model state and initialized the engine in 0.00 seconds. This run is evidence
for the real vLLM control path, not for a GPU-invisible host.

## Bypass and fatal guards

The final no-replay rerun reproduced all four frozen VLLM-13 JSONL hashes
exactly:

- `r1_p4_steps.jsonl`,
  `e61fbcbda575adb68b7a6cb0b68581eb11ece8341cf413ecbca590d1853b3807`;
- `r1_p16_steps.jsonl`,
  `b77ebcb411a4fb2bdded4dcdb740eb9a5f64771484982508aeb7ef4a538b5b92`;
- `r3_p4_steps.jsonl`,
  `8d33f45905ab16d1445c1f59a30e627b497ef33e19dde3e654bea9761b3be527`;
- `r3_p16_steps.jsonl`,
  `cfd9202cccdc2caa9db8f54b95c96d71965fc971066b4f4ea03fd04320313cc5`.

The off path constructs no replay state and still calls the original token
fabricator. The metric study also retained exact fabricated token 512 and the
frozen baseline and replay schedule shapes. Both drain guards passed.
Speculative decoding and structured output keep their existing refusal paths.
All of these checks are fatal and unscored.

## Genuine-risk fraction

All four scored families, 100 percent, could plausibly fail in a competent
implementation:

- B1 risked cursor drift, position-based identity, partial batch mutation and
  divergence between executor and skeleton serving.
- B2 risked changing tokens without moving scheduler completion, or moving
  completion without reaching `StepResult`, TTFT and TPOT.
- B3 risked a constant metric adjustment or an incorrect count of the removed
  decode work.
- B4 risked upstream request-ID rewriting, runner-boundary mismatch and drain
  loss. Two actual failed attempts demonstrated that external-runtime risk
  before the final pass.

## Deliberate omission and residual work

This slice does not implement SGLang replay. PLAY-7
`(Completeness; P2; M)` owns that adapter half, including its identity off
mode and a real in-process smoke. PLAY-4 and PLAY-5 remain responsible for
routing projection and the broader oracle-consistency study. No PLAY-8 or
PLAY-9 task was needed.
