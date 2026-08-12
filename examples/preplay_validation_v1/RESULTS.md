# Pre-play validation v1 results

The PLAY-5 expectations were frozen in commit `bc5eb9e` before the comparison
implementation or any result-producing inference, scheduler, or backend run.
The final registered run used implementation commit `6c76dd8` on 2026-08-11.
It completed the replay half exactly and reached an environmental blocker in
the independent CPU half.

The 96-send counts below are the historical pre-TRAF-25 observations. The
corrected replay used 48 sends per step while retaining every JCT and all 13
scored relations; see
[the token ownership results](../token_ownership_v1/RESULTS.md#preplay_validation_v1).

## Post-specified fix round 1 corrections

The notes in this section were authored after the original results were
observed. They correct the evidence description without changing commit
`bc5eb9e`, the preserved runs, or the original 13/13 replay headline.

### Scored-total accounting

Commit `d50cc19` fixes a real latent reporting defect. A future executed
oracle comparison now contributes the number of rows whose `passed` field is
true, rather than its full executed denominator, to `passed_scored`. A focused
regression presents one passing and one failing oracle row and requires a
contribution of one. The original run had zero executed oracle rows, so its
persisted 13/13 count and summary hash do not change. The summary is still
written before a failing-oracle assertion so the failure remains preserved,
but it can no longer publish an inflated pass count.

### EOS chronology and replay observability

The EOS normalization in commit `6c76dd8` was authored after the preserved
11/13 run exposed two EOS stop-reason mismatches. It is a post-specified
semantic normalization, not a frozen prediction. vLLM v1 represented each
EOS completion as `finish_reason="stop"`, `stop_reason=None`, while retaining
the EOS token as the scheduler-owned final token. Treating that form as `eos`
is faithful because the same scored row independently requires equality of
the complete token sequence, including that final token. Commit `d50cc19`
also guards the degenerate case: `choice.stop_reason == eos_token_id` cannot
identify EOS when both values are `None`.

Replay admission pins `max_tokens` to the oracle output length. A late length
overshoot is therefore structurally unobservable, and the length-cap reason
follows from exact tokens at that admission cap. The scored completion rows
still carry genuine risk through early termination, wrong tokens, missing or
duplicate completion, and non-length stop reasons. The preserved 11/13 run
demonstrates that the stop-reason predicate can fail live.

### PLAY-B3 input authority

The primary PLAY-B3 evaluator read `replay/routed-experts.json` and used the
same context-indexed slice shape as the traffic projection. Calling that check
independent of the projection was too strong, and that claim is withdrawn.
It still compared separately calculated pair tables with the emitted GOAL
files, but it could not by itself detect a defect already present in the
raw-trace-to-routing projection.

Commit `d50cc19` adds a stronger post-specified recomputation. It reads the
raw greedy JSONL request and `forward-token` rows, advances its own per-request
prefill and decode cursors from the preserved scheduler records, applies only
the fixed expert ownership table and the frozen 2,048-byte closed form, and
never reads `routed-experts.json`. All five GOAL tables at each bandwidth
matched, for 10/10 exact post-specified rows with genuine-risk fraction
`10/10 = 100%`. Scheduler membership and expected tables were identical across
bandwidths. The new `raw_goal_recheck.json` SHA-256 is
`5bde5ec30bf33b144a64e0ac90360ff7a50cb821d9c7294e0ecdaa2f8635b411`.
No primary study half was rerun, and the original 13/13 artifact has no delta.

The post-specified command is:

```text
.venv/bin/python examples/preplay_validation_v1/recompute_raw_goal_tables.py \
  --trace "${SIMLLM_PLAY5_RUN_ROOT:?configure SIMLLM_PLAY5_RUN_ROOT}/transformers/greedy.jsonl" \
  --steps-200 "${SIMLLM_PLAY5_RUN_ROOT:?configure SIMLLM_PLAY5_RUN_ROOT}/replay/200g/steps.jsonl" \
  --goals-200 "${SIMLLM_PLAY5_RUN_ROOT:?configure SIMLLM_PLAY5_RUN_ROOT}/replay/200g/htsim" \
  --steps-400 "${SIMLLM_PLAY5_RUN_ROOT:?configure SIMLLM_PLAY5_RUN_ROOT}/replay/400g/steps.jsonl" \
  --goals-400 "${SIMLLM_PLAY5_RUN_ROOT:?configure SIMLLM_PLAY5_RUN_ROOT}/replay/400g/htsim" \
  --out "${SIMLLM_PLAY5_FIX1_RUN_ROOT:?configure SIMLLM_PLAY5_FIX1_RUN_ROOT}/raw_goal_recheck.json"
```

The same command with `--check-only` prints its confirmation by design and
produces no artifacts.

### Independent CPU invocation chronology

The registered command was invoked four times. Each invocation wrote a CPU
blocker record, and none ran a vLLM model forward:

- At `652989e`, `EngineArgs` rejected the unsupported `device="cpu"` keyword.
  Its blocker SHA-256 is
  `75e747945b0e725890a04edafb3f84b8057ed183ee6aab8235fc5cb468168a25`.
- At `9dc343d`, after removing that keyword, the class-object worker argument
  was serialized to bytes and rejected by `ParallelConfig`, which requires a
  dotted string. Its blocker SHA-256 is
  `f0eec68d6d9e9992044ebc4ec013ca329a7470c2ad7ad1afad41e5a6a91e0d7f`.
- At `0f0c29f`, the dotted worker string was correct, but importing `LLM`
  before installing `CpuPlatform` bound the wrong platform state and reached
  a `VllmConfig` assertion. Its blocker SHA-256 is
  `531655510e6028116a11f9ffa8d512b83204ca58d5f67b999cf716fa6c606f79`.
- At `6c76dd8`, `LLM` was first imported after the CPU override. This was the
  first invocation to fulfill the registered platform and dotted-worker
  configuration. It reached `CPUWorker.__init__` and stopped only at the
  missing `init_cpu_memory_env` operator. Its blocker SHA-256 is
  `2a38ae45271c99b67d53fff8badfda6e4ff03910a3c571ef640118e6f8f62ab4`.

The last invocation is the registered CPU attempt used for the frozen
disposition because it is the only one that reached the intended stock-worker
boundary. This accounting does not retroactively demote or erase the first
three invocations.

### Incremental evidence and tighter guards

The 13/13 replay set partially re-derives relations already landed by
`routed_supply_v1` on the same trace, specifically the routed pair-table
closed form and the `-20 ps/byte` bandwidth relation. Its incremental evidence
is the real scheduler and joined replay chain through exact oracle completion,
captured supply, `StepRecord`, backend `StepResult`, TTFT, and TPOT, not the
novelty of those two relations.

Commit `d50cc19` also requires prompt-token equality before assigning a seeded
sampler difference and requires both routing observations to place the two
changed experts on opposite sides of their own top-8 versus top-9 boundary.
No oracle row executed in the preserved environment, so these tightenings
change no observed classification. Finally, the harness pins
`vllm/platforms/__init__.py` and `vllm/platforms/cpu.py` by SHA-256 even though
those two pins were not frozen in `expectations.md`. They are post-freeze
source-identity tightenings and do not increase any score.

## Independent framework result

The Transformers runner completed both frozen sampling modes. Its greedy rows
retained the accepted PLAY-1 prompt lengths `(15, 22, 20)`, output lengths
`(3, 1, 5)`, and stop reasons `(eos, length-cap, stop-string)`. The seeded rows
also completed, with output lengths `(16, 1, 5)`. These captures are inputs to
the independent comparison, not comparison passes by themselves.

The final independent-framework invocation selected vLLM `CpuPlatform` and
entered the dotted validation subclass of the stock `CPUWorker`. Construction
then failed at `vllm/v1/worker/cpu_worker.py:71`: the installed CUDA build does
not export `torch.ops._C.init_cpu_memory_env`. The exception was
`AttributeError`, no vLLM model forward ran, and no partial framework oracle
artifact was accepted. Consequently PLAY-B1 executed 0 scored rows, six rows
are blocked outside the denominator, and no divergence was silently accepted
or classified without evidence.

The required environment to execute the remaining half is a vLLM 0.26.0 CPU
build that selects `CpuPlatform`, exports `init_cpu_memory_env`, constructs the
stock `CPUWorker` and `CPUModelRunner`, and loads the pinned Granite snapshot
entirely on CPU. A CUDA build with a platform override is not sufficient.

## Routed replay result

Both live cells submitted all three requests together to the real vLLM
scheduler. They used the joined greedy trace, the device-free `SimWorker`, the
captured routing supply, and the `rnic-nn-fluid` backend.

| Evidence family | Passed | Executed |
|---|---:|---:|
| Scheduler-visible completion | 6 | 6 |
| Captured all-to-all stream | 2 | 2 |
| TTFT bandwidth relation | 3 | 3 |
| TPOT bandwidth relation | 2 | 2 |
| Total | 13 | 13 |

Every bandwidth cell returned the exact oracle token sequences and normalized
stop reasons. Completion order was `length-cap`, `eos-brief`, then
`stop-string`; bandwidth did not change request membership, completion order,
tokens, lengths, or stop reasons.

The original pair-table calculation matched every emitted GOAL exactly.
Each bandwidth produced five scheduler steps, each with 48 dispatch/combine
tags and 96 sparse sends. All ten backend runs reported captured routing,
placement epoch zero, and physical quiescence.

| Step | JCT at 200 Gbit/s (ps) | JCT at 400 Gbit/s (ps) | Measured and expected 400 minus 200 (ps) |
|---:|---:|---:|---:|
| 0 | 320,157,120 | 208,090,560 | -112,066,560 |
| 1 | 103,888,320 | 99,956,160 | -3,932,160 |
| 2 | 103,888,320 | 99,956,160 | -3,932,160 |
| 3 | 99,956,160 | 97,990,080 | -1,966,080 |
| 4 | 99,956,160 | 97,990,080 | -1,966,080 |

All three TTFT instances had an observed and expected 400 minus 200 Gbit/s
difference of `-112,066,560 ps`. The two TPOT instances also matched exactly:
`eos-brief` was `-3,932,160 ps` and `stop-string` was `-2,949,120 ps`.
Every signed relation was strictly negative as frozen.

## Evidence accounting

The executed behavioral headline is 13/13, with genuine-risk fraction
`13/13 = 100%`. The six blocked independent-oracle rows are not passes and do
not enter that denominator. Fatal unscored guards for source identity, greedy
oracle identity, captured-token conservation, bandwidth scheduler identity,
step/result cardinality, replay completion, GOAL uniqueness, captured routing,
placement epoch, and backend quiescence all passed.

The final external `summary.json` SHA-256 is
`d70ebd254c9fe6556ae820d77dfc43124446ba17cd8e37470a98663cc65af54a`.
The greedy and seeded trace hashes are respectively
`e354323912e4544fecb60974ec88d57b916d75baa0c9b3a3ddc5ccf5761bdf24`
and `56dadfc7e785c326744d173a8f146d9daae6e6b5541f50278776061c5c33f10c`.
Bulk traces, GOAL files, backend CSV files, and logs remain below the configured
`SIMLLM_PLAY5_RUN_ROOT` and are not tracked.

## Run chronology

Commit `652989e` implemented the frozen evaluator. The first three of four
registered-command invocations were preserved externally because they exposed
harness plumbing defects before the final result:

- the first invocation printed its input confirmation and diagnostic logs,
  then stopped before a replay cell because the companion GOAL converter was
  not discoverable;
- the second printed its diagnostics, selected a stale converter, and stopped
  when that converter faulted before the backend run;
- the third completed the two replay cells and exposed vLLM's raw EOS form,
  `finish_reason="stop"`, `stop_reason=None`, with the EOS token still present.
  It reported 11/13 because the harness had not normalized that audited form.

Commits `9dc343d`, `0f0c29f`, and `6c76dd8` corrected converter discovery,
selected the converter paired with the supplied htsim build, fixed the CPU
platform import boundary, and normalized EOS from the scheduler-owned final
token. None changed a frozen request, relation, threshold, bandwidth, pair-size
formula, expected direction, or acceptance band. Each of the four CPU children
wrote a diagnostic blocker record but produced no model inference artifact or
scored comparison row. The exact CPU chronology and post-specified distinction
above supersede the earlier shorthand that called only the last invocation
valid. The final invocation passed all executable frozen relations and
recorded the stock CPU-worker blocker above.

PLAY-5 remains open only because the independent-framework comparison did not
execute. The replay end-to-end half is complete. PLAY-6 remains the separate
production framework-runner feature and was deliberately not implemented.
