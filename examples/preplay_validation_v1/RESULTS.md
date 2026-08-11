# Pre-play validation v1 results

The PLAY-5 expectations were frozen in commit `bc5eb9e` before the comparison
implementation or any result-producing inference, scheduler, or backend run.
The final registered run used implementation commit `6c76dd8` on 2026-08-11.
It completed the replay half exactly and reached an environmental blocker in
the independent CPU half.

## Independent framework result

The Transformers runner completed both frozen sampling modes. Its greedy rows
retained the accepted PLAY-1 prompt lengths `(15, 22, 20)`, output lengths
`(3, 1, 5)`, and stop reasons `(eos, length-cap, stop-string)`. The seeded rows
also completed, with output lengths `(16, 1, 5)`. These captures are inputs to
the independent comparison, not comparison passes by themselves.

The one valid independent-framework attempt selected vLLM `CpuPlatform` and
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

The independent pair-table calculation matched every emitted GOAL exactly.
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

Commit `652989e` implemented the frozen evaluator. Three preliminary registered
invocations were preserved externally because they exposed harness plumbing
defects before a valid final result:

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
formula, expected direction, or acceptance band. Each preliminary CPU child
wrote a diagnostic blocker record but produced no model inference artifact or
scored comparison row. The final invocation passed all executable frozen
relations and recorded the stock CPU-worker blocker above.

PLAY-5 remains open only because the independent-framework comparison did not
execute. The replay end-to-end half is complete. PLAY-6 remains the separate
production framework-runner feature and was deliberately not implemented.
