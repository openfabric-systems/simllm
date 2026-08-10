# Review-triggered regression expectations

Date: 2026-08-10

This amendment follows implementation commit `5329332` and the integration
review. It is post-specified relative to the original COMP-16 and VLLM-15
study, but it precedes the review fixes and their result-producing runs. It
does not change the frozen 11-row scored denominator in `expectations.md`.

## Replay and live sample-count relation

For the frozen B1 mixed batch, the live step sink and the serial replay
lowerer must both use `StepRecord.num_sampled=1`. The fused one-flop-per-ps
estimate is exactly `880,128 ps`, and both render `440 ns` per layer. Their
rendered GOAL text must be byte-identical.

Removing only `num_sampled` preserves the historical fallback to two
scheduled rows. Its fused estimate remains `912,896 ps`, its rendered layer
cost remains `456 ns`, and its replay GOAL SHA-256 remains
`7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6`.
Thus the exact field changes the fused estimate by `-32,768 ps` and the
rendered two-layer compute service by `-32,000 ps`; the absent-field bytes do
not change.

## Post-specified shape-sensitive relation

This is a regression check, not additional frozen behavioral evidence. In
the two-layer, TP-width-two Check A cell, the disabled even split emits
`(17, 17) ns` and the LM-head-in-last-layer split emits `(14, 21) ns`. The
right edge of the final layer's calc service interval, measured from entry to
that layer, therefore moves later by exactly

`21 - 17 = +4 ns`.

The runner must assert this value from the rendered GOAL reached through both
complete backend runs. It must also retain the separately frozen absolute
TTFT delta of `+1,000 ps`. A zero final-layer delta would reveal an even LM
head placement even if cumulative truncation happened to retain the TTFT
relation.

## Quiescence and evidence accounting

Every study CSV quiescence value must be derived from the backend wrapper's
`RnicRunResult.quiescent` projection. No CSV row may supply a constant
`"verified"` value.

Check A's four scored rows plausibly test cumulative-boundary truncation, so
their genuine-risk fraction remains 4 of 4 for that rule. Their folded layer
shape assertions are structural and unscored. In Check B2, only the
mid-prompt row distinguishes exact attribution from a blanket scheduled-row
count; its genuine-risk fraction is therefore 1 of 5. With B1 and the live
vLLM row each remaining 1 of 1, the overall plausible-risk count becomes 7
of 11. The post-specified shape regression remains outside this denominator.

## Registered review commands

The deterministic study command remains the command in `RESULTS.md`. The
touched Python gates are:

```bash
.venv/bin/pytest -q tests/test_step_lowerer.py tests/test_step_sink.py
/data3/yifeng/simllm-dev/venv-vllm/bin/python -m pytest -q \
  tests/test_step_lowerer.py tests/test_step_sink.py
```

Before this amendment is committed, the study command is executed with
`--check-only`, and both test commands are executed with `--collect-only`.
These modes parse the registered surfaces and produce no study results.
There is no external-system expectation in this amendment, so no new
external-source audit applies.
