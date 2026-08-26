# Deployment-curve anchor freeze

This is the expectations-only anchor freeze for the CORE-54 scaffold. It
precedes the study implementation, the granite dry run, every generated curve,
every fitted constant and every scored flagship run. The machine-readable
authority is `expectations.json`; this file explains its split and its limits.

## Published evidence

The freeze transcribes every numeric disclosure in
`docs/papers/deepseek-deployment-disclosures.md` with a source ID and source
locator. It records both published systems: the 96-H100 SGLang reproduction
target and DeepSeek's H800 production deployment. Configuration counts,
parallelism degrees, prompt and cache lengths, batch sizes, expert counts,
throughputs, latency statements, cost, comparison values and stated percentage
deltas remain distinct fields. Rounded and approximate values are labeled as
such rather than silently converted into exact measurements.

No external data is downloaded. The two source URLs identify the public record;
the repository dossier is the transcription used by the harness.

## Calibration and held-out split

Two exact SGLang table values are calibration anchors: the 1K prefill row at
57,674 tokens per second per node and the standard decode row at 22,282 tokens
per second per node. Only these IDs may be visible to a constants fit.

Three exact SGLang table values are held out: the 2K prefill row at 54,543, the
4K prefill row at 50,302 and the simulated multi-token prediction decode row at
17,373 tokens per second per node. Only these IDs may be visible to the scoring
path. Each held-out simulated value must lie within 5 percent of its published
value. When the simulated value carries an uncertainty interval, that interval
must intersect the closed 5 percent disclosure interval.

The rounded headline values, the 2 to 5 second time-to-first-token range, the
approximately 100 millisecond inter-token latency, the estimated 0.20 dollars
per million output tokens, the stated comparison deltas and DeepSeek's
production averages are context-only. DeepSeek's values define the second
legend, not fit inputs for the SGLang target. Context-only anchors cannot enter
either fitting or held-out scoring.

## Curve contract

Each configuration traces one `simllm-deployment-curve-v1` record as offered
load increases. The horizontal axis is aggregated output throughput in tokens
per second, increasing rightward. It is all terminal output tokens divided by
the interval from first admission to last terminal completion.

The vertical axis is inverse per-token request delay in tokens per second,
increasing upward. The raw record stores per-token request delay in
picoseconds, so the plotted coordinate is `1,000,000,000,000 / delay_ps`. The
upper-right corner is optimal.

## Uncertainty freeze boundary

This commit freezes the 5 percent disclosure bar, but it freezes no numerical
component-derived uncertainty band. Such a band depends on compute, collective,
host-submission, fabric and KV-handoff pricing records that do not all exist on
this scaffold branch. The later scored-run expectations commit must name those
records and freeze the resulting numerical bands before the first scored run.
It may not edit the anchors, their split, the axes or the 5 percent bar frozen
here.

## Scope

This freeze does not claim a DeepSeek result, a calibrated constant, a
publication figure or CORE-54 closure. It establishes immutable inputs for the
scaffold. CORE-54 remains open on its registered dependencies.
