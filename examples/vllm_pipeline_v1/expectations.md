# Pipeline executor expectations

These expectations precede implementation and the first study run. The study
qualifies a model-side vLLM v0.27.1 adapter, not physical GPU performance.

## Contract and fatal guards

- PP1 retains its existing outputs, durations, record order and supported
  configuration behavior exactly. The PP path is selected only for PP > 1.
- PP2 and PP4 use the real engine's batch queue. Execute/sample interleavings
  retain FIFO identity. A drain has no sampling slot; duplicate consumption,
  unsupported modes and excess in-flight work fail explicitly.
- Enqueueing reserves simulated work without advancing the scheduler clock.
  Consuming a future advances the sole clock to its completion exactly once.
  Re-reading a future is idempotent. Exceptions cannot become valid tokens.
- Each layer belongs to one stage, including uneven splits. Only the last
  stage contains the LM-head term. Stage rank groups are disjoint and cover
  the selected tensor and pipeline world. Captured stage outcomes join to
  their parent step and preserve request identity.
- The dense Llama/Qwen declaration carries hidden states and residuals at
  each boundary. For divisible eager tensors each TP lane sends its shard to
  the matching lane, followed by the receiver's TP all-gathers. The summed
  forward payload is twice tokens times hidden width times dtype bytes per
  boundary. Unsupported transfer layouts are refused rather than guessed.
- A stage cannot begin its next batch before its prior outgoing send has
  completed. Computation on distinct pipeline stages may overlap. No stage
  computes before its incoming activations are available.
- The existing execution graph and device runtime own service and completion.
  Backend outcomes are read-only projections, not a second timing calendar.
  This first path uses a declared coarse runtime; it does not qualify an AMD
  peer profile, packet-level CX6 accuracy or detailed PP request attribution.

Any violated fatal guard voids the study. Structural guard counts are never
added to the behavioral score. No result is pre-registered by rewriting its
history after observation.

## Numerical experiment

Vary PP width over 2 and 4, TP width over 1 and 2, and declared link rate over
100 and 200 Gb/s. Use fixed dense geometry, fixed token IDs, two output tokens
or more, and at least two independent requests. Include one uneven layer
split and a prompt that requires multiple prefill chunks. Exercise TP4 x PP2
as an additional eight-rank smoke. The real-engine driver uses a small local
Llama configuration with fabricated output tokens, no GPU and no weights.

Before inspecting each result, the lower bound is its causal compute chain
plus mandatory communication serialization on that chain. An independent
serial schedule of the same work is an upper envelope for pipeline overlap.
For an isolated boundary, doubling link rate halves its serialization term
to integer picosecond rounding; any fixed service stays fixed. For a fixed
workload, reducing only link bandwidth cannot improve completion time. A
network delay must reach the returned future and the reported time to first
token (TTFT) or time per output token (TPOT).

For homogeneous compute-only stages of service C and M independent batches,
the reference completion envelope is (P + M - 1) * C, compared with the
serial M * P * C. Adding an outgoing send of service H holds its source
stage until that send completes; compare against an independent recurrence
start[m,p] = max(arrival[m,p], previous_source_release[p]). This recurrence
is an exact-oracle class separate from behavioral parameter relations.

The report cites this expectations commit, publishes the actual axes and
observed TTFT/TPOT, identifies every unsupported scope by an owning task,
and distinguishes engine execution from transcribed/unit inputs. Full lint,
unit tests and documentation-format checks precede publication.
