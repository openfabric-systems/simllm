# SGLang worker seam v1 frozen expectations

This file and `expectations.json` are the expectations-only freeze for the
SGL-12 sampled-count fix in the SGLang worker seam. They precede the
implementation of that fix, the harness that measures it, and every run. No
value below was read from a simllm execution: every predicted number is
derived by hand from the closed-form model stated here, and
`run_study.py --check-only` re-derives all of them from the frozen row shapes
and physical constants so an arithmetic slip in the literals fails before the
freeze rather than after the measurement.

## The defect being priced

`SglStepTranslator.translate` emits a `StepRecord` with `num_sampled` and
`sampled_request_ids` both absent. `simllm.core.completion.sampled_request_ids`
reads an absent `num_sampled` as "every scheduled request produced a token",
so a mid-prompt chunked-prefill row counts as a token-producing row. The
consequence is not a warning or an exception: the `CompletionReducer` closes
the request's first-token interval at the first extend step, and the reported
TTFT is the completion of a step at which SGLang produced no token for that
request. The same absence prices the LM head for every scheduled row in
`simllm.compute.step_kernel`, whether or not the row samples.

The fix populates both fields from the rows for which SGLang consumes a
generated token, keeps the absent-field record as an explicitly selected
compatibility path, and leaves every other record field untouched.

## Transcribed SGLang rule, pinned commit 8f2a3ad

The count is not modeled. It is transcribed from the scheduler code that
consumes the worker's `next_token_ids`, at the pinned commit:

- `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`,
  `process_batch_result_prefill`: a row is skipped when
  `(req.finished() and req.inflight_middle_chunks <= 0) or req.is_retracted`;
  otherwise the token is appended only inside `if req.inflight_middle_chunks
  <= 0:`, and the `else` branch decrements `inflight_middle_chunks` without
  appending. An extend or mixed row therefore consumes a generated token
  exactly when `inflight_middle_chunks <= 0 and not finished() and not
  is_retracted`.
- The same file, `process_batch_result_decode`: with overlap scheduling off,
  which this adapter requires, every decode row runs
  `req.output_ids.extend(next_token_id)`. Every row of a decode batch
  consumes a token.
- `python/sglang/srt/managers/scheduler.py`: `self.chunked_req
  .inflight_middle_chunks += 1` runs immediately before `ScheduleBatch
  .init_new`, so the mid-prompt request already carries a positive counter at
  forward time; and `python/sglang/srt/managers/schedule_batch.py`
  `prepare_for_extend` sets `req.is_retracted = False` for every admitted row,
  so a resumed prefill after retraction is a token-producing row again.
- A MIXED batch appends its running decode requests after the prefill rows and
  lists them in `decoding_reqs`; they reach the same prefill result path with
  `inflight_middle_chunks == 0` and therefore consume a token.
- A radix hit changes `Req.cached_tokens` and the chunk sizes, never this
  predicate.

This freeze asserts that rule as the specification. The implementation reads
`inflight_middle_chunks`, `finished()` and `is_retracted` with `getattr` at
batch-observation time, which is the same non-overlap iteration in which the
scheduler later applies the rule.

## Fixture and metric chain

One rank, one framework-neutral chain, no backend binary and no SGLang import:

```
stub ScheduleBatch -> observe_schedule_batch -> SglStepTranslator
  -> StepRecord -> ObservedStepLowerer (serial fallback) -> ExecutionGraph
  -> CoarseDeviceRuntime -> CompletionEvent -> CompletionReducer
  -> StepResult.request_metrics -> TTFT and TPOT
```

The stub batches carry the pinned SGLang attribute names, so the transcribed
rule above is exercised from batch observation through to TTFT. The live
SGLang scheduler is not in this loop; see the closing section.

Fixture constants, frozen in `expectations.json`:

- `ModelDims`: 32 layers, hidden 4096, intermediate 14336, 32 heads, 8 KV
  heads, head size 128, vocab 128256, 2 activation bytes. This is the
  per-rank geometry of a Llama-3.1-8B-shaped dense model, chosen so the
  predicted magnitudes can be checked against a real deployment.
- Derived: attention parameters 1,342,177,280; MLP parameters 5,637,144,576;
  weight bytes 13,958,643,712; LM head bytes 1,050,673,152; base step bytes
  15,009,316,864; KV bytes per context token 131,072.
- `RooflineProvider(efficiency=0.5)` without the layer breakdown, peak
  2.0e15 FLOP/s, `HostInitiationModel.ideal()`, `tp_ranks=(0,)`, so the
  lowered graph is 32 serial compute operations on one rank with no
  collective and no host initiation term.
- Step duration model, exactly as the code composes it: the roofline picks
  `max(flops / (peak * efficiency), bytes / (bandwidth * efficiency))`,
  truncates to whole picoseconds, the serial lowerer with no provider layer
  breakdown floors that to `per_layer_ns = duration_ps // (32 * 1000)` and
  repeats it per layer, and the runtime serializes the 32 operations. The
  step latency is therefore `32000 * (duration_ps // 32000)`.

## Physical sanity before precision

Stated before any measurement:

- Floor. Every step in this model streams the 13,958,643,712 weight bytes and
  the 1,050,673,152 LM-head bytes once. At the fast bandwidth (2.0e12 bytes/s
  at 0.5 efficiency, i.e. 1.0e12 effective) no step can complete faster than
  15,009,316,864 ps, i.e. 15.01 ms. No step may be below that floor.
- Ceiling. The largest scheduled context in the sweep is 1,205 tokens, so no
  fast-bandwidth step may exceed 15,009,316,864 + 1,205 * 131,072 =
  15,167,258,624 ps, i.e. 15.17 ms. The slow cells double both bounds.
- Cross-check against the real system. 15.14 ms for the decode step is 66
  tokens per second for an 8B model at an effective 1.0 TB/s. A real 8B bf16
  model reads about 15 GB per decode token, so a 3.35 TB/s H100 at
  60 to 80 percent achieved bandwidth lands at 5 to 8 ms, i.e. 130 to 200
  tokens per second. The fixture is slower in the same proportion as its
  effective bandwidth is lower. It is the right order of magnitude and it is
  not the 50,000 tokens per second that a mis-scaled model would produce.
- The 600-token prefill chunk needs 8.47 ms of compute against 15.09 ms of
  weight streaming, so the roofline reports it memory-bound. A real prefill
  of that size is compute-bound because weight reads overlap arithmetic
  across the batch; the `max()` roofline is coarse there. That coarseness is
  a property of the accepted compute model, it is identical in both arms of
  this study, and it does not touch the attribution being measured.

## Parameter sweep

Three parameters, twelve cells, two arms per cell:

1. Batch composition: `solo` (the chunked request alone), `mixed` (a running
   decode companion `D` in every step, the MIXED-batch case), `paired` (a
   second prefill `S` that completes its whole 100-token prompt in the first
   step, so the step has one sampling and one non-sampling prefill row).
2. Chunk count for request `R`, whose prompt is 1,000 tokens: 2 chunks
   (600 then 400) or 3 chunks (400, 300, 300). One decode step follows the
   last chunk in every cell.
3. Memory bandwidth: 2.0e12 bytes/s (`fast`) or 1.0e12 bytes/s (`slow`).

Arms: `fixed` populates `num_sampled` and `sampled_request_ids`; `compat`
selects the explicit compatibility path in which both fields stay absent.
The two arms differ in nothing else, and the harness builds one batch stream
per cell and replays it through both.

The `paired` composition is the case that forces the identity list: its first
step has `num_sampled == 1` with an empty scheduled decode set, which
`simllm.core.completion.sampled_request_ids` refuses as ambiguous unless
`sampled_request_ids` is present. A fix that populated only the count would
raise there.

## Frozen predictions

`expectations.json` carries, per cell: the row shapes, the byte count of each
step, its roofline memory and compute picoseconds, its quantization margin,
its step latency, the cumulative step completion times, and for request `R`
the sampling steps, TTFT, token count and TPOT in each arm, plus the same for
the control request. Two examples, both fast bandwidth:

| cell | step latencies (ps) | compat TTFT | fixed TTFT | error |
|---|---|---:|---:|---:|
| solo-2chunk-fast | 15,087,936,000 / 15,140,384,000 / 15,140,512,000 | 15,087,936,000 | 30,228,320,000 | 15,140,384,000 |
| solo-3chunk-fast | 15,061,728,000 / 15,101,056,000 / 15,140,384,000 / 15,140,512,000 | 15,061,728,000 | 45,303,168,000 | 30,241,440,000 |

Expected shapes, stated before the run:

- The compatibility arm reports `R`'s TTFT at the completion of the first
  extend step in every cell, which is one step early with 2 chunks and two
  steps early with 3 chunks. The error is not a small bias: in the solo cells
  the compatibility arm reports 49.9 percent of the true TTFT with 2 chunks
  and 33.2 percent with 3 chunks.
- The fixed arm reports `R`'s TTFT at the completion of the step carrying the
  last prompt chunk.
- Halving the bandwidth doubles every memory-bound step, so every predicted
  time in a `slow` cell is its `fast` partner doubled up to the 32,000 ps
  quantization, i.e. the ratio lies in [2.0, 2.000005]. This is a derived
  consequence of the per-step predictions, not an independent claim.
- The control requests `D` and `S` sample in both arms, so their TTFT is
  identical in both arms and equal to the first step completion.

## Scored relations and the pre-freeze entailment answer

The scored basis is chosen so that no scored row is implied by another scored
row or by a fatal guard.

**E1, 42 rows.** The step latency of every step of every cell, in
picoseconds, exactly as frozen. Twenty one steps per bandwidth, both
bandwidths. The two arms must report the same value (guard G1), so each value
is scored once.

*Can this fail?* Yes. It is an absolute prediction of the roofline selection,
the picosecond truncation, the per-layer nanosecond floor, the 32-way
repetition and the serial runtime composition, from first principles. Any
error in the frozen geometry arithmetic, in the quantization rule, or in the
graph composition moves it. It is not implied by any guard: the guards fix
that the arms agree and that the step is memory-bound, not what the value is.

**E2, 24 rows.** Request `R`'s TTFT in each arm of each cell, in
picoseconds, exactly as frozen.

*Can this fail?* Yes. Given E1 the magnitudes of the candidate completions
are known, but which completion the reducer attributes the first token to is
exactly the behavior under test. In the fixed arm the value can land on any
of the 3 or 4 step completions, and a wrong sampled-row rule lands it on the
first. In the compatibility arm it can move if the fix leaks into the
compatibility path. The rows within one arm are correlated: they test one
rule twelve times under different compositions, not twelve independent
mechanisms.

**E3, 16 rows.** The control request's TTFT (`D` in the four mixed cells,
`S` in the four paired cells) in each arm, exactly as frozen.

*Can this fail?* Yes, and in the opposite direction from E2: a rule that
classified MIXED decode rows or a completing prefill row as non-sampling
would move these values later or drop the metric entirely. These rows are
what keeps the fix from being "report fewer tokens" instead of "report the
right tokens".

**Not scored, because they are entailed.** Given E1, E2, E3 and the fixture's
step composition, the following are arithmetic consequences and are reported
as conformance findings only, fatal if violated: `R`'s token count per arm,
`R`'s TPOT per arm, the sampling-step tuples, the TTFT error per cell, and
the fast-to-slow ratios. The 2.0 bandwidth relation in particular is entailed
by the per-step predictions and carries no independent risk.

## Fatal guards, unscored

A violation of any guard voids the run for the purpose of closing anything.
No pass fraction over these is ever reported.

- **G1.** In every cell the two arms report identical per-step latencies. The
  fixture is memory-bound in both arms, and `num_sampled` reaches only the
  LM-head FLOP term, so the sampled count cannot move a step latency here.
- **G2.** Every step is memory-bound: its roofline compute picoseconds are
  strictly below its memory picoseconds, computed from the frozen shapes. The
  smallest margin in the sweep is a factor of 1.53.
- **G3.** Every step's roofline duration is at least 2 ps away from a
  32,000 ps boundary, so the picosecond truncation inside the provider cannot
  move a predicted step latency. The smallest frozen margin is 1,536 ps.
- **G4.** No compatibility-arm record carries `num_sampled` or
  `sampled_request_ids` in its canonical JSON, and every fixed-arm record
  carries both. The compatibility record stream is byte-identical to the
  accepted baseline; the tracked LF fixture under `tests/fixtures/sglang`
  holds that baseline and a pytest, not only this harness, enforces it.
- **G5.** Every fixed-arm record whose sampled count is a strict nonempty
  subset of its scheduled rows carries `sampled_request_ids`. This is
  by construction and is what keeps `sampled_request_ids` from raising.
- **G6.** The chain raises nothing. `CompletionReducer` validates interval
  attribution conservation, execution-graph structure and completion-event
  projection on every step; any raised error voids the run.
- **G7.** Each cell consumes exactly its frozen number of steps in both arms,
  and each arm starts from a fresh clock, runtime and reducer at time zero.

## Evidence classes and the genuine-risk denominator

- Exact-oracle rows, scored: 82 (E1 42, E2 24, E3 16). This is the genuine
  risk denominator of the study.
- Fatal guards: 7 named predicates, never scored, never reported as a
  fraction.
- Entailed conformance findings: token counts, TPOT, sampling tuples, TTFT
  errors and bandwidth ratios. Reported, never added to the scored total.
- Structural evidence, reported separately: the import-free unit tests for the
  transcribed rule, the compatibility byte lock, the replay token source and
  the configuration reset.

Counts from these classes are never summed into one headline.

## What this study does not demonstrate

- It does not run a live SGLang scheduler. The batch observation is exercised
  against stubs that carry the pinned attribute names, and the rule is
  transcribed from the pinned source. Observed agreement with a live
  scheduler is a separate claim and is not made here.
- It does not exercise the pre-play replay token source through TTFT. The
  replay path is a separate mechanism landing on the same branch with its own
  import-free tests and its identity off-path proof; PLAY-7's in-process live
  smoke is explicitly out of scope for this branch.
- It does not touch the vLLM adapter, the shared lowerer, the sink or the
  reducer. Every framework-neutral component in the chain above is used
  unchanged.

## Closure map

- SGL-12 clause "source and populate exact `StepRecord.num_sampled` at the
  worker seam": E1 through E3 plus the unit tests.
- SGL-12 clause "distinguish a mid-prompt extend row from the extend step
  that reaches `origin_input_ids`, including radix hits, retracted prefills
  and MIXED batches": the sweep's compositions plus the import-free tests for
  radix hits and retraction resume.
- SGL-12 clause "prove the count matches the rows for which SGLang consumes a
  generated token": transcription evidence plus stub tests. Observed live
  agreement is not demonstrated and moves to a residual ID.
- SGL-12 clause "keep the absent field as the explicit compatibility path":
  G4 and the compatibility arm.
