# MiniMax expert-parallel scaling, second freeze

The first run is void against FG-4 and its headline comparison rests on
a premise my first freeze got wrong. This second freeze states the void,
corrects the premise, and defines the study that actually answers the
maintainer's question. No band that fired is widened.

## Why the first run is void, and what the first freeze got wrong

FG-4 required every table and figure to state both traffic abstractions
and voided a run that presented them as equivalent. The prose did
explain the distinction, but the result tables, the CSV and the figure
footer did not, and the implemented check only compared two constants
and a fixed string. FG-4 true was therefore not earned, and under the
fatal-means-void rule the published nonvoid state is incorrect.

The deeper error is mine. The first freeze said both arms price "the
same dispatch and combine collective". They do not. The external model
prices a DENSE strategy: a half-precision all-gather of every token to
every expert-parallel rank plus a reduce-scatter, with volume growing as
tokens times hidden times expert-parallel width. Our arm priced a SPARSE
strategy: routed payload to top-k experts only, volume independent of
width. Both strategies are real. The dense all-gather plus
reduce-scatter path is the documented general fallback in TensorRT-LLM
itself, and the external source deliberately selects it for SM90 while
keeping separate sparse branches for SM100 and for DeepEP.

So the measured 0.2742607736975033 is a comparison between two different
communication strategies. It is not evidence that the external planner
omits a mechanism, and it is not a precision claim. Nothing in the run
determines which strategy an actual MiniMax H200 deployment selects; the
study holds no runtime trace, engine configuration or kernel trace.

## The corrected design: same strategy, different contention modelling

To answer the question the maintainer actually asked, both arms must
price the SAME traffic, so that the only difference left is contention.

- Arm D-external: the dense all-gather plus reduce-scatter as the
  external planner prices it, through its NCCL table with
  effective_ranks = min(E, 8) and its fixed bandwidth tiers.
- Arm D-packet: the identical dense collective, identical bytes,
  identical rank count and identical placement, priced by packet
  simulation.

The ratio of those two arms isolates contention on equal traffic. That
is the frozen comparison of this study.

The sparse-versus-dense strategy difference becomes a separate, clearly
labelled secondary result. It is reported as a strategy comparison, never
as a precision or omission claim, and it carries an explicit statement
that the study does not know which strategy the real deployment uses.

## Corrected fatal guards

- FG-4 corrected and made checkable: every published table row, CSV row,
  figure series and figure caption naming a step or ratio also names the
  communication strategy and traffic definition of each arm. The guard
  is implemented by inspecting the generated artifacts, not by comparing
  constants. Any artifact that names a comparison without naming both
  strategies voids the run.
- FG-8 new, routing geometry: the packet arm's messages must follow the
  strategy it claims. For a sparse arm, a source may only send to the
  ranks its routed assignments actually reach; all-pairs fluidization of
  fractional bytes is forbidden. The run publishes the realized
  distinct-destination count per source and the realized simultaneous
  sender count per receiver, beside their analytical expectations. For
  the study's parameters the expected distinct destinations per source
  is 255 * (1 - (31/32)^4) = 30.41 and the expected cross-node senders
  per receiver is 248 * (1 - (31/32)^4) = 29.58. A realized fan-in more
  than 1.2 times its analytical expectation voids the arm.
- FG-9 new, combine precision: dispatch and combine precisions are
  declared separately and justified from the represented implementation.
  Symmetric FP8 in both directions is not assumed. Where the represented
  implementation returns BF16 combine, the arm prices BF16 combine.
- FG-10 new, population honesty: a scored packet cell must either
  simulate the full population or carry a measured full-population
  anchor at a smaller width plus a stated extrapolation rule. A cell
  sampled without such an anchor may be published as a diagnostic but
  may not be scored.
- FG-1, FG-2, FG-3, FG-5, FG-6 and FG-7 carry forward unchanged.

## Families

- Family E (dispatch pricing parity) and Family C (decode step parity)
  carry forward unchanged. Family C's result is reported as an
  end-to-end parity check that reuses the dispatch code E validates, not
  as independent confirmation of E.
- Family D, the corrected contention comparison, scored: across the
  simulated widths, the ratio of arm D-packet to arm D-external on
  identical dense traffic. Frozen expectation: the ratio is greater than
  or equal to 1.0 at every width, because the external arm prices no
  contention while the packet arm prices it on the same bytes. A ratio
  below 1.0 at any width is published as a refutation with its mechanism
  named, and is a genuine finding about the external NCCL extrapolation
  being conservative rather than optimistic at that width.
- Family S, the strategy comparison, published and labelled, unscored:
  the sparse arm beside the dense arm at each width, with both strategies
  named in every row, and an explicit statement that the study does not
  establish which strategy the real deployment selects.
- Family W carries forward.

## Closure

A full pass quantifies what contention modelling adds on identical
traffic as expert parallelism widens, and separately documents how much
the choice of communication strategy changes the answer. It does not
determine which strategy any real deployment uses, does not validate
either stack against hardware, and closes no task.
