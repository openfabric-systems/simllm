# MiniMax-M2.5 expert-parallel scaling expectations

These expectations freeze the maintainer-directed scaling study: run
MiniMax-M2.5 across expert-parallel widths, hence across node counts, so
that the network mechanism our stack prices and the external planner's
does not becomes visible where it actually dominates. They are committed
before any implementation exists.

The Qwen3-32B matched-seam study measured a packet-to-unpriced step ratio
of 1.042715399805 because its only cross-fabric traffic was a modest KV
handoff. The probe of the external planner shows that at expert-parallel
widths of 32 and above, expert dispatch is 72 to 84 percent of its decode
step, and that it prices dispatch with no fan-in term of any kind.

## Frozen external facts, established by the executed probe

On h200_sxm with backend trtllm 1.3.0rc10, their MoEDispatch never
consults a TRT-LLM all-to-all table; that table ships only for gb200.
H200 is SM90 and falls through to NCCL delegation, pricing dispatch as
one half-precision all-gather plus one half-precision reduce-scatter
against a NCCL 2.26.2 table whose measured ranks are 2, 4 and 8 only.
Above eight ranks it uses effective_ranks = min(E, 8) and multiplies by
F(E) = ((E-1)/E) * (8/7) * (BW(8)/BW(E)), with BW(8) = 450000000000 and
BW(E>8) = 50000000000 bytes per second. Their dispatch ignores topk and
prices a half-precision buffer proportional to tokens * hidden * E, not
routed payload proportional to tokens * topk * hidden.

Their live decode sweep at a 256-GPU budget, MTP nextn 3, isl 256,
local batch 1 per attention-DP rank:

| EP | decode step ms | dispatch ms | dispatch share |
|---:|---:|---:|---:|
| 8 | 13.984132942232176 | 1.92205 | 13.744506062262868% |
| 32 | 27.51711787974335 | 19.82220267857143 | 72.03589694676378% |
| 128 | 44.86945704576469 | 36.77934174107143 | 81.96966079522261% |
| 256 | 61.028924458726934 | 51.39544921875 | 84.21490248203223% |

EP 1 is illegal at this budget: 216.83251190185547 GiB per GPU against
141 GiB capacity.

## Fatal guards

- FG-1 no SimLLM-authored timing model in the matched arm: every compute
  duration resolves from the imported measured database, and every
  external adjustment applied is declared in the tracked adjustment
  table with its source file and line. Our own roofline provider must be
  provably bypassed. This carries forward the corrected specification
  from the matched-seam wave, not the mis-specified original.
- FG-2 evidence class: measured compute carries MEASURED-EXTERNAL;
  packet-derived terms carry SIM-DERIVED; the two are never merged into
  one label.
- FG-3 MTP declared: nextn is set explicitly to 3 for the faithful arm.
  Any no-MTP arm is separately labelled. A run that leaves nextn at its
  zero default while claiming faithfulness is void.
- FG-4 traffic-model disclosure: their dispatch abstraction (a
  half-precision all-gather plus reduce-scatter over tokens * hidden * E
  bytes) and our packet traffic model (routed expert payload) are
  different abstractions. Every table and figure states both. A run that
  presents them as equivalent traffic is void.
- FG-5 subset labelling: any expert-parallel width whose packet arm is
  sampled rather than fully simulated is labelled as sampled in the
  record, every table row and every figure point, with the sampling rule
  stated. An unlabelled sampled point voids the run.
- FG-6 determinism: every scored quantity reproduces bit-for-bit across
  two full evaluations in fresh processes, wall time excluded by name.
- FG-7 chronology.

## Family E: reproducing their dispatch pricing (exact, scored)

Our reimplementation of their NCCL delegation, including the rank
extrapolation and its fixed bandwidth tiers, reproduces their published
dispatch values bit-equally at expert-parallel widths 8, 32, 128 and 256,
compared as IEEE-754 hex against values obtained live from the pinned
sdk and frozen in the study configuration before the implementation
exists. Four scored cells.

## Family C: reproducing their decode step (scored)

Our composition over the imported database reproduces their published
decode step at each of the four legal widths with a quotient in
[0.98, 1.02]. Four scored cells. This is the matched seam at MoE scale,
and it is predicted to hold because both sides are then the same
arithmetic over the same measured rows plus the same declared
adjustments.

## Family N: the network mechanism against expert-parallel width (scored)

For each simulated width, the packet arm prices the same dispatch and
combine collective that their arm prices, at the same operating point,
and the study publishes the ratio of our packet-priced step to their
unpriced-network step.

- N1 monotone direction: the ratio is non-decreasing in expert-parallel
  width across the simulated widths. A non-monotone point is published
  as a refutation with its mechanism named.
- N2 separation grows: the ratio at the widest simulated width is at
  least 1.25, that is, at least a quarter of a step of network cost that
  their pricing does not carry. The Qwen workload measured 1.0427 with
  only a KV handoff on the fabric; expert dispatch at 72 to 84 percent
  of their step is expected to separate far harder. If the widest
  measured ratio falls below 1.25, that is published as a refutation of
  this expectation, the band is not widened, and the measured value
  stands as the result.
- N3 fan-in attribution: at the widest simulated width the study
  publishes the receiver ingress occupancy and the maximum simultaneous
  senders per receiver, so the mechanism is named by measurement rather
  than asserted.

## Family W: wall time (scored, generous)

The complete sweep, both arms, all scored families and the figure
complete in at most 3600 s, machine disclosed. Any width requiring
sampling under FG-5 counts its sampling budget inside this bound.

## The figure

Two panels in the established grammar. Left: their decode step and our
packet-priced step against expert-parallel width, with the dispatch
share annotated. Right: the ratio against expert-parallel width, with
the Qwen3-32B value 1.042715399805 marked as the single-workload
reference point so the reader sees the mechanism growing from the
earlier study's regime into this one. Sampled widths are marked
distinctly. Series named AIConfigurator and SimLLM.

## Closure

A full pass establishes that at expert-parallel scale the term their
planner class does not price becomes the dominant term of the step, and
quantifies it on their own timing base. It does not validate either
stack against hardware, does not import another system or version, and
does not claim their measured compute rows are wrong. Scored families
are E, C, N and W, in their classes, never summed.
