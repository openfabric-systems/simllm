# CORE-54 scored flagship worker sizing

## CORE-54 scored run

- As-of commit: `c41065b709dc600055b4aafe73222557ef9ffdeb`.
- Scope: preregister the scored DeepSeek-V3 deployment run, configure separate
  SGLang prefill and decode experiments, execute load sweeps through the landed
  placement, session and content-addressed pricing paths, fit only calibration
  anchors, score the priced held-out anchors once, propagate intervals, render
  the publication figure, and publish the literal verdict.
- Assumptions: the three dependency merges at HEAD remain the runtime and
  evidence authority; the candidate record digest remains
  `ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52`;
  the largest faithful harness scale can be established without downloading
  model weights; bulk run artifacts remain outside Git.
- Exclusions: no model-weight download, no MTP price imputation, no widening of
  physical envelopes, no simultaneous 4-node prefill plus 9-node decode claim
  as a 96-GPU disclosure configuration, no remote dispatch, no deletion, and
  no README prose outside mechanical task-progress and open-count cells.
- Owner: CORE-54 Codex worker on `codex/core54run_flagship` in worktree
  `core54run`.
- Dependencies: merged CORE-53 `pd_session_kernel_cycle_v1` binding, merged
  SGL-33 `SglangDisaggregatedSession`, merged candidate record and pricing
  projections, `disaggregated_target_topology_v1`, and the frozen
  `deployment_curve_v1` anchor split and axis contract.
- First reviewable slice: expectations-only scored-run freeze containing the
  physical envelopes, tunable list, load grid, pre-fit bands, separate
  experiment mapping, stable cross-run identity field set, and one-shot held-out
  decision rule.

### Expected files

- Created: a scored configuration, expectations-only freeze, external-run
  manifest or tracked digest summary, published result payload, flagship PDF
  and PNG if repository policy permits publication copies, and the final study
  report.
- Modified: the deployment-curve runner, plotting and scoring helpers only as
  required by the landed SGLang and lookup-pricing paths; focused tests; owning
  registry documents; mechanical README task-progress and open-count cells if
  CORE-54 changes state.
- Bulk evidence: session work directories, per-request traces, intermediate
  records and reproducibility logs live under
  `/data3/yifeng/simllm-dev/wave-runs/core54run/` and are not tracked.

### Expected handwritten line ranges

- Production or reusable study code: 350 to 850 lines.
- Tests and compact fixtures: 300 to 700 lines.
- Freeze, configuration, study, registry and figure documentation: 900 to
  1,700 lines.
- Mechanically generated records, fitted output, score payloads, manifests and
  figure binaries: listed in completion accounting but counted as zero
  handwritten lines.

### Confidence and uncertainty

- Confidence: medium before the landed session and pricing APIs are exercised
  together.
- Dominant uncertainty: the largest faithful executable rank mapping supported
  by the local SGLang harness and how its conserved scaling affects the frozen
  pre-fit bands. The absent MTP candidate cell is already known and remains a
  named COMP-72 blocker rather than a sizing uncertainty.
- External work not represented by line counts: COMP-72 resumable Merlin MTP
  execution and any future promotion of the candidate pricing record.
