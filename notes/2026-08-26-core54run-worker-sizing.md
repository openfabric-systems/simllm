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
  `$SIMLLM_CORE54_RUN_ROOT` and are not tracked. Dispatch maps that variable to
  the requested external `wave-runs/core54run` directory.

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

## Completion accounting

- Final scored verdict: `SCORABLE_HELD_OUT_REFUTED_MTP_BLOCKED`; maximum
  priced held-out error 69.20%, with MTP unpriced on COMP-72.
- Scored run: `$SIMLLM_CORE54_RUN_ROOT/attempt-5`; post-score no-fit,
  no-anchor-value and no-score binding qualification:
  `$SIMLLM_CORE54_RUN_ROOT/attempt-6-binding-qualification`.
- Preserved bring-up attempts: attempts 1 through 4, each terminated before a
  held-out score was written.
- Publication payload: 64 KiB compact JSON, 33 KiB PDF and 237 KiB PNG in the
  study directory. Request traces and packet intermediates remain external.
- Actual tracked change against the requested base before generated binary and
  compact-result accounting: approximately 3,250 insertions and 120 deletions
  across freeze, implementation, tests, study and literal registry updates.
- The reusable implementation exceeded the initial range because the scored
  path needed process-safe content-addressed provider transfer, tokenizer-free
  pretokenized scheduler setup, an explicit audited dimension override, exact
  token-budget qualification, a no-rescore binding mode and compact publishing.
  Tests remained inside the expected range; documentation and freeze material
  remained near the expected range.
- Final task movement: CORE-57 closed; CORE-54 and CORE-56 remain open. New
  residuals are CORE-59, COMP-74 and SGL-38. Existing COMP-72, SGL-36 and
  TRAF-64 retain their non-duplicated scopes.
