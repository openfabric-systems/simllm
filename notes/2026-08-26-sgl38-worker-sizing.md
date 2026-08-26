# SGL-38 decode request shape worker sizing

## SGL-38 remote KV projection

- As-of commit: `61ef185cac5e381998606bcfcdebaa783c085a33`.
- Scope: carry the immutable driver-level remote KV prefix length through the
  SGLang disaggregated-session join into the decode scheduler's
  `KernelRequestShape`, without fabricating KV tensors or changing scheduling
  and batching authority.
- Assumptions: the merged kernel-cycle candidate record remains authoritative;
  the accepted standard-decode shape has prior KV length 2000; and the
  preregistered CORE-58 stable identity field set is the comparison authority
  across repetitions.
- Exclusions: pricing mechanisms, constant envelopes, the
  `deployment_curve_v1` study, scored flagship execution, model-weight
  downloads, remote dispatch, deletion, and README prose beyond mechanical
  task progress or open-count cells.
- Owner: SGL-38 Codex worker on `codex/sgl38_decode_shape` in worktree `sgl38`.
- Dependencies: the merged SGLang driver-level disaggregated session, the
  `pd_session_kernel_cycle_v1` binding, and candidate record digest
  `ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52`.
- First reviewable slice: an expectations-only freeze for the exact candidate
  key, comparator-miss count, stable identity field set, and feature-disabled
  byte identity, before any scored comparison.

### Expected files

- Created: this sizing note and, if the existing test fixtures do not already
  provide a suitable authority, one compact expectations-only fixture.
- Modified: the SGLang session join or pump at the narrow handoff-to-shape seam,
  focused tests, the SGLang registry entry, and mechanical task-ledger or
  README cells only if literal closure changes them.
- Bulk evidence: any untracked diagnostic output stays under
  `$SIMLLM_SGL38_RUN_ROOT`; local dispatch maps that variable to the requested
  external `wave-runs/sgl38` directory.

### Expected handwritten line ranges

- Production code: 20 to 90 lines.
- Focused tests and compact fixtures: 100 to 260 lines.
- Sizing, freeze, proof, and literal registry updates: 60 to 180 lines.

### Confidence and uncertainty

- Confidence: medium-high because the defect is localized to an existing
  driver-level join and the scheduler shape is already content-addressed.
- Dominant uncertainty: whether the remote prefix is represented on the
  immutable handoff object or must be added to its internal join projection
  without affecting feature-disabled serialization.
- External work not represented by line counts: the later integrator-dispatched
  scored flagship rerun and the parallel CORE-59 pricing-mechanism work.
