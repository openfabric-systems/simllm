# TRAF-66 finite overlap boundary worker sizing

## TRAF-66 boundary derivation and calibration-only check

- As-of commit: `9ee51a327eaff8a3d09fdf3e39c03791140ba4cf`.
- Scope: capture the pinned SGLang two-child prefill schedule from prologue
  through every per-layer dispatch and combine yield, steady-state interleave,
  and epilogue; derive exposed boundary service from those events and the
  existing component compute and packet services; freeze the form and envelope
  before comparing with the visible 1K calibration residual.
- Assumptions: CORE-60's source-backed EP32 composition and component services
  are immutable inputs; the pinned SGLang commit is the only external schedule
  authority; all boundary arithmetic follows child-stage and event
  conservation without a fitted overlap scale or boundary fraction.
- Exclusions: no held-out numeric access, no scored flagship rerun, no decode
  pricing, no NVLink work, no mutation of prior records, no model-weight
  download, no web access, no deletion, and no README prose beyond permitted
  mechanical progress and open-count cells.
- Owner: TRAF-66 Codex worker on `codex/traf66_overlap_boundary` in worktree
  `traf66`.
- Dependencies: the COMP-75 preregistered source protocol, a committed TRAF-66
  extension naming exact pinned batch-overlap ranges before inspection,
  CORE-60's frozen composition and packet evidence, and calibration-only
  component compute records.
- First reviewable slice: a source-boundary extension followed by an
  expectations-only freeze that states the event-derived boundary form, exact
  conservation identities, envelope, and signed residual movement before any
  visible comparison.

### Expected files

- Created: this sizing note; a TRAF-66 source-boundary extension; compact
  expectations, event ledger, result, and preservation-lock records; a
  calibration-only runner; and focused tests.
- Modified after the expectations freeze: a narrowly scoped reusable boundary
  service helper, the literal TRAF-66 registry entry only if acceptance is
  complete, and permitted mechanical task-progress/open-count cells.
- Preserved: CORE-60, COMP-75, packet evidence, all prior freeze and result
  records, and every scored-run artifact remain byte-identical.
- Bulk evidence: generated diagnostics live under `<TRAF66_RUN_ROOT>/`, bound
  by the task environment to the required external `wave-runs/traf66` root,
  and are not tracked.

### Expected handwritten line ranges

- Production or reusable study code: 80 to 220 lines.
- Focused tests and compact fixtures: 180 to 420 lines.
- Source freeze, expectations, event ledger, result, registry, preservation,
  and handoff documentation: 260 to 620 lines.
- Generated calibration diagnostics and external bulk evidence: counted as
  zero handwritten lines.

### Confidence and uncertainty

- Confidence: medium before the exact pinned yield sequence is transcribed and
  reconciled with CORE-60's component service definitions.
- Dominant uncertainty: whether the source event structure exposes one whole
  boundary phase, a compute-limited remainder, or a different finite prologue
  and epilogue service. The visible residual is not authority for that choice.
- Closure rule: publish the independently derived signed 1K movement and its
  honest residual. Close TRAF-66 only if the registry entry is met literally;
  otherwise register the exact remainder under an available reserved owner
  without consuming TRAF-67 or CORE-61.
