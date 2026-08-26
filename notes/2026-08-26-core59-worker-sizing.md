# CORE-59 role mechanism worker sizing

## CORE-59 mechanism freeze and implementation

- As-of commit: `61ef185cac5e381998606bcfcdebaa783c085a33`.
- Scope: replace the CORE-54 flagship's shared collective-surcharge residual
  with role- and shape-specific mechanisms identified from component-complete
  calibration evidence, freeze every new mechanism and physical envelope before
  comparison, and implement signed calibration-row movement without reading or
  rescoring held-out anchors.
- Assumptions: the existing placement, fabric, traffic, packet and
  content-addressed candidate paths remain authoritative; EP32 prefill is
  missing per-MoE-layer expert dispatch and combine service that must be priced
  through those paths; decode receives no new mechanism unless
  component-complete decode evidence identifies one independently of SGL-38.
- Exclusions: no SGLang join or shape-key derivation changes, no scored flagship
  rerun, no held-out anchor access, no mutation of the first scored run or its
  refutation artifacts, no model-weight download, no remote dispatch, no file
  deletion, and no README prose beyond permitted mechanical cells.
- Owner: CORE-59 Codex worker on `codex/core59_role_mechanisms` in worktree
  `core59`.
- Dependencies: CORE-54's immutable first scored result, the
  `deployment_curve_v1` calibration split, the content-addressed per-layer
  candidate record, deployment projection `ee154ed5...`, preserved packet
  evidence, and the existing collective and traffic implementations. SGL-38 is
  parallel work and remains outside this branch.
- First reviewable slice: an expectations-only freeze that names each role and
  shape gate, derives every constant and envelope from declared hardware or
  evidence authority, records the expected signed movement of both calibration
  rows, and proves that no held-out values were consulted.

### Expected files

- Created: this sizing note and a compact CORE-59 expectations-only mechanism
  freeze.
- Modified: the reusable flagship pricing or projection helper, focused tests,
  and the literal CORE-59 registry entry only if acceptance is complete.
- Preserved: all first-run configurations, records, score payloads, plots and
  reports remain byte-identical.
- Bulk evidence: any generated diagnostic output lives under
  `$SIMLLM_CORE59_RUN_ROOT` and is not tracked. Dispatch maps that variable to
  the requested external `wave-runs/core59` directory.

### Expected handwritten line ranges

- Production or reusable study code: 80 to 240 lines.
- Focused tests and compact fixtures: 140 to 360 lines.
- Freeze, sizing, registry and handoff documentation: 180 to 420 lines.
- Generated diagnostics and external bulk evidence: counted as zero handwritten
  lines.

### Confidence and uncertainty

- Confidence: medium before component and packet authorities are reconciled.
- Dominant uncertainty: whether the existing packet evidence carries enough
  role and shape identity to price both EP dispatch and combine without a new
  fitted constant. If it does not, the mechanism must remain frozen but
  unimplemented rather than acquire a free residual.
- Decode expectation: zero new decode-side mechanism unless complete decode
  component evidence shows missing service after the measured decode row binds.
  SGL-38 owns that binding and CORE-59 must compose with it.

## Completion accounting

- Frozen mechanism: one EP32 prefill dispatch-and-combine service through the
  existing placement, NVLink and htsim path; zero DP-attention synchronization
  mechanisms; zero standard-decode mechanisms.
- External evidence: point and sensitivity service runs plus calibration-only
  output live under `$SIMLLM_CORE59_RUN_ROOT`. No model weights were downloaded
  or loaded.
- Actual reusable study code: 753 lines across the validator, projection helper
  and compact native runner. This exceeds the initial range because the branch
  includes independent arithmetic validation, historical hash locking, strict
  shape gates and a reproducible two-arm htsim driver.
- Actual focused tests: 267 lines, inside the initial range.
- Freeze and result records: 591 JSON and Markdown lines. The compact result
  JSON is generated accounting rather than handwritten model logic; registry,
  EOL and mechanical progress edits add 42 lines before replacements.
- Calibration movement: EP32 prefill decreases 70.8964 percent and moves from
  66.7072 percent high to 51.4821 percent low; EP72 standard decode remains
  exactly unchanged at 59.8341 percent low.
- Final task movement: CORE-59 meets its literal acceptance and closes. CORE-60
  owns the newly isolated prefill service-composition overcorrection. SGL-38
  remains the sole owner of the decode bind.
