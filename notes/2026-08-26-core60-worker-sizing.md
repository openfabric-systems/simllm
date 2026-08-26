# CORE-60 prefill composition worker sizing

## CORE-60 composition freeze and implementation

- As-of commit: `03da15515b868951ca63b0cd615dec80109da719`.
- Scope: identify which independently justified EP32 physical contracts explain
  CORE-59's serial prefill-service overcorrection, freeze their expected signed
  effects before comparison, and compose the calibration-only 1K movement
  through the existing placement and traffic authorities without fitting.
- Candidate contracts: per-rank token ownership, FP8 dispatch and BF16 combine
  wire precision, same-destination expert deduplication, and framework-supported
  compute and communication overlap. Each adopted contract requires a pinned
  source or architecture derivation; unsupported contracts remain rejected.
- Assumptions: CORE-59's immutable mechanism freeze and result are the baseline;
  the deployment projection owns EP32 token placement; uniform top-8 routing
  supplies architecture arithmetic only where explicitly frozen; available
  measured compute time, not an anchor fit, bounds any hidden communication.
- Exclusions: no held-out numeric access, no scored flagship rerun, no SGLang
  join or decode pricing changes, no mutation of CORE-59 or first scored-run
  artifacts, no model-weight download, no remote dispatch, no deletion, and no
  README prose beyond permitted mechanical progress and open-count cells.
- Owner: CORE-60 Codex worker on `codex/core60_composition` in worktree
  `core60`.
- Dependencies: CORE-59's frozen EP32 dispatch/combine mechanism and calibration
  output, the pinned SGLang/DeepEP source authority, the deployment projection,
  the existing placement and traffic implementations, and calibration-only
  component compute evidence. SGL-38 remains the sole owner of the decode bind.
- First reviewable slice: an expectations-only freeze that states each adopted
  contract, cites its source or formula, predicts its signed effect before any
  movement is computed, locks historical artifacts, and rejects free scales.

### Expected files

- Created: this sizing note and compact CORE-60 expectations-only JSON and
  Markdown freezes.
- Modified or created after the freeze: a narrowly scoped prefill composition
  helper, calibration-only runner, focused tests, result records, and the
  literal CORE-60 registry entry only if acceptance is complete.
- Preserved: CORE-59's freeze and result plus all nine locked first scored-run
  artifacts remain byte-identical.
- Bulk evidence: generated diagnostics live under the configured
  `wave-runs/core60/` root and are not tracked.

### Expected handwritten line ranges

- Production or reusable study code: 100 to 300 lines.
- Focused tests and compact fixtures: 160 to 420 lines.
- Freeze, sizing, result, registry and handoff documentation: 220 to 520 lines.
- Generated calibration diagnostics and external bulk evidence: counted as
  zero handwritten lines.

### Confidence and uncertainty

- Confidence: medium before the pinned framework source and component timing
  envelopes are reconciled with CORE-59's byte ledger.
- Dominant uncertainty: whether pinned source establishes an overlap schedule
  whose hidden service can be bounded entirely by existing component compute
  evidence. If not, overlap stays rejected rather than acquiring a fitted
  fraction.
- Closure rule: publish the independently justified composed movement and its
  honest residual. Close CORE-60 only if the registry entry is met literally;
  otherwise register the exact remainder under reserved ID TRAF-66 or COMP-75
  according to ownership.
