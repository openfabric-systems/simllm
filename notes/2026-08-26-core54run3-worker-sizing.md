# CORE-54 third scored flagship worker sizing

## CORE-54 third scored run

- As-of commit: `ef7b10a85e1b728984b84e86f924bfdd267d92e8`.
- Scope: preregister the third DeepSeek-V3 scored run, add the clean TRAF-66
  and TRAF-67 overlap-exposure envelope to the second-run pricing model, fit
  the exposed fraction only on calibration anchors, freeze the admissible
  benchmark-bias attenuation layer, execute the unchanged disclosed
  configurations, score the 2K and 4K prefill anchors once, render the
  three-layer publication figure, and publish the literal verdict.
- Assumptions: the merged TRAF-67 head makes the perfect-overlap floor and
  two-child ceiling calibration-clean; the COMP-75 traffic arithmetic, SGL-38
  remote-KV binding, CORE-58 identity surface, MTP-absent ruling, and all
  second-run execution settings remain authoritative and byte-identical.
- Exclusions: no held-out numeric access before the serialized fit, no
  envelope widening, no in-run adjustment, no MTP imputation, no decode
  attenuation, no model-weight download, no web access, no deletion, no remote
  Git mutation, and no README prose outside mechanical task progress and open
  counts.
- Owner: CORE-54 Codex worker on `codex/core54run3_scored` in worktree
  `core54run3`.
- Dependencies: merged TRAF-66 and TRAF-67 boundary evidence, the second scored
  run and its preservation lock, COMP-75 clean composition, SGL-38 exact shape
  binding, the frozen disclosure anchor split, and the maintainer's 2026-08-26
  attenuation policy.
- First reviewable slice: an expectations-only commit containing the physical
  overlap envelope, complete tunable list, every derived refinement and
  rejection, the independently quantified attenuation factor and uncertainty,
  three pre-fit band layers, preservation locks, and the one-shot scoring rule.

### Expected files

- Created before scoring: this sizing note, third-run Markdown and JSON
  expectations, LF attribute pins, and focused freeze tests.
- Created after the freeze commit: a compact run configuration, pure fitting
  and scoring helpers, the scored runner, publication renderer and publisher,
  focused result tests, a content-addressed compact result, PDF and PNG figures,
  and the third-run report.
- Modified after scoring: literal owning registry entries, task-ledger state,
  permitted mechanical README task progress and open counts, and this note's
  completion accounting.
- Preserved: every first-run, second-run, CORE-59, CORE-60, COMP-75, TRAF-66,
  and TRAF-67 freeze, code, result, and figure named by the preservation-lock
  class remains byte-identical.
- Bulk evidence: frozen fits, one-shot scores, live session traces, packet
  intermediates, manifests, and reproducibility diagnostics live under
  `$SIMLLM_CORE54RUN3_RUN_ROOT`; dispatch binds it to the requested external
  `wave-runs/core54run3` root, and none of that bulk is tracked.

### Frozen mechanism and attenuation sizing

- The only new fitted constant is the exposed fraction `f` in the closed
  interval `[0, 1/2]`, with service
  `max(C, P) + f * min(C, P)`. The endpoints are exactly the clean COMP-75
  perfect-overlap floor and clean TRAF-67 two-child ceiling.
- The existing topology-derived locality split is retained: seven same-node
  and 24 fabric peers are derived from the four-node, eight-rank-per-node EP32
  layout and are already priced through the COMP-75 locality machinery. The
  A100 packet candidate is not substituted into the H100 target because that
  would be an unjustified cross-architecture refinement.
- One attenuation factor is admitted for the disclosure's in-distribution
  expert-balance simplification. Its point is the uniform-routing expected
  unique destination-rank incidence divided by top-k eight,
  `939691952959 / 1034504281000`, and its uncertainty is two standard errors
  from the exact hypergeometric indicator covariance over 16,384 tokens. It
  touches all three prefill anchors, so one factor is fewer than the three
  anchors it touches.
- An exact-length packing factor is rejected. Both the disclosure and the
  configured run already pack exact lengths to 16,384 tokens per rank, and no
  independent per-request overhead magnitude exists without using anchor
  values.

### Expected handwritten line ranges

- Reusable study code and renderer: 650 to 1,300 lines.
- Focused freeze, fit, score, chronology, preservation, and publication tests:
  350 to 800 lines.
- Expectations, configuration, report, registry, and completion documentation:
  900 to 1,700 lines.
- Generated compact records and publication binaries are listed in completion
  accounting but counted as zero handwritten lines.

### Confidence and uncertainty

- Confidence: medium-high because the execution path and all runtime bindings
  are inherited from the completed second run; the new scored logic is pure
  exact arithmetic around the clean boundary and attenuation layers.
- Dominant uncertainty: whether the independently frozen routing-balance
  correction places both held-out point predictions within the 5 percent bar
  after the calibration-only fit. No held-out number is used to tune that
  factor or its uncertainty.
- Closure rule: publish all three held-out layers and their propagated bands.
  Move CORE-54 only as far as its literal registry wording permits, retain the
  MTP blocker and registered decode residual, and register the dominant next
  mechanism without widening or adjusting any layer if a scorable point misses.

### Completion accounting

- Freeze commit: `45251494fa7c9dc0b872bf5324619380cf516a7b`.
- Scored-runner commit: `3d13cde15b19d105d91df4986751c03bebdb56b7`.
- Dispatch: one scored attempt under
  `$SIMLLM_CORE54RUN3_RUN_ROOT/attempt-1`, after the serialized fit and before
  any held-out access. No rerun or in-run layer adjustment was performed.
- Outcome: `SCORABLE_HELD_OUT_PASS_MTP_BLOCKED`. The attenuated errors are
  -4.519707 percent at 2K and +3.530310 percent at 4K. The corresponding
  unattenuated errors remain +5.113992 percent and +13.976233 percent.
- MTP and decode: the MTP numeric anchor was not read and remains blocked on
  COMP-72. The standard-decode calibration error remains -59.834128 percent,
  unattenuated and unadjusted.
- Handwritten implementation: 1,751 lines across the reusable tools, runner,
  renderer and publisher. This exceeds the 650 to 1,300 estimate because the
  field-addressed access boundary, live-session conservation evidence and
  full provenance projection remained explicit in the run-specific code.
- Handwritten tests: 772 lines across freeze, mechanism, access chronology,
  publication and byte-lock tests, inside the 350 to 800 estimate.
- Expectations and documentation: 1,041 new lines across the JSON and Markdown
  freeze, report, sizing note and registry additions, inside the 900 to 1,700
  estimate. Generated compact JSON and publication binaries count as zero
  handwritten lines.
- Publication bytes: compact JSON 116 KiB, PDF 36 KiB and PNG 240 KiB. Bulk
  evidence remains external; the full scored result is 408 KiB.
- Preservation: all 33 inherited record locks pass byte for byte. The task
  ledger was inspected and correctly left unchanged because CORE-54 and every
  registered residual remain open. No README open-count cell changed.
- Focused validation: 26 passed and 1 skipped for the freeze, runner and
  publication suites. Final validation is Ruff green and 3,123 passed with 13
  skipped in the full pytest suite, using pytest's direct exit status.
