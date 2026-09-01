# TRAF-74 second-capture worker sizing

## Long-flow NV4 incast revalidation

- Date: 2026-09-01.
- As-of commit: `65593131a0448d2b33f51018d5972c918dad3493`.
- Scope: freeze, capture and score a second six-cell A100 NVLink incast
  comparison at degrees one, two and three using larger long-flow rungs that
  leave demonstrated physical margin below the inherited ten percent launch
  skew ceiling. Preserve the complete first result byte-identically, pin the
  merged simultaneous-release domain and its flow policy, publish every
  required hardware ledger and raw observable, and close TRAF-74 only when the
  non-void comparison is literal.
- Assumptions: the merged model at the as-of commit remains the scored
  surrogate; `release_aware_round_robin` is the explicit flow policy; the
  corrected TRAF-70 persistent peer-write producer is reused without change;
  the maintainer's persistent Merlin connection remains healthy; one qualified
  exclusive `a100-hourly` allocation is sufficient.
- Exclusions: no edit under `simllm/backends`, no producer or inherited study
  mutation, no README change, no model weights, no web access, no remote Git
  mutation, no H200, MiniMax, deployment-curve, deploy or serving-band work,
  and no claim for degrees four, eight or sixteen.
- Owner: TRAF-74 worker on `codex/traf74_incast_revalidation`.
- Dependencies: the retained TRAF-69, TRAF-70 and TRAF-72 artifacts; the first
  TRAF-74 freeze and Merlin job `200456`; the scored candidate profile; the
  merged model at the as-of commit; a free paced `a100-hourly` slot after the
  staged TRAF-77 captures; and TRAF-86 remaining unused if a precision
  residual must be registered.
- First reviewable slice: an expectations-only commit containing the larger
  rung choice, physical floor and ceiling, launch-skew margin arithmetic,
  six frozen simulator predictions, aggregate and per-source bands, immutable
  input digests, and focused freeze checks. It precedes the second runner and
  every new hardware observation.

### Expected files

- Create `examples/nvlink_incast_validation_v1/build_expectations_run2.py`,
  `expectations_run2.json` and `expectations_run2.md` for the second freeze.
- Create `examples/nvlink_incast_validation_v1/run_campaign_run2.py`,
  `run_merlin_cell_run2.sbatch`, `score_study_run2.py` and
  `plot_study_run2.py` for the separate second-capture path.
- Create `examples/nvlink_incast_validation_v1/RESULTS_RUN2.md`,
  `results_run2.json`, `comparison_run2.csv` and two run-two figure files for
  the compact publication.
- Create or materially extend up to three focused
  `tests/test_nvlink_incast_validation_run2_*.py` files.
- Modify `docs/modules/traffic.md` only after scoring, either to remove TRAF-74
  or to move any identified residual to the preassigned free TRAF-86.
- Modify this sizing note at completion with actual accounting.
- Keep every currently tracked file under
  `examples/nvlink_incast_validation_v1/` byte-identical; all run-two names are
  additive.

### Expected handwritten line ranges

- Production lines: zero.
- Tests and fixtures: 180 to 420 lines.
- Studies, configuration and documentation: 2,200 to 4,000 lines across the
  run-two generator, capture and scoring path, compact report, registry effect
  and this note.
- Mechanically generated expectations JSON, score JSON, comparison CSV,
  figures, external raw rows, scheduler logs, build artifacts and digest
  inventories are listed evidence and count as zero handwritten lines.

### Confidence and uncertainty

- Confidence: medium.
- Dominant uncertainty: the larger flows must amortize the observed serialized
  PCIe launch cost enough for FG11 while remaining within the producer's
  counter and completion limits, and the endpoint may still expose a real
  model miss after launch skew becomes negligible.
- Hardware and waiting work not expressed by line counts: queue occupancy,
  one exclusive allocation, CUDA compilation, capture duration, persistent SSH
  health, lean evidence retrieval and scheduler accounting.
- Scope-change rule: update this section before continuing if the chosen rungs
  require a producer change, a model interface change, another hardware cell,
  or files outside the bounded families above.

### Completion accounting

- Pending.
