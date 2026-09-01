# Decode kernel ladder sizing

- As of commit: `598e124`
- Scope: build and run the vLLM aarch64 GH200 decode kernel ladder for batch
  32 and KV length 2,000, preserve the confirmatory evidence, derive the
  per-layer composition, and publish the resulting standard-decode calibration
  movement under CORE-66 or a fresh CORE owner if registry discovery requires
  one.
- Assumptions: the retained 46-row inventory is sufficient to recover exact
  kernel shapes; the existing GH200 CUDA 12.9, Python 3.11, Torch cu129 and
  vLLM lane remains runnable; dummy weights fit one GH200; short gh-hourly
  cells can be submitted serially; Nsight Systems remains available; optional
  hardware counters may be denied without voiding timing evidence when that
  survivable case is frozen explicitly.
- Exclusions: model-weight downloads, SGLang execution, expert parallel or
  network traffic, tensor parallelism above one, the held-out simulated-MTP
  value, changes under `simllm/deploy`, `worktrees/p2loggopsim`, deploy P2
  branches, the A100 partition, and the fifth scored flagship run.
- Owner: decode kernel ladder worker on `codex/decode_kernel_ladder`.
- Dependencies: CORE-65 retained kernel inventory; CORE-61 GH200 capture
  tooling and environment; Merlin `gh-hourly` allocation; vLLM's native
  aarch64 kernel identities; maintainer git identity already configured.
- First reviewable slice: an expectations-only commit that freezes the
  surviving scratch hypotheses, rung matrix, physical bounds, evidence
  classes, access guards, failure semantics and acceptance tolerances before
  the confirmatory run.

## Expected repository changes

- Create one bounded study family under
  `examples/deployment_curve_v1/core66_decode_kernel_ladder_*` or a dedicated
  `examples/decode_kernel_ladder_v1/` directory: capture runner, analyzer,
  expectation records, result records, preservation manifest and report.
- Modify `docs/modules/core.md`, `docs/task-ledger.json`, and the minimum
  mechanical registry/index surfaces required by their existing contracts.
- Create or modify focused tests under
  `tests/test_deployment_curve_core66_*` or
  `tests/test_decode_kernel_ladder*.py`.
- Modify no framework-agnostic production module unless the study proves that
  a supported calibration interface needs a narrow extension. If that occurs,
  update this sizing note before the edit.

## Handwritten line ranges

- Production: 0 lines expected. Contingency 40 to 140 lines only for a narrow,
  supported calibration-interface extension proven necessary by the study.
- Tests and fixtures: 250 to 550 lines.
- Studies, configuration and documentation: 900 to 1,900 lines across 12 to
  24 files.
- Mechanically generated and zero-counted: raw Nsight reports, exported trace
  databases, CSV timing samples, run logs, rendered figures, digest manifests,
  copied environment inventories and generated result JSON.

## Confidence and uncertainty

- Confidence: low.
- Dominant uncertainty: whether isolated vLLM-native kernels can be invoked at
  every retained shape without importing model weights, and whether the fused
  layer boundary is exposed cleanly enough to capture dense and MoE layers
  separately.
- External and waiting work not represented by line counts: several short
  Merlin queue waits, a few GPU-hours of GH200 service, SSH interruptions,
  possible performance-counter permission denial, and scratch iteration until
  the mega-kernel residual meets the frozen tolerance.

## Completion actuals

- Final commit: `361e7db69dfb2dbc90675659fe73492620062e47`.
- Actual tracked delta from `598e124`: 14 files, 2,495 added lines and one
  deleted line.
- Production: zero lines.
- Tests and fixtures: 448 lines across three files.
- Studies, configuration and documentation: 2,047 added lines and one deleted
  line across eleven files.
- Bulk evidence excluded from handwritten counts: Nsight Systems reports and
  SQLite exports, scheduler logs, timing samples, weight snapshots and the
  local scratch ledger.
- Variance: the initial study ceiling was too low because the final portable
  reproduction surface needed a scheduler hook, native correlation scorer and
  standalone launcher. The pre-publication scope update covered the final
  total. No production interface was needed.

## Scope update before publication commit, 2026-09-01

The portable published harness requires a capture runner, native trace scorer,
scheduler hook and Slurm launcher in addition to the initially expected study
records. The original studies/configuration/documentation ceiling no longer
covers that explicit reproducibility surface. The revised range is 1,800 to
2,600 handwritten study, configuration and documentation lines across 12 to
22 files. The test range becomes 300 to 650 lines. Production remains zero.
Hardware waiting and bulk profile evidence remain outside line counts. The
dominant uncertainty is resolved: all twelve family seams and both full-layer
boundaries execute under vLLM, while direct hardware counters remain denied.
