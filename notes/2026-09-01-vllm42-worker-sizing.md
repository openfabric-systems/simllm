# VLLM-42 worker sizing

## Plan as of ffc9bbc

- As-of commit: `ffc9bbc` (`Guard VLLM-42 record access`).
- Scope: replace the refuted batching-service component-band predictor with a
  service-only mechanism derived from independently measured batch service and
  arrival inputs, freeze held-out cells and acceptance bands before comparison,
  run the local successor study, publish separate timing fields and conservation
  evidence, and close VLLM-42 only if every frozen requirement is literal.
- Assumptions: the merged VLLM-41 local serving harness remains reusable; its
  records remain immutable; the independent batch-service measurements already
  exist in tracked inputs; the successor needs no product-runtime behavior
  change; VLLM-50 is free on main if a residual must be registered.
- Exclusions: cluster time, model-weight downloads, NVLink code, `simllm/deploy`,
  H200 collectives, MiniMax work, the `deployment_curve_v1` scored lineage,
  README files, and all backend submodule changes.
- Owner: VLLM-42 worker on `codex/vllm42_service_bands`.
- Dependencies: VLLM-41 merged evidence and harness lineage; independently
  measured batch-service inputs; VLLM-50 availability for an honest residual.
- First reviewable slice: an expectations-only commit containing the predictor
  derivation, signed mechanism claim, load and ratio holdouts, physical bounds,
  and exact scoring rule, with no successor observation or implementation.

## Expected files

Created or materially modified production and study logic:

- `examples/pd_session_batching_service_v1/predictor.py`
- `examples/pd_session_batching_service_v1/freeze_expectations.py`
- `examples/pd_session_batching_service_v1/run_study.py`
- `examples/pd_session_batching_service_v1/publish_results.py`
- a bounded family of helper modules in
  `examples/pd_session_batching_service_v1/` if the inherited harness requires
  one portability or projection adapter

Created or materially modified tests and fixtures:

- `tests/test_pd_session_batching_service_freeze.py`
- `tests/test_pd_session_batching_service_study.py`
- `tests/test_pd_session_batching_service_result.py`
- a bounded family of small deterministic fixtures under
  `tests/fixtures/vllm/` only if the local harness cannot use frozen inputs
  directly

Created or materially modified study, configuration, and documentation files:

- `examples/pd_session_batching_service_v1/EXPECTATIONS.md`
- `examples/pd_session_batching_service_v1/expectations.json`
- `examples/pd_session_batching_service_v1/access_ledger.jsonl`
- `examples/pd_session_batching_service_v1/preservation_manifest.json`
- `examples/pd_session_batching_service_v1/results.json`
- `examples/pd_session_batching_service_v1/RESULTS.md`
- `docs/modules/adapters-vllm.md`
- `docs/task-ledger.json` if registry bookkeeping requires it
- `notes/2026-09-01-vllm42-worker-sizing.md`

Mechanically generated outputs, not counted as handwritten lines:

- frozen expectation matrices emitted by `freeze_expectations.py`
- raw successor cell records under the configured bulk-output root
- published result matrices and access-ledger rows
- preservation hashes and any deterministic summary tables

Read-only preservation set:

- every file under `examples/pd_session_queue_onset_v1/`
- every earlier tracked `pd_session` study record

## Handwritten line ranges

- Production and study logic: 700 to 1,200 lines.
- Tests and hand-authored fixtures: 350 to 650 lines.
- Study configuration and documentation: 450 to 850 lines.

Generated matrices, ledgers, hashes, and result tables have zero handwritten
line count. Runtime configuration consumed only by this study is accounted in
the study and configuration range.

## Confidence and uncertainty

- Confidence: medium.
- Dominant uncertainty: the exact inherited harness seam and whether the
  independent batch-service record can be projected without a small adapter.
- External waiting work: none expected. No hardware, cluster allocation,
  external repository, or network work is in scope.
- Scope-change trigger: update this plan before continuing if product runtime
  code becomes necessary, if more than one inherited interface must change, or
  if the expected handwritten total leaves these ranges.

## Completion accounting

- Completion baseline: the final working tree based on `8825e46`, before the
  held-out and combined publication commit.
- Actual tracked delta: 25 files, 16,272 added lines and 3 deleted lines.
- Handwritten production and study logic: 2,575 lines, above the planned 700 to
  1,200 lines.
- Tests: 704 lines, 54 lines above the planned upper bound.
- Handwritten study configuration and documentation: 154 lines, below the
  planned 450 to 850 lines.
- Mechanically generated expectations, result JSON, and result Markdown:
  12,839 added lines with zero handwritten line count.
- The dominant estimate miss was the inherited virtual harness throughput. The
  sequential runner was valid but too slow for the 48-cell split, so closure
  required the VLLM-41 local one-process-per-cell seam, a strict shard merger,
  and their tests. The reader and final self-contained publisher were also
  larger than anticipated. Generated reports carried the expected study prose,
  leaving fewer separately handwritten documentation lines than estimated.
