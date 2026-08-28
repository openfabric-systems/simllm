# CORE-63 clean repetition worker sizing

Date: 2026-08-28

## Scope

This worker will reproduce the CORE-63 residency derivation under a clean,
field-addressed access protocol. The work includes the protocol and reader,
the frozen expectation and preservation manifests, the clean derivation
artifacts, the registry update, and verification evidence.

## Expected change size

- Small protocol and access-reader implementation with focused tests.
- Small frozen input, expectation, and preservation manifests.
- Medium clean evidence report and machine-readable companion artifacts.
- Small mechanical registry and task-progress updates.

## Portable paths

- Repository root: `<repo>`
- CORE-63 registry: `<repo>/docs/modules/core.md`
- Merged void study: `<repo>/examples/deployment_curve_v1/core63_calibration_result.md`
- Clean study area: `<repo>/examples/deployment_curve_v1/`
- Bulk scratch and generated intermediates: `<wave-runs>/core63c/`

No generated bulk data will be placed in the repository. Local absolute paths
will not be recorded in committed evidence.

## Isolation

This worker will not modify `simllm/deploy`, `worktrees/p2loggopsim`, any
`codex/deploy_p2_*` branch, or `nvcompare` work. No model weights or web pages
will be downloaded.

## Access and verification budget

The protected records will be accessed only after the expectations-only
protocol and freeze commits exist. Each permitted field access will be logged
contemporaneously with byte accounting. Whole-file selectors will be rejected,
and the held-out MTP value will remain unread and unscored.

Each commit will be checked with Ruff and the full pytest suite in the
worktree Python 3.10 virtual environment. Pytest's direct exit status will be
recorded without a pipe. End-of-line attributes and POSIX text rendering will
also be checked.
