# COMP-74 worker sizing

- Date: `2026-08-27`
- Scope: replace the CORE-54 zero-width DeepSeek candidate distribution
  placeholder with per-key uncertainty derived from independent retained
  repetitions, then propagate that term through the existing flagship interval
  engine without changing scored point predictions or verdicts.
- Initial size: medium.
- Expected implementation: one expectations-only freeze and field-addressed
  reader, one distribution estimator integrated with the existing interval
  path, one independently generated study artifact, focused tests, and the
  mechanical compute-registry update.
- Expected risk: moderate. The main risks are accidental data inspection before
  the freeze, pooling observations across priced keys, changing preservation-
  locked records, and allowing display-band movement to alter a scored verdict.
- Validation budget: targeted tests while iterating, then `ruff` and the full
  `pytest` suite with their direct exit statuses for every local commit.
- Scope exclusions: traffic work, NVLink profiles, Merlin submissions, model
  downloads, and all published-throughput inputs.
