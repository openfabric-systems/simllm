# TRAF-65 Merlin execution worker sizing

## Frozen A100 NVLink campaign

- Date: 2026-08-27.
- As-of branch head: `e716f082c50d509c461a3f3843bc7c77ee0e174f`.
- Execution head: `2ab092f9255d77c00c547446b65534a3b273ec82`.
- Expectations SHA-256:
  `212a7a26f54e444c9b18f1e528bd0d00b5a28e4f9e005b0dc137f477ad642571`.
- Size: large.
- Scope: stage the exact merged TRAF-65 study, execute the pending prefix of
  its 86 frozen A100 NVLink cells on one qualified four-A100 `NV4` Merlin
  node, pull content-addressed evidence, run the study's own scoring path, and
  publish only the task and candidate-profile state supported by the measured
  rows.
- Assumptions: the integrator's verified 2026-08-27 node state supersedes the
  written `2026-08-28T06:30` submission hold; the `a100-hourly` partition and
  the merged `%1`, six-minute, task-indexed pacing remain mandatory; and a
  digest-complete cell is immutable and skipped on every resume.
- Exclusions: no COMP-72 or `gh` partition work, no deployment-curve study,
  no htsim NVLink module-code change, no expectations amendment, no model
  weights, no web access, no deletion, no remote Git mutation, and no README
  prose beyond the permitted mechanical progress block and open counts.
- Owner: TRAF-65 execution worker on `codex/traf65_execution` in worktree
  `traf65x`.
- First reviewable slice: this sizing note plus a read-only freeze, staging,
  pending-index, scheduler, and result-root audit before any scored state is
  published.

### Expected evidence and files

- External bulk evidence: Merlin results under the configured
  `simllm-data/traf65` remote root and the local mirror under the requested
  `wave-runs/traf65x` evidence root.
- External lean evidence: compact scoring and completion records under the
  requested `simllm-kernelprobe/traf65` evidence root.
- Created or modified after measurement: study-owned score outputs and report
  artifacts, the candidate profile's parameter values and status fields, the
  TRAF-65 registry wording, and exact remainder metadata if the scheduler
  permits only a completed prefix.
- Preserved: the frozen expectations and decision rules, every digest-complete
  attempt, the deployment-curve studies, COMP-72 state, and htsim NVLink
  implementation code.

### Expected handwritten line ranges

- Cluster dispatch or evidence-transfer helpers: zero to 120 lines, only if
  the merged entry point cannot express a required read-only audit.
- Study scoring and focused tests: zero to 500 lines, preferring the existing
  study-owned path and fixtures.
- Result, candidate-profile, registry, and handoff documentation: 180 to 700
  lines.
- Generated measurement rows and bulk evidence count as zero handwritten
  lines and remain outside Git.

### Confidence and uncertainty

- Confidence: high that staging and resumption are deterministic because the
  archive head, expectations digest, cell catalog, manifest closure, and
  pending-index command are all frozen.
- Dominant uncertainty: shared-node occupancy may allow only a prefix within
  the available paced windows, and the measured counters may refute one or
  more declared candidate parameters.
- Publication rule: score every digest-complete cell without changing a band,
  publish an honest per-corner and per-module verdict, and register the exact
  pending indices. Promote the A100 candidate only where the frozen decision
  rules classify a parameter as measured; otherwise retain it as declared or
  publish its refutation.
