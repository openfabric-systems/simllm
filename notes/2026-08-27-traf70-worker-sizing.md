# TRAF-70 worker sizing

## Corrected A100 NVLink packet capture

- Date: 2026-08-27.
- As-of branch head: `bbce394df22c498c5c55492cc54be789a790f031`.
- Size: large.
- Scope: freeze a new 80-case A100 NVLink packet-identification protocol,
  implement the corrected batched producer and digest-complete runner, execute
  all 80 isolated cases plus five ordered corner frames and one all-corners
  frame on one qualified four-A100 `NV4` Merlin node, score only the frozen
  identification rules, and publish the exact parameter and evidence-class
  changes supported by the observations.
- Required corrections: every named control reaches the hardware producer;
  copy-engine work is batched; every row records observed raw and data bytes,
  per-link and per-direction counter deltas, replay, recovery and error deltas,
  a destination-byte checksum and ordering ledger, and an explicit throttle
  verdict; candidate-derived fields remain separate from observations; and
  every frozen fatal guard is decidable.
- Execution contract: use the qualified `NV4` class, exclusive
  `a100-hourly`, short `%1` pacing, digest-complete immutable resumption, the
  occupancy rule, and a clean stop record on SSH loss. Bulk evidence lives
  under `<SIMLLM_DEVELOPMENT_ROOT>/wave-runs/traf70/` locally and
  `simllm-data/traf70` relative to the Merlin login home. Only compact score
  and manifest evidence is pulled into the worktree.
- Freeze order: commit the expectations-only record before the corrected
  harness; include per-case bands, the observation-to-parameter decision rules,
  and the complete fatal-guard observable schema; never amend expectations
  after any scored cell starts.
- Publication boundary: keep the TRAF-65 tree and the existing A100 candidate
  profile byte-identical until the TRAF-70 score is published. After scoring,
  change only values and evidence classes that the frozen rules literally
  identify, including explicit candidate refutations through the study-owned
  htsim scoring path.
- Exclusions: no model weights, no web access, no remote Git mutation, no
  deletion, no unrelated traffic or decode-pricing change, and no README prose
  outside the mechanical task-progress block and open-count cells.
- Owner: TRAF-70 worker on `codex/traf70_corrected_capture` in the dedicated
  `traf70` worktree.

### Expected evidence and files

- Frozen study inputs: a new versioned study directory containing the stable
  case catalog, expectations JSON and Markdown, per-case acceptance bands,
  decision rules, fatal guards, and immutable input digests.
- Post-freeze implementation: a CUDA producer, GPU-free build check, resumable
  local and Merlin runners, short paced scheduler entry point, scorer, and
  focused tests.
- External bulk evidence: CUDA binaries, row streams, logs, manifests, and
  attempt records under the two requested `traf70` roots.
- Tracked publication: compact score JSON, human-readable result and resume
  records, permitted task-registry progress, and only score-authorized htsim
  parameter changes.
- Preserved: all TRAF-65 expectations, captures, results and profile bytes
  until the score exists, plus every digest-complete TRAF-70 attempt.

### Expected handwritten line ranges

- Expectations, decision rules and case catalog: 1,200 to 2,800 lines,
  including generated JSON whose source and digest are reviewable.
- Corrected producer and execution helpers: 900 to 1,900 lines.
- Scorer, publication path and focused tests: 900 to 1,800 lines.
- Result, resume, registry and handoff documentation: 250 to 800 lines.
- Generated measurement rows and external bulk logs count as zero handwritten
  lines and remain outside Git.

### Confidence and uncertainty

- Confidence: high that the prior 86-cell catalog and content-addressed
  resumption pattern can be retained while replacing the invalid observation
  schema and producer behavior.
- Dominant uncertainty: public CUDA and NVML interfaces may not expose true
  NVLink packet, replay or recovery counters on A100. The freeze must make this
  limitation a decidable guard rather than infer candidate packet fields from
  elapsed bytes.
- Hardware risk: shared occupancy and SSH availability may require multiple
  short `%1` windows. Complete immutable cells are skipped, never overwritten.
- Publication rule: a completed campaign can still be an honest no-promotion
  result. A parameter moves only if its named frozen rule identifies it from
  observation fields that pass every applicable fatal guard.

### Completion accounting

- Sizing note created before the TRAF-70 expectations tree.
- Expectations freeze, implementation, Merlin job IDs, exact completion
  counts, evidence digests, score, parameter changes and flow-dynamics gate
  verdict will be appended only as those milestones become true.
