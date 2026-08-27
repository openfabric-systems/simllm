# VLLM-40 clean load-delay worker sizing

## VLLM-40 clean repetition

- As-of commit: `e0388434f3f5ed14451f0393d887b0e4e7414063`.
- Scope: repeat the frozen VLLM-39 load-delay study under the committed
  field-addressed access protocol, using only the named measured granite batch
  rows, and publish the per-segment direction, held-out band, and monotonic
  claim verdicts exactly as observed.
- Assumptions: the merged VLLM-39 pre-run sweep, decomposition rule, held-out
  set, band definitions, and field-addressed reader are authoritative and will
  be reused without amendment or refit unless the VLLM-40 registry entry
  explicitly requires otherwise.
- Exclusions: no record reconnaissance, whole-file reads, unlogged accesses,
  DeepSeek rows, held-out-shape rows outside the declared comparison, web
  fetches, model-weight downloads, deletion, scored flagship execution or
  edits, comparator edits, CORE-51 control edits, or README prose beyond the
  permitted mechanical progress and open-count cells.
- Owner: VLLM-40 Codex worker on `codex/vllm40_clean_load_delay` in worktree
  `vllm40`.
- Dependencies: the literal VLLM-40 registry specification, the merged
  VLLM-39 pre-run artifacts, the committed field-addressed reader, and the
  task-owned external bulk root supplied as `SIMLLM_VLLM40_RUN_ROOT`.
- First reviewable slice: identify and hash the frozen study inputs and
  protected artifacts, then execute only the committed reader's declared
  field accesses while writing the access ledger before interpreting any
  result.

### Expected files

- Created: this sizing note plus a compact clean-run access ledger, exact
  result artifact, and maintainer-facing report if the frozen runner does not
  already generate them.
- Modified: only the literal VLLM-40 result surfaces and registry/task-progress
  cells justified by clean evidence, including VLLM-39 and VLLM-35 no farther
  than the observed result permits.
- Preserved byte-identically: CORE-51 control, deterministic comparator, every
  scored flagship artifact, all frozen pre-run inputs, and VLLM-41 unless the
  declared sweep resolves it without amendment.
- Bulk evidence: task-owned run output stays under
  `SIMLLM_VLLM40_RUN_ROOT` and remains outside Git; the local launcher maps
  that variable to the brief's required external `wave-runs/vllm40` directory.

### Expected handwritten line ranges

- Production or reusable study code: 0 to 80 lines; the committed reader and
  frozen runner should be reused as-is wherever possible.
- Focused tests: 0 to 180 lines, limited to any newly exposed deterministic
  result or registry contract.
- Access ledger, exact result, registry updates, and handoff documentation: 120
  to 420 lines.
- External bulk run evidence: counted as zero handwritten lines.

### Confidence and uncertainty

- Confidence: medium-high that the merged freeze and reader support a literal
  clean repetition without code changes.
- Dominant uncertainty: whether the clean measured rows support all frozen
  directional and band claims; null, mixed, or refuting evidence will be
  published without amendment or refit.
- Closure rule: close VLLM-40 only on literal acceptance, move VLLM-39 and
  VLLM-35 only as far as the clean evidence carries, and leave VLLM-41
  untouched unless the already frozen clean sweep resolves its sub-250 onset
  question for free.
