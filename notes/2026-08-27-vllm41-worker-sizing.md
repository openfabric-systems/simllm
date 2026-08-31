# VLLM-41 queue-onset worker sizing

## VLLM-41 lower-load qualification

- As-of commit: `2b5a092943d9ee48bda341327feac2dc975584ed`.
- Scope: freeze and run a lower offered-load ladder through the existing
  concurrent vLLM session, predict the first queue-dominated segment from the
  imported measured batch-service surface and arrival process alone, and
  publish the observed onset and decompositions without reopening the
  validated 250 to 8,000 requests/s monotonic direction.
- Assumptions: the merged VLLM-40 qualification, committed field-addressed
  reader, measured Granite surface, candidate no-calibration status, and
  decomposition boundary are authoritative. The lower ladder, held-outs,
  prediction method, and quantitative bands must be committed before any
  VLLM-41 execution.
- Exclusions: no fit or rescore from observed load-delay curves, imported
  surface mutation, scored flagship execution or edits, prior-record edits,
  traffic-module changes, NVLink capture work, Merlin submission, deployment
  curve changes, web fetches, model-weight downloads, deletions, or README
  prose outside the permitted mechanical progress and open-count cells.
- Owner: VLLM-41 Codex worker on `codex/vllm41_queue_onset` in worktree
  `vllm41`.
- Dependencies: the literal VLLM-41 registry specification, the merged
  VLLM-39/40 load-delay lineage, the committed field-addressed reader, the
  concurrent-session driver, and the task-owned external bulk root supplied
  through an environment-root variable.
- First reviewable slice: commit an expectations-only freeze containing the
  lower load ladder, at least one sub-250 load and one pool-ratio held-out,
  arrival-and-surface-only queue bands, the first queue-dominated-segment
  prediction, and the exact service-versus-wait decomposition rule.

### Expected files

- Created: this sizing note; VLLM-41 expectations, access protocol and ledger,
  frozen runner inputs, exact results, and a maintainer-facing report in a
  task-specific load-delay study directory if isolating the lineage is safer
  than extending the VLLM-39/40 directory.
- Modified: reusable load-delay study code and focused tests only where needed,
  followed by the literal VLLM-41 registry and permitted mechanical task
  progress surfaces if the observed evidence satisfies them.
- Preserved byte-identically: the VLLM-39 and VLLM-40 frozen inputs and result
  records, every scored flagship artifact, deployment-curve artifacts, and all
  parallel NVLink/traffic-lane files.
- Bulk evidence: task-owned run output stays under an environment-root form
  that maps to `wave-runs/vllm41` outside Git.

### Expected handwritten line ranges

- Expectations-only freeze and derivation: 180 to 500 lines.
- Production or reusable study code: 40 to 240 lines, favoring reuse of the
  existing concurrent-session driver and field-addressed reader.
- Focused tests: 100 to 360 lines for freeze ordering, no-fit derivation,
  access logging, decompositions, bands, preservation, and registry honesty.
- Exact result, registry updates, and handoff documentation: 180 to 520 lines.
- External bulk run evidence: counted as zero handwritten lines.

### Confidence and uncertainty

- Confidence: medium that the existing driver can expose the lower-load onset
  without changing scheduler semantics or the imported surface.
- Dominant uncertainty: at sufficiently low finite runs, zero or sparse queue
  waits may make the first queue-dominated segment absent or bracketed rather
  than point-identified. Such evidence will be reported literally.
- Closure rule: close VLLM-41 only if the frozen ladder and held-outs identify
  the onset under their predeclared bands and decomposition. Otherwise keep it
  open and register the residual on VLLM-42 or VLLM-43 without rescoring the
  observed curve.
