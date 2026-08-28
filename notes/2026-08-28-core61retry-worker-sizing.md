# CORE-61 depth-8 retry worker sizing

## Pre-scoring startup-harness amendment and registered retry

- As-of commit: `a7b992be19f744dc7134cc88152c0e401e63404b`.
- Scope: audit both retained failed depth-8 manifests, freeze a pre-scoring
  startup-harness amendment, submit the registered base-then-decode pair, and
  score the exact batch-32, remote-KV-2000 decode step against the unchanged
  3,751,359,511 ps prediction and 5 percent acceptance rule.
- Assumptions: no depth-8 cell has scored; the 65,536-token vLLM dummy startup
  pass is harness scaffolding; and a lower startup scheduler cap plus calibrated
  staggered prompts can expose one exact full-batch decode step with all 32
  cached KV lengths equal to 2,000.
- Exclusions: no prediction or comparison-rule change, no scored-artifact edit,
  no COMP-76 change, no model-weight download, no web fetch, no deployment or
  p2loggopsim work, no deletion, and no README prose beyond permitted
  mechanical progress or open-count cells.
- Owner: CORE-61 worker on `codex/core61_depth8_retry` in worktree `core61r`.
- Dependencies: the two pinned failed-attempt manifests, the staged pinned vLLM
  0.27.1 environment on Merlin, the existing exact-KV scheduler marker route,
  short `gh-hourly` availability, and a fresh Merlin MFA session.
- First reviewable slice: an expectations-only supplement that records the two
  exact startup allocation sites, the amended startup and exact-alignment
  contract, the immutable prediction and tolerance, the base-then-decode order,
  and the fail-closed outcome before any new harness is staged or job submitted.

### Expected files

- Created: this sizing note; a machine-readable and prose CORE-61 retry freeze;
  a narrow task-owned capture harness and scorer if the staged generic harness
  cannot express the exact aligned boundary; and a compact result ledger.
- Modified after the freeze: focused CORE-61 tests plus the CORE-61 and compute
  registry entries exactly as the measured result permits.
- Preserved: the original CORE-61 expectations, 3,751,359,511 ps prediction,
  5 percent rule, all prior failed attempt trees, every scored result, COMP-76,
  integrator-owned README prose, and neighboring deployment worktrees.
- Bulk evidence: local output stays below `wave-runs/core61r`; remote output
  stays below the task campaign root named by `SIMLLM_CORE61_RETRY_RUN_ROOT`;
  only digest-bearing lean evidence is considered for the repository.

### Expected handwritten line ranges

- Task-owned capture and scoring code: 220 to 520 lines.
- Focused tests and compact fixtures: 100 to 260 lines.
- Freeze, result, registry, and handoff documentation: 180 to 420 lines.
- Nsys reports, SQLite exports, scheduler logs, digest manifests, and other
  mechanically emitted evidence: counted as zero handwritten lines.

### Confidence and uncertainty

- Confidence: high that a 4,096-token startup cap fits the depth-8 dummy model,
  because the retained 16,384-token base capture already initialized and ran.
- Dominant uncertainty: whether the existing exact-KV alignment and profiler
  markers resolve a single 32-request, KV-2000 decode boundary under the pinned
  DeepSeek legacy runner without another task-local hook adjustment.
- External waiting: inspect hourly occupancy before each short submission, run
  base then decode, and park with exact resume state if MFA or SSH expires.
- Closure rule: within 5 percent validates linear depth scaling and leaves the
  decode-family gap in expert-parallel residency shape or decode-side overlap.
  A miss publishes its signed non-linearity and registers only the literal
  residual under reserved identifier CORE-63. A second startup failure remains
  a published exact error with CORE-61 open.
