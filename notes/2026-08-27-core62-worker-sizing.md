# CORE-62 and TRAF-68 worker sizing

Date: 2026-08-27
Branch: `codex/core62_analytical_gate`
Base: `97d34b0aa34fb25f55c24cfd39b923119cf69c95`
Source: maintainer directive of 2026-08-27

## Scope

Register and execute CORE-62 as the versioned analytical deployment-frontier
plot contract and exact roofline consistency gate, and TRAF-68 as the
two-network bottleneck study. CORE-61 remains the separately registered
full-depth decode extrapolation task. The new figure uses per-request decode
speed on the x axis and aggregate output throughput normalized per GPU on the
y axis, both logarithmic. It keeps analytical batch-sweep lines, simulated
operating-point dots, published marker evidence and y-only horizontal anchors
distinct.

The analytical step floor is the maximum of declared HBM weight-read time,
declared peak-compute time and ideal exact-byte serialization over declared
nominal links, with perfect overlap only where the declared schedule permits.
The gate freezes an integer-picosecond accounting identity before its first
run. Every roofline-simulation shortfall must equal the inter-node excess plus
the intra-node excess plus an exactly explained byte-level residual. TRAF-68
publishes those attributed terms and the binding mechanism per swept point.

## Assumptions and exclusions

- The existing disaggregated-session, roofline provider, packet-evidence and
  three-module NVLink candidate profile remain the implementation authorities.
- Kernel simulation is disabled in the gated arm. The existing roofline pricing
  path supplies compute service without a new fitted parameter.
- No prior scored flagship record, figure, expectation or result is rerun or
  modified. The full existing preservation-lock class is verified before
  publication.
- No model weights or web pages are downloaded. No backend submodule is edited.
- Compact reproducible rows and figures may be tracked. Bulk backend output and
  scratch diagnostics live under the task-specific `wave-runs/core62/` root.
- README prose and study indexes remain integrator-owned. Only the mechanical
  task-progress block and module-status open-count cells may change there.
- Reserved residual IDs are TRAF-69 and COMP-77. No residual is folded into
  CORE-62 or TRAF-68 merely to close them.

## Owner, dependencies and first reviewable slice

- Owner: CORE-62 Codex worker in the `core61` worktree.
- Dependencies: `RooflineProvider`, transformer model dimensions and GPU
  envelopes, the disaggregated placement/session projection, packet-level
  completion evidence, the candidate NVLink module profile, and the current
  flagship preservation lock.
- First reviewable slice: an expectations-only commit containing the versioned
  axis and line/marker contract, every analytical input and source locator, the
  batch and configuration sweep, the integer attribution identity, extraction
  rules, fatal guards, expected directions, physical bounds, task registration
  and preservation digests. It contains no implementation, generated result or
  measured value.

## Expected files

- Created before the run: `examples/deployment_frontier_v1/expectations.md`,
  `examples/deployment_frontier_v1/expectations.json`, and EOL pins for the new
  study family.
- Created after the freeze: a narrow analytical and accounting module, study
  runner, renderer, compact JSON and Markdown results, and PDF and PNG figures
  under `examples/deployment_frontier_v1/`.
- Created or modified after the freeze: focused freeze, model, runner and
  renderer tests under `tests/`; `docs/modules/core.md` and
  `docs/modules/traffic.md` for literal registry status only; mechanical
  progress cells in `docs/README_PRO.md`.
- Preserved byte for byte: every file in the existing flagship preservation
  lock and all earlier deployment figures and records.
- Bulk evidence: task-specific packet output and scratch render diagnostics
  under `wave-runs/core62/`, counted as generated output.

## Expected handwritten line ranges

- Production and reusable study code: 350 to 800 lines.
- Focused tests and compact fixtures: 300 to 700 lines.
- Freeze, registry and result documentation: 400 to 850 lines.
- Compact generated result JSON and rendered figures: zero handwritten lines.
- Bulk packet traces and backend output: zero handwritten lines.

## Confidence and uncertainty

- Confidence: medium.
- Dominant uncertainty: whether the current session exposes packet-fabric and
  NVLink module timestamps at one lossless join, or whether the study must
  construct a narrow read-only projection from existing evidence classes.
- Network physics and integer accounting are high confidence once that join is
  identified. Publication layout is low risk but requires visual inspection at
  final physical size.
- External or hardware work: none expected. The packet backend binary must be
  available locally; if absent, the run remains blocked rather than being
  replaced with downloaded or invented evidence.
- Closure rule: CORE-62 closes only when every swept roofline point has zero
  unexplained residual and satisfies the versioned plot contract. TRAF-68
  closes only when every point has both attributed terms, one declared
  bottleneck class and module or mechanism evidence. Any exact remainder is
  registered under TRAF-69 or COMP-77 according to ownership.

## Completion accounting

Completed on 2026-08-27 with 23 touched repository files: 16 handwritten
source, test, freeze or registry files and seven generated publication files.
The external packet run retained 56 files and 332 KiB under
`wave-runs/core62/gated-run-1/`; no bulk file entered the repository.

The final handwritten delta is 2,829 inserted lines and 4 removed lines, or
2,825 net. Reusable study code is 1,679 lines, 879 above the estimated upper
bound because the implementation keeps exact arithmetic, guarded packet
execution, plotting and publication as four auditable surfaces. Focused tests
are 503 lines and remain inside the 300 to 700 estimate. Freeze and registry
documentation remains inside its estimated range.

The dominant uncertainty resolved to a compact exact projection: two
independent four-endpoint candidate domains reproduce the packet-object
timestamps and byte ledgers without retaining millions of objects. All 18
accounting residuals are 0 ps. CORE-62 closes literally. TRAF-68 stays open
because its frozen nine-node arm exposes raw incast but no elapsed inter-node
bottleneck, so the first study is published as a refutation.
