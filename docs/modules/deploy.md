# simllm.deploy

The deployment planner is SimLLM's lowest and fastest fidelity rung. It turns
a content-addressed model, declared role pools, device and fabric capacities,
a workload point, service-level targets and physical budgets into stamped
capacity estimates and a deterministic operating frontier without starting a
backend, allocating a GPU or importing a serving framework.

## Interface

- `DeploymentCandidate` is the immutable
  `simllm-deployment-candidate-v1` declaration. The top-level declaration
  rejects wrong schema tags and duplicate pool roles; its nested `ModelRef`,
  `PoolSpec`, `FabricSpec`, `WorkloadPoint`, `SlaSpec` and `BudgetSpec`
  records reject unknown fields, booleans in integer fields, nonpositive
  widths, devices absent from `GPU_ENVELOPES` and non-ASCII strings (the v1
  restriction that keeps the canonical identity total over valid
  candidates). `to_json` and `from_json` are the strict wire boundary,
  while `candidate_key` is the SHA-256 of its canonical JSON object.
- `check_feasibility` rejects pipeline-parallel pools that this rung cannot
  price, static per-rank state that meets or exceeds device high-bandwidth
  memory capacity, and candidates above declared GPU or node budgets. The v1
  node count is the number of declared engine slots; an engine spanning more
  than one physical node still counts as one slot, so the node-budget
  refusal is a lower bound until DEPLOY-2 returns rendered host packing. An
  accepted `FeasibilityReport` has `accepted=True` and an empty reason
  tuple; every refusal uses stable reason codes.
- The capacity estimator prices prefill, decode, handoff and queue terms in
  integer picoseconds or exact fractions. `ModelWork` and `EnvelopeSpec`
  identify the content-addressed inventory and declared GPU roofline inputs.
  `estimate_decode_step` reproduces the frozen 8, 16 and 72 GPU byte
  partitions, ideal link floors and inter-then-intra telescoping identity.
  Optional `BatchServicePoint` rows remain a separate measured scheduler
  service term and are interpolated only through the calibration module's
  installed surface function.
- `estimate_prefill_request` requires an explicit source-carrying prefill
  service and declared handoff. Its `step_ps` is the prefill service, while
  `request_ps` adds the causally subsequent handoff. `queue_delay_ps` uses the
  exact deterministic D/D/c overload form, and `match_pools` returns ceil-form
  engine requirements, exact utilization, capacities and service-level
  agreement membership for separate or combined role pools.
- `EstimateStamp` is the strict `simllm-deployment-estimate-v1` evidence
  record. Every stamped result record (`StepEstimate`, `RateMatchReport`)
  carries the candidate key, estimator class `ESTIMATE`, and a uniquely
  named source for every consumed duration; the bare exact-arithmetic
  helpers (`queue_delay_ps`, `queue_occupancy`,
  `decode_capacity_requests_per_second`) return unstamped exact values by
  contract and take no candidate. Roofline work, declared link rates,
  measured surface entry keys and tracked simulation-record excess remain
  distinct evidence classes. Supplying no source for an enabled term is an
  error, and supplied batch-service points must state their evidence class
  explicitly.
- `ScanInputs` supplies the feasibility bounds plus one `EstimatorInputs`
  record or a per-point resolver. `scan` retains candidate and batch
  declaration order, emits no points for rejected candidates, and records
  their stable reason codes. Accepted points carry exact reduced fractions,
  the estimator stamp, and class `ESTIMATE` or `SIMULATED` according to
  whether tracked `SIM-DERIVED` excess terms were consumed.
- `FrontierRecord` is the strict
  `simllm-deployment-frontier-record-v1` wire boundary. It nests every
  candidate decision and point and carries measured external context as
  paired or y-only anchors; a y-only production anchor is never converted
  into an invented point. `pareto_front` maximizes both exact axes, keeps
  coordinate ties, and returns a canonical order independent of input order.
- `prepare_plot_v3` emits
  `simllm-deployment-frontier-plot-contract-v3` data. It preserves the version
  2 analytical lines, simulated dots, measured white diamonds and dashed
  y-only anchors, then adds point-class marker metadata and Pareto-front
  emphasis. The installed scan and plot preparation are in-process closed
  forms with zero subprocesses.
- Promotion renders an accepted candidate through the repository placement,
  execution and simulation contracts. The candidate key remains the stable
  join between planning evidence and the promoted run.

## Estimator model class

Deployment pricing is a distinct model class, not a simulation shortcut. Each
estimate carries `simllm-deployment-estimate-v1`, the candidate key, the
resolved inputs and a per-term evidence label. Frontier points say `ESTIMATE`
for closed-form prices. A point says `SIMULATED` only when it consumes tracked
simulation-derived terms, and its stamp identifies those source records. The
planning scan itself never becomes a simulation result merely because it
reuses such evidence.

`PrecisionConfig` continues to own duration-only pricing seams such as compute,
locality and network service. A deployment estimator may resolve a level at
those seams and record it, but it does not create a parallel precision surface.
Queue occupancy, pool sizing and rate matching are scheduler-decision
surrogates: they live in the estimator stamp and never enter `PrecisionConfig`,
because changing them can change which deployment is selected rather than only
how long an unchanged execution takes.

## Status

The strict candidate schema, canonical identity, backend-free feasibility
gate, stamped capacity estimator, deterministic frontier scan, exact Pareto
selection and plot-contract v3 preparation are installed with their v1
boundaries locked by tests. The estimator and frontier driver reproduce the
frozen roofline, byte partition, ideal network, telescoping, coordinate and
legacy plot-series forms without starting a process. The wave-level study
completes the planning rung under DEPLOY-1. Structural placement rendering,
SGLang-side candidate construction, parallel scanning and promoted simulation
remain explicit optional integrations under DEPLOY-2, DEPLOY-3, DEPLOY-7 and
DEPLOY-8.

## Open tasks

### Precision

- DEPLOY-4 (Precision; P1; M): Replace the single input-local decode batch
  surface with content-addressed per-width coverage beyond the frozen 8, 16
  and 72 GPU shapes. Identify every width by measured entry keys and accept
  interpolation only when held-out service is within 15 percent; preserve the
  current fail-closed unsupported-width path outside that coverage.
- DEPLOY-5 (Precision; P1; M): Replace the explicit prefill-service term with
  measured prefill batch surfaces keyed by prompt and parallel width. Preserve
  the declared source-carrying input as the identity bypass, and require the
  measured path to predict held-out request service within 15 percent before
  it becomes the default for a covered cell.

### Completeness

- DEPLOY-1 (Completeness; P1; L): Install the estimator, queue and rate-match
  forms, deterministic frontier scan, plot projection and frozen deployment
  scan study, while leaving every existing simulator and adapter entry point
  unchanged when deployment planning is not selected.
- DEPLOY-2 (Completeness; P2; M): Wire accepted candidates into the structural
  placement and fabric renderers and return rendered host packing to
  feasibility; preserve the declared one-engine-per-node arithmetic exactly
  when structural rendering is disabled.
- DEPLOY-3 (Completeness; P2; M): Construct strict deployment candidates from
  SGLang configuration and inventory evidence without moving scheduling
  authority into `simllm.deploy`; preserve manual candidate construction and
  the adapter-disabled path exactly.
- DEPLOY-6 (Completeness; P2; M): In wave P-2, connect the analytical
  collective profile to deployment pricing through the existing compute and
  traffic contracts. Preserve the three frozen byte partitions, link floors
  and every baseline estimate exactly when the profile connection is disabled.
- DEPLOY-7 (Completeness; P2; M): In wave P-2, add an optional multiprocess
  runner for large deployment scans. Preserve the single-process runner as the
  identity off path, including candidate and batch order, exact fractions,
  stamps, rejection records and the zero-subprocess guarantee when parallel
  execution is disabled.
- DEPLOY-8 (Completeness; P2; L): In wave P-4, define the promotion protocol
  that joins an accepted frontier point and estimator stamp to the structural
  placement, execution and simulation records by candidate key. Depend on
  DEPLOY-2 for rendered host packing, and preserve the planning-only record
  exactly when promotion is disabled.
