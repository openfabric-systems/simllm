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
  Zero collective work is an exact identity at TP2, TP4 and TP8, which lets a
  device-only scan retain its declared tensor-parallel structure without
  inventing network bytes; nonzero work and every wider width keep the
  existing sourced-partition requirements and fail-closed behavior.
  Optional `MEASURED` `BatchServicePoint` rows remain a separate scheduler
  service term. A `MEASURED-EXTERNAL` surface instead owns the decode
  `kernel_floor` directly, retains the imported slice and entry-key identity,
  and admits no positive roofline or declared fitted term into that scored
  stamp. Both forms interpolate only through the calibration module's
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
  `weak_dominance_pareto` applies the same coordinate-deduplicated rule to
  pool-composed throughput curves whose y axis is not `batch * decode_speed`.
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
legacy plot-series forms without starting a process. The frozen
`deployment_scan_v1` study passes every scored family with 0 ps maximum error
across both 18-cell compatibility reproductions; its process guards fire zero
times, its 72-point primary pricing takes 0.030626657 seconds and its
6,000-point throughput grid takes 2.084882394 seconds on the disclosed study
machine. Structural placement rendering, SGLang-side candidate construction,
parallel scanning and promoted simulation remain explicit optional
integrations under DEPLOY-2, DEPLOY-3, DEPLOY-7 and DEPLOY-8.

The nonvoid
[frontier comparison](../../examples/frontier_comparison_v1/RESULTS.md) binds
the exact Qwen3-32B-FP8 inventory to the declared H200 roofline and scans 5,070
bounded disaggregated candidates at each of three efficiency arms with zero
pricing subprocesses. Its corrected per-rank accounting produces a MIXED
result: decode e-star 0.586068 remains inside the frozen band, while prefill
e-star 0.142552 falls outside it and X2 passes 3 of 4 rows. X3b passes 10 of 10
through a disclosed two-point step-frontier degeneracy, while X3c passes only 3
of 10 against the frozen minimum of 8. The study validates the decode bracket
and refutes the prefill matched-point premise.

The first published
[matched-seam frontier](../../examples/matched_seam_frontier_v1/RESULTS.md) is
void against its own FG-1. That guard forbade roofline and fitted terms anywhere
in the scored arm, but the imported external resolver is speed-of-light
normalized and its serving composition applies empirical factors. The guard
was not widened. The corrected freeze narrows the valid claim to proving that
SimLLM adds no timing model of its own, keeps the original bands unchanged and
retains every first-run number as void evidence. DEPLOY-12 is reopened for the
corrected evidence and the missing priced-network reference. DEPLOY-13 retains
the rounded-axis residual. DEPLOY-9 through DEPLOY-11 retain their breadth and
silicon-precision scopes.

## Open tasks

### Precision

- DEPLOY-12 (Precision; P0; M): replace the void matched-seam closure with a
  corrected run that proves every scored value bypasses SimLLM's
  `RooflineProvider`, declares every applied external adjustment and publishes
  remove-one Family R sensitivity. Preserve the first run and all of its
  numbers as void evidence. The corrected zero-network arm is an explicit
  unpriced diagnostic only: before attributing its packet gap to receiver-side
  serialization, freeze and execute a third arm that charges LogGOPSim L, o, g
  and G service. Acceptance requires byte-for-byte determinism across two full
  fresh-process scored records with wall time excluded by name, the original
  bands unchanged, and no isolated-network-mechanism claim until the priced
  reference exists.
- DEPLOY-13 (Precision; P1; M): replace the rounded external x coordinate as
  an exact step-frontier threshold with source-carried unrounded coordinates
  or explicit publication intervals. The matched-seam study's F-2-09 surrogate
  compares external 168.131 tokens/s/user with the exact matching point at
  168.130792, excludes that point, selects row 10 and reports 0.607495 against
  the frozen 0.75 floor. Freeze the representation and lookup rule before the
  successor run, retain the current refutation unchanged, and require every
  matched configuration to remain selected throughout its declared rounding
  interval without weakening any quotient band.
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
- DEPLOY-11 (Precision; P1; L): Calibrate the declared H200 device envelope
  used by the Qwen3-32B frontier comparison against retained per-operation and
  end-to-end service observations on H200 silicon. The surrogate being
  replaced is the single declared roofline efficiency applied to inventory
  FLOPs and logical HBM bytes. Identify compute-bound, memory-bound, prefill
  and decode parameters from source-complete framework traces, then require
  held-out service within 15 percent across the registered prompt, context,
  batch and TP2, TP4 and TP8 cells. Keep the declared envelope as an explicit
  comparison-only bypass until the measured profile passes; installing either
  report-only e-star value is forbidden.

### Completeness

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
- DEPLOY-9 (Completeness; P2; M): Generalize deployment candidate enumeration
  beyond the frontier comparison's frozen TP2, TP4 and TP8 widths, positive
  role-worker splits and bounded decode batch ladder. Accept caller-declared,
  source-stamped width and batch families in deterministic order, reject any
  width whose work or network partition is unavailable, and reproduce the
  frozen comparison candidate keys, points and frontiers byte exactly when the
  generalized enumerator is disabled.
- DEPLOY-10 (Completeness; P2; M): Add display-only comparison adapters for
  external deployment planners and serving systems beyond the frozen
  aiconfigurator 0.11.0 rows. Each adapter pins tool, version, database,
  workload and source bytes, stamps every row `MEASURED-EXTERNAL`, and proves
  by process interception and estimator-input inspection that no external row
  enters pricing. With no adapter selected, preserve the accepted frontier
  record and figure bytes exactly.
