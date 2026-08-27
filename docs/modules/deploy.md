# simllm.deploy

The deployment planner is SimLLM's lowest and fastest fidelity rung. It turns
a content-addressed model, declared role pools, device and fabric capacities,
a workload point, service-level targets and physical budgets into stamped
capacity estimates and a deterministic operating frontier without starting a
backend, allocating a GPU or importing a serving framework.

## Interface

- `DeploymentCandidate` is the immutable
  `simllm-deployment-candidate-v1` declaration. Its nested `ModelRef`,
  `PoolSpec`, `FabricSpec`, `WorkloadPoint`, `SlaSpec` and `BudgetSpec`
  records reject unknown fields, wrong schema tags, booleans in integer
  fields, duplicate pool roles, nonpositive widths and devices absent from
  `GPU_ENVELOPES`. `to_json` and `from_json` are the strict wire boundary,
  while `candidate_key` is the SHA-256 of its canonical JSON object.
- `check_feasibility` rejects pipeline-parallel pools that this rung cannot
  price, static per-rank state that meets or exceeds device high-bandwidth
  memory capacity, and candidates above declared GPU or node budgets. The v1
  node count is the number of declared engine slots. An accepted
  `FeasibilityReport` has `accepted=True` and an empty reason tuple; every
  refusal uses stable reason codes.
- The capacity estimator prices prefill, decode, handoff and queue terms in
  integer picoseconds or exact fractions. Every duration carries its evidence
  class and source, and rate matching reports required role-pool engine counts,
  exact utilization and service-level agreement membership.
- The frontier scan evaluates feasible candidates and batch widths in stable
  order, reports exact tokens per second per request and tokens per second per
  GPU, keeps Pareto ties, and prepares the versioned plotting projection. The
  scan is an in-process closed form with zero subprocesses.
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

The strict candidate schema, canonical identity and backend-free feasibility
gate are installed with every v1 refusal locked by tests. The capacity
estimator, frontier scan and wave-level promotion gate complete the planning
rung under DEPLOY-1. Structural placement rendering and SGLang-side candidate
construction remain explicit optional integrations under DEPLOY-2 and
DEPLOY-3.

## Open tasks

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
