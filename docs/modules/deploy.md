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
  carries the candidate key, estimator class `ESTIMATE` or `ESTIMATE-LOOP`,
  and a uniquely named source for every consumed duration; the bare exact-arithmetic
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
- `SurrogateServingLoop` is the framework-free continuous-batching estimator.
  Its immutable causal tuple fixes the scheduled-token budget, sequence cap,
  chunked-prefill mode, prefix-cache enablement, long-prefill threshold, model
  length, queue policy, 16-token scheduler block geometry, block count,
  reservation mode and watermark. Ordered `SurrogateRequest` rows enter through
  `RequestAdmissionGate`. The loop owns scheduling, preemption, recompute and
  prefix-cache decisions, emits complete `StepRecord` rows plus ordered
  `KvCacheWork`, and advances one virtual clock only to the `StepResult`
  returned by the existing pricing sink.

## Estimator model class

Deployment pricing is a distinct model class, not a simulation shortcut. Each
estimate carries `simllm-deployment-estimate-v1`, the candidate key, the
resolved inputs and a per-term evidence label. Frontier points say `ESTIMATE`
for closed-form prices. A point says `SIMULATED` only when it consumes tracked
simulation-derived terms, and its stamp identifies those source records. The
planning scan itself never becomes a simulation result merely because it
reuses such evidence.

The continuous-batching engine is registered as estimator model class and
point class `ESTIMATE-LOOP`. It changes scheduler decisions while delegating
all duration pricing to the existing compute, locality and network surfaces.
Its native KV stream uses the same lifecycle vocabulary as the live vLLM
normalization bridge and enters the shared lowerer, device runtime, lifecycle
ledger and completion reducer without a framework precision setting.

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

The single-engine `ESTIMATE-LOOP` path implements the pinned vLLM v0.27.1
synchronous continuous-batching rules over virtual arrivals:
running-before-waiting budget subtraction, FCFS and native priority queues,
sequence admission limits, chunked and threshold-limited prefill, recompute
preemption, full-extent prefix hashing, lazy least-recently-used eviction,
full-input reservation and watermark headroom. When prefix caching is off,
released hashless blocks append to the free queue for locality; the enabled
path retains its hashless-prepend and cached-tail ordering. The supported pin
has no `max_num_partial_prefills` or `max_long_partial_prefills` fields, so
neither is a surrogate input or provenance field. Records carry explicit
sampled identities, one-time cached counts, post-step contexts, same-step
preemptions and next-step finished identities. Multi-pool sessions, additional
priority-policy modes, selectable prefix salts and interpreter pins, and
wall-timed arrivals are separate opt-in extensions under DEPLOY-14 through
DEPLOY-17.

The nonvoid
[surrogate conformance study](../../examples/surrogate_conformance_v1/RESULTS.md)
has a corrected, nonvoid post-specified scoring record. F1 is 4 of 4, F2 is 8
of 8, F3 is 4 of 4, F5 is 4 of 4 and F7 is 5 of 5. F4 is 0 of 3, F6 is 0 of
3 and W is 0 of 1, so the loop is not certified. Adversarial review traced the
original F3 failures to the oracle recording allocation order before the
pinned engine freed each manager group in reverse. The corrected capture makes
every F3 row exact and closes DEPLOY-18, withdrawing the phantom surrogate
allocator defect. Cache-enabled F7 omits only FREE because VLLM-43 records
that the bridge cannot distinguish reclaimable cached blocks from discarded
content; its 4, 4 and 6 FREE divergences remain visible as unscored
observations. DEPLOY-19 owns only the genuine one-step-late finished identities
in F4. DEPLOY-20 owns the remaining prefix decision-step and exact KV-accounting
failures in F6, with VLLM-42 through VLLM-44 providing the missing native
observability. The corrected wall-time median is 77,114,203 ns for the
surrogate versus 175,782,543 ns for the live vLLM loop, a ratio of 0.438690906
against the frozen maximum of 0.01; DEPLOY-21 remains open. All 78 fatal guards
were evaluated and passed, including a KV control that starts from a passing F3
row and changes it to a failing row. The qualified estimator claim is limited
to F1, F2, F3, F5 and F7.

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
was not widened, and every first-run number remains visible as void evidence.
The corrected run is nonvoid: all eight fatal guards hold, the two complete
fresh-process scored records are byte-identical, and the original register is
MIXED with S 13 of 13, R 10 of 10, F 12 of 13, M 2 of 2 and W 1 of 1. The valid
claim is narrow: every scored value bypasses SimLLM's `RooflineProvider`, all
eight external adjustments and their remove-one sensitivities are published,
and the packet-priced network is compared only with an arm that charges zero
network service. DEPLOY-12 remains open solely for the missing
LogGOPSim-priced reference required before an isolated network-mechanism claim.
DEPLOY-13 retains the rounded-axis residual. DEPLOY-9 through DEPLOY-11 retain
their breadth and silicon-precision scopes.

## Open tasks

### Precision

- DEPLOY-12 (Precision; P1; M): replace the explicit zero-network diagnostic
  with the corrected freeze's third arm that charges LogGOPSim L, o, g and G
  service before attributing any packet gap to receiver-side serialization.
  Preserve both the void first publication and the corrected nonvoid record
  byte-for-byte, keep the original bands unchanged, and retain the current
  no-isolated-mechanism wording until the third arm identifies a residual
  beyond the priced LogGOP terms. Acceptance requires two complete
  fresh-process scored records to remain byte-identical with wall time excluded
  by name and the explicit bypass to reproduce the corrected unpriced arm.
- DEPLOY-13 (Precision; P1; M): replace the rounded external x coordinate as
  an exact step-frontier threshold with source-carried unrounded coordinates
  or explicit publication intervals. The matched-seam study's F-2-09 surrogate
  compares external 168.131 tokens/s/user with the exact matching point at
  168.130792, excludes that point, selects row 10 and reports 0.607495 against
  the frozen 0.75 floor. Freeze the representation and lookup rule before the
  successor run, retain the current refutation unchanged, and require every
  matched configuration to remain selected throughout its declared rounding
  interval without weakening any quotient band.
- DEPLOY-19 (Precision; P1; M): emit prefix-request finished identities in the
  same decision step as the pinned scheduler. The corrected record withdraws
  the former block-lifecycle attribution: every cache-enabled F7 row passes on
  the authoritative alphabet, while the bridge's 4, 4 and 6 FREE divergences
  are unscored observations owned by VLLM-43. The genuine surrogate finding is
  that finished identities arrive one decision step late in all three F4
  cells, visible in both the normalized record and retained native comparison.
  Acceptance requires exact finish-step identity on all three cells without
  changing F1, F2, F3, F5 or the authoritative F7 rows. Full F4 closure also
  requires VLLM-43 to make the cache-enabled FREE projection authoritative.
- DEPLOY-20 (Precision; P1; M): preserve native decision-step identity and
  identical KV accounting when records enter the shared pricing chain. Frozen
  F6 is honestly 0 of 3: all rows expose surrogate RESERVE and WRITE accounting
  absent from the live sidecar, while the prefix cell also labels its second
  priced row step 2 instead of step 1 and differs in FREE accounting. Keep the
  clause exact. Use VLLM-42 through VLLM-44 to observe the native service,
  content state and pre-decision reserve identity rather than dropping any
  compared field. Acceptance requires all three frozen F6 rows to match
  `StepResult`, time to first token, time per output token and KV accounting
  exactly through the identical lowerer and device runtime, with record, KV
  and pricing mutation controls still firing.
- DEPLOY-21 (Precision; P1; M): reduce the framework-free surrogate's steady
  loop cost from the corrected 77,114,203 ns median on the frozen 128-request
  workload. The live vLLM median is 175,782,543 ns, so the corrected ratio is
  0.438690906 rather than the frozen maximum 0.01. The first publication's
  0.416430536 miss remains preserved and reaches the same verdict. Profile only
  the steady loop, keep construction and capture outside the timed region, and
  replace the identified Python hot paths without changing any decision, KV,
  timestamp or metric byte. Acceptance is a median surrogate-to-live ratio at
  or below 0.01 over seven runs on the same disclosed machine and pin, with
  every exact passing family still byte-identical.
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
- DEPLOY-14 (Completeness; P2; L): Add the multi-pool surrogate session slice
  beneath the existing `pd_session` records and handoff policy. Keep
  `RequestAdmissionGate` as the sole external-arrival authority and handoff
  completion as the sole decode-eligibility authority. When multi-pool mode is
  disabled, preserve every single-engine `ESTIMATE-LOOP` record, timestamp,
  KV operation, price and request metric exactly.
- DEPLOY-15 (Completeness; P2; M): Add priority-policy extensions beyond the
  pinned native FCFS and priority choices behind the scheduler policy seam.
  Mandatory readiness and capacity checks run before any extension may reorder
  candidates. Selecting either installed native policy must preserve its
  waiting order, preemption victim, timestamps and KV lifecycle exactly.
- DEPLOY-16 (Completeness; P2; M): Add explicit prefix-salt and
  interpreter-pinning modes to the surrogate cache identity. Record the salt,
  Python version, pickle protocol and hash implementation in the model input,
  compare lifecycle decisions under one stable block-ID bijection, and retain
  the installed salt-free process-local mode exactly when neither option is
  selected.
- DEPLOY-17 (Completeness; P2; M): Add optional wall-timed request admission
  for concurrent serving experiments. Keep virtual-time admission as the exact
  identity off path, isolate wall time from virtual timestamps and TTFT/TPOT,
  and accept only the separately frozen adjacent-step boundary rule and
  aggregate batch-size band for the wall-timed mode.
