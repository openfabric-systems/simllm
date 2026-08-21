# simllm.compute

Pluggable compute-time providers plus the host initiation model. The scalar
compatibility path needs one number per GOAL `calc` node: how long a rank
computes before it hands data over. Higher-fidelity execution resolves compact
immutable service entries and advances the selected batch or incremental
device service. Trace replay, profiling, fitting and external Accel-Sim use
remain offline; none runs once per serving step.

## Interface

- `KernelSpec`: fused work plus its stable shape key. A fused transformer step
  also carries the exact `family_kernels` projection used to apportion work;
  ordinary kernels leave it empty.
- `ComputeProvider.estimate(kernel: KernelSpec, gpu: GpuSpec) -> DurationEstimate`
- `ComputeProvider.estimate_layers(kernel, gpu, num_layers)`: optional ordered
  layer estimates for the same fused kernel. The default returns `None` and
  preserves scalar callers exactly. An implemented breakdown must contain one
  nonnegative duration per layer and sum to `estimate()` exactly; consumers
  validate both invariants before using it.
- `ProfileTableProvider`: measured (kernel name, config, GPU) duration
  tables from real captures or offline SASS simulation. Exact-match
  lookups return the entry; a miss interpolates log-linearly along one
  numeric config axis between the nearest bracketing entries of the same
  kernel and GPU, with the uncertainty inflated to
  `max(0.15, neighbors')` (interpolation never claims tighter error than
  its inputs). Queries outside the covered range, or differing on more
  than one axis (COMP-4), raise `KeyError`. Tables round-trip through a
  versioned JSON artifact (schema `simllm-profile-table-v1`) with
  mandatory provenance (`source` e.g. "accel-sim" or "capture",
  simulator/capture `version`, `gpu`, caller-supplied `created` date; the
  library never reads the clock). `enable_family_sum=True` is an explicit
  opt-in that sums a fused kernel's declared family projections and propagates
  conservative uncertainty. The default ignores that projection, retains the
  historical miss behavior and serializes the same table byte for byte.
- `ComputeCalibrationArtifact`: strict
  `simllm-compute-calibration-v1` capture record. It binds GPU, driver, CUDA,
  profiler, source, binary, static-SASS and capture-manifest identities to an
  immutable train or held-out split, launch metadata and every raw duration
  sample. Its compiler emits the existing profile-table schema using train
  medians and held-out error to set family uncertainty.
- `RooflineProvider`: analytical `max(flops/peak, bytes/bw)` with an
  efficiency derate; classifies compute- vs memory-bound from the kernel
  configuration alone. `enable_layer_breakdown=True` apportions the fused
  duration using family work on the selected roof. Repeated transformer
  families divide evenly and the complete LM-head family belongs to the last
  layer. Cumulative integer boundaries guarantee that the nonnegative layer
  durations sum to the scalar estimate exactly. The default is disabled and
  retains the scalar compatibility path byte for byte.
- `TraceCalibratedGpuProvider`: validates and replays its exact trace catalog
  once at construction, then serves O(1) cached estimates behind the existing
  `ComputeProvider` interface. `gpu_model_artifact_to_profile_table` compiles
  validated replays into the smaller immutable online table artifact.
- `ModelDims`: per-rank transformer geometry, dense or MoE. MoE fields
  (`num_experts`, `top_k`, `moe_intermediate_size`, `local_num_experts`)
  default to the dense model; when declared, per-token MLP flops count
  `top_k` experts and weight bytes count only the experts resident on
  this rank under expert parallelism (`local_num_experts`; 0 means all).
- `step_kernel`: one engine step as a single fused kernel (what the
  adapters price today).
- `step_kernels`: the same step split into named kernel families
  (`attn_gemm`, `attn_score`, `mlp_gemm`, `lm_head`, `kv_read`), each
  carrying its shape key (`new_tokens`, `kv_tokens`, `sampled` as
  applicable). Family flops and bytes sum to the fused kernel exactly
  (unit-tested invariant; weights counted once, in the family that
  streams them). This is the COMP-1 groundwork: offline SASS runs
  populate per-family profile tables, and the step loop sums per-family
  estimates instead of pricing one opaque blob.
- `GpuDeviceConfig` and `GpuDevice`: the versioned GPU composition entry point.
  A device is an architecture profile plus typed `GpuPortConfig` ports, each
  carrying protocol, role, direction, declared capabilities, an optional
  declared ceiling and the provenance of the ceiling it ends up with.
  `default_gpu_device_config` derives the port set an architecture's own
  mechanisms already imply. With no declared ceiling, `GpuDevice.architecture`
  is the input object itself, so `sm_scheduler_model()` and
  `copy_engine_service()` reproduce every accepted artifact exactly.
  `GpuPortProtocol` names PCIe and NVLink-C2C on the host link and NVLink,
  PCIe, xGMI and UALink on the peer link. Naming a protocol is not supporting
  it: xGMI and UALink have no first-party measurement and no declared profile
  here, so a port claiming either is rejected during configuration with a
  diagnostic naming COMP-35, which owns vendor instantiation for both.
- `HostInitiationModel`: the exact-zero `ideal` profile, legacy additive
  constants, and two device-bound fixed-step launch-throughput profiles.
  `turing_cuda_graph(N)` and `turing_eager_host(N)` compose provider service
  `C` as `max(C, N * g)`, because host launch demand can overlap device
  service. Each calibrated estimate retains its raw provider duration, launch
  floor, empirical bounds and exposed host contribution. The named Turing
  profiles accept only `GpuSpec.name="gtx1660-ti-sm75"`; a B100 or H100
  selection fails during configuration instead of borrowing the constant.
- `HardwareCollector`: an offline-only producer of environment records,
  physical launch observations and silicon measurements. Vendor backends may
  use CUDA/CUPTI/Nsight, ROCm/rocprofiler or another declared toolchain, but
  they emit the same typed evidence records and never enter the serving import
  path.
- `OfflineKernelSimulator`: an offline-only adapter over a pinned external
  simulator. It consumes content-addressed captured inputs and emits typed
  simulator observations. It is optional, capability queried and never a
  source of silicon truth.
- `CalibrationCompiler`: the deterministic offline compiler from validated
  evidence and an immutable split to a compact device model. It applies
  source precedence, fits only the training partition, scores validation and
  test partitions without fitting them, and emits no model when a fatal guard
  fails.
- `simllm-device-calibration-bundle-v1`: the contributor and campaign
  evidence envelope. It is an acyclic content-addressed record graph whose
  raw traces and profiler exports remain external blobs named by digest. It
  carries capture and simulator environments, exact graph and implementation
  observations, silicon measurements, optional simulator observations,
  immutable train/validation/test membership, fit inputs and validation
  results. An evidence manifest hashes only its evidence records; a fit names
  that manifest; a release manifest names the evidence manifest, fit, compact
  model and validation result, so no record hashes itself or forms a cycle.
- `OperationImplementationBinding`: an observed-capture sidecar joining an
  exact `(instance_graph_sha256, operation_id, launch_ordinal)` to the
  `implementation_ref` actually reported by the model runner and profiler and
  its typed `shape_vector`. Those five fields are the complete record; fitted
  demand and service-entry identity are forbidden. The instance graph hash is
  evidence provenance only.
- `CollectiveDeviceStageBinding`: an observed-capture sidecar joining
  `(instance_graph_sha256, collective_operation_id,
  collective_plan_integrity_sha256, rank, launch_ordinal)` to one observed
  GPU-resident NCCL or RCCL `ImplementationRef` and typed `ShapeVector`. Demand
  stays in measurement records and compiled service entries. The binding is an
  internal physical stage of its parent `CollectiveWork`, not a second graph
  operation or an independent completion authority.
- `simllm-resolved-operation-service-binding-set-v1`: an immutable
  per-execution sidecar produced before device state mutates. It contains
  exactly `schema`, `instance_graph_sha256`, `dispatch_context_sha256` and
  `bindings`. The immutable dispatch context includes the validated rank/device
  assignment and a total tuple of selected models keyed by device instance. The
  applicable selected model resolves every compute operation from its semantic
  work and typed shape to exactly one implementation and service entry. Online service
  lookup uses `(implementation_id, shape_vector)`, never a graph hash or a
  kernel-name fallback. Each member contains exactly `instance_graph_sha256`,
  `operation_id`, `launch_ordinal`, `device_instance_id`,
  `device_model_sha256`, normalized `semantic_key`, `shape_vector`,
  `implementation_ref`, `service_entry_id`, `resolution_source` exactly
  `selector` or `observed-binding`, and required-nullable
  `observed_implementation_binding_sha256`. Bindings follow graph operation
  tuple order, then launch ordinal. Device/model identity must match the
  selected-model tuple.
- `simllm-device-dispatch-context-v1`: the canonical preflight identity shared
  by both resolved sets. It contains exactly `schema`,
  `instance_graph_sha256`, `rank_device_assignments` and
  `selected_device_models`. A rank assignment contains exactly `rank` and
  `device_instance_id`; assignments are unique, sorted by integer rank and
  total over every graph participant rank. A model selection contains exactly
  `device_instance_id`, `device_model_id`, `device_model_sha256` and
  `dispatch_signature_sha256`; selections are unique, sorted by device
  instance, and cover exactly the devices used by the assignments.
  `simllm-dispatch-signature-v1` contains exactly `schema`, `framework_id`,
  `framework_version`, `backend_id`, `backend_version`, `kernel_library_id`,
  `kernel_library_version`, `algorithm_policy_id`, `device_isa`,
  `numeric_traits` and `layout_traits`. Every identifier and version is
  nonblank. Each trait follows the nested
  `simllm-typed-dispatch-trait-v1` contract and is the strict object
  `{trait_id, value_type, value}`;
  its type is exactly `integer`, `string` or `boolean`, its value matches that
  type, and each tuple is sorted with unique nonblank trait IDs. Launch mode is
  forbidden because it changes host initiation only. The dispatch-context
  digest is the SHA-256 of this exact canonical context record, and each model
  selection's signature digest resolves to the exact signature used by that
  model.
- `OperationServiceResolver.resolve(graph, selected_device_models,
  dispatch_context) ->
  ResolvedOperationServiceBindingSet`: validates the frozen model envelope,
  normalizes every typed shape, resolves every implementation and exact service
  entry, and proves totality before any runtime or service state mutates. It is
  pure and deterministic.
- `CollectiveDeviceStageResolver.resolve(planned_graph,
  selected_device_models,
  dispatch_context) -> ResolvedCollectiveDeviceStageSet`: the device model's
  stage selector resolves the ordered GPU-resident implementations, shapes and
  service entries for every supported semantic collective and rank. The
  traffic-owned `CollectivePlan` supplies topology and frontier identities, not
  implementation choice. The resolver proves totality before mutation and
  never creates a `ComputeWork` node. The semantic collective remains the sole
  graph lifecycle and completion authority; compute owns only stage
  feasibility, SM/HBM reservation, service and release. Version 1 rejects a
  stage entry with nonzero peer-port or data-mover demand; those intervals stay
  with the existing traffic plan and later observed-mover composition. Each
  fresh `ResolvedCollectiveDeviceStage` contains `instance_graph_sha256`,
  `collective_operation_id`, `collective_plan_integrity_sha256`, rank,
  `launch_ordinal`, `device_instance_id`, `device_model_sha256`, selected
  `implementation_ref`, normalized `shape_vector`, `service_entry_id` and
  `resolution_source` exactly `selector`. It requires no
  historical capture binding; validation compares the two records separately.
- `CollectiveDeviceRankFrontier`: the conservative version-1 handoff nested in
  the resolved stage set. It carries `collective_operation_id`, exact
  `collective_plan_integrity_sha256`, rank, `ordered_stage_ordinals`, and the
  exact `entry_action_ids` and `terminal_action_ids` copied from that rank's
  traffic-owned plan. Version 1 requires exactly one resolved resident stage
  per rank and rejects a multi-stage online composition until stream and
  dependency evidence identifies its order. There is one frontier for every
  plan rank, every action resolves to that rank, the copied tuples match the
  plan byte for byte, and every resolved stage occurs exactly once with no
  extra.
- `simllm-resolved-collective-device-stage-set-v1`: the nonempty canonical
  collective set with exactly `schema`, `instance_graph_sha256`,
  `dispatch_context_sha256`, `stages` and `rank_frontiers`. Stages are ordered
  by graph collective tuple order, plan rank order and launch ordinal;
  frontiers are ordered by graph collective tuple order and plan rank order.
  Every stage device/model pair matches the selected-model tuple. A graph with
  no resolved collective stage omits this record and uses null in the closure;
  an empty record is invalid.
- `simllm-resolved-device-binding-closure-v1`: a canonical immutable record
  with exactly `schema`, `instance_graph_sha256`,
  `operation_service_binding_set_sha256` and nullable
  `collective_device_stage_set_sha256`. Run provenance hashes this complete
  closure, so a communication-stage choice cannot disappear from a result.
  The operation set must name the closure's graph. A nonnull collective set
  must name that same graph and the operation set's exact dispatch-context
  digest. Every device/model pair in either set must match the dispatch
  context and the selected-model tuple copied into run provenance. A mismatch
  is a splice error and rejects the whole closure before publication.
  For a resolved collective rank, `submitted_at` is parent collective launch
  completion and `eligible_at` is the maximum of that boundary and rank-local
  graph-predecessor readiness. The device-stage grant at `started_at` releases
  that rank's existing plan entry actions. The device engine reaches
  `device_work_finished_at_ps` when its throughput demands and floors finish,
  but compute retains the stage's lifetime residency and exclusive
  reservations. Traffic alone owns every plan action, extent, byte, peer-port
  and network timestamp. Define `traffic_terminal_at_ps` as the maximum of the
  rank's `eligible_at` identity element and every copied terminal-action
  completion, so a legal sparse rank with no terminal action remains defined.
  The read-only traffic boundary gates the compute-owned lease release:
  `rank_release_at_ps = max(device_work_finished_at_ps,
  traffic_terminal_at_ps)`. Compute releases reservations there, rank
  completion equals that boundary, and the parent emits one graph completion
  at the maximum across ranks. No plan action depends on release, so the gate
  is acyclic. A disabled
  path, or an entry whose every active-axis demand is known zero in every epoch
  and whose every epoch floor is null or zero, constructs no device visit,
  delays no entry action beyond the same preexisting legal boundary and
  preserves the accepted traffic schedule exactly. A positive floor prevents
  that bypass even when every demand is zero.
  The authoritative stage `QueueVisit` remains under the parent
  `collective_operation_id` and uses the stable subject object ID
  `<operation-id>:rank:<rank>:stage:<launch-ordinal>`. Its `finished_at` and
  `completed_at` both equal `rank_release_at_ps`. Version 1 identifies this
  composite visit with the existing
  legacy `GPU_SCHEDULER` `ResourceRef`; internal service axes do not become
  separate queue visits. CORE-8 loss-checks its normal completion and
  bookkeeping projection. CORE-50 is required only if a later model projects
  an otherwise-unrepresentable outer resource. The stage emits no independent
  graph-operation completion; only the semantic collective emits that event.
  The interval from `device_work_finished_at_ps` to `rank_release_at_ps` is
  lease-held occupancy evidence, never another additive kernel or network
  latency term. CORE-26 implements this through an incremental request with
  `release_mode=external-frontier`; after traffic publishes its terminal fact,
  runtime calls the compute-owned transaction's `release_held` method. Traffic
  never mutates the reservation directly.
- `simllm-device-model-v1`: the deterministic compact online artifact. It
  contains model identity, closed `acceptance_status` exactly `candidate` or
  `validated`, closed `target_basis` exactly `target-silicon` or
  `architecture-derived`, a support envelope, shape schemas, implementation
  and collective-stage selectors, exact service entries, a typed resource-axis
  registry with an explicit known mask, and a per-entry evidence ledger.
  `architecture-derived` requires `acceptance_status=candidate`; target-silicon
  models may be either candidate or validated.
  Unknown demand is distinct from an explicit zero. The artifact contains
  neither raw traces nor an Accel-Sim dependency.

The exact `simllm-device-model-v1` object contains `schema`,
`device_model_id`, `device_kind_id`, `acceptance_status`, `target_basis`,
`device_identity_sha256`, `operating_envelope_sha256`,
`support_envelope_sha256`, `evidence_manifest_sha256`, `fit_sha256`,
`expectations_commit`, `dispatch_signature_sha256s`, `shape_schemas`,
`implementation_selector_sha256`, `collective_stage_selector_sha256`,
`resource_registry`, `interaction_contract`,
`host_initiation_profile_sha256`, `service_entries`,
`service_entry_evidence`, `scalar_profile_table_sha256`, `gpu_spec_sha256`,
`gpu_architecture_profile_sha256`, `gpu_device_config_sha256`,
`validation_record_sha256`, `validation_summary_sha256`,
`acceptance_bars_sha256` and `model_limits`, with no extra members. The
expectations commit is lowercase hexadecimal with length 40 or 64. Dispatch
signature digests are sorted, unique and nonempty. Shape schemas are nonempty
and sorted uniquely by schema ID.

The implementation selector is a required content-addressed declarative-data
record with no code, callback, expression or binary. The collective selector
is nullable and is null exactly when the support envelope claims no
collective-device stage. The resource registry and
`independent-resource-v1` interaction contract are inline and name the model's
device kind. Each service-entry member is exactly `{service_entry_id, entry}`;
members are nonempty, sorted uniquely by nonblank ID and unique by
`(implementation_id, shape_vector)`. Each evidence-ledger member contains
exactly `service_entry_id`, `source_selection`, `source_record_sha256s`,
`residual_record_sha256`, `support_envelope_sha256`,
`operating_envelope_sha256`, `isolated_duration_ps` and
`uncertainty_bound`, sorted one-to-one with service entries. Source selection
is exactly `silicon`, `silicon-fit`, `accel-sim`, `analytical-transfer` or
`simulator-derived`; source digests are sorted, unique and nonempty, duration
is nonnegative and uncertainty is an exact nonnegative rational.

Device identity, both envelopes, evidence manifest, fit, implementation
selector, validation record, validation summary and acceptance-bars
references are required. Collective selector, host initiation, scalar profile
table, GPU spec, GPU architecture profile and GPU device configuration
references are explicitly nullable. `model_limits` contains exactly
`max_shape_schemas`, `max_shape_axes_per_schema`, `max_resource_axes`,
`max_service_entries`, `max_epochs_per_entry` and `max_resident_entries`, each
a positive signed-128 integer that bounds the corresponding count. Every
nonnull reference resolves inside the release closure and cross-record device,
envelope, selector, entry and validation identities agree. Raw traces, sample
vectors, profiler rows and simulator dependencies stay outside this artifact.
- `simllm-device-resource-registry-v1`: the canonical registry carried by a
  device model. Its exact members are `schema`, `device_kind_id`,
  `active_axis_ids` and `axes`. Axis IDs are unique; `axes` sort
  lexicographically by `axis_id`; `active_axis_ids` are a sorted unique subset.
  Each strict `DeviceResourceAxis` has exactly `axis_id`, `axis_class`,
  `service_scope`, `base_unit`, `clock_domain_id`, `capacity_source_id`,
  `rate`, `residency_capacity` and `exclusive_capacity`. The class is exactly
  `throughput`, `residency` or `exclusive`; service scope is exactly
  `device-internal`, `peer-port` or `data-mover`. All three capacity members
  are required, and exactly the class-appropriate one is nonnull. `rate` is the
  exact object `{numerator, denominator}` with reduced nonnegative numerator
  and positive denominator; residency capacity is a nonnegative integer and
  exclusive capacity is a positive integer slot count. The registry SHA-256 is
  over this exact canonical record.
  A cycle-denominated throughput rate requires a clock domain; a wall-time
  rate leaves it null. Non-throughput axes reject a rate. An active axis with
  any positive accepted demand requires positive class-appropriate capacity at
  load time.
- `DeviceResourceVector`: a `registry_sha256`, `device_kind_id`, integer values
  and Boolean known bits aligned to the registry's complete axis order. A known
  value is nonnegative; a negative demand is a load error. A known zero means
  that service entry has no demand on the axis. Every active-axis value must be
  known; inactive-axis entries use an unknown bit and canonical zero placeholder
  that carries no demand. Unknown never becomes a known zero.
  Throughput demand, residency demand and exclusive occupancy are never
  interchanged. Load and evaluation reject any numerator, denominator,
  cross-product or accumulation outside the signed 128-bit domain; one ceiling
  occurs only at the externally visible integer-picosecond boundary.
- `DeviceServiceEntry`: an immutable `(implementation_id, shape_vector)` key
  and a nonempty ordered tuple of immutable `ServiceEpochDefinition` values.
  Each epoch carries one aligned `DeviceResourceVector` of resource-native
  demands and an optional fixed floor in integer picoseconds. Entries contain
  no start time or editable service rate. Throughput values are consumable work;
  residency and exclusive values are held requirements and never decrement.
  Admission reserves the per-axis maximum residency and exclusive requirement
  across all epochs for the entry's full lifetime. Runtime resident state owns
  admission, epoch start, current epoch and remaining throughput demand. An
  epoch advances only after every throughput demand reaches zero and its fixed
  floor has elapsed. An ordinary entry releases its lifetime reservations at
  final work completion; a collective stage follows the rank lease-release
  rule above.
- `independent-resource-v1`: the only version-1 resource law. Every throughput
  axis divides its one registry capacity equally among resident epochs with
  positive remaining demand on that axis, using exact rational arithmetic.
  Axes otherwise progress independently. `interaction_terms` must be empty and
  a nonempty term set is rejected pending a versioned interface amendment.
  Mechanistic service lookup uses exact `DeviceServiceEntry` cells and never
  interpolates resource demands or reservations. An optional compiled scalar
  table may declare one integer shape axis for duration interpolation. Between
  canonical cells `x0 < x < x1`, it evaluates
  `y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)` with exact reduced rationals,
  inclusive support, exact hits first, lower-cell tie choice and one ceiling
  only when `y` becomes an externally visible integer-picosecond duration.
  Generic multi-axis interpolation remains COMP-4.
- `BatchKernelService`: the first live capability. It consumes one fully
  resolved co-runnable tuple and preserves the current complete-tuple order,
  cycle-to-picosecond ceiling, composition cursors and barrier mechanics. Its
  enabled service may intentionally change completion and metric values. The
  scalar and legacy off paths preserve their service calls, cursors, reports
  and result bytes exactly. Its pure call is
  `dispatch_batch(requests: tuple[ResolvedDeviceServiceRequest, ...],
  common_start_ps: int, snapshot: DeviceServiceSnapshot) ->
  BatchKernelServiceResult`. Inputs are immutable and ordered; the result
  contains an ordered `ServiceFact` tuple, `DeviceAccounting` and
  `next_snapshot`. The service mutates no live state and emits no callback or
  graph completion. Runtime validates the complete result before atomically
  adopting the snapshot and facts. Failure leaves live state unchanged.
  `ResolvedDeviceServiceRequest` contains exactly a stable `subject_key`,
  `service_entry_id` and `release_mode`, whose closed values are `work-finish`
  and `external-frontier`. `DeviceServiceSnapshot` is the strict object
  `{device_instance_id, device_model_sha256, registry_sha256,
  resident_states}`; runtime composition cursors never enter it. Resident
  states are canonically ordered by `(admission_sequence, subject_key)`, but
  version-1 batch dispatch accepts only an empty input tuple and a successful
  total result must return it empty. Nonempty resumable state belongs to
  CORE-12 incremental service. Batch service accepts only `work-finish` and
  rejects `external-frontier`; the initial COMP-25 batch path is therefore
  ordinary noncollective service.
  At `common_start_ps`, and after every release, batch admission first releases
  completed reservations, then scans pending requests in original tuple order,
  admitting each request that is feasible against the updated snapshot and
  skipping the rest. Zero-duration completions and newly feasible requests are
  drained to a finite same-time fixed point before time advances. Preflight
  rejects a request whose lifetime reservation maxima exceed device capacity.
  Every `ServiceFact` has exactly `subject_key`, `epoch_index`,
  `submitted_at`, `eligible_at`, `started_at`, `work_finished_at`,
  `finished_at` and `completed_at`. For an ordinary batch request,
  `work_finished_at=finished_at`.
  Facts are serialized in request tuple order, then ascending epoch index. For
  the first epoch, `submitted_at=eligible_at=common_start_ps` and `started_at`
  is its admission grant. For every later epoch,
  `submitted_at=eligible_at=started_at` equals the preceding epoch's
  `work_finished_at`. An intermediate epoch has
  `finished_at=completed_at=work_finished_at`; the ordinary final epoch also
  has all three equal.
  `DeviceAccounting` has exactly `registry_sha256`, aligned reduced-rational
  `admitted_throughput` and `served_throughput` totals, and aligned integer
  `acquired_reservations`, `released_reservations` and
  `held_reservation_ps` totals. Throughput totals are zero on non-throughput
  axes; reservation totals are zero on throughput axes. Held reservation is
  demand units times lease duration and is also zero on throughput axes.
  Validation requires contiguous declared epochs, exactly one final
  completion per input subject, no extra field, subject or epoch, monotonic
  boundaries, served equal to admitted, released equal to acquired, exact
  agreement with the input entries, no resident leakage, and a next snapshot
  with the same device, model and registry identities.
  Transactional admission of a later operation into an active device is a
  separate CORE-12 capability rather than an implied property of this batch
  interface.
- `IncrementalDeviceService.begin(snapshot) ->
  IncrementalDeviceServiceTransaction`: the CORE-12 capability. A transaction
  exposes pure `admissible(request, now_ps) -> Feasibility`, mutating
  `dispatch_granted(request, admission_sequence, now_ps)`,
  `peek_next_event_ps() -> int | None`, `advance(to_ps) ->
  tuple[DeviceServiceEvent, ...]`, mutating
  `release_held(subject_key, release_at_ps) -> ServiceFact`, read-only
  `accounting() -> DeviceAccounting`, and `prepare()`, `commit()` and `abort()`.
  `now_ps` equals runtime's current logical time and transaction time never
  decreases. `DeviceServiceEvent` is a closed union: a
  `WorkFinishedEvent` has exactly `{kind: "work-finished", subject_key,
  epoch_index, at_ps}`, while a `ServiceFactEvent` has exactly
  `{kind: "service-fact", fact}` with one strict `ServiceFact`. Intermediate
  epochs and ordinary final epochs emit a service fact when work finishes. An
  external-frontier final epoch emits only `WorkFinishedEvent` at work finish.
  A `work-finish` request releases normally. An `external-frontier` request
  retains its lifetime reservations after work finish; `release_held` is valid
  exactly once, only for that mode's final epoch, only after its
  `WorkFinishedEvent`, and at a boundary no earlier than work finish. Runtime
  first advances the transaction through that boundary;
  `release_held` then releases at the transaction's current time and returns
  the one final fact whose `work_finished_at` preserves the earlier boundary
  and whose `finished_at=completed_at=release_at_ps`. CORE-26 alone supplies
  that release boundary from the read-only traffic terminal. Runtime alone
  selects a grant. If the traffic terminal is already known, release follows
  work finish immediately; otherwise runtime advances to the later terminal.
  `peek_next_event_ps()=None` while a subject is held means that compute has no
  internal event, not that the global runtime is quiescent or dead. Events sort
  by time, admission sequence, epoch index and then variant, with
  `work-finished` before `service-fact`. Every participant
  prepares before the fixed infallible adoption order: device engine,
  arbitration policy, runtime state, one bookkeeping batch, then callbacks.

Every estimate carries an honest uncertainty so results can report error
bounds.

## Kernel-time determinism

This is the model's kernel-time semantics, and it is a contract, not an
implementation detail. It follows the maintainer ruling of 2026-08-18.

**A compute kernel's isolated service law and resource demand are deterministic
constants with no tail.** They are a pure function of exactly four inputs:

1. the **kernel family** (`attn_gemm`, `attn_score`, `mlp_gemm`, `lm_head`,
   `kv_read`, or the fused `llm_step` that projects onto them),
2. the **phase**, prefill or decode,
3. the **token and shape inputs**, i.e. `new_tokens`, `kv_tokens`, `sampled`
   and the per-rank `ModelDims` geometry, and
4. the **architecture profile**, i.e. the `GpuSpec` envelope or the
   `GpuArchitectureProfile` the mechanistic replay is calibrated to.

Nothing else may enter. The same four inputs give the same isolated floor and
demand vector on every rank, in every worker, through either frontend adapter,
on every repeat, and in every process. No provider draws a random number, reads
a wall clock or reads the environment, and no pricing entry point accepts a
rank, a worker ID or an adapter identity.

The constant does not erase deterministic contention. The device engine
combines immutable demand with its one capacity registry, admitted resident
state and the closed resource law. The resulting `started_at` to `finished_at`
resource interval may lengthen when co-resident work changes, but identical
model, graph and admission state produces the identical interval. Reports keep
the isolated floor and demand distinct from that realized contended interval;
neither is sampled.

**Rank and runner independence is a statement about the function, not about the
shape.** Two ranks may legitimately carry different shape inputs and therefore
different constants. Uneven expert parallelism is the case already in the
repository: vLLM spreads global experts over the expert-parallel world and gives
the low ranks the remainder, so 30 experts over 8 ranks leaves ranks 0 to 5 with
four resident experts and ranks 6 and 7 with three. Those ranks stream different
weight bytes and their decode steps cost different amounts. That is an input
difference, and the contract is unaffected by it. What the contract forbids is a
provider whose answer depends on who asked.

**Memory-bound kernels are pinned to the HBM bound.** In the roofline provider
a memory-bound estimate is exactly `bytes_moved / (mem_bandwidth * efficiency)`
with no compute term leaking in, and it reports `bound="memory"`. In the
mechanistic `SmSchedulerModel` a kernel whose limiter is the flat per-GPU HBM
cursor takes exactly its cursor occupancy plus the profile's fixed HBM return
latency, and adding SMs changes nothing.

**CUDA-graph launch and eager launch differ only in the host launch cost.** The
COMP-2 profiles already distinguish `turing-cuda-graph` from
`turing-eager-host`, and both compose as `max(C, N * g)` over an unchanged
provider service `C`. The launch class never reaches kernel service time. The
`ideal` host profile contributes exactly zero, so a study with no host profile
selected is reading kernel service time and nothing else.

> Launch-mode ownership is explicit. First-party A100 measurement in the
> [graph launch study](../../examples/a100_graph_launch_v1/RESULTS.md) finds a
> device-side per-kernel cost that is 1.415 to 1.506 microseconds larger in
> eager mode than in a graph, of which a null kernel accounts for 1.080. The
> standing ruling assigns every launch-mode effect to the host launch path in
> this model. COMP-48 identifies the host term without changing kernel service
> or adding a device-front-end stage. The kernel-service clause above is
> unchanged.
>
> That assignment needs a host composition able to carry the term. The
> [host launch composition study](../../examples/host_launch_composition_v1/RESULTS.md)
> shows the shipped `max(C, N * g)` form predicts a launch-mode delta of
> exactly zero whenever per-kernel service reaches the eager per-launch
> constant, which covers every real kernel that study measured, so its
> relative error against the measurement is exactly 1.0 and about 2,000 GPU
> cycles. The zero is structural: it holds for any per-launch constants, so
> installing A100 constants does not change it. COMP-44 owns giving a
> calibrated profile a non-overlappable term, and COMP-48 cannot meet its bar
> before it lands.

**There is no per-kernel tail, and the rationale is that tails are emergent.**
Reported TTFT and TPOT distributions have wide tails in real deployments, and
this model produces them from the network, from batching decisions and from
queueing at contended resources, which is where they physically come from.
Attributing a tail to per-kernel stochasticity would double count: the same
spread would appear once in the kernel constant and again in the queueing that
constant feeds. It would also be unfalsifiable at the metric, because a p99
TTFT can be reproduced by an arbitrary mix of kernel noise and queue noise. So
kernel service time carries a mean-valued constant with an honest uncertainty
for error bounds, and every tail claim is owned by the network, batching and
queueing chain (COMP-9).

**Collective work is the one declared exception**, and it is owned by the
traffic and collective side rather than here. Its destiny is a packetized path
over the GPU's NVLink, xGMI or UALink ports; until then collectives complete
through the deterministic ATLAHS and htsim chain with no-tail constant
completion.

Enforcement is in `tests/test_kernel_determinism.py`, which locks all four
clauses with a mutation control for each, against the fixtures and exact
constants pre-registered by the
[kernel determinism study](../../examples/kernel_determinism_v1/RESULTS.md).

Two limits of that enforcement, so the locks are not read as stronger than they
are. First, the runner-independence evidence is asymmetric: the vLLM executor's
own pricing method is invoked, while the SGLang worker is not importable without
SGLang installed, so its half drives SGLang's own geometry reader into the same
shared call its `_settle` makes rather than invoking `_settle` itself. Second,
the "no random source, wall clock or environment read" check is a static fence
over statically resolvable references: import statements and their aliases,
`from` imports at any relative level, dotted attribute uses resolved through the
alias map, and run-time imports with a constant name, with a computed import
name rejected outright. It cannot see a source reached through a name it cannot
resolve statically, such as a callable passed in as an argument. It is a fence
against introducing one, not a proof that none can exist.

## Fixed per-step host profiles

The fixed-step calibration is scoped to an NVIDIA GeForce GTX 1660 Ti
(`gtx1660-ti-sm75`, compute capability 7.5) on an AMD Ryzen 9 3950X host with
driver 550.90.07 and CUDA 12.4.99. It installs two explicit launch classes:

| Profile | Launch class | Point (ps/launch) | Sample-limited empirical range (ps/launch) |
|---|---|---:|---:|
| `turing-cuda-graph` | `cuda-graph-node` | 809,306 | 624,665 to 809,306 |
| `turing-eager-host` | `eager-host-bound` | 2,364,255 | 2,327,730 to 2,544,074 |

The empirical range is the minimum and maximum of five observations, not a
confidence interval. GPU UUID, host CPU, driver, CUDA version, launch class,
source study and uncertainty kind travel with each profile. The profile point
is a sensitivity constant for this measured Turing device and host only. It
is not a H100 or B100 calibration. Scheduler, sampler and Python-side costs
outside the measured launch classes remain unknown.

The serial step lowerer is the one timing authority. For provider service
`C`, launch count `N` and per-launch point `g`, it computes
`F = max(C, N * g)`. Since GOAL represents whole nanoseconds, calibrated
service is the smallest enclosure `Q = ceil(F / 1,000) * 1,000` ps. The
packet-level sink selects and exposes that same model, while coordinator
dispatch validates that the adapter and sink share it and does not add the
term again. A nonideal profile is rejected on a fallback that has no
host-model-aware timing sink. The default in `SerialStepLowererConfig` and
`HtsimStepSinkConfig` remains `HostInitiationModel.ideal()`, which contributes
exactly zero. Legacy explicit scalar constants retain their historical
additive behavior.

## Trace-driven GPU service boundary

The first SASS service slice models one isolated kernel at a time. Its input
contains stable implementation and trace identities, launch grid and CTA
resource use, plus explicit CTA trace classes. Each class binds a per-warp
instruction stream and dependencies to exact linear block IDs, so edge or
data-dependent CTAs need not be cloned from a representative block. The model
has four replaceable mechanisms:

1. **CTA admission and assignment.** Resident CTAs are limited by the minimum
   of SM block, warp, thread, per-warp register allocation, static and total
   shared-memory capacities. Per-block thread and per-thread register limits
   are checked separately. CTAs are assigned deterministically to SMs as
   capacity becomes available. A launch that cannot admit one CTA fails
   instead of returning a precise-looking duration.
2. **Warp scheduling and SM service.** Ready warps issue through a declared
   number of schedulers and per-cycle issue width. Dependency scoreboards
   preserve RAW and WAW producer ordering. Instruction classes map to
   replaceable latency, initiation interval and execution-port parameters, so
   later calibration can improve tensor, scalar, special-function and memory
   behavior independently. Warp selection is an explicit calibration choice;
   v1 provides deterministic loose round-robin and greedy-then-oldest policies.
   The bootstrap profiles use loose round-robin without claiming NVIDIA's
   undisclosed subpartition policy. The current model handles synchronous
   normalized per-warp instructions only. Barriers, `cp.async`, TMA, warpgroup
   async issue/commit/wait, cooperative launches and thread-block clusters
   fail closed under COMP-10.
3. **HBM service.** Global-memory instructions create explicit byte demand.
   The first slice separates logical lane-request bytes from physical
   transacted bytes, then applies a fixed return latency plus sustained service
   bandwidth to the latter. It reports requested, transacted and serviced
   bytes plus request-instruction count. One flat GPU-wide cursor serializes
   HBM demand across every kernel passed to `estimate_concurrent`, which is the
   first explicit cross-kernel contention mechanism. An input trace may label
   L1, L2 or shared-memory service and receive an explicit fixed latency, but
   v1 does not predict cache hits, partitions or bank conflicts. Those deeper
   mechanisms remain unsupported under COMP-10, not hidden efficiency factors.
   CORE-4 decides which graph operations are released together and arbitrates
   kernel traffic against explicit DMA; the compute model prices the kernel set
   it is given.
4. **Copy service.** A copy descriptor declares direction, endpoints and
   bytes. Isolated service is setup time plus byte serialization in the copy
   engine's own declared clock domain and directional bandwidth. API launch
   delay, engine selection, queue waiting, simultaneous copies, compute/copy
   overlap and shared-HBM arbitration belong to CORE-4. This is external
   device DMA service. In-kernel async copy and TMA are not approximated as
   external DMA.
5. **NVLink egress service.** A store may name the `nvlink` memory space,
   which serializes on one per-GPU egress cursor with its own latency and
   bandwidth, exactly as HBM stores serialize on the HBM cursor. This is
   the intra-node path that keeps NVLink traffic off the fabric backend
   (TRAF-10). It is one flat same-generation egress serializer: peer
   topology, per-link routing, ingress service and reduction lanes are
   absent under COMP-31. A calibration without an `nvlink` profile rejects
   NVLink instructions rather than pricing them as HBM.

### Concurrent task scheduling

`estimate_concurrent` replays several `GpuTask` records on one GPU.
A task is a kernel launch plus a `GpuTaskKind` label (compute, memory or
network) used only for attribution: the replay prices every task by its
instructions and the resources it touches, never by its label. Tasks
share SM residency, per-SM issue budgets, pipelines, the HBM cursor and
the NVLink cursor, and CTAs of a later task backfill capacity an earlier
task cannot use. The result carries the makespan plus per-task admitted
and completion cycles, issued instructions and byte counts.

Each task now also carries logical submission and eligibility cycles. The
concurrent service admits no CTA before eligibility, includes a newly eligible
task in the same deterministic replay as resident kernels, and projects both
input cycles into its per-task estimate. Default zero cycles preserve every
accepted replay. Idle time before the first eligible task advances virtual
time but is not misreported as dependency, pipeline or completion drain.

`simllm.compute.rnic` uses this timed service for optional RNIC submission
production. CPU-proxy mode submits a light GPU descriptor-store and
publication task. GPU-initiated mode submits a WQE-store, doorbell-record
store and publication task. Both use the network task class and contend with
surrounding kernels for SM residency, issue and HBM service. The surrounding
NCCL egress task retains its NVLink cursor and can be delayed through the
shared issue path. Compute completion is resolved against the caller's
submission deadline, then projected into the native RNIC record as an
immutable link. Coupling is disabled by default, and host-CPU mode never
constructs a task or invokes the scheduler.

Replay order is a declared input to this coupling. `RnicProducerCoupling`
passes caller-supplied concurrent tasks first in caller order, followed by
non-host producer tasks in request order. The deterministic baseline scheduler
uses task index to break admission and issue ties. Producer-last order lets the
frozen residency-saturated background claim the full SM before the producer,
which creates the registered +20 and +23 cycle submission delays. Reversing
that order admits the producer first and does not preserve those rows. The
COMP-13 concurrent artifact must therefore serialize and validate the exact
task order rather than reconstruct it from task kind or identity.

NCCL collectives enter through `simllm.compute.nccl`, which builds the
per-GPU egress kernel of a ring all-reduce: `2 * (W - 1) * P / W` bytes
per GPU, chunked across channel CTAs and their warps, each chunk loaded
from HBM and stored to NVLink. This makes a collective a schedulable
kernel like any other, so it contends with compute and memory work
instead of being priced in isolation. Proxy operations, ingress and
multi-ring topologies are COMP-31.

The [task-mix study](../../examples/gpu_task_mix/RESULTS.md) measures
what limits each kind: compute scales with SMs and with the pipeline
initiation interval, memory is pinned to the HBM cursor and gains nothing
from more SMs, and a double-buffered ring egress kernel falls from 6.1
times its own egress bound with one warp per channel to within 2.4 percent
at eight warps. At that point a ring-first run hides a 132-cycle memory task
under its NVLink drain while conserving all HBM and NVLink counters.
The study also ledgers two registration misses that name real shared
resources: concurrent tasks contend for the issue path, and SM residency is
itself contended, so a co-scheduled kernel is free only while the SM has room
for it.

The result reports total cycles and picoseconds together with occupancy,
instruction issue, HBM demand and per-SM counters. Scheduler pressure counts
wall cycles in which an SM exhausts its dispatch budget. Dependency idle and
pipeline idle count whole-SM idle wall cycles; final instruction or memory
completion is reported separately as completion drain. These counters are
model observables, not aliases for Nsight's per-warp stall metrics.
Deterministic replay of the same artifact must be bit-identical. Unknown
opcodes, missing trace identity, impossible residency, unsupported
cooperative or cluster launches, and incompatible copy directions fail
loudly. The model does not infer a SASS stream from the five aggregate
`step_kernels` accounting families; exact per-invocation records remain
COMP-6.

This boundary is deliberately below the online `ComputeProvider` lookup and
above a full device runtime. Provider construction can replay a catalog once,
or an offline run can populate `simllm-profile-table-v1`. `ExecutionGraph` keeps
CUDA streams and dependencies. CORE-4 composes service calls, selects physical
engines, arbitrates resources and determines inter-operation overlap. Neither
package duplicates the other's scheduler.

### Registered mixed-makespan forms

A concurrent makespan is not the maximum of the isolated durations. The
task-mix study measured two reasons, and `decompose_mixed_makespan` names the
terms of a replay that already happened so a study or regression can compare
them. It is a read-only projection of one `GpuConcurrentEstimate` against the
single-task controls of the same architecture, never a second estimator.
`MixedMakespanForm` reports the regime, both physical bounds, the issue delay
and the residency decomposition.

The G1 issue-order form. When every task admits its first CTAs at its own
eligibility cycle, the tasks overlap and the makespan is

```text
T_mixed = max(isolated durations) + delta_issue,
```

where `delta_issue` follows the actual ordered tuple the caller submitted.
For the frozen 8-CTA memory and NVLink egress pair, memory-first measures 329
cycles against a 328-cycle egress control and network-first measures 328. The
delay survives widening the per-SM scheduler budget alone and widening the
load/store issue width alone; only widening both together removes it, so the
binding resource is whichever per-SM issue currency is scarcer. This is not a
label rule: `GpuTaskKind`, priority and a canonical memory-before-network sort
are all irrelevant, and reconstructing the order from any of them would
reproduce the number for the wrong reason.

The G2 residency form. When an SM's shared memory cannot hold both tasks'
CTAs, the second task does not backfill; it waits and then pays its whole
isolated duration:

```text
T_mixed = admitted_cycle(gated task) + isolated duration(gated task).
```

With each CTA claiming half an SM's shared memory the isolated controls are 14
and 229 cycles, the memory task admits at cycle 14 exactly when the compute
task finishes, and the makespan is their 243-cycle sum. Removing the shared
memory demand restores backfill: isolated 7 and 132, makespan 133, i.e. the
maximum plus the same one-cycle G1 term. The admission equality is part of the
form, because a 243-cycle makespan on its own would not identify residency as
the cause.

Submission order is therefore an input CORE-4 owns, not a property the compute
service may infer. `CoarseDeviceRuntime` fixes the membership of a co-runnable
compute group, orders it by repeated arbitration grants, and passes that
ordered tuple to `estimate_concurrent`, so the measured G1 term follows the
order the runtime actually chose. Under the identity policy every grant is the
deterministic baseline sequence, which is `ExecutionGraph` tuple order, and
permuting priority labels changes nothing. A class-aware policy reorders only
legal ready candidates, and
[the arbitrated-order study](../../examples/arbitrated_order_v1/RESULTS.md)
measured the same one-cycle G1 term following that reordered tuple through the
live metric chain.

Both forms are the behavior of the exact frozen fixtures, replicated by
[the mixed-makespan study](../../examples/mixed_makespan_v1/RESULTS.md)
through the component scheduler and through the live CORE-4 metric chain.
Neither extrapolates to other shared-memory fractions, launch shapes,
instruction mixes or GPU architectures, and the synthetic 1 GHz profile is a
mechanism fixture rather than any silicon calibration.

## Offline device calibration

Device calibration has one evidence path and one compact serving path. The
offline path captures production executions on real devices, qualifies any
optional simulator observations, fits and validates a model, and publishes a
small immutable artifact. The serving path loads that artifact and performs
deterministic resolution and service lookup. It never imports a profiler,
launches a subprocess, replays an instruction trace or invokes Accel-Sim.

`simllm-execution-graph-v1` remains the only runnable DAG format and remains
byte-for-byte unchanged. Its exact canonical unbound JSON bytes produce
`instance_graph_sha256`, which joins capture evidence only. In that preimage,
every `ComputeWork` retains its semantic fields and serializes both
`nominal_duration_ps` and `uncertainty_fraction` as explicit `null` members.
The calibration canonical writer encodes the exact JSON object returned by the
strict graph-v1 serializer. Because graph v1 permits finite float config
scalars but the calibration canonical grammar deliberately has no binary-float
spelling, a graph admitted for calibration binding rejects every float-valued
`ComputeWork.config` member. Integer, Boolean and string config members remain
valid; the existing graph-v1 reader is unchanged.
A separately
versioned `simllm-execution-graph-template-v1` projection is written by the
calibration canonical writer. Operations receive integer ordinals in original
graph tuple order; distinct ranks map to `0..n-1` in ascending source-rank
order; within each normalized rank, distinct logical queues receive ordinals in
first-operation occurrence order. Every dependency and completion frontier is
rewritten to those operation ordinals. Its `template_graph_sha256` groups
equivalent structures and defines held-out splits; it never selects service.
The strict top-level record contains exactly `schema`, `operations`,
`completion_operation_ordinals` and `collective_plans`, including an empty
collective tuple. Operations preserve source tuple order, and array position is
the operation ordinal. Each operation contains exactly `rank_ordinal`,
`logical_queue_ordinal`, `priority`, `work`,
`depends_on_operation_ordinals` and
`participant_local_depends_on_operation_ordinals`. Dependency tuples are
sorted unique ordinals. An empty source completion frontier normalizes to all
operation ordinals; an explicit frontier becomes a sorted unique tuple.

`work` is a closed strict union. Its variants are exactly `{kind: "compute",
kernel}`, `{kind: "kv-cache", action}`, `{kind: "dma", source_role,
destination_role}`, `{kind: "collective", collective, algorithm_hint,
rank_ordinals, channel_ordinal, pair_rank_ordinals}` and `{kind: "control",
mode, message, destination_rank_ordinals}`. Nullable `algorithm_hint` remains
present. Collective and control rank tuples preserve source order. Each sparse
collective pair is exactly `[source_rank_ordinal, destination_rank_ordinal]` in
the source aggregate-pair order with its bytes removed; an empty pair table
stays empty. Effective collective channels use `channel_hint` or the canonical
`default` value and receive graph-global ordinals by first collective-operation
occurrence, preserving channel equality while removing spelling. DMA uses
source and destination roles from the
`simllm-device-endpoint-role-v1` normalizer. It accepts `host`, `host:pinned`,
`host:pageable`, `gpu:<rank>`, `gpu:<rank>:hbm` and `cuda:<rank>`; GPU rank
tokens rewrite through the same rank map and must resolve. Every other endpoint
role rejects the projection rather than entering a hash.

Each projected collective plan contains exactly `operation_ordinal`,
`algorithm`, `channel_ordinal`, `rank_order`, `rounds`, `actions`, `extents`,
`entry_action_ordinals` and `terminal_action_ordinals`. Plans preserve source
tuple order, which graph validation aligns to collective-operation order.
`rank_order` contains normalized rank ordinals while preserving its source
order.
Their `channel_ordinal` repeats the semantic collective's effective channel as
a loss check. Round transfer channels use a separate graph-global ordinal
namespace assigned by first plan and round occurrence because they identify a
different traffic resource. A round contains exactly
`{transfer_channel_ordinal}` and its array position is its ordinal. An action
contains exactly `{rank_ordinal, kind, extent_ordinal,
depends_on_action_ordinals}` and its array position is its ordinal. An extent
contains exactly `{round_ordinal, source_rank_ordinal,
destination_rank_ordinal, send_action_ordinal, receive_action_ordinal}` and its
array position is its ordinal. Entry and terminal maps contain exactly
`{rank_ordinal, action_ordinals}` in plan rank order, with each action tuple
sorted. Every reference is rewritten and total.

The normalized rank map uses the ascending union of every operation anchor,
collective participant, control destination, accepted DMA endpoint and plan
rank. The projection excludes execution, step, operation, queue, channel,
action, extent, request, pool, block, descriptor and correlation identities;
release and not-before timestamps; placement epochs; compute config, FLOPs,
HBM, duration and uncertainty; every other KV field; DMA bytes; collective,
control, plan and extent payloads; request attribution; round tags and indices;
and integrity hashes. Changing only an excluded value or consistently renaming
an ordinalized identity leaves the hash unchanged. Dependency and completion
tuple permutations, empty versus explicit-all completion, and null versus
`default` channel spelling are invariant. Changing retained source tuple order
other than the explicitly sorted dependency, completion and frontier-set
fields, priority, dependency scope, effective completion, retained rank or
queue equivalence, work kind or family, DMA role, collective rank order,
sparse-pair support, channel sharing, plan presence or any retained round,
action, extent or frontier edge changes it. Rank relabeling is invariant only
when it preserves source-rank order. Projection is idempotent, leaks no raw
identity, rejects an
unknown role or unresolved reference, and never selects service. These are
validation guards, not examples.

Observed and synthetic binding are distinct:

- A physical capture records the implementation reported by the framework and
  profiler in `OperationImplementationBinding`. No selector may rewrite that
  observation. A supported physical graph lowers every noncollective physical
  launch to its own compute operation with explicit order and dependencies. A
  noncollective semantic operation that still hides two or more launches is
  unsupported rather than receiving several service entries.
- Physical NCCL and RCCL launches remain internal stages of their parent
  semantic `CollectiveWork`. Capture records each in
  `CollectiveDeviceStageBinding`, with exact plan hash, rank and launch ordinal,
  and proves that the observed stage set is complete. The traffic-owned
  `CollectivePlan` remains the only algorithm, chunk and transfer expansion.
  Online version 1 resolves exactly one stage per rank plus its copied rank
  frontier; device service owns its SM and HBM interval, while runtime and
  traffic emit only the parent collective's graph completion and retain chunk,
  peer-port and mover timing. A multi-stage rank is unsupported, and no
  physical stage is also scheduled or charged as a `ComputeWork` node.
- A simulated execution normalizes semantic work through a registered
  `ShapeSchema`. Framework, backend, kernel-library version, numeric and layout
  support are frozen in the selected device-model envelope; launch mode is
  excluded from kernel dispatch. A model-owned selector then resolves semantic
  work and typed shape to one `ImplementationRef`. It produces a fresh total
  `ResolvedOperationServiceBindingSet` for that graph before scheduling starts.
- An implementation reference is structured and content addressed. A target
  implementation names vendor, ISA, module or code-object hash, function or
  code hash, backend or algorithm identity and a trusted launch-template
  identifier in `launch_formula_id`. An analytical reference instead names a
  declarative model hash and target applicability;
  it is allowed only for the explicit architecture-derived COMP-52 path.
- Calibrated mode requires exactly one binding for every compute operation,
  rejects extras and forbids kernel-name fallback. If one dispatch signature
  maps to multiple observed implementations, the context is incomplete and
  validation rejects it rather than choosing arbitrarily.

Within one envelope, semantic work and typed shape may select different
implementations. Envelope labels validate applicability; they are not service
inputs. Once `(implementation_id, shape_vector)` is fixed, framework, adapter,
backend, library and launch-mode labels cannot change service. This preserves the
kernel-time contract: the selected device model fixes its implementation
selector, and repeated family, phase and shape inputs on that profile resolve
the same constant.

The measurement campaign makes the three workload strata physical, rather
than treating their labels as model inputs:

| Stratum | Required production measurements | Boundary |
|---|---|---|
| Compute | High-arithmetic-intensity kernels over shape, occupancy and SM-pressure sweeps; FLOPs, pipeline issue, residency and separate host-launch observations | Kernel service and device demand only |
| Memory | KV, attention and streaming kernels over bytes, cache state and concurrency; requested and transacted HBM bytes, cache counters and HBM service | Enforce FLOPs/peak and compulsory-bytes/HBM floors |
| Communication | GPU-resident NCCL or RCCL kernels over payload, participants and channels, isolated and co-running with compute and memory; SM, HBM, peer-egress and observed mover demand | Excludes collective expansion, wire serialization, RNIC queues, congestion and FCT |

Every full or communication-validated campaign contains isolated controls for
all three strata, every pairwise mixture, a three-way mixture and held-out
physical framework graphs. A scalar compute-memory campaign may omit
communication only by declaring that capability unsupported. Splits are by
shape and graph identity, never repeated launches of the same cell.
Graph-level silicon makespan and critical path are holdout oracles. Resource
labels remain attribution only: relabeling an otherwise identical task does
not change service.
Before observation, a mixed cell's physical floor is exactly the maximum of
the applicable isolated floor for every authored member copy. Width changes
copy multiplicity but cannot reduce that maximum. Its finite ceiling is the sum
of the applicable member ceilings over every authored copy. The deliberately
serialized same-members control is a separate measured upper bound checked
after observation, never the pre-observation ceiling, because a
pre-observation bound may never borrow a measured value.

Selection is explicit and fail closed. An exact silicon point wins, followed
by a point covered by an existing validated silicon fit. Per-entry fit support
is a declared subset of the train-defined target envelope and may be exact-only
or exclude a categorical implementation region. A qualified Accel-Sim
observation may fill only a declared missing exact kernel cell inside the
train-defined, silicon-validated SM80 target envelope and between the required
real anchors when no validated silicon fit covers that cell. It records
`coverage-gap` as its reason and carries its measured calibration residual. If
a valid silicon fit later covers the cell, that fit outranks the simulator. An
analytical roofline is used only when the selected policy
explicitly permits it. Otherwise lookup fails. Sources are never averaged or
blended silently.

The optional Accel-Sim sidecar is an untouched upstream checkout pinned to
[`3016c658f810bdae9a14bf4534ee99e9945eedae`](https://github.com/accel-sim/accel-sim-framework/commit/3016c658f810bdae9a14bf4534ee99e9945eedae).
That upstream development snapshot is selected because the latest official
release, [v1.3.0](https://github.com/accel-sim/accel-sim-framework/releases/tag/v1.3.0),
does not contain the A100 configuration. All SimLLM wrappers, configurations
and contributor tooling live outside the upstream tree. The associated
upstream A100 archive is pinned at statistics commit
`ee21104be44ad55dfde789111d3b94372be8435f` and GPGPU-Sim commit
`6c3cf4ff32110908386d605a7034fc67666a92de`. Its site-local A100 hardware and
trace inputs are not publicly fetched, so archive-golden conformance is the
initial gate. Exact upstream reproduction waits for those inputs to be obtained
and hash locked; a fresh project capture is labeled project reproduction.

| Target or use | Real-device evidence | Accel-Sim use |
|---|---|---|
| A100, SM80 compute and memory | Required silicon anchors | Optional missing-region fill only after correlation qualifies the exact tool and hardware envelope |
| H100 or later NVIDIA ISA | Required | Rejected until an upstream target is independently qualified |
| AMD ROCm device | Required; scalar compact profiles may publish before vendor peer ports | Rejected because the CUDA, NVBit and NVIDIA SASS path does not support ROCm |
| Communication and data movers | Required on the target stack | Rejected for every communication-stratum observation, including GPU demand, wire service, congestion and FCT |
| Serving loop or online model build | Compact model only; validated by default, candidate only under explicit experimental opt-in | Rejected |

The capability-matrix name `amd-rocm-target` denotes one parameterized campaign
slot rather than one GPU. Before measurement, a campaign binds it to one exact
immutable target ID and content-addressed envelope with SKU, architecture or
ISA, driver, ROCm, profiler and collective-library identities. All rows in one
claim use that same binding. A different target has a separate campaign,
denominator and model identity; rows from two AMD targets cannot satisfy one
claim together.

Contributor uploads are deterministic small evidence releases, not executable
code submissions. The offline tool validates schema versions, duplicate and
unknown fields, finite numeric domains, content hashes, safe relative paths,
archive count and size limits, total bindings, immutable split isolation,
physical floors, licenses and source provenance before producing a reviewable
model diff. Raw traces stay outside Git. Candidate status never implies
validated status, and promotion requires an untouched held-out run plus live
TTFT or TPOT provenance. The complete roadmap and ownership are summarized in
[Offline device calibration](../design/offline-device-calibration.md).

## Seed profiles and calibration ledger

`GpuArchitectureProfile` contains structural limits. Its swappable
`GpuCalibrationProfile` is explicitly bound to one target architecture profile
and contains the target core and optional memory clock, instruction/pipeline
timing, memory timing and bandwidth, warp selection, copy-engine timing,
provenance and uncertainty. The provenance GPU may identify a transferred
evidence source, e.g. H800 timing used as an H100 prior, without changing the
target identity. Recalibration therefore leaves architecture and trace
identity unchanged, and attaching an A100 calibration to an H100 structure
fails at construction.

The A100 SXM 80 GB and H100 SXM 80 GB profiles are bootstrap artifacts. Their
documented occupancy limits and SKU peaks come from NVIDIA's
[Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html),
[Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html),
[A100 data sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf)
and [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/).
Instruction and memory context comes from the open
[Ampere study](https://arxiv.org/abs/2208.11174),
[Hopper/H800 study](https://arxiv.org/abs/2402.13499), and the later
[A100/H800 microbenchmark study](https://arxiv.org/abs/2501.12084). The
numeric memory-latency priors are transferred from the last study. Its Hopper
device is H800 PCIe, not H100 SXM, and its A100 measurements are not treated as
an exact A100 SXM 80 GB match. The profile provenance says so and assigns 50
percent relative uncertainty. A public peak is a capacity constraint, not a
claim that an arbitrary kernel reaches it. Public documentation does not
expose a complete copy-engine timing or selection contract, so the seed
profiles intentionally contain no copy engines until capture supplies them.

The optional A100 SM80 gap-filling lane uses the
[Accel-Sim paper](https://doi.org/10.1109/ISCA45697.2020.00047) and
[Accel-Sim framework](https://github.com/accel-sim/accel-sim-framework) for
external SASS replay and counter correlation. Every target-silicon campaign
must close the applicable production ledger below before COMP-1 can close;
only a simulator-filled A100 campaign must also close the Accel-Sim fields.
H100 and AMD use their target-native capture evidence and have no Accel-Sim
dependency:

| Component | Required evidence |
|---|---|
| Run envelope | framework and commit, model, exact GPU SKU and UUID, driver, CUDA, libraries, dtype/quantization, eager or graph mode, numeric observed core/memory clocks, lock policy and warm-up policy |
| Kernel identity | binary and function hash, semantic operation, launch order, stream, grid/block dimensions, registers, static/dynamic shared memory, cooperative/cluster flags |
| Instruction and scheduler | target-native tracer/version and trace or code-object hash where supported, warp/wave and CTA/workgroup identities, instruction classes and dependencies, elapsed cycles, eligible/active execution units, issue utilization and stall reasons; an A100 simulator fill additionally names the qualified SASS/Accel-Sim records |
| Memory | requested and transacted bytes, cache hit/miss counters, HBM throughput, latency probes, cache-state protocol and memory-clock state |
| Copy | API kind, direction and endpoints, bytes, stream/event order, reported device engine capabilities, setup samples, sustained bandwidth and concurrent-copy experiment |
| Fit | immutable train/held-out split, raw samples, sample count, fitted parameters, residuals by component, uncertainty and creation date |

The v2 artifact enforces the capture environment, model/GPU identity,
framework/tool/library versions, clock and warm-up policy, numeric observed
core and memory clocks, hashes, semantic attributes, launch resources,
CTA/warp traces, stream order, requested and transacted bytes, copy
direction/endpoints, raw duration samples, deterministic replay, split and
residual. A captured artifact must use the calibration's exact core and target
memory clocks; seed calibrations without a numeric memory-clock target cannot
claim captured measurements. This legacy import-only schema stays frozen
without profiler cache counters, per-warp eligible/active samples, 3D launch
coordinates or concurrent-copy experiments. The production
`simllm-device-calibration-bundle-v1` carries those typed records under the
COMP-1 and COMP-10 acceptance boundaries. Bulk counter exports remain
content-addressed outside Git.

The ledger keeps structural facts separate from fitted timing parameters. A
future capture can replace instruction latencies, throughput corrections,
cache/HBM behavior and copy parameters without changing the trace, service or
provider interfaces.

### Artifact boundary

`simllm-gpu-model-artifact-v2` remains a strict compatibility and import record
for the existing internal SASS replay model. New physical captures, external
simulator observations and concurrent evidence use
`simllm-device-calibration-bundle-v1`; new compact releases use
`simllm-device-model-v1`. The legacy artifact complements
`simllm-profile-table-v1`: it keeps one internal replay auditable, while the
profile table remains a supported scalar online lookup surface. The reader
promotes v1 artifacts by
renaming the clarified per-SM completion counter and filling the absent NVLink
profile and counters with `null` and zero; compatibility writers continue to
emit v2 byte for byte. A GPU-model artifact retains:

- the architecture-profile identity, exact SKU, structural limits, fitted
  parameter set, source links and declared uncertainty;
- capture envelope and calibration provenance, including framework,
  toolchain, tracer/simulator versions, observed core/memory clocks and
  creation date;
- SASS trace identity, kernel binary/function identity, semantic catalog key,
  launch shape and resource declaration;
- simulated cycles and picoseconds, replay counters, occupancy, issued
  instructions, per-SM idle/pressure/drain counters, and requested,
  transacted and serviced HBM bytes;
- explicit copy transfers and service replays with direction, endpoints,
  selected engine, independent clock domain and stream order;
- measured duration samples, sample count and summary statistics when silicon
  measurements exist; absent measurements remain explicitly absent rather
  than being synthesized from the model;
- immutable train or held-out split and the fitted residual/uncertainty when
  the artifact participates in calibration.

The strict loader normalizes hash spellings, rejects duplicate semantic keys
and stream orders, recomputes sample summaries, checks capture split isolation,
and reruns every deterministic kernel/copy estimate before accepting it.
Changing an identity, source, fit or split produces a new artifact. Small
synthetic fixtures may live with tests and studies. Raw production SASS traces,
profiler exports and bulk replay outputs live under the external root
configured by `SIMLLM_DATA_ROOT`, never in
Git; the public artifact records their content hashes and provenance.

## COMP-1: offline device calibration plan

Strictly offline; the step loop never invokes a cycle-level simulator.

- Anchor every target on its own silicon capture. Target-native measurement is
  the evidence for A100, H100 and AMD alike. Only the A100 lane may add a
  qualified SASS replay, and only to fill a declared missing exact cell inside
  the silicon-validated SM80 envelope. Raw cycle-simulator output is
  never treated as silicon truth. Pin a support envelope for every table:
  framework and commit, model, GPU architecture, CUDA/toolchain, dtype and
  quantization, eager or CUDA-graph mode, kernel implementation, tensor
  parallel width, batch/new-token/context shapes, KV dtype and MoE shape.
  Unsupported combinations miss loudly rather than borrowing a precise-
  looking number.
- Capture the exact production run first. Nsight/CUPTI metadata records
  kernel identity, launch order, streams, shapes and silicon durations. On
  the A100 lane only, NVBit supplies the SASS traces the sidecar needs; the
  H100 and AMD lanes stop at this capture. Key table entries
  by kernel binary/hash plus the semantic shape, not by a family label alone,
  so a framework or compiler kernel change invalidates the correct entries.
- Build one replayable microbenchmark per captured kernel implementation.
  It must reproduce launch parameters, tensor layout, dtype, workspace,
  stream/graph mode and relevant cache state. Sweep the captured shape axes,
  not synthetic square GEMMs that the real framework never launches.
- On the A100 lane, replay traces offline with the pinned Accel-Sim and
  GPGPU-Sim configuration inside its qualified SM80 support region. Every
  other target skips this step entirely and fails closed on a simulator
  request. Fit and report calibration residuals
  against silicon using train shapes, then evaluate held-out shapes. Launch
  overhead, host delay and queueing are measured separately from kernel
  service, so the SASS table cannot hide a missing runtime queue.
- Populate `simllm-device-calibration-bundle-v1` with capture and kernel
  hashes, target identity, tool versions, typed shape, measured samples,
  optional simulated cycles, calibrated duration, uncertainty, immutable
  split and contributor-supplied creation date. Compile it into
  `simllm-device-model-v1` and, where scalar service is sufficient, an existing
  `simllm-profile-table-v1`. Every output retains a per-entry evidence ledger;
  changing an identity field produces a new content-addressed record.
- Initial acceptance bars, to be tightened from evidence: 100 percent kernel
  identity coverage for the supported run; the stability bar below; held-out
  per-kernel median absolute percentage error below 10 percent and p95 below
  20 percent; per-phase median below 5 percent and p95 below 10 percent;
  compute-only step error below 5 percent. Every miss is reported, never
  averaged away.
- Stability bar, environment-scoped. In a **controlled** environment, defined
  as a non-display device with locked application clocks and exclusive compute
  access, the bar is the original one: measured coefficient of variation below
  2 percent over every sample of a cell. That remains the bar the production
  target-architecture capture must meet. In an environment explicitly declared
  as a **shared display GPU without clock control**, a cell is stable when its
  excursion-trimmed coefficient of variation is below 2 percent, its excursion
  fraction is below 10 percent of the cell's samples, and its maximum excursion
  ratio is below 1.35, where an excursion is a sample above 1.05 times the cell
  median. Every cell additionally reports its all-sample coefficient of
  variation and its full excursion census; no sample is ever discarded from the
  artifact. The
  [fidelity study](../../examples/compute_fidelity_v1/RESULTS.md) froze this
  form before evaluating it, and measured why the second form is the one that
  identifies kernel service-time stability on a display GPU: across the tracked
  Turing capture, 7 samples out of 2,050 exceed the excursion threshold, one in
  each of 7 cells, and the three cells that failed the all-sample bar have
  trimmed coefficients of variation of 0.172, 0.212 and 0.842 percent. A fresh
  4,000-launch probe of the worst of those cells attributes 93.4 percent of its
  excursions to longer block residency at an unchanged 1,869 MHz effective SM
  clock, and the remainder to clock-state drops to 76.9 percent of that clock.
  Neither is kernel service-time variation, and neither is removable without
  the administrator action COMP-5 requires.
- Fixed per-step cost. Kernel service time is not step time. A modeled step is
  exactly the sum of its kernel service: `RooflineProvider` returns 0 ps for a
  zero-work kernel and uses a homogeneous roofline formula above that, with its
  public integer-picosecond result subject to rounding,
  `ProfileTableProvider` returns a measured kernel duration, and
  `HostInitiationModel` is a per-send network initiation delay rather than a
  per-kernel launch cost, so nothing in this package prices kernel launch,
  scheduling or sampling. The fidelity study bounds what that omission is worth
  for a 24-layer top-8 MoE decode step: 440 to 567 device-visible launches in
  eager mode, at a Turing-measured 630 ns per CUDA-graph node, 1,603 ns of
  device-side inter-kernel gap, or 2,332 ns per host-bound eager launch, which
  leaves an omitted excess of 1.79 to 12.31 times the whole modeled decode
  compute of that step. The launch count is a property of the model geometry
  and the framework rather than of the GPU, but the constant itself is Turing
  evidence and does not transfer. At 440 launches, the omitted excess remains
  at least one modeled compute only above 451.7 ns per launch and disappears at
  or below 225.8 ns. Calibrating the production constant on
  the target architecture belongs to this task's "launch overhead, host delay
  and queueing are measured separately from kernel service" clause, and no knob
  is added to the step path until it is measured.
- Simulator starting point: the official Accel-Sim development commit
  `3016c658f810bdae9a14bf4534ee99e9945eedae` in SASS trace-driven mode with
  its pinned GPGPU-Sim and compatible NVBit tracer. The pin is qualified only
  for the supported SM80 region against silicon anchors. The official v1.3.0
  release lacks that A100 configuration, while H100, later NVIDIA ISA and AMD
  ROCm targets have no qualified path and reject simulator use.
- Architecture-derived targets are separate COMP-52 candidates. They use a
  content-addressed declarative analytical implementation, an accepted anchor
  and explicit parameter deltas with inflated uncertainty. They are never the
  default, never described as Accel-Sim-backed and never promoted without
  target-specific validation.
- Hard dependency (COMP-5): local CUDA 12.4 and CUPTI activity timing work on
  the GTX 1660 Ti with driver 550.90.07. Nsight Compute attaches but returns
  `ERR_NVGPUCTRPERM` because the loaded driver has
  `RmProfilingAdminOnly: 1`; no performance counters are collected. The
  display GPU also produced isolated timing outliers above the original
  all-sample stability ceiling, and the fidelity study identified their two
  mechanisms: blocks resident longer because the desktop shares the SM, and
  discrete drops of the effective SM clock. Both need permissions this project
  does not have, so the controlled-environment form of the stability bar cannot
  be met here at all. Production closure needs counter permission, a non-display
  device with lockable clocks, and allocation on the exact target architecture
  with target-native activity and counter evidence. A simulator-filled A100
  campaign additionally needs the qualified dynamic-SASS and Accel-Sim path;
  H100 and AMD do not. The
  [A100 environment qualification](../../examples/a100_environment_qualification_v1/RESULTS.md)
  now proves that one Merlin A100 allocation supports CUDA activity, basic
  performance counters, static SASS and exact environment provenance. It does
  not yet prove controlled-cell stability, dynamic tracing, Accel-Sim replay
  or a production kernel.

## GPU device composition and typed ports

The NIC has been a device with typed ports since BACK-18. The GPU now is too:
`GpuDeviceConfig` composes an architecture profile with typed ports over the
two link mechanisms that already exist, and adds nothing to their timing. The
design statement is
[the packet-device model](../design/packet-device-model.md); the validated slice
is [gpu_device_ports_v1](../../examples/gpu_device_ports_v1/RESULTS.md).

A port carries protocol (`pcie`, `nvlink_c2c`, `nvlink`, `xgmi`), role (host
link or peer link), direction (ingress, egress or bidirectional, relative to the
GPU), declared capabilities, and a ceiling with the provenance of that ceiling.
The mechanism behind a capability stays authoritative: `copy_engine_transfer`
names the per-direction `CopyDirectionProfile` entries of one `CopyEngineProfile`
and `peer_store_egress` names the flat `NvlinkProfile` egress cursor.

Four rules make the port layer safe to add under a byte-identical off path.

1. **Reading a ceiling is not declaring one.** A port with no declared ceiling
   reads its ceiling out of the mechanism and reports
   `calibration_derived` provenance. A device whose ports declare no ceiling
   returns the input architecture object itself, so every accepted timestamp,
   counter and byte count is reproduced by object identity rather than by
   equality. A declared ceiling replaces the mechanism parameter for the
   directions that one port carries, and only those; the derived architecture is
   renamed (`<profile>+<port>@<value>bpc`) so no artifact can claim the base
   profile identity while carrying a rescoped parameter.
2. **A disabled port is a declaration that is absent, not a mechanism that is
   off.** Disabling a port never rescopes the copy engine or the egress cursor.
   The port keeps its interface and is still reported with `not_applicable`
   applicability, its own parameters are inert, and every request made of it is
   rejected with a diagnostic naming it. A disabled port carrying a declared
   ceiling is itself a configuration error.
3. **One mechanism has one port authority.** Two enabled ports may not claim the
   same copy direction of the same engine, and two may not claim the one
   per-GPU egress cursor.
4. **Anything without a mechanism fails closed at configuration time.** A
   peer-store port on a calibration with no `nvlink` profile, a copy direction
   the engine does not declare, an unknown engine, a `device_to_device` copy
   (which stays inside one GPU and crosses no port), an xGMI port (COMP-35 owns
   vendor instantiation), a transport-control capability such as ECN marking
   (BACK-48 owns making the ABI v2 packet vocabulary reachable from a non-wire
   port), and a single bidirectional port over two disagreeing mechanism
   ceilings are all rejected during configuration rather than at first use. The
   last of those is why the measured Grace C2C asymmetry, 419.93 GB/s inbound
   against 169.96 GB/s outbound, has to be declared as two ports instead of one
   averaged rate.

The ports declare and negotiate; they do not emit packets. Carrying an extent
and attempt identity in the ABI v2 vocabulary across a non-wire port is BACK-48
with COMP-40 as its compute-side half, and attaching measured per-port ceilings
to a shipped profile is COMP-41.

## Status

The module exposes one vendor-neutral offline calibration contract from real
framework DAG capture through a content-addressed evidence bundle to a compact
deterministic device model. The model records typed shape and implementation
selection, exact source provenance and a support envelope; optional Accel-Sim
use is isolated to qualified A100 gaps and is absent from online execution.

The kernel-time determinism contract above is stated publicly and enforced. The
pre-registered
[kernel determinism study](../../examples/kernel_determinism_v1/RESULTS.md) is
nonvoid: all 23 fatal guards held, all 3 controls discriminated, and all 8
frozen scored instances passed with a zero residual, with 5 derived rows and 8
raw observations reported separately and never added in. It fixes the exact
prefill and decode constants of its own fixture, shows the memory-bound pin on
both the roofline and the SM-scheduler paths (including that the pin does not
notice SM count), and shows the vLLM and SGLang readers pricing one step to the
identical picosecond. Its findings are that the contract constrains the pricing
function and not the per-rank shape assignment (an uneven expert split is an
input difference, not a violation), that COMP-9's original per-kernel
distribution scope is refuted rather than unfinished, and that the two adapter
readers store two optional dtype fields differently while resolving them
identically, which is COMP-42. The study makes no silicon claim, prices no
collective, and validates no tail: locating the tail is COMP-9, which is open.
A static import and attribute-reference audit found no random source, wall clock
or environment read reachable by a statically resolvable name anywhere in
`simllm/compute`, so the guard that forbids them is a fence rather than a fix.
Review widened that audit after showing its first form could be stepped around
by a bare `numpy` import with a `numpy.random` use, by
`importlib.import_module` or `__import__`, or by a relative import; its residual
blind spot, a source reached through a name that cannot be resolved statically,
is stated with the contract above rather than left implied.

Both providers, the transformer step model (fused and family-decomposed),
the host model, and the trace-driven GPU service are implemented and
tested. The service covers isolated-kernel replay, copy descriptors, the
NVLink egress cursor, concurrent multi-task scheduling
(`estimate_concurrent`) and the NCCL ring-collective builder. The
[service-model study](../../examples/gpu_service_model/RESULTS.md) validates
22 post-specified exact-oracle rows to zero-cycle residual, and the
[task-mix study](../../examples/gpu_task_mix/RESULTS.md) reports 36 passing
exact-oracle rows and 6 passing behavioral relation families over 17
instances. Its 21 structural invariants are unscored, and its two superseded
registration misses remain visible as the chronology behind findings G1 and
G2. Those two findings now have registered forms: the
[mixed-makespan study](../../examples/mixed_makespan_v1/RESULTS.md) replicates
them through the component scheduler and through the live CORE-4 metric chain,
passing 11 genuine-risk instances across four families with all 124 fatal
guards holding. Its residuals are COMP-24 (the forms cover one fixture and one
residency-gated task), COMP-25 (no production step path selects the concurrent
kernel service) and CORE-49, which closed with
[the arbitrated-order study](../../examples/arbitrated_order_v1/RESULTS.md):
the co-runnable group is now ordered by repeated arbitration grants rather than
by graph order. The built-in
A100/H100 profiles are unvalidated bootstrap seeds and do not establish
production accuracy: their pipeline initiation intervals are derived from
published per-SM unit counts, not measured.

The GPU device composition entry point with typed PCIe and NVLink ports is
landed and closes COMP-34. The
[device-port study](../../examples/gpu_device_ports_v1/RESULTS.md) passes 11 of
11 scored instances across four families with all 54 fatal guards holding: a
declared host-link ceiling moves the job completion time of a `DmaWork`
descriptor through the live CORE-4 chain by the exact registered amount, a
declared peer-link ceiling moves the NVLink egress term of the accepted task-mix
cells onto values that study already published, the override never leaves the
direction its port carries, and every accepted `gpu_task_mix`,
`gpu_service_model` and `mixed_makespan_v1` artifact reproduces byte for byte
through the composed device with default ports, locked by
`tests/test_gpu_device_ports.py` with a mutation control per artifact. Four
further identity-path cells are retained as an unscored baseline register, which
is the correction the study's own correction section records against its first
publication of 15 of 15. Its
residuals are COMP-40 (the ports declare capabilities but emit no packet event)
and COMP-41 (no shipped profile carries a measured per-port ceiling). Finding F1
of that study is a constraint on later registrations: halving the egress ceiling
of the accepted ring cell added the full serialization delta with nothing hidden
by overlap, because at eight warps per channel the kernel is already within 101
cycles of its own egress bound. Finding F3 is a constraint on how a freeze is
written: entailment has to be checked per parameterized instance, because a
relation can be unlosable in some of its cells and genuinely at risk in others.

The
[A100 environment qualification](../../examples/a100_environment_qualification_v1/RESULTS.md)
is `QUALIFIED` at SimLLM commit
`3c829c660ec6d48a627447632ee99bd40f001784`. One nonexclusive Merlin
allocation exposed exactly one A100 SXM4 80 GB, stable disabled MIG state, no
foreign process, a nonempty Nsight Systems CUDA trace, numeric Nsight Compute
basic counters and static `sm_80` SASS. This establishes the capability gate
for an A100 production study. It populates no profile table, transfers no
duration to H100 or B100, and leaves dynamic NVBit capture, Accel-Sim
compatibility and registered-cell stability unproven.

The [Turing calibration study](../../examples/compute_calibration_v1/RESULTS.md)
lands the first real activity-timing pipeline and populated table. On the
available GTX 1660 Ti it captured 50 family, dtype and shape cells with 2,050
target samples. Held-out calibrated median and p95 error were 0.674 percent
and 1.773 percent versus 17.782 percent and 25.069 percent for the roofline
bootstrap. The frozen study is nevertheless an overall failure: isolated
high-duration samples put 3 of 50 final cells above the 2 percent coefficient
of variation ceiling, and the preceding post-fix capture missed 2 of 50.
These Turing numbers validate the method and do not transfer to Hopper.

The [fidelity study](../../examples/compute_fidelity_v1/RESULTS.md) is void with
findings because frozen fatal guard XFER-G4's exact proportionality predicate
failed by a 1 ps integer-quantization residual. Its behavioral pass fraction is
therefore uninterpretable. The measurement layer still changes what is known
about the earlier stability failure and the modeled step. Re-reading the same
immutable capture shows the ceiling was failed by 7 samples out of 2,050, one
in each of 7 cells, while the worst excursion-trimmed coefficient of variation
anywhere in the capture is 1.054 percent. A 4,000-launch device probe that
records each block's own cycle span and residency alongside its wall duration
attributes 93.4 percent of a fresh excursion population to longer block
residency at an unchanged 1,869 MHz effective SM clock and the remainder to
clock-state drops to 76.9 percent of that clock, so the tail is the display GPU
rather than the kernel. The stability bar above is refrozen accordingly, with
the original all-sample form retained unchanged for the controlled environment
the production capture must use. The same study measures a fixed per-step cost
whose omitted excess is 1.79 to 12.31 times the whole modeled decode compute of
a 24-layer top-8 MoE step. It registers no new task ID: COMP-1 and COMP-5 both
stay open and keep every clause they registered.

The two A100 calibration studies of the same campaign are void beside it, and
neither closes anything. The
[A100 kernel constants study](../../examples/a100_kernel_constants_v1/RESULTS.md)
violated three stability guards across two runs and deliberately withholds the
`simllm-profile-table-v1` artifact it was built to produce, because a table
from a void run is one a provider would load without noticing. Its retained
evidence is a measured 1818.21 GB/s HBM roof, per-family roofline efficiency
spanning 0.125 to 0.951 where the surrogate is a flat 0.7, captured MoE expert
cells at 5.17 to 12.20 times their own memory roof, a bimodal 1275 and 1410 MHz
SM clock that moves compute constants by the clock ratio and leaves
memory-limited ones still, and a 2.34 microsecond device cost for one CUDA
event placed between two launches. The
[A100 graph launch study](../../examples/a100_graph_launch_v1/RESULTS.md)
violated one dispersion guard and installs neither of the two
`HostInitiationModel` profiles it measured. Its retained evidence separates
host submission, 1,629,633 ps per eager launch against a flat 1.6 microseconds
per graph replay at any chain length, from a device-side per-kernel cost that
is 1.415 to 1.506 microseconds larger in eager mode than in a graph. The
standing ruling assigns that last number to the modeled host launch path,
which is why COMP-48 exists. Neither
study registers a closure; between them they register COMP-43, COMP-44,
COMP-45, COMP-46, COMP-47 and COMP-48.

The [fixed host-step study](../../examples/host_step_cost_v1/RESULTS.md)
re-established that measurement under a corrected freeze before installing
anything. Corrected calibration attempt three was nonvoid and accepted: all
3 genuinely risky relations plus 1 post-specified replication passed (CAL-1,
whose band was widened after the attempt-two miss at 809,068 ps), and all six
fatal guards held. It measured 809,306 ps per CUDA-graph node and 2,364,255 ps
per host-bound eager launch on the declared Turing device, with the empirical
ranges and provenance recorded above. The live `a-ep8-200g` holdout is a
nonvoid end-to-end conformance and reach demonstration with a genuine-risk
denominator of zero and 12 retained entailed rows. Across graph versus eager
launch and 440 versus 567 launches, decode multipliers were 2.2011, 2.6813,
5.3978 and 6.8006; TPOT multipliers were 2.2019, 2.6825, 5.4008 and 6.8047.
Those values show that the installed cost reaches TTFT and TPOT, not that its
magnitude was independently predicted.

The ideal compatibility guard is separate, fatal and unscored. A fresh
five-cell `end_to_end_replay_v1` replay was nonvoid, retained all 13 of that
study's exact-oracle relations, and reproduced its aggregate canonical digest
plus every `steps.jsonl` byte stream. The first calibrated live attempt was
void because repeated per-layer integer floors underrepresented
`max(C, N * g)` by 6,640 to 20,502 ps. The corrected second attempt verifies
the exact whole-nanosecond enclosure, but its magnitude rows are unscored
because fatal exact-row oracles entail them. The held-out third attempt, not
that regression, supplies live conformance and reach evidence but no magnitude
score.

For the mission error budget, item 1 moves from zero to a measured launch
floor only in the device-bound Turing sensitivity. Correlating that launch
term in the simulated and plausible-real expressions leaves a point residual
optimism range of 1.424953 to 3.891039 times; propagating the sample-limited
empirical endpoints gives 1.396964 to 4.508550 times. These ranges assume all
unmeasured scheduler, sampler and Python costs are zero and sit beside, rather
than replace, the mission's generic 5 to 22 times budget. The reference B100
host cost is unknown, so no absolute B100 composed optimism range is supported.
The fixed 99,024,000 ps input is B100-derived. Its 554,631,168 bytes need
1,925,802,667 ps on the Turing device's 288 GB/s roof and 2,751,146,667 ps at
the 0.7 derate, above all four launch floors, so the hybrid rows are not a
device-consistent Turing step prediction. The reported rows use
`network + max(C, N * g)`.

The [composed step budget study](../../examples/composed_step_budget_v1/RESULTS.md)
settled the composition by measurement instead of arithmetic. Running the
mission chain with the host profile and the TRAF-11 collective floor both
enabled shows that the merged code computes
`max(C, N * g) + collective floor + raw fabric`: the launch demand overlaps
provider compute and nothing else. The alternative `max(C + network, N * g)`
reading, which would have given 1.650672 ms for every profile, appears in none
of the study's 93 decode-step observations. Attempt one of that study is void
because one of its own fatal predicates compared a raw provider value against a
quantized literal; attempt two held all ten guards, passed 3 of 3 scored
families, and reproduced attempt one's raw values exactly. A case A decode step
at 400 Gbit/s measures 1.916754 ms at CUDA graph with 440 launches and
2.901192 ms at eager host with 567, against 0.204527 ms with both features
disabled, which the same run reproduced byte for byte. Composition is exact:
over 31 matched decode compositions the two host profiles separate by exactly
984,438,000 ps, the difference of their quantized launch demands, in every
pair. The composition is consistent with the launch count's own registered
exclusion of collective launches, so it is not a defect and no task ID was
registered for it. What the study makes plain is that the modeled compute
contributes zero exposed picoseconds once a calibrated host profile is
selected, because the launch floor masks every provider estimate below it, and
that 94.03 to 96.05 percent of the composed step is transferred constants.

The M5 first slice landed the COMP-1 groundwork: `step_kernels`, the
`simllm-profile-table-v1` artifact with provenance, and 1D log-linear
interpolation (closing COMP-3; the multi-axis extension is COMP-4). The
production SASS pipeline itself (above) has not run yet. Nsight Systems
activity timing works locally, while Nsight Compute counters fail with
`ERR_NVGPUCTRPERM`, the display GPU misses the frozen stability guard and
TU116 cannot supply target-architecture evidence. COMP-5 records those exact
hardware requirements. Therefore COMP-1, COMP-5 and COMP-6 remain open. MoE
geometry
landed with the same slice and is exercised by the examples/m5 studies
together with the MoE traffic mapping
([traffic](traffic.md), [M5 results](../../examples/m5/RESULTS.md)).

COMP-16 is complete. The roofline provider now supplies an explicit opt-in
layer breakdown from the fused step's exact family projection. The
[latent-knob study](../../examples/step_sink_latent_knobs/RESULTS.md) sweeps
two layer counts and two TP widths on the live fluid step sink: every enabled
row moves first-token latency later by the frozen 1,000 ps with zero residual,
while the default path retains the historical GOAL SHA-256 exactly. Profile
table and trace-calibrated layer estimates still require COMP-6 and remain
open as COMP-17.

The COMP-15 first slice is implemented in `simllm.compute.nccl_stack`. Its
function identities were audited against NVIDIA NCCL release `v2.30.7-1`,
commit `73cf112295c33aee2b895f329f592f2a9b4b0f97`. It adds name-mirrored
`ncclCommInitRank` and `ncclAllReduce` entry points and a planner with the same
explicit `2 * (world_size - 1)` ring-step decomposition and strict lane
divisibility as `simllm.compute.nccl`. The send connector follows NCCL's
head/tail convention: the GPU publishes ready state and advances `tail`, while
the CPU proxy advances `head` only after a separately produced network
completion is observed. `ncclProxySaveOp` queues operations before kernel
launch, independent proxy progression permits FIFO occupancy above one, and a
doorbell separates verbs posting from the fake external completion source.

The intra-node route stays inside `ncclKernelMain`, `runRing` and `genericOp`.
The inter-node route traverses the GPU send FIFO, CPU proxy, `ncclNet.isend`,
verbs post, doorbell, external CQE, `ncclNet.test`, CQ poll and head-credit
return. The receive leg is explicitly absent from this slice. Every call,
proactive signal store, and successful poll observation emits a strict
`simllm-nccl-stack-event-v1` record from one caller-supplied `VirtualClock`.
The [NCCL stack skeleton study](../../examples/nccl_stack_v1/RESULTS.md)
reports 5 of 5 passing behavioral relation families over all 35 instances and
10 of 10 fatal unscored structural invariants. This zero-time component stream
is not yet projected onto the live TTFT/TPOT metric chain.

One boundary of that skeleton now carries an opt-in gate. `ncclNetRegMr`
mirrors the net plugin's `regMr` together with the channel FIFO establishment
that follows it, and a communicator built with `require_buffer_registration`
refuses a collective whose destination buffer is not registered on every
channel. The seam declares the registration's one-time cost and, as everywhere
else in this module, never advances the caller's clock. The cost model, the
identity and re-registration rules, and the ledger that spends that cost on the
live metric chain are traffic-owned and are stated in
[the interim collective completion and registration contract](traffic.md#collective-completion-and-registration-the-interim-contract).
Only two claims there rest on the ABI, that a registration entry point exists
and that one seam serves NCCL and RCCL; the one-time charging rule, the
per-buffer identity scope, the channel factor and the three re-registration
events are declared model choices. This gate keeps its own registered-buffer
state, which carries no generation and which the live chain never consults, so
the seam and the traffic-owned ledger are two states that agree by convention
until TRAF-58 unifies them. A communicator that does not ask for the gate emits
exactly the events it emitted before the gate existed. BACK-47 still owns the
device-facing packet emission contract at this same seam.

The same [collective latency floor study](../../examples/collective_latency_floor_v1/RESULTS.md)
closes COMP-11, with its undemonstrated mechanism clauses moved exactly to
COMP-31. The selectable profile replaces the flat local endpoint rate, adds
one participant-indexed base latency at the semantic collective boundary and
reports that base separately from raw fabric transport and the 2.000
microsecond propagation reference. The one-charge and exact identity guards
show that local and fabric projections do not advance or price the same
collective twice, including in a two-node mixed-placement collective with
simultaneous positive local and fabric service. The study does not demonstrate
peer topology, per-link routing, receiving-HBM interaction, reduction lanes or
proxy operations.

### NCCL stack name audit

SimLLM mirrors names and causal boundaries only. It copies no NCCL source.
Every event function is either an audited NCCL symbol or has a `simllm` prefix
and an explicit reason:

| Mirrored event name | NCCL source and symbol, or SimLLM reason |
|---|---|
| `ncclCommInitRank` | `src/init.cc`, `ncclCommInitRank` |
| `ncclBuildRings` | `src/graph/rings.cc`, `ncclBuildRings` |
| `initChannel` | `src/channel.cc`, `initChannel` |
| `ncclAllReduce` | `src/collectives.cc`, `ncclAllReduce` |
| `ncclEnqueueCheck` | `src/enqueue.cc`, `ncclEnqueueCheck` |
| `scheduleCollTasksToPlan` | `src/enqueue.cc`, `scheduleCollTasksToPlan` |
| `calcCollChunking` | `src/enqueue.cc`, `calcCollChunking` |
| `ncclProxySaveOp` | `src/proxy.cc`, `ncclProxySaveOp`; upload call in `src/enqueue.cc` |
| `ncclLaunchKernel` | `src/enqueue.cc`, `ncclLaunchKernel` |
| `ncclKernelMain` | `src/device/common.h`, `ncclKernelMain` |
| `runRing` | `src/device/all_reduce.h`, `runRing` |
| `waitPeer` | `src/device/prims_simple.h`, `waitPeer` |
| `genericOp` | `src/device/prims_simple.h`, `genericOp` |
| `postPeer` | `src/device/prims_simple.h`, `postPeer` |
| `ncclProxyProgress` | `src/proxy.cc`, `ncclProxyProgress` |
| `sendProxyProgress` | `src/transport/net.cc`, `sendProxyProgress` |
| `ncclNet.isend` | `src/include/plugin/net/net_v12.h`, `isend` member; `ncclIbIsend` in `src/transport/net_ib/p2p.cc` is the audited IB implementation |
| `ncclNet.test` | `src/include/plugin/net/net_v12.h`, `test` member; called by `sendProxyProgress` in `src/transport/net.cc` |
| `ncclNet.regMr` | `src/include/plugin/net/net_v12.h`, `regMr` member, in the same audited NCCL `v2.30.7-1` release as the `isend` and `test` rows above; the entry NCCL calls so an RDMA NIC can prepare a buffer. The published `ncclNet_v6` form of the same member, and its RCCL equivalent, are quoted in [the AMD GPU fabric note](../papers/amd-gpu-fabric.md) |
| `simllmChannelBufferRegistered` | simllm-invented: the one-time (communicator, channel, buffer) registration boundary, where the mirrored seam declares a cost that the traffic-owned ledger spends |
| `wrap_ibv_post_send` | `src/include/ibvwrap.h`, `wrap_ibv_post_send`; called by `ncclIbIsend` in `src/transport/net_ib/p2p.cc` |
| `wrap_ibv_poll_cq` | `src/include/ibvwrap.h`, `wrap_ibv_poll_cq`; called by `ncclIbTest` in `src/transport/net_ib/p2p.cc` |
| `simllmRnicRingDoorbell` | simllm-invented: exposes the RNIC notification hidden inside the verbs provider's post operation |
| `simllmNetworkComplete` | simllm-invented: deterministic external completion injection until a native RNIC session supplies CQEs |
| `simllmKernelComplete` | simllm-invented: stack-internal kernel-completion observation until runtime projection lands |

## Open tasks

### Precision

- COMP-1 (Precision; P1; L): complete production compute calibration.
  This task is the numerical capstone for every selected target: it owns
  target-silicon compute and memory anchors, per-device fit inputs, untouched
  test error and live TTFT/TPOT evidence for A100, H100 and AMD campaigns. Its
  Accel-Sim correlation and selective missing-region filling apply only to the
  supported A100 SM80 envelope. COMP-50 owns the generic record schemas,
  canonicalizer, compiler and validators; COMP-1 supplies their device-specific
  evidence and acceptance results rather than widening those interfaces. The Turing method anchor lands
  activity capture, immutable raw samples,
  train-only table compilation, interpolation and the provider seam, but its
  numbers are synthetic TU116 evidence. Its final run passed held-out
  calibrated median and p95 error at 0.674 percent and 1.773 percent versus
  17.782 percent and 25.069 percent for the flat 0.7 roofline surrogate.
  Stability is no longer the reason this task is open: the fidelity study
  showed the 3-of-50 miss came from 7 samples in 2,050 against a worst trimmed
  coefficient of variation of 1.054 percent, and refroze the bar in the
  environment-scoped form above. Two things now block it. First, no complete
  target-architecture evidence exists. Replace an active bootstrap or the flat
  0.7 roofline surrogate only after capturing exact production framework
  kernels on the target, collecting the full activity and counter ledger plus
  target-native implementation or code-object identity, and validating
  immutable held-out kernels. The A100 SM80 lane additionally collects its
  dynamic-SASS ledger and may qualify pinned Accel-Sim replay inside the
  supported envelope to fill declared gaps. H100 and AMD lanes neither require
  nor consume Accel-Sim. Second, the fixed-step seam now has
  calibrated Turing CUDA-graph and eager-host profiles, but no H100 or B100
  constant. The fidelity study's omitted excess of 1.79 to 12.31 times the
  modeled decode compute therefore remains unbounded on the production target,
  so the compute-only step error clause is unreachable until launch overhead,
  host delay and queueing are measured on that exact architecture. Do not
  transfer the Turing launch constants in the meantime. Acceptance remains
  the environment-scoped stability bar with the controlled form required for the
  production capture, held-out kernel median error below 10 percent and p95
  below 20 percent, per-phase median below 5 percent and p95 below 10 percent,
  and compute-only step error below 5 percent. The roofline and calibration-off
  paths must retain accepted artifacts and timestamps byte for byte. The
  [A100 hardware envelope](../../examples/a100_hardware_envelope_v1/RESULTS.md)
  narrows the second blocker without removing it. On one A100-SXM4-80GB with
  clocks observed at 1410 MHz through every timed block, the launch constants
  are 1.806 us for a pipelined eager launch, 6.069 us for a synchronized
  roundtrip and 0.791 us for a CUDA-graph replay node, so the graph path costs
  0.44 of the eager path on the target architecture rather than on Turing.
  The same lane bounds the flat 0.7 roofline surrogate directly: BF16 GEMM
  reaches 302.22 TFLOP/s at 16384 cubed, which is 96.9 percent of the 311.87
  TFLOP/s clock-derived peak, while HBM read reaches 86.8 percent of the
  2,039.04 GB/s memory-clock-derived peak, and the memory-to-compute crossover
  for `N` = `K` = 8192 measures at `M` = 256 against an ideal 158.9. A single
  0.7 efficiency constant cannot span 0.47 percent of peak at `M` = 1 and 96.9
  percent at 16384 cubed. The
  [GH200 hardware envelope](../../examples/gh200_hardware_envelope_v1/RESULTS.md)
  adds the Hopper constants from the identical sweep: 1.304 us pipelined, 6.126
  us synchronized roundtrip and 0.589 us per CUDA-graph replay node, with the
  roofline crossover at `M` = 512 against an ideal 284.6 and 918.66 TFLOP/s at
  16384 cubed. Two architecture pairs now bound the transfer question the task
  asks. The host-issue constants move with the host, not the GPU: the aarch64
  Grace launches 28 percent faster than the x86 EPYC while the synchronized
  roundtrip is unchanged within one percent, so a launch constant may not be
  carried across hosts even at fixed GPU generation. Both are microbenchmark
  evidence only: no production framework kernel, no dynamic SASS, no Accel-Sim
  calibration and no held-out kernel matrix, so COMP-1 stays open on its first
  blocker.
  The [A100 kernel constants study](../../examples/a100_kernel_constants_v1/RESULTS.md)
  is reviewed `VOID` and therefore closes nothing, but its retained evidence
  narrows the surrogate question further. Its measured HBM roof is 1818.21 GB/s,
  89.17 percent of nameplate, and it publishes clock-conditioned constants
  because application clock control is denied on that allocation. Three of its
  findings bear directly on this task. The flat 0.7 roofline derate is wrong in
  opposite directions for different shapes: measured roofline efficiency spans
  0.315 to 0.763 on the granite QKV family and 0.820 to 0.951 on an
  8192-squared synthetic family, so no single constant covers both. Captured
  MoE expert GEMMs at the granite population's expert loads run 5.17 to 12.20
  times their own memory roof over all 18 captured cells, because at those loads
  the kernel is bound by a fixed per-kernel cost rather than by bandwidth or
  arithmetic; COMP-43 owns that term and COMP-7's entry carries the same trap
  for the per-rank load work. And the operand layout, not the shape, produced a factor 2.9 swing
  between neighbouring token counts in the study's first run, which is why any
  future table must record the layout its constants were measured under.
  The [A100 graph launch study](../../examples/a100_graph_launch_v1/RESULTS.md)
  is also reviewed `VOID` and installs nothing. It retains evidence on part of
  this task's second blocker, the launch and host-delay terms, on the target
  architecture and host; the queueing term that blocker also names is not
  measured by it, so the blocker is narrowed and not removed. As retained
  evidence, the eager per-launch host cost on this A100 and EPYC pair is
  1,629,633 ps, 31.07 percent below the Turing `eager-host-bound` point, which
  alongside the already recorded Grace against EPYC difference is evidence that
  a launch constant tracks the host and driver rather than the GPU generation.
  Two host pairs are not a proof of that rule, and the rule stays a hypothesis
  the next host measurement can refute. CUDA
  graph replay costs the host 1.6 microseconds regardless of chain length, so
  at 256 nodes the host pays 6.5 nanoseconds per enqueued kernel, a factor 251
  below eager. The same study observes a real kernel period 1.42 to 1.51
  microseconds larger in eager mode than in a graph, of which a null kernel
  accounts for 1.08. The standing kernel-time ruling keeps that launch-mode
  effect outside service; COMP-48 owns identifying it as a host launch term.
- COMP-5 (Precision; P1; L): provide the production capture
  environment required by COMP-1. This task owns qualification policy,
  validity and stability evidence, fatal void rules and the hardware harness.
  COMP-50 owns the typed CUDA and ROCm doctor records and backend
  implementations that this task scores. The policy applies to A100, H100 and
  AMD campaigns without making COMP-5 the owner of each device calibration.
  The local GTX 1660 Ti still cannot qualify:
  Nsight Compute returns `ERR_NVGPUCTRPERM`, and display sharing produces the
  residency and clock-state excursions measured by the fidelity study. The
  [A100 environment qualification](../../examples/a100_environment_qualification_v1/RESULTS.md)
  removes the corresponding basic-capability uncertainty for one Merlin A100.
  Job `195283` produced a nonempty activity trace, numeric basic counters,
  exact tool and GPU provenance, matching disabled MIG and allowed-clock policy
  immediately before and after profiling, no foreign process and static
  `sm_80` SASS. All three probe executions agreed on device identity and
  checksum. This evidence is A100-scoped and must reject H100 or B100 use.
  The task remains open because the qualification intentionally omitted
  production SGLang kernels, dynamic NVBit tracing, Accel-Sim compatibility,
  controlled-clock evidence and the registered-cell stability sweep. The next
  expectations-only production study must exercise those mechanisms, retain
  the exact calibration-off path, and keep every registered cell below the
  controlled-environment stability ceiling before COMP-1 may consume an A100
  profile. Nsight Systems warned that device-side CUDA-event completion tracing
  can add overhead or false cross-stream dependencies. The production freeze
  must explicitly set `--cuda-event-trace=false`, or defend a frozen
  alternative, before interpreting multi-stream dependency evidence. The
  [A100 hardware envelope](../../examples/a100_hardware_envelope_v1/RESULTS.md)
  adds one measured constraint the production freeze must respect: every
  event-bracketed kernel in that study carried about 6 microseconds of fixed
  cost, matching its own 6.069 microsecond launch roundtrip, which silently
  destroyed the L2-residency signature it had predicted at an 8 MiB working
  set. Any registered cell whose kernel is shorter than roughly 60 microseconds
  measures the launch path as much as the kernel, so the production capture
  must amortize inside the timed region or declare a minimum kernel duration.
  Clocks were not locked there either, so the controlled-clock requirement is
  untouched. The
  [A100 kernel constants study](../../examples/a100_kernel_constants_v1/RESULTS.md)
  establishes why: on that allocation `nvidia-smi --lock-gpu-clocks` and
  `nvidia-smi -ac` are both refused with "The current user does not have
  permission to change clocks". The refusal was observed on three allocations
  of this account on `a100-hourly`, on nodes `gpu101` and `gpu105`, so the
  controlled-environment form of the stability bar cannot be met on the
  allocations this project has obtained, without an administrator action it
  does not have. Whether another account or another partition would be refused
  is not established by that evidence. The study substituted a
  clock-conditioned form, publishing constants per SM clock state over
  clock-stationary batches, and that substitute itself failed on 16 of 97
  scored cells, which is evidence about the environment rather than about the
  kernels. It also measured two facts a production capture must respect: the
  SM clock under load is bimodal at 1275 and 1410 MHz with a 283 to 432
  millisecond transition, so a cell that spans the boost boundary mixes two
  constants; and one `cudaEventRecord` placed between two consecutive launches
  costs 2.34 microseconds of device time, so per-kernel event instrumentation
  is not a free observation of a short kernel.
- COMP-7 (Precision; P1; M): MoE compute assumes perfectly balanced routing:
  every rank computes `top_k` experts' flops for its own tokens and streams all
  resident experts once. Consume the landed `simllm-routed-experts-v1`
  projection through `RoutedMoeSupply`, using the same selected placement
  epoch as traffic, to drive per-rank effective expert load and hot-expert
  imbalance. Pricing trap, from first-party A100 measurement: at the captured
  granite expert loads the roofline is not the binding term. The
  [A100 kernel constants study](../../examples/a100_kernel_constants_v1/RESULTS.md)
  measured all 18 captured expert cells at 5.17 to 12.20 times their own memory
  roof, because a load of 1 to 54 rows sits far below the 218 and 277 row
  roofline knees of those two shapes and the kernel is bound by a fixed
  per-kernel cost instead. That evidence is from a void run and closes nothing,
  but any work here that makes per-rank expert load more precise is refining an
  input to a term whose magnitude is wrong by 5 to 12 times, so COMP-43 should
  land alongside it rather than after it.
- COMP-9 (Precision; P1; L): locate and validate latency-tail fidelity in the
  network, batching and queueing chain, which is where the standing kernel-time
  determinism decision (maintainer, 2026-08-18) puts every tail. This task
  previously promised a measured or fitted service-time distribution on
  `DurationEstimate` and the profile artifacts so CORE-5 could claim
  kernel-level p99 and p99.9 accuracy. That scope is refuted for compute. A
  kernel's service time is a deterministic constant with no tail, so a
  per-kernel distribution would double count spread that the queueing it feeds
  already produces, and a reported p99 TTFT could then be reproduced by an
  arbitrary mix of kernel noise and queue noise, which makes the attribution
  unfalsifiable at the metric. `DurationEstimate` keeps one nominal value plus
  an honest uncertainty, and that uncertainty stays an error bound on a
  constant, never a sampling distribution.
  The surrogate now being replaced is the repository's silence about where a
  reported tail comes from: p50 through p99.9 TTFT and TPOT are named as
  milestone deliverables in `docs/architecture.md`, `adapters-vllm.md` and
  `adapters-sglang.md` with no statement of which mechanism owns each
  percentile. The identifying observables are per-visit queue waits under the
  one queue-visit contract (`submitted_at`, `eligible_at`, `started_at`,
  `finished_at`, `completed_at`), per-flow FCT from the packet-level backend,
  and batch composition per step. Acceptance: each reported TTFT and TPOT
  percentile is attributed to network, batching or queueing terms selected on
  the realized critical path with no additive mixing of wait reductions; a
  held-out workload's tail is predicted within a declared band; and removing all
  fabric contention and all batching collapses the distribution onto the
  deterministic constant this module guarantees. The deterministic compute path
  stays byte-identical throughout.
- COMP-31 (Precision; P1; L): complete the mechanism detail retained from
  COMP-11 after the calibrated endpoint serializer and semantic collective
  floor landed. The active selectable model still projects local traffic onto
  one endpoint serializer and folds unresolved stack work into a
  participant-indexed base. Add peer topology and per-link routing, ingress
  service and receiving-HBM interaction, priced reduction lanes and proxy
  operations. Identify those terms from pinned B200 per-link traffic, HBM
  counters, reduction-kernel timing and proxy timestamps over payload and
  participant sweeps with held-out cells. Require exact byte and work
  conservation, one timing authority for every term, no local/fabric double
  count, and held-out phase-completion error no larger than 10 percent or
  1 microsecond, whichever is larger. Report the reduced-form profile's
  before error and preserve the exact `legacy` and all-remote identity paths.
- COMP-17 (Precision; P1; M): audit and close the calibrated per-layer timing
  gap only after COMP-6 supplies exact per-invocation captured shapes and
  bindings and COMP-25 selects their resolved graph pricing on the live path.
  The current surrogate is the step sink's even split whenever
  `ProfileTableProvider` or `TraceCalibratedGpuProvider` is selected. On the
  primary exact-binding path, acceptance requires that no supported run uses
  that split, that measured per-invocation durations reconcile their integer
  sum to the fused estimate exactly, and that every rendered cumulative
  boundary stays within the declared capture uncertainty. COMP-6 and COMP-25
  own the production behavior; this entry owns the evidence-gated closure
  audit and adds no parallel projection. If a supported fallback still needs
  `estimate_layers`, register that residual under a new P2 task before closing
  this entry, with a measured layer-heterogeneity anchor and the original
  normalized-shape acceptance. The explicit no-breakdown path must retain the
  accepted GOAL bytes and TTFT exactly.
- COMP-21 (Precision; P1; L): calibrate the active optional RNIC producer
  task shapes that currently use a synthetic normalized trace. The v1
  surrogate charges one 64-byte descriptor store plus publication for a CPU
  proxy, or one 64-byte WQE store, one 4-byte doorbell-record store and
  publication for GPU initiation, with one CTA, one warp and minimal
  residency. Calibration must resolve the current GPU-initiated overlap: the
  producer task charges that 4-byte doorbell-record update before effective
  submission, then the native path charges the same physical update as a
  `DoorbellRecord` host store starting at submission. Assign its service to
  one timing authority and retain only the ordering projection at the other
  boundary. Capture GPU descriptor publication and mapped-UAR submission on
  the selected production GPU while sweeping batch sizes 1, 4 and 16 and
  idle, half-resident and residency-saturated neighbors. Use task admission,
  producer completion and RNIC-visible doorbell time as the identifying
  observables. Replace the trace and profile entries only when an independent
  validation capture predicts completion and queue wait within the larger of
  two GPU cycles or 10 percent in every cell. Report the synthetic
  before-versus-calibrated after error for every cell. The disabled coupling
  and host-CPU paths must retain every accepted timestamp and artifact byte.
- COMP-22 (Precision; P1; L): calibrate the GPU resource demand of the active
  cross-node collective path before CORE-26 and CORE-27 replace TRAF-7's
  independent-resource surrogate. This task owns only the communication
  stratum's GPU-resident demand. Collective expansion, wire serialization,
  RNIC service, congestion and FCT remain with their existing traffic and
  backend authorities. Capture pinned NCCL or RCCL collectives across
  payload, participant and channel-count sweeps, alone and beside compute- and
  HBM-bound kernels. Use kernel residency, channel occupancy, SM issue, HBM
  read/write traffic, network ingress/egress and any copy-engine or GPUDirect
  activity as identifying observables. Record an explicit zero for resources
  absent from the measured path, including whether any data mover is present,
  and supply those observed zero or nonzero demands to CORE-26 and CORE-27
  without charging downstream service here. Replace the synthetic demands only when an
  independent holdout predicts task completion and queue wait within the larger
  of two GPU cycles or 10 percent in every cell, and report the surrogate's
  before-versus-calibrated error. The calibration-off path must preserve every
  accepted TRAF-7 timestamp and artifact byte.
- COMP-23 (Precision; P2; L): record the calibrated per-kernel duration spread
  as capture evidence beside the mean-valued table. The landed profile table and
  trace-calibrated service model return one value per input, which cannot
  express the run-to-run spread that clock, cache and scheduling variation
  produce on real silicon, and a calibration that reports only a median cannot
  say how well identified its own constant is. The Turing method anchor supplies
  41 raw samples per family, dtype and shape cell and demonstrates why the
  record must retain outliers rather than only a mean. Those synthetic TU116
  samples validate the artifact shape but do not calibrate production kernels.
  Calibrate the uncertainty envelope per production kernel family after COMP-1
  and COMP-5 provide the target capture, carry the fit provenance and
  calibration envelope, and validate held-out quantiles against raw silicon
  samples. Report the
  deterministic point-table error before the distributional result.
  Scope constraint from the standing kernel-time determinism decision
  (maintainer, 2026-08-18): the fitted spread is calibration evidence about how
  well the constant is identified and about capture-environment stability, and
  it feeds the estimate's honest uncertainty. It is not a sampling source. No
  provider may draw from it to price a kernel, no seed enters a service path,
  and a reported latency tail is owned by COMP-9's chain rather than by this
  entry. The deterministic providers remain exact compatibility levels and their
  accepted artifacts stay byte-identical. COMP-50 owns the generic observation
  and validation records, while COMP-6, VLLM-12 and SGL-10 own capture
  production. This entry closes only when production observations retain the
  external sample-blob SHA-256 and byte count, integer summaries, a complete
  spread and excursion census, validation-calibrated uncertainty and untouched
  test evidence, and validated promotion recomputes and verifies the external
  raw blob. It adds no second record schema or capture path.
- COMP-24 (Precision; P1; M): qualify the closed
  `independent-resource-v1` mixed-service form beyond the single frozen fixture
  on which its explicit axes and residency rules were measured. COMP-12
  registered one issue-order pair and one residency-gated pair, so
  `decompose_mixed_makespan` refuses a replay in which more than one task
  waited for residency, and no measured row covers other shared-memory
  fractions, register or warp pressure, launch shapes or instruction mixes.
  The surrogate being replaced is the assumption that those explicit resource
  axes and the two-task rows generalize. Use isolated controls, admission cycles
  and concurrent makespans as the identifying observables, sweep the residency
  currencies independently so the binding one is identified rather than
  assumed, and require `independent-resource-v1` to predict each held-out cell
  exactly on the synthetic fixture before any silicon claim. The registered
  two-task rows must stay exact. COMP-6 then supplies real isolated, pairwise
  and three-way graphs as silicon identifying inputs. COMP-50 owns generic
  split handling, fit orchestration and compilation; COMP-24 owns only this
  task-specific numerical identification and acceptance. If held-out evidence
  requires any nonempty interaction term, reject it under v1 and land a
  versioned interface amendment plus a new expectations-only freeze before
  fitting that form. On untouched silicon cells, both mixed completion and
  queue wait must remain within the larger of two GPU cycles or 10 percent;
  every synthetic fixture row remains exact.
- COMP-25 (Precision; P1; M): connect the concurrent kernel service to a
  production step path. The trace-driven SM scheduler is reachable through
  `CoarseDeviceRuntime(kernel_services=...)` and COMP-12 demonstrated the
  chain to `StepResult`, TTFT and TPOT, but no production study or step sink
  selects it. Every reported production step therefore takes the scalar
  `ComputeWork.nominal_duration_ps` path, whose concurrent makespan is the
  independent-resource maximum and carries neither registered form. Implement
  the device-model-owned synthetic `ImplementationSelector`, resolve and
  validate one immutable total `ResolvedOperationServiceBindingSet` plus
  digest per graph before scheduling, and select the complete-tuple
  `BatchKernelService` on the live path while preserving the existing batch
  order, cursor and cycle-ceiling mechanics. COMP-6 owns observed
  physical bindings and shapes; CORE-12 owns incremental later-arrival
  admission. Freeze the pure selector and resolved-set digest first, let
  CORE-45 consume that digest, then close both tasks through shared live
  integration. Before implementation, freeze one no-contention single-request
  prefill and decode fixture. The selected critical-path device-service delta
  must equal the `StepResult` latency delta and signed TTFT delta exactly, and
  the fixed decode chain's TPOT delta must equal its per-step selected service
  delta exactly. Report the before-and-after TTFT and TPOT values. The
  explicit scalar off path must keep every accepted service call tuple,
  cursor, visit, report, timestamp and result byte exactly.
- COMP-28 (Precision; P2; L): After COMP-21 supplies device-bound structural
  captures for CPU-proxy and GPU-initiated network submission, fit and
  validate their scalar host-initiation projections for the analytical
  fallback used only while structural submission is disabled. Carry GPU,
  host, RNIC and submission-class provenance plus predeclared capture
  uncertainty; held-out ready-to-RNIC-visible latency must remain within that
  uncertainty. The ideal zero-cost profile remains the exact compatibility
  path.
- COMP-41 (Precision; P2; M): attach measured per-port ceilings to a shipped
  architecture profile. This task owns shipped, measured port ceilings and
  their envelopes, never kernel calibration or a fabricated transport rate.
  COMP-34 landed ports that carry a ceiling with its
  provenance, but every ceiling reachable today is either read out of a
  synthetic study calibration (`calibration_derived`) or declared by a study
  (`model_configuration`); no shipped profile carries a `first_party_measured`
  port ceiling, and the A100 and GH200 seed profiles declare no copy engine and
  no NVLink profile at all, so they compose to a device with no ports. The
  surrogate being replaced is the absence of a port ceiling on any shipped
  profile. The identifying observables are the measured cells already published
  by
  [a100_hardware_envelope_v1](../../examples/a100_hardware_envelope_v1/RESULTS.md)
  and
  [gh200_hardware_envelope_v1](../../examples/gh200_hardware_envelope_v1/RESULTS.md):
  26.78 GB/s host to device and 26.19 GB/s device to host on PCIe generation 4
  by 16, 419.93 GB/s inbound against 169.96 GB/s outbound on Grace C2C, 94.00 to
  94.07 GB/s per NVLink3 ordered pair with 281.65 GB/s of per-GPU egress, and
  133.24 to 133.27 GB/s per NVLink4 pair with 398.71 GB/s of egress.
  Acceptance: each shipped ceiling carries its envelope study as provenance and
  its own validity window, the asymmetric host link is expressed as two ports
  rather than one averaged rate, a request for an architecture with no measured
  ceiling is rejected rather than borrowing another architecture's number, and
  every accepted artifact stays byte-identical. This is P2 while no study
  selects a measured port ceiling and becomes P1 when one does.
- COMP-43 (Precision; P1; M): price the fixed per-kernel cost that neither
  compute provider carries. The surrogate being replaced is the absence of any
  floor: `RooflineProvider` returns `max(flops/peak, bytes/bandwidth)` and
  `ProfileTableProvider` returns a table entry, so a kernel whose work is
  smaller than the device's own per-kernel cost is priced below what the device
  can do. The identifying observables are first-party and already measured on
  the target architecture by the
  [A100 kernel constants study](../../examples/a100_kernel_constants_v1/RESULTS.md):
  the uninstrumented back-to-back period of an empty kernel is 1.904
  microseconds, and the captured granite MoE expert GEMMs at their captured
  expert loads measure 4.725 to 9.227 microseconds against memory roofs of
  0.578 to 1.275 microseconds, a factor of 5.17 to 12.20 over all 18 cells.
  Acceptance: a per-kernel floor whose value is measured on the architecture it
  is applied to and refuses an architecture it was not measured on, an explicit
  off path that reproduces every accepted artifact and timestamp byte for byte,
  and a
  reported before and after on the decode step of the granite fixture with the
  omitted excess bounded rather than estimated. The evidence this task consumes
  comes from a void run, so a non-void measurement (COMP-45) is a prerequisite
  for the calibrated value even though the mechanism can land first.
- COMP-45 (Precision; P1; M): produce a non-void A100 kernel-constant run. The
  [A100 kernel constants study](../../examples/a100_kernel_constants_v1/RESULTS.md)
  is void twice on its stability preconditions, so its constants close nothing
  and its profile table is deliberately withheld. Two causes are identified and
  neither is a kernel property. Application clock control is denied on the
  Merlin A100 partition, so a cell that spans the 1275 to 1410 MHz boost
  boundary mixes two constants; and one `cudaEventRecord` between two
  consecutive launches costs 2.34 microseconds of device time, so a
  per-repetition chain does not measure a short kernel. The surrogate being
  replaced is the flat 0.7 roofline derate on `a100`. Acceptance: a protocol
  whose stability precondition is achievable without clock control, stated and
  frozen before the run; every scored cell inside its own frozen dispersion
  ceiling; and a `simllm-profile-table-v1` artifact loadable by
  `ProfileTableProvider` whose held-out interpolation error meets COMP-1's
  registered median 10 percent and p95 20 percent bars. The void run already
  reaches 0.70 percent median and 18.53 percent p95 on its held-out shapes, so
  the bars are reachable; what is missing is a run whose guards hold.
  Boundary. COMP-45 owns producing the artifact; COMP-1 is its consumer and
  keeps the surrogate claim, so COMP-45 does not restate it. The registry
  answers the consumption question ONE way, stated here: COMP-5 remains the
  gate, and COMP-1 may consume an A100 profile only once COMP-5's
  environment-scoped stability bar is met on the cells that profile contains.
  COMP-45 is the work that makes that possible on an allocation without clock
  control; it does not bypass COMP-5's gate, and closing COMP-45 does not by
  itself license consumption.
- COMP-47 (Precision; P1; L): reach a non-void A100 graph-launch run and
  install the two host profiles it produces. The
  [A100 graph launch study](../../examples/a100_graph_launch_v1/RESULTS.md) is
  reviewed `VOID`: fatal guard `GG7` was violated, so its behavioral score is
  uninterpretable and no fraction of it is a result. Fourteen of its 15 scored
  expectations passed and one failed, and that 14 is not a score; it is written
  down only so a reader can see which relations survived. `GG7` bounds the
  block-mean dispersion of every reported period at 4 percent, which a chain of
  one to eight kernels cannot meet against the device's 1024 ns event quantum.
  This is L rather than S because closing it needs a fresh allocation on the
  target hardware and a re-frozen protocol, which is hardware evidence.
  The surrogate being replaced is the absence of any A100 entry in
  `HostInitiationModel`, whose calibrated profiles today accept only
  `gtx1660-ti-sm75`. Acceptance: a freeze whose dispersion guard is scoped to
  the periods the study actually publishes, stated before the run; a run whose
  guards hold; and `a100-epyc-eager-host` and `a100-epyc-cuda-graph` installed
  with the same fail-closed device check the Turing profiles carry, rejecting
  every key except `a100`. The measured values are already published as
  retained evidence: 1,629,633 ps per eager launch over an empirical 1,625,986
  to 1,927,260 ps, and 1,647,674 ps per graph replay independent of chain
  length. Installing them from the void run is refused on purpose.
- COMP-48 (Precision; P1; M): identify the launch-mode residual as host
  initiation. The standing kernel-time
  determinism ruling rejects the service-time reading: CUDA-graph versus eager
  mode never changes kernel service, so no launch-class field may reach a
  kernel-service key. The
  [A100 graph launch study](../../examples/a100_graph_launch_v1/RESULTS.md)
  measured a device-side per-kernel cost that is 1.415 to 1.506 microseconds
  larger in eager mode than under CUDA-graph replay, roughly constant across
  kernels whose own periods span 8.9 to 89.6 microseconds, of which a null
  kernel accounts for 1.080 microseconds. The identifying observable is
  activity timing across graph replay and eager execution with host submission
  measured separately. The existing driver cannot record an event during
  stream capture, so the new protocol must use a non-perturbing source such as
  profiler activity rows and must distinguish unchanged service from host
  launch visibility rather than subtracting two compound periods. No
  device-front-end service stage exists. Freeze null and real-kernel families,
  eager and graph modes, at
  least two chain lengths and an immutable train/validation/test split. Fit no
  value from test. Acceptance requires measured host initiation to predict the
  observed launch-mode delta within the larger of two GPU cycles or 10 percent
  in every supported cell. Also require no double charge with COMP-43's kernel floor or COMP-47's
  host profiles, and byte-identical kernel service and off-path results.
  That bar is unreachable under today's calibrated composition and this task
  is blocked on COMP-44 until it is not. The
  [host launch composition study](../../examples/host_launch_composition_v1/RESULTS.md)
  is nonvoid and refutes the reachability: `max(C, N * g)` returns a
  launch-mode delta of exactly zero for every per-kernel service at or above
  the eager per-launch constant, so its error against the measured 1.415 to
  1.506 microseconds is exactly 1.0 and roughly 2,000 GPU cycles, and the
  study's `R7` shows the zero holds for any per-launch constants rather than
  only the installed ones. A measurement campaign run before COMP-44 supplies
  a non-overlappable term would therefore fail this bar by construction and
  could close nothing.
- COMP-53 (Precision; P0; S): amend the frozen `transformer-dag-v1` physical
  sanity and guard contract before its first campaign cell runs. Four defects
  make the current freeze unevaluable or circular, and no cell has been
  observed yet, so the amendment lands as a new expectations-only freeze
  rather than an in-place edit of the existing record. First, `EQ5` maxes over
  `kernel_floor_ps`, which no equation, contract field or prose defines
  anywhere in the repository; COMP-43 owns the term it means, but the freeze
  never says so or says how it is obtained. Second, `EQ6` consumes
  `applicable_stage_floors_ps`, which no equation produces. Third, the floor
  half of the contract reads `compute_rate`, `hbm_rate` and `peer_rate`
  without declaring them, while every ceiling-side input is declared in
  `finite_campaign_envelope_fields` and bound by `finite_evidence_rule` to
  preexisting qualified evidence and never the current cell outcome. A floor
  derived from the measurement it bounds is circular, and a physical-bound
  violation is a fatal guard, so the asymmetry can void or silently spare a
  campaign for the wrong reason. Fourth, `G11` separates the timeline, counter
  and dynamic-instruction passes but omits the mixed pass the same freeze
  declares, leaving the fourth pass outside the only guard that enforces pass
  separation. Also settle the mixed-cell denominator a scalar compute-memory
  envelope inherits: the design statement drops all 28 mixed cells while the
  suite's own `mixed_rule` requires every cell whose member capabilities are
  ready, which is the 12 with no communication member. Acceptance: the amended
  freeze defines every term its equations use, declares and evidence-binds
  every floor-side input on the same footing as the ceiling side, covers all
  four passes in the pass-separation guard, states the reduced denominator,
  and locks its own schema string and the closed `preflight_states` enum in
  the freeze test. No measured value may be read before it lands, and the
  amendment cites this chronology.

### Completeness

- COMP-4 (Completeness; P2; M): add generic multi-axis interpolation as an
  explicit optional path without weakening the device-model v1 boundary.
  Version 1 supports exact cells or one declared affine axis, and
  `ProfileTableProvider` likewise permits one config axis while every other
  axis is pinned; a query differing on two or more axes fails closed. The new
  path is unavailable today, and that explicit refusal remains the exact off
  path. Before implementation, freeze target-silicon grid cells that vary at
  least two axes independently, the interpolation equation and a quantitative
  untouched-cell error band. Enabling the path must meet that band, while
  disabling it must reproduce every accepted exact lookup, one-axis
  interpolation, rejection and artifact byte exactly. No accepted calibration
  or runtime path may invoke generic multi-axis interpolation silently.
- COMP-6 (Completeness; P1; M): produce generic physical capture identities
  and typed invocation shapes. Join every observed noncollective physical
  launch by exact `(instance_graph_sha256, operation_id, launch_ordinal)` and
  preserve an immutable `OperationImplementationBinding`. For each supported
  semantic collective, preserve the ordered GPU-resident stages as
  `CollectiveDeviceStageBinding` records joined by exact
  `(instance_graph_sha256, collective_operation_id,
  collective_plan_integrity_sha256, rank, launch_ordinal)`.
  Validate both noncollective operation-to-launch totality and collective
  plan-to-stage totality without changing `simllm-execution-graph-v1` or
  creating a second collective completion authority. Capture preserves every
  observed launch when one rank has multiple stages. The online v1 resolver's
  exactly one resident stage per plan rank is a separate resolver and CORE-26
  composition constraint, never permission to coalesce or drop capture rows.
  COMP-50 owns both binding schemas plus their identity, shape and selector
  schemas; this task additionally owns `simllm-moe-routing-sidecar-v1`, the
  routing evidence a captured MoE graph needs.
  VLLM-12 and SGL-10 are thin producers. This task owns capture
  projection, topology normalization, activity joins and validation of both
  frozen ledgers. It does not absorb COMP-17's per-layer timing projection.
  When physical capture is disabled, every existing graph, aggregate record,
  timestamp and completion remains byte-identical.
- COMP-10 (Completeness; P1; L): extend SimLLM trace replay beyond synchronous
  normalized per-warp instructions. Add subpartition-aware scheduler
  ownership, CTA barriers, `cp.async`, Hopper TMA and warpgroup asynchronous
  issue, commit and wait semantics, plus calibrated cache partitions, bank
  conflicts and hit/miss behavior. Until each mechanism lands with capture
  evidence, its opcode or launch form fails closed rather than borrowing a
  scalar latency. The synchronous normalized replay is the exact compatibility
  path. Test every enabled supported mechanism beside that bypass, require
  unsupported opcodes and launch forms to reject, and preserve every accepted
  synchronous record and timestamp exactly. SimLLM replay remains distinct
  from the external Accel-Sim sidecar and never patches or substitutes for it.
- COMP-13 (Completeness; P1; M): add narrow content-addressed COMP-50
  concurrent input and output records for `GpuTask` and
  `GpuConcurrentEstimate`. Preserve exact task order and the five queue-visit
  times (submitted, eligible, started, finished and completed), requested and
  transacted HBM/NVLink bytes, request counts and deterministic replay
  validation. `simllm-gpu-model-artifact-v2` remains import-only and every
  accepted legacy artifact stays byte-identical. Until the new records land,
  concurrent demo CSVs are reviewed evidence but are not calibration-bundle
  records.
- COMP-14 (Completeness; P2; L): add optional NCCL algorithm builders for
  tree all-reduce, all-to-all, reduce-scatter and all-gather behind an
  explicit algorithm selection. The ring builder remains the identity
  baseline: selecting or omitting the default ring path must preserve every
  accepted ring timestamp, counter and task order exactly.
- COMP-15 (Completeness; P1; L): model the NCCL software stack with the real
  stack's functional names and interfaces, trimmed to the main path. The
  audited zero-time first slice is landed: communicator and ring setup,
  explicit ring-step chunk planning, GPU send-FIFO tail publication,
  `ncclProxySaveOp` queueing, independent CPU proxy progression,
  `ncclNet.isend`, verbs post, RNIC doorbell, external CQE production, CQ poll,
  proxy head-credit return, and distinct intra-node and inter-node call loops
  all emit strict events on one caller-owned clock. The
  [study](../../examples/nccl_stack_v1/RESULTS.md) freezes and validates the
  exact call sequences and planner relations.
  Remaining work is to replace deliberate zero-time boundaries and
  metadata-only movement with calibrated service mechanisms connected to the
  existing GPU, PCIe, native RNIC and fabric authorities; add the
  GPU-initiated leg; project selected events through the supported runtime and
  metric chain; and land the VLLM-14 and SGL-11 adapter callers. Receive-leg
  progression must wire `recvProxyProgress`, `ncclNet.irecv`, `ncclIbIrecv`,
  `wrap_ibv_post_recv`, receive completion through `ncclNet.test` and
  `wrap_ibv_poll_cq`, receive-connector tail publication, and GPU `waitPeer`
  plus `postPeer` head-credit return. These additions must retain one timing
  authority and the explicit bypass behavior.
  Intra-node collectives must compose with the NVLink-class egress model and
  stay off the fabric. Inter-node transfer and receive completion must project
  through CORE-4 and CORE-5 to `CompletionEvent`, `StepResult`, TTFT and TPOT.
  Boundary against BACK-47: this task owns the stack's own calibrated service,
  its receive leg and its metric projection, while BACK-47 owns the
  device-facing packet-emission contract at the plugin ABI seam. Neither may
  claim the other's half.
  Add the BACK-20 GPU-initiated leg behind the same upper interface while
  preserving the CPU-host proxy path as the default identity baseline. The
  VLLM-14 and SGL-11 simulated communicators remain the adapter callers that
  must connect to this stack. Function and event identities must remain stable
  so later captures, timing calibration and adapter traces align with this
  first slice.
- COMP-35 (Completeness; P1; M): instantiate vendor peer ports, so an AMD ROCm
  GPU and a UALink pod can be expressed at all. Once COMP-34 lands port objects,
  a vendor instantiation names the peer port xGMI or UALink rather than NVLink,
  names the collective producer RCCL rather than NCCL on the AMD arm, and
  supplies the envelope slots those names need. Neither protocol has a
  first-party measurement in this repository and the only figures available for
  either are vendor or consortium nameplate, so the instantiation must fail
  closed exactly as a calibrated B100 or H100 host-cost request already does,
  rejecting during configuration instead of borrowing an NVLink ceiling or an
  NVLink efficiency. UALink is the sharper case of the same rule: the UALink 200G
  1.0 specification states a 200 GT/s per-lane data rate carried at a 212.5 GT/s
  signalling rate, so taking the headline figure as a payload ceiling repeats
  exactly the NVLink4 signalling-versus-payload error the port taxonomy already
  records. Acceptance: a declared xGMI or UALink profile carrying its own
  provenance and validity window is required before any cell on that protocol
  runs; an undeclared or unmeasured request is rejected with a diagnostic naming
  the missing profile and the port it belongs to; and every accepted NVIDIA cell
  stays byte-identical. The accepted AMD calibration roadmap makes this P1,
  while every unqualified AMD or UALink port remains fail closed. This task
  owns only xGMI or UALink peer-port instantiation and RCCL identity. An AMD
  scalar compute-model candidate does not wait for it. COMP-34 landed the port
  objects and made the xGMI protocol nameable; the kernel-time determinism
  contract added UALink beside it on the peer-link role, and both are rejected
  at configuration time with a diagnostic naming this task. What remains is a
  declared ceiling per protocol with its own provenance and validity window,
  plus the RCCL producer
  naming.
- COMP-42 (Completeness; P2; S): normalize how the two adapter geometry readers
  spell the optional dtype widths on `ModelDims`. The vLLM reader resolves
  `weight_dtype_bytes` and `kv_dtype_bytes` from the quantization and cache
  configs and stores explicit floats; the SGLang reader stores `None` and lets
  `ModelDims` fall back to the activation width. Both resolve to the same number
  through `weight_element_bytes` and `kv_element_bytes`, so no reported
  picosecond moves today, which is measured by
  [kernel_determinism_v1](../../examples/kernel_determinism_v1/RESULTS.md)
  and pinned by `tests/test_kernel_determinism.py`. The unavailable path is a
  consumer that compares or hashes `ModelDims` itself: two adapters describing
  one identical rank would disagree, which is the failure mode BACK-50 already
  records for the effective-hardware snapshot. Give SGLang the same quantization
  and cache-dtype resolution vLLM has, or make both store the resolved width,
  and keep the explicit unresolved path testable. The off path is the current
  behavior: every accepted artifact and every priced step must stay
  byte-identical, and the pinning test must be updated in the same change rather
  than deleted.
- COMP-40 (Completeness; P2; M): the landed GPU ports declare capabilities but
  emit no packet event, so an intra-node leg still cannot report an extent, an
  attempt, a TX boundary or an arrival in the same language a wire port uses.
  The three transport-control capabilities (ECN marking, priority flow control,
  congestion notification) exist today only to be rejected by name, and the
  rejection diagnostic points at BACK-48. Boundary against BACK-48: that task
  owns making the ABI v2 vocabulary reachable from a non-wire port at all, while
  this one owns binding the GPU host and peer ports to it, including which
  capabilities a GPU port may then honestly advertise. Acceptance: an intra-node
  transfer emits session-unique extent and attempt identity through a GPU port,
  loss, duplication and double-charged bytes are detectable from those events,
  and the no-emission path preserves every accepted timestamp, counter and
  artifact byte exactly. This is P2 while no study consumes port events and
  becomes P1 when TRAF-45 packetizes the intra-node leg.
- COMP-44 (Completeness; P2; S): let a calibrated host profile carry a fixed
  per-invocation cost beside its per-launch constant. `HostInitiationModel`'s
  calibrated form has exactly one term, `point_ps_per_launch`, composed as
  `max(C, N * g)`, which is the right shape for eager launching and the wrong
  shape for CUDA graph replay. The
  [A100 graph launch study](../../examples/a100_graph_launch_v1/RESULTS.md)
  measured the graph host cost at 1.574 to 1.686 microseconds per replay across
  chain lengths 1 to 256, a fitted per-node slope of 0.000297 microseconds at
  an R-squared of 0.516, so there is no per-launch constant to fit: the cost is
  a fixed per-replay term plus a per-node term indistinguishable from zero.
  Expressing it as a per-launch constant makes the published point depend on
  the chain length, which is why that study's `a100-epyc-cuda-graph` point is
  declared scoped to a reference chain length rather than universal. The
  surrogate being replaced is that scoping. The
  [host launch composition study](../../examples/host_launch_composition_v1/RESULTS.md)
  widens the question from which term to which operator. Every calibrated term
  today composes as `max(C, N * g)`, whose exposed contribution is exactly zero
  once per-kernel service reaches the per-launch constant, so no calibrated
  term of any magnitude can express a per-kernel cost that the measurement
  shows is flat across a factor-ten range of kernel period. The shipped
  `legacy-fixed-step` branch already composes additively and returns exactly
  that shape, so the repository has the operator but no calibrated profile is
  allowed to use it. Acceptance: a calibrated profile may declare a fixed
  per-invocation term and a per-launch term, each term declares whether it
  overlaps device service or is non-overlappable, the composition states which
  terms a launch class uses, an additive term reproduces a launch-mode delta
  independent of provider service, the exact `ideal` zero
  profile and both Turing profiles reproduce every accepted artifact and
  timestamp byte for byte, and a graph profile built from a fixed term is
  independent of the launch count it is asked about. This is P2 while no study
  selects an A100 host profile and becomes P1 when COMP-47 installs one or when
  COMP-48 opens its measurement campaign, whichever comes first, because
  COMP-48's acceptance cannot be met without it.
- COMP-46 (Completeness; P2; M): supply a production-grade decode attention
  microbenchmark. The decode lane of the
  [A100 kernel constants study](../../examples/a100_kernel_constants_v1/RESULTS.md)
  reached 5.5 to 13.3 percent of the measured HBM roof even after its warps
  carried four independent online-softmax accumulators, and its time grew by
  3.02 between batch 64 and batch 256 where the KV bytes grew by 4, so it is
  still gaining efficiency with occupancy rather than sitting on the roof. Its
  constants therefore describe that microbenchmark and not a paged or flash
  decoding kernel, and the study says so. The surrogate being replaced is that
  lane's own kernel. The path this adds is a second, selectable decode kernel
  in the study harness; the existing kernel stays reachable and is the explicit
  off path, so the current lane's constants remain reproducible byte for byte
  after the new kernel lands. Acceptance: a decode kernel whose achieved KV
  bandwidth reaches a frozen fraction of the measured roof over the whole batch
  and cache-length grid, with the fraction stated before the run; a published
  comparison against the current kernel on the identical grid; and the off path
  reproducing the published constants of this study exactly. This is P2 while
  no study consumes a decode attention constant and becomes P1 when one does.
- COMP-49 (Completeness; P1; M): reify the xPU's inter-subsystem
  communication as a streaming crossbar behind the common interface. The
  README states the device as pluggable subsystems, the hardware scheduler,
  HBM, the copy engines, the PCIe host port and the scale-up ports,
  communicating over one common interface. Today the service model couples
  them through direct cursor and budget references with no reified
  interconnect object. Add a crossbar of point-to-point streaming lanes: a
  subsystem pushes work descriptors down a lane and consumes them from the
  far end, with no shared bus, since the model deliberately has no NoC on
  the GPU; the crossbar is contention-free by design and every crossing
  emits an observability event. The default composition must preserve every
  accepted baseline byte for byte. It may transport typed resource-vector
  descriptors, but it owns no calibration coefficients, resource registry or
  service-model fit. BACK-53 owns the RNIC counterpart, a
  NoC-like signal-slot bus whose contention is a registered future upgrade.
- COMP-50 (Completeness; P1; L): deliver the vendor-neutral offline calibration
  package and compact device-model contract. Own
  `simllm-device-calibration-bundle-v1`, `simllm-device-model-v1`, canonical
  record and content-hash rules including
  `simllm-calibration-canonical-bytes-v1`, the narrower
  `simllm-calibration-canonical-ascii-conformance-v1` subset and the
  independent C++17 verifier that covers only that subset,
  `simllm-calibration-token-fixture-v1`,
  noncollective and collective-stage binding
  schemas and pure resolvers, shape and implementation identities, typed
  resource axes with known masks, strict validators, generic split and fit
  orchestration, compilers, the `HardwareCollector` and
  `OfflineKernelSimulator` protocol interfaces, CUDA and ROCm doctor backends,
  the local CLI and the contributor submission contract. Keep GPU and
  simulator scripts, workloads and configuration under top-level
  `offline/calibration/`; keep the lazy offline record/compiler package under
  `simllm/calibration/`; publish reviewed compact data under `devices/`.
  Keep those tracked roots as the single authorities; an installed release may
  carry only a manifest-digest-checked immutable archive built from them, with
  explicit suite and registry root overrides and no second editable copy.
  Compile both scalar profile-table service and optional mechanistic service
  without making a precision claim; COMP-1, COMP-22 and COMP-24 own their
  evidence and numerical acceptance.
  Raw traces remain external and uploads contain data only. Enabling,
  disabling or omitting this optional path preserves accepted scalar,
  GPU-model-v2, default-import, validation and serving bytes and timestamps
  exactly. Candidate and validated status are distinct, and validation fails
  closed on unknown fields, duplicate keys, nonfinite values, unsafe archives,
  hash mismatch, incomplete bindings, split leakage or physical-floor
  violation.
- COMP-51 (Completeness; P1; M): add the official Accel-Sim framework unchanged
  as an optional offline submodule and close its reproducible dependency
  envelope at `third_party/accel-sim-framework`. Pin upstream commit
  `3016c658f810bdae9a14bf4534ee99e9945eedae`; keep every SimLLM wrapper,
  configuration and script outside the checkout; record upstream and
  transitive licenses; provide a hash-locked prefetch bundle and an offline
  smoke that runs after dependencies are present. Pin the associated upstream
  statistics archive at `ee21104be44ad55dfde789111d3b94372be8435f` and
  GPGPU-Sim at `6c3cf4ff32110908386d605a7034fc67666a92de`. Check the
  archive golden without claiming an exact public A100 rerun because its
  site-local hardware statistics and trace inputs are not distributed; claim
  exact reproduction only after those inputs are acquired and hash locked.
  Reject recursive initialization, an unexpected nested submodule state and
  dirty edits beneath the official gitlink. Default imports, validators,
  serving, tests and contributor checks never fetch or require the submodule.
  Separate networked release verification must prove that the exact framework
  pin is fetchable and branch-reachable; default offline CI remains network
  free. Reject H100, later NVIDIA ISA, AMD ROCm,
  communication and serving-loop invocation. Numerical SM80 correlation and
  selective filling remain COMP-1. Absence, disablement or an unsupported
  request preserves every accepted default byte and timestamp exactly.
- COMP-52 (Completeness; P2; L): support explicit architecture-derived device
  candidates when target silicon is not yet available. Require a validated
  anchor, declared architecture deltas, a content-addressed analytical
  `ImplementationRef`, target code-object availability when a code-object
  reference is selected, applicability guards, inflated uncertainty and
  anchor/delta evidence joins. The result is candidate-only, nondefault and
  never described as Accel-Sim or silicon calibrated. Every run records the
  model hash, status, target basis and envelope, and an unsupported target
  fails closed. Disabling this path preserves the validated-anchor model and
  every accepted result byte exactly.

### Uncategorized

- COMP-8: the fused-vs-family sum invariant test compares in float; above
  2 to the 53rd flops (a 32k-token prefill chunk on a 100B-class dense
  rank) ULP effects could mask a real mismatch even though the integer
  identity is exact. Assert the sums in the integer domain when such
  shapes enter scope (audit note, examples/m5/RESULTS.md).
