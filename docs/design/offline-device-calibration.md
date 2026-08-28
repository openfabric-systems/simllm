# Offline device calibration

SimLLM builds device models offline from exact framework executions and loads
only compact deterministic artifacts in serving simulations. Accel-Sim is an
optional, untouched upstream sidecar for qualified A100 gaps. It is neither an
online dependency nor a substitute for target-silicon evidence.

This note freezes the architecture, ownership and execution order. The module
registries remain the source of truth for open work. The live fill state of
the measurement campaign, per target, framework and model, is tracked in the
[calibration coverage matrix](calibration-coverage.md).

## Boundary

The system has four distinct layers:

| Layer | Mutable authority | Output |
|---|---|---|
| Framework capture | The real model runner and profiler own the implementation that executed | Unchanged `simllm-execution-graph-v1` plus observed implementation bindings |
| Offline evidence | The calibration package owns validation, content identities and immutable splits | `simllm-device-calibration-bundle-v1` |
| Offline compilation | The calibration compiler owns fitting, source selection and validation | `simllm-device-model-v1` plus an optional scalar profile table |
| Online execution | `DeviceRuntime` owns graph legality, arbitration and result projection; the selected device service owns internal feasibility, reservation, rates and release | Per-graph resolved bindings, completion facts and model provenance |

`StepRecord` remains semantic input and `ExecutionGraph` remains the runnable
DAG. Profiler rows, implementation identities, calibration evidence and device
models do not enter either record. The same semantic graph can therefore carry
separate NVIDIA and AMD bindings without becoming vendor specific.

The repository layout keeps those layers visible:

```text
third_party/accel-sim-framework/   untouched optional upstream submodule
offline/calibration/               GPU and simulator scripts, workloads and configs
offline/calibration/kernel/        kernel requests and compact candidate results
offline/calibration/network/       transport and fabric calibration configuration
simllm/calibration/                strict records, compiler, validators and CLI
devices/<vendor>/<device>/<ver>/   reviewed compact device releases
```

Nothing below `offline/calibration/` is imported by the serving path. Project
Accel-Sim configurations and wrappers live there rather than inside the
submodule. `simllm.calibration` is a lazy offline tool package; ordinary
`simllm` imports and device-model loading do not import its collector or
simulator backends. Published `devices/` entries contain compact data and an
evidence ledger, never raw traces or contributor executables.

The tracked `offline/calibration/` suites and `devices/` releases are the
single source authorities. A release build may place their manifest-selected
files in an immutable package archive, but it records the source-manifest
digest and never maintains a second editable tracked copy. The CLI resolves an
explicit `--suite-root` or `--registry-root` first, then the corresponding
`SIMLLM_CALIBRATION_SUITE_ROOT` or `SIMLLM_DEVICE_REGISTRY_ROOT`, then a
repository checkout, then the digest-checked packaged archive. It rejects
conflicting roots or a packaged digest mismatch. This makes an installed
contributor command usable without making package resources a second registry
authority.

## Local-shard collection

`simllm.calibration.local_shard` provides one framework-neutral request and
result boundary for rank-local kernel capture. A request carries the existing
dispatch signature, exact model revision, logical tensor, pipeline, data and
expert parallel sizes, the physical rank coordinates and device ordinal,
phase and shape, launch mode and a deterministic synthetic-token recipe. The
logical parallel sizes never imply that one GPU executed the whole distributed
configuration.

The framework target is an external command and remains the authority for the
model class, sharding implementation, compilation and kernel launches it
actually executed. The common launcher uses no shell. It writes canonical
request bytes, requires an empty caller-supplied output root, and accepts only
a canonical candidate result whose model, dispatch, shard, input hash and
device ISA match the request exactly. Each kernel observation carries a
relative sample-blob name, byte count and SHA-256; the launcher reads and
verifies every blob before completing the run.

An A100 target therefore emits SM80 evidence only. A target that cannot
materialize the declared rank-local shard rejects the request instead of using
another architecture or silently executing a different parallel
configuration. Distributed collectives and network service are explicit
exclusions. Their dependencies may be visible at the framework boundary, but
their time is calibrated by the communication and network authorities.

The command surface is:

```bash
simllm-calibrate run \
  --request request.json \
  --target framework-target \
  --output-root "$SIMLLM_KERNEL_CALIBRATION_RUN_ROOT/cell"
```

The tracked `offline/calibration/kernel/` and
`offline/calibration/network/` namespaces keep reviewed kernel and network
configuration separate. Raw traces and sample blobs remain outside Git. The
local-shard result is candidate evidence and compiles through the established
kernel-cycle and device-model authorities only after their existing validation
gates.

## Identity and binding

Three identities have separate purposes:

1. `instance_graph_sha256` hashes the exact canonical unbound
   `simllm-execution-graph-v1` bytes. Every `ComputeWork` in that preimage
   retains its semantic fields and serializes `nominal_duration_ps` and
   `uncertainty_fraction` as explicit null members. The hash joins evidence and
   never selects service. The calibration canonical writer encodes the exact
   strict graph-v1 JSON object. A graph admitted for calibration binding rejects
   float-valued `ComputeWork.config` members because the calibration grammar has
   no binary-float spelling; this does not change the graph-v1 reader.
2. `template_graph_sha256` hashes canonical
   `simllm-execution-graph-template-v1` bytes. Operations receive ordinals in
   original graph tuple order; distinct ranks map to `0..n-1` in ascending
   source-rank order; and each normalized rank's logical queues receive
   ordinals in first-operation occurrence order. Its exact top level is
   `schema`, `operations`, `completion_operation_ordinals` and
   `collective_plans`, including an empty plan tuple. Operations preserve
   source tuple order and array position is the ordinal. An operation contains
   only `rank_ordinal`, `logical_queue_ordinal`, `priority`, `work`, sorted
   unique `depends_on_operation_ordinals` and sorted unique
   `participant_local_depends_on_operation_ordinals`. An empty source
   completion frontier normalizes to all operations; an explicit one becomes a
   sorted unique ordinal tuple.

   `work` is a strict union of `{kind: "compute", kernel}`, `{kind:
   "kv-cache", action}`, `{kind: "dma", source_role, destination_role}`,
   `{kind: "collective", collective, algorithm_hint, rank_ordinals,
   channel_ordinal, pair_rank_ordinals}` and `{kind: "control", mode, message,
   destination_rank_ordinals}`. Nullable `algorithm_hint` stays present.
   Collective and control ranks preserve source order. Sparse collective pairs
   retain only normalized endpoint pairs in source aggregate-pair order.
   Effective collective channels use `channel_hint` or canonical `default` and
   receive graph-global ordinals by first collective-operation occurrence.
   DMA endpoint roles use the
   `simllm-device-endpoint-role-v1` normalizer: `host`, `host:pinned`,
   `host:pageable`, `gpu:<rank>`, `gpu:<rank>:hbm` and `cuda:<rank>` are the
   only accepted forms, and every GPU rank rewrites through the same map.

   A projected collective plan contains only `operation_ordinal`,
   `algorithm`, `channel_ordinal`, `rank_order`, `rounds`, `actions`, `extents`,
   `entry_action_ordinals` and `terminal_action_ordinals`, preserving source
   plan order. `rank_order` contains normalized rank ordinals in source order.
   The channel repeats the semantic channel as a loss check. Round
   transfer channels use a separate graph-global first-occurrence namespace.
   Round array position is its ordinal and the strict member contains only
   `transfer_channel_ordinal`. Action and extent array positions are their
   ordinals. An action contains only `rank_ordinal`, `kind`, `extent_ordinal`
   and sorted dependency ordinals. An extent contains only its round,
   normalized endpoints and send/receive action ordinals. Entry and terminal
   members contain only rank and sorted action ordinals in plan-rank order.

   The rank map is the ascending union of every operation anchor, collective
   participant, control destination, accepted DMA endpoint and plan rank. The
   projection excludes all raw instance and plan-local identity spellings,
   timestamps, correlations, placement epochs, config, shape, payload, bytes,
   FLOPs, service, demand, request attribution, tags and integrity hashes.
   Excluded-only changes, consistent identity renames, dependency/frontier
   tuple permutations, empty versus explicit-all completion and null versus
   `default` channel spelling preserve the hash. Retained source tuple order
   other than the explicitly sorted dependency, completion and frontier-set
   fields, priority, dependency scope, effective completion, rank or queue
   equivalence, work family, DMA role, collective rank order, sparse-pair
   support, channel sharing, plan presence and every retained plan edge are
   sensitive. Rank
   renaming is invariant only when source-rank order is preserved. Projection
   is idempotent, rejects unresolved references and never selects service.
3. A service key is `(implementation_id, shape_vector)`. A
   `DispatchSignature` validates the selected model's frozen framework,
   backend, library, numeric, layout and device envelope. Within that envelope,
   semantic work and typed shape choose the implementation. Launch mode never
   participates in kernel dispatch.

An observed run records the exact implementation from the model runner and
profiler. A selector cannot replace it. A supported physical graph contains
one compute operation per noncollective physical launch, with explicit order
and dependencies; a noncollective semantic operation that still hides several
launches is unsupported. Physical NCCL and RCCL launches instead use
`CollectiveDeviceStageBinding`, keyed by instance graph, parent collective,
exact plan integrity hash, rank and launch ordinal. The binding carries the
observed implementation and typed shape, not fitted demand.

A synthetic run resolves every compute operation before scheduling. Its
canonical `simllm-resolved-operation-service-binding-set-v1` record has exactly
`schema`, `instance_graph_sha256`, `dispatch_context_sha256` and `bindings`.
Each binding has exact graph operation and launch identity, device instance and
selected-model SHA-256, semantic key, shape, implementation, service entry,
resolution source and a required-nullable observed-binding SHA-256. Bindings
follow graph tuple order, then launch ordinal. The immutable dispatch context
contains the validated rank/device assignment and a total selected-model tuple
keyed by device instance.

The dispatch context is the strict canonical
`simllm-device-dispatch-context-v1` record with exactly `schema`,
`instance_graph_sha256`, `rank_device_assignments` and
`selected_device_models`. Rank assignments contain only `rank` and
`device_instance_id`, sort by integer rank and cover every graph participant
exactly once. Model selections contain only `device_instance_id`,
`device_model_id`, `device_model_sha256` and `dispatch_signature_sha256`, sort
by device instance and cover exactly the assigned devices. Each signature is
the strict `simllm-dispatch-signature-v1` record with exactly `schema`,
`framework_id`, `framework_version`, `backend_id`, `backend_version`,
`kernel_library_id`, `kernel_library_version`, `algorithm_policy_id`,
`device_isa`, `numeric_traits` and `layout_traits`. Every identity and version
is nonblank. A nested `simllm-typed-dispatch-trait-v1` trait contains only
`trait_id`, `value_type` and `value`; the type is `integer`, `string` or
`boolean`, the value matches it, and each tuple sorts by unique nonblank trait
ID. Launch mode
is forbidden from this record because it affects host initiation only.

A supported planned collective also resolves a nonempty canonical
`simllm-resolved-collective-device-stage-set-v1` with exactly `schema`,
`instance_graph_sha256`, `dispatch_context_sha256`, `stages` and
`rank_frontiers`. Every stage carries its device instance and selected-model
SHA-256. Stages follow graph collective tuple, plan rank and launch-ordinal
order; frontiers follow graph collective tuple and plan rank order. A graph
with no resolved stage omits the record and puts null in the graph-total
closure. Both pure resolvers consume the total selected-model tuple, not one
global model. The traffic plan supplies topology but never chooses an
implementation. Version 1 supports exactly one resident stage per rank and
rejects multi-stage composition until stream and dependency evidence identifies
its order. Each rank carries a `CollectiveDeviceRankFrontier` whose parent
collective operation, plan hash, rank, one stage ordinal, entry action IDs and
terminal action IDs match the traffic plan exactly. A static device model
never enumerates future graph instance hashes. Calibrated mode rejects a
missing or extra binding, a
device/model mismatch, a stage also scheduled as `ComputeWork`, and every
kernel-name fallback.

For each rank, `submitted_at` is parent collective launch completion and
`eligible_at` is the maximum of that boundary and rank-local graph-predecessor
readiness. The device-stage grant at `started_at` releases that rank's existing
plan entry actions. The device engine reaches `device_work_finished_at_ps` when
its throughput demands and floors finish, but compute retains the stage's
lifetime residency and exclusive reservations. Traffic owns every plan action,
extent, byte, peer-port and network timestamp.
Version 1 rejects nonzero peer-port or data-mover demand in the resolved stage;
measured mover evidence is retained for later composition.
`traffic_terminal_at_ps` is
the maximum of the rank's `eligible_at` identity element and every copied
terminal-action completion. This keeps a legal sparse rank with an empty
terminal tuple defined. The read-only traffic boundary gates the compute-owned
lease release: `rank_release_at_ps = max(device_work_finished_at_ps,
traffic_terminal_at_ps)`. Compute releases reservations there and rank
completion equals it. No plan action depends on release, so the gate is
acyclic. The semantic collective emits one graph completion at the maximum
across ranks. A disabled path, or an entry whose every active-axis
demand is known zero in every epoch and whose every epoch floor is null or
zero, creates no device visit and leaves entry actions at the same preexisting
legal boundary while preserving the accepted traffic schedule. A positive
floor prevents that bypass. The canonical
`simllm-resolved-device-binding-closure-v1` record has exactly `schema`,
`instance_graph_sha256`, `operation_service_binding_set_sha256` and nullable
`collective_device_stage_set_sha256`; its hash enters run provenance. The
operation set must name the closure graph. A nonnull collective set must name
the same graph and the operation set's dispatch-context digest. Every
device/model pair in both sets must match the dispatch context and run
provenance; a cross-record splice rejects the closure.

The authoritative stage `QueueVisit` remains under the parent collective
operation and uses stable subject object identity
`<operation-id>:rank:<rank>:stage:<launch-ordinal>`. Its `finished_at` is
`rank_release_at_ps`; its `completed_at` is the same boundary. Version 1
identifies this composite visit with the existing legacy
`GPU_SCHEDULER` `ResourceRef`; internal service axes do not become separate
queue visits. CORE-8 loss-checks its normal completion and bookkeeping
projection. CORE-50 is required only for a later otherwise-unrepresentable
outer resource. The stage never emits an independent graph-operation
completion. Only the semantic collective emits that completion at the maximum
rank boundary. The interval from `device_work_finished_at_ps` to
`rank_release_at_ps` is lease-held occupancy evidence, never another additive
kernel or network latency term.

`ShapeSchema` defines ordered integer axes, units and domains once. A
`ShapeVector` carries that schema identity and its values. Numeric, layout,
quantization traits are typed dispatch traits, not anonymous shape dimensions.
Framework, backend, library, numeric and layout compatibility are frozen in the
selected device-model envelope; launch mode is excluded from kernel dispatch.
The model selector then uses semantic work and typed shape. Once
`(implementation_id, shape_vector)` is fixed, none of those envelope labels may
change service. `ImplementationRef` has two closed variants:

- a target code-object reference with vendor, ISA, module or code-object hash,
  function or code hash, backend or algorithm identity and a trusted
  launch-template identifier in `launch_formula_id`;
- a declarative analytical reference with its own model hash, exact target
  applicability and anchor/delta evidence, allowed only for COMP-52 candidates.

## Evidence closure

The bundle is an acyclic content-addressed record graph:

```text
evidence manifest -> capture, graph, binding, measurement and simulator records
fit record        -> evidence manifest
release manifest  -> evidence manifest + fit + compact model + validation
```

A record identifier is the SHA-256 of that record's canonical bytes. The
identifier is external to the record, so there is no self-hash. Raw traces,
profiler exports and replay output remain external blobs referenced by digest.
Canonical authoritative bytes use UTF-8 and normalize every object key and
string value to Unicode NFC. Object keys sort lexicographically by normalized
Unicode scalar sequence; arrays retain schema-defined order. The strict reader
rejects duplicate keys before object construction, keys that collide after
normalization, unpaired surrogates, unknown fields and nonfinite numbers.
Strings escape quote and backslash, use the five JSON short-control escapes,
encode every other U+0000 through U+001F scalar as lowercase `\u00xx`, never
escape slash and emit every other scalar directly as UTF-8. Integers use the
shortest base-10 spelling with no plus sign, leading zero or negative zero.
Booleans and null use lowercase JSON literals. Canonical objects contain no
insignificant whitespace or terminal newline.

Exact picoseconds, cycles, bytes, counts and FLOPs are arbitrary-precision
signed JSON integer tokens until each schema applies its field bounds. No
producer or consumer converts them through IEEE-754. Rates and exact ratios are
reduced numerator and denominator integer pairs. Fitted coefficients are
decimal strings at declared precision with no exponent, plus sign, redundant
leading or trailing zero, or negative zero. The external record identifier is
the lowercase hexadecimal SHA-256 of exactly these bytes.

The Python implementation is the sole full-Unicode canonicalizer and locks its
supported Python and Unicode database versions. An independent C++17 verifier
covers only the separately named
`simllm-calibration-canonical-ascii-conformance-v1` subset. It parses ASCII
keys and values, all JSON structural types, duplicate keys, control escapes and
arbitrary-length integer lexemes itself, rejects non-ASCII input, and uses a
project-owned SHA-256 implementation derived from FIPS 180-4. It uses no JSON,
cryptographic or Unicode dependency. Full Unicode NFC and
normalization-collision vectors remain Python-only; ASCII-subset agreement is
never described as a full native record loader.

Train, validation and test membership is immutable. The split groups by shape
and graph identity, not repeated launches of one cell. Fitting reads only
train. Model choice reads validation. Untouched test and graph-level silicon
makespan are promotion evidence. A separately identified all-data refit may be
published only with a new model identity and candidate status after those
results are frozen. It inherits no accuracy or validated-status claim from the
evaluated model and requires fresh untouched evidence before promotion.

## Measurement design

The frozen transformer DAG suite covers three device-demand strata:

- Compute uses high-arithmetic-intensity production operations across token,
  tensor and occupancy pressure. It measures FLOPs, issue, residency, launch
  and host-launch observations separately.
- Memory uses production KV, attention and streaming operations across bytes,
  cache state and concurrency. It records requested and transacted bytes,
  cache counters and HBM service.
- Communication uses GPU-resident NCCL or RCCL kernels across payload,
  participants and channels. It records SM, HBM, peer-egress and any observed
  mover demand.

The suite includes isolated controls, every pairwise mixture, a three-way
mixture and held-out framework graphs. Communication evidence records
peer-egress and mover observations, but the version-1 resident-stage barrier
charges only SM/HBM demand. Traffic retains chunk and peer-port timing;
collective expansion, wire serialization, RNIC work, fabric congestion and FCT
retain their existing authorities. Version 1 gates plan entry on the device
stage grant and joins device release with the exact plan terminal frontier.
CORE-13 or CORE-27 may consume retained port or mover evidence only after its
own versioned handoff and no-double-charge freeze.

The first authored suite uses the exact pinned Granite MoE checkpoint named in
its `suite.json`. It derives every request token deterministically from graph
case, request and position ordinals, uses greedy decoding with no random draw,
and content-addresses the resulting token fixture. Capture records exact
per-layer top-k expert routing; divergent routing inside one claimed envelope
is fatal or requires a distinct envelope. The memory bucket warms code and
cache metadata but rotates data buffers through a working set at least twice
the measured L2 capacity. HBM and cache counters must confirm the declared
steady-HBM state; an L2-resident cell cannot identify an HBM-bound entry.

Communication cells name runnable `all-reduce` plus `ring` and `all-to-allv`
plus `pairwise` plans, with a positive channel count in every cell. The matrix
sweeps payload, participant count and channels. An all-reduce scalar is the
full reduced input bytes per rank; an all-to-allv scalar is bytes per ordered
source-destination pair. Rank `i` maps to node `i`, local GPU zero, on the
vendor-native GPU-direct RDMA class; a target unable to execute that placement
declares communication unsupported rather than substituting an intra-node path.

The authored counts of 15 graph cells, 20 communication cells and 28 mixed
cells are logical topology denominators, not already-expanded run counts. Each
device release declares its supported framework envelopes. Every claimed
`(target, framework)` envelope executes all 15 graph cells in eager mode and
also in captured-graph mode when that backend reports support. A full
communication-validated envelope additionally executes all 20 communication
and all 28 mixed cells for each claimed framework and supported launch mode. A
scalar compute-memory envelope instead declares communication unsupported and
does not inherit the 20-cell communication denominator. Amendment (maintainer,
2026-08-24): under the frozen `mixed_rule`, that envelope inherits the reduced
mixed denominator of 12, comprising all four widths of `mix-compute`,
`mix-memory` and `mix-compute-memory`, because those are every cell whose member
capabilities are ready. The 16 cells with a communication member remain
excluded. This dated amendment corrects only the scalar envelope denominator;
it changes no suite topology, interface, acceptance bar or evidence rule. Two
framework claims may not satisfy one logical denominator with disjoint subsets.

Each measurement starts with physical bounds. Compute cannot beat FLOPs over
the applicable peak; memory cannot beat compulsory bytes over measured HBM
bandwidth; port service cannot beat payload over its measured directional
ceiling. A mixed cell's pre-observation floor is exactly the maximum of the
applicable isolated floor for every authored member copy; width changes copy
multiplicity but cannot reduce that maximum. Before reading a result, every
cell also records either a defensible
finite first-principles ceiling, or, for an isolated or graph cell with no
defensible finite bound, the explicit value `unbounded`. A mixed cell's finite
ceiling is the sum of the applicable member ceilings over every authored copy.
The deliberately serialized same-members control is a separate measured upper
bound checked after observation, never the pre-observation ceiling, because a
pre-observation bound may never borrow a measured value. Fatal
identity, conservation, clock, split and physical-bound guards void the
campaign rather than reducing a score.

## Source selection and support

The compiler applies one visible precedence rule:

1. exact silicon observation;
2. silicon fit inside that fit entry's validated support;
3. qualified Accel-Sim fill for an explicit missing exact A100 point inside the
   train-defined, silicon-validated SM80 support region and between required
   real anchors;
4. explicit analytical roofline fallback, only when policy permits it;
5. unsupported.

It never blends sources silently. A simulator fill records `coverage-gap`, the
upstream commit, simulator configuration and trace hashes, cycles, calibrated
residual and uncertainty. Per-entry fit support is a declared subset of the
train-defined target envelope and may be exact-only or exclude a categorical
implementation region. A simulator fill is reachable only when the target
envelope and real-anchor bracket cover the exact cell but no validated silicon
fit does; a later valid silicon fit outranks it. Graph-level silicon makespan
remains the oracle.

The Accel-Sim sidecar pins the official upstream development commit
[`3016c658f810bdae9a14bf4534ee99e9945eedae`](https://github.com/accel-sim/accel-sim-framework/commit/3016c658f810bdae9a14bf4534ee99e9945eedae).
The official v1.3.0 release lacks its A100 configuration. The sidecar is
qualified only for a declared SM80 compute and memory region after correlation
against project silicon anchors. H100 and later NVIDIA ISA, AMD ROCm,
every communication-stratum observation and online invocation reject it. A measured-only A100
model can promote without the sidecar; an A100 model containing a simulator
fill cannot promote without the Wave 1B sidecar gate and qualified replay and
correlation evidence.

The upstream A100 archive is locked separately at statistics commit
`ee21104be44ad55dfde789111d3b94372be8435f` and GPGPU-Sim commit
`6c3cf4ff32110908386d605a7034fc67666a92de`. Its A100 hardware statistics and
traces are referenced from site-local storage and are not published by the
upstream fetch helper. The first sidecar gate therefore checks conformance to
the checked-in archive golden only. An exact upstream rerun is not claimed
until those inputs are obtained and hash locked; a fresh project A100 capture
is labeled project reproduction, not upstream reproduction.

`amd-rocm-target` is a parameterized campaign slot, not a device identity.
Before any cell runs, one campaign binds it to one immutable target ID and
content-addressed envelope containing the exact SKU, architecture or ISA,
driver, ROCm, profiler and collective-library identities. Every row in one
claim uses that same binding. A different AMD target starts a separate
campaign, denominator and device-model identity; rows from two targets never
satisfy one claim together.

## Compact model and deterministic execution

`simllm-device-model-v1` contains closed `acceptance_status` exactly
`candidate` or `validated`, closed `target_basis` exactly `target-silicon` or
`architecture-derived`, a support envelope, shape schemas, selectors, exact
service entries, a typed resource registry and a per-entry evidence ledger. Its
`architecture-derived` basis requires candidate status; only target-silicon
evidence may carry validated status. The strict top-level object contains
exactly `schema`, `device_model_id`,
`device_kind_id`, `acceptance_status`, `target_basis`,
`device_identity_sha256`, `operating_envelope_sha256`,
`support_envelope_sha256`, `evidence_manifest_sha256`, `fit_sha256`,
`expectations_commit`, `dispatch_signature_sha256s`, `shape_schemas`,
`implementation_selector_sha256`, `collective_stage_selector_sha256`,
`resource_registry`, `interaction_contract`,
`host_initiation_profile_sha256`, `service_entries`,
`service_entry_evidence`, `scalar_profile_table_sha256`, `gpu_spec_sha256`,
`gpu_architecture_profile_sha256`, `gpu_device_config_sha256`,
`validation_record_sha256`, `validation_summary_sha256`,
`acceptance_bars_sha256` and `model_limits`. The expectations commit is
lowercase hexadecimal of length 40 or 64. Dispatch signatures are a sorted,
unique, nonempty digest tuple; shape schemas are nonempty and sorted uniquely
by schema ID.

The required implementation-selector reference resolves to declarative data
only. Code, callbacks, expressions and binaries reject. The collective-stage
selector reference is nullable exactly when the support envelope claims no
collective device stage. The resource registry and
`independent-resource-v1` contract are inline and match the model device kind.
Each service-entry member is exactly `{service_entry_id, entry}`, sorted and
unique by nonblank ID with unique `(implementation_id, shape_vector)` keys.
Each evidence member contains exactly `service_entry_id`,
`source_selection`, `source_record_sha256s`, `residual_record_sha256`,
`support_envelope_sha256`, `operating_envelope_sha256`,
`isolated_duration_ps` and exact rational `uncertainty_bound`, sorted
one-to-one with the entries. Source selection is exactly `silicon`,
`silicon-fit`, `accel-sim`, `analytical-transfer` or `simulator-derived`.

Device identity, operating and support envelopes, evidence manifest, fit,
implementation selector, validation record, validation summary and acceptance
bars are required content references. Collective selector, host initiation,
scalar profile table, GPU spec, GPU architecture profile and GPU device
configuration are explicitly nullable references. `model_limits` contains
exactly `max_shape_schemas`, `max_shape_axes_per_schema`,
`max_resource_axes`, `max_service_entries`, `max_epochs_per_entry` and
`max_resident_entries`; every value is a positive signed-128 integer and
bounds the corresponding count. Every nonnull reference is reachable from the
release closure and cross-record identities agree. Raw traces, sample vectors,
profiler rows and simulator dependencies stay outside the model.

The `simllm-device-resource-registry-v1` has exactly `schema`, `device_kind_id`,
sorted unique `active_axis_ids` and `axes` sorted by unique `axis_id`; active
IDs are a subset. Every strict axis has exactly `axis_id`, `axis_class`,
`service_scope`, `base_unit`, `clock_domain_id`, `capacity_source_id`, `rate`,
`residency_capacity` and `exclusive_capacity`. Class is exactly `throughput`,
`residency` or `exclusive`; service scope is exactly `device-internal`,
`peer-port` or `data-mover`. All capacity members are present and exactly the
class-appropriate member is nonnull. A rate is exact `{numerator, denominator}`
with reduced nonnegative numerator and positive denominator; residency capacity
is a nonnegative integer and exclusive capacity is a positive integer slot
count. The registry SHA-256 hashes this exact canonical record. A
cycle-denominated throughput rate requires a clock
domain while a wall-time rate leaves it null; other classes reject a rate. An
active axis with any positive accepted demand requires positive
class-appropriate capacity at load time.

Every `DeviceResourceVector` carries the registry hash and device kind plus
integer values and Boolean known bits aligned to the complete axis order. Every
known value is a nonnegative integer and every active-axis value is known; a
negative demand is a load error. A known zero means that service entry has no
demand on that axis; inactive axes use an unknown bit and canonical zero
placeholder that carries no demand. Unknown never becomes a known zero.
Throughput demand, residency demand and exclusive occupancy stay distinct from
each other and from core queue or bookkeeping resource references.

An immutable `DeviceServiceEntry` contains only its
`(implementation_id, shape_vector)` key and a nonempty ordered tuple of
immutable `ServiceEpochDefinition` values. Each epoch contains an aligned
resource-native demand vector and an optional fixed floor in integer
picoseconds. Entries contain no start time or editable rate. Throughput values
are consumable work; residency and exclusive values are held requirements and
never decrement. Admission reserves each axis's maximum residency and
exclusive requirement across all epochs for the entry's full lifetime. Runtime
resident state owns admission, epoch start, current epoch and remaining
throughput demand. An epoch advances only after every throughput demand reaches
zero and its fixed floor has elapsed. An ordinary entry releases its lifetime
reservations at final work completion; a collective stage follows the rank
lease-release rule above.

The only version-1 resource law is `independent-resource-v1`. Each throughput
axis divides its one registry capacity equally among resident epochs with
positive remaining demand on that axis, using exact rational arithmetic; axes
otherwise progress independently. `interaction_terms` is required to be empty.
A nonempty term set is rejected until a versioned interface amendment and a new
expectations-only freeze define and identify another law.
Launch mode affects only the host launch path. There is no device-front-end
service stage, and no residual is charged by subtraction alone. COMP-48
identifies the host term while preserving kernel service exactly.

The mechanistic version-1 model uses exact `DeviceServiceEntry` cells and never
interpolates resource demand or reservations. Its optional scalar profile-table
form may declare one integer shape axis for duration interpolation. For
`x0 < x < x1`, it evaluates exactly
`y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)`, with inclusive support,
canonical lower and upper cells, reduced rational arithmetic and one ceiling
when `y` becomes an externally visible integer picosecond duration. Exact hits
precede interpolation; a bracketing tie chooses the lower cell ID. It uses no
floating-point logarithm or exponential. Exact lookup is expected constant
time and declared one-axis bracketing is logarithmic in that axis. Generic
multi-axis interpolation remains COMP-4.

All canonical integers and rational numerators, denominators, cross-products
and accumulations must fit a signed 128-bit domain. Overflow is a load or
evaluation error, never saturation or wraparound. Rational comparisons remain
exact through a service epoch and apply one ceiling only when the complete
externally visible boundary becomes integer picoseconds.

Load builds and validates indexes once in `O(N log N)`. The hot path performs
no artifact parsing, fitting, subprocess execution or instruction replay.
For `k` resident requests and `R` active resource axes, one device event is
bounded by `O(kR)` and resident state by `O(kR)` plus the immutable model index.
Each artifact declares finite limits for entries, axes and resident requests
before it is accepted.

The initial `BatchKernelService` accepts the complete co-runnable tuple and
preserves current tuple order, cycle-native replay, integer ceiling,
composition-cursor and barrier mechanics. Enabled service may intentionally
change completions and metrics. The scalar and legacy off paths preserve their
service calls, cursor state, visits, reports and result bytes exactly. The
batch call is pure: immutable ordered requests, common start and immutable
snapshot enter; ordered service facts, accounting and a next snapshot return.
It mutates no live state and emits no callback or graph completion. Runtime
validates the whole result before atomic adoption, and failure changes nothing.
Incremental service is a separate
CORE-12 transaction. In incremental mode, `DeviceRuntime` alone selects and
orders a graph-level grant. The device service exposes a pure feasibility
query, then atomically reserves, services and releases the granted request. It
owns no graph dependency or callback.

The narrow capability is fixed as follows:

```text
OperationServiceResolver.resolve(graph, selected_device_models, dispatch_context)
  -> ResolvedOperationServiceBindingSet
CollectiveDeviceStageResolver.resolve(
  planned_graph, selected_device_models, dispatch_context
) -> ResolvedCollectiveDeviceStageSet | None
BatchKernelService.dispatch_batch(
  requests: tuple[ResolvedDeviceServiceRequest, ...],
  common_start_ps: int,
  snapshot: DeviceServiceSnapshot
) -> BatchKernelServiceResult[ordered ServiceFact, DeviceAccounting, next_snapshot]
IncrementalDeviceService.begin(snapshot)
  -> IncrementalDeviceServiceTransaction
transaction.admissible(request, now_ps) -> Feasibility       # pure
transaction.dispatch_granted(request, admission_sequence, now_ps)  # mutating
transaction.peek_next_event_ps() -> int | None
transaction.advance(to_ps) -> tuple[DeviceServiceEvent, ...]
transaction.release_held(subject_key, release_at_ps) -> ServiceFact
transaction.accounting() -> DeviceAccounting                # read-only
transaction.prepare(); transaction.commit(); transaction.abort()
```

Each `ResolvedDeviceServiceRequest` contains exactly a stable `subject_key`,
`service_entry_id` and `release_mode`, whose closed values are `work-finish`
and `external-frontier`. Batch accepts only `work-finish`. The compute-owned
`DeviceServiceSnapshot` has exactly
`device_instance_id`, `device_model_sha256`, `registry_sha256` and
`resident_states`; runtime composition cursors never enter it. Resident states
are canonically ordered by `(admission_sequence, subject_key)`, but version-1
batch dispatch accepts only an empty tuple and a successful total result
returns it empty. Nonempty resumable state belongs to CORE-12.
At common start and after every release, batch admission releases completed
reservations, scans pending requests in original tuple order, admits each
currently feasible request and skips the rest. It drains zero-duration
completions and newly feasible requests to a finite same-time fixed point before
advancing. Preflight rejects any request whose lifetime reservation maxima
exceed capacity.
Every service fact has exactly `subject_key`, `epoch_index`, `submitted_at`,
`eligible_at`, `started_at`, `work_finished_at`, `finished_at` and
`completed_at`. Ordinary batch service has `work_finished_at=finished_at`; the
separate value supports later collective lease projection.
Facts serialize in request tuple order, then ascending epoch index. The first
epoch has `submitted_at=eligible_at=common_start_ps` and starts at its admission
grant. Every later epoch has submitted, eligible and started equal to the prior
epoch's work finish. An intermediate epoch finishes and completes at its work
finish; the ordinary final epoch does the same. Batch rejects
`release_mode=external-frontier`; collective rank leases use the incremental
transaction below.
`DeviceAccounting` has exactly `registry_sha256`, aligned reduced-rational
`admitted_throughput` and `served_throughput`, and aligned integer
`acquired_reservations`, `released_reservations` and `held_reservation_ps`.
Nonapplicable axis classes carry zero; held reservation is demand units times
lease duration. Batch-result validation requires contiguous declared epochs,
exactly one final completion per input subject, no extra field, subject or
epoch, monotonic boundaries, served equal to admitted, released equal to
acquired, exact agreement with input entries, no resident leakage and a next
snapshot with the same device, model and resource-registry identities.

An incremental transaction resolves every binding before mutation, uses
prepare, commit and abort, and emits no callback before commit. Its device
engine is quiescent when runtime requests physical closure of that transaction;
it does not force unrelated network or background work to drain or strengthen
the framework-completion boundary. Runtime stages or clones arbitration state and one
atomic bookkeeping batch; engine, policy, runtime and bookkeeper all prepare
before the fixed, infallible adoption order: device engine, arbitration policy,
runtime state, one bookkeeping batch, then callbacks. A failure before adoption leaves
every live participant unchanged. At one timestamp, it releases and publishes all
device-internal completions before runtime applies readiness and arbitration.
Candidate enumeration is permutation invariant, same-time closure is finite
and a pending graph with neither an admissible request nor a future event is a
fatal dead state.

Incremental time is monotone. A grant's `now_ps` equals runtime's current
logical time. `DeviceServiceEvent` is a closed union of strict
`WorkFinishedEvent {kind: "work-finished", subject_key, epoch_index, at_ps}`
and strict `{kind: "service-fact", fact: ServiceFact}`. Intermediate epochs
and ordinary final epochs emit a service fact at work finish. An
`external-frontier` final epoch emits only the work-finished event and retains
its lifetime reservations. Runtime advances through the read-only external
frontier, then `release_held` releases at that current time and returns the
single final fact with the earlier work finish and
`finished_at=completed_at=release_at_ps`. Batch service accepts only
`work-finish`; this held-release path belongs solely to the CORE-12
transaction used by CORE-26.
Release is exactly once, final-epoch-only and follows that subject's
work-finished event. If traffic finished first, it occurs immediately at work
finish; otherwise runtime advances to the later terminal before release. A
held subject with no next compute event is waiting for an external event, not
quiescent or a fatal device dead state. Events sort by time, admission
sequence, epoch index and variant, with `work-finished` before `service-fact`.

Runtime candidate ties retain earliest eligibility followed by the existing
deterministic baseline order. Device-internal ties use runtime admission
sequence, then epoch index and canonical resource-axis order. Neither path may
depend on mapping or set iteration, workload labels or floating-point
comparison. The legacy SASS batch adapter retains its existing task, block,
warp and lane rules exactly.

## Contributor workflow

The stable offline command surface is:

```text
simllm-calibrate doctor
simllm-calibrate run
simllm-calibrate validate
simllm-calibrate pack
simllm-calibrate submit
```

`doctor` emits a typed environment record. `run` executes only a declared
local collector or optional simulator backend. `validate` and `pack` require no
GPU and no initialized simulator checkout. `submit` shows the generated diff
and requires an explicit authenticated action before opening a small data-only
review. CI validates content but never executes contributor binaries or raw
traces.

Archive validation rejects absolute or traversal paths, symlinks, duplicate
semantic keys, excessive file count, compressed or expanded size, unsupported
vendor or tool capabilities, missing rights and license metadata, mismatched
hashes, incomplete bindings, split leakage and physical-bound violations. A
candidate remains distinct from a validated profile throughout review.

## Execution waves

The waves separate interfaces, physical evidence and runtime behavior so no
lane consumes an unfrozen contract.

| Wave | Work | Exit gate |
|---|---|---|
| 0 | Freeze public interfaces, task ownership, support policy and the authored transformer-DAG expectation suite | Documentation and freeze checks pass; no behavior, simulator checkout or measured result lands |
| 1A | COMP-50 canonical records, validators, collector and doctor protocols, compiler shell and inert local CLI | Canonical vectors, malicious records, absent-tool behavior, split isolation and exact off-path checks pass; archive safety remains in Wave 5B |
| 1B | COMP-51 untouched Accel-Sim sidecar, licenses, pinned dependencies and offline smoke | Integration waits for 1A's canonical writer; default offline CI still requires no sidecar; separate networked release verification proves the exact pin fetchable and branch-reachable |
| 2 | COMP-6 vendor activity joins plus physical identity projection, with VLLM-12 and SGL-10 capture producers | Both frameworks emit total bindings for the same contract; disabled producers are exact |
| 3 | COMP-50 CUDA/ROCm doctor backends, COMP-5 qualification and A100, H100 and AMD silicon campaigns; COMP-22 communication-demand capture | Every retained campaign is non-void and content addressed; silicon lanes require 1A, not 1B |
| 4 | COMP-50 generic compilation; COMP-1, COMP-22 and COMP-24 target-specific fitting and acceptance; COMP-25 batch selection; CORE-45 provenance | Untouched test, live TTFT/TPOT and exact batch off-path pass |
| 5P | CORE-8 loss-checked projection for authoritative device visits; when an outer resource lacks a legacy kind, CORE-50 lands its strict registered reference before that resource's projection | Projected visits pass cross-language checks; any v2 reference also passes strict-old-byte checks |
| 5I | CORE-12 transactional later-arrival execution | Fault injection, quiescence, deterministic event order and arrival-offset sweeps pass |
| 5B | COMP-50 safe pack and explicit submit workflow, rights and license checks, and complete reachable-object validation | One data-only release validates identically on Linux and Windows without a GPU or initialized sidecar, and an external contributor produces the review with one explicit command |
| 6 | Conditional communication composition after 5I: CORE-11 shared HBM when demand is nonzero, CORE-13 intra-node peer service, CORE-26 cross-node GPU-to-RNIC composition through the external-frontier transaction, then CORE-27 only for an observed mover | No double charge and live communication metrics pass; CORE-50 joins only when a new outer resource kind is observed |
| 7 | Publish versioned reference profiles after Gate 5B; add COMP-52 candidate-only architecture derivation after a validated anchor exists | The identical data-only release passes contribution validation, status and provenance are visible on every promoted result, and unsupported paths fail closed |

Wave 1B development may proceed beside 1A, but its integration gate waits for
the canonical writer. A simulator-filled A100 lane joins Wave 4 only after both
1B and its qualified replay finish. Measured-only A100, H100 and AMD lanes do
not wait for Accel-Sim. Wave 5B may begin with Wave 1A synthetic objects and
finishes against a real Wave 4 release; it runs beside 5P and 5I. Offline
communication capture, fitting and batch-only checks can branch from Wave 4.
Live resident-stage composition waits for 5I because its compute-owned lease
uses CORE-12's external-frontier transaction even when no later kernel arrives.
A communication profile that emits the resident-stage visit consumes the
CORE-8 part of Wave 5P. CORE-50 joins only
when that profile emits an otherwise-unrepresentable outer queue resource;
registering internal service axes alone never requires it. Scalar and
compute-memory profiles with no new authoritative visit bypass Waves 5P, 5I
and 6, while communication-enabled profiles consume only the applicable gates.

Amendment (maintainer, 2026-08-24): execution priority inside these waves
follows measured silicon on the reachable cluster targets. The A100 and GH200
lanes run their capture and measurement campaigns first, tracked per target,
framework and model in the [calibration coverage matrix](calibration-coverage.md),
with COMP-54 supplying the offline model-workload extraction that enumerates
each column. Wave 1B proceeds only when a measured A100 column exposes an
explicitly missing exact point that the source precedence already reserves for
the sidecar: a kernel that can be measured on the target is measured, never
simulated. This amendment reorders execution and changes no interface,
schema, ownership or precedence rule above.

## Ownership map

This map names the lanes this document creates or reshapes. It is not the
registry: the module docs under `docs/modules/` remain the source of truth for
every task, including the ones the calibration suite depends on but does not
reshape.

- The calibration package (COMP-50) owns schemas, canonicalization,
  validation, compilation, both pure binding resolvers and the contributor
  workflow, including the canonical-bytes and ASCII-conformance schemas, the
  native ASCII verifier and the token-fixture schema. The untouched external
  sidecar is COMP-51.
- Generic noncollective and collective-stage capture joins, plus the MoE
  routing sidecar schema, belong to COMP-6; thin framework observations belong
  to VLLM-12 and SGL-10.
- Environment qualification is COMP-5. Per-target compute and memory numerical
  acceptance for A100, H100 and AMD is COMP-1; only the A100 lane owns
  Accel-Sim correlation and selective filling. Communication GPU demand is
  COMP-22. Whether the closed `independent-resource-v1` axes and residency
  rules explain held-out mixtures is identified and accepted by COMP-24. Any
  nonempty interaction form requires a versioned interface amendment and a new
  expectations-only freeze.
- Synthetic noncollective selection, resolved operation binding sets and live
  batch reachability are COMP-25. Collective-stage evidence comes from
  COMP-22, and its live rank-barrier composition is CORE-26. The complete
  resolved-device binding-closure provenance is CORE-45. Incremental admission
  and the compute-owned external-frontier lease capability that CORE-26
  consumes are CORE-12.
- Versioned core projection of registered device resources is CORE-50 and
  nothing else; loss-checked queue-visit projection is CORE-8.
- Vendor peer ports are COMP-35, measured port ceilings COMP-41, device
  composition CORE-11 and CORE-13, cross-node composition CORE-26, and only
  actually observed mover visits CORE-27.
- Explicit architecture-derived candidates are COMP-52, which never changes
  the validated default.
- The preflight physical-sanity amendment is frozen in
  `offline/calibration/suites/transformer-dag-v1/expectations-amendment-2026-08-24.json`;
  it closes COMP-53 before the first campaign cell runs.
