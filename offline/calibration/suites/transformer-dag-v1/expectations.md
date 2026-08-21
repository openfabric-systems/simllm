# Transformer DAG device calibration v1 expectations

## Freeze scope and chronology

This is the expectations-only record for `transformer-dag-v1`. It is authored
before the calibration package, capture adapters, measurement harnesses,
Accel-Sim sidecar, compact-model compiler and live service integration exist.
It also precedes every hardware or simulator execution governed by this suite.
No value produced by those later activities may be written back into this
freeze.

This directory contains authored inputs and expected relations only. It
contains no execution graph, content-addressed output object, canonical digest
golden, measured duration, profiler row, instruction trace, simulator cycle,
fit residual, result report or task closure. The machine freeze states
`closes: []` and names every registry task governed by this freeze.

The freeze test reads these documents only. It imports no SimLLM module and
does not invoke a subprocess, GPU, profiler, simulator or network operation.

## Boundary under test

The suite freezes three operation strata from one exact model substrate:

- `compute-prefill` selects representative dense matrix operations from a
  captured prefill DAG;
- `memory-decode` selects representative decode attention and streaming
  operations;
- `moe-communication-decode` captures routed MoE device work and its real
  collective boundaries.

These labels report coverage. They do not select a service equation. Resource
classification comes from later evidence, and relabeling a task cannot change
its predicted timestamps.

The first suite does not claim a separate dense checkpoint. It uses the exact
Granite MoE checkpoint already pinned by the repository and selects its dense
operation strata. Adding another checkpoint changes the authored suite and
requires a new expectations freeze before its first run.

The execution DAG remains `simllm-execution-graph-v1` byte for byte. Shape,
implementation and service records remain immutable sidecars. The compact
online artifact is `simllm-device-model-v1`; no profiler or simulator is
reachable from its service path.

## Frozen model and framework identities

| Item | Frozen identity |
| --- | --- |
| model | `ibm-granite/granite-3.0-1b-a400m-instruct` |
| model revision | `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445` |
| config SHA-256 | `ca4bb3a5c1bdef988ab413e0d731640446da65316e4ed16de3666cd96ecc3a0b` |
| weight SHA-256 | `f7ae1cee56a9ea6c5360437b1c0407f8d84816b2cc75470f4e7e5236fa2a07dc` |
| weight bytes | 2,669,283,096 |
| arithmetic | BF16, unquantized |
| model geometry | 24 layers, hidden 1024, intermediate 512, 16 query heads, 8 KV heads, head size 64, 32 experts, top-k 8, vocabulary 49155 |
| vLLM | version 0.26.0, source commit `568afb3a13806beb53bb2e6bd518269357b237c0` |
| SGLang | source commit `8f2a3ad6d7d68c58ae65b61a75bb2115449addca`, tree `5be26db1f559064c0f9e724e78c1a8f619754867` |

The single-device base has TP, PP, DP and EP width one. The
`moe-communication-decode` cells use four expert participants. Eager capture
is required. A CUDA graph or ROCm equivalent is required only after the
collector reports it as supported, and it is recorded as a distinct launch
mode. Unsupported mode is an explicit capability result, never a silent skip.

Case ordinal is the zero-based `graph_cells` array order. Request ordinal is
zero-based within the case, and request identity is
`<case-id>:request:<request-ordinal>`. Compute-prefill uses the authored request
count and prompt length; decode families use authored batch and context length.
For zero-based position, the exact token fixture is
`token_id = 256 + ((104729 * case_ordinal + 8191 * request_ordinal + 131 *
position) mod (vocab_size - 256))`. Every token is active, position IDs are
zero through length minus one and there is no padding beyond the authored
shape. Sampling is greedy at temperature zero with no random number generator.
One decode forward is captured and its output token is not recursively
consumed. The external fixture identity is lowercase SHA-256 of the later
canonical fixture record; this expectations-only freeze authors the formula and
does not contain that generated identifier. Capture records exact top-k MoE
routing in `simllm-moe-routing-sidecar-v1` rows containing case, request, layer,
token position and ordered expert IDs. Route divergence inside one claimed
envelope is fatal or requires a separate envelope.

The observation matrix expands every authored graph, communication and mixed
cell over every frozen target, framework and required launch mode. Each target
uses its supported real-silicon collector. Eager is always required; captured
graph is additionally required when the backend reports support. A
communication cell requires its authored participant capability, and a mixed
cell requires every member capability. A `blocked`, `not-applicable` or
`rejected` context produces no observation and never a zero. A missing required
context cannot support a validated claim. The denominators 15, 20 and 28 count
authored topology cells, not the expanded observations.

## Frozen graph cells

The 15 graph cells vary one primary axis within each family. Train cells alone
define support. Validation selects model form, source policy and uncertainty.
Test remains unopened until all those choices are frozen.

### Compute-prefill cells

Every cell carries four requests with an equal prompt partition.

| Cell | Split | Tokens per request | Total prompt tokens |
| --- | --- | ---: | ---: |
| `cp-train-r4-t128` | train | 32 | 128 |
| `cp-train-r4-t768` | train | 192 | 768 |
| `cp-train-r4-t2048` | train | 512 | 2048 |
| `cp-validation-r4-t512` | validation | 128 | 512 |
| `cp-test-r4-t1024` | test | 256 | 1024 |

### Memory-decode cells

Every cell carries batch four and one new token per request.

| Cell | Split | Context tokens |
| --- | --- | ---: |
| `md-train-b4-c128` | train | 128 |
| `md-train-b4-c1024` | train | 1024 |
| `md-train-b4-c8192` | train | 8192 |
| `md-validation-b4-c512` | validation | 512 |
| `md-test-b4-c2048` | test | 2048 |

### MoE-communication-decode cells

Every cell carries context 2048, one new token per request and four expert
participants.

| Cell | Split | Batch |
| --- | --- | ---: |
| `mc-train-b1-c2048` | train | 1 |
| `mc-train-b16-c2048` | train | 16 |
| `mc-train-b64-c2048` | train | 64 |
| `mc-validation-b4-c2048` | validation | 4 |
| `mc-test-b8-c2048` | test | 8 |

## Frozen communication cells

Communication uses real NCCL or RCCL evidence. Accel-Sim never certifies this
stratum. The runnable pairs are ring `all-reduce` and pairwise `all-to-allv`.
For ring all-reduce, scalar `payload_bytes` is the full reduced input bytes per
rank. For pairwise all-to-allv with an empty pair table, it is the bytes on
every ordered source-destination pair. Peer-port service bytes and EQ4 floors
are derived only after applying that operation and algorithm convention. The
payload slice fixes four participants and two channels. The participant slice
fixes 1,048,576 bytes and two channels. Their four-participant validation cell
is shared. The channel slice fixes four participants and 1,048,576 bytes: train
uses one and four channels, validation uses two, and untouched test uses three.
There are ten unique cells per operation and 20 in total.

Every cell is cross-node with ranks zero through participants minus one. Rank
`i` maps to node `i`, local GPU zero, over vendor-native GPU-direct RDMA. A
campaign without that placement or transport declares communication
unsupported rather than substituting another locality.

| Cell | Operation | Split | Participants | Payload bytes | Channels |
| --- | --- | --- | ---: | ---: | ---: |
| `comm-all-reduce-p4-b65536-ch2` | all-reduce | train | 4 | 65,536 | 2 |
| `comm-all-reduce-p4-b16777216-ch2` | all-reduce | train | 4 | 16,777,216 | 2 |
| `comm-all-reduce-p4-b1048576-ch2` | all-reduce | validation | 4 | 1,048,576 | 2 |
| `comm-all-reduce-p4-b4194304-ch2` | all-reduce | test | 4 | 4,194,304 | 2 |
| `comm-all-reduce-p2-b1048576-ch2` | all-reduce | train | 2 | 1,048,576 | 2 |
| `comm-all-reduce-p8-b1048576-ch2` | all-reduce | train | 8 | 1,048,576 | 2 |
| `comm-all-reduce-p6-b1048576-ch2` | all-reduce | test | 6 | 1,048,576 | 2 |
| `comm-all-reduce-p4-b1048576-ch1` | all-reduce | train | 4 | 1,048,576 | 1 |
| `comm-all-reduce-p4-b1048576-ch4` | all-reduce | train | 4 | 1,048,576 | 4 |
| `comm-all-reduce-p4-b1048576-ch3` | all-reduce | test | 4 | 1,048,576 | 3 |
| `comm-all-to-allv-p4-b65536-ch2` | all-to-allv | train | 4 | 65,536 | 2 |
| `comm-all-to-allv-p4-b16777216-ch2` | all-to-allv | train | 4 | 16,777,216 | 2 |
| `comm-all-to-allv-p4-b1048576-ch2` | all-to-allv | validation | 4 | 1,048,576 | 2 |
| `comm-all-to-allv-p4-b4194304-ch2` | all-to-allv | test | 4 | 4,194,304 | 2 |
| `comm-all-to-allv-p2-b1048576-ch2` | all-to-allv | train | 2 | 1,048,576 | 2 |
| `comm-all-to-allv-p8-b1048576-ch2` | all-to-allv | train | 8 | 1,048,576 | 2 |
| `comm-all-to-allv-p6-b1048576-ch2` | all-to-allv | test | 6 | 1,048,576 | 2 |
| `comm-all-to-allv-p4-b1048576-ch1` | all-to-allv | train | 4 | 1,048,576 | 1 |
| `comm-all-to-allv-p4-b1048576-ch4` | all-to-allv | train | 4 | 1,048,576 | 4 |
| `comm-all-to-allv-p4-b1048576-ch3` | all-to-allv | test | 4 | 1,048,576 | 3 |

A target without enough devices records the missing capability and cannot
claim validated communication coverage. It may retain valid single-device
evidence as candidate evidence, but it may not replace communication with a
zero or simulator result.

## Frozen collective device-stage seam

A strict `simllm-typed-dispatch-trait-v1` member has exactly `trait_id`,
`value_type` and `value`. Trait ID is nonblank; value type is exactly `integer`,
`string` or `boolean`, and the value matches it. Trait tuples sort uniquely by
trait ID. Each device has a strict `simllm-dispatch-signature-v1` record with
exactly `schema`, `framework_id`, `framework_version`, `backend_id`,
`backend_version`, `kernel_library_id`, `kernel_library_version`,
`algorithm_policy_id`, `device_isa`, `numeric_traits` and `layout_traits`.
Every string is nonblank, both trait members are strict typed-trait tuples, and
launch mode is forbidden.

The exact `simllm-device-dispatch-context-v1` record has `schema`,
`instance_graph_sha256`, `rank_device_assignments` and
`selected_device_models`. Each strict assignment has `rank` and
`device_instance_id`; assignments sort uniquely by rank and cover every graph
participant rank. Each strict selection has `device_instance_id`,
`device_model_id`, `device_model_sha256` and `dispatch_signature_sha256`;
selections sort uniquely by device instance and contain exactly the used
devices with no extras. Both resolved sets name the SHA-256 of this exact same
context. The closure and every present set agree on instance graph; both
present sets agree on dispatch context. Run provenance device/model tuples
match the selections and every resolved-set member.

A noncollective `OperationImplementationBinding` has exactly
`instance_graph_sha256`, `operation_id`, `launch_ordinal`,
`implementation_ref` and `shape_vector`. It is capture evidence only and
carries no measured demand, resource vector, fixed floor or service entry.
A fresh `ResolvedOperationServiceBinding` has `instance_graph_sha256`,
`operation_id`, `launch_ordinal`, `device_instance_id`,
`device_model_sha256`, normalized `semantic_key`, `shape_vector`,
`implementation_ref`, `service_entry_id`, `resolution_source` and required
nullable `observed_implementation_binding_sha256`. Resolution source is
exactly `observed-binding` or `selector`; the nullable hash is nonnull for the
former and null for the latter. The immutable
`simllm-resolved-operation-service-binding-set-v1` record has exactly
`schema`, `instance_graph_sha256`, `dispatch_context_sha256` and `bindings`.
Bindings follow graph operation tuple order, then launch ordinal. The resolver
consumes a total selected-model tuple keyed by device instance; every binding's
device and model match it. The total set resolves before runtime state mutates.

A `CollectiveDeviceStageBinding` joins the exact
`(instance_graph_sha256, collective_operation_id,
collective_plan_integrity_sha256, rank, launch_ordinal)` identity to the
observed `ImplementationRef` and typed `ShapeVector`. Those seven fields are
the complete observed binding. It carries no demand, service entry or action
frontier. Missing, duplicate and extra physical stages reject.

A fresh online `ResolvedCollectiveDeviceStage` contains the seven identity,
implementation and shape fields plus `device_instance_id`,
`device_model_sha256`, `service_entry_id` and `resolution_source` exactly
`selector`. It requires no historical observed binding; validation compares
the capture and resolved records separately. Only the device-model stage
selector may choose the entry, using the semantic collective, traffic-owned
plan topology and frozen dispatch context. The resolver consumes a total
selected-model tuple keyed by device instance, and every stage device/model
pair matches it. Version 1 accepts exactly one resolved resident stage per plan
rank. Multiple stages per rank reject until a later freeze has stream and
dependency evidence.

The resolved set carries exactly one `CollectiveDeviceRankFrontier` per plan
rank. It names collective operation, plan integrity hash and rank; its ordered
stage ordinals contain exactly one value in version 1. Its entry and terminal
action IDs are copied byte-identically from the plan. Validation proves every
plan rank and referenced action resolves, every stage is used exactly once and
there is no extra rank, action or stage.

The nonempty `simllm-resolved-collective-device-stage-set-v1` record has
exactly `schema`, `instance_graph_sha256`, `dispatch_context_sha256`, `stages`
and `rank_frontiers`. Stages follow graph collective tuple order, plan rank
order and launch ordinal. Frontiers follow graph collective tuple order and
plan rank order. A graph with no resolved collective stage omits this record
and puts null in the binding closure; an empty set record rejects.

Every resource axis carries closed `service_scope` exactly `device-internal`,
`peer-port` or `data-mover`; scope is never inferred from an axis ID. Version 1
permits positive resolved-stage demand only on `device-internal` axes and
rejects it on `peer-port` or `data-mover` axes.
The stage's `submitted_at` is parent collective launch completion and
`eligible_at_ps = max(submitted_at_ps,
rank_local_graph_predecessor_ready_at_ps)`. Device grant releases the existing
plan entry actions for that rank. The device engine reaches
`device_work_finished_at_ps` when its throughput demands and epoch floors
finish, but compute retains the stage's lifetime residency and exclusive
reservations. Traffic alone owns plan actions, extents and network service.

For a legal sparse rank, define
`traffic_terminal_at_ps = max({eligible_at_ps} union
{terminal_action_completed_at_ps for terminal_action_id in
terminal_action_ids})`. The singleton identity element keeps an empty terminal
tuple defined. Rank completion is
`rank_release_at_ps = max(device_work_finished_at_ps,
traffic_terminal_at_ps)`. A collective stage is an incremental request with
`release_mode=external-frontier`. Runtime calls the compute-owned
`release_held(subject_key, rank_release_at_ps)` after reading the traffic
terminal and advancing through that boundary; the method returns the final
fact directly, and traffic never mutates the reservations. Compute releases
the lifetime residency and exclusive reservations at that boundary, and
`rank_completed_at_ps = rank_release_at_ps`. No plan action depends on rank
release, so the gate is acyclic. The
parent completion is
`collective_completed_at_ps = max(rank_completed_at_ps for all plan ranks)`.
The device stage is neither `ComputeWork` nor an independent `CompletionEvent`
and emits no graph completion. Its one authoritative composite `QueueVisit`
remains under the parent collective operation and uses subject identity
`<operation-id>:rank:<rank>:stage:<launch-ordinal>`. Its `finished_at` equals
`rank_release_at_ps`; its `completed_at` equals the same boundary. Version 1
uses the existing legacy `GPU_SCHEDULER` `ResourceRef`.
Internal service axes never become separate queue visits. The semantic
collective remains the sole graph lifecycle and completion authority.
The interval from `device_work_finished_at_ps` to `rank_release_at_ps` is
lease-held occupancy evidence, never another additive kernel or network
latency term.

An entry whose every active-axis demand is known zero in every epoch and whose
every epoch floor is null or zero, or a disabled stage, emits no device visit,
delays no plan entry and preserves every accepted artifact and timestamp byte
for byte. A positive epoch floor prevents the bypass. The canonical
`ResolvedDeviceBindingClosure` is exactly the four-field
`simllm-resolved-device-binding-closure-v1` record: `schema`,
`instance_graph_sha256`, `operation_service_binding_set_sha256` and nullable
`collective_device_stage_set_sha256`. Its external identifier hashes those
exact canonical bytes.

`simllm-run-provenance-v2` preserves every version-1 field unchanged and adds top-level
`instance_graph_sha256`, `resolved_device_binding_closure_sha256` and
`device_models`. For compact-device execution, inherited `source_schema` is
exactly `simllm-execution-graph-v1` and inherited `source_sha256` equals
`instance_graph_sha256`; any disagreement rejects. Each device-model entry has
exactly `device_instance_id`,
`device_model_id`, `device_model_sha256`, `acceptance_status`, `target_basis`
and `operating_envelope_sha256`, with no extras. Entries sort canonically by
device instance, exactly one per device; heterogeneous entries are legal.
Status and basis copy unchanged. The selected model and envelope records are
reachable and verified in result-artifact closure. This complete provenance
reaches `StepResult`, TTFT and TPOT. Its bytes use the core version-1-family
compact UTF-8 JSON convention plus exactly one terminal LF. The `StepResult`
provenance reference hashes the full bytes including that LF. This convention
does not change calibration canonical records, which have no terminal newline.

## Frozen mixed-resource matrix

Representative bindings are selected without timing information:

- compute is the captured dense GEMM binding with greatest authored FLOPs;
- memory is the captured decode-attention or streaming binding with greatest
  authored compulsory bytes;
- communication is the captured collective binding with greatest logical
  payload;
- ties use canonical implementation identity.

The seven arms are compute, memory, communication, compute plus memory,
compute plus communication, memory plus communication and all three together.
Width is the number of copies of every arm member. Copy identity is
`<cell-id>:<member>:<rep>`, where `rep` is zero-based. The tuple is flattened
member-major in canonical compute, memory, communication order, then by
repetition ordinal. This tuple is the exact `BatchKernelService` request order.
Every copy has the same submitted and eligible timestamp and shares one device
instance. Each arm runs at width one and four in train, width two in validation
and width three in test. Cell identity is `<arm-id>-w<width>`, which fixes 28
mixed cells.

The single-resource arms are isolated controls. Pair and triple arms identify
sharing. The deliberately serialized run of the same members supplies the
control ceiling. The mixed physical floor is the maximum of all applicable
member-copy floors. A counter replay never supplies the concurrency timeline.

## Split rules

Kernel fit units are `(implementation_id, shape_vector)`. If one unit appears
in several authored graph cases, all occurrences receive the first split in
the precedence train, validation, test. Repetitions of one cell never cross a
split. A claimed family must retain at least one unique test unit.

Full-graph and mixed rows use `template_graph_sha256` plus case identity, so a
new shape case is distinct even when topology is unchanged. Stable tail
kernels may use a singleton exact train entry. A singleton does not
interpolate, extrapolate or participate in concurrent sharing without measured
resource demand.

Support is the region defined by train rows only. Validation may choose model
form, simulator source policy and uncertainty but cannot expand that region.
Test is opened only after fit and validation choices are immutable. An outside
test row is `unsupported_by_train_envelope`; it has no residual and counts
against support coverage.

## Measurement protocol

Each timing cell has ten warmup launches and 41 retained repetitions. Exact
clock policy and observed clock bands are environment identity. A validated
controlled environment requires exclusive compute access and the frozen
population coefficient-of-variation bar.

Four passes remain separate:

1. The timeline pass observes graph, stream, event and completion boundaries
   without per-kernel timing events.
2. The counter pass replays faithful isolated implementations and cannot serve
   as overlap evidence.
3. The NVIDIA dynamic-instruction pass traces only selected supported cells.
4. The mixed pass measures isolated, pairwise, triple and full-graph
   makespans with low-overhead activity.

Code and cache state are steady-warm. Data buffers rotate across a working set
at least twice the measured L2 capacity recorded in the controlled environment
before cell execution. The isolated counter pass validates authored compulsory
HBM bytes plus cache and HBM counters. A cell requested for the HBM bucket
rejects when its observed data working set is L2-resident.

A preflight returns `ready`, `blocked`, `not-applicable` or `rejected`.
Blocked or inapplicable work produces no observation. Expected rejection is a
capability result, not a zero-duration cell.

## Exact arithmetic and physical equations

All authored numbers in the machine freeze are integers. Rates and thresholds
are reduced rational numerator and denominator pairs. Internal evaluation is
exact, and one ceiling converts the complete external boundary to integer
picoseconds.

Canonical record bytes use UTF-8 without a byte-order mark and normalize all
Unicode to NFC. Object keys sort lexicographically by normalized Unicode scalar
sequence; arrays retain their schema-defined order. A loader rejects duplicate
keys both before and after normalization and rejects unpaired surrogates.
Strings escape quote, backslash and the five JSON short controls, encode every
other U+0000 through U+001F control as lowercase `\u00xx`, never escape slash,
and emit every other scalar directly as UTF-8. Integers use minimal base-10
spelling with no plus sign, leading zero or negative zero. Boolean and null
literals are lowercase. Canonical bytes contain no insignificant whitespace,
byte-order mark or terminal newline.

Exact picoseconds, cycles, bytes, counts and FLOPs are JSON integer tokens and
are parsed losslessly before schema bounds apply. Fitted coefficients are
precision-declared decimal strings with no exponent, plus sign, redundant
leading or trailing zero, or negative zero. The external record identifier is
lowercase hexadecimal SHA-256 over exactly those canonical bytes and is never a
JSON member. The full-domain authority is CPython 3.10 with Unicode database
13.0.0. The independent C++17 ASCII conformance verifier rejects non-ASCII input
and covers every structural type, duplicate rejection, control escapes,
arbitrary-length integer lexemes and a project SHA-256 implementation; it makes
no full-Unicode claim.

The calibration canonical writer serializes the exact strict
`simllm-execution-graph-v1` JSON object. Exact canonical unbound bytes produce
`instance_graph_sha256`; every unbound compute record retains
`nominal_duration_ps` and `uncertainty_fraction` as explicit null members.
Calibration binding rejects a graph whose `ComputeWork.config` contains any
float-valued scalar because the calibration canonical grammar has no float
spelling. The existing version-1 graph reader remains unchanged. The
calibration canonical writer separately emits
`simllm-execution-graph-template-v1`. Its strict top level has exactly
`schema`, `operations`, `completion_operation_ordinals` and
`collective_plans`; the collective tuple remains present when empty. Each
operation has exactly `rank_ordinal`, `logical_queue_ordinal`, `priority`,
`work`,
`depends_on_operation_ordinals` and
`participant_local_depends_on_operation_ordinals`. Operations preserve source
tuple order, and array position is the operation ordinal. The normalized rank
map is the ascending union of every operation anchor, collective participant,
control destination, accepted DMA endpoint and plan rank. Within each
normalized rank, queues map to ordinals by first operation occurrence. Both
dependency tuples are sorted unique ordinals. An empty source completion
frontier normalizes to all operation ordinals; an explicit frontier becomes a
sorted unique tuple.

`work` is a closed strict union. Its exact variants are `{kind: "compute",
kernel}`, `{kind: "kv-cache", action}`, `{kind: "dma", source_role,
destination_role}`, `{kind: "collective", collective, algorithm_hint,
rank_ordinals, channel_ordinal, pair_rank_ordinals}` and `{kind: "control",
mode, message, destination_rank_ordinals}`. Nullable `algorithm_hint` remains
present. Collective and control rank tuples preserve source order. Each sparse
collective pair is exactly `[source_rank_ordinal, destination_rank_ordinal]` in
source aggregate-pair order with its bytes removed; an empty pair table stays
empty. An effective collective channel is `channel_hint` or canonical
`default`, mapped to a graph-global ordinal by first collective-operation
occurrence. This preserves channel equality while removing spelling. DMA
endpoint-role normalization is
`simllm-device-endpoint-role-v1`; its exact values are `host`, `host:pinned`,
`host:pageable`, `gpu:<rank>`, `gpu:<rank>:hbm` and `cuda:<rank>`. GPU and CUDA
rank tokens rewrite through the same map and must resolve; an unknown role
rejects.

Each projected collective plan has exactly `operation_ordinal`, `algorithm`,
`channel_ordinal`, `rank_order`, `rounds`, `actions`,
`extents`, `entry_action_ordinals` and `terminal_action_ordinals`. Plans follow
source plan tuple order, which graph validation aligns to collective-operation
order. `rank_order` contains normalized ordinals preserving source rank order.
The plan channel repeats its semantic collective's effective channel as a loss
check. Round transfer channels use a separate graph-global namespace assigned
by first plan and round occurrence. A round has exactly
`{transfer_channel_ordinal}` and its array position is its ordinal. An action
has exactly `{rank_ordinal, kind, extent_ordinal,
depends_on_action_ordinals}` with a sorted unique dependency tuple, and its
array position is its ordinal. An extent
has exactly `{round_ordinal, source_rank_ordinal,
destination_rank_ordinal, send_action_ordinal, receive_action_ordinal}` and its
array position is its ordinal. Entry and terminal maps have exactly
`{rank_ordinal, action_ordinals}` in plan rank order, with sorted action tuples.
Every reference is ordinal-rewritten and total.

The projection excludes execution, step, operation, queue, channel, action,
extent, request, pool, block, descriptor and correlation identities; release
and not-before timestamps; placement epochs; compute config, FLOPs, HBM,
duration and uncertainty; every other KV field; DMA bytes; collective,
control, plan and extent payloads; request attribution; round tags and indices;
shape, service and demand values; and integrity hashes. Excluded-only changes,
consistent ordinalized-identity renames, dependency, completion or frontier
action-set tuple permutations, empty versus explicit-all completion and null
versus `default` channel spelling preserve the hash.
Changing any retained source tuple order except explicitly sorted dependency,
completion and frontier action-set fields changes it. Priority, dependency
scope, effective completion, retained rank or queue equivalence, work kind or
family, DMA role, collective rank order, sparse-pair support, channel sharing,
plan presence and every retained round, action, extent or frontier edge are
also sensitive. Rank relabeling is invariant only when source-rank order is
preserved. The projection is idempotent, leaks no raw identity, rejects an
unknown role or unresolved reference, groups splits and never selects service.
These are fatal validation guards.

The canonical `simllm-device-resource-registry-v1` record has exactly
`schema`, `device_kind_id`, `active_axis_ids` and `axes`. Axis IDs are unique,
axes sort lexicographically by axis ID, and active axis IDs are a sorted unique
subset. Every strict axis has exactly `axis_id`, `axis_class`, `service_scope`,
`base_unit`, `clock_domain_id`, `capacity_source_id`, `rate`,
`residency_capacity` and `exclusive_capacity`. Service scope is exactly
`device-internal`, `peer-port` or `data-mover` and is never inferred from axis
identity. All three capacity members are required, and exactly the
class-appropriate one is nonnull. A rate has exactly `numerator` and
`denominator`, with a reduced nonnegative integer numerator and positive
integer denominator. `throughput` requires a rate and a clock domain only when
its denominator is device cycles; a wall-time rate leaves the clock domain
null. `residency` carries nonnegative integer capacity units and null rate;
`exclusive` carries a positive integer slot count and null rate. The registry
SHA-256 hashes this exact canonical record.

A resource vector carries registry hash, device kind, integer values and known
bits aligned to every registry axis. Known values are nonnegative integers;
negative values reject. Every active-axis value is known: positive means this
entry demands the axis and zero means it does not. An inactive axis uses an
unknown bit and canonical zero placeholder, and an unknown active axis rejects
rather than becoming known zero. A `DeviceServiceEntry` is an immutable
`(implementation_id, shape_vector)` key plus an ordered nonempty tuple of
immutable `ServiceEpochDefinition` values. Each definition carries one aligned
resource vector and a fixed floor that is either null or a nonnegative integer
number of picoseconds. An entry or definition stores no start time or editable
service rate. Runtime resident state alone owns the current epoch index, integer
start time and aligned remaining demands after admission. It derives rates from
registry capacities, current residency and the declared interaction law, never
from the entry or epoch definition.

The interaction contract has exactly `interaction_law` and
`interaction_terms`. Version 1 accepts only `independent-resource-v1`;
`interaction_terms` is required empty and a nonempty value rejects. One event
divides each throughput axis's one registry capacity equally among resident
epochs with positive remaining demand on that axis, using exact rational
arithmetic. Resource axes progress independently. One event costs `O(kR)` for
`k` resident entries and `R` active axes. It has no interaction-term dimension.
Throughput components are consumable remaining work and decrement. Residency
and exclusive components are held requirements, never remaining work.
Admission reserves the per-axis maximum residency and exclusive requirement
across all epochs for the entry lifetime, before service begins. An epoch
advances in order only when all throughput demands reach zero and its fixed
floor has elapsed. Ordinary entries release every reservation at final work
completion; collective stages instead release at the frozen rank lease
boundary.

If any accepted entry has positive demand on an active axis, model load
requires positive class-appropriate capacity on that axis before admission.
Zero throughput or residency capacity is legal only when no accepted entry
demands that active axis, or when the axis is inactive. Exclusive capacity is
always positive. This rejects permanent dead states at load. Core references
remain the closed
`ResourceRef` or `RegisteredDeviceResourceRef` union and never accept an
unregistered axis string. The registered form identifies registry, device kind,
concrete device, axis, concrete resource instance and latency owner.

Mechanistic `DeviceServiceEntry` lookup is exact-cell-only. Resource demands
and reservations are never interpolated. Only an optional scalar duration
profile table may declare one integer shape axis while every other axis remains
pinned. Exact hits take precedence. Between canonical cells `x0 < x < x1`,
inclusive support uses EQ7 with reduced rational arithmetic and selects the
lower cell ID on a bracketing tie. It applies one ceiling only when `y` becomes
an externally visible integer-picosecond duration and uses no floating-point
logarithm or exponential. Exact lookup is expected constant time; declared
one-axis bracketing is logarithmic in that axis.
Generic multi-axis interpolation is unavailable, and a query differing on two
or more axes fails closed.

The compact model wire vocabulary is closed. `acceptance_status` is exactly
`candidate` or `validated`, and `target_basis` is exactly `target-silicon` or
`architecture-derived`; aliases and alternative spellings reject.
`architecture-derived` requires `candidate`, and the combination `validated`
plus `architecture-derived` rejects. Run provenance copies both wire values
unchanged. Runtime selects validated models by default and admits a candidate
only under explicit experimental opt-in.

`BatchKernelService` has one exact pure call:
`dispatch_batch(requests: tuple[ResolvedDeviceServiceRequest,...],
common_start_ps: int, snapshot: DeviceServiceSnapshot) ->
BatchKernelServiceResult`. Its inputs are immutable; requests are ordered and
retain that order. Scalar and legacy off paths preserve service calls,
composition cursors, barriers, visits, reports, result bytes and timestamps
byte for byte. Every request has stable `subject_key` and exact
`service_entry_id` plus `release_mode`; those are its exact three fields.
Release mode is closed to `work-finish` or `external-frontier`. Batch service
accepts only `work-finish` and rejects `external-frontier`.
`DeviceServiceSnapshot`
has exactly `device_instance_id`, `device_model_sha256`, `registry_sha256` and
`resident_states`; runtime composition cursors are excluded. Residents sort by
`(admission_sequence, subject_key)`. Version-1 batch input requires the resident
tuple empty and a successful next snapshot returns it empty. At common start
and after every release, admission first releases completed reservations, then
scans pending requests in original tuple order. It admits each feasible request
and skips each infeasible request, which is stable first-fit. Zero-duration
completions and newly feasible requests drain to a finite same-time fixed point
before time advances. Preflight rejects a request whose lifetime reservation
maxima exceed device capacity. The result contains `service_facts` as an ordered
`tuple[ServiceFact,...]`, one `DeviceAccounting` and `next_snapshot` as a
`DeviceServiceSnapshot`. Every fact repeats subject key, epoch index and
submitted, eligible, started, work-finished, finished and completed boundaries.
Batch service has `work_finished_at=finished_at`. Facts serialize in request
tuple order, then ascending epoch index. Epoch zero has
`submitted_at=eligible_at=common_start_ps` and `started_at` equal to its
admission grant. Every later epoch has
`submitted_at=eligible_at=started_at` equal to the prior epoch's
`work_finished_at`. An intermediate epoch and an ordinary final epoch have
`finished_at=completed_at=work_finished_at`. Validation
proves contiguous declared epochs, each expected subject and epoch appears
exactly once, every input completes exactly once, no extra subject or epoch
or field exists, and boundaries are monotonic. `DeviceAccounting` has exactly
`registry_sha256`, aligned reduced-rational `admitted_throughput` and
`served_throughput`, plus aligned integer `acquired_reservations` and
`released_reservations` and `held_reservation_ps`. Throughput totals are zero
on non-throughput axes; reservation and held-reservation totals are zero on
throughput axes. Held reservation is demand units times lease duration. Served
equals admitted, released equals acquired and every vector agrees with input
service entries.
There is no resident-state leakage.
The next snapshot keeps the input device, model-hash and registry identities.
It is compute-owned, with the exact outer fields above and no runtime cursors.
The service mutates no input,
invokes no callback and emits no graph completion. Runtime validates the
complete result and only then atomically adopts its next snapshot and service
facts. Failure leaves live state unchanged. Later-arrival methods remain
exclusively on the incremental device-service transaction interface.

`IncrementalDeviceService.begin(snapshot)` returns an
`IncrementalDeviceServiceTransaction`. It exposes pure
`admissible(request, now_ps) -> Feasibility`, mutating
`dispatch_granted(request, admission_sequence, now_ps)`,
`peek_next_event_ps() -> int | None`,
`advance(to_ps) -> tuple[DeviceServiceEvent,...]`, mutating
`release_held(subject_key, release_at_ps) -> ServiceFact`, read-only
`accounting() -> DeviceAccounting`, and `prepare()`, `commit()` and `abort()`.
The snapshot is the only `begin` argument and owns device, model and registry
identity. The exact `DeviceServiceEvent` union has strict
`WorkFinishedEvent {kind: "work-finished", subject_key, epoch_index, at_ps}`
and strict `ServiceFactEvent {kind: "service-fact", fact}`, where `fact` is one
strict `ServiceFact`. Transaction time never decreases, and every grant's
`now_ps` equals runtime's current logical time. Intermediate epochs and
ordinary final epochs emit `ServiceFactEvent` at work finish.
A `work-finish` request releases normally and emits its final
`ServiceFactEvent`. An
`external-frontier` request retains lifetime reservations after work finish,
and its final epoch emits only `WorkFinishedEvent` instead of
`ServiceFactEvent`. `release_held` is valid exactly once, only for that mode's
final epoch, and only after that subject's work-finished event. Runtime first
advances through `release_at_ps`; it equals the transaction's current time and
is no earlier than work finish. The returned final fact preserves the earlier
`work_finished_at` and has
`finished_at=completed_at=release_at_ps`. The method returns that fact directly
and never stages a same-time advance. If the traffic terminal is already known,
release follows work finish immediately; otherwise runtime advances to the
later terminal first. A held subject with `peek_next_event_ps()=None` waits for
an external event; it is not global quiescence or a fatal device dead state.
Events sort by time, admission sequence, epoch index and kind, with
`work-finished` before `service-fact`. Runtime alone selects a grant.

All integer values, rational parts, cross-products and accumulations fit signed
128-bit arithmetic. Overflow rejects on load or evaluation. Comparisons use
exact cross-products and apply one ceiling only at the complete externally
visible picosecond boundary.

- **EQ1** `record_id = lowercase_hex(SHA256(canonical_record_bytes))`.
- **EQ2** `compute_floor_ps = ceil(flops * compute_rate_den * 10^12 / compute_rate_num)`.
- **EQ3** `memory_floor_ps = ceil(hbm_bytes * hbm_rate_den * 10^12 / hbm_rate_num)`.
- **EQ4** `peer_floor_ps = ceil(peer_bytes * peer_rate_den * 10^12 / peer_rate_num)`.
- **EQ5** `isolated_floor_ps = max(compute_floor_ps, memory_floor_ps, peer_floor_ps, kernel_floor_ps)`.
- **EQ6** `graph_floor_ps = longest_dependency_path_sum(applicable_stage_floors_ps)`.
- **EQ7** `y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)` only for the optional scalar duration table's one declared integer axis.
- **EQ8** `ape = abs(predicted_ps - silicon_ps) / silicon_ps`.
- **EQ9** `support_coverage = supported_required_test_units / all_required_test_units`.

Before its first observation, each finite-bound campaign content-addresses and
freezes positive reduced-rational minimum compute, HBM, peer-port and transport
rates, plus nonnegative integer maximum host-launch, device-fixed and
per-transport-action fixed times. Every term cites preexisting qualified
evidence, never the current cell outcome; a contributor never invents a finite
guarantee. A kernel ceiling serially sums both fixed terms and the
ceilings of FLOPs, HBM bytes and peer bytes at those minimum rates. A
communication ceiling serially sums its host and device fixed terms, derived
peer bytes at minimum peer rate, and every traffic action's fixed term plus
bytes at minimum transport rate. A graph ceiling is the sum of applicable
kernel and communication ceilings in a fully serialized topological order. A
mixed physical ceiling sums the member ceiling for every authored copy.
When no defensible finite ceiling exists, an isolated or graph cell declares
`unbounded` before observation and may not borrow its eventual measured value.

Every cell materializes its applicable floor and stratum ceiling before its
timing is read. A simultaneous mixed cell must also be no slower than the
measured deliberately serialized same-members control, whose own timing must
pass its physical ceiling. A timing outside an applicable interval, or a mixed
makespan above that control, is fatal. Being inside the bounds is necessary and
not sufficient.

## Acceptance bars

The evidence classes remain separate. Matrix cell counts are authored-input
counts, not scores. Fatal guards are unscored. Exact structural checks,
behavioral relation families and native executables are never summed into one
headline denominator.

- Supported graph operation coverage is exactly 1/1.
- Supported graph physical-launch coverage is exactly 1/1.
- Validated test support coverage is exactly 1/1.
- Controlled-environment population coefficient of variation is below 1/50.
- Untouched kernel-test median APE is below 1/10 and nearest-rank p95 APE is
  below 1/5.
- Untouched phase median APE is below 1/20 and nearest-rank p95 APE is below
  1/10.
- Compute-only full-step APE is below 1/20.
- Mixed completion and queue-wait error is no larger than the greater of two
  GPU cycles or 1/10 of the silicon value.
- Untouched three-channel completion and queue-wait error reuse that same
  greater-of-two-cycles-or-1/10 bar.
- On untouched launch-mode cells, host initiation predicts the observed
  residual within the greater of two GPU cycles or 1/10 while kernel service
  remains identical.

## Expected behavioral relation families

There are twelve relation families. Their parameterized instances are reported
within their own families and are never added to fatal guards or matrix-cell
counts.

- **R1 Physical-floor monotonicity.** For one implementation, increasing an
  authored work quantity cannot lower the corresponding physical floor.
- **R2 Capacity scaling.** Halving one identified throughput capacity doubles
  that resource service term exactly and cannot reduce complete service.
- **R3 Train-defined support.** A validated claimed capability supports every
  required test unit inside the support envelope defined from train rows only.
- **R4 Kernel accuracy.** Untouched supported kernel test rows meet the frozen
  median and p95 APE bars.
- **R5 Phase and step accuracy.** Untouched phase rows and the compute-only
  full step meet their separately frozen error bars.
- **R6 Mixed-resource bounds.** Every simultaneous mixed-resource makespan
  lies between its applicable floor and deliberately serialized control.
- **R7 Mixed-resource accuracy.** Mixed completion and queue wait stay within
  the larger of two GPU cycles or ten percent on untouched cells.
- **R8 Communication scaling.** At fixed participants, scaling payload by an
  authored integer ratio scales the peer-port service floor by that exact ratio
  and cannot reduce completion.
- **R9 Selective A100 simulator fill.** Inside the train-defined target and
  qualified SM80 envelopes, an exact-only categorical region with no exact
  silicon row or validated silicon-fit entry selects qualified Accel-Sim only
  between real anchors; a later valid silicon fit for that exact region
  outranks it, and extrapolation remains forbidden.
- **R10 Live and disabled paths.** On a serial compute-only graph, changing
  exactly one critical-path compact-service duration by signed `delta_ps` while
  holding every other term fixed changes `StepResult` completion and prefill
  TTFT or decode TPOT by exactly `delta_ps`; the disabled path preserves every
  accepted byte and timestamp exactly.
- **R11 Host-launch residual identification.** On untouched launch-mode cells,
  measured host initiation predicts the observed residual within the greater
  of two GPU cycles or ten percent while kernel service remains identical.
- **R12 Channel repartitioning.** At fixed operation, algorithm, participants
  and payload, changing channel count preserves logical bytes and the peer-port
  serialization floor exactly. No completion monotonicity is assumed.
  Untouched channel-count-three completion and queue wait meet the larger of
  two GPU cycles or ten percent.

## Fatal guards, unscored

No guard is survivable. One violation voids the affected device campaign,
retains its evidence and closes no task. A void campaign for one target does
not invalidate an independent target campaign.

- **G1 Identity envelope.** Target, framework, model, toolchain, launch mode
  and environment identities equal the declared run envelope.
- **G2 Environment stationarity.** Before and after device UUID, SKU,
  partition, clock policy, power policy and foreign-process state agree.
- **G3 Graph validity.** Every execution graph is strict
  `simllm-execution-graph-v1`, acyclic and has a complete completion frontier;
  its token fixture equals the authored generator, every referenced collective
  plan has exact rank membership and valid entry and terminal action identities,
  and MoE routing does not diverge inside one claimed envelope.
- **G4 Physical-launch totality.** Activity and correlation joins are total:
  every noncollective physical launch has exactly one operation binding and
  every bound noncollective operation has exactly one physical launch; a
  semantic noncollective operation hiding two or more launches is unsupported.
  Every collective physical launch has exactly one stage binding, and neither
  ledger has a missing, duplicate or extra launch.
- **G5 Binding totality.** Every supported capture has a total observed binding
  for each noncollective operation and collective device stage; every online
  modeled operation and collective plan rank has exactly one resolved service
  entry, version 1 has exactly one resident stage per rank, and validation never
  rewrites an observed implementation.
- **G6 Split isolation.** No kernel, graph or repetition unit crosses the
  frozen train, validation and test boundary.
- **G7 Physical interval.** Every timing lies between its precomputed physical
  floor and frozen stratum ceiling, and every simultaneous mixed-resource
  makespan also lies at or below its deliberately serialized same-members
  control.
- **G8 Raw evidence integrity.** Every external raw-blob digest, byte count and
  recomputed compact summary agrees with its evidence record.
- **G9 Conservation.** Task, operation, launch, collective rank, plan action,
  device stage and byte accounting conserve exactly with no hidden loss,
  duplication or extra member.
- **G10 One timing authority.** No GPU-port, collective, RNIC or fabric
  interval or byte is charged by two timing authorities; version-1 collective
  device stages permit positive demand only on explicit `device-internal` axes
  and reject it on `peer-port` or `data-mover` axes, which traffic alone owns.
- **G11 Pass separation.** Timeline, counter and dynamic-instruction passes
  remain separate, and counter replay is never used as the concurrency
  timeline.
- **G12 Nonperturbing capture.** No per-kernel timing event perturbs a short
  production kernel, device-side event tracing is disabled unless separately
  frozen, code and cache are steady-warm, data rotation spans at least twice
  measured L2, and compulsory-HBM plus cache and HBM counters reject an
  L2-resident requested HBM bucket.
- **G13 Resource-state validity.** Resource vectors reject unknown active or
  negative known demand; demanded active axes have positive capacity; only
  `independent-resource-v1` with empty interaction terms loads; epoch
  reservation, progress, ordered advance and final release follow the frozen
  semantics.
- **G14 Record graph integrity.** The content-addressed record graph is
  acyclic, complete and consistent with the one canonical byte contract.
- **G15 Arithmetic integrity.** Exact integer and reduced-rational arithmetic
  applies one final picosecond ceiling and remains inside the declared overflow
  bound.
- **G16 Simulator support.** Accel-Sim serves only supported SM80 cells and
  rejects H100, AMD and unsupported SM80 features.
- **G17 Simulator communication exclusion.** Accel-Sim supplies no
  communication-stratum evidence or fabric timing.
- **G18 Compatibility identity.** Calibration-off, absent-profile,
  absent-submodule, batch scalar and legacy off paths, disabled
  collective-stage and collective-stage paths whose every active-axis demand
  is known zero in every epoch and every epoch floor is null or zero preserve
  service calls, composition cursors, barriers, visits, reports, result bytes
  and timestamps exactly; a positive floor prevents bypass, a bypass emits no
  device visit and delays no plan entry, and a rejected batch result leaves
  live state unchanged.
- **G19 Live provenance.** `ResolvedDeviceBindingClosure`, model identity,
  acceptance status, target basis and operating envelope reach `StepResult`,
  TTFT and TPOT provenance without disagreement.
- **G20 Fatal means void.** One fatal violation makes the affected device
  campaign void, retains its evidence and closes no task.

## Accel-Sim capability boundary

The official upstream repository is
`https://github.com/accel-sim/accel-sim-framework.git`. The first qualified
development snapshot is
`3016c658f810bdae9a14bf4534ee99e9945eedae`. The v1.3.0 release pin is
`c5296df152c99a28dd64e5d9560bd58a8fd2e774`. Both are authored source
identities, not output artifacts. No submodule or downloaded dependency lands
with this freeze.

The associated upstream A100 statistics archive is pinned at
`ee21104be44ad55dfde789111d3b94372be8435f` and its GPGPU-Sim dependency at
`6c3cf4ff32110908386d605a7034fc67666a92de`. Upstream's A100 hardware
statistics and traces resolve through site-local storage and are not provided
by its fetch helper. Wave 1B therefore freezes archive-golden conformance as
the first reproducible gate. It cannot claim an exact public rerun until those
inputs are acquired and hash locked. A fresh SimLLM A100 capture is a project
reproduction rather than an upstream reproduction.

| Backend | NVIDIA A100 SM80 | NVIDIA H100 SM90 | AMD ROCm target |
| --- | --- | --- | --- |
| CUDA silicon collector | supported | supported | reject |
| ROCm silicon collector | reject | reject | supported |
| Accel-Sim | conditional supported cells only | reject | reject |
| compact model compiler | supported | supported | supported |

`amd-rocm-target` is a parameterized campaign slot. Before its first
observation, one campaign binds it to one exact immutable concrete target and
operating envelope. Different concrete AMD targets cannot share an authored
denominator, evidence split or device model.

Accel-Sim runs only after real A100 anchors exist. A simulated point must lie
inside both the simulator capability envelope and the train-defined,
silicon-validated support region. It must be bracketed by real anchors on its
declared interpolation axis and may fill only an explicitly missing point. It
never supplies communication, H100 or AMD evidence.

The discriminating `a100-sm80-exact-only-gap` expectation places its query
inside the train-defined target and qualified SM80 envelopes and strictly
between real silicon anchors, while no exact silicon row exists. The affected
implementation region is categorical and exact-only, so there is no validated
silicon-fit entry and the fit source is inapplicable. Closed precedence is
exact silicon, validated silicon fit, then qualified Accel-Sim; qualified
Accel-Sim must therefore be selected without extrapolation. Adding a valid
silicon fit for that exact region would outrank the simulator source. A
qualified Accel-Sim fill alone never promotes a candidate. Validated status
also requires qualified sidecar replay and correlation, untouched physical
test evidence and live TTFT and TPOT evidence.

## What this freeze does not establish

- It implements no interface, collector, canonicalizer, compiler or runtime.
- It captures no graph and identifies no production implementation.
- It qualifies no environment, simulator configuration or device profile.
- It measures no compute, memory, communication, launch or queue term.
- It creates no content-addressed object, model or compatibility projection.
- It makes no candidate or validated device available to users.
- It closes none of COMP-1, COMP-4, COMP-5, COMP-6, COMP-10, COMP-13, COMP-17,
  COMP-22, COMP-23, COMP-24, COMP-25, COMP-35, COMP-41, COMP-43, COMP-45,
  COMP-47, COMP-48, COMP-49, COMP-50, COMP-51, COMP-52, CORE-8, CORE-11,
  CORE-12, CORE-13, CORE-26, CORE-27, CORE-45, CORE-50, VLLM-12, SGL-10 or
  SGL-24.
