# simllm.traffic

Semantic collectives to physical flows. Consumes three inputs: the
collective trace, the placement manifest and the fabric manifest; produces
the flow-level work the GOAL emitter renders.

## Interface

- Collective trace (`simllm-collective-trace-v1`, JSONL): one record per
  communication op, i.e. `{step, layer, op, group_type, group_global_ranks,
  send_counts, element_bytes, hidden_size, placement_epoch,
  release_time_ns}` with `op` in ALL_REDUCE / ALL_GATHER / REDUCE_SCATTER /
  ALL_TO_ALLV / SEND_RECV, plus KV-transfer records.
- For MoE, `expert_owners[layer][global_expert_id]` (from the placement
  manifest, per placement epoch) turns routed tokens into all-to-allv
  destinations.
- Semantic collectives are expanded into the algorithm actually used (ring,
  tree, pairwise all-to-allv, or a custom collective-network schedule) as
  chunked send/recv chains.
- `step_comm` (M4): one `StepRecord` plus a per-rank `ModelDims` plus a
  tensor-parallel group of GOAL ranks maps to the step's TP collective
  work. `step_tp_allreduces` lists two ring allreduces per transformer
  layer (attention output, MLP output), each of payload
  `total_new_tokens * hidden_size * dtype_bytes`; a TP world of size 1 or
  a zero-token drain record produces no ops.
- `step_moe_alltoalls`: the same record plus MoE
  `ModelDims` plus an expert-parallel group of W GOAL ranks maps to the
  step's MoE traffic: per MoE layer, a dispatch pairwise all-to-allv then
  a combine pairwise all-to-allv. Without an optional `RoutedMoeSupply`,
  the first EP rank is the one modeled engine and uses the uniform payload
  `total_new_tokens * top_k * hidden_size * dtype_bytes // W` to every
  other rank. The router spreads this one engine's (token, expert)
  assignments evenly over the EP group and its own 1/W share stays local,
  off the fabric. `RoutedMoeSupply` instead declares one `engine_rank`, then
  joins either the strict `simllm-routed-experts-v1` projection or the packed
  routing arena to immutable placement snapshots and a step-to-epoch map. It
  slices each scheduled prefill or decode phase, emits one hidden vector from
  that engine per token and remote destination rank, pre-reduces combine to
  the transposed pair table and records the selected epoch on the graph
  operation. Peer EP ranks own experts but carry zero scheduled tokens in
  this isolated projection. Full peer-engine population requires explicit,
  independently routed peer workloads under TRAF-26. Dense dims, an EP world
  below 2 or a zero-token record produce no ops.
- `step_moe_message_sequences` is the explicit
  `captured-message-sequence` precision level. It retains request identity and
  every contributing `(token_index, top_k_index)` position while emitting
  either one message per token and unique remote destination or one
  whole-layer request/source/destination group ordered by its first
  contribution. `render_sequenced_step_goal` projects that tuple order into
  source-local issue dependencies. The aggregate APIs above remain the
  default and expose no second fidelity selector; `simllm.core.PrecisionConfig`
  is the one selection surface.
- `plan_step_locality` expands TP ring rounds and MoE dispatch/combine tables
  into ordered directed phases over semantic global ranks, then joins an
  optional placement manifest through `RankMapper.is_intra_node` before any
  GOAL-rank projection. Missing placement is the exact accepted all-remote
  compatibility classification. An explicit all-remote placement under
  `gpu-rank` mapping takes the same identity path. This direct `StepRecord`
  planner remains independently executable through `render_step_goal`. It is
  the ordering authority for a standalone direct-GOAL run and an observational
  cross-check, never the active graph-projected sink's ordering authority.
- `plan_execution_graph_locality` expands only graph-owned collective
  operations, tags and request partitions, then classifies their directed
  segments as local or fabric traffic. Its exact verifier recomputes the plan
  from the graph and rejects a lost, duplicated or mutated operation, edge,
  rank, payload, tag or request partition.
- In the placement-enabled path, each phase's intra-node segments use a flat
  analytic per-endpoint serializer. `ClassifiedCommunicationPhase` carries an
  explicit sorted `nvlink_endpoint_bytes` ledger of
  `(rank, egress_bytes, ingress_bytes)` built from the local segments
  themselves, with no transpose or symmetry assumption, and rebuilds it in
  `__post_init__` so a ledger, byte conservation or service that disagrees with
  its own segments cannot be constructed. The declared first-cut rate is
  450,000,000,000 bytes/s. The modeled port is full duplex, matching NVLink and
  NVSwitch ports, so one endpoint's load is
  `max(egress_bytes, ingress_bytes)`, its service is
  `ceil(endpoint_load * 1e9 / rate)` whole nanoseconds, and the serial phase
  costs the largest of them. The rejected alternative, a shared half-duplex port
  charged `egress_bytes + ingress_bytes`, would double a symmetric exchange the
  hardware serves on independent lanes; see
  [the endpoint service results](../../examples/endpoint_service_v1/RESULTS.md).
  Only cross-node segments reach `render_fabric_phase_goal` and htsim.
  `HtsimStepSink`
  executes graph-ordered causal artifacts. Within a placement-split collective
  it uses `max(local_service, fabric_service)` for each directed phase and sums
  those phase services. An all-intra-node step invokes no fabric backend. The
  450,000,000,000 bytes/s value remains the exact `None` and `legacy`
  compatibility level. Selecting `b200-nccl-2.27-local-v1` replaces its
  endpoint rate and adds one width-indexed semantic-collective base latency
  outside the phase-local maximum. TRAF-31 owns the missing same-generation
  point-to-point capture.
- `lower_step_observations` joins that traffic plan to framework-neutral
  `ExecutionObservations`. The adapter tuple order, logical queues, dependency
  edges, gates, priorities, correlations and completion frontier pass through
  unchanged. Each observed collective identifies one layer and semantic site;
  traffic validates its group and payload, supplies the algorithm, routed pair
  table and placement epoch, and requires every planned site exactly once.
  `ObservedStepLowerer` exposes this path through the standard
  `ExecutionLowerer` contract. Omitting observations delegates directly to
  `SerialStepLowerer` as the exact compatibility off path.
- `render_step_goal` renders the serial per-rank chain (per layer: `calc`,
  the two TP allreduces when the TP world produces them, then for MoE dims
  with `ep_ranks` given the dispatch and combine all-to-allvs) through the
  existing `ring_allreduce` and `pairwise_all_to_allv` patterns; tags are
  disjoint per collective. The calc input may be one compatibility scalar or
  an ordered value per layer, and `num_goal_ranks` idle-fills a larger GOAL
  layout. A scalar step without MoE work renders byte-identically to the
  pre-M5 emitter (golden test). The direct renderer deliberately constructs
  its own ATLAHS schedule so it can remain an independent debug cross-check;
  it is not used to repair or override graph-projected ordering.
- `plan_execution_graph_collectives` attaches one immutable `CollectivePlan`
  per collective operation and returns the planned `ExecutionGraph`. The plan
  is the sole explicit-plan authority for algorithm, rank order, rounds, tags,
  channels, chunk sizes, endpoint actions, their rank-local predecessors and
  the directed extents with their request partitions. A canonical SHA-256 over
  that content is its integrity identity, so a partially changed handoff is
  rejected before scheduling, and the semantic fields are compared against the
  `CollectiveWork` they join to, so a byte-conserving rank-order or tag change
  cannot pass. Coverage is all or nothing: a graph that plans only some of its
  collectives is invalid. Tags come from the accepted `collective_goal_tags`
  allocator rather than a second implementation, and that allocator reads them
  back out of the plan once it is attached. `collective_plan_by_operation`
  returns the validated inventory and `render_collective_plan` renders the
  declared actions, choosing only how a finished round is exposed to a
  successor. A graph with no plan keeps the accepted compatibility path,
  including the coarse runtime's own expansion, byte for byte.
- `render_serial_execution_graph_goal` is the CORE-2 graph-only diagnostic
  replay. It accepts validated per-rank compute, ring allreduce and pairwise
  all-to-allv operations, preserves `participant_local_depends_on` edges and
  representable single-rank FIFO predecessors, and never reduces a distributed
  whole-operation barrier through rank-local ancestry. It fails loudly on
  explicit or implicit cross-rank barriers and work or timing semantics the
  serial GOAL subset cannot represent. Sparse pair tables preserve exact idle
  rank frontiers, including a zero-byte semantic collective when every routed
  destination is local. It consumes no `StepRecord` after lowering.
- `project_execution_graph_goal` is the checked active projection. It assigns
  graph operations to canonical causal-level artifacts, renders supported
  rank-local relations with structured provenance, records distributed
  whole-operation edges as ordered artifact boundaries and inventories other
  cross-artifact edges. `verify_execution_goal_projection` independently
  checks the canonical partition and exact operation, edge, rank, message,
  payload, tag, request-partition and completion-boundary inventories.

Deliberately out of scope: exact TP weight-storage intervals (packed QKV,
gate/up packing, quantization padding); group memberships plus activation
shapes suffice for communication simulation.

## Status

Pattern expansion landed with M1 (`simllm.traffic.patterns`): scatter,
gather, ring allreduce (reduce-scatter + allgather, 2(W-1) chained rounds),
pairwise all-to-allv, and binomial-tree broadcast, all rendered as GOAL
send/recv chains with explicit dependencies (TRAF-1 closed). Backend
validation coverage differs by pattern: scatter/gather are validated end to
end against the packet-level backends with picosecond-exact closed forms
(examples/m1/RESULTS.md), the M4 studies did the same for ring allreduce on
both null-network profiles (examples/m4/RESULTS.md checks A and C), and the
M5 studies closed the pairwise all-to-allv part of TRAF-4 on the fluid
profile (examples/m5/RESULTS.md check A: symmetric all-to-allv exact to
0 ps across size x width, using the whole-bps floor and whole-ps ceil
quantization of the fluid manifold read from the backend source); the
binomial tree still has structural unit tests only. The `step_comm` TP
mapping landed with M4 and is validated end to end by the examples/m4 grid
and the live tp=8 closed-loop run; the MoE mapping (uniform-routing first
half of TRAF-2) landed with M5 and is validated by the examples/m5 step
grid (fluid MoE step makespans exact to 0 ps across EP x step-shape). The
JSONL collective-trace consumer is not yet implemented (TRAF-5).

The captured-routing half of TRAF-2 is implemented behind the explicit
`RoutedMoeSupply` seam. Its absent path is the single-engine uniform
approximation, while its enabled path is live through `SerialStepLowerer`,
`render_step_goal` and `HtsimStepSink`. The original combined Granite study
passed exact graph and GOAL pair tables at two placement epochs and four
fluid-JCT cells with 0 ps residual, closing TRAF-2. TRAF-25 later corrected
both paths' source population and reran the traffic section with 48 rather
than 96 positive flows while retaining all four JCT values; see
[the routing supply results](../../examples/routed_supply_v1/RESULTS.md).

CORE-2 additionally proved that serial GOAL rendered only from a
JSON-round-tripped `ExecutionGraph` is byte- and timing-equivalent to the
legacy step path over TP width and link-rate sweeps, including a MoE sentinel
([results](../../examples/core2_lowering/RESULTS.md)).

The BACK-5/BACK-7 step-sink study additionally validates unequal ordered calc
values over layer-count and TP-width sweeps, plus explicit 64-rank padding on
the fluid and physical-topology paths. Every valid comparison has 0 ps timing
residual; see
[examples/step_sink_precision/RESULTS.md](../../examples/step_sink_precision/RESULTS.md).

The TRAF-10 first-cut locality split is live through `HtsimStepSink` and
`StepResult`; see
[the NVLink locality results](../../examples/nvlink_locality_v1/RESULTS.md).
Across a fixed captured Granite step, raw fabric bytes increased and local
bytes decreased exactly from one node to two nodes to all remote at both
payloads. Single-node TP widths 1 through 8 emitted no fabric bytes, while the
explicit all-remote cells matched omitted placement. TRAF-12 made
`ExecutionGraph` the semantic authority for the active sink and aligned the
coarse runtime, locality and backend projection to one effective edge
inventory. The corrected dependency rerun retained the graph census exactly:
144 operations, 423 effective edges, 72 causal artifacts, 47 required
distributed FIFO boundaries and 376 other serialized edges. The corrected
direct diagnostic is 20,392 bytes and 144 flows at both payloads. At 1,024 and
2,048 vector bytes, it completed in 150,838,767 ps and 205,653,487 ps, while
the graph-authoritative path completed in 155,702,768 ps and 215,381,488 ps.
The graph-minus-direct changes are therefore 4,864,001 ps and 9,728,001 ps;
see
[the dependency authority results](../../examples/dependency_authority_v1/RESULTS.md).
The authority conclusion is unchanged. In the serial sink,
`dependency_cross_check="atlahs-goal"` keeps the `ExecutionGraph` projection
authoritative and does not change the `StepResult`. Its structural comparator
inspects all 423 canonical effective edges and reports 94 differences: 47
whole-operation logical-queue FIFO differences and 47 participant-local
syntactic-frontier differences. The raw timing subset remains the 47
whole-operation boundaries, with 32/47 unequal, early gaps. These are
diagnostic findings rather than values folded into the live result. The
default-off path preserves the accepted artifacts and results exactly. The
current cross-check is restricted to the all-remote compatibility
classification; a placement with local NVLink work is rejected, and TRAF-16
owns the participant-local frontier precision needed before that comparison
is meaningful. The corrected single-node `AAAA` values remain pending
CORE-41's ingress-aware analytic service correction and are not a precision
acceptance oracle. `simllm.core.PrecisionConfig` and `RunProvenance` own the
unified fidelity selection and record; this option is only a traffic/backend
diagnostic switch and names no seam level.
Historical
`examples/breakdown` fabric-TP columns remain byte-unchanged and are the
all-remote, cross-node what-if under this model.
TRAF-7 is complete for observation-driven step lowering and the coarse live
metric chain. The frozen two-layer study crossed `C/D` from 1/2 to 2 and
realized independent, two-stage pipeline and serial graphs at their exact
closed forms. Pipeline TTFT and TPOT were exactly 5/6 of serial in all four
registered metric cells. A separate shared-versus-split NCCL-channel fixture
changed JCT by exactly 999 ps, and the absent-observation graph and GOAL bytes
retained their accepted hashes. All 16 scored relations, 22 exact-oracle rows
and 12 fatal unscored guards passed; see
[the overlap results](../../examples/compute_comm_overlap_v1/RESULTS.md).
The vLLM adapter now emits one implemented source-backed schedule. Its current
qualification is void, as recorded below. The runtime's GPU-side contention
gaps remain explicit under CORE-26, CORE-27 and COMP-22.

The 2026-08-13 TRAF-14 qualification closed the duplicated collective
expansion. Ring rounds and pairwise extents now live in one immutable
traffic-owned `CollectivePlan` carried through `ExecutionGraph`, and the
coarse runtime schedules those declared extents instead of re-deriving them.
Compared with the shipped `ring_allreduce` and `pairwise_all_to_allv`
expansions over worlds 2 and 4, payloads 3, 4 and 4,096 bytes, the three
routed sparse cases and both GOAL frontier modes, all 18 comparisons are
identical in messages, dependencies, tags, chunks, per-rank frontiers and
rendered text. The two registered byte-conserving perturbations, a changed
plan tag and a changed semantic rank order, are both rejected by validation
and by runtime preflight with zero work requests submitted, while the same
rank-order change is absorbed silently by the absent-plan surrogate at an
unchanged 120 ps and unchanged 24 bytes. The explicit plan carries a 3-byte
four-rank ring that the compatibility runtime rejects outright and reaches
TTFT and TPOT of exactly 120 ps at 400 Gbit/s and 240 ps at 200 Gbit/s across
one prefill and two decode steps. All six genuine-risk instances in two
families passed and no fatal guard was violated; the absent-plan arm keeps its
559-byte v1 wire form and its exact runtime timing, including under a nonzero
collective channel service. See
[the collective plan results](../../examples/collective_plan_v1/RESULTS.md).
CORE-48 owns the missing cross-node destination-ingress serializer that keeps a
converging combine structural rather than physical evidence.

The 2026-08-13 TRAF-28 qualification then made that plan the lowering default
and closed the task. `SerialStepLowererConfig.attach_collective_plan` and the
matching `lower_step_observations` keyword default to True, so both shipped
lowerers hand the runtime a fully planned graph and the runtime's own semantic
reconstruction is unreachable on the production path. Setting the flag False is
the explicit bypass: the graph then carries no plan, its v1 wire form omits the
field so the accepted 559-byte anchor is unchanged, and the runtime falls back
to its own expansion. The first run is void because its physical bound charged
the swept fabric rate to a live arm placed entirely on one node, where the
coarse runtime serves same-node sends over a fixed NVLink rate; the transport
refreeze pinned one rank per node and made the bound charge each extent to the
link the model selects. The corrected run passed 4/4 scored families over 20
instances with every fatal guard held. Default and bypass produced identical
completion times, quiescence, completion events and every WQE timestamp across
two tensor-parallel widths, two rates and both lowering paths. A changed plan
tag and a changed semantic rank order are both rejected with zero work requests
submitted on the default path, while the bypass absorbs the same rank-order
change silently at unchanged 196,608 bytes and unchanged 4,730,040 ps. With the
absent-plan branch made fatal, every default cell still executes and every
bypass cell raises. On the replayed 54-token Granite step the live TTFT and
TPOT are 709,803,840 ps and 132,794,880 ps at 400 Gbit/s and exactly the
inverse-rate pair at 200 Gbit/s, with both network terms scaling by 2.0000. The
reconstruction is now dead code on the production path and live code on the
compatibility path: the explicit bypass, deserialized v1 graphs without a plan
field, and directly constructed `ExecutionGraph` values still reach it, so
deleting it requires first retiring the absent-plan wire form. See
[the plan default results](../../examples/collective_plan_default_v1/RESULTS.md).
The 2026-08-12 TRAF-13 qualification added `DeviceRuntimeStepSink`, which binds
the adapter's sole `VirtualClock` and carries optional observations through
`ObservedStepLowerer`, `CoarseDeviceRuntime`, `CompletionEvent`,
`RuntimeReport`, `CompletionReducer` and request-attributed `StepResult`
metrics. Its first component qualification passed the accepted exact serial
graph and GOAL identity checks but emitted no framework schedule, so its
historical behavioral result remains `0/0, blocked before behavioral
execution`. See
[the observed-schedule qualification results](../../examples/observed_schedule_v1/RESULTS.md).
A separate pre-VLLM-22 live diagnostic confirmed that earlier boundary across
one prefill and one decode step. It is historical component evidence and does
not change that blocked denominator.

TRAF-13 is complete. The source-backed vLLM qualification on 2026-08-12 was
void because the fatal `ttft_exact_single_batch` guard tested a
serial-versus-observed arm-equivalence premise that is false, and that premise
was also what attributed its decode reductions to DBO. The 2026-08-13
requalification does not assume the premise. It adds a third arm: the observed
operation tuple with only cross-microbatch serialization edges added, holding
every operation identity, queue, work object, correlation and completion
endpoint byte-identical. Overlap is `control - observed`, structure is
`serial - control`, and structure splits at the last collective completion into
a layer-ordering term and a terminal-frontier term.

The two structural causes are now measured rather than named. On a decode step
the TRAF-9 whole-layer ordering moves the MoE phase by about +17.97
microseconds and the observed arm's terminal logits plus `requests-visible`
frontier moves the tail by about -17.97 microseconds, because the producer
relocates the LM-head compute while conserving the per-rank total. They cancel
to -903.913 ps single-node and -7,123.478 ps cross-node, and that remainder is
the extra summed peak per-source egress created by splitting each collective
into two microbatch collectives, predicted at 791.5 and 7,123.5 ps from the
frozen routed table. Overlap is 1,450,472.652 and 13,051,993.043 ps, which is
99.61 and 99.59 percent of the arithmetic communication ceiling registered
before the run. The cross-node to single-node overlap ratio is 8.998 against
the exact 9.0 link-rate ratio, so B2 discriminates the overlap mechanism by its
response to bandwidth. The terminal term's ratio is exactly 1.000, but B3 is
reclassified post-specified as a fatal-unscored structural invariant carrying
no evidential weight for this frozen producer: its interval starts after the
last collective completion, and the only later operations are logits compute
and zero-duration visibility compute. The producer emits no control work, and
all placement-dependent link-rate service defines the collective frontier
instead. All 19 registered fatal guards and the reclassified invariant held,
and 3 of 5 genuine-risk instances passed; the two failures are one refuted
expectation that the serial arm would stay strictly slower on the single-batch
prefill, where the two arms are in fact exactly equal. See
[the observed-overlap results](../../examples/vllm_observed_overlap_v1/RESULTS.md).

The vLLM wrapper shows event waits and no wrapper-level global barrier, but
`deep_ep` itself was not installed; lower level rank-local completion is
inferred, not directly source-backed.

The retained 440,115,200 directed bytes are a pre-TRAF-25 conservation
identity over the source-multiplied table and are not portable. The duration
model keys on maximum per-endpoint load, not total bytes. It keyed on maximum
per-source egress until CORE-41, which was the same quantity only while the
local traffic matrix stayed symmetric. Under TRAF-25,
dispatch egress from the owning rank stays fixed while combine collapses, so
the communication term changes by roughly a factor of two rather than the
eightfold total-byte change. `_validate_microbatch_partition` conserves the
inflated planner table against itself and cannot detect the defect; it is not a
byte-correctness guard. VLLM-24 later added the guard that can:
`simllm.traffic.routed_conservation` checks the planned routed table against
independently derived token ownership. Both reduction bands are consequently
non-portable across TRAF-25. See
[the source-backed results](../../examples/vllm_observed_schedule_v1/RESULTS.md).

TRAF-25 corrected the source-multiplicity defect in both captured and uniform
MoE rendering. One `StepRecord` now has one engine source, while peer EP ranks
remain expert owners with zero scheduled tokens in the isolated projection.
The Granite EP-width sweep passed 3/3 genuine-risk families and 9/9 instances.
At EP width 8, total traffic changed from 207,499,264 to 25,563,136 bytes while
peak-rank egress changed from 27,060,224 to 12,781,568 bytes. The live 400
Gbit/s fluid and packet results were 706,622,768 and 724,527,360 ps, both above
the 255,631,360 ps corrected serialization floor. Five affected traffic
consumers were rerun, the framework oracle was audited as unaffected, and all
old source-multiplied numeric surfaces are listed in
[the token ownership results](../../examples/token_ownership_v1/RESULTS.md).

VLLM-24 added the independent routed-byte conservation guard in
`simllm.traffic.routed_conservation`, and both `lower_step_observations` and
`render_step_goal` now run it on the full-step routed plan. Its ownership side
comes from the record's per-request scheduled token counts, the declared
`RoutedMoeSupply.engine_rank` and the model geometry, so it never consumes the
per-token routing walk it inspects. Five rules apply to every routed
representation, including the uniform destination approximation:
`source-attribution`, `destination-legality`, `owner-egress`,
`transpose-symmetry` and the `step-hop-bound`
`bytes <= total_new_tokens * top_k * num_layers * 2 * vector_bytes`. Four more,
`vector-granularity`, `request-identity`, `per-request-hop-bound` and
`per-layer-hop-bound`, need deduplicated captured routing and are deliberately
not applied to the uniform approximation, which exceeds
`min(top_k, W - 1)` per token by construction because it never merges experts
that share a destination. A violated rule is fatal and unscored.

The qualifying study replayed the captured Granite routing at EP worlds 2 and 8
against a source-replicated arm reproducing the pre-TRAF-25 shape. At world 8
the replicated arm emitted 42,656 hops against an 8,448-hop bound and was
caught; at world 2 it emitted 2,112 against the same bound and was not, because
a two-rank world admits at most one remote owner per token-layer, so
`2 * hops_A(2) <= 4224` for any routing at all. The captured-only per-layer
rule caught it at both worlds, but it is unavailable to the uniform
approximation, which is why the wide-world step bound is the rule that
generalizes. See
[the conservation results](../../examples/routed_byte_conservation_v1/RESULTS.md).

TRAF-21's explicit captured-message sequence interface remains complete.
TRAF-25 now makes the aggregate and sequenced projections consume one ordered
contribution authority owned by `RoutedMoeSupply.engine_rank`. The
EP-width-eight regression requires exact ordered-pair equality and an
independent hop ceiling for both grouping modes. The old source loop would add
non-engine source pairs and emit 101,318 Granite hops against the 20,736
ceiling, so it fails both guards.

The corrected ownership refreeze reached every native cell, but the run is
void because four aggregate or expert-group fluid cells violated a fatal
floor that incorrectly treated full-duplex dispatch and combine loads as
serial. No corrected behavioral fraction is reported. All exact ownership,
pair, request, hop, input-identity and quiescence guards passed. At Granite
scale the three modes emitted 336, 1,008 and 12,482 messages with 25,563,136
bytes each. The per-token mode rendered and compiled in 17.020 seconds, used
17.166 MiB peak traced memory, produced 2.140 MiB of GOAL text and completed
its packet and fluid backend runs in 32.890 and 24.592 seconds. That run keeps
its void record and its raw observations. See
[the dispatch sequence results](../../examples/dispatch_sequence_v1/RESULTS.md).

The 2026-08-13 TRAF-22 requalification is a fresh qualification on the
corrected floor and closes the task. The modeled port is full duplex, so the
payload floor of a single-home-rank step is one endpoint charged in its busier
direction, `max(egress, ingress) * 8 / rate`: 655,360 ps at 200 Gbit/s and
327,680 ps at 400 Gbit/s on the retained synthetic fixture, exactly half the
summed floors the void freeze used. Every bound is now computed from the
actually rendered messages, the packet ceiling from the backend's
full-envelope calendar, and the freeze added a held-out routing shape and
1,024-byte payload that had never been rendered or executed. All six fatal
guard groups held and all 34 scored instances in 5 families passed across 24
synthetic and 12 Granite cells. The registered but previously unexecuted
200/400 Gbit/s Granite scaling check passed on all six cells, with every
network-term ratio within 0.06 percent of two after subtracting the
rate-independent `24 * 4,139 ns` compute. Granite at 200 Gbit/s completes in
908,419,960 ps packet and 879,134,570 ps fluid for the aggregate default,
1,131,907,000 and 1,027,865,000 for expert groups, and 2,092,043,000 and
1,121,909,000 per token. The aggregate default reproduced its GOAL identity and
both retained 400 Gbit/s completions exactly, which is also the evidence that
the pairwise frontier documentation alignment and the collective-plan lowering
default moved no rendered byte. The per-token 200 Gbit/s packet backend ran
58.02 seconds against the 60-second limit, which is the measured practical
boundary of this scale point. The documented `pairwise_all_to_allv` source-only
frontier now matches its implementation, a rank's first send; moving it to the
last send would change accepted timing and needs its own freeze. See
[the requalification results](../../examples/dispatch_sequence_v2/RESULTS.md).

The [collective latency floor study](../../examples/collective_latency_floor_v1/RESULTS.md)
closes TRAF-11, with its sole undemonstrated source clause moved to TRAF-31.
Attempt one is retained as void because two fatal sensitivity oracles encoded
the fluid backend's whole-picosecond result one picosecond low. Attempt two
changed only those harness oracles and preserved every raw measurement, but
closure review found that C3 was mathematically entailed and that the fatal
harness lacked a mixed-placement cell. Attempt three changed no modeled
behavior. The final classification replay passed both non-entailed genuine-risk
families, C1 and C2; C3 and C4 passed as exact-unscored relations; and every
fatal guard held, including a two-node collective with simultaneous local and
fabric service. TTFT and TPOT reach is established by
`step_service_conservation`.
The 4 KiB held-out errors at participant widths 2, 4 and 8 were 0.261, 0.347
and 0.080 microseconds. The selected profile adds 1.446145392 ms across the
reference step's 48 collectives, moving the mission network budget from
0.106 ms to 1.552145392 ms. The 1.651145392 ms whole-step figure is arithmetic
on main's published 0.205 ms literal, not a measured composed run. It applies a
DGX B200 intra-node NVLink ALL-REDUCE intercept unchanged to cross-node
pairwise ALL-TO-ALLV. The 1.446145392 ms addition is 0.4 percent above the
mission budget's nominal 1.44 ms endpoint, so the direction of residual error
remains ambiguous. The same-wave fixed host term was not composed here, and
additivity versus overlap remains unresolved. The legacy and all-remote
identity paths remained exact. The public evidence supplies a B200 collective
capture and a vendor capacity ceiling, but not the same-generation
point-to-point payload capture required by the original clause.

## Open tasks

### Precision

- TRAF-31 (Precision; P1; L): obtain the same-generation point-to-point
  payload capture absent from the `b200-nccl-2.27-local-v1` calibration. The
  selectable profile currently identifies its 70,027,079,100 bytes/s endpoint
  serializer from a public eight-B200 all-reduce capture and uses the DGX B200
  aggregate NVSwitch specification only as a physical ceiling. Capture pinned
  B200 NVLink point-to-point completion across the profile's 8-byte to
  256-KiB payload envelope and representative peer placements on the
  eight-GPU node, reserving at least one payload as holdout. Validate or refit
  the serializer and require held-out completion error no larger than
  10 percent or 1 microsecond, whichever is larger. Report the current-profile
  before error, rerun the collective holdouts after any refit, and preserve the
  exact `legacy` and all-remote identity paths.
- TRAF-16 (Precision; P1; L): preserve participant-local per-rank frontiers
  across graph-artifact and placement-subphase process boundaries. Current
  process quiescence strengthens 284 participant-local edges to artifact-wide
  order. Acceptance must compare raw per-rank starts and completions with the
  graph scope, move live JCT by the registered direction and magnitude, and
  retain the current supported artifact bytes and timing as the explicit off
  path.
- TRAF-23 (Precision; P1; L): measure per-rank completion frontiers on the
  VLLM-22 path. Capture dispatch and combine return, next-compute eligibility,
  final-logits completion and terminal `requests-visible` fan-in on a B100
  NVLink node and a 400 Gbit/s cross-node placement. Fit only identifiable
  latency and frontier terms, hold out at least one payload and step shape, and
  require registered held-out error bounds on those measured frontiers. Any
  one-edge perturbation criterion must be no larger than one measured target
  collective's service time; a whole-step reduction fraction is not an
  admissible minimum. The current source schedule and producer-disabled serial
  identities remain the exact off paths.
- TRAF-19 (Precision; P2; L): add a statistical flow-completion level
  beside the fluid and packet-level network models. Fit a completion-time
  distribution offline from packet-level runs over a declared topology,
  load and collective shape, then draw from it, so a large sweep keeps
  network side effects such as ECMP hash collisions, incast tails and
  link failures as a measured tail instead of deleting them by assuming
  an infinite pipe. The fit must carry its calibration envelope and be
  refused outside it, the draw must be seeded and reproducible, and the
  packet-level path stays the exact reference the fit is validated
  against. Acceptance compares fitted quantiles against held-out
  packet-level runs at registered accuracy, and states plainly that a
  marginal fit does not reproduce correlations it never observed.
- TRAF-20 (Precision; P2; M): add a fluid LogGOPSim fast level for
  schedule-shape studies that do not need per-flow transport behavior.
  The GOAL already compiles to the LogGOPSim toolchain, so this level
  reuses it analytically and bypasses the event-driven RNIC path. Its
  purpose is sweep throughput, so acceptance must state the measured
  wall-clock gain and the measured error against the packet-level
  reference on the same schedules, and it must refuse configurations
  whose questions it cannot answer rather than returning a number.

### Completeness

- TRAF-26 (Completeness; P2; L): extend the isolated one-engine routed-step
  projection to a full DP times EP group population. Each peer engine must
  carry an explicit captured workload or a reproducible independently sampled
  workload, and its routing must be independently observed or sampled.
  Replaying one engine's routing table on every peer is forbidden because it
  manufactures correlated hot-expert incast. Acceptance compares group bytes,
  peak egress, incast fan-in, TTFT and TPOT against a multi-engine capture,
  while selecting the isolated mode preserves every accepted TRAF-25 byte,
  timestamp and completion order exactly.

- TRAF-15 (Completeness; P2; M): project arbitrary legal forward, non-monotone
  and general non-contiguous or fan-in DAGs through the step sink. The current
  projector rejects unsupported order classes before writing an artifact.
  Acceptance must preserve that explicit rejection as the off path, avoid
  inventing order between independent operations and retain every supported
  projection byte and timestamp exactly.

### Uncategorized

- TRAF-3: KV-transfer records for PD-disaggregation and cache-miss
  re-prefill (milestone M6).
- TRAF-4 (ring-allreduce part closed by examples/m4, pairwise-a2av part
  closed by examples/m5): end-to-end closed-form validation of binomial
  broadcast against the fluid backend, extending the M1/M4/M5 study
  pattern.
- TRAF-5: the JSONL collective-trace consumer (parse
  `simllm-collective-trace-v1` records and hand them to pattern expansion).
- TRAF-6: sequence parallelism in the step model. `step_comm` reduces the
  full activation on every rank; SP would replace each allreduce with a
  reduce-scatter plus allgather of 1/W the bytes around the norm/dropout
  regions.
- TRAF-8: pipeline-parallel activation traffic from step records. Records
  carry no PP stage attribution yet, so `step_comm` emits TP and EP
  collectives only; the M1 workload-B GOAL shows the target activation-chain
  shape.
- TRAF-9: MoE layer op ordering. `render_step_goal` renders one calc per
  layer followed by the TP allreduces and then dispatch and combine back
  to back; a real MoE layer splits its compute around the all-to-alls
  (attention and router before dispatch, expert MLP between dispatch and
  combine) and may overlap shared-expert work with the a2avs. The serial
  whole-layer calc keeps the makespan correct only to first order.
