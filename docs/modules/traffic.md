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
  default and expose no second fidelity selector; CORE-36 owns that future
  selection surface.
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
  analytic per-source egress serializer. The declared first-cut rate is
  450,000,000,000 bytes/s and source service is
  `ceil(local_egress_bytes * 1e9 / rate)` whole nanoseconds. Only cross-node
  segments reach `render_fabric_phase_goal` and htsim. `HtsimStepSink`
  executes graph-ordered causal artifacts. Within a placement-split collective
  it uses `max(local_service, fabric_service)` for each directed phase and sums
  those phase services. An all-intra-node step invokes no fabric backend. The
  analytic value is uncalibrated; TRAF-11 owns its replacement with
  same-generation evidence.
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
acceptance oracle. CORE-36 owns the future unified fidelity selector and
provenance record; this option is only the present traffic/backend seam
switch.
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
qualification is void, as recorded below. The runtime's
physical collective expansion and GPU-side contention gaps remain explicit
under TRAF-14, CORE-26, CORE-27 and COMP-22.
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

TRAF-13 remains open after the source-backed vLLM qualification on 2026-08-12.
The eight-rank Granite replay emitted observations for all 32 nonempty steps
and reached traffic binding, the coarse runtime, completion events,
`StepResult`, TTFT and TPOT. The run is void with findings because the fatal
`ttft_exact_single_batch` guard was violated. That guard tested the same
serial-versus-observed arm-equivalence premise needed to attribute the two
in-band TPOT reductions to DBO, so the failure is not orthogonal to the raw
behavioral findings. Retained DBO-off steps 24 through 31 bound the measured
non-DBO residual at 1.231 percent of the mean DBO reduction on both placements,
but do not restore a score.

The residual is not caused by participant-local dependency scope, which both
lowerers use. The structural differences are the open TRAF-9 whole-layer MoE
ordering approximation and the observed arm's terminal logits plus
`requests-visible` fan-in. The vLLM wrapper shows event waits and no
wrapper-level global barrier, but `deep_ep` itself was not installed; lower
level rank-local completion is inferred, not directly source-backed.

The retained 440,115,200 directed bytes are a pre-TRAF-25 conservation
identity over the source-multiplied table and are not portable. The duration
model keys on maximum per-source egress, not total bytes. Under TRAF-25,
dispatch egress from the owning rank stays fixed while combine collapses, so
the communication term changes by roughly a factor of two rather than the
eightfold total-byte change. `_validate_microbatch_partition` conserves the
inflated planner table against itself and cannot detect the defect; it is not a
byte-correctness guard. Both reduction bands are consequently non-portable
across TRAF-25. See
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

TRAF-21 is complete. The explicit captured-message sequence level copies the
strict v2 framework-returned top-k order, retains routing ordinals and exact
per-request and per-pair projections, and supports per-token and whole-layer
expert grouping. Its 12-cell native decision fixture preserved all exact
bytes and changed packet completion in the registered positive direction.
The frozen byte-only packet bands missed because the backend reserves full
packet-calendar envelopes, while the frozen fluid comparison exposed an
80,201 ps capture-order effect at 400 Gbit/s. The inverse-rate family passed
2/2; the complete genuine-risk result is 1/3 family classes and 2/10
instances. The accepted 334,432-byte Granite aggregate GOAL retained its
SHA-256. Aggregate and expert-group Granite rendering remained practical, but
the 101,318-message per-token render exceeded 60 seconds. See
[the dispatch sequence results](../../examples/dispatch_sequence_v1/RESULTS.md).

## Open tasks

### Precision

- TRAF-11 (Precision; P1; L): calibrate the current flat 450 GB/s,
  zero-propagation, per-source NVLink egress surrogate against
  same-generation point-to-point and collective captures. Sweep payload and
  participant count on the reference eight-GPU node, hold out at least one
  payload per participant width, and replace the constant with the smallest
  identifiable bandwidth, latency and concurrency form whose held-out phase
  completion error is at most 10 percent or 1 microsecond, whichever is
  larger. Report the before/after TTFT and TPOT effect and retain the exact
  all-remote identity path.
- TRAF-14 (Precision; P1; M): move ring-round and pairwise-extent expansion
  from the coarse runtime's current semantic-work surrogate into one immutable
  traffic-owned collective plan carried through `ExecutionGraph`. The runtime
  may schedule those extents but may not choose or reconstruct their
  algorithm, chunk sizes, rank order or tags. Compare the plan against the
  existing GOAL pattern expansion over payload, world-size and routed sparse-pair
  sweeps with exact byte, round, dependency and tag conservation. The absent
  explicit plan must preserve the accepted v1 wire bytes and serial timing
  exactly.
- TRAF-16 (Precision; P1; L): preserve participant-local per-rank frontiers
  across graph-artifact and placement-subphase process boundaries. Current
  process quiescence strengthens 284 participant-local edges to artifact-wide
  order. Acceptance must compare raw per-rank starts and completions with the
  graph scope, move live JCT by the registered direction and magnitude, and
  retain the current supported artifact bytes and timing as the explicit off
  path.
- TRAF-22 (Precision; P1; M): make `captured-message-sequence` practical for
  the measured 54-token, 24-layer Granite step without weakening message or
  routing identity. The current 101,318-message plan finishes in 1.86 s, but
  GOAL rendering exceeds 60 s because per-message label validation scans the
  growing rank program. Replace that quadratic validation with a checked bulk
  or indexed path, then pre-register and execute the omitted per-token packet
  and fluid cells. Acceptance requires render plus compile within 30 s, peak
  traced memory within 1 GiB, GOAL text within 64 MiB, exact pair and request
  conservation, and byte-identical aggregate and expert-group off paths.
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

- TRAF-13 (Completeness; P0; L): qualify at least one real framework schedule
  producer through `ObservedStepLowerer`. The 2026-08-12 VLLM-22 run reached
  the full metric chain but is void because `ttft_exact_single_batch` violated
  the frozen arm-equivalence premise. A future expectations-only qualification
  must distinguish DBO from the TRAF-9 and terminal-frontier differences,
  preserve every captured order and dependency fact through traffic binding,
  `DeviceRuntime`, `CompletionEvent`, `StepResult`, TTFT and TPOT, and show a
  registered live-metric relation. Disabling the producer must select the
  serial lowerer and preserve every accepted serial graph, GOAL byte,
  timestamp and completion order exactly. Routed-byte acceptance depends on
  TRAF-25 and VLLM-24.
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
