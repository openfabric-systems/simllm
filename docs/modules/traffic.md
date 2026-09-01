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
  work. `step_tp_allreduces` lists, per transformer layer, the ring allreduces
  `layer_tp_allreduce_sites` reports for the model and its expert-parallel
  declaration, each of payload
  `total_new_tokens * hidden_size * dtype_bytes`; a TP world of size 1 or
  a zero-token drain record produces no ops.
- `layer_tp_allreduce_sites` is that site rule. Every layer reduces its MLP
  output exactly once and the rule names which mechanism does it. A dense
  layer reduces at both row-parallel outputs, the attention output projection
  and the MLP down projection. A routed layer whose output arrives through a
  combine all-to-all reduces only after attention, because that combine
  returns finished expert vectors and the token's home rank forms the layer
  output by a local weighted sum, so no partial sum spans the TP group. The
  condition is exactly `renders_expert_combine`, the shared predicate
  `step_moe_alltoalls` takes its own early exits from, so the two inventories
  cannot disagree. That includes the degenerate uniform case: when the
  per-pair share floors to zero bytes no all-to-all is rendered at all, so the
  layer keeps both sites and its output is still reduced once.
- Declaring the expert-parallel group to a renderer therefore asserts an
  all-to-all whose combine returns an already reduced output, which is a
  narrower claim than expert parallelism, and two pinned vLLM 0.26.0
  conditions must both hold for it. First,
  `model_executor/layers/fused_moe/config.py:1052-1055` makes
  `use_all2all_kernels` require expert parallelism AND one of `dp_size > 1`,
  `pcp_size > 1` or sequence parallelism, so a `tp=8, ep=8, dp=1` deployment
  runs naive expert parallelism with no all-to-all at all. Second, the
  selected backend must reduce: `config/parallel.py:186` defaults
  `all2all_backend` to `allgather_reducescatter`, whose prepare-finalize
  returns `output_is_reduced()` False
  (`model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:109` and
  `:242`), while the deepep, mori, nixl and flashinfer families return True.
  When either fails,
  `model_executor/layers/fused_moe/runner/moe_runner.py:436-465` all-reduces
  the fused output over the TP group. The naive shape is rendered by not
  declaring the group: two allreduce sites and no all-to-all, which is what
  the framework executes. The vLLM producer classifies both conditions before
  it binds a group, and refuses outright when an all-to-all path exists whose
  allgather and reduce-scatter traffic this repository renders nothing for,
  rather than pricing it as a pairwise all-to-allv.
  TRAF-40 owns turning the mode into an explicit declaration and rendering
  that refused path. Sequence parallelism is out of scope here and owned by
  TRAF-6: under it the framework skips the TP reduction entirely and the model
  performs its own allgather.
- Shared experts are the documented exception, all-reduced over the TP group
  even on the reducing all-to-all path
  (`model_executor/layers/fused_moe/runner/moe_runner.py:416-433`); both
  frontend readers refuse shared-expert geometries rather than dropping that
  reduction, and VLLM-25 and SGL-18 own supporting them. TRAF-34 owns mixed
  dense and routed layer schedules and TRAF-35 owns `moe_tp` above 1; neither
  is expressible in `ModelDims` and neither is guessed.
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
- `NetworkLevel.LOGGOPSIM_IDEAL` selects the ideal-network step sink. It
  preserves the graph-owned collective plan, GOAL rendering, analytic
  intra-node service and standard `StepResult` metric path while replacing
  only remote packet execution with the LogGOP cost model. The level records
  declared `L`, `o`, `g`, `G`, `O` and `S` values, including the exact `G`
  string derived from bit rate. It rejects composed-native RNIC hardware at
  precision validation because those two remote-network authorities cannot
  own the same step.
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
- `CollectiveFixedCostEnvelope` is that same selection expressed as a named
  bracket rather than one silently chosen constant. An envelope names a
  `lower` and an `upper` profile beside the `off` arm that charges nothing,
  and it refuses a pair that does not isolate the fixed cost: both arms must
  share the endpoint rate, the source payload interval, the propagation
  reference and the supported widths, both must carry provenance, and the
  lower arm must be strictly cheaper at every width. The bracket is over the
  arms a study can select, not over the physical value: an arm's own declared
  band may reach past the arm above it, and the envelope's `claim` string has
  to say what the bracket does and does not assert. Two envelopes ship.
  `intra-node-fixed-cost-v1` runs from `collective-fixed-cost-floor-v1`, which
  adds no surcharge so the claimed fixed cost is exactly the propagation the
  backend already charges, to `b200-nccl-2.27-local-v1`.
  `cross-node-fixed-cost-provisional-v1` runs from `b200-nccl-2.27-local-v1`,
  a floor because a fabric hop cannot be cheaper than the NVLink hop it
  replaces, to `b200-nccl-2.27-cross-node-provisional-v1`, which is the
  pessimistic selectable edge rather than a ceiling: its own band reaches
  77,487,789 ps at width 8, 57 percent above the 49,487,789 ps it charges, and
  no evidence here establishes a ceiling at all. `HtsimStepSinkConfig` takes
  the envelope and the arm as one selection, mutually exclusive with the bare
  profile spelling; the `off` arm resolves to no profile and is exactly the
  default path.
- Every profile that joins an envelope carries a
  `CollectiveLatencyProvenance` record: an evidence class of `calibrated`,
  `transferred-at-use`, `provisional-transferred` or `structural-floor`, the
  source, the locator inside that source, the transfer performed, and an
  inclusive uncertainty band per participant width. A profile whose point
  value falls outside its own declared band is refused at construction, and a
  width no band anchors fails closed. An envelope additionally declares a
  point-of-use class per arm, because a number calibrated on one operation
  shape or topology is not calibrated when an envelope charges it for another:
  `b200-nccl-2.27-local-v1` is `calibrated` as an object and both envelopes
  publish it as `transferred-at-use`, which is the class the run record
  carries. Only a `calibrated` profile may be downgraded and every downgrade
  states its reason.
- `b200-nccl-2.27-cross-node-provisional-v1` is provisional-transferred and
  never calibrated. Every fabric step is the measured 2,000,000 ps propagation
  reference plus a per-step initiation term: nothing at the lower edge,
  1,000,000 ps at the point estimate (one half of the about 2 us commodity
  RDMA round-trip anchor of Kalia et al. ATC'16), and 3,000,000 ps at the
  upper edge (the top of the p50 ACK turnaround in UCCL Table 2, restricted to
  that table's Light columns, whose message sizes match this workload). Each
  such step replaces one 1,617,160 ps NVLink step from the two-point slope of
  the source table. The `2(W-1)` decomposition is this repository's own
  expansion model rather than an attribute of the capture, which names no
  algorithm; a `2 log2(W)` tree at the same per-step delta would move the
  width-8 point estimate from 49.49 to 38.43 us. TRAF-36 owns the missing
  cross-node measurement and the algorithm question with it.
- `a100-nccl-2.31-cross-node-socket-v1` is the first `calibrated` cross-node
  profile, measured on two A100 nodes whose only inter-node path is NCCL's
  kernel socket transport over Cray Cassini Slingshot 200Gb ports with
  GPUDirect RDMA disabled. It charges a 40,140,799 ps width-2 floor, banded to
  the 55,808,000 ps isolated reading, and carries a four-anchor
  `CollectiveBandwidthCurve`. It supports width 2 and fails closed at every
  other width, so it cannot be an envelope arm beside the B200 profiles, which
  is registered as TRAF-49. The measured floor is 3.744 times the
  `b200-nccl-2.27-local-v1` width-2 intercept the cross-node envelope charges on
  its `lower` arm, and the measured 20.070 us fabric ring step is 4.0 times the
  5.000 us upper edge this module prices a fabric step at. That is one transport
  on one machine and not the 400 Gbit/s RDMA path the reference configuration
  assumes; TRAF-48 owns that capture. Nothing selects the profile, no envelope
  contains it, and no reported TTFT or TPOT moves.
- The regime break that
  [collective regime curve](../../examples/collective_regime_curve_v1/RESULTS.md)
  could only hypothesize is now a measured mechanism. NCCL logs its algorithm
  and protocol per call, and on the cross-node socket path it switches from `LL`
  to `SIMPLE` between 786,432 and 1,048,576 bytes, exactly where the measured
  completion time *falls* from 1456.49 to 822.84 us while the payload grows.
  `LL` carries a 4-byte flag for every 4 bytes of payload, so it puts twice the
  bytes on the wire: nearly free on NVLink, dominant on a bandwidth-starved
  socket transport. The intra-node dip and this cross-node step are the same
  mechanism with opposite signs, and the boundary is observable at communicator
  init without any fitting.
- The fabric side now has a first-party flow-level reference dataset: the
  [Merlin fabric flow capture](../../examples/merlin_fabric_flow_capture_v1/RESULTS.md)
  publishes byte-locked long-running NCCL per-chunk completion series over
  the Cray Cassini Slingshot fabric (kernel socket transport, GDR
  disabled): a 300-second solo stream at 4.991 GB/s steady, a degree-2
  incast whose aggregate reaches 1.71 times the solo rate because each
  flow rides its own source stack and destination port, a two-node cell
  in which four same-node source stacks sharing one port reach 11.1 GB/s
  (aggregate scales with stack count, not port count), a mixed
  A100-plus-GH200 pair with a 2.77x direction asymmetry (the Grace-sourced
  leg is the slow one), and a post-specified, unscored two-flow join in which the
  established flow loses nothing at the join. It changes no profile, no
  envelope and no reported metric; TRAF-51 owns the htsim comparison
  against it and TRAF-52 the families still queued behind cluster
  reservations.
- The TRAF-51 comparison ran in wave 19:
  [merlin ss fabric calibration](../../examples/merlin_ss_fabric_calibration_v1/RESULTS.md)
  pinned the htsim ss-dragonfly fabric merge, declared a single-switch
  Merlin instance with per-parameter provenance, and validated the
  instance's exact serialization arithmetic and a frozen composition
  rule separating the measured endpoint host-stack floor (78.7 to 84.5
  percent of an 8 MiB chunk's life) from fabric serialization for the
  captured steady-state families. The fabric side passed 11 of 11 rows,
  three of them hand-derived exact oracles confirmed to the bin, and
  every composed steady quantity landed within 7.3 percent of its
  measured counterpart from solo anchors alone. The study states its
  own limit as its discrimination statement: the captured loads run
  each stack at under a fifth of a port, so any non-bottleneck fabric
  model yields identical composed verdicts, and what is validated is
  the composition rule and instance arithmetic, not fabric-model
  discrimination. The 119-second simultaneous-start transient is
  registered as un-modeled endpoint dynamics (TRAF-53) rather than
  fitted, the mixed pair stays out of the fabric's scored scope by the
  frozen exclusion, and the open-loop shared-egress artifact regime is
  echoed only as a positive control. It changes no profile, no envelope
  and no reported metric.
- The wave-21 load-bearing recalibration
  ([merlin ss fabric loadbearing](../../examples/merlin_ss_fabric_loadbearing_v1/RESULTS.md))
  reran the comparison with the fabric genuinely carrying risk through
  the pinned load harness: the measured per-stack endpoint floors enter
  as closed-loop think times and the sharing waits are simulated, so
  the composed quantities are simulator outputs rather than external
  arithmetic. All 8 scored rows passed with no guard fired: the
  captured x4 shared-egress aggregate is reproduced within the frozen
  band (composed 10.63 against measured 11.10 GB/s; the whole 4.21
  percent residual is simulated queueing, and the disclosed fluid
  napkin predicted it to 0.4 percent; the band is coarse by
  construction, separating fluid-like sharing from a chunk-serializing
  egress near 0.74 while tolerating up to roughly 2.5 times the
  observed wait, so the sharing-mechanism class is validated rather
  than a 4-percent tolerance), two buffer configurations
  byte-identical at capture-shaped load produce opposite registered
  verdicts on the composed x4 cell (4 MiB faults by the registered
  closed-loop drop signature, 32 MiB completes in band) plus banded
  saturating-arm separations, and the p50-static endpoint floor
  overshoots the measured aggregate by the registered 12.7 percent,
  refuting static p50 floors for skewed shared-port families (TRAF-53
  evidence). None of this claims which buffer value the Merlin switch
  physically has: the closed-loop abstraction carries no loss
  recovery while the real transport does. A frozen late-arrival path
  stands ready to score any tranche-2 shared-egress group with no
  code change. It changes no profile, no envelope and no reported
  metric.
- Because the table is a surcharge on a transport that already contains one
  propagation delay, `realized_fixed_cost_ps` is what a run actually charges,
  and it exceeds a source capture that was itself a complete fixed cost by up
  to `propagation_reference_ps`. TRAF-37 owns that over-count.
- `arm_ratio_envelope` is the reporting helper an envelope study publishes
  with. Given one `(arm, numerator_ps, denominator_ps)` row per arm it returns
  the per-arm ratios, the interval they span, and whether that interval
  brackets 1, i.e. whether the evidence determines the sign of the comparison
  at all. Quotients are exact before conversion to float, so a large fixed
  cost cannot swallow the low-order digits of a ratio.
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
  the layer's TP allreduce sites when the TP world produces them, then for MoE
  dims with `ep_ranks` given the dispatch and combine all-to-allvs) through the
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

## Collective completion and registration, the interim contract

This section is the public statement of the collective carve-out in the
maintainer's kernel-time determinism ruling of 2026-08-18. Everything else in
that ruling makes a kernel's service time a deterministic constant with no
tail. Collective work is the one exception, and this is what the exception
means while it lasts.

**Completions are deterministic no-tail constants.** For a given traffic and a
given fabric state, a collective's completion time through the ATLAHS and htsim
chain is a constant. It is not sampled, it carries no per-call jitter, and it
is identical across ranks and across runs for the same inputs. Latency spread
in a served deployment comes from the network, from batching and from queueing,
never from a collective drawing a different number twice. An executed artifact
composes as `registration + base_latency + max(local_service,
fabric_transport)`, and every term of that sum is a deterministic function of
its inputs.

**Registration gates the completion.** A collective may not complete on memory
the communication stack has not registered. The first collective to use a
`(communicator, generation, channel, buffer)` identity pays an explicit
one-time registration cost, serialized ahead of its own completion; every later
collective on a registered identity pays nothing. Exactly three events force a
re-registration: a new buffer, a new peer set, and a communicator rebuild.

**Where the halves live, and where they are not yet joined.** The cost model,
the identity rules and the ledger are traffic-owned, in
`simllm.traffic.collective_registration`. The live charge reaches TTFT and TPOT
through `HtsimStepSinkConfig.collective_registration`, which is off by default;
with no model named, the ledger charges zero, records nothing, and every
accepted artifact, timestamp and metric stays byte-identical to the baseline.
Those two are one authority with an explicit projection. The registration
boundary is separately mirrored at the plugin seam in
`simllm.compute.nccl_stack`, where `ncclNetRegMr` mirrors the net plugin's
`regMr` together with the channel FIFO establishment that follows it and
`require_buffer_registration` is the gate; that seam declares a cost and never
advances a clock, which is that module's standing contract. The seam is not
joined to the ledger: it keeps its own per-communicator registered-buffer
state, that state carries no generation, and the live chain never consults it.
TRAF-58 unifies them.

**Almost all of this is declared, not measured.** Exactly two things rest on
evidence. `regMr` and `regMrDmaBuf` are members of the documented `ncclNet_v6`
struct and NCCL calls them so an RDMA NIC can prepare a buffer, so a
registration entry point exists at this seam; and RCCL exposes the same ABI, so
one seam serves both stacks (see
[the AMD GPU fabric note](../papers/amd-gpu-fabric.md)). Everything else is a
model choice this repository declares: that the cost is paid once rather than
per call, that the identity is scoped to a buffer, that a channel belongs to
that identity, that exactly three events force a re-registration, and the 20
microsecond duration. Asking the shipped cost for a calibrated value raises
rather than returning the declared constant under a calibrated label; TRAF-56
is the calibration, and it has to measure the model choices as well as the
constant.

**This is interim.** The registered destiny is a packetized NCCL and RCCL
collective path over the GPU's own NVLink, xGMI and UALink ports, where a
channel is bound to a port, a chunk becomes packets, and the registration
handshake becomes port traffic instead of a declared constant. TRAF-54,
TRAF-55 and TRAF-57 carry that work, above the packetized intra-node leg of
TRAF-45 and on the port events of COMP-40 and the port-kind-independent
vocabulary of BACK-48. When they land, the constant completion becomes an
emergent one and this section is replaced rather than amended.

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

The `loggopsim-ideal` level passes the frozen
[ideal-network study](../../examples/loggopsim_ideal_v1/RESULTS.md): 30 of 30
exact arithmetic observables, 3 of 3 live metric-chain identities and 3 of 3
wall-time ceilings in separate evidence classes, with all fatal guards held.
The live step's independently reproduced network makespan and its TTFT delta
against the zero-collective control are both exactly 202,000 ps. The
[frontier ladder](../../examples/frontier_ladder_v1/RESULTS.md) measures
modeled error against pinned packet observations through M-1, M-2 and M-3,
but executes no packet reference and therefore measures no packet wall clock.
The level refuses overlapping multi-source receiver fan-in by default, with
an explicit acknowledgment and provenance stamp for deliberate envelope
measurements. The separately frozen
[acceptance study](../../examples/loggopsim_acceptance_v1/RESULTS.md) executes
both arms seven times on each of the twelve byte-identical flow sets. The
packet arm reproduces all twelve pinned completions exactly and all three
enforcement predicates pass, but the full qualification is honestly REFUTED:
1.088866981 packet seconds over 0.029767114 ideal seconds is 36.579528x,
below the frozen 50x floor. All four fatal guards hold, so TRAF-20 stays open
on that speed miss rather than on packet anchoring or envelope enforcement.

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
remains ambiguous. The legacy and all-remote
identity paths remained exact. The public evidence supplies a B200 collective
capture and a vendor capacity ceiling, but not the same-generation
point-to-point payload capture required by the original clause.

The [composed step budget study](../../examples/composed_step_budget_v1/RESULTS.md)
then measured the composition that study left open. Running the mission chain
with the floor and the same-wave host term both enabled charges the floor
additively: a step is `max(C, N * g)` plus one 30,128,029 ps base latency per
semantic collective plus the raw fabric transport, so the arithmetic
1.651145392 ms whole-step figure above understates the measured 1.916754 ms
CUDA-graph composition by 13.9 percent and describes no host profile. The floor
stays outside the raw transport: over 31 matched decode compositions
`2 * fabric(400G) - fabric(200G)` returns 96,000,006 to 96,000,048 ps, the
mission study's own 48 propagation charges of 2.000 us plus backend
quantization. It also compresses bandwidth sensitivity as an additive term
should, moving the 400 to 200 Gbit/s decode-step ratio from the published
1.0441 to 1.4760 down to 1.0047 to 1.0148. Two limits are now quantified rather
than asserted: the floor is 74.73 to 75.45 percent of the composed step at the
CUDA-graph profile, so the transferred ALL-REDUCE intercept dominates a number
this repository reports, and the width-8 endpoint envelope's 458,752-byte
ceiling was reached to within 17 percent by a 34-token prefill step, so a
larger case would be rejected at planning time.

The [fixed-cost envelope study](../../examples/collective_fixed_cost_envelope_v1/RESULTS.md)
then replaced the silent choice of that intercept with a bracket. Sixteen cells
over four named arms, two link rates and two expert-parallel widths, replayed
through the fluid backend on an all-remote placement, put the per-collective
surcharge at width 8 between 0 and 49,487,789 ps and the realized fixed cost,
propagation included, between 2,000,000 and 51,487,789 ps, a factor of 25.7.
Across the fixture's 48 collectives that is 46.0 to 95.6 percent of a decode
step, so the fixed cost dominates every arm including the default, and the
default is the only arm that does not say so. The arm envelope of the
ep4-to-ep8 decode ratio is 0.547454 to 1.248990 at 400 Gbit/s and 0.549095 to
1.224748 at 200 Gbit/s: both brackets contain 1, so the sign of the
expert-parallel width ordering is not determined by the available evidence. The
compression is equally large, moving the 200 to 400 Gbit/s ep8 decode ratio
from 1.065947 under the default to 1.005326 under the provisional cross-node
arm. Every one of the 40 simulated steps is reproduced to the picosecond by
`compute_service_ps + 48 * fabric_service_ps + 48 * surcharge`, with the fluid
backend returning exactly one picosecond above the closed-form fabric service
at all eight width, phase and rate points. Attempt one is retained as void
because the structural fatal guard compared a list of tuples against a list of
lists and could not hold for any data; attempt two repaired only that
comparison and reproduced every raw measurement exactly.

The 2026-08-14 TRAF-33 qualification corrected the tensor-parallel allreduce
site inventory and closes the task. A layer's sites come from
`layer_tp_allreduce_sites`, so a layer whose output arrives through a combine
all-to-all reduces once, after attention, instead of twice. Over 54 frozen
cells crossing model kind, tensor-parallel width, layer count and token count,
the GOAL renderer, the communication-phase planner and the graph lowerer with
its collective plan each reproduced one frozen closed form exactly, and the
all-to-all tag base moved down with the shortened ring list without sharing a
tag with any ring block. All 120 pre-registered scored instances in four
families passed and no fatal guard was violated, though only 48 of the 120
exercise a layer the change alters; the other 72 are dense and
expert-tensor-sharded regression arms whose value is that they did not move.
The reference 24-layer cell with an 8-rank tensor-parallel group and a declared
8-rank all-to-all expert-parallel group renders 24 allreduces plus 48
all-to-alls, that is 72 collectives and 8,257,536 tensor-parallel bytes,
against 96 collectives and 16,515,072 bytes before. Per site the rendered
344,064 bytes sit exactly on the `2(W-1)P` bandwidth-optimal floor at width 8
and a factor of four below the naive all-gather ceiling. Every arm that renders
no all-to-all, dense, expert-tensor-sharded and naive expert-parallel alike, is
byte-identical to the pre-change renderer with its digest pinned in the tests.
See [the allreduce site results](../../examples/moe_tp_sites_v1/RESULTS.md).

Two integrator reviews then corrected the merged rule, the producer feeding it
and two of that report's statements; all of the following is post-specified.
The first rule keyed on the dims alone and was wrong for naive expert
parallelism, so it now keys on the declared group as described in the interface
above, and a further 18 post-specified cells covering that shape passed 36 of
36 in the same three path families. Those instances are a separate evidence
class and are not summed with the 120. The second review found three defects
one level out from the rule. The vLLM producer bound the group unconditionally,
so it still declared an all-to-all for a `dp=1` deployment and for the default
non-reducing backend; it now classifies both pinned conditions and refuses the
backend whose allgather and reduce-scatter traffic this repository renders
nothing for. The reader's refusal list was a strict subset of the SGLang list
it claimed to mirror and now reaches parity with it on the shared-expert and mixed-schedule
families with per-field predicates, the MLA, speculative and
quantization refusals staying outside this guard's scope. And the one-reduction invariant had a representable
counterexample, a uniform per-pair share flooring to zero bytes, which left a
layer reduced zero times; both inventories now take their exits from one shared
predicate and a new fatal guard measures the invariant on all 72 cells. The
freeze's registered invariance clauses for rank relabeling and expert-parallel
group width, unexecuted in the first run, now run at every cell along with a
sweep of the resident-expert count the corrected rule no longer reads, and the
routed byte totals are asserted against a closed form confirmed identical to
the merge base over all 72 cells. The end-to-end weight is the per-collective
base latency rather than the bytes: removing 24 phantom collectives removes
`24 * 30,128,029` ps, that is 0.723073 ms of additive base latency, which is
21.50 percent of the 3.362899 ms defective step and 27.39 percent of the
2.639827 ms corrected step once a tensor-parallel group is declared. The
earlier 38 percent figure divided by a 1.916754 ms step measured at
`tp_ranks=(0,)`, a configuration with no allreduce at all, and is retracted, as
is the freeze's napkin line that charged an aggregate byte count to a single
link and overstated the serialization surplus eightfold.

One rendered surface and one prose projection are stale across TRAF-33, and
TRAF-41 owns both. The Granite live cells of the collective plan default study
declare one 8-rank group as both the tensor-parallel and the expert-parallel
group over expert-parallel dims, so their 709,803,840 ps TTFT, 132,794,880 ps
TPOT and transport rows were measured with 48 rather than 24 allreduces per
step and must be rerun. The composed step budget study's measured
1,446,145,392 ps collective floor is 48 MoE all-to-alls at `tp_ranks=(0,)`,
which this change does not touch and a rerun reproduces exactly; what is stale
there is its counterfactual paragraph projecting that a 24-layer model's
tensor-parallel allreduces would add another 1,446,145,392 ps and be 74.73 to
75.45 percent of the composed step, since under a declared reducing all-to-all
group the projected addition is 723,072,696 ps. That paragraph needs amending
rather than rerunning, and the measured floor is never halved. Both studies'
dense cells, including the 196,608-byte and 4,730,040 ps rank-order row, are
unaffected, as are the MoE studies that render an expert-parallel group with a
tensor-parallel world of one.

The corrected inventory and the fixed-cost envelope were exercised together on
a live chain for the first time by
[examples/sglang_composed_deployment_v1](../../examples/sglang_composed_deployment_v1/RESULTS.md).
Its intra-node cell declares one 8-rank group as both the tensor-parallel and
the expert-parallel group over expert-parallel dims, which is the shape TRAF-41
has to requalify elsewhere, and it reported the corrected 24 attention
allreduces plus 48 MoE all-to-alls on every one of its steps, 408 executed
artifacts, with the summed base equal to the collective count times the arm
constant exactly, which is the check that a ring is charged once rather than
once per round. Two limits of the fixed-cost surface became measurable there
and are registered as TRAF-42. The realized bracket the envelope publishes,
`[2,000,000, 32,128,029]` ps at width 8, adds a propagation reference that an
all-intra-node step never charges, so the local path realizes
`[0, 30,128,029]` and its `off` arm has no physical floor; the cell measures
885 ns per collective on its first prefill step, all of it endpoint
serialization. And the applied evidence class is per arm, so the same cell has
to publish `transferred-at-use` for a surcharge that is 723,072,696 ps at the
capture's own operation and interconnect and 1,446,145,392 ps by transfer. The
study also confirms that a resolved profile replaces the declared NVLink
endpoint rate as well as adding a surcharge: the same step's local service is
10,052,000 ps at 450,000,000,000 bytes/s and 63,746,000 ps at the profile's
70,027,079,100 bytes/s, so on a locality-bearing cell the `lower` arm is a
bandwidth arm rather than a null arm. That coupling is intended, since the
intercept and the slope were fitted together, and it is recorded here so a
study reports all three arms rather than treating `lower` as `off`.

The interim collective-registration cost is implemented in
`simllm.traffic.collective_registration` and gated at the mirrored plugin seam
by `ncclNetRegMr` and `require_buffer_registration`. It is off by default: a
`CollectiveRegistrationLedger` built with no model charges zero, records
nothing, and leaves every accepted artifact, timestamp and metric exactly where
it was. The declared cost is 20,000,000 ps per identity, `calibrated_cost_ps`
fails closed on it, and TRAF-56 is the calibration. The
[registration study](../../examples/nccl_registration_v1/RESULTS.md) is
interpretable with all nine fatal guards held, 6 of 6 exact-oracle rows and 3
of 3 behavioral families over 7 instances, in two classes never summed. The
opt-in moves TTFT by exactly 1,280,000,000 ps in a TP2 prefill cell whose 64
collectives register 64 identities, by exactly twice that at two channels, and
by exactly 80,000,000 ps in a two-node TP4 cell where every collective is split
into several executed artifacts and a real `htsim_rnic` process decides each
fabric term. Every later step, every GOAL artifact digest and the
default-constructed arm are unchanged.

The declared-constant half of TRAF-61 is live in the CORE-51 session. One
handoff record derives 393,216 or 786,432 bytes from the Granite KV geometry
at prompt lengths 8 and 16, advances the shared clock by the selected 100 or
200 microseconds, and emits no backend run or packet artifact. Doubling the
constant moves every TTFT by exactly 100,000,000 ps and moves no TPOT. Its
explicit packet arm renders eight pairwise shards through GOAL and the packet
backend after one 20,000,000 ps PCIe submission term. The bounded study
conserves every byte and endpoint, completes at the last arrival, moves TTFT
with 0 ps residual and leaves decode TPOT unchanged. PLACE-5 now supplies the
complete declared target topology, so the bounded TRAF-62 packet mechanism is
closed and TRAF-64 is unblocked for target-scale qualification. TRAF-61 stays
open until that qualification; see
[the constant-arm results](../../examples/pd_session_v1/RESULTS.md),
[the packet-arm results](../../examples/pd_session_fabric_handoff_v1/RESULTS.md)
and [the target-topology result](../../examples/disaggregated_target_topology_v1/RESULTS.md).

The 2026-08-26
[clean finite-boundary repetition](../../examples/deployment_curve_v1/traf67_calibration_result.md)
closes TRAF-66 and TRAF-67 under their literal clean-pass clause. The
field-addressed access ledger contains one visible 1K COMP-75 row and the
held-out ledger is empty; the reader stopped after 2,193 of 8,415 bytes, before
any later record field. The unchanged two-child form remains
`max(C, P) + min(C, P) / 2`, with all 116 dispatch and 116 combine launches and
completions, 232 yields and 234 child-stage advances conserved. Its independent
visible residual moves from -0.592425 percent to -23.423673 percent, exactly
reproducing the -22.831248 percentage-point movement, and all 27 prior-artifact
locks pass. No 2K or 4K value was accessed or compared, no scored run was
performed, and decode pricing, TRAF-65 and the NVLink scope remain untouched.
The third scored run alone owns the held-out comparison. The frontier ladder
below carries the next fabric-specific comparison without rescoring those
held-out anchors.

The first frozen
[two-network bottleneck map](../../examples/deployment_frontier_v1/RESULTS.md)
publishes both elapsed attributions and both raw excesses at all 18 CORE-62
points. The raw fabric mechanism is serialization in the two-node arm and
incast in the nine-node arm; the candidate intra-node module is TX credits and
packetization throughout, with zero pass-through switch contention. Seventeen
points are roofline-bound, one B100 batch-32 point is intra-node-bound and no
point is inter-node-bound. The nine-node batch-32 raw incast excess is
6.743004420 ms but remains off critical below the 9.535537623 ms H100 roofline.
That misses the frozen positive elapsed-incast direction, so the step-level
study remains an honest refutation of its positive elapsed-incast prediction.
The A100 NVLink3 profile remains cross-architecture candidate evidence, not
H100 or B100 measurement evidence. The TX, switch and RX queues, analytic
identity bypass, frozen identification map and current evidence split are
documented in the [NVLink domain-model study](../design/nvlink-domain-model.md).

The
[frontier ladder](../../examples/frontier_ladder_v1/RESULTS.md) carries 24
ESTIMATE and 30 SIMULATED points without merging their authorities. The six
B100 ideal-rung points are closed-form ESTIMATE points with no execution; the
twelve H100 ideal-rung points execute from rendered GOALs through the
digest-pinned binary with the exact `G` string `0.02`. All twelve native legs
match the frozen literals across seven repetitions, and all four fatal guards
reject genuine mutations.
At batch 32, the packet-to-ideal quotient is 1.015637 for the serialized
two-node shape, 8.110405 for eight-into-one incast and 1.015682 for the
isolated incast control. The mechanism is explicit: the ideal receiver charges
no per-byte gap, while the packet receiver serializes the shared ingress. At
step level, the H100 kernel masks every fabric difference and the sole rung
movement remains the pinned B100 batch-32 intra-node excess. The original
published median is 0.034612 seconds for all twelve ideal legs. The quantified
packet-error envelope completes TRAF-20's modeled-error half. The
[acceptance study](../../examples/loggopsim_acceptance_v1/RESULTS.md) executes
the same twelve byte-identical flow sets through both arms with seven samples
per arm. B passes 12 of 12 with every packet quotient exactly 1.0, and C passes
3 of 3 for refusal, acknowledged fan-in, and clean-path identity. A passes
only 1 of 2: the 0.029767114-second ideal total meets its one-second ceiling,
but the 1.088866981-second packet total yields 36.579528x rather than the
required 50x. The valid refutation keeps TRAF-20 open. The fabric-leg view
that exposes the contention masked by the original step map closes TRAF-68.
Neither result makes an absolute-accuracy claim against silicon.

The scored NV4 domain now publishes its packet-level flow joins, convergence,
divergence and flow-completion-time distributions in
[the TRAF-69 result](../../examples/nvlink_flow_dynamics_v1/RESULTS.md).
The 1-to-2 open identity lands at 13,886 ps and the 2-to-1 solo-target identity
at 16,684 ps, both with 0 ps residual. The reverse-target schedule completes
flow C, flow B, then flow A, and all 219 stable raw-bin rate checks pass. Every
one of the 21 incast-degree-by-size CDF cells passes its frozen p50 and p95
bands. Simulated payload reaches 94.009808, 187.880751 and 194.562756 GB/s at
degrees one through three, respectively, without exceeding the frozen
94.117647, 188.235294 and 194.919456 GB/s ceilings. The published 281.65 GB/s
sender fan-out row remains separate and is honestly refuted by a 46.334975
percent miss. All 13 fatal guards and all 60 preservation locks pass, including
the byte-identical default flow-inactive path, so TRAF-69 closes. The result
keeps the measured TX and RX plateaus, ten declared candidate parameters and
the structural pass-through switch, the score's eleven unchanged internals in
total, visibly separate; TRAF-65 remains open on its live held-out integration
bar.

The [TRAF-71 comparison](../../examples/nvlink_rnic_comparison_v1/RESULTS.md)
places that scored three-module credit domain and pinned htsim `rnic-nn` on one
zero-fit NVLink physical mapping and the byte-identical seven-rung, nine-seed
staggered workload. rnic-nn is tighter in 8 of 21 rung-degree cells, NVLink in
11, with two ties; at 64 KiB and above rnic-nn is tighter in only 5 of 9, so
the frozen 7-of-9 smoothness prediction is honestly refuted. The merged NVLink
raw sample and CDF hashes reproduce exactly. No credit-window stall occurs:
the first credit returns at 210.880 ns while one 256-packet bonded-link round
spans 696.320 ns. The small-rung roughness instead follows finite empirical-CDF
steps and frozen stagger alignment, with RX arbitration contributing to the
NVLink transport shape. Both arms share the degree-3-left-of-degree-1 sign only
at 1 KiB; the five larger NVLink-only signs stay attributed to release-aware
packet round robin and stable RX admission order. The pinned rnic-nn profile is
central max-min packet-slot pacing with zero ACK events and reverse bytes, not
ACK pacing. All 16 fatal guards pass, the inherited 60 artifacts and all 18
merged flow-dynamics files remain byte-identical, both final figure pairs pass
visual inspection, the two frozen scientific misses E3 and E5 remain visible,
and TRAF-71 closes.

TRAF-72 supersedes TRAF-71's degree-3 interpretation without editing that
merged study. The pre-run audit finds no capacity-value deficit: the legacy
rnic-nn receiver received 100.000, 200.000 and 207.101921876 GB/s at degrees
1, 2 and 3, so degree 3 divided exactly the same 207.101921876 GB/s aggregate
that limits the NVLink composition. Its 1.000000 raw capacity ratio cannot
explain the reported `30.203976 / 18.145397 = 1.664553` p50 ratio. The deficit
is instead the mapped fair-share entity. TRAF-71 admitted each overlapping
transfer as an independent max-min flow while the NVLink side queued extents
within an ordered-pair source class. On the frozen release interval `3S/4`,
the normalized nearest-rank p50 ratio is `(601S/160)/(9S/4) = 601/360 =
1.669444`, within 0.3 percent of the observation. TRAF-72 freezes one active
fair-share entity per ordered pair, the htsim fluid null reference, and the
incast extension to degrees 4, 8 and 16.

The [TRAF-72 result](../../examples/nvlink_rnic_comparison_v2/RESULTS.md)
confirms the mapping-deficit verdict. At degree 3 and 512 KiB the corrected
rnic-nn p50 is 18.120617 us, left of the regenerated NVLink value 18.145397
us, and the legacy-to-corrected ratio is 1.66683 against the frozen 1.669444
prediction. The fluid arm agrees with its independent continuous-service
oracle to 0 ps in all 42 cells. Its stronger location hypothesis is honestly
refuted in 13 of 252 comparisons because an indivisible packet slot can
complete a selected 256 B or 1 KiB flow before equal continuous fluid shares
finish. The mesh tail hypothesis is also honestly refuted: both fair-share
references beat NVLink in all 36 frozen small-flow p99 and worst-flow
comparisons, but none of the 12 relative advantages grows monotonically with
degree. The fairness hypothesis passes 11 of 24 clauses; in particular, 256 B
packet-slot discreteness makes both references less fair than NVLink through
degree 16. All 126 capacity and byte-ledger cells, 12 fatal guards and 16
TRAF-71 preservation locks pass, so TRAF-72 closes while its three behavioral
refutations remain part of the result.

TRAF-73 separates credit ownership from downstream arbitration. Public NVLink
documents establish credit flow control and describe multiple virtual-channel
classes, but they do not identify the A100 credit quantum, allocation scope,
virtual-channel count, buffer depth or product arbiter. The model carries one
implicit virtual channel and keeps the existing 256-credit, 272-byte and
200,000 ps values as declared candidates on each physical link. Incast
contention sits at destination ingress and memory acceptance on NV4, plus a
crossbar output on an NVSwitch path. Release-aware round robin is the declared
baseline candidate, with static interleave and greedy capture selectable as
alternatives. None of these policy labels is a product fact or hardware
measurement.

The [TRAF-73 simulation result](../../examples/nvlink_credit_arbitration_v1/RESULTS.md)
passes all 15 frozen policy and degree instances with all 105 fatal guards
intact. The physical degree-3 cell predicts raw wire shares of 87.159, 59.921
and 59.921 GB/s for release-aware round robin; 60.000 GB/s per source for
static interleave; and 99.760, 53.621 and 53.621 GB/s for greedy capture.
Degrees 4, 8 and 16 remain labeled simulated mesh extrapolations. These values
are frozen predictions for the registered hardware discriminator, not a
promotion of any arbitration policy.

**Expert-parallel collectives are priced over both intra-node accelerator
transport and cross-node fabric, with fixed launch, synchronization and
algorithm-selection overheads carried once in the common metric chain.**

The MiniMax expert-parallel packet arm routes a study-only balanced population
as whole token-expert assignments, emits messages only to realized
destinations, and declares FP8 dispatch separately from BF16 combine. The
[MiniMax scaling result](../../examples/minimax_ep_scaling_v1/RESULTS.md)
publishes the original run as void, then compares two cost models on the same
requested generic half-precision all-gather plus reduce-scatter element count.
The external arm is an opaque eight-rank NCCL table measurement scaled by a
rank factor, with no source, destination, path or message ledger. The packet
arm is a direct all-pairs realization on a concrete Clos placement. Their
ratio is not contention isolation. The aggregate collective authority corrects
the measured ratios from the superseded 0.025905, 0.353015 and 0.802618 to
1.109143, 0.435919 and 0.847299 at expert-parallel widths 8, 32 and 128. The
unchanged lower bound is 1.0, so EP 8 passes while EP 32 and EP 128 remain
refuted. EP 8 has zero cross-node senders per receiver and is not a contention
cell. The EP 256 component-wise diagnostic ratio is 1.218997, superseding
1.187022. It remains unscored because its extrapolation rule was specified
after the corrected expectations freeze.

The external table identifies its dtype only as `half`, not BF16. Its source
coordinate is an element count despite the SDK interpolation label
`message_bytes`; the corrected aggregate authority converts that coordinate
to true bytes using the dtype width before fitting. The
[aggregate calibration](../../examples/collective_floor_calibration_v1/RESULTS.md)
reduces held-out median error from 91.6161 percent for the actual current ring
path to 3.2671 percent. Its frozen precision family remains refuted because 12
of 63 held-out cells exceed 10 percent and p95 is 19.7972 percent. The D8
coordinate freeze maps the external query to its operation buffer: 98,304 half
elements, or 196,608 bytes per phase. That matched query prices the EP-8
zero-fan-in cell at 2.131828400 ms, 1.109143050 of its 1.922050 ms external
arm, and refutes the unchanged `[0.90, 1.10]` band. The unscored physical
endpoint reading at 172,032 bytes is 2.060523530 ms, or 1.072044707 of the
external arm. The earlier 344,064-byte query doubled already physical bytes
and does not support a pass. The earlier Family D publication omitted this
aggregate term from its packet arm. Its old EP 8 refutation was therefore an
artifact of the omission, not a finding about the external planner. The old
EP 256 rule also multiplied the whole EP 128 phase by `31 / 15`; the corrected
rule scales only fabric service, queries the EP 256 aggregate byte coordinate,
and adds each fixed floor once. The corrected sequence is not monotone because
EP 8 passes before EP 32 and EP 128 refute, while EP 256 is unscored. The
published single crossover near expert parallelism 200 is withdrawn, and the
study makes no contention-attribution claim.

The unscored sparse arm simulates every realized message at every width and
uses the same aggregate authority through acknowledged FP8-dispatch and
BF16-combine donor transfers. Its corrected sparse-to-dense step ratios are
1.072932, 0.489749, 0.336183 and 0.302797 at widths 8, 32, 128 and 256; its
floor-omitting values remain labeled superseded. At EP 256 it reconstructs
29.78125 cross-node senders per receiver from simulator completion rows,
against an analytical 29.576912, and carries 97,920 dispatch plus 195,840
combine bytes per rank. The isolated-engine default is unchanged. The
existing MiniMax publication is complete only for its pinned legacy authority.
Its EP 8 PASS holds only under that legacy authority pin; explicit successor
binding gives quotient 0.946736591 against the MiniMax freeze's
packet-over-external floor of 1.0 and therefore refutes the cell. The rebinding
and superseding publication are registered in TRAF-83 rather than silently
defaulted. TRAF-76's aggregate surface records a post-specified regression on
an adaptively reused 63-cell evaluation set with training-cell-only numeric
evaluation: all 63 H200 cells pass with 2.5010 percent median, 8.6958 percent
p95 and 9.9262 percent maximum relative error. The matched D8 coordinate moves
from quotient 1.109143050 to 0.946736591 inside that Family D8 comparison's
unchanged `[0.90, 1.10]` band. Its opaque completion is one charge with zero
exposed serialization, and the accepted floor-plus-slope donor transfers
remain byte-identical. TRAF-76 stays open for packet integration, explicit
MiniMax successor rebinding and genuinely independent H200 validation, owned
by TRAF-82, TRAF-83 and TRAF-84 respectively.
TRAF-78 owns observed routing geometry, TRAF-75 owns supported-path directional
precision, TRAF-77 owns hardware transport calibration, and TRAF-26 owns
complete production peer workloads.

The [second TRAF-74 capture](../../examples/nvlink_incast_validation_v1/RESULTS_RUN2.md)
validates the scored base domain at the only incast degrees a four-A100 `NV4`
node can realize. Merlin job `202466` completes all 42 rows, all 13 fatal
guards pass, and the worst sequential launch-skew fraction is 1.129 percent
against the frozen 10.000 percent ceiling. All six cells then miss the frozen
plus or minus 16 percent band: hardware aggregate goodput spans 4.461874 to
14.288373 GB/s while simultaneous-release simulation spans 94.104154 to
194.808553 GB/s, with a worst absolute signed relative error of 2011.175
percent. The frozen size-dependent rule attributes every miss to
packetization. TRAF-74 closes as a literal non-void validation, and TRAF-86
owns the identified packetization precision residual.

NVLink hardware incast identification is long-flow only. Sender launches on
the real node serialize through sequential PCIe writes, so nanosecond-scale
true synchronous small-flow co-arrival cannot be constructed. Simulated
small-flow incast is a model prediction with no direct hardware check. Degrees
4, 8 and 16 are a declared simulated mesh extrapolation with no hardware
counterpart on an NV4 node; an NVSwitch-class configuration is the physical
route to those higher degrees.

TRAF-79 reverse engineers the NVLink packet domain from public documents. Its
28-choice reconciliation confirms 10 current choices, contradicts 6 and
leaves 12 undocumented. The deciding correction is that the documented family
unit is a 128-bit, 16-byte flit and a packet occupies 1 through 18 flits; 272
bytes is therefore one 17-flit occupancy, not the universal unit. The
[mechanism record](../design/nvlink-mechanism-reverse-engineering.md) makes the
public evidence boundary literal and closes TRAF-79. TRAF-80 owns the model
alignment, while TRAF-73 remains open to identify the A100-specific credit and
arbitration parameters after that structure lands.

## Open tasks

### Precision

- TRAF-81 (Precision; P1; L): complete the blocked rank-16 cell in the
  [independent collective-floor extrapolation study](../../examples/collective_floor_extrapolation_v1/RESULTS.md).
  The frozen rank-2 and rank-4 training cells and rank-8 holdout completed on
  Merlin A100 GPUs with all fatal guards held and every measured rate inside
  its physical envelope. At the first locality crossing, rank-4 donor
  extrapolation misses rank-8 all-gather by 62.889 percent median and 155.593
  percent p95, and reduce-scatter by 61.111 percent median and 144.544 percent
  p95, against frozen 25 and 50 percent bands. The floor-fraction family also
  fails both operations. This refutes normalized-efficiency and
  floor-versus-slope transfer from one NV4 node to two nodes on the measured
  A100 system. The formal verdict remains `BLOCKED`, not a completed
  refutation, because Merlin's per-job quality-of-service limits admit no more
  than eight GPUs while rank 16 needs 16, leaving the frozen error-growth and
  sign family unevaluated. Obtain a conforming four-node rank-16 allocation,
  run only the frozen missing cell without substituting another topology or
  transport, then publish its descriptive fit, byte-level errors and S3
  result. The evidence remains shape-only across architectures: no A100
  absolute latency, bandwidth, floor or slope calibrates H200. This study
  changes no installed authority or signature metric by itself.
- TRAF-80 (Precision; P1; L): align the three-module NVLink packet, credit and
  switch domain with the public mechanism boundary established by TRAF-79.
  The surrogate being replaced treats one fixed 272-byte extent as the packet
  and credit unit, frees the sender credit on a fixed timer, models one
  implicit virtual channel, exposes destination service without explicit
  receive-buffer ownership, and gives the switch no port, virtual-output-queue
  or crossbar contention. Replace it with generation-scoped 16-byte flit
  packetization with optional control flits, link-local acknowledgement and
  replay accounting, explicit traffic-class and virtual-channel state,
  receiver-owned credit release, ordering visibility, and an NVSwitch policy
  seam over input ports, virtual output queues and crossbar outputs. Do not
  promote an exact A100 credit quantum, pool scope, virtual-channel count,
  buffer depth, credit-return encoding, striping granularity or product arbiter:
  public documents leave those parameters unidentified, so TRAF-73 remains
  their measurement owner. Preserve the profile-absent path and the NV4
  direct-mesh structural pass-through path exactly, and make the identity
  arbitration policy preserve every accepted timestamp, wire byte, random
  draw and completion order. Land an expectations-only commit before the
  behavioral implementation. Its minimum physical oracles are 94.117647 GB/s
  payload for repeated 17-flit packets and 88.888889 GB/s for repeated 18-flit
  packets on four 25 GB/s A100 links, a nonnegative 5.882 percent serialization
  change when the optional flit is present, credit release never preceding
  receive-buffer release, and error-free replay producing zero added bytes and
  time while injected errors add neither negative quantity. The sanity study
  varies packet occupancy and link rate, measures one fixed job-completion
  time, and checks the frozen serialization relation. Publish the signed shift
  in every inherited envelope before TRAF-73 begins identification; a result
  that violates any conservation, ownership or identity guard is void.

- TRAF-86 (Precision; P1; L): replace the NVLink domain's declared packetization
  service conversion identified by the completed TRAF-74 comparison. The
  surrogate being replaced maps each logical flow directly onto the profile's
  256-byte payload plus 16-byte header and then services those packets at the
  measured TX egress and RX ingress plateaus. On base commit `6559313`, it
  predicts 94.104154 to 194.808553 GB/s for the six simultaneous-release cells.
  Merlin job `202466` instead measures 4.461874 to 14.288373 GB/s at 4 MiB and
  8 MiB, with all 13 fatal guards passing and every per-source completion plus
  aggregate comparison outside the frozen plus or minus 16 percent band. The
  size-dependent miss shrinks by more than five percentage points at 8 MiB for
  every degree, so the frozen TRAF-74 rule names packetization rather than the
  TX plateau, RX plateau, credit round or pass-through switch identity. Treat
  that capture as identification evidence, not as a held-out acceptance set.
  After TRAF-80 lands the public-document packet and credit structure, freeze
  the effective logical-message to packet-service conversion, its counter or
  trace observable, and new 16 MiB and 32 MiB held-out long-flow cells before
  implementation or capture. The replacement must keep checksum, ordering,
  per-link data and raw counters, replay, recovery, throttle, topology and
  competing-process guards decidable; put the base conversion behind an
  explicit compatibility mode that reproduces every frozen base completion,
  aggregate, byte count and order exactly. Acceptance requires one new non-void
  degree-1, degree-2 and degree-3 comparison whose aggregate and every
  per-source median are inside its pre-run physical band, plus an end-to-end
  TTFT or TPOT change through the supported metric chain. Degrees 4, 8 and 16
  remain declared simulation-only extrapolations and are not validated by an
  NV4 result.

- TRAF-77 (Precision; P1; L): replace the MiniMax scaling study's borrowed
  32 MiB switch-wide buffer and uncalibrated rnic-cn transport service with
  independently observed multi-node phase timing, queueing and buffering
  evidence. The corrected study path supplies full realized populations,
  explicit routing geometry and directional byte widths to the transport, but
  it has no H200 hardware capture and still selects its switch buffer from
  another physical RNIC runtime. Capture phase release and completion times,
  receiver ingress occupancy, path choices, queue waits and buffer high-water
  marks for at least two routing concentrations and two expert-parallel widths.
  Freeze the sweep and expected directions before capture. Acceptance reports
  the transport surrogate's before error, fits no scored holdout, reproduces
  phase makespan and receiver occupancy inside frozen quantitative bands, and
  demonstrates an end-to-end TTFT or TPOT change through the supported metric
  chain. The explicit uncalibrated transport mode remains selectable and
  reproduces the corrected result exactly when hardware calibration is
  disabled. TRAF-78 owns observed assignment geometry, TRAF-75 owns
  directional precision selection, TRAF-26 supplies complete peer workloads,
  and COMP-89 separately owns the donor NCCL extrapolation.
- TRAF-78 (Precision; P1; L): replace the MiniMax full-population packet
  arm's deterministic balanced assignment surrogate with independently
  observed per-rank expert assignments. The corrected surrogate routes whole
  token-expert assignments only to destinations they reach and preserves its
  exact off path, but it does not identify a deployed engine's routing
  distribution. Freeze at least two routing concentrations and two
  expert-parallel widths before capture. Acceptance reports distinct
  destinations per source, cross-node senders per receiver, phase makespan and
  the resulting TTFT or TPOT change against held-out multi-node evidence.
  TRAF-26 owns supplying complete peer workloads; this entry owns the active
  surrogate's routing-geometry precision.
- TRAF-75 (Precision; P1; M): propagate separate expert-dispatch and
  expert-combine element widths from supported framework configuration through
  the execution graph and traffic lowering. The traffic seam accepts explicit
  directional precision, while callers that omit it retain the exact symmetric
  model-dtype baseline. Acceptance covers FP8 dispatch with ordinary BF16
  combine, an explicitly enabled low-precision combine mode, and the symmetric
  bypass, then demonstrates the expected byte and TTFT or TPOT changes through
  the supported metric chain. This entry owns combine-precision selection;
  TRAF-78 owns destination geometry and TRAF-77 owns transport calibration.
- TRAF-76 (Precision; P1; L): complete H200 intra-node collective precision
  beyond the selectable aggregate completion authority. The
  [completion publication](../../examples/collective_floor_calibration_v1/COMPLETION_RESULTS.md)
  records a post-specified regression on an adaptively reused 63-cell
  evaluation set with training-cell-only numeric evaluation. The completed
  surface moves that regression from 51 of 63 to 63 of 63 inside the larger of
  10 percent or two GPU cycles, with 2.5010 percent median, 8.6958 percent p95
  and 9.9262 percent maximum relative error. It is not an untouched holdout
  qualification because the model form, trough preservation, reduce-scatter
  floor, transition coordinates and rank-8 branches were selected after this
  evaluation set had been observed. At the matched 196,608-byte D8 coordinate
  it predicts 1.819675065 ms, quotient 0.946736591 inside the unchanged
  `[0.90, 1.10]` Family D8 band, which completes Leg B without tuning its band.
  It charges one whole completion and exposes zero serialization service. The
  original floor-plus-slope authority remains byte-identical for explicit
  donor transfers, all 16 published MiniMax legacy queries reproduce exactly,
  and the pre-wave timestamps, byte counts, completion order, backend
  invocation order and random state remain exact. Those reproductions do not
  establish successor validity: the existing MiniMax EP 8 PASS holds only
  under the legacy authority pin, while successor binding yields quotient
  0.946736591 against the MiniMax freeze's packet-over-external requirement of
  at least 1.0 and refutes it. MiniMax Family D is 0 of 3 under
  successor binding; EP 32 and EP 128 remain refuted legacy rank-8 donor
  transfers, and the successor rejects their unfitted ranks. TRAF-83 owns the
  explicit rebinding and superseding publication with unchanged bands, so no
  successor is silently defaulted. Attempt 0005 remains visible as a 46-of-63
  refutation of the paired-operation trend ratio. TRAF-76 stays open for the
  packet integration in TRAF-82, the MiniMax rebinding in TRAF-83 and genuinely
  independent H200 validation in TRAF-84. TRAF-77 owns cross-node transport
  calibration, TRAF-78 owns destination geometry, TRAF-75 owns directional
  precision selection, and COMP-89 owns independent calibration of the
  external NCCL extrapolation.
- TRAF-82 (Precision; P1; L): complete the packetized H200 intra-node
  collective. The
  [design boundary](../design/traf82-h200-packet-collective.md) freezes
  zero-fan-in participants 2 and 8 and nonzero-fan-in participants 4 and 8 at
  65,536-byte and 1,048,576-byte payloads. The completion wave did not execute
  or score those cells because the opaque H200 table cannot independently
  identify credit, queue, port, switch or arbitration values, and the current
  tree lacks the prerequisite generation-scoped flit, receiver-owned credit,
  explicit virtual-channel, replay, receive-order, virtual-output-queue and
  two-sided crossbar structure. Land that structure through TRAF-80, identify
  every H200 product parameter from independent evidence, then run the frozen
  PZ and PN families. Acceptance requires the larger of 10 percent or two GPU
  cycles for every matched phase completion, before and after errors, exact
  byte conservation, one timing authority, zero aggregate charge when packet
  timing is enabled, deterministic replay and a byte-identical identity off
  path. Until then the nonzero-fan-in packet mechanism remains an explicitly
  transferred local component and TRAF-76 stays open. This entry's packet
  scope is unchanged; landing it alone does not close TRAF-76 while TRAF-83
  and TRAF-84 remain open.
- TRAF-83 (Precision; P1; M): explicitly rebind and supersede the MiniMax
  expert-parallel scaling publication under the completed aggregate authority,
  without changing any frozen band. Preserve the legacy publication and its
  authority pin, then predeclare the successor binding and republish Family D.
  The exact EP 8 coordinate uses the same operations, rank, 196,608 operation
  bytes, 65 repeats and 1.922050 ms external arm; its successor quotient is
  0.946736591, below the frozen packet-over-external floor of 1.0, so the
  legacy EP 8 PASS reverses to a refutation. EP 32 and EP 128 remain refuted
  rank-8 donor transfers under the legacy authority, while the successor
  rejects their unfitted ranks. Acceptance makes the authority choice explicit
  at every consumer, preserves the legacy off path byte for byte, reports
  successor Family D as 0 of 3 unless new predeclared evidence changes a cell,
  and publishes the superseding MiniMax result without silently changing a
  default.
- TRAF-84 (Precision; P1; L): independently validate the completed H200
  aggregate surface on cells never used in any fitting, model-form selection,
  branch selection, threshold selection or prior evaluation decision. Freeze
  the independent matrix, physical floor and ceiling checks, and the unchanged
  larger of 10 percent or two GPU cycles band before measuring or importing
  any cell. Publish every cell and fatal guard, with a violated fatal guard
  voiding the run. The existing 63-of-63 result remains a post-specified
  regression on an adaptively reused 63-cell evaluation set with
  training-cell-only numeric evaluation and never enters the independent
  denominator. Acceptance requires every newly pre-specified independent cell
  to pass and preserves the current surface and legacy donor authority as
  explicit, separately selectable paths.
  Independent A100 evidence now tests the shape premise behind that wider-rank
  transfer. A rank-4 donor misses untouched rank-8 all-gather and
  reduce-scatter curves by 62.889 and 61.111 percent median respectively, and
  both floor-fraction decompositions fail their frozen bands at the first
  locality crossing, a boundary that on the measured cluster combines node
  locality with a transport-stack change (NVLink inside the node, NCCL Socket
  over Slingshot with GPU Direct RDMA disabled between nodes). TRAF-81 remains
  formally blocked on rank 16, so it does not supply an error-growth result,
  but the available rank-8 evidence removes unqualified shape confidence from
  the EP 32 and EP 128 donor transfers. It neither transfers an A100 absolute
  band to H200 nor changes the exact H200 rank-8 measurements.
- TRAF-20 (Precision; P2; M): qualify the delivered `loggopsim-ideal`
  fast level for schedule-shape studies that do not need per-flow transport
  behavior. The
  [frontier ladder](../../examples/frontier_ladder_v1/RESULTS.md) measures the
  modeled-error half against pinned packet observations through M-1, M-2 and
  M-3: batch-32 packet over ideal is 1.015637 for serialized traffic, 8.110405
  for incast and 1.015682 for the isolated incast control. It executes no
  packet reference and therefore measures no packet wall clock. The level now
  computes receiver fan-in from each rendered GOAL and refuses overlapping
  flows from multiple sources by default, naming the unmodeled receiver
  per-byte gap, the about 8x frozen-cell optimism and the ladder evidence. An
  explicit `acknowledge_fan_in=True` permits a deliberate run and stamps both
  fan-in and acknowledgment in provenance; fan-in-free runs preserve their
  accepted GOAL bytes. The
  [acceptance study](../../examples/loggopsim_acceptance_v1/RESULTS.md)
  satisfies the live packet anchoring and enforcement clauses: all twelve
  packet completions reproduce the pinned record exactly, and default refusal,
  explicit acknowledgment with stamp, and the byte-identical clean path all
  pass. Its full verdict is REFUTED because the measured packet-over-ideal
  wall-clock gain is 36.579528x against the frozen 50x floor, even though the
  0.029767114-second ideal total meets its one-second ceiling. All four fatal
  guards hold. TRAF-20 therefore remains open solely on the registered speed
  qualification; neither the passing subfamilies nor a future repetition may
  be presented as closure unless every frozen scored predicate passes.
- TRAF-51 (Precision; P1; L): complete the Slingshot fabric calibration
  beyond the steady-state partial. The wave-19 comparison
  ([merlin ss fabric calibration](../../examples/merlin_ss_fabric_calibration_v1/RESULTS.md),
  freeze `7f7550e`) validated, on a declared single-switch Merlin
  instance with per-parameter provenance, the instance's exact
  per-packet serialization arithmetic (three hand-derived oracles) and
  a frozen composition rule that separates the measured endpoint
  host-stack floor (1256.4 and 1858.2 us per 8 MiB chunk on the two
  A100 pairs, 78.7 to 84.5 percent of chunk life) from the 340.4 us
  fabric term: all 11 simulation rows passed and all 10 conditional
  consistency rows confirmed, with every captured steady quantity
  within 7.3 percent of its solo-anchor prediction. The operative claim
  is composition-rule validity given a non-bottleneck fabric: the
  captured loads (each stack under a fifth of a port) cannot
  discriminate between fabric models, and the conditional rows are
  simulation-insensitive by the disclosed cancellation, so no
  fabric-model discrimination is claimed. Addressed without scoring,
  under the freeze's entailment rule: the registered 1.71x aggregate
  and 0.9795/0.991 Jain clauses are arithmetic functions of the scored
  per-flow rows and were reported as explicitly-unscored derived rows
  (composed 1.77 and 0.983 against measured 1.72, 0.979 and 0.991); the
  mixed pair's 2.77x clause took the frozen out-of-scope branch with
  its endpoint floors handed to TRAF-53. What remains before the
  registered clause is fully met, after the wave-21 load-bearing
  recalibration
  ([merlin ss fabric loadbearing](../../examples/merlin_ss_fabric_loadbearing_v1/RESULTS.md),
  freeze `179cdc9`) closed the load-bearing gap for the captured
  shared-egress family: on the same declared instance, with the
  pinned load harness's think-time seam carrying the per-stack
  rate-derived endpoint floors, all 8 scored rows passed with no
  guard fired, the x4 aggregate landed at 0.958 of measured inside
  the frozen [0.90, 1.001] band with the entire residual being
  simulated sharing waits (a coarse band by construction: it
  distinguishes the sharing-mechanism class, tolerating up to roughly
  2.5 times the observed wait), composed-level discrimination between two
  buffer configurations was demonstrated (byte-identical at
  capture-shaped load, opposite registered verdicts under the
  composed x4 cell, banded saturating separations), and the
  p50-static floor was refuted for skewed shared-port families (the
  registered 12.7 percent overshoot, handed to TRAF-53). What remains
  before the registered clause is fully met: reproducing the
  119-second simultaneous-start convergence transient (impossible
  under a static endpoint floor; needs the TRAF-53 endpoint-dynamics
  term), the tranche-2 families (their captures are TRAF-52; the
  recalibration's frozen late-arrival path scores any shared-egress
  group among them with no code change), any multi-switch
  adaptive-routing claim (structurally unreachable at the
  single-switch shape that five-node discovery determines), the true
  source-shared x4 mapping (inexpressible until HTSIM-32 and HTSIM-33
  land; the destination-shared abstraction is declared with its cost
  stated), and any statement about which fabric configuration Merlin
  physically has, which needs fabric-side measurements the capture
  does not contain. Acceptance: the remaining behaviors within
  tolerances stated in a freeze chain against byte-locked captures,
  with the endpoint floor an explicit separate term and no validated
  steady quantity regressing outside its accepted residual.
- TRAF-53 (Precision; P2; L): model the endpoint host-stack dynamics
  the TRAF-51 composition holds static. The calibration study separated
  a per-pair static endpoint floor (1256.4 and 1858.2 us per chunk on
  the A100 pairs; 2164.5 and 6515.0 us on the mixed pairs, derived and
  unscored there) and showed that floor plus the calibrated fabric
  reproduces every captured steady quantity within 7.3 percent; what no
  static floor can produce is the measured dynamics: the 119-second
  simultaneous-start incast convergence against 0-second staggered-join
  settling on the same pairs, the source-identity rate asymmetry
  (gpu105 sustains 4.89 to 4.99 GB/s where gpu103 sustains 3.66 to
  4.02), burst-versus-sustained differences of tens of percent with
  pair-dependent sign, and the mixed pair's 2.775x direction asymmetry.
  The wave-21 recalibration added a measured refutation of the p50
  form of the static floor on the shared-port x4 family: p50-derived
  think times overshoot the measured aggregate by the registered 12.7
  percent because the captured cadences are right-skewed (mean over
  median 1.082 to 1.248 per stack), so no single static think time
  reproduces both the mean and the median; an endpoint-process model
  with genuine cadence dispersion is what closes that gap.
  The identifying observables are in the byte-locked capture dataset;
  the surrogate being replaced is the static floor. Acceptance: an
  endpoint-process model that reproduces the i2 convergence transient
  and the staggered-join contrast within a frozen band while every
  calibrated steady quantity stays inside its accepted residual.
- TRAF-52 (Precision; P2; M): complete the GH200-to-GH200 flow-capture
  family, the last family the wave-18 windows could not run. Every frozen
  A100 cell is now captured, scored against the unchanged bands and
  folded into the byte-locked dataset (x4 first, then i3, i4 and j3 in
  the 2026-08-18 tranche-2 landing; the full 18-relation denominator is
  evaluated at 16 pass and 2 honest failures, E-J-2 and E-J-3, classified
  in the study's RESULTS as a specification error of the freeze's
  shared-bottleneck and stationarity premises, with the bands left
  unwidened). The GH cells were declared but never frozen, because their
  jitter ladder (job 195700, still pending) cannot run before the
  psicourse02 reservation lifts and the chunk size is not frozen before
  the ladder is measured; they arrive only under a freeze-2
  expectations-only commit with their own guard list and scored
  denominator. Acceptance: the GH family captured under freeze 2 or its
  impossibility documented with scheduler evidence, scored against its
  own frozen bands, and folded into the published dataset through a
  packaging commit that preserves every existing byte lock.
- TRAF-48 (Precision; P1; L): capture a cross-node collective on an RDMA fabric
  with GPUDirect RDMA active, which is the fabric the shipped cross-node
  envelope actually targets. The
  [cross-node collective envelope](../../examples/crossnode_collective_envelope_v1/RESULTS.md)
  supplies the repository's first measured cross-node numbers, but the only
  inter-node path this project can reach is four Cray Cassini Slingshot 200Gb
  ports per node carrying NCCL's kernel socket transport with GDR disabled,
  because no NCCL network plugin is installed and the nodes expose no
  InfiniBand device. That measurement anchors a port ceiling and a stack
  efficiency on a named machine; it does not anchor the 400 Gbit/s RDMA path
  the deployment reference configuration assumes, and the transport is a
  first-order term rather than a detail. The identifying observable is the
  completion time of a small-payload ALL-REDUCE and pairwise ALL-TO-ALLV over
  an RDMA fabric with the GDR status recorded from `NCCL_DEBUG=INFO`, at the
  widths a cluster can express with one rank per node. Report the socket
  profile's before error at every width the RDMA capture reaches, and preserve
  the socket profile as an explicitly transport-labeled selection rather than
  replacing it, since a kernel-socket deployment is a real configuration and
  not only a limitation.
- TRAF-32 (Precision; P1; M): widen or refit the endpoint-byte envelope the
  calibrated profiles accept, which is currently too narrow for the mission
  workload. Registered late and by a different change from the one that
  declared it: the composed-step-budget freeze reserved this ID in advance for
  a collective floor defect and its study then reported the ID unused, so it
  was never entered in any module registry. The declared trigger, that freeze's
  G8 endpoint-envelope guard firing, never fired; this registration rests
  instead on fresh evidence from the fixed-cost envelope study, which found the
  width-8 ceiling reached exactly by a 32-token prefill on the reference
  geometry. Of the two clauses that freeze declared, only the envelope-width
  clause is live. The double-charge clause is closed by construction, because
  `StepCollectiveTimingOutcome` already raises when a semantic collective
  receives more than one base latency, with both the pass branch and the raise
  branch driven by tests. The live clause: the width-8 envelope
  ceiling of 458,752 bytes is reached by a 34-token prefill step and reached
  exactly by a 32-token prefill on the eight-wide expert-parallel reference
  geometry, and the width-4 ceiling of 393,216 bytes is lower still, so a
  realistic prefill is rejected at planning time rather than priced. Capture or
  derive collective completion across the endpoint loads a mission prefill
  actually produces, extend the profile's validity interval to cover them with
  stated held-out error, and keep the explicit rejection for loads outside the
  extended interval. Acceptance must show a mission-scale prefill step
  completing under an active arm and must preserve every accepted decode-step
  timestamp exactly.
- TRAF-36 (Precision; P1; L): measure the real cross-node per-collective fixed
  cost and replace `b200-nccl-2.27-cross-node-provisional-v1`, which is
  transferred from an intra-node capture and carries no cross-node
  measurement. Identifying observable: the completion time of a small-payload
  collective, at participant widths 2, 4 and 8 with one rank per node, over a
  400 Gbit/s fabric, as a function of payload across the profile's 8-byte to
  256-KiB interval, so the width-indexed intercept separates from the endpoint
  serializer by the same regression the intra-node profile used. Capture both
  ring ALL-REDUCE and pairwise ALL-TO-ALLV, reserve at least one payload and
  one width as holdout, and require held-out completion error no larger than
  10 percent or 2 microseconds, whichever is larger. The same capture has to
  settle the algorithm question the current transfer papers over: neither the
  intra-node source nor this profile records which algorithm NCCL selected, the
  `2(W-1)` decomposition is this repository's own expansion model, and NCCL on
  an eight-GPU NVSwitch node ordinarily selects NVLS or a tree for a small
  ALL-REDUCE, which at the same per-step delta would move the width-8 point
  estimate from 49.49 to 38.43 us. Record the selected algorithm per width,
  either from `NCCL_DEBUG=INFO` topology output or by pinning `NCCL_ALGO`, and
  refit the step count to match it. Report the provisional-transferred
  profile's before error at every measured width, relabel the refitted profile
  `calibrated` only if the holdouts pass, and preserve the `off` arm and the
  existing intra-node arm exactly.

  Partial evidence now exists and the task stays open. The
  [cross-node collective envelope](../../examples/crossnode_collective_envelope_v1/RESULTS.md)
  measured the width-2 one-rank-per-node cell on two A100 nodes and found the
  transferred profile a factor 2.976 too small there, with the measurement above
  even that profile's own declared upper band edge of 17.488 us. It also settles
  the algorithm question at that width: NCCL's log names `RING`, and twice the
  measured one-way point-to-point time reproduces the measured all-reduce floor
  to 6.8 percent, so the `2(W-1)` decomposition is confirmed at width 2 on the
  algorithm NCCL actually chose. Three clauses remain unsatisfiable on the
  hardware this project can reach and the reason is recorded so the dead end is
  not re-attempted: the only inter-node path is NCCL's kernel socket transport
  over Cray Cassini Slingshot ports with GPUDirect RDMA disabled rather than a
  400 Gbit/s RDMA fabric, and width 8 at one rank per node needs eight nodes
  on a five-node cluster. The mixed-ring width-8 cell (two nodes by four
  GPUs) did land late, on 2026-08-18, and its frozen relations all passed;
  the one-rank-per-node width-8 clause stays physically unsatisfiable here.
  TRAF-48 owns the RDMA capture and TRAF-50 the profile fold-in of the
  measured width.
- TRAF-50 (Precision; P2; M): fold the measured width-8 mixed ring into the
  first-party cross-node profile behind a fresh freeze. The
  [cross-node collective envelope](../../examples/crossnode_collective_envelope_v1/RESULTS.md)
  measured width 2 at publication; its queued width-8 cell (two nodes by
  four GPUs) landed 2026-08-18 and was scored by the study's committed
  reproduction path, so every width-scaling relation its freeze wrote now
  passes (8 B floor 50.790 us, width-8 over width-2 bus bandwidth 3.194,
  the transferred width-8 intercept within 2.6 percent of measurement) and
  the capture half of this task is done. A ring with some NVLink hops and some
  fabric hops is the shape a real two-node tensor-parallel deployment has, and
  it is the shape the composed SGLang deployment study prices with a transferred
  constant, so it is the width that matters most. The identifying observable is
  the 8-byte ALL-REDUCE and ALL-TO-ALLV floor and the asymptotic bus bandwidth
  at four ranks per node on two nodes, with the per-call protocol read from
  `NCCL_DEBUG=INFO` as that study did. Acceptance adds the measured widths to
  `a100-nccl-2.31-cross-node-socket-v1` behind a fresh freeze and preserves the
  width-2 row exactly.
- TRAF-37 (Precision; P1; M): stop charging one propagation delay twice under
  an active arm. The intercept is added outside `max(local_service,
  fabric_service)` while the fabric service already contains one propagation
  delay, so the realized per-collective fixed cost is `intercept +
  propagation_reference_ps`, which over-counts a source capture that was
  itself a complete fixed cost by up to 2.000 us per collective: 6.6 percent
  at width 8, 12.7 percent at width 4 and 18.7 percent at width 2 of the
  intra-node arm. Identify how much of each source intercept is transport that
  the backend already prices, subtract only that part, and require the
  corrected arm to reproduce the source capture's held-out completions at least
  as well as the current one. The uncorrected charge stays available as the
  explicit accepted baseline so the composed-step-budget and collective-floor
  results remain reproducible byte for byte.
- TRAF-39 (Precision; P1; M): identify the per-collective fixed cost of a
  pairwise ALL-TO-ALLV separately from the ring ALL-REDUCE the profiles are
  fitted on. Every shipped profile applies an ALL-REDUCE intercept unchanged to
  the MoE dispatch and combine all-to-alls, which are the only collectives the
  expert-parallel reference geometry emits, so the operation-shape transfer is
  a first-class contributor to the width of the published bracket rather than a
  secondary caveat. Capture both operations at the same widths, payloads and
  stack, report the ratio between their intercepts per width, and either add a
  per-operation table or state with evidence that one table serves both.
  Acceptance must move the published envelope width and must preserve the
  existing arms as explicit selections.
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
  exact `legacy` and all-remote identity paths. The
  [A100 hardware envelope](../../examples/a100_hardware_envelope_v1/RESULTS.md)
  supplies the analogous A100 capture and a warning that applies to the B200
  fit as much as to its own: fitting `t = alpha + S / beta` over an 8-byte to
  256-KiB window on a 4-GPU A100 `NV4` mesh returns 68.10 GB/s, within 3
  percent of the B200 profile's 70,027,079,100 bytes/s, while that window's
  own achieved algorithm bandwidth at 256 KiB is only 14.46 GB/s and the
  asymptotic value at 1 GiB is 141.93 GB/s. A slope fitted inside the
  latency-dominated regime is not a fabric bandwidth, so the B200 refit must
  extend past the payload where bus bandwidth flattens rather than only adding
  point-to-point samples inside the existing window.
- TRAF-43 (Precision; P1; M): replace the single-slope collective serializer
  with a regime-aware form. The shipped model charges one
  `bandwidth_bytes_per_second` at every payload, and the
  [A100 hardware envelope](../../examples/a100_hardware_envelope_v1/RESULTS.md)
  measured how wrong that is on real intra-node NVLink. Anchoring the intercept
  at the measured 8-byte floor and the slope at the 1-GiB algorithm bandwidth
  reproduces both anchors exactly and is optimistic everywhere between them, by
  50.8 percent at width 2 and 1 MiB and by 45.8 percent at width 4 and 2 MiB.
  The cause is identified: bus bandwidth is still climbing across that window,
  passing half its asymptote only at 2.45 MiB at width 2 and 8.24 MiB at width
  4, and reaching 90 percent of it only at 128 MiB. A high fit R-squared does
  not detect this, because the largest payloads dominate an ordinary least
  squares fit; the A100 fit scored 0.9997 while placing an 87.36 microsecond
  intercept where the measured floor is 9.11 microseconds. The identifying
  observable is measured completion time across the full payload decade range
  at a fixed width. Acceptance requires a form whose worst signed error over
  the measured sweep is at most 15 percent at both widths, an unchanged result
  at the two anchors, and a bypass that preserves the accepted single-slope
  artifacts exactly. This is now confirmed on two architectures. The
  [GH200 hardware envelope](../../examples/gh200_hardware_envelope_v1/RESULTS.md)
  froze the reproduction as a pre-run prediction with an explicit falsifier and
  both held: on a 4-GPU GH200 `NV6` mesh the same two-anchor model is
  optimistic at every payload between its anchors, worst at -48.1 percent at
  1 MiB and width 2 against the A100's -50.8 percent at the same point, and the
  same wide-window fit places a 56.44 microsecond intercept where the measured
  floor is 6.22 microseconds, a factor 9.07 against the A100's 9.59, at an
  R-squared of 0.99968. A defect that survives a change of link generation,
  link count, channel count and host architecture is a property of the model
  form, so this task needs no further hardware evidence to justify it.
  The first candidate replacement is landed and refuted. The
  [collective regime curve](../../examples/collective_regime_curve_v1/RESULTS.md)
  added `CollectiveBandwidthCurve`, a payload-indexed serialization bandwidth
  with geometric interpolation between measured anchors, and scored `16 of 20`
  against a frozen five-anchor rule and a 16-payload held-out split. It clears
  the bar at width 4 on both machines, at -11.86 and -9.95 percent, and misses
  it at width 2 on both, at -27.76 and -27.35 percent, in both cases at exactly
  1 MiB. The cause is identified and is itself a finding: measured
  serialization bandwidth is **not monotone in payload**. It dips 26 percent on
  the A100 and 22 percent on the GH200 at 1 MiB at width 2, and about 7 percent
  at 2 MiB at width 4, so any interpolation between anchors that straddle the
  dip predicts a faster collective than the hardware delivers. The dip has the
  shape of a protocol transition but this study did not instrument NCCL's
  selection, so that mechanism is a hypothesis. The next candidate must anchor
  at the transition rather than on a log-spaced grid, keep a held-out payload
  inside the dip so the check still measures generalization, and be frozen
  before its error is computed. The mechanism stays landed and inert: no
  shipped profile carries a curve, an uncurved width charges exactly the flat
  slope it always did, and no reported TTFT or TPOT moves.
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
- TRAF-56 (Precision; P1; M): calibrate the collective registration cost and
  the model choices around it. `DECLARED_NCCL_CHANNEL_REGISTRATION_COST` charges
  20,000,000 ps per `(communicator, generation, channel, buffer)` identity, and
  the ABI behind it establishes only that a registration entry point exists at
  the plugin seam and that one seam serves NCCL and RCCL. The surrogate being
  replaced is therefore larger than the constant: the once-per-identity
  charging rule, the per-buffer identity scope, the channel factor and the
  three-event re-registration set are declared model choices with no
  measurement behind them, and a calibration that fits the duration while
  leaving them assumed has calibrated the smaller half. `calibrated_cost_ps`
  already fails closed so no consumer can read the constant as measured. The
  identifying observable is the wall time between the plugin entering `regMr`
  for a buffer and the first collective on that buffer becoming eligible,
  captured at more than one buffer size and more than one channel count so the
  size and channel dependence the declared constant ignores is either measured
  or refuted, and repeated across a buffer reuse and a communicator rebuild so
  the once-per-identity rule is tested rather than assumed. Acceptance: a
  `calibrated` cost with its measurement named, held-out error reported against
  the capture, each surviving model choice restated as measured or explicitly
  retained as declared, the declared constant kept selectable for reproducing
  accepted runs, and a before-and-after TTFT delta on the
  `nccl_registration_v1` live cell. P1 because
  `examples/nccl_registration_v1` opts the registration on, which makes its
  calibration active-path precision.
- TRAF-65 (Precision; P1; L): identify and land the A100 NVLink3 packet
  service that replaces the active flat endpoint serializer. The
  [A100 hardware envelope](../../examples/a100_hardware_envelope_v1/RESULTS.md)
  measured 94.00 to 94.07 GB/s on every four-link ordered pair and
  281.65 GB/s under three-way fan-out, but those rates identify only an
  envelope. They do not identify packet boundaries, four-link bonding,
  request and response direction, finite credits, FIFO placement or
  head-of-line blocking. The current model therefore cannot say why the
  measured pair rate is within 0.13 percent of the candidate
  `100 * 256 / (256 + 16) = 94.1176 GB/s`, and it cannot distinguish that
  packet-overhead explanation from copy-engine coalescing. Run a dedicated
  study on one qualified four-A100-SXM4-80GB `NV4` Merlin node. Freeze the
  expectations-only commit before the harness and first timed run. Use a
  persistent SM peer-write producer, a dependent SM peer-read producer and a
  copy-engine reference, with NCCL send and receive as a protocol validation
  rather than as the packet-format authority. Each named parameter sweep below
  is one case. The frozen catalog has five corners with exactly 16 cases each:

  - **Packetization, `CORNER_NVPKT_001` through `CORNER_NVPKT_016`:** every
    payload byte from 1 through 512; candidate-boundary neighbours through
    4096; `256k+r` residuals; destination alignment; source alignment; access
    width; active warp lanes; lane-mask shape; address stride; fixed total
    bytes with varied message size; fixed message size with varied count;
    address reuse versus separation; peer-write length and alignment; peer-read
    length and alignment; copy-engine versus SM producers; and a seeded blind
    holdout of lengths, masks and alignments.
    Stable names in case order: `CORNER_NVPKT_001_payload_bytes`,
    `CORNER_NVPKT_002_candidate_boundaries`,
    `CORNER_NVPKT_003_256b_residuals`,
    `CORNER_NVPKT_004_destination_alignment`,
    `CORNER_NVPKT_005_source_alignment`, `CORNER_NVPKT_006_access_width`,
    `CORNER_NVPKT_007_active_warp_lanes`,
    `CORNER_NVPKT_008_lane_mask_shape`, `CORNER_NVPKT_009_address_stride`,
    `CORNER_NVPKT_010_fixed_total_bytes`,
    `CORNER_NVPKT_011_fixed_message_size`,
    `CORNER_NVPKT_012_address_reuse`, `CORNER_NVPKT_013_peer_write`,
    `CORNER_NVPKT_014_peer_read`, `CORNER_NVPKT_015_producer_comparison` and
    `CORNER_NVPKT_016_blind_holdout`.
  - **Bond and wire, `CORNER_NVBOND_017` through `CORNER_NVBOND_032`:** all
    ordered pairs; per-link balance; stream count; producer concurrency;
    bandwidth ramp; offered rate; burst length; source fan-out; destination
    fan-in envelope; symmetric bidirectional traffic; asymmetric
    bidirectional traffic; disjoint unidirectional pairs; disjoint
    bidirectional pairs; all ordered mesh flows; cold versus warm startup; and
    node-and-time repeatability.
    Stable names in case order: `CORNER_NVBOND_017_ordered_pair_matrix`,
    `CORNER_NVBOND_018_per_link_balance`, `CORNER_NVBOND_019_stream_count`,
    `CORNER_NVBOND_020_producer_concurrency`,
    `CORNER_NVBOND_021_bandwidth_ramp`, `CORNER_NVBOND_022_offered_rate`,
    `CORNER_NVBOND_023_burst_length`, `CORNER_NVBOND_024_source_fanout`,
    `CORNER_NVBOND_025_destination_fanin`,
    `CORNER_NVBOND_026_symmetric_bidirectional`,
    `CORNER_NVBOND_027_asymmetric_bidirectional`,
    `CORNER_NVBOND_028_disjoint_unidirectional`,
    `CORNER_NVBOND_029_disjoint_bidirectional`,
    `CORNER_NVBOND_030_full_mesh`, `CORNER_NVBOND_031_startup_state` and
    `CORNER_NVBOND_032_node_time_repeatability`.
  - **Incast and destination FIFO, `CORNER_NVINC_033` through
    `CORNER_NVINC_048`:** one-source baseline; two-source simultaneous incast;
    three-source simultaneous incast; fixed aggregate rate across fan-in;
    per-source offered rate; start skew; step-wise join and leave; burst depth;
    equal message sizes; unequal sizes; one elephant with two mice; two
    elephants with one mouse; push incast into distinct buffers; pull gather
    from three buffers; a hot destination region with a dispersed-address
    control; and a long fairness, tail and drain soak. A 3-to-1 `NV4` incast
    uses three distinct four-link bundles, so any shared limit is attributed
    first to destination ingress, its merge FIFO or memory acceptance, never
    to one contended wire.
    Stable names in case order: `CORNER_NVINC_033_one_source`,
    `CORNER_NVINC_034_two_source`, `CORNER_NVINC_035_three_source`,
    `CORNER_NVINC_036_fixed_aggregate_rate`,
    `CORNER_NVINC_037_per_source_rate`, `CORNER_NVINC_038_start_skew`,
    `CORNER_NVINC_039_join_leave`, `CORNER_NVINC_040_burst_depth`,
    `CORNER_NVINC_041_equal_message_size`,
    `CORNER_NVINC_042_unequal_message_size`,
    `CORNER_NVINC_043_one_elephant_two_mice`,
    `CORNER_NVINC_044_two_elephants_one_mouse`,
    `CORNER_NVINC_045_push_distinct_buffers`,
    `CORNER_NVINC_046_pull_gather`, `CORNER_NVINC_047_hot_destination` and
    `CORNER_NVINC_048_long_soak`.
  - **Credit depletion and return, `CORNER_NVCRD_049` through
    `CORNER_NVCRD_064`:** dependent round trip; outstanding-write sweeps at
    16, 32, 64, 128 and 256 bytes; outstanding reads; outstanding atomics;
    burst length at a fixed window; inter-message gap; inter-burst recovery
    gap; duty cycle; a predeclared adaptive zoom around the first 95-percent
    throughput knee; two streams on one pair; one source to two peers; and
    opposite directions on one pair.
    Stable names in case order: `CORNER_NVCRD_049_dependent_round_trip`,
    `CORNER_NVCRD_050_outstanding_write_16b`,
    `CORNER_NVCRD_051_outstanding_write_32b`,
    `CORNER_NVCRD_052_outstanding_write_64b`,
    `CORNER_NVCRD_053_outstanding_write_128b`,
    `CORNER_NVCRD_054_outstanding_write_256b`,
    `CORNER_NVCRD_055_outstanding_read`,
    `CORNER_NVCRD_056_outstanding_atomic`,
    `CORNER_NVCRD_057_burst_length`, `CORNER_NVCRD_058_inter_message_gap`,
    `CORNER_NVCRD_059_inter_burst_gap`, `CORNER_NVCRD_060_duty_cycle`,
    `CORNER_NVCRD_061_adaptive_knee_zoom`,
    `CORNER_NVCRD_062_two_streams_one_pair`,
    `CORNER_NVCRD_063_one_source_two_peers` and
    `CORNER_NVCRD_064_opposite_directions`.
  - **FIFO partition and head-of-line behavior, `CORNER_NVHOL_065` through
    `CORNER_NVHOL_080`:** small behind large; large behind small; separate
    streams on one pair; alternating sizes; a seeded bimodal mix; a latency
    flow under same-pair bulk; a latency flow under other-peer bulk from the
    same source; a latency flow under remote incast; write with write; read
    with read; read and write whose payloads use the same direction; read and
    write whose payloads use opposite directions; distinct memory regions; a
    shared-cache-line hotspot with a dispersed control; post-burst drain; and
    a seeded blind mixed soak.
    Stable names in case order: `CORNER_NVHOL_065_small_behind_large`,
    `CORNER_NVHOL_066_large_behind_small`,
    `CORNER_NVHOL_067_separate_streams`,
    `CORNER_NVHOL_068_alternating_sizes`, `CORNER_NVHOL_069_bimodal_mix`,
    `CORNER_NVHOL_070_same_pair_bulk`, `CORNER_NVHOL_071_other_peer_bulk`,
    `CORNER_NVHOL_072_remote_incast`, `CORNER_NVHOL_073_write_write`,
    `CORNER_NVHOL_074_read_read`,
    `CORNER_NVHOL_075_same_direction_read_write`,
    `CORNER_NVHOL_076_opposite_direction_read_write`,
    `CORNER_NVHOL_077_distinct_regions`,
    `CORNER_NVHOL_078_shared_cache_line`,
    `CORNER_NVHOL_079_post_burst_drain` and
    `CORNER_NVHOL_080_blind_mixed_soak`.

  Every case runs both isolated and in ordered continuous `corner_frame` mode;
  the complete catalog also runs once as `all_corners_frame`. Record logical
  bytes, per-link and per-direction raw and data counter deltas, batch time,
  issuer-local latency, completion and drain time, per-source rate, checksum,
  topology, clocks, competing processes and CRC, replay and recovery deltas.
  Repeat small operations until counter quantization is negligible, and never
  launch or synchronize once per message. Fatal guards are payload corruption,
  a non-`NV4` or fallback path, raw bytes below data bytes, a non-monotone
  counter, an unexplained nominal replay or error, a rate above 25 GB/s per
  physical link, 100 GB/s per ordered pair or 300 GB/s per GPU direction, and
  a measurement contaminated by throttling or another process. A fired guard
  voids the affected run rather than reducing a pass fraction.

  Fit only what the observations identify: payload and overhead granularity,
  maximum packet payload, four directional serializers and their bond policy,
  an effective credit unit and window with return latency, per-link ingress
  FIFO service and any shared destination merge service. Exact undocumented
  virtual-channel counts, credit counts, buffer depths, bit fields and
  arbitration rules remain unnamed unless independent observables identify
  them. A fitted knee is reported as effective capacity, not promoted to a
  literal hardware register. Acceptance requires exact logical-byte and
  request/response conservation, packet-overhead steps within counter
  resolution, held-out throughput and completion error no larger than
  10 percent or 1 microsecond, whichever is larger, and the first saturation
  knee within one frozen sweep point. The resulting structural model keeps
  packetization, credits, FIFO visits and wire serialization explicit, then
  reaches the supported TRAF-45 and TRAF-54 path so a registered contention
  cell changes TTFT or TPOT in the frozen direction. The analytic path and any
  uncalibrated architecture remain exact selectable bypasses. Capture alone is
  component evidence and does not close this task. The handoff is explicitly
  two-phase rather than a completion cycle: the hardware run publishes a
  versioned `candidate` profile and conformance fixtures while TRAF-65 remains
  open; TRAF-45 and TRAF-54 consume that candidate without waiting for this
  task to close; the live held-out result promotes it to `calibrated` and closes
  TRAF-65. No downstream task depends on TRAF-65 being closed.

  Local-arm progress on 2026-08-26 does not close this entry. The final
  expectations-only commit is `d74b123`, with SHA-256
  `212a7a26f54e444c9b18f1e528bd0d00b5a28e4f9e005b0dc137f477ad642571`.
  The GPU-free arm compiled the three-producer mock harness and completed all
  86 resumable cells: 80 isolated, five ordered `corner_frame` and one
  `all_corners_frame`, totaling 14,035 rows whose manifests verify. Those rows
  carry no measurement claim. The additive htsim candidate composes separate
  TX, switch and RX modules, makes the A100 direct-mesh switch an exact
  pass-through, and returns the caller's analytic result by object identity
  when no candidate is selected. Its comparison uses only the already
  published 94.00 to 94.07 GB/s ordered-pair and 281.65 GB/s fan-out envelope
  rows. The 80-case on-silicon campaign remains this entry's own resumable
  remainder, so reserved IDs TRAF-67 and TRAF-68 are not consumed. Maintenance
  reservation `SD26082026` holds every Merlin GPU node down until
  `2026-08-28T06:30`; the exact digest-pinned staging, pending-index and
  `sbatch` commands are in the
  [TRAF-65 resume record](../../examples/a100_nvlink_packet_v1/RESUME.md).

  The on-silicon remainder executed early after the integrator verified on
  2026-08-27 that reservation `SD26082026` had lifted and the A100 partitions
  were visible in mixed and allocated states. Merlin job `198968` ran the exact
  frozen head on one exclusive `NV4` node with the registered `%1` pacing. All
  86 cells completed on `gpu105`, all 14,035 rows and manifests verify, every
  scheduler task exited `0:0`, no stop record exists, and the post-run pending
  set is empty. The written maintenance date and its verified early
  supersession are both retained in the resume record.

  The literal score is
  [COMPLETE_VOID_86_OF_86](../../examples/a100_nvlink_packet_v1/RESULTS.md),
  not a calibration. The hardware row schema records elapsed time but no
  observed per-row raw/data, link, direction, replay, recovery or error
  deltas. Its checksum is a point-id hash rather than a destination-byte
  comparison, it derives `candidate_packet_count` and `candidate_raw_bytes`
  from the hypothesis, it parses but does not apply access width, lane mask,
  stream count, outstanding window, burst length, gap or offered rate in the
  hardware path, and the copy-engine loop enqueues once per message. Five
  frozen fatal guards are therefore undecidable and the completed timings are
  void for promotion. The visible rate reductions refute the capture
  procedure's three scored bond bands and all 16 incast bands, not the physical
  NVLink constants. Packet overhead versus copy-engine coalescing remains
  unidentifiable; every TX and RX parameter remains declared and unmeasured;
  the A100 pass-through switch stands as a structural direct-mesh invariant,
  not a hardware measurement. The candidate profile stays `candidate` with no
  parameter value changed, and TRAF-65 stays OPEN.

  TRAF-70 replaced that void identification capture without changing the
  TRAF-65 expectations. The corrected expectations-only freeze is SHA-256
  `f0ab026e054873a56614af63ab3a7ae3219dc0b045423808cb41522910fa6da6`.
  Merlin jobs `199957` and `199960` completed all 86 cells and 11,542 rows on
  one qualified exclusive `NV4` node; every accepted task exited `0:0`, the
  pending set is empty, and the producer binary has one verified digest. The
  [corrected score](../../examples/a100_nvlink_packet_v2/RESULTS.md) is
  `COMPLETE_VALID_86_OF_86`. All ten fatal guards are decidable passes, every
  result-row throttle verdict is `CLEAR`, and observations never consume
  candidate-derived packet fields.

  The frozen rules identify an effective TX endpoint egress rate of
  `160795737454` bytes per second and an effective RX ingress rate of
  `207101921876` bytes per second, refuting and replacing the two 300 GB/s
  candidates. They confirm the existing request and response direction,
  `extent_sequence` reassembly and `per_extent` delivery with measured
  rule-specific evidence. The packet size and header, link count and rate,
  bond policy, effective credits, RX buffer and return latency, and queue scope
  remain inconclusive declared candidates. The direct-mesh switch remains
  structural. The pre-score profile is preserved at its original
  `899712c4...e354f` digest; the live profile moved only after the score was
  committed and now exposes parameter-specific evidence. The flow-dynamics
  gate is `OPEN`. TRAF-65 remains open for its separately required live
  held-out integration result.
  This entry was registered as TRAF-73 and renumbered to TRAF-74 by the
  integrator at merge time, because a concurrently dispatched study
  claimed the same identifier; the study's own frozen artifacts keep the
  original string so their digests stay intact.

- TRAF-73 (Precision; P1; M): identify the effective NVLink credit window,
  credit-pool scope and downstream incast arbitration on the qualified NV4
  node. The surrogate being replaced is the pair-keyed credit ledger plus an
  undocumented scheduling choice. Public documents establish credit flow
  control and multiple virtual-channel classes, but not the A100 allocation
  scope or quantum. Until TRAF-80 replaces the fixed-unit and timer structure,
  the unchanged 256-credit, 272-byte candidate stays keyed per link for one
  implicit modeled virtual channel. It does not claim a physical
  virtual-channel count. Release-aware round robin is the declared baseline
  candidate; static interleave and greedy capture are explicit alternatives.
  TRAF-73 measures among those candidates only after the structure is aligned.

  The expectations-only specification is
  [nvlink_credit_arbitration_v1](../../examples/nvlink_credit_arbitration_v1/expectations.md).
  H1 sweeps 31 payload sizes from 4 KiB through 8 MiB on every directed NV4
  pair and zooms around the 262,144-byte payload counterpart of the four-link
  candidate window. A repeated persistent latency break identifies an
  effective window and return delay; no break is inconclusive because the
  declared return is shorter than one link's window serialization. H2 repeats
  the zoom with one, two and three senders into receiver 3. Stable per-sender
  knees with aggregate outstanding bytes proportional to sender count support
  per-link pools; knees divided by sender count with constant aggregate bytes
  support a shared pool. H3 uses 500 ms steady streams at raw offers 100, 60
  and 60 GB/s and rotates the greedy source. Small senders at their offered
  rates with the greedy sender receiving the remainder support fair
  arbitration; a greedy sender at least 95 GB/s with a small sender below
  57 GB/s supports greedy capture; equal 57 to 63 GB/s shares with unused
  receiver service support static non-borrowing interleave.

  Before hardware runs, publish all three simulation arms at degrees 2, 3, 4,
  8 and 16, labeling 4, 8 and 16 as simulated mesh extrapolations with no NV4
  counterpart. Acceptance for the implementation slice is full-suite green,
  exact logical and wire conservation, the expected directional separation,
  the scored candidate profile unchanged, and all 89 files in the merged
  packet, flow and comparison studies byte-identical. Acceptance for closure
  is a nonvoid hardware classification under the frozen rules, including an
  honest inconclusive result with no promotion. Until then all credit numbers
  and arbitration policies remain declared candidates and TRAF-73 stays open.

  The simulation slice is complete: all 15 policy and degree instances pass
  their frozen per-source share bands, all 105 fatal guards pass, the candidate
  profile is unchanged, and the 89-file merged-study preservation lock is
  byte-identical. The first run retained two false aggregate refutations
  because its aggregate quantization bound allowed one packet for a sum over
  16 sources. The post-specified correction multiplies that already frozen
  per-source packet bound by degree; workloads, windows, per-source results and
  expectations remain unchanged. H1, H2 and H3 are registered but not run, so
  no candidate value or policy is promoted and TRAF-73 remains open.

### Completeness

- TRAF-49 (Completeness; P2; M): let a profile that supports only the widths it
  measured join a fixed-cost envelope. `CollectiveFixedCostEnvelope` requires
  both arms to support identical participant counts, and every shipped arm
  supports 2, 4 and 8 because it is derived from one capture that covers all
  three. `a100-nccl-2.31-cross-node-socket-v1` supports only the widths its
  study realized and fails closed elsewhere, which is the correct behavior for a
  measurement and makes it ineligible as an arm, so the first measured
  cross-node profile cannot be bracketed against the transferred one it exists
  to check. Acceptance needs an envelope that brackets on the intersection of
  its arms' supported widths and refuses a width only one arm anchors, with the
  refusal driven by a test, and it must preserve both existing envelopes'
  `bracket_ps` and `realized_bracket_ps` at every width exactly. The off path is
  the current refusal to construct such an envelope at all.
- TRAF-34 (Completeness; P2; M): render mixed dense and routed layer
  schedules. `ModelDims` carries one whole-model mixture geometry, so
  `num_experts > 0` means every layer is routed and both the allreduce-site
  rule and the MoE all-to-all inventory apply uniformly over
  `range(num_layers)`. A model with a dense prefix or a periodic dense layer
  (`first_k_dense_replace`, `num_dense_layers`) cannot be expressed, and the
  SGLang reader currently refuses those sentinel fields outright rather than
  pricing them as routed; SGL-18 owns the reader half. Acceptance needs a
  per-layer routed schedule carried on or beside the dims, a rendered
  inventory giving each dense layer two allreduce sites and no all-to-all,
  and byte-identical whole-model dense and whole-model routed renders as the
  explicit off path.
- TRAF-35 (Completeness; P2; M): represent an expert-parallel group that
  still tensor-shards the expert weights, i.e. `moe_tp` above 1. `ModelDims`
  has no field for that width, so a declared expert-parallel group is taken to
  mean `moe_tp` equal to 1, which is the only shape either adapter produces
  today: the vLLM geometry sets `moe_tp_size` to 1 whenever expert
  parallelism is in use, and the SGLang reader refuses anything but
  TP = EP = MoE-DP = 1. A `moe_tp` above 1 needs a third reduction site, an
  allreduce of the expert output over the `moe_tp` subgroup sitting between
  dispatch and combine, which no current path renders. Acceptance requires the
  subgroup membership on the dims or the group inputs, that rendered subgroup
  allreduce, and byte-identical `moe_tp` equal to 1 and dense renders as the
  off path.
- TRAF-38 (Completeness; P2; M): make the fixed-cost arms selectable outside
  the all-remote fluid path. An arm currently also selects its profile's
  endpoint bandwidth, and the sink refuses any network profile other than
  `rnic-nn-fluid`, so switching arms isolates the fixed cost only when the
  placement produces no NVLink-local bytes. In a mixed or all-local placement
  the `lower` and `upper` arms of `intra-node-fixed-cost-v1` differ from the
  `off` arm by the endpoint rate as well as by the surcharge, and the published
  bracket therefore stops being a bracket on the fixed cost alone. Separate the
  endpoint-rate selection from the arm selection, keep the current coupled
  behavior as the explicit off path so accepted local and mixed-placement
  timestamps are preserved exactly, and demonstrate an arm sweep on a mixed
  placement whose non-arm terms are byte-identical across arms.
- TRAF-40 (Completeness; P2; M): make the all-to-all mode an explicit
  declaration and render the path that is currently refused. The correctness
  half is closed: the vLLM producer now classifies both pinned conditions
  before binding an expert group, `use_all2all_kernels` from
  `model_executor/layers/fused_moe/config.py:1052-1055` and the backend's own
  `output_is_reduced()`, so a `tp=8, ep=8, dp=1` deployment binds no group and
  renders the two allreduces and no all-to-all that
  `model_executor/layers/fused_moe/runner/moe_runner.py:436-465` executes.
  Three residuals remain. The renderers still infer the mode from the presence
  of the group rather than from a declared indicator, so a hand-written caller
  can still assert a reducing combine that its configuration does not have.
  Naive expert parallelism carries no rendered expert traffic, so a naive-EP
  run cannot express per-rank expert ownership to the placement layer at all.
  And a non-reducing all-to-all backend such as the default
  `allgather_reducescatter` under data parallelism moves expert activations
  through an allgather and a reduce-scatter, a traffic shape this repository
  renders nothing for, so binding refuses it outright rather than pricing a
  pairwise all-to-allv the deployment never executes; that refusal is the off
  path acceptance must preserve. The refusal protects the traffic shape and
  not the allreduce count, which is correctly empty under this backend either
  way: combined with data parallelism and `tp > 1` it makes the expert input
  sequence parallel (`config/parallel.py:653-668`) and
  `model_executor/layers/fused_moe/runner/moe_runner.py:459` then skips the
  final all-reduce entirely, while at `tp == 1` that all-reduce is a one-rank
  no-op the site rule renders as none. Acceptance needs the explicit indicator
  on the render inputs, a refusal when a declared group contradicts it, the
  allgather and reduce-scatter rendering with its own byte oracle, and
  byte-identical renders for the declared and omitted-group paths that exist
  today.
- TRAF-41 (Completeness; P1; M): requalify the one rendered surface and amend
  the one prose projection that the TRAF-33 inventory correction made stale.
  The rendered surface is the Granite live cells of the collective plan
  default study, which declare one 8-rank group as both the tensor-parallel
  and the expert-parallel group, so their 709,803,840 ps TTFT, 132,794,880 ps
  TPOT and transport rows were measured with 48 rather than 24 allreduces per
  step; those cells must be rerun. The prose projection is the composed step
  budget study's counterfactual paragraph, which reasons that adding a
  24-layer model's tensor-parallel allreduces would add
  `48 * 30,128,029 = 1,446,145,392` ps and be 74.73 to 75.45 percent of the
  composed step; under a declared reducing all-to-all group the counterfactual
  addition is `24 * 30,128,029 = 723,072,696` ps and that paragraph must be
  amended rather than rerun. That study's measured 1,446,145,392 ps collective
  floor is 48 MoE all-to-alls at `tp_ranks=(0,)`, which this change does not
  touch and a rerun reproduces exactly, so acceptance must never halve the
  measured floor. Keep both studies' dense cells byte-identical.

- TRAF-42 (Completeness; P1; M): stop the collective fixed-cost surface from
  describing itself as if every collective crossed the fabric. One root cause
  with two observable consequences, both first exposed by the intra-node cell
  of
  [examples/sglang_composed_deployment_v1](../../examples/sglang_composed_deployment_v1/RESULTS.md).
  First, `CollectiveLatencyProfile.realized_fixed_cost_ps` adds
  `propagation_reference_ps` unconditionally, so
  `intra-node-fixed-cost-v1` publishes `realized_bracket_ps` of
  `[2,000,000, 32,128,029]` ps at width 8 while an all-intra-node step invokes
  no fabric backend and therefore realizes `[0, 30,128,029]`. That cell
  measures 885 ns per collective on its first prefill step, all of it endpoint
  serialization, so the `off` arm on the local path carries no physical floor
  at all, and the envelope's own claim string, the floor profile's transfer
  text and the interface paragraph of this document each overstate the lower
  edge by exactly one propagation reference. Second, the applied evidence
  class is per arm rather than per operation shape, so a cell that charges the
  same profile at its captured operation and interconnect for some collectives
  and by transfer for others has to publish the conservative single class for
  all of them: that cell charges the intra-node NVLink ALL-REDUCE intercept to
  24 ALL-REDUCEs per step at their captured shape, worth 723,072,696 ps, and
  to 48 pairwise ALL-TO-ALLVs by transfer, worth 1,446,145,392 ps, and the
  split had to be derived by hand from the inventory. Identifying observables:
  the realized fixed cost a locality-bearing cell actually charges, and the
  evidence class attached to each executed collective in the run record.
  Acceptance requires the realized bracket to match what the selected path
  charges, one evidence class per executed collective, and every accepted
  all-remote run record to stay byte-identical. This is the record surface,
  not the measurement: TRAF-39 owns capturing an ALL-TO-ALLV intercept of its
  own, and this task is what lets a record say which collectives a single
  table was and was not fitted on, whether or not that capture ever lands.

- TRAF-44 (Completeness; P2; M): add a selectable A100-scoped intra-node
  collective profile so an A100 study stops borrowing B200 numbers. The only
  calibrated local profile today is `b200-nccl-2.27-local-v1`, fitted from a
  published third-party capture of hardware this project cannot reach. The
  [A100 hardware envelope](../../examples/a100_hardware_envelope_v1/RESULTS.md)
  measured the equivalent first-party numbers on a 4-GPU A100-SXM4-80GB `NV4`
  mesh under NCCL 2.31.2: per-collective latency floors of 9,113,600 ps at
  width 2 and 12,953,600 ps at width 4, and an asymptotic all-reduce algorithm
  bandwidth of 72,774,312,725 and 141,927,693,992 bytes/s. The width-4
  intercept sits close to the B200 profile's 15,745,167 ps, so intercepts
  transfer between NVLink generations far better than slopes do. The new
  profile must carry its own provenance record, declare a validity window that
  states where its bandwidth term holds rather than implying one slope
  everywhere, and refuse widths it did not measure, which here excludes width
  8. This is P2 while no study selects it and becomes P1 when an A100-scoped
  study opts in. Land it after or together with TRAF-43, since adding a second
  single-slope profile would propagate the mid-range error that task exists to
  remove. Acceptance requires the new arm to be an explicit selection whose
  absence preserves every accepted artifact byte for byte.

  A second first-party profile is now also available and belongs in the same
  change. The
  [GH200 hardware envelope](../../examples/gh200_hardware_envelope_v1/RESULTS.md)
  measured latency floors of 6,220,800 ps at width 2 and 8,457,600 ps at width
  4, with asymptotic all-reduce algorithm bandwidths of 115,151,100,868 and
  224,623,611,127 bytes/s, on a 4-GPU GH200 120GB `NV6` mesh. Landing both
  profiles together is what makes the selection meaningful, because the pair
  demonstrates what a single profile cannot: ring efficiency against a GPU's
  own link ceiling is 71.0 percent on Ampere and 74.9 percent on Hopper, 3.9
  percentage points apart, while the ceiling itself moves by exactly 1.5 times,
  from 300 to 450 GB/s. The transferable quantity is much closer to the
  efficiency than to the bandwidth. Use the payload rate of 25 GB/s per NVLink
  link per direction on both generations: `nvidia-smi nvlink -s` reports 25 for
  NVLink3 but the 26.5625 GB/s raw signalling rate for NVLink4, and taking that
  report at face value overstates a Hopper ceiling by 6.25 percent.

- TRAF-26 (Completeness; P2; L): extend the isolated one-engine routed-step
  projection to a full DP times EP group population. Each peer engine must
  carry an explicit captured workload or a reproducible independently sampled
  workload, and its routing must be independently observed or sampled.
  Replaying one engine's routing table on every peer is forbidden because it
  manufactures correlated hot-expert incast. The MiniMax study's explicit
  uniform token count per rank is a controlled symmetric surrogate, not an
  independently routed workload, so it does not close this entry. Acceptance
  compares group bytes, peak egress, incast fan-in, TTFT and TPOT against a
  multi-engine capture, while selecting the isolated mode preserves every
  accepted TRAF-25 byte, timestamp and completion order exactly.

- TRAF-15 (Completeness; P2; M): project arbitrary legal forward, non-monotone
  and general non-contiguous or fan-in DAGs through the step sink. The current
  projector rejects unsupported order classes before writing an artifact.
  Acceptance must preserve that explicit rejection as the off path, avoid
  inventing order between independent operations and retain every supported
  projection byte and timestamp exactly.

- TRAF-45 (Completeness; P1; L): add a packetized intra-node leg behind the
  analytic locality path. `classify_step_locality` charges local segments from a
  per-endpoint byte ledger at one declared flat rate, which is exact arithmetic
  over a surrogate port: there is no packet, no attempt identity, no
  destination-side arrival and no per-link routing, so an NVLink or xGMI segment
  cannot be observed the way a wire segment can. Land the packet path over the
  COMP-34 GPU ports using the BACK-48 port-kind-independent vocabulary, and keep
  the analytic split as a byte-identical off path. The packetized leg must
  charge destination ingress explicitly at the receiving port instead of
  inheriting the analytic `max(egress_bytes, ingress_bytes)` endpoint-load
  surrogate that CORE-41 installed; CORE-48 keeps the cross-node
  destination-ingress serializer, TRAF-65 owns the A100 packet, bond, credit,
  FIFO and wire calibration, and COMP-31 owns the remaining compute-side local
  mechanism and cross-architecture generalization. None is closed here. The
  implementation consumes TRAF-65's versioned `candidate` profile when selected
  and never invents a measured constant from a study case name. The candidate
  handoff does not require TRAF-65 to be closed. Acceptance: the off path
  reproduces every accepted
  `nvlink_locality_v1` and `mixed_attribution_v1` byte, timestamp and component
  attribution exactly; the enabled path conserves bytes against the same
  endpoint ledger, emits one terminal per extent with no double charge against
  the semantic collective, and moves per-request TTFT and TPOT in the registered
  direction; and a converging combine, where the analytic surrogate is weakest,
  is the registered cell rather than a symmetric exchange. The design statement
  is [the packet-device model](../design/packet-device-model.md). This is P1
  because the accepted TRAF-65 study requires the packet path for live metric
  closure; its exact analytic bypass remains supported throughout.
- TRAF-54 (Completeness; P1; L): land the packetized NCCL and RCCL collective
  protocol layer over the GPU ports. TRAF-45 owns the leg below this one, which
  turns a directed intra-node segment into packets on a port; this task owns
  the collective protocol above it, which today has no packet form at all: a
  logical channel is a planner index rather than a thing bound to a port, a
  chunk is a byte count rather than a sequence of packets on that port, and the
  ring step order exists only in the analytic plan. Bind each channel of a
  communicator to a named GPU port, map the chunk expansion of
  `simllm.compute.nccl_stack` onto packets emitted through that port, and let
  the ring step order fall out of per-port arrival rather than out of a
  precomputed sum. Scope boundary: this task adds no port, no packet vocabulary
  and no port-kind taxonomy; it consumes TRAF-45's leg, COMP-40's port events
  and BACK-48's port-kind-independent vocabulary. It also consumes TRAF-65's
  candidate A100 transport profile and NCCL validation rows, while keeping an
  NCCL logical channel distinct from a physical link and from any
  transaction-layer virtual channel the hardware study cannot observe.
  Acceptance: a collective's
  completion is the arrival time of its last packet rather than a composed
  constant, byte conservation holds against the same endpoint ledger the
  analytic path uses, the analytic path stays selectable and reproduces every
  accepted `nvlink_locality_v1`, `mixed_attribution_v1` and
  `nccl_registration_v1` artifact byte-identically, and the interim
  constant-completion section of this document is replaced rather than amended.
  Trigger: TRAF-45 lands the packetized intra-node leg, COMP-40 emits port
  packet events and TRAF-65 publishes its candidate profile and conformance
  fixtures for the selected architecture. That publication is an intermediate
  handoff, not TRAF-65 closure. The hardware capture may run before those
  software triggers; only TRAF-65's live metric closure waits on them. This is
  P1 because that
  accepted live closure requires the collective protocol, while the analytic
  protocol remains the exact bypass.
- TRAF-55 (Completeness; P2; M): make the registration handshake port traffic.
  `CollectiveRegistrationLedger` charges a declared constant that no packet
  carries, so a registration is invisible to every port, occupies no link and
  contends with nothing. Model the handshake as what it is, the descriptor and
  completion exchange a plugin performs when it registers a buffer with a
  device, emitted through the same port the registered channel will use and
  carrying its own extent and attempt identity. Scope boundary: BACK-47 owns
  the device-facing emission contract at the plugin seam and this task owns
  only the collective-protocol side of it, i.e. which registrations happen, in
  what order, on which channel. Acceptance: a registration's cost is the
  completion time of its own port traffic rather than a configuration constant,
  the declared-constant path stays selectable and byte-identical, the three
  re-registration events each produce their own observable exchange, and
  TRAF-56's calibration target moves from a constant to a per-size model.
  Trigger: TRAF-54 binds channels to ports and BACK-47 lands the seam emission
  contract.
- TRAF-57 (Completeness; P2; L): make the collective path port-kind
  independent across NVLink, xGMI and UALink. The registration model, the
  channel plan and the chunk expansion all currently assume one intra-node
  medium and name it NVLink, in the medium label, in the declared bandwidth
  default and in the fixed-cost envelope claims. A collective whose channels
  run over xGMI or UALink ports must be expressible without a second code path
  and without a port-kind switch in any consumer. Scope boundary: the port-kind
  taxonomy itself belongs to the compute module and BACK-48 owns making the
  packet vocabulary reachable from a non-wire port; this task owns the
  collective layer's use of them, including which per-port capability a
  collective channel may require and how a request for an unavailable one is
  refused. Acceptance: one collective plan runs unchanged over all three port
  kinds, a capability a port cannot honor is rejected before any state mutation
  rather than silently ignored, the medium labels a run publishes name the port
  kind that actually carried the traffic, and every accepted NVLink-only
  artifact stays byte-identical. Trigger: BACK-48 lands the
  port-kind-independent vocabulary and the compute-side port taxonomy carries a
  UALink row.
- TRAF-58 (Completeness; P2; S): give collective registration one gate and one
  authority. Two registration states exist today and agree only by convention.
  `CollectiveRegistrationLedger` keys
  `(communicator, generation, channel, buffer)` and owns the charge, while
  `simllm.compute.nccl_stack`'s `require_buffer_registration` gate keeps a
  separate per-communicator `registered_buffers` map that carries no
  generation, is never invalidated by a rebuild, and is never consulted by the
  live chain. A run can therefore satisfy the seam gate and pay nothing, or pay
  and fail the seam gate, with nothing detecting either. Make the ledger the
  sole authority and the seam a read-only projection joined by the same
  identity, so the gate refuses exactly the collectives the ledger has not
  charged. Acceptance: one identity type serves both, a rebuild invalidates the
  seam gate as well as the ledger, a gated collective that the ledger has not
  charged is refused, and both the ungated seam path and the disabled ledger
  path stay byte-identical, including every accepted `nccl_stack_v1` sequence
  and every `nccl_registration_v1` artifact. Trigger: any study that opts the
  seam gate and the live charge in at once, which none does today.
- TRAF-59 (Completeness; P2; S): teach the dependency cross-check about the
  registration offset. `HtsimStepSink._execute_plan` advances
  `artifact_offset_ps` by the composed service, which now includes the
  registration charge, so every authority flow timestamp handed to
  `complete_dependency_cross_check` is shifted by that charge while the
  independently rendered comparison schedule carries none. Selecting
  `dependency_cross_check` together with `collective_registration` therefore
  reports a completion disagreement equal to the charge and calls a correct run
  wrong. The class is pre-existing, since the calibrated base latency shifts
  the same offsets, and that spelling is refused by an explicit configuration
  guard while this one is not. Either refuse the pair the way the calibrated
  profile is refused, or subtract the semantic terms before comparing.
  Acceptance: the two selections either compose with no spurious disagreement
  or are refused before the workdir exists, and the accepted cross-check
  artifacts stay byte-identical. Trigger: a study selects both.
- TRAF-60 (Completeness; P2; S): reconcile the registration ledger with
  prepared-but-unconsumed replays. `HtsimPersistentStepSink.prepare` lowers
  every record in the batch up front, so the ledger is charged for all of them,
  while `StepCollectiveRegistrationOutcome` values are published only as each
  record is consumed. A batch that is prepared and then abandoned, or consumed
  in part, leaves the ledger's charged total ahead of every published
  projection, which is exactly the authority-and-projection divergence the
  repository's one-authority rule exists to prevent. Decide which of the two is
  authoritative for an unconsumed step and make the other its exact projection,
  or refuse preparation while a registration model is selected. Acceptance: the
  ledger total and the published outcomes agree after any prefix of a prepared
  batch is consumed, and the serial sink stays byte-identical. Trigger: a study
  uses the persistent sink with a registration model, which none does today.
- TRAF-61 (Completeness; P1; M): render the prefill-to-decode KV transfer of
  the disaggregated session as fabric traffic. Per-request KV bytes derive
  from the model geometry and the request's context length, sourced at the
  prefill rank set and sunk at the decode rank set per the placement, and
  ride the existing flow rendering and GOAL machinery with no second traffic
  authority. A declared-constant bypass arm prices the handoff without the
  fabric and its off path preserves every accepted artifact byte for byte.
  Composes with CORE-51's session and PLACE-4's placement; per-pair sizes
  and chunking follow the same conventions as the existing collective
  renderers. The declared-constant arm is delivered with exact KV-byte
  accounting, one timing authority and zero backend runs. The closed TRAF-62
  packet-rendered half delivers the bounded mechanism. This task stays open
  until TRAF-64 qualifies that mechanism on the fixed target topology.
- TRAF-64 (Completeness; P1; L): qualify the delivered packet KV handoff on
  PLACE-5's fixed physical target topology. Derive source and destination
  ranks, chunks and paths from the accepted role-aware placement rather than
  the bounded pairwise cell, then run the target topology through the same
  GOAL, flow and last-arrival authority. Acceptance requires exact aggregate
  byte, endpoint and chunk conservation, the declared PCIe term before every
  packet service interval, the exact signed TTFT effect on the live metric
  chain and unchanged decode TPOT. The bounded packet cell and the constant
  and off arms remain byte-for-byte and timestamp-for-timestamp identical.
  PLACE-5 supplies the complete declared physical graph, so this task is
  unblocked. CORE-54 adds a DeepSeek MLA bounded cell with 281,088,000 bytes,
  eight conserved 35,136,000-byte rank-pair messages and quiescent 200 and 400
  Gbit/s arms. It derives the role rank subsets from PLACE-5 but does not feed
  the complete fabric manifest and paths into the packet driver, so TRAF-64
  remains open and this evidence is not a target-path qualification.

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
