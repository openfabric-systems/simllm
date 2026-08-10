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
- `step_moe_alltoalls` (M5, TRAF-2 first half): the same record plus MoE
  `ModelDims` plus an expert-parallel group of W GOAL ranks maps to the
  step's MoE traffic: per MoE layer, a dispatch pairwise all-to-allv then
  a combine pairwise all-to-allv, each rank sending
  `total_new_tokens * top_k * hidden_size * dtype_bytes // W` to every
  other rank. This is the uniform-routing assumption: the router spreads
  (token, expert) assignments evenly over the EP group and each rank's
  own 1/W share stays local, off the fabric; replacing it with per-token
  routed-experts captures (including EPLB placement-epoch snapshots) is
  the TRAF-2 second half. Dense dims, an EP world below 2 or a zero-token
  record produce no ops.
- `render_step_goal` renders the serial per-rank chain (per layer: `calc`,
  the two TP allreduces when the TP world produces them, then for MoE dims
  with `ep_ranks` given the dispatch and combine all-to-allvs) through the
  existing `ring_allreduce` and `pairwise_all_to_allv` patterns; tags are
  disjoint per collective. The calc input may be one compatibility scalar or
  an ordered value per layer, and `num_goal_ranks` idle-fills a larger GOAL
  layout. A scalar step without MoE work renders byte-identically to the
  pre-M5 emitter (golden test).
- `render_serial_execution_graph_goal` is the CORE-2 graph-only diagnostic
  replay. It accepts validated per-rank compute, ring allreduce and pairwise
  all-to-allv operations, preserves `participant_local_depends_on` edges and
  FIFO predecessors, and never reduces a separate `depends_on`
  whole-operation barrier through rank-local ancestry. It fails loudly on
  cross-rank barriers, work or timing semantics the serial GOAL subset cannot
  represent. It consumes no `StepRecord` after lowering.

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

CORE-2 additionally proved that serial GOAL rendered only from a
JSON-round-tripped `ExecutionGraph` is byte- and timing-equivalent to the
legacy step path over TP width and link-rate sweeps, including a MoE sentinel
([results](../../examples/core2_lowering/RESULTS.md)).

The BACK-5/BACK-7 step-sink study additionally validates unequal ordered calc
values over layer-count and TP-width sweeps, plus explicit 64-rank padding on
the fluid and physical-topology paths. Every valid comparison has 0 ps timing
residual; see
[examples/step_sink_precision/RESULTS.md](../../examples/step_sink_precision/RESULTS.md).

## Open tasks

- TRAF-4 (ring-allreduce part closed by examples/m4, pairwise-a2av part
  closed by examples/m5): end-to-end closed-form validation of binomial
  broadcast against the fluid backend, extending the M1/M4/M5 study
  pattern.
- TRAF-2 (second half; first half landed with M5): MoE dispatch/combine
  from routed-experts captures, including EPLB epoch-snapshot handling in
  trace records, replacing the uniform-routing assumption of
  `step_moe_alltoalls` (and its floor-divided equal per-pair payload)
  with observed per-token destinations. Named sub-approximation (audit
  finding): the current mapping sends one hidden-vector copy per (token,
  expert) assignment, while real dispatch kernels dedupe to at most one
  copy per (token, destination rank) and pre-reduce the combine, so with
  top_k > W the emitted a2av bytes are inflated by up to top_k / W; the
  routed-captures half removes this along with uniform routing.
- TRAF-3: KV-transfer records for PD-disaggregation and cache-miss
  re-prefill (milestone M6).
- TRAF-5: the JSONL collective-trace consumer (parse
  `simllm-collective-trace-v1` records and hand them to pattern expansion).
- TRAF-6: sequence parallelism in the step model. `step_comm` reduces the
  full activation on every rank; SP would replace each allreduce with a
  reduce-scatter plus allgather of 1/W the bytes around the norm/dropout
  regions.
- TRAF-7: communication/compute overlap in the step model. `step_comm`
  chains each layer's compute and its collectives strictly serially;
  real engines overlap the MLP allreduce with the next layer's start under
  some schedules. Implement this after CORE-3/4, by lowering compute and
  collective work onto the framework-observed logical streams with explicit
  event/dependency edges in `ExecutionGraph`. The adapter owns observed
  program order and legal concurrency; the traffic planner owns collective
  algorithm/chunk expansion; `DeviceRuntime` owns realized overlap after
  CUDA-stream, GPU, HBM, copy-engine, NCCL-channel, WQE and NIC contention.
  No layer stores or learns an overlap percentage. First validate an ideal
  independent-resource graph at exact `max(compute, communication)` versus
  the serial graph at exact `compute + communication`, then add one resource
  contention mechanism at a time.
- TRAF-8: pipeline-parallel activation traffic from step records. Records
  carry no PP stage attribution yet, so `step_comm` emits TP and EP
  collectives only; the M1 workload-B GOAL shows the target
  activation-chain shape.
- TRAF-10: intra-node NVLink locality split (maintainer direction,
  2026-08-04). Intra-node segments of a collective should not ride the
  fabric: model them as a point-to-point NVLink-class network (first cut:
  a flat same-generation NVLink per-GPU bandwidth, analytic, no packet
  simulation), and send only inter-node segments to htsim. Consequence:
  single-node TP (any width up to 8 on the 8-GPU reference node) has no
  fabric component at all, and the fabric story applies to cross-node
  placements. Needs the locality knowledge of the placement manifest
  (`is_intra_node`) and composes with the `unique-nic` GOAL-rank mapping
  (PLACE-2); the committed examples/breakdown fabric-TP columns become
  the cross-node what-if under this model.
- TRAF-9: MoE layer op ordering. `render_step_goal` renders one calc per
  layer followed by the TP allreduces and then dispatch and combine back
  to back; a real MoE layer splits its compute around the all-to-alls
  (attention and router before dispatch, expert MLP between dispatch and
  combine) and may overlap shared-expert work with the a2avs. The serial
  whole-layer calc keeps the makespan correct only to first order.
