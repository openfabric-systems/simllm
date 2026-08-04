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
  a zero-token drain record produces no ops. `render_step_goal` renders
  the serial per-rank chain (per layer: `calc`, then the two allreduces)
  through the existing `ring_allreduce` pattern; tags are disjoint per
  allreduce.

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
(examples/m1/RESULTS.md), and the M4 studies did the same for ring
allreduce on both null-network profiles (examples/m4/RESULTS.md checks
A and C: fluid exact to 0 ps at six size/width points, packetized nn
matching its per-round point form to 0 ps), which closes the
ring-allreduce part of TRAF-4; pairwise all-to-allv and the binomial tree
still have structural unit tests only. The `step_comm` mapping landed with
M4 and is validated end to end by the examples/m4 grid and the live tp=8
closed-loop run. The JSONL collective-trace consumer is not yet
implemented (TRAF-5).

## Open tasks

- TRAF-4 (ring-allreduce part closed by examples/m4): end-to-end
  closed-form validation of pairwise all-to-allv and binomial broadcast
  against the fluid backend (per-round forms are exact there), extending
  the M1/M4 study pattern.
- TRAF-2: MoE dispatch/combine from routed-experts captures, including EPLB
  epoch-snapshot handling in trace records (milestone M5).
- TRAF-3: KV-transfer records for PD-disaggregation and cache-miss
  re-prefill (milestone M6).
- TRAF-5: the JSONL collective-trace consumer (parse
  `simllm-collective-trace-v1` records and hand them to pattern expansion).
- TRAF-6: sequence parallelism in the step model. `step_comm` reduces the
  full activation on every rank; SP would replace each allreduce with a
  reduce-scatter plus allgather of 1/W the bytes around the norm/dropout
  regions.
- TRAF-7: communication/compute overlap in the step model. `step_comm`
  chains each layer's compute and its two allreduces strictly serially;
  real engines overlap the MLP allreduce with the next layer's start under
  some schedules.
- TRAF-8: pipeline-parallel activation traffic from step records. Records
  carry no PP stage attribution yet, so `step_comm` emits TP collectives
  only; the M1 workload-B GOAL shows the target activation-chain shape.
