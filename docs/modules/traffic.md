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
(examples/m1/RESULTS.md); ring allreduce, pairwise all-to-allv and the
binomial tree have structural unit tests only so far (TRAF-4). The JSONL
collective-trace consumer is not yet implemented.

## Open tasks

- TRAF-4: end-to-end closed-form validation of ring allreduce, pairwise
  all-to-allv and binomial broadcast against the fluid backend (per-round
  forms are exact there), extending the M1 study pattern.
- TRAF-2: MoE dispatch/combine from routed-experts captures, including EPLB
  epoch-snapshot handling in trace records (milestone M5).
- TRAF-3: KV-transfer records for PD-disaggregation and cache-miss
  re-prefill (milestone M6).
