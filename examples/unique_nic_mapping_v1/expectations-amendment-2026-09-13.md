# Unique-NIC mapping amendment, 2026-09-13

This expectations amendment is post-specified to a harness-feasibility
finding made before any U3 implementation or run, and is frozen before that
implementation. The original freeze stays unchanged; this file replaces the
path cell U3 takes and fixes two readings.

## Finding

The freeze asked for U3's all-pairs traffic "through the sink's
expert-parallel path". `HtsimStepSink` renders expert-parallel traffic with
one engine rank today: uniform routing dispatches from rank 0 and combines
back to it, captured routing supplies one engine, and the full-population
variant of the lowerer is not threaded into the sink configuration. Under
that traffic a NIC shared by `g` GPUs still carries rank 0's eight remote
flows, so the frozen `8 g` flows-per-endpoint table cannot be observed
through the sink. Threading a full-population option through the lowerer
and sink is a separate scope, registered rather than pulled into this slice.

## Replacement path for cell U3

U3 exercises exactly the mechanism under test through the traffic layer
and the backend, not through the sink's routing: build one
`CollectiveCommunicationPhase` holding every remote ordered GPU pair of the
16-rank placement with `S` bytes each, classify it with
`classify_step_locality` under the `unique-nic` mapper (all 16 x 15 pairs
minus the intra-node ones stay on the fabric; the intra-node ones are local
and carry no fabric bytes), render it with `render_fabric_phase_goal`
using the collapsed-pair tag rule and the projection table, convert with
`to_binary`, run `run_htsim_rnic` on `rnic-nn-fluid` at 400 Gbit/s, and
join every completion row through the projection table. The U3 table, its
ratios, the fabric-byte and NVLink-byte invariants, the `g = 1` byte identity
with `gpu-rank` and the exactly-once join are unchanged. The sink's own
unique-nic evidence is U1 and U2, which stay as frozen.

## Fixed readings

- `M` is the largest number of segments sharing one GOAL endpoint pair
  within one phase, maximized over the step's phases. Tags already differ
  between phases, so `tag * M + j` with `j < M` is unique across the step,
  and `M = 1` remains the identity for every accepted `gpu-rank` artifact.
- U2 uses the m5 `decode8x2048` and `prefill2048` step shapes with
  `moe_dims(8)` and `ep_ranks` 0 through 7.

## What does not change

Every literal of U1, U2, U3, U4 and U5, the seven digests and the six m5
makespans. The full-population expert-parallel traffic through the sink is
registered as part of the residual task this slice creates.
