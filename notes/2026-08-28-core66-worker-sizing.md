# CORE-66 worker sizing

## Frozen cell

The feasible capture uses three GH200 nodes with four GPUs per node. It runs
one expert-parallel rank per GPU, for EP12 and twelve ranks total. Each rank
keeps four routed experts resident, so the reduced model has 48 routed experts.
The run keeps batch 32 and KV length 2,000 per rank, disables MTP, enables
data-parallel attention and the DeepEP all-to-all backend, and measures one
decode iteration. Dummy weights are mandatory and model-weight downloads are
forbidden.

The reduced model depth is four transformer layers. It preserves the standard
DeepSeek-V3 first-three-dense composition and includes the following MoE layer,
so dense and MoE launches can be captured separately. This is an identity and
physics capture. It is not an EP72 service measurement.

## Resource envelope

- Allocation: three nodes, four GH200 GPUs per node, one hourly partition.
- Scheduler submissions: one. A retry requires an integrator decision and is
  not part of this worker envelope.
- Planned occupied time: no more than 45 minutes inside the hourly partition.
- Measured work: one decode iteration after setup and warmup, with the counter
  collection performed inside the same scheduled cell when permission allows.
- Local bulk root: `$SIMLLM_WAVE_RUN_ROOT/core66/`.
- Remote campaign root: `$MERLIN_CAMPAIGN_ROOT/core66/`.
- Repository artifacts: `examples/deployment_curve_v1/` and `notes/`.
- Retrieval: manifests, profiler summaries, routing ledgers, logs and selected
  trace slices only. Raw traces remain in the bulk roots.

The runner must terminate child processes and release the allocation when its
controlling SSH session disappears. Every stage writes a resume marker, but a
resume may only continue the same allocation and must not resubmit the cell.

## Storage estimate

The repository-facing evidence should remain below 20 MiB. Local or campaign
bulk may use up to 20 GiB for short-lived profiler traces. No checkpoint or
model-weight shard is copied, downloaded or retained.

