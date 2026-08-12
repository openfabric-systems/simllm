# Routed-MoE byte conservation v1 expectations

This document freezes the VLLM-24 expectations before the independent
conservation guard, its traffic-side wiring, the study harness, or any
measured result exists. The repository source at this boundary is commit
`aeb40ac95cdd8163942297335948c94df0376e04`.

The evidence is authored against official vLLM v0.26.0 commit
`568afb3a13806beb53bb2e6bd518269357b237c0`. A run records the source files
and version it actually observes. The authored-against identity and the
observed identity are independent provenance fields; no check requires a live
checkout, package, or submodule pin to equal the authored-against commit.

## Why this task exists, stated before the run

The project shipped an 8x routed-MoE byte defect for its entire history
because every byte check compared the renderer against itself. The observed
schedule path then loosened the adapter-versus-plan byte agreement check for
exactly the collective class its own producer emits: the Granite producer
emits zero-byte semantic all-to-allv markers, and
`_validate_observed_collective` accepts an empty pair table and an empty
request-pair table as agreement. `_validate_microbatch_partition` recombines
microbatch plans that the same traffic planner produced from the same record,
so it cannot see a defect shared by both sides.

An independent validation is one that does not consume the projection it is
checking. The projection under test is the routed byte table the traffic
planner produces on the observed path. The independent inputs are:

- `StepRecord.scheduled[i].num_new_tokens`, the adapter's source-observed
  per-request token ownership carried across the adapter seam;
- `RoutedMoeSupply.engine_rank`, the explicit declaration that exactly one
  rank dispatches the scheduled tokens while peer EP ranks own experts and
  carry no scheduled tokens;
- `ModelDims.top_k`, `ModelDims.num_layers`, `ModelDims.hidden_size` and
  `ModelDims.dtype_bytes`, the model geometry;
- the EP group `ep_ranks`.

None of those is derived from the per-token routing walk that builds the byte
table, so a defect in that walk cannot hide inside the check.

## Frozen conservation rules

Nine named rules. A rule either holds or is a violation; the guard is a fatal
unscored invariant and is never reported as a fraction.

| Rule | Statement | Applies |
|---|---|---|
| `source-attribution` | every directed pair's source rank equals the declared owner rank | always |
| `destination-legality` | every destination is in `ep_ranks` and differs from the source | always |
| `owner-egress` | bytes leaving the declared owner equal the total directed bytes of the step | always |
| `transpose-symmetry` | each layer's combine pair table is the exact transpose of its dispatch pair table | always |
| `step-hop-bound` | `emitted_bytes <= total_new_tokens * top_k * num_layers * 2 * vector_bytes` | always |
| `vector-granularity` | every directed byte count is a positive multiple of `vector_bytes` | captured supply only |
| `request-identity` | every request in the request-pair table is a scheduled request with nonzero new tokens | captured supply only |
| `per-request-hop-bound` | per layer and phase, a request's bytes are at most `its tokens * min(top_k, W - 1) * vector_bytes` | captured supply only |
| `per-layer-hop-bound` | per layer and phase, total bytes are at most `total_new_tokens * min(top_k, W - 1) * vector_bytes` | captured supply only |

`vector_bytes = hidden_size * dtype_bytes` is one hidden activation vector.
`W = len(ep_ranks)` is the EP world.

The four captured-supply-only rules are deliberately excluded from the uniform
destination approximation, which spreads `total_new_tokens * top_k`
assignments evenly over `W` ranks without deduplicating several selected
experts that land on the same destination. That approximation therefore
exceeds `min(top_k, W - 1)` per token whenever `top_k > W`, by construction and
not by defect. The five always-on rules do hold for it, and
`step-hop-bound` is the rule this freeze relies on.

## Why EP world 8 and not EP world 2

The historical defect replicated the dispatch table so that every EP rank
appeared as a source of the same tokens, multiplying total directed bytes by
`W`. Let `hops_A(W)` be the correct owner-attributed hop count of a step and
`hops_B(W) = W * hops_A(W)` the replicated one. `step-hop-bound` detects the
fault exactly when `W * hops_A(W) > total_new_tokens * top_k * num_layers * 2`.

With the captured Granite geometry (`top_k = 8`, `num_experts = 32`,
`num_layers = 24`) the per-token per-layer per-phase hop count is at most
`min(top_k, W - 1)`:

- at `W = 2` that ceiling is 1, so `hops_A(2) <= T * 1 * 24 * 2` and
  `hops_B(2) = 2 * hops_A(2) <= T * 8 * 24 * 2`, which is the bound itself.
  The bound therefore **cannot** detect the replication at EP world 2, for any
  routing whatsoever. This is a first-principles certainty, not a measurement.
- at `W = 8` that ceiling is 7 and 32 experts sit in 8 owner blocks of 4, so a
  token's 8 distinct selected experts reach several owners. The bound detects
  the replication whenever `hops_A(8) > T * 1 * 24 * 2`, i.e. whenever the mean
  number of distinct remote owners per token-layer exceeds 1.

That is the whole reason the identity is evaluated at EP world 8.

## Frozen study registry

Input capture: the tracked `examples/preplay_trace_v1/granite_length_cap.jsonl`
Granite trace, `simllm-preplay-trace-v1`, `expert_count = 32`, `top_k = 8`,
24 MoE layers, request `length-cap` with a 22-token prompt.

Geometry: `num_layers = 24`, `hidden_size = 1024`, `dtype_bytes = 2`, so
`vector_bytes = 2048`.

Expert placement: contiguous owner blocks, `32 // W` experts per rank on every
layer, at placement epoch 0.

Steps:

| Step | Phase | `T` = total_new_tokens |
|---|---|---|
| `prefill` | PREFILL | 22 |
| `decode` | DECODE | 1 |

Worlds: `W in {2, 8}`. Arms: `A` owner-attributed (the traffic planner's
captured routed table) and `B` source-replicated (arm A's table replicated so
that every EP rank appears as a source of the same tokens, the pre-TRAF-25
shape).

Cells: 2 steps x 2 worlds x 2 arms = 8 evaluations.

### Exact oracle rows (closed form, no measurement)

| Quantity | Closed form | prefill | decode |
|---|---|---|---|
| `step_hop_bound` (hops) | `T * top_k * num_layers * 2` | 8448 | 384 |
| `step_hop_bound` (bytes) | above `* 2048` | 17301504 | 786432 |
| `per_layer_hop_bound` at `W = 2` (hops) | `T * min(top_k, W - 1)` | 22 | 1 |
| `per_layer_hop_bound` at `W = 8` (hops) | `T * min(top_k, W - 1)` | 154 | 7 |
| `hops_A(W)` ceiling | `T * min(top_k, W - 1) * num_layers * 2` at `W = 2` | 1056 | 48 |
| `hops_B(W)` | `W * hops_A(W)` exactly | | |
| directed bytes | `hops * 2048` | | |

### Frozen expectations

E1 (fatal unscored). Arm A conserves at both worlds and both steps: all nine
rules hold, `source-attribution` reports exactly one source rank equal to the
declared `engine_rank`, and `owner_egress_bytes == total_directed_bytes`.

E2 (fatal unscored). `hops_B(W) == W * hops_A(W)` and
`bytes_B(W) == W * bytes_A(W)` exactly, in all four (step, world) cells. This
is the fault-injection identity; if it fails the injected fault is not the
historical defect and the run is void.

E3 (scored, genuine risk). At `W = 8`, `step-hop-bound` is violated by arm B on
both steps. Quantitatively this requires `hops_A(8) > 1056` on the prefill step
and `hops_A(8) > 48` on the decode step, i.e. a mean of more than one distinct
remote owner per token-layer. Direction: `hops_B(8) / step_hop_bound > 1`.

E4 (scored, genuine risk). At `W = 2`, `step-hop-bound` is NOT violated by
arm B on either step, because `hops_B(2) <= 8448` and `<= 384` respectively
hold by construction. Direction: `hops_B(2) / step_hop_bound <= 1`. This is the
counter-cell that shows the identity has to be evaluated at EP world 8.

E5 (scored, genuine risk). `source-attribution` is violated by arm B at BOTH
worlds and both steps, reporting `W` source ranks instead of 1. The two rules
are not redundant: the source rule sees the replication structurally, the hop
bound sees it dimensionally, and only the hop bound survives a fault that
keeps a single source but inflates its bytes.

E6 (scored, genuine risk). The Granite observed-schedule producer's zero-byte
semantic all-to-allv markers no longer bypass byte evidence. Lowering the same
Granite observations through `lower_step_observations` with a captured routed
supply at `W = 8` runs the conservation guard on the planned table and reports
`evidence_mode = "no-byte-evidence"` for the observation itself. An
observation carrying a pair table that disagrees with the plan still fails
exactly as before.

Scored behavioral instances: E3 (2 instances), E4 (2 instances), E5 (4
instances), E6 (1 instance) = 9. E1 and E2 are fatal unscored guards and never
enter that denominator.

### Entailment analysis

E5 is not entailed by any earlier fatal oracle: E1 asserts arm A's properties,
not arm B's, and E2 asserts only the byte multiplier of the injected fault, not
which rule detects it. E3 and E4 are not entailed by E2 either: E2 fixes
`hops_B = W * hops_A` but says nothing about `hops_A`, and whether
`W * hops_A(8)` clears 8448 depends on the captured routing distribution, which
is not frozen here. E4's direction IS a first-principles certainty and is
scored as a genuine-risk instance only because the implementation could
compute the bound wrongly; its refutation would mean the guard is wrong, not
that the physics changed.

### Physical sanity, floors and ceilings stated first

Floor. A step cannot move fewer than zero routed bytes, and it moves zero only
when every selected expert of every token is resident on the owner rank. With
32 experts in 8 blocks of 4 and 8 distinct selected experts, at least
`ceil(8 / 4) = 2` owner blocks are touched, so at least one remote hop per
token-layer exists at `W = 8`: `hops_A(8) >= 22 * 1 * 24 * 2 = 1056` on the
prefill step. At `W = 2` there are only 2 blocks of 16 experts, so all 8
selected experts can be resident on the owner rank: the floor there is 0.

Ceiling. `hops_A(8) <= 22 * 7 * 24 * 2 = 7392` on the prefill step, since a
token reaches at most 7 remote owners out of 8. `hops_A(2) <= 1056`, since a
token reaches at most 1 remote owner out of 2.

So the prefill arm A directed bytes must land in
`[1056 * 2048, 7392 * 2048] = [2162688, 15138816]` bytes at `W = 8` and in
`[0, 1056 * 2048] = [0, 2162688]` bytes at `W = 2`. A measured value outside
those intervals is proof of a defect in the model, the harness or the reading,
regardless of how exactly any other number matches.

Scaling companion. Moving from `W = 2` to `W = 8` must not lower arm A's
directed bytes, because a wider EP world can only split a token's experts over
more owners. If the measured ratio `bytes_A(8) / bytes_A(2)` comes out below 1
the relation is not what the model claims.

## Registry discipline

This freeze registers no new task ID. An ID is registered after the run only
for a registered VLLM-24 acceptance clause the run did not demonstrate.
