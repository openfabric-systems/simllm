# MoE tensor-parallel all-reduce site inventory, v1 expectations

This expectations-only record freezes the TRAF-33 study before any change to
the tensor-parallel all-reduce lowering. It is a mechanism qualification of a
collective inventory: how many all-reduce call sites one transformer layer
executes, and what the renderer, the phase planner and the graph lowerer
therefore put on the wire. It makes no claim that the ring expansion, the
coarse runtime or any latency profile is calibrated to a current GPU or
collective library, and it runs no network backend at all.

## Working-tree status at freeze time

The freeze is authored on branch `codex/traf33_moe_tp_sites` at
`e18b9b0102808e9b8e0f276c2b82c51ed8c5b51d`, with `git status --porcelain`
empty apart from the two files this commit adds. `ruff check .`,
`python scripts/check_docs_format.py` and `pytest -q` were green on that
revision before the freeze was written: 1,675 passed and 8 skipped. No
implementation of the behavior below exists in the tree at this commit, and no
cell of the matrix below has been executed.

## Source audit of every premise

Every premise is read out of this repository at the freeze revision.

- `simllm/traffic/step_comm.py:102` fixes
  `TP_ALLREDUCE_SITES = ("attention", "mlp")`, and
  `simllm/traffic/step_comm.py:615-633` emits both sites for every layer in
  `range(dims.num_layers)` with no reference to the model's mixture geometry.
  A routed MoE model with a real tensor-parallel group therefore renders two
  all-reduces per layer today.
- `simllm/compute/transformer.py:59-72` is the whole of what `ModelDims` says
  about mixtures. `num_experts > 0` means "every layer's MLP is a routed
  mixture", `top_k` is the activated count, `moe_intermediate_size` is one
  expert's already tensor-parallel-sharded intermediate size, and
  `local_num_experts` is "the number of experts per MoE layer resident on THIS
  rank", whose default 0 means "all `num_experts` are local, i.e. no expert
  parallelism". `simllm/compute/transformer.py:141-146` exposes that as
  `resident_experts`. There is no per-layer field of any kind: the dims carry
  one whole-model mixture geometry and cannot express a schedule in which some
  layers are dense and others routed.
- `simllm/adapters/vllm/executor.py:795-806` resolves the vLLM MoE parallel
  shape as `use_ep = enable_expert_parallel and flatten_tp_size > 1`,
  `ep_size = flatten_tp_size if use_ep else 1` and
  `moe_tp_size = 1 if use_ep else flatten_tp_size`. Expert parallelism in use
  therefore always implies that expert weights are not tensor-sharded, and
  `simllm/adapters/vllm/executor.py:735-742` states the same in prose. When
  expert parallelism is out of use, the experts are tensor-sharded over the
  whole flattened set and `simllm/adapters/vllm/executor.py:939` divides the
  per-expert intermediate size by that width.
- `simllm/adapters/sglang/worker.py:752-766` derives SGLang's implicit MoE
  tensor-parallel width as `moe_tp = tp_size // (moe_ep_size * moe_dp_size)`,
  which is 1 exactly when the expert-parallel group spans the whole
  tensor-parallel group. `simllm/adapters/sglang/worker.py:884-901` currently
  refuses any SGLang MoE geometry other than `TP = EP = MoE-DP = 1` and defers
  the rest to SGL-18, so no SGLang path can reach `moe_tp > 1` today.
- `simllm/traffic/step_comm.py:1065-1103` renders, for a routed `dims` and an
  expert-parallel group of at least two ranks, one dispatch and one combine
  pairwise all-to-allv per layer, and `simllm/traffic/step_comm.py:1104-1108`
  returns nothing at all for dense dims or an expert-parallel group below two.
- TRAF-9 in `docs/modules/traffic.md:578-583` reads, in full: "MoE layer op
  ordering. `render_step_goal` renders one calc per layer followed by the TP
  allreduces and then dispatch and combine back to back; a real MoE layer
  splits its compute around the all-to-alls (attention and router before
  dispatch, expert MLP between dispatch and combine) and may overlap
  shared-expert work with the a2avs. The serial whole-layer calc keeps the
  makespan correct only to first order." It is about where the compute sits
  inside a layer, not about how many all-reduce call sites the layer has, so
  it does not cover this defect.
- `simllm/traffic/patterns.py:84-104` fixes the ring expansion used by every
  path below: `2(W-1)` rounds, one successor message per rank and round,
  `chunk = max(1, payload // W)`, consecutive tags from the operation's base.
- The three consumers that must agree are independent implementations.
  `simllm/traffic/step_comm.py:1434-1445` indexes the all-reduce list
  positionally as `layer * 2 + site_index` when rendering GOAL,
  `simllm/traffic/step_comm.py:1750-1760` repeats that positional index when
  building communication phases, and
  `simllm/backends/step_lowerer.py:386-390` instead looks the site up by key
  and skips a missing one. Both positional sites also feed the MoE tag base
  through `moe_base_tag = base_tag + len(tp_ops) * tag_stride`
  (`simllm/traffic/step_comm.py:1405-1406` and `1741-1746`), so a shortened
  all-reduce list moves that base.

### The premise this study narrows

The brief asks for suppression keyed on routedness alone, i.e. on
`num_experts > 0`. The audit does not support that, and implementing it would
replace one defect with another. With expert parallelism out of use, a routed
model's experts are tensor-sharded across the whole tensor-parallel group
(`simllm/adapters/vllm/executor.py:802`, `moe_tp_size = flatten_tp_size`),
every rank holds a column or row shard of every expert, every rank computes a
partial sum of the layer output, and the reduction over the tensor-parallel
group after the expert down projection is real. Nothing renders it if the
mlp site is dropped, because that configuration also renders no all-to-all
(`simllm/traffic/step_comm.py:1104-1105`, expert-parallel group below two).

The premise that does hold is the one the brief's own premise list states:
under combined tensor and expert parallelism with `moe_tp = 1`, each expert's
down projection is computed whole on its owner rank, the combine all-to-all
returns finished per-expert vectors, and the token's home rank forms the
layer output by a local weighted sum. There is no partial sum spread across
the tensor-parallel group, so no mlp-site all-reduce exists. The dims express
exactly that condition, and only that condition, through
`resident_experts < num_experts`.

The frozen rule is therefore: a layer emits the mlp-site all-reduce unless the
dims declare a routed mixture whose experts are expert-parallel, i.e. unless
`num_experts > 0 and resident_experts < num_experts`. The rule reads dims
fields only. It never reads `len(tp_ranks)`, `len(ep_ranks)` or any group
width.

## Frozen matrix

Three model kinds, all per-rank geometries with `hidden_size = 4096`,
`num_heads = 32`, `num_kv_heads = 8`, `head_size = 128`,
`intermediate_size = 1024`, `vocab_size = 32000`, `dtype_bytes = 2`:

| Kind | MoE fields | Expert-parallel group | Sites per layer |
|---|---|---|---|
| `dense` | none | none | 2 |
| `routed-tp` | `num_experts=32, top_k=2, moe_intermediate_size=512, local_num_experts=32` | none | 2 |
| `routed-ep` | `num_experts=32, top_k=2, moe_intermediate_size=512, local_num_experts=4` | ranks 0 to 7 | 1 |

Axes: tensor-parallel width `W` in {1, 2, 8} with `tp_ranks = range(W)`;
`num_layers` `L` in {1, 4, 24}; new tokens `T` in {3, 12} on one decode
request. That is 3 x 3 x 3 x 2 = 54 cells. Every cell renders through three
independent paths and writes its raw row to the external run root configured
by `SIMLLM_DATA_ROOT`. No backend binary is invoked.

`W = 1` produces no tensor-parallel collective at all
(`simllm/traffic/step_comm.py:623-625`), so the 18 cells at `W = 1` carry no
all-reduce. The 12 of them with no expert-parallel group additionally have no
collective work at all and `render_step_goal` must refuse them.

## Closed forms frozen before the run

With `P = T * hidden_size * dtype_bytes = T * 8192` bytes, and
`s(kind)` the sites-per-layer column above:

```text
n_sites      = L * s(kind)
chunk_bytes  = max(1, P // W)
tp_messages  = n_sites * 2 * (W - 1) * W
tp_bytes     = tp_messages * chunk_bytes
tag_stride   = 2 * (W - 1)
moe_base_tag = base_tag + n_sites * tag_stride
moe_ops      = 2 * L  when the expert-parallel group has at least two ranks
```

`P` is 24,576 bytes at `T = 3` and 98,304 bytes at `T = 12`; both divide
exactly by 2 and by 8, so `chunk_bytes` is exact at every cell and the one-byte
floor of `max(1, ...)` is never reached.

## Napkin bounds, written before any measured digit

- Floor. A bandwidth-optimal all-reduce over `W` ranks must move at least
  `2(W-1)/W * P` bytes out of every rank, because reduce-scatter and all-gather
  each move `W-1` chunks of `P/W`. Aggregated over the group that is
  `2(W-1) * P` bytes per call site, and no schedule beats it. At `W = 8`,
  `T = 3` that floor is `14 * 24,576 = 344,064` bytes per site.
- Ceiling. The naive all-gather-then-reduce-locally schedule moves `(W-1) * P`
  per rank, i.e. `W(W-1) * P = 1,376,256` bytes per site at the same point.
  Any ring must sit at or below it.
- Placement. The frozen closed form predicts `2(W-1) * P` exactly, i.e. sitting
  on the floor, because the chunk division is exact and the ring is
  bandwidth-optimal. A measured value above the ceiling or below the floor is a
  defect in the model, the harness or the reading, whichever the follow-up
  finds.
- Covariate that must scale with it. Going from `W = 2` to `W = 8` must
  multiply per-site bytes by exactly `(8-1)/(2-1) = 7`, and going from `T = 3`
  to `T = 12` must multiply them by exactly 4. A per-site figure that matches
  at one point but scales by anything else is not the relation the model
  claims.
- End-to-end plausibility. At `L = 24`, `W = 8`, `T = 3` the corrected
  `routed-ep` step carries `24 * 344,064 = 8,257,536` bytes of tensor-parallel
  all-reduce traffic and the defect added the same amount again. On the wave's
  reference 400 Gbit/s rail that surplus is
  `8,257,536 * 8 / 4e11 = 165.2` microseconds of pure serialization, against
  the roughly 1.9 millisecond composed decode step recorded in the traffic
  status section. An 8.7 percent step-level error is material and well above
  any measurement noise in a deterministic renderer, which is why the defect is
  worth a correctness fix rather than a note.
- System plausibility. A deployed routed-MoE engine under expert parallelism
  does not reduce its MLP output across the tensor-parallel group. Its combine
  all-to-all already returns finished expert vectors, and the surviving
  collective per layer is the attention output-projection all-reduce. One
  all-reduce plus two all-to-alls per layer is the shape the corrected model
  must produce; two plus two is not a shape any such deployment executes.

## Scored relations, and the entailment question answered for each

The question asked of every candidate relation is: given the guards already
registered below, can this relation fail? Only relations that can fail are
scored.

- **S1, renderer against the closed form (scored).** For each of the 36 cells
  with `W > 1`, the messages `render_step_goal` records for all-reduce
  operations number exactly `tp_messages` and carry exactly `tp_bytes`.
  *Can it fail?* Yes. The renderer indexes the all-reduce list positionally,
  the corrected list is shorter for `routed-ep`, and a wrong index map, a
  dropped or duplicated site, or a changed chunk each break the equality. No
  other registered guard forces it.
- **S2, phase planner against the closed form (scored).** For the same 36
  cells, `step_communication_phases` yields exactly `n_sites * 2 * (W - 1)`
  all-reduce phases whose directed segment bytes sum to `tp_bytes`.
  *Can it fail?* Yes, independently of S1. It is a separate implementation with
  its own positional index at `simllm/traffic/step_comm.py:1754`, and it can
  disagree with the renderer.
- **S3, graph lowerer and collective plan against the closed form (scored).**
  For the same 36 cells, the `ExecutionGraph` from `SerialStepLowerer` carries
  exactly `n_sites` all-reduce operations, and the attached `CollectivePlan`
  extents sum to `tp_bytes`.
  *Can it fail?* Yes, independently of S1 and S2. The lowerer looks sites up by
  key rather than by position, and the plan's rounds and extents are built in
  another module, so agreement with the renderer is a real risk rather than a
  restatement.
- **S4, tag disjointness and the moved MoE tag base (scored).** For each of the
  12 cells that render both all-reduce and all-to-allv work, no tag is shared
  by two distinct operation ids, and the smallest all-to-allv tag equals
  `base_tag + n_sites * 2 * (W - 1)`.
  *Can it fail?* Yes, and this is the relation the refactor most plausibly
  breaks. Under the current positional scheme a suppressed site leaves its tag
  block allocated while `len(tp_ops)` shrinks, which drives the MoE base into
  an all-reduce block. S1 to S3 are byte and count statements and do not
  constrain tags.

Scored total: 4 families, 36 + 36 + 36 + 12 = 120 parameterized instances.

## Fatal guards, void and never scored

A violated fatal guard voids the run for the purpose of closing TRAF-33. None
of these is reported as a fraction.

- **F1, the site rule itself.** `layer_tp_allreduce_sites` returns
  `("attention", "mlp")` for `dense` and `routed-tp` dims and `("attention",)`
  for `routed-ep` dims. This is by construction of the function under test and
  carries no evidential weight.
- **F2, no group-width dependence.** For a fixed kind and `L`, the per-layer
  site tuple is identical at `W` in {1, 2, 8}, and is unchanged when
  `tp_ranks` is relabeled to a disjoint rank set of the same width and when
  the expert-parallel group width is changed from 8 to 2 with the dims held
  fixed. A violation means the rule read a group width and the study is void.
- **F3, the MoE inventory is untouched.** Every `routed-ep` cell with a
  two-or-more-rank expert-parallel group renders exactly `2 * L` all-to-allv
  operations with unchanged uniform pair tables. This change may not move one
  expert byte.
- **F4, the dense path is byte-identical.** `pytest -q` stays green and no
  accepted artifact that involves only dense dims requires an edit. The study
  additionally records the SHA-256 of each dense cell's rendered GOAL text so a
  later change can diff it.
- **F5, refusal is preserved.** At `W = 1` with no expert-parallel group,
  `render_step_goal` still raises `ValueError` naming "no tensor-parallel
  collectives", at every kind and `L`.

## Relations that are entailed, reported and not scored

Each of these follows arithmetically from S1 together with F1, so scoring them
would inflate a denominator with restatements.

- The token ratio `tp_bytes(T=12) / tp_bytes(T=3)` is exactly 4 at every cell.
- The width ratio `tp_bytes(W=8) / tp_bytes(W=2)` is exactly 7 at every cell,
  at fixed kind, `L` and `T`.
- The `routed-ep` arm carries exactly half the `dense` arm's tensor-parallel
  bytes at matched `W`, `L` and `T`.
- The headline inventory the brief asks for, a 24-layer routed model with an
  8-rank tensor-parallel group and an 8-rank expert-parallel group rendering
  24 all-reduces plus 48 all-to-alls, i.e. 72 collectives and not 96, is
  entailed by F1 and F3 and is reported as a structural cell.

## What the run may not do

The run may not change any modeled behavior after observing a number, may not
retune a closed form to a measurement, and may not convert a violated fatal
guard into a score. If a scored family fails, the finding is reported, TRAF-33
stays open, and the cause is named. Refuting a frozen expectation is a result,
not a failure to be edited away.
