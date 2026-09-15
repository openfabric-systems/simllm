# Declared expert variant result

## Outcome

What ran: `examples/declared_expert_variants_v1`, the frozen PLACE-7 and
PLACE-11 qualification, built declared placements from implementation commit
`0d8fb907` for the two mixture-of-experts (MoE) deployments the pinned vLLM
0.27.1 runs and the declared builder could not describe, a MoE model with
expert parallelism switched off and a model whose framework-side conditions
downgrade a declared `round_robin` map to `linear`, and drove the PLACE-3 and
PLACE-13 studies' own `--check` runs as the live identity arms. The
expectations-only commit is `841dfb38`, its baseline amendment is `e08bed05`,
its cell V13 amendment is `5ffe5606`, and the harness commit is `8d1b9e66`.

What came out: the result is `PASS` with no finding. There is no deciding
number, because the scored denominator is zero by design and was frozen that
way; the deciding fact is that all ten structural cells were exact, all eight
rejection controls refused, the five reference manifests kept their bytes and
digests, and both the PLACE-3 and the PLACE-13 studies reproduced their
tracked results unchanged. The one quantity worth carrying is the ownership
conservation: with expert parallelism off the worked example holds 12,288
ownership entries at exactly eight owners per `(layer, expert)` pair, where
the accepted expert-parallel-on geometry holds 1,536 at exactly one.

What it changes for the project: PLACE-7 and PLACE-11 both close. The
placement doc's statement that a declared layout states the deployments the
framework really runs becomes literal:
`DeclaredExpertLayout(expert_parallel=False)` emits the stage's `ep` group with
all-expert ownership, so a declared manifest of an expert-parallel-disabled run
and an extracted one now agree on group inventory instead of differing by a
whole group; and `DeclaredExpertMapExceptions` states each framework-side and
model-side rule as its own condition, so a layout that would really run
`linear` records `linear`. The module docstring's sentence that the fallback
"is not modeled here" is withdrawn in the same change.

What it does not change: nothing about any metric. The enabled PLACE-7 path is
not live-reachable and this study did not make it so. No backend ran, no
makespan moved, and no step record was produced. The accepted m5 identity is
untouched by construction, and the two `--check` arms are the evidence of that
rather than a new measurement. The all-owner geometry is representable in the
manifest and reaches no consumer, which stays with PLACE-8. The remaining
declared MoE dimensions that carry no manifest field are registered as
PLACE-15.

## Pinned semantics that the builder reproduces

Read from the installed vLLM 0.27.1 package, file names relative to the
installed `vllm` package.

- The `ep` group survives the flag. `distributed/parallel_state.py`
  (`initialize_model_parallel`) creates it under
  `config.model_config is None or config.model_config.is_moe`, commented
  "Don't create EP group for dense models". `enable_expert_parallel` does not
  appear in that condition, so the group and every member's index in it are
  what they are with expert parallelism on.
- The off switch. `model_executor/layers/fused_moe/config.py`
  (`FusedMoEParallelConfig.make`) computes
  `use_ep = dp_size_ * pcp_size_ * tp_size_ > 1 and
  vllm_parallel_config.enable_expert_parallel`, so expert parallelism is off
  both when the launcher declines it and when the flattened width is one.
- What replaces it. The `if not use_ep` branch keeps `tp_size` and `tp_rank`
  from `flatten_tp_across_dp_and_pcp` (`flatten_tp_size = dp_size * pcp_size *
  tp_size`) and sets `ep_size=1`. `FusedMoEConfig.__post_init__` then asserts
  `intermediate_size % tp_size == 0` and shards by that flattened width, and
  `determine_expert_map` returns `(global_num_experts, None, None)` at
  `ep_size == 1`, so every rank of the stage holds every expert.
- The two-step strategy resolution.
  `model_executor/layers/fused_moe/expert_map_manager.py`
  (`determine_expert_placement_strategy`), called only when
  `moe_parallel_config.use_ep`, returns `linear` when
  `(num_expert_group is not None and num_expert_group > 1) and
  num_redundant_experts == 0 and not enable_eplb` is false, and again when
  `use_all2all_kernels and not needs_round_robin_routing_tables`. Then
  `ExpertMapManager._determine_placement_strategy` runs unconditionally and
  returns `linear` when `ep_size == 1`.
- The backend condition. In `config.py`,
  `use_all2all_kernels = use_ep and (dp_size > 1 or pcp_size > 1 or
  is_sequence_parallel)` and `needs_round_robin_routing_tables =
  use_deepep_ll_kernels or use_nixl_ep_kernels`, so the fallback fires on any
  all-to-all backend other than `deepep_low_latency` and `nixl_ep`.
- The EPLB refusal and group. `layer.py` raises its divisibility `ValueError`
  only inside `if enable_eplb:` and only when
  `use_ep and global_num_experts % ep_size != 0`; its `else` arm asserts
  `num_redundant_experts == 0`. `initialize_model_parallel` creates the `eplb`
  group from the same rank lists under
  `if config.parallel_config.enable_eplb`.

Two claims in the registry entries did not survive this reading, and the
builder follows the source:

- PLACE-11 said the fallback applies when the model "has at most one expert
  group". The condition is `num_expert_group is not None and num_expert_group
  > 1`, so a model that never sets the field, which is every model outside the
  grouped-topk family, also falls back.
- PLACE-11 said "some model implementations refuse expert counts their
  expert-parallel size does not divide". None does. The literal model-scoped
  refusal is `ValueError(f"Tensor parallel size {self.tp_size} is greater than
  the number of experts {...}")`, raised by sixteen sparse-MoE blocks under
  `model_executor/models` including `qwen3_moe.py`, `mimo_v2.py`, `laguna.py`,
  `cohere2_moe.py` and `minimax_m2.py`, and it compares the declared tensor
  width, not the expert-parallel size. What the model implementations really
  carry is a silent uniform-block assumption,
  `n_local_physical_experts = n_physical_experts // ep_size` with
  `physical_expert_start = ep_rank * n_local_physical_experts`, in
  `deepseek_v2.py`, `mixtral.py`, `qwen3_moe.py` and `transformers/moe.py`,
  which nothing raises on. The builder therefore offers
  `model_refuses_tp_above_experts` for the real refusal and
  `model_uniform_expert_blocks` for the assumption, and the second refuses a
  remainder rather than recording ownership such a model would not honor.

## Evidence

| Evidence class | Result |
|---|---|
| Scored exact-oracle rows | 0 of 0, by design and as frozen; see "Why nothing is scored" |
| Structural exact guards (V1, V3, V4, V5, V6, V7, V8, V10, V11, V12) | 10 of 10 cells exact |
| Rejection controls (V9) | 8 of 8 refused, each a `ValueError` naming the field, with the four matching build controls all building unchanged |
| Fatal off-path defaults (V2) | `expert_parallel` defaults true, `exceptions` defaults `None`, every condition defaults false |
| Fatal compatibility digests | 5 of 5 reference manifests byte identical to the pre-change record |
| Fatal study checks | PLACE-3 and PLACE-13 both reproduced their tracked results |
| Recorded finding (V13) | the consumers' state on the day of the run, not a claim about this slice |

Counts in different evidence classes are not added, and none of them is a
score. The tracked [results](results.json) hold every cell; bulk manifests
stay under the configured data root.

V1, the worked example with expert parallelism off (`tp=4, pp=2, dp=2`, 48
layers all MoE, 32 experts, EP size 8): rank 9 keeps the `ep` group
`[0, 1, 2, 3, 8, 9, 10, 11]` at index 5 and the layer range `[0, 24)`, and
owns the full `[0, ..., 31]` in each of layers 0 through 23; rank 15 keeps
`[4, 5, 6, 7, 12, 13, 14, 15]` at index 7 and `[24, 48)` and owns the full
range in each of layers 24 through 47. The expert-parallel-on twin gives those
same two ranks `[20, 21, 22, 23]` and `[28, 29, 30, 31]`, which is the
accepted PLACE-3 cell C1. Summed over all sixteen ranks the ownership entries
number exactly 12,288, with exactly eight owners for every `(layer, expert)`
pair of a stage.

V3, the group-inventory gap: the V1 manifest and
`declared_manifest(tp=4, pp=2, dp=2)` agree rank by rank on global rank,
hostname, local rank and the `tp`, `pp` and `dp` memberships, and differ in
exactly the `ep` membership, the layer range and the expert ownership. The
declared epoch is zero here, so it is equal rather than differing. This is the
gap the task existed to close: the deployment was declarable before only by
omitting `experts`, which gave the right ownership and no `ep` group at all.

V4, the degenerate identity (`tp=1, pp=1, dp=1`): the two settings of the flag
produce byte-identical manifests, the `ep` group is `[0]` at index 0 and the
ownership is the full range under both. This is the declared image of
`use_ep`'s own width test.

V5, the arange refusal following the resolved strategy: `tp=1, dp=8` with 4
experts and `round_robin` is refused with "num_experts 4 must be >= ep_size - 1
(7) under round_robin placement"; the same layout with `expert_parallel=False`
resolves to `linear`, builds, and gives all eight ranks the identical
`[0, 1, 2, 3]` (one distinct ownership row across the world). The framework
never reaches the failing `torch.arange` in such a run, so refusing there
would have rejected a layout the framework runs.

V6, the same deployment under SGLang: `declared_sglang_manifest(tp=8,
ep_size=8, experts=<expert_parallel=False>)` is refused with "expert_parallel
False has no SGLang spelling; declare ep_size=1 instead". The contrast at
`ep_size=1` builds: rank 5's `ep` group is the singleton `[5]` at index 0, it
owns the full `[0, ..., 31]`, and its `moe_tp` group is the whole tensor group
`[0, ..., 7]`. The vLLM twin of the same all-expert geometry gives rank 5 the
`ep` group `[0, ..., 7]`. One flag could not have served both builders,
because SGLang's all-expert geometry is a singleton group and vLLM's is the
whole stage.

V7, the fallback rows (`tp=1, dp=8`, 24 MoE layers, 32 experts, declared
`round_robin`): with `exceptions=None` and with an all-false object, rank 0
owns `[0, 8, 16, 24]` and rank 5 owns `[5, 13, 21, 29]` and the resolution is
`round_robin`. With any one of `single_expert_group`, `redundant_experts`,
`eplb` and `all2all_without_round_robin` true, and with all four true, rank 0
owns `[0, 1, 2, 3]`, rank 5 owns `[20, 21, 22, 23]` and the resolution is
`linear`. Six declarations, two frozen ownership rows each.

V8, a declared `linear` is never touched: with all six conditions true, rank 5
owns `[20, 21, 22, 23]`, identical to the plain layout's ownership, and the
resolution is `linear`. The four framework conditions are inert outside a
declared `round_robin`, mirroring the resolver's own first line.

V9, the rejection controls, each a `ValueError` naming the field:

| Control | Message |
|---|---|
| `eplb` with 30 experts at EP size 8 | `num_experts 30 must be divisible by ep_size 8 under eplb` |
| `model_uniform_expert_blocks` with 30 experts at EP size 8 | `num_experts 30 must be divisible by ep_size 8 under model_uniform_expert_blocks` |
| `model_refuses_tp_above_experts` at `tp=8` with 4 experts | `tp 8 must be <= num_experts 4 under model_refuses_tp_above_experts` |
| `expert_parallel=1` | `expert_parallel must be a bool, got 1` |
| `single_expert_group=1` | `single_expert_group must be a bool, got 1` |
| `exceptions="round_robin"` | `exceptions must be a DeclaredExpertMapExceptions or None, got 'round_robin'` |
| `expert_parallel=False` under the SGLang builder | `expert_parallel False has no SGLang spelling; declare ep_size=1 instead` |
| a non-default `exceptions` under the SGLang builder | `exceptions must be None under the SGLang expert map, which has no round_robin placement to fall back from` |

The matching build controls all built and none changed: 30 experts at EP size
8 without `eplb` and without `model_uniform_expert_blocks` reproduces the
PLACE-3 amendment's cell C8 exactly, per-rank counts `4, 4, 4, 4, 4, 4, 3, 3`
with rank 6 owning `[24, 25, 26]` and rank 7 `[27, 28, 29]`; and
`model_refuses_tp_above_experts` at `tp=4` with 32 experts leaves ownership
identical to the plain layout.

V10, the EPLB group: rank 9 of `tp=4, pp=2, dp=2` carries an `eplb` membership
with `global_ranks` `[0, 1, 2, 3, 8, 9, 10, 11]` at index 5, identical to its
`ep` membership, and every rank of the manifest mirrors its own `ep` group.
The `groups` keys are exactly `tp`, `pp`, `dp`, `ep`, `eplb` in that order,
and no `eplb` key exists without the condition.

V11, the two selections composed (`tp=1, dp=8`, `expert_parallel=False` with
`eplb=True`): at 32 experts the resolution is `linear` through the
`ep_size == 1` branch whatever the conditions say, every rank owns the full
range, and the `eplb` group equals the `ep` group `[0, ..., 7]`. At 30 experts
the same declaration builds rather than being refused, and every rank owns the
full `[0, ..., 29]`, because the framework's EPLB divisibility `ValueError` is
guarded by `use_ep`, which is false there. This is the row a reader is most
likely to get wrong, and it is the one cell whose failure would have voided
both tasks.

V12, wire identity: the V1 and V10 manifests round-trip through
`PlacementManifest.save` and `load` to equal objects, `source` is `declared`,
the group keys appear in the frozen order in every rank, and
`local_expert_ids` keys serialize ascending.

## Physical sanity

Nothing here is timed, so there is no measured value to bound. The napkin
check the freeze put in its place is a conservation count stated before the
run: with expert parallelism on, the owners of one MoE layer partition the
expert set, so a stage's ownership entries equal its MoE layer count times the
expert count; with it off, every rank of the stage owns every expert, so the
total is that number multiplied by `DP x TP`. Both held exactly. V1's 12,288
is `24 * 32 * 8` per stage over two stages, and V11's 6,144 is `24 * 32 * 8`
over one. An implementation landing on any other count would have changed the
geometry rather than its representation.

The floor a reader should check first is that switching expert parallelism off
can only add ownership and must never move a group. It did not: the `ep`
group's ranks and index are the accepted PLACE-3 values in every cell, and V3
asserts that field by field over all sixteen ranks. An implementation that
moved an EP rank while switching the flag would have misread
`initialize_model_parallel`, which does not consult the flag at all.

## Why nothing is scored

The scored denominator is zero, frozen that way before the run, and it is not
reported as a passing fraction of anything.

- The m5 identity that PLACE-3 and PLACE-13 both score declares `linear` at 32
  experts over a world that divides it, with no condition and with expert
  parallelism on. Neither selection can move one byte of its GOAL text or one
  picosecond of its six makespans, by construction. Running it a third time
  would have re-measured frozen numbers and answered nothing about the
  variants.
- The enabled PLACE-7 path is not live-reachable. The vLLM step schedule
  refuses a replicated geometry, because `dims.local_num_experts * len(ranks)`
  is `32 * 8` and not 32, and the routed expert snapshot refuses to project
  one at all. Under the repo's live-reachability rule a mechanism no supported
  path connects to the metric chain carries no metric claim, and manufacturing
  one here would have been a false closure.
- The evidence that the selections changed nothing reachable is therefore the
  two study `--check` arms, which are live, already scored in their own
  studies, and fatal here. Both reproduced.

## Choices the freeze left open

- **How the resolved strategy is exposed.** The freeze allowed either a
  documented manifest field or nothing new. The implementation exposes it only
  through `declared_resolved_placement_strategy(experts, ep_size)` and adds no
  manifest field: the schema has no strategy field today, adding one would
  move the bytes of every manifest ever emitted, and `local_expert_ids`
  already records the map that runs, which is the only thing a consumer can
  act on.
- **What the SGLang builder does with the two selections.** The freeze asked
  whether the builder should refuse `expert_parallel=False` or require
  `ep_size == 1` with it. It refuses, and refuses a non-default `exceptions`
  as well. Requiring `ep_size == 1` would have given one deployment two
  spellings whose emitted geometries differ, since SGLang's all-expert `ep`
  group is a singleton and vLLM's is the whole stage. V6 freezes that
  divergence as the reason.
- **Whether `eplb=True` emits the `eplb` group.** The freeze decided it
  should, against the narrower reading that PLACE-11 owns only the expert map.
  The reasoning is the same one PLACE-7 rests on: a declared manifest is
  meant to equal an extracted one, `initialize_model_parallel` really creates
  that group, and omitting it would have left the same group-inventory gap the
  other half of this slice exists to close.
- **When the round-robin arange refusal is evaluated.** Against the resolved
  strategy, not the declared one, so a layout refused before this slice builds
  once a condition forces `linear`. V5 is the frozen row. No existing manifest
  moves, because the resolution is the identity without the new fields.

## Reproduction

```bash
source .env.local.sh
python examples/declared_expert_variants_v1/run_study.py
python examples/declared_expert_variants_v1/run_study.py --check --output-root <fresh>
```

`SIMLLM_HTSIM_RNIC` and `SIMLLM_TXT2BIN` select the pinned binaries that the
two nested study checks need, `SIMLLM_SGLANG_ENV` selects the interpreter that
answers the PLACE-13 oracle arm inside its check, and `SIMLLM_DATA_ROOT` or
`--output-root` owns the bulk artifacts. The harness refuses to run unless the
freeze and both amendments are ancestors of `HEAD`. The check mode rebuilds
every cell and compares the summary to the tracked results, ignoring only the
implementation commit stamp.
