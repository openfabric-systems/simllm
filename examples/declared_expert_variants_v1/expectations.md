# Declared expert variant expectations

Date: 2026-09-15

This is the expectations-only freeze for PLACE-7 and PLACE-11. It precedes the
builder changes, their tests, the study harness and every result-producing run.
Both tasks add a declared selection that the accepted builder cannot express
today, and both must leave every manifest that does not name the selection byte
identical. The declared layout is a what-if placement computed from the pinned
framework's layout rules; it is not an extracted record of a live run.

## Question

Can `DeclaredExpertLayout` state the two mixture-of-experts (MoE) deployments
that the pinned vLLM 0.27.1 really runs and the declared builder currently
cannot describe, namely a MoE model with expert parallelism switched off and a
model whose framework-side conditions downgrade a declared `round_robin` map to
`linear`, so that a declared manifest and a manifest extracted from such a run
agree on group inventory and on expert ownership, without changing one byte of
any manifest that does not ask for either selection?

## Why the two tasks share one freeze

They are the same seam read twice. PLACE-7 turns expert parallelism off, and
the pinned framework's own strategy resolver then returns `linear` through its
`ep_size == 1` branch, which is the first line of PLACE-11's fallback. PLACE-11
declares the model-scoped conditions under which the same resolver returns
`linear` with expert parallelism still on. Freezing them apart would state that
shared branch twice and leave the composition of the two unspecified, which is
exactly the row this freeze has to pin (cell V11). Both are Completeness P2
tasks whose off path is the accepted baseline, so they share one compatibility
identity and one refusal family.

## Frozen source and compatibility identity

The implementation starts from commit
`2a4abea67b9b1ce793781b76f330574059ffe817`. The JSON registry records the
pre-change SHA-256 identities of the declared builder, the placement package
entry, the manifest module, the focused declared-expert test, the PLACE-3
harness, the vLLM step schedule and the placement registry.

The compatibility authority is the UTF-8 JSON emitted by
`PlacementManifest.save`, including its terminal line feed. The five reference
manifests of the PLACE-3 freeze must keep their byte lengths and digests:

| Record | Builder call | Bytes | SHA-256 |
|---|---|---:|---|
| worked example | `declared_manifest(tp=4, pp=2, dp=2)` | 10,832 | `2e46eadcaccd83de1778ea98c368585b452b83067489b9a2df064e25a14d857e` |
| m4 tensor group | `declared_manifest(tp=8)` | 5,698 | `3812aef241d93f2af8d86da0f6bbb606dae9c43c768ceb3efb8d649dc3bacfd6` |
| rail pipeline | `declared_pipeline_placement(8)` | 52,310 | `8d38cf4b6990bfd75fc90dcfe994180c3c1150adce297b6f71983fd8c9c877db` |
| m5 expert world | `declared_manifest(tp=1, dp=8)` | 5,698 | `0894fae1687217d88466b5692034a8d1863fb8c658afdfde2c317d1c412f60a5` |
| width tail | `declared_manifest(tp=64, nodes=8, gpus_per_node=8)` | 102,050 | `bed8d26c72feaf117f48b651ec18b4fb0caf39df2e47866a550f74252972aecd` |

The five digests were recomputed at the baseline commit and agree with the
PLACE-3 and PLACE-13 freezes, which is what lets this slice reuse them. The
PLACE-13 slice adds a second entry point rather than an option and therefore
does not move any of the five; the identity holds whether or not that slice has
landed when this one starts, and if it has landed then the declared builder,
the placement package entry and the placement registry carry the digests the
PLACE-13 freeze records instead of the ones below.

Two study `--check` runs are the second fatal identity: the PLACE-3 declared
expert placement study and, once it has landed, the PLACE-13 SGLang declared
layout study must both reproduce their tracked results with no change. Both
need the backend binaries named by `SIMLLM_HTSIM_RNIC` and `SIMLLM_TXT2BIN`,
which come from the local environment file and never from a tracked file.

Both identities are fatal and unscored. A manifest built without the two new
selections must be byte identical to the pre-change output, and a manifest
built with either present must differ only in the fields this freeze names. A
single violation voids the run, and the owning task stays open.

## Pinned framework semantics

The rules below are read from the installed vLLM 0.27.1 package and are the
specification of the two declared selections. File names are given relative to
the installed `vllm` package. Prefill context parallelism (PCP) is one
throughout, and external data parallelism is one.

### PLACE-7: the EP group exists with expert parallelism off

- Group creation. `distributed/parallel_state.py`
  (`initialize_model_parallel`) creates the `ep` group under the single
  condition `config.model_config is None or config.model_config.is_moe`,
  commented "Don't create EP group for dense models." The
  `enable_expert_parallel` flag does not appear in that condition, so a MoE
  model gets the group whether or not expert parallelism is on. Its ranks are
  the same transpose as today, `all_ranks.transpose(1, 2).reshape(-1,
  data_parallel_size * prefill_context_model_parallel_size *
  tensor_model_parallel_size)`, so the group holds the `DP * TP` ranks of one
  pipeline stage, data-parallel major and tensor-parallel minor, and a
  member's index in it is `dp * TP + tp`. This is the accepted declared EP
  group, unchanged.
- The off switch. `model_executor/layers/fused_moe/config.py`
  (`FusedMoEParallelConfig.make`) computes
  `use_ep = dp_size_ * pcp_size_ * tp_size_ > 1 and
  vllm_parallel_config.enable_expert_parallel`. Two things follow. Expert
  parallelism is off when the launcher does not ask for it, and it is also off
  when the flattened width `DP * PCP * TP` is one, whatever the launcher asks.
- What replaces it. In the `if not use_ep` branch the returned config keeps
  `tp_size` and `tp_rank` from
  `FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp`, which returns
  `flatten_tp_size = dp_size * pcp_size * tp_size` and
  `flatten_tp_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size +
  tp_rank`, and sets `ep_size=1`, `ep_rank=0`, `use_ep=False`. So the MoE
  layer's tensor width is the whole flattened `DP * TP` group, not the tensor
  group alone.
- Tensor sharding. `FusedMoEConfig.__post_init__` in the same file asserts
  `self.intermediate_size % tp_size == 0` and sets
  `self.intermediate_size_per_partition = self.intermediate_size // tp_size`
  against that flattened `tp_size`, so every expert's MLP intermediate
  dimension is cut into `DP * TP` shards, one per rank of the stage.
- Local experts. `model_executor/layers/fused_moe/expert_map_manager.py`
  (`determine_expert_map`) returns `(global_num_experts, None, None)` at
  `ep_size == 1`, and `model_executor/layers/fused_moe/layer.py` passes
  `num_local_experts=expert_map_manager.local_num_experts` into the layer. The
  expert map is `None`, so there is no local-to-global renumbering: every rank
  of the stage holds every expert of every MoE layer it owns.

The declared selection therefore emits the same `ep` group as today and the
full `[0, num_experts)` list for every MoE layer of the stage. The tensor shard
of each expert is deliberately not represented: the placement manifest schema
records which global expert ids live on a rank, not which slice of an expert's
weights lives there. That omission is stated here so a reader does not mistake
all-expert ownership for unsharded experts.

### PLACE-11: the framework-side fallback and the model-scoped rules

- The resolver. `expert_map_manager.py`
  (`determine_expert_placement_strategy`) is called from
  `ExpertMapManager.__init__` only when `moe_parallel_config.use_ep`. For a
  requested `round_robin` it computes
  `round_robin_supported = (num_expert_group is not None and num_expert_group
  > 1) and num_redundant_experts == 0 and not enable_eplb` and returns
  `"linear"` when that is false, logging that round-robin placement "is only
  supported for models with multiple expert groups and no redundant experts".
  It then returns `"linear"` a second time when
  `moe_parallel_config.use_all2all_kernels and not
  moe_parallel_config.needs_round_robin_routing_tables`.
- The backend condition. In `config.py`,
  `use_all2all_kernels = self.use_ep and (self.dp_size > 1 or self.pcp_size >
  1 or self.is_sequence_parallel)` and
  `needs_round_robin_routing_tables = self.use_deepep_ll_kernels or
  self.use_nixl_ep_kernels`, so the second fallback fires whenever an
  all-to-all path is in use and its backend is neither `deepep_low_latency`
  nor `nixl_ep`.
- The second resolver. `ExpertMapManager._determine_placement_strategy` runs
  unconditionally on the result. For a `round_robin` request it returns
  `"linear"` when `self.ep_size == 1`, and again when the same all-to-all
  condition holds. The `ep_size == 1` branch is the line PLACE-7 shares.
- The EPLB divisibility refusal. `layer.py` raises
  `ValueError("EPLB currently only supports even distribution of experts
  across ranks. ...")` when `use_ep and global_num_experts % ep_size != 0`,
  and only inside its `if enable_eplb:` branch. The same branch's `else` arm
  asserts `num_redundant_experts == 0`, so redundant experts exist only under
  EPLB. This is the refusal the PLACE-3 amendment of 2026-09-13 correctly
  scoped to EPLB, and it is guarded by `use_ep`, so it cannot fire with expert
  parallelism off.
- The model-scoped refusal. Sixteen sparse-MoE blocks under
  `model_executor/models` raise
  `ValueError(f"Tensor parallel size {self.tp_size} is greater than the number
  of experts {config.num_experts}.")`, among them `qwen3_moe.py`
  (`Qwen3MoeSparseMoeBlock`), `mimo_v2.py`, `laguna.py`, `cohere2_moe.py` and
  `minimax_m2.py`. The compared width is `self.tp_size =
  get_tensor_model_parallel_world_size()`, the declared tensor width, not the
  expert-parallel size.
- The model-scoped uniform-block assumption. The `MixtureOfExperts` bookkeeping
  in `deepseek_v2.py`, `mixtral.py`, `qwen3_moe.py` and the
  `transformers/moe.py` wrapper computes
  `n_local_physical_experts = n_physical_experts // ep_size` and
  `physical_expert_start = ep_rank * n_local_physical_experts`, a uniform block
  per rank. `gpt_oss.py` does the same in its weight loader with
  `experts_per_rank = num_experts // ep_size`. None of these raises: with a
  remainder they silently disagree with `determine_expert_map`, which hands
  `base + 1` experts to the ranks below the remainder.

Two statements in the registry entries do not survive this reading, and the
declared selection follows the source rather than the entries:

- PLACE-11 says the fallback applies when "the model has at most one expert
  group". The source condition is `num_expert_group is not None and
  num_expert_group > 1`, so a model that does not set the field at all, which
  is every model outside the grouped-topk family, also falls back. The
  declared boolean is named for the condition, not for the count.
- PLACE-11 says "some model implementations refuse expert counts their
  expert-parallel size does not divide". No model does. The literal
  model-scoped refusal is `tp_size > num_experts`, and the divisibility
  refusal lives in the fused MoE layer behind EPLB. What the model
  implementations carry is the silent uniform-block assumption above. The
  declared selection therefore offers one boolean for the real refusal and one
  for the uniform-block assumption, and the second refuses a remainder rather
  than modeling the framework's silent disagreement, so the declared manifest
  never records ownership the model implementation would not honor.

## Declared interface

```text
DeclaredExpertMapExceptions(
    single_expert_group: bool = False,          # num_expert_group unset or 1
    redundant_experts: bool = False,            # num_redundant_experts > 0
    eplb: bool = False,                         # enable_eplb
    all2all_without_round_robin: bool = False,  # all2all path, not deepep-ll or nixl-ep
    model_uniform_expert_blocks: bool = False,  # n_physical_experts // ep_size blocks
    model_refuses_tp_above_experts: bool = False,  # tp_size > num_experts raises
)

DeclaredExpertLayout(
    num_layers: int,
    moe_layers: tuple[int, ...],
    num_experts: int,
    placement_strategy: str = "linear",
    placement_epoch: int = 0,
    expert_parallel: bool = True,
    exceptions: DeclaredExpertMapExceptions | None = None,
)

declared_resolved_placement_strategy(
    experts: DeclaredExpertLayout, ep_size: int
) -> str
```

`DeclaredExpertMapExceptions` is frozen, hashable and all-false by default. The
two new `DeclaredExpertLayout` fields default to the accepted behavior, so
every call written today keeps its meaning and its bytes.

The resolved strategy is

```text
resolved = placement_strategy
if placement_strategy == "round_robin" and (
    not expert_parallel
    or ep_size == 1
    or (exceptions is not None and (
        exceptions.single_expert_group
        or exceptions.redundant_experts
        or exceptions.eplb
        or exceptions.all2all_without_round_robin))
):
    resolved = "linear"
```

with `ep_size = DP * TP` under the vLLM builder. `declared_resolved_placement_strategy`
is the single authority for that rule and the only place the resolution is
exposed. Nothing new is added to the manifest: the manifest schema carries no
placement-strategy field today, adding one would move every manifest's bytes,
and `local_expert_ids` already records the map that really runs, which is the
only thing a consumer can act on. A study that wants to report the resolution
calls the function.

With `expert_parallel=False` under the vLLM builder, every rank keeps the `ep`
group of its stage with the same `global_ranks` and the same `rank_in_group` as
today, and its `local_expert_ids` for every MoE layer of its stage is the full
`list(range(num_experts))`. The tensor shard of each expert is not represented.
With `eplb=True` every rank additionally carries an `eplb` membership,
inserted after `ep`, with the same `global_ranks` and the same `rank_in_group`
as its `ep` membership, because `initialize_model_parallel` creates that second
group with the same rank lists under `if config.parallel_config.enable_eplb`.
A declared manifest that omitted it would leave the same group-inventory gap
between a declared and an extracted manifest that PLACE-7 exists to close.
The redundant physical experts that EPLB permits are not represented: a
declared layout states logical expert ids only, and `redundant_experts=True`
declares the condition for the fallback, not the extra experts.

The refusal of a `round_robin` layout whose expert count the framework's
`torch.arange(r, num_experts, ep_size)` cannot serve is evaluated against the
resolved strategy, not the declared one. A layout that is refused today
therefore builds when a selection forces `linear`, because the framework never
reaches that arange in those runs. This changes no existing manifest, since the
resolution is the identity without the new fields.

Construction refuses, with a `ValueError` naming the field: `expert_parallel`
or any `DeclaredExpertMapExceptions` field spelled as anything but a bool
(1 and 0 are not booleans here, mirroring the existing rule that True is not a
width of one); `exceptions` that is neither `None` nor a
`DeclaredExpertMapExceptions`; `eplb=True` with `num_experts % ep_size != 0`
while `expert_parallel` is true; `model_uniform_expert_blocks=True` with
`num_experts % ep_size != 0`; and `model_refuses_tp_above_experts=True` with
`tp > num_experts`. Every field-local refusal of PLACE-3 and its amendment
applies unchanged.

The SGLang builder `declared_sglang_manifest` refuses both selections with a
`ValueError` naming the field. SGLang's own off switch is `--ep-size 1`, which
the builder already takes: at `ep_size=1` the EP group is the rank's own
singleton and the block map is the full expert range, so `expert_parallel` has
nothing left to say there and its vLLM meaning, a group that still spans
`DP * TP` while every rank owns everything, has no SGLang counterpart. The
exceptions object is refused because the pinned SGLang expert map has no
round-robin placement at all and the builder already refuses `round_robin`, so
no SGLang run can take the fallback these booleans describe.

## Frozen cells

Cell V1, the PLACE-7 worked example: `tp=4, pp=2, dp=2` with 48 layers, all 48
carrying MoE blocks, 32 experts, `expert_parallel=False`. The EP size is 8.
Rank 9 is `(dp=1, pp=0, tp=1)`: its `ep` group is `[0, 1, 2, 3, 8, 9, 10, 11]`
with `rank_in_group` 5 and its layer range is `[0, 24)`, all three unchanged
from the accepted output, and its `local_expert_ids` keys are exactly the
layers 0 through 23, each mapping to the full `[0, 1, ..., 31]`. Rank 15 is
`(dp=1, pp=1, tp=3)`: `ep` group `[4, 5, 6, 7, 12, 13, 14, 15]`,
`rank_in_group` 7, layer range `[24, 48)`, and the same full list in each of
layers 24 through 47. With `expert_parallel=True` the same two ranks own
`[20, 21, 22, 23]` and `[28, 29, 30, 31]`, which is the accepted PLACE-3 cell
C1. Summed over all 16 ranks the ownership entries number exactly
`16 * 24 * 32 = 12,288`, and every `(layer, expert)` pair of a stage is owned
by all 8 ranks of that stage.

Cell V2, the off-path identity: `expert_parallel` defaults to true and
`exceptions` defaults to `None`, so every builder call written before this
slice keeps its bytes. The five reference digests above are the guard, and a
`DeclaredExpertLayout` built without naming either field compares equal to the
same layout built before the slice.

Cell V3, the group-inventory gap PLACE-7 names: the V1 manifest and
`declared_manifest(tp=4, pp=2, dp=2)` with no `experts` agree field by field on
`global_rank`, `hostname`, `local_rank` and the `tp`, `pp` and `dp`
memberships of every rank, and differ exactly in the added `ep` membership, the
`pipeline_layer_range`, the `local_expert_ids` and, when the declared epoch is
not zero, the `placement_epoch`. Today that deployment can only be declared by
omitting `experts`, which gives the right ownership and the wrong group
inventory; this cell is what closing the gap means.

Cell V4, the degenerate identity: at `tp=1, pp=1, dp=1` with 24 MoE layers and
32 experts, the two settings of `expert_parallel` produce byte-identical
manifests. The `ep` group is `[0]` with `rank_in_group` 0 and the ownership is
the full `[0, 1, ..., 31]` in each of the 24 layers under both. This is the
declared image of `use_ep = dp_size_ * pcp_size_ * tp_size_ > 1 and
enable_expert_parallel`, which is false at a flattened width of one whatever
the launcher asks for.

Cell V5, PLACE-7 and the round-robin arange: `tp=1, dp=8` with 24 MoE layers,
4 experts and `round_robin` is refused today with "num_experts 4 must be >=
ep_size - 1 (7) under round_robin placement". The same layout with
`expert_parallel=False` builds, its resolved strategy is `linear`, and every
one of the 8 ranks owns `[0, 1, 2, 3]` in every one of the 24 layers, because
the framework's `torch.arange` is never reached at `ep_size == 1`.

Cell V6, PLACE-7 under SGLang: `declared_sglang_manifest(tp=8, ep_size=8,
experts=<48 layers, 32 experts, expert_parallel=False>)` is refused with a
`ValueError` naming `expert_parallel` and pointing at `ep_size=1`. For
contrast, `declared_sglang_manifest(tp=8, ep_size=1, experts=<the same layout
with the default flag>)` builds: every rank's `ep` group is its own singleton
`[rank]` with `rank_in_group` 0 and it owns the full `[0, 1, ..., 31]` in every
layer, and its `moe_tp` group is the whole tensor group `[0, ..., 7]`. The
frozen contrast is that SGLang's all-expert geometry gives a singleton `ep`
group where vLLM's gives the `DP * TP` group of the stage, which is why one
flag cannot serve both builders.

Cell V7, the PLACE-11 fallback rows: `tp=1, pp=1, dp=8` with 24 MoE layers, 32
experts and a declared `round_robin`, so EP size 8. Rank 5 owns
`[5, 13, 21, 29]` and rank 0 owns `[0, 8, 16, 24]` with `exceptions=None` and
with an all-false `DeclaredExpertMapExceptions`. With any one of
`single_expert_group`, `redundant_experts`, `eplb` and
`all2all_without_round_robin` true on its own, and with all four true together,
rank 5 owns `[20, 21, 22, 23]` and rank 0 owns `[0, 1, 2, 3]`, and
`declared_resolved_placement_strategy` returns `linear`. That is six
declarations and two frozen ownership rows each.

Cell V8, the exceptions never touch a declared `linear`: with
`placement_strategy="linear"` and all six booleans true, the V7 world's rank 5
owns `[20, 21, 22, 23]` and `declared_resolved_placement_strategy` returns
`linear`, exactly as with `exceptions=None`. The two model-scoped booleans add
refusals and nothing else; the four framework booleans are inert outside a
declared `round_robin`, which mirrors the resolver's first line,
`if requested_strategy != "round_robin": return requested_strategy`.

Cell V9, refusals, each a `ValueError` naming the field and raised before the
first rank is built: `eplb=True` with 30 experts at EP size 8;
`model_uniform_expert_blocks=True` with 30 experts at EP size 8;
`model_refuses_tp_above_experts=True` at `tp=8` with 4 experts;
`expert_parallel=1`; `single_expert_group=1`; `exceptions="round_robin"`;
`expert_parallel=False` under the SGLang builder; and a non-default
`exceptions` under the SGLang builder. The matching controls must build and
must not change: 30 experts at EP size 8 with `eplb=False` and
`model_uniform_expert_blocks=False` reproduce the amendment's cell C8 exactly,
per-rank counts `4, 4, 4, 4, 4, 4, 3, 3`, rank 6 owning `[24, 25, 26]` and rank
7 owning `[27, 28, 29]` under `linear`; and `model_refuses_tp_above_experts=True`
at `tp=4` with 32 experts builds unchanged.

Cell V10, the EPLB group: with `eplb=True` at `tp=4, pp=2, dp=2`, rank 9
carries an `eplb` membership with `global_ranks` `[0, 1, 2, 3, 8, 9, 10, 11]`
and `rank_in_group` 5, identical to its `ep` membership, and the `groups` keys
of every rank are exactly `tp`, `pp`, `dp`, `ep`, `eplb` in that order. With
`eplb=False` no rank carries an `eplb` key.

Cell V11, the two selections composed: `tp=1, dp=8` with 24 MoE layers and an
`exceptions` object carrying `eplb=True`, under `expert_parallel=False`. At 32
experts the resolved strategy is `linear` through the `ep_size == 1` branch
whatever the exceptions say, every rank owns the full `[0, 1, ..., 31]` in
every layer, and the `eplb` group of each rank equals its `ep` group
`[0, 1, ..., 7]`. At 30 experts the same declaration builds rather than being
refused, and every rank owns the full `[0, 1, ..., 29]`, because the
framework's EPLB divisibility `ValueError` is guarded by `use_ep`, which is
false here. That last row is the one a reader is most likely to get wrong, and
it is frozen as a literal.

Cell V12, wire identity: the V1 and V10 manifests round-trip through
`PlacementManifest.save` and `PlacementManifest.load` to equal objects;
`source` is `declared`; `ep` follows `dp` and `eplb` follows `ep` in every
rank's `groups`; and `local_expert_ids` keys serialize in ascending layer
order.

Cell V13, the consumer consequence, recorded as a finding rather than as a
builder claim: over the `tp=1, dp=8` manifest of cell V11 at 32 experts,
`ExpertPlacementSnapshot.from_manifest(manifest, ep_ranks)` carries exactly
`8 * 24 * 32 = 6,144` owner entries, and `owner_map()` collapses them to 768
keys, so seven of every eight entries are lost by a projection that assumes one
owner per `(layer, expert)` pair. In the same geometry the vLLM step schedule
refuses the layout, because `dims.local_num_experts * len(ranks)` is
`32 * 8`, not 32. Both are stated as the frozen state of the consumers on the
day of the freeze, not as behavior this slice changes.

## Physical sanity before observation

Nothing in this freeze is timed, so there is no measured value to bound. The
napkin check that replaces it is a conservation count, stated before the run:
with expert parallelism on, the owners of one MoE layer partition the expert
set and the total ownership entries of a stage equal `num_experts` times its
MoE layer count; with it off, every rank of the stage owns every expert and the
total is that number multiplied by `DP * TP`. Cell V1's 12,288 entries are
`48 / 2 * 32 * 8 * 2` stages, and cell V11's 6,144 are `24 * 32 * 8`. An
implementation that lands on any other count has changed the geometry, not the
representation, and the run is void.

The floor a reader should check first is that the enabled PLACE-7 path can only
add ownership, never move a group: the `ep` group's `global_ranks` and
`rank_in_group` are the accepted PLACE-3 values in every cell above, and cell
V3 asserts that field by field. If an implementation moves an EP rank while
switching expert parallelism off, it has misread
`initialize_model_parallel`, which does not consult the flag at all.

## Why this study is structural only

No live cell is warranted, and the study runs no backend. Three reasons, all
frozen before the run:

- The m5 identity that PLACE-3 and PLACE-13 both score uses `linear` at 32
  experts over a world that divides it, with no exception declared and expert
  parallelism on. Neither selection can move one byte of its Group Operation
  Assembly Language (GOAL) text or one picosecond of its six makespans, by
  construction: the resolution is the identity and the ownership is unchanged.
  Running it a third time would re-measure frozen numbers and answer nothing
  about the variants.
- The enabled PLACE-7 path is not live-reachable today. Cell V13 records why:
  the vLLM step schedule refuses a replicated expert geometry, and the routed
  expert snapshot's `owner_map` is single-owner. Under the repo's
  live-reachability rule a mechanism that no supported path connects to the
  metric chain carries no metric claim, so manufacturing one here would be a
  false closure.
- The evidence that the two selections changed nothing reachable is exactly the
  two study `--check` runs named above, which are live, already scored, and
  fatal here.

The scored denominator of this study is therefore zero, deliberately. It is not
reported as a passing fraction of anything, and closure below is stated in
terms of structural exactness and the two fatal identities only.

## Evidence accounting and closure

V1, V3, V4, V5, V6, V7, V8, V10, V11 and V12 are structural exact guards. V9 is
a rejection control family with its matching build controls. V2, the five
reference digests and the PLACE-3 and PLACE-13 study `--check` runs are fatal,
unscored by-construction identities. V13 is a recorded finding about the
consumers and asserts their current behavior, not a claim about this slice.
Counts from these classes are never added into one total, and no class here is
scored.

PLACE-7 closes only if every V1 through V6 literal is exact, the field-level
comparison of V3 holds, the SGLang refusal of V6 fires, and the five digests
and the two study checks remain exact. Its closure is a statement about
manifest content: a declared manifest of an expert-parallel-disabled MoE
deployment now carries the same group inventory and the same ownership as an
extracted one. It is explicitly not a claim that the enabled path drives a
metric; cell V13 names the two consumers that refuse it, and the live arm stays
with PLACE-8, which owns the non-uniform and replicated expert geometries in
the step-sink consumers.

PLACE-11 closes only if every V7 through V11 literal is exact, the refusals and
the build controls of V9 both behave as frozen, and the same identities remain
exact. Its closure makes the module docstring's sentence that the framework's
own fallback "is not modeled here" false, so that sentence is rewritten in the
closing change rather than left standing.

If either task's cells fail, that task stays open and the other may still
close, because their cells are disjoint except for V11, which belongs to both:
a V11 failure voids both.

The implementation registers residual tasks for what these two selections
deliberately leave out: the tensor shard of an expert under disabled expert
parallelism, which no manifest field can express today; the redundant physical
experts EPLB creates, which the declared layout states only as a condition; and
the sequence-parallel and prefill-context-parallel terms of
`use_all2all_kernels`, which this freeze folds into one declared boolean rather
than modeling as widths. PLACE-3, PLACE-8, PLACE-10 and PLACE-13 are otherwise
untouched.
