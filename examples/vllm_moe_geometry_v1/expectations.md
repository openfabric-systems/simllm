# vLLM MoE geometry and expert group v1 expectations

This document freezes the VLLM-6 expectations before the geometry fix, the
expert-group derivation, the sink binding, the study harness, or any measured
result exists. The repository source at this boundary is commit
`aeb40ac95cdd8163942297335948c94df0376e04`.

The evidence is authored against official vLLM v0.26.0 commit
`568afb3a13806beb53bb2e6bd518269357b237c0`. A run records the source files and
version it actually observes. The authored-against identity and the observed
identity are independent provenance fields; no check requires a live checkout,
package, or submodule pin to equal the authored-against commit.

An earlier change closed part of this clause with a post-specified component
test and was reverted. Every geometry fact below is read off the pinned vLLM
v0.26.0 source with file and line citations, not off a plausible account of
how MoE layers are usually shaped.

## Pinned-source audit

Paths and line numbers are relative to the installed `vllm` package.

- `model_executor/models/granitemoe.py:247-254` builds the routed block with
  `num_experts=config.num_local_experts`, `top_k=config.num_experts_per_tok`,
  `hidden_size=config.hidden_size` and `intermediate_size=config.intermediate_size`.
  The Hugging Face field named `num_local_experts` is the GLOBAL expert count of
  a Granite MoE layer, and the expert intermediate size is the model's plain
  `intermediate_size`, i.e. Granite has no separate `moe_intermediate_size`
  field.
- `model_executor/layers/fused_moe/config.py:1113-1121`
  (`flatten_tp_across_dp_and_pcp`) defines
  `flatten_tp_size = dp_size * pcp_size * tp_size` and
  `flatten_tp_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank`.
- `model_executor/layers/fused_moe/config.py:1204-1207` defines
  `use_ep = dp_size * pcp_size * tp_size > 1 and enable_expert_parallel`.
  Enabling the flag in a world of one device does not enable expert
  parallelism.
- `model_executor/layers/fused_moe/config.py:1217-1231` is the no-EP branch: it
  keeps `tp_size = flatten_tp_size` and sets `ep_size = 1`. Without expert
  parallelism the experts are tensor-sharded across the FLATTENED
  `dp * pcp * tp` device set, not across `tp` alone.
- `model_executor/layers/fused_moe/config.py:1233-1252` is the EP branch: it
  sets `tp_size = 1`, `ep_size = flatten_tp_size` and
  `ep_rank = flatten_tp_rank`. Under expert parallelism a device owns whole
  experts and the expert weights are NOT tensor-sharded.
- `model_executor/layers/fused_moe/config.py:1324-1329` computes
  `intermediate_size_per_partition = intermediate_size // moe_parallel_config.tp_size`,
  which is the MoE-local `tp_size` from the two branches above.
- `model_executor/layers/fused_moe/layer.py:73-96` (`determine_expert_counts`)
  sets `global_num_experts = num_experts + num_redundant_experts`. Redundant
  experts require EPLB; `config/parallel.py:70` carries
  `EPLBConfig.num_redundant_experts` with default 0.
- `model_executor/layers/fused_moe/expert_map_manager.py:62-69`
  (`determine_expert_map`) returns the whole global count when `ep_size == 1`,
  and otherwise `base = global // ep_size`, `remainder = global % ep_size`,
  `local_num_experts = base + 1 if ep_rank < remainder else base`. An uneven
  division is legal for the expert map; only EPLB rejects it
  (`layer.py:249-256`).
- `distributed/parallel_state.py:1789-1801` fixes the rank layout order as
  `ExternalDP x DP x PP x PCP x TP` and builds
  `all_ranks = arange(world).reshape(-1, DP, PP, PCP, TP)`.
- `distributed/parallel_state.py:1893-1919` builds the EP group as
  `all_ranks.transpose(1, 2).reshape(-1, DP * PCP * TP).unbind(0)`, and only
  when `model_config.is_moe`. The EP group of a rank is therefore every rank
  sharing its ExternalDP index and its PP index, ordered by `(dp, pcp, tp)`.
- `config/parallel.py:328` carries `ParallelConfig.rank`, the global rank the
  geometry reader can key its per-rank answers on.

Audited file identities:

| Source file | SHA-256 |
|---|---|
| `vllm/model_executor/models/granitemoe.py` | `b60e452c3f28b25aa104c88869daa25c06a7fb6ed45bd34e908fa6a8395efda1` |
| `vllm/model_executor/layers/fused_moe/config.py` | `f229a423db043aed9f0f7e09d586f82c5a05cbba5871784f0f46500059c745e6` |
| `vllm/model_executor/layers/fused_moe/layer.py` | `c888133b47507d6dc0a8e74671ef6e6b2c60c53e615e27098e354ad5b961d725` |
| `vllm/model_executor/layers/fused_moe/expert_map_manager.py` | `308c00f63fcb6f44518ec3a56d8f902c7bba75928c2e3aea30fbe1f3f473973b` |
| `vllm/distributed/parallel_state.py` | `ecfa5eeda697e9982591c93eca835fcc53463267c1c22110700819efcecee74f` |
| `vllm/config/parallel.py` | `a6581c267ab265e24905d2f5caa514482c28359f71380c6f894ceab25aa22541` |

## Frozen mapping, stated as closed forms

For a config with `dp`, `pcp`, `tp`, `enable_expert_parallel`, global `rank`,
text config fields `num_local_experts`, `num_experts_per_tok`,
`intermediate_size`, and `eplb_config.num_redundant_experts`:

```
moe                 = num_local_experts is not None and num_local_experts > 0
num_experts         = 0 if not moe else num_local_experts + num_redundant_experts
top_k               = 0 if not moe else num_experts_per_tok
flatten_tp          = dp * pcp * tp
use_ep              = moe and enable_expert_parallel and flatten_tp > 1
ep_size             = flatten_tp if use_ep else 1
moe_tp_size         = 1 if use_ep else flatten_tp
moe_intermediate    = None if not moe else intermediate_size // moe_tp_size
ep_rank             = flatten index of rank inside its EP group
base, remainder     = divmod(num_experts, ep_size)
local_num_experts   = 0 if not moe else (base + 1 if ep_rank < remainder else base)
```

The dense (non-MoE) `intermediate_size` field of `ModelDims` keeps its existing
`intermediate_size // tp` meaning; only the MoE per-expert size follows
`moe_tp_size`.

The EP group of global rank `r`, when the model is MoE, in the
`ExternalDP x DP x PP x PCP x TP` layout with `S = PCP * TP`:

```
block   = DP * PP * S
extdp   = r // block
within  = r %  block
pp      = (within // S) % PP
d, c, t enumerate DP, PCP, TP in that order
member(d, c, t) = extdp * block + ((d * PP + pp) * PCP + c) * TP + t
ep_ranks = tuple(member(d, c, t) for d in range(DP) for c in range(PCP) for t in range(TP))
ep_rank  = index of r inside ep_ranks
```

### Frozen cells

Granite `ibm-granite/granite-3.0-1b-a400m-instruct` shape: 24 layers,
`hidden_size = 1024`, `intermediate_size = 512`, `num_local_experts = 32`,
`num_experts_per_tok = 8`, `num_redundant_experts = 0`. The world is always
`dp * pp * pcp * tp` with `ExternalDP = 1` and `pp = 1` unless a cell says
otherwise.

| Cell | dp | pcp | tp | EP flag | rank | num_experts | top_k | moe_intermediate_size | local_num_experts | ep_ranks |
|---|---|---|---|---|---|---|---|---|---|---|
| `g-dense-world1` | 1 | 1 | 1 | off | 0 | 32 | 8 | 512 | 32 | `(0,)` |
| `g-ep-off-tp2` | 1 | 1 | 2 | off | 1 | 32 | 8 | 256 | 32 | `(0, 1)` |
| `g-ep-off-dp2-tp2` | 2 | 1 | 2 | off | 3 | 32 | 8 | 128 | 32 | `(0, 1, 2, 3)` |
| `g-ep-flag-world1` | 1 | 1 | 1 | on | 0 | 32 | 8 | 512 | 32 | `(0,)` |
| `g-ep-dp8` | 8 | 1 | 1 | on | 5 | 32 | 8 | 512 | 4 | `(0,...,7)` |
| `g-ep-dp2-tp2` | 2 | 1 | 2 | on | 3 | 32 | 8 | 512 | 8 | `(0, 1, 2, 3)` |
| `g-ep-dp2-pcp2-tp2` | 2 | 2 | 2 | on | 7 | 32 | 8 | 512 | 4 | `(0,...,7)` |
| `g-ep-dp8-uneven` | 8 | 1 | 1 | on | 0 | 30 | 8 | 512 | 4 | `(0,...,7)` |
| `g-ep-dp8-uneven-hi` | 8 | 1 | 1 | on | 7 | 30 | 8 | 512 | 3 | `(0,...,7)` |
| `g-ep-dp2-eplb` | 2 | 1 | 1 | on | 1 | 34 | 8 | 512 | 17 | `(0, 1)` |
| `llama-dense` | 1 | 1 | 1 | off | 0 | 0 | 0 | `None` | 0 | `None` |
| `llama-dense-tp2` | 1 | 1 | 2 | off | 1 | 0 | 0 | `None` | 0 | `None` |

`g-ep-dp8-uneven` uses `num_local_experts = 30` so that
`divmod(30, 8) == (3, 6)`: ranks 0 through 5 own 4 experts and ranks 6 and 7
own 3. `g-ep-dp2-eplb` uses `num_local_experts = 32` with
`num_redundant_experts = 2` under EPLB, so `global = 34` and each of the 2 EP
ranks owns 17.

`llama-dense` and `llama-dense-tp2` have no `num_local_experts` field at all;
their `ep_ranks` is `None` because vLLM does not create an EP group for a
non-MoE model (`parallel_state.py:1895`).

### EP group layout cells

Derived purely from the frozen closed form above.

| Cell | ExternalDP | DP | PP | PCP | TP | rank | ep_ranks | ep_rank |
|---|---|---|---|---|---|---|---|---|
| `layout-dp2-tp2` | 1 | 2 | 1 | 1 | 2 | 3 | `(0, 1, 2, 3)` | 3 |
| `layout-dp2-pp2-tp2` | 1 | 2 | 2 | 1 | 2 | 6 | `(2, 3, 6, 7)` | 2 |
| `layout-dp2-pp2-tp2-lo` | 1 | 2 | 2 | 1 | 2 | 1 | `(0, 1, 4, 5)` | 1 |
| `layout-dp2-pcp2-tp2` | 1 | 2 | 1 | 2 | 2 | 5 | `(0,...,7)` | 5 |
| `layout-extdp2-dp2-tp2` | 2 | 2 | 1 | 1 | 2 | 5 | `(4, 5, 6, 7)` | 1 |

`layout-dp2-pp2-tp2` is the discriminating cell: the EP group excludes the
other pipeline stage, so it is `(2, 3, 6, 7)` for rank 6, not `(0,...,7)` and
not `(4, 5, 6, 7)`.

## Frozen expectations

F1 (scored, genuine risk). Every one of the 12 geometry cells maps exactly as
tabulated for `num_experts`, `top_k`, `moe_intermediate_size` and
`local_num_experts`. 12 instances.

F2 (scored, genuine risk). Every one of the 5 EP layout cells produces exactly
the tabulated `ep_ranks` tuple and `ep_rank` index. 5 instances.

F3 (scored, genuine risk). The three cells that the current reader gets wrong
change in the stated direction relative to the pre-change reader:

- `g-ep-dp2-tp2`: `moe_intermediate_size` rises from 256 to 512 (EP does not
  tensor-shard experts).
- `g-ep-dp2-tp2`: `local_num_experts` falls from 16 to 8 (the EP world is
  `dp * pcp * tp = 4`, not `dp = 2`).
- `g-ep-off-dp2-tp2`: `moe_intermediate_size` falls from 256 to 128 (without
  EP the experts shard over the flattened `dp * tp = 4` devices).

3 instances.

F4 (scored, genuine risk). `SimExecutor` derives its EP group from the active
parallel configuration and binds it to an expert-group-capable sink exactly
once, before the first step, and only when `use_ep` is true. With `use_ep`
false the executor performs no binding at all and the sink keeps whatever
`ep_ranks` its own configuration declared. 2 instances (bound and not bound).

F5 (fatal unscored). Default and failure behavior is preserved: a text config
whose accessors are missing still stamps the defaulted field names on
`ModelDims.defaulted_fields` and never raises, and a non-MoE config yields
`num_experts = 0`, `top_k = 0`, `moe_intermediate_size = None`,
`local_num_experts = 0`.

F6 (fatal unscored). The VLLM-22 `SimWorker` schedule identity and the
producer-disabled serial identity are preserved exactly: with the observation
producer off, the rendered serial `ExecutionGraph` and GOAL bytes of the
accepted Granite step are unchanged, and with it on, the emitted operation
identifiers, order, logical queues and dependency edges of the accepted step
are unchanged.

Scored behavioral instances: F1 (12) + F2 (5) + F3 (3) + F4 (2) = 22. F5 and
F6 are fatal unscored guards and never enter that denominator.

### Entailment analysis

F3 is entailed by F1 in value, since F1 already fixes the post-change numbers.
It is retained as a separate scored family only for its DIRECTION claim
relative to the pre-change reader, which F1 does not state, and its instances
are counted once each rather than doubled into F1. F2 is not entailed by F1:
the geometry cells fix counts, the layout cells fix membership, and a reader
could get every count right with the wrong group. F5 is evaluated before F1 so
that a defaulting regression voids the run rather than scoring as a cell miss.

### Physical sanity, floors and ceilings stated first

Floor. `local_num_experts >= 1` on every rank of a legal EP world, since a rank
owning zero experts would receive no dispatched token and the routed all-to-all
would be degenerate. With 32 experts the largest legal EP world under this
constraint is 32.

Ceiling. `local_num_experts <= num_experts` always, and
`sum over EP ranks of local_num_experts == num_experts` exactly. That sum rule
is what the uneven cell tests: `6 * 4 + 2 * 3 == 30`.

Weight-byte companion. `moe_intermediate_size * local_num_experts` is
proportional to the per-rank resident expert weight bytes, which sets the
decode memory floor. At Granite scale the corrected `g-ep-dp2-tp2` cell gives
`512 * 8 = 4096` where the pre-change reader gave `256 * 16 = 4096`: the
product is unchanged, so the roofline decode floor does not move, while the
per-expert GEMM shape and the routed hidden-vector geometry both do. Stating
this before the run keeps a matching TTFT from being read as evidence that the
old mapping was right.

## Registry discipline

This freeze registers no new task ID. An ID is registered after the run only
for a registered VLLM-6 acceptance clause the run did not demonstrate.
