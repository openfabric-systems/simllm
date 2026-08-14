# vLLM MoE geometry and expert group v1 results (VLLM-6)

Expectations were frozen in commit `20f6017`, before the implementation and
every result-producing run, with a check-only dry run that validated all 17
frozen cells against their own closed forms and produced no artifacts. The
measured run below observed repository commit `88a57f7` with a clean working
tree. The evidence was authored against vLLM v0.26.0 commit
`568afb3a13806beb53bb2e6bd518269357b237c0`; the pinned arm observed vLLM
`0.26.0` and no check requires the two to be equal.

## Verdict

**VLLM-6 closes.** The run is not void: all 5 fatal guard cases held. Scored
behavioral instances: **22/22**, matching the frozen denominator exactly.

## Fatal unscored guards, 5 of 5 held

| Guard | Case | Held |
|---|---|---|
| `default-and-dense-behavior` | a dense config yields `num_experts = 0`, `top_k = 0`, `moe_intermediate_size = None`, `local_num_experts = 0`, and no expert group | yes |
| `default-and-dense-behavior` | a missing accessor is stamped on `defaulted_fields` rather than raised | yes |
| `vllm22-schedule-identity` | 24 layers x 2 phases of semantic MoE sites | yes |
| `vllm22-schedule-identity` | every routed site stays a zero-byte semantic marker | yes |
| `vllm22-schedule-identity` | one visibility frontier for the unbatched slice | yes |

These ran before the scored families, so a defaulting or schedule regression
would have voided the run rather than scoring as a cell miss.

## F1: the 12 geometry cells, 12/12

Every cell mapped exactly as frozen for expert count, top-k, per-expert
intermediate size, local experts and the expert group.

| Cell | dp / pcp / tp | EP | rank | experts | top-k | per-expert intermediate | local experts |
|---|---|---|---|---|---|---|---|
| `g-dense-world1` | 1/1/1 | off | 0 | 32 | 8 | 512 | 32 |
| `g-ep-off-tp2` | 1/1/2 | off | 1 | 32 | 8 | 256 | 32 |
| `g-ep-off-dp2-tp2` | 2/1/2 | off | 3 | 32 | 8 | 128 | 32 |
| `g-ep-flag-world1` | 1/1/1 | on | 0 | 32 | 8 | 512 | 32 |
| `g-ep-dp8` | 8/1/1 | on | 5 | 32 | 8 | 512 | 4 |
| `g-ep-dp2-tp2` | 2/1/2 | on | 3 | 32 | 8 | 512 | 8 |
| `g-ep-dp2-pcp2-tp2` | 2/2/2 | on | 7 | 32 | 8 | 512 | 4 |
| `g-ep-dp8-uneven` | 8/1/1 | on | 0 | 30 | 8 | 512 | 4 |
| `g-ep-dp8-uneven-hi` | 8/1/1 | on | 7 | 30 | 8 | 512 | 3 |
| `g-ep-dp2-eplb` | 2/1/1 | on | 1 | 34 | 8 | 512 | 17 |
| `llama-dense` | 1/1/1 | off | 0 | 0 | 0 | none | 0 |
| `llama-dense-tp2` | 1/1/2 | off | 1 | 0 | 0 | none | 0 |

`g-ep-flag-world1` is the one that catches a common misreading: enabling
`--enable-expert-parallel` on a single device does not enable expert
parallelism, because vLLM's condition is
`dp * pcp * tp > 1 and enable_expert_parallel`
(`fused_moe/config.py:1204-1207`).

## F2: the 5 rank layout cells, 5/5

| Cell | ExternalDP / DP / PP / PCP / TP | rank | ep_ranks | ep_rank |
|---|---|---|---|---|
| `layout-dp2-tp2` | 1/2/1/1/2 | 3 | `(0, 1, 2, 3)` | 3 |
| `layout-dp2-pp2-tp2` | 1/2/2/1/2 | 6 | `(2, 3, 6, 7)` | 2 |
| `layout-dp2-pp2-tp2-lo` | 1/2/2/1/2 | 1 | `(0, 1, 4, 5)` | 1 |
| `layout-dp2-pcp2-tp2` | 1/2/1/2/2 | 5 | `(0, ..., 7)` | 5 |
| `layout-extdp2-dp2-tp2` | 2/2/1/1/2 | 5 | `(4, 5, 6, 7)` | 1 |

`layout-dp2-pp2-tp2` is the discriminating cell. The expert group excludes the
other pipeline stage, so rank 6's group is `(2, 3, 6, 7)`, not the whole world
and not the contiguous `(4, 5, 6, 7)` that a naive block split would give.

## F3: the three corrected directions, 3/3

| Cell and field | before | after |
|---|---|---|
| `g-ep-dp2-tp2` `moe_intermediate_size` | 256 | 512 |
| `g-ep-dp2-tp2` `local_num_experts` | 16 | 8 |
| `g-ep-off-dp2-tp2` `moe_intermediate_size` | 256 | 128 |

The "before" column is the reader this branch replaces, reproduced in the
harness and run side by side, not quoted from memory. Root cause of the first
two: the old reader divided the per-expert intermediate size by
`tensor_parallel_size` whatever the expert-parallel state and sized the expert
world from `data_parallel_size` alone. vLLM does neither. Under expert
parallelism a device owns whole experts and the MoE tensor-parallel size is 1
(`fused_moe/config.py:1233-1252`), while the expert world is the flattened
`dp * pcp * tp` (`fused_moe/config.py:1113-1121`). Root cause of the third:
without expert parallelism the experts shard across that same flattened set,
not across `tp` alone (`fused_moe/config.py:1217-1231`).

## F4: executor binding, 2/2

With `dp = 8` and expert parallelism in use, the executor derived
`(0, ..., 7)` and bound it to an expert-group-capable sink exactly once. With
the flag set but a device set of one, it performed no binding at all, so the
sink keeps whatever group its own configuration declared. This is component
evidence: the binding method was exercised with the exact attributes
`_init_executor` sets, because `SimExecutor` itself is only constructible
inside a real vLLM process (VLLM-5 still tracks that CI harness).

### Post-specified chronology note, 2026-08-14

Binding gained a precondition after this study was published, and the study was
rerun rather than left stale. TRAF-33 made a declared expert group assert an
all-to-all whose combine returns an already reduced output, so
`_bind_expert_group` now binds only when the pinned configuration implies one:
all-to-all kernels enabled (`fused_moe/config.py:1052-1055`) and a backend
whose prepare-finalize reduces. The cells here declared no
`all2all_backend` at all, so the reader applied vLLM's own
`allgather_reducescatter` default (`config/parallel.py:186`), whose combine
does not reduce (`fused_moe/prepare_finalize/naive_dp_ep.py:109` and `:242`),
and the binding arm raised instead of binding.

Exactly one field changed: `_vllm_config` now sets
`all2all_backend = "deepep_high_throughput"` on every cell's parallel config,
as the module constant `CELL_ALL2ALL_BACKEND`. A real `ParallelConfig` always
carries a backend, and the cells' intent is expert-group derivation and
binding, not backend selection, so declaring a reducing one preserves what F4
was written to test rather than reinterpreting it. No expected value, no cell
input tuple and no closed form was touched.

The rerun reproduces this study exactly. Both F4 rows pass with the same
observed groups, `(0, ..., 7)` and none, and the scored total is again 22/22
against the frozen denominator of 22 with no fatal guard violated. Every other
row is byte-identical to the pre-change run: all 2 fatal-guard rows, 3
schedule-identity rows, 12 F1 geometry rows, 5 F2 layout rows and 3 F3
direction rows compare equal field for field, which is expected because
`moe_intermediate_size`, `local_num_experts` and `ep_ranks` are functions of
`dp`, `pcp`, `tp`, `enable_expert_parallel`, `rank` and the expert count alone
and read no backend. The pinned-source arm is likewise unaffected, because its
probe receives the raw cell tuples rather than this fixture; it reran with the
same three refusals named below.

The gap that let this reach a published study is closed too:
`tests/test_adapters_vllm.py` now executes these binding cells directly, so a
future change to the binding semantics fails the suite instead of only the
study.

## Physical sanity, checked before the digits

`sum over EP ranks of local_num_experts == num_experts` on every cell where
expert parallelism is in use. The uneven cell is the one that tests it:
`6 * 4 + 2 * 3 == 30`. `local_num_experts >= 1` holds on every rank of every
cell, and an expert world wider than the expert count is now refused with an
explicit message rather than handing a rank zero experts.

Weight-byte companion, stated in the freeze before the run:
`moe_intermediate_size * local_num_experts` is unchanged by the correction at
`g-ep-dp2-tp2` (`512 * 8 == 256 * 16 == 4096`). The per-rank resident expert
weight bytes, and therefore the decode memory floor, do not move. What moves is
the per-expert GEMM shape and the expert world size, which is what the routed
traffic geometry depends on. A TTFT that matches after this change is therefore
not evidence that the old mapping was right, and was pre-registered as such.

## Pinned-source corroboration, and what it refused

The parallel-side of every cell was re-derived in a second interpreter against
the real pinned `vllm.config.ParallelConfig`, so the field names, defaults and
validation come from the package rather than from a stand-in. 10 of 12 geometry
cells and 3 of 5 layout cells constructed and agreed exactly on
`ep_size`, `moe_tp_size`, `ep_ranks` and `ep_rank`.

Three cells were refused by vLLM v0.26.0 itself:

- `g-ep-dp2-pcp2-tp2` and `layout-dp2-pcp2-tp2`:
  `PCP does not support data parallelism yet`. Prefill context parallelism
  combined with data parallelism is not a reachable configuration in this
  release, so those two cells describe the geometry reader's answer for a shape
  the framework will not build. Their scored pass is stand-in evidence only.
- `layout-extdp2-dp2-tp2`: `data_parallel_rank (2) must be in the range [0, 2)`.
  The probe drives `external_launcher`, which derives the data-parallel rank
  from the `RANK` environment variable, and the ExternalDP dimension needs a
  launcher convention the probe does not supply. The layout arithmetic itself
  is unchanged; only its corroboration is missing.

This is recorded because it is a real limit on the evidence, not because a
frozen expectation failed. The frozen expectations are about the reader's
mapping, and the reader answers those three cells exactly as frozen.

## Genuine-risk analysis

All 22 scored instances are genuine risk. F3 is entailed by F1 in value, which
the freeze said before the run; it is retained for its DIRECTION claim against
the replaced reader, which F1 does not make, and its 3 instances are counted
once rather than doubled into F1's 12. F2 is not entailed by F1: a reader can
get every count right with the wrong group, which is exactly what
`layout-dp2-pp2-tp2` tests. The fatal guards were evaluated first and are not
part of the 22.

## Registered IDs

Zero new IDs. Every registered VLLM-6 acceptance clause is demonstrated: exact
mappings for expert count, top-k, per-expert intermediate size and local
experts across enabled and no-MoE configurations, including the failure and
default behavior; the executor deriving the exact expert group from the active
parallel configuration and passing it to its sink; and the explicit no-EP path
plus the VLLM-22 schedule and serial-off identities preserved.

Two limitations sit in prose rather than in a new ID, because no registered
clause claimed them: the PCP-plus-DP shape is unreachable in vLLM v0.26.0 and
therefore uncorroborated, and the executor-level binding is component evidence
until the VLLM-5 CI harness can construct `SimExecutor` without a live engine.

Raw report: `report.json` under the wave-10 run root, outside the repository.
