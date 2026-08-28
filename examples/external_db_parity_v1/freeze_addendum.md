# External database parity review freeze addendum

This append-only freeze responds to findings A-1, B-2, B-3 and B-4 from the
adversarial review of the external database parity study. The original
`expectations.md` remains immutable. The inventory and query points below are
frozen before the repaired comparison run.

## I1 payload recount oracle

The local study worker independently decompresses the converted payload and
recounts its rows by the table name stored in each row. These literals come
from the executed source audit and are the expected side of the scored I1
count comparisons. The manifest inventory is a separate claim that must equal
this recount or the run is void.

| Table | Rows |
|---|---:|
| `compute_scale` | 1,628 |
| `context_attention` | 50,574 |
| `context_mla` | 1,760 |
| `context_mla_module` | 3,873 |
| `custom_allreduce` | 69 |
| `encoder_attention` | 6,314 |
| `gdn` | 1,862 |
| `gemm` | 101,010 |
| `generation_attention` | 24,438 |
| `generation_dsa_module` | 2,944 |
| `generation_mla` | 2,896 |
| `generation_mla_module` | 5,888 |
| `mamba2` | 469 |
| `mla_bmm` | 848 |
| `moe` | 74,358 |
| `scale_matrix` | 1,628 |
| `wideep_moe` | 4,158 |
| **Total** | **284,717** |

## Added I2 query points

The machine-readable copies of these points are in `query_points.json` under
`review_addendum.queries`.

### Cross-site distance-cap discrimination

- `I2-27`, rule: GEMM cross-site inverse-square utilization blending excludes
  every site beyond the maximum log2-space distance of 2.0. Query
  `{m=257, n=131072, k=131072, quant_mode=bfloat16}`. The capped local resolver
  returned `0x1.d16e300d9dc77p+3`. A local resolver with only the cap disabled
  admitted `(n=16384, k=65536)` at distance `3.1622776601683795` as its fourth
  site and returned `0x1.cc9259aaacb10p+3`. The two results differ by
  85,475,858,518,375 ULP, so the point discriminates the frozen cap rule. The
  cap-off value is an unscored diagnostic; the scored row remains capped local
  versus capped live bit equality.

### Load-time clamp exact hits

- `I2-28`, rule: an exact GEMM table hit returns the load-time
  speed-of-light-clamped served value. Query
  `{m=32768, n=64, k=512, quant_mode=bfloat16}`.
- `I2-29`, rule: an exact generation-attention table cell returns the
  load-time speed-of-light-clamped served value. Cell query
  `{b=128, s=1024, n=128, n_kv=8, kv_quant_mode=bfloat16,
  window_size=0, head_size=128}`.

### Ignored-dimension invariance

Each pair is scored only when both endpoints are bit-equal between the local
and live resolvers and the two endpoints are bit-equal to each other.

- `I2-30` and `I2-31`, rule: generation-attention `attn_dtype` is ignored.
  Both use `{b=64, s=4000, n=96, n_kv=8, kv_quant_mode=fp8,
  window_size=0, head_size=64}` and vary only `attn_dtype` between
  `bfloat16` and `fp8`.
- `I2-32` and `I2-33`, rule: the dispatched standard MoE query ignores
  non-low-latency `kernel_source`. Both use
  `{num_tokens=256, hidden_size=4096, inter_size=1536, topk=8,
  num_experts=128, moe_tp_size=4, moe_ep_size=4,
  quant_mode=bfloat16, workload_distribution=balanced}` and vary only
  `kernel_source` between `moe_torch_flow` and
  `moe_torch_flow_cutlass`.
- `I2-34` and `I2-35`, rule: GDN `model_name` is ignored. Both use the
  frozen generation recurrence exact hit and vary only `model_name` between
  `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-397B-A17B`.
- `I2-36` and `I2-37`, rule: GDN `num_tokens` is ignored. Both use the
  same frozen generation recurrence exact hit and vary only `num_tokens`
  between `512` and `1024`.

## Extended FG-5 cell

Alongside the original GEMM cell, FG-5 now freezes this clamped
generation-attention cell:

- normalized key
  `(bfloat16, 8, 128, 0, 128, 128, 1024)`;
- raw latency `0x1.b78732aaaaaabp-4`;
- served latency `0x1.d0d73a2abadb5p-4`.
