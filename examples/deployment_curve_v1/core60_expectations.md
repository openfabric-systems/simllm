# CORE-60 EP32 prefill composition freeze

Status: **EXPECTATIONS_ONLY, PROTOCOL VOID**. The physical service below was
derived before the CORE-60 calibration comparison. The visible calibration
result was already public from CORE-59, but it was not used to select a
contract, tune a value or compute the movement below. During source inspection,
however, the official SGLang page rendered its evaluation table and exposed
the forbidden 2K and 4K prefill rows. Those numbers were not used, but literal
no-held-out-access acceptance is not met. No scored comparison is authorized
and no fitted or free constant exists. COMP-75 owns a clean independent
repetition.

## Adopted contracts and expected signed effects

| Contract | Frozen evidence and rule | Expected effect versus CORE-59 |
|---|---|---|
| Per-rank token ownership | The deployment projection case `sglang-prefill-ep32-r16-t16384` declares 16,384 new tokens per rank, 4,194,304 routed visits per layer globally and 131,072 per rank. These equal `32 * 16384 * 8` and `16384 * 8`. | Unchanged. CORE-59 already priced the right per-rank token population. |
| Routed wire precision | Pinned SGLang commit `bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3`: `python/sglang/srt/layers/moe/utils.py:227` through `:291` selects FP8 by default for DeepSeek-V3; `srt/environ.py:988` through `:992` enables JIT DeepGEMM by default; `token_dispatcher/deepep.py:511` through `:609` quantizes and dispatches FP8 plus scales; `fp8_kernel.py:462` through `:517` and `:605` through `:641` allocate 128-element float32 group scales on H100; `moe_runner/deep_gemm.py:450` through `:474` produces BF16 expert output for combine. | Increase throughput by reducing communication bytes. Dispatch is 7,168 FP8 bytes plus 56 four-byte scales, or 7,392 bytes per token-destination. Combine remains 14,336 BF16 bytes. |
| Same-destination expert deduplication | Pinned `token_dispatcher/deepep.py:570` through `:609` obtains and dispatches `is_token_in_rank`. Existing simllm captured routing documents the same once-per-destination dispatch and reverse combine at `simllm/traffic/step_comm.py:1207` through `:1221`. | Increase throughput by reducing both dispatch and combine rank-pair payloads. |
| Compute and communication overlap | Pinned `batch_overlap/operations_strategy.py:89` through `:132` places yields between dispatch launch/completion and combine launch/completion around expert and shared-expert compute. `batch_overlap/operations.py:38` through `:71` advances two child batches together; `two_batch_overlap.py:880` through `:941` splits and invokes that executor; `models/deepseek_v2.py:2832` through `:2883` selects it for runnable TBO batches. The [SGLang large-scale EP report](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/) identifies this two-microbatch overlap as part of the EP32 prefill configuration. | Increase throughput relative to CORE-59's serial addition. The steady-state composition is max-like and the independently measured candidate compute service is the complete hiding budget. |

The composed model still has positive exposed communication service, so its
prefill throughput must decrease relative to candidate-only. It must increase
relative to CORE-59. Decode must remain exactly unchanged because SGL-38 is
the sole owner of the decode bind. These signs are frozen before the CORE-60
movement is computed.

## Destination and wire arithmetic

The uniform-routing assumption is explicit: each token selects eight distinct
experts uniformly without replacement from 256 logical experts, and each of
32 ranks owns eight logical experts. For any destination rank,

```text
p_rank = 1 - C(248, 8) / C(256, 8)
       = 939691952959 / 4138017124000

E[unique destinations] = 32 * p_rank
                       = 939691952959 / 129313035125
                       = 7.26679991735288

E[remote destinations] = 31 * p_rank
                       = 7.039712419935602
```

The exact expected aggregate ordered-pair payload is
`16384 * p_rank * vector_bytes`. It is fractional. The selected integer is its
floor, matching the existing uniform traffic renderer's floor convention, and
the adjacent ceiling is propagated as a physical rounding envelope.

| Phase | Bytes per token-destination | Exact expected pair bytes | Selected pair bytes | Integer envelope |
|---|---:|---:|---:|---:|
| Dispatch | 7,392 | 27,502,686.714405112 | 27,502,686 | 27,502,686 to 27,502,687 |
| Combine | 14,336 | 53,338,543.93096749 | 53,338,543 | 53,338,543 to 53,338,544 |

Rank zero retains the existing four-node EP32 placement: seven peers are
local and 24 are remote. On the selected floor, dispatch carries 192,518,802
local bytes and 660,064,464 fabric bytes per phase; combine carries
373,369,801 local bytes and 1,280,125,032 fabric bytes. The upper rounding arm
adds at most one byte per ordered peer.

## Frozen physical service and overlap

The existing 450 GB/s endpoint serializer and pinned `rnic-nn` backend produce
the following services. The 400 Gbit/s PLACE-5 arm is selected; 200 Gbit/s is
the existing pessimistic sensitivity, not a fit choice.

| Arm | Dispatch per layer | Combine per layer | 58-layer communication | Candidate compute hidden | Exposed increment |
|---|---:|---:|---:|---:|---:|
| 400 Gbit/s point | 13,410,556,120 ps | 26,006,336,300 ps | 2,286,179,760,360 ps | 1,363,249,960,000 ps | 922,929,800,360 ps |
| 200 Gbit/s sensitivity | 26,819,112,240 ps | 52,010,672,600 ps | 4,572,127,520,720 ps | 1,363,249,960,000 ps | 3,208,877,560,720 ps |

The composition is frozen as

```text
total_step_service_ps = max(candidate_compute_service_ps,
                            communication_service_ps)
incremental_service_ps = max(communication_service_ps
                             - candidate_compute_service_ps, 0)
```

This is not an overlap fraction. It uses only the component evidence's
1,363,249,960,000 ps compute service as hiding capacity. The point rounding
envelope for total service is 2,286,179,760,360 to 2,286,179,762,680 ps; the
sensitivity envelope is 4,572,127,520,720 to 4,572,127,525,360 ps.

## Comparison and preservation lock

Only `sglang_prefill_1k` and `sglang_decode_standard` may be read by the later
calibration-only comparison. The 2K prefill, 4K prefill and simulated-MTP IDs
remain forbidden. The comparison must publish its accessed IDs, an empty
forbidden list, zero fitted parameters, the signed movement from candidate-only
and CORE-59, and the honest remaining visible-row error.

The JSON freeze pins all four CORE-59 freeze/result artifacts and the first
scored run's nine locked artifacts. The later runner must verify all thirteen
digests without rewriting them. It may neither invoke the flagship runner nor
alter decode pricing.
