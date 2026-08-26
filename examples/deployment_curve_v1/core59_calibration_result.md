# CORE-59 calibration-only result

## Mechanism list

1. **EP32 prefill dispatch and combine:** one uniform-routing dispatch and one
   combine all-to-allv through the existing placement, NVLink and `rnic-nn`
   htsim path for each of 58 MoE layers. The point service is
   3,320,872,128,000 ps per candidate step.
2. **EP32 data-parallel attention synchronization:** zero. TP is one and each
   attention rank owns separate requests, so no rank-spanning attention result
   is identified.
3. **EP72 standard decode:** zero new mechanisms. SGL-38 owns the measured
   decode shape bind, and this task does not absorb its error.

The historical `intra_node_collective_surcharge_ps` remains unchanged in the
first scored run. Its CORE-59 application count is zero. There are no fitted
parameters and no new free constants.

## Constant envelopes and justification

| Constant | Envelope | Physical basis |
|---|---:|---|
| MoE layer count | 58 to 58 | Exact DeepSeek-V3 architecture after three dense early layers |
| Dispatch/combine phases | 2 to 2 per MoE layer | Existing traffic semantics |
| Prefill EP width | 32 to 32 | Calibration deployment projection |
| New tokens per rank | 16,384 to 16,384 | Candidate key and deployment case agree |
| Routed vector | top-k 8, hidden 7,168, 2 bytes | Pinned model and existing traffic payload rule |
| Local endpoint rate | 450 GB/s to 450 GB/s | Existing full-duplex NVLink serializer |
| Fabric rate | 200 to 400 Gbit/s | Existing sensitivity arm and PLACE-5 point arm |
| Total prefill mechanism service | 3,320,872,128,000 to 6,641,512,256,000 ps | Pinned htsim output over 116 phases |

The point and sensitivity runs each rendered two one-layer phases, 48 fabric
flows, 14 local segments and exact byte conservation. Both reached physical
quiescence. The compact evidence is under
`$SIMLLM_CORE59_RUN_ROOT/mechanism-evidence-1/`.

## Signed calibration-row movements

| Calibration row | Published | Candidate only | CORE-59 point | Signed movement | Signed error before | Signed error after |
|---|---:|---:|---:|---:|---:|---:|
| `sglang_prefill_1k` | 57,674.0000 | 96,146.7111 | 27,982.1912 | -68,164.5198 tokens/s/node, -70.8964% | +66.7072% | -51.4821% |
| `sglang_decode_standard` | 22,282.0000 | 8,949.7597 | 8,949.7597 | exactly 0 | -59.8341% | -59.8341% |

The prefill mechanism moves in the frozen negative direction but overcorrects
the visible calibration row. No compensating scale or overlap fraction was
introduced. CORE-60 owns identifying the component-backed service composition
needed to resolve that remaining calibration miss before another scored run.

## Access and preservation

The comparison accessed only `sglang_prefill_1k` and
`sglang_decode_standard`. The forbidden access list is empty, no held-out
numeric value was accessed, and no held-out scorer ran. All nine pinned
first-run artifacts matched their SHA-256 identities, including the published
refutation, configuration, runner, tools and figures.

CORE-59's literal acceptance is met: the shared residual is replaced on the
successor path, the physical mechanisms and envelopes were frozen before this
comparison, both calibration rows have explicit signed movements, and the
first scored run is unchanged.
