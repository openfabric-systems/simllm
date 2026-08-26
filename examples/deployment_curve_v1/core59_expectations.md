# CORE-59 role and shape mechanism freeze

Status: **EXPECTATIONS_ONLY**. This freeze was written before either visible
calibration target value was read for CORE-59. No held-out target value is an
input, no held-out score is authorized, and the first scored run remains
immutable history.

## Frozen mechanism list

| Role and shape | Mechanism | Count | Expected calibration movement |
|---|---|---:|---|
| EP32 prefill, 16,384 new tokens per rank | Per-MoE-layer all-to-allv dispatch and combine through the existing placement, NVLink and htsim traffic path | 116 phases, from 58 layers times 2 | Decrease throughput |
| EP72 standard decode | None | 0 | Exactly unchanged |
| EP32 data-parallel attention | No synchronization service. TP is one and each attention rank owns separate requests, so the disclosed DP-attention shape does not shard one attention result across ranks. | 0 | Exactly unchanged |

The prefill candidate calls its only aggregate component
`aggregate_noncollective_step_service`. The full-depth value is a declared
61/4 depth extrapolation, and its routing availability is `not-captured`.
The deployment projection independently fixes EP32, 16,384 new tokens per
rank, top-k 8 and 58 routed layers. Those component-complete calibration
surfaces identify the omitted dispatch and combine traffic without a
throughput target.

The standard-decode evidence does not identify a positive missing service
independently of the unbound measured shape. SGL-38 owns that bind. CORE-59
therefore freezes zero decode mechanisms instead of adding a term that would
absorb SGL-38's error or make the already low calibration projection lower.

## Constants and physical envelopes

No new free constant and no tunable constant exists.
The historical `intra_node_collective_surcharge_ps` remains byte-identical in
the first-run artifacts but has application count zero in the CORE-59 path.

| Constant | Frozen envelope | Justification |
|---|---:|---|
| MoE layers | 58 to 58 | DeepSeek-V3 has three dense early layers within 61 total layers. |
| Phases per MoE layer | 2 to 2 | Existing traffic semantics render dispatch then combine. |
| EP ranks | 32 to 32 | The calibration deployment is EP32 over four eight-GPU nodes. |
| New tokens per rank | 16,384 to 16,384 | The visible candidate key and deployment case agree exactly. |
| Top-k | 8 to 8 | Pinned model architecture. |
| Hidden width | 7,168 to 7,168 | Pinned model architecture. |
| Activation bytes | 2 to 2 bytes per element | Existing `ModelDims` wire payload rule. No FP8 wire claim is present. |
| Local endpoint rate | 450 GB/s to 450 GB/s | Existing full-duplex NVLink serializer constant. |
| Fabric rate | 200 to 400 Gbit/s | Existing sensitivity and PLACE-5 point arms, both exercised by the first packet session. |

The existing uniform-routing traffic authority computes

```text
per_pair_bytes = 16384 * 8 * 7168 * 2 // 32
               = 58,720,256 bytes
```

For rank zero's four-node EP32 placement, seven peers are local and 24 are
remote. Each phase therefore carries 411,041,792 local bytes and
1,409,286,144 fabric bytes. The local endpoint serializer costs 913,427,000 ps.
The pinned `rnic-nn` htsim binary costs 28,628,208,000 ps per phase at the
400 Gbit/s point arm and 57,254,416,000 ps at 200 Gbit/s. The fabric dominates
the local maximum at both endpoints.

Applied to 116 phases, the mechanism service envelope is
3,320,872,128,000 to 6,641,512,256,000 ps per candidate step. The selected
point is the 400 Gbit/s PLACE-5 arm. The 200 Gbit/s endpoint remains a
propagated sensitivity, not a fit option.

## Comparison and access lock

Only `sglang_prefill_1k` and `sglang_decode_standard` may be read by the later
calibration comparison. The 2K prefill, 4K prefill and simulated-MTP IDs are
forbidden. A comparison must publish its accessed ID list, an empty forbidden
list, the exact signed movement for both visible rows and zero fitted
parameters.

The JSON freeze pins SHA-256 identities for the first scored expectations,
configuration, result, report, runner, tools and publication figures. CORE-59
must validate those hashes without rewriting any of them.
