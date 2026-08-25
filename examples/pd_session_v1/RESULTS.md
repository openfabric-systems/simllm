# Disaggregated serving session v1 results

The first CORE-51 slice passes. One vLLM v0.27.1 prefill engine and one
separate decode engine, each exposing eight simulated workers, ran four
requests through the real scheduler-side KV connector seam on one shared
virtual clock. Every time-to-first-token (TTFT) decomposition had a zero
picosecond residual, all six scored behavioral instances held, and no packet
backend ran.

This result lands the declared-constant and one-plus-one halves of CORE-51,
TRAF-61 and PLACE-4. It does not close those tasks. CORE-52 owns a live
56-engine run, CORE-53 owns lookup-record compute pricing, TRAF-62 owns the
fabric-rendered KV transfer, PLACE-5 owns target locations resolved through a
complete fabric topology, and VLLM-35 owns concurrent multi-request batching
through both pools.

## Provenance and chronology

The complete expectations were frozen in commit
`303a958f80062726573ab0717decb84895cad8f9`. The worktree was clean before the
freeze, apart from the required gitignored sizing note. The freeze preceded
the implementation and every scored run.

The mechanism landed in commit
`f25fc8fe6612c938fcef4a6c62f6d709f7bf77a0`. The committed harness is
`5c34e985fc68dacf0ce6d668c26685e84cc13f8d`. Its check-only command validated
the cached Granite configuration and fixture, vLLM 0.27.1 and the frozen
source hashes without creating its selected run directory.

Two scored runs are retained:

1. The first run passed 4 of 4 exact rows and 6 of 6 behavioral instances.
   Its result SHA-256 is
   `f6cb4faae2357d0bb87a4adaf610916dc39726984f50a159853b7c9eb7254f29`.
   Review then found that import-time peak resident set size exceeded every
   later engine sample, so the derived peak increment was censored at zero.
2. Commit `22c76eb6b1314643f6a77f6b3918744e659b9b07` changed only the descriptive
   scale summary to use current resident set size and to report the peak
   censoring explicitly. This was post-specified after the first result. No
   acceptance relation or fatal guard changed. A fresh second run again
   passed 4 of 4 exact rows and 6 of 6 behavioral instances. Its result
   SHA-256 is
   `870071e532e3aa587cce55eb40e2969007063cce43a65cf2280ee0746027d4c4`.

The compact tracked projection is [results.json](results.json). The full
second result remains in the configured bulk-data root under
`core51/scored-v2/result.json`.

## What ran

The live session used cached
`ibm-granite/granite-3.0-1b-a400m-instruct` revision
`ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`, tensor parallel width eight,
64 logical KV blocks per engine, four client-visible decode tokens, no prefix
caching, no chunked prefill and no asynchronous scheduling.

The prefill and decode engines were separate in-process vLLM instances. The
driver called `reset_configuration()` before each construction, declared the
engine and connector roles, and injected the same `VirtualClock` object into
both executors. vLLM randomized a different internal request ID in each pool.
The connector carried the stable session request ID explicitly, so the two
internal scheduler identities remained distinct while the handoff and result
joined losslessly.

The connector exercised vLLM's scheduler-side interface. It reported the
producer's original prompt coverage to the decode scheduler, which computed
only the final bootstrap prompt position before sampling. SimExecutor owns no
paged KV tensor, so worker-side load and save calls deliberately moved no
tensor. One core `KvHandoffEvent` remained the only timing authority.

Compute came from the existing B100 roofline provider. Tensor-parallel work
used the existing `intra-node-fixed-cost-v1` lower arm. The handoff used the
TRAF-61 declared constant. The packet-rendered handoff was disabled.

## Exact timing rows

Admission, both scheduler releases, the handoff and every decode-token
completion came from the shared virtual clock. Decode admission wait was zero
in all four cells.

| Prompt tokens | KV bytes | Handoff (ps) | Prefill service (ps) | First decode service (ps) | TTFT (ps) | TPOT (ps) |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 393,216 | 100,000,000 | 95,424,000 | 77,952,000 | 273,376,000 | 77,952,000 |
| 8 | 393,216 | 200,000,000 | 95,424,000 | 77,952,000 | 373,376,000 | 77,952,000 |
| 16 | 786,432 | 100,000,000 | 114,936,000 | 77,976,000 | 292,912,000 | 77,976,000 |
| 16 | 786,432 | 200,000,000 | 114,936,000 | 77,976,000 | 392,912,000 | 77,976,000 |

For the first row, the deciding identity is:

```text
273,376,000 ps
  = 0 ps prefill queue
  + 95,424,000 ps prefill service
  + 100,000,000 ps KV handoff
  + 0 ps decode admission wait
  + 77,952,000 ps first decode service
```

All four rows had a zero-picosecond residual. These are exact-oracle rows and
do not increase the behavioral denominator.

## Scored behavioral relations

Both handoff-movement instances passed. Raising the declared handoff from
100,000,000 to 200,000,000 ps moved TTFT by exactly 100,000,000 ps at both
prompt lengths. Prefill service, decode admission, decode service, token IDs,
token order and TPOT moved by exactly zero.

All four prompt-movement instances passed. Doubling the prompt doubled KV
bytes exactly, raised prefill service by 19,512,000 ps, raised TTFT by
19,536,000 ps and did not reduce TPOT. The prefill-service and TTFT comparisons
remain separate instances at each handoff setting.

The scored result is therefore 2 of 2 behavioral families and 6 of 6
instances. Source identity, placement, role agreement, shared-clock identity,
exact decomposition, physical bounds and packet-backend absence are fatal or
structural evidence and are not added to that denominator.

## Physical sanity

The handoff carries keys plus values over all 24 layers, eight KV heads and a
64-element head in two-byte elements. That is 49,152 bytes per prompt token.
At 400 Gbit/s, serialization alone takes at least 7,864,320 ps for eight
tokens and 15,728,640 ps for 16. The 100,000,000 and 200,000,000 ps constants
are respectively 12.7 to 25.4 times the eight-token floor and 6.4 to 12.7
times the 16-token floor. Both remain below the frozen deliberately slow-path
ceilings of 364,572,800 and 679,145,600 ps. This bounds the surrogate; it does
not calibrate it.

The per-rank Granite model used by the roofline carries 320,864,256 resident
weight bytes. Reading those bytes once at the B100 envelope's 8 TB/s memory
rate takes 40,108,032 ps, or 57,297,189 ps after the provider's 0.7 efficiency
derate. The observed 77,952,000 to 77,976,000 ps decode steps are 1.36 times
that derated floor. The 95,424,000 to 114,936,000 ps prefill steps are also
above it and rise with prompt work.

All 20 nonempty steps lay between 77,952,000 and 114,936,000 ps, inside the
frozen 1,000,000 to 100,000,000,000 ps physical interval. The modeled decode
cadence is 12,824 to 12,828 tokens per second for this 400M-active-parameter
model on the B100 envelope, inside the frozen 10 to 100,000 range. Finally,
all 20 locality projections reported zero fabric backend runs: both tensor
parallel groups stay within their own eight-GPU node, so inter-node traffic
would have been a defect in this slice.

These three checks are independent: weight movement bounds compute service,
KV bytes over link rate bound the handoff, and decode cadence checks the
composed end-to-end scale. They establish plausibility, not hardware
calibration. CORE-53 and TRAF-62 own the missing measured compute and network
paths.

## Placement

The same builder produced both shapes. The live one-plus-one shape contains
16 ranks, 16 GPUs and 16 GPU-affine NICs. The manifest-only target contains
448 dense ranks, 448 GPUs and 448 GPU-affine NICs over 16 prefill plus 40
decode nodes. Ranks 0 through 127 are prefill and 128 through 447 are decode.
Every tensor-parallel group is one consecutive eight-rank node, every
data-parallel group stays within one role, and GPU-rank GOAL mapping is
identity.

The current fabric locations are deterministic simulated labels. They do not
resolve through a complete switch-and-link topology. PLACE-4 therefore stays
open, with PLACE-5 carrying that exact residual.

## Engine-count feasibility

Fresh child processes retained all engines at two measured points:

| Shape | Retained engines | Current RSS delta (KiB) | KiB per engine | Construction wall time (s) | Seconds per engine |
|---|---:|---:|---:|---:|---:|
| 1 prefill + 1 decode | 2 | 63,476 | 31,738 | 1.4733 | 0.7367 |
| 2 prefill + 2 decode | 4 | 64,112 | 16,028 | 1.5185 | 0.3796 |

The per-engine averages include one cold first construction in each child.
The individual first engines took 0.9710 and 0.9544 seconds; later engines
took 0.0297 to 0.0317 seconds. Import-time peak resident set size was 979,124
KiB in both children and remained above every retained-engine sample, so the
peak increment is censored at zero and is not used as a memory estimate.

Using the measured current-resident averages gives a descriptive 56-engine
increment of 897,568 to 1,777,328 KiB and a sequential construction range of
21.26 to 41.25 seconds. This extrapolation assumes the same model metadata,
64-block KV pool, tensor-parallel width, allocator behavior, process topology
and sequential construction. It does not establish that 56 engines fit or
that a 448-rank workload runs on this host. CORE-52 remains open for that
measurement.

## What changes

- CORE-51 gains its live one-plus-one vLLM slice, exact TTFT and TPOT
  reduction and a measured scale bound. It remains open.
- TRAF-61 gains the live geometry-sized declared-constant handoff with zero
  packet invocations. It remains open on TRAF-62.
- PLACE-4 gains the role-carrying one-plus-one builder and a structurally
  validated 448-rank projection. It remains open on PLACE-5.
- VLLM-35 registers the unproven concurrent batching path. CORE-52 and CORE-53
  register the full live scale and lookup-pricing residuals.
- M6 moves from registered to first slice landed.

## What does not change

No 56-engine session or 448-rank workload ran. No KV packet was rendered, no
handoff constant was calibrated, no kernel-cycle lookup record priced a step,
and no SGLang engine was constructed. The connector controls the real vLLM
scheduler seam but does not claim a worker tensor copy. CORE-51, TRAF-61 and
PLACE-4 all stay open, and no full-target capacity or accuracy claim follows.
