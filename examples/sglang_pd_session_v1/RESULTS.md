# SGLang disaggregated session result

The SGLang session mechanism is live, but the frozen study is REFUTED. Every
fatal guard held after correcting an analyzer that had incorrectly required
two independent prefill schedulers to complete handoffs in admission order.
All 144 admissions, 144 handoffs, 144 terminals and 576 decode tokens were
conserved across 18 cells, and the maximum time-to-first-token decomposition
residual was 0 ps. Throughput was nondecreasing in 4 of 6 frozen curves, so
SGL-33 stays open and SGL-36 owns identification of the physical curve shape.

## What ran

Pinned SGLang `0.5.19.dev345+gbfeae4e79` drove the cached Granite checkpoint
through one-prefill/one-decode, one-prefill/two-decode and
two-prefill/one-decode pools. Each ratio crossed 8-token and 16-token prompts
with eight concurrent requests at 8,000, 16,000 and 32,000 offered requests
per second. Each request produced four decode tokens. Isolated controls priced
the key-value cache handoff with the accepted 100,000,000 and 200,000,000 ps
constant arms and the existing TRAF-62 packet arm.

The pinned native prefill/decode connector was audited before the freeze. Its
prefill role rejects the fake transfer backend, while its decode role requires
registered key-value metadata and tensors that the bufferless simulated worker
does not own. The run therefore used the frozen driver-level fallback. Each
pool engine retained its own stock SGLang scheduler in an isolated process,
and one parent virtual clock owned session time.

The expectations-only commit was
`191c237700431507639f963a0ee69e94292611c6`. The scored observation ran at
`9e9e61808db0a6dc26985b717fca24a5eafd13e0`; its retained raw result has
SHA-256 `007814fb2a9df1580f58303a5a8eaa058478573384b267a5cead4e3d4345783f`.
That file records the original VOID classification. Commit
`623195eb5ceb38d626dcabdd473592af9795efa3` corrected only the analyzer's
evidence interpretation: exact-once conservation compares identity sets, and
a failed frozen throughput relation is a refutation rather than a fatal guard.
No observation or expectation changed.

## What came out

Every exact cell has 8 admissions, 8 handoffs, 8 terminals, 32 decode tokens
and a 0 ps time-to-first-token decomposition residual. All three pool ratios
exposed a genuine multi-request prefill batch and decode batch. Increasing the
constant handoff by 100,000,000 ps added exactly 100,000,000 ps to time to
first token and 0 ps to time per output token; every other term was identical.

The packet arm moved time to first token from 264,304,000 ps to 187,385,600 ps.
Its exact signed delta was -76,918,400 ps, equal to the packet handoff duration
of 23,081,600 ps minus the 100,000,000 ps constant. The metric residual was
0 ps and time per output token was unchanged. Eight flows carried eight
49,152-byte shards, conserving 393,216 bytes.

One complete machine-readable curve is:

| Offered requests/s | Aggregate output tokens/s | Per-token request delay |
|---:|---:|---:|
| 8,000 | 21,852.431 | 242,029,000 ps |
| 16,000 | 30,459.481 | 193,268,500 ps |
| 32,000 | 35,552.711 | 182,986,250 ps |

Throughput was nondecreasing in that curve and three others. It dipped in
`sglang-p1-d2-prompt8` and `sglang-p2-d1-prompt16`. The latter fell from
19,071.038 output tokens/s at 16,000 offered requests/s to 17,818.108 at
32,000. The fixed eight-request grid changes how stock schedulers form batches
and is not a calibrated saturation surface. The frozen relation is not widened
after seeing the result.

## Physical sanity

The resident-weight read floor is 52,691,712 ps. Observed nonempty steps span
76,800,000 to 548,799,360 ps, so none completes faster than that memory floor.
Each 49,152-byte packet shard needs at least 983,040 ps to serialize at
400 Gbit/s; observed packet service is 3,081,600 ps. Client-visible decode
cadence spans 1,438.9 to 11,911.6 tokens/s, inside the frozen 10 to 100,000
tokens/s interval for this 400M-active-parameter model. These are independent
memory, network and end-to-end plausibility checks, not calibration claims.

## What it changes for the project

SGL-33 gains the complete concurrent session, structural arrangement
projection, constant and packet handoff arms, exact request metrics and
reusable curve records, but stays open on its frozen throughput refutation.
SGL-35 owns native connector reachability without changing the accepted
driver fallback. SGL-36 owns a calibrated load-delay and throughput surface
before CORE-54 may treat the curves as physical evidence. CORE-57 owns the
flagship arithmetic finding: four eight-GPU prefill nodes plus nine eight-GPU
decode nodes render 104 ranks, 8 more than CORE-54's current 96-rank claim.

## What it does not change

This result does not calibrate batching or saturation, execute real tensor
operations, run the flagship deployment, close SGL-33 or validate CORE-54's
physical curve. The vLLM session, every accepted SGLang worker record and all
prior study results retain their frozen SHA-256 digests. The two earlier
stopped attempts and the original analyzer classification remain retained; no
failed behavioral relation is converted into a fatal score or a closure.
