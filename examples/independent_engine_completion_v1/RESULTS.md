# Independent-engine completion

**All 112 native requests qualify within the unchanged process limits.** The
`independent_engine_completion_v1` campaign runs serialized and independent
engine timing at one, two and four engines per pool, with two prompt lengths,
two handoff durations, later arrivals and unequal requests. CORE-68 closes:
separate simulated GPU groups now complete through one clock and reveal native
request output at their own due events.

## Outcome and physical bounds

Four requests with eight prompt tokens and a 100-microsecond handoff complete
in 1,442.656, 819.040 and 507.232 microseconds at independent pool widths one,
two and four. One prompt takes 95.424 microseconds and each decode token takes
77.952 microseconds under the frozen declared service model. The expected
last completion is `prompt + handoff + (4 / width) * 4 * decode`.
Width one to two therefore removes exactly eight decode intervals, 623.616
microseconds; width two to four removes four, 311.808 microseconds. Both
relations hold exactly. Startup work and handoff remain, so four times the
engine width does not promise four times whole-request throughput.

| Prompt tokens | Handoff (microseconds) | Width 1 completion | Width 2 completion | Width 4 completion |
|---|---:|---:|---:|---:|
| 8 | 100 | 1,442.656 | 819.040 | 507.232 |
| 8 | 200 | 1,542.656 | 919.040 | 607.232 |
| 16 | 100 | 1,462.552 | 838.744 | 526.840 |
| 16 | 200 | 1,562.552 | 938.744 | 626.840 |

All completion columns are microseconds from the cell origin. In these
independent burst cells, increasing the handoff by 100 microseconds moves
every first-token time, token completion and cell completion by exactly
100 microseconds. Time per output token (TPOT)
remains the declared decode interval; this forced identity is unscored.
The serialized comparator takes 1,628.928 or 1,707.360 microseconds for these
bursts and gains no overlap from adding engines.

The floor is the larger of every request's causal chain and every engine's
charged service. The ceiling is the last arrival plus all unique service and
all handoffs. Every native cell lies inside these pre-run bounds. Compute
service also exceeds the conditional 40.108032-microsecond resident-weight
streaming floor. That bound applies to the current surrogate that charges all
resident experts; COMP-7 owns demand from the experts actually selected.

The independent network check starts from bytes over rate. Cache sizes of
393,216 and 786,432 bytes require at least 7.864320 and 15.728640 microseconds
at 400 Gbit/s. A declared 10-Gbit/s route plus 50 microseconds gives ceilings
364.572800 and 679.145600 microseconds. Both chosen handoffs lie inside these
bounds. They are declared transfers, without packet contention or measured
hardware calibration.

The causal system check uses the later short request. Its prompt finishes at
96.424 microseconds and its first decode token at 274.376 microseconds from
the mixed cell origin, before the earlier long request's 114.936 and 292.912
microseconds. The serialized comparator has the opposite order. The short
request arrives at one microsecond, so its time to first token (TTFT) is
273.376 microseconds. This distinction checks actual arrival and visibility,
not a post-hoc throughput multiplier.

## Evidence and chronology

The executed source is `b6620435f25e39b65a424592b019cf9b1c1c8307`.
The original prospective behavior freeze is
`73270f6699440aa4a17ec7b2af5155698ab643b2`, before implementation and any
campaign. Representation amendment
`7d756ad7cefdd8270c2a6b28e2b74dbe84139e6e` follows the first failed attempt.
The final pre-run expectation commit,
`1d7f5a9c1f1908f55e065f5d8ca77339bc5d17cb`, freezes complete first-exit receipt
retention after the second failure. Neither changes the behavior grid or caps.

The complete evidence classes remain separate:

- 14 component configurations and 34 native cells, including four historical
  controls, cover 112 requests in six fresh processes.
- All 18 required admission stages finish, with no violated fatal guard among
  562,493 unscored structural checks.
- All 36 exact oracle vectors agree: two component vectors, thirty native
  cell timelines and four historical comparisons.
- All 25 behavioral instances pass in six families: eight component instances
  in two families and seventeen native instances in four families.
- All sixteen deliberate corruption controls reject at their required
  boundaries. These controls are not additional behavioral points.

The three independent processes retain 335 complete engine steps, including
65 genuine zero-service drains, with 1,340 native checkpoints. Every step
joins to one receipt, five completion events and one engine visit. The
checkpoints retain native request/cache ownership, in-flight tokens, block
references, free queues and emitted frontend output. Full sink publications
remain unchanged until retirement and then appear exactly once. All
independent native work is drained before successful teardown; the serialized
compatibility path preserves its existing finished-request bookkeeping.

The slowest process finishes in 1,586.313 seconds, below 1,800 seconds. Maximum
sampled current resident memory is 1,456,748 KiB, below sixteen gibibytes.
These host observations establish campaign feasibility; they do not establish
a controlled wall-clock speedup. The separately qualified
[publication reader](../publication_snapshot_v1/RESULTS.md) and
[primitive dispatch](../snapshot_dispatch_v1/RESULTS.md) studies preserve full
model results and corruption checks while removing repeated host work.

The [portable publication](results.json) retains all 97 raw process-file
receipts, totaling 167,629,425 bytes, and ten root evidence receipts. Every
first-exit inventory matches its final inventory and the retained bytes.
The raw summary SHA-256 is
`a43d3e969c6d4445e72a93c6e4a77e501b80a86954459055f171ea750910e8e6`.

## Earlier retained attempts

| Attempt | Executed source | Fatal finding | Retained publication |
|---|---|---|---|
| First | `594fdfb001dc056b93fca5bef0595d548e754034` | Float/integer profile comparison mismatch after 24 requests; no process admitted | [Original record](void-frozen-v1.json) |
| Second | `ce8bf357e7477291d2a3cfc55b8a6875dd789c06` | Timeout after eight complete independent request rows; three serialized processes admitted | [Original record](void-frozen-v2.json) |
| Third | `72a395677ec56e9814a85518473940f2f95313c3` | Timeout after eighteen complete independent request rows; no independent process admitted | [Original record](void-frozen-v3.json) |

All three attempts remain VOID, with null behavioral scores and unchanged
portable receipts. The second failed process has a final receipt but no
initial pre-admission lock; its chronology remains explicit. The fourth run
supplies new complete evidence and does not rescore any earlier attempt.

## Project effect and limits

CORE-68's live independent completion claim is now literal for in-process
vLLM 0.27.1, one request per engine step, simulated workers and declared
isolated compute/collective/handoff service. This unblocks independent timing
as a prerequisite for the larger disaggregated serving loop.

CORE-52 retains the 448-worker host-scale campaign. CORE-51 and CORE-54 retain
their complete deployment and frontier obligations. CORE-69 retains pending
cancellation and recovery; CORE-70 retains shared-resource and wider native
composition. No real GPU, model weights or packet backend runs in this
campaign. These results qualify scheduling and visibility, without closing
large-model calibration, shared-fabric behavior or deployment prediction.
