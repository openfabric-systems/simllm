# Matched-seam aggregate-arm result

## Outcome

What ran: append-only attempt 0002 composed the 25 frozen aggregate operating
points twice in fresh processes from the imported Qwen3-32B-FP8 measured
operation database, evaluated every declared remove-one sensitivity, and
rendered both aggregate-qualified figure pairs.

What came out: the run is nonvoid and Family AR passes 25 of 25. The composed
TPOT quotient spans **0.999930063 to 1.000074536** against the frozen aggregate
table, so the largest absolute difference from one is 0.007454 percent. The two
fresh-process evaluation payloads are byte-identical at SHA-256
`0f37ed4ce38ca2b985501fc3072be046917cd3c57f272f0cd10c5579e08ed16f`.
Every TTFT residual is inside the frozen plus or minus 0.0005 ms publication
rounding bound.

What it changes for the project: DEPLOY-22 closes. The publication now carries
a SimLLM aggregate counterpart for the external grey series, using both the
unpriced and packet traffic definitions. Those two aggregate arms coincide
exactly because the co-located pool has zero prefill/decode handoff bytes and
starts no packet process. Agreement is carried chiefly by the aggregate TTFT
queueing heuristic, the TensorRT-LLM three-step TPOT count correction, and the
imported 3 microsecond memory-operation constant. No DEPLOY-23 or DEPLOY-24
residual is registered.

What it does not change: this is composition parity at one imported measured
seam, not an H200 hardware validation or a new kernel calibration. It does not
change the protected disaggregated scores, the F-2-09 rounded-axis refutation,
DEPLOY-13, or any earlier matched-seam artifact. The measured per-operation TP
collectives stay inside the imported pass timing; zero P/D handoff does not
mean zero internal collective cost.

## Aggregate TPOT quotient

The quotient is SimLLM composed TPOT divided by the published aggregate TPOT.
The frozen acceptance band is [0.98, 1.02].

| Row | TP | Batch | Context budget | Published TPOT (ms) | SimLLM TPOT (ms) | Quotient |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 20 | 4000 | 10.009 | 10.008580 | 0.999958050 |
| 2 | 4 | 16 | 4000 | 8.810 | 8.809773 | 0.999974265 |
| 3 | 4 | 15 | 4000 | 8.563 | 8.563240 | 1.000028075 |
| 4 | 4 | 14 | 4000 | 8.236 | 8.235983 | 0.999997899 |
| 5 | 4 | 13 | 4000 | 8.028 | 8.027812 | 0.999976582 |
| 6 | 4 | 12 | 4000 | 7.750 | 7.750053 | 1.000006806 |
| 7 | 4 | 11 | 4000 | 7.482 | 7.481962 | 0.999994979 |
| 8 | 4 | 10 | 4000 | 7.255 | 7.255134 | 1.000018454 |
| 9 | 4 | 9 | 4000 | 6.946 | 6.945523 | 0.999931345 |
| 10 | 8 | 16 | 8000 | 6.204 | 6.203566 | 0.999930063 |
| 11 | 8 | 15 | 8000 | 6.163 | 6.163348 | 1.000056418 |
| 12 | 8 | 14 | 8000 | 5.848 | 5.848004 | 1.000000630 |
| 13 | 8 | 13 | 8000 | 5.801 | 5.800811 | 0.999967472 |
| 14 | 8 | 12 | 8000 | 5.512 | 5.511697 | 0.999945090 |
| 15 | 8 | 11 | 8000 | 5.450 | 5.450003 | 1.000000469 |
| 16 | 8 | 10 | 8000 | 5.130 | 5.130113 | 1.000021940 |
| 17 | 8 | 9 | 8000 | 5.075 | 5.075095 | 1.000018627 |
| 18 | 8 | 8 | 8000 | 4.789 | 4.789322 | 1.000067318 |
| 19 | 8 | 7 | 8000 | 4.754 | 4.754354 | 1.000074536 |
| 20 | 8 | 6 | 8000 | 4.712 | 4.711735 | 0.999943843 |
| 21 | 8 | 5 | 4000 | 4.703 | 4.703292 | 1.000062095 |
| 22 | 8 | 5 | 8000 | 4.667 | 4.667058 | 1.000012514 |
| 23 | 8 | 4 | 8000 | 4.645 | 4.644774 | 0.999951385 |
| 24 | 8 | 4 | 4000 | 4.540 | 4.539957 | 0.999990558 |
| 25 | 8 | 3 | 4000 | 4.485 | 4.484978 | 0.999995105 |

The agreement is a composition result. Every positive service component comes
from the same imported measured database used by the external table. SimLLM
adds no kernel duration and never reaches its `RooflineProvider` on this arm.

## Operating-point TTFT decomposition

Published aggregate TTFT is not isolated prefill service. It is the pure
prefill step times the required pass count, followed by the queueing component
for that operating point. `Residual` is published TTFT minus composed TTFT.

| Row | Pure prefill (ms) | Passes | Base prefill (ms) | Queue factor | Queue component (ms) | SimLLM TTFT (ms) | Published (ms) | Residual (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100.764594 | 1 | 100.764594 | 2.850 | 186.414499 | 287.179094 | 287.179 | -0.000093785 |
| 2 | 100.562205 | 1 | 100.562205 | 2.650 | 165.927638 | 266.489843 | 266.490 | +0.000157174 |
| 3 | 100.495318 | 1 | 100.495318 | 2.600 | 160.792509 | 261.287827 | 261.288 | +0.000172548 |
| 4 | 100.428431 | 1 | 100.428431 | 2.550 | 155.664067 | 256.092498 | 256.092 | -0.000497932 |
| 5 | 100.361542 | 1 | 100.361542 | 2.500 | 150.542313 | 250.903854 | 250.904 | +0.000145572 |
| 6 | 100.294652 | 1 | 100.294652 | 2.450 | 145.427245 | 245.721897 | 245.722 | +0.000102897 |
| 7 | 100.227761 | 1 | 100.227761 | 2.400 | 140.318865 | 240.546626 | 240.547 | +0.000373880 |
| 8 | 100.160869 | 1 | 100.160869 | 2.350 | 135.217173 | 235.378042 | 235.378 | -0.000041644 |
| 9 | 100.093976 | 1 | 100.093976 | 2.300 | 130.122168 | 230.216144 | 230.216 | -0.000143837 |
| 10 | 127.620006 | 1 | 127.620006 | 2.250 | 159.525007 | 287.145013 | 287.145 | -0.000013410 |
| 11 | 127.591664 | 1 | 127.591664 | 2.250 | 159.489580 | 287.081243 | 287.081 | -0.000243221 |
| 12 | 127.563321 | 1 | 127.563321 | 2.200 | 153.075986 | 280.639307 | 280.639 | -0.000306962 |
| 13 | 127.534979 | 1 | 127.534979 | 2.200 | 153.041975 | 280.576954 | 280.577 | +0.000046115 |
| 14 | 127.506637 | 1 | 127.506637 | 2.150 | 146.632632 | 274.139269 | 274.139 | -0.000268969 |
| 15 | 127.478294 | 1 | 127.478294 | 2.150 | 146.600039 | 274.078333 | 274.078 | -0.000333004 |
| 16 | 127.449952 | 1 | 127.449952 | 2.100 | 140.194947 | 267.644899 | 267.645 | +0.000100569 |
| 17 | 127.433849 | 1 | 127.433849 | 2.100 | 140.177234 | 267.611084 | 267.611 | -0.000083657 |
| 18 | 127.417747 | 1 | 127.417747 | 2.050 | 133.788634 | 261.206381 | 261.206 | -0.000380550 |
| 19 | 127.401644 | 1 | 127.401644 | 2.050 | 133.771726 | 261.173370 | 261.173 | -0.000369910 |
| 20 | 127.385541 | 1 | 127.385541 | 2.000 | 127.385541 | 254.771082 | 254.771 | -0.000082212 |
| 21 | 74.773096 | 1 | 74.773096 | 2.100 | 82.250406 | 157.023501 | 157.024 | +0.000498550 |
| 22 | 127.364771 | 1 | 127.364771 | 2.000 | 127.364771 | 254.729542 | 254.730 | +0.000457587 |
| 23 | 127.344001 | 1 | 127.344001 | 1.950 | 120.976801 | 248.320803 | 248.321 | +0.000197453 |
| 24 | 74.757160 | 1 | 74.757160 | 2.050 | 78.495018 | 153.252178 | 153.252 | -0.000177512 |
| 25 | 74.741222 | 1 | 74.741222 | 2.000 | 74.741222 | 149.482444 | 149.482 | -0.000444423 |

Pure prefill spans 74.741222 to 127.620006 ms. Queueing adds 74.741222
to 186.414499 ms, with factors from 1.95 to 2.85. This is why matching the
published 149.482 to 287.179 ms TTFT values directly against an isolated
prefill pass would be wrong.

## Aggregate adjustment sensitivity

Each row removes exactly one declared factor. The quotient ranges remain
against the published aggregate table. `Complete baseline identity` applies
when the factor is unreachable by both metrics; metric-specific byte identity
is also enforced for each one-sided factor in the record.

| Factor removed | TPOT reachable | TTFT reachable | TPOT quotient range | TTFT quotient range | Complete baseline identity |
|---|:---:|:---:|---:|---:|:---:|
| `prefill_latency_correction` | no | no | 0.999930063 to 1.000074536 | 0.999996825 to 1.000002973 | yes |
| `decode_latency_correction` | no | no | 0.999930063 to 1.000074536 | 0.999996825 to 1.000002973 | yes |
| `prefill_rate_matching_degradation` | no | no | 0.999930063 to 1.000074536 | 0.999996825 to 1.000002973 | yes |
| `decode_rate_matching_degradation` | no | no | 0.999930063 to 1.000074536 | 0.999996825 to 1.000002973 | yes |
| `autoscale_ttft_correction` | no | no | 0.999930063 to 1.000074536 | 0.999996825 to 1.000002973 | yes |
| `memory_bandwidth_empirical_scale` | yes | yes | 0.994353381 to 0.999240408 | 0.981435704 to 0.985975253 | no |
| `memory_empirical_constant_latency` | yes | yes | 0.870898115 to 0.942110113 | 0.992253423 to 0.995463140 | no |
| `context_attention_extra_latency_correction` | yes | yes | 0.999170510 to 0.999977012 | 0.996915199 to 0.998194901 | no |
| `aggregate_ttft_queueing_heuristic` | no | yes | 0.999930063 to 1.000074536 | 0.350877308 to 0.512820105 | no |
| `trtllm_tpot_mixed_step_reduction` | yes | no | 1.052782053 to 1.154867076 | 0.999996825 to 1.000002973 | no |

The mechanism attribution is direct:

- Removing the three-step mixed-count correction raises TPOT to 1.052782 to
  1.154867 of publication. It is the dominant aggregate-specific TPOT term.
- Removing the 3 microsecond memory-operation constant lowers TPOT to
  0.870898 to 0.942110. It is the dominant imported pass-level TPOT factor.
- Removing the queueing heuristic lowers TTFT to 0.350877 to 0.512820. It is
  the dominant reason published aggregate TTFT differs from pure prefill.
- The HBM bandwidth derating and context-attention correction have smaller,
  still observable effects. All five disaggregated-only factors reproduce the
  complete aggregate baseline byte for byte.

## Physical sanity

The frozen napkin bounds are satisfied from independent angles:

- Weight streaming gives 1.666667 ms at TP4 and 0.833333 ms at TP8. Observed
  generation-only service is 4.343617 to 6.794305 ms and TPOT is 4.484978 to
  10.008580 ms, all above the applicable floor.
- Mixed-step compute floors span 16.186047 to 32.509606 ms. Observed mixed
  service spans 74.741222 to 127.620006 ms.
- Published request-latency ceilings span 2,387.486 to 5,281.461 ms. Every
  mixed step, generation-only step, TTFT and TPOT is positive and below its
  row's ceiling.
- The optimistic batch-amortized HBM ceilings span 450 to 3,000 tokens/s/GPU.
  Observed aggregate throughput spans 78.524919 to 472.846115 tokens/s/GPU and
  satisfies both plotted axes and the request-rate identity exactly.

These bounds reject unit, repeat-count and physically impossible throughput
failures. They do not by themselves validate silicon accuracy.

## Network arms and figure

The unpriced aggregate arm and packet aggregate arm each record zero handoff
bytes, zero flows and zero native invocations. Their complete scheduling,
TTFT, TPOT, throughput, axes and completion projection are byte-identical.

The new study and publication figures retain the protected panels and add a
hollow purple marker for the SimLLM unpriced aggregate arm plus a pink cross
for the SimLLM packet aggregate arm. The markers occupy identical coordinates
but remain separately visible. Every legend entry names aggregate or
disaggregated strategy and its traffic definition. Both PDF/PNG pairs passed
visual inspection with no clipping, hidden labels or panel overlap.

## Guards, attempts and disposition

All eleven fatal guards pass. Attempt 0002 ran from implementation commit
`6450a656e826bea325ec1127d4b2f7ab9be910e0`, finished in 283.608026 seconds
against the 600 second bound, and records W 1 of 1. The expectation freeze is
commit `923c384b786d55530084b33dfeedb2790752cb22`.

Attempt 0001 was numerically nonvoid with the same 25 of 25 TPOT result, but
post-run visual inspection found its publication legend clipped at both page
edges. Its append-only evidence remains under attempt 0001. The legend layout
was corrected and committed before attempt 0002, so only attempt 0002 supplies
the completed figure criterion and closes DEPLOY-22.
