# TRAF-72 corrected transport comparison and incast mesh

## Outcome

What came out: `PASS_WITH_HONEST_REFUTATIONS`. The mapping-audit verdict is
`MAPPING_DEFICIT_IN_FAIR_SHARE_ENTITY_NOT_CAPACITY_VALUE`. TRAF-71 gave its
degree-3 rnic-nn receiver exactly 207.101921876 GB/s, the same aggregate that
limits the NVLink composition. The capacity ratio was 1.000000 and could not
cause the legacy 512 KiB p50 ratio `1.664553`. The corrected
ordered-pair mapping moves rnic-nn from 30.203976 us to
18.120617 us while the regenerated NVLink value is 18.145397
us. The legacy-to-corrected ratio is `1.666829` against the
frozen `601/360 = 1.669444` queue-mapping prediction.

The fluid-reference verdict is `REFUTED`:
239 of 252
frozen location comparisons pass, and the independent continuous-service
oracle agrees to at most 0 ps. The
higher-degree small-flow tail hypothesis is `REFUTED` and the
fairness hypothesis is `REFUTED`. Frozen honest refutations:
H2, H3, H4.

What ran: all 42 rung-degree cells, nine frozen seeds and three transports,
for 77,112 flow samples. Degrees 1 through 3 retain the
TRAF-69 releases. Degrees 4, 8 and 16 instantiate the same scored constants on
the declared simulated mesh.

What it changes: TRAF-72 closes because the mapping correction, fluid null,
mesh extension, tail metrics, fairness metric, preservation guards, and
publication disclosures all execute. TRAF-71's degree-3 transport-effect
interpretation is superseded by this mapping audit.

What it does not change: the merged TRAF-71 directory remains byte-identical.
This study creates no new hardware evidence and does not close TRAF-65. The
degree-4, degree-8 and degree-16 topology has no NV4 hardware counterpart.

## Mapping audit

| Degree | Legacy receiver grant | Binding value |
|---:|---:|---|
| 1 | 100.000000 GB/s | ordered-pair class cap |
| 2 | 200.000000 GB/s | ordered-pair class cap |
| 3 | 207.101922 GB/s | RX ingress plateau |

At degree 3 the max-min allocator divided the full receiver plateau, not an
aggregate below it. The right-shift came from admitting every overlapping
application transfer as another max-min flow. The corrected adapter keeps one
active flow per ordered pair and queues later transfers in that class. On the
frozen `3S/4` release interval, the class-queued nearest-rank median is `9S/4`
while the legacy per-transfer processor-sharing median is `601S/160`. Their
ratio is `601/360 = 1.669444`, within
0.294
percent of the legacy observation before the correction.

## Fluid null reference

The exact continuous-byte oracle contains the same 100 GB/s source-class caps,
207.101921876 GB/s destination cap, release tuples, and ordered-pair queues as
the htsim fluid arm. It contains no packet, header, ACK, reverse byte, credit,
or propagation term. Its literal comparison result is reported above; any H2
miss is a harness or mapping finding unless the result mechanically identifies
a transport mechanism.

## Frozen hypothesis verdicts

| Hypothesis | Passed | Required | Verdict |
|---|---:|---:|---|
| H1 | 2 | 2 | PASS |
| H2 | 239 | 252 | REFUTED |
| H3 | 36 | 48 | REFUTED |
| H4 | 11 | 24 | REFUTED |
| H5 | 126 | 126 | PASS |

H5 is an exact/fatal result, not a directional score inferred from noisy
samples. It passed in all 126 transport cells: every flow completed, source
and destination allocations stayed within their caps, packet wire ledgers
were exact, and fluid carried payload bytes without packet or control bytes.

## Refutation diagnosis

H2 is refuted only by the 13 fluid-versus-rnic-nn packet comparisons below.
All 126 fluid-versus-NVLink comparisons pass, and all remaining
fluid-versus-packet comparisons pass. The deviations are mechanically
attributed to indivisible packet slots: a selected short packet flow can
finish before the equal continuous fluid shares finish. The zero-picosecond
fluid-oracle error shows that this is not a fluid harness defect. It refutes
the stronger claim that a capacity-bound fluid null must minimize every order
statistic of packetized fair service.

| Degree | Rung | Statistic | Fluid us | Packet reference us | Fluid/reference |
|---:|---|---|---:|---:|---:|
| 1 | 256 B | p99 | 0.004265 | 0.003164 | 1.347777 |
| 1 | 256 B | worst | 0.004265 | 0.003164 | 1.347777 |
| 1 | 1 KiB | p50 | 0.012370 | 0.011510 | 1.074698 |
| 1 | 1 KiB | p99 | 0.022272 | 0.019939 | 1.117042 |
| 1 | 1 KiB | worst | 0.022272 | 0.019939 | 1.117042 |
| 2 | 256 B | p99 | 0.004742 | 0.004217 | 1.124546 |
| 2 | 256 B | worst | 0.004742 | 0.004217 | 1.124546 |
| 8 | 256 B | p50 | 0.004864 | 0.004471 | 1.087798 |
| 8 | 256 B | p99 | 0.013738 | 0.012866 | 1.067801 |
| 8 | 256 B | worst | 0.013738 | 0.012866 | 1.067801 |
| 16 | 256 B | p50 | 0.013073 | 0.010237 | 1.277054 |
| 16 | 256 B | p99 | 0.027162 | 0.025572 | 1.062196 |
| 16 | 256 B | worst | 0.028140 | 0.027686 | 1.016402 |

For H3, both corrected references are strictly left of NVLink in every one of
the 36 frozen small-flow mesh tail comparisons. The refutation is the
"increasingly" clause: all 12 required nondecreasing-advantage checks fail.
The NVLink credit schedule's source rotation approaches the same receiver
sharing as degree grows, so the relative advantage shrinks or remains nearly
flat instead of increasing.

| Rung | Reference | p99 NV/reference at d4, d8, d16 | Worst NV/reference at d4, d8, d16 | Both nondecreasing |
|---|---|---|---|---|
| 256 B | rnic-nn packet | 2.227857, 1.541980, 1.294333 | 2.227857, 1.541980, 1.224253 | no |
| 256 B | rnic-nn fluid | 2.291562, 1.444070, 1.218545 | 2.291562, 1.444070, 1.204496 | no |
| 1 KiB | rnic-nn packet | 1.182627, 1.025295, 1.027997 | 1.182627, 1.025295, 1.018746 | no |
| 1 KiB | rnic-nn fluid | 1.337168, 1.306889, 1.376549 | 1.337168, 1.306889, 1.371179 | no |
| 4 KiB | rnic-nn packet | 1.024121, 1.008602, 1.005392 | 1.024121, 1.008602, 1.003724 | no |
| 4 KiB | rnic-nn fluid | 1.388609, 1.287104, 1.259961 | 1.388609, 1.287104, 1.259348 | no |

H4 passes 11 of 24 frozen comparisons. The 256 B packet-slot discreteness
makes both fair-share references less fair than the rotating NVLink schedule,
with the gap worsening through degree 16. At 1 KiB and 4 KiB the reference
gaps improve with degree, but several degree-4 and degree-8 no-lower checks
still fail. The table separates those two clauses.

| Rung | Reference | Reference Jain at d4, d8, d16 | NVLink Jain at d4, d8, d16 | No-lower passes | Gap nondecreasing |
|---|---|---|---|---:|---|
| 256 B | rnic-nn packet | 0.964180, 0.936229, 0.835371 | 0.996958, 0.993412, 0.971800 | 0/3 | no |
| 256 B | rnic-nn fluid | 0.976525, 0.938314, 0.887620 | 0.996958, 0.993412, 0.971800 | 0/3 | no |
| 1 KiB | rnic-nn packet | 0.991069, 0.992207, 0.995978 | 0.991995, 0.977629, 0.972645 | 2/3 | yes |
| 1 KiB | rnic-nn fluid | 0.994938, 0.993591, 0.996854 | 0.991995, 0.977629, 0.972645 | 3/3 | yes |
| 4 KiB | rnic-nn packet | 0.998603, 0.999087, 0.999650 | 0.999762, 0.999422, 0.999155 | 1/3 | yes |
| 4 KiB | rnic-nn fluid | 0.998524, 0.999093, 0.999724 | 0.999762, 0.999422, 0.999155 | 1/3 | yes |

## Tail and fairness tables

Each table reports p50, nearest-rank p99, worst-flow FCT, and Jain fairness.
Values are nine-seed means with the seed min-max in brackets.

### 256 B

| Degree | Transport | p50 us [seed min, max] | p99 us [seed min, max] | Worst us [seed min, max] | Jain fairness [seed min, max] |
|---:|---|---:|---:|---:|---:|
| 1 | NVLink credit | 0.012194 [0.012194, 0.012194] | 0.013067 [0.012194, 0.014282] | 0.013067 [0.012194, 0.014282] | 1.000000 [1.000000, 1.000000] |
| 1 | rnic-nn packet | 0.002627 [0.002627, 0.002627] | 0.003164 [0.002627, 0.003958] | 0.003164 [0.002627, 0.003958] | 1.000000 [1.000000, 1.000000] |
| 1 | rnic-nn fluid | 0.002560 [0.002560, 0.002560] | 0.004265 [0.002560, 0.006384] | 0.004265 [0.002560, 0.006384] | 1.000000 [1.000000, 1.000000] |
| 2 | NVLink credit | 0.012194 [0.012194, 0.012194] | 0.013895 [0.013310, 0.015038] | 0.013895 [0.013310, 0.015038] | 0.999298 [0.998470, 0.999667] |
| 2 | rnic-nn packet | 0.002627 [0.002627, 0.002627] | 0.004217 [0.003697, 0.004951] | 0.004217 [0.003697, 0.004951] | 0.991382 [0.980262, 0.995843] |
| 2 | rnic-nn fluid | 0.002560 [0.002560, 0.002560] | 0.004742 [0.003310, 0.006653] | 0.004742 [0.003310, 0.006653] | 0.987786 [0.978769, 0.999983] |
| 3 | NVLink credit | 0.012381 [0.012194, 0.013187] | 0.015977 [0.013642, 0.021076] | 0.015977 [0.013642, 0.021076] | 0.997134 [0.990706, 0.999287] |
| 3 | rnic-nn packet | 0.002801 [0.002627, 0.003619] | 0.006973 [0.004413, 0.012319] | 0.006973 [0.004413, 0.012319] | 0.974089 [0.946201, 0.989544] |
| 3 | rnic-nn fluid | 0.002752 [0.002560, 0.003529] | 0.006445 [0.004239, 0.011290] | 0.006445 [0.004239, 0.011290] | 0.979528 [0.944779, 0.995690] |
| 4 | NVLink credit | 0.012444 [0.012194, 0.013135] | 0.016464 [0.014119, 0.019260] | 0.016464 [0.014119, 0.019260] | 0.996958 [0.993579, 0.998846] |
| 4 | rnic-nn packet | 0.002865 [0.002627, 0.003545] | 0.007390 [0.005065, 0.010783] | 0.007390 [0.005065, 0.010783] | 0.964180 [0.938984, 0.982387] |
| 4 | rnic-nn fluid | 0.002823 [0.002560, 0.003407] | 0.007185 [0.003723, 0.011654] | 0.007185 [0.003723, 0.011654] | 0.976525 [0.949657, 0.996416] |
| 8 | NVLink credit | 0.014023 [0.013313, 0.016091] | 0.019839 [0.016631, 0.026689] | 0.019839 [0.016631, 0.026689] | 0.993412 [0.990476, 0.995548] |
| 8 | rnic-nn packet | 0.004471 [0.003734, 0.006546] | 0.012866 [0.007060, 0.025062] | 0.012866 [0.007060, 0.025062] | 0.936229 [0.919670, 0.949131] |
| 8 | rnic-nn fluid | 0.004864 [0.003962, 0.006611] | 0.013738 [0.008542, 0.023731] | 0.013738 [0.008542, 0.023731] | 0.938314 [0.921925, 0.960934] |
| 16 | NVLink credit | 0.019834 [0.017793, 0.022389] | 0.033098 [0.030302, 0.040505] | 0.033894 [0.031221, 0.041117] | 0.971800 [0.968159, 0.976839] |
| 16 | rnic-nn packet | 0.010237 [0.008192, 0.012813] | 0.025572 [0.021507, 0.035310] | 0.027686 [0.022158, 0.036387] | 0.835371 [0.807138, 0.870692] |
| 16 | rnic-nn fluid | 0.013073 [0.011425, 0.014427] | 0.027162 [0.023356, 0.033656] | 0.028140 [0.024809, 0.035422] | 0.887620 [0.830991, 0.943683] |

### 1 KiB

| Degree | Transport | p50 us [seed min, max] | p99 us [seed min, max] | Worst us [seed min, max] | Jain fairness [seed min, max] |
|---:|---|---:|---:|---:|---:|
| 1 | NVLink credit | 0.023717 [0.017270, 0.040751] | 0.035821 [0.023374, 0.052290] | 0.035821 [0.023374, 0.052290] | 1.000000 [1.000000, 1.000000] |
| 1 | rnic-nn packet | 0.011510 [0.010507, 0.015347] | 0.019939 [0.014925, 0.027622] | 0.019939 [0.014925, 0.027622] | 1.000000 [1.000000, 1.000000] |
| 1 | rnic-nn fluid | 0.012370 [0.010240, 0.021105] | 0.022272 [0.015704, 0.033330] | 0.022272 [0.015704, 0.033330] | 1.000000 [1.000000, 1.000000] |
| 2 | NVLink credit | 0.024938 [0.019142, 0.047464] | 0.036700 [0.029355, 0.053929] | 0.036700 [0.029355, 0.053929] | 0.991270 [0.983402, 0.998604] |
| 2 | rnic-nn packet | 0.013852 [0.011385, 0.026243] | 0.023870 [0.016582, 0.035180] | 0.023870 [0.016582, 0.035180] | 0.988796 [0.979495, 0.998585] |
| 2 | rnic-nn fluid | 0.012068 [0.010240, 0.023573] | 0.022425 [0.017078, 0.031430] | 0.022425 [0.017078, 0.031430] | 0.986944 [0.973867, 0.998310] |
| 3 | NVLink credit | 0.022114 [0.020078, 0.023823] | 0.030245 [0.025878, 0.036402] | 0.030245 [0.025878, 0.036402] | 0.995269 [0.991538, 0.997998] |
| 3 | rnic-nn packet | 0.014396 [0.013134, 0.015904] | 0.023258 [0.017545, 0.029116] | 0.023258 [0.017545, 0.029116] | 0.994472 [0.990652, 0.997221] |
| 3 | rnic-nn fluid | 0.013065 [0.011715, 0.014122] | 0.021151 [0.015938, 0.026453] | 0.021151 [0.015938, 0.026453] | 0.995605 [0.992289, 0.998118] |
| 4 | NVLink credit | 0.025940 [0.023726, 0.034824] | 0.038206 [0.029509, 0.048351] | 0.038206 [0.029509, 0.048351] | 0.991995 [0.988436, 0.994831] |
| 4 | rnic-nn packet | 0.018898 [0.015969, 0.029349] | 0.032306 [0.026511, 0.045220] | 0.032306 [0.026511, 0.045220] | 0.991069 [0.987290, 0.993350] |
| 4 | rnic-nn fluid | 0.017102 [0.015164, 0.025378] | 0.028572 [0.023462, 0.035930] | 0.028572 [0.023462, 0.035930] | 0.994938 [0.992395, 0.997214] |
| 8 | NVLink credit | 0.053621 [0.043292, 0.072701] | 0.076685 [0.054124, 0.097524] | 0.076685 [0.054124, 0.097524] | 0.977629 [0.964201, 0.989321] |
| 8 | rnic-nn packet | 0.050388 [0.040377, 0.070242] | 0.074793 [0.052697, 0.098939] | 0.074793 [0.052697, 0.098939] | 0.992207 [0.988540, 0.995054] |
| 8 | rnic-nn fluid | 0.040826 [0.035034, 0.054621] | 0.058678 [0.044331, 0.071419] | 0.058678 [0.044331, 0.071419] | 0.993591 [0.992497, 0.996361] |
| 16 | NVLink credit | 0.135224 [0.097163, 0.156159] | 0.222083 [0.175095, 0.257820] | 0.222784 [0.175369, 0.257846] | 0.972645 [0.962092, 0.982129] |
| 16 | rnic-nn packet | 0.140505 [0.102831, 0.161487] | 0.216035 [0.168745, 0.249349] | 0.218685 [0.172042, 0.250033] | 0.995978 [0.994390, 0.997711] |
| 16 | rnic-nn fluid | 0.117403 [0.079111, 0.140242] | 0.161333 [0.119705, 0.193629] | 0.162476 [0.124141, 0.194460] | 0.996854 [0.995294, 0.998285] |

### 4 KiB

| Degree | Transport | p50 us [seed min, max] | p99 us [seed min, max] | Worst us [seed min, max] | Jain fairness [seed min, max] |
|---:|---|---:|---:|---:|---:|
| 1 | NVLink credit | 0.127262 [0.107886, 0.178430] | 0.172411 [0.143525, 0.225471] | 0.172411 [0.143525, 0.225471] | 1.000000 [1.000000, 1.000000] |
| 1 | rnic-nn packet | 0.074590 [0.064511, 0.095219] | 0.106975 [0.085420, 0.134821] | 0.106975 [0.085420, 0.134821] | 1.000000 [1.000000, 1.000000] |
| 1 | rnic-nn fluid | 0.067020 [0.056350, 0.087499] | 0.094516 [0.071295, 0.121116] | 0.094516 [0.071295, 0.121116] | 1.000000 [1.000000, 1.000000] |
| 2 | NVLink credit | 0.135003 [0.074555, 0.178918] | 0.177737 [0.094401, 0.232403] | 0.177737 [0.094401, 0.232403] | 0.998555 [0.996715, 0.999804] |
| 2 | rnic-nn packet | 0.077424 [0.048985, 0.097133] | 0.112070 [0.058437, 0.160521] | 0.112070 [0.058437, 0.160521] | 0.998572 [0.997733, 0.999448] |
| 2 | rnic-nn fluid | 0.069626 [0.042531, 0.088193] | 0.099062 [0.050791, 0.145173] | 0.099062 [0.050791, 0.145173] | 0.998640 [0.997868, 0.999579] |
| 3 | NVLink credit | 0.103118 [0.089660, 0.122452] | 0.147855 [0.122864, 0.176517] | 0.147855 [0.122864, 0.176517] | 0.999847 [0.999749, 0.999941] |
| 3 | rnic-nn packet | 0.098280 [0.084741, 0.119015] | 0.141838 [0.115794, 0.172167] | 0.141838 [0.115794, 0.172167] | 0.998717 [0.996744, 0.999314] |
| 3 | rnic-nn fluid | 0.076324 [0.063607, 0.096536] | 0.102817 [0.075003, 0.129066] | 0.102817 [0.075003, 0.129066] | 0.998713 [0.997456, 0.999305] |
| 4 | NVLink credit | 0.141302 [0.118791, 0.175780] | 0.224902 [0.182639, 0.282611] | 0.224902 [0.182639, 0.282611] | 0.999762 [0.999633, 0.999833] |
| 4 | rnic-nn packet | 0.137896 [0.112170, 0.177390] | 0.219605 [0.180526, 0.274324] | 0.219605 [0.180526, 0.274324] | 0.998603 [0.997566, 0.999462] |
| 4 | rnic-nn fluid | 0.108716 [0.085242, 0.146470] | 0.161962 [0.120021, 0.215299] | 0.161962 [0.120021, 0.215299] | 0.998524 [0.995427, 0.999538] |
| 8 | NVLink credit | 0.347263 [0.323800, 0.362818] | 0.551329 [0.534437, 0.571835] | 0.551329 [0.534437, 0.571835] | 0.999422 [0.999322, 0.999488] |
| 8 | rnic-nn packet | 0.348451 [0.321862, 0.367335] | 0.546627 [0.529044, 0.568218] | 0.546627 [0.529044, 0.568218] | 0.999087 [0.998893, 0.999305] |
| 8 | rnic-nn fluid | 0.290893 [0.265205, 0.309536] | 0.428348 [0.411493, 0.449623] | 0.428348 [0.411493, 0.449623] | 0.999093 [0.998768, 0.999450] |
| 16 | NVLink credit | 0.714100 [0.697864, 0.723074] | 1.171357 [1.138727, 1.190496] | 1.172168 [1.139150, 1.191464] | 0.999155 [0.999030, 0.999230] |
| 16 | rnic-nn packet | 0.715983 [0.703329, 0.724628] | 1.165074 [1.131569, 1.187403] | 1.167819 [1.136864, 1.189251] | 0.999650 [0.999454, 0.999789] |
| 16 | rnic-nn fluid | 0.600374 [0.585872, 0.608784] | 0.929677 [0.897432, 0.950075] | 0.930774 [0.897741, 0.950817] | 0.999724 [0.999498, 0.999850] |

### 16 KiB

| Degree | Transport | p50 us [seed min, max] | p99 us [seed min, max] | Worst us [seed min, max] | Jain fairness [seed min, max] |
|---:|---|---:|---:|---:|---:|
| 1 | NVLink credit | 0.606642 [0.578655, 0.650124] | 0.828549 [0.799682, 0.879961] | 0.828549 [0.799682, 0.879961] | 1.000000 [1.000000, 1.000000] |
| 1 | rnic-nn packet | 0.360281 [0.341410, 0.372644] | 0.574806 [0.543833, 0.620694] | 0.574806 [0.543833, 0.620694] | 1.000000 [1.000000, 1.000000] |
| 1 | rnic-nn fluid | 0.309704 [0.290833, 0.322067] | 0.474966 [0.443993, 0.520854] | 0.474966 [0.443993, 0.520854] | 1.000000 [1.000000, 1.000000] |
| 2 | NVLink credit | 0.617702 [0.598146, 0.637427] | 0.844053 [0.817783, 0.872790] | 0.844053 [0.817783, 0.872790] | 0.999860 [0.999673, 0.999969] |
| 2 | rnic-nn packet | 0.361997 [0.345279, 0.375121] | 0.600727 [0.584046, 0.613509] | 0.600727 [0.584046, 0.613509] | 0.999893 [0.999785, 0.999966] |
| 2 | rnic-nn fluid | 0.303086 [0.285939, 0.316329] | 0.485844 [0.468876, 0.499222] | 0.485844 [0.468876, 0.499222] | 0.999891 [0.999769, 0.999954] |
| 3 | NVLink credit | 0.522291 [0.504281, 0.539675] | 0.847714 [0.828097, 0.872791] | 0.847714 [0.828097, 0.872791] | 0.999996 [0.999994, 0.999999] |
| 3 | rnic-nn packet | 0.521196 [0.503126, 0.543446] | 0.842318 [0.821062, 0.863901] | 0.842318 [0.821062, 0.863901] | 0.999902 [0.999777, 0.999951] |
| 3 | rnic-nn fluid | 0.430228 [0.412849, 0.451906] | 0.664370 [0.643601, 0.686239] | 0.664370 [0.643601, 0.686239] | 0.999908 [0.999793, 0.999965] |
| 4 | NVLink credit | 0.724247 [0.709524, 0.739014] | 1.184116 [1.152722, 1.218203] | 1.184116 [1.152722, 1.218203] | 0.999992 [0.999989, 0.999995] |
| 4 | rnic-nn packet | 0.723506 [0.704584, 0.734080] | 1.179072 [1.146401, 1.211295] | 1.179072 [1.146401, 1.211295] | 0.999899 [0.999775, 0.999959] |
| 4 | rnic-nn fluid | 0.605053 [0.585707, 0.614438] | 0.941682 [0.908592, 0.974828] | 0.941682 [0.908592, 0.974828] | 0.999882 [0.999730, 0.999969] |
| 8 | NVLink credit | 1.479572 [1.459979, 1.497248] | 2.440441 [2.405048, 2.471733] | 2.440441 [2.405048, 2.471733] | 0.999974 [0.999970, 0.999977] |
| 8 | rnic-nn packet | 1.477930 [1.458188, 1.496184] | 2.433701 [2.401453, 2.466145] | 2.433701 [2.401453, 2.466145] | 0.999970 [0.999960, 0.999984] |
| 8 | rnic-nn fluid | 1.242866 [1.224776, 1.259281] | 1.959566 [1.926026, 1.990912] | 1.959566 [1.926026, 1.990912] | 0.999975 [0.999952, 0.999987] |
| 16 | NVLink credit | 2.994881 [2.968235, 3.026810] | 4.970473 [4.950629, 4.997563] | 4.971324 [4.951376, 4.997838] | 0.999950 [0.999944, 0.999956] |
| 16 | rnic-nn packet | 2.993949 [2.961322, 3.023232] | 4.957320 [4.936197, 4.984490] | 4.959098 [4.937998, 4.986277] | 0.999977 [0.999956, 0.999986] |
| 16 | rnic-nn fluid | 2.522418 [2.493340, 2.554161] | 4.010812 [3.991696, 4.036876] | 4.011777 [3.992666, 4.037394] | 0.999981 [0.999957, 0.999989] |

### 64 KiB

| Degree | Transport | p50 us [seed min, max] | p99 us [seed min, max] | Worst us [seed min, max] | Jain fairness [seed min, max] |
|---:|---|---:|---:|---:|---:|
| 1 | NVLink credit | 2.582308 [2.550218, 2.599424] | 3.519333 [3.490391, 3.546432] | 3.519333 [3.490391, 3.546432] | 1.000000 [1.000000, 1.000000] |
| 1 | rnic-nn packet | 1.539132 [1.503068, 1.554520] | 2.551251 [2.520707, 2.569910] | 2.551251 [2.520707, 2.569910] | 1.000000 [1.000000, 1.000000] |
| 1 | rnic-nn fluid | 1.301364 [1.265300, 1.316752] | 2.077029 [2.046485, 2.095688] | 2.077029 [2.046485, 2.095688] | 1.000000 [1.000000, 1.000000] |
| 2 | NVLink credit | 2.589703 [2.550254, 2.618862] | 3.533085 [3.484615, 3.576117] | 3.533085 [3.484615, 3.576117] | 0.999993 [0.999990, 0.999998] |
| 2 | rnic-nn packet | 1.549361 [1.516909, 1.565459] | 2.559526 [2.524737, 2.581516] | 2.559526 [2.524737, 2.581516] | 0.999995 [0.999993, 0.999998] |
| 2 | rnic-nn fluid | 1.306801 [1.270905, 1.327202] | 2.076679 [2.049641, 2.102886] | 2.076679 [2.049641, 2.102886] | 0.999995 [0.999992, 0.999998] |
| 3 | NVLink credit | 2.237491 [2.213899, 2.249593] | 3.700773 [3.670871, 3.723215] | 3.700773 [3.670871, 3.723215] | 1.000000 [1.000000, 1.000000] |
| 3 | rnic-nn packet | 2.233527 [2.205339, 2.246804] | 3.689740 [3.655434, 3.714923] | 3.689740 [3.655434, 3.714923] | 0.999993 [0.999988, 0.999998] |
| 3 | rnic-nn fluid | 1.876427 [1.848847, 1.888446] | 2.977534 [2.942781, 3.002441] | 2.977534 [2.942781, 3.002441] | 0.999994 [0.999985, 0.999998] |
| 4 | NVLink credit | 2.997931 [2.987710, 3.006294] | 4.975058 [4.935485, 5.017074] | 4.975058 [4.935485, 5.017074] | 0.999999 [0.999999, 1.000000] |
| 4 | rnic-nn packet | 2.994094 [2.978167, 3.006193] | 4.961621 [4.922948, 5.000714] | 4.961621 [4.922948, 5.000714] | 0.999995 [0.999990, 0.999998] |
| 4 | rnic-nn fluid | 2.519844 [2.504906, 2.532522] | 4.012383 [3.973715, 4.050307] | 4.012383 [3.973715, 4.050307] | 0.999995 [0.999990, 0.999998] |
| 8 | NVLink credit | 6.019517 [6.007739, 6.042168] | 10.016596 [9.992751, 10.057924] | 10.016596 [9.992751, 10.057924] | 0.999998 [0.999998, 0.999999] |
| 8 | rnic-nn packet | 6.014287 [6.001046, 6.040767] | 9.996907 [9.972360, 10.038189] | 9.996907 [9.972360, 10.038189] | 0.999997 [0.999994, 0.999998] |
| 8 | rnic-nn fluid | 5.066938 [5.052994, 5.091330] | 8.098149 [8.074112, 8.139349] | 8.098149 [8.074112, 8.139349] | 0.999997 [0.999992, 0.999999] |
| 16 | NVLink credit | 12.088984 [12.075619, 12.104586] | 20.120428 [20.094050, 20.143851] | 20.121277 [20.094938, 20.145164] | 0.999997 [0.999997, 0.999997] |
| 16 | rnic-nn packet | 12.076946 [12.061533, 12.091934] | 20.084372 [20.058586, 20.106750] | 20.086152 [20.061668, 20.108004] | 0.999999 [0.999998, 0.999999] |
| 16 | rnic-nn fluid | 10.181565 [10.168816, 10.198894] | 16.288987 [16.263491, 16.311256] | 16.289988 [16.263999, 16.312252] | 0.999999 [0.999999, 0.999999] |

### 256 KiB

| Degree | Transport | p50 us [seed min, max] | p99 us [seed min, max] | Worst us [seed min, max] | Jain fairness [seed min, max] |
|---:|---|---:|---:|---:|---:|
| 1 | NVLink credit | 10.404909 [10.376299, 10.439683] | 14.233747 [14.189850, 14.269370] | 14.233747 [14.189850, 14.269370] | 1.000000 [1.000000, 1.000000] |
| 1 | rnic-nn packet | 6.228539 [6.202192, 6.249227] | 10.358911 [10.333698, 10.384924] | 10.358911 [10.333698, 10.384924] | 1.000000 [1.000000, 1.000000] |
| 1 | rnic-nn fluid | 5.257768 [5.231421, 5.278456] | 8.418683 [8.393470, 8.444696] | 8.418683 [8.393470, 8.444696] | 1.000000 [1.000000, 1.000000] |
| 2 | NVLink credit | 10.414850 [10.368044, 10.436268] | 14.252626 [14.203874, 14.298117] | 14.252626 [14.203874, 14.298117] | 0.999999 [0.999999, 1.000000] |
| 2 | rnic-nn packet | 6.243833 [6.213463, 6.259574] | 10.388654 [10.346262, 10.417913] | 10.388654 [10.346262, 10.417913] | 1.000000 [0.999999, 1.000000] |
| 2 | rnic-nn fluid | 5.264457 [5.234499, 5.279736] | 8.433119 [8.391587, 8.462051] | 8.433119 [8.391587, 8.462051] | 1.000000 [0.999999, 1.000000] |
| 3 | NVLink credit | 9.044894 [9.023967, 9.067481] | 15.063676 [15.034296, 15.098985] | 15.063676 [15.034296, 15.098985] | 1.000000 [1.000000, 1.000000] |
| 3 | rnic-nn packet | 9.032608 [9.010344, 9.058515] | 15.034660 [15.001961, 15.069541] | 15.034660 [15.001961, 15.069541] | 1.000000 [0.999999, 1.000000] |
| 3 | rnic-nn fluid | 7.607423 [7.585039, 7.632388] | 12.186763 [12.153599, 12.221167] | 12.186763 [12.153599, 12.221167] | 1.000000 [0.999999, 1.000000] |
| 4 | NVLink credit | 12.086650 [12.068665, 12.111189] | 20.123545 [20.097275, 20.163138] | 20.123545 [20.097275, 20.163138] | 1.000000 [1.000000, 1.000000] |
| 4 | rnic-nn packet | 12.070293 [12.054088, 12.092828] | 20.087616 [20.059570, 20.128577] | 20.087616 [20.059570, 20.128577] | 1.000000 [0.999999, 1.000000] |
| 4 | rnic-nn fluid | 10.172083 [10.154722, 10.194177] | 16.290128 [16.262713, 16.331185] | 16.290128 [16.262713, 16.331185] | 1.000000 [0.999999, 1.000000] |
| 8 | NVLink credit | 24.203078 [24.170617, 24.235332] | 40.314895 [40.276125, 40.343740] | 40.314895 [40.276125, 40.343740] | 1.000000 [1.000000, 1.000000] |
| 8 | rnic-nn packet | 24.171591 [24.143043, 24.200140] | 40.250000 [40.208752, 40.280282] | 40.250000 [40.208752, 40.280282] | 1.000000 [1.000000, 1.000000] |
| 8 | rnic-nn fluid | 20.376600 [20.347365, 20.405888] | 32.655028 [32.613967, 32.684975] | 32.655028 [32.613967, 32.684975] | 1.000000 [1.000000, 1.000000] |
| 16 | NVLink credit | 48.441771 [48.433646, 48.464879] | 80.736252 [80.707291, 80.767565] | 80.737146 [80.708376, 80.768714] | 1.000000 [1.000000, 1.000000] |
| 16 | rnic-nn packet | 48.380762 [48.369364, 48.409323] | 80.606185 [80.574505, 80.637474] | 80.609048 [80.575743, 80.641650] | 1.000000 [1.000000, 1.000000] |
| 16 | rnic-nn fluid | 40.790421 [40.780265, 40.816431] | 65.419593 [65.389044, 65.452366] | 65.420708 [65.389219, 65.452437] | 1.000000 [1.000000, 1.000000] |

### 512 KiB

| Degree | Transport | p50 us [seed min, max] | p99 us [seed min, max] | Worst us [seed min, max] | Jain fairness [seed min, max] |
|---:|---|---:|---:|---:|---:|
| 1 | NVLink credit | 20.862777 [20.836455, 20.923820] | 28.555776 [28.533094, 28.631845] | 28.555776 [28.533094, 28.631845] | 1.000000 [1.000000, 1.000000] |
| 1 | rnic-nn packet | 12.497167 [12.469719, 12.531167] | 20.811098 [20.789905, 20.842782] | 20.811098 [20.789905, 20.842782] | 1.000000 [1.000000, 1.000000] |
| 1 | rnic-nn fluid | 10.541179 [10.513731, 10.575179] | 16.900435 [16.879242, 16.932119] | 16.900435 [16.879242, 16.932119] | 1.000000 [1.000000, 1.000000] |
| 2 | NVLink credit | 20.876967 [20.842674, 20.929017] | 28.581078 [28.525304, 28.632787] | 28.581078 [28.525304, 28.632787] | 1.000000 [1.000000, 1.000000] |
| 2 | rnic-nn packet | 12.510574 [12.482478, 12.548916] | 20.840368 [20.795342, 20.860758] | 20.840368 [20.795342, 20.860758] | 1.000000 [1.000000, 1.000000] |
| 2 | rnic-nn fluid | 10.546611 [10.519923, 10.584104] | 16.914623 [16.869827, 16.934705] | 16.914623 [16.869827, 16.934705] | 1.000000 [1.000000, 1.000000] |
| 3 | NVLink credit | 18.145397 [18.127387, 18.162071] | 30.217367 [30.194122, 30.239789] | 30.217367 [30.194122, 30.239789] | 1.000000 [1.000000, 1.000000] |
| 3 | rnic-nn packet | 18.120617 [18.100158, 18.136524] | 30.166705 [30.144002, 30.184571] | 30.166705 [30.144002, 30.184571] | 1.000000 [1.000000, 1.000000] |
| 3 | rnic-nn fluid | 15.270949 [15.250320, 15.286468] | 24.470691 [24.447170, 24.488924] | 24.470691 [24.447170, 24.488924] | 1.000000 [1.000000, 1.000000] |
| 4 | NVLink credit | 24.193948 [24.171881, 24.211592] | 40.318119 [40.300339, 40.363379] | 40.318119 [40.300339, 40.363379] | 1.000000 [1.000000, 1.000000] |
| 4 | rnic-nn packet | 24.160819 [24.140537, 24.175654] | 40.252751 [40.235118, 40.294233] | 40.252751 [40.235118, 40.294233] | 1.000000 [1.000000, 1.000000] |
| 4 | rnic-nn fluid | 20.363677 [20.343926, 20.379240] | 32.658177 [32.639921, 32.700831] | 32.658177 [32.639921, 32.700831] | 1.000000 [1.000000, 1.000000] |
| 8 | NVLink credit | 48.435331 [48.412640, 48.456164] | 80.729660 [80.689258, 80.763599] | 80.729660 [80.689258, 80.763599] | 1.000000 [1.000000, 1.000000] |
| 8 | rnic-nn packet | 48.373098 [48.347419, 48.397497] | 80.601181 [80.563577, 80.633256] | 80.601181 [80.563577, 80.633256] | 1.000000 [1.000000, 1.000000] |
| 8 | rnic-nn fluid | 40.780680 [40.756661, 40.805957] | 65.412016 [65.373211, 65.443685] | 65.412016 [65.373211, 65.443685] | 1.000000 [1.000000, 1.000000] |
| 16 | NVLink credit | 96.913135 [96.898905, 96.933448] | 161.548597 [161.517681, 161.570009] | 161.549432 [161.518526, 161.570952] | 1.000000 [1.000000, 1.000000] |
| 16 | rnic-nn packet | 96.789678 [96.770394, 96.808454] | 161.293445 [161.261321, 161.314784] | 161.295480 [161.262087, 161.318103] | 1.000000 [1.000000, 1.000000] |
| 16 | rnic-nn fluid | 81.604512 [81.588762, 81.622146] | 130.916900 [130.887992, 130.935438] | 130.918569 [130.888163, 130.941042] | 1.000000 [1.000000, 1.000000] |


## Topology and measurement limits

SIMULATED MESH EXTRAPOLATION at degrees 4/8/16; no NV4 hardware counterpart; NVSwitch-class hardware is the physical route.

SMALL-FLOW INCAST IS A MODEL PREDICTION; real true-sync launch is not constructible through sequential PCIe writes; hardware identification is long-flow only.

The simulated-mesh constants are a topology extrapolation, not a claim that
an NV4 node can host more than three senders into its fourth GPU. An
NVSwitch-class configuration is the physical route to higher degrees.

## Figures

- [`figures/nvlink-rnic-fluid-fct-cdf-physical.pdf`](figures/nvlink-rnic-fluid-fct-cdf-physical.pdf)
- [`figures/nvlink-rnic-fluid-fct-cdf-mesh.pdf`](figures/nvlink-rnic-fluid-fct-cdf-mesh.pdf)
- [`figures/nvlink-rnic-fluid-tail.pdf`](figures/nvlink-rnic-fluid-tail.pdf)
- [`figures/nvlink-rnic-fluid-fairness.pdf`](figures/nvlink-rnic-fluid-fairness.pdf)
- [`figures/nvlink-rnic-mapping-audit-degree-3.pdf`](figures/nvlink-rnic-mapping-audit-degree-3.pdf)

Every PDF has a matching PNG. Every figure identifies simulated, measured,
declared and structural evidence, and carries the applicable topology and
measurement disclosure. The final PNGs were visually inspected after render.

## Preservation and reproducibility

Two pre-score runs are retained for audit. `traf72-final-attempt1` stopped
before scoring when queued pair classes were not admitted after release
exhaustion. `traf72-final-attempt2` stopped before scoring when the canonical
NV4 endpoint-count guard rejected the declared degree-4 mesh. Both harness
defects were corrected with tests. `traf72-final-attempt3` is the sole
evaluation of record, and neither stopped run contributes a reported sample.

All 16 merged TRAF-71 files pass
their frozen byte hashes. The run authority is expectations commit
`8e69696ba22a600a9aefab21c9f5d93e3f977a77` with SHA-256
`9724d405c400d5e38582fd869f24866f31fc6e0907d4b1b558b620eb411324bb`. The adapter is built from htsim
commit `1dcbfec36a33753bf978cf6323bade1a6645fe4f` and has executable SHA-256
`963d089491fd1ed15b0a14bad149c8c81e89d78af4806f7e7383431c9a9faae2`. Bulk samples, CDF rows,
per-cell schedules and manifests remain outside Git; their hashes are recorded
in `results.json`.
