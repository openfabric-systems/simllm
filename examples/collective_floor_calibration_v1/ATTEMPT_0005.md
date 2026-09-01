# Collective completion attempt 0005

What ran: the frozen training-only paired-operation anchor from commit
`427b443` was evaluated once over the unchanged 63 Family H holdouts and the
unchanged D8 matched coordinate.

What came out: Family H is refuted at 46 of 63, with 4.2288 percent median
relative error and 23.4354 percent p95 relative error. D8 passes at quotient
0.946736591 against the unchanged `[0.90, 1.10]` band.

What it changes: the paired-operation local-trend candidate is retired. It
shows that all-gather and reduce-scatter do not carry the same local regime
closely enough for one operation to serve as the other's anchor after a broad
trend-ratio correction. TRAF-76 stays open, and another expectations-only
freeze is required before a replacement is implemented.

What it does not change: no prior artifact, band, cell membership, accepted
bypass, compatibility transfer, MiniMax consumer, or packet claim changes.
The serialized authority hash before holdout loading is
`f204dd7c34a2d99db02c8eb9d58ea5abe1e4f83a02a7fddf7cbfeedfad06eeed`.

## Physical sanity before the score

The floor for every row is ring endpoint bytes divided by 450 GB/s. Every
measured and modeled completion is above that floor. The source still supplies
no finite algorithm-progress ceiling, so the ceiling remains unbounded.

## Family H ledger

| Cell | Bytes | Measured us | Attempt 0004 error | Attempt 0005 us | Attempt 0005 error | Result |
|---|---:|---:|---:|---:|---:|---|
| half/all_gather/r2/i01 | 1,024 | 5.310000 | 2.2478% | 5.274928 | 0.6605% | PASS |
| half/all_gather/r2/i03 | 4,096 | 5.380000 | 3.2175% | 5.461505 | 1.5150% | PASS |
| half/all_gather/r2/i05 | 16,384 | 5.780000 | 21.1583% | 6.271667 | 8.5063% | PASS |
| half/all_gather/r2/i07 | 65,536 | 6.240000 | 2.5084% | 6.124291 | 1.8543% | PASS |
| half/all_gather/r2/i09 | 262,144 | 7.020000 | 0.8444% | 7.420415 | 5.7039% | PASS |
| half/all_gather/r2/i11 | 1,048,576 | 10.960000 | 0.9351% | 10.899199 | 0.5548% | PASS |
| half/all_gather/r2/i13 | 4,194,304 | 26.500000 | 1.8691% | 29.431424 | 11.0620% | FAIL |
| half/all_gather/r2/i15 | 16,777,216 | 54.000000 | 2.0093% | 52.079050 | 3.5573% | PASS |
| half/all_gather/r2/i17 | 67,108,864 | 156.380000 | 2.6838% | 163.668725 | 4.6609% | PASS |
| half/all_gather/r2/i19 | 268,435,456 | 546.520000 | 0.5005% | 505.923416 | 7.4282% | PASS |
| half/all_gather/r4/i01 | 1,024 | 7.160000 | 1.4303% | 6.959137 | 2.8053% | PASS |
| half/all_gather/r4/i03 | 4,096 | 7.240000 | 0.2512% | 7.333149 | 1.2866% | PASS |
| half/all_gather/r4/i05 | 16,384 | 8.360000 | 5.7573% | 8.107090 | 3.0252% | PASS |
| half/all_gather/r4/i07 | 65,536 | 8.770000 | 19.7972% | 9.188870 | 4.7762% | PASS |
| half/all_gather/r4/i09 | 262,144 | 8.970000 | 7.2874% | 9.200362 | 2.5681% | PASS |
| half/all_gather/r4/i11 | 1,048,576 | 13.540000 | 0.4256% | 11.902522 | 12.0936% | FAIL |
| half/all_gather/r4/i13 | 4,194,304 | 35.770000 | 17.5472% | 42.760018 | 19.5416% | FAIL |
| half/all_gather/r4/i15 | 16,777,216 | 64.400000 | 0.6748% | 57.595151 | 10.5665% | FAIL |
| half/all_gather/r4/i17 | 67,108,864 | 191.240000 | 4.4738% | 196.375262 | 2.6852% | PASS |
| half/all_gather/r4/i19 | 268,435,456 | 660.790000 | 0.4888% | 602.276254 | 8.8551% | PASS |
| half/all_gather/r8/i01 | 1,024 | 10.310000 | 0.3183% | 11.646645 | 12.9645% | FAIL |
| half/all_gather/r8/i03 | 4,096 | 11.460000 | 5.3136% | 9.784300 | 14.6222% | FAIL |
| half/all_gather/r8/i05 | 16,384 | 13.340000 | 3.4180% | 12.966315 | 2.8012% | PASS |
| half/all_gather/r8/i07 | 65,536 | 14.820000 | 41.8081% | 18.299815 | 23.4805% | FAIL |
| half/all_gather/r8/i09 | 262,144 | 17.710000 | 19.6411% | 13.135061 | 25.8325% | FAIL |
| half/all_gather/r8/i11 | 1,048,576 | 18.650000 | 7.1207% | 17.861324 | 4.2288% | PASS |
| half/all_gather/r8/i13 | 4,194,304 | 35.800000 | 17.0846% | 33.487778 | 6.4587% | PASS |
| half/all_gather/r8/i15 | 16,777,216 | 66.520000 | 18.9575% | 76.011324 | 14.2684% | FAIL |
| half/all_gather/r8/i17 | 67,108,864 | 207.640000 | 0.2021% | 214.197596 | 3.1582% | PASS |
| half/all_gather/r8/i19 | 268,435,456 | 722.100000 | 0.2288% | 696.762993 | 3.5088% | PASS |
| half/reduce_scatter/r2/i00 | 512 | 5.260000 | 1.2886% | 5.388715 | 2.4471% | PASS |
| half/reduce_scatter/r2/i02 | 2,048 | 5.280000 | 1.2250% | 5.201622 | 1.4844% | PASS |
| half/reduce_scatter/r2/i04 | 8,192 | 5.570000 | 2.8318% | 6.088898 | 9.3159% | PASS |
| half/reduce_scatter/r2/i06 | 32,768 | 6.070000 | 6.3819% | 5.581145 | 8.0536% | PASS |
| half/reduce_scatter/r2/i08 | 131,072 | 6.170000 | 9.6277% | 6.507616 | 5.4719% | PASS |
| half/reduce_scatter/r2/i10 | 524,288 | 8.200000 | 0.8130% | 8.031089 | 2.0599% | PASS |
| half/reduce_scatter/r2/i12 | 2,097,152 | 15.100000 | 3.2671% | 17.725157 | 17.3851% | FAIL |
| half/reduce_scatter/r2/i14 | 8,388,608 | 31.360000 | 3.8480% | 37.094219 | 18.2851% | FAIL |
| half/reduce_scatter/r2/i16 | 33,554,432 | 91.690000 | 5.0631% | 92.256072 | 0.6174% | PASS |
| half/reduce_scatter/r2/i18 | 134,217,728 | 304.820000 | 0.0497% | 312.620810 | 2.5592% | PASS |
| half/reduce_scatter/r2/i20 | 536,870,912 | 1094.320000 | 7.5249% | 1105.266355 | 1.0003% | PASS |
| half/reduce_scatter/r4/i00 | 512 | 7.130000 | 4.0247% | 6.925918 | 2.8623% | PASS |
| half/reduce_scatter/r4/i02 | 2,048 | 7.090000 | 4.7010% | 7.277577 | 2.6457% | PASS |
| half/reduce_scatter/r4/i04 | 8,192 | 7.260000 | 2.5989% | 7.356606 | 1.3307% | PASS |
| half/reduce_scatter/r4/i06 | 32,768 | 8.060000 | 6.3254% | 8.832001 | 9.5782% | PASS |
| half/reduce_scatter/r4/i08 | 131,072 | 8.040000 | 1.0430% | 8.101011 | 0.7588% | PASS |
| half/reduce_scatter/r4/i10 | 524,288 | 9.800000 | 2.2445% | 9.420295 | 3.8745% | PASS |
| half/reduce_scatter/r4/i12 | 2,097,152 | 18.380000 | 12.5375% | 20.323724 | 10.5752% | FAIL |
| half/reduce_scatter/r4/i14 | 8,388,608 | 39.250000 | 4.7134% | 49.855366 | 27.0200% | FAIL |
| half/reduce_scatter/r4/i16 | 33,554,432 | 102.190000 | 10.9893% | 103.150821 | 0.9402% | PASS |
| half/reduce_scatter/r4/i18 | 134,217,728 | 348.540000 | 0.4007% | 357.013107 | 2.4310% | PASS |
| half/reduce_scatter/r4/i20 | 536,870,912 | 1292.600000 | 0.8848% | 1269.849527 | 1.7601% | PASS |
| half/reduce_scatter/r8/i00 | 512 | 11.910000 | 8.9774% | 12.438614 | 4.4384% | PASS |
| half/reduce_scatter/r8/i02 | 2,048 | 10.410000 | 4.7389% | 10.954628 | 5.2318% | PASS |
| half/reduce_scatter/r8/i04 | 8,192 | 12.580000 | 11.3400% | 11.141741 | 11.4329% | FAIL |
| half/reduce_scatter/r8/i06 | 32,768 | 13.330000 | 8.8233% | 14.928358 | 11.9907% | FAIL |
| half/reduce_scatter/r8/i08 | 131,072 | 13.390000 | 20.6537% | 13.417999 | 0.2091% | PASS |
| half/reduce_scatter/r8/i10 | 524,288 | 12.190000 | 17.1352% | 11.287786 | 7.4013% | PASS |
| half/reduce_scatter/r8/i12 | 2,097,152 | 21.260000 | 1.8201% | 26.242376 | 23.4354% | FAIL |
| half/reduce_scatter/r8/i14 | 8,388,608 | 44.430000 | 6.3474% | 43.289841 | 2.5662% | PASS |
| half/reduce_scatter/r8/i16 | 33,554,432 | 111.150000 | 1.6079% | 123.737969 | 11.3252% | FAIL |
| half/reduce_scatter/r8/i18 | 134,217,728 | 376.840000 | 0.5761% | 381.022135 | 1.1098% | PASS |
| half/reduce_scatter/r8/i20 | 536,870,912 | 1394.160000 | 1.9680% | 1366.609405 | 1.9761% | PASS |

## D8 disposition

At 196,608 operation-buffer bytes, reduce-scatter contributes 13.186667 us
and all-gather contributes 14.808334 us per layer. Across 65 layers the model
gives 1.819675065 ms against the unchanged 1.922050 ms external arm. The
quotient is 0.946736591, so Leg B's model-form-bias hypothesis passes without
a D8-specific constant. The physical endpoint floor is 0.382293 us per phase,
and both contributions are above it.
