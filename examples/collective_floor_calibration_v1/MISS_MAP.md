# Collective floor Family H miss map

This diagnostic records where attempt 0004 missed before a replacement model
form is selected. It is derived only from the immutable 63-cell ledger in
`record.json`. It changes no membership, band, fit, runtime value, or accepted
bypass byte.

## Physical bounds before the residuals

For every cell, the physical ring floor is `(ranks - 1) / ranks` times true
bytes divided by 450 GB/s. Every measured completion is above that floor. The
source supplies no finite algorithm-progress ceiling, so the honest ceiling is
unbounded. Ten percent is larger than two H200 GPU cycles for every cell.

## The 12 misses

| Rank | Operation | Source index | True bytes | Measured us | Attempt 0004 us | Relative error |
|---:|---|---:|---:|---:|---:|---:|
| 2 | all-gather | 5 | 16,384 | 5.780000 | 7.002951 | 21.1583% |
| 4 | all-gather | 7 | 65,536 | 8.770000 | 10.506216 | 19.7972% |
| 4 | all-gather | 13 | 4,194,304 | 35.770000 | 29.493384 | 17.5472% |
| 8 | all-gather | 7 | 65,536 | 14.820000 | 21.015954 | 41.8081% |
| 8 | all-gather | 9 | 262,144 | 17.710000 | 14.231569 | 19.6411% |
| 8 | all-gather | 13 | 4,194,304 | 35.800000 | 29.683709 | 17.0846% |
| 8 | all-gather | 15 | 16,777,216 | 66.520000 | 79.130559 | 18.9575% |
| 4 | reduce-scatter | 12 | 2,097,152 | 18.380000 | 16.075604 | 12.5375% |
| 4 | reduce-scatter | 16 | 33,554,432 | 102.190000 | 90.960000 | 10.9893% |
| 8 | reduce-scatter | 4 | 8,192 | 12.580000 | 11.153426 | 11.3400% |
| 8 | reduce-scatter | 8 | 131,072 | 13.390000 | 16.155536 | 20.6537% |
| 8 | reduce-scatter | 10 | 524,288 | 12.190000 | 14.278781 | 17.1352% |

The misses span three ranks, both operations, and 8 KiB through 32 MiB. Seven
are all-gather and five are reduce-scatter. They do not form one payload
cluster. The rank-8 mid-size subset is non-monotone: all-gather falls from
15.090 us at 128 KiB to 13.400 us at 512 KiB while its held-out 256 KiB cell is
17.710 us, and reduce-scatter oscillates over the adjacent region. That subset
is consistent with a transition region, but the miss map alone does not name
the protocol or fit a correction.

## Scope of this diagnostic

This file selects no replacement, fits no value, and evaluates no new model.
The original 63 training cells and 63 holdout cells remain disjoint and
unchanged.
