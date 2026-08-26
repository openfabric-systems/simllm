# CORE-54 second scored flagship expectations

Status: **EXPECTATIONS_ONLY**. This freeze precedes the second scored runner,
its fit, every second-run held-out read, every second-run result and the second
publication figure. The machine-readable authority is
`scored_run2_expectations.json`.

## Inherited disclosure and execution rulings

The first scored run's allocation, identity and missing-MTP rulings are
unchanged. Prefill and decode are separate experiments on the disclosed
12-node cluster: four eight-GPU nodes at EP32 for prefill and nine eight-GPU
nodes at EP72 for decode. The 13-node, 104-rank simultaneous rendering remains
a structural comparator only. It is not the 96-GPU disclosure. The PLACE-5
16-prefill plus 40-decode deployment remains second-legend what-if context.

The live execution remains one eight-rank prefill engine plus one eight-rank
decode engine under the serialized parent clock. Per-node observables map to
the separate disclosure experiments by the exact factors four and nine. The
CORE-58 stable identity field set is inherited without modification. It
excludes pool-local request IDs, process IDs and complete serialized result
bytes.

The offered request-rate grids are also unchanged:

| Configuration | Offered requests/s |
|---|---|
| SGLang prefill 1K | 32, 64, 128, 256 |
| SGLang prefill 2K | 16, 32, 64, 128 |
| SGLang prefill 4K | 8, 16, 32, 64 |
| SGLang standard decode | 2,000, 4,000, 8,000, 16,000, 32,000 |
| SGLang simulated MTP | 1,000, 2,000, 4,000, 8,000, 16,000 |

Every rate has an integer-picosecond interarrival. The MTP grid stays frozen
but is not executed because the exact COMP-72 cell remains absent.

## Second-run pricing declaration

Prefill uses the clean COMP-75 composition record. Each routed token dispatches
one FP8 hidden vector plus its 128-element float32 scales, combines one BF16
hidden vector, and sends at most one vector to the same destination rank after
expert selections are deduplicated there. The two-batch schedule composes the
communication service with the measured compute budget by `max`, with no
fitted overlap fraction.

The 400 Gbit/s deployment-point communication record is
2,286,179,760,360 to 2,286,179,762,680 ps. Its selected value is
2,286,179,760,360 ps. The exact per-rank token total is 16,384 at 1K, 2K and
4K, so the communication term transfers unchanged across the three disclosed
prompt lengths. Only their measured compute rows differ:
1,363,249,960,000 ps, 1,420,296,672,000 ps and 1,595,133,392,750 ps. All are
below the communication record, so the max-composed point is the same for all
three prompt lengths. The 200 Gbit/s record remains a named sensitivity arm;
it is not substituted for the fixed PLACE-5 400 Gbit/s deployment point.

Decode enables SGL-38's default-off `project_remote_kv_length` option for this
run. The scheduler-shaped batch of 32 requests at remote KV length 2,000 must
bind the exact candidate key
`05d1c33cdef9c12e25eb9159adc9dc80f1cd57b6333778f9efb5fb24cd6a74aa`
once with zero comparator misses. The selected declared 61-layer service is
28,604,120,000 ps.

## Constants and pre-fit predictions

The first freeze's tunable list and closed envelope are inherited. The
composition adds zero free or fitted tunables.

| Constant | Initial | Closed envelope | Second-run application |
|---|---:|---:|---|
| Intra-node collective surcharge | 15,064,014 ps | 0 to 30,128,029 ps | Zero applications on both CORE-59 mechanism paths. The fit is therefore flat and the inherited smaller-value tie rule selects 0 ps. |

The PCIe submission and 400/200 Gbit/s point/sensitivity declarations remain
fixed, not fitted. The communication services are backend records, not
adjustable constants.

Before any fit, the exact point formula is
`per-node tokens * 10^12 / max(measured compute ps, communication ps)` for
prefill and `256 * 10^12 / 28,604,120,000` for standard decode. The prefill
lower endpoint uses the upper integer-rounding service record. Its upper
endpoint uses the lower service record. The decode record is exact.

| Anchor | Lower | Point | Upper | Published role |
|---|---:|---:|---:|---|
| SGLang prefill 1K | 57,332.324492 | 57,332.324550 | 57,332.324550 | calibration |
| SGLang prefill 2K | 57,332.324492 | 57,332.324550 | 57,332.324550 | held out |
| SGLang prefill 4K | 57,332.324492 | 57,332.324550 | 57,332.324550 | held out |
| SGLang standard decode | 8,949.759685 | 8,949.759685 | 8,949.759685 | calibration |
| SGLang simulated MTP | BLOCKED | BLOCKED | BLOCKED | held out |

The bands propagate the selected record interval, the inherited constant
envelope and the record's distribution interval. The distribution contribution
is the COMP-74-registered zero-width placeholder marked
`insufficient-replays`; it makes no stability or zero-variance claim.

## Decode reality and registered resolution

CORE-59 identified zero evidence-backed decode mechanisms. SGL-38 binds the
measured shape but does not change its price. The candidate's measured
four-layer basis is 1.875680 ms, and the declared `61 / 4` extrapolation prices
one batch-32 decode step at 28.604120 ms. The visible EP72 calibration
throughput implies `256 / 22,282 = 11.489094` ms per node step. The declared
price is therefore expected to underpredict throughput by 59.8341 percent.

This signed calibration miss is not adjustable during the run. TRAF-66 owns
the finite-overlap prefill residual, COMP-72 owns the measured DeepSeek cells
including MTP, and CORE-61 will own the validity of the full-depth decode
extrapolation. Decode reproduction remains required before CORE-54 can close.

## Fit, one-shot score and figure

The fit reads only `sglang_prefill_1k` and `sglang_decode_standard`, writes and
content-addresses its artifact, then freezes. The held-out scorer may then read
the 2K and 4K prefill values once. Each point is scored against the 5 percent
bar, and the maximum point error decides the scoped verdict. Bands cannot turn
a point miss into a pass. The MTP row is not read or imputed; it remains
`BLOCKED` on COMP-72 with no prediction.

The publication figure keeps the ordered rightward throughput and upward
inverse-delay axes, simulated curves with deterministic bands, published
anchors and the second legend for DeepSeek H800 production plus the PLACE-5
16-prefill plus 40-decode what-if. It has no watermark. It states the scoped
held-out verdict, the MTP blocker and the disclosed decode calibration miss.

The first scored artifacts and every CORE-59, CORE-60, COMP-75 and SGL-38
record listed in the JSON preservation lock must remain byte-identical. A fatal
guard violation voids the second run rather than becoming a lost score.
