# Collective floor calibration result

## Chronology

Attempt 0001 is void. The first freeze pinned the external table coordinate to
bytes, while the pinned software development kit and the table itself prove
that coordinate is an element count. The run stopped before regime selection,
fitting or implementation. The fatal axis guard worked as designed, and no
result from that attempt contributes here.

The second freeze corrected only the axis. The corrected configuration converts
elements to true bytes before fitting, freezes all six boundary sets from the
63 training cells, and retains every original fatal guard, scored family and
band. Attempt 0002 ran that configuration twice in fresh processes. The two
records were byte-identical after excluding only the field named
`wall_time_seconds`.

## Outcome

What ran: attempt 0002 fit six half-precision H200 NCCL curves on the corrected
byte axis, exercised the live aggregate-floor seam and exact bypass, ran all
five frozen families, and repeated the complete evaluation in a fresh process.

What came out: the run is interpretable because all seven fatal guards held,
but Family H refuted the requested precision with 51 of 63 held-out cells
inside the 10 percent band. The calibrated median error is 3.2671 percent,
down from 88.1114 percent for bare serialization, while the calibrated p95 is
19.7972 percent and therefore fails the unchanged band. Families B, D8, M and
W pass.

What it changes: TRAF-76 narrows from an absent aggregate collective cost to a
selectable, source-identified aggregate authority with an exact off path. It
remains open on the 12-cell held-out residual and on the packet-mechanism work
for credits, geometry, switch behavior, arbitration and nonzero-fan-in
integration. No task closes and no milestone moves to complete.

What it does not change: TRAF-77 is untouched; the EP-32 and EP-128 mixed-width
rows remain unscored transfers with unchanged fabric service; no A100 packet
candidate enters an H200 calibrated value; and the default disabled path keeps
the accepted timestamps, bytes, ordering and random state exactly.

## Corrected axis and source identity

The source coordinate is `ELEMENTS`. The fitted physical coordinate is
`BYTES`, with `true_bytes = source_elements * dtype_width_bytes`. The three
pinned SDK observations are recorded in
[study_config.json](study_config.json):

- `aiconfigurator_core/sdk/operations/communication.py:516` passes tokens times
  elements per token directly to the NCCL query;
- `communication.py:394-395` multiplies that same count by the dtype width on
  the analytical path;
- `aiconfigurator_core/sdk/common.py:1112-1113` assigns width 2 to half and
  width 1 to int8.

At an equal physical size of 512 bytes, the half query resolves source element
256 at 0.00517 ms, while the int8 query resolves source element 512 at
0.00519 ms. They are distinct measured cells. The probe's section-4 half
intercepts remain zero-size floors, but its slopes and bandwidths were
element-axis artifacts and are not used here.

The source is NVIDIA AIConfigurator 0.11.0, `h200_sxm`, NCCL database 2.26.2,
row version 2.29.2, artifact
`e432db694195110aa39c1e1eccf1accda012e69ef68e95210d049809bb93f015`,
with the first source row winning duplicate coordinates. Fitted estimates are
`calibrated`; operation, dtype, rank or range transfers are
`transferred-at-use`, never `MEASURED` or `MEASURED-EXTERNAL`.

## Frozen regimes on true bytes

Every line is `T_ps = floor_ps + true_bytes * slope_ps_per_byte`. Boundaries
are inclusive starts of following regimes and came only from training cells.
The bandwidth column is the slope's derived effective byte rate, not a new
fitted term.

| Operation | Ranks | Regime true-byte range | Floor, us | Slope, ps/B | Effective GB/s |
|---|---:|---:|---:|---:|---:|
| all-gather | 2 | 512 to 32,767 | 5.069818 | 117.989003548 | 8.475366 |
| all-gather | 2 | 32,768 to 8,388,607 | 5.751542 | 5.064907134 | 197.436986 |
| all-gather | 2 | 8,388,608 to 536,870,912 | 19.825641 | 1.972279630 | 507.027495 |
| all-gather | 4 | 512 to 131,071 | 7.002849 | 53.457134203 | 18.706577 |
| all-gather | 4 | 131,072 to 8,388,607 | 8.299037 | 5.053125983 | 197.897302 |
| all-gather | 4 | 8,388,608 to 536,870,912 | 24.392450 | 2.358732889 | 423.956441 |
| all-gather | 8 | 512 to 131,071 | 10.173404 | 165.444182711 | 6.044335 |
| all-gather | 8 | 131,072 to 33,554,431 | 13.201426 | 3.929682544 | 254.473482 |
| all-gather | 8 | 33,554,432 to 536,870,912 | 37.263676 | 2.545058736 | 392.918240 |
| reduce-scatter | 2 | 512 to 262,143 | 5.322149 | 11.000653469 | 90.903691 |
| reduce-scatter | 2 | 262,144 to 4,194,303 | 6.153333 | 4.030863444 | 248.085804 |
| reduce-scatter | 2 | 4,194,304 to 536,870,912 | 14.406432 | 2.164878056 | 461.919782 |
| reduce-scatter | 4 | 512 to 4,194,303 | 7.414845 | 4.129771436 | 242.144151 |
| reduce-scatter | 4 | 4,194,304 to 67,108,863 | 24.480000 | 1.981258392 | 504.729723 |
| reduce-scatter | 4 | 67,108,864 to 536,870,912 | 31.903333 | 2.369532983 | 422.024090 |
| reduce-scatter | 8 | 512 to 262,143 | 10.819952 | 40.707267445 | 24.565638 |
| reduce-scatter | 8 | 262,144 to 16,777,215 | 12.080689 | 4.192525715 | 238.519706 |
| reduce-scatter | 8 | 16,777,216 to 536,870,912 | 25.693230 | 2.600072605 | 384.604645 |

The exact rational floors, slopes, effective bandwidths and training-cell
lists are in [record.json](record.json).

## Fatal guards

| Guard | Outcome | What the generated run checked |
|---|---|---|
| FG-1, no invented terms | PASS | The authority contains only source identity, frozen boundaries, fitted floors, fitted byte slopes and derived bandwidth; the equal-byte axis guard also passed. |
| FG-2, no double counting | PASS | Constructed semantic-base, registration, host-launch and second-NVLink-rate selections were rejected; the active projection carried zero duplicate charge. |
| FG-3, evidence classes | PASS | Exact-domain use was calibrated, while operation, dtype, rank and range transfers all downgraded with reasons. |
| FG-4, exact bypass | PASS | The generated feature-absent and explicit-off records had the same `ef294263694dfc1a0475f32fd129416bde7933e4cc86cc6d0083959b8be9e0ab` digest. |
| FG-5, A100 fence | PASS | No geometry, credit, link, buffer, arbitration or A100 candidate field appears in the calibration input surface. |
| FG-6, determinism | PASS | Two fresh-process deterministic records both hashed to `d85e56ee7de9a3622c38d6c1ce0a3789375308e7121afebc40d1092719e06ee9`. |
| FG-7, chronology | PASS | Configuration commit `fdffaec` precedes implementation commit `a983c8c`; the fit module is absent before the configuration and present at implementation. |

No guard was skipped. The bypass comparison executed four backend calls over
two steps, inspected four GOAL and four completion artifacts, and conserved
532,480 application send bytes. Its completion order, backend order, segment
tuples and random-generator state were identical.

## Family H: held-out reproduction

Physical sanity before the score: each held-out completion has a lower bound
of true bytes divided by 450 GB/s. All 63 measurements are above that floor.
The available source has no algorithm progress guarantee, so no finite
first-principles upper completion bound can be derived; the honest ceiling is
unbounded. The sourced NVIDIA DGX H200 core clock is 1.98 GHz, making two GPU
cycles `100000/99` ps, or 1,011 ps after ceiling. Ten percent dominates that
cycle term for every held-out cell. The clock is informational and comes from
the NVIDIA NIM hardware specification linked in `study_config.json`.

Family H is refuted: 51 of 63 cells pass, not 63 of 63.

| Error view over the same 63 cells | Median | Nearest-rank p95 |
|---|---:|---:|
| Bare 450 GB/s serialization | 88.1114% | 99.9779% |
| Corrected aggregate calibration | 3.2671% | 19.7972% |

The median improves by 26.9692 times, exceeding the frozen order-of-magnitude
direction, but 12 individual cells still exceed 10 percent. The largest miss
is all-gather rank 8 at 65,536 bytes: 21.015954 us modeled against 14.820000 us
measured, a 41.8081 percent error in the first regime. Curve tallies are 9/10,
8/10 and 6/10 for all-gather ranks 2, 4 and 8; and 11/11, 9/11 and 8/11 for
reduce-scatter ranks 2, 4 and 8. The complete cell ledger is in
[results.csv](results.csv).

## Family B: exact bypass

Physical sanity before the score: a disabled timing feature has a zero drift
floor and a zero drift ceiling. Any nonzero timestamp, byte, order or random
state difference is a defect.

Family B passes. The feature-absent and explicit-off records are byte-identical
across every frozen field and generated artifact. Both calibrated-outcome
lists remain empty.

## Family D8: zero-fan-in repricing

Physical sanity before the score: 172,032 local endpoint bytes over 450 GB/s
take 382.293333 ns per phase before integer rounding. Two phases over 65
layers therefore cannot beat 0.049698 ms; the accepted current path is
0.049790 ms. The source exposes no finite algorithm progress bound, so the
upper completion bound is unbounded.

Family D8 passes. The calibrated aggregate path prices 1.824983875 ms against
the fixed 1.92205 ms external arm, a quotient of 0.949498647 inside
[0.90, 1.10]. The corrected query uses 172,032 source elements, hence 344,064
true half bytes. The old quotient remains 0.025904633.

The unscored mixed-width publications keep their native fabric service
unchanged. EP 32 moves from 6.997536 to 8.64087354 ms, with 53.8272 us fabric
service in each phase. EP 128 moves from 29.519776 to 31.16311354 ms, with
227.0752 us fabric service in each phase. Both use an explicit rank-8 donor
and are labeled `transferred-at-use`; neither is an end-to-end H200
calibration or a TRAF-77 result.

## Family M: signature metric chain

Physical sanity before the score: the on arm cannot complete below the off arm
because every bare local serialization term is replaced by a positive fitted
completion floor plus serialization. The source supplies no finite software
progress ceiling, so the upper bound is unbounded.

Family M passes through `StepRecord`, `HtsimStepSink`, `StepResult` and
`HtsimRequestMetricReducer`. The explicit off arm reproduces the
feature-absent path exactly.

| Metric | Calibration off | Calibration on | Delta |
|---|---:|---:|---:|
| Time to first token (TTFT) | 5,914,368,000 ps | 8,548,601,088 ps | 2,634,233,088 ps |
| Time per output token (TPOT) | 2,409,008,000 ps | 3,140,441,664 ps | 731,433,664 ps |

The first-step floor-plus-serialization replacement is exactly 2,634,233,088
ps, equal to the TTFT delta. Each of the two decode steps adds exactly
731,433,664 ps, so their mean is the observed TPOT delta. The projection
carries zero semantic base, zero registration and zero host launch floor.

## Family W: wall time

Physical sanity before the score: elapsed evaluation time has a zero-second
floor and the frozen 600-second ceiling.

Family W passes at 592.484152 seconds for the slower fresh-process evaluation,
7.515848 seconds below the ceiling. Wall time is excluded by name from the
determinism comparison and is never added to another evidence denominator.

## Artifacts and reproduction

The portable tracked artifacts are:

- [study_config.json](study_config.json), SHA-256
  `83de9c82dea918d09e9bb474529f33b146b3a14a6f010b8d85ea1cfa83b35f57`;
- [record.json](record.json), SHA-256
  `9a65bfb520d6cedac8710b36b556ae46277c6cc2a62a5a93c60646fb7dfed28b`;
- [results.csv](results.csv), SHA-256
  `bbc96f433eb107d501a986551bf8c4a3573913c96792a6a5bdc045b20e4e521e`.

Run artifacts remain under append-only `attempt-0002`; attempt 0001 remains
the recorded void. Reproduce into a new attempt directory with:

```bash
python examples/collective_floor_calibration_v1/run_study.py \
  --workdir <new-attempt-directory>
python examples/collective_floor_calibration_v1/run_study.py --check
```

The runner inserts the repository root into `sys.path`, so it needs no
`PYTHONPATH`. It records a named skip when a required native executable is
unavailable. This accepted attempt skipped nothing.
