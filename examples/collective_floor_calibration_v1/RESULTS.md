# Collective floor calibration result

## Chronology

Attempt 0001 is void. The first freeze pinned the external table coordinate to
bytes, while the pinned software development kit and the table itself prove
that coordinate is an element count. The run stopped before regime selection,
fitting or implementation. The fatal axis guard worked as designed.

No attempt-0001 directory exists by construction. The worker stopped before
creating any artifact, so the worker report is its only evidence.

The second freeze corrected the source axis. Attempt 0002 is superseded because
its publication made four mistakes: D8 doubled the physical endpoint byte count
instead of querying the matched operation-buffer coordinate, Family B compared
two post-wave all-remote paths instead of the pre-wave mixed-locality path,
Family H mislabeled a bare physical floor as the current ring implementation,
and the production consumer accepted transferred-at-use timing without an
explicit acknowledgement.

The third freeze, commit `6df3688`, pins the D8 coordinate mapping without
changing its `[0.90, 1.10]` band. The MiniMax arm passes
`tokens_per_rank * hidden * width` elements to the NCCL table. At expert
parallel width 8 that is 98,304 half elements, or 196,608 operation-buffer
bytes per phase. The 172,032-byte physical endpoint reading remains a useful
unscored diagnostic.

Attempt 0003 is void. Its second fresh evaluation took 657.147230 seconds,
above the unchanged 600-second Family W ceiling, while its first took
564.127896 seconds. W therefore failed and FG-6 also differed. The corrected
H, B, D8 and M findings from that attempt cannot support publication.

Attempt 0004 is the corrected publication. It uses all three freezes, the
generated pre-wave golden, the consumer transfer fence and parallel execution
of the two independent native semantic halves. Both fresh evaluations are
deterministically identical after excluding only `wall_time_seconds`.

## Outcome

What ran: attempt 0004 fit six half-precision H200 NCCL curves on the corrected
byte axis, exercised the live aggregate-floor seam and the pre-wave bypass
golden, ran all five frozen families, and repeated the complete evaluation in
a fresh process.

What came out: the run is interpretable because all seven fatal guards held.
Family H refuted the requested precision with 51 of 63 held-out cells inside
the 10 percent band. Its calibrated median error is 3.267106 percent, down from
91.616079 percent for the actual current ring path, while calibrated p95 is
19.797218 percent. Family D8 also refuted its unchanged band: the matched
196,608-byte query costs 2.131828400 ms, or 1.109143050 of the external arm.
Families B, M and W pass.

What it changes: TRAF-76 narrows from an absent aggregate collective cost to a
selectable, source-identified aggregate authority with an exact pre-wave off
path and a consumer fence on transferred timing. It remains open on the
12-cell held-out residual, the refuted D8 coordinate, and packet-mechanism work
for credits, geometry, switch behavior, arbitration and nonzero-fan-in
integration. No task closes and no milestone moves to complete.

What it does not change: TRAF-77 is untouched; the EP-32 and EP-128 mixed-width
rows remain unscored transfers with unchanged fabric service; no A100 packet
candidate enters an H200 calibrated value; and the default disabled path keeps
the pre-wave timestamps, application and wire bytes, ordering and random state
exactly.

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

[expectations_v3.md](expectations_v3.md) pins a separate coordinate mapping for
D8. The scored coordinate is the MiniMax operation buffer, not the bytes sent
by one physical endpoint. Its arithmetic is `12 * 4096 * 2 = 98,304` half
elements and `98,304 * 2 = 196,608` bytes per operation per phase. The
172,032-byte physical endpoint coordinate and the incorrect attempt-0002
344,064-byte query are not scored.

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
| FG-3, evidence classes | PASS | Exact-domain use was calibrated, while operation, dtype, rank and range transfers all downgraded with reasons. The consumer refused a transfer by default and stamped the explicitly acknowledged outcome. |
| FG-4, exact bypass | PASS | The pre-wave golden and post-wave default-off records had the same `ac952c0c0f3e9f427fb892711d716c1f93d826a86faabca70221b7b767f03f2d` digest. |
| FG-5, A100 fence | PASS | No geometry, credit, link, buffer, arbitration or A100 candidate field appears in the calibration input surface. |
| FG-6, determinism | PASS | Two fresh-process deterministic records both hashed to `0898dba617af210034deccc34c2edf18ea845897edea20d4a763af88b5df970e`. |
| FG-7, chronology | PASS | Configuration commit `fdffaec` precedes implementation commit `a983c8c`, and coordinate freeze commit `6df3688` precedes repair commit `b3913e0`. |

No guard was skipped. The bypass comparison executed 24 backend calls over two
steps and inspected a scenario containing both intra-node NVLink segments and
fabric segments. Its completion order, backend order, phase and step
timestamps, local and fabric segment tuples, application and wire bytes, and
random-generator state were identical to the pre-wave golden.

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
| Actual current ring path | 91.6161% | 99.9579% |
| Corrected aggregate calibration | 3.2671% | 19.7972% |

The before column prices the current implementation as payload divided by
world-size chunks over `world - 1` rounds for each semantic half. It is not a
bare physical floor. The median improves by 28.0420 times, exceeding the frozen
order-of-magnitude direction, but 12 individual cells still exceed 10 percent.
The largest miss
is all-gather rank 8 at 65,536 bytes: 21.015954 us modeled against 14.820000 us
measured, a 41.8081 percent error in the first regime. Curve tallies are 9/10,
8/10 and 6/10 for all-gather ranks 2, 4 and 8; and 11/11, 9/11 and 8/11 for
reduce-scatter ranks 2, 4 and 8. The complete cell ledger is in
[results.csv](results.csv).

## Family B: exact bypass

Physical sanity before the score: a disabled timing feature has a zero drift
floor and a zero drift ceiling. Any nonzero timestamp, byte, order or random
state difference is a defect.

Family B passes. The golden was generated by executing pre-wave commit
`06fc199783e364c2eaa6a7c917a1f9f2c84d79ac` over a two-step scenario with both
intra-node NVLink segments and fabric segments. The tracked golden file hashes
to `1303b6bffa6f345dad6b374e1507314fe18c9b895cc03c85f12aa16e76a2616b`.
Its canonical record and the post-wave default-off record both hash to
`ac952c0c0f3e9f427fb892711d716c1f93d826a86faabca70221b7b767f03f2d`.

The first step carries 1,572,864 application bytes, split equally into 786,432
local and 786,432 fabric bytes. The second carries 24,576 application bytes,
split equally into 12,288 local and 12,288 fabric bytes. The GOAL files carry
798,720 fabric wire send bytes across 24 backend invocations. Phase and step
timestamps, local and fabric segment tuples, application and wire byte counts,
completion order `[0, 1]`, backend invocation order and random-generator state
are byte-identical. There is no first divergent field.

## Family D8: zero-fan-in repricing

Physical sanity before the score: 172,032 local endpoint bytes over 450 GB/s
take 382.293333 ns per phase before integer rounding. Two phases over 65
layers therefore cannot beat 0.049698 ms; the accepted current path is
0.049790 ms. The source exposes no finite algorithm progress bound, so the
upper completion bound is unbounded.

Family D8 refutes the unchanged `[0.90, 1.10]` band. The two readings are:

| Coordinate interpretation | Query bytes per phase | Calibrated dispatch plus combine | Quotient to 1.922050 ms | Scored |
|---|---:|---:|---:|---|
| Physical per-endpoint bytes | 172,032 | 2.060523530 ms | 1.072044707 | No |
| Matched operation-buffer coordinate | 196,608 | 2.131828400 ms | 1.109143050 | Yes, refuted |

The matched query follows the frozen MiniMax call coordinate: 98,304 half
elements times two bytes per element. Attempt 0002 instead doubled the already
physical 172,032-byte count and queried 344,064 bytes. That mistake produced
the unearned 1.824983875 ms and 0.949498647 pass. The original current-path
quotient remains 0.025904633.

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

The production consumer refuses any `transferred-at-use` estimate by default
with `CollectiveFloorTransferError`, before publishing an outcome into the
metric chain. A deliberate run must set
`acknowledge_collective_floor_transfer=True`; every accepted transferred
outcome then stamps `transferred_at_use_acknowledged=True`. FG-3 and the unit
tests exercise both the refusal and acknowledgement paths.

## Family W: wall time

Physical sanity before the score: elapsed evaluation time has a zero-second
floor and the frozen 600-second ceiling.

Family W passes at 358.232721 seconds for the slower fresh-process evaluation,
241.767279 seconds below the ceiling. The first evaluation took 355.702865
seconds. Wall time is excluded by name from the
determinism comparison and is never added to another evidence denominator.

## Artifacts and reproduction

The portable tracked artifacts are:

- [study_config.json](study_config.json), SHA-256
  `83de9c82dea918d09e9bb474529f33b146b3a14a6f010b8d85ea1cfa83b35f57`;
- [pre_wave_bypass_golden.json](pre_wave_bypass_golden.json), generated by
  commit `06fc199783e364c2eaa6a7c917a1f9f2c84d79ac`, SHA-256
  `1303b6bffa6f345dad6b374e1507314fe18c9b895cc03c85f12aa16e76a2616b`;
- [record.json](record.json), SHA-256
  `3e41e6ec80e67eed851ca68884da0244ac8f79c338bef06272d8ccb97113026f`;
- [results.csv](results.csv), SHA-256
  `9dc7e5abfb955e6fc731e86fd51c9cc44e24d8b921506e76063f0e9ca723d3e5`.

Bulk artifacts remain append-only. Attempt 0002 is the superseded publication,
attempt 0003 is the retained void run, and attempt 0004 is the corrected
publication. There is no attempt-0001 directory by construction. Reproduce
into a new attempt directory with:

```bash
python examples/collective_floor_calibration_v1/run_study.py \
  --workdir <new-attempt-directory>
python examples/collective_floor_calibration_v1/run_study.py --check
```

The runner inserts the repository root into `sys.path`, so it needs no
`PYTHONPATH`. It records a named skip when a required native executable is
unavailable. This accepted attempt skipped nothing.
