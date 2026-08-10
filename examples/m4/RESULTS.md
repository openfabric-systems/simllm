# M4 closed-loop slice: results against pre-registered expectations

Runs of 2026-08-04. Backend binary: `htsim_rnic` from a machine-local
HTSIM-rnic-private build pinned at `c03e1f2`, stacked on the merged
2026-08-03 simllm-addon rounds; the resolved historical build path is
intentionally omitted. Current reproductions select the build root with
`SIMLLM_HTSIM_BUILD`. The `rnic-nn` / `rnic-nn-fluid` profiles exercised
here reproduce the
M1-calibrated constants exactly, which checks A and C bound empirically.
GOAL conversion with the prebuilt `txt2bin`. All sections ran in one
`run_m4.py` invocation (36 rows in `summary.csv`, with raw GOALs and
completion CSVs beside it). Those artifacts remain outside Git in the
machine-local directory used for the historical run; its resolved historical path is
intentionally omitted. New runs default to `${SIMLLM_DATA_ROOT}/m4`.
The backends are deterministic, so every number reproduces exactly.
Checks A1-A6, B1-B6 (plus B-q1/B-q2), C1-C6, D1-D2 and E1-E3 are the
registered predictions of [expectations.md](expectations.md) (kept frozen;
deviations would be disclosed here, never edited there).

Verdict: **every registered check passes. All 6 A cells, all 6 B cells,
both D metrics and all 6 C cells reproduce their closed forms with 0 ps
residual; the C band and floor hold with the point form exact; the E
properties hold on all 17 replayed steps.** The live tp=8 stretch run
(below) also closed the loop against a real vLLM with 0 ps residual per
step.

Audit and rerun note (2026-08-04): an independent mathematician audit
rederived every closed form digit-exact and re-ran four registered points
fresh, and together with the house audit found three defects in this
report's first draft, all fixed here: (1) the first draft of `run_m4.py`
derived the B and C expected values from the runtime cost model instead of
the frozen registered constants, making the 0 ps residual self-referential;
the harness now compares against the frozen `FROZEN_B_INPUTS` table and
additionally checks the runtime estimate and calc against the frozen
values per cell (`estimate_matches_frozen`, `calc_matches_frozen`, all
true), and the full grid was rerun with identical measurements. (2) The
live-run TPOT had averaged only 6 of the 7 decode deltas (a zip
off-by-one); the script is fixed and the live run rerun, corrected number
below. (3) This verdict paragraph had miscounted the C grid as 12 cells.
Also disclosed per the audit: the registered E1 second clause
(`completed_at_ps = virtual_time_ps + step_latency_ps`) and the equivalent
D replay assertion are true by construction of the sink and cannot fail;
they are kept because they were registered, but they carry no evidential
weight.

## Check A: standalone ring allreduce on rnic-nn-fluid

Registered: JCT = 2(W-1) * (chunk * 20 + 2,000,000) ps, exact.

| Check | S bytes | W | predicted ps | measured ps | residual |
|---|---|---|---|---|---|
| A1 | 65,536 | 2 | 5,310,720 | 5,310,720 | 0 |
| A2 | 65,536 | 4 | 13,966,080 | 13,966,080 | 0 |
| A3 | 65,536 | 8 | 30,293,760 | 30,293,760 | 0 |
| A4 | 16,777,216 | 2 | 339,544,320 | 339,544,320 | 0 |
| A5 | 16,777,216 | 4 | 515,316,480 | 515,316,480 | 0 |
| A6 | 16,777,216 | 8 | 615,202,560 | 615,202,560 | 0 |

All pass. This validates the fluid ring-allreduce closed form end to end
and closes the ring-allreduce third of TRAF-4 (pairwise all-to-allv and
binomial broadcast remain).

## Check B: full step makespan on rnic-nn-fluid

Registered: makespan = L * c * 1000 + 2L * 2(W-1) * (chunk * 20 + P),
exact, with the frozen roofline inputs E and c.

| Check | shape | TP | predicted ps | measured ps | residual |
|---|---|---|---|---|---|
| B1 | decode8x2048 | 2 | 1,965,550,080 | 1,965,550,080 | 0 |
| B2 | decode8x2048 | 4 | 1,800,453,120 | 1,800,453,120 | 0 |
| B3 | decode8x2048 | 8 | 2,485,904,640 | 2,485,904,640 | 0 |
| B4 | prefill2048 | 2 | 33,512,148,480 | 33,512,148,480 | 0 |
| B5 | prefill2048 | 4 | 38,871,326,720 | 38,871,326,720 | 0 |
| B6 | prefill2048 | 8 | 42,318,915,840 | 42,318,915,840 | 0 |

The measured whole-step roofline estimates and per-layer calc units also
matched the frozen inputs exactly (e.g. B1: E = 1,625,667,291 ps,
c = 50,802 ns). All pass.

- B-q1 (decode non-monotonic in TP): 1.966 / 1.800 / 2.486 ms for
  TP = 2/4/8, i.e. TP=4 < TP=2 < TP=8 as registered. Pass.
- B-q2 (prefill monotone increasing in TP): 33.51 / 38.87 / 42.32 ms.
  Pass.

## Check C: the same grid on rnic-nn

Registered: point form (Npkt + 1) * 83,200 + P per round, band at most one
slot per round in either direction, and makespan_nn >= makespan_fluid.

| Check | shape | TP | point ps | measured ps | deviation | band | >= fluid |
|---|---|---|---|---|---|---|---|
| C1 | decode8x2048 | 2 | 1,977,510,400 | 1,977,510,400 | 0 | 10,649,600 | yes |
| C2 | decode8x2048 | 4 | 1,834,368,000 | 1,834,368,000 | 0 | 31,948,800 | yes |
| C3 | decode8x2048 | 8 | 2,562,745,600 | 2,562,745,600 | 0 | 74,547,200 | yes |
| C4 | prefill2048 | 2 | 33,858,342,400 | 33,858,342,400 | 0 | 10,649,600 | yes |
| C5 | prefill2048 | 4 | 39,406,592,000 | 39,406,592,000 | 0 | 31,948,800 | yes |
| C6 | prefill2048 | 8 | 42,980,665,600 | 42,980,665,600 | 0 | 74,547,200 | yes |

All pass, and the point form turned out exact (0 ps deviation in every
cell, band unused). The registered caution came from the M1 C7
final-packet pipelining seen under *contending* flows; a ring round is W
disjoint full-rate flows with strict `requires` chaining, so no pipelining
across rounds is possible and the store-and-forward slot of the M1 C3
single-flow intercept applies per round unchanged. The band was
over-cautious, not wrong; noted, not a finding.

## Check D: TTFT and TPOT of a replayed step sequence

Registered: TTFT = B6, every decode latency = B3, exact.

- D1: TTFT = 42,318,915,840 ps = B6 exactly. Pass.
- D2: all 8 decode-step latencies identical (one distinct value),
  TPOT = 2,485,904,640 ps = B3 exactly. Pass.

Final virtual time of the 9-step replay: 62,206,152,960 ps
(= B6 + 8 * B3, consistent).

## Check E: replay of the recorded adapter smoke JSONLs

Both files loaded through `step_records_from_jsonl` (8 vLLM records, 9
SGLang records, schema-validated), replayed at declared tp=8 on
rnic-nn-fluid.

- E1 (monotonic virtual time, completed_at = virtual_time + latency):
  holds for all 17 records. Pass.
- E2 (every step simulated, latency strictly above the compute-only
  estimate): holds for all 17 steps; e.g. vLLM step 0: estimate
  499,232,182 ps vs simulated 2,511,432,960 ps. Pass.
- E3 (network share, report only): vLLM: 0.8012 (step 0, 12 tokens),
  0.7872 (steps 1-7, 3 tokens). SGLang: 0.7856 (step 0, 2 tokens), 0.7983
  (step 1, 10 tokens), 0.7872 (steps 2-8, 3 tokens). At these tiny smoke
  batches the TP=8 allreduce chain is propagation-dominated
  (64 * 14 * ~2 us ~ 1.8 ms per step), hence the ~79 percent share; this
  is the declared llama-8b-on-tp8 what-if, not a statement about the
  1B-parameter smoke model itself.

Unregistered observation: the replayed step latencies also match the
check-B closed form evaluated per record to 0 ps (e.g. the 3-token decode
steps: c = 15,601 or 15,602 ns as the growing context nudges the roofline
estimate across a 32,000 ps flooring boundary, giving 2,346,282,240 vs
2,346,314,240 ps; the 32,000 ps split is exactly L * 1 ns). The form
generalizes beyond the two registered step shapes.

## Stretch: live vLLM tp=8 closed loop

vLLM v0.26.0 in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`,
granite-3.0-1b-a400m-instruct, `num_gpu_blocks_override=2048`,
`tensor_parallel_size=8`) **accepted tp=8 under `SimExecutor` on a 1-GPU
box**: the executor fabricates 8 workers and touches no device, and no
config validation objected (the engine served all 8 workers' init RPCs and
pinned the KV pool per worker). With
`configure(step_sink=HtsimStepSink(...))` on rnic-nn-fluid, the packet
simulator sat inside the live step loop: 8 steps, 24 generated tokens,
and every step latency equals the check-B closed form with **0 ps
residual**. The live schedule reproduced the E-section vLLM replay
row-for-row (same 12-token first step, same 3-token decode steps, same
latencies), a direct replay-fidelity cross-check.

Sim-native metrics with network time included: TTFT = 2,511,432,960 ps,
TPOT = 2,346,300,526 ps (mean over all 7 decode deltas; the first draft
reported 2,346,303,573 ps by dropping the first delta, caught in audit and
rerun with the fixed script), final virtual time 18,935,536,640 ps.
Wall-clock cost was ~8 s per step, the documented per-step-subprocess
diagnostic-mode overhead (BRIDGE-1 tracks the persistent co-simulator that
removes it). The live artifacts remain outside Git in the machine-local
directory used for the historical run; its resolved historical path is intentionally
omitted. New live runs default to
`${SIMLLM_DATA_ROOT}/m4/live-vllm-tp8`.
