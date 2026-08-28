# CORE-61 depth-8 retry result

Status: complete. CORE-61 closes; CORE-63 is not registered.

## Signed depth residual and verdict

The signed depth residual is **-121,791,511 ps** (-3.355537 percent, measured minus predicted), with the measurement below the preregistered 3,751,359,511 ps prediction. The frozen five-percent comparison is **validated linear depth scaling**.

Validate linear depth scaling for the frozen decode family. The remaining decode-family gap then belongs to expert-parallel residency shape or decode-side overlap.

| Quantity | Value |
|---|---:|
| Preregistered prediction | 3,751,359,511 ps |
| Measured depth-8 service | 3,629,568,000 ps |
| Signed `measured - predicted` residual | **-121,791,511 ps** |
| Signed residual over measured service | **-3.355537 percent** |
| Frozen acceptance | absolute residual at or below 5 percent |
| Exact accepted interval | 3,572,723,344 to 3,948,799,485 ps |
| Verdict | **VALIDATED LINEAR DEPTH SCALING** |

The measured depth ratio against the retained four-layer 1,875,680,000 ps
basis is 1.935068. Exact doubling would be 3,751,360,000 ps, so the measured
step is 121,792,000 ps faster than that simple comparator. The scored residual
differs by 489 ps because the frozen prediction separates the per-step fixed
term instead of multiplying it.

## Physical sanity before scoring

Before reading the measurement, the rejection floor was fixed at
1,115,000,000 ps. The eight-layer graph contains the same four-layer prefix as
the retained study, so it cannot beat that prefix's published GH200
roof-derived floor. The absolute ceiling was 1,200,000,000,000,000 ps because
a successfully retained selected step had to finish inside the registered
20-minute allocation. These bounds and the distinct hypothesis band were
written to the remote control record before the measurement was read.

The 3,629,568,000 ps result is 3.255218 times the physical floor, far below the
allocation ceiling, and inside the frozen scoring interval. At batch 32 it is
8,816.476231 tokens per second for this single-rank measured step. This is a
measurement sanity value, not a new published calibration target.

## Frozen comparison

The pre-scoring harness amendment was frozen at `c7523b8b3d522f5e066f0b392e15881b9f41f5c4`. It changed startup scaffolding only. The prediction, sign convention, tolerance and exact batch-32, remote-KV-2000 boundary remained unchanged.

No depth-8 cell had scored at that commit. The amendment reduced only vLLM's
dummy startup scheduler cap from 65,536 to 4,096 tokens, then used calibrated
staggered prompts to reach one full decode batch. The retained scheduler marker
proves 32 requests, no new requests, one scheduled token per request, and all
32 cached KV lengths exactly 2,000. Prompt lengths from 1,985 through 2,000
offset scheduler age; they do not change the selected cached-KV state.

The original failures were startup allocations, not measurements. Job
`200123` requested 896 MiB for a BF16 `65,536 x 7,168` hidden-state output.
Warm retry `200128` reached a different final site and requested 3 GiB for a
BF16 `65,536 x 24,576` FlashInfer DeepGEMM output. Both inherited 65,536 from
`MAX_NUM_BATCHED_TOKENS`; `REDUCED_LAYERS=8` changes depth and does not change
that token cap. The merged result's statement that both final allocations were
the same 896 MiB is corrected by the pre-scoring supplement, while the
historical artifact remains untouched.

## Retained execution and digests

The registered sequence ran on `gh-hourly` with one task-owned job at a time.

- Base job `200137` completed in 4 minutes 54 seconds. Nsys capture succeeded;
  its older compact analyzer could not resolve positive service for the split
  batch and remained unscored. The full retained-tree manifest has SHA-256
  `cd7452645bcfc3ab3af7f4cba18aea3cafe062f7dc06385c2008b9b8f0abf30d`.
- Decode job `200138` completed in 5 minutes 16 seconds. Alignment, profile and
  analysis statuses are all zero. The exact boundary contains 36 runtime
  correlations and 236 GPU kernel records. Noncollective service is
  3,629,568,000 ps and collective service is zero.
- The decode finalize manifest has SHA-256
  `a7be6cd2ca9ddb33580f39390895d22923fe89811fb801458e84dbe89c67cfa4`.
  The post-accounting retained-tree manifest has SHA-256
  `4a236851f5b41e82a567186bb52b9b7088b9a395e16b51eba05914f74c684280`.
  The scored `measurement.json` has SHA-256
  `be5368d450a325913b9050d337e303310d03cf93674dbcbf951f45bb7176581b`.

The machine-readable result carries the exact profile, SQLite, Nsys,
alignment, harness, empty weight-file list and analysis-time log digests. The
wrapper then appended its verdict and finish record to the log, so the final
manifest, rather than the earlier analysis-time log digest, identifies the
completed log. Complete profiler payloads remain below
`${SIMLLM_CORE61_RETRY_RUN_ROOT}/decode-200138` on Merlin. The lean local pull
under `wave-runs/core61r/decode-200138` is 184 KiB and has manifest SHA-256
`f62aa1616f15d65e47ed26fe91fd0c9a9221f64ddeec98994a071df3018799d4`.
No model weight file appeared, no model weights were downloaded and no web
page was fetched.

## Residual separation

The signed value above belongs only to CORE-61 depth scaling. TRAF-66's finite compute and communication overlap was not recomputed and remains a separate ledger term. COMP-76 is unchanged.

| Owner | Residual term | Signed value | State |
|---|---|---:|---|
| CORE-61 | held-out depth scaling | -121,791,511 ps | validated within 5 percent |
| TRAF-66 | finite compute and communication overlap | not recomputed | unchanged and separate |

CORE-61 therefore closes exactly on its literal acceptance. The larger
decode-family gap is not explained by a failure of linear depth scaling at
this held-out shape; the remaining mechanism is expert-parallel residency
shape or decode-side overlap. CORE-63 remains reserved and is not registered.
COMP-72 and COMP-78 remain open on their independent Granite campaign
acceptance, and COMP-76 remains untouched.
