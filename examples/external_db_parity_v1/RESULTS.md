# External database parity result

The imported operation database remains bit-identical to the pinned external
Python resolver at the frozen seam after the adversarial review repairs. The
repaired comparison is nonvoid: all four scored families pass, all eight fatal
guards hold, and the combined interpolation and pass-composition ledger has no
unit in the last place (ULP) finding.

## What ran

`external_db_parity_v1` converted the pinned H200 SXM TensorRT-LLM 1.3.0rc10
slice, loaded the converted artifact without the external software development
kit (SDK), independently decompressed and recounted the payload, and compared
37 resolver queries plus four Qwen3-32B-FP8 pass compositions against live SDK
calls. Both sides ran twice in fresh processes. Bulk evidence is retained at
`${SIMLLM_P3X_T1_BULK_ROOT}/attempt-0002`; the portable compact evidence is in
[record.json](record.json) and [results.csv](results.csv).

## What came out

The deciding result is zero ULP difference across all 37 resolver queries and
all four pass totals. No per-term pass-composition difference was found. The
scored families remain separate:

| Family | Result | Meaning |
|---|---:|---|
| I1 | 25 / 25 | Seven frozen raw binary64 rows, 17 independently recounted table sizes and the independently recounted 284,717-row total match their frozen oracles. |
| I2 | 37 / 37 | All 26 original points and 11 review-addendum points are bit-equal to the live SDK. |
| P1 | 4 / 4 | Both prefill and both decode pass totals match the live SDK and frozen literals; their local and live terms also match. |
| W | 1 / 1 | Conversion plus both repeated local and live evaluations took 86.032942 seconds, below 120 seconds. |

FG-1 through FG-8 all passed. The manifest's row inventory equals the payload
recount, but it is a voiding cross-check rather than the source of the scored
counts. The distance-cap diagnostic changed GEMM query `I2-27` from the capped
`0x1.d16e300d9dc77p+3` to the cap-off
`0x1.cc9259aaacb10p+3`, a difference of 85,475,858,518,375 ULP. The cap-off
value is not scored.

The study has discriminating coverage for exactly these rules: the 2.0 GEMM
site-distance cap, the GEMM and generation-attention load-time
speed-of-light clamps, generation-attention `attn_dtype` invariance, standard
mixture-of-experts non-low-latency `kernel_source` invariance, and GDN
`model_name` and `num_tokens` invariance. The other I2 rows are local-versus-
live parity points, not rule-removal discrimination controls.

The frozen BF16 GEMM cell at `m=32768, n=64, k=512` serves
`0x1.02253ae9a795bp-7` after loading instead of its raw
`0x1.eb4af55555555p-8`. The frozen generation-attention cell serves
`0x1.d0d73a2abadb5p-4` instead of its raw
`0x1.b78732aaaaaabp-4`. Three GEMM cells and 367 generation-attention cells are
raised to their analytical speed-of-light floor. Every served local value
carries `MEASURED-EXTERNAL` and the complete frozen source identity. The exact,
composite and gap mapping rejects both undeclared families and composite plus
constituent double charging.

## What the review changed

The three adversarial lenses preserved the numerical result and exposed gaps
in what the study actually proved. The repaired evidence addresses them as
follows:

- A-1 recounts the compressed payload directly and voids on any manifest
  disagreement. A-2 publishes any local-versus-live P1 term mismatch even if
  the total agrees. A-3 compares the working bytes of every freeze file with
  its recorded Git blob. A-4 retains live-worker stderr in a compact void
  record before failing. A-6 regenerates the local scored hex values in the
  SDK-free continuous-integration test and compares them with the tracked
  record.
- B-2 adds a cap-discriminating GEMM point and an unscored cap-off value. B-3
  scores exact hits on both clamped families and extends FG-5 to the second
  family. B-4 adds four ignored-dimension pairs. B-9 combines I2 and P1 in the
  top-level ULP findings. B-5 is registered as COMP-87 because WideEP MoE, MLA
  BMM, Mamba2, MLA and generation-MLA dispatch remain outside this resolver.
- C-1 preserves the NVIDIA 2025-2026 source-file notice as well as the 2026
  collection notice. C-2 adds file-local conversion notices to both converted
  JSON files and enumerates every artifact derivation in `MODIFIED`. C-3 uses
  exact hashes and exact notice lines with truncated-license and wrong-year
  negative controls. C-5 records the complete byte-producing JSON and XZ
  recipe. C-6 binds and verifies all three converted JSON files. C-7 proves a
  rehashed donor-version row is rejected. C-8 requires the evidence class at
  every constructor call and rejects explicit `MEASURED`. C-11 pins every
  study implementation and test surface to LF line endings.

## Physical sanity

Before reading the prefill result, the tensor-core floor is about 28.3 ms:
`2 * 32 billion parameters * 3,500 tokens / TP4`, divided by the declared
1.978 PFLOP/s FP8 peak. The measured 99.282611 ms is 3.51 times that floor,
equivalent to about 28.5 percent peak use after kernel and communication work.

Before reading the decode result, one TP4 share of roughly 32 GB of FP8 weights
is about 8 GB. Streaming it once from 4.8 TB/s high-bandwidth memory has a floor
near 1.67 ms. The measured 11.102129 ms is 6.65 times that floor, so it does not
imply impossible memory bandwidth. Halving batch while doubling context
changes the decode step only from 11.102129 ms to 11.207792 ms, consistent with
the fixed weight stream remaining large while key-value-cache attention grows.

At the pass level, the batch-64 decode is about 90.07 generated tokens per
second per request and about 5,764 aggregate tokens per second. The 3,500-token
prefill processes about 35,253 prompt tokens per second. These are plausibility
checks only. They rule out gross unit mistakes; they do not validate a serving
runtime or an end-to-end deployment.

## Artifact and licensing evidence

The artifact directory identity is
`85e72f990f00ea457de522d0b773e678f5e067740689912df5646f6296273284`.
Its 2,119,044-byte XZ payload is still content-addressed by SHA-256
`0f606718c5e413e898d9ad33a3d7e803e532d5d5bd8a1c29929e7f8b2458e8ef`.
The review did not change one payload byte. The loader verifies that payload
and the converted `system.json`, `model-config.json` and
`family-mapping.json` hashes before use.

The exact Apache 2.0 license hash and both exact NVIDIA copyright lines passed.
The repository notice now carries 2025-2026, both converted JSON files name
their source and point to `MODIFIED`, and the negative licensing controls
failed closed. The artifact licensing hashes are:

| File | SHA-256 |
|---|---|
| `LICENSE` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| `THIRD_PARTY_NOTICE` | `8883f78c9d753e647963f468145231e9c14e42e8642fda36b0b5c9b6f89e9c57` |
| `MODIFIED` | `bfe7e227252dc60f86bd78f9427ac9930fd2f25255308b05f1c8de695e826446` |

## Freeze chronology

The original expectations commit `44959ef`, original 26-query freeze
`afe7ee6` and load-mutation guard freeze `f7ec05a` still precede the importer
and first comparison. The immutable `expectations.md` bytes retain SHA-256
`efecb44a4cc51a0c4f9c4dec74fe6aeedc1151d7ddfede560bc145b8554b4253`.

After review and before the repair implementation or second comparison, commit
`25dc6b5` froze the independent table-count inventory, the cap-discriminating
point, both clamp hits, all four invariance pairs and the second FG-5 cell in
`freeze_addendum.md` and `query_points.json`. Repair commit `bba456c` then ran
append-only attempt 0002. FG-8 verified both ancestry and current working bytes
against the recorded Git blobs.

## What it changes for the project

P3X-T1 remains a literal matched pricing seam for the frozen external Python
surface, and its scored coverage and guard claims now match the evidence that
produces them. The compute module can consume this artifact without PyArrow,
PyYAML or the external SDK, independently verify its payload and converted JSON
files, retain source identity on every result, and distinguish exact mappings
from declared composites and gaps. COMP-87 newly owns the imported operation
families that still have no dispatched resolver path. No existing compute
calibration task closes and no milestone advances beyond the already matched
offline seam.

## What it does not change

This result does not import another system, backend or database version
(COMP-82), enable shared-layer HYBRID inheritance (COMP-83), claim parity for
the compiled estimator (COMP-84), invent absent power observations (COMP-85),
or place the external pass on the supported `ExecutionGraph` to
`CompletionEvent` to `StepResult` to time to first token (TTFT) and time per
output token (TPOT) chain (COMP-86). It does not dispatch WideEP MoE, MLA BMM,
Mamba2, MLA or generation-MLA operations (COMP-87), reproduce external Pareto
rows, or compare serving mechanisms. The payload, original raw-row oracles,
four pass totals and their zero-ULP numerical result are unchanged.
