# External database parity result

The imported operation database is bit-identical to the pinned external Python
resolver at the frozen seam. The first comparison run is nonvoid, all four
scored families pass, all eight fatal guards hold, and no unit in the last
place (ULP) finding exists.

## What ran

`external_db_parity_v1` converted the pinned H200 SXM TensorRT-LLM 1.3.0rc10
slice, loaded the converted artifact without the external SDK, and compared 26
resolver queries plus four Qwen3-32B-FP8 pass compositions against live SDK
calls. Both sides ran twice in fresh processes. Bulk evidence is retained at
`${SIMLLM_P3X_T1_BULK_ROOT}/attempt-0001`; the portable compact evidence is in
[record.json](record.json) and [results.csv](results.csv).

## What came out

The deciding result is zero ULP difference across all 26 resolver queries and
all four pass totals. The scored families remain separate:

| Family | Result | Meaning |
|---|---:|---|
| I1 | 25 / 25 | Seven frozen raw binary64 rows, all 17 table counts and the 284,717-row total match. |
| I2 | 26 / 26 | Every audited interpolation, clamp, smoothing, hold and rekey rule is bit-equal to the live SDK. |
| P1 | 4 / 4 | Both prefill and both decode pass totals match the live SDK and the frozen literals. |
| W | 1 / 1 | Conversion plus both repeated local and live evaluations took 78.428689 seconds, below 120 seconds. |

FG-1 through FG-8 all passed. In particular, the frozen BF16 GEMM mutation at
`m=32768, n=64, k=512` serves
`0x1.02253ae9a795bp-7` after loading instead of its raw
`0x1.eb4af55555555p-8`, on both sides. Three GEMM cells and 367 generation
attention cells are raised to their analytical speed-of-light floor. Every
served local value carries `MEASURED-EXTERNAL` and the complete frozen source
identity. The exact, composite and gap mapping rejects both undeclared families
and composite plus constituent double charging.

## Physical sanity

Before reading the prefill result, the tensor-core floor is about 28.3 ms:
`2 * 32 billion parameters * 3,500 tokens / TP4`, divided by the declared
1.978 PFLOP/s FP8 peak. The measured 99.282611 ms is 3.51 times that floor,
equivalent to about 28.5 percent peak use after kernel and communication work.

Before reading the decode result, one TP4 share of roughly 32 GB of FP8 weights
is about 8 GB. Streaming it once from 4.8 TB/s HBM has a floor near 1.67 ms.
The measured 11.102129 ms is 6.65 times that floor, so it does not imply
impossible memory bandwidth. Halving batch while doubling context changes the
decode step only from 11.102129 ms to 11.207792 ms, consistent with the fixed
weight stream remaining large while KV-cache attention grows.

At the pass level, the batch-64 decode is about 90.07 generated tokens per
second per request and about 5,764 aggregate tokens per second. The 3,500-token
prefill processes about 35,253 prompt tokens per second. These are plausibility
checks only. They rule out gross unit mistakes; they do not validate a serving
runtime or an end-to-end deployment.

## Artifact and licensing evidence

The artifact directory identity is
`85e72f990f00ea457de522d0b773e678f5e067740689912df5646f6296273284`.
Its 2,119,044-byte XZ payload is content-addressed by SHA-256
`0f606718c5e413e898d9ad33a3d7e803e532d5d5bd8a1c29929e7f8b2458e8ef`.
The artifact preserves the Apache 2.0 license, NVIDIA SPDX and copyright
notice, and the SimLLM modified-file statement. Their SHA-256 values are:

| File | SHA-256 |
|---|---|
| `LICENSE` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| `THIRD_PARTY_NOTICE` | `3abf9540ea9cebf33fae0a796c52097a0b591cb1cccf0718705d0dd8a4f3a4bb` |
| `MODIFIED` | `0f93928e72da11eae92a95d3396884b3edb1b75ab58cbb329eee4f3be0dd3133` |

## What it changes for the project

P3X-T1 establishes a literal matched pricing seam for the frozen external
Python surface. The compute module can consume this artifact without PyArrow,
PyYAML or the external SDK, retain the source identity on every result, and
distinguish exact mappings from declared composites and gaps. COMP-82 through
COMP-86 register the five residual surfaces: additional identities, HYBRID
inheritance, the compiled estimator, power fields and live TTFT/TPOT
composition.

The expectations commit is `44959ef`, the 26-query freeze is `afe7ee6`, and
the load-mutation guard freeze is `f7ec05a`. All precede the importer, runner
and first comparison. The first comparison used runner commit `df5344a`.

## What it does not change

This result does not import another system, backend or database version
(COMP-82), enable shared-layer HYBRID inheritance (COMP-83), claim parity for
the compiled estimator (COMP-84), invent absent power observations (COMP-85),
or place the external pass on the supported `ExecutionGraph` to
`CompletionEvent` to `StepResult` to TTFT/TPOT chain (COMP-86). It does not
reproduce external Pareto rows or compare serving mechanisms. No existing
compute calibration task closes.
