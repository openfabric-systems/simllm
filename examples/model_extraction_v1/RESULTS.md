# Offline model extraction v1 results

## Outcome

What ran: the frozen `model_extraction_v1` study drove the pinned vLLM and
SGLang CPU-only configuration seams twice each over all 15 Granite suite cases,
then scored their family projections, launch counts, shape sweeps, graph
templates and canonical bytes.

What came out: the run is nonvoid and produced **two complete framework
inventories**, one per pinned framework. Every case carries 97 logical family
invocations, all four behavioral relation families passed, the independent
byte-determinism oracle passed for both repeat pairs, and no fatal guard was
violated.

What it changes: the Granite model column now has published offline workload
denominators for vLLM and SGLang, so that first COMP-54 column is literal and
unblocks its later target-silicon joins. COMP-54 remains open for the planned
dense Llama-class and larger routed-MoE columns. COMP-6, VLLM-12 and SGL-10
remain open and continue to own physical implementation identity and observed
launch capture.

What it does not change: no GPU, cluster, physical launch, code object,
Accel-Sim, device timing, time to first token (TTFT), time per output token
(TPOT) or calibration matrix cell was exercised. The transformer-dag-v1 suite
bytes remain identical. This result does not validate or close any physical
capture or device-model task.

## Chronology and frozen input

The isolated path-scanner repair landed first at
`7196435081841c77c6271f41c16923f070d32023`. The expectations-only freeze is
`d5ec23ed13380df6e2fafbb2267494c55fc64380`, which precedes implementation
commit `7928f084a0da21bd2aa6a4329e8e0d1008896f3f` and every scored run. The
freeze records that the worktree was clean before its files were authored.

The unchanged suite bytes have SHA-256
`5ec3296dd34ef42c65bc3677916aedc284585ec5b6b11ea2ecd5873a3e5d2266`.
The cached checkpoint is
`ibm-granite/granite-3.0-1b-a400m-instruct` revision
`ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`; its config, weight and byte-count
identities all matched the freeze before extraction.

## Published inventories

| Framework | Canonical inventory | Cases | Bytes |
|---|---|---:|---:|
| vLLM 0.26.0 | [`e74e995a...7119af9`](../../offline/calibration/model-inventories/e74e995a89588a304aa852593d3505cfab9a94d2c068c82dbe9c776da7119af9.json) | 15 | 24,911 |
| SGLang `0.0.0.dev1+g8f2a3ad6d` | [`147fe439...36c54c`](../../offline/calibration/model-inventories/147fe4398d5615afe7954c9199134de37f706da2cecda8fc37d6514ad936c54c.json) | 15 | 24,959 |

Both records pass the strict typed reader, reproduce their filename SHA-256
from their canonical bytes and have identical denominators after removing the
framework identity and framework-specific physical-join task. Their record
hashes differ, as required, because framework provenance is part of identity.

## Evidence by class

### Behavioral relations

- R1 checked 75 family entries per framework. Family order, typed shapes,
  integer FLOPs and integer HBM bytes matched `step_kernels()` exactly, and
  family sums matched the fused `step_kernel()` work.
- R2 found exactly four once-per-layer families and one once-per-step family.
  At 24 layers every one of the 30 framework-case rows therefore contains
  exactly `4 * 24 + 1 = 97` logical invocations.
- R3 propagated all three five-point sweeps exactly: prefill prompt length,
  memory-decode context length and MoE-decode batch all changed their declared
  shape coordinates in suite order.
- R4 produced exactly two graph-template classes per framework: one shared by
  the ten single-rank cases and one shared by the five four-rank MoE cases.
  The two hashes differ, and the corresponding classes agree across
  frameworks.

All four behavioral relation families passed. This count excludes exact
oracles, fatal guards, structural invariants and native tests.

### Exact oracle

R5 passed: each framework's two fresh extractions were byte identical, while
the vLLM and SGLang inventory bytes differed because their pinned framework
identities differ.

### Fatal guards

No fatal guard was violated. Suite bytes, checkpoint identity, framework
identity, case totality, family totality, integer work and absent physical
identity markers all held. Ordinary `simllm` imports in both framework
environments loaded neither framework runtime. Had any guard failed, the run
would have been void and no behavioral score would be interpreted.

### Structural negative controls

The live unflagged vLLM command rejected with status 2 before creating an
object. Targeted native tests separately proved rejection for an unsupported
checkpoint or framework geometry, a partial StepRecord reload and an unknown
family; the lazy-import control also passed. Those four tests passed and are
not added to the behavioral denominator.

## Sanity bounds

The deciding 97 is a logical structure count, not an observed launch or time.
Its first-principles floor is four required layer-repeated families over 24
layers plus one LM-head family, or 97. Its ceiling under this exact frozen
family abstraction is the same 97 because no optional family exists. Every
case sits exactly on both bounds.

Three independent checks support that structural number. Both framework
configuration surfaces recover the suite's exact 24-layer and 32-expert model
geometry. Family FLOPs and HBM bytes conserve exactly against the independently
computed fused step. Graph normalization yields the expected single-rank and
four-rank collective topology classes. None is a physical plausibility check:
the run contains no service time, bandwidth, rate or measured launch count to
compare with silicon, and the strict absent-by-design markers prevent such a
claim.

## Reproduction

Machine-specific paths belong in the gitignored local environment file. With
`VLLM_PYTHON`, `SGLANG_PYTHON`, `CHECKPOINT_ROOT` and `COMP54_RUN_ROOT` bound to
the pinned local runtimes, exact checkpoint snapshot and a fresh external
output directory, run:

```bash
.venv/bin/python examples/model_extraction_v1/run_study.py \
  --vllm-python "$VLLM_PYTHON" \
  --sglang-python "$SGLANG_PYTHON" \
  --suite-root offline/calibration \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --output-root "$COMP54_RUN_ROOT"
```

The runner keeps raw logs, repeated StepRecord streams, canonical objects and
its canonical result record outside Git. Only the two small content-addressed
inventory objects are promoted into the repository.
