# Framework pin bump v1 expectations

## Freeze scope and chronology

This expectations-only freeze precedes every adapter change, pin change,
generated inventory and live run for VLLM-30 and SGL-32. Immediately before
this file was authored, `git status --porcelain=v1` produced no rows at commit
`cce58f384b73cdcde088b14a934a2f25eb81677f`. The worktree was clean. Failed
attempts to allocate the required external runtimes changed no repository
file and produced no conformance result.

Source discovery selects vLLM 0.27.1 at commit
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` and SGLang main at commit
`bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3`, source tree
`9ffe149f40e1cd5bff7dadc6806ad1927d312e69`. The installed SGLang version is
recorded from the completed editable environment before its pin identity is
published. Discovery may identify source facts, but no source inspection is a
substitute for the live and adapter conformance gates below.

No measured value, generated inventory identifier, observed pass count or
outcome-dependent exception appears in this freeze. A failed fatal guard voids
the affected framework half, leaves its owning task open and prevents its pin
from being published as verified.

## Frozen source and registry expectations

Both selected sources must expose all three required model surfaces through
their ordinary model registries:

| Framework | Kimi K3 | Qwen3.5 | Granite |
|---|---|---|---|
| vLLM 0.27.1 | `KimiK3ForConditionalGeneration` | `Qwen3_5ForConditionalGeneration` | dense, MoE, shared-MoE and hybrid Granite entries remain registered |
| SGLang main | `kimi_k3.py` contributes `KimiK3ForConditionalGeneration` through `EntryClass` | `qwen3_5.py` contributes `Qwen3_5ForConditionalGeneration` through `EntryClass` | dense, MoE and hybrid Granite modules continue to contribute their entry classes |

An architecture that is merely documented, present in an unregistered module
or reachable only through a silent fallback does not satisfy this gate. The
ordinary registry must select its native implementation. Import or registry
failure is fatal for that framework half.

## Conformance matrix

Every row below is required. Source inspection freezes the upstream shape,
adapter tests exercise positive and negative contracts, and the live smoke
proves that the installed runtime reaches the seam. A source-only pass cannot
replace a failed live row.

| Framework | Seam | Required conformance |
|---|---|---|
| vLLM | executor RPC | `SimExecutor` remains a valid v1 executor with the complete required RPC surface; structured scheduler output and device-requiring paths continue to refuse loudly |
| vLLM | flagged worker skeleton | the explicit skeleton flag remains mandatory; the source construction sequence and every load-bearing runner, cache, scheduler and output field are either re-anchored exactly or rejected before plausible output |
| vLLM | simulated coordinator | all-reduce, all-gather, broadcast, send, receive and rank-membership signatures mirror the selected `GroupCoordinator`; invalid groups and unservable payloads fail before observation |
| vLLM | step serializer | every `StepRecord` field used by the scheduler, sampled-request identity, padding count, replay and completion remains round-trippable; the old identity stream fixture remains byte exact |
| vLLM | placement exporter | the worker extension still exposes the required public entry, discovers layer and expert ownership from the selected source fields and rejects malformed inputs without inventing placement |
| vLLM | COMP-54 extraction | the CPU-safe configuration seam loads the cached Granite checkpoint, projects all 15 authored suite cases and emits a total canonical inventory under the new framework identity |
| SGLang | plugin entry | package entry-point discovery and direct `install()` remain inert unless explicitly enabled, then replace exactly the intended TP worker construction boundary |
| SGLang | TP worker and stub | `SimTpModelWorker`, its stub runner and scheduler-visible construction template carry every load-bearing field used by the selected source; missing or moved fields refuse before a plausible batch result |
| SGLang | CPU engine | the selected runtime constructs Granite with `device="cpu"`, no usable CUDA GPU and the explicit plugin gate, then reaches the simulated worker without silently choosing a CUDA path |
| SGLang | streaming driver | `SglangSchedulerPump` still mirrors one normal event-loop body, projects only finished output rows, preserves request identity and refuses unsupported overlap or chunked-prefill shapes |
| SGLang | simulated coordinator | all mirrored collective signatures, including output-list all-gather, match the selected source; the communicator flag-off path remains the exact identity bypass |
| SGLang | COMP-54 extraction | the CPU engine configuration seam loads the cached Granite checkpoint, projects all 15 authored suite cases and emits a total canonical inventory under the new commit and tree identity |

## Required test families

The new vLLM environment must pass
`tests/test_adapters_vllm.py`, `tests/test_vllm_communicator.py`,
`tests/test_vllm_oracle.py`, `tests/test_model_inventory.py`,
`tests/test_model_inventory_artifacts.py` and the calibration command tests.
The new SGLang environment must pass `tests/test_adapters_sglang.py`,
`tests/test_adapters_sglang_pump.py`, `tests/test_sglang_communicator.py`,
`tests/test_sglang_oracle.py`, `tests/test_sglang_client.py`, the two model
inventory families and the calibration command tests. Tests must run with the
worktree first on `PYTHONPATH`, so an older installed SimLLM package cannot
mask drift.

Repository-wide `.venv/bin/ruff check .` and `.venv/bin/pytest -q` must pass
before the freeze commit and before every later commit. Documentation format,
task registry reconciliation and content-addressed artifact checks are part of
that fatal gate, not scored adapter relations.

## Required live smokes

The vLLM runtime runs offline against the cached Granite revision
`ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445` with `HF_HOME` bound to the
existing cache, `HF_HUB_OFFLINE=1` and
`VLLM_ENABLE_V1_MULTIPROCESSING=0`. One executor-path smoke and one explicitly
flagged worker-skeleton smoke must each construct in process, generate two
fabricated tokens for one request, emit two ordered StepRecords and shut down
without allocating CUDA memory. The unflagged skeleton control must reject
before worker construction.

The SGLang runtime uses the same cached checkpoint, `device="cpu"`,
`HF_HUB_OFFLINE=1` and `SIMLLM_SGLANG_ENABLE=1`. Its in-process scheduler pump
must construct `SimTpModelWorker`, submit one tokenized request, generate two
fabricated tokens, emit two ordered StepRecords and project one terminal
streaming result with the same request identity. The plugin-disabled control
must leave the selected SGLang construction target untouched. Unsupported
chunked-prefill or overlap shapes remain explicit refusals.

These are reachability checks, not device-timing studies. They carry no
physical kernel, bandwidth, time to first token (TTFT) or time per output token
(TPOT) claim.

## Granite re-extraction expectations

The unchanged 15-case `transformer-dag-v1` authored grid is extracted twice
per new framework identity with the COMP-54 command. Each repeat pair must be
byte identical. Every case must retain the ordered families `attn_gemm`,
`attn_score`, `mlp_gemm`, `lm_head`, `kv_read`, with four once-per-layer
families and one once-per-step family. At 24 layers the exact logical count is
therefore `4 * 24 + 1 = 97` per case. The ten single-rank cases must share one
normalized graph template, the five four-rank cases another, and the two
templates must differ.

The floor on the logical family count is 97 because all four layer-repeated
families and the LM head are mandatory. The ceiling is also 97 because this
suite declares no optional family. A result outside that exact structural
bound is fatal. The count is not a physical launch count and has no timing
interpretation.

The vLLM and SGLang inventories may differ only through framework provenance
after their structural denominators are compared. Any framework-specific
structural difference must be explained as selected-source behavior or treated
as a defect. Code-object hashes and observed physical launches remain
`absent-by-design`.

## Historical byte identity and bypass

Existing content-addressed inventory objects are historical evidence. They
must not be removed, rewritten, copied under a new identity or relabeled:

| Historical identity | Path and required SHA-256 |
|---|---|
| vLLM 0.26.0 | `offline/calibration/model-inventories/e74e995a89588a304aa852593d3505cfab9a94d2c068c82dbe9c776da7119af9.json`, `e74e995a89588a304aa852593d3505cfab9a94d2c068c82dbe9c776da7119af9` |
| SGLang `8f2a3ad` | `offline/calibration/model-inventories/147fe4398d5615afe7954c9199134de37f706da2cecda8fc37d6514ad936c54c.json`, `147fe4398d5615afe7954c9199134de37f706da2cecda8fc37d6514ad936c54c` |

The entire prior `model_extraction_v1` evidence record is also immutable. Its
four file hashes at freeze time are:

- `expectations.md`:
  `7299914b10bcb3f914a74796a4af82092fa39e1fbddc44c09d9fbc3e7c72d11f`;
- `expectations.json`:
  `87ba5bb20b9d10846364ac988eae88ef8c0a7c80a56e53282dc7d5dc83e5cd77`;
- `run_study.py`:
  `d05cab50ffa9e0af0057bced1bdc8666cfc458d3b56f9abd8becade975b2a08a`;
- `RESULTS.md`:
  `6eee291299387c5e13f1d88bbe0a6b97ad5237f7145e50e795bb1e390dc7c7dd`.

Existing old runtime environments and the old SGLang checkout remain
untouched and reproducible beside the new environments. Current pin authorities,
coverage headers and current Models cells switch to the new identities only
after conformance passes. Historical study prose keeps the version it actually
ran.

## Task effect

VLLM-30 closes only if every vLLM row, its two live smokes, repeated Granite
extraction and historical byte guards pass. SGL-32 closes only if every SGLang
row, its live smoke, repeated Granite extraction and historical byte guards
pass. The halves are independent. A failed half retains its original task and
records the exact newly discovered breakage under an available reserved ID.

Closing either task changes only framework support and the Granite offline
inventory identity for that half. It does not validate Kimi K3 weights, GPU
kernels, physical launch identity, device timing, TTFT, TPOT, VLLM-12, SGL-10,
COMP-6 or the remaining COMP-54 model columns.
