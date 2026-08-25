# Framework pin bump v1 results

## Outcome

What ran: the frozen VLLM-30 and SGL-32 conformance matrix inspected both
selected source trees, exercised every named adapter family in the new
framework environments, ran the required CPU-only live smokes, and extracted
the 15-case Granite inventory twice per new identity.

What came out: the qualification is nonvoid for both frameworks. vLLM 0.27.1
and SGLang `bfeae4e` both resolve Kimi K3, Qwen3.5 and the required Granite
families through their native registries. Every targeted test and live smoke
passed. Both repeated inventory pairs are byte identical, and every case sits
at the deciding exact structural bound of **97 logical invocations**.

What it changes: VLLM-30 and SGL-32 close. The current framework pins, module
status and coverage headers now name the qualified identities, and the
Granite Models cell publishes the two new content-addressed inventories beside
the old ones. Kimi K3 is no longer blocked by framework registry support;
COMP-54 still owns exact checkpoint binding and COMP-59 owns its reduced-depth
physical capture envelope.

What it does not change: no model checkpoint was downloaded, no Kimi K3 model
was instantiated, and no GPU kernel, code object, physical launch, device
timing, time to first token (TTFT), time per output token (TPOT), VLLM-12,
SGL-10, COMP-6, COMP-54 or COMP-59 was validated or closed. Every old-identity
inventory and prior extraction-study byte remains unchanged historical
evidence.

## Chronology and selected identities

The expectations-only freeze is
`f8501f66b6feeef7fad0e3eed8be7744be7c2163`, which precedes every adapter
repair, pin edit, live smoke and extraction. It records the clean status of the
base worktree at `cce58f384b73cdcde088b14a934a2f25eb81677f`.

The original `transformer-dag-v1` suite remains byte exact at SHA-256
`5ec3296dd34e...e5d2266`, preserving the prior extraction freeze. The new
`transformer-dag-v1-frameworks-2026-08-24` suite has SHA-256
`1282207c5ad8...ddedd9dce` and differs only in its two framework identity
rows. Its model identity, 15 cases and every authored calibration field equal
the historical suite exactly.

| Framework | Installed identity | Source identity |
|---|---|---|
| vLLM | stable release 0.27.1 | commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| SGLang | `0.5.19.dev345+gbfeae4e79` | commit `bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3`, tree `9ffe149f40e1cd5bff7dadc6806ad1927d312e69` |

Both environments used Python 3.10, no pip cache and the ordinary framework
wheel composition. SGLang was installed editable from the exact detached
source commit. SimLLM was installed editable into both environments. The
previous pinned environments and checkout were not modified.

## Source and registry conformance

The vLLM registry reports all six inspected architecture names as supported:
`KimiK3ForConditionalGeneration`, `Qwen3_5ForConditionalGeneration`,
`GraniteForCausalLM`, `GraniteMoeForCausalLM`,
`GraniteMoeHybridForCausalLM` and `GraniteMoeSharedForCausalLM`.

SGLang's ordinary dynamic `EntryClass` registry reports Kimi K3, Qwen3.5,
dense Granite, Granite MoE, shared Granite MoE and hybrid Granite. In
particular, `kimi_k3.py` contributes `KimiK3ForConditionalGeneration`; it is
not a documentation-only or fallback registration.

| Framework | Seam | Result |
|---|---|---|
| vLLM | executor RPC | Passed: the three abstract methods and the complete simulated RPC table remain valid; unknown optional RPCs retain their accounted compatibility behavior |
| vLLM | flagged worker skeleton | Passed after carrying the renamed sleep buffers and worker sentinel; the new device-backed fault-tolerance path now refuses loudly under VLLM-13 |
| vLLM | simulated coordinator | Passed: all-reduce, all-gather, broadcast, send and receive signatures match `GroupCoordinator` |
| vLLM | step serializer | Passed: current and historical identity fixtures round-trip and remain byte exact |
| vLLM | placement exporter | Passed: the single public extension entry and source-backed optional discovery fields remain valid |
| vLLM | COMP-54 extraction | Passed twice over all 15 Granite cases with identical bytes |
| SGLang | plugin entry | Passed: one installed plugin entry remains inert while disabled and replaces only the TP worker boundary when enabled |
| SGLang | TP worker and stub | Passed after mirroring context length, draft attention, current graph accounting and weight-load fields; draft, multi-layer EAGLE and hidden-state paths refuse loudly under SGL-5 |
| SGLang | CPU engine | Passed: the explicit CPU configuration reached `SimTpModelWorker` and selected the CPU-native attention backend without loading weights |
| SGLang | streaming driver | Passed: the in-process pump preserved request identity and returned one terminal result |
| SGLang | simulated coordinator | Passed: signatures match the selected source, including output-list all-gather, and the disabled path remains exact identity |
| SGLang | COMP-54 extraction | Passed twice over all 15 Granite cases with identical bytes |

## Native tests and live reachability

The new vLLM environment ran the six frozen test families: 119 passed and 2
absence-only tests skipped. The new SGLang environment ran its eight frozen
test families: 169 passed and 2 absence-only tests skipped. The installed
source-shape checks compare the adapter constructor and forward signatures
directly with the selected framework classes.

The vLLM skeleton live smoke forced the framework CPU platform, resolved the
cached Granite MoE configuration, reached `SimWorker` and `SimModelRunner`,
generated token id 24577 twice, and emitted two ordered
`atlahs-closed-loop-step-v1` records. Its unflagged control rejected before
worker construction. The separate `SimExecutor` smoke generated two tokens,
emitted two records and ended with an empty `unhandled_rpcs` counter.

The SGLang live smoke used its CPU-native engine path, reached
`SimTpModelWorker`, generated two tokens for request `r0`, emitted two ordered
records and returned one terminal result with the same request identity. Its
disabled plugin control found exactly one SimLLM entry point and left
`Scheduler.init_tp_model_worker` unchanged.

All live commands used offline Hugging Face mode and the already-cached
Granite configuration and tokenizer. They loaded no weight tensor and made no
network request. SGLang printed warnings while probing an unusable installed
CUDA stack before retaining its requested CPU path; no CUDA allocation or GPU
execution occurred.

## Published Granite inventories

| Framework | Canonical inventory | Cases | Bytes |
|---|---|---:|---:|
| vLLM 0.27.1 | [`33758c3c...d8725e2`](../../offline/calibration/model-inventories/33758c3c71d5dacae8f6a82cb937f5e70b0d28eaa7c2358c13baccbd8d8725e2.json) | 15 | 24,911 |
| SGLang `0.5.19.dev345+gbfeae4e79` | [`3998b208...e2b9b9a`](../../offline/calibration/model-inventories/3998b208bef6498709a9a4b6b2ca2e1825a9db54918186f4fc4387a9ee2b9b9a.json) | 15 | 24,962 |

Each framework's two extractions produced the same canonical record SHA-256.
Every case retains the ordered family list `attn_gemm`, `attn_score`,
`mlp_gemm`, `lm_head`, `kv_read`. The first four families repeat once over 24
layers and the last occurs once, so every total is exactly
`4 * 24 + 1 = 97`. The ten single-rank cases share one graph template, the
five four-rank cases share one other template, and those templates differ.
After removing framework provenance and the framework-specific physical join
task, the two complete inventories are structurally identical. Code-object
hashes and observed launches remain `absent-by-design`.

The structural floor is 97 because all four layer-repeated families and the LM
head are mandatory. The ceiling is also 97 because the suite declares no
optional family. All 30 framework-case rows sit exactly on both bounds. This
is a configuration-derived workload count, not a physical launch count or a
timing measurement.

## Historical identity guards

The old vLLM inventory still hashes to
`e74e995a89588a304aa852593d3505cfab9a94d2c068c82dbe9c776da7119af9`,
and the old SGLang inventory still hashes to
`147fe4398d5615afe7954c9199134de37f706da2cecda8fc37d6514ad936c54c`.
The four frozen `model_extraction_v1` files retain their registered hashes:
`7299914b10bc...e7c72d11f`, `87ba5bb20b9d...dc83e5cd77`,
`d05cab50ffa9...975b2a08a` and `6eee29129938...90dc7c7dd` in the order
listed by the freeze. No old object was rewritten or relabeled.

## Deviations

Two superseded environment destinations hit `ENOSPC` before either new
runtime completed. The integrator selected the final environment volume, and
both clean builds then completed with pip cache disabled. No failed attempt
changed a tracked repository file or an old runtime.

The first executor and SGLang live-smoke wrappers each asserted a convenience
attribute that the repository type does not expose, after the underlying live
run had already passed and written its external record stream. The corrected
wrappers inspected the public schema and completion fields under fresh output
names; both final runs passed. The earlier external outputs were retained.
These were harness assertion defects, not seam or framework failures.

## Reproduction

Machine-specific paths belong in the gitignored local environment file. Bind
`VLLM_PYTHON`, `SGLANG_PYTHON`, `CHECKPOINT_ROOT`, `HF_HOME` and a fresh
`PINBUMP_RUN_ROOT`, keep `HF_HUB_OFFLINE=1`, then invoke
`simllm-calibrate extract` twice for each framework against
`offline/calibration` with
`--suite transformer-dag-v1-frameworks-2026-08-24`. The vLLM calls
additionally set `SIMLLM_VLLM_WORKER_MODE=skeleton` and
`VLLM_ENABLE_V1_MULTIPROCESSING=0`; the SGLang calls set
`SIMLLM_SGLANG_ENABLE=1`. Compare each repeat pair byte for byte and verify
that each object filename equals the SHA-256 of its canonical bytes.
