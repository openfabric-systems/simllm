# simllm.adapters.sglang

SGLang frontend adapter. The seam is the TP worker.

## Interface (planned)

`SimTpModelWorker` implements SGLang's `BaseTpWorker`:
`forward_batch_generation(batch)` returns a `GenerationBatchResult` with a
fabricated int64 `next_token_ids` tensor and simulated timing. It is
selected at the scheduler's worker-construction point
(`Scheduler.init_tp_model_worker`), the same seam SGLang uses for
platform-specific workers; the first iteration runs with
`--disable-overlap-schedule` and CPU-resident pool tensors.

RadixCache prefix matching, eviction, and the token/request pool accounting
are scheduler-side index bookkeeping and stay real, so radix hit rates and
vRAM pressure respond to the workload exactly as in production. SGLang's
scripted-runtime test harness (`SGLANG_TEST_SCRIPTED_RUNTIME`) is the
intended deterministic validation driver.

## Status

Design and seam validated against SGLang main as of 2026-08-03 (note: the
old `TpModelWorkerClient` overlap worker no longer exists; overlap is now
in-scheduler dual-stream with a `FutureMap`). Package is a documented
placeholder with no import-time SGLang dependency. Implementation is
milestone M3.

## Open tasks

- SGL-1: implement `SimTpModelWorker` against a pinned SGLang version.
- SGL-2: upstream (or carry) the small worker-class selection flag at
  `Scheduler.init_tp_model_worker`.
- SGL-3: RadixCache-aware studies: prefix-hit rate and re-prefill traffic
  vs shared-prefix workload structure.
