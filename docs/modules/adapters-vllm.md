# simllm.adapters.vllm

vLLM frontend adapter. No fork required: vLLM's v1 engine resolves the
executor class from a dotted import path.

## Interface (planned)

```
vllm serve <model> --distributed-executor-backend simllm.adapters.vllm.SimExecutor
```

`SimExecutor` subclasses `vllm.v1.executor.abstract.Executor`:

- serves the init-time RPCs (`get_kv_cache_spec`,
  `determine_available_memory`, `initialize_from_config`,
  `compile_or_warm_up_model`, `initialize_cache`, `get_supported_tasks`)
  with model-derived values, pinning the simulated vRAM pool via
  `CacheConfig.num_gpu_blocks_override`;
- fabricates `ModelRunnerOutput(req_ids, req_id_to_index,
  sampled_token_ids)` per step with simulated timing;
- exports the placement manifest from the workers via `collective_rpc`.

The v1 scheduler, KV-cache manager, block pool and prefix hashing are
CPU-side bookkeeping in the scheduler process and run unmodified. Timing has
two modes: paced (delay the future by the simulated latency; stock metrics
work unchanged) and virtual (return immediately; report sim-native metrics).

## Status

Design and seam validated against vLLM v0.14.0 (2026-08-03 research);
package is a documented placeholder with no import-time vLLM dependency.
Implementation is milestone M2.

## Open tasks

- VLLM-1: implement `SimExecutor` (offline mode first), pinned to
  vLLM v0.14.0, validated against the granite capture numbers.
- VLLM-2: worker-side placement-manifest exporter (`collective_rpc`
  callable or worker extension class), shared schema with declared
  manifests (see [placement](placement.md), PLACE-3).
- VLLM-3: sim-native metrics export via a `vllm.stat_logger_plugins`
  stat logger for virtual-time runs.
