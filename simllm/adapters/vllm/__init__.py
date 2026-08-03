"""vLLM adapter (milestone M2).

vLLM's v1 engine selects its executor class from a dotted import path, so no
fork is required::

    vllm serve <model> --distributed-executor-backend simllm.adapters.vllm.SimExecutor

``SimExecutor`` will subclass ``vllm.v1.executor.abstract.Executor``: it
serves the init-time RPCs with model-derived values, pins the simulated KV
pool via ``CacheConfig.num_gpu_blocks_override``, and fabricates a
``ModelRunnerOutput`` per step with simulated timing. The v1 scheduler,
KV-cache manager and prefix-cache hashing run unmodified.

This package intentionally has no import-time vLLM dependency; the executor
is only importable once the adapter lands and vLLM is installed.
"""
