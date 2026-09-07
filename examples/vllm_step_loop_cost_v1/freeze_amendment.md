# Pre-run CPU distribution identity amendment

The supplied environment imports successfully. Its module version is
`0.27.1` and its installed distribution version is `0.27.1+cpu`, the same
CPU build identifier used by `examples/vllm_collective_timing_v1/run_study.py`.
The scheduler source and model configuration hashes match the original
freeze exactly.

Attempt `attempt-001` stopped in version preflight before engine construction,
the attribute-access microbenchmark or any timed workload. No phase or loop
measurement exists from that attempt. The original expectations file and
failed-attempt log are preserved.

This expectations-only amendment pins the distribution version to exactly
`0.27.1+cpu` and the module version to exactly `0.27.1`. Both checks must pass,
alongside the unchanged exact scheduler source and model configuration hashes.
No other build suffix or version is accepted. Record both versions in the
result provenance. Every workload, sweep, timer, repetition, relation, fatal
guard and reporting clause in the original freeze remains unchanged.

This amendment precedes the corrected version-check implementation and the
first measurement. The result cites its commit as the final pre-run freeze
and cites the original expectations commit separately.
