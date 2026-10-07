# Atomic receive successor study

The study checks earned-credit transfer when the final packet debit and a
successful successor admission share one timestamp. Its expectations are
frozen at `dbb60ad3fc9f864a372188c143b44b282b22530c` before implementation and
execution. The native device and C facade use identical public configurations
and literal packet inputs.

Run from the repository root with CMake, a C++17 compiler and Python available.
Choose a new external output directory:

```sh
python examples/rnic_rx_atomic_successor_v1/run_native.py \
    --output "${SIMLLM_DATA_ROOT}/rx-atomic-successor-new" --mutants
```

The runner refuses existing output directories, verifies the committed
expectation, builds with warnings as errors, runs RNIC CTest, checks both
frontends, and compiles five separately identified negative source copies.
It records source, tool, binary, command, trace and ABI provenance. Process
guards bound each stage and clean up owned descendants on timeout.

[RESULTS.md](RESULTS.md) separates exact rows, relation instances, fatal
controls, executables and compatibility artifacts. This is continuously ready
receive-service component evidence. Framework selection and request metrics
remain BACK-76 in [the owning module](../../docs/modules/backends.md).
