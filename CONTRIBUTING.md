# Contributing to SimLLM

Thanks for your interest in SimLLM. The project is young and contributions of
every size are welcome, from a topology config to a new frontend adapter.

## Development setup

```bash
git clone https://github.com/openfabric-systems/simllm.git
cd simllm
git submodule update --init third_party/atlahs third_party/htsim
pip install -e .[dev]
```

Run the checks locally before opening a PR:

```bash
ruff check .
pytest -q
```

Changes under `simllm/backends/rnic/` also run the dependency-free native
gate:

```bash
cmake -S simllm/backends/rnic -B build/rnic -DCMAKE_BUILD_TYPE=Debug \
  -DSIMLLM_RNIC_WARNINGS_AS_ERRORS=ON
cmake --build build/rnic --parallel
ctest --test-dir build/rnic --output-on-failure
```

## Where to contribute

- **Workload generators** (`simllm/workload/`): arrival processes, length
  distributions, shared-prefix structure, trace replay.
- **Compute-time providers** (`simllm/compute/`): measured prefill/decode
  latency profiles for specific GPUs and models, roofline refinements,
  offline SASS-level (Accel-Sim/GPGPU-Sim) table generation.
- **Placement & fabric** (`simllm/placement/`): manifest extraction for new
  framework versions, fabric topology descriptions, NIC-selection logic.
- **Traffic models** (`simllm/traffic/`): TP/PP collectives, MoE
  dispatch/combine, KV-cache transfers, collective-algorithm expansion.
- **Frontend adapters** (`simllm/adapters/`): vLLM and SGLang first; the
  adapter contract is one scheduler-step record in, one step result out.
- **Backends** (`simllm/backends/`): invocation and result parsing for
  htsim profiles and LogGOPSim.
- **Docs, examples, plotting.**

## Ground rules

- Keep the core framework-agnostic: nothing in `simllm/core`, `simllm/goal`
  or `simllm/backends` may import vLLM or SGLang.
- Changes to the backend simulators belong in their own repos
  (`third_party/atlahs`, `third_party/htsim`); SimLLM only pins refs.
- Add a test for every new module. CI must stay green.

## Code style

- Python ≥ 3.10, `ruff` for linting, type hints on public APIs.
- Small, focused PRs with a clear description of the modeling assumption
  being added or changed.
