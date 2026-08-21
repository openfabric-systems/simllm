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

## Developer Certificate of Origin

SimLLM uses the [Developer Certificate of Origin 1.1](DCO) instead of a
Contributor License Agreement. Every commit in a contribution must include a
`Signed-off-by` trailer certifying that the contributor has the right to submit
the work under the repository's license.

Add the trailer automatically with Git's `--signoff` option:

```bash
git commit --signoff -m "Describe the change"
```

The resulting commit message ends with:

```text
Signed-off-by: Your Name <you@example.com>
```

Use your real name and an email address that matches the commit author. The
sign-off is a certification under the DCO, not a copyright assignment or a
Contributor License Agreement.

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
- Module docs under `docs/modules/` follow the skeleton in
  [docs/modules/FORMAT.md](docs/modules/FORMAT.md). `pytest -q` checks it; run
  `python scripts/check_docs_format.py` for the messages on their own.
- Changes to the backend simulators belong in their own repos
  (`third_party/atlahs`, `third_party/htsim`); SimLLM only pins refs.
- Add a test for every new module. CI must stay green.

## Code style

- Python ≥ 3.10, `ruff` for linting, type hints on public APIs.
- Small, focused PRs with a clear description of the modeling assumption
  being added or changed.
