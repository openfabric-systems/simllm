# CORE-66 EP4 environment retry expectations

## Unchanged capture cell

This retry changes only the execution environment and its fail-fast checks.
The cell in `core66_ep4_expectations.json` and every signed deviation remain
unchanged: one GH200 node, four GPUs and ranks, four routed experts resident
per rank, 16 routed experts total, batch 32, key-value cache length 2,000,
multi-token prediction disabled, dummy weights, data-parallel attention and
language-model head, DeepEP, three dense layers, one mixture-of-experts layer
and one measured decode iteration. SGLang remains pinned at
`bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3`.

The prior EP12 and EP8 refusal records and job `200879` remain immutable. This
is one newly authorized submission, not an amendment or continuation of that
failed job.

## Recovered environment

The environment authority is the successful CORE-61 depth retry. Its merged
batch script and retained Merlin script both hash to
`6f467a4bf4d0fee0d390113c4bf580edfbda3bbe48b4bf6bb0cfec78a5c9020e`.
Decode job `200138` completed on a GH200 node and recorded nonempty CUDA
activity with NVIDIA Nsight Systems version 2025.1.3.140.

The retry therefore uses these commands and no substituted module selection:

```bash
module purge
module load gcc/12.3.0
module load cuda/12.9.1
export PATH="${SIMLLM_CORE66E_GH200_VENV}/bin:${PATH}"
SIMLLM_CORE66E_PYTHON="${SIMLLM_CORE66E_GH200_VENV}/bin/python"
```

The interpreter is the retained ARM aarch64 Python 3.11.11 binary used by
CORE-61. It must hash to
`71eb688df288d106c65112c2d8a69bb22c0e4737951f8c0946476fd0f43dde7b`.
It must report Torch 2.13.0+cu129 and Torch CUDA 12.9. The driver may advertise
CUDA API 13.1; that does not replace the deliberately selected cu129 runtime
and CUDA 12.9.1 toolchain.

## Fail-fast preflight

Before any profiler call, the job checks the module commands, CUDA 12.9
compiler, exact `nsys` version, presence of `ncu`, PATH-selected interpreter,
interpreter hash and architecture, Torch and CUDA versions, visible GH200,
pinned SGLang commit, capture-module import and an ARM aarch64 DeepEP
extension. A failed check writes its exact command and observation, finalizes
the small evidence manifest and exits without invoking the profiler. No second
submission follows.

The previously staged CPython 3.12 x86-64 DeepEP target is not acceptable on
the ARM aarch64 compute node. The preflight must reject it rather than allow a
later import or loader failure.

## Publication gate and guard

The standard-decode movement may be published only when DeepEP dispatch and
combine service has a zero-parameter registered-cell projection and
rank-preserving high-bandwidth memory read and write bytes price the other
correction direction. Otherwise the movement remains null. The downward
correction is never published alone, and no parameter is fitted.

Held-out use in arithmetic, prediction comparison, fitting or a published
reproduction is fatal. Incidental exposure without use is survivable when it
is disclosed, because the zero-parameter arithmetic remains independently
checkable. Pytest, ruff, documentation checkers and git plumbing remain
automated-process exemptions.
