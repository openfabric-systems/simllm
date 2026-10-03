# Serial SM70 component results

The expectations commit is `5def25e7`. The component preserves the existing
provider, runtime and GPU-envelope files. The optional Volta envelope lives
in `simllm.compute.volta` so retained historical source freezes remain exact.

| Gate | Terminal result | Evidence class |
| --- | --- | --- |
| Replay adapter and frozen count/clock/cycle grid | 29 passed | Unit fixtures, no simulator execution |
| Historical decode-HBM and deployment source guards | 24 passed | Regression |
| Affected compute, calibration, runtime, execution and overlap checks | 127 passed | Regression |
| Module format | 11 modules pass | Documentation |
| Changed Python lint and whitespace | Pass | Static checks |
| Complete final `pytest -q` | 6,984 passed, 45 skipped | Full regression |
| Complete `ruff check .` | Pass | Static checks |
| CUDA 12.4.99 compilation with `-arch=sm_70` | Pass | Compilation |
| Code object inspection | Two SM70 ELF objects | Binary inspection |
| Execution on a non-SM70 verification GPU | Exit 1 with explicit SM70 requirement | Capability refusal |

The probe source SHA-256 is
`befcb9e597ff49adac4bbfe879637e18bce9c0059ff34d01388f7e0b0e6e16b8`.
It checks every GPU matrix and reduction value for finiteness before reducing
the error. No CUDA numerical result or calibrated timing is claimed here.
Its GEMM/reduction chain is a capture diagnostic, not a model implementation.

An initial interrupted full suite reported 2,264 passed, 19 failed and 23
skipped. All 19 failures came from historical source-hash guards after the
first additive envelope edit touched the protected transformer source. The
untouched `622188a1` baseline passed both implicated files, 24 cases total.
Moving the optional envelope into its own module restored their 24 passes.
The interrupted full suite is retained evidence, not a complete-suite pass.
The repaired complete suite finishes separately in 1,534.87 seconds, with
4,689 warnings and exit 0. Its retained log is `build/full-pytest-final.log`.

Real replay remains gated on authentic captured SM70 traces and a supported
installed simulator. No trace is synthesized by the adapter. The official
framework and GPGPU-Sim pins identify the QV100-SASS simulation profile;
they do not establish Tesla V100 PCIe calibration. The adapter verifies
config dependency closure and verifies input hashes again after execution.
Completed observations remain candidates and carry the config, executable
and capture identities. Raw logs and binaries belong in ignored `build/`.
