# Local-shard kernel collector result

What ran: the frozen `local_shard_kernel_collector_v1` contract study ran an
external nonphysical target twice over tensor parallel sizes 1 and 4 and batch
sizes 1 and 8.

What came out: **all 4 of 4 cells passed**. Every request and result retained
its content identity, synthetic row count equaled batch size, the fixture's
local GEMM output width changed from 4096 at tensor parallel one to 1024 at
tensor parallel four, repeated kernel order and result bytes were identical,
and all sample blobs matched their recorded byte count and SHA-256.

What it changes for the project: the local `simllm-calibrate run` slice of
COMP-50 is executable through one framework-neutral external target contract.
The request distinguishes logical parallelism from the physical shard and the
result rejects a different model revision, framework dispatch signature,
physical device or instruction-set architecture (ISA). COMP-50 remains open on
its other package, doctor, compiler and submission surfaces. COMP-1 remains
open on target-silicon evidence.

What it does not change: no graphics processing unit (GPU) timing was measured,
no vLLM or SGLang model column was filled, no kernel or network constant was
installed, and no time to first token (TTFT) or time per output token (TPOT)
changed. The fixture durations are nonphysical contract values and carry no
performance meaning.

## Hardware boundary

On a real GPU, the framework target owns the actual compilation and execution.
It receives deterministic synthetic token rows and one exact physical shard
coordinate. The generic collector then checks what came back. An A100 request
accepts only an SM80 result. It cannot be labeled as an SM90 or AMD result, and
an isolated shard cannot report distributed collective or network service as
measured work.

This separation is deliberate. Model and framework support lives in target
adapters, while request identity, synthetic input, content closure and
architecture checks stay common. A target that cannot prove the requested
rank-local mapping rejects the request instead of pretending that one GPU ran
the distributed configuration.

## Chronology and evidence

The expectations-only commit is
`808125133ebfcd930b28a0b4f962ecad111e6d3f`. The machine-readable result is
[`result.json`](result.json). Its only scored evidence class is nonphysical
contract conformance. The physical-sanity check is not applicable because no
fixture duration is interpreted as hardware time.
