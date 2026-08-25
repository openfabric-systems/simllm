# Kernel-cycle lookup retained-fixture result

## Outcome

What ran: `kernel_cycle_lut_v1` analyzed the retained Granite ordered-kernel,
kernel-summary, clock and Nsight Compute excerpts, emitted one candidate
`simllm-kernel-cycle-lut-v1` record, and compiled it through both established
service forms.

What came out: the run is nonvoid. All five kernels seen by both instruments
met the frozen factor-two elapsed-time agreement band; the deciding maximum
ratio was 1.739130. The max-plus-fixed decomposition reconstructed every
per-kernel time and the 2,047,488,000 ps partial step total with a maximum
error of 0 ps. Repeating the analysis produced the identical record digest
`e495f3ca5d0858cf371b19205ae6b7747d633695020d10f58645c5f245086070`.

What it changes: the COMP-64 retained-fixture slice now has a strict unified
record, deterministic analyzer, candidate scalar and device-service
compilers, portable full campaign plan, and code-object double-harvest check.
COMP-64 stays open for the registered GPU campaign. COMP-65 owns static
compile-graph inference and its runtime cross-check, while COMP-66 owns
program-counter attribution on a profiler and target that grant it.

What it does not change: no GPU ran, no calibration or validated service claim
was made, and no serving default moved. The retained component pass did not
measure DRAM bytes, route loads or code-object bytes, so those fields remain
explicit candidate-state nulls. Sixteen graph replays are below the frozen 256
minimum, so the distribution verdict is `insufficient-replays`. The external
probe tree was read only; this workflow made no writes to it.

## Physical checks

The chip cannot complete a kernel in zero time, and an individual kernel from
this cell cannot be longer than its 2,323,678,000 ps enclosing decode step.
The largest retained kernel was 17,984,000 ps, which is positive and 129 times
below that ceiling. The observed streaming-multiprocessor clock was 1.410 GHz
and the observed memory clock was 1.593 GHz, both positive.

The five candidate Nsight Compute to Nsight Systems elapsed ratios were:

| Kernel family | Ratio | Frozen band |
|---|---:|---:|
| FlashAttention combine | 1.739130 | 0.5 to 2.0 |
| FlashAttention split KV | 1.262626 | 0.5 to 2.0 |
| fused MoE | 1.441026 | 0.5 to 2.0 |
| cuBLAS GEMV | 1.375744 | 0.5 to 2.0 |
| MoE top-k gating | 1.355311 | 0.5 to 2.0 |

This comparison checks the fixture join, not numerical calibration. Nsight
Compute replays and instruments a selected launch, while Nsight Systems
records the framework stream with lower detail. Their agreement within the
frozen broad band says the names and units join plausibly. It does not make
either instrument an exact oracle for the other.

## Evidence accounting

The scored denominator is one behavioral relation family with five
parameterized instances, and it passed. Seven fatal guards all held:
source-digest verification, canonical content identity, decomposition
conservation, key rejection, byte determinism, compiler round trips and
physical bounds. None of those guards was added to the behavioral score.

The expectations-only commit is
`10f4ad2b450d5d559cd67d50ccb87e2557e7123d`. Its frozen file SHA-256 is
`bdb910e63fa6315ca9d91c1ff448e7810bc05c29ae8641838f35292fcd39af78`.
