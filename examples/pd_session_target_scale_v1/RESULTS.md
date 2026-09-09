# Live target engine-scale result

The live native campaign is **VOID**: its first process reached the frozen
1,200-second host time limit before completing the one-prefill, one-decode
scale. No scale was admitted, and CORE-52 stays open. The result does not
establish a memory limit, independent GPU speed or target deployment capacity.

## What ran

`pd_session_target_scale_v1` attempted the frozen five-scale campaign, starting
with two real vLLM frontends over sixteen simulated workers. Each engine uses
the existing deterministic compute price and declared local collective cost.
The process loads configuration without model weights. Its first four requests
repeat the accepted historical controls, followed by the first planned
80-request serial cell.

The final expectations-only commit is
`416fe5ba0386f2770ccd9fc4d0b1ac09a098ef0c`; execution uses
`c0f529c87c4f573cc580d53f223abe09b815a5aa`. The freeze precedes this harness
implementation and first campaign. Existing small-session values are explicitly
known regression oracles. The timeout, workload and acceptance contract were
not altered after observing progress.

## Stopping point and interpretation

| Retained observation | Value |
|---|---|
| Native process wall time at termination | 1,200.143 seconds |
| Stop reason | Frozen 1,200-second timeout |
| Process exit | Killed, exit code -9 |
| Constructed native engines | Two |
| Distinct observed simulated workers | Sixteen |
| Last construction observation | 12.601 seconds from observation origin |
| Clock advance during construction | Zero |
| Sampled maximum current resident memory | 1,068,044 KiB, approximately 1.02 GiB |
| Completed historical control requests | Four |
| Completed requests in first serial cell | 65 of its required 80 |
| Next partial request | Index 65 has persisted step records |
| Complete admitted cells or scales | None |
| 16-prefill, 40-decode target attempted | No |

The 69 completed progress rows comprise four controls and first-cell request
indices 0 through 64. A partly executed next request is not another completed
request. The partial rows and sampled memory describe this stopping point;
they are not qualified scale results.

The process stops before writing its final native result and receipt. Its final
source-origin, object-retention and GPU-state checks therefore do not run.
Post-run inspection finds the four retained historical comparison bytes equal
to the frozen CORE-58 hashes, but that inspection cannot replace the missing
campaign admission. No measured value is promoted from the partial run.

The full campaign has no behavioral score. Its summary contains the protocol
and manifest stages only, zero admitted exact-oracle rows and zero admitted
behavioral relation rows. There is no meaningful fraction of passing numerical
checks to report. [results.json](results.json) records the stopping observations,
raw artifact identities and the separate software validation status. The raw
summary SHA-256 is
`75a67c30b8d76bea7f06db61c0dfa8942151b38ba863b771f09d73364adb07c4`.

## What changes for the project

CORE-52 requires a host execution investigation before a new target campaign.
Source inspection identifies repeated graph construction, collective-plan
validation, locality projection, dependency-edge lookup and progress-file
synchronization as candidates. It does not measure their contribution to the
timeout. A separate prospectively frozen diagnostic compares uninstrumented
execution with passive phase timing and bounded call profiling, preserving
complete scheduler results and the historical comparisons.

The smallest potential optimization is a call-local dependency-edge index in
the GOAL projection verifier. Its existing search scans all effective edges
for every serialized edge. A measured attribution and new exact-semantics
freeze must precede such an optimization. Neither whole-result caching nor a
shape-only cache is justified: request identity, release time, placement,
dependency scope and mutable sink state all affect valid reuse.

CORE-68 remains the independent-engine timing task. The current driver advances
one shared clock through a whole engine step before selecting another; this
campaign tests host retention and routing under that disclosed serialized
compatibility model. It cannot establish concurrent GPU throughput even if a
later host campaign retains all 448 workers.

CORE-51 and CORE-54 retain whole-deployment and large-model frontier closure.
Accepted single-session behavior and manifest-only target rendering stay
unchanged. No GPU calibration, hardware reservation, model-service constant or
public frontier tolerance changes as a result of this timeout.

## Reproduction boundary

The original source and final freeze identify the retained campaign. Select
an unused output directory and configure the exact native interpreter, vLLM
source and config-only checkpoint cache through local environment values:

```bash
python -m examples.pd_session_target_scale_v1.run_study \
  --native-python "$SIMLLM_VLLM_PYTHON" \
  --vllm-source "$SIMLLM_VLLM_SOURCE" \
  --hf-hub-cache "$SIMLLM_HF_HUB_CACHE" \
  --output-root "$SIMLLM_PD_TARGET_OUTPUT"
```

Keep the original outputs immutable. A successor campaign has a separate
freeze, source identity, output directory and verdict.

The publication software gate at source
`fba9436f1de1a93d016dd9abfdccb3022b1f3e84` passes: 5,844 tests passed,
31 skipped and Ruff clean. This validates the software publication without
rescoring the retained VOID campaign. All six pull-request CI checks are
required before merge.
