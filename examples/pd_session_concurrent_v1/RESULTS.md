# Concurrent disaggregated serving session result

The concurrent session mechanism is live, but the frozen curve study is
REFUTED. The amended run held every fatal guard, conserved all 144 admissions,
144 handoffs, 144 terminals and 576 decode tokens across 18 cells, and produced
a maximum TTFT decomposition residual of 0 ps. Every pool ratio exposed a real
multi-request stock-scheduler batch in both roles. Per-token request delay did
not follow the frozen nondecreasing direction in any of the six curves, so
VLLM-35 stays open and VLLM-39 owns identification of the load-delay shape.

## What ran

The offline vLLM 0.27.1 runtime drove the cached Granite checkpoint through
one-prefill/one-decode, one-prefill/two-decode and two-prefill/one-decode pool
ratios. Each ratio crossed 8-token and 16-token prompts with eight concurrent
requests at 8,000, 16,000 and 32,000 offered requests per second. Each request
produced four decode tokens and used the accepted 100,000,000 ps declared KV
handoff. The driver admitted requests and released completed handoffs; each
stock vLLM scheduler alone selected its engine-local batches.

The original 8, 16 and 32 requests-per-second freeze also ran to completion.
It is retained as a VOID result with SHA-256
`7121ab1b99eeb4809de8e2546351fd03653cd7acf30cee99a0c50155d401d5c5`.
Its three batching guards failed because the highest-load interarrival was
31,250,000,000 ps while prefill service was at most 114,936,000 ps. The
1000-fold unit correction was committed before the amended run, which is
therefore labeled a post-specified regression rather than preregistration.
The amended raw result has SHA-256
`2306cdc7a5700bf6e6a9ca34c07a5a0a8bd3877726f168c8611fe28a993a5b62`.

## What came out

All fatal guards held in the amended run. Every one of the 18 exact rows has 8
admissions, 8 handoffs, 8 terminals, 32 terminal decode tokens and 0 ps TTFT
decomposition residual. Pool-local identities are unique and remain joined to
one stable session identity. The accepted one-request CORE-51 control remains
exact at 273,376,000 ps TTFT and 77,952,000 ps TPOT, and all six tracked
CORE-51 study artifacts retain their frozen SHA-256 values.

The scheduler-batch maxima are:

| Prefill engines | Decode engines | Prefill maximum | Decode maximum |
|---:|---:|---:|---:|
| 1 | 1 | 4 | 8 |
| 1 | 2 | 4 | 4 |
| 2 | 1 | 3 | 8 |

One complete machine-readable curve, for one prefill engine, one decode engine
and an 8-token prompt, is:

| Offered requests/s | Aggregate output tokens/s | Per-token request delay |
|---:|---:|---:|
| 8,000 | 19,665.007 | 212,327,500 ps |
| 16,000 | 26,257.057 | 199,582,000 ps |
| 32,000 | 32,270.817 | 196,310,000 ps |

Throughput is nondecreasing in all six curves. The exact curve records are
usable by CORE-54, but the delay direction is refuted: all six curves either
decrease or dip. Shared batches amortize modeled service enough to outweigh
queue growth on this short eight-request grid. That is a mechanism finding,
not a failed conservation guard, and the frozen direction is not widened after
seeing it.

## Physical sanity

Each rank's 320,864,256 resident weight bytes take at least 40,108,032 ps to
read at the B100 8 TB/s envelope, or 57,297,189 ps after the provider's 0.7
efficiency factor. Observed nonempty steps span 77,952,000 to 317,776,000 ps,
above that floor. The wider values are real batched services rather than a
faster-than-memory result.

The 8-token handoff carries 393,216 bytes. Serializing that aggregate on one
400 Gbit/s link takes at least 7,864,320 ps; the 100,000,000 ps declared term is
12.7 times that floor. This remains a bounded surrogate, not packet
calibration.

The client-visible decode cadence spans 3,146.9 to 12,681.3 tokens per second,
inside the frozen 10 to 100,000 interval and plausible for the modeled
400M-active-parameter checkpoint. Weight movement, KV serialization and
end-to-end cadence are independent checks.

## What it changes for the project

VLLM-35 gains the complete concurrent identity, independent completion,
multi-request batching and deployment-curve mechanism, but remains open on its
frozen curve refutation. VLLM-39 is registered to identify the calibrated
load-delay shape and either validate a corrected direction against deployment
evidence or withdraw the monotonic claim. CORE-54 can consume the exact curve
schema and the recorded points, but cannot present this result as a validated
flagship delay curve.

## What it does not change

This result does not change the one-request CORE-51 artifact or timestamps,
does not calibrate batching or saturation, does not run the 56-engine target,
does not render the KV handoff as packets and does not close VLLM-35. The
original void result stays retained, and no failed fatal guard is converted to
a score.
