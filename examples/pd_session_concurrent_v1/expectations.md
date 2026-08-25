# Concurrent disaggregated serving session expectations

This is the expectations-only freeze for VLLM-35. It predates the concurrent
session implementation, its harness, every generated result and every scored
run. The accepted one-request CORE-51 study remains a byte-locked baseline and
is not edited by this study.

## Question

Can one disaggregated session keep several requests in flight across real vLLM
prefill and decode schedulers, while those schedulers remain the only batching
authorities, each producer completion releases only its matching consumer, and
the resulting run emits the load-versus-delay curve mechanics needed by
CORE-54?

## Frozen runtime and source identity

The frontend is vLLM 0.27.1 on Python 3.10.18. The model is the cached
`ibm-granite/granite-3.0-1b-a400m-instruct` revision
`ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`. Network access stays disabled.
The JSON registry freezes the model configuration, prompt fixture, adapter,
core, scheduler, engine and connector source hashes.

The run is valid only with `HF_HUB_OFFLINE=1`,
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, and `PYTHONPATH` naming the selected clean
worktree. Machine-specific paths are command-line inputs or local environment
variables and never enter tracked records.

## Byte-locked CORE-51 baseline

All six tracked files under `examples/pd_session_v1` are frozen by SHA-256 in
the JSON registry. The concurrent implementation adds a new bulk API. It does
not change the one-request study, its accepted timestamps, its compact result,
its report, or its line-ending contract. A digest disagreement is fatal and
voids this study.

The accepted prompt-8, 100 microsecond constant cell remains the live
one-request timing control: 95,424,000 ps of prefill service, 100,000,000 ps
of handoff, 77,952,000 ps of first-decode service, 273,376,000 ps TTFT and
77,952,000 ps TPOT. It is an exact baseline control, not a scored curve point.

## Frozen sweep

Each engine has tensor-parallel width eight and a stock maximum sequence count
of eight. The pool shapes are one prefill plus one decode, one prefill plus two
decode, and two prefill plus one decode. Every shape runs both prompt lengths,
8 and 16 tokens, at offered loads of 8, 16 and 32 requests per second. These
loads correspond exactly to interarrival intervals of 125,000,000,000,
62,500,000,000 and 31,250,000,000 ps.

Each of the 18 cells admits eight requests. Every request asks for four decode
tokens and uses the accepted 100,000,000 ps declared handoff. Request IDs are
stable session IDs. vLLM may assign different internal identities in the two
pools, but neither identity may be reused or confused with another request.

Requests that arrive while an engine step is in service retain their original
admission timestamp and become scheduler-visible at the next driver boundary.
The driver may choose which pool engine to step next, but it may not assemble,
split, reorder or price a scheduler batch. Only each stock vLLM scheduler may
do that.

## Independent completion contract

One producer completion schedules one immutable KV handoff at that producer's
completion timestamp. Scheduling several handoffs at the same timestamp does
not advance the shared clock once per request. Every handoff completes at its
own submitted timestamp plus the declared duration. Its matching decode request
becomes eligible then, even if another producer or consumer remains active.

The driver drains neither an individual request nor a whole producer pool
before it can expose ready consumers. A request may wait for a scheduler grant,
but it may not wait merely because the driver is finishing an unrelated
request.

## Exact conservation

For each cell, the admission ledger, handoff ledger and terminal ledger have
the same eight stable request IDs exactly once. Each stable ID maps to one
distinct prefill-local identity and one distinct decode-local identity. Every
terminal request carries exactly four generated token IDs in stable prefix
order. The cell therefore conserves 8 admissions, 8 handoffs, 8 terminals and
32 terminal decode tokens.

Every request's TTFT is its first decode-token completion minus admission. Its
decomposition is prefill queue, prefill service, complete handoff duration,
decode admission wait and first decode-token service. The residual must be
zero picoseconds for every request.

## Genuine batching

At the highest offered load, every pool ratio must expose at least one prefill
`StepRecord` and one decode `StepRecord` whose scheduled set contains two or
more distinct requests. The scheduled IDs must come from vLLM's scheduler
output and must match the stable requests through the recorded pool-local
identity map. A hand-built driver batch does not satisfy this guard.

## Curve record

Every prompt and pool-ratio configuration emits one
`simllm-deployment-curve-v1` record containing the three offered-load points.
Each point stores exact numerator and denominator pairs for:

- aggregated output throughput, all terminal output tokens divided by the
  interval from first admission to last terminal completion;
- per-token request delay, the arithmetic mean across requests of terminal
  completion minus admission divided by that request's four output tokens.

The plot orientation is throughput rightward and inverse per-token request
delay upward. The raw delay remains in the record so an integrator can choose
the axis transform without reconstructing request timelines.

Within each of the six curves, aggregated output throughput is nondecreasing
from the lowest to highest offered load. Per-token request delay is also
nondecreasing. The curve may be stepped or flat; strict curvature and a
calibrated saturation knee are not claimed by VLLM-35.

## Prompt relation and physical sanity

Granite carries 49,152 KV bytes per original prompt token. Doubling the prompt
from 8 to 16 tokens therefore moves every handoff from 393,216 to 786,432
bytes exactly. It must not change the requested terminal token count.

Before reading the modeled steps, every nonempty step is bounded between
1,000,000 and 100,000,000,000 ps. The client-visible decode cadence must stay
between 10 and 100,000 tokens per second. The result report must also compare
the observed compute terms with the resident-weight read floor, the handoff
with bytes over link rate, and the composed curve with a plausible serving
rate. These are three separate sanity angles and are not scored relations.

## Fatal guards and evidence accounting

A source, runtime, baseline, role, clock, identity, conservation, timestamp,
decomposition, batching or physical-bound failure voids the study. Fatal
guards are never reported as a fraction. Curve points, behavioral relation
families and exact conservation remain separate evidence classes.

If the run is nonvoid but a scored direction fails, VLLM-35 stays open and the
result is reported as a refutation. If all literal acceptance clauses hold,
VLLM-35 closes. Any remaining adapter-specific gap is registered as VLLM-39;
any remaining framework-neutral session gap is registered as CORE-55.

## Scope

A valid result establishes concurrent requests in the current vLLM session,
real scheduler batches in both roles, exact request conservation and reusable
curve records. It does not establish the 56-engine target, SGLang, DeepSeek
compute prices, calibrated saturation behavior, packetized KV traffic or the
public flagship anchors.
