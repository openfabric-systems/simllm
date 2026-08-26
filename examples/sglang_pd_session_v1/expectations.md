# SGLang disaggregated serving session expectations

This is the expectations-only freeze for SGL-33. It predates the SGLang
session implementation, process-isolated pool driver, arrangement projection,
study harness, generated results and every scored run. The accepted vLLM
session, SGLang worker records and all prior studies remain read-only
baselines.

## Question

Can separate stock SGLang schedulers serve the prefill and decode roles over
simulated GPUs under the framework-neutral CORE-51 session contract, with one
stable request identity, both accepted key-value cache handoff pricing arms,
exact time to first token and time per output token, and structural
data-parallel-attention and expert-parallel arrangements suitable for the
CORE-54 deployment curve?

## Frozen source and runtime identity

The frontend is SGLang `0.5.19.dev345+gbfeae4e79` at commit
`bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3` on Python 3.10.18. The model is
the cached `ibm-granite/granite-3.0-1b-a400m-instruct` revision
`ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`. The run uses CPU, float32,
offline model resolution and the simulated worker. Network access and model
downloads are forbidden.

The JSON registry freezes the source hashes that define the worker, pump,
CORE-51 reducer, constant and packet handoffs, existing role-aware placement,
the native SGLang disaggregation seam, the model configuration and the prompt
fixture. Machine-specific paths are inputs and never enter tracked records.
Every executed path renders with POSIX separators.

## Seam audit and disclosed fallback

The seam audit ran before this freeze against the pinned source and runtime.
SGLang exposes native `prefill` and `decode` disaggregation modes plus
Mooncake, NIXL, MORI, Ascend and fake transfer backends. That native pair is
not reachable with the current bufferless simulated worker and no GPU:

- the argument resolver rejects `disaggregation_mode="prefill"` with the
  fake backend before scheduler construction;
- the decode scheduler reaches native disaggregation initialization, then
  asks the bufferless model runner for registered KV-cache metadata beginning
  with `kv_cache_dtype_str`; the runner intentionally owns no transferable KV
  tensors or registrations.

The study therefore freezes the driver-level join allowed by SGL-33 and
CORE-51. One producer completion schedules one existing CORE-51 handoff. Its
matching consumer becomes eligible only after that event completes. The join
does not fabricate native connector success, and every result identifies the
fallback. Native-seam reachability remains separate residual work.

## One timing authority and process isolation

Pinned SGLang stores model-parallel groups in process-global state. Retaining a
second scheduler in the same process fails because the tensor-parallel group is
already initialized. Each pool engine therefore retains one stock SGLang
scheduler in its own child process. The parent owns the sole session
`VirtualClock`. Before a selected engine step, the parent supplies its current
timestamp; the child runs exactly one stock-scheduler step and returns the
immutable `StepRecord`, completion rows and finishing timestamp. The parent
then advances to that timestamp. Child clocks are step-local executors of the
parent grant, never independent session clocks.

The driver may choose which engine to step and which already-arrived requests
to submit. It may not assemble, split, reorder or price a framework batch.
Every batch recorded by the study must come directly from a stock SGLang
scheduler `StepRecord`.

## Frozen concurrent sweep

The pool ratios are one prefill plus one decode, one prefill plus two decode,
and two prefill plus one decode. Every SGLang scheduler executes at CPU
`tp_size=1`. Each simulated pool engine represents one eight-GPU node only in
the placement and traffic projections. No real tensor operation is implied by
that structural width.

Each ratio crosses prompt lengths 8 and 16 at offered loads 8,000, 16,000 and
32,000 requests per second, exact interarrival intervals of 125,000,000,
62,500,000 and 31,250,000 ps. Each of the 18 cells admits eight requests. Each
request asks for four decode tokens and uses the 100,000,000 ps declared
handoff. Prompt prefixes are distinct across cells so retained SGLang radix
state does not create cross-cell prefix hits.

Every role's structural arrangement enables data-parallel attention. Its
attention data-parallel, dense data-parallel and expert-parallel sizes equal
the number of simulated GPUs in that role for the cell. The placement carries
`attn_dp`, `dense_dp` and `ep` groups of those exact sizes. The actual stock
scheduler remains width one and performs no distributed tensor operation.

At the highest offered load, every pool ratio must expose at least one genuine
multi-request prefill batch and one genuine multi-request decode batch.

## Exact request and timing conservation

Within every concurrent cell, the admission, handoff and terminal ledgers
carry the same eight stable request identities exactly once. Every stable
identity maps to one distinct prefill-local identity and one distinct
decode-local identity. Every terminal carries four generated token IDs in
stable prefix order, so each cell conserves 8 admissions, 8 handoffs, 8
terminals and 32 decode tokens.

For every request, time to first token is first decode-token completion minus
admission. Its exact additive decomposition is prefill queue, prefill service,
complete handoff duration, decode scheduler wait and first decode-token
service. The residual must be zero picoseconds. Time per output token is the
exact fraction from the first through last decode-token completion.

## Handoff sensitivity and packet arm

Two isolated one-request controls use the same 8-token shape and differ only
in the declared handoff, 100,000,000 versus 200,000,000 ps. The larger constant
must add exactly 100,000,000 ps to time to first token and zero to time per
output token. Every other decomposition term must be identical.

A third isolated one-request control uses the existing TRAF-62 packet policy
at 400 Gbit/s with its 20,000,000 ps PCIe submission term. Its source ranks,
destination ranks and eight chunks come from the one-plus-one SGLang placement,
not a separate rank formula. Aggregate KV bytes are 393,216. The exact signed
relation is:

`packet TTFT - constant TTFT = packet handoff duration - 100,000,000 ps`.

The residual must be zero picoseconds and decode TPOT must be identical. The
packet policy is the only arm allowed to emit backend artifacts. The constant
arm emits none.

## Machine-readable deployment curves

Each prompt and pool-ratio configuration emits one
`simllm-deployment-curve-v1` record with three
`simllm-deployment-curve-point-v1` points. Exact numerator and denominator
pairs carry aggregate output throughput and mean per-token request delay using
the same definitions, field names and orientation as the VLLM-35 study.
Throughput is expected to be nondecreasing over the three offered loads. The
delay direction is reported but not used to close SGL-33 because the current
roofline bootstrap has no calibrated contention surface. A residual task owns
that calibration before CORE-54 treats the curve shape as physical evidence.

## Structural flagship render and arithmetic finding

The arrangement projection renders the two disclosed SGLang role units
literally: four eight-GPU prefill nodes with attention DP32, dense DP32 and
EP32, plus nine eight-GPU decode nodes with attention DP72 and EP72. Combined,
those role units contain 13 nodes and 104 ranks. This is a structural
juxtaposition, not a claim that the public 12-node cluster retained all 13
units simultaneously. CORE-54 currently calls the same four-plus-nine shape a
96-GPU deployment. The arithmetic disagreement is fatal to a flagship
allocation claim and must be registered without changing this study's
structural acceptance.

## Physical sanity

Before reading results, every nonempty modeled step must lie between 1,000,000
and 100,000,000,000 ps, and client-visible decode cadence must lie between 10
and 100,000 tokens per second. The result also checks three independent
angles: resident weight bytes over the B100 HBM envelope, KV bytes over link
rate, and end-to-end request cadence against the scale of a 400M-active-
parameter model.

At 400 Gbit/s, 393,216 aggregate bytes cannot serialize faster than 7,864,320
ps on one link. Each 49,152-byte rank-local shard cannot serialize faster than
983,040 ps. Packet service must sit above the shard floor and below the broad
81,457,280 ps upper sanity bound inherited from the TRAF-62 bounded cell.

## Fatal guards and evidence accounting

A source, runtime, baseline byte, role, process-isolation, parent-clock,
identity, conservation, timestamp, decomposition, arrangement, endpoint,
chunk, packet ordering, physical-bound or metric disagreement voids the run.
Fatal guards are never scored. Exact metric rows, behavioral relation families
and structural projections remain separate evidence classes.

SGL-33 closes only if every fatal guard holds and every frozen exact or
behavioral relation holds. The allowed driver fallback does not block closure
when it is disclosed exactly. Native-seam reachability, physical curve-shape
calibration and the CORE-54 allocation arithmetic use only the reserved
residual IDs SGL-35, SGL-36 and CORE-57.

## Scope

A valid result establishes the SGLang-side disaggregated session over cached
Granite on CPU, process-isolated stock scheduler pools, exact CORE-51 timing,
TRAF-61 and TRAF-62 handoff reuse, structural SGLang arrangements and reusable
curve records. It does not run real distributed tensor operations, calibrate
SGLang batching, validate the published DeepSeek anchors, run the physical
flagship allocation, close TRAF-61 or TRAF-62, or make the native KV connector
reachable.
