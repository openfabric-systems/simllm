# Disaggregated serving session v1 expectations

This is the expectations-only record for the first slice of CORE-51, TRAF-61,
and PLACE-4. It freezes one prefill engine plus one decode engine, with eight
simulated GPUs in each engine. No target implementation or scored run existed
when this file was committed.

## Chronology

The source state is repository commit
`d3f909bc2b7e69b3e6127a13ea195e50b3c0f4ea`. Immediately before these freeze
files were authored, `git status --porcelain=v1` printed no tracked or
untracked path. The required local sizing note was already present under the
gitignored working layer. No CORE-51 behavior, scored output, or observed
number informed the relations below.

The result report must name the final expectations-only commit. A later source
audit that changes a frozen hash is a fatal guard, not permission to update the
expected relation after seeing a run.

## Seam audit and decision

The pinned vLLM v0.27.1 surface supports a real scheduler-side seam under
`SimExecutor`:

- `vllm/config/kv_transfer.py` declares a connector and a producer, consumer,
  or both role per engine instance.
- `vllm/v1/core/sched/scheduler.py` constructs the scheduler connector, asks it
  how many remote prompt tokens exist, hands its metadata to the executor, and
  calls `request_finished` before releasing producer blocks.
- `vllm/outputs.py` carries the producer's `kv_transfer_params` to the driver.
  `vllm/v1/request.py` reconstructs the same field from the decode request's
  `SamplingParams.extra_args`.
- `SimExecutor` already accepts scheduler output that contains opaque
  connector metadata. Its simulated workers do not allocate a paged KV tensor
  and return no worker-side connector output.

The decision is therefore to use the real vLLM connector and request surfaces
for scheduler control, not a driver-only join. The connector is an explicit
simulation connector: producer completion returns the remote prompt coverage,
the decode scheduler admits exactly that external coverage, and the driver
advances the sole shared virtual clock through one SimLLM KV-handoff event.
There is no claim that a worker connector copied tensor bytes. This is the
declared-constant bypass arm of TRAF-61.

The adapter audit is frozen by SHA-256 in `expectations.json`. The audited
files are `simllm/adapters/vllm/executor.py`, `vllm/config/kv_transfer.py`,
`vllm/distributed/kv_transfer/kv_connector/v1/base.py`,
`vllm/v1/core/sched/scheduler.py`, `vllm/v1/engine/core.py`, and
`vllm/outputs.py`.

Each pool's vLLM scheduler remains its only admission and batching authority.
The session driver chooses a pool instance, submits the request, observes the
producer completion, applies the handoff event, and submits the augmented
request to decode. It does not construct a second batch or advance a request
inside either scheduler.

## Deployment and placement

One serving node is one in-process vLLM engine with tensor parallel width
eight. The first slice constructs one prefill engine and one decode engine,
for 16 simulated ranks. Both executors must hold the identical
`VirtualClock` object. `reset_configuration()` is called before each engine's
per-pool hooks are installed, including the boundary between the two roles.

The same placement builder must also render, without constructing engines, the
target of 16 prefill nodes followed by 40 decode nodes. The exact structural
oracles are:

- 448 ranks total, ranks 0 through 127 prefill and 128 through 447 decode;
- eight GPUs and eight one-to-one GPU-affine NICs per node;
- each tensor-parallel group is the eight consecutive ranks of one node;
- each data-parallel group contains equal local ranks from nodes of the same
  pool only, never a prefill and decode rank together;
- every rank carries its pool role, simulated GPU identity, node identity, NIC
  identity, and physical fabric location;
- GPU-rank GOAL mapping remains identity for all 448 ranks.

The concrete placement and fabric records use
`simllm-placement-manifest-v1` and `simllm-fabric-topology-v1`. This slice
does not implement PLACE-1's general inventory discovery or PLACE-2's
general unique-NIC mapping.

## Timing and pricing contract

The physical sequence is simple. The prefill engine reads the prompt and
model weights, performs its node-local collectives, and emits one bootstrap
token so vLLM can finish the producer request. The producer's KV coverage is
the original prompt. The decode engine receives the original prompt plus that
bootstrap token, loads the original prompt coverage through its real
connector seam, computes the remaining bootstrap position, and emits the
first client-visible decode token. Later tokens come only from the decode
pool.

Per-step compute uses the existing provider selected by `SimExecutor` and the
step sink. The default is the existing roofline bootstrap. A caller may pass
the existing lookup-backed provider when the calibration campaign fills it.
No price is fitted here. Tensor-parallel work stays on the existing declared
node-local collective arm. No packet backend should run for those all-local
groups.

For request `r`, with original admission `A`, prefill eligibility `PE`,
prefill completion `PF`, handoff completion `H`, first decode eligibility
`DE`, and first decode-token completion `D1`, the exact registered form is:

```text
prefill_queue = PE - A
prefill_service = PF - PE
handoff = H - PF
decode_admission_wait = DE - H
decode_first_token_service = D1 - DE

TTFT = D1 - A
     = prefill_queue + prefill_service + handoff
       + decode_admission_wait + decode_first_token_service
```

Time to first token (TTFT) is therefore admission through the whole chain to
the first client-visible token produced by the decode pool. Time per output
token (TPOT) is the exact rational mean of consecutive decode-pool token
completion deltas. The prefill bootstrap token is not a client-visible token
and never enters TPOT.

## Frozen sweep

The cached Granite revision is
`ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`. The tracked prompt fixture has
SHA-256
`36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341`.
The four behavior cells are the Cartesian product:

| Parameter | Values |
|---|---|
| Original prompt tokens | 8, 16 |
| Declared KV handoff | 100,000,000 ps, 200,000,000 ps |
| Client-visible decode tokens | 4 in every cell |

The connector sees 8 or 16 remote tokens. The decode prompt contains one
additional bootstrap token. Prefix caching, chunked prefill, asynchronous
scheduling, speculative decoding, and packet-rendered KV traffic are off.

## Exact and behavioral relations

The four per-cell TTFT decompositions are exact-oracle rows and are reported
separately from scored behavior.

Two genuine-risk behavioral families contain six parameterized instances:

1. Handoff movement, two instances: at each prompt length, moving from the
   100,000,000 ps arm to the 200,000,000 ps arm adds exactly 100,000,000 ps
   to every TTFT. It changes prefill service, decode admission wait, decode
   service, token values, token order, and TPOT by exactly zero.
2. Prompt movement, four instances: at each handoff arm, 16 prompt tokens
   produce exactly twice the KV bytes of 8 prompt tokens, strictly increase
   prefill service and TTFT, and do not decrease TPOT. The service and TTFT
   comparisons are separate instances, so a missed strict relation cannot be
   hidden by the byte identity.

The 448-rank placement relation is an unscored structural invariant. Native
engine construction, exact decomposition, and scale measurements remain
separate evidence classes and are never added to the behavioral denominator.

## Physical bounds before modeled values

Granite has 24 layers, 8 key-value heads, a 64-element head, and two-byte KV
elements. Keys plus values therefore occupy exactly 49,152 bytes per original
prompt token across the full model. The 8-token and 16-token handoffs carry
393,216 and 786,432 bytes.

The physical floor is bytes over one 400 Gbit/s link: 7,864,320 ps for 8
tokens and 15,728,640 ps for 16 tokens. No modeled handoff may be below it.
The frozen bypass ceiling assumes a deliberately slow 10 Gbit/s effective
path plus 50,000,000 ps fixed overhead: 364,572,800 ps and 679,145,600 ps.
Both selected constants sit inside both intervals. This ceiling is an
acceptance envelope for the constant surrogate, not a hardware calibration.

Every nonempty engine step must lie between 1,000,000 ps and
100,000,000,000 ps. The lower edge is intentionally below the weight-read and
kernel-launch floors of the 400M-active-parameter Granite model. The upper
edge permits a path more than an order of magnitude slower than reading the
resident weights at a conservative accelerator-memory rate. The report must
also state the measured decode rate and reject a result outside 10 through
100,000 client-visible tokens per second. Being inside these broad bounds is
not proof of calibration.

## Engine-count feasibility measurement

Fresh child processes construct 1 prefill plus 1 decode engine and 2 prefill
plus 2 decode engines. Each child retains every engine. After each
construction it records role, simulated worker count, elapsed wall time,
current resident set size, and monotonic peak resident set size. The study
reports total and per-added-engine memory and wall time at both points.

The 56-engine extrapolation is descriptive. It uses the observed range of
per-engine increments and states the assumptions that model metadata, KV
block count, tensor-parallel width, allocator behavior, and process topology
remain unchanged. It must not say the target fits unless a 56-engine run was
actually measured.

## Fatal guards

The run is void if any of these fails:

- a pinned vLLM version, model, fixture, or audited source hash disagrees;
- an engine's pool role and KV producer or consumer role disagree;
- any constructed executor does not share the one clock object;
- a request identity is lost, one request emits other than one handoff, or a
  timestamp is nonmonotonic;
- any TTFT decomposition has a nonzero residual;
- any handoff constant lies outside its frozen physical interval;
- an all-local collective or the KV bypass invokes a packet backend;
- the one-plus-one or target placement violates a structural oracle;
- an engine-count cell is missing, has the wrong role or worker count, or its
  retained-engine peak memory is nonmonotonic.

One fatal failure voids the run. The evidence is retained and CORE-51,
TRAF-61, and PLACE-4 stay open. Fatal guards are never reported as a pass
fraction.

## Scope of a valid result

A valid run demonstrates the live one-plus-one vLLM session, exact constant
handoff movement, node-local pricing reachability, and structural rendering of
the 56-node target. It does not demonstrate 56 live engines, packetized KV
traffic, calibrated KV transfer, lookup-record compute pricing, SGLang, or a
physical 448-GPU deployment.
