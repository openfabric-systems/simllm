# Live target engine-scale qualification

This prospective execution protocol owns CORE-52. It measures the existing
session at 16 prefill and 40 decode engines, with 448 simultaneously retained
simulated workers. Its first run follows the final expectations-only commit.
The session implementation and accepted small-session results already exist;
the exact historical numbers are explicitly known regression oracles.

## Physical mechanism and scope

An engine holds a native serving frontend, its scheduler, an executor and eight
simulated worker objects. Separate prefill engines produce the prompt's key
and value cache. A declared transfer makes it available to a separate decode
engine, which emits four tokens. There are no model weights, GPU kernels or
packet-backend runs in this host feasibility experiment.

The existing concurrent driver finishes one engine step before choosing the
next and advances one shared clock through that complete service. This is a
**globally serialized compatibility model**. Its measured request timing can
prove routing and conservation. It cannot predict the speedup from independent
GPUs working at the same time. CORE-68 owns independent-engine completion
scheduling; CORE-51 and CORE-54 retain that dependency.

Each live sink uses its existing local eight-rank placement. The study joins
actual worker identity to the declared global manifest through role, engine
ordinal and local rank. This is a read-only projection. It does not make the
448-rank physical fabric a live timing authority.

## Frozen sweep and retention proof

Run fresh native processes at 1+1, 2+5, 4+10, 8+20 and 16+40 prefill/decode
engines. Retain all engines in a session until every request finishes. Observe
current resident memory, construction time and clock state after every added
engine. Keep the peak-memory high-water mark separate. Neither memory nor
host throughput is assumed to scale linearly or monotonically.

The construction observer keeps only scalar IDs and weak references. It must
not keep an engine or worker alive on the session's behalf. Re-enumerate the
session-owned pools before and after requests. Join each executor to the
actual frontend core and obtain worker identity through the frontend callable
collective remote procedure call. The target requires 448 distinct actual
worker objects with the expected ranks and configuration identities; adding
56 declared world sizes is insufficient. All engines share the same clock,
and constructing them advances it by exactly zero.

If resident memory reaches the frozen cap or a process exceeds its timeout,
retain every completed construction row and the exact stopping point. A target
that cannot retain all 56 engines leaves CORE-52 open. Do not extrapolate a
pass from smaller points.

## Request execution and metric chain

At each scale, run 80 serial requests for each of four cells: prompt lengths
8 and 16 crossed with declared handoff durations 100 and 200 microseconds.
Eighty is divisible by every pool size, so every cell routes requests through
every engine and returns both round-robin cursors to their initial positions.
Then admit one simultaneous burst of 80 requests with prompt length 16 and a
100-microsecond handoff. Every declared engine must also serve that burst.

A simultaneous burst has no finite offered request rate. Report its completed
requests and tokens divided by its actual virtual makespan, and label the
arrival mode. Do not supply an invented offered rate to a deployment-curve
constructor. Host request throughput uses measured execution wall time and
remains a separate descriptive metric.

Retain complete request results, unique executor records and step results,
actual scheduler memberships and clock-advance observations. Join a request's
stable ID to its two native internal IDs and the steps that really scheduled
it. Concurrent per-request record slices overlap; never sum those slices.
Count service once by `(engine_id, step_index)` and reconstruct the union of
its intervals plus separately justified wall-idle gaps. Keep additive service
work, per-request causal waits and global wall-idle under distinct names.

The one-plus-one process first repeats the original four requests under their
original labels and order. Its complete comparison hashes must equal the
accepted CORE-58 hashes, excluding only the two declared opaque root IDs.
The original one-plus-one and target placement bytes also remain exact when
physical rendering is disabled. All historical study files remain byte-locked.

## Bounds written before execution

Cache storage is 49,152 bytes per prompt token, hence 393,216 or 786,432 bytes.
At 400 Gbit/s the transfer floors are 7,864,320 and 15,728,640 picoseconds.
A declared slower route at 10 Gbit/s plus 50 microseconds gives ceilings of
364,572,800 and 679,145,600 picoseconds. Both frozen handoff durations fit.
These declared bounds do not constitute a network measurement.

The active per-rank weight count is 320,864,256 bytes. Dividing by the B100
profile's 8 TB/s memory ceiling gives a 40,108,032-picosecond decode-service
floor. The existing accepted decode values exceed it. A serial request cannot
finish before prefill, handoff and all four positive decode services. A loose
ceiling is five frozen per-step ceilings plus handoff. For a simultaneous
burst, the compatibility model's unique service sum is a makespan floor, and
that sum plus all handoff durations is a ceiling. These are explicitly bounds
on this serialized model, not an upper bound for independent GPUs.

## Acceptance and evidence classes

The 20 serial cells reproduce the accepted timing values for every one of
their 80 requests. Those historical comparisons and the four complete baseline
hashes are exact-oracle evidence, separate from scoring. Ten handoff-response
instances require an added 100 microseconds to move each request's time to
first token and job completion by exactly that amount, leave time per output
token unchanged, and add 8 milliseconds to the serial cell. Ten prompt-response
instances require twice the cache bytes, strictly greater prefill time, first
token time and completion time, and nondecreasing time per output token. Cache-byte scaling, fixed-service
equalities and the time-per-output-token comparison are separate unscored
guards. One joint moved-outcome relation is one behavioral instance; additional
metrics or matched requests do not increase the denominator.

All identities, stages, roles, source joins, conservation checks, inactive paths
and corruption controls are fatal and unscored. Their finite domains are in
`expectations.json`. Missing evidence is fatal, even if the reported totals or
verdict look correct. A single failed guard or relation makes the whole run
VOID with a null behavioral score. Preserve partial evidence and the traceback.

A complete valid result with independent review and software gates closes
CORE-52's live host-scale qualification. CORE-68 retains physical cross-engine
concurrency, CORE-53 retains accepted lookup pricing, and CORE-51/CORE-54 retain
the end-to-end deployment obligations. No GPU calibration or large-model
frontier is qualified by this campaign.
