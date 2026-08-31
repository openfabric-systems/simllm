# CORE-66 EP8 capture expectations

## Frozen cell

The single feasible cell uses two GH200 nodes with four GPUs per node, eight
ranks, expert parallel width eight and four routed experts resident per rank.
The reduced model has 32 routed experts, three dense layers and one mixture of
experts layer. Each rank runs batch 32 at key-value cache length 2,000, with
multi-token prediction disabled, dummy weights, data-parallel attention, the
data-parallel language-model head and DeepEP. One decode iteration is measured.
The source is pinned at
`bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3`. Network fetches, model-weight
downloads and backend fallback are forbidden.

CUDA graph replay is disabled because graph replay does not re-enter the
per-layer Python wrappers that record routing identities and semantic launch
ranges. Eager launch raises host overhead. Kernel and DeepEP service remain
deterministic across launch modes, so those services are identifiable, but the
raw eager step duration is not a registered graph-mode step measurement.

The scheduler request is frozen to cluster `gmerlin7`, partition `gh-hourly`,
QoS `gpu_hourly`, account `merlin`, two nodes and eight GPUs total. The cell is
submitted once. A scheduler refusal ends the attempt and is published exactly.

## Declared deviation ledger

The EP8 capture identifies physical launches and service mechanisms. It is not
an EP72 service measurement.

- Eight rather than 72 expert-parallel peers biases measured dispatch and
  combine service downward.
- Thirty-two rather than 256 unique routed experts increases the local share
  under uniform routing and biases remote traffic downward. The effect on
  grouped-kernel occupancy is indeterminate.
- Thirty-two unique physical slots omit the registered 288-slot population,
  including the three-plus-one-redundant cohort. The effects on locality,
  duplicate residency and grouped kernels are indeterminate.
- Four rather than 61 transformer layers lowers raw step service by
  construction. Only separately identified per-layer services may enter the
  frozen multiplier ledger.
- Two rather than nine nodes lowers participant count and fabric contention,
  biasing aggregate communication service downward.
- Four rather than eight GPUs per node raises the fraction of cross-node peers.
  That can raise service per byte even while fewer peers and nodes lower
  aggregate communication service.
- Four routed expert slots per rank match the registered residency, so no
  per-rank routed-weight residency bias is expected.
- Disabling CUDA graph replay raises host launch overhead. It does not change
  deterministic kernel or DeepEP service, but it prevents promotion of the raw
  eager step time to registered graph-mode service.
- Any fallback from DeepEP invalidates communication pricing and forces null
  signed movement.
- Dummy weights preserve tensor shapes and expected byte demand, but their
  routed IDs are not production routing statistics.

## Evidence and comparison gate

The capture must bind the 37 semantic rows left physically unbound by CORE-65,
resolve the attention, mixture of experts and data-parallel language-model-head
paths, measure DeepEP dispatch and combine with rank, peers, payload and
duration, preserve rank identity for per-kernel and per-step high-bandwidth
memory reads and writes, and record each layer's routed expert IDs, assignment
counts and local physical slots.

The `1/64` count-and-resident-weight candidate and `1/9` assignment candidate
are checked from those physical records. No direction is selected for the byte
candidate before the counter pass. More high-bandwidth memory bytes than the
candidate increases modeled service and moves throughput downward; fewer bytes
does the opposite. Adding the DeepEP work absent from the earlier vLLM capture
increases modeled service and moves throughput downward.

Signed movement is published only when both DeepEP dispatch/combine service and
rank-preserving high-bandwidth memory bytes are available. If either is
unavailable, movement remains null. The downward DeepEP correction is never
published alone. The frozen multipliers are common `61/4`, dense `1`, mixture
of experts `58`, step `1` and output `1`; no constant is fitted.

## Guard and disclosure

A held-out value entering arithmetic, a prediction comparison, fitting or a
published reproduction is fatal and not survivable. Such a run is void and
CORE-66 remains open.

Incidental exposure without use is survivable. It is logged with what was seen
and where. The physical identities, measured services, routing checks and
zero-parameter signed derivation remain interpretable because the arithmetic
has no free or fitted parameter and is independently checkable from the
published record. The exposure degrades disclosure quality, not the result.
Broad repository searches, documentation sweeps and unguarded protected-record
reads remain forbidden because avoiding them is cheap.

Pytest, ruff, documentation and task-progress checkers, and git plumbing are
automated-process exemptions and do not count as protected-record access.

## Physical sanity

Before inspecting a measured duration, the memory floor is the measured read
plus write bytes divided by the GH200 peak high-bandwidth memory rate. The
communication floor is payload bytes divided by the applicable directed link
rate. A constituent launch cannot exceed the decode-step wall time that
contains it. Review uses three independent views: kernels against memory bytes,
DeepEP against payload and peer locality, and end-to-end decode service against
the standard-decode anchor and frozen layer composition.
