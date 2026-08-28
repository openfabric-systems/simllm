# CORE-66 EP4 capture expectations

## Frozen cell

The fourth feasible cell uses one GH200 node with four GPUs, four ranks, expert
parallel width four and four routed experts resident per rank. The reduced
model has 16 routed experts, three dense layers and one mixture of experts
layer. Each rank runs batch 32 at key-value cache length 2,000, with
multi-token prediction disabled, dummy weights, data-parallel attention, the
data-parallel language-model head and DeepEP. One decode iteration is measured.
The source is pinned at
`bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3`. Network fetches, model-weight
downloads and backend fallback are forbidden.

The scheduler request is cluster `gmerlin7`, partition `gh-hourly`, QoS
`gpu_general`, account `merlin`, one node, four GPUs and 55 minutes. The cell is
submitted once. EP8, EP12 and EP72 are not retried. All earlier refusal records
remain unchanged.

CUDA graph replay is disabled because graph replay does not re-enter the
per-layer Python wrappers that record routing identities and semantic launch
ranges. Eager launch raises host overhead. Kernel and DeepEP service remain
deterministic across launch modes, but raw eager step time is not registered
graph-mode service.

## Declared deviation ledger

The EP4 capture identifies physical launches and the local DeepEP timing
domain. It is not an EP72 service measurement.

- Four rather than 72 expert-parallel peers biases dispatch and combine
  service downward.
- Sixteen rather than 256 unique routed experts increases locality under
  uniform routing and biases remote traffic downward. Grouped-kernel occupancy
  is indeterminate.
- Sixteen unique physical slots omit the registered 288-slot population and
  its three-plus-one-redundant cohort. Duplicate-residency effects are
  indeterminate.
- Four rather than 61 transformer layers lowers raw step service by
  construction. Only separately identified per-layer services may enter the
  frozen multiplier ledger.
- One rather than nine nodes eliminates fabric serialization, switch traversal
  and cross-node contention. Every DeepEP peer is local, so EP4 communication
  is a downward-biased timing domain rather than registered EP72 service.
- Four rather than eight GPUs per node reduces the local participant count.
- Four routed expert slots per rank match the registered residency, so no
  routed-weight residency bias is expected.
- Disabling CUDA graph replay raises host launch overhead but does not change
  deterministic kernel or DeepEP service.
- Any fallback from DeepEP invalidates communication pricing and forces null
  signed movement.
- Dummy weights preserve tensor shapes and byte demand, but their routed IDs
  are not production routing statistics.

## Evidence and comparison gate

The capture must bind the 37 semantic rows left physically unbound by CORE-65,
resolve attention, mixture of experts and the data-parallel language-model
head, measure DeepEP dispatch and combine with rank, peers, payload and
duration, preserve rank identity for per-kernel and per-step high-bandwidth
memory reads and writes, and record each layer's routed expert IDs, assignment
counts and local physical slots.

The `1/64` count-and-resident-weight candidate and `1/9` assignment candidate
are checked from those physical records. No direction is selected for the byte
candidate before the counter pass. More high-bandwidth memory bytes than the
candidate increases modeled service and moves throughput downward; fewer bytes
does the opposite. Adding DeepEP work absent from the earlier vLLM capture
increases modeled service and moves throughput downward.

Signed movement is published only when rank-preserving high-bandwidth memory
bytes and a zero-parameter registered-cell projection of DeepEP service are
both available. Counter denial limits publication to the timing domain and
keeps movement null. A local-only DeepEP duration without a registered-cell
projection also keeps movement null. The downward DeepEP correction is never
published alone. The frozen multipliers are common `61/4`, dense `1`, mixture
of experts `58`, step `1` and output `1`; no constant is fitted.

## Guard and disclosure

A held-out value entering arithmetic, a prediction comparison, fitting or a
published reproduction is fatal and not survivable. Such a run is void and
CORE-66 remains open.

Incidental exposure without use is survivable. It is logged with what was seen
and where. Physical identities, measured services, routing checks and any
zero-parameter signed derivation remain interpretable because the arithmetic
has no free or fitted parameter and is independently checkable. The exposure
degrades disclosure quality, not the result. Broad repository searches,
documentation sweeps and unguarded protected-record reads remain forbidden.

Pytest, ruff, documentation and task-progress checkers, and git plumbing are
automated-process exemptions and do not count as protected-record access.

## Physical sanity

Before inspecting a measured duration, the memory floor is measured read plus
write bytes divided by the GH200 peak high-bandwidth memory rate. The local
communication floor is remote payload bytes divided by the measured directed
GPU-interconnect rate. A constituent launch cannot exceed the decode-step wall
time that contains it. Review uses kernels against memory bytes, DeepEP against
remote payload and local links, and end-to-end decode service against the
standard-decode anchor and frozen layer composition.
