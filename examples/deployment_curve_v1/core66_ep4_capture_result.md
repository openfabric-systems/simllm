# CORE-66 EP4 capture result

## Allocated cell and deviation ledger

The scheduler allocated the frozen EP4 cell on one node with four NVIDIA GH200
GPUs, but no physical capture ran. The allocated topology was four ranks, four
routed experts resident per rank and 16 experts total. Each rank was configured
for batch 32 and key-value cache length 2,000. Multi-token prediction was
disabled. The command requested dummy weights, data-parallel attention, the
data-parallel language-model head, DeepEP, three dense layers, one mixture of
experts layer and one measured decode iteration. SGLang was pinned at
`bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3`.

Job `200879` resolved to cluster `gmerlin7`, partition `gh-hourly`, quality of
service `gpu_general`, node `gpu002` and four allocated GH200 GPUs. It was the
only real submission. It ran for 14 seconds and failed with exit code `127:0`,
consuming 56 GPU-seconds, or about 0.0156 GPU-hours. The deciding capture number
is zero measured decode iterations.

The signed deviation ledger was frozen before submission:

- Four rather than 72 expert-parallel peers biases dispatch and combine
  service downward.
- Sixteen rather than 256 unique routed experts raises uniform-routing
  locality and biases remote traffic downward. Grouped-kernel occupancy is
  indeterminate.
- Sixteen unique slots omit the registered 288-slot population for 256 unique
  experts and its three-plus-one-redundant cohort. Locality and
  duplicate-residency effects are indeterminate.
- Four rather than 61 transformer layers lowers raw step service by
  construction. Only separately identified per-layer services may use the
  frozen multipliers.
- One rather than nine nodes omits fabric serialization, switch traversal and
  cross-node contention, biasing communication service downward.
- Four rather than eight GPUs per node reduces the number of intra-node
  participants.
- The EP4 DeepEP transport domain is entirely local, while the registered cell
  has intra-node and cross-node peers. Local DeepEP service cannot directly
  price EP72.
- Four routed expert slots per rank match the registered residency, so no
  routed-weight residency bias is expected.
- Eager semantic instrumentation raises host launch overhead. Deterministic
  kernel and DeepEP service would remain identifiable, but raw eager step time
  could not become registered graph-mode step service.
- Any fallback from DeepEP would invalidate communication pricing and force
  null signed movement.
- Dummy weights preserve tensor shapes and byte demand but do not provide
  production routing statistics.

None of these differences promotes an EP4 duration to measured EP72 service.

## Launch failure and physical evidence

The batch launcher requested module `cuda/13.2.1`. The compute node reported
`module load: module does not exist -- cuda/13.2.1`, continued to the node
launcher and then reported `nsys: command not found`. The timing pass returned
127 before the SGLang process started. The memory-counter pass was marked
`not-run`.

Consequently, zero of the 37 semantically classified but physically unbound
rows received a kernel binding. The attention, mixture-of-experts and
data-parallel language-model-head physical paths remain unavailable. There are
zero DeepEP dispatch launches, zero DeepEP combine launches, no peer or payload
records and no durations. No routing record exists, so routed expert IDs,
assignment counts and local slot IDs remain unavailable. The `1/64`
count-and-weight and `1/9` assignment candidates remain physically unchecked.

The high-bandwidth memory counter permission was not tested. This is neither a
permission grant nor a permission denial: the missing `nsys` executable stopped
the timing pass before the counter pass. With no kernels, payloads, memory bytes
or decode duration, the frozen memory, communication and end-to-end physical
sanity bounds have no measured value to test.

## Signed movement

The calibration-only movement of the standard-decode anchor remains null, not
zero. Neither required correction direction was priced. DeepEP dispatch and
combine service, including the zero-parameter projection from the all-local
EP4 domain to the registered mixed transport domain, is absent. Rank-preserving
high-bandwidth memory read and write bytes are also absent. The downward
DeepEP correction was not published alone. The common `61/4`, dense `1`,
mixture-of-experts `58`, step `1` and output `1` multipliers were not applied to
a measurement, and no constant was fitted.

## Project disposition

CORE-66 stays open. The feasible capture is blocked on a profiler environment
that has both NVIDIA Nsight Systems (`nsys`) and NVIDIA Nsight Compute (`ncu`)
available on the GH200 compute node, plus explicit authorization for any new
frozen attempt. The exact registered EP72 capture also remains impossible on
this project cluster. No fifth scored run occurred and no milestone moved.

The EP12 and EP8 refusal records remain intact. This result does not bind the
37 launch identities, price DeepEP, decide the high-bandwidth memory candidate,
check either routing scale, move the standard-decode anchor or claim EP72
service.

## Guard and disclosure

Expectations-only commit `a428996` preceded staging, scheduler preflight and the
single real submission. No held-out value was used in arithmetic, prediction,
fitting or publication, and no incidental held-out exposure occurred. Every
bounded reader access has paired begin and end events, and every forbidden
ledger is empty. Pytest, ruff, documentation checkers and git plumbing remain
automated-process exemptions. The physical capture is void because the
profiler executable was unavailable, not because of held-out exposure.
