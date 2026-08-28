# CORE-66 EP4 environment retry result

## Allocated cell and frozen deviation ledger

The scheduler allocated the unchanged EP4 cell on one node with four NVIDIA
GH200 GPUs, but the fail-fast package check stopped before profiling. The
requested topology remained four ranks, four routed experts resident per rank
and 16 experts total, batch 32, key-value cache length 2,000, multi-token
prediction disabled, dummy weights, data-parallel attention and language-model
head, DeepEP, three dense layers, one mixture-of-experts layer and one measured
decode iteration. SGLang remained pinned at
`bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3`. Zero decode iterations ran.

Job `200891` was the only real submission. It resolved to `gmerlin7`,
`gh-hourly`, `gpu_general`, node `gpu003` and four allocated GPUs. It ran for
75 seconds and exited `4:0`, consuming 300 GPU-seconds, or 0.0833 GPU-hours.
Together with the earlier 14-second EP4 launch, the two EP4 attempts consumed
about 0.0989 GPU-hours.

The signed deviation ledger did not change:

- Four rather than 72 expert-parallel peers biases dispatch and combine
  service downward.
- Sixteen rather than 256 unique routed experts raises uniform-routing
  locality and biases remote traffic downward. Grouped-kernel occupancy is
  indeterminate.
- Sixteen unique slots omit the registered 288-slot population for 256 unique
  experts and its three-plus-one-redundant cohort. Locality and
  duplicate-residency effects are indeterminate.
- Four rather than 61 transformer layers lowers raw step service by
  construction.
- One rather than nine nodes omits fabric serialization, switch traversal and
  cross-node contention, biasing communication service downward.
- Four rather than eight GPUs per node reduces the number of intra-node
  participants.
- The EP4 DeepEP transport domain is entirely local, while the registered cell
  has intra-node and cross-node peers. Local service cannot directly price
  EP72.
- Four routed expert slots per rank match the registered residency.
- Eager semantic instrumentation raises host launch overhead. Raw eager step
  time cannot become registered graph-mode step service.
- Any fallback from DeepEP invalidates communication pricing.
- Dummy weights preserve tensor shapes and byte demand but do not provide
  production routing statistics.

No EP4 duration is promoted to measured EP72 service.

## Recovered environment and exact failure

The hard scheduler and profiler environment path is now verified. The job
loaded `gcc/12.3.0` and `cuda/12.9.1`. The CUDA compiler reported 12.9.86.
NVIDIA Nsight Systems (`nsys`) reported the exact successful CORE-61 version,
2025.1.3.140. NVIDIA Nsight Compute (`ncu`) was also present at 2025.2.1. The
PATH-selected interpreter was the retained CORE-61 Python 3.11.11 ARM aarch64
binary. Ordered assertions reached the DeepEP import only after Torch
2.13.0+cu129, Torch CUDA 12.9 and a visible GH200 passed.

DeepEP then failed immediately with `This wheel requires CUDA 13, but PyTorch
uses CUDA 12`. The staged `sgl-deep-ep` 0.1.2 build declares `cu130` and CUDA
major 13. Its wheel tags are CPython 3.12 x86-64, while the successful CORE-61
environment is CPython 3.11 ARM aarch64. The CUDA-major check fired first, so
the extension loader and SGLang capture-module import were not reached.

The preflight therefore proves that the CUDA 12.9 modules, profiler binaries
and interpreter are available on the GH200 image. It also narrows the next
blocker to a DeepEP build compatible with cu129, CPython 3.11 and ARM aarch64,
followed by the still-unreached SGLang import check. No profiler command was
invoked.

## Physical identities and signed movement

Zero of the 37 semantically classified but physically unbound rows received a
kernel binding. There are zero DeepEP dispatch launches, zero combine launches,
no peer, payload or duration records, and no routing or local-slot records. The
high-bandwidth memory counter permission was not tested because neither the
timing nor counter pass started. The `1/64` count-and-weight and `1/9`
assignment candidates remain physically unchecked.

The calibration-only movement remains null, not zero. DeepEP service and its
zero-parameter registered-cell projection are absent, and rank-preserving
high-bandwidth memory read and write bytes are absent. The common `61/4`, dense
`1`, mixture-of-experts `58`, step `1` and output `1` multipliers were not
applied. No downward correction was published alone and no parameter was fit.

## Project disposition

CORE-66 stays open. Its feasible path is blocked on a cu129, CPython 3.11, ARM
aarch64 DeepEP package that passes the frozen import preflight. The subsequent
SGLang capture import also remains untested. The registered EP72 capture remains
impossible on this project cluster. No fifth scored run occurred, no milestone
moved and no further submission is authorized by this result.

The EP12 and EP8 refusal records and job `200879` remain intact. This result
does not bind the 37 identities, price DeepEP, decide high-bandwidth memory
bytes, check either routing scale, move the standard-decode anchor or claim
EP72 service.

## Guard and disclosure

Expectations-only commit `9332023` preceded staging and submission. No held-out
value entered arithmetic, prediction, fitting or publication, and no incidental
held-out exposure occurred. One bounded reader selector was unavailable and
returned no value; its paired access events record the rejection, and its
forbidden ledger is empty. Every other bounded access passed and every
forbidden ledger is empty. Pytest, ruff, documentation checkers and git plumbing
remain automated-process exemptions.
