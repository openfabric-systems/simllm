# CORE-66 EP4 fallback capture expectations

## Frozen cell

This is a new cell. It does not amend the EP8 refusal, the first EP4 launch
failure or the EP4 DeepEP environment failure. One GH200 node runs four GPUs,
four ranks and expert-parallel width four. Each rank owns four routed experts,
so the reduced model has 16 routed experts. It has dense layers 0 through 2
and one mixture-of-experts layer at index 3. Each rank runs batch 32 at a
key-value cache length of 2,000. Multi-token prediction is disabled. Weights
are dummy-only. Attention and the language-model head are data-parallel. CUDA
graph replay is disabled. One decode iteration is measured.

The pinned SGLang commit is
`bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3`. The command explicitly selects
`--moe-a2a-backend none`. In this source, that choice instantiates
`sglang.srt.layers.moe.token_dispatcher.standard.StandardDispatcher`. The
dispatcher maps global routed-expert IDs to each rank's four local slots. The
normal mixture-of-experts path then combines routed output through SGLang's
standard post-expert tensor-parallel collective. This is the fallback path,
not DeepEP. No DeepEP or NVSHMEM source build is permitted.

The scheduler request is cluster `gmerlin7`, partition `gh-hourly`, QoS
`gpu_general`, account `merlin`, one node and four GPUs. It is submitted once.
The job first verifies the recovered CORE-61 CUDA 12.9 modules, `nvcc`, Nsight
Systems, Nsight Compute, the exact Python 3.11 ARM interpreter, the pinned
source and the non-DeepEP SGLang imports. A failure stops before profiling.

## Declared deviation ledger

This cell identifies fallback-path physics and identities. It never claims an
EP72 measurement.

- Four rather than 72 expert-parallel participants reduce collective work and
  omit the registered peer topology. No DeepEP peer service is inferred.
- Sixteen rather than 256 unique routed experts changes routing frequencies
  and can increase repeated expert selection. Grouped-kernel occupancy remains
  indeterminate.
- Sixteen unique slots omit the registered 288-slot population for 256 unique
  experts. The three-plus-one-redundant cohort and duplicate-residency effects
  remain absent.
- Four rather than 61 transformer layers lowers raw step service. Only
  separately identified per-layer services could enter the frozen multiplier
  ledger.
- One rather than nine nodes removes fabric serialization, switch traversal
  and cross-node contention. Four rather than eight GPUs also reduces the
  intra-node participant count.
- Four routed expert slots per rank match the registered residency count.
- Eager launch raises host overhead but leaves deterministic kernel service
  identifiable. Raw eager step time is not registered graph-mode service.
- The standard `none` backend replaces DeepEP. Its launches and durations are
  backend-specific. DeepEP dispatch and combine remain unpriced, so signed
  calibration movement is null by construction.
- Dummy weights preserve tensor shapes and measurable byte demand. Their
  routed IDs are not production routing statistics.

## Evidence and candidate decisions

The capture targets physical bindings for the 37 semantically classified rows
that lack SGLang identities. The attention, multi-head latent attention, dense
feed-forward, router, shared-expert, routed-expert and data-parallel
language-model-head paths do not require DeepEP and may bind here. The four
DeepEP dispatch and combine counterpart families remain absent by construction.

The counter pass preserves process and rank identity. If GH200 counter access
is permitted, it publishes each kernel's high-bandwidth memory reads and writes
and their per-rank and per-step totals. Those bytes decide the actual
fallback-cell weight-read volume. If permission is denied, the exact denial is
published and the weight-read candidate remains undecidable.

The timing pass records every routed expert ID, assignment count, owner rank
and local slot for the mixture-of-experts layer. Those records check how the
`1/64` resident-count and `1/9` assignment interpretations map to physical
behavior, with the explicit limitation that dummy routing is not a production
routing distribution. Semantic ranges check the three-dense then one-MoE layer
composition.

DeepEP remains blocked on a CUDA 12.9, CPython 3.11, ARM aarch64 build. Because
one required correction direction is unavailable by construction, the
calibration-only signed movement stays null even if HBM counters succeed. The
frozen multipliers remain common `61/4`, dense `1`, mixture of experts `58`,
step `1` and output `1`. The downward correction is never published alone.

## Guard and disclosure

A held-out value entering arithmetic, prediction comparison, fitting or a
published reproduction is fatal. The run is then void.

Incidental exposure without use is survivable and disclosed. Physical kernel
identities, HBM bytes and routing records remain interpretable because the
derivation contains zero free or fitted parameters and is independently
checkable. Calibration movement is already forced null. Broad searches and
unguarded protected-record reads remain forbidden. Pytest, ruff, documentation
and task-progress checkers, and git plumbing are automated-process exemptions.

Two incidental exposures occurred before this freeze. A narrow test file read
for candidate-key discovery displayed retained calibration-output fields. The
guarded CORE-66 task read displayed the registered standard-decode anchor.
Neither value entered arithmetic, comparison, fitting or reproduction here.

## Physical sanity

Before reading a duration, the memory floor is measured read plus write bytes
divided by GH200 peak high-bandwidth memory rate. A kernel cannot outlast the
decode-step interval that contains it. Review also checks fallback collective
launches against the one-node participant topology and verifies that semantic
ranges contain exactly three dense layers followed by one mixture-of-experts
layer.
