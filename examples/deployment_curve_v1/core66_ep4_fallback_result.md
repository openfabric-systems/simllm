# CORE-66 EP4 fallback capture result

## What ran

The new fallback cell requested one GH200 node, four GPUs and four ranks under
`gh-hourly` with QoS `gpu_general`. Expert parallel width was four, with four
routed experts per rank and 16 total. The dummy-weight model requested batch 32
and key-value cache length 2,000 per rank, data-parallel attention and
language-model head, three dense layers followed by one mixture-of-experts
layer, multi-token prediction disabled, and one measured decode iteration.
DeepEP was disabled and `--moe-a2a-backend none` selected the pinned source's
standard dispatcher path. Job `200961` was the only scheduler submission.

The cell allocated on `gpu002` with all four requested GPUs. The fail-fast
check verified the recovered CUDA 12.9 modules, compiler, Nsight Systems,
Nsight Compute and Python 3.11 ARM interpreter. Importing pinned SGLang then
failed in `sglang.srt.utils.common` because the environment has no `orjson`
module. The job exited `4:0` after 49 seconds, before any profiler call. It used
196 GPU-seconds, or 0.054444 GPU-hours. It was not retried.

## Declared deviation ledger

No measured value silently represents EP72. The frozen differences and their
expected directions remained:

- Four rather than 72 expert-parallel participants reduce collective work and
  omit the registered peer topology. No DeepEP service can be inferred.
- Sixteen rather than 256 routed experts changes routing frequencies and can
  increase repeated selection. Grouped-kernel occupancy is indeterminate.
- Sixteen unique slots omit the registered 288 slots for 256 unique experts,
  including the three-plus-one-redundant cohort.
- Four rather than 61 layers lowers raw step service by construction.
- One rather than nine nodes removes fabric serialization, switch traversal
  and cross-node contention. Four rather than eight GPUs also reduces the
  intra-node participant count.
- Four routed slots per rank match the registered residency count.
- Eager launch would raise host overhead, while kernel service remains
  deterministic. No kernel launched here.
- The standard `none` backend replaces DeepEP. Its physical service would be
  backend-specific, and DeepEP remains unpriced.
- Dummy weights preserve intended tensor shapes, but dummy routing would not
  be a production distribution. Model construction was not reached.

## Bindings, bytes and routing

Zero of the 37 semantically classified but physically unbound rows received a
physical SGLang identity. The runtime never resolved attention, multi-head
latent attention, mixture-of-experts runner or data-parallel language-model
head backends. Source inspection still identifies `StandardDispatcher` as the
requested fallback dispatcher, but that is not a runtime binding.

The timing pass did not start. The counter pass therefore did not start, so GH
counter permission was never tested. Per-kernel and per-step high-bandwidth
memory reads and writes are unavailable. No routed expert IDs, assignment
counts, owner ranks or local slot IDs were recorded.

The four CORE-65 candidate families receive no new physical decision:

- The three-dense then one-MoE layer composition remains structurally frozen,
  but no physical execution checked it.
- The `1/64` resident-count and weight interpretation remains unchecked
  because neither routing nor HBM bytes exist.
- The inherited `1/9` assignment interpretation remains unchecked because no
  routing record exists.
- The actual weight-read volume remains undecidable because the counter pass
  did not run. No non-DeepEP counterpart row bound, while the four DeepEP
  dispatch and combine families were absent by construction.

## Signed movement

The calibration-only signed movement is null. DeepEP dispatch and combine are
unpriced by construction in this fallback cell, and rank-preserving HBM bytes
are also unavailable after the import failure. Neither correction direction
can be priced. The common `61/4`, dense `1`, mixture-of-experts `58`, step `1`
and output `1` multiplier structure was not applied. No downward correction is
published alone.

## Guard, sanity and project effect

No held-out value entered arithmetic, comparison or fitting. Four incidental
exposure events are disclosed without numeric reproduction. They do not affect
the environment failure, zero binding count or absence of counter and routing
records. There is no kernel duration or byte count against which to evaluate a
memory floor or constituent ceiling.

CORE-66 remains open. This result adds the exact final environment remainder:
the CUDA 12.9, CPython 3.11, ARM environment that runs the profiler also needs
a runnable pinned-SGLang dependency set, beginning with `orjson`. DeepEP still
requires a compatible CUDA 12.9, CPython 3.11, ARM build, and the registered
EP72 cell still exceeds project hardware. No further submission is authorized
by this result. It does not bind a row, decide a byte or routing candidate,
price DeepEP, move the standard-decode calibration, perform the fifth scored
run, close CORE-65 or close CORE-66.
