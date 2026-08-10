# Step-sink latent knobs: results

Date: 2026-08-10

COMP-16 and VLLM-15 are complete. The enabled roofline family split changed
first-token latency by the frozen `+1,000 ps` in all four fluid cells. The
vLLM translator's exact sample count changed the mixed chunked-prefill cell by
the frozen `-32,000 ps`. Every exact residual is zero, the real vLLM smoke
passed, and the explicit compatibility path retained its historical GOAL
digest.

## Expectations and chronology

The admissible expectations-only commit is
`25d098c997f078eb92dcf155cd36c44d9d6b2313` (`Freeze the latent knob
expectations`). It precedes every implementation edit and every result-producing
run. Its commit message records that the working tree contained only the
expectations file and no implementation file.

Before the freeze, both registered runner modes passed `--check-only`. The
output directory remained absent, so those dry runs produced no measured
values or result artifacts. The vLLM source audit and its installed-source
digests are frozen in [expectations.md](expectations.md).

Raw GOAL, binary, completion-ledger, JSONL, and summary artifacts are under
`/data3/yifeng/simllm-dev/wave2-runs/comp16_latent_knobs/`. The directory has
36 files totaling 272 KiB. None is Git content.

## Reproduction

Deterministic provider, adapter, backend, and compatibility checks:

```bash
SIMLLM_HTSIM_RNIC=/data3/yifeng/simllm-dev/build-htsim/datacenter/htsim_rnic \
SIMLLM_TXT2BIN=/data3/yifeng/simllm-dev/tools/txt2bin.prebuilt \
.venv/bin/python examples/step_sink_latent_knobs/run_study.py \
  --mode deterministic \
  --out /data3/yifeng/simllm-dev/wave2-runs/comp16_latent_knobs
```

Pinned external-runtime smoke:

```bash
env PYTHONPATH=. VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  VLLM_USE_V2_MODEL_RUNNER=0 SIMLLM_VLLM_WORKER_MODE=skeleton \
  SIMLLM_VLLM_MODE=virtual HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_HOME=/home/yifeng/packages/vllm-rnic-capture/hf-cache \
  CUDA_VISIBLE_DEVICES= \
  /data3/yifeng/simllm-dev/venv-vllm/bin/python \
  examples/step_sink_latent_knobs/run_study.py \
  --mode live-vllm \
  --out /data3/yifeng/simllm-dev/wave2-runs/comp16_latent_knobs
```

Every successful backend row reported `physical_quiescence=verified`; the
backend wrapper rejects a run whose completion ledger is not quiescent.

## Evidence accounting

Evidence classes remain separate.

| Evidence class | Result | Scored meaning |
|---|---:|---|
| Run configurations | 4 roofline cells plus fixed adapter and live cells | Unscored inputs |
| Check A exact end-to-end rows | 4/4 pass | Scored behavioral rows |
| Check B1 adapter-to-TTFT relation | 1/1 pass | Scored behavioral relation |
| Check B2 attribution instances | 5/5 pass | Scored behavioral instances |
| Check C real vLLM relation | 1/1 pass | Scored live integration relation |
| Check D and conservation guards | pass | Fatal, unscored structural invariants |
| Repository unit and integration tests | 455/455 pass | Separate test executable |
| Pinned-vLLM adapter tests | 35/35 applicable pass | Separate external-runtime test executable |

The scored headline is 11 of 11 passing relations or rows. Configuration
echoes, exact-sum conservation, byte identity, schema behavior, source pin,
and quiescence are not added to that denominator.

## Check A: roofline family split

Both frozen shapes were memory-bound. The provider apportioned fused service
using bytes from the exact family projection, divided all repeated families
over the transformer layers, and assigned the LM-head family to the last
layer. Integer cumulative boundaries preserved the fused estimate exactly.

| layers | TP width | layer duration ps | disabled calc ns | enabled calc ns | disabled TTFT ps | enabled TTFT ps | delta ps | residual ps |
|---:|---:|---|---|---|---:|---:|---:|---:|
| 2 | 2 | 14,811; 20,663 | 17; 17 | 14; 21 | 16,044,240 | 16,045,240 | +1,000 | 0 |
| 2 | 4 | 14,811; 20,663 | 17; 17 | 14; 21 | 48,049,360 | 48,050,360 | +1,000 | 0 |
| 4 | 2 | 14,811; 14,811; 14,812; 20,663 | 16; 16; 16; 16 | 14; 15; 15; 21 | 32,084,480 | 32,085,480 | +1,000 | 0 |
| 4 | 4 | 14,811; 14,811; 14,812; 20,663 | 16; 16; 16; 16 | 14; 15; 15; 21 | 96,094,720 | 96,095,720 | +1,000 | 0 |

All ranks rendered the listed calc sequence. Flow counts were exactly 16, 96,
32, and 192 in table order. At fixed TP width, four layers had greater TTFT
than two. At fixed layer count, TP width four had greater TTFT than width two.
The final allreduce boundary moved later by exactly 1,000 ps in every cell, as
registered.

The decision-relevant guard also passed: every family projection conserves
both fused flops and bytes, every returned duration is nonnegative, and each
layer vector sums to the scalar estimate exactly. It remains fatal and
unscored rather than being counted as another behavioral pass. The observed
evidence therefore supports the existing `estimate_layers` contract; no
redesign to a separate layer-work object is needed for the roofline path.

## Check B: exact sample attribution

The mixed scheduler-shaped input went through the actual vLLM translator. Its
mid-prompt request did not sample, its attached decode request did, and the
produced record carried `num_sampled=1`. Removing only that optional field
created the compatibility comparison.

| quantity | absent-field bypass | adapter exact field | exact minus bypass | frozen delta | residual |
|---|---:|---:|---:|---:|---:|
| sample count | 2 | 1 | -1 | -1 | 0 |
| fused estimate ps | 912,896 | 880,128 | -32,768 | -32,768 | 0 |
| rendered calc ns per layer | 456 | 440 | -16 | -16 | 0 |
| JCT/TTFT ps | 16,963,200 | 16,931,200 | -32,000 | -32,000 | 0 |

The five attribution instances also matched the nonempty fabricated output
rows exactly:

| case | scheduled rows | record samples | output rows | residual |
|---|---:|---:|---:|---:|
| mid-prompt chunk | 1 | 0 | 0 | 0 |
| prompt-completing chunk after prefix hit | 1 | 1 | 1 | 0 |
| prefix-cache completion on admission | 1 | 1 | 1 | 0 |
| decode | 1 | 1 | 1 | 0 |
| attach-mid-flight fallback | 1 | 1 | 1 | 0 |

The equality to fabricated rows supports keeping attribution at the
translator rather than moving it to a post-fabrication output seam.

## Check C: pinned live vLLM smoke

The scored external-runtime run used vLLM v0.26.0, the real in-process engine,
the dotted `SimWorker`, a `SimModelRunner`, a two-token scheduling budget, one
explicit three-token prompt, and two requested outputs. It observed:

```text
scheduled_tokens=(2, 1, 1)
exact_samples=(0, 1, 1)
sampled_token_ids=(24577, 24577)
record_count=3
```

The exact sample counts sum to the two live generated tokens. This is scored
integration evidence because vLLM could have ignored the budget, selected a
different worker path, or exposed an attribution mismatch. The host still
identified a GTX 1660 Ti after `CUDA_VISIBLE_DEVICES=`; this run does not
claim GPU invisibility and does not close VLLM-16.

## Check D: compatibility and structural guards

The default provider still returns no layer breakdown. Its sink-generated
GOAL retained SHA-256
`f8aade109ba8e3a581b7d965b3a0c76c1247016a1e37491fa84efbbf377677a5`
and the 12,030 ns even layer value exactly. An ordinary kernel with no family
metadata returns no breakdown even when the provider flag is enabled, and a
nonconserving family projection is rejected.

An absent `num_sampled` field remains absent on the wire and round-trips
through the v1 reader; a present zero remains present. These guards all
passed. They are fatal but unscored.

## Genuine-risk fraction

This estimate asks which scored relations a competent implementation could
plausibly have failed, not which checks happened to fail.

| Family | Plausibly at risk | Scored total | Fraction | Why it could fail |
|---|---:|---:|---:|---|
| A, roofline to TTFT | 4 | 4 | 100% | Summing per-family roofline maxima, rounding layers independently, or placing the LM head evenly would change boundaries or violate the closed form. |
| B1, adapter to TTFT | 1 | 1 | 100% | Leaving the record field absent would retain two samples and produce a zero TTFT delta instead of `-32,000 ps`. |
| B2, attribution matrix | 5 | 5 | 100% | Prompt completion, cached admission, and attach-mid-flight state use different translator branches; a blanket scheduled-row count would fail at least the mid-prompt instance. |
| C, real vLLM | 1 | 1 | 100% | The real worker or stream path could bypass the edited construction site, or vLLM could schedule a different chunk shape. |
| Overall | 11 | 11 | 100% | Every scored row exercised behavior absent or materially different before this change. |

## Repository gates

```text
$ .venv/bin/ruff check .
All checks passed!
```

```text
$ SIMLLM_HTSIM_RNIC=... SIMLLM_TXT2BIN=... .venv/bin/pytest -q
........................................................................ [ 15%]
........................................................................ [ 31%]
........................................................................ [ 47%]
........................................................................ [ 63%]
........................................................................ [ 79%]
........................................................................ [ 94%]
.......................                                                  [100%]
455 passed in 13.76s
```

The pinned-vLLM adapter test executable separately reported 35 passed and two
absence-only skips in 7.14 seconds. No C++ source changed, so no native CMake
or CTest gate applies.

## Deliberate omissions and residual work

Profile-table and trace-calibrated per-layer estimates remain COMP-17 after
COMP-6 supplies captured per-invocation shapes. Those providers keep the
accepted no-breakdown path. SGLang sample-count attribution remains SGL-12;
this slice deliberately did not touch that adapter. VLLM-16 remains open for
the equivalent worker smoke on a genuinely GPU-invisible host.
