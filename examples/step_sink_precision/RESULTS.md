# Step-sink precision results

Date: 2026-08-10

The expectations were frozen before implementation and before the first run
at commit `9a8c05ec583b26b4ce4dc5081f35114920fe0f26`. Checks A and B matched
every registered value exactly. Check C contained an invalid backend
configuration: `htsim_rnic` does not allow a physical topology file with
`rnic-nn-fluid`. The frozen file was not edited after this observation. The
corrected fluid comparison and the actual `rnic-cn` topology comparison are
reported separately as post-specified regression checks.

Raw GOAL, binary and completion-ledger artifacts plus `summary.csv` are under
`/data3/yifeng/simllm-dev/wave1-runs/back567_step_sink_precision/`. They are
not Git content.

## Reproduction

```bash
SIMLLM_HTSIM_RNIC=/data3/yifeng/simllm-dev/build-htsim/datacenter/htsim_rnic \
SIMLLM_TXT2BIN=/data3/yifeng/simllm-dev/tools/txt2bin.prebuilt \
.venv/bin/python examples/step_sink_precision/run_study.py \
  --out /data3/yifeng/simllm-dev/wave1-runs/back567_step_sink_precision
```

Every successful backend row reported `physical_quiescence=verified`; the
Python wrapper rejects a run that does not.

## Check A: unequal provider layer durations

The optional provider surface returned the registered unequal picosecond
durations. The sink validated their count and exact fused sum, converted
cumulative layer boundaries to whole GOAL nanoseconds, and rendered the same
sequence on every participating rank.

| layers | TP width | provider layer ps | rendered calc ns | estimate ps | measured JCT ps | frozen JCT ps | residual ps | flows |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 2 | 2 | 2,600; 4,600 | 2; 5 | 7,200 | 16,017,240 | 16,017,240 | 0 | 16 |
| 2 | 4 | 4,600; 8,600 | 4; 9 | 13,200 | 48,028,360 | 48,028,360 | 0 | 96 |
| 4 | 2 | 2,600; 4,600; 6,600; 8,600 | 2; 5; 6; 9 | 22,400 | 32,042,480 | 32,042,480 | 0 | 32 |
| 4 | 4 | 4,600; 8,600; 12,600; 16,600 | 4; 9; 12; 17 | 42,400 | 96,072,720 | 96,072,720 | 0 | 192 |

At fixed width, increasing the layer count increased JCT. At fixed layer
count, increasing TP width increased JCT. Every calc sequence was strictly
increasing, and its sum equaled `floor(estimate_ps / 1000)` exactly. These are
the four exact-oracle rows of Check A; structural provider-contract guards are
not included in that row count.

## Check B: exact sample attribution

The existing record fields were source-checked before adding the optional
count. Phase, new tokens and post-step context do not reveal total prompt
length. The vLLM translator retains `prompt_len` in private request state and
already computes `produces_token`; the SGLang batch row records post-step
context but not total prompt length. Exactness therefore needs new record
content.

| relation | absent-field result | exact-field result | frozen relation | observed |
|---|---:|---:|---:|---:|
| chunked-prefill sample count | 2 | 1 | decrease by 1 | decrease by 1 |
| fused estimate ps | 912,896 | 880,128 | delta 32,768 | delta 32,768 |
| represented calc ns per layer | 456 | 440 | delta 16 | delta 16 |
| fluid JCT ps | 16,963,200 | 16,931,200 | delta 32,000 | delta 32,000 |
| all-sample fused estimate ps | 424,960 | 424,960 | equal | equal |
| all-sample fluid JCT ps | 16,444,480 | 16,444,480 | residual 0 | residual 0 |

The all-sample GOAL texts were byte-identical. The chunked-prefill fused
delta is the exact one-request LM-head term. Its GOAL delta is 768 ps smaller
because the unchanged compatibility path independently truncates the scalar
estimate to one whole-nanosecond value per layer, exactly as registered.

`num_sampled` is optional in `atlahs-closed-loop-step-v1`. An absent value is
omitted by the writer, old payloads load with `None`, and the sink falls back
to `len(scheduled)` exactly. A present value round-trips and is range checked.
Adapter population remains VLLM-15 and SGL-12.

## Check C: GOAL-rank padding

The first study attempt completed A and B, then stopped when the registered
fluid topology row exited with code 2. The backend states that physical Clos
options are valid only for physical profiles. This is a preregistration defect
because the expectations required `rnic-nn-fluid` and a topology file in the
same command.

Two post-specified comparisons preserve the intended mechanism test:

| evidence | profile | topology flag | TP width | explicit-knob JCT ps | workaround JCT ps | residual ps | normalized flow ledger |
|---|---|---|---:|---:|---:|---:|---|
| fluid correction | rnic-nn-fluid | no | 2 | 16,214,240 | 16,214,240 | 0 | exact |
| fluid correction | rnic-nn-fluid | no | 4 | 48,219,360 | 48,219,360 | 0 | exact |
| topology supplement | rnic-cn | yes | 2 | 68,930,560 | 68,930,560 | 0 | exact |
| topology supplement | rnic-cn | yes | 4 | 206,401,920 | 206,401,920 | 0 | exact |

The fluid correction compares a 64-rank GOAL with active ranks `0..W-1`
against the old `64-W..63` placement workaround, without the illegal topology
flag. The topology supplement performs that comparison on the committed
64-node Clos with `rnic-cn`, which is the live BACK-7 use case. Both GOALs in
every pair declared 64 ranks. After subtracting the old placement's rank
offset, source, destination, tag, payload, start, completion and FCT rows were
identical.

These rows are not claimed as preregistered passes. They are post-specified
regression evidence that the explicit knob reproduces the old workaround on
both the exact fluid anchor and the physical topology path.

## Check D: fatal compatibility and structural guards

The default sink GOAL retained the pre-change SHA-256 digest
`f8aade109ba8e3a581b7d965b3a0c76c1247016a1e37491fa84efbbf377677a5` and
the 12,030 ns even per-layer calc. The absent sample field remained omitted,
and a present field round-tripped through v1. Three invalid provider
breakdowns were rejected: wrong count, negative duration and fused-sum
mismatch.

These checks are fatal but unscored. They do not increase any behavioral
denominator.

## Repository gates

```text
$ .venv/bin/ruff check .
All checks passed!
```

```text
$ .venv/bin/pytest -q
........................................................................ [ 89%]
...................................                                      [100%]
323 passed in 1.35s
```

No native C++ source changed, so no native build or CTest gate applies.
