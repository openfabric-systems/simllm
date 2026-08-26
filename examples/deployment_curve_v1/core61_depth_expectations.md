# CORE-61 depth-extrapolation expectations

Status: expectations only. No field from the retained candidate record has
been read for CORE-61.

## Frozen hypothesis and signed direction

The existing rule is:

```text
T_linear(61) = 61 / 4 x T(4)
```

It multiplies launch, scheduler, and other per-step fixed service by 15.25 even
though that service occurs once per decode step. Before computing any retained
value, CORE-61 predicts that the separated rule

```text
T(4) = F + 4 x p
T_separated(61) = F + 61 x p
```

is strictly smaller when `F > 0`, unchanged when `F = 0`, and differs from the
linear rule by exactly `57 / 4 x F`.

## Frozen field-addressed exposure

The only permitted record is
`examples/hopper_kernel_cycle_candidate_v1/candidate-record.json`, whose
published manifest names SHA-256
`ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52`.
The reader selects only JSON entry `entries[7]`, expected to be
`deepseek-v3-reduced4-vllm-ep72-decode-b32-c2000`, and projects only the exact
fields enumerated in `core61_depth_expectations.json`.

The projection includes the exact candidate `key` field, evidence classes,
measured step service, median SM clock, and the first kernel's identity, launch
count, measured elapsed service, compute cycles, memory service, fixed service,
and recorded method. The key is then checked for the exact decode pool, launch
mode, physical TP1/DP1/EP1 parallelism, and batch-32 KV-length vector. The
implementation identity carries the EP72 deployment-shape label, but the
reduced physical-envelope key itself remains TP1/DP1/EP1; this study does not
rewrite that key. The reader rejects a second kernel without decoding it. It
skips every unselected value without decoding it and stops after the selected
entry, before the remainder of the record. Every attempted access is appended
with `newline="\n"` to the external access ledger. No whole-record read is
permitted, including for a fresh digest calculation.

## Frozen component classification

The retained record's component equation is evaluated in its own domains:

```text
compute_ps = ceil(compute_sm_cycles x 10^12 / median_sm_hz)
kernel_repeatable_ps = max(compute_ps, memory_service_ps)
kernel_service_ps = kernel_repeatable_ps + fixed_overhead_ps
```

`F` includes `fixed_overhead_ps x launch_count` only when the entry is a
complete kernel stream and the kernel is the explicit one-launch
`aggregate_noncollective_step_service` whose recorded method identifies
retained additive noncollective service. Its compute and memory terms remain the
four-layer repeatable basis. A fixed field without that whole-step identity is
not moved into `F`; it stays in the four-layer repeatable basis and the result
must disclose that the record could not prove it was per-step fixed.

The selected entry must reconstruct exactly as `T(4) = F + 4 x p`. Any shape,
identity, evidence-class, kernel-count, component-sign, or conservation
mismatch fails closed. The separated result is rounded half up only at the
published picosecond boundary.

The result evidence class is `DECLARED`: a derivation from a `MEASURED`
decomposition at one depth. It is not a second measurement and cannot validate
the depth rule.

## Comparison-only calibration context

The published 22,282 tokens per second value is comparison context only. It
may not select a component, alter the classification, tune a scale, or enter
either extrapolation. If its implied step is displayed later, the result must
label it as context and show that it was not an input.

## Resumable Merlin remainder

CORE-61 cannot close in the local arm. Its held-out depth is frozen at eight
layers with the identical batch-32, remote-KV-2000 decode shape. Execution is
forbidden before `2026-08-28T06:30` in `Europe/Zurich` and belongs under the
existing COMP-72 freeze. Output goes below the task-owned external root named
by `SIMLLM_CORE61_RUN_ROOT`; the machine-readable freeze pins its exact value.

The base submission is:

```text
ssh merlin sbatch -M gmerlin7 --partition=gh-hourly --time=00:25:00 --job-name=gh-core61-d8-base --export=ALL,MODEL=deepseek-ai/DeepSeek-V3,MODEL_KEY=deepseek-v3,SHAPE_SET=deepseek,REVISION=e815299b0bcbac849fa540c768ef21845365c9eb,REDUCED_LAYERS=8,GPU_MEMORY_UTILIZATION=0.88,MODE=graph,DEEPSEEK_SUITE=base,MAX_MODEL_LEN=8192,MAX_NUM_BATCHED_TOKENS=16384,RUN_WALL=0 $SIMLLM_MERLIN_STAGE_ROOT/gh200lane/run_vllm_capture.sbatch
```

The exact held-out decode submission is:

```text
ssh merlin sbatch -M gmerlin7 --partition=gh-hourly --time=00:20:00 --job-name=gh-core61-d8-decode --export=ALL,MODEL=deepseek-ai/DeepSeek-V3,MODEL_KEY=deepseek-v3,SHAPE_SET=deepseek,REVISION=e815299b0bcbac849fa540c768ef21845365c9eb,REDUCED_LAYERS=8,GPU_MEMORY_UTILIZATION=0.88,MODE=graph,DEEPSEEK_SUITE=decode,MAX_MODEL_LEN=8192,MAX_NUM_BATCHED_TOKENS=65536,MAX_NUM_SEQS=64,RUN_WALL=0 $SIMLLM_MERLIN_STAGE_ROOT/gh200lane/run_vllm_capture.sbatch
```

After the maintenance gate, submit base then decode. Preserve all
digest-complete output and stop cleanly on SSH loss. Acceptance still requires
the preregistered eight-layer prediction to land within 5 percent of measured
service and a signed residual ledger separating depth scaling from TRAF-66
overlap. COMP-76 remains untouched. `CORE-63` is reserved if the held-out
measurement later exposes a residual that needs its own owner.

## Scope locks

No scored run, analytical-gate artifact, decode session code, NVLink module, or
COMP-76 surface is changed. No model weights are downloaded and no web page is
fetched.
