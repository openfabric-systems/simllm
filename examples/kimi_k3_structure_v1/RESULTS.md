# Kimi K3 logical structure: retained refutation

**The original study is void. Its frozen shared-branch calculation omits
92 final additions, one per routed layer, and underestimates both logical
completion times by 92,000 picoseconds.** COMP-54 remains open. The corrected
checks are explicitly post-specified and do not change the original verdict.

The study drives the exact Kimi K3 checkpoint through the pinned vLLM and
SGLang configuration seams, extracts the full logical text graph, and runs
synthetic serial operator service through completion events and request
metrics. It loads no checkpoint weights and measures no GPU kernel.

## Frozen contract and chronology

The initial expectations-only commit is `c661a85`. The final pre-run freeze is
`727d78c68e38c4f0eecb6e0e75fe37f13932a0b3`, which distinguishes a logical
longest-path check from the explicitly serial runtime. Both precede behavior
implementation and the first component execution. The initial implementation
and formal run source is `5761474`.

The component check refuted the shared-branch oracle before the full study.
Independent source review confirmed that the graph preserves the final
addition required by both native implementations. The frozen files remain
unchanged. The formal study retains those failed guards with a null behavioral
score; it never subtracts them from a pass denominator. Its exact-oracle rows,
behavioral relations and unscored structural guards remain separate artifacts.

The later retained-evidence audit adds checks after observing the run. It
checks the frozen grid and operator partition, graph-to-inventory shapes and
arithmetic, state capacity, exact rational token times, negative execution
boundaries and installed source identity. Those are post-specified regression
checks, not a new claim of pre-registration or task closure.

The formal run completes all 24 structural cells, four native inventory
extractions and eight synthetic request cells containing 24 steps. Its only
fatal findings are the shared-join error in both phases of both frameworks.
The retained-evidence audit at `0e41c8e4957795b8de926e34c1f82bea6d9bfc8a`
passes without findings. It joins every lifecycle event, exact input record,
inventory and request boundary, and verifies 11 pinned native source files
and 40 critical import origins. That later pass leaves the original run void.

The [compact publication](results.json) separates the original 157 exact
oracle rows, 64 retained relation instances across four families, and 292
unscored structural guards. These counts are never added into a score. The
original summary has SHA-256
`11a59db533fbaab49498b091cf29e49846e17e872b6990515f2fe8386d26fcd0`.

The final software regression passes 5,466 tests with 31 skips. Ruff, module
format and task-registry checks pass. These software gates are separate from
the study's evidence classes and do not change its void verdict.

## Why the frozen oracle is wrong

The shared branch has two parallel input projections, then its activation
and output projection: three sequential service regions. The routed branch
has nine sequential regions before the final shared/routed addition. Giving
each shared region 10,000 picoseconds makes that branch determine the join.
The addition itself still costs the ordinary 1,000 picoseconds.

| Phase | Ordinary graph depth | Frozen slow-shared time (ps) | Source-consistent time (ps) |
|---|---:|---:|---:|
| Cold prefill | 1,812 | 3,652,000 | 3,744,000 |
| One-token decode | 1,860 | 3,700,000 | 3,792,000 |

The correct prefill expression is
`(1812 - 9 * 92) * 1000 + 3 * 92 * 10000`. Decode replaces 1812 with 1860.
The frozen calculation subtracted ten ordinary regions per routed layer and
therefore removed the final additions as well. Removing those graph nodes,
setting their service to zero, or changing the frozen numbers would conceal
the error. These graph times assume independent resource grants and do not
establish concurrent admission in the coarse runtime, which remains CORE-12.

## Structure and physical bounds

The checkpoint is
[`moonshotai/Kimi-K3` at `f831ab66814297da540d832a5235f8e904f29d06`](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json).
The [authored suite](../../offline/calibration/suites/kimi-k3-text-v1-frameworks-2026-09-09/suite.json)
pins its complete configuration and the metadata identities of all 96 weight
shards. This binds checkpoint metadata without claiming local weight-content
verification.

Three independent checks constrain the interpretation:

1. Tensor shapes and encoding: one routed expert has
   `3 * 3584 * 3072 = 33,030,144` values. Four-bit values plus one eight-bit
   scale per 32 values require at least 17,547,264 bytes. A packed checkpoint
   does not establish runtime layout or bytes read per step.
2. Retained state: 69 linear-attention layers retain 434,110,464 recurrent
   bytes plus 15,261,696 convolution-history bytes per sequence. The combined
   capacity is 449,372,160 bytes, independent of history length. The 24 latent
   attention layers add exactly 27,648 bytes per cached token. Allocation
   padding, workspace and access traffic are outside these capacity quantities.
3. Causal work: every one of the 93 original layers has its own dependencies.
   A result cannot precede either input to its shared/routed addition. There
   are 3,244 logical visits in prefill and 3,268 in decode. Exact causal pairs
   are `n * prior_context + n * (n + 1) / 2`, so cold prefill grows
   quadratically in new-token count while one-token decode grows linearly in
   history. These statements do not identify a physical launch schedule.

The two framework projections agree on geometry. Their final normalization
has a source-defined invocation difference: vLLM gathers sampled rows before
the output normalization; SGLang normalizes all newly computed rows. That
difference stays in the graph and inventory identities.

## Synthetic request interpretation

The runtime diagnostic gives each logical region the same declared service
per newly computed step token. It records zero memory arbitration explicitly,
uses a separate synthetic graph identity and grants one operation per compute
group. Under this declared resource model, both the floor and ceiling of a
step are exactly `logical_visits * new_tokens * service`.

For one prompt token and 1,000 picoseconds per logical region, the expected
time to first token is 3,244,000 picoseconds. Each decode step then costs
3,268,000 picoseconds. Doubling service doubles all completion times; increasing
the prompt from one to four tokens multiplies the first-token time by four
and preserves the one-token decode service. These numbers are synthetic
request-path checks and must not be interpreted as GPU throughput.

All eight observed cells satisfy those exact synthetic relations in both
frameworks. For the one-token, 1,000-picosecond cell, prefill followed by two
decode steps completes at 9,780,000 picoseconds. The additive sum of queue
waits is a separate work quantity; it is not a wall-latency decomposition.

Raw native repetitions, complete event streams, request records, graph
identities, compatibility records and audit findings are retained outside Git.
The compact publication records their content hashes and verdicts.

## Project consequence

COMP-54 supplies a complete logical K3 text representation and retains its
structural qualification gap after this refutation. The candidate inventory
does not enter the accepted coverage registry. Existing accepted Granite,
Qwen3.8, DeepSeek-V3 and Qwen3-32B columns remain the compatibility controls.
All eight fresh native compatibility records reproduce their accepted
content identities, and all 19 pinned legacy artifacts remain byte-identical.

COMP-59 and COMP-64 still own physical capture and service coverage. CORE-54
still owns realistic deployment curves. This study supplies no multi-GPU
kernel measurements, no calibrated K3 time, no distributed rank projection
and no Pareto-frontier closure.

## Reproduction

Use pinned native environments, a weight-free local K3 configuration directory,
the frozen source copies, and a local JSON map from each of the four legacy
suite IDs to its exact checkpoint directory. Machine-specific values belong
in ignored local configuration. Select a fresh external output directory.

```bash
python -m examples.kimi_k3_structure_v1.run_study \
  --vllm-python "$VLLM_PYTHON" --sglang-python "$SGLANG_PYTHON" \
  --checkpoint-root "$K3_CHECKPOINT_ROOT" --source-root "$K3_SOURCE_ROOT" \
  --legacy-checkpoints "$K3_LEGACY_CHECKPOINTS" --output-root "$K3_RUN_ROOT"
```

The unchanged frozen shared-branch guards make the source-consistent run exit
with status 2 and verdict `VOID`. The post-specified audit preserves that
verdict:

```bash
python -m examples.kimi_k3_structure_v1.audit_retained \
  --source-root "$K3_RUN_ROOT" --native-sources "$K3_NATIVE_SOURCE_PROOF" \
  --output-root "$K3_AUDIT_ROOT"
```
