# Step-sink precision: pre-registered expectations

Written and frozen before the BACK-5, BACK-6 and BACK-7 implementation and
before any run of this study. The immutable pre-change reference is SimLLM
commit `4d486d1`. No value below is fitted to a simulation result.

## Evidence classes

The study keeps these classes separate:

- Check A is an exact analytical oracle over a two-parameter matrix.
- Check B contains two behavioral sample-attribution relations.
- Check C is a two-row exact equivalence comparison on a live backend.
- Check D and the validation bullets are fatal compatibility or structural
  invariants. They are unscored and do not increase any behavioral pass
  denominator.

All backend rows use `rnic-nn-fluid` at 400 Gbit/s. The reused M1 constants
are 20 ps per payload byte, fixed propagation `P = 2,000,000 ps`, and one
GOAL `calc` unit equal to 1,000 ps. Raw outputs belong under
a machine-local external directory outside Git; the resolved historical target
is intentionally omitted. As a post-freeze portability convention, new runs
default to `${SIMLLM_DATA_ROOT}/step_sink_precision/`.

## Check A: unequal provider layer durations

The two swept parameters are transformer layer count `L` in `{2, 4}` and TP
width `W` in `{2, 4}`. Every cell schedules one decode token at context 16
with an exact sample count of one. Geometry is hidden size 64, intermediate
size 128, four attention and KV heads of width 16, vocabulary size 256 and
two-byte elements. The host-initiation delay is zero.

A registered provider implements both the existing fused estimate and the
optional per-layer surface. Layer `i`, numbered from zero, has duration

```text
d_i(L, W) = 1000 * W * (i + 1) + 600 ps.
```

Its fused duration is exactly `E = sum_i d_i`. The provider contract must
reject a breakdown with the wrong layer count, a negative duration, or a sum
different from the fused estimate. Those rejection checks are structural and
unscored.

GOAL accepts only whole nanoseconds. For an opt-in breakdown, truncation is
registered at cumulative layer boundaries:

```text
b_i = floor(sum_{j=0..i} d_j / 1000)
c_0 = b_0
c_i = b_i - b_{i-1}
```

Thus `sum_i c_i = floor(E / 1000)` exactly. A nonzero host delay, if used,
belongs to the first layer before this boundary conversion. This study keeps
it zero to isolate the provider seam. A provider that does not implement the
optional surface retains the old rule exactly:

```text
c_even = floor(E / (L * 1000))
```

The registered opt-in rows are:

| L | W | provider durations ps | E ps | rendered calc ns by layer | sum calc ns |
|---:|---:|---|---:|---|---:|
| 2 | 2 | 2,600; 4,600 | 7,200 | 2; 5 | 7 |
| 2 | 4 | 4,600; 8,600 | 13,200 | 4; 9 | 13 |
| 4 | 2 | 2,600; 4,600; 6,600; 8,600 | 22,400 | 2; 5; 6; 9 | 22 |
| 4 | 4 | 4,600; 8,600; 12,600; 16,600 | 42,400 | 4; 9; 12; 17 | 42 |

Every participating rank must render that cell's exact layer sequence. The
sequence is strictly increasing, increasing `W` increases every layer share,
and the first two raw durations at `L = 4` equal the `L = 2` raw durations for
the same `W`.

The collective payload is `S = 1 * 64 * 2 = 128 bytes`. Each layer has two
ring allreduces, each with `2(W-1)` serial rounds and chunk `S/W`. The exact
end-to-end form is

```text
J(L, W) = 1000 * sum_i c_i
          + 2L * 2(W-1) * ((S/W) * 20 + P).
```

| L | W | expected JCT ps | expected flows |
|---:|---:|---:|---:|
| 2 | 2 | 16,017,240 | 16 |
| 2 | 4 | 48,028,360 | 96 |
| 4 | 2 | 32,042,480 | 32 |
| 4 | 4 | 96,072,720 | 192 |

Every timing residual is exactly 0 ps. JCT increases with either `L` or `W`
on this registered matrix. A rendered layer mismatch, fused-sum mismatch,
flow-count mismatch, or nonzero JCT residual is a failure.

## Check B: exact sample attribution

This check uses `L = 2`, `W = 2` and the same geometry. A deterministic fused
provider maps one modeled FLOP to one picosecond and ignores bytes, so the LM
head delta cannot be hidden by a roofline maximum. It does not implement the
optional layer surface, which deliberately exercises the unchanged even-split
fallback.

The chunked-prefill case schedules two requests:

- one mid-prompt prefill row with four new tokens and post-step context 8;
- one decode row with one new token and post-step context 32.

The old absent-field approximation counts two samples. The exact optional
record field counts one. The registered LM-head difference is

```text
delta_head = 2 * (2 - 1) * hidden * vocab = 32,768 FLOPs
           = 32,768 ps under this provider.
```

The fused estimates are 912,896 ps for the absent-field approximation and
880,128 ps for exact attribution. Historical even splitting renders 456 ns
per layer versus 440 ns per layer. Independent whole-nanosecond truncation
therefore makes the full GOAL and fluid JCT decrease by exactly 32,000 ps:
16,963,200 ps to 16,931,200 ps. Both the 32,768 ps fused-estimate delta and
the 32,000 ps represented/JCT delta must match exactly.

The identity case schedules two decode rows, both of which sample. An absent
sample field and an explicit count of two must produce the same 424,960 ps
fused estimate, 212 ns per-layer calc, byte-identical GOAL and 16,444,480 ps
JCT. This proves that exact attribution changes nothing when every scheduled
request samples.

The existing `StepRecord` fields cannot identify prompt completion: phase,
new-token count and post-step context reveal the computed boundary but not the
request's total prompt length. The vLLM translator retains that length in
private request state and already derives `produces_token`; SGLang's recorded
batch row also omits total prompt length. The shared field is therefore
optional. Its absence uses `len(record.scheduled)` exactly as before. The v1
JSON writer must omit an absent field, old v1 payloads must still load, and a
present field must round-trip. These wire checks are fatal and unscored.

## Check C: explicit GOAL-rank padding

Use the committed 64-node topology
`examples/m1/topologies/clos_64_400g.topo`, `L = 2`, TP width in `{2, 4}` and
one decode token at context 16. For each width compare two logically identical
fluid runs:

1. active ranks `0..W-1` with `num_goal_ranks = 64`;
2. the old workaround, active ranks `64-W..63` with `num_goal_ranks` absent.

Both rendered GOALs must declare exactly 64 ranks. After translating the old
active rank IDs back to `0..W-1`, flow source, destination, tag, payload,
start, completion and FCT ledgers must match exactly. JCT residual between the
two runs is 0 ps for both widths. This equivalence is the behavioral check;
idle-rank `calc 0` population and exact rank count are structural guards.

## Check D: default byte baseline

The fatal baseline uses the existing two-layer small dense geometry: hidden
size 1024, intermediate size 4096, eight attention and KV heads of width 128,
vocabulary size 32000 and two-byte elements. It schedules one 256-token
prefill at context 256 over TP ranks `(0, 1)`, default B100 roofline provider,
default zero host delay, no EP group, no topology, no sample field, no
per-layer provider implementation and `num_goal_ranks = None`.

The sink-produced GOAL bytes must equal the bytes produced by commit
`4d486d1` for that exact record and configuration. The expected per-layer
calc remains 12,030 ns. This comparison is fatal and unscored. It protects the
accepted M4 and CORE-2 compatibility path independently of the opt-in checks.

## Acceptance

- Check A passes all four exact rendered-layer and JCT rows, with the stated
  directions across both swept parameters.
- Check B passes the exact chunked-prefill deltas and the all-sample identity.
- Check C passes both exact rank-padding equivalence rows.
- Check D and every structural or wire invariant pass, but none contributes
  to a behavioral pass count.
- Any unexplained residual, baseline byte change, schema incompatibility,
  provider-contract acceptance of an invalid breakdown, or unverified backend
  quiescence is a failure.
