# Decode HBM crossover v1 expectations

A graphics processing unit (GPU) reads the model's weights and each request's
cached keys and values from high-bandwidth memory (HBM), then performs the
arithmetic that generates the next token. Batching reuses the same weights,
but each additional request adds cache reads and arithmetic. This study pins
down when reading bytes or doing arithmetic limits an ideal decode step.
The reported time per output token (TPOT) is one batch step's service time,
because every request receives one token per step. It is not step time divided
by batch. No host, network, queue, scheduler or memory-capacity claim is made.

The owning tasks are DEPLOY-4 and DEPLOY-5 in
[the deployment module](../../docs/modules/deploy.md). This study supplies
analytical evidence, not the measured surfaces either task requires.
DEPLOY-5 concerns prefill and cannot close from a decode-only study.

## Chronology and authority

This expectations-only commit precedes study implementation and the first
pricing call. The input discovery read source and existing evidence only.
The adjacent [expectations.json](expectations.json) declares every input,
source digest and exact rational crossover. The adjacent
[expected_cells.csv](expected_cells.csv) contains exact integers for every
cell, derived algebraically without importing or calling SimLLM. Those are
predictions, not simulator output. The independent preparation arithmetic
uses only the literal geometry and coefficients below. RESULTS.md must name
this commit and report deviations without modifying any frozen file.

Only installed pricing surfaces are used. Dense cells call `ModelDims`,
`step_kernel` and `RooflineProvider(efficiency=1.0)` against `GPU_ENVELOPES`
and bandwidth-scaled `GpuSpec` objects. The DeepSeek inventory retains its
compressed-attention and nonuniform rank geometry through `KernelSpec`.
Substituting an approximate dense-attention `ModelDims` would be incorrect.
`StepEstimate` and `EstimatorInputs` are used for TP8 and the auxiliary TP4
row, where zero represented collective bytes have an installed partition.
The TP1 and zero-network EP72 estimator paths have no installed partition;
they use the compute provider directly. This does not enable those paths.

`FrontierPoint` requires throughput equal to its `batch_per_gpu` times request
speed, with an integer batch. TP1 shared batch B has that meaning, whereas
TP8 shared batch B has only B/8 output tokens per GPU. Inventing a TP8
FrontierPoint with B would overstate throughput eightfold. This study records
exact rational coordinates directly and leaves that installed record alone.

## Frozen sweep

- Shared decode batch B: 1, 2, 4, 8, 16, 32, 64, 128, 256.
- Devices: a100, h100, h200, b100, b200, using the shipped dense BF16
  (two-byte floating-point) peaks and HBM bandwidths.
- HBM scale s: 0.5, 1, 2, with arithmetic throughput held fixed.
- Dense contexts C: 1 and 2048 tokens, including this step's new token.
  C=1 is a minimal-cache control, not a typical serving prompt.
- Dense geometries, fields already local to a rank:

| Geometry | Layers | Hidden | Intermediate | Heads | KV heads | Head size | Vocabulary | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7B-class | 32 | 4096 | 14336 | 32 | 8 | 128 | 128256 | 1 |
| 70B-class | 80 | 8192 | 3584 | 8 | 1 | 128 | 128256 | 8 |

TP means tensor parallelism, where GPUs jointly process the same requests.
These are declared class geometries, not claims to exact checkpoint counts.
The 7B-class geometry follows the breakdown study. The 70B-class uses a
28672-wide full intermediate layer divided by eight. As in that example,
the language-model output head remains replicated, including its full
vocabulary. It must be included in weight bytes and token arithmetic.
All dense activation, cache and weight elements occupy two bytes.

- DeepSeek-V3 uses the existing worst rank class by logical weight bytes,
  `sglang-decode-ep72-dp-attention`, `rank-class-0`, context 2000, with
  W=27,446,643,040 bytes, F=112,322,823,926 floating-point operations per
  local token and K=140,544,000 cache bytes per local request. The batch-32
  anchor must reconstruct 3,594,330,365,632 operations and 31,944,051,040
  bytes. These are the exact existing deployment projection coefficients,
  extended linearly in batch as in deployment_scan_v1; routing and cache
  mechanisms are not recalibrated. Expert parallelism (EP72) spreads experts
  across 72 ranks while each rank serves its own attention requests. Thus B
  is local requests per rank and throughput per GPU is B/T, not B/(72T).
- For every finite B*, price floor(B*) and ceil(B*) as separate knee probes
  unless that batch is already in the main grid. This adds 105 unique cells;
  the largest is batch 4279. They validate arithmetic classification only,
  and do not assert that such a batch fits memory.
- One auxiliary external comparison cell described below.

There are 675 main configurations, 105 additional knee configurations, and
one external comparison configuration. Every cell's identity, operation
count, byte count, component times, total time, regime and serial ceiling
are frozen in expected_cells.csv. Configuration counts are not pass scores.

## Closed forms and units

Let P be device operations per second, H=s times device HBM bytes per second,
W all per-rank resident weight bytes including the output head, and K(C)
cache bytes per request. For a dense geometry with layers L, hidden h,
intermediate i, query heads n, cache heads k, head size d and vocabulary V:

```
A = L * (h * (n*d + 2*k*d) + n*d*h + 3*h*i) + h*V
W = 2*A
K(C) = 4*C*L*k*d
F(C) = 2*A + 4*(C-1)*L*n*d
M(B) = (W + B*K(C)) / H
Q(B) = B*F(C) / P
T(B) = max(M(B), Q(B))
throughput_per_gpu = B / (TP*T(B))
B* = W*P / (F(C)*H - K(C)*P), if the denominator is positive
```

F already counts two operations per multiply-accumulate; multiplying it by
two again would double-count arithmetic. The brief's `2 * flops_per_token`
form is dimensionally interpreted as two times multiply-accumulates per
token, equivalently F as defined here. EP72 uses its frozen F, W and K and
TP=1 in the throughput formula. The auxiliary FP8 row uses its own frozen
coefficients instead of the dense W=2*A relation.

When F*H <= K*P, there is no finite compute crossover: memory stays slower
for every batch. Otherwise T=M below B* and T=Q at and above B*. Memory
service is affine in batch, not exactly flat: relative excess over the
weight-only time is B*K/W. The phrase "flat below B*" is valid only when
that fraction is negligible, here explicitly defined as at most one percent.
At K=0 the ideal closed form is exactly flat. No realistic cache read is
silently removed to obtain that picture. A second useful scale, W/K, marks
equal weight and cache traffic; it is not the compute crossover.

In the memory regime throughput increases with diminishing returns toward
H/(TP*K). In the compute regime it reaches P/(TP*F), with TPOT linear in B.
The smaller of these two rates is the asymptotic throughput ceiling.
Doubling H halves the memory term, leaves arithmetic unchanged, and moves a
finite B* earlier. A memory-only asymptote can look like saturation without
any arithmetic crossover, so saturation alone does not identify the regime.

All durations are picoseconds (ps), 10^12 ps per second. The exact API oracle
uses the installed floating arithmetic order `int(work / float(rate) * 10^12)`
on each component before taking the maximum, with zero-ps acceptance.
An independent rational floor uses integer division on work*10^12/rate.
Its difference from the API projection is bounded by 1 ps, stated before
observation. This quantization allowance is not an extra behavioral pass.

## Physical bounds, stated before pricing

Floor: no represented step can beat max((W+B*K)/H, B*F/P), and in particular
no TPOT can beat W/H, apart from sub-picosecond integer representation.
Ceiling: the ideal model's fully serialized memory-plus-arithmetic schedule
costs (W+B*K)/H + B*F/P, so it bounds the fused ideal step from above.
There is no finite real-serving latency ceiling derivable from peak rates
alone; host stalls, communication and queuing may exceed this model ceiling.

The nominal weight-only floors below are exact integer floor projections,
not timings. The half-bandwidth and double-bandwidth arms use W/H directly,
never multiply a previously rounded integer.

| Geometry | A100 ps | H100 ps | H200 ps | B100 ps | B200 ps |
|---|---:|---:|---:|---:|---:|
| 7B-class | 7361116657 | 4480393093 | 3126941013 | 1876164608 | 1876164608 |
| 70B-class TP8 | 9423298981 | 5735554216 | 4002938880 | 2401763328 | 2401763328 |
| DeepSeek-V3 EP72 | 13460835232 | 8193027773 | 5718050633 | 3430830380 | 3430830380 |

These floors assume the represented resident bytes are streamed from HBM.
The peak envelopes provide no memory-capacity surface, so working-set byte
counts are reported but deployment feasibility is unchecked. The crossover
probes beyond batch 256 are especially not capacity recommendations.

## Evidence classes and acceptance

Exact-oracle class E1: all 781 cells must match the frozen FLOPs, bytes,
component times, TPOT integer and regime at 0 ps. Work conservation itself
is separately a fatal precondition and never increases a behavioral score.

Behavioral relations, reported by family and parameterized instance:

- B1, bandwidth response: every adjacent bandwidth pair satisfies
  T_fast <= T_slow <= 2*T_fast+2 ps. If both are memory-bound,
  abs(T_slow-2*T_fast) <= 2 ps. Only memory or transition pairs are scored;
  pairs that are both compute-bound have a fatal, unscored identity guard.
- B2, memory batch growth: adjacent batches that remain memory-bound have
  abs((T2-T1)-(B2-B1)*K*10^12/H) <= 2 ps. This quantifies the deviation from
  flatness, including contexts where it is large.
- B3, compute batch growth: adjacent compute-bound doubled batches satisfy
  abs(T2-2*T1) <= 2 ps. The throughput ratio differs from one by at most
  2/T2; any tiny nonmonotonicity inside this rounding bound is reported.
- B4, frontier direction: adjacent main batches have nondecreasing TPOT and
  nondecreasing throughput to relative tolerance 2/min(T1,T2). Rounding can
  make the ideal constant-throughput segment weakly wobble inside this band.
- B5, crossover: the frozen rational B* brackets the provider's memory to
  compute transition at its integer neighbors. An exact integer tie is
  compute-bound. For absent crossovers all sampled cells remain memory-bound.
- B6, context response: for paired dense cells T(2048) >= T(1), with strict
  growth whenever cache reads or attention arithmetic affect the maximum.

No families from different evidence classes are summed into a headline score.
A failed behavioral relation yields FAIL; any failed fatal guard yields VOID
and makes every behavioral score uninterpretable. No guard is survivable.

Fatal guards: frozen file and source digests; complete unique cell identities;
input geometry and EP72 anchor reconstruction; exact emitted work; integer
physical bounds (at most 1 ps rational floor difference); only ROOFLINE or
DECLARED pricing provenance; correct TP and EP output normalization; exact
StepEstimate-to-provider agreement where used; zero represented network,
host and surface terms; no subprocess creation in pricing; frozen commit
ancestry and unchanged expectations. Metadata-only git checks occur outside
the intercepted pricing region. Disabled or forced-zero checks stay unscored.

## External sanity angle

The pinned frontier_comparison_v1 result contains published external-tool
row 4: Qwen3-32B FP8 on H200, TP4 decode, batch 64, average context 4250,
TPOT 9,179,000,000 ps. It is an offline performance-database estimate from
aiconfigurator 0.11.0, database h200_sxm trtllm 1.3.0rc10, not a fresh live
serving measurement. The upstream tool describes its data-based method in
[its project documentation](https://github.com/ai-dynamo/aiconfigurator).

The auxiliary cell uses that record's exact per-rank coefficients:
W=7,995,883,520 bytes, K=278,528,000 bytes/request, F=18,218,903,040
operations/token, and the same dense FP8 peak 1,979,000,000,000,000
operations/s (twice the shipped BF16 peak), at 4,800,000,000,000 bytes/s.
Its physical memory floor is 5,379,515,733 ps after integer projection;
its arithmetic floor is 589,191,407 ps, and the serial model ceiling is
5,968,707,141 ps. Expect the roofline TPOT to be at or below the external
TPOT; this comparison is unscored plausibility evidence, not calibration.
Compare TPOT only: that external system-wide throughput also accounts for
prefill workers and cannot be compared with decode-only per-GPU throughput.
The retained attention-projection inconsistency is inherited evidence and
is not repaired here; it does not change the binding memory term.

## Figure and scope

One plain matplotlib figure, PNG plus PDF: log-log horizontal TPOT in ms,
vertical output tokens/s/GPU, device-colored nominal-bandwidth curves for
all three models. Dense C=1 curves are solid and C=2048 curves dashed;
DeepSeek uses C=2000. Diamonds mark finite analytical B*, including markers
outside the nine-batch sweep, clearly distinguished from sampled circles.
The frozen integer probes support those marks. Large-context knees outside
the main plot window are reported numerically instead of distorting axes.
All bandwidth arms remain available in the results table.

This supplies a roofline bottleneck map and a reproducible estimator check.
It does not create a calibrated decode surface, price prefill, install a
profile, validate hardware, measure communication, certify memory capacity,
change a production default or close DEPLOY-4 or DEPLOY-5.
