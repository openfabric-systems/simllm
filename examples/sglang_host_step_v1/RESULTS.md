# Results: the SGLang chain's per-step host cost (SGL-23)

Frozen by expectations-only commit `79b03da`, which landed before the
selector, the harness and the tests existed. The measuring run happened on
2026-08-14 against the seam commit `d803d71`.

**Verdict: the run is not void, and all three scored relations pass.** No
fatal guard was violated. In two evidence classes that are never summed:
63 of 63 exact-oracle rows and 18 of 18 behavioral instances agreed with the
freeze, to the picosecond. Sixty-three replayed steps drove 9,072
`htsim_rnic` invocations across seven cells, plus 1,296 more for the pre-seam
reference sink.

## What ran

Nine tracked SGLang step records
(`examples/m4/fixtures/sglang-m3-steps.jsonl`, SHA-256
`656772148c...cb7c26f`) replayed through `HtsimStepSink` on `rnic-nn-fluid`
at 400 Gbit/s, with `tp_ranks=(0,)` and `ep_ranks=(0..7)`, against the
per-rank Granite MoE geometry the SGLang end-to-end demo declares at
expert-parallel width 8. Seven cells: the `ideal` off arm, and the CUDA-graph
and eager host classes at three launch counts each, chosen so that one launch
straddles the masking crossover.

An eighth arm, `reference-preseam`, built the same sink by hand with
`HostInitiationModel.ideal()`, `GPU_ENVELOPES["b100"]` and
`RooflineProvider(0.7)`, i.e. exactly what a study wrote before the selector
existed.

## Scored relations

| relation | class | instances | result |
|---|---|---:|---|
| R0, provider service equals the frozen roofline | exact-oracle | 9 | 9 pass |
| R1, step latency delta equals the enclosed compute delta | exact-oracle | 54 | 54 pass |
| R2, the regime flips one launch wide | behavioral | 18 | 18 pass |

The 63 exact-oracle rows and the 18 behavioral instances are separate
evidence classes and are not added.

Every frozen delta reproduced exactly. The measured table, in picoseconds
relative to the `ideal` arm of the same record:

| step | new tokens | `ideal` (us) | fabric (us) | `graph122` | `graph123` | `graph440` | `eager41` | `eager42` | `eager567` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12 | 360.103 | 261.151 | 21,000 | 593,000 | 257,143,000 | 21,000 | 347,000 | 1,241,581,000 |
| 1 | 1 | 208.739 | 109.763 | 5,000 | 569,000 | 257,119,000 | 5,000 | 323,000 | 1,241,557,000 |
| 2 | 7 | 291.410 | 192.338 | 24,000 | 473,000 | 257,023,000 | 24,000 | 227,000 | 1,241,461,000 |
| 3 | 2 | 222.621 | 123.525 | 17,000 | 449,000 | 256,999,000 | 17,000 | 203,000 | 1,241,437,000 |
| 4 | 1 | 208.739 | 109.763 | 14,000 | 569,000 | 257,119,000 | 14,000 | 323,000 | 1,241,557,000 |
| 5 | 11 | 346.460 | 247.388 | 15,000 | 473,000 | 257,023,000 | 15,000 | 227,000 | 1,241,461,000 |
| 6 | 2 | 222.621 | 123.525 | 8,000 | 449,000 | 256,999,000 | 8,000 | 203,000 | 1,241,437,000 |
| 7 | 1 | 208.715 | 109.763 | 21,000 | 593,000 | 257,143,000 | 21,000 | 347,000 | 1,241,581,000 |
| 8 | 1 | 208.739 | 109.763 | 5,000 | 569,000 | 257,119,000 | 5,000 | 323,000 | 1,241,557,000 |

R2 is the relation that mattered most and it held at full resolution. At 122
CUDA-graph launches every record reported zero exposed host time and kept the
provider's own `memory` bound; at 123 every record reported
`host-initiation` with a positive exposure, 563,701 ps on step 1. The eager
class flipped between 41 and 42 launches the same way. A selector that had
built the profile outside its named factory would have taken the legacy
additive branch and moved the masked cells by their whole launch floor,
roughly 98.7 us instead of 5 ns. It did not.

## The pre-registered warning that mattered

The freeze registered, before the run, that a fully masked calibrated cell is
**not** identical to the ideal arm, because the two arms quantize differently:
the ideal arm floors each layer independently and the calibrated arm encloses
the whole step to the next whole nanosecond. The measurement confirmed it
exactly. `graph122` and `eager41` both add 5,000 to 24,000 ps per step even
though their launch demand is entirely hidden behind the compute. Only the
`ideal` profile is an exact identity, which is why the off arm of this seam is
`ideal` and not "a calibrated profile with a small launch count".

## Physical sanity, three independent angles

The napkin bounds were written down before any digit was read.

**Compute and memory physics.** The step streams 553,654,272 bytes of
resident weights and LM head, plus 49,152 bytes per context token. Over the
B100 envelope's 8.0e12 bytes/s that is 69.21 us, and 98.87 us at the frozen
0.7 derate. Every record's measured provider service landed between
98.972160 us and 99.112594 us, just above that floor, and every one was
memory bound. R0 checked all nine to the picosecond.

**Network and serialization physics.** The frozen floor was 48 collectives
times one 2 us propagation, i.e. 96 us, and the ceiling was 1.68 ms. The
measured fabric term is 109.763 us for a one-token step and 261.151 us for
the twelve-token step, inside the bracket and near its floor. It is also fully
explained by hand: one collective sends `new_tokens * 2,048` bytes to each of
7 peers, so at 400 Gbit/s the closed form is
`48 * (new_tokens * 2,048 * 7 * 20 ps + 2,000,000 ps)`. That predicts
109,762,560 ps and 261,150,720 ps against measured 109,762,608 ps and
261,150,768 ps: a constant residual of 48 ps, i.e. exactly 1 ps per
collective, which is the fluid model's integer rounding. The covariate scales
correctly too: going from 1 to 12 new tokens adds
`11 * 2,048 * 7 * 20 * 48 = 151,388,160` ps, and the measurement added
151,388,160 ps.

**End-to-end plausibility.** Published single-stream decode of a 1B-class
model on a datacenter GPU runs at roughly 100 to 400 tokens per second, i.e.
2.5 to 10 ms per step. The `ideal` arm's 208.7 to 360.1 us implies 2,800 to
4,800 tokens per second, 12 to 48 times too fast. `graph440` at 465.9 to
617.2 us is 5 to 21 times too fast, and `eager567` at 1.450 to 1.602 ms is
1.7 to 6.9 times too fast. The direction is the pre-registered one: adding
the host term moves the model toward the plausible range without reaching it,
which is what should happen while scheduler Python, sampling and small-batch
inefficiency remain unmodeled. No cell came out faster than the ideal arm and
none came out slower than 5 ms, the two outcomes that would have indicated a
defect.

## What the composed step is made of

The most frequent record shape is a one-token decode step (four of the nine
records). Decomposed:

| term | `ideal` | `graph440` | `eager567` |
|---|---:|---:|---:|
| host launch demand, GTX 1660 Ti point times vLLM 0.26.0 count | 0 ps | 356,095,000 ps (76.44%) | 1,340,533,000 ps (92.43%) |
| packet-level fabric service, actually simulated | 109,762,608 ps (52.58%) | 109,762,608 ps (23.56%) | 109,762,608 ps (7.57%) |
| modeled B100 compute, exposed | 98,976,000 ps (47.42%) | 0 ps (0.00%) | 0 ps (0.00%) |

Once the transferred launch demand is selected it masks the modeled compute
completely, exactly as the vLLM composed study found on its own chain: the
step is 76 or 92 percent one transferred constant, and the B100 roofline this
repository spent a milestone building contributes nothing at all. That is the
most useful thing this study reports, and it is an argument for measuring
SGLang's own launch count (SGL-24) rather than for trusting these magnitudes.

One difference from the vLLM composed study is worth naming: this study
selected no calibrated collective latency profile, so its fabric term is
entirely simulated packets rather than a second transferred constant. Had the
calibrated width-8 intercept been selected, 48 collectives would have carried
`48 * 30,128,029 = 1,446,145,392` ps of base latency, which would dominate
even the eager launch floor. Isolating the host term was deliberate; the
collective floor on this chain belongs to other work.

## Fatal guards

All seven held, so the run is not void. They are recorded as held or violated
and never as a fraction.

| guard | what it asserted | outcome |
|---|---|---|
| G1 | the input stream hashes to the frozen digest and holds 9 records | held |
| G2 | the off arm is an exact identity against the pre-seam construction | held |
| G3 | every enabled cell reports the Turing device key and the b100 envelope | held |
| G4 | every record simulated, positive latency, completion equals arrival plus latency | held |
| G5 | the provider service of a record is identical in all seven cells | held |
| G6 | no SGLang import | held |
| G7 | no record carries an exact sampled count, because the stream predates SGL-12 | held |

## Three qualifications a harsh reader should raise, raised here

**R1 was less risky than the freeze implied.** The freeze said R1 tested that
the layer computes and the 48 collectives "serialize without the host term
perturbing the fabric schedule". They do serialize, but by construction rather
than by simulation: each rendered artifact is one collective, and
`HtsimStepSink._execute_plan` composes the step as
`represented_compute_ps + sum(collective services)`
(`simllm/backends/step_sink.py`, lines 1151 to 1159). htsim executes the
collectives; it never sees the compute. So of R1's three sub-claims, that the
selected host term reaches the sink's step-latency authority unchanged and
that the enclosure is the lowerer's own were genuinely at risk, while the
additive composition was not. The fabric invariance was still measured rather
than assumed: the fabric residual is bit-identical in all seven cells and in
the reference arm.

**G2's artifact half is uninformative in this lowering.** Every cell rendered
byte-identical artifacts, all 1,296 files rolling up to the same SHA-256,
because the GOAL input is a function of the record and the geometry only. That
is a real property worth recording, but it means artifact identity cannot
distinguish the ideal arm from a calibrated one. The informative half of G2 is
the per-step value equality against the hand-built pre-seam sink, which held.

**Nothing here was measured on SGLang, and no live scheduler was in the
loop.** The per-launch point is a GTX 1660 Ti capture with an AMD Ryzen 9
3950X host, the launch count is a static enumeration of vLLM 0.26.0 sources,
and the compute envelope is B100: a three-source device hybrid, in the same
sense the vLLM composed study called its own number a three-device chimera.
The record stream is a recorded schedule replayed against a declared
geometry, the same what-if `examples/m4` check E ran. This study demonstrates
that the SGLang chain can now select a per-step host cost and that the term
arrives where the closed form says it should. It does not claim the term is
the right size for SGLang. SGL-24 owns the launch count, SGL-26 owns the live
in-process selection, and neither is claimed here.

## Reproducing

```bash
python examples/sglang_host_step_v1/run_study.py --check-only
python examples/sglang_host_step_v1/run_study.py --run-dir <writable directory>
```

The first command re-derives every frozen literal in `expectations.json` from
the frozen constants and imports no simllm code. The second needs
`SIMLLM_HTSIM_RNIC` and `SIMLLM_TXT2BIN`, writes one workdir per cell and one
`results.json`, and prints the scored summary. The tracked `results.json` is
that file with the per-artifact digest list replaced by a per-cell rollup, so
the repository carries no bulk run output.
