# Expectations: the SGLang chain's per-step host cost (SGL-23)

Frozen before the seam exists and before any cell runs. Working tree at freeze
time: the SGLang host-model selector, its tests, the study harness and every
result file are absent; only this file, `expectations.json` and the SGL-23 to
SGL-26 registry entries in
[docs/modules/adapters-sglang.md](../../docs/modules/adapters-sglang.md) are
present. Nothing below was derived from a measurement.

## What is being tested and why

The SGLang chain prices a model step as roofline compute and nothing else. The
per-step host cost is exactly zero because
`examples/sglang_end_to_end_v1/run_study.py` line 650 builds
`HostInitiationModel.ideal()`, and no SGL task owns choosing anything else.
SGL-23 adds an owned selector. This study is its evidence: it replays a tracked
SGLang step-record stream through `simllm.backends.HtsimStepSink` with the
selector in seven states and checks that what arrives at the simulated step
latency is exactly what the closed form says should arrive.

The mechanism under test is `simllm.compute.host.HostInitiationModel`, whose
exposure rule is `F = max(C, N * g)` for provider service `C`, launch count `N`
and per-launch point `g` (`simllm/compute/host.py`, line 314). The overlap-aware
`max` is the whole point: a launch demand smaller than the compute it hides
behind must change nothing, and a launch demand larger than it must replace the
compute entirely. Neither half has ever been exercised on the SGLang chain.

## Honest provenance of the transferred constants

This study composes constants from three different places and none of them was
measured on SGLang. The disclosure has to be at least as blunt as the vLLM
chain's, which called its own composed number "a three-device chimera"
(`examples/composed_step_budget_v1/RESULTS.md`).

- The per-launch point `g` is a CUDA-graph node replay time (809,306 ps) or an
  eager host-bound launch (2,364,255 ps) measured on one GTX 1660 Ti with an
  AMD Ryzen 9 3950X host in `examples/host_step_cost_v1`. COMP-1 refuses to
  treat it as a calibration for any other device, and this study does not.
- The launch count `N` is a static enumeration of vLLM 0.26.0 sources for the
  pinned Granite MoE geometry, frozen as the bracket `[440, 567]`
  (`examples/compute_fidelity_v1/expectations.md`, line 310). SGLang's model
  runner, its fused MoE path and the pump's unrolled `event_loop_normal` issue
  their own launches, and nobody has counted them. Using vLLM's count on the
  SGLang chain is a third-party surrogate, registered as SGL-24.
- The compute envelope is B100 (`GPU_ENVELOPES["b100"]`), while the calibrated
  host profiles refuse every GPU key but `gtx1660-ti-sm75`. The selector
  therefore presents the Turing key to the host model and pins the provider to
  the B100 envelope, exactly as `examples/end_to_end_replay_v1/run_study.py`
  does. Every enabled row is a disclosed device hybrid.

Nothing here is a calibration, and no cell of this study is a prediction for
any deployment. The claim under test is that the selector composes the terms
the way the closed form says, not that the composed number is right.

## Inputs

The record stream is the tracked SGLang smoke capture
`examples/m4/fixtures/sglang-m3-steps.jsonl`, 9 records, SHA-256
`656772148cd8fbda71a25af08215d806f38f3886abb068f72c9e0ddc8cb7c26f`. It is a
schedule, not a geometry: it carries request identities, phases, token counts
and contexts and nothing about the model. It predates SGL-12, so no record
carries `num_sampled`, and every consumer reads the whole scheduled batch as
having sampled. That fallback is exact for this stream because chunked prefill
was disabled in the capture, so no scheduled row is a mid-prompt chunk.

The geometry is the per-rank Granite MoE geometry the SGLang end-to-end study
declares at expert-parallel width 8 (`examples/sglang_end_to_end_v1/run_study.py`,
`_dims(8)`): 24 layers, hidden 1,024, 16 heads, 8 KV heads, head size 64, vocab
49,155, 2 activation bytes, 32 routed experts, top-8, expert intermediate 512,
4 resident experts. Pairing a recorded schedule with a declared geometry is the
same what-if `examples/m4` check E ran, and it is labelled as one here.

The sink is `HtsimStepSink` on `rnic-nn-fluid` at 400 Gbit/s with `tp_ranks=(0,)`
and `ep_ranks=(0..7)`, i.e. the demo's expert-parallel-only topology with
uniform routing. `tp_ranks=(0,)` emits no tensor-parallel all-reduce, which
keeps this study clear of the routed-MoE all-reduce over-count that another
wave-14 task owns.

## Frozen physical constants

| quantity | value | source |
|---|---:|---|
| B100 peak dense FLOP/s | 1.8e15 | `GPU_ENVELOPES["b100"]` |
| B100 HBM bytes/s | 8.0e12 | `GPU_ENVELOPES["b100"]` |
| roofline derate | 0.7 | the study's frozen efficiency |
| CUDA-graph point `g` | 809,306 ps | `turing-cuda-graph` |
| eager host point `g` | 2,364,255 ps | `turing-eager-host` |
| attention parameters, all layers | 75,497,472 | `24 * (1024 * 2048 + 1024 * 1024)` |
| resident MoE parameters, all layers | 150,994,944 | `3 * 1024 * 512 * 4 * 24` |
| activated MoE parameters, all layers | 301,989,888 | `3 * 1024 * 512 * 8 * 24` |
| resident weight bytes | 452,984,832 | `(75,497,472 + 150,994,944) * 2` |
| LM head bytes | 100,669,440 | `1024 * 49,155 * 2` |
| KV bytes per context token | 49,152 | `2 * 24 * 8 * 64 * 2` |

## Napkin bounds, written before any digit is read

- **Compute floor.** A decode step cannot beat its own resident bytes over
  memory bandwidth. 553,654,272 bytes over 8.0e12 bytes/s is 69.21 us with no
  derate and 98.87 us at the frozen 0.7. Every record's provider service must
  land just above 98.87 us, and it must be memory bound: the largest record
  moves 9.17e9 FLOPs, which is 7.28 us of B100 compute, an order of magnitude
  under the memory term.
- **Network floor.** Each of the 24 layers emits an MoE dispatch and a combine,
  so 48 collectives sit between the layer computes. Nothing crosses the fabric
  faster than one propagation, so the network adds at least 48 * 2 us = 96 us
  to a step. The ideal arm's step latency therefore cannot be below about
  195 us.
- **Network ceiling.** The largest record sends 12 * 8 * 1,024 * 2 / 8 = 24,576
  bytes per pair. At 400 Gbit/s a pair takes 0.49 us of serialization. If every
  collective serialized all `2 (W - 1) = 14` rounds at one propagation plus its
  chunk each, one collective costs at most 34.9 us and 48 of them 1.68 ms. The
  ideal arm's step latency is therefore bracketed by [0.195 ms, 1.78 ms], and
  the eager-567 arm by [1.44 ms, 3.02 ms].
- **Covariate.** The term that must scale with the launch count is the launch
  floor itself, at slope exactly `g`. Going from 123 to 440 CUDA-graph launches
  must move every record's step by exactly 256,550,000 ps, the same number for
  every record, because the floor has replaced the compute in both cells.
- **System plausibility.** Published single-stream decode of a 1B-class model
  on a datacenter GPU runs at roughly 100 to 400 tokens per second, i.e. 2.5 to
  10 ms per step. The ideal arm's step of about 0.2 ms is 12 to 50 times too
  fast, which is the expected direction: the model omits the host, the
  scheduler and small-batch inefficiency. The eager-567 arm at 1.4 ms or more
  moves toward that range without reaching it. A cell that came out *faster*
  than the ideal arm, or slower than about 5 ms, would indicate a defect.

## Derived per-record provider service

Computed here from the frozen constants, not measured. `new` is the step's new
tokens, `kv` its context tokens, `pairs` its query-key pairs and `samp` its
sampled rows (the fallback, i.e. the scheduled row count).

| step | new | kv | pairs | samp | `C` (ps) |
|---:|---:|---:|---:|---:|---:|
| 0 | 12 | 12 | 72 | 1 | 98,972,160 |
| 1 | 1 | 13 | 12 | 1 | 98,980,937 |
| 2 | 7 | 26 | 67 | 2 | 99,095,040 |
| 3 | 2 | 28 | 26 | 2 | 99,112,594 |
| 4 | 1 | 14 | 13 | 1 | 98,989,714 |
| 5 | 11 | 25 | 64 | 2 | 99,086,262 |
| 6 | 2 | 27 | 25 | 2 | 99,103,817 |
| 7 | 1 | 12 | 11 | 1 | 98,972,160 |
| 8 | 1 | 13 | 12 | 1 | 98,980,937 |

Every one is memory bound, and every one sits between 98.97 us and 99.12 us,
i.e. just above the 98.87 us floor as required.

## The frozen sweep

Two parameters vary: the launch class and the launch count. The counts are not
round numbers, they are the crossover of the `max` rule. `C / g` is 122.3 to
122.5 CUDA-graph launches and 41.9 to 41.94 eager launches across the nine
records, so one launch either side of those points flips every record at once.

| cell | profile | `N` | launch floor `N * g` (ps) | regime |
|---|---|---:|---:|---|
| `ideal` | `ideal` | 0 | 0 | off arm |
| `graph122` | `turing-cuda-graph` | 122 | 98,735,332 | masked, all records |
| `graph123` | `turing-cuda-graph` | 123 | 99,544,638 | host bound, all records |
| `graph440` | `turing-cuda-graph` | 440 | 356,094,640 | host bound, all records |
| `eager41` | `turing-eager-host` | 41 | 96,934,455 | masked, all records |
| `eager42` | `turing-eager-host` | 42 | 99,298,710 | host bound, all records |
| `eager567` | `turing-eager-host` | 567 | 1,340,532,585 | host bound, all records |

`graph440` and `eager567` are the transferred vLLM bracket endpoints, so their
enclosed totals must reproduce the two numbers the vLLM composed study already
reported: 356,095,000 ps and 1,340,533,000 ps.

## The enclosure, and why the masked cells are not silent

GOAL represents whole nanoseconds, and the two arms quantize differently. With
`RooflineProvider` layer breakdown disabled, `SerialStepLowerer.timing` gives an
ideal arm `24 * floor(C / 24,000)` nanoseconds of layer compute, while a
calibrated arm encloses the whole step as `ceil(F / 1,000)` nanoseconds and puts
the shortfall in layer 0 (`simllm/backends/step_lowerer.py`, lines 205 to 246).

The consequence is registered here in advance because it is counterintuitive: a
fully masked calibrated cell is **not** identical to the ideal arm. It is longer
by `ceil(C / 1,000) - 24 * floor(C / 24,000)` nanoseconds, which is 5 to 24 ns
for these records. Only the `ideal` profile is an exact identity. Anyone reading
"the launch floor is below the compute so nothing changed" without this
paragraph would be wrong by up to 24 ns per step.

Frozen enclosed totals, in nanoseconds of layer compute per step:

| step | `ideal` | `graph122` | `graph123` | `graph440` | `eager41` | `eager42` | `eager567` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 98,952 | 98,973 | 99,545 | 356,095 | 98,973 | 99,299 | 1,340,533 |
| 1 | 98,976 | 98,981 | 99,545 | 356,095 | 98,981 | 99,299 | 1,340,533 |
| 2 | 99,072 | 99,096 | 99,545 | 356,095 | 99,096 | 99,299 | 1,340,533 |
| 3 | 99,096 | 99,113 | 99,545 | 356,095 | 99,113 | 99,299 | 1,340,533 |
| 4 | 98,976 | 98,990 | 99,545 | 356,095 | 98,990 | 99,299 | 1,340,533 |
| 5 | 99,072 | 99,087 | 99,545 | 356,095 | 99,087 | 99,299 | 1,340,533 |
| 6 | 99,096 | 99,104 | 99,545 | 356,095 | 99,104 | 99,299 | 1,340,533 |
| 7 | 98,952 | 98,973 | 99,545 | 356,095 | 98,973 | 99,299 | 1,340,533 |
| 8 | 98,976 | 98,981 | 99,545 | 356,095 | 98,981 | 99,299 | 1,340,533 |

## Scored relations

Two evidence classes, never summed.

### Exact-oracle rows (63)

**R0, 9 rows.** For every record, the provider service the sink reports equals
the `C` column above, to the picosecond, in every cell.

**R1, 54 rows.** For every record and every calibrated cell, the simulated step
latency the sink returns, minus the same record's simulated step latency in the
`ideal` cell, equals exactly
`1,000 * (enclosed_ns(cell) - enclosed_ns(ideal))` picoseconds, i.e. the frozen
delta table below.

| step | `graph122` | `graph123` | `graph440` | `eager41` | `eager42` | `eager567` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 21,000 | 593,000 | 257,143,000 | 21,000 | 347,000 | 1,241,581,000 |
| 1 | 5,000 | 569,000 | 257,119,000 | 5,000 | 323,000 | 1,241,557,000 |
| 2 | 24,000 | 473,000 | 257,023,000 | 24,000 | 227,000 | 1,241,461,000 |
| 3 | 17,000 | 449,000 | 256,999,000 | 17,000 | 203,000 | 1,241,437,000 |
| 4 | 14,000 | 569,000 | 257,119,000 | 14,000 | 323,000 | 1,241,557,000 |
| 5 | 15,000 | 473,000 | 257,023,000 | 15,000 | 227,000 | 1,241,461,000 |
| 6 | 8,000 | 449,000 | 256,999,000 | 8,000 | 203,000 | 1,241,437,000 |
| 7 | 21,000 | 593,000 | 257,143,000 | 21,000 | 347,000 | 1,241,581,000 |
| 8 | 5,000 | 569,000 | 257,119,000 | 5,000 | 323,000 | 1,241,557,000 |

### Behavioral relation instances (18)

**R2, 18 instances.** The regime flips exactly where the closed form puts it,
one launch wide. For all 9 records the CUDA-graph class reports the provider's
own bound with zero exposed host time at `N = 122` and reports
`bound == "host-initiation"` with positive exposed host time at `N = 123`; the
eager class does the same between `N = 41` and `N = 42`. Nine instances per
class pair.

## The entailment question

For each scored relation: given the guards already registered below, can it
fail?

- **R0 can fail.** The `C` column is an independent hand derivation of the
  roofline from the geometry. No registered guard pins the provider's answer to
  it; G5 only requires the answer to be the same in every cell. A slip in the
  derivation, or a provider that prices the Turing envelope instead of the
  pinned B100 one, breaks R0.
- **R1 can fail.** R1 asserts two things beyond R0: that the whole-nanosecond
  enclosure is the one the lowerer applies, and that the end-to-end simulated
  latency moves by exactly that enclosure and by nothing else, i.e. that the 48
  collectives and the layer computes serialize without the host term perturbing
  the fabric schedule. Neither is guarded. Overlap, a clamped layer duration, or
  a shortfall placed differently all break R1.
- **R2 can fail.** R2 is not entailed by R0 plus arithmetic, because the
  composition rule itself is under test on this chain for the first time.
  `HostInitiationModel` has a second, additive branch for non-calibrated
  constants (`simllm/compute/host.py`, lines 317 to 323); a selector that built
  the profile by hand instead of through the named factory would compose
  `C + N * g` and every masked cell would move. R2 is the relation that
  distinguishes the two.

Relations deliberately **not** scored because they cannot fail given the above:

- The slope in the launch count. `latency(graph440) - latency(graph123)` must be
  256,550,000 ps for every record, and `latency(eager567) - latency(eager42)`
  must be 1,241,234,000 ps. Both follow from R1 applied twice. They are reported
  as covariate checks and carry no score.
- The cross-class ratio of the launch floors, 2,364,255 / 809,306 = 2.9214, is
  fixed by the two frozen constants. Reported, not scored.
- The share of the composed step made of transferred constants. Reported,
  because it is the number a reader most needs, and unscored, because it is
  arithmetic over R1's outputs.

## Fatal guards, void and not scored

A violation voids the run for the purpose of closing SGL-23. None of these is
reported as a fraction.

- **G1.** The input stream's SHA-256 is
  `656772148cd8fbda71a25af08215d806f38f3886abb068f72c9e0ddc8cb7c26f` and it
  holds exactly 9 records with the frozen shapes.
- **G2.** The off arm is an exact identity. A sink built through the selector
  with `ideal` produces byte-identical rendered GOAL artifacts, identical
  `layer_calc_ns`, identical makespans and identical `StepResult` values to a
  sink built with the pre-seam explicit defaults (`HostInitiationModel.ideal()`,
  `GPU_ENVELOPES["b100"]`, `RooflineProvider(0.7)`).
- **G3.** Every calibrated cell reports `device_key == "gtx1660-ti-sm75"` and a
  `b100` provider envelope, so no enabled row can be read as device consistent.
- **G4.** Every record is simulated in every cell: the sink returns a
  `StepResult` for all 9 records in all 7 cells, with a positive latency and
  `completed_at_ps == virtual_time_ps + step_latency_ps`.
- **G5.** The provider service reported for one record is identical across all
  7 cells. This is the guard that makes R1's deltas mean "the host term did
  this" rather than "the device key moved the compute".
- **G6.** The study imports no SGLang and needs no GPU.
- **G7.** No record's sampled count is exact (`sample_count_exact` is false for
  all 63 rows), because the stream predates SGL-12. This is recorded so nobody
  later reads these rows as evidence about the sampled-identity path.

This freeze declares no survivable fatal guard. Any violation voids the run.

## What a refutation would mean

If R2 fails in the masked direction, i.e. the masked cells move by their full
launch floor, the selector is composing additively and SGL-23 does not close.
If R1 fails while R0 and R2 hold, the composition is right and the lowering or
the fabric schedule is not, which would be a finding about
`SerialStepLowerer`, not about the selector. If R0 fails, the geometry the
study declares is not the geometry the sink prices, which is the same class of
defect SGL-25 registers for the end-to-end study's control cell.
