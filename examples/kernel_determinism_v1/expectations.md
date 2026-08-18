# Kernel determinism v1 expectations

This expectations-only change precedes every line of the COMP-42 implementation
and every run of this study. The results report must cite the commit that first
contains this file. No implementation, generated row or measured value is part
of this freeze. Every number below is either an integer derived by hand from a
closed form stated here and checked with a calculator that does not import
`simllm`, or a value already published by an accepted study and quoted as an
identity target.

## What is being frozen

The maintainer ruled on 2026-08-18 that a compute kernel's service time is a
deterministic constant with no tail: a pure function of four inputs, namely the
kernel family, the phase (prefill or decode), the token and shape inputs, and
the architecture profile. It is identical across ranks and across GPU runners
for the same inputs. Memory-bound kernels are pinned to the HBM bound.
CUDA-graph versus normal launch differs only in the host launch cost, never in
kernel service time. Latency tails emerge from the network, batching and
queueing, never from per-kernel stochasticity. Collective work is the declared
exception and is owned elsewhere.

This study freezes the observable consequences of that ruling before the code
that enforces them is written. It covers three things:

1. the roofline pricing path (`RooflineProvider` over
   `simllm.compute.transformer.step_kernel`), which is what both frontend
   adapters price a step through today;
2. the mechanistic path (`SmSchedulerModel` in `simllm.compute.gpu_model`) and
   its one flat HBM cursor;
3. the port taxonomy's fail-closed discipline, extended to UALink beside xGMI.

Out of scope, explicitly: collectives (NCCL and RCCL), the network, batching and
queueing. Those are where a tail is allowed to come from, and none of them is
measured here. Also out of scope: any claim about silicon. Every fixture below
is synthetic and exists so the closed form can be checked by hand.

## The distinction this study must not blur

The ruling constrains the *function*, not the *shape assignment*. Two ranks may
legitimately carry different shape inputs. The clearest case already in the
repository is uneven expert parallelism: vLLM spreads global experts over the
expert-parallel world and gives the low ranks the remainder
(`expert_map_manager.py` behavior mirrored in
`simllm/adapters/vllm/executor.py`), so with 30 experts over 8 ranks, ranks 0
to 5 own 4 experts and ranks 6 and 7 own 3. Those ranks hold different weight
bytes, so their decode steps legitimately cost different amounts.

That is not a violation of the ruling and this study does not treat it as one.
What the ruling forbids is a *provider* that keys on a rank, a worker id or an
adapter identity: given the same `ModelDims`, the same `StepRecord`, the same
provider, the same `GpuSpec` and the same host profile, the picoseconds must be
identical no matter who asked. The invariant is stated over inputs, and the
per-rank geometry is one of the inputs.

## Fixtures

### Fixture R, the roofline step

A dense per-rank geometry small enough that every product below is checkable by
hand:

```text
num_layers        = 2
hidden_size       = 128
intermediate_size = 256
num_heads         = 4
num_kv_heads      = 2
head_size         = 32
vocab_size        = 1024
dtype_bytes       = 2
```

which gives, from the closed forms in `simllm/compute/transformer.py`:

```text
q_dim            = num_heads * head_size                       =    128
kv_dim           = num_kv_heads * head_size                    =     64
attention_params = (hidden * (q_dim + 2*kv_dim)
                    + q_dim * hidden) * num_layers             = 98,304
mlp_params       = 3 * hidden * intermediate * num_layers      = 196,608
weight_bytes     = (attention_params + mlp_params) * 2         = 589,824
lm_head_bytes    = hidden * vocab * 2                          = 262,144
```

The GPU envelope is a synthetic device whose nameplates are exact powers of two,
so that the provider's IEEE-754 double arithmetic is exact and the frozen
integers cannot be missed by a one-unit truncation artifact:

```text
GpuSpec(name="kernel-determinism-fixture",
        peak_flops     = 2**48 = 281,474,976,710,656 FLOP/s  (281.5 TFLOP/s)
        mem_bandwidth  = 2**41 =   2,199,023,255,552 bytes/s (2.199 TB/s))
RooflineProvider(efficiency = 0.5)
```

Both nameplates sit inside the range of real datacenter parts (an A100 is
312 TFLOP/s dense BF16 and 2.039 TB/s), so the napkin math below is meaningful
even though no silicon claim is made. The 0.5 derate is a fixture choice, not a
measured efficiency. With `efficiency = 0.5` the effective roofs are exactly
`2**47` FLOP/s and `2**40` bytes/s.

The provider's closed form is

```text
t_compute = flops / (peak_flops * efficiency)
t_memory  = bytes_moved / (mem_bandwidth * efficiency)
bound     = "compute" if t_compute >= t_memory else "memory"
duration_ps = int(max(t_compute, t_memory) * 1e12)
```

and the host profile is `HostInitiationModel.ideal()`, which contributes
exactly zero.

Four step cells:

| Cell | Phase | Requests | New tokens each | Context each | Sampled |
|---|---|---:|---:|---:|---:|
| R1 | decode | 2 | 1 | 64 | 2 |
| R2 | decode | 4 | 1 | 64 | 4 |
| R3 | prefill | 1 | 512 | 512 | 1 |
| R4 | prefill | 1 | 1024 | 1024 | 1 |

### Fixture M, the mechanistic HBM cursor

The accepted synthetic 1 GHz fixture of
[gpu_task_mix](../gpu_task_mix/RESULTS.md), imported from its harness so the
architecture is that study's own object and not a lookalike: `HBM_LATENCY = 100`
cycles, one flat HBM cursor, `memory_launch(warps=8, per_warp=4, ...)`, i.e. 32
HBM load instructions. The transaction size is new: 192 bytes, which appears in
no accepted cell of that study. Cursor bandwidths 64 and 32 bytes per cycle.

The closed form that study already published for this launch shape is

```text
duration_cycles = 32 * ceil(transaction_bytes / bandwidth) + 100
duration_ps     = duration_cycles * 1000
```

### Fixture A, the two adapter geometry readers

One dense config object per adapter, shaped to fixture R's geometry, driven
through `model_dims_from_vllm_config` and `model_dims_from_sglang` at tensor
parallel size 1 with no quantization and an `auto` KV-cache dtype. Neither
config may default any geometry field, so both readers must report
`defaulted_fields == ()`.

## Physical bounds, stated before any measurement

Napkin math first, digits second.

- **Decode floor.** A decode step cannot beat its own weight and cache read.
  Cell R1 must stream 589,824 weight bytes plus 262,144 LM-head bytes plus
  65,536 KV bytes, i.e. 917,504 bytes. At the derated 2^40 bytes per second that
  is 834,465 ps, and at the full 2^41 nameplate with no derate it would be
  417,232 ps. The measurement must sit at the derated figure exactly; anything
  below the undercut nameplate figure is proof of a defect.
- **The decode pin is not a knife edge.** R1's compute term is 13,023.96 ps
  against a memory term of 834,465.03 ps, a factor of 64. A classification that
  came out "compute" here would not be a rounding accident, it would be a bug.
- **Prefill ceiling.** Cell R3 does 436,469,760 FLOP. At the derated 2^47 FLOP
  per second that is 3,101,304 ps, against a memory term of 1,013,278 ps, so it
  is compute bound by 3.06 times. A prefill chunk cannot be faster than its own
  arithmetic.
- **Prefill is superlinear in chunk length.** Doubling the chunk from 512 to
  1024 tokens multiplies the projection term by 2 and the attention score and
  value term by 4, because the query-key pair count is quadratic. The predicted
  FLOP ratio is 1,141,112,832 / 436,469,760 = 2.6144, and the predicted time
  ratio is the same 2.6144 because both cells are compute bound. A measured
  ratio of 2.0 would mean the quadratic term is missing; a ratio of 4.0 would
  mean the linear term is.
- **HBM cursor floor.** 32 loads of 192 bytes is 6,144 bytes. At 64 bytes per
  cycle no ordering of those loads can occupy the cursor for fewer than 96
  cycles, and halving the bandwidth to 32 exactly doubles that to 192, because
  192 is a multiple of both. The fixed 100-cycle return latency is additive and
  is not a bandwidth term.
- **System plausibility.** The fixture is a two-layer toy, not a deployment, so
  its absolute picoseconds carry no claim. What must match real serving
  behavior is the direction: decode memory bound, prefill compute bound, prefill
  superlinear in chunk length. All three are frozen above.

## Fatal guards

A single violated guard voids the run. None of these is survivable, and none is
ever reported as a fraction. Their evidence is retained and the owning task
stays open.

- **G1, repeat determinism.** For each of the four R cells and each of the two M
  cells, 64 repeated estimates through one provider instance must serialize to
  one byte string with zero variance.
- **G2, fresh-instance determinism.** The same six cells priced through a
  freshly constructed provider (and a freshly constructed `SmSchedulerModel`)
  must serialize to the same byte string as G1.
- **G3, host identity.** With `HostInitiationModel.ideal()` every R cell's
  composed duration must equal the provider duration exactly and its exposed
  host contribution must be zero, so nothing in this study is measuring a launch
  cost.
- **G4, no nondeterminism source.** No module in `simllm/compute` may import a
  random-number source, a wall clock or a process environment reader. The audit
  is over the module abstract syntax trees, not over a grep of comments.
- **G5, no rank in the provider surface.** No `ComputeProvider` pricing entry
  point, and no function in the shared step-cost model, may accept a rank, a
  worker id, a device index or an adapter identity as a parameter.
- **G6, UALink fails closed.** A UALink port must be rejected during
  configuration with a diagnostic that names the task owning vendor
  instantiation, exactly as an xGMI port already is, and the rejection must not
  depend on the port ever being used.
- **G7, accepted artifacts stay byte-identical.** `examples/gpu_task_mix`'s
  accepted artifacts and `examples/gpu_service_model/results.csv` must
  regenerate byte for byte with the UALink protocol landed. Adding a protocol
  name to an enum must move nothing.

Entailment note for G1 and G2, stated before the run: the determinism of a
deterministic function is entailed by its construction. These are pure Python
closed forms over frozen dataclasses with no RNG and no clock, so a variance
here would mean the construction assumption is false, not that a prediction was
wrong. That is precisely why they are guards and not scored rows. They are not
counted in any behavioral denominator and their passing earns no point.

## Controls

Unscored structural evidence whose job is to prove the checks above are not
vacuous. A control that fails to fail is itself a finding.

- **C1.** A provider built with a different efficiency must serialize to a
  different byte string on cell R1, so G1's byte equality is discriminating.
- **C2.** Changing one geometry field in one adapter's config must change the
  priced picoseconds, so the adapter agreement below is discriminating.
- **C3.** Halving the HBM cursor bandwidth must change the mechanistic duration,
  so the pin is a measurement and not a constant the harness prints.
- **C4.** The accepted artifact byte lock already carries its own mutation
  control in `tests/test_gpu_device_ports.py`; this study reuses it rather than
  inventing a second one.

## Scored denominator

Eight instances across four families. Every one is an exact integer predicted
above from a closed form stated above. The entailment answer is given per row
and is the reason the row is in or out of the denominator.

### Family A: phase and token keying move the constant (4 instances)

| ID | Cell | Predicted duration_ps | Predicted bound | Entailed by another row? |
|---|---|---:|---|---|
| A1 | R1, decode, 2 requests | 834,465 | memory | No. First exact prediction of this fixture. |
| A2 | R2, decode, 4 requests | 894,069 | memory | No. A model that ignored the KV read would give A1 again. |
| A3 | R3, prefill, 512 tokens | 3,101,304 | compute | No. A model that ignored the phase would keep the memory bound. |
| A4 | R4, prefill, 1024 tokens | 8,108,094 | compute | No. A step model linear in chunk length would give 2 x 3,101,304 = 6,202,608. |

Derived and therefore unscored: that A3 and A4 differ from A1 and A2 at all;
that the A4 over A3 ratio is 2.6144; that A2 minus A1 equals the 65,536-byte KV
delta. Each is arithmetic over rows already scored above.

### Family B: the memory-bound pin on the mechanistic HBM cursor (2 instances)

| ID | Cell | Predicted duration_cycles | Predicted duration_ps | Entailed by another row? |
|---|---|---:|---:|---|
| B1 | 192 bytes at 64 bytes per cycle | 196 | 196,000 | No. New transaction size; no accepted cell publishes 196. |
| B2 | 192 bytes at 32 bytes per cycle | 292 | 292,000 | No. Independent bandwidth. |

Derived and therefore unscored: that `(B2 - 100) == 2 * (B1 - 100)`. Once both
rows are scored the ratio cannot fail independently.

### Family C: the pin ignores SM count (1 instance)

| ID | Cell | Predicted duration_cycles | Entailed by another row? |
|---|---|---:|---|
| C1 | B1's launch on a two-SM architecture | 196 | No. A defect that let SM count leak into a memory-bound duration would pass B1 and fail here. This replicates the accepted `gpu_task_mix` B2 family at a new cell. |

### Family D: runner and adapter independence (1 instance)

| ID | Claim | Entailed by another row? |
|---|---|---|
| D1 | The vLLM and the SGLang geometry readers, given equivalent dense configs, produce the same integer geometry and the same `weight_element_bytes` and `kv_element_bytes`. | No. Two independent readers of two different config vocabularies can drift, and nothing else in this study would catch it. |

Derived and therefore unscored: that the two adapters then price cell R1 to the
identical 834,465 ps. Once D1 holds, both adapters call the same
`estimate_step_latency_ps` on equal inputs, so equal output is entailed by the
function being a function. It is reported as a derived row, not as a point.

Predicted representation difference, recorded now so it cannot be discovered
after the fact and presented as a prediction: the two readers will *not* produce
equal `ModelDims` dataclasses. Reading the two sources shows the vLLM reader
passes `weight_dtype_bytes` and `kv_dtype_bytes` as explicit floats resolved
from the quantization and cache configs, while the SGLang reader leaves both
`None` and lets `ModelDims` fall back to the activation width. Both spellings
resolve to 2.0 through `weight_element_bytes` and `kv_element_bytes`, so no
picosecond moves. D1 is therefore stated over the resolved widths, not over
dataclass equality. If this prediction is wrong in either direction it is a
finding and is reported as one.

## What a pass and a failure each mean

A pass means the ruling's observable consequences hold on these fixtures and
that the enforcement is discriminating. It does not mean any absolute duration
is calibrated, does not transfer to silicon, and says nothing about collectives,
the network, batching or queueing, which is where the ruling puts every tail.

A failure of any scored row means the closed form and the implementation
disagree, and the row's own prediction says which of the two is wrong. A
violated fatal guard voids the run: the behavioral score becomes uninterpretable
because the guard asserted the precondition under which the scored numbers mean
what they claim.

## Artifacts this study will produce

- `run_study.py`, the harness.
- `results.json`, schema `simllm-kernel-determinism-v1`, carrying every row with
  its evidence class, predicted value, measured value and residual.
- `RESULTS.md`, the report, citing this freeze's commit.
- `tests/test_kernel_determinism_study.py`, the lock, which regenerates
  `results.json` and requires byte equality with the committed artifact.
