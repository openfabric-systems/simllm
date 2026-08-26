# CORE-61 local depth-extrapolation result

Status: local derivation complete; CORE-61 remains open.

## Result first

The retained four-layer decomposition separates only `489 ps` of per-step
fixed service. The remaining `1,875,679,511 ps` is the four-layer repeatable
component, giving exactly `468,919,877.75 ps` per layer.

| Quantity | Exact or published value |
|---|---:|
| Measured four-layer service | 1.875680000 ms |
| Per-step fixed `F` | 489 ps, or 0.489 ns |
| Four-layer repeatable service | 1.875679511 ms |
| Per-layer repeatable `p` | 0.46891987775 ms |
| Existing linear 61-layer declaration | 28.604120000 ms |
| Separated 61-layer declaration, exact | 28.60411303175 ms |
| Separated 61-layer declaration, published | **28.604113032 ms** |
| Signed movement from linear, exact | -6.96825 ns |
| Absolute movement relative to linear | 0.243610 ppm |

The expected sign is present, but the magnitude is a null result for the
decode blocker. The fixed component is nonzero and only 0.260705 ppm of the
four-layer step. Avoiding its repeated multiplication changes the 61-layer
declaration by only 6.96825 ns. The retained decomposition does not support a
per-step fixed component large enough to materially change a millisecond-scale
decode step.

## Exact derivation

The retained component record reconstructs without error:

```text
compute_ps = ceil(3,066,736 x 10^12 / 1,635,000,000)
           = 1,875,679,511 ps
memory_ps  = 0 ps
F          = 489 ps
T(4)       = max(compute_ps, memory_ps) + F
           = 1,875,680,000 ps
p          = (T(4) - F) / 4
           = 468,919,877.75 ps per layer
```

Therefore:

```text
T_linear(61)    = 61 / 4 x T(4)
                = 28,604,120,000 ps
T_separated(61) = F + 61 x p
                = 28,604,113,031.75 ps
published       = 28,604,113,032 ps, round half up
```

The exact signed correction is `-57 / 4 x F = -6,968.25 ps`, matching the
frozen direction and identity.

## Evidence class and access

The corrected service is `DECLARED`: a derivation from a `MEASURED` service
decomposition at one depth. The component attribution retains its record class
`DISCLOSED` because GH200 counters were denied and the component method carries
an A100 bound. This is not a full-depth measurement, a second-depth validation,
or a promotion of the candidate record.

The field-addressed access selected JSON entry `entries[7]`, retained the exact
candidate key and the preregistered component fields, and consumed 21,700 of
57,417 record bytes. Its single access-ledger row reports `PASS`,
`whole_record_loaded=false`, and `unselected_values_decoded=false`. The
expectations-only access contract was committed first as `a6ba146`.

The exact candidate is vLLM `0.27.1+cu129`, DeepSeek-V3 reduced depth four at
revision `e815299b0bcbac849fa540c768ef21845365c9eb`, CUDA graph decode,
TP1/DP1/EP1 physical-envelope parallelism, batch 32, and 32 KV lengths of 2,000.
Its implementation identity retains the EP72 deployment-shape label. Routed
expert loads remain `not-captured` and were not invented.

## Calibration context

The published 22,282 tokens per second value was copied into the result only as
comparison context. It did not enter selection, component classification, or
arithmetic. No throughput-implied step is displayed.

## Registry movement and Merlin remainder

CORE-61 remains open because only one depth is measured. The local arm lands
the separated derivation and freezes the held-out eight-layer prediction at
**3.751359511 ms** for the identical batch-32, remote-KV-2000 shape. The
measured value and signed residual remain absent.

The second-depth base and decode submissions join COMP-72's resumable Merlin
remainder. They are forbidden before `2026-08-28T06:30` in `Europe/Zurich`, use
the same pinned DeepSeek revision and `gh-hourly` window, and write only below
the configured task-owned external root. The exact commands remain frozen in
`core61_depth_expectations.json` and are copied next to COMP-72's registry
commands.

Acceptance still requires the eight-layer prediction to be within 5 percent
of measured service and a signed residual ledger that keeps TRAF-66 overlap
separate. COMP-76 is unchanged. CORE-63 remains reserved and is not registered
by this null local result.
