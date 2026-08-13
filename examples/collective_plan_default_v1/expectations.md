# Collective plan lowering default expectations

Date: 2026-08-13

This is an expectations-only freeze for TRAF-28. It was written before the
lowering default was changed, before the bypass existed, and before any new
measurement was taken.

## What is being changed and why

TRAF-14 landed one immutable traffic-owned `CollectivePlan` and made it the
sole authority for algorithm, rank order, rounds, tags, channels, chunk sizes,
endpoint actions and directed extents **whenever it is present**. No shipped
lowerer attaches one, so every default graph still reaches the coarse runtime's
own reconstruction in `simllm/core/runtime.py`. Two expansions therefore exist
for exactly the graphs nobody opted in, and they can drift.

TRAF-28 makes the plan the default on the production lowering path so that the
reconstruction can be retired.

## Frozen seam

`SerialStepLowererConfig` gains one boolean field, and
`traffic.lower_step_observations` gains the matching keyword. Both default to
attaching the plan. The bypass is explicit and preserves the accepted
compatibility artifacts exactly.

```text
SerialStepLowererConfig(..., attach_collective_plan: bool = True)
lower_step_observations(..., attach_collective_plan: bool = True)
```

The plan is attached with the accepted `traffic.plan_execution_graph_collectives`
entry point. No second planner, tag allocator or expansion is authored.

## Frozen configurations

Two parameters vary, as the validation discipline requires.

- Link rate: 200 and 400 Gbit/s.
- Tensor-parallel width: 2 and 4 semantic ranks, placed on distinct coarse
  RNICs at `(0, 8)` and `(0, 8, 16, 24)`.

Both lowering paths are exercised at every cell: `SerialStepLowerer` with no
observations, and `ObservedStepLowerer` with observations, which reaches
`traffic.lower_step_observations`.

The live arm replays the real 54-token, 24-layer, EP-width-eight Granite
prefill step and the two decode steps that follow it, at both rates, rather
than the TRAF-14 three-byte sentinel.

## Exact oracles, fatal and unscored

- **E1 coverage.** Every default-lowered graph that contains a collective
  carries one plan per collective operation, and `validate_execution_graph`
  accepts it. Coverage is all or nothing, so a partially planned graph is not
  representable.
- **E2 bypass emptiness.** Every bypass-lowered graph has
  `collective_plans == ()`, and its v1 wire JSON omits the `collective_plans`
  key entirely.
- **E3 plan is the only difference.** For every cell,
  `replace(default_graph, collective_plans=()) == bypass_graph`. This is the
  clause that says the default changed the physical authority and nothing else
  about the lowered work.
- **E4 equivalence.** For every cell,
  `plan_execution_graph_collectives(bypass_graph) == default_graph`, and
  attaching twice is idempotent.
- **E5 integrity.** Every attached plan's canonical SHA-256 equals its
  recomputed integrity identity.
- **E6 legacy wire anchor.** The accepted absent-plan v1 graph still serializes
  to 559 bytes with SHA-256
  `f4a5a70f5bd4a0c2fed874baa88f3035266a54f386a59927e115872c2bcff0a3`, still
  omits the plan field, and still round-trips.
- **E7 physical bounds.** Every live completion sits inside the floor and
  ceiling stated below.

A violation of any of these voids the run.

## Scored behavioral relations

Four families and twenty instances are registered. Every relation is evaluated
from raw runtime records before any exact oracle above is applied.

**A, default and bypass runtime identity (8 instances).** Two rates times two
tensor-parallel widths times two lowering paths. The explicit plan scheduler and
the absent-plan reconstruction are separate implementations, so this equality is
a genuine risk rather than a construction. For each cell the following must be
identical between the default and bypass graphs:

```text
completed_at_ps, quiesced_at_ps, the full CompletionEvent tuple,
and every WQE tuple (operation, source, destination, payload, tag, channel,
submitted_at, eligible_at, started_at, finished_at, completed_at)
```

**B, perturbation rejection on the default path (6 instances).** The TRAF-14
negative-control family, now applied to graphs the shipped lowerer produced.
Both perturbations conserve total bytes.

1. Change one plan round's tag while retaining the plan's original integrity
   identity. Graph validation and runtime preflight must reject it with zero
   work requests submitted. Two instances, one per lowering path.
2. Change the semantic rank order carried by `CollectiveWork` from
   `(0, 8, 16, 24)` to `(0, 16, 8, 24)` while retaining the original plan. The
   participant set and total bytes are unchanged, so only a plan-authoritative
   runtime can see it. Two instances, one per lowering path.
3. Absorption control: the same rank-order change applied to the **bypass**
   graph must execute successfully with unchanged total bytes, demonstrating
   that the surrogate cannot see it. Two instances, one per lowering path.

**C, surrogate unreachability (4 instances).** The runtime's absent-plan
collective reconstruction is replaced by a sentinel that raises on entry. With
the sentinel installed, every default-path cell at both rates and both lowering
paths must still execute to completion, and every bypass cell must raise. Two
default instances and two bypass instances.

**D, live inverse-rate relation on the replayed Granite step (2 instances).**
Represented compute is rate independent and every remaining term is
serialization, so for the prefill step and for a decode step:

```text
(latency(200 Gbit/s) - compute_ps) / (latency(400 Gbit/s) - compute_ps)
    in [1.95, 2.05]
```

`compute_ps` is read from the lowered graph's own `ComputeWork` nominal
durations on the critical rank, not from a literal.

TTFT and TPOT are reported for every live cell at both rates and both modes,
and the default and bypass values must agree exactly, which is family A applied
to the reduced metrics.

## Physical sanity before any digit is read

For a live cell, define the total directed collective bytes `B`, the maximum
per-endpoint full-duplex load `L = max_e max(egress(e), ingress(e))`, and the
represented compute `compute_ps`.

```text
floor_ps   = compute_ps + ceil(L * 8 * 1e12 / rate_bps)
ceiling_ps = compute_ps + ceil(B * 8 * 1e12 / rate_bps) + 1000 * message_count
```

The floor is the single critical endpoint charged in its busier direction and
cannot be beaten. The ceiling serializes every byte of the group onto one
endpoint and adds a nanosecond of quantization per message, and cannot be
exceeded. Halving the rate must move the network term by two, which is exactly
what family D checks.

A live value outside those bounds is a defect in the model, the harness or the
reading, and it voids the run.

## Retirement question this study must answer

TRAF-28 exists so the runtime reconstruction can be deleted. The run must say
explicitly whether it is now deletable and, if not, exactly what still reaches
it. Family C establishes unreachability on the default path; it does not by
itself establish that no supported caller reaches the branch. The result
enumerates the remaining callers rather than asserting a deletion that the
evidence does not support.

## Fatal and failed outcomes

A violated fatal guard voids the run: no behavioral fraction is published and
TRAF-28 stays open with findings. If every fatal guard passes and a scored
relation misses, the run is failed, publishes its scored fraction, and TRAF-28
stays open. A missed band is never converted into a fatal guard and is never
refrozen after observation.

## Registered command and dry run

```bash
.venv/bin/python examples/collective_plan_default_v1/run_study.py \
  --out "$SIMLLM_WAVE10_RUN_ROOT/collective_plan_default_v1" \
  --granite-root "$SIMLLM_GRANITE_REPLAY_ROOT"
```

The study needs no native simulator: the coarse device runtime is the timing
authority for every cell. Before this expectations-only commit the complete
command is run with `--check-only`, which validates only the frozen registries
and arithmetic above, imports no SimLLM module, reads no external artifact and
writes nothing.
