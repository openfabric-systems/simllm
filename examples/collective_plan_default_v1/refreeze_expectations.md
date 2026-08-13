# Collective plan lowering default, transport refreeze

Date: 2026-08-13

This expectations-only supplement corrects one invalid physical literal in
[expectations.md](expectations.md) after the first result-producing run went
**void**. It was written before the corrected placement was executed and
before any new measurement existed. The void run's record and its raw
observations stay unmodified.

## What the void run refuted

The frozen bound treated every directed collective byte of the live arm as
fabric traffic charged at the swept `rnic_rate_bps`. The live arm placed the
eight-rank group at semantic ranks 0 through 7. The coarse device profile maps
a global rank to `(node, gpu) = divmod(rank, 8)`, so all eight ranks sat on
node 0. `CoarseDeviceRuntime` routes a same-node semantic send over NVLink at
the profile's fixed `nvlink_rate_bps`, which the sweep never varied.

Every live cell therefore moved 100 percent of its bytes on a 900 Gbit/s
NVLink port that no swept parameter touched. Two registered facts failed as a
direct consequence:

- **E7 physical bounds.** The measured 370,655,040 ps prefill step latency sits
  below the frozen 656,719,680 ps floor at 400 Gbit/s, because that floor
  charged 27,869,184 bytes of endpoint load to the fabric rate. Against the
  transport the model actually used, the same load costs 247,726,080 ps and the
  measured network term of 271,319,040 ps is above it. The defect was in the
  bound, not in the mechanism.
- **D live inverse-rate.** Both ratios came out at exactly 1.0. A rate sweep
  cannot test serialization scaling when the swept rate binds nothing.

The refuted literal is a physical bound, so the run is void and publishes no
behavioral fraction. Families A, B and C were evaluated from raw records
before the bound was consulted and their truth values are retained as findings
of a void run. They are not a score.

## What this supplement changes, and what it must not

Two things change, both of them corrections of an invalid physical literal:

1. **The live arm's placement is pinned.** The eight-rank group becomes
   `(0, 8, 16, 24, 32, 40, 48, 56)`, one rank per node under the repository's
   standing reference configuration, where intra-node traffic is NVLink and
   stays off the fabric. This is the same one-rank-per-RNIC discipline the
   TRAF-14 study already used for `(0, 8, 16, 24)`. The original freeze named
   the step and the rates but left the placement unstated, which is the gap
   this closes.
2. **The bound formula becomes transport aware.** Each directed extent is
   charged to the link the model actually selects for it:

```text
fabric(e)   = extents whose endpoints sit on different nodes
nvlink(e)   = extents whose endpoints sit on the same node
floor_ps    = compute_ps
            + max over endpoints of max(
                  ceil(fabric_load(e) * 8 * 1e12 / rnic_rate_bps),
                  ceil(nvlink_load(e) * 8 * 1e12 / nvlink_rate_bps))
ceiling_ps  = compute_ps
            + ceil(fabric_bytes * 8 * 1e12 / rnic_rate_bps)
            + ceil(nvlink_bytes * 8 * 1e12 / nvlink_rate_bps)
            + 1000 * message_count
```

With the pinned one-rank-per-node placement every extent is cross-node, so the
NVLink terms are zero and the formula reduces to the original one. The general
form is written down anyway so the bound cannot silently mis-model a placement
again.

Nothing else changes. In particular:

- The registered inverse-rate band stays `[1.95, 2.05]`, unchanged and not
  widened after seeing a ratio of 1.0.
- The four scored families, their twenty instances and their predicates are
  unchanged.
- Every exact oracle E1 through E6 is unchanged.
- The 559-byte absent-plan wire anchor is unchanged.

## Fatal and failed outcomes

Unchanged from the original freeze. A violated fatal guard voids the rerun and
closes nothing. A missed scored band is a failed run that publishes its
fraction, and is never converted into a fatal guard or refrozen afterwards.

## Registered command and dry run

```bash
.venv/bin/python examples/collective_plan_default_v1/run_study.py \
  --out "$SIMLLM_WAVE10_RUN_ROOT/collective_plan_default_v1-transport-refreeze" \
  --granite-root "$SIMLLM_GRANITE_REPLAY_ROOT"
```

Before this expectations-only commit the complete command is run with
`--check-only`, which validates only the frozen registries and arithmetic,
imports no SimLLM module, reads no external artifact and writes nothing.
