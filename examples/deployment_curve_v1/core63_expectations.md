# CORE-63 decode expert-residency expectations

Status: expectations only. The retained component record and retained kernel
summary have not been opened for CORE-63.

## Frozen residency hypothesis and sign

The current decode basis is a tensor-parallel-one, expert-parallel-one
four-layer capture at batch 32. Each token visits top 8 routed experts, so the
capture represents 256 routed expert-token assignments on a rank.

The disclosed standard-decode workload has 256 tokens per node. Under the
uniform-routing assumption, stated exactly here before component access, those
2,048 assignments are uniform over the deployment projection's 288 physical
expert slots. EP72 places four physical slots on each rank. The expected work
for one rank is therefore:

```text
256 tokens/node x top 8 x 4 resident slots/rank / 288 slots = 256/9
```

Relative to the capture's 256 routed assignments, the routed-expert GEMM scale
is exactly `1/9`. No calibration value enters this factor. The expected signed
direction is a smaller decode step, higher standard-decode throughput and a
less-negative calibration residual before any possible crossing of the anchor.

Attention and MLA remain local under data-parallel attention. Router and top-k
work, the shared expert, the dense early layers, normalization and other
noncollective kernels also remain at scale one. Only recorded kernel rows whose
name contains `fused_moe_kernel` are classified as routed-expert GEMM work and
scaled. Generic GEMMs are deliberately retained so shared-expert or attention
work cannot be reduced by an ambiguous name.

## Frozen component composition

The candidate record's selected entry must reconstruct the retained four-layer
service from its compute-cycle, memory-service and fixed-overhead components.
The separately retained standard-decode kernel summary must reconstruct the
same noncollective service from `total_duration_per_step_ns` at a tolerance of
one picosecond. No held-out record is permitted.

CORE-61's validated fixed treatment is retained: the aggregate fixed component
occurs once per step. The remaining four-layer repeatable service is separated
into routed-expert and retained kernel families. The corrected full-depth step
is frozen as:

```text
T63 = F + 61/4 x (retained_4 + routed_4 / 9)
```

All arithmetic is exact rational arithmetic until the final published
picosecond, which is rounded half up. There is no free or fitted constant.

## Clean exposure and calibration-only comparison

The committed field reader is the only permitted record-access path. It reads
candidate entry `entries[7]` and streams only the standard-decode,
batch-32, device-0, noncollective rows from the byte-pinned retained kernel
summary. It logs both accesses, never loads a whole record, never decodes an
unselected CSV payload and records an empty held-out ledger.

Only after the corrected step exists may the calibration result compare it to
the published standard-decode anchor of 22,282 tokens per second per node. The
current 8,949.76 prediction and its 28,604,120,000 ps step are comparison
context and cannot select a component or tune a scale. The MTP anchor is owned
by the fifth scored run and must never be read or compared here.

## Decode-side overlap ruling

The current decode price is compute-only. No communication term enters it, so
overlap composition is not the binding CORE-63 mechanism. A decode-side overlap
task may be registered only after a decode communication service term exists;
CORE-63 does not derive one speculatively.

## Closure and preservation

CORE-63 closes only if the access protocol, reconstruction, architecture
arithmetic, signed direction, preservation lock and calibration-only
publication all hold literally. An overcorrection or undercorrection is
published with its signed residual rather than hidden. CORE-64 is reserved for
the exact residual if one remains.

The 93-entry preservation manifest covers the inherited scored-publication
class and the later decode-lineage records. Every entry must remain
byte-identical. No scored run, model-weight download or web access is allowed.
