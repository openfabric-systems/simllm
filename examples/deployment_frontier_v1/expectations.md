# CORE-62 analytical frontier and TRAF-68 bottleneck expectations

These expectations freeze the new deployment-frontier plot contract, the
closed-form roofline reference, and the exact two-network accounting identity
before implementation or packet observation. Their source is the maintainer
directive of 2026-08-27. CORE-61 remains the separate full-depth decode
extrapolation task; this work is CORE-62 and TRAF-68.

## Plot contract amendment

The batch-sweep frontier is a new figure. It does not alter the frozen ordered
axes or any record of the three prior flagship figures. Its contract is
`simllm-deployment-frontier-plot-contract-v2`:

- X is per-request decode speed in tokens per second per request. It is
  `10^12 / step_time_ps`, so 100 means 10 ms inter-token latency.
- Y is aggregate output throughput normalized per GPU in tokens per second per
  GPU. It is `batch_per_gpu * 10^12 / step_time_ps`.
- Both axes are logarithmic and increase toward the upper-right optimum.
- Batch per GPU sweeps through 1, 2, 4, 8, 16 and 32. The analytical reference
  is a line. The roofline simulation is a dot at each swept point.
- Published paired evidence retains the white diamond with black edge used by
  the flagship figure. The DeepSeek production decode value has no disclosed
  batch or context, so it is a dashed horizontal line and never an invented
  point.

The analytical line is a floor-style step-time bound. Its step time is the
worst declared floor, so a comparable real or simulated point must sit on or
below it. That interpretation appears in each figure caption.

## Analytical kernel reference

The sole work source is the content-addressed DeepSeek deployment projection
`ee154ed5...fd5a2.json`, unit `sglang-decode-ep72-dp-attention`, case
`sglang-decode-ep72-b32-c2000`, worst logical-HBM rank class
`rank-class-0`. At batch 32 and context 2,000 it declares exactly
3,594,330,365,632 FLOPs and 31,944,051,040 logical HBM bytes. The static HBM
term is 27,446,643,040 bytes. Both batch increments divide exactly:

```text
flops(B)     = 112,322,823,926 * B
hbm_bytes(B) = 27,446,643,040 + 140,544,000 * B
```

The kernel simulator is off. `RooflineProvider(efficiency=1.0)` prices these
quantities against the declared H100 or B100 peak envelope. The provider's
integer projection is retained exactly. Compute and memory floors are the
declared work divided by the corresponding envelope, converted to integer
picoseconds; kernel service is their maximum. There is no fitted quantity.

## Ideal network reference

Each batch item contributes exactly
`58 * 2 * 8 * 7168 * 2 = 13,303,808` logical collective bytes per GPU: 58 MoE
layers, dispatch and combine, top-8 routing, hidden width 7,168 and two-byte
elements. For a placement denominator `D`, local bytes are `total // D` and
remote bytes are the exact remainder. Remote bytes split in stable source-rank
order over the declared fan-in. Every byte must be conserved.

The ideal inter-node floor is the largest single flow's exact payload over the
nominal 400 Gbit/s link. It assumes zero contention, zero protocol overhead
and no propagation charge. The ideal intra-node floor is the largest transfer
payload over the nominal four-link 100 GB/s ordered-pair rate. It assumes zero
headers, credits, stalls, launch or completion overhead. The schedule permits
perfect overlap, so the analytical step is:

```text
A = max(compute_floor, memory_floor, ideal_fabric, ideal_intra_node)
```

Three configurations are frozen before observation: a B100 one-node
intra-node-heavy arm, an H100 two-node serialized split, and the comparable
H100 nine-node EP72 incast arm. These expected regime names are directions,
not pass conditions that can overwrite an observation.

## Exact consistency identity

The inter-node side runs through the packetized htsim `rnic-nn` profile. The
intra-node side runs through the existing three-module A100 NVLink3 candidate.
Where that candidate contributes, the result and figure must say that it is
candidate evidence used across architecture, not H100 or B100 measurement
evidence.

The gate freezes this inter-then-intra telescoping identity:

```text
A  = max(kernel, ideal_fabric, ideal_intra)
FI = max(kernel, simulated_fabric, ideal_intra)
S  = max(kernel, simulated_fabric, simulated_intra)

inter_node_attributed = FI - A
intra_node_attributed = S - FI
residual              = S - A - inter_node_attributed
                             - intra_node_attributed
```

Both attributed terms must be nonnegative and the residual must be exactly
zero at all 18 points. The fixed order prevents an off-critical medium from
being charged as elapsed time. Raw service excess for each medium is still
published even when it is masked.

For the fabric, the runner extracts an isolated largest-flow reference through
the same binary. The fixed 2,000,000 ps part of isolated excess is protocol,
remaining isolated excess is serialization, and concurrent excess over the
isolated result is incast. For the candidate intra-node profile, ideal-to-TX,
TX-to-switch and switch-to-RX completion deltas identify TX credits and
packetization, switch contention, and RX return respectively. The frozen
pass-through switch must contribute zero.

A nonzero residual is not absorbed. The run is published as `REFUTED` with
exact payload, logical-byte and wire-byte ledgers plus the first mismatch.
TRAF-69 is reserved for a fabric residual and COMP-77 for a compute or
composition residual.

## Bottleneck map and figures

Every point publishes the analytical and simulated step times, both elapsed
attributions, both raw network excesses, the fabric mechanism, the candidate
module and one binding classification. Inter-node binds when its simulated
service strictly exceeds kernel and intra-node service. Intra-node binds under
the symmetric rule. Neither is material when kernel service is at least both
network services. Ties are `co-critical` and list all owners.

The deployment figure is a 7 by 4.33 inch two-column figure. The TRAF-68 study
figure is a 7 by 6 inch two-panel figure: analytical lines and simulation dots
above, stacked inter-node and intra-node elapsed attribution below. Both render
to PDF and PNG from the same compact result.

## Preservation and closure

The complete 33-artifact preservation class from the third flagship freeze is
inherited by its exact digest, and ten third-run expectations, tools, records
and figures are added. All 43 artifacts are checked byte for byte. No prior
flagship runner is executed and no prior record or figure is modified.

CORE-62 closes only if all 18 residuals are zero and the versioned plot
contract renders literally. TRAF-68 closes only if all 18 points carry both
network terms, mechanism or module evidence and a bottleneck classification,
and the dedicated two-panel figure renders. Otherwise the result remains an
explicit refutation or the tasks stay open.
