# CORE-62 analytical frontier and TRAF-68 two-network result

## Gate verdict

**Accounting gate PASS; frozen bottleneck expectation REFUTED.**
All 18 residuals are exactly zero after the two frozen network terms. The maximum absolute unexplained residual is
**0 ps** across all 18 swept
points. Both attributed terms use the frozen inter-then-intra telescoping order;
no residual was absorbed. The overall study is `REFUTED` because the frozen
nine-node arm did not produce positive elapsed inter-node attribution.

This is a roofline replay of the declared disaggregated-session decode step,
not a live SGLang frontend run. Kernel simulation is off and
`RooflineProvider(efficiency=1.0)` is the only kernel price.

## Bottleneck map

The binding classifications are intra-node 1, neither 17. Dominant fabric mechanism rows
are incast 6, none 6, serialization 6. Dominant candidate-module rows are TX credits and packetization 18. A raw network
excess remains in [results.csv](results.csv) even when the roofline or the other
network masks it from elapsed step time.

The intra-node timing is cross-architecture candidate evidence. The existing
A100 NVLink3 three-module profile prices two independent four-endpoint domains
for each eight-GPU node. It is not H100 or B100 measurement evidence.

## Figures

- [Deployment frontier](figures/deployment-frontier.pdf) and
  [PNG](figures/deployment-frontier.png)
- [Two-network bottleneck attribution](figures/two-network-bottleneck.pdf) and
  [PNG](figures/two-network-bottleneck.png)

The analytical reference is a line and the roofline simulation is a dot at each
batch. Both frontier axes are logarithmic: X is per-request decode speed and Y
is aggregate output throughput normalized per GPU. Analytical lines are
floor-style step-time bounds, so comparable real and simulated points sit on or
below them. The published standard-decode measurement retains its white diamond
marker. The y-only production anchor is a dashed horizontal line because its
batch and context were not disclosed.

## Per-point accounting

Times are milliseconds except for the exact residual column.

| Configuration | B/GPU | Analytical | Simulated | Inter attributed | Intra attributed | Residual ps | Fabric mechanism | Candidate module | Binds |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| b100-one-node-intra | 1 | 3.448398 | 3.448398 | 0.000000 | 0.000000 | 0 | none | TX credits and packetization | neither |
| b100-one-node-intra | 2 | 3.465966 | 3.465966 | 0.000000 | 0.000000 | 0 | none | TX credits and packetization | neither |
| b100-one-node-intra | 4 | 3.501102 | 3.501102 | 0.000000 | 0.000000 | 0 | none | TX credits and packetization | neither |
| b100-one-node-intra | 8 | 3.571374 | 3.571374 | 0.000000 | 0.000000 | 0 | none | TX credits and packetization | neither |
| b100-one-node-intra | 16 | 3.711918 | 3.711918 | 0.000000 | 0.000000 | 0 | none | TX credits and packetization | neither |
| b100-one-node-intra | 32 | 4.257219 | 4.523298 | 0.000000 | 0.266080 | 0 | none | TX credits and packetization | intra-node |
| h100-two-node-serialized | 1 | 8.234981 | 8.234981 | 0.000000 | 0.000000 | 0 | serialization | TX credits and packetization | neither |
| h100-two-node-serialized | 2 | 8.276935 | 8.276935 | 0.000000 | 0.000000 | 0 | serialization | TX credits and packetization | neither |
| h100-two-node-serialized | 4 | 8.360842 | 8.360842 | 0.000000 | 0.000000 | 0 | serialization | TX credits and packetization | neither |
| h100-two-node-serialized | 8 | 8.528655 | 8.528655 | 0.000000 | 0.000000 | 0 | serialization | TX credits and packetization | neither |
| h100-two-node-serialized | 16 | 8.864283 | 8.864283 | 0.000000 | 0.000000 | 0 | serialization | TX credits and packetization | neither |
| h100-two-node-serialized | 32 | 9.535538 | 9.535538 | 0.000000 | 0.000000 | 0 | serialization | TX credits and packetization | neither |
| h100-nine-node-incast | 1 | 8.234981 | 8.234981 | 0.000000 | 0.000000 | 0 | incast | TX credits and packetization | neither |
| h100-nine-node-incast | 2 | 8.276935 | 8.276935 | 0.000000 | 0.000000 | 0 | incast | TX credits and packetization | neither |
| h100-nine-node-incast | 4 | 8.360842 | 8.360842 | 0.000000 | 0.000000 | 0 | incast | TX credits and packetization | neither |
| h100-nine-node-incast | 8 | 8.528655 | 8.528655 | 0.000000 | 0.000000 | 0 | incast | TX credits and packetization | neither |
| h100-nine-node-incast | 16 | 8.864283 | 8.864283 | 0.000000 | 0.000000 | 0 | incast | TX credits and packetization | neither |
| h100-nine-node-incast | 32 | 9.535538 | 9.535538 | 0.000000 | 0.000000 | 0 | incast | TX credits and packetization | neither |

## Frozen direction checks

- PASS: per-request speed is nonincreasing with batch
- PASS: aggregate per-GPU throughput is nondecreasing with batch
- MISS: nine-node incast has positive elapsed inter-node attribution
- PASS: pass-through candidate switch attribution is zero
- PASS: at least one point is roofline-bound with neither network material

## Provenance and preservation

- Expectations commit: `a7c86086dff5b3039cb84cfe0fa84875404f397d`
- Expectations SHA-256: `54295c81cebe36ee32d12b8ab1432c9fc060094ddf98403152b0d619cc37438f`
- Implementation run commit: `227e24dac2c786cbf937bbbf0460b001fae3be23`
- htsim rnic-nn binary SHA-256: `388415f92d6ef54c84bb5d2b7f7dabcaad27574ec235d62260f08175f3958bd9`
- txt2bin SHA-256: `f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b`

All 43 artifacts in the expanded
preservation class remained byte-identical. No prior flagship runner was
invoked and no prior record or figure was rewritten. TRAF-69 and COMP-77 remain
reserved for a fabric residual or a compute and composition residual.
