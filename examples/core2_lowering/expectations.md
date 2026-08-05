# CORE-2 lowering and wire replay: pre-registered expectations

Written and frozen before the CORE-2 implementation and before any run of
this study. The graph replay must consume only a graph after its JSON round
trip. It must not inspect the source `StepRecord` or call `HtsimStepSink`.

## System under test

One synthetic scheduler step uses the existing small dense geometry from the
step-sink tests:

- two transformer layers;
- hidden size 1024, intermediate size 4096;
- eight attention and KV heads of width 128;
- vocabulary size 32000 and two-byte elements;
- one 256-token prefill request at context length 256;
- roofline efficiency 0.7 on the B100 envelope;
- the `rnic-nn-fluid` profile with propagation P = 2,000,000 ps.

The two swept parameters are tensor-parallel width W in `{2, 4}` and endpoint
link rate R in `{200, 400}` Gbit/s. The model geometry stays fixed across W so
this study isolates graph and network replay rather than claiming a realistic
resharding study.

The existing compute model gives whole-step estimate E = 24,061,074 ps.
The diagnostic GOAL path splits it using
`c = floor(E / (L * 1000)) = 12,030 ns` per layer, so graph replay must carry
critical-path compute K = 24,060,000 ps. The 1,074 ps difference is the
already documented BACK-5 whole-nanosecond truncation, not an unexplained
residual.

Each of the two TP allreduces per layer moves
`S = 256 * 1024 * 2 = 524,288` bytes. A ring over W ranks has `2(W-1)`
strictly chained rounds, each moving chunk `S/W` at full endpoint rate.

## Check A: graph sufficiency and wire identity

For every W, lowering must produce a valid graph whose operation IDs are
unique, whose dependencies and completion IDs resolve, and whose explicit
plus implicit FIFO edges are acyclic. Its JSON encoding and decoding must
round-trip to exact dataclass equality.

The serial phase order on every participating rank is:

```text
layer compute -> attention ring allreduce -> MLP ring allreduce
              -> next-layer compute
```

Every allreduce must retain its layer correlation, participants, 524,288-byte
payload and ring algorithm hint. With no adapter observations, the graph must
contain no KV, DMA or control work. For fixed W, the graph and its canonical
JSON must be identical at both link rates because link service belongs to the
backend, not lowering.

## Check B: dense 2 by 2 JCT sweep

The frozen closed form is:

```text
J(W,R) = K + 2L * 2(W-1) * ((S/W) * 8e12/R + P)
```

| W | R Gbit/s | expected JCT ps | expected flows |
|---:|---:|---:|---:|
| 2 | 400 | 82,003,040 | 16 |
| 2 | 200 | 123,946,080 | 16 |
| 4 | 400 | 134,974,560 | 96 |
| 4 | 200 | 197,889,120 | 96 |

Three independent values must agree in every cell:

1. The current `HtsimStepSink` JCT.
2. JCT from graph-only GOAL replay after JSON serialization and parsing.
3. The frozen closed form above.

Acceptance is exact:

- legacy minus frozen: 0 ps;
- graph replay minus frozen: 0 ps;
- graph replay minus legacy: 0 ps;
- physical quiescence verified in both backend runs;
- sorted flow ledgers identical in source, destination, tag, payload, start,
  completion and FCT;
- flow count equal to the table.

Define serialization-only time:

```text
Q(W,R) = J(W,R) - K - 2L * 2(W-1) * P
```

The parameter relations must hold exactly:

- `Q(W,200) = 2 * Q(W,400)`;
- `Q(4,R) = 1.5 * Q(2,R)`;
- doubling rate does not halve total JCT because compute and propagation stay
  fixed.

Any nonzero timing residual, flow-ledger difference, graph difference across
rates, or failed relation is a bug. No tolerance band is registered for the
deterministic fluid backend.

## Check C: MoE compatibility sentinel

The current serial path also supports expert-parallel traffic. One sentinel
uses the existing two-layer small MoE geometry, a two-request decode step,
TP width 1, EP ranks `{0, 1, 2, 3}` and 400 Gbit/s. Each dispatch and combine
phase sends 2,048 bytes per ordered rank pair. The fluid allocator grants
each of the three simultaneous sends per endpoint
`floor(400,000,000,000 / 3)` bit/s and rounds serialization upward to whole
picoseconds.

Frozen acceptance:

- legacy JCT: 25,811,524 ps;
- graph-replay JCT: 25,811,524 ps;
- residual between either path and the frozen value: 0 ps;
- 48 flows in each ledger;
- sorted flow ledgers identical field for field;
- physical quiescence verified.

This is a compatibility sentinel, not a parameter sweep. Check B supplies the
required two-parameter experiment.

## WQE bookkeeping addendum, frozen before the backend change

The read-only audit at pinned HTSIM commit `70151bc` found no explicit RNIC
SQ/RQ/CQ layer, DCQCN QP table or `rnic-cn` directed link-pair table. The
backend addition is required to be timing-neutral: SQ post and dispatch occur
at the existing send timestamp, while CQ post and immediate virtual-CPU
consumption occur at the existing WQE completion timestamp. RQ is a zero-use
placeholder. No packet event is exported.

Consequently, every JCT and full flow ledger above must remain exactly
unchanged. The public completion row count is also the WQE count: 16, 16, 96
and 96 for the dense matrix, and 48 for the MoE sentinel. Aggregate queue
counters must show one SQ post and dispatch plus one CQ post and consume per
row, zero RQ activity and zero final queue depth. The fluid baseline reports
transport kind `none`. In focused backend tests, same-direction WQEs reuse a
transport identity while reverse direction is distinct. DCQCN identities are
QPs; `rnic-cn` identities are directed L2 link pairs.
