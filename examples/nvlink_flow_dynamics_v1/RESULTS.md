# TRAF-69 scored NV4 flow dynamics

## Outcome

What ran: the frozen three-module NVLink domain study exercised one ordered-pair
join and exit schedule, the seven-rung flow-completion-time ladder over nine
seeds, and physical incast degrees one through three on the NV4 topology.

What came out: `PASS_WITH_EXPECTED_FANOUT_REFUTATION`. The deciding exact checks are
convergence residual 0 ps and divergence residual 0 ps.
All 13 fatal guards passed and all 60 prior artifacts stayed byte-identical.

The final expectations authority is commit
`32a49805546bd038af5e49fd68b5d2ed0cea6174`, with expectations SHA-256
`6e6e8f0ed7c79572f1ef893f7f7869d8a4e854200bdee514b4338b87955e1261`.

What it changes: TRAF-69 closes and the scored NV4 flow-dynamics claim becomes literal.

What it does not change: this is a scored-profile simulation, not new hardware
evidence. The TX and RX plateaus remain measured, ten parameters remain
declared candidates, the pass-through switch remains structural, TRAF-65 stays
open on its separate live held-out integration bar, and no analytical default
path or prior result moves.

## Exact convergence and divergence identities

The 1-to-2 open identity is
`0 + 1,692 + 10,880 + 1,314 = 13,886 ps`: zero credit wait, one packet
admission, one candidate link serialization and one measured RX serialization.
Observed 13,886 ps, residual 0 ps, PASS.

The 2-to-1 target identity is
`0 + 2 * 10,880 - 3 * 1,692 + 0 = 16,684 ps`: zero credit wait, two
four-link cadences, the phase-3 subtraction of three endpoint admissions and
common RX serialization canceled. Observed 16,684 ps, residual 0 ps, PASS.

The overall schedule completed in order
`flow-c`, `flow-b`, `flow-a`. Its reverse target rule is PASS;
219 raw-bin steady checks ran with 0 misses. Rate bins are fixed and raw with no smoothing:
696,320 ps for the overall schedule and 10,880 ps for both transition panels.

## FCT CDF verdicts by size rung

Each cell is the verdict for a mean empirical cumulative distribution function
with a pointwise min-max shaded band across nine frozen seeds. The table reports
every frozen size rung rather than reducing them to one headline count.

| Flow size | Degree 1 | Degree 2 | Degree 3 | Mean p50 range across degrees | Verdict |
|---:|---|---|---|---:|---|
| 256 B | PASS | PASS | PASS | 0.012194 to 0.012381 us | PASS |
| 1 KiB | PASS | PASS | PASS | 0.022114 to 0.024938 us | PASS |
| 4 KiB | PASS | PASS | PASS | 0.103118 to 0.135003 us | PASS |
| 16 KiB | PASS | PASS | PASS | 0.522291 to 0.617702 us | PASS |
| 64 KiB | PASS | PASS | PASS | 2.237491 to 2.589703 us | PASS |
| 256 KiB | PASS | PASS | PASS | 9.044894 to 10.414850 us | PASS |
| 512 KiB | PASS | PASS | PASS | 18.145397 to 20.876967 us | PASS |

## Incast to the physical ceiling

| Degree | Simulated payload | Frozen ceiling | Ceiling fraction | Owner | Verdict |
|---:|---:|---:|---:|---|---|
| 1 | 94.009808 GB/s | 94.117647 GB/s | 0.998854 | `tx_pair_links` | PASS |
| 2 | 187.880751 GB/s | 188.235294 GB/s | 0.998116 | `tx_pair_links` | PASS |
| 3 | 194.562756 GB/s | 194.919456 GB/s | 0.998170 | `rx` | PASS |

Degree one and degree two are ordered-pair-link limited. Degree three is the
only receiver-limited cell and uses the measured 207.101921876 GB/s raw RX
plateau. The payload ceilings include the declared-candidate 256/272 packet
efficiency.

The separate one-sender fan-out check simulated 151.147543
GB/s against the published 281.65 GB/s, a 46.334975
percent miss and `REFUTED` verdict. This is the expected honest
refutation of the sender-side row. It is not used as an incast receiver ceiling.

## Evidence split and guards

- Measured: TX endpoint plateau, RX ingress plateau, request-response direction,
  extent-sequence reassembly and per-extent delivery.
- Declared candidates: maximum payload, header, link count and rate, bond
  policy, credit unit and count, RX buffer and return latency, and queue scope.
- Structural: the NV4 direct-mesh switch is pass-through.
- Preservation: 60 locked artifacts and
  the default-flow canonical digest
  `2f2af64619ed3c6341b209d877d9f1e6984a67e44b97b5eb176a157294a6c252` passed.
- CDF definition: 9 seeds; the shaded interval is
  pointwise minimum to maximum empirical CDF across seeds.

## Run chronology and retained misses

The first execution at `b808a6b` stopped before scoring when the degree-3
large-flow cell exposed missing receiver-side credit backpressure in the new
opt-in path. The first complete run at `97cb90d` emitted an invalid scorer
refutation: audit found that membership-transition bins were being treated as
steady and exact floating-point containment was being used for CDF means. The
scorer-only correction at `4f4022e` changed no raw CSV evidence. Both earlier
attempt directories remain retained outside the checkout.

The first complete and final runs have identical SHA-256 for every raw table:

| Raw table | SHA-256 |
|---|---|
| `convergence-rate.csv` | `0932b93e6f2df6c347681afbf0543b06a37d0e59db3d944828c662fd649e85e5` |
| `divergence-rate.csv` | `e079bc095ae81182e2557a369944c67f42b04ae48e6b876742ce2b47fb3b1417` |
| `fct-cdf.csv` | `500f4a2a2aa971d33e31b7769199d4107ec7cf29f617d3d36524b1ea8283ab65` |
| `fct-samples.csv` | `d051bb65d802d4a3e90a65f7dbf3ba573bee32b7d76ea5956c51d172f841bef8` |
| `incast-degree-1-rate.csv` | `10fd2c85fb13dff27fee4ed224f72163cee453d8b8fb301b81541388bd6d6536` |
| `incast-degree-2-rate.csv` | `231992b0fe21108f39dd4c4808e3ebae31c5def0a983d31841e0846d811bbfcb` |
| `incast-degree-3-rate.csv` | `00bc0d34a12c7e05b9ff43806b3ce30356478d97707f085882f56b122a974825` |
| `overall-rate.csv` | `8cf9bc0d6db61f7a190ccdf2d00ca3f3794ccaea2177a9b01e38d3992201780b` |

The one scientific miss is still published rather than normalized away: the
separate sender-side 281.65 GB/s fan-out row is `REFUTED` by 46.334975 percent.

## Figures

- [`figures/nvlink-flow-dynamics.pdf`](figures/nvlink-flow-dynamics.pdf)
- [`figures/nvlink-fct-cdf.pdf`](figures/nvlink-fct-cdf.pdf)
- [`figures/nvlink-incast-degree-1.pdf`](figures/nvlink-incast-degree-1.pdf)
- [`figures/nvlink-incast-degree-2.pdf`](figures/nvlink-incast-degree-2.pdf)
- [`figures/nvlink-incast-degree-3.pdf`](figures/nvlink-incast-degree-3.pdf)

Every PDF has a matching PNG. The final PNGs were inspected at publication
size for clipping, overlap, legends, shaded-band visibility and border contact.
