# MiniMax-M2.5 expert-parallel scaling result

The frozen study is nonvoid with honest refutations. All four external
dispatch cells are bit-equal, all four composed decode quotients are exactly
1.0, all seven fatal guards hold, and the wall-time family passes. The network
family passes only its fan-in measurement. Its deciding widest ratio is
`0.2742607736975033`, below the frozen `1.25` floor and opposite the frozen
non-decreasing direction.

## What ran

`minimax_ep_scaling_v1` ran the pinned AIConfigurator software development kit
(SDK), the imported operation and NCCL databases, and the rnic-cn packet
backend twice in fresh processes at expert-parallel widths 8, 32, 128 and
256. Each packet point represents 65 serial layer executions. Widths 8, 32
and 128 simulate every directed rank pair for one layer. Width 256 simulates
one receiver at GPU-local index zero on each of 32 nodes, retains every one of
the 256 senders, and represents the faithful full-step ledger without claiming
to simulate its omitted receiver destinations. Bulk evidence is retained at
`${SIMLLM_MINIMAX_T1_BULK_ROOT}/attempt-0002`; the portable result is in
[record.json](record.json) and [results.csv](results.csv).

## Physical sanity before the verdicts

Before reading the EP 256 packet result, four decode candidates times top-8
routing times hidden size 3072, divided across 256 destinations at one FP8 byte
per element, gives exactly 384 bytes per directed rank pair. One rank sends
and receives `2 * 255 * 384 = 195,840` bytes across dispatch plus combine. At
50 GB/s, no full-rank realization can beat `195,840 / 50e9 = 3.9168`
microseconds per layer. Across 65 layer equivalents, the per-rank floor is
254.592 microseconds. The sampled packet arm measures 54.648960 microseconds
for each phase, or 109.297920 microseconds per layer and 7.1043648 ms across
65 executions. It is 27.9049 times the serialization floor, so the result does
not imply impossible link service. The excess is where control, propagation,
path sharing, receiver fan-in and queues enter.

Before reading either widest step, the routed expert weights alone provide an
independent memory floor. One gated expert has
`3 * 3072 * 1536 = 14,155,776` FP8 bytes. Eight active experts across 65
layer equivalents carry at least 7,361,003,520 weight bytes, even under perfect
reuse across the four speculative candidates. At 4.8 TB/s high-bandwidth
memory, that is a 1.5335424 ms floor before attention, router, logits or
communication. The 16.737840 ms SimLLM step and the 61.028924 ms
AIConfigurator step both remain above it.

Before interpreting throughput, the widest AIConfigurator step permits
16.3857 decode steps per second, while the packet-priced step permits 59.7449.
With Multi-Token Prediction (MTP) nextn 3, their absolute ceilings are 65.5427
and 238.9795 candidate tokens per second per request before acceptance losses.
Neither number is an impossible tens-of-thousands-of-tokens result. This check
rules out a gross time-unit error; it does not validate either timing model
against an H200 deployment.

## What came out

The result refutes the proposed omission-cost interpretation. AIConfigurator
prices a half-precision NCCL all-gather plus reduce-scatter over a buffer
proportional to `tokens * hidden * E`. SimLLM prices uniform routed FP8 expert
payload proportional to `tokens * topk * hidden`, with placement, paths,
queues and receiver fan-in. These are different traffic abstractions. At EP
256, AIConfigurator assigns 51.39544921875 ms to dispatch, while the sampled
routed packet arm assigns 7.1043648 ms. Replacing the former with the latter
on the same 9.633475239976931 ms non-dispatch timing base lowers the step from
61.028924458726934 ms to 16.73784003997693 ms. The run therefore cannot claim
that the omitted receiver mechanism adds at least a quarter-step. It shows
instead that the donor's much larger buffer abstraction dominates the
comparison.

The scored families remain separate:

| Family | Result | Verdict |
|---|---:|---|
| E | 4 / 4 | PASS: every dispatch value is bit-equal to the frozen live SDK oracle. |
| C | 4 / 4 | PASS: every composed decode quotient is exactly 1.0 inside `[0.98, 1.02]`. |
| N | 1 / 3 | REFUTED: N1 and N2 fail; N3 measures the required fan-in. |
| W | 1 / 1 | PASS: both fresh arms, scoring and figure rendering completed in 298.814700 seconds, below 3,600 seconds. |

All fatal guards FG-1 through FG-7 pass. A fatal guard is not part of the
behavioral denominator.

## Family E: external dispatch parity

| EP | Frozen dispatch ms | Frozen hex | Local hex | Outcome |
|---:|---:|---|---|---|
| 8 | 1.92205 | `0x1.ec0b780346dc6p+0` | `0x1.ec0b780346dc6p+0` | PASS |
| 32 | 19.82220267857143 | `0x1.3d27bdfef25dcp+4` | `0x1.3d27bdfef25dcp+4` | PASS |
| 128 | 36.77934174107143 | `0x1.263c1785d279dp+5` | `0x1.263c1785d279dp+5` | PASS |
| 256 | 51.39544921875 | `0x1.9b29e147ae148p+5` | `0x1.9b29e147ae148p+5` | PASS |

The NCCL collection identity is H200 SXM, NCCL collection 2.26.2, with source
rows that declare version 2.29.2. Its content-addressed slice is
`e432db694195110aa39c1e1eccf1accda012e69ef68e95210d049809bb93f015`
and its compressed payload SHA-256 is
`12ed4c1dc12b3d9f0f04ffecf025d0dea5599946fa36944bcb60117035d70efb`.
The record preserves NCCL as the source identity of both dispatch operations.

## Family C: external pass composition

| EP | Live SDK step ms | Local composed step ms | Quotient | Outcome |
|---:|---:|---:|---:|---|
| 8 | 13.984132942232176 | 13.984132942232176 | 1.0 | PASS |
| 32 | 27.51711787974335 | 27.51711787974335 | 1.0 | PASS |
| 128 | 44.86945704576469 | 44.86945704576469 | 1.0 | PASS |
| 256 | 61.028924458726934 | 61.028924458726934 | 1.0 | PASS |

The generic composer retains the merged Qwen3-32B-FP8 behavior. The
post-refactor parity replay at
`${SIMLLM_MINIMAX_T1_BULK_ROOT}/qwen-parity-after-generic/attempt-0002`
is nonvoid with I1 25/25, I2 26/26, I2S 13/13, P1 4/4 and W 1/1, with no
unit-in-the-last-place finding.

## Family N: routed packet pricing

| EP | Packet sampling | AIConfigurator step ms | Packet dispatch plus combine ms | SimLLM step ms | SimLLM / AIConfigurator | Outcome |
|---:|---|---:|---:|---:|---:|---|
| 8 | one full-peer layer of 65 | 13.984132942232176 | 0.0249600 | 12.087042942232175 | 0.8643398194341548 | N1 sequence cell |
| 32 | one full-peer layer of 65 | 27.51711787974335 | 4.0350336 | 11.72994880117192 | 0.4262782480503487 | N1 decreases |
| 128 | one full-peer layer of 65 | 44.86945704576469 | 5.5890432 | 13.679158504693262 | 0.3048657016451342 | N1 decreases |
| 256 | one receiver per node, every sender, one layer of 65 | 61.028924458726934 | 7.1043648 | 16.73784003997693 | 0.2742607736975033 | N1 and N2 refuted |

N1 required a non-decreasing ratio. The measured sequence is strictly
decreasing: 0.864340, 0.426278, 0.304866 and 0.274261. N1 is refuted.

N2 required the widest ratio to be at least 1.25. The measured value is
0.2742607736975033. N2 is refuted without changing the band.

N3 passes. At EP 256, both dispatch and combine measure 54.648960 microseconds
of receiver ingress occupancy and 248 simultaneous cross-node senders at the
selected receiver. Seven same-node peers remain on the analytic NVLink leg,
so the logical receiver has all 255 possible peer senders while the htsim
measurement correctly reports only the 248 fabric senders.

The faithful EP 256 ledger remains 130,560 directed messages and 50,135,040
bytes per layer, hence 8,486,400 messages and 3,258,777,600 bytes across 65
executions. The scored EP 256 sample executes 16,320 messages per layer, one
eighth of the receiver destinations, while retaining all sources and all 32
nodes. A prior full-population probe reached the packet backend but timed out
after 900 seconds in dispatch alone, before producing a completion row. It is
retained at `${SIMLLM_MINIMAX_T1_BULK_ROOT}/runner-probe-006` and contributes
no scored value.

## Determinism, chronology and artifacts

Both fresh evaluation payloads have the same SHA-256,
`31c092cf2e1264c55820b58a2d942c428c2c33f5414026952357ccadb725c461`.
The study ran from commit `df47c6532c71312d36eb96ed528f6ebd772e5952`.
The immutable expectation commit `61b66c4` and exact-oracle configuration
commit `5a29bb0` both precede the implementation and run commit. The tracked
expectations SHA-256 is
`9b355278c779c7834d18eaf3b19d16929f7b1800926e0ba1ba271f14a5d613ed`.

The tracked artifact hashes are:

| Artifact | SHA-256 |
|---|---|
| `record.json` | `6f980eaea513bb532723e5a0cd66740002a5f7d4b3c78317c95c745aa0921f68` |
| `results.csv` | `b806306e2b2bc9ff81f4a0895fbc5694845e423fd37171d32c45ff98a5250467` |
| `figures/minimax_ep_scaling.png` | `e7c2b72fa85bf4cd9bf1169e6371bb164382e39016c0613d3e8029d9c81ebf85` |
| `figures/minimax_ep_scaling.pdf` | `3efe36eda7d08706176e6169162827e8cfcb7e15e9d3d3d913df46ffaebd01f1` |

The publication figure is available as
[PNG](figures/minimax_ep_scaling.png) and
[PDF](figures/minimax_ep_scaling.pdf). Visual inspection confirmed that both
panels, all dispatch-share annotations, the exact Qwen reference and the
distinct receiver-subset marker are legible without clipping.

## What it changes for the project

The model-configured external pass composer and MiniMax-M2.5 MoEDispatch
resolver are now literal offline capabilities, and COMP-87 is not their owner.
The frozen E and C matched seam is established at all four widths. The study's
network claim does not advance: N1 and N2 are refuted because the two tools
price different traffic quantities. COMP-88 opens for independent H200 NCCL
rank and message-size calibration of the donor extrapolation. TRAF-73 opens
for hardware-identified routed expert traffic, switch buffering and receiver
occupancy. TRAF-26 remains open for independently routed production peer
workloads rather than this study's uniform per-rank surrogate. No existing
milestone advances to a hardware-valid MiniMax scaling claim.

## What it does not change

This result does not validate either stack against hardware, does not claim
the external measured compute rows are wrong, and does not present the two
traffic abstractions as equivalent. It does not simulate all EP 256 receiver
destinations, does not close TRAF-26, and does not place the external pass on
the supported `ExecutionGraph` through `CompletionEvent`, `StepResult`, time
to first token (TTFT) and time per output token (TPOT) chain owned by COMP-86.
It does not dispatch the WideEP MoE, MLA BMM, Mamba2, MLA or generation-MLA
families owned by COMP-87. The Qwen parity result, imported operation artifact
and all accepted default traffic timestamps remain unchanged.
