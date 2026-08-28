# MiniMax-M2.5 expert-parallel scaling result

The first run is VOID against FG-4. Its prose explained that the external and
packet arms moved different traffic, but its result tables, CSV and figure
footer did not. Its implemented guard inspected two constants and a fixed
string instead of the generated artifacts, so FG-4 was never earned. The
deeper correction is that the first freeze claimed both arms price the same
dispatch and combine collective, and they do not. The external arm prices a
dense half-precision all-gather plus reduce-scatter whose volume grows as
tokens times hidden size times expert-parallel width. The packet arm priced a
sparse routed payload whose logical volume is independent of width.

Both are real strategies. The dense path is TensorRT-LLM's documented general
fallback, and the external source deliberately selects it for SM90 while
retaining separate sparse branches for SM100 and DeepEP. The published
`0.2742607736975033` is therefore a strategy comparison, not evidence of an
omitted mechanism. This study does not know which strategy a real deployment
selects. Every first-run number remains visible below as void evidence.

## What ran

Before any Family D value is read, its ratio compares two cost models. It does
not execute the same physical realization twice, and it is NOT evidence that
contention is the only difference. The external arm is an opaque eight-rank
NCCL table measurement scaled by a rank factor. It has no source, destination,
path or message ledger. The packet arm expands the requested logical element
count into direct all-pairs transfers on a concrete Clos placement.

The corrected `minimax_ep_scaling_v1` study ran two fresh evaluations at
expert-parallel widths 8, 32, 128 and 256. Family D requested the same generic
half-precision all-gather plus reduce-scatter element count from both cost
models. EP 8, 32 and 128 used measured full packet populations. EP 256 used a
post-specified `31 / 15` extrapolation from the full EP 128 anchor and is an
unscored diagnostic because that rule first appeared in implementation commit
`a6ba97f`, after corrected expectations commit `4d1e41c`. Family S ran full
realized sparse populations at every width with FP8 dispatch and BF16 combine.
Bulk evidence is retained at
`${SIMLLM_MINIMAX_FIX_BULK_ROOT}/attempt-0002`; the portable rows are in
[record.json](record.json) and [results.csv](results.csv).

## What came out

Family D scores 0 of 3 measured widths. EP 8, 32 and 128 are all REFUTED
against the unchanged lower bound of 1.0, at ratios
`0.02590463307406155`, `0.3530150565741419` and
`0.8026183885459625`. EP 256's ratio `1.187022158460092` is an UNSCORED
DIAGNOSTIC. It is not a pass anywhere in this report or its generated
artifacts.

The apparent rise does not establish a contention crossover. At EP 128, the
external extrapolator starts from an eight-rank donor latency of
`55.445 microseconds` per layer. Treating the caller's request as generic
two-byte half elements, ideal two-phase ring serialization accounts for only
`12.233386666666666 microseconds`; the remaining fixed and algorithmic
residual is `43.21161333333334 microseconds`. The rank multiplier is
`10.205357142857142`. Multiplying that residual and then repeating it for 65
layers produces `28.664346541071428 ms`, which exceeds the observed EP 128
external-minus-packet gap of `7.259565741071427 ms` by a factor of
`3.948493279549929`. The remaining gap is therefore fully explainable by the
two models' different treatment of fixed and algorithmic overhead. The study
cannot attribute it to contention.

The external table identifies its dtype only as `half`; it does not identify
BF16. Its interpolation axis is named `message_bytes`, while the caller passes
`tokens * hidden * EP` as a count of elements. That unit question remains
unresolved and is disclosed rather than silently choosing the axis name or the
caller's semantics as authoritative.

### Family D cost-model comparison

At eight GPUs per node, every EP 8 rank is intra-node. Its cross-node sender
count is exactly zero, so it is not a contention cell. At every width the
broader Family D comparison also remains a cost-model comparison rather than
contention isolation.

The repository already carries NVLink domain modelling merged from
`nvlink_flow_dynamics_v1` and `nvlink_rnic_comparison_v1`. This study's packet
arm did not use that domain for its intra-node legs. TRAF-76 owns binding that
landed model into the packet arm and pricing the fixed collective overheads.

| EP | Interpretation | D-external strategy, traffic and realization | D-packet strategy, traffic and realization | External ms | Packet ms | Packet / external | Population | Outcome |
|---:|---|---|---|---:|---:|---:|---|---|
| 8 | two cost models; not a contention cell because cross-node fan-in is `0.000000` | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP`; opaque eight-rank measurement with no flow ledger | packet Clos cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP`; direct all-pairs transfers | 1.92205 | 0.04979 | 0.02590463307406155 | measured full rank and message population, 112 messages per layer | REFUTED |
| 32 | two cost models, not contention isolation | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP`; opaque eight-rank measurement with no flow ledger | packet Clos cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP`; direct all-pairs transfers | 19.82220267857143 | 6.997536 | 0.3530150565741419 | measured full rank and message population, 1,984 messages per layer | REFUTED |
| 128 | two cost models, not contention isolation | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP`; opaque eight-rank measurement with no flow ledger | packet Clos cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP`; direct all-pairs transfers | 36.77934174107143 | 29.519776 | 0.8026183885459625 | measured full rank and message population, 32,512 messages per layer | REFUTED |
| 256 | two cost models, not contention isolation | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP`; opaque eight-rank measurement with no flow ledger | packet Clos cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP`; direct all-pairs transfers | 51.39544921875 | 61.00753706666667 | 1.187022158460092 | post-specified `31 / 15` extrapolation from measured full EP 128 population | UNSCORED DIAGNOSTIC |

Family E remains 4 of 4 bit-equal. Family C remains 4 of 4 at quotient 1.0,
but it is an end-to-end parity check that reuses the dispatch code Family E
validates, not independent confirmation of E. Family W passes in
907.220454105176 seconds. Family S is published and unscored. All fatal guards
FG-1 through FG-10 hold, so the corrected run is nonvoid.

## What it changes for the project

The original MiniMax result remains published as void, and its omission-cost
interpretation stays withdrawn. Family D now records 0 of 3 measured cells,
with EP 256 reclassified as an unscored diagnostic. The contention-isolation
and crossover claims are withdrawn. Family D is now a disclosed comparison of
an opaque external NCCL-table cost model and a direct packet Clos cost model.

No existing task changes status and no milestone advances. TRAF-76 is
registered to own intra-node collective transport and fixed collective
overhead pricing in the packet arm, including binding the landed NVLink domain
into intra-node legs. TRAF-74 owns replacing the deterministic balanced routing
geometry with observed per-rank
assignments. TRAF-75 explicitly owns propagating dispatch and combine
precision from supported framework configuration. TRAF-73 is narrowed to
hardware transport calibration, including queue service, phase makespan,
buffering and receiver occupancy. TRAF-26 continues to own complete production
peer workloads, and COMP-88 continues to own independent calibration of the
external NCCL extrapolation.

## What it does not change

The run does not isolate contention, determine which communication strategy a
MiniMax deployment uses, validate either timing model against H200 hardware,
or turn Family S into a precision claim. It does not make Family C independent
evidence for Family E. It does not close TRAF-26, TRAF-73, TRAF-74, TRAF-75,
TRAF-76 or COMP-88, and it does not change accepted default traffic timestamps,
the imported operation artifact or the prior Qwen parity result.

## Physical sanity before detailed interpretation

At EP 256, one dense half buffer is
`4 tokens * 3072 hidden * 256 ranks * 2 bytes = 6,291,456 bytes`. The two
direct collective phases put 12,189,696 bytes per rank on the fabric. At
50 GB/s, no packet realization can beat 243.79392 microseconds per layer, or
15.8466048 ms across 65 layers. The post-specified diagnostic extrapolation is
61.00753706666667 ms, 3.8499 times that floor. The external estimate is
51.39544921875 ms, 3.2433 times the same floor. Both are physically possible;
the floor alone does not identify which overhead model is right.

The corrected sparse EP 256 arm sends 97,920 FP8 dispatch bytes and returns
195,840 BF16 combine bytes per rank, for 293,760 bytes total. Of those,
286,524 bytes cross the fabric. Its 50 GB/s serialization floor is
5.73048 microseconds per layer, or 0.3724812 ms across 65 layers. The measured
packet dispatch-plus-combine value is 7.4813024 ms, 20.0850 times the floor.
This remains physically possible and leaves room for propagation, packet
service, path sharing and queueing.

An independent memory bound reaches the same broad conclusion. One gated
expert carries `3 * 3072 * 1536 = 14,155,776` FP8 weight bytes. Eight active
experts over 65 layers require at least 7,361,003,520 bytes, or 1.5335424 ms
at 4.8 TB/s before attention, routing, logits or communication. The corrected
dense external step is 61.028924458726934 ms and the sparse packet-composed
step is 17.114777639976932 ms, both above that floor.

For end-to-end plausibility, those two steps imply 16.3857 and 58.4290 decode
steps per second per request. Even treating each Multi-Token Prediction step
as four candidates gives ceilings of 65.5427 and 233.7162 candidate tokens per
second before acceptance losses. Neither implies an impossible
tens-of-thousands-of-tokens rate. These bounds rule out gross byte, time and
unit defects; they do not validate either model against a deployment.

## First-run void evidence

Every row below remains void. It cannot be scored because FG-4 failed, the
strategies differ, the sparse geometry was all-pairs fluidized, combine was
incorrectly FP8, and the EP 256 cell simulated only one eighth of receiver
destinations without a full-population anchor.

| EP | External strategy and traffic | Packet strategy and traffic | External step ms | Packet communication ms | Packet step ms | Packet / external | Population |
|---:|---|---|---:|---:|---:|---:|---|
| 8 | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP` | sparse all-pairs fluidized FP8 dispatch and FP8 combine over `tokens * topk * hidden` | 13.984132942232176 | 0.02496 | 12.087042942232175 | 0.8643398194341548 | full all-pairs fluidized population, 112 messages per layer |
| 32 | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP` | sparse all-pairs fluidized FP8 dispatch and FP8 combine over `tokens * topk * hidden` | 27.51711787974335 | 4.0350336 | 11.72994880117192 | 0.4262782480503487 | full all-pairs fluidized population, 1,984 messages per layer |
| 128 | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP` | sparse all-pairs fluidized FP8 dispatch and FP8 combine over `tokens * topk * hidden` | 44.86945704576469 | 5.5890432 | 13.679158504693262 | 0.3048657016451342 | full all-pairs fluidized population, 32,512 messages per layer |
| 256 | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP` | sparse all-pairs fluidized FP8 dispatch and FP8 combine over `tokens * topk * hidden` | 61.028924458726934 | 7.1043648 | 16.73784003997693 | 0.2742607736975033 | 16,320 of 130,560 messages per layer, one eighth and unanchored |

The old 248-sender EP 256 fan-in was `248 / 29.57691192626953 = 8.3849`
times the realizable analytical expectation. Its published headline ratio is
retained solely as void strategy-comparison evidence.

The invalid EP 256 sample reported 54.648960 microseconds of ingress occupancy
in each direction, 248 fabric senders and 255 logical senders after including
seven same-node peers. Its represented ledger was 130,560 messages and
50,135,040 bytes per layer, hence 8,486,400 messages and 3,258,777,600 bytes
over 65 layers, but it executed only 16,320 messages per layer. Those values
remain findings about the void construction, not scored evidence.

## Family S: published strategy comparison, unscored

Family S compares the external dense SM90 fallback with the corrected sparse
routed expert payload. It does not know which strategy a real deployment
selects.

| EP | S-dense strategy and traffic | S-sparse strategy and traffic | Dense step ms | Sparse packet communication ms | Sparse step ms | Sparse / dense | Population |
|---:|---|---|---:|---:|---:|---:|---|
| 8 | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP` | sparse realized top-k routing, FP8 dispatch plus BF16 combine over `tokens * topk * hidden` | 13.984132942232176 | 0.037375 | 12.099457942232176 | 0.8652276113373988 | full rank and realized-message population, 112 messages per layer |
| 32 | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP` | sparse realized top-k routing, FP8 dispatch plus BF16 combine over `tokens * topk * hidden` | 27.51711787974335 | 4.4170048 | 12.11192000117192 | 0.4401594692476161 | full rank and realized-message population, 1,340 messages per layer |
| 128 | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP` | sparse realized top-k routing, FP8 dispatch plus BF16 combine over `tokens * topk * hidden` | 44.86945704576469 | 5.6296448 | 13.719760104693265 | 0.30577058444678235 | full rank and realized-message population, 7,444 messages per layer |
| 256 | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP` | sparse realized top-k routing, FP8 dispatch plus BF16 combine over `tokens * topk * hidden` | 61.028924458726934 | 7.4813024 | 17.114777639976932 | 0.28043714995422264 | full rank and realized-message population, 15,640 messages per layer |

The sparse-to-dense ratios remain a strategy comparison only. They do not
identify an omitted mechanism or establish which path runs in production.

## Routing geometry and directional precision

The balanced surrogate routes whole token-expert assignments only to
destinations they reach. It never manufactures fractional bytes to every peer.
The analytical expectation uses independent assignment landing. The realized
cross-node values are reconstructed from simulator completion rows, and the
total destination values add the analytically completed same-node segments.
They are not read back from the planned fabric segment structure.

| EP | Expected distinct destinations per source | Realized distinct destinations per source | Expected cross-node senders per receiver | Realized cross-node senders per receiver | Maximum realized |
|---:|---:|---:|---:|---:|---:|
| 8 | 7.000000 | 7.000000 | 0.000000 | 0.000000 | 0 |
| 32 | 21.191406 | 20.937500 | 16.406250 | 16.437500 | 24 |
| 128 | 28.895523 | 29.078125 | 27.302856 | 27.578125 | 32 |
| 256 | 30.411744 | 30.546875 | 29.576912 | 29.781250 | 32 |

At EP 256 the expected destination count is exactly
`255 * (1 - (31 / 32)^4) = 30.411744117736816`, and the expected cross-node
sender count is
`248 * (1 - (31 / 32)^4) = 29.57691192626953`. The realized cross-node
fan-in is 29.78125, only 0.69 percent above expectation and far below the FG-8
ceiling of 1.2 times expectation.

Dispatch and combine precision are declared separately. The represented sparse
implementation quantizes dispatch to FP8 at one byte per element. Its ordinary
combine return is BF16 at two bytes per element; low-precision combine is a
separate mode and is disabled here.

| EP | FP8 dispatch bytes per rank | BF16 combine bytes per rank | Total bytes per rank |
|---:|---:|---:|---:|
| 8 | 86,016 | 172,032 | 258,048 |
| 32 | 95,232 | 190,464 | 285,696 |
| 128 | 97,536 | 195,072 | 292,608 |
| 256 | 97,920 | 195,840 | 293,760 |

## External parity, determinism and guards

Family E reproduces all four frozen external dispatch cells bit-for-bit:

| EP | Frozen dispatch ms | Frozen hex | Local hex |
|---:|---:|---|---|
| 8 | 1.92205 | `0x1.ec0b780346dc6p+0` | `0x1.ec0b780346dc6p+0` |
| 32 | 19.82220267857143 | `0x1.3d27bdfef25dcp+4` | `0x1.3d27bdfef25dcp+4` |
| 128 | 36.77934174107143 | `0x1.263c1785d279dp+5` | `0x1.263c1785d279dp+5` |
| 256 | 51.39544921875 | `0x1.9b29e147ae148p+5` | `0x1.9b29e147ae148p+5` |

Both fresh evaluation payloads have SHA-256
`ed5c4be84e3c243255ec45be1b224a8a08e5479d98ee1f7848e1c9831de95882`.
The corrected run commit is
`7eff88a4efa68c4d2ad8233201d18e43b97d8d77`. The first freeze
`61b66c4`, exact-oracle commit `5a29bb0`, and corrected freeze `4d1e41c`
all precede implementation and execution. The two expectation files retain
SHA-256 values
`9b355278c779c7834d18eaf3b19d16929f7b1800926e0ba1ba271f14a5d613ed`
and
`b237945a945e1b1500ab299cf81faf20e704541f6c3e591b1cf90c418b5bb116`.

FG-4 inspected four record rows, four CSV rows, five figure series, the figure
caption and extracted PDF text. FG-8 validated realized routing geometry,
FG-9 validated directional precision, and FG-10 validated every scored
population or anchor. FG-1 through FG-3 and FG-5 through FG-7 also pass. The
failed rendering-only `attempt-0001` remains append-only evidence and was not
scored; `attempt-0002` is the nonvoid corrected run.

The tracked artifact hashes are:

| Artifact | SHA-256 |
|---|---|
| `record.json` | `d99b615cbcf36c60b12e806266f5d4281db3964b39a7134b8a94a12ca9f59cc9` |
| `results.csv` | `dd2b0c9be299338636a91b0a958f172687a2a3ef6ccc77788ed0776933905ab8` |
| `figures/minimax_ep_scaling.png` | `deedf3b85aa8077566a40ed38b16d1ca42c85957839223b9a945fa9d6ebd91da` |
| `figures/minimax_ep_scaling.pdf` | `238ffa5132890dd5304005e667a29f3aa4339578052ab078fb937f59a356e7cf` |
| `figures/minimax_ep_scaling.metadata.json` | `024b2789720a9afd87451bbfad2361a226d8f6a6c093b8e24b3e7909a56ed372` |

The corrected comparison and contention-ratio panel are available as
[PNG](figures/minimax_ep_scaling.png) and
[PDF](figures/minimax_ep_scaling.pdf). Visual inspection confirmed that every
series, point, extrapolation marker, bound, label and footer is legible without
clipping or overlap.
