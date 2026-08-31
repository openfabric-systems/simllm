# MiniMax-M2.5 expert-parallel scaling result

The original first run remains VOID against FG-4. Its headline
`0.2742607736975033` was a strategy comparison whose tables and figure did not
name both traffic definitions, and it is NOT evidence about contention or an
external-planner omission. The study still does not know which communication
strategy a real deployment selects, and therefore does not know which strategy
is the production path. Every first-run value remains visible as
void evidence below.

The later merged Family D publication was nonvoid, but its packet arm omitted
intra-node collective transport and fixed collective overheads. Those values
are now superseded, not deleted. That floor-omitting publication scored 0 of 3
widths. The corrected study compares the same two cost models and re-evaluates
the unchanged `expectations_v2.md` predicate. Family D now passes 1 of 3 scored
widths. EP 8 changes from `0.02590463307406155` to
`1.1091430503889075`, a correction factor of `42.816396866840726`, and flips
from REFUTED to PASS. EP 32 and EP 128 remain REFUTED. EP 256 remains an
UNSCORED DIAGNOSTIC under a corrected component-wise rule.

## What ran

Implementation commit `e4626a9` binds the landed
`h200-nccl-2.26.2-aggregate-floor-v1` authority into Family D and Family S.
Each semantic phase queries its operation-buffer byte coordinate, charges the
fitted aggregate floor once outside the maximum of calibrated local byte-slope
service and unchanged packet fabric service, and keeps the floor-omitting
composition beside it as superseded evidence.

Family D uses half-precision all-gather and reduce-scatter donor curves.
EP 8 is an exact rank-8 calibrated use. EP 32, EP 128 and EP 256 deliberately
use rank-8 donor curves and stamp `transferred-at-use` acknowledgement. Family
S maps FP8 dispatch to the half all-gather donor and BF16 combine to the half
reduce-scatter donor. Both sparse semantic uses are explicit transfers, not
direct H200 measurements.

The external arm is unchanged. It still queries
`tokens_per_rank * hidden_size * expert_parallel` elements. The collective
calibration established that the raw table coordinate is an element count
despite its `message_bytes` interpolation label, and converts it to true bytes
using dtype width. The MiniMax external reproduction already used that element
coordinate and remains bit-equal.

The publication evaluates expert-parallel widths 8, 32, 128 and 256. EP 8, 32
and 128 use full dense rank and message populations. EP 256 remains an unscored
diagnostic derived from the full EP 128 dense anchor. Family S executes the
full realized sparse population at every width. Bulk evidence is append-only
under `${SIMLLM_MINIMAX_FIX_BULK_ROOT}`; portable evidence is in
[record.json](record.json) and [results.csv](results.csv).

## What came out

The binding corrects the merged Family D ratios as follows:

| EP | Superseded packet / external | Corrected packet / external | Change in outcome |
|---:|---:|---:|---|
| 8 | 0.02590463307406155 | 1.1091430503889075 | REFUTED to PASS |
| 32 | 0.3530150565741419 | 0.4359189379766115 | REFUTED remains REFUTED |
| 128 | 0.8026183885459625 | 0.8472993823377812 | REFUTED remains REFUTED |
| 256 | 1.187022158460092 | 1.2189965368336635 | remains UNSCORED DIAGNOSTIC |

The original frozen lower bound is still 1.0. It was not widened or
re-specified. The correction restores that expectation at EP 8 only. It does
not restore it at every scored width, so the earlier EP 32 and EP 128
refutations are not artifacts of this omission. The earlier EP 8 refutation
was entirely our own missing term and was not a finding about the external
planner.

Family E remains 4 of 4 bit-equal. Family C remains 4 of 4 at quotient 1.0,
but it reuses the dispatch code Family E validates and is not independent
confirmation. Family S remains published and unscored, with every sparse value
corrected and every earlier value labeled superseded.

## What it changes for the project

TRAF-76 no longer owns an unbound MiniMax packet arm at the widths published
here. The aggregate floor and byte-slope authority reaches the full Family D
populations at EP 8, 32 and 128, the component-wise EP 256 diagnostic, and all
four full Family S populations. Every transferred use is acknowledged and
recorded.

TRAF-76 remains open and narrows to its unsatisfied precision bars and packet
mechanism remainder. Twelve of 63 held-out calibration cells still miss the
10 percent band, the calibration study's D8 quotient 1.109143050 still exceeds
its separate 1.10 upper bound, and credits, product geometry, switch behavior,
arbitration and nonzero-fan-in H200 calibration remain unresolved. Closing the
task on this binding would claim partial coverage as complete.

The merged crossover claim does not survive. The corrected cross-node ratios
rise from 0.435918938 at EP 32 to 0.847299382 at EP 128 and the unscored
1.218996537 diagnostic at EP 256. Linear interpolation in expert-parallel width
would cross 1.0 at 180.584957, while interpolation on the plotted log2 width
axis would cross at 170.168475. Neither is scored, and EP 8 already sits above
1.0 before the curve falls below it at EP 32. There is no single monotone
crossover near expert parallelism 200 to publish.

## What it does not change

The correction does not isolate contention, determine the strategy a deployed
MiniMax engine uses, validate either timing model against H200 hardware, or
turn Family S into a precision claim. It does not change the external arm, its
four frozen cells, the operation database, accepted default traffic
timestamps, TRAF-77, TRAF-78, TRAF-75, TRAF-26 or COMP-89. It does not close a
milestone.

### Family D cost-model comparison

EP 8 has zero cross-node traffic and is not a contention cell. At every width
the ratio remains a comparison of an opaque external NCCL-table cost model and
a packet Clos cost model with a transferred aggregate collective component.
It is not evidence that contention is their only difference.

| EP | Interpretation | D-external strategy, traffic and realization | D-packet strategy, traffic and realization | External ms | Corrected packet ms | Corrected ratio | Superseded packet ms and ratio | Population | Outcome |
|---:|---|---|---|---:|---:|---:|---:|---|---|
| 8 | two cost models; not a contention cell because cross-node fan-in is `0.000000` | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP`; opaque eight-rank measurement with no flow ledger | packet Clos cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP`; exact rank-8 aggregate floor and byte slopes composed with direct all-pairs transport | 1.92205 | 2.1318284 | 1.1091430503889075 | 0.04979 and 0.02590463307406155 | measured full rank and message population, 112 messages per layer | PASS |
| 32 | two cost models, not contention isolation | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP`; opaque eight-rank measurement with no flow ledger | packet Clos cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP`; acknowledged rank-8 transfer composed with direct all-pairs fabric service | 19.82220267857143 | 8.640873540000001 | 0.4359189379766115 | 6.997536 and 0.3530150565741419 | measured full rank and message population, 1,984 messages per layer | REFUTED |
| 128 | two cost models, not contention isolation | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP`; opaque eight-rank measurement with no flow ledger | packet Clos cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP`; acknowledged rank-8 transfer composed with direct all-pairs fabric service | 36.77934174107143 | 31.163113539999998 | 0.8472993823377812 | 29.519776 and 0.8026183885459625 | measured full rank and message population, 32,512 messages per layer | REFUTED |
| 256 | two cost models, not contention isolation | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP`; opaque eight-rank measurement with no flow ledger | packet Clos cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP`; acknowledged rank-8 transfer with component-wise extrapolated fabric service | 51.39544921875 | 62.65087460666667 | 1.2189965368336635 | 61.00753706666667 and 1.187022158460092 | unscored diagnostic from full EP 128 anchor | UNSCORED DIAGNOSTIC |

## EP 256 extrapolation correction

The old `31 / 15` rule multiplied the whole EP 128 phase duration. Once fixed
floors are present, that is invalid because it multiplies an additive constant
as if it were cross-node bytes. The corrected unscored rule acts on components:

1. Scale each EP 128 fabric service by
   `(256 - 8) / (128 - 8) = 31 / 15`.
2. Query the rank-8 all-gather or reduce-scatter donor at the EP 256
   6,291,456-byte operation-buffer coordinate.
3. For each semantic half, compute
   `floor + max(calibrated byte-slope service, extrapolated fabric service)`.
4. Add the two halves and multiply by 65 represented layers.

The extrapolated fabric service dominates the calibrated byte-slope service in
both halves. The corrected total therefore equals the superseded
61.00753706666667 ms fabric projection plus 1.64333754 ms of once-charged
floors, or 62.65087460666667 ms. The old value remains visible but its whole-
phase linearity rule is withdrawn.

## Family S: published strategy comparison, unscored

Family S compares the external dense SM90 fallback with sparse realized top-k
routing. The corrected sparse packet arm carries FP8 dispatch, BF16 combine,
and explicitly transferred aggregate collective floors and byte slopes. It
does not establish which strategy a real deployment selects.

| EP | S-dense strategy and traffic | S-sparse strategy and traffic | Dense step ms | Corrected sparse communication ms | Corrected sparse step ms | Corrected sparse / dense | Superseded sparse step ms and ratio | Population |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 8 | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP` | sparse realized top-k routing, FP8 dispatch plus BF16 combine over `tokens * topk * hidden`; acknowledged operation and dtype transfer | 13.984132942232176 | 2.94193614 | 15.004019082232176 | 1.0729316679277223 | 12.099457942232176 and 0.8652276113373988 | full rank and realized-message population, 112 messages per layer |
| 32 | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP` | sparse realized top-k routing, FP8 dispatch plus BF16 combine over `tokens * topk * hidden`; acknowledged operation, dtype and rank transfer | 27.51711787974335 | 5.781573005 | 13.47648820617192 | 0.48974926317020284 | 12.11192000117192 and 0.4401594692476161 | full rank and realized-message population, 1,340 messages per layer |
| 128 | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP` | sparse realized top-k routing, FP8 dispatch plus BF16 combine over `tokens * topk * hidden`; acknowledged operation, dtype and rank transfer | 44.86945704576469 | 6.994213005 | 15.084328309693262 | 0.3361825460537214 | 13.719760104693265 and 0.30577058444678235 | full rank and realized-message population, 7,444 messages per layer |
| 256 | external NCCL-table cost model: dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP` | sparse realized top-k routing, FP8 dispatch plus BF16 combine over `tokens * topk * hidden`; acknowledged operation, dtype and rank transfer | 61.028924458726934 | 8.845870605 | 18.479345844976933 | 0.3027965183537566 | 17.114777639976932 and 0.28043714995422264 | full rank and realized-message population, 15,640 messages per layer |

The sparse dispatch operation-buffer coordinate is 98,304 FP8 bytes. The
combine coordinate is 196,608 BF16 bytes. Those logical buffer coordinates are
not the bytes physically sent after self assignments and same-node routing are
removed. The floor authority is an explicit transferred proxy for sparse
semantics, so these values remain unscored.

## Physical sanity before detailed interpretation

At EP 256, one dense half buffer is
`4 * 3072 * 256 * 2 = 6,291,456` bytes. The two phases put 12,189,696 bytes per
rank on the fabric. At 50 GB/s, the 65-layer dense communication has a
15.8466048 ms floor. The source exposes no finite progress ceiling, so the
honest ceiling is unbounded. The corrected 62.65087460666667 ms lies inside
those bounds at 3.9536 times the serialization floor.

The sparse EP 256 arm sends 97,920 FP8 dispatch bytes and returns 195,840 BF16
combine bytes per rank. Of the 293,760 total bytes, 286,524 cross the fabric.
Its 65-layer fabric serialization floor is 0.3724812 ms and its ceiling is
unbounded. The corrected 8.845870605 ms communication is 23.7485 times that
floor, leaving room for propagation, packet service, sharing and the
transferred collective completion terms.

An independent memory floor remains 1.5335424 ms for eight active experts over
65 layers at 4.8 TB/s before attention, routing, logits or communication. The
external dense step is 61.028924458726934 ms and the corrected sparse step is
18.479345844976933 ms, both above it. They imply 16.3857 and 54.1145 decode
steps per second per request. Even treating each Multi-Token Prediction step as
four candidates gives ceilings of 65.5427 and 216.4579 candidate tokens per
second before acceptance losses. These checks rule out gross byte and time
unit errors; they do not validate the transfers against hardware.

## First-run void evidence

Every row below remains void. FG-4 failed, the strategies differ, sparse
geometry was all-pairs fluidized, combine was incorrectly FP8, and EP 256 used
an unanchored one-eighth receiver sample.

| EP | External strategy and traffic | Packet strategy and traffic | External step ms | Packet communication ms | Packet step ms | Packet / external | Population |
|---:|---|---|---:|---:|---:|---:|---|
| 8 | dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP` | sparse all-pairs fluidized FP8 dispatch and FP8 combine over `tokens * topk * hidden` | 13.984132942232176 | 0.02496 | 12.087042942232175 | 0.8643398194341548 | full all-pairs fluidized population |
| 32 | dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP` | sparse all-pairs fluidized FP8 dispatch and FP8 combine over `tokens * topk * hidden` | 27.51711787974335 | 4.0350336 | 11.72994880117192 | 0.4262782480503487 | full all-pairs fluidized population |
| 128 | dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP` | sparse all-pairs fluidized FP8 dispatch and FP8 combine over `tokens * topk * hidden` | 44.86945704576469 | 5.5890432 | 13.679158504693262 | 0.3048657016451342 | full all-pairs fluidized population |
| 256 | dense SM90 fallback, generic half-precision all-gather plus reduce-scatter over `tokens * hidden * EP` | sparse all-pairs fluidized FP8 dispatch and FP8 combine over `tokens * topk * hidden` | 61.028924458726934 | 7.1043648 | 16.73784003997693 | 0.2742607736975033 | 16,320 of 130,560 messages per layer, one eighth and unanchored |

The void EP 256 ledger remains 8,486,400 represented messages and
3,258,777,600 bytes over 65 layers. Those are findings about the invalid
construction only.

## Routing geometry and directional precision

The balanced sparse surrogate still routes whole assignments only to reached
destinations. The collective-floor binding changes timing, not messages,
destinations, payload bytes or completion geometry.

| EP | Expected destinations per source | Realized destinations per source | Expected cross-node senders per receiver | Realized cross-node senders per receiver | Maximum realized |
|---:|---:|---:|---:|---:|---:|
| 8 | 7.000000 | 7.000000 | 0.000000 | 0.000000 | 0 |
| 32 | 21.191406 | 20.937500 | 16.406250 | 16.437500 | 24 |
| 128 | 28.895523 | 29.078125 | 27.302856 | 27.578125 | 32 |
| 256 | 30.411744 | 30.546875 | 29.576912 | 29.781250 | 32 |

At EP 256 the realized cross-node fan-in is 29.78125, 0.69 percent above the
29.576911926 analytical expectation and below the FG-8 ceiling. Dispatch stays
FP8 at one byte per element and ordinary combine stays BF16 at two bytes per
element.

## External parity, determinism and guards

Family E reproduces all four frozen dispatch cells bit-for-bit:

| EP | Frozen dispatch ms | Frozen hex |
|---:|---:|---|
| 8 | 1.92205 | `0x1.ec0b780346dc6p+0` |
| 32 | 19.82220267857143 | `0x1.3d27bdfef25dcp+4` |
| 128 | 36.77934174107143 | `0x1.263c1785d279dp+5` |
| 256 | 51.39544921875 | `0x1.9b29e147ae148p+5` |

The final record publishes the two fresh-process hashes, wall time, all fatal
guards and exact artifact identities. The first freeze `61b66c4`, oracle
commit `5a29bb0`, and binding freeze `4d1e41c` all precede implementation
commit `e4626a9`. Correction attempt `attempt-0001` retained valid numerical
results but its publication was void because the figure caption did not name
half precision, so FG-4 failed closed. Disclosure commit `af9b82a` corrected
the caption before fresh `attempt-0002`. That accepted attempt ran from
`af9b82a18e5f23f951df683777f7b548e9447bd8`, completed Family W in
`803.6440525054932` seconds, and produced two bit-equal fresh-process hashes:
`0f4f6b5f2a37def38fd4de0dc96a58ba03765dd972fd9bcd960cccd0a82076e0`.

Both immutable expectation files retain SHA-256 values
`9b355278c779c7834d18eaf3b19d16929f7b1800926e0ba1ba271f14a5d613ed`
and `b237945a945e1b1500ab299cf81faf20e704541f6c3e591b1cf90c418b5bb116`.

The accepted portable artifact SHA-256 values are:

| Artifact | SHA-256 |
|---|---|
| `record.json` | `7f8a3a07867faf18a4f7f307889a9f90e6780eb7c06591a05fad6163ca381f02` |
| `results.csv` | `0fe423d2e0639791864eb69ffec5e1da2ec45e2dc2d3cf813a8aadfca3d8ecad` |
| `minimax_ep_scaling.png` | `babc1559c8e5baa60dc8d76aee93d49b5db0e3152d433190110f910f215370a3` |
| `minimax_ep_scaling.pdf` | `ee0d12bb8e21f5a7e644fc36f573617a849d44d3daac6bad2c19cb8c31db66c2` |
| `minimax_ep_scaling.metadata.json` | `7050def9357ba24b20cbbc9719d0f5944c1bc1d7bf6e638c5e4758f53372d1e5` |

The figure is available as [PNG](figures/minimax_ep_scaling.png) and
[PDF](figures/minimax_ep_scaling.pdf). Its three panels show corrected and
superseded Family S steps, corrected and superseded Family D collective times,
and corrected and superseded Family D ratios. Every series names its strategy
and traffic definition.
