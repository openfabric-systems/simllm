# KV lifecycle accounting v1 results

Run of `examples/kv_cache_strategies/run_study.py --out <run directory>` against
the expectations frozen in [expectations.md](expectations.md).

**Headline: a replayed request's timing now responds to memory pressure.**
Constraining the pool from 64 to 32 blocks raises the replayed request's TTFT
from 10,244,194,304 ps to 20,488,388,608 ps, exactly 2.0000x, and preemption
raises TPOT from 5,845,808,128 ps to 7,935,808,128 ps, exactly 1.3575x. Both
moves reproduce the frozen tables to 0 ps. Above the constraint threshold the
metric does not move at all: capacities 48, 56 and 64 give bit-identical TTFT.

- 20 of 20 scored genuine-risk instances passed.
- 17 of 17 entailed relations held.
- 0 of 56 fatal guards violated. The run is interpretable.

## Chronology and pre-registration

- `5a36cbe` froze the expectations. No implementation existed; the check-only
  dry run confirmed that no KV pool type was exported and that every
  byte-carrying KV read or write was refused in preflight.
- `e33f265` landed the ledger and the HBM lowering.
- `6e791db` amended the freeze, before any run of the study existed, after
  re-deriving family B's token cursor against the vLLM sources. The amendment
  is disclosed below and recorded in full in `expectations.md`.
- This file reports the first and only executed measurement.

## The amendment, stated plainly

The original freeze mis-modeled the preemption cursor by one token. A decode
step writes the KV of the token it queries, so after two decode steps the
resident context is 258 tokens while the sequence is 259 tokens long.
Preemption zeroes `num_computed_tokens` (`vllm/v1/core/sched/scheduler.py:1225`)
and keeps `num_tokens`, so the resumed step recomputes 259 tokens, not 258.

The correction made the registration stricter rather than easier. The original
predicted that the resumed step moves one token equivalent fewer KV bytes than
the decode step it replaces, so the TPOT rise would grow as bandwidth fell.
With the correct cursor the resumed step writes exactly as many token
equivalents as the decode step reads and writes, the KV terms cancel, and the
rise is bandwidth-independent. The run confirms the corrected form: the rise is
2,090,000,000 ps at 8 TB/s and 2,090,000,000 ps at 4 TB/s, identical to the
picosecond.

## Two harness defects, both in the checker

The first execution reported 12 fatal-guard violations and 2 scored failures.
Both were comparison defects in the study's own checker, not in the model or
the frozen values, and neither frozen number changed:

- the guard compared the ledger's cumulative `write_bytes` counter, which
  spans all three replayed requests, against one request's write;
- the family B accounting instance read the cumulative `released_references`
  counter, which also carries the pressure request's release, instead of the
  preempted request's own release demand.

Both now compare like with like: the byte guard uses the whole replay's
expected write total, and the accounting instance reads the preempted
request's `RELEASE` demand from the KV report. Every metric guard passed on
the first execution and is unchanged.

## Physical sanity, three independent angles

Bounds were stated in the freeze before any measurement.

**Memory physics.** The cold prefill writes 512 x 131,072 B = 67,108,864 B.
At 8 TB/s no engine beats 8,388,608 ps; the model reports exactly that, so it
sits on the floor and is a lower bound on reality. Real HBM efficiency of 70 to
90 percent would put the true cost at 9.3 to 12 us. Halving the rate exactly
doubles the term in all 12 cells while leaving the compute term bit-identical,
which is the signature of a serialization-bound quantity.

**Compute physics.** 512 tokens x 2 x 8e9 FLOP = 8.192 TFLOP has a floor of
4.55 ms at the B100 dense fp16 peak of about 1.8 PFLOP/s. The fixture spends
20.48 ms, about 22 percent of peak, inside the plausible band for a
small-batch prefill. The decode step spends exactly 2.0 ms, which is the
16 GB weight read at 8 TB/s, i.e. 100 percent of bandwidth and therefore an
optimistic bound; real single-request decode reaches 60 to 80 percent.

**System plausibility.** TTFT of 10.24 ms for a 256-token prefill and 499
tokens per second of undisturbed decode for one 8B request on one B100-class
GPU are at the optimistic end of published single-request behavior, and the KV
terms are sub-percent corrections to both: the KV write is 0.0409 percent of
TTFT and the decode KV read is 0.21 percent of a decode step. That ratio is
the physically correct one at batch size one, where weight traffic dominates
KV traffic; a KV term comparable to the compute term here would have been a
defect. The measured TPOT of 5.85 ms (171 tokens per second) is dominated not
by KV bytes but by the 15.37 ms scheduler gap while the pressure request
prefills, which is exactly how a real deployment's per-token latency degrades
under load.

## Family A: prefix reuse under capacity pressure

| C | hit blocks | evictions | TTFT at 8 TB/s | kv_ps | kernel_ps | TTFT at 4 TB/s | kv_ps |
|---|---|---|---|---|---|---|---|
| 32 | 0 | 24 | 20,488,388,608 | 8,388,608 | 20,480,000,000 | 20,496,777,216 | 16,777,216 |
| 40 | 8 | 16 | 15,366,291,456 | 6,291,456 | 15,360,000,000 | 15,372,582,912 | 12,582,912 |
| 44 | 12 | 12 | 12,805,242,880 | 5,242,880 | 12,800,000,000 | 12,810,485,760 | 10,485,760 |
| 48 | 16 | 8 | 10,244,194,304 | 4,194,304 | 10,240,000,000 | 10,248,388,608 | 8,388,608 |
| 56 | 16 | 0 | 10,244,194,304 | 4,194,304 | 10,240,000,000 | 10,248,388,608 | 8,388,608 |
| 64 | 16 | 0 | 10,244,194,304 | 4,194,304 | 10,240,000,000 | 10,248,388,608 | 8,388,608 |

The kernel term is identical at both HBM rates in every row, which is why it
appears once.

Every cell equals its frozen value to 0 ps. The eviction ladder is the
independent evidence that the vLLM ordering was modeled correctly: the filler
request reclaims A's private tail before the shared prefix, because vLLM frees
a request's blocks in reverse (`single_type_kv_cache_manager.py:503`) and
allocates from the LRU front (`block_pool.py:661`). If that ordering were
inverted, C=48 would lose the whole shared prefix and TTFT would double.

## Family B: preemption and recompute

| Arm | k | TTFT | TPOT | token-4 interval | evictions of D | D release | D freed |
|---|---|---|---|---|---|---|---|
| unconstrained | 16,384 | 10,244,194,304 | 5,845,808,128 | 17,370,534,912 | 0 | 0 | 0 |
| preempted | 16,384 | 10,244,194,304 | 7,935,808,128 | 25,730,534,912 | 16 | 17 | 1 |
| unconstrained | 32,768 | 10,248,388,608 | 5,851,616,256 | 17,381,069,824 | 0 | 0 | 0 |
| preempted | 32,768 | 10,248,388,608 | 7,941,616,256 | 25,741,069,824 | 16 | 17 | 1 |

All six step latencies in each arm match the frozen column to 0 ps. The first
three steps and the pressure step are bit-identical across arms, so the whole
TPOT difference is attributable to the resumed step, which runs 5.1711 times
longer than the decode step it replaces.

The accounting rows carry no time at all and are therefore independent
evidence: the preempted request releases 17 blocks, of which 16 hold full
blocks and become reclaimable while the partially filled tail block is freed
outright, and the pressure request then evicts exactly those 16. That
asymmetry is vLLM's own (`block_pool.py:738-740`, unhashed blocks are consumed
first), and it is why `FREE` exists in the vocabulary separately from `EVICT`.

## Evidence classes

Never summed across classes.

**Scored genuine-risk instances, 20 of 20 passed.** Twelve family A exact rows
(TTFT, kv_ps and kernel_ps per capacity and rate), four family B exact rows
(six step latencies, TTFT, exact rational TPOT and the token-4 interval per arm
and rate), two eviction-ladder rows and two preemption-accounting rows.

**Entailed relations, 17 of 17 held, unscored.** Direction, plateau, hit
ladder, bandwidth doubling of only the KV term, the 2.0000x magnitude, the
TPOT rise and its closed form, the unchanged TTFT under late constraint, and
the unconstrained control.

**Fatal and unscored, 0 of 56 violated.** The pool conservation identity
(`live + reclaimable + free == capacity`) after every cell, zero scheduler gap
in family A, the whole-replay write-byte total, the hit-token total, the
recomputed-token total, and the cross-arm identity of the four steps that
precede the divergence.

## Entailment analysis

The freeze scored the relation families themselves. That reading was too
generous and this report tightens it. An exact per-cell oracle already pins
every relation derived from the same measurement: once TTFT equals the frozen
table, its direction, its plateau, its 2.0000x magnitude and its bandwidth
scaling all follow arithmetically. The scored set is therefore the exact rows,
which nothing else pins, plus the two accounting facts that carry no time and
so cannot be entailed by any timing oracle. The derived relations are still
reported, and all held, but they are not counted as separate evidence. The
change is conservative: it lowers what is claimed as independent risk rather
than raising it, and no instance moved from failing to passing.

## Closure scope: CORE-3 stays open

Each registered clause, mapped to evidence.

1. "implement explicit KV lifecycle accounting before resource contention" -
   **met**. `KvLifecycleLedger` consumes a graph's KV work in preflight, before
   any resource is scheduled.
2. "Consume adapter observations for RESERVE, ALLOCATE, BIND_PREFIX, TOUCH,
   READ, WRITE, RETAIN/RELEASE, EVICT, FREE, SWAP, TRANSFER and RECOMPUTE" -
   **met for the consumption contract**, which handles all thirteen members and
   fails construction if the vocabulary grows without a rule. No live adapter
   emits them yet; the capture halves are VLLM-11 and SGL-9, and this study's
   observation stream comes from a vLLM-policy fixture that lives in the study,
   not in `simllm.core`.
3. "Enforce allocation, ownership, reference-count and byte-conservation
   invariants" - **met**, as I1 to I7, with unit coverage for each refusal.
4. "Add `examples/kv_cache_strategies/` with pre-registered vLLM and SGLang
   cases: no reuse, repeated system prefixes, competing prefix pools,
   multi-turn sessions, chunked prefill, capacity pressure, eviction,
   preemption/recompute, mixed contexts and bursts" - **partly met**. No reuse,
   repeated system prefixes, capacity pressure, eviction and
   preemption/recompute are pre-registered and executed. Competing prefix
   pools, multi-turn sessions, chunked prefill, mixed contexts, bursts and
   every SGLang case are not.
5. "Sweep capacity, block size, arrival rate, length, sharing and concurrency" -
   **partly met**. Capacity is swept over six levels and HBM rate over two.
   Block size, arrival rate, length, sharing and concurrency are not.
6. "report live/reserved/reclaimable bytes, fragmentation, hits, eviction
   reason and age, reads/writes, transfers, recompute, preemption, capacity
   wait and TTFT/TPOT tails" - **partly met**. Live, reserved, reclaimable and
   free blocks and bytes, hits, eviction reason, reads, writes, transfers and
   recompute are reported. Fragmentation, eviction age, a preemption counter,
   capacity wait and distribution tails are not.
7. "Acceptance must enable those same fixtures through the HBM service and
   preserve the explicit zero-byte path exactly" - **met**. Byte-carrying KV
   work is served from the HBM queue and attributed to `kv_ps`; with no pool
   declared it is still refused before authority mutation, and with a pool
   declared a zero-byte observation keeps every timestamp and every completion
   event bit-identical to the accepted baseline.

Because clauses 4, 5 and 6 are only partly met, **CORE-3 does not close**. Its
registry entry is narrowed to the remaining scope rather than removed, and
**zero new task IDs were registered**: a clause that its own entry still
carries does not need a second identity. CORE-50 and CORE-51 remain unused.

## Contradiction sweep

Statements this change makes stale, reported rather than edited:

- `docs/architecture.md:275`, "CORE-3 still owns explicit KV lifecycle". CORE-3
  still owns the case matrix, the remaining sweeps and the remaining reporting
  surface, but no longer the accounting or its path to a metric.
- `docs/architecture.md:210-213`, which says physical KV reads and writes lower
  to HBM operations, swap and remote movement lower to DMA plus network work,
  and recompute lowers to compute plus a KV write. The first is now true. Swap
  and transfer are accounted and byte-carrying but lower to the HBM queue
  rather than to DMA plus network; recompute is accounted as tokens and is not
  lowered at all.
- `docs/README_PRO.md:193-199`, which lists explicit KV lifecycle as a roadmap
  item to be "validated in a dedicated `examples/kv_cache_strategies/` study
  before KV bytes couple to resource contention". That study now exists and
  those bytes now couple to the HBM queue; what remains on this roadmap line
  is the adapter capture.
- `README.md:292`, "Planned on this axis: explicit KV-lifecycle capture". The
  capture halves VLLM-11 and SGL-9 are indeed still planned; the core-side
  accounting they will feed is not.

## What this does not claim

The serialized KV write is an upper bound on the KV term. A real attention
kernel overlaps the KV store with compute, so the same byte accounting under
an overlapped lowering yields identical bytes and a smaller time contribution;
the range runs from zero added time to the value reported here. The fixture's
32 to 64 block pools are a mechanism knob, not deployment sizing: a real 8B
deployment holds thousands of blocks. The recompute term that dominates both
headline moves is a replayed framework decision, priced by the compute
provider; what this change owns is the byte term, the pool state that decides
how many bytes there are, and the invariants that make an illegal stream fail
closed rather than quietly cost nothing.
