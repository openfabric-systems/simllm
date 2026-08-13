# KV lifecycle accounting v1 expectations

This is the expectations-only record for CORE-3, explicit KV lifecycle
accounting before resource contention. It freezes the KV vocabulary's
semantics, the pool state machine and its invariants, the one seam through
which KV bytes reach a reported metric, and the exact numbers a live run must
reproduce.

Nothing here was measured. Every number below comes from the arithmetic in
this file, and no value in it may be edited to agree with an observation.

## Claim boundary and chronology

The evidence is authored against SimLLM commit
`4e1be35af5327c27db53ed002dc420e1de6f613b`. At that commit
`simllm/core/runtime.py` treats every `KvCacheWork` operation as a zero-cost
lifecycle marker and refuses a byte-carrying `READ` or `WRITE` during
preflight, so no KV lifecycle state exists anywhere in the repository and no
KV byte can reach TTFT or TPOT. The check-only run of `run_study.py` in this
same commit asserts exactly that absence before any implementation lands.

This registration claims four things and nothing more:

1. a pool-level KV lifecycle authority consumes the observed vocabulary in
   graph order, before any resource contention, and enforces allocation,
   ownership, reference-count, capacity and byte-conservation invariants;
2. that authority's byte accounting reaches TTFT and TPOT through one new
   lowering, the HBM queue, and its explicit off path preserves the accepted
   zero-byte behavior exactly;
3. under a constrained cache a replayed request's TTFT and TPOT move up by
   the exact frozen amounts, and above the constraint threshold they do not
   move at all;
4. the KV byte term of a resumed step after preemption is *smaller* than the
   term of the decode step it replaces, by exactly one token equivalent,
   while the step as a whole is much longer.

It does not model contention (CORE-27 owns data-mover resources), does not
decide admission, eviction or prefix reuse, does not capture from a live
framework (VLLM-11 and SGL-9 own the capture halves), and does not touch
`simllm/traffic`. The fixture below is a mechanism fixture with deliberately
tiny pools; it is not a deployment sizing claim.

## Derived semantics: pinned vLLM 0.26.0

Every rule in the state machine is derived from the pinned sources at
`vllm==0.26.0` (`Version: 0.26.0` in the installed `dist-info/METADATA`;
`vllm/version.py` only re-exports `._version`). Paths are relative to the
installed `vllm` package root. Line numbers were read at the cited offsets.

| Fact | Citation | Semantics |
|---|---|---|
| Pool is a fixed array of blocks | `vllm/v1/core/block_pool.py:175` | `self.blocks = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]`; capacity is fixed at construction |
| Free list is the eviction order | `vllm/v1/core/block_pool.py:181`, `vllm/v1/core/kv_cache_utils.py:184` | `free_block_queue = FreeKVCacheBlockQueue(self.blocks)` |
| LRU end is the front | `vllm/v1/core/kv_cache_utils.py:195` | "1. The least recent used block is at the front (LRU)." |
| A block carries a reference count | `vllm/v1/core/kv_cache_utils.py:124` | `ref_cnt: int = 0`; there is no `incr_ref`/`decr_ref` in 0.26.0, the pool mutates it inline |
| Allocation pops the LRU front | `vllm/v1/core/block_pool.py:661` | `ret = self.free_block_queue.popleft_n(num_blocks)` |
| Eviction is lazy and happens at reuse | `vllm/v1/core/block_pool.py:666`, `:679` | `self._maybe_evict_cached_block(block)` runs inside `get_new_blocks`, immediately before `block.ref_cnt += 1` at `:668` |
| Eviction resets the hash | `vllm/v1/core/kv_cache_utils.py:159` | `KVCacheBlock.reset_hash()`, "Reset the block hash when the block is evicted." |
| Reuse increments the count and leaves the free queue | `vllm/v1/core/block_pool.py:702-715` | `touch()`: "increases its reference count by 1, and may remove the block from the free queue"; the removal is guarded by `if block.ref_cnt == 0` |
| A prefix hit touches the hit blocks | `vllm/v1/core/single_type_kv_cache_manager.py:260` | `self.block_pool.touch(new_computed_blocks)`, "make sure they won't be evicted" |
| Release decrements, it does not discard | `vllm/v1/core/block_pool.py:731-740` | `block.ref_cnt -= 1`; at zero a hashed block is `append_n`-ed (stays reusable) and an unhashed one is `prepend_n`-ed ("Blocks without hash always get evicted first") |
| Release order is reversed per request | `vllm/v1/core/single_type_kv_cache_manager.py:503` | `free_blocks(reversed(pop_blocks_for_free(request_id)))`, so a request's tail block is evicted before its head, and a shared prefix survives longest |
| The hit is a chained-hash prefix run | `vllm/v1/core/single_type_kv_cache_manager.py:708-711` | "A missing block implies every later block misses too (chained hashes)" |
| A full hit still recomputes one token | `vllm/v1/core/kv_cache_manager.py:237` | `max_cache_hit_length = request.num_tokens - 1` |
| The hit is realized by fast-forwarding the cursor | `vllm/v1/core/sched/scheduler.py:1031` | `request.num_computed_tokens = num_computed_tokens`, so only `num_tokens - num_computed_tokens` tokens reach the model |
| Capacity refusal is a scheduler-visible fact | `vllm/v1/core/kv_cache_manager.py:462-466` | `available_blocks = get_num_free_blocks() - reserved_blocks`; `if required_blocks > available_blocks: return None` |
| Touch precedes new allocation | `vllm/v1/core/kv_cache_manager.py:474`, `:481` | `allocate_new_computed_blocks(...)` runs before `allocate_new_blocks(...)`, so hit blocks cannot be evicted by the same request's own allocation |
| Preemption frees and zeroes the cursor | `vllm/v1/core/sched/scheduler.py:1212`, `:1221`, `:1225`, `:1233` | `_preempt_request` frees the blocks, sets `request.num_computed_tokens = 0` and prepends the request to the waiting queue |
| Preemption is recompute-only in v1 | `vllm/v1/core/sched/scheduler.py` contains no `swap`; `PreemptionMode` does not exist anywhere in 0.26.0 | the v0 swap fallback is gone |
| Host-tier KV movement is an opt-in offload connector | `vllm/v1/kv_offload/cpu/gpu_worker.py:394`, `vllm/_custom_ops.py:2656` | `swap_blocks(...)` exists only on the offload path, not in the scheduler |
| Bytes per block per layer | `vllm/v1/kv_cache_interface.py:203-218` | `2 * block_size * num_kv_heads * head_dim * dtype_size` |
| Blocks from memory | `vllm/v1/core/kv_cache_utils.py:1005` | `num_blocks = int(available_memory // page_size // num_layers)` |

Two vocabulary decisions follow directly from those sources and are frozen
here because a different split would double count:

- `BIND_PREFIX` is the reuse *decision* (`find_longest_cache_hit` and the hit
  length), and `TOUCH` is the *mechanism* that raises the reference count and
  removes a reclaimable block from the free queue (`block_pool.touch` with
  `ref_cnt == 0`). `RETAIN` is the same call's other branch, an additional
  owner on a block that is already live (`ref_cnt > 0`, no queue removal).
  Only `BIND_PREFIX` accrues hit counters, and only `TOUCH`/`RETAIN` change
  reference counts, so a hit is never counted twice.
- `EVICT` and `FREE` both discard cached content at reference count zero.
  `EVICT` is involuntary reclamation and carries a required cause
  (`_maybe_evict_cached_block` inside `get_new_blocks`); `FREE` is voluntary
  return of a block that holds no reusable content (the
  `prepend_n(blocks_without_hash)` half of `free_blocks`, and
  `reset_prefix_cache`). `RELEASE` is only the reference-count drop.

`SWAP` and `TRANSFER` are accounted but are not exercised by a vLLM v1
capture unless the offload connector is enabled, and this study does not
enable it.

## Frozen state machine

One pool has a fixed `capacity_blocks`, a fixed `block_bytes` and a fixed
`block_tokens`. Every block is in exactly one of three states:

- `FREE`: reference count 0, holds no reusable content;
- `LIVE`: reference count at least 1, held by one or more owners;
- `RECLAIMABLE`: reference count 0, holds reusable content, may be hit by
  `BIND_PREFIX` or reclaimed by `EVICT`.

Frozen transitions, with the observation that causes each one:

| Action | Precondition | Effect |
|---|---|---|
| `RESERVE` | request named, no block IDs | records a block demand for the request; fails if the demand exceeds `capacity_blocks - live_blocks` |
| `ALLOCATE` | every named block `FREE` | `LIVE`, reference count 1, owner is the request; consumes the request's outstanding reservation |
| `BIND_PREFIX` | every named block `RECLAIMABLE` or `LIVE` | no state change; accrues hit blocks and hit tokens |
| `TOUCH` | every named block `RECLAIMABLE` | `LIVE`, reference count 1, owner added |
| `RETAIN` | every named block `LIVE` | reference count + 1, owner added |
| `READ` / `WRITE` | every named block `LIVE` and owned by the request | byte-carrying; no state change |
| `RELEASE` | every named block `LIVE` and owned by the request | reference count - 1, owner removed; at zero becomes `RECLAIMABLE` if it holds content, else `FREE` |
| `EVICT` | every named block `RECLAIMABLE`, cause required | `FREE`, content discarded |
| `FREE` | every named block `RECLAIMABLE` or `FREE` | `FREE`, content discarded |
| `SWAP` | every named block `LIVE` or `RECLAIMABLE` | byte-carrying; changes the block's tier |
| `TRANSFER` | every named block `LIVE` | byte-carrying; no state change |
| `RECOMPUTE` | request named | records recomputed tokens; no block state change |

A block acquires reusable content when the tokens it covers are written, so
`ALLOCATE` followed by `WRITE` leaves it `LIVE` with content, and its later
`RELEASE` therefore lands in `RECLAIMABLE`. A block that is allocated and
never written holds no content and its `RELEASE` lands in `FREE`; this is the
partial-tail-block case, and it is the reason `FREE` exists separately from
`EVICT`.

Frozen invariants. Every one of these is fatal and unscored: a violation
voids the run rather than costing a point.

- I1 allocation: `ALLOCATE` never names a block that is `LIVE` or
  `RECLAIMABLE`. Reclaiming a cached block requires an explicit preceding
  `EVICT`, which is what makes eviction visible instead of implied.
- I2 capacity: `live + reclaimable + free == capacity_blocks` after every
  observation, and no observation drives any of the three negative.
- I3 ownership: `READ`, `WRITE`, `RELEASE` and `TRANSFER` name only blocks the
  observing request currently holds.
- I4 reference count: never negative; `EVICT` and `FREE` require zero; a
  supplied `reference_count` field must equal the ledger's own value.
- I5 byte conservation: a byte-carrying action satisfies
  `byte_count == (token_end - token_start) * block_bytes / block_tokens` when
  it declares a token interval, and otherwise
  `byte_count <= len(block_ids) * block_bytes`. A zero-byte action stays
  exactly zero-byte.
- I6 reservation: a request's total allocated blocks never exceed its
  outstanding reservation when it made one.
- I7 ordering: the ledger consumes operations in graph order, entirely inside
  preflight, and a failed preflight leaves the previous pool state unchanged.

## Frozen live seam and its off path

`KvCacheWork` byte accounting reaches a metric through exactly one new
lowering: a byte-carrying `READ`, `WRITE`, `SWAP` or `TRANSFER` occupies the
per-GPU HBM queue for

```
service_ps = ceil(byte_count * 8 * 10**12 / hbm_rate_bps)
```

and the operation completes at `finished + completion_delivery_ps`, the same
shape the DMA path already uses. `_attribution_field` already maps an
HBM-queue visit owned by `KvCacheWork` to `kv_ps`, so the term arrives in
`LatencyAttribution.kv_ps`, then in `RequestMetric`, then in TTFT and TPOT.

Off path, frozen exactly:

- with no KV pool configured, or with `hbm_rate_bps == 0`, a byte-carrying
  `READ` or `WRITE` is refused in preflight exactly as it is today, before any
  authority mutation;
- a zero-byte `KvCacheWork` operation keeps its current timing to the
  picosecond, with or without a pool configured, and adds no HBM visit;
- configuring a pool changes no timestamp of any graph that carries no
  byte-carrying KV work.

## Fixture geometry

A single-rank mechanism fixture. Its geometry is the published Llama-3.1-8B
attention shape, chosen so the byte arithmetic is checkable against real
hardware, not so the pool sizes below resemble a deployment.

| Symbol | Value | Source |
|---|---|---|
| layers | 32 | Llama-3.1-8B |
| KV heads | 8 | Llama-3.1-8B grouped-query attention |
| head size | 128 | Llama-3.1-8B |
| dtype size | 2 B | fp16 |
| `block_tokens` | 16 | vLLM default page size in tokens |
| page bytes per layer | 65,536 B | `2 * 16 * 8 * 128 * 2`, `vllm/v1/kv_cache_interface.py:213-218` |
| `block_bytes` | 2,097,152 B | 32 layers x 65,536 B |
| bytes per token | 131,072 B | `block_bytes / block_tokens` |
| `hbm_rate_bps` fast | 64,000,000,000,000 | 8 TB/s, B100-class HBM3e |
| `hbm_rate_bps` slow | 32,000,000,000,000 | 4 TB/s, the halved comparator |
| k, KV ps per token, fast | 16,384 ps | 131,072 B / 8 B per ps |
| k, KV ps per token, slow | 32,768 ps | 131,072 B / 4 B per ps |
| prefill compute | 40,000,000 ps per token | 16 GFLOP per token at 400 TFLOP/s effective |
| decode compute | 2,000,000,000 ps per step | 16 GB of weights at 8 TB/s |

## Physical sanity, floors and ceilings

Stated before any measurement.

- KV write floor. R's cold prefill writes 512 x 131,072 B = 67,108,864 B. At
  8 TB/s no engine beats 8,388,608 ps (8.39 us); at 4 TB/s, 16,777,216 ps.
  The model sits exactly on that floor, so it is a lower bound on reality;
  real HBM efficiency of 70 to 90 percent puts the true cost at 9.3 to 12 us.
- Prefill compute floor. 512 tokens x 2 x 8e9 FLOP = 8.192 TFLOP. At the
  B100 dense fp16 peak of about 1.8 PFLOP/s the floor is 4.55 ms. The fixture
  spends 20.48 ms, i.e. about 22 percent of peak, inside the plausible 15 to
  50 percent band for a small-batch prefill.
- Decode floor. An 8B fp16 model reads 16 GB of weights per step; at 8 TB/s
  the floor is 2.0 ms. The fixture spends exactly 2.0 ms, i.e. 100 percent of
  bandwidth, which is an optimistic bound; real single-request decode runs at
  60 to 80 percent, i.e. 2.5 to 3.3 ms.
- Decode KV read share. 256 tokens x 131,072 B = 33,554,432 B, 4.19 us at
  8 TB/s, which is 0.21 percent of a 2 ms step. That is the correct order for
  a single request: KV traffic only rivals weight traffic at large batch.
- End-to-end plausibility. TTFT of 10.24 ms for a 256-token prefill and about
  500 tokens per second of decode for one 8B request on one B100-class GPU are
  at the optimistic end of published single-request behavior, and the KV terms
  are sub-percent corrections to both. A KV term that came out comparable to
  the compute term at batch size one would be a defect, not a finding.
- Pool sizing. 32 to 64 blocks is 64 to 128 MiB of KV, far below a real
  deployment's tens of gigabytes. It is the constraint knob of this fixture,
  chosen so eviction is reachable in a three-step replay.

## Family A: prefix reuse under capacity pressure, TTFT

Three requests replay on one rank, one step each, on the same pool.

- Step 0: request A prefills 384 tokens, i.e. the 256-token shared system
  prefix S in blocks `a0..a15` plus 128 private tokens in `a16..a23`. A then
  releases all 24 blocks in reverse order.
- Step 1: request F prefills 512 unrelated tokens, 32 blocks, then releases
  them in reverse order.
- Step 2: request R prefills 512 tokens whose first 256 are S.

The free queue after step 0 is `[unused 24..C-1] ++ [a23 .. a0]`, so F's 32
blocks first consume the never-used blocks and then evict A's blocks
tail-first. With `E = max(0, 32 - (C - 24))` evictions, the surviving prefix of
S is `16 - max(0, E - 8)` blocks, and R's chained-hash hit is exactly that run.
`max_cache_hit_length = 511` caps the hit at 31 blocks and is not binding here.

| Capacity C | Evictions E in step 1 | R hit blocks h | R hit tokens | R new blocks | R recomputed tokens |
|---|---|---|---|---|---|
| 32 | 24 | 0 | 0 | 32 | 512 |
| 40 | 16 | 8 | 128 | 24 | 384 |
| 44 | 12 | 12 | 192 | 20 | 320 |
| 48 | 8 | 16 | 256 | 16 | 256 |
| 56 | 0 | 16 | 256 | 16 | 256 |
| 64 | 0 | 16 | 256 | 16 | 256 |

R's step is the only step that samples R, so R's TTFT is exactly its step
latency:

```
TTFT(C, k) = recomputed_tokens(C) * (40,000,000 + k)
kv_ps(C, k) = recomputed_tokens(C) * k
kernel_ps(C)  = recomputed_tokens(C) * 40,000,000
```

Frozen values, in picoseconds:

| C | TTFT at k=16,384 | kv_ps at k=16,384 | TTFT at k=32,768 | kv_ps at k=32,768 |
|---|---|---|---|---|
| 32 | 20,488,388,608 | 8,388,608 | 20,496,777,216 | 16,777,216 |
| 40 | 15,366,291,456 | 6,291,456 | 15,372,582,912 | 12,582,912 |
| 44 | 12,805,242,880 | 5,242,880 | 12,810,485,760 | 10,485,760 |
| 48 | 10,244,194,304 | 4,194,304 | 10,248,388,608 | 8,388,608 |
| 56 | 10,244,194,304 | 4,194,304 | 10,248,388,608 | 8,388,608 |
| 64 | 10,244,194,304 | 4,194,304 | 10,248,388,608 | 8,388,608 |

Registered relations for family A:

- A1 direction: TTFT is non-increasing in C, and TTFT(32) > TTFT(64).
- A2 plateau: TTFT(48) == TTFT(56) == TTFT(64) to 0 ps at both rates. Above
  the constraint threshold, capacity does not move the metric at all.
- A3 exactness: every cell equals the table above to 0 ps.
- A4 bandwidth: `kv_ps` at 4 TB/s is exactly twice `kv_ps` at 8 TB/s in every
  cell, while `kernel_ps` is bit-identical across the two rates.
- A5 decomposition: the reported `kv_ps` equals `recomputed_tokens * k` and
  the reported `kernel_ps` equals `recomputed_tokens * 40,000,000`, so the
  KV term is separated from the replayed compute term rather than inferred.
- A6 accounting: the ledger reports hit blocks, evictions and written bytes
  matching the ladder table exactly, and the eviction count in step 1 equals
  E.
- A7 magnitude: at C=32 the constrained TTFT is 2.0000 times the
  unconstrained TTFT (both terms are linear in recomputed tokens), and the
  KV share of TTFT is 0.041 percent at 8 TB/s.

## Family B: preemption and recompute, TPOT

One request D and one pressure request G replay on one rank.

- Step 0: D prefills 256 tokens, 16 blocks. This is D's first token.
- Step 1: D decodes token 2. Context 256; it allocates one new block.
- Step 2: D decodes token 3. Context 257.
- Step 3: G prefills 384 tokens, 24 blocks. D is not scheduled in this step in
  either arm, so the resulting scheduler gap is identical in both arms and
  cancels from the comparison.
- Step 4: in the unconstrained arm D decodes token 4 with context 258. In the
  constrained arm D was preempted during step 3, so step 4 recomputes all 258
  tokens, rewrites their KV and yields token 4.
- Step 5: D decodes token 5. Context 259.

A prefill step emits `WRITE` only, because its attention reads the KV it is
writing inside the same kernel. A decode step emits a `READ` of the resident
context and a `WRITE` of the single new token.

Step latencies in picoseconds, with k the KV cost of one token:

```
step 0 = 256 * 40,000,000 + 256 * k
step 1 = 2,000,000,000 + 257 * k
step 2 = 2,000,000,000 + 258 * k
step 3 = 384 * 40,000,000 + 384 * k
step 4 unconstrained = 2,000,000,000 + 259 * k
step 4 preempted     = 258 * 40,000,000 + 258 * k
step 5 = 2,000,000,000 + 260 * k
```

D's four inter-token intervals are steps 1, 2, (3 + 4) and 5, and TPOT is
their exact rational mean.

| Arm | k | TTFT | TPOT | token-4 interval |
|---|---|---|---|---|
| unconstrained | 16,384 | 10,244,194,304 | 5,845,808,128 | 17,370,534,912 |
| preempted | 16,384 | 10,244,194,304 | 7,925,804,032 | 25,690,518,528 |
| unconstrained | 32,768 | 10,248,388,608 | 5,851,616,256 | 17,381,069,824 |
| preempted | 32,768 | 10,248,388,608 | 7,931,608,064 | 25,701,037,056 |

Registered relations for family B:

- B1 direction: preemption raises TPOT. The rise is 2,079,995,904 ps at
  k=16,384 and 2,079,991,808 ps at k=32,768, i.e. a factor of 1.3558 and
  1.3554.
- B2 closed form: `TPOT_delta = (8,320,000,000 - k) / 4` exactly.
- B3 sign of the byte term: the resumed step writes 258 token equivalents
  while the decode step it replaces moves 259 (a 258-token read plus a
  one-token write), so the resumed step's KV byte term is *smaller* by exactly
  one token equivalent, k ps, even though the step is 5.15 times longer. A
  model that assumed constraint always adds KV bytes gets this sign wrong.
- B4 TTFT unchanged: D's TTFT is identical in both arms to 0 ps, because the
  preemption happens after the first token. Constraint that arrives late must
  not retroactively move TTFT.
- B5 unconstrained control: with capacity above the threshold no preemption is
  observed, no eviction of D's blocks is accounted, and every step latency
  equals the unconstrained column to 0 ps.
- B6 accounting: the ledger reports 17 released blocks and 16 evicted blocks
  for D in the preempted arm (its partial tail block holds no full-block
  content and is freed, not evicted) and zero evictions of D's blocks in the
  unconstrained arm.

## Evidence classes and scoring

Separate classes, never summed:

- Fatal and unscored: I1 to I7, the off-path preservation guards, the
  conservation identity `live + reclaimable + free == capacity`, and the
  by-construction zero assertions (zero-byte timing neutrality, identical
  scheduler gaps across the family B arms).
- Behavioral, genuine risk, scored: A1, A2, A4, A5, A6, B1, B2, B3, B4, B6.
- Entailed and therefore reported but unscored: A3 is entailed by A5 plus the
  step-latency identity once A6 fixes the ladder, and A7 is entailed by A3.
  B5 is entailed by B4 plus the unconstrained column of A. They are reported
  as consistency rows, not as separate scored instances.

Entailment analysis. The exact TTFT table (A3) is not independent evidence
once the ladder (A6) and the decomposition (A5) both hold, because
`TTFT = kernel_ps + kv_ps` is a step-latency identity the reducer already
enforces through its attribution conservation check. The genuinely risky
predictions are the ladder itself, which depends on getting vLLM's reversed
free order and lazy eviction right; the plateau, which fails if capacity leaks
into timing anywhere; the bandwidth scaling of only the KV term; and the sign
of B3, which no by-construction argument supplies.

## Not claimed

The registered CORE-3 entry asks for more than this study delivers, and the
following clauses are deliberately out of this freeze: SGLang cases, arrival
rate and concurrency sweeps, fragmentation and reclaimable-byte tails,
competing prefix pools, chunked prefill, mixed contexts and bursts, and the
swap and cross-pool transfer paths. This freeze does not claim them, and the
closure report must state which registered clauses remain open rather than
implying the whole entry is met.

The serialized KV write in this fixture is an upper bound on the KV term: real
attention kernels overlap the KV store with compute, so the same byte
accounting under an overlapped lowering yields the same bytes and a smaller
time contribution. The lower bound of that range is zero added time, and the
upper bound is the value this study reports.
