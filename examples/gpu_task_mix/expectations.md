# GPU task-mix expectations

This study has no public expectations-only ancestor. Its initial harness also
computed the D2 and D3 comparison values from model runs before those values
were replaced with literals. The equations below are therefore a
post-specified regression specification, not publicly auditable
preregistration. The original D2 and D3 misses remain in the result ledger,
while corrected mechanism families guard what those misses taught us.

## Question

The GPU service model now schedules three kinds of task concurrently:
compute kernels, memory-bound kernels, and NCCL network kernels whose
stores leave the GPU on NVLink. Each kind is limited by a different
resource. This study registers what limits each kind in isolation, and
what happens to the makespan when two kinds share one GPU.

## Fixture

One synthetic 1 GHz GPU, so one cycle is exactly 1000 ps and every form
below is exact integer arithmetic. Built in `run_gpu_task_mix.py`:

| Property | Value |
|---|---|
| `sm_count` | swept, 1 or 2 |
| `scheduler_count_per_sm` x `dispatch_width_per_scheduler` | 4 issues per SM per cycle |
| ALU pipeline | latency 4 cycles, 4 lanes, initiation interval swept (1, 2, 4) |
| LOAD_STORE pipeline | latency 1 cycle, 4 lanes, initiation interval 1 |
| HBM | latency 100 cycles, bandwidth swept (32, 64) bytes per cycle |
| NVLink | latency 200 cycles, bandwidth swept (8, 16) bytes per cycle |
| residency limits | 16 blocks, 64 warps, 2048 threads per SM |

Registers and shared memory are left unconstrained except in check D3,
which uses shared memory deliberately.

Notation: `T` is pipeline latency in cycles, `I` its initiation interval,
`L` the number of pipeline lanes (4 here), `Q` the per-SM issue budget
(4 here), `s_hbm = ceil(D / b_hbm)` the HBM service of one `D`-byte
transaction, `s_nv = ceil(C / b_nv)` the NVLink service of one `C`-byte
chunk, `L_hbm = 100` and `L_nv = 200` the two latencies.

## A. Compute-intensive: the issue path is the limit

A compute kernel with no memory traffic is limited by how fast warps can
be issued into the pipelines, so its duration must scale with the number
of SMs and with the initiation interval, and must not depend on any
bandwidth.

- **A1 (pipeline throughput).** `W` warps of `K` independent ALU
  instructions each on one SM, with `W >= L`. Each cycle that a lane is
  free, `L` instructions issue; a lane taken at cycle `c` returns at
  `c + I`, so bursts of `L` land every `I` cycles. With `N = W * K`
  instructions the last burst issues at `(ceil(N / L) - 1) * I` and
  retires `T` cycles later:

  `duration = (ceil(N / L) - 1) * I + T`

  Swept over `I` in (1, 2, 4) at `W = 8`, `K = 4`, so `N = 32`:
  `I = 1` gives 11 cycles, `I = 2` gives 18, `I = 4` gives 32.

- **A2 (dependency latency).** One warp of `K` instructions each
  dependent on its predecessor issues one instruction per completed
  instruction, so `duration = K * T`. At `K = 8`, `T = 4`: 32 cycles.
  This cell must not move when `I` changes, because a dependent chain
  never has two instructions in flight.

- **A3 (SM scaling).** The same `W = 8` warps of `K = 4` spread over 2
  SMs put 4 warps on each, so `N` per SM halves and
  `duration = (ceil(N / (2 * L)) - 1) * I + T`. At `I = 1`: 7 cycles
  against 11 on one SM. Compute time must fall when SMs are added.

## B. Memory-intensive: one bandwidth cursor is the limit

Every HBM transaction is serialized on one GPU-wide cursor, so a
bandwidth-bound kernel must not care how many SMs or issue slots it has.

- **B1 (serialization form).** `N` transactions of `D` bytes with issue
  never the constraint: transaction `i` occupies the cursor over
  `[i * s_hbm, (i + 1) * s_hbm]` and retires `L_hbm` later, so

  `duration = N * s_hbm + L_hbm`

  Swept at `N = 32` over `D` in (64, 128) and `b_hbm` in (32, 64):
  `(64, 64)` gives 132 cycles, `(64, 32)` and `(128, 64)` both give 164,
  `(128, 32)` gives 228.

- **B2 (SM independence).** Every B1 cell must produce the identical
  duration at `sm_count = 1` and `sm_count = 2`. Adding SMs to a
  bandwidth-bound kernel must buy exactly nothing. This is the
  qualitative opposite of A3 and is the point of the check.

- **B3 (bandwidth scaling).** The serialization term is exactly
  proportional to service: `(duration(b) - L_hbm) = 2 * (duration(2b) -
  L_hbm)` whenever `s_hbm` doubles exactly, i.e. 64 against 32 cycles of
  serialization at `D = 64`.

## C. Network-intensive: the NVLink egress cursor is the limit

- **C1 (egress byte closed form).** A ring all-reduce of `P` bytes over
  `W` ranks sends exactly `2 * (W - 1) * P / W` bytes per GPU. The
  built kernel's `nvlink_transacted_bytes` must equal that integer
  exactly, and `hbm_transacted_bytes` must equal it too, because each
  chunk is loaded once and stored once. Swept over `P` in (65536,
  131072) and `W` in (2, 4, 8).

- **C2 (pure egress serialization).** A kernel of `N` independent
  NVLink stores of `C` bytes has the same cursor form as B1:

  `duration = N * s_nv + L_nv`

  Swept at `N = 32` over `C` in (64, 128) and `b_nv` in (8, 16):
  `(64, 16)` gives 328 cycles, `(64, 8)` and `(128, 16)` both give 456,
  `(128, 8)` gives 712.

- **C3 (NCCL is bounded by its egress and converges to it).** The
  NCCL ring kernel cannot beat the pure egress bound of C2 for the same
  byte count and chunk size, so its duration is at least
  `N_st * s_nv + L_nv`. Its excess over that bound is the cost of the
  load-to-store chain and must shrink monotonically as
  `warps_per_channel` rises through (1, 2, 4, 8), because more warps
  hide more of the chain behind the cursor. Registered as a direction,
  not a closed form: the model's convergence point is what this study
  measures, and a non-monotone excess is a failure.

## D. Mixed: what two kinds do to each other

- **D1 (two memory tasks are exactly additive).** Two memory tasks of
  `N1` and `N2` transactions run concurrently share one cursor, so the
  makespan must equal a single task of `N1 + N2` transactions:
  `(N1 + N2) * s_hbm + L_hbm`. At `N1 = N2 = 16`, `D = 64`,
  `b_hbm = 64`: 132 cycles, matching the B1 `N = 32` cell exactly.

- **D2 (shared issue path).** The memory task and pure-egress network task
  use different data cursors but share scheduler and load-store issue
  resources. The post-specified exact regression is 329 cycles when memory
  is submitted first. The corrected behavioral family has four instances:
  network-first measures 328 cycles; memory-first is exactly one cycle
  longer; widening only the scheduler budget or only the load-store lanes
  preserves 329 cycles; widening both removes the delay and recovers 328.

- **D3 (shared-memory residency).** Giving every CTA half of the SM's shared
  memory changes both isolation controls: compute takes 14 cycles and memory
  takes 229. The tasks cannot be co-resident, so their concurrent makespan is
  exactly `14 + 229 = 243` cycles and memory admission occurs at cycle 14.
  With shared-memory demand removed, the isolated controls are 7 and 132
  cycles and the concurrent result is 133, i.e. full overlap plus D2's
  one-cycle issue delay.

- **D4 (attribution conservation).** In every concurrent replay, the sum
  over tasks of `issued_instructions` must equal the replay's total, and
  the same must hold for `hbm_transacted_bytes` and
  `nvlink_transacted_bytes`. Exact equality, no tolerance.

## E. Follow-up: an actual NCCL task beside memory

This post-specified extension follows the correction that made the ring trace
genuinely double buffered. The isolated controls come from B1 and C3; the
regression ledger retains the mixed-result band below.

The network task is `nccl_ring_allreduce_task` at payload 65,536 bytes,
world size 2, two channels, 64-byte chunks and eight warps per channel.
It is submitted first beside the B1 memory task with 32 64-byte HBM
transactions on the same two-SM fixture.

- **E1 (real ring attribution).** The ring task must retain kind `network`,
  issue 1,024 HBM loads and 1,024 NVLink stores, and attribute exactly
  65,536 requested and transacted bytes to each cursor.
- **E2 (mixed conservation).** The concurrent result must contain 2,080
  issued instructions, 67,584 requested and transacted HBM bytes over
  1,056 HBM requests, and 65,536 requested and transacted NVLink bytes
  over 1,024 NVLink requests. Per-task sums must equal every total.
- **E3 (overlap band).** The established isolated controls are 4,397
  cycles for the ring task and 132 cycles for the memory task. Adding the
  HBM-sharing task cannot make the ring finish sooner, so the mixed
  makespan must be at least 4,397 cycles. Because both tasks admit
  together and the memory return latency overlaps the ring's independent
  NVLink drain, it must remain strictly below the serialized sum of 4,529
  cycles. The post-specified regression band is `[4,397, 4,529)`.

## Evidence classes and chronology

The harness executes 38 distinct deterministic run configurations through 46
replay invocations. It reports four classes separately and never adds their
counts together:

- Exact-oracle rows compare a replay field with an independently written
  integer oracle.
- Behavioral relations compare two or more replay configurations. A family
  passes only if every parameterized instance passes.
- Structural invariants check conservation, builder symmetry or a bound that
  follows from the same mechanism under test. They are fatal when violated
  but unscored.
- Historical ledger rows preserve superseded registrations and are never
  interpreted as current failures.

The initial 2026-08-06 run registered D2 as 328 cycles and measured 329. It
registered D3 as 132 cycles and measured 243. Those targets were written after
model execution, and the original harness had previously computed comparison
values from model outputs. On 2026-08-07 the two misses were retained as
`H-D2` and `H-D3`, while the corrected D2 and D3 relations above became the
active post-specified regression. This chronology does not support a
preregistration claim.

## Failure policy

The default harness exits non-zero if any active exact oracle, behavioral
relation or structural invariant misses. Historical ledger rows do not affect
the exit status. Structural and by-construction rows are unscored and carry no
behavioral evidential weight.
