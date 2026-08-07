# GPU service model: post-specified regression expectations

This file, the initial implementation, the runner and the first results were
added together in commit `12bfe8b` on 2026-08-06. There is no earlier public
expectations-only ancestor, so these checks are post-specified regression
expectations, not preregistration. They use a synthetic 1 GHz GPU, so one cycle
is exactly 1,000 ps. Kernel launch overhead is zero. These checks validate
mechanisms, not A100/H100 timing accuracy.

## Boundary under test

The study exercises isolated service components only:

- CTA admission and deterministic assignment to SMs
- warp scheduler issue and dependency scoreboards
- register, warp, thread, block, and shared-memory residency limits
- isolated-kernel HBM latency plus bandwidth service
- isolated external copy-engine descriptor service

CUDA stream ordering, inter-kernel scheduling, copy-engine selection and
queueing, compute/copy overlap, and cross-operation HBM contention remain the
CORE-4 `DeviceRuntime` boundary and are not claimed here.

## A: partial CTA waves

Each CTA has one warp and consumes all shared memory, so exactly one CTA may
reside per SM. Its trace is 64 dependent ALU instructions with four-cycle
producer latency. One CTA therefore takes exactly 256 cycles. For B CTAs and S
SMs:

    cycles(B, S) = ceil(B / S) * 256

| B CTAs | S SMs | expected cycles |
|---:|---:|---:|
| 4 | 1 | 1,024 |
| 4 | 4 | 256 |
| 9 | 1 | 2,304 |
| 9 | 4 | 768 |

The residual must be zero in every cell. The final partial wave must remain
visible for B = 9, S = 4.

## B: scheduler width and dependency hiding

One resident CTA contains W ready warps. Each warp issues one independent ALU
instruction with latency four. With Q schedulers, one issue per scheduler per
cycle, and enough ALU ports:

    cycles(W, Q) = ceil(W / Q) - 1 + 4

| W warps | Q schedulers | expected cycles |
|---:|---:|---:|
| 4 | 1 | 7 |
| 4 | 4 | 4 |
| 16 | 1 | 19 |
| 16 | 4 | 7 |

A separate one-warp chain of eight dependent four-cycle instructions must take
exactly 32 cycles for both Q = 1 and Q = 4. More schedulers cannot shorten a
true dependency chain.

## C: occupancy is the minimum active constraint

The synthetic SM permits 16 CTAs, 64 warps, 2,048 threads, and 65,536 32-bit
registers. Register allocation granularity is one register for this fixture;
shared memory is non-limiting.

| threads/CTA | registers/thread | expected resident CTAs/SM |
|---:|---:|---:|
| 128 | 32 | 16 |
| 128 | 128 | 4 |
| 256 | 32 | 8 |
| 256 | 128 | 2 |

Every value must equal the minimum of the block, thread, warp, register, and
shared-memory limits. A launch that cannot admit one CTA must fail loudly.

## D: isolated HBM latency and serialization

One warp issues one dependent global-memory transaction at cycle zero. Fixed
return latency is L = 100 cycles. For D bytes and sustained service b
bytes/cycle:

    cycles(D, b) = L + D / b

| D bytes | b bytes/cycle | expected cycles |
|---:|---:|---:|
| 4,096 | 32 | 228 |
| 4,096 | 64 | 164 |
| 8,192 | 32 | 356 |
| 8,192 | 64 | 228 |

After subtracting L, doubling D doubles service exactly and doubling b halves
it exactly. Serviced HBM bytes must equal submitted bytes.

## E: isolated copy-engine service

The copy model excludes API launch delay and queue waiting. A descriptor has a
20-cycle engine setup and streams at b bytes/cycle:

    cycles(D, b) = 20 + D / b

| D bytes | b bytes/cycle | expected cycles |
|---:|---:|---:|
| 4,096 | 32 | 148 |
| 4,096 | 64 | 84 |
| 8,192 | 32 | 276 |
| 8,192 | 64 | 148 |

The residual must be zero. Effective throughput `D / cycles` must rise with D
and approach, but never exceed, the configured copy bandwidth.

## F: replay and seed-profile honesty

- Repeating any exact fixture must produce bit-identical cycles, duration,
  residency, issue counts, stall counts, and byte counts.
- Unknown opcodes, missing trace identity, impossible residency, unsupported
  cooperative/cluster launches, and incompatible copy directions must fail.
- The built-in A100 SXM 80 GB and H100 SXM 80 GB seed profiles must round-trip
  through the versioned artifact without losing source or uncertainty.
- Their architecture facts must match their declared public sources, but no
  bootstrap duration is accepted as silicon validation.
- COMP-1 and COMP-5 remain open. Later rented-GPU acceptance retains the
  existing bars: exact production-kernel identity coverage 100 percent,
  controlled measurement CV below 2 percent, held-out per-kernel median
  absolute percentage error below 10 percent and p95 below 20 percent.

All exact checks require zero-cycle residual and no capacity violation.
