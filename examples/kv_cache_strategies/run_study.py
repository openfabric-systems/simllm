"""Replay one request against a constrained KV cache and measure TTFT/TPOT.

CORE-3: SimLLM had no KV lifecycle state at all. Every ``KvCacheWork``
operation was a zero-cost marker and a byte-carrying READ or WRITE was refused
in preflight, so a replayed request's timing could not respond to memory
pressure. This study lands the pool authority, its invariants and the one
lowering that lets its bytes reach a reported metric.

Every literal below was frozen in ``expectations.md`` before any
implementation existed, and none of them may be edited to agree with an
observation. ``--check-only`` proves the arithmetic is self consistent and
that the mechanism is still absent; it is the pre-implementation dry run.

The fixture geometry is the published Llama-3.1-8B attention shape so the byte
arithmetic is checkable against real hardware. The pool sizes are a mechanism
knob, not a deployment sizing claim.
"""

from __future__ import annotations

import argparse
import itertools
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Frozen fixture geometry. Quoted from expectations.md.
# ---------------------------------------------------------------------------

LAYERS = 32
KV_HEADS = 8
HEAD_SIZE = 128
DTYPE_BYTES = 2
BLOCK_TOKENS = 16

#: 2 * block_tokens * kv_heads * head_size * dtype_bytes, per layer
PAGE_BYTES_PER_LAYER = 2 * BLOCK_TOKENS * KV_HEADS * HEAD_SIZE * DTYPE_BYTES
BLOCK_BYTES = LAYERS * PAGE_BYTES_PER_LAYER
BYTES_PER_TOKEN = BLOCK_BYTES // BLOCK_TOKENS

#: 8 TB/s and the halved comparator, expressed in bits per second
HBM_RATE_FAST_BPS = 64_000_000_000_000
HBM_RATE_SLOW_BPS = 32_000_000_000_000

#: KV picoseconds per token at each rate
K_FAST = 16_384
K_SLOW = 32_768

PREFILL_PS_PER_TOKEN = 40_000_000
DECODE_STEP_PS = 2_000_000_000

# ---------------------------------------------------------------------------
# Frozen family A ladder and tables.
# ---------------------------------------------------------------------------

A_PREFIX_TOKENS = 256
A_PRIVATE_TOKENS = 128
F_TOKENS = 512
R_TOKENS = 512

CAPACITIES = (32, 40, 44, 48, 56, 64)

#: capacity -> (evictions during step 1, R hit blocks)
FROZEN_LADDER: dict[int, tuple[int, int]] = {
    32: (24, 0),
    40: (16, 8),
    44: (12, 12),
    48: (8, 16),
    56: (0, 16),
    64: (0, 16),
}

#: (capacity, k) -> (ttft_ps, kv_ps)
FROZEN_FAMILY_A: dict[tuple[int, int], tuple[int, int]] = {
    (32, K_FAST): (20_488_388_608, 8_388_608),
    (40, K_FAST): (15_366_291_456, 6_291_456),
    (44, K_FAST): (12_805_242_880, 5_242_880),
    (48, K_FAST): (10_244_194_304, 4_194_304),
    (56, K_FAST): (10_244_194_304, 4_194_304),
    (64, K_FAST): (10_244_194_304, 4_194_304),
    (32, K_SLOW): (20_496_777_216, 16_777_216),
    (40, K_SLOW): (15_372_582_912, 12_582_912),
    (44, K_SLOW): (12_810_485_760, 10_485_760),
    (48, K_SLOW): (10_248_388_608, 8_388_608),
    (56, K_SLOW): (10_248_388_608, 8_388_608),
    (64, K_SLOW): (10_248_388_608, 8_388_608),
}

# ---------------------------------------------------------------------------
# Frozen family B tables.
# ---------------------------------------------------------------------------

D_PROMPT_TOKENS = 256
D_DECODE_TOKENS = 4
G_TOKENS = 384

#: capacity that forces the preemption, and the capacity that does not
B_CAPACITY_TIGHT = 24
B_CAPACITY_LOOSE = 64

#: (arm, k) -> (ttft_ps, tpot_ps, token_4_interval_ps)
FROZEN_FAMILY_B: dict[tuple[str, int], tuple[int, Fraction, int]] = {
    ("unconstrained", K_FAST): (
        10_244_194_304,
        Fraction(5_845_808_128),
        17_370_534_912,
    ),
    ("preempted", K_FAST): (
        10_244_194_304,
        Fraction(7_935_808_128),
        25_730_534_912,
    ),
    ("unconstrained", K_SLOW): (
        10_248_388_608,
        Fraction(5_851_616_256),
        17_381_069_824,
    ),
    ("preempted", K_SLOW): (
        10_248_388_608,
        Fraction(7_941_616_256),
        25_741_069_824,
    ),
}

#: the frozen TPOT rise caused by the preemption, identical at every HBM rate
#: because the resumed step and the decode step it replaces move exactly the
#: same 259 token equivalents of KV
FROZEN_TPOT_DELTA_PS = 2_090_000_000

#: blocks D releases and blocks of D that are evicted, preempted arm
FROZEN_D_RELEASED_BLOCKS = 17
FROZEN_D_EVICTED_BLOCKS = 16


def _fail(name: str, expected: object, observed: object) -> None:
    raise AssertionError(f"{name}: expected {expected!r}, observed {observed!r}")


def _equal(name: str, expected: object, observed: object) -> None:
    if expected != observed:
        _fail(name, expected, observed)


def kv_ps_per_token(rate_bps: int) -> int:
    """Return the frozen HBM service of one token of KV at ``rate_bps``."""

    return -(-BYTES_PER_TOKEN * 8 * 10**12 // rate_bps)


def family_a_ladder(capacity: int) -> tuple[int, int]:
    """Return (evictions in step 1, R hit blocks) for one pool capacity.

    Derived from vLLM 0.26.0: a request's blocks are freed in reverse order
    (``single_type_kv_cache_manager.py:503``) and allocation pops the LRU front
    (``block_pool.py:661``), so A's private tail is reclaimed before the shared
    prefix, and the surviving hit is a chained-hash prefix run
    (``single_type_kv_cache_manager.py:708-711``).
    """

    a_blocks = (A_PREFIX_TOKENS + A_PRIVATE_TOKENS) // BLOCK_TOKENS
    f_blocks = F_TOKENS // BLOCK_TOKENS
    prefix_blocks = A_PREFIX_TOKENS // BLOCK_TOKENS
    private_blocks = a_blocks - prefix_blocks
    evictions = max(0, f_blocks - (capacity - a_blocks))
    hit_blocks = prefix_blocks - max(0, evictions - private_blocks)
    return evictions, max(0, hit_blocks)


def family_a_expected(capacity: int, k: int) -> tuple[int, int]:
    """Return the frozen (TTFT, kv_ps) closed form for one family A cell."""

    _, hit_blocks = family_a_ladder(capacity)
    recomputed = R_TOKENS - hit_blocks * BLOCK_TOKENS
    return recomputed * (PREFILL_PS_PER_TOKEN + k), recomputed * k


def family_b_steps(arm: str, k: int) -> tuple[int, ...]:
    """Return the frozen step latencies of family B, in graph order."""

    prompt = D_PROMPT_TOKENS
    #: preemption zeroes the token cursor while the sequence keeps its three
    #: generated tokens, so the resumed step recomputes every one of them
    resumed_tokens = prompt + 3
    step_4 = (
        resumed_tokens * PREFILL_PS_PER_TOKEN + resumed_tokens * k
        if arm == "preempted"
        else DECODE_STEP_PS + (prompt + 3) * k
    )
    return (
        prompt * PREFILL_PS_PER_TOKEN + prompt * k,
        DECODE_STEP_PS + (prompt + 1) * k,
        DECODE_STEP_PS + (prompt + 2) * k,
        G_TOKENS * PREFILL_PS_PER_TOKEN + G_TOKENS * k,
        step_4,
        DECODE_STEP_PS + (prompt + 4) * k,
    )


def family_b_expected(arm: str, k: int) -> tuple[int, Fraction, int]:
    """Return the frozen (TTFT, TPOT, token-4 interval) for one family B arm."""

    steps = family_b_steps(arm, k)
    intervals = (steps[1], steps[2], steps[3] + steps[4], steps[5])
    return steps[0], Fraction(sum(intervals), len(intervals)), intervals[2]


def check_only() -> None:
    """Validate the frozen registry against its own arithmetic, run no study.

    This is the pre-implementation dry run. It also asserts that the mechanism
    the study measures does not exist yet, so a later claim of
    pre-registration is checkable rather than asserted.
    """

    _equal("page bytes per layer", 65_536, PAGE_BYTES_PER_LAYER)
    _equal("block bytes", 2_097_152, BLOCK_BYTES)
    _equal("bytes per token", 131_072, BYTES_PER_TOKEN)
    _equal("k fast", K_FAST, kv_ps_per_token(HBM_RATE_FAST_BPS))
    _equal("k slow", K_SLOW, kv_ps_per_token(HBM_RATE_SLOW_BPS))
    _equal("k doubling", 2 * K_FAST, K_SLOW)

    for capacity in CAPACITIES:
        _equal(f"ladder C={capacity}", FROZEN_LADDER[capacity], family_a_ladder(capacity))
    for (capacity, k), frozen in FROZEN_FAMILY_A.items():
        _equal(f"family A C={capacity} k={k}", frozen, family_a_expected(capacity, k))

    for k in (K_FAST, K_SLOW):
        plateau = {FROZEN_FAMILY_A[(capacity, k)] for capacity in (48, 56, 64)}
        if len(plateau) != 1:
            _fail(f"plateau k={k}", "one distinct cell", sorted(plateau))
        ttfts = [FROZEN_FAMILY_A[(capacity, k)][0] for capacity in CAPACITIES]
        if any(left < right for left, right in itertools.pairwise(ttfts)):
            _fail(f"monotonicity k={k}", "non-increasing in capacity", ttfts)
        _equal(
            f"constrained ratio k={k}",
            2 * FROZEN_FAMILY_A[(64, k)][0],
            FROZEN_FAMILY_A[(32, k)][0],
        )
    for capacity in CAPACITIES:
        _equal(
            f"kv doubling C={capacity}",
            2 * FROZEN_FAMILY_A[(capacity, K_FAST)][1],
            FROZEN_FAMILY_A[(capacity, K_SLOW)][1],
        )
        kernel_fast = (
            FROZEN_FAMILY_A[(capacity, K_FAST)][0] - FROZEN_FAMILY_A[(capacity, K_FAST)][1]
        )
        kernel_slow = (
            FROZEN_FAMILY_A[(capacity, K_SLOW)][0] - FROZEN_FAMILY_A[(capacity, K_SLOW)][1]
        )
        _equal(f"kernel invariance C={capacity}", kernel_fast, kernel_slow)

    for (arm, k), frozen in FROZEN_FAMILY_B.items():
        _equal(f"family B {arm} k={k}", frozen, family_b_expected(arm, k))
    for k in (K_FAST, K_SLOW):
        delta = FROZEN_FAMILY_B[("preempted", k)][1] - FROZEN_FAMILY_B[("unconstrained", k)][1]
        _equal(f"tpot delta k={k}", Fraction(FROZEN_TPOT_DELTA_PS), delta)
        _equal(
            f"tpot delta closed form k={k}",
            Fraction(10_360_000_000 - 2_000_000_000, 4),
            delta,
        )
        _equal(
            f"ttft unchanged k={k}",
            FROZEN_FAMILY_B[("unconstrained", k)][0],
            FROZEN_FAMILY_B[("preempted", k)][0],
        )
        resumed_kv = (D_PROMPT_TOKENS + 3) * k
        replaced_kv = (D_PROMPT_TOKENS + 2) * k + k
        _equal(f"resumed kv cancels k={k}", replaced_kv, resumed_kv)

    state = _check_mechanism_absent()
    print(
        "check-only validated the frozen KV ladder, both metric tables and the "
        f"physical byte arithmetic, and confirmed the {state}"
    )


def _check_mechanism_absent() -> str:
    """Assert a KV byte cannot reach a metric without a declared pool.

    Before the implementation landed this was the absence proof that made the
    pre-registration claim checkable: no pool type existed and every
    byte-carrying KV operation was refused. Afterwards the same assertion is
    the off-path guard, because a runtime with no declared pool must still
    refuse that operation exactly as it did then.
    """

    from simllm import core
    from simllm.core import (
        CoarseDeviceRuntime,
        ExecutionGraph,
        ExecutionOperation,
        KvCacheAction,
        KvCacheWork,
    )

    landed = all(hasattr(core, name) for name in ("KvPoolSpec", "KvLifecycleLedger"))
    graph = ExecutionGraph(
        "kv-absent",
        0,
        0,
        (
            ExecutionOperation(
                "kv",
                0,
                "kv",
                KvCacheWork(KvCacheAction.WRITE, "pool", byte_count=BLOCK_BYTES),
            ),
        ),
    )
    try:
        CoarseDeviceRuntime().execute(graph)
    except ValueError:
        return "off path intact" if landed else "mechanism absent"
    _fail("byte-carrying KV without a pool", "refused in preflight", "accepted")
    raise AssertionError("unreachable")


def _run(out: Path) -> dict[str, Any]:
    raise RuntimeError(
        "the CORE-3 KV lowering has not landed in this commit; "
        "run with --check-only"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the frozen registry and exit without running the study",
    )
    args = parser.parse_args()
    if args.check_only:
        check_only()
        return
    if args.out is None:
        parser.error("--out is required unless --check-only is given")
    payload = _run(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
