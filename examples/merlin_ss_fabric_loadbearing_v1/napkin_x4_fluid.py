#!/usr/bin/env python3
"""Disclosed fluid napkin model for the x4 shared-egress closed-loop cells.

Committed with the freeze (expectations.md) and run before any
htsim_ss_dragonfly load cell of this study: pure arithmetic over declared
instance constants and byte-locked dataset measurements, with no htsim
invocation of any kind. Its outputs are the disclosed point predictions
the freeze quotes; the registered bands are set around them and the
packet-level simulator is scored against the bands, never against this
script.

Model: the shared switch-to-destination egress serves at the wire rate,
split by water-filling equal shares over flows that can take service
(the round-robin grant loop at packet granularity, taken to fluid);
each bursting closed-loop flow arrives at the wire rate through its own
injection link; shared-buffer occupancy is total arrived minus total
served wire bytes; a capacity crossing is recorded as the fluid drop
instant. Completion of a burst is its served end plus the constant tail
K = F - 938 * ser; the next burst starts at completion plus the flow's
declared think time.
"""

from fractions import Fraction

SER = 361_520                    # ps per 9038 B wire packet at 200 Gbit/s
N_PKT = 938                      # packets per 8 MiB chunk
CHUNK_WIRE = N_PKT * 9038        # 8,477,644 wire bytes per chunk
ARR = N_PKT * SER                # arrival window of one burst, ps
K_TAIL = 1_311_520               # completion tail: F - ARR
F = 340_417_280                  # greedy chunk completion on an idle fabric, ps
B = 8_388_608                    # chunk payload bytes
C_WIRE = Fraction(9038, SER)     # wire bytes per ps (exactly 25 GB/s)

# Byte-locked x4 measurements (examples/merlin_fabric_flow_capture_v1):
# per-flow completions in the final-20-second stage window and whole-series
# successive-delta p50s, re-derived from the series bytes at freeze time.
CHUNKS_20S = [4014, 7704, 7682, 7053]
P50_PS = [5_138_805_000, 2_123_632_000, 2_175_749_000, 2_134_978_000]
MEASURED_AGG = Fraction(sum(CHUNKS_20S) * B, 20)   # bytes per second

START_PS = [0, 400_000_000, 800_000_000, 1_200_000_000]


def think_rate_arm():
    """Rate-derived endpoint floors: round(20e12 / n_i) - F."""
    return [round(Fraction(20 * 10**12, n)) - F for n in CHUNKS_20S]


def think_p50_arm():
    """p50-derived endpoint floors: p50_i - F."""
    return [p - F for p in P50_PS]


def rates(active, burst_left, queue):
    """Water-filling equal-share service rates; returns (arr, srv) lists."""
    n = len(active)
    arr = [C_WIRE if (active[i] and burst_left[i] > 0) else Fraction(0) for i in range(n)]
    contenders = [i for i in range(n) if active[i] and (queue[i] > 0 or arr[i] > 0)]
    srv = [Fraction(0)] * n
    cap = C_WIRE
    todo = list(contenders)
    while todo and cap > 0:
        share = cap / len(todo)
        limited = [i for i in todo if queue[i] == 0 and arr[i] <= share]
        if not limited:
            for i in todo:
                srv[i] += share
            cap = Fraction(0)
            break
        for i in limited:
            srv[i] += arr[i]
            cap -= arr[i]
            todo.remove(i)
    return arr, srv


def simulate(think, buffer_bytes, horizon_ps, starts=START_PS):
    n = len(think)
    next_start = [Fraction(s) for s in starts]
    active = [False] * n
    burst_left = [Fraction(0)] * n
    queue = [Fraction(0)] * n
    completions = [[] for _ in range(n)]
    occupancy = Fraction(0)
    max_occ = Fraction(0)
    first_drop = None
    horizon = Fraction(horizon_ps)
    t = Fraction(0)

    while t < horizon:
        for i in range(n):
            if not active[i] and next_start[i] is not None and next_start[i] <= t:
                active[i] = True
                burst_left[i] = Fraction(CHUNK_WIRE)
                next_start[i] = None
        arr, srv = rates(active, burst_left, queue)
        events = [horizon]
        for i in range(n):
            if not active[i] and next_start[i] is not None and next_start[i] > t:
                events.append(next_start[i])
            if active[i]:
                if arr[i] > 0:
                    events.append(t + burst_left[i] / arr[i])
                if arr[i] - srv[i] < 0 and queue[i] > 0:
                    events.append(t + queue[i] / (srv[i] - arr[i]))
                if arr[i] == 0 and queue[i] > 0 and srv[i] > 0:
                    events.append(t + queue[i] / srv[i])
        occ_rate = sum(arr) - sum(srv)
        if occ_rate > 0 and first_drop is None:
            events.append(t + (Fraction(buffer_bytes) - occupancy) / occ_rate)
        t_next = min(e for e in events if e > t)
        dt = t_next - t
        for i in range(n):
            if active[i]:
                a = arr[i] * dt
                s = srv[i] * dt
                burst_left[i] -= a
                queue[i] += a - s
                occupancy += a - s
        max_occ = max(max_occ, occupancy)
        if first_drop is None and occupancy >= buffer_bytes and occ_rate > 0:
            first_drop = t_next
        t = t_next
        for i in range(n):
            if active[i] and burst_left[i] == 0 and queue[i] == 0:
                comp = t + K_TAIL
                completions[i].append(comp)
                active[i] = False
                next_start[i] = comp + think[i]
    return {
        "completions": completions,
        "max_occ_bytes": float(max_occ),
        "first_drop_ps": None if first_drop is None else float(first_drop),
    }


def report(label, think, buffer_bytes, horizon_ps, lo_ps, hi_ps):
    r = simulate(think, buffer_bytes, horizon_ps)
    window = hi_ps - lo_ps
    per_flow = [sum(1 for c in f if lo_ps <= c < hi_ps) for f in r["completions"]]
    agg = Fraction(sum(per_flow) * B) * Fraction(10**12, window)
    print(f"== {label} buffer={buffer_bytes}")
    print(f"   chunks in window per flow: {per_flow}")
    print(f"   aggregate GB/s {float(agg) / 1e9:.6f} ratio {float(agg / MEASURED_AGG):.5f}")
    print(f"   max occupancy bytes {r['max_occ_bytes']:.0f}")
    print(f"   first fluid capacity crossing ps {r['first_drop_ps']}")


if __name__ == "__main__":
    tr = think_rate_arm()
    tp = think_p50_arm()
    print("rate-arm think_ps:", tr)
    print("p50-arm think_ps:", tp)
    zero_rate = float(sum(Fraction(B * 10**12, F + z) for z in tr)) / 1e9
    zero_p50 = float(sum(Fraction(B * 10**12, F + z) for z in tp)) / 1e9
    print("zero-wait rate-arm aggregate GB/s =", zero_rate)
    print("zero-wait p50-arm aggregate GB/s  =", zero_p50)
    print("measured aggregate GB/s           =", float(MEASURED_AGG) / 1e9)
    print("worst-case occupancy bound bytes  =", 3 * CHUNK_WIRE)
    HORIZON = 6_000_000_000_000
    LO, HI = 500_000_000_000, 6_000_000_000_000
    report("rate-arm 32MiB", tr, 33_554_432, HORIZON, LO, HI)
    report("p50-arm 32MiB", tp, 33_554_432, HORIZON, LO, HI)
    report("rate-arm 4MiB", tr, 4_194_304, HORIZON, LO, HI)
    report("p50-arm 4MiB", tp, 4_194_304, HORIZON, LO, HI)
