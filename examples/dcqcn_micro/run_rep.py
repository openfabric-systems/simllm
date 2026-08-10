"""Repeated-WQE stream collapse: run the checks of expectations-rep.md.

Rank 0 posts n independent WQEs of size S to rank 15 at 400G; DCQCN
runs both modes at the 1 MiB lossy buffers, cn and fluid deterministic.
Emits rep.csv (per-cell goodput and counters) and rep-summary.csv
(check verdicts) for the plot script.

Usage:
    SIMLLM_HTSIM_RNIC=... SIMLLM_HTSIM_DCQCN=... SIMLLM_TXT2BIN=... \\
    SIMLLM_DATA_ROOT=... python examples/dcqcn_micro/run_rep.py
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

from simllm._local_config import path_from_env
from simllm.backends import (
    HtsimDcqcnConfig,
    HtsimRnicConfig,
    RnicRunResult,
    run_htsim_dcqcn,
    run_htsim_rnic,
)
from simllm.goal import GoalTrace, to_binary

HERE = Path(__file__).resolve().parent
TOPO = HERE.parent / "m1" / "topologies" / "clos_64_400g.topo"
G = 1_000_000_000
C_GBS = 50.0
SMALL_BUFFERS = {"-shared_buffer_bytes": "1048576", "-egress_buffer_bytes": "1048576"}
CN_RECOVERY = ["rnic_cn_gap_nacks_dispatched", "rnic_cn_late_data_packets",
               "rnic_cn_deterministic_retransmissions",
               "rnic_cn_maximum_retry_attempt_observed"]
DQ_KEYS = ["ns_tm3_dropped_packets", "silent_rtos", "dcqcn_pfc_pause_frames"]
SIZES = [16 << 10, 64 << 10]
REPS = [10, 100, 1000, 10000]
BUFFER = 1 << 20


def rep_goal(size: int, n: int) -> GoalTrace:
    trace = GoalTrace(64)
    for i in range(n):
        trace.rank(0).send(size, to=15, tag=1 + i)
        trace.rank(15).recv(size, source=0, tag=1 + i)
    for r in set(range(64)) - {0, 15}:
        trace.rank(r).calc(0)
    return trace


def counters(result: RnicRunResult, keys: list[str]) -> dict[str, int]:
    out = dict.fromkeys(keys, 0)
    for line in result.manifest:
        for token in line.split():
            for key in keys:
                if token.startswith(key + "="):
                    try:
                        out[key] = int(token.split("=", 1)[1])
                    except ValueError:
                        pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out is None:
        data_root = path_from_env("SIMLLM_DATA_ROOT")
        if data_root is None:
            parser.error("--out is required when SIMLLM_DATA_ROOT is not set")
        args.out = data_root / "dcqcn_micro"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    checks: list[dict] = []

    def emit(**kw) -> None:
        checks.append(kw)
        print(" ".join(f"{k}={v}" for k, v in kw.items()))

    def record(engine, size, n, result, extra=None):
        goodput = n * size / (result.job_completion_time_ps() * 1e-12) / 1e9
        row = {"engine": engine, "size": size, "n": n,
               "goodput_GBs": round(goodput, 4),
               "jct_ps": result.job_completion_time_ps()}
        row.update(extra or {})
        rows.append(row)
        print(row)
        return goodput, row

    table: dict[tuple[str, int, int], float] = {}
    drops: dict[tuple[str, int, int], int] = {}
    pauses: dict[tuple[int, int], int] = {}
    cn_recovery_total = 0
    for size in SIZES:
        for n in REPS:
            goal_bin = to_binary(rep_goal(size, n).write(out / f"rep{n}-s{size}.goal"))
            r = run_htsim_rnic(HtsimRnicConfig(
                goal_bin=goal_bin, profile="rnic-nn-fluid", linkspeed_bps=400 * G,
                completion_csv=out / f"rep{n}-s{size}.fluid.csv"))
            table[("fluid", size, n)], _ = record("fluid", size, n, r)
            try:
                r = run_htsim_rnic(HtsimRnicConfig(
                    goal_bin=goal_bin, profile="rnic-cn", linkspeed_bps=400 * G,
                    completion_csv=out / f"rep{n}-s{size}.cn.csv", topology=TOPO))
            except subprocess.TimeoutExpired:
                # rnic-cn does not complete 10k simultaneous same-pair flows
                # within the 600 s per-run budget (concurrent-flow scaling
                # limit of the reservation ledger; recorded as a finding).
                rows.append({"engine": "cn", "size": size, "n": n,
                             "goodput_GBs": None, "jct_ps": None,
                             "timeout": True})
                print(f"cn size={size} n={n}: TIMEOUT (600 s budget)")
                r = None
            if r is not None:
                cn_recovery_total += sum(counters(r, CN_RECOVERY).values())
                table[("cn", size, n)], _ = record("cn", size, n, r)
            if n == 1000 and size == 64 << 10 and r is not None:
                r2 = run_htsim_rnic(HtsimRnicConfig(
                    goal_bin=goal_bin, profile="rnic-cn", linkspeed_bps=400 * G,
                    completion_csv=out / f"rep{n}-s{size}.cn2.csv", topology=TOPO))
                emit(check="P2-determinism",
                     ok=[f.fct_ps for f in r.flows] == [f.fct_ps for f in r2.flows])
            for mode in ("ecn-only", "ecn-pfc"):
                seeds = (1, 2) if n == 100 else (1,)
                for seed in seeds:
                    flags = dict(SMALL_BUFFERS)
                    flags["-pfc"] = "on" if mode == "ecn-pfc" else "off"
                    flags["-seed"] = str(seed)
                    flags["-ecn_seed"] = str(seed)
                    r = run_htsim_dcqcn(HtsimDcqcnConfig(
                        goal_bin=goal_bin, topology=TOPO, link_bps=400 * G,
                        completion_csv=out / f"rep{n}-s{size}.dcqcn-{mode}-s{seed}.csv",
                        extra_flags=flags))
                    cnt = counters(r, DQ_KEYS)
                    goodput, _ = record(f"dcqcn-{mode}", size, n, r, cnt)
                    if seed == 1:
                        table[(f"dcqcn-{mode}", size, n)] = goodput
                        drops[(mode == "ecn-pfc", size, n)] = (
                            cnt["ns_tm3_dropped_packets"] + cnt["silent_rtos"])
                        if mode == "ecn-pfc":
                            pauses[(size, n)] = cnt["dcqcn_pfc_pause_frames"]

    with open(out / "rep.csv", "w", newline="") as handle:
        fields: list[str] = []
        for row in rows:
            fields += [k for k in row if k not in fields]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # P1: fluid line rate (boundary cell at 0.94 C per the registration).
    p1 = all(
        table[("fluid", size, n)]
        >= (0.94 if (n, size) == (100, 16 << 10) else 0.95) * C_GBS
        for size in SIZES for n in REPS if n >= 100)
    emit(check="P1", ok=p1)
    # P2: cn band and losslessness.
    cn_cells = [(size, n) for size in SIZES for n in REPS
                if n >= 100 and ("cn", size, n) in table]
    cn_missing = [(size, n) for size in SIZES for n in REPS
                  if n >= 100 and ("cn", size, n) not in table]
    p2 = all(0.80 * C_GBS <= table[("cn", size, n)] <= 1.00 * C_GBS
             for size, n in cn_cells)
    emit(check="P2", cn_recovery_total=cn_recovery_total,
         unmeasured_cells=cn_missing,
         ok=p2 and cn_recovery_total == 0 and not cn_missing)
    # P3: buffer-absorbed cells (n = 10 both sizes).
    p3 = True
    for size in SIZES:
        for mode in ("ecn-only", "ecn-pfc"):
            good = table[(f"dcqcn-{mode}", size, 10)]
            fluid = table[("fluid", size, 10)]
            no_events = drops[(mode == "ecn-pfc", size, 10)] == 0 and (
                mode == "ecn-only" or pauses[(size, 10)] == 0)
            if not (no_events and good >= 0.75 * fluid):
                p3 = False
    emit(check="P3", ok=p3)
    # P4: overflow cells drop; goodput < 0.1 C at n = 100.
    overflow = [(size, n) for size in SIZES for n in REPS if n * size > BUFFER]
    p4_drops = all(drops[(False, size, n)] > 0 for size, n in overflow)
    p4_bar = all(table[("dcqcn-ecn-only", size, 100)] < 0.1 * C_GBS
                 for size in SIZES)
    emit(check="P4", all_overflow_cells_drop=p4_drops,
         n100_goodputs=[table[("dcqcn-ecn-only", size, 100)] for size in SIZES],
         ok=p4_drops and p4_bar)
    # P5: non-monotonic shape at 64 KiB.
    emit(check="P5",
         g100=table[("dcqcn-ecn-only", 64 << 10, 100)],
         g10000=table[("dcqcn-ecn-only", 64 << 10, 10000)],
         ok=(table[("dcqcn-ecn-only", 64 << 10, 10000)]
             > table[("dcqcn-ecn-only", 64 << 10, 100)]))
    # P6: PFC mode pauses on overflow and stays below cn.
    p6 = all(pauses[(size, n)] > 0
             and (("cn", size, n) not in table
                  or table[("dcqcn-ecn-pfc", size, n)] < table[("cn", size, n)])
             for size, n in overflow)
    emit(check="P6", pauses={f"{s}x{n}": pauses[(s, n)] for s, n in overflow},
         ok=p6)
    # P7: cn > 2x the better DCQCN mode on every overflow cell.
    p7_cells = [(size, n) for size, n in overflow if ("cn", size, n) in table]
    p7 = all(table[("cn", size, n)] > 2 * max(
        table[("dcqcn-ecn-only", size, n)], table[("dcqcn-ecn-pfc", size, n)])
        for size, n in p7_cells)
    emit(check="P7", unmeasured_cells=[c for c in overflow if c not in p7_cells],
         ok=p7 and len(p7_cells) == len(overflow))

    with open(out / "rep-summary.csv", "w", newline="") as handle:
        fields = []
        for row in checks:
            fields += [k for k in row if k not in fields]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(checks)
    print(f"\n{len(checks)} check rows -> {out / 'rep-summary.csv'}")


if __name__ == "__main__":
    main()
