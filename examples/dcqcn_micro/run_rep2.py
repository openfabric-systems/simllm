"""Contended repeated-WQE streams: run the checks of expectations-rep2.md.

Two same-leaf senders (ranks 0 and 1) each post n independent WQEs of
size S to rank 15 at 400G: offered load 2 C into a C bottleneck for the
burst duration. Emits rep2.csv and rep2-summary.csv.

Usage:
    SIMLLM_HTSIM_RNIC=... SIMLLM_HTSIM_DCQCN=... SIMLLM_TXT2BIN=... \\
    SIMLLM_DATA_ROOT=... python examples/dcqcn_micro/run_rep2.py
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
DQ_KEYS = ["ns_tm3_dropped_packets", "silent_rtos", "dcqcn_pfc_pause_frames",
           "dcqcn_pfc_max_cascade_depth"]
SIZES = [16 << 10, 64 << 10]
REPS = [10, 100, 1000]
BUFFER = 1 << 20


def rep2_goal(size: int, n: int) -> GoalTrace:
    trace = GoalTrace(64)
    for sender in (0, 1):
        for i in range(n):
            tag = 1 + sender * 100000 + i
            trace.rank(sender).send(size, to=15, tag=tag)
            trace.rank(15).recv(size, source=sender, tag=tag)
    for r in set(range(64)) - {0, 1, 15}:
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
        goodput = 2 * n * size / (result.job_completion_time_ps() * 1e-12) / 1e9
        row = {"engine": engine, "size": size, "n": n,
               "goodput_GBs": round(goodput, 4),
               "jct_ps": result.job_completion_time_ps()}
        row.update(extra or {})
        rows.append(row)
        print(row)
        return goodput

    table: dict[tuple[str, int, int], float] = {}
    drops: dict[tuple[int, int], int] = {}
    pauses: dict[tuple[int, int], int] = {}
    cascade: dict[tuple[int, int], int] = {}
    cn_recovery_total = 0
    for size in SIZES:
        for n in REPS:
            goal_bin = to_binary(rep2_goal(size, n).write(
                out / f"rep2-{n}-s{size}.goal"))
            r = run_htsim_rnic(HtsimRnicConfig(
                goal_bin=goal_bin, profile="rnic-nn-fluid", linkspeed_bps=400 * G,
                completion_csv=out / f"rep2-{n}-s{size}.fluid.csv"))
            table[("fluid", size, n)] = record("fluid", size, n, r)
            try:
                r = run_htsim_rnic(HtsimRnicConfig(
                    goal_bin=goal_bin, profile="rnic-cn", linkspeed_bps=400 * G,
                    completion_csv=out / f"rep2-{n}-s{size}.cn.csv",
                    topology=TOPO))
                cn_recovery_total += sum(counters(r, CN_RECOVERY).values())
                table[("cn", size, n)] = record("cn", size, n, r)
            except subprocess.TimeoutExpired:
                rows.append({"engine": "cn", "size": size, "n": n,
                             "goodput_GBs": None, "jct_ps": None,
                             "timeout": True})
                print(f"cn size={size} n={n}: TIMEOUT (600 s budget)")
            for mode in ("ecn-only", "ecn-pfc"):
                seeds = (1, 2) if n == 100 else (1,)
                for seed in seeds:
                    flags = dict(SMALL_BUFFERS)
                    flags["-pfc"] = "on" if mode == "ecn-pfc" else "off"
                    flags["-seed"] = str(seed)
                    flags["-ecn_seed"] = str(seed)
                    r = run_htsim_dcqcn(HtsimDcqcnConfig(
                        goal_bin=goal_bin, topology=TOPO, link_bps=400 * G,
                        completion_csv=out /
                        f"rep2-{n}-s{size}.dcqcn-{mode}-s{seed}.csv",
                        extra_flags=flags))
                    cnt = counters(r, DQ_KEYS)
                    goodput = record(f"dcqcn-{mode}", size, n, r, cnt)
                    if seed == 1:
                        table[(f"dcqcn-{mode}", size, n)] = goodput
                        if mode == "ecn-only":
                            drops[(size, n)] = (cnt["ns_tm3_dropped_packets"]
                                                + cnt["silent_rtos"])
                        else:
                            pauses[(size, n)] = cnt["dcqcn_pfc_pause_frames"]
                            cascade[(size, n)] = cnt["dcqcn_pfc_max_cascade_depth"]

    with open(out / "rep2.csv", "w", newline="") as handle:
        fields: list[str] = []
        for row in rows:
            fields += [k for k in row if k not in fields]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    overflow = [(size, n) for size in SIZES for n in REPS
                if n * size > BUFFER]
    emit(check="Q1", ok=all(table[("fluid", size, n)] >= 0.90 * C_GBS
                            for size in SIZES for n in REPS if n >= 100))
    cn_missing = [(size, n) for size in SIZES for n in REPS
                  if n >= 100 and ("cn", size, n) not in table]
    emit(check="Q2", cn_recovery_total=cn_recovery_total,
         unmeasured_cells=cn_missing,
         ok=(not cn_missing and cn_recovery_total == 0
             and all(0.75 * C_GBS <= table[("cn", size, n)] <= 1.00 * C_GBS
                     for size in SIZES for n in REPS if n >= 100)))
    emit(check="Q3", ok=all(
        drops.get((size, 10), 0) == 0 and pauses.get((size, 10), 0) == 0
        for size in SIZES))
    emit(check="Q4",
         drops={f"{s}x{n}": drops[(s, n)] for s, n in overflow},
         n100_goodputs=[table[("dcqcn-ecn-only", size, 100)] for size in SIZES],
         ok=(all(drops[(s, n)] > 0 for s, n in overflow)
             and all(table[("dcqcn-ecn-only", size, 100)] < 0.1 * C_GBS
                     for size in SIZES)))
    emit(check="Q5",
         pauses={f"{s}x{n}": pauses[(s, n)] for s, n in overflow},
         cascade={f"{s}x{n}": cascade[(s, n)] for s, n in overflow},
         ok=all(pauses[(s, n)] > 0
                and (("cn", s, n) not in table
                     or table[("dcqcn-ecn-pfc", s, n)] < table[("cn", s, n)])
                for s, n in overflow))
    q6_cells = [(s, n) for s, n in overflow if ("cn", s, n) in table]
    emit(check="Q6",
         unmeasured_cells=[c for c in overflow if c not in q6_cells],
         ok=(len(q6_cells) == len(overflow) and all(
             table[("cn", s, n)] > 2 * max(table[("dcqcn-ecn-only", s, n)],
                                           table[("dcqcn-ecn-pfc", s, n)])
             for s, n in q6_cells)))
    emit(check="Q7", ok=all(
        table[("dcqcn-ecn-only", size, 1000)]
        <= 2 * table[("dcqcn-ecn-only", size, 100)]
        for size in SIZES))

    with open(out / "rep2-summary.csv", "w", newline="") as handle:
        fields = []
        for row in checks:
            fields += [k for k in row if k not in fields]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(checks)
    print(f"\n{len(checks)} check rows -> {out / 'rep2-summary.csv'}")


if __name__ == "__main__":
    main()
