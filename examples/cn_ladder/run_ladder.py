"""rnic-cn incast ladder versus rnic-nn (see expectations.md).

Grid: fan-in x size, cn at the default and calibrated control deadlines,
identical GOALs against the nn baseline. One CSV row per (cell, metric);
disqualifier columns (late admissions, gap NACKs) are scraped from the cn
manifest so a failing cell is visible in the same table.

Usage: python run_ladder.py [--out DIR] [--mixed]  (mixed = phase 2, the
all-to-all with lognormal sizes, only after the ladder holds)
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from simllm.backends import (
    HtsimRnicConfig,
    RnicRunResult,
    normalized_fct,
    run_htsim_rnic,
)
from simllm.goal import GoalTrace, to_binary
from simllm.workload import LogNormalLengths

HERE = Path(__file__).resolve().parent
TOPO = HERE.parent / "m1" / "topologies" / "clos_64_400g.topo"
G400 = 400_000_000_000
FAN_INS = [1, 2, 4, 8, 16, 32, 63]
SIZES = [16 << 20, 4 << 20, 1 << 20, 256 << 10, 64 << 10, 16 << 10, 4096]
DEADLINES = {"k10": None, "k4p5": "4500000"}


def incast_goal(fan_in: int, size: int) -> GoalTrace:
    trace = GoalTrace(64)
    for s in range(1, fan_in + 1):
        trace.rank(s).send(size, to=0, tag=1)
        trace.rank(0).recv(size, source=s, tag=1)
    for r in set(range(64)) - set(range(fan_in + 1)):
        trace.rank(r).calc(0)
    return trace


def mixed_all_to_all_goal(seed: int = 7) -> GoalTrace:
    """All-to-all over 16 ranks with lognormal sizes (mean 256 KiB)."""
    sizes = LogNormalLengths(mean=256 << 10, sigma=1.5, seed=seed, minimum=4096)
    trace = GoalTrace(64)
    ranks = list(range(16))
    draw = iter(sizes.sample(len(ranks) * (len(ranks) - 1)))
    for s in ranks:
        for d in ranks:
            if s == d:
                continue
            size = int(next(draw))
            trace.rank(s).send(size, to=d, tag=100 + s)
            trace.rank(d).recv(size, source=s, tag=100 + s)
    for r in range(16, 64):
        trace.rank(r).calc(0)
    return trace


def manifest_counter(result: RnicRunResult, key: str) -> int:
    for line in result.manifest:
        for token in line.split():
            if token.startswith(key + "="):
                try:
                    return int(token.split("=", 1)[1])
                except ValueError:
                    return -1
    return 0


def run_cell(name: str, goal: GoalTrace, out: Path, writer, repeat_check: bool = False):
    goal_bin = to_binary(goal.write(out / f"{name}.goal"))
    nn = run_htsim_rnic(HtsimRnicConfig(
        goal_bin=goal_bin, profile="rnic-nn", linkspeed_bps=G400,
        completion_csv=out / f"{name}.nn.csv"))
    for variant, deadline in DEADLINES.items():
        extra = {"-rnic_cn_control_deadline_ps": deadline} if deadline else {}
        results = []
        for rep in range(3 if repeat_check else 1):
            results.append(run_htsim_rnic(HtsimRnicConfig(
                goal_bin=goal_bin, profile="rnic-cn", linkspeed_bps=G400,
                completion_csv=out / f"{name}.{variant}.csv", topology=TOPO,
                extra_flags=dict(extra))))
        first = {f.flow_id: f.completion_time_ps for f in results[0].flows}
        deterministic = all(
            {f.flow_id: f.completion_time_ps for f in r.flows} == first
            for r in results[1:])
        cn = results[0]
        norm = normalized_fct(cn.flows, nn.flows)
        slow = sorted(n.slowdown for n in norm)
        excess = sorted((n.fct_ps - n.baseline_fct_ps) / 1e6 for n in norm)
        row = {
            "cell": name, "variant": variant, "flows": len(slow),
            "slow_min": round(slow[0], 4),
            "slow_med": round(statistics.median(slow), 4),
            "slow_max": round(slow[-1], 4),
            "excess_med_us": round(statistics.median(excess), 3),
            "excess_max_us": round(excess[-1], 3),
            "nn_med_fct_us": round(statistics.median(
                f.fct_ps for f in nn.flows) / 1e6, 3),
            "late_admissions": manifest_counter(cn, "rnic_cn_late_data_packets"),
            "gap_nacks": manifest_counter(cn, "rnic_cn_gap_nacks_dispatched"),
            "deterministic": deterministic,
        }
        writer.writerow(row)
        print(" ".join(f"{k}={v}" for k, v in row.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/data3/yifeng/simllm-dev/cn-ladder")
    parser.add_argument("--mixed", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fields = ["cell", "variant", "flows", "slow_min", "slow_med", "slow_max",
              "excess_med_us", "excess_max_us", "nn_med_fct_us",
              "late_admissions", "gap_nacks", "deterministic"]
    with open(out / ("mixed.csv" if args.mixed else "ladder.csv"), "w",
              newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        if args.mixed:
            run_cell("a2a16-lognormal", mixed_all_to_all_goal(), out, writer,
                     repeat_check=True)
        else:
            for fan_in in FAN_INS:
                for size in SIZES:
                    run_cell(f"in{fan_in}-s{size}", incast_goal(fan_in, size),
                             out, writer, repeat_check=(fan_in in (1, 63)))


if __name__ == "__main__":
    main()
