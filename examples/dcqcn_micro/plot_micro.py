"""Render the DCQCN micro-behavior figures from the study CSVs.

Usage:
    python examples/dcqcn_micro/plot_micro.py \\
        --runs /data3/yifeng/simllm-dev/dcqcn-micro-runs \\
        --out examples/dcqcn_micro/plots
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CN, DQ, AQ = "#2a78d6", "#eb6834", "#1baf7a"
NN, INK, MUTED, SURFACE = "#999999", "#222222", "#666666", "#fcfcfb"
UCCL = {32768: 28.0, 65536: 45.0, 131072: 49.0, 262144: 50.0}


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10.5, color=INK, pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.set_axisbelow(True)
    ax.grid(color="#e6e6e3", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="/data3/yifeng/simllm-dev/dcqcn-micro-runs")
    parser.add_argument("--out", default=str(Path(__file__).parent / "plots"))
    args = parser.parse_args()
    runs = Path(args.runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Figure 1: message size vs goodput at 400G, with the paper target.
    msg = defaultdict(dict)
    with open(runs / "msg.csv") as handle:
        for row in csv.DictReader(handle):
            msg[(int(row["q"]), row["engine"])][int(row["size"])] = float(
                row["goodput_GBs"])
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    sizes = sorted(msg[(1, "dcqcn")])
    kb = [s / 1024 for s in sizes]
    ax.plot(kb, [msg[(1, "fluid")][s] for s in sizes], color=NN,
            linestyle="--", linewidth=1.5, marker="o", markersize=4)
    ax.plot(kb, [msg[(1, "cn")][s] for s in sizes], color=CN,
            linewidth=1.8, marker="o", markersize=4)
    ax.plot(kb, [msg[(1, "dcqcn")][s] for s in sizes], color=DQ,
            linewidth=1.8, marker="o", markersize=4)
    ax.plot(kb, [msg[(16, "dcqcn")][s] for s in sizes], color=DQ,
            linewidth=1.8, marker="s", markersize=5, linestyle=":")
    grid = [2 ** i * 1024 for i in range(2, 13)]
    a2_q16 = [s * 16 / (5.2e-6 + s * 16 / 50e9) / 1e9 for s in grid]
    ax.plot([s / 1024 for s in grid], a2_q16, color=AQ, linewidth=1.6,
            linestyle="-.")
    ax.scatter([s / 1024 for s in UCCL], list(UCCL.values()), color=INK,
               marker="*", s=90, zorder=5)
    ax.set_xscale("log")
    ax.axhline(50, color="#cccccc", linewidth=0.8)
    ax.text(4.2, 50.8, "line rate 50 GB/s", fontsize=8.5, color=MUTED)
    style(ax, "Message size vs goodput, 400G cross-leaf pair: current models, "
              "the A2 target law and the UCCL anchors",
          "message size (KiB, log)", "goodput (GB/s)")
    ax.legend(handles=[
        Line2D([], [], color=NN, linestyle="--", marker="o", label="fluid ideal (Q=1)"),
        Line2D([], [], color=CN, marker="o", label="rnic-cn (Q=1)"),
        Line2D([], [], color=DQ, marker="o", label="DCQCN model (Q=1)"),
        Line2D([], [], color=DQ, marker="s", linestyle=":",
               label="DCQCN model (Q=16, aggregate)"),
        Line2D([], [], color=AQ, linestyle="-.",
               label="A2 target law (T0=5.2us, Q=16)"),
        Line2D([], [], color=INK, marker="*", linestyle="",
               label="UCCL Fig. 14 anchors (Q=16)"),
    ], frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "msg_size_vs_goodput.png", dpi=160)
    plt.close(fig)

    # Figure 2: incast fair-share drop-off at 40G.
    inc = []
    with open(runs / "incast.csv") as handle:
        for row in csv.DictReader(handle):
            inc.append({k: (v if k in ("engine",) else float(v))
                        for k, v in row.items()})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ns = sorted({int(r["n"]) for r in inc})
    fair = [5.0 / n for n in ns]
    ax1.set_facecolor(SURFACE)
    ax1.plot(ns, fair, color=NN, linestyle="--", linewidth=1.5)
    for engine, color, marker in (("cn", CN, "o"), ("dcqcn", DQ, "s")):
        for n in ns:
            values = [r["mean_GBs"] for r in inc
                      if r["engine"] == engine and int(r["n"]) == n]
            ax1.scatter([n] * len(values), values, color=color, marker=marker,
                        s=30, zorder=3)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xticks(ns, [str(n) for n in ns])
    style(ax1, "Per-sender throughput vs fan-in\n(40G, 40 MB per sender)",
          "senders N", "mean per-sender GB/s")
    ax1.legend(handles=[
        Line2D([], [], color=NN, linestyle="--", label="fair share C/N"),
        Line2D([], [], color=CN, marker="o", linestyle="", label="rnic-cn"),
        Line2D([], [], color=DQ, marker="s", linestyle="",
               label="DCQCN (3 seeds)"),
    ], frameon=False, fontsize=8.5, loc="lower left")
    ax2.set_facecolor(SURFACE)
    for engine, color, marker in (("cn", CN, "o"), ("dcqcn", DQ, "s")):
        for n in ns:
            values = [r["utilization"] for r in inc
                      if r["engine"] == engine and int(r["n"]) == n]
            ax2.scatter([n] * len(values), values, color=color, marker=marker,
                        s=30, zorder=3)
    ax2.axhline(1.0, color="#cccccc", linewidth=0.8)
    ax2.axhline(0.85, color=NN, linewidth=0.9, linestyle=":")
    ax2.text(2.05, 0.853, "T2 bar 0.85", fontsize=8, color=MUTED)
    ax2.set_xscale("log")
    ax2.set_xticks(ns, [str(n) for n in ns])
    ax2.set_ylim(0.6, 1.05)
    style(ax2, "Receiver-link utilization vs fan-in\n(Jain fairness 0.993 to "
               "1.000 everywhere)", "senders N", "sum of rates / C")
    fig.tight_layout()
    fig.savefig(out / "incast_fairshare.png", dpi=160)
    plt.close(fig)

    # Figure 3: join/exit rate time series (the paper Figure 10 analog).
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 7.4), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, engine, label in zip(
            axes, ("fluid", "dcqcn", "cn"),
            ("fluid ideal (instant fair sharing)",
             ("DCQCN model: cut at 10.9 ms, fair at 12.8 ms, "
              "recovery 0.41 ms after exit"),
             ("rnic-cn: fair at 11.5 ms; solo chunked rate capped at 0.68 C "
              "by per-WQE declare cost (HTSIM-6)"))):
        ax.set_facecolor(SURFACE)
        series = defaultdict(list)
        with open(runs / f"join-{engine}.csv") as handle:
            for row in csv.DictReader(handle):
                series[row["flow"]].append(
                    (float(row["t_ms"]), float(row["rate_GBs"])))
        for flow, color in (("A", CN if engine == "cn" else DQ), ("B", AQ)):
            points = sorted(series[flow])
            ax.step([t for t, _ in points], [r for _, r in points],
                    where="pre", color=color, linewidth=1.4,
                    label=f"flow {flow}")
        ax.axhline(5.0, color="#cccccc", linewidth=0.8)
        ax.axhline(2.5, color="#e0e0dd", linewidth=0.8, linestyle=":")
        ax.axvline(10.0, color="#cccccc", linewidth=0.8, linestyle=":")
        style(ax, label, "", "GB/s")
        ax.set_ylim(0, 5.6)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    axes[-1].set_xlabel("time (ms); B joins at 10 ms, exits when its 100 MB "
                        "finish", fontsize=9, color=MUTED)
    fig.suptitle("Join/exit convergence to fairness, 40G shared receiver link "
                 "(per-1MB-chunk achieved rate)", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "join_exit_convergence.png", dpi=160)
    plt.close(fig)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
