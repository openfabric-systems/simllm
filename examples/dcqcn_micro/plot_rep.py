"""Render the repeated-WQE figures from rep.csv and rep2.csv.

Usage:
    python examples/dcqcn_micro/plot_rep.py \\
        --runs /data3/yifeng/simllm-dev/dcqcn-micro-runs \\
        --out examples/dcqcn_micro/plots
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CN, DQ_ECN, DQ_PFC = "#2a78d6", "#eb6834", "#1baf7a"
NN, INK, MUTED, SURFACE = "#999999", "#222222", "#666666", "#fcfcfb"
SIZES = [16 << 10, 64 << 10]
BUFFER = 1 << 20


def load(path: Path) -> dict[tuple[str, int, int], float]:
    table: dict[tuple[str, int, int], float] = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            if not row["goodput_GBs"]:
                continue
            key = (row["engine"], int(row["size"]), int(row["n"]))
            table.setdefault(key, float(row["goodput_GBs"]))  # seed 1 first
    return table


def style(ax, title):
    ax.set_title(title, fontsize=10.5, color=INK, pad=8)
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
    single = load(runs / "rep.csv")
    contended = load(runs / "rep2.csv")
    reps = [10, 100, 1000]

    fig, axes = plt.subplots(1, 2, figsize=(9.9, 4.9), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, size in zip(axes, SIZES):
        ax.set_facecolor(SURFACE)
        for engine, color, ls, marker in (
                ("fluid", NN, "--", "o"), ("cn", CN, "-", "o"),
                ("dcqcn-ecn-only", DQ_ECN, "-", "s"),
                ("dcqcn-ecn-pfc", DQ_PFC, "-", "^")):
            values = [contended[(engine, size, n)] for n in reps]
            ax.plot(reps, values, color=color, linestyle=ls, marker=marker,
                    markersize=5, linewidth=1.7)
        solo = [single[("dcqcn-ecn-only", size, n)] for n in reps]
        ax.plot(reps, solo, color=DQ_ECN, linestyle=":", marker="s",
                markersize=4, linewidth=1.2, alpha=0.7)
        overflow_n = BUFFER / size
        ax.axvline(overflow_n, color="#bbbbbb", linewidth=0.9, linestyle=":")
        ax.text(overflow_n * 1.12, 2.6, "per-sender burst\n= 1 MiB buffer",
                fontsize=8, color=MUTED)
        ax.axhline(50, color="#cccccc", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(reps, [str(n) for n in reps])
        ax.set_xlabel("repetitions of the same WQE per sender (log)",
                      fontsize=9, color=MUTED)
        style(ax, f"S = {size >> 10} KiB per WQE")
    axes[0].set_ylabel("aggregate goodput (GB/s, log)", fontsize=9,
                       color=MUTED)
    axes[0].annotate("collapse: 0.064 GB/s\n(one 50 ms RTO tail)",
                     xy=(100, 0.064), xytext=(150, 0.6), fontsize=8.5,
                     color=MUTED,
                     arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    axes[1].legend(handles=[
        Line2D([], [], color=NN, linestyle="--", marker="o",
               label="fluid ideal (2 senders)"),
        Line2D([], [], color=CN, marker="o", label="rnic-cn (2 senders, lossless)"),
        Line2D([], [], color=DQ_ECN, marker="s",
               label="DCQCN ECN-only (2 senders)"),
        Line2D([], [], color=DQ_PFC, marker="^",
               label="DCQCN ECN+PFC (identical: PFC never engages)"),
        Line2D([], [], color=DQ_ECN, linestyle=":", marker="s", alpha=0.7,
               label="DCQCN, single sender (no contention)"),
    ], frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Repeated same-size WQEs, two senders into one receiver "
                 "(1 MiB buffers): DCQCN collapses past the buffer boundary",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "repeated_wqe_collapse.png", dpi=160)
    plt.close(fig)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
