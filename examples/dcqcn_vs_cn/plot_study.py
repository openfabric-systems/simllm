"""Render the DCQCN vs rnic-cn figures from the study CSVs.

Usage:
    python examples/dcqcn_vs_cn/plot_study.py \\
        --runs /data3/yifeng/simllm-dev/dcqcn-vs-cn-runs \\
        --out examples/dcqcn_vs_cn/plots
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

# Validated categorical palette, all-pairs colorblind-safe in this order.
CN, DQ_ECN, DQ_PFC = "#2a78d6", "#eb6834", "#1baf7a"
NN, INK, MUTED, SURFACE = "#999999", "#222222", "#666666", "#fcfcfb"


def load_dists(path: Path) -> dict[tuple[str, str], list[float]]:
    table = defaultdict(list)
    with open(path) as handle:
        for row in csv.DictReader(handle):
            table[(row["scenario"], row["run"])].append(float(row["ratio"]))
    return {k: sorted(v) for k, v in table.items()}


def cdf(ax, values, **kw):
    n = len(values)
    ax.step(values, [(i + 1) / n for i in range(n)], where="post", **kw)


def style(ax, title, xlabel):
    ax.set_title(title, fontsize=10.5, color=INK, pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_axisbelow(True)
    ax.grid(color="#e6e6e3", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="/data3/yifeng/simllm-dev/dcqcn-vs-cn-runs")
    parser.add_argument("--out", default=str(Path(__file__).parent / "plots"))
    args = parser.parse_args()
    runs = Path(args.runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dists = load_dists(runs / "distributions.csv")

    # Figure 1: a2a16 mixed lognormal CDF (the tail figure).
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for seed in range(1, 9):
        for mode, color in (("ecn-only", DQ_ECN), ("ecn-pfc", DQ_PFC)):
            key = ("a2a16", f"dcqcn-{mode}-s{seed}")
            if key in dists:
                cdf(ax, dists[key], color=color, alpha=0.32, linewidth=1.0)
    cdf(ax, dists[("a2a16", "rnic-nn")], color=NN, linewidth=1.6, linestyle="--")
    cdf(ax, dists[("a2a16", "rnic-cn")], color=CN, linewidth=2.2)
    ax.set_xscale("log")
    ax.axvline(1.0, color="#cccccc", linewidth=0.8)
    style(ax, "Mixed lognormal all-to-all (16 ranks, 1 MiB buffers): "
              "per-flow FCT normalized to the fluid ideal",
          "normalized FCT (log scale)")
    ax.set_ylabel("fraction of flows", fontsize=9, color=MUTED)
    ax.legend(handles=[
        Line2D([], [], color=CN, linewidth=2.2, label="rnic-cn (deterministic)"),
        Line2D([], [], color=NN, linewidth=1.6, linestyle="--", label="rnic-nn"),
        Line2D([], [], color=DQ_ECN, alpha=0.6, label="DCQCN ECN-only (8 seeds)"),
        Line2D([], [], color=DQ_PFC, alpha=0.6, label="DCQCN ECN+PFC (8 seeds)"),
    ], frameon=False, fontsize=9.5, loc="lower right")
    ax.annotate("DCQCN RTO tail\n(p99 3067x, max 3528x)", xy=(420, 0.955),
                fontsize=9.5, color=MUTED, ha="center")
    ax.annotate("cn max 24.9x", xy=(25, 0.90), fontsize=9.5, color=CN, ha="left")
    fig.tight_layout()
    fig.savefig(out / "a2a16_nfct_cdf.png", dpi=160)
    plt.close(fig)

    # Figure 2: incast p99 per cell, log scale (the boundary figure).
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    cells = [("inc8-s65536", "8 x 64 KiB\n(fits buffer)"),
             ("inc32-s65536", "32 x 64 KiB"),
             ("inc8-s2097152", "8 x 2 MiB"),
             ("inc32-s2097152", "32 x 2 MiB")]
    p99 = {}
    with open(runs / "summary.csv") as handle:
        for row in csv.DictReader(handle):
            if row["check"] == "R1a":
                p99[(row["cell"], row["mode"])] = (
                    float(row["cn_p99"]), float(row["dq_p99"]))
    for i, (cell, label) in enumerate(cells):
        cn_val = p99[(cell, "ecn-only")][0]
        ax.bar(i - 0.27, cn_val, 0.24, color=CN, edgecolor=SURFACE)
        ax.text(i - 0.27, cn_val * 1.15, f"{cn_val:.2f}", ha="center",
                fontsize=9, color=INK)
        for j, mode in enumerate(("ecn-only", "ecn-pfc")):
            value = p99[(cell, mode)][1]
            x = i + 0.03 + j * 0.27
            ax.bar(x, value, 0.24, color=DQ_ECN if mode == "ecn-only" else DQ_PFC,
                   edgecolor=SURFACE)
            ax.text(x, value * 1.15, f"{value:.5g}", ha="center", fontsize=9,
                    color=INK)
    ax.set_yscale("log")
    ax.set_xticks(range(4), [label for _, label in cells])
    style(ax, "Incast p99 normalized FCT (log scale); the left cell fits the\n"
              "1 MiB buffer and DCQCN wins it, the rest overflow and it collapses", "")
    ax.set_ylabel("p99 normalized FCT", fontsize=9, color=MUTED)
    ax.legend(handles=[
        Line2D([], [], color=CN, marker="s", linestyle="", label="rnic-cn"),
        Line2D([], [], color=DQ_ECN, marker="s", linestyle="", label="DCQCN ECN-only"),
        Line2D([], [], color=DQ_PFC, marker="s", linestyle="", label="DCQCN ECN+PFC"),
    ], frameon=False, fontsize=9.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "incast_p99.png", dpi=160)
    plt.close(fig)

    # Figure 3: ECMP collisions per seed and the contention-free TP boundary.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    fig.patch.set_facecolor(SURFACE)
    seed_maxes = []
    tp = {}
    with open(runs / "summary.csv") as handle:
        for row in csv.DictReader(handle):
            if row["check"] == "R2b":
                seed_maxes = [float(v) for v in
                              row["seed_maxes"].strip("[]").split(",")]
            if row["check"] == "R2a":
                cn_max = float(row["cn_max"])
    with open(runs / "tp_totals.csv") as handle:
        for row in csv.DictReader(handle):
            tp[row["run"]] = int(row["total_ps"])
    ax1.set_facecolor(SURFACE)
    ax1.scatter(range(1, 9), seed_maxes, color=DQ_PFC, zorder=3,
                label="DCQCN max per seed")
    ax1.axhline(cn_max, color=CN, linewidth=2.0, label=f"rnic-cn max {cn_max:.2f}")
    ax1.axhline(2.0, color="#cccccc", linewidth=0.8, linestyle=":")
    ax1.text(8.1, 2.0, "2x = one\ncollision", fontsize=8.5, color=MUTED,
             va="center")
    style(ax1, "ECMP collisions: cross-leaf permutation,\n16 x 8 MiB, "
               "max normalized FCT per seed", "DCQCN seed")
    ax1.set_ylabel("max normalized FCT", fontsize=9, color=MUTED)
    ax1.set_ylim(0, 4.2)
    ax1.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax2.set_facecolor(SURFACE)
    bars = [("fluid ideal", tp["fluid"], NN),
            ("DCQCN", tp["dcqcn-s1"], DQ_PFC),
            ("rnic-cn", tp["rnic-cn"], CN)]
    for i, (label, value, color) in enumerate(bars):
        ax2.bar(i, value * 1e-9, 0.55, color=color, edgecolor=SURFACE)
        ax2.text(i, value * 1e-9 + 2, f"{value * 1e-9:.1f}", ha="center",
                 fontsize=9, color=INK)
    ax2.set_xticks(range(3), [b[0] for b in bars])
    style(ax2, "Contention-free cross-node TP request:\nthe registered DCQCN "
               "outperformance", "")
    ax2.set_ylabel("request total, ms", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / "ecmp_and_tp_boundary.png", dpi=160)
    plt.close(fig)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
