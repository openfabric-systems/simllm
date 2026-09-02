#!/usr/bin/env python3
"""Plot the drain window: goodput and receiver ingress discards against the
inter-burst gap, at two message sizes.

    python tools/plot_drain.py --data data/gapsweep.csv --out figures/drain.pdf
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# Render without a display; matplotlib allows the backend to be selected after
# pyplot is imported.
matplotlib.use("Agg")

STYLE = {
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "legend.fontsize": 6,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.2,
    "grid.linewidth": 0.4,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        lines = [line for line in fh if not line.startswith("#")]
    return list(csv.DictReader(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rows = read_rows(args.data)
    by_size: dict[int, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_size[int(r["size_bytes"])].append(r)
    gaps = sorted({int(r["gap_us"]) for r in rows})
    positions = {gap: i for i, gap in enumerate(gaps)}

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(3.35, 2.35))

    series = [
        (8192, "8 KiB", "o", "#a33b20"),
        (65536, "64 KiB", "s", "#1f4e79"),
    ]
    for size, label, marker, colour in series:
        points = sorted(by_size[size], key=lambda r: int(r["gap_us"]))
        xs = [positions[int(r["gap_us"])] for r in points]
        ax.plot(
            xs,
            [float(r["burst_goodput_gbps"]) for r in points],
            marker=marker,
            color=colour,
            linestyle="-",
            label=f"{label}, in burst",
        )
        ax.plot(
            xs,
            [float(r["wall_goodput_gbps"]) for r in points],
            marker=marker,
            color=colour,
            linestyle="--",
            markerfacecolor="white",
            markeredgewidth=0.7,
            label=f"{label}, over the wall clock",
        )
        lossy = [r for r in points if float(r["rx_discards_phy"]) > 0]
        ax.scatter(
            [positions[int(r["gap_us"])] for r in lossy],
            [float(r["burst_goodput_gbps"]) for r in lossy],
            marker="x",
            s=26,
            linewidths=0.9,
            color="black",
            zorder=5,
            label="_nolegend_",
        )

    ax.scatter([], [], marker="x", s=26, linewidths=0.9, color="black",
               label="receiver discards moved")

    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([str(g) for g in gaps])
    ax.set_xlabel(r"inter-burst gap ($\mu$s)")
    ax.set_ylabel("goodput (Gb/s)")
    ax.set_ylim(0, 116)
    ax.set_xlim(-0.35, len(gaps) - 0.35)
    ax.grid(True, axis="y", color="0.9")
    ax.set_axisbelow(True)
    ax.annotate(
        "a 4 $\\mu$s gap raises\n8 KiB goodput 13.8%",
        xy=(1, 88.5),
        xytext=(0.05, 103),
        fontsize=5.8,
        color="0.25",
        arrowprops={"arrowstyle": "-", "linewidth": 0.5, "color": "0.5"},
    )
    ax.legend(loc="lower left", frameon=False, handlelength=1.8, ncol=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
