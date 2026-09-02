#!/usr/bin/env python3
"""Plot goodput against message size at queue depth 1 and 1024.

Reads the curated extract written by the campaign curation step and writes one
PDF. Paths are relative to whatever the caller passes; the script contains no
absolute path.

    python tools/plot_msgsize.py --data data/msgsize.csv --out figures/msgsize.pdf
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

# Canonical fit from the true depth-1 arm (RESULTS-p4-kernels, test E1D1).
T_EFF_US = 4.48
C_GBPS = 97.1

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
    "lines.markersize": 3.0,
    "grid.linewidth": 0.4,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        lines = [line for line in fh if not line.startswith("#")]
    return list(csv.DictReader(lines))


def fixed_offset(size_bytes: float, t_eff_us: float, c_gbps: float) -> float:
    """B = S / (T_eff + S / C), in Gb/s for S in bytes."""
    bits = size_bytes * 8.0
    seconds = t_eff_us * 1e-6 + bits / (c_gbps * 1e9)
    return bits / seconds / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rows = read_rows(args.data)
    arms: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        arms[r["arm"]].append((float(r["size_bytes"]), float(r["goodput_gbps"])))
    for points in arms.values():
        points.sort()

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(3.35, 2.35))

    model_x = [2.0**e for e in range(9, 24)]
    ax.plot(
        model_x,
        [fixed_offset(x, T_EFF_US, C_GBPS) for x in model_x],
        color="0.35",
        linestyle="-",
        linewidth=0.9,
        zorder=1,
        label=rf"fit $T_{{eff}}={T_EFF_US}\,\mu$s, $C={C_GBPS}$ Gb/s",
    )

    series = [
        ("p4_engine_depth1024", "queue depth 1024", "s", "#1f4e79", "-"),
        ("p4_engine_depth1", "queue depth 1", "o", "#a33b20", "-"),
        ("p2_perftest_depth1", "depth 1, benchmark loop", "^", "#7a7a7a", "--"),
    ]
    for arm, label, marker, colour, dash in series:
        points = [p for p in arms.get(arm, []) if p[0] >= 512]
        if not points:
            continue
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker=marker,
            color=colour,
            linestyle=dash,
            markerfacecolor="white" if arm.startswith("p2") else colour,
            markeredgewidth=0.7,
            label=label,
            zorder=3,
        )

    s_half = T_EFF_US * 1e-6 * C_GBPS * 1e9 / 8.0
    ax.axvline(s_half, color="0.6", linestyle=":", linewidth=0.7, zorder=0)
    ax.annotate(
        r"$S_{1/2}=T_{eff}\,C\approx 54$ KiB",
        xy=(s_half * 1.25, 88),
        fontsize=5.8,
        color="0.35",
    )
    ax.axhline(97.7, color="0.6", linestyle=":", linewidth=0.7, zorder=0)
    ax.annotate(
        "multi-queue-pair ceiling 97.7 Gb/s",
        xy=(9e6, 99.5),
        fontsize=5.8,
        color="0.35",
        ha="right",
    )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("message size (bytes)")
    ax.set_ylabel("goodput (Gb/s)")
    ax.set_ylim(0, 112)
    ax.set_xlim(400, 1.2e7)
    ax.set_xticks([2**e for e in (10, 13, 16, 19, 22)])
    ax.set_xticklabels(["1 KiB", "8 KiB", "64 KiB", "512 KiB", "4 MiB"])
    ax.grid(True, which="major", axis="both", color="0.9")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, handlelength=1.8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
