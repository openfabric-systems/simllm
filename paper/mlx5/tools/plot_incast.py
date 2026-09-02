#!/usr/bin/env python3
"""Plot the incast tax (top) and the switch egress buffer identity (bottom).

    python tools/plot_incast.py --data data/incast.csv \
        --buffer data/buffer.csv --out figures/incast.pdf
"""

from __future__ import annotations

import argparse
import csv
import statistics
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
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.2,
    "grid.linewidth": 0.4,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
}

CASES = [
    ("solo_1MiB", "solo\n1 MiB"),
    ("incast2to1_1MiB", "2 to 1\n1 MiB"),
    ("fanout1to2_1MiB", "1 to 2\n1 MiB"),
    ("solo_512B", "solo\n512 B"),
    ("incast2to1_512B", "2 to 1\n512 B"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        lines = [line for line in fh if not line.startswith("#")]
    return list(csv.DictReader(lines))


def maybe_float(value: str) -> float | None:
    return float(value) if value not in ("", None) else None


def plot_tax(ax, rows: list[dict[str, str]]) -> None:
    by_case = {r["case"]: r for r in rows}
    xs = range(len(CASES))
    width = 0.38
    wire, app, labels = [], [], []
    for key, label in CASES:
        row = by_case[key]
        w = maybe_float(row["wire_gbps"])
        a = maybe_float(row["app_goodput_gbps"])
        # The fan-out control reverses the roles, so the receiver-side wire
        # counter is not the quantity of interest there.
        wire.append(0.0 if key.startswith("fanout") else (w or 0.0))
        app.append(a or 0.0)
        labels.append(label)

    ax.bar([x - width / 2 for x in xs], wire, width, color="#c8d6e5",
           edgecolor="#1f4e79", linewidth=0.6, label="wire (physical layer)")
    ax.bar([x + width / 2 for x in xs], app, width, color="#e8c6b8",
           edgecolor="#a33b20", linewidth=0.6, label="application goodput")

    tax_index = next(i for i, (k, _) in enumerate(CASES) if k == "incast2to1_1MiB")
    ax.annotate(
        "",
        xy=(tax_index + width / 2, app[tax_index]),
        xytext=(tax_index + width / 2, wire[tax_index]),
        arrowprops={"arrowstyle": "<->", "linewidth": 0.6, "color": "black"},
    )
    ax.annotate(
        "26.9% of the link:\n1.65% loss amplified 16x",
        xy=(tax_index + 0.55, (app[tax_index] + wire[tax_index]) / 2),
        fontsize=5.8,
        va="center",
    )
    ax.annotate("no receiver-side wire", xy=(2 - width / 2, 6), rotation=90,
                fontsize=5.5, ha="center", va="bottom", color="0.35")

    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Gb/s")
    ax.set_ylim(0, 118)
    ax.grid(True, axis="y", color="0.9")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, handlelength=1.4, ncol=2)


def plot_buffer(ax, rows: list[dict[str, str]]) -> None:
    fill: dict[float, list[float]] = defaultdict(list)
    drain: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        excess = round(float(r["excess_gbps"]), 1)
        fill[excess].append(float(r["b_fill_mb"]))
        drain[excess].append(float(r["b_drain_mb"]))

    xs = sorted(fill)
    ax.errorbar(
        xs,
        [statistics.mean(fill[x]) for x in xs],
        yerr=[
            (max(fill[x]) - min(fill[x])) / 2 if len(fill[x]) > 1 else 0.0
            for x in xs
        ],
        marker="o",
        color="#1f4e79",
        capsize=2,
        linestyle="-",
        label="excess rate $\\times$ time to full",
    )
    ax.errorbar(
        xs,
        [statistics.mean(drain[x]) for x in xs],
        yerr=[
            (max(drain[x]) - min(drain[x])) / 2 if len(drain[x]) > 1 else 0.0
            for x in xs
        ],
        marker="^",
        color="#7a7a7a",
        capsize=2,
        linestyle="--",
        markerfacecolor="white",
        markeredgewidth=0.7,
        label="drain-tail estimate",
    )
    mean_fill = statistics.mean(v for x in xs for v in fill[x])
    ax.axhline(mean_fill, color="#a33b20", linestyle=":", linewidth=0.8)
    ax.annotate(
        f"$B_{{port}} \\approx {mean_fill:.1f}$ MB, tail drop, 0 CE marks",
        xy=(xs[0] - 0.4, mean_fill + 3.1),
        fontsize=5.8,
        color="#a33b20",
    )
    ax.set_xlabel("excess offered rate above the port rate (Gb/s)")
    ax.set_ylabel("buffer estimate (MB)")
    ax.set_ylim(0, 11)
    ax.grid(True, axis="y", color="0.9")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, handlelength=1.8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--buffer", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    plt.rcParams.update(STYLE)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.35, 3.9))
    plot_tax(top, read_rows(args.data))
    plot_buffer(bottom, read_rows(args.buffer))
    fig.tight_layout(h_pad=1.2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
