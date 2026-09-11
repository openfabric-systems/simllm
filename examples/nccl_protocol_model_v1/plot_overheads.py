"""Show isolated timing controls and the independent visibility illustration."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text())
    with args.summary.open() as stream:
        rows = list(csv.DictReader(stream))
    lookup = {
        (
            r["architecture"],
            r["lane"],
            int(r["width"]),
            int(r["bytes"]),
            int(r["warm"]),
            int(r["timed"]),
            int(r["persistent"]),
            int(r["rotate"]),
        ): float(r["median_us"])
        for r in rows
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.8))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.78, bottom=0.16, hspace=0.65, wspace=0.30)
    fig.text(
        0.10,
        0.96,
        "Timing boundaries and GPU visibility overhead",
        fontsize=12,
        weight="bold",
        color="#25313b",
    )
    fig.text(
        0.10,
        0.91,
        "Controlled factors | Five process repeats | Each plotted time is a median",
        fontsize=8.5,
        color="#657380",
    )
    arms = [
        ("Original counts, fresh workers", (5, 20, 0, 0), "#454f5b"),
        ("Only iterations: 20 → 100", (5, 100, 0, 0), "#bc7626"),
        ("Counts, reuse and rotation", (20, 100, 1, 1), "#1878a1"),
    ]
    fig.legend(
        handles=[
            Line2D([], [], color=color, lw=1.2, marker="o", ms=3, label=label)
            for label, _, color in arms
        ],
        loc="upper left",
        bbox_to_anchor=(0.09, 0.87),
        ncol=2,
        frameon=False,
        fontsize=7.5,
    )
    sizes = [524288, 786432, 1048576, 1310720, 1572864, 2097152, 3145728, 4194304]
    for ax, width in zip(axes[0], (2, 4), strict=True):
        for label, settings, color in arms:
            delta = [
                lookup[("a100", "event", width, size, *settings)]
                - lookup[("a100", "reference", width, size, 20, 100, -1, -1)]
                for size in sizes
            ]
            ax.plot(np.array(sizes) / 2**20, delta, color=color, lw=1.1, marker="o", ms=3)
        ax.axhline(0, color="#8896a3", lw=0.7, ls="--")
        ax.grid(axis="y", color="#e2e7eb", lw=0.6)
        ax.set_title(f"A100 | {width} GPUs", loc="left")
        ax.set_xlabel("Payload per GPU (MiB)")
        ax.set_ylabel("Event − benchmark (µs)")
        ax.set_xticks([0.5, 1, 2, 3, 4])
    ax = axes[1, 0]
    probes = report["visibility_probe"]
    colors = ["#647da0", "#b36a76"]
    for i, variant in enumerate(("embedded_ready", "separate_fenced")):
        chosen = [
            next(r for r in probes if r["architecture"] == arch and r["variant"] == variant)
            for arch in ("a100", "gh200")
        ]
        x = np.arange(2) + (i - 0.5) * 0.32
        median = np.array([r["median_ns"] / 1000 for r in chosen])
        low = np.array([r["q1_ns"] / 1000 for r in chosen])
        high = np.array([r["q3_ns"] / 1000 for r in chosen])
        ax.bar(
            x,
            median,
            width=0.29,
            color=colors[i],
            yerr=[median - low, high - median],
            capsize=2,
            error_kw={"elinewidth": 0.8},
        )
        for position, value in zip(x, median, strict=True):
            ax.text(position, value + 0.3, f"{value:.2f}", ha="center", fontsize=7)
    ax.set_title("Acknowledged GPU exchange", loc="left")
    ax.set_xticks([0, 1], ["A100", "GH200"])
    ax.set_ylabel("Round-trip time (µs)")
    ax.set_ylim(0, 12.3)
    ax.grid(axis="y", color="#e2e7eb", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Patch(facecolor=colors[0], label="Embedded ready"),
            Patch(facecolor=colors[1], label="Separate + fence"),
        ],
        loc="upper center",
        fontsize=6.8,
        ncol=2,
        frameon=False,
    )
    ax = axes[1, 1]
    ax.set_title("Data and readiness in NCCL", loc="left")
    data = [0.5, 120 / 128, 1]
    labels = ["LL", "LL128", "Simple"]
    ax.barh([2, 1, 0], data, color="#447d9b", height=0.48)
    ax.barh([2, 1, 0], [1 - v for v in data], left=data, color="#d39c60", height=0.48)
    ax.set_yticks([2, 1, 0], labels)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_xlabel("Fraction of protocol-buffer bytes")
    for y, text in [
        (2, "8 B data + 8 B flags"),
        (1, "120 B data + 8 B flag"),
        (0, "Data; separate head/tail counters"),
    ]:
        ax.text(0, y + 0.31, text, fontsize=6.8, va="bottom")
    ax.set_ylim(-0.5, 2.65)
    fig.text(
        0.10,
        0.081,
        "No isolated timing control explains the A100 offset across all 16 main payload/width cases.",
        fontsize=7.3,
        color="#657380",
    )
    fig.text(
        0.10,
        0.042,
        "Exchange probe: one lane, system atomics and a watchdog. Its latency is not an identified NCCL RTT.",
        fontsize=7.3,
        color="#657380",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output.with_suffix(suffix))


if __name__ == "__main__":
    main()
