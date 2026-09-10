"""Render the protocol estimate, declared envelope and every measured payload."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simllm.traffic.collective_protocol import NcclRingProtocolModel

COLORS = {2: "#b73846", 4: "#176ba0"}


def load_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def configure():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#84909c",
            "text.color": "#25313b",
            "axes.labelcolor": "#25313b",
            "xtick.color": "#667481",
            "ytick.color": "#667481",
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def curves(models, points, output, label):
    fig, axes = plt.subplots(2, 2, figsize=(7, 6.5), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.17, top=0.79, hspace=0.30, wspace=0.24)
    fig.text(0.09, 0.966, "NCCL protocol work explains the timing step", fontsize=13, weight="bold")
    fig.text(0.09, 0.930, label, fontsize=9, color="#657380")
    handles = [
        Line2D([], [], color="#25313b", lw=1.3, label="Model center"),
        Patch(facecolor="#aac9df", alpha=0.5, label="Software + method envelope"),
        Line2D(
            [],
            [],
            marker="o",
            ms=3,
            mfc="white",
            mec="#25313b",
            ls="none",
            label="Original GPU-event timing",
        ),
        Line2D(
            [], [], marker="+", ms=4, color="#657380", ls="none", label="NVIDIA benchmark on Merlin"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.08, 0.904),
        ncol=2,
        frameon=False,
        fontsize=8,
        columnspacing=1.4,
    )
    for model in models:
        ax = axes[0 if model.width == 2 else 1, 0 if model.architecture == "a100" else 1]
        rows = [
            r
            for r in points
            if r["architecture"] == model.architecture and int(r["width"]) == model.width
        ]
        grid = sorted({int(r["bytes"]) for r in rows})
        if not grid:
            raise ValueError("missing plotted curve")
        # Evaluate at quarter-KiB intervals so source discontinuities remain
        # vertical changes rather than long interpolated diagonal segments.
        dense = sorted(set(range(min(grid), max(grid) + 1, 256)) | set(grid))
        estimates = [model.predict(b) for b in dense]
        x = np.asarray(dense) / 2**20
        ax.fill_between(
            x,
            [e.lower_ps / 1e6 for e in estimates],
            [e.upper_ps / 1e6 for e in estimates],
            color=COLORS[model.width],
            alpha=0.16,
            lw=0,
        )
        ax.plot(
            x, [e.central_ps / 1e6 for e in estimates], color=COLORS[model.width], lw=1.2, zorder=4
        )
        for lane, marker in [("timing", "+"), ("legacy", "o")]:
            subset = sorted([r for r in rows if r["lane"] == lane], key=lambda r: int(r["bytes"]))
            ax.scatter(
                [int(r["bytes"]) / 2**20 for r in subset],
                [float(r["measured_us"]) for r in subset],
                s=8 if lane == "timing" else 7,
                marker=marker,
                facecolors="none" if lane == "legacy" else "#657380",
                **({"edgecolors": "#3b4650"} if lane == "legacy" else {}),
                linewidths=0.5,
                alpha=0.70,
                zorder=3,
            )
        missed = [r for r in rows if not float(r["lower_us"]) <= float(r["measured_us"]) <= float(r["upper_us"])]
        ax.scatter([int(r["bytes"]) / 2**20 for r in missed],
                   [float(r["measured_us"]) for r in missed], marker="x", color="#b67600",
                   s=15, linewidths=.8, zorder=6)
        for start, protocol in model.protocol_starts[1:]:
            ax.axvline(start / 2**20, color="#8896a3", ls=(0, (2, 3)), lw=0.7)
            ax.text(
                start / 2**20 + 0.055,
                0.84,
                protocol.title() if protocol == "SIMPLE" else protocol,
                transform=ax.get_xaxis_transform(),
                fontsize=7,
                color="#657380",
                va="top",
            )
        ax.set_title(f"{model.architecture.upper()}  |  {model.width} GPUs", loc="left", pad=8)
        ax.set_xlim(0.20, 4.08)
        ax.margins(y=0.10)
        ax.grid(axis="y", color="#e0e6eb", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xticks([0.25, 1, 2, 3, 4], ["0.25", "1", "2", "3", "4"])
        ax.set_ylabel("All-reduce time (µs)")
        if model.width == 4:
            ax.set_xlabel("Payload per GPU (MiB)")
        covered = sum(
            float(r["lower_us"]) <= float(r["measured_us"]) <= float(r["upper_us"]) for r in rows
        )
        ax.text(
            0.03,
            0.96,
            f"{covered}/{len(rows)} medians inside",
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
        )
    fig.text(
        0.09,
        0.066,
        "Band: calibration residuals, repeat spread and method allowance. Gold crosses mark uncovered medians.",
        fontsize=7.3,
        color="#657380",
    )
    fig.text(
        0.09,
        0.038,
        "LL carries data and flags together; Simple fences data, publishes counters, and polls in the same kernel.",
        fontsize=7.3,
        color="#657380",
    )
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--label",
        default="Retrospective dense comparison | 241 payloads per hardware curve | NCCL 2.31.2",
    )
    args = parser.parse_args()
    configure()
    models = tuple(NcclRingProtocolModel.from_json(v) for v in json.loads(args.models.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    curves(models, load_rows(args.predictions), args.output, args.label)


if __name__ == "__main__":
    main()
