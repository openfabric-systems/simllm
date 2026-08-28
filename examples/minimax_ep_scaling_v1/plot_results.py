# SPDX-FileCopyrightText: Copyright (c) 2026 Yifeng Wang
# SPDX-License-Identifier: Apache-2.0
"""Render the MiniMax expert-parallel scaling publication figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

QWEN_REFERENCE_RATIO = 1.042715399805


def render(record_path: Path, output_dir: Path) -> tuple[Path, Path]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    rows = record["rows"]
    widths = [row["expert_parallel"] for row in rows]
    external_steps = [row["aiconfigurator_step_ms"] for row in rows]
    simllm_steps = [row["simllm_step_ms"] for row in rows]
    ratios = [row["ratio"] for row in rows]
    dispatch_shares = [100 * row["dispatch_share"] for row in rows]
    full_peer_indices = [
        index for index, row in enumerate(rows) if not row["peer_subset"]
    ]
    subset_indices = [index for index, row in enumerate(rows) if row["peer_subset"]]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
        }
    )
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.25, 3.35))
    blue = "#2F6B9A"
    orange = "#D56A2C"

    left.plot(
        widths,
        external_steps,
        color=blue,
        marker="o",
        linewidth=1.8,
        markersize=5,
        label="AIConfigurator",
    )
    left.plot(
        widths,
        simllm_steps,
        color=orange,
        linewidth=1.8,
        label="SimLLM",
    )
    left.scatter(
        [widths[index] for index in full_peer_indices],
        [simllm_steps[index] for index in full_peer_indices],
        color=orange,
        marker="D",
        facecolors="white",
        linewidths=1.4,
        s=28,
        zorder=3,
    )
    left.scatter(
        [widths[index] for index in subset_indices],
        [simllm_steps[index] for index in subset_indices],
        color=orange,
        marker="s",
        facecolors="white",
        linewidths=1.4,
        s=31,
        zorder=3,
    )
    for width, step, share in zip(
        widths, external_steps, dispatch_shares, strict=True
    ):
        left.annotate(
            f"{share:.1f}% dispatch",
            (width, step),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            va="top",
            color=blue,
            fontsize=7,
        )
    left.set_title("Decode step on the external timing base")
    left.set_xlabel("Expert-parallel width")
    left.set_ylabel("Step time (ms)")
    left.set_xscale("log", base=2)
    left.set_xticks(widths)
    left.xaxis.set_major_formatter(ScalarFormatter())
    left.grid(axis="y", color="#D8D8D8", linewidth=0.6)
    left.legend(frameon=False, loc="upper left")

    right.plot(
        widths,
        ratios,
        color=orange,
        linewidth=1.8,
        label="SimLLM / AIConfigurator",
    )
    right.scatter(
        [widths[index] for index in full_peer_indices],
        [ratios[index] for index in full_peer_indices],
        color=orange,
        marker="D",
        facecolors="white",
        linewidths=1.4,
        s=28,
        zorder=3,
    )
    right.scatter(
        [widths[index] for index in subset_indices],
        [ratios[index] for index in subset_indices],
        color=orange,
        marker="s",
        facecolors="white",
        linewidths=1.4,
        s=31,
        zorder=3,
    )
    right.axhline(
        QWEN_REFERENCE_RATIO,
        color="#555555",
        linestyle=(0, (4, 3)),
        linewidth=1.1,
        label=f"Qwen3-32B reference {QWEN_REFERENCE_RATIO:.4f}",
    )
    right.set_title("Packet-priced step ratio")
    right.set_xlabel("Expert-parallel width")
    right.set_ylabel("Step-time ratio")
    right.set_xscale("log", base=2)
    right.set_xticks(widths)
    right.xaxis.set_major_formatter(ScalarFormatter())
    right.grid(axis="y", color="#D8D8D8", linewidth=0.6)
    right.legend(frameon=False, loc="upper left")

    figure.text(
        0.5,
        0.015,
        (
            "Hollow diamonds use full peers for one layer of 65. The hollow square "
            "uses one receiver per node with every sender. Traffic abstractions differ."
        ),
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#333333",
    )
    figure.tight_layout(rect=(0, 0.075, 1, 1), w_pad=2.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "minimax_ep_scaling.png"
    pdf = output_dir / "minimax_ep_scaling.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png, pdf


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    png, pdf = render(args.record, args.output_dir)
    print(png)
    print(pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
