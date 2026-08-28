# SPDX-FileCopyrightText: Copyright (c) 2026 Yifeng Wang
# SPDX-License-Identifier: Apache-2.0
"""Render the MiniMax expert-parallel scaling publication figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def render(record_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    rows = record["rows"]
    widths = [row["expert_parallel"] for row in rows]
    external_steps = [row["aiconfigurator_step_ms"] for row in rows]
    sparse_steps = [row["family_s_packet_step_ms"] for row in rows]
    dense_external = [row["family_d_external_ms"] for row in rows]
    dense_packet = [row["family_d_packet_ms"] for row in rows]
    contention_ratios = [row["family_d_ratio"] for row in rows]
    dense_measured_indices = [
        index
        for index, row in enumerate(rows)
        if row["family_d_simulated_messages_per_layer"] > 0
    ]
    dense_extrapolated_indices = [
        index
        for index, row in enumerate(rows)
        if row["family_d_simulated_messages_per_layer"] == 0
    ]

    dense_label = "Dense fallback: half all-gather + reduce-scatter"
    sparse_label = "Sparse routed: FP8 dispatch + BF16 combine"
    packet_dense_label = "Dense packet: same half all-gather + reduce-scatter"
    ratio_label = "Dense packet / dense external, identical half traffic"
    caption = (
        "Family D compares identical dense half-precision all-gather plus "
        "reduce-scatter bytes. Family S compares that dense fallback with sparse "
        "routed FP8 dispatch plus BF16 combine. Every point samples one layer of "
        "65; D at EP 256 extrapolates from the measured full EP 128 population."
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 6.2,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
        }
    )
    figure, (left, middle, right) = plt.subplots(1, 3, figsize=(7.5, 3.25))
    blue = "#2F6B9A"
    orange = "#D56A2C"
    green = "#2D7D63"

    left.plot(
        widths,
        external_steps,
        color=blue,
        marker="o",
        linewidth=1.8,
        markersize=4.5,
        label=dense_label,
    )
    left.plot(
        widths,
        sparse_steps,
        color=orange,
        marker="D",
        linewidth=1.8,
        markersize=4.2,
        label=sparse_label,
    )
    left.set_title("Family S strategy steps")
    left.set_xlabel("Expert-parallel width")
    left.set_ylabel("Step time (ms)")
    left.set_xscale("log", base=2)
    left.set_xticks(widths)
    left.xaxis.set_major_formatter(ScalarFormatter())
    left.grid(axis="y", color="#D8D8D8", linewidth=0.6)
    left.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.94,
        loc="upper left",
    )

    middle.plot(
        widths,
        dense_external,
        color=blue,
        marker="o",
        linewidth=1.8,
        markersize=4.5,
        label=dense_label,
    )
    middle.plot(
        widths,
        dense_packet,
        color=green,
        linewidth=1.8,
        label=packet_dense_label,
    )
    middle.scatter(
        [widths[index] for index in dense_measured_indices],
        [dense_packet[index] for index in dense_measured_indices],
        color=green,
        marker="o",
        s=23,
        zorder=3,
    )
    middle.scatter(
        [widths[index] for index in dense_extrapolated_indices],
        [dense_packet[index] for index in dense_extrapolated_indices],
        color=green,
        marker="s",
        facecolors="white",
        linewidths=1.4,
        s=28,
        zorder=3,
    )
    middle.set_title("Family D identical traffic")
    middle.set_xlabel("Expert-parallel width")
    middle.set_ylabel("Collective time (ms)")
    middle.set_xscale("log", base=2)
    middle.set_yscale("log")
    middle.set_xticks(widths)
    middle.xaxis.set_major_formatter(ScalarFormatter())
    middle.grid(axis="y", color="#D8D8D8", linewidth=0.6)
    middle.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.94,
        loc="upper left",
    )

    right.plot(
        widths,
        contention_ratios,
        color=green,
        linewidth=1.8,
        label=ratio_label,
    )
    right.scatter(
        [widths[index] for index in dense_measured_indices],
        [contention_ratios[index] for index in dense_measured_indices],
        color=green,
        marker="o",
        s=23,
        zorder=3,
    )
    right.scatter(
        [widths[index] for index in dense_extrapolated_indices],
        [contention_ratios[index] for index in dense_extrapolated_indices],
        color=green,
        marker="s",
        facecolors="white",
        linewidths=1.4,
        s=28,
        zorder=3,
    )
    right.axhline(
        1.0,
        color="#555555",
        linestyle=(0, (4, 3)),
        linewidth=1.1,
        label="Frozen lower bound 1.0",
    )
    for width, ratio in zip(widths, contention_ratios, strict=True):
        right.annotate(
            f"{ratio:.3f}",
            (width, ratio),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=6.2,
            color=green,
        )
    right.set_title("Family D contention ratio")
    right.set_xlabel("Expert-parallel width")
    right.set_ylabel("Packet dense / external dense")
    right.set_xscale("log", base=2)
    right.set_xticks(widths)
    right.xaxis.set_major_formatter(ScalarFormatter())
    right.grid(axis="y", color="#D8D8D8", linewidth=0.6)
    right.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
        loc="best",
    )

    figure.text(
        0.5,
        0.01,
        caption,
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#333333",
        wrap=True,
    )
    figure.tight_layout(rect=(0, 0.12, 1, 1), w_pad=1.7)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "minimax_ep_scaling.png"
    pdf = output_dir / "minimax_ep_scaling.pdf"
    metadata = output_dir / "minimax_ep_scaling.metadata.json"
    figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    metadata.write_text(
        json.dumps(
            {
                "caption": caption,
                "series": [
                    {
                        "panel": "family-s-strategy-steps",
                        "label": dense_label,
                        "strategy": rows[0]["family_s_dense_strategy"],
                        "traffic_definition": rows[0][
                            "family_s_dense_traffic_definition"
                        ],
                    },
                    {
                        "panel": "family-s-strategy-steps",
                        "label": sparse_label,
                        "strategy": rows[0]["family_s_sparse_strategy"],
                        "traffic_definition": rows[0][
                            "family_s_sparse_traffic_definition"
                        ],
                    },
                    {
                        "panel": "family-d-identical-traffic",
                        "label": dense_label,
                        "strategy": rows[0]["family_d_external_strategy"],
                        "traffic_definition": rows[0][
                            "family_d_external_traffic_definition"
                        ],
                    },
                    {
                        "panel": "family-d-identical-traffic",
                        "label": packet_dense_label,
                        "strategy": rows[0]["family_d_packet_strategy"],
                        "traffic_definition": rows[0][
                            "family_d_packet_traffic_definition"
                        ],
                    },
                    {
                        "panel": "family-d-contention-ratio",
                        "label": ratio_label,
                        "strategy": "same dense strategy in numerator and denominator",
                        "traffic_definition": rows[0][
                            "family_d_packet_traffic_definition"
                        ],
                    },
                ],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return png, pdf, metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    png, pdf, metadata = render(args.record, args.output_dir)
    print(png)
    print(pdf)
    print(metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
