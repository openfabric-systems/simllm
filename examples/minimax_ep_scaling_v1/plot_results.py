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
    superseded_sparse_steps = [
        row["family_s_superseded_packet_step_ms"] for row in rows
    ]
    dense_external = [row["family_d_external_ms"] for row in rows]
    dense_packet = [row["family_d_packet_ms"] for row in rows]
    superseded_dense_packet = [
        row["family_d_superseded_packet_ms"] for row in rows
    ]
    cost_model_ratios = [row["family_d_ratio"] for row in rows]
    superseded_cost_model_ratios = [
        row["family_d_superseded_ratio"] for row in rows
    ]
    dense_scored_indices = [
        index
        for index, row in enumerate(rows)
        if row["family_d_score_status"] == "scored measured cell"
    ]
    dense_diagnostic_indices = [
        index
        for index, row in enumerate(rows)
        if row["family_d_score_status"] == "unscored post-specified diagnostic"
    ]
    ep8_index = widths.index(8)
    scored_cross_node_indices = [
        index
        for index in dense_scored_indices
        if rows[index]["family_d_cross_node_contention_present"]
    ]

    dense_label = (
        "External NCCL cost model\nDense fallback: half all-gather + reduce-scatter"
    )
    sparse_label = (
        "Corrected sparse routed payload\n"
        "FP8 dispatch + BF16 combine + transferred collective floor"
    )
    superseded_sparse_label = (
        "Superseded sparse routed payload\nFP8 dispatch + BF16 combine, floor omitted"
    )
    packet_dense_label = (
        "Corrected packet Clos cost model\n"
        "Direct all-pairs half traffic + aggregate collective floor"
    )
    superseded_packet_dense_label = (
        "Superseded packet Clos cost model\n"
        "Direct all-pairs half traffic, floor omitted"
    )
    ratio_label = (
        "Corrected packet Clos / external NCCL\n"
        "same requested half-element count + collective floor"
    )
    superseded_ratio_label = (
        "Superseded packet Clos / external NCCL\n"
        "same requested count, collective floor omitted"
    )
    caption = (
        "Family D corrected ratios are "
        + ", ".join(f"{ratio:.3f}" for ratio in cost_model_ratios)
        + " at EP 8, 32, 128 and 256; the superseded floor-omitting ratios are "
        + ", ".join(f"{ratio:.3f}" for ratio in superseded_cost_model_ratios)
        + ". The corrected packet arm charges aggregate all-gather and "
        "reduce-scatter floors once outside local/fabric max service. EP 8 has "
        "no cross-node traffic. EP 256 uses an unscored component-wise diagnostic. "
        "Family S compares the dense fallback with corrected sparse routed FP8 "
        "dispatch plus BF16 combine; its floor-omitting series is superseded."
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
    figure, (left, middle, right) = plt.subplots(1, 3, figsize=(11.25, 4.0))
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
    left.plot(
        widths,
        superseded_sparse_steps,
        color="#777777",
        marker="x",
        linestyle=(0, (3, 2)),
        linewidth=1.2,
        markersize=4.0,
        label=superseded_sparse_label,
    )
    left.set_title("Family S strategy steps")
    left.set_xlabel("Expert-parallel width")
    left.set_ylabel("Step time (ms)")
    left.set_xscale("log", base=2)
    left.set_xticks(widths)
    left.xaxis.set_major_formatter(ScalarFormatter())
    left.set_ylim(9.0, 67.0)
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
    middle.plot(
        widths,
        superseded_dense_packet,
        color="#777777",
        marker="x",
        linestyle=(0, (3, 2)),
        linewidth=1.2,
        markersize=4.0,
        label=superseded_packet_dense_label,
    )
    middle.scatter(
        [widths[index] for index in dense_scored_indices],
        [dense_packet[index] for index in dense_scored_indices],
        color=green,
        marker="o",
        s=23,
        zorder=3,
    )
    middle.scatter(
        [widths[index] for index in dense_diagnostic_indices],
        [dense_packet[index] for index in dense_diagnostic_indices],
        color=green,
        marker="s",
        facecolors="white",
        linewidths=1.4,
        s=28,
        zorder=3,
    )
    middle.set_title("Family D cost models, same logical count")
    middle.set_xlabel("Expert-parallel width")
    middle.set_ylabel("Collective time (ms)")
    middle.set_xscale("log", base=2)
    middle.set_yscale("log")
    middle.set_xticks(widths)
    middle.xaxis.set_major_formatter(ScalarFormatter())
    middle.set_ylim(0.03, 100.0)
    middle.grid(axis="y", color="#D8D8D8", linewidth=0.6)
    middle.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.94,
        loc="lower right",
    )

    right.plot(
        [widths[index] for index in dense_scored_indices],
        [cost_model_ratios[index] for index in dense_scored_indices],
        color=green,
        linewidth=1.8,
        label=ratio_label,
    )
    right.plot(
        widths,
        superseded_cost_model_ratios,
        color="#777777",
        marker="x",
        linestyle=(0, (3, 2)),
        linewidth=1.2,
        markersize=4.0,
        label=superseded_ratio_label,
    )
    right.plot(
        [widths[dense_scored_indices[-1]], widths[dense_diagnostic_indices[0]]],
        [
            cost_model_ratios[dense_scored_indices[-1]],
            cost_model_ratios[dense_diagnostic_indices[0]],
        ],
        color=green,
        linewidth=1.2,
        linestyle=(0, (2, 2)),
    )
    right.scatter(
        [widths[index] for index in scored_cross_node_indices],
        [cost_model_ratios[index] for index in scored_cross_node_indices],
        color=green,
        marker="o",
        s=23,
        zorder=3,
    )
    right.scatter(
        [widths[ep8_index]],
        [cost_model_ratios[ep8_index]],
        color="#9B3A32",
        marker="X",
        s=32,
        zorder=4,
        label="EP 8: no cross-node traffic",
    )
    right.scatter(
        [widths[index] for index in dense_diagnostic_indices],
        [cost_model_ratios[index] for index in dense_diagnostic_indices],
        color=green,
        marker="s",
        facecolors="white",
        linewidths=1.4,
        s=28,
        zorder=3,
        label="EP 256: unscored component diagnostic",
    )
    right.axhline(
        1.0,
        color="#555555",
        linestyle=(0, (4, 3)),
        linewidth=1.1,
        label="Frozen scored lower bound 1.0",
    )
    for width, ratio in zip(widths, cost_model_ratios, strict=True):
        right.annotate(
            f"{ratio:.3f}",
            (width, ratio),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=6.2,
            color=green,
        )
    right.set_title("Family D cost-model ratio")
    right.set_xlabel("Expert-parallel width")
    right.set_ylabel("Packet Clos / external NCCL")
    right.set_xscale("log", base=2)
    right.set_xticks(widths)
    right.xaxis.set_major_formatter(ScalarFormatter())
    right.set_ylim(-0.03, 1.3)
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
    figure.tight_layout(rect=(0, 0.12, 1, 1), w_pad=2.4)
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
                        "panel": "family-s-strategy-steps",
                        "label": superseded_sparse_label,
                        "strategy": rows[0]["family_s_sparse_strategy"],
                        "traffic_definition": (
                            rows[0]["family_s_sparse_traffic_definition"]
                            + "; superseded omission of intra-node transport and "
                            "fixed collective overheads"
                        ),
                    },
                    {
                        "panel": "family-d-cost-model-times",
                        "label": dense_label,
                        "strategy": rows[0]["family_d_external_strategy"],
                        "traffic_definition": rows[0][
                            "family_d_external_traffic_definition"
                        ],
                    },
                    {
                        "panel": "family-d-cost-model-times",
                        "label": packet_dense_label,
                        "strategy": rows[0]["family_d_packet_strategy"],
                        "traffic_definition": rows[0][
                            "family_d_packet_traffic_definition"
                        ],
                    },
                    {
                        "panel": "family-d-cost-model-times",
                        "label": superseded_packet_dense_label,
                        "strategy": rows[0]["family_d_packet_strategy"],
                        "traffic_definition": (
                            rows[0]["family_d_packet_traffic_definition"]
                            + "; superseded omission of intra-node transport and "
                            "fixed collective overheads"
                        ),
                    },
                    {
                        "panel": "family-d-cost-model-ratio",
                        "label": ratio_label,
                        "strategy": (
                            "same requested dense logical element count, different "
                            "physical realizations and cost models"
                        ),
                        "traffic_definition": rows[0][
                            "family_d_packet_traffic_definition"
                        ],
                    },
                    {
                        "panel": "family-d-cost-model-ratio",
                        "label": superseded_ratio_label,
                        "strategy": (
                            "same requested dense logical element count with the "
                            "packet collective floor omitted"
                        ),
                        "traffic_definition": rows[0][
                            "family_d_packet_traffic_definition"
                        ],
                    },
                ],
                "point_annotations": {
                    "ep8": "zero cross-node traffic; corrected floor-bound ratio",
                    "ep256": "unscored post-specified component-wise diagnostic",
                },
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
