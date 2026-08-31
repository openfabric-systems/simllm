#!/usr/bin/env python3
"""Render the frozen deployment scan through plot contract version 3."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from simllm.deploy import (
    frontier_record_from_json,
    pareto_front,
    prepare_plot_v3,
)

STUDY_DIR = Path(__file__).resolve().parent
RESULT_SCHEMA = "simllm-deployment-scan-result-v1"
COLORS = {
    "b100-one-node-intra": "#2563a6",
    "h100-two-node-serialized": "#d17c0f",
    "h100-nine-node-incast": "#27845c",
}
FIXED_TIMESTAMP = datetime(2026, 8, 27, tzinfo=timezone.utc)


def prepare_study_plot(result: dict[str, Any]) -> dict[str, Any]:
    """Build one renderer-independent view from the two frozen scan records."""

    if result.get("schema") != RESULT_SCHEMA:
        raise ValueError(f"unexpected deployment scan schema: {result.get('schema')!r}")
    if result.get("status") != "PASS":
        raise ValueError("a void or failing study cannot produce a passing figure")
    records = result.get("records")
    if not isinstance(records, dict):
        raise TypeError("results.records: expected an object")
    analytical = frontier_record_from_json(records["analytical"])
    simulated = frontier_record_from_json(records["simulated"])
    analytical_plot = prepare_plot_v3(analytical)
    simulated_plot = prepare_plot_v3(simulated)
    if analytical_plot["schema"] != "simllm-deployment-frontier-plot-contract-v3":
        raise ValueError("analytical plot did not use contract version 3")
    if simulated_plot["schema"] != "simllm-deployment-frontier-plot-contract-v3":
        raise ValueError("simulated plot did not use contract version 3")
    if [curve["id"] for curve in analytical_plot["curves"]] != [
        curve["id"] for curve in simulated_plot["curves"]
    ]:
        raise ValueError("analytical and simulated curve identities differ")
    return {
        "schema": "simllm-deployment-scan-plot-v1",
        "analytical": analytical_plot,
        "simulated": simulated_plot,
        "pareto_points": [
            {
                "configuration_id": point.configuration_id,
                "batch_per_gpu": point.batch_per_gpu,
                "x": float(point.x_tokens_per_second_per_request),
                "y": float(point.y_tokens_per_second_per_gpu),
            }
            for point in pareto_front(simulated.points)
        ],
    }


def _matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.6,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.2,
            "legend.fontsize": 6.4,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def render(plot: dict[str, Any], output_stem: Path) -> tuple[Path, Path]:
    """Render a two-column PDF and PNG with all authority classes visible."""

    analytical_contract = plot["analytical"]
    simulated_contract = plot["simulated"]
    plt = _matplotlib()
    from matplotlib.lines import Line2D

    figure, axis = plt.subplots(figsize=(7.0, 4.33))
    figure.subplots_adjust(left=0.105, right=0.975, bottom=0.24, top=0.68)

    for analytical_curve, simulated_curve in zip(
        analytical_contract["curves"],
        simulated_contract["curves"],
        strict=True,
    ):
        configuration_id = analytical_curve["id"]
        color = COLORS[configuration_id]
        is_two_node = configuration_id == "h100-two-node-serialized"
        analytical_points = analytical_curve["points"]
        simulated_points = simulated_curve["points"]
        axis.plot(
            [point["analytical_x"] for point in analytical_points],
            [point["analytical_y"] for point in analytical_points],
            color=color,
            linewidth=3.2 if is_two_node else 1.7,
            zorder=2 if is_two_node else 3,
        )
        axis.scatter(
            [point["simulated_x"] for point in analytical_points],
            [point["simulated_y"] for point in analytical_points],
            marker="o",
            facecolor="none",
            edgecolor=color,
            linewidth=0.9,
            s=49 if is_two_node else 31,
            zorder=4 if is_two_node else 5,
        )
        axis.scatter(
            [point["simulated_x"] for point in simulated_points],
            [point["simulated_y"] for point in simulated_points],
            marker="o",
            facecolor=color,
            edgecolor="white",
            linewidth=0.45,
            s=38 if is_two_node else 22,
            zorder=5 if is_two_node else 6,
        )

    front = plot["pareto_points"]
    axis.plot(
        [point["x"] for point in front],
        [point["y"] for point in front],
        color="#111111",
        linewidth=3.8,
        alpha=0.28,
        zorder=2,
    )
    axis.scatter(
        [point["x"] for point in front],
        [point["y"] for point in front],
        marker="o",
        facecolor="none",
        edgecolor="#111111",
        linewidth=1.15,
        s=42,
        zorder=7,
    )
    for index in (0, len(front) - 1):
        point = front[index]
        axis.annotate(
            f"B={point['batch_per_gpu']}",
            (point["x"], point["y"]),
            xytext=(5, 4 if index == 0 else -10),
            textcoords="offset points",
            fontsize=6.2,
            color="#111111",
        )

    paired = simulated_contract["paired_marker"]
    if paired is not None:
        axis.scatter(
            [paired["x"]],
            [paired["y"]],
            marker="D",
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
            s=42,
            zorder=8,
        )
    y_only = simulated_contract["y_only_anchor"]
    if y_only is not None:
        axis.axhline(
            y_only["y"],
            color="#4a4a4a",
            linestyle="--",
            linewidth=1.1,
            zorder=1,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Per-request decode speed (token/s/request)")
    axis.set_ylabel("Output throughput per GPU (token/s/GPU)")
    axis.grid(True, which="both", color="#d5d9df", linewidth=0.5, alpha=0.8)
    axis.margins(x=0.07, y=0.11)
    figure.suptitle("Backend-free DeepSeek deployment scan", y=0.985, fontsize=10.2)
    configuration_handles = [
        Line2D([0], [0], color=COLORS[configuration_id], linewidth=2.1)
        for configuration_id in COLORS
    ]
    configuration_legend = figure.legend(
        configuration_handles,
        ("B100 1-node", "H100 2-node serialized", "H100 9-node incast"),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
        columnspacing=2.2,
        handlelength=2.5,
    )
    figure.add_artist(configuration_legend)
    semantic_handles = (
        Line2D([0], [0], color="#333333", linewidth=1.7),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="#333333",
            linestyle="none",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="#777777",
            markeredgecolor="white",
            linestyle="none",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="#111111",
            markeredgewidth=1.3,
            linestyle="none",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            markerfacecolor="white",
            markeredgecolor="black",
            linestyle="none",
        ),
        Line2D([0], [0], color="#4a4a4a", linestyle="--", linewidth=1.1),
    )
    figure.legend(
        semantic_handles,
        ("Analytical", "ESTIMATE", "SIMULATED", "Pareto", "Paired", "Y-only"),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.84),
        ncol=6,
        frameon=False,
        columnspacing=1.3,
        handlelength=2.0,
    )
    figure.text(
        0.105,
        0.055,
        "Solid lines are analytical floor composition. Hollow dots are ESTIMATE points; "
        "filled dots consume tracked SIM-DERIVED excess.\nThe black outline emphasizes "
        "the exact Pareto front. The production reference remains y-only, so no x value "
        "is invented.",
        ha="left",
        va="bottom",
        fontsize=6.5,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = output_stem.with_suffix(".pdf")
    png = output_stem.with_suffix(".png")
    figure.savefig(
        pdf,
        metadata={"CreationDate": FIXED_TIMESTAMP, "ModDate": FIXED_TIMESTAMP},
    )
    figure.savefig(
        png,
        dpi=220,
        metadata={"Software": "simllm deployment_scan_v1"},
    )
    plt.close(figure)
    return pdf, png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=STUDY_DIR / "results.json")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=STUDY_DIR / "figures" / "deployment-scan-frontier",
    )
    arguments = parser.parse_args()
    result = json.loads(arguments.result.read_text(encoding="utf-8"))
    pdf, png = render(prepare_study_plot(result), arguments.output_stem)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
