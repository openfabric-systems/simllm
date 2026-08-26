#!/usr/bin/env python3
"""Render the versioned CORE-62 and TRAF-68 frontier figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

COLORS = {
    "b100-one-node-intra": "#2563a6",
    "h100-two-node-serialized": "#d17c0f",
    "h100-nine-node-incast": "#27845c",
}


def _coordinate(point: dict[str, Any], kind: str, axis: str) -> float:
    return float(point[f"{kind}_operating_point"][axis]["decimal"])


def prepare_plot(result: dict[str, Any]) -> dict[str, Any]:
    """Build a renderer-independent plot description from a compact result."""

    if result.get("schema") != "simllm-deployment-frontier-result-v1":
        raise ValueError("unexpected deployment-frontier result schema")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for point in result["points"]:
        grouped.setdefault(point["configuration_id"], []).append(point)
    curves = []
    for configuration_id, points in grouped.items():
        points.sort(key=lambda point: point["batch_per_gpu"])
        curves.append(
            {
                "id": configuration_id,
                "label": points[0]["configuration_label"],
                "color": COLORS[configuration_id],
                "points": [
                    {
                        "batch_per_gpu": point["batch_per_gpu"],
                        "analytical_x": _coordinate(
                            point,
                            "analytical",
                            "x_tokens_per_second_per_request",
                        ),
                        "analytical_y": _coordinate(
                            point,
                            "analytical",
                            "y_tokens_per_second_per_gpu",
                        ),
                        "simulated_x": _coordinate(
                            point,
                            "simulated",
                            "x_tokens_per_second_per_request",
                        ),
                        "simulated_y": _coordinate(
                            point,
                            "simulated",
                            "y_tokens_per_second_per_gpu",
                        ),
                        "inter_node_attributed_ps": point["accounting"][
                            "inter_node_attributed_ps"
                        ],
                        "intra_node_attributed_ps": point["accounting"][
                            "intra_node_attributed_ps"
                        ],
                        "bottleneck": point["bottleneck"]["classification"],
                    }
                    for point in points
                ],
            }
        )
    contract = result["plot_contract"]
    if contract["x"]["scale"] != "log" or contract["y"]["scale"] != "log":
        raise ValueError("the frozen frontier axes must both be logarithmic")
    paired = result["published_context"]["paired"][0]
    y_only = result["published_context"]["y_only"][0]
    return {
        "curves": curves,
        "paired_marker": {
            "label": paired["label"],
            "x": paired["tokens_per_second_per_node"] / paired["batch_per_node"],
            "y": paired["tokens_per_second_per_node"] / paired["gpus_per_node"],
        },
        "y_only_anchor": {
            "label": y_only["label"],
            "y": y_only["tokens_per_second_per_node"] / y_only["gpus_per_node"],
        },
        "status": result["status"],
        "candidate_disclosure": result["intra_node_candidate_disclosure"],
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


def _draw_frontier(axis: Any, plot: dict[str, Any], *, annotate_batches: bool) -> None:
    for curve in plot["curves"]:
        points = curve["points"]
        analytical_x = [point["analytical_x"] for point in points]
        analytical_y = [point["analytical_y"] for point in points]
        simulated_x = [point["simulated_x"] for point in points]
        simulated_y = [point["simulated_y"] for point in points]
        axis.plot(
            analytical_x,
            analytical_y,
            color=curve["color"],
            linewidth=1.8,
            label=f"{curve['label']} analytical",
            zorder=3,
        )
        axis.scatter(
            simulated_x,
            simulated_y,
            color=curve["color"],
            edgecolor="white",
            linewidth=0.45,
            s=24,
            label=f"{curve['label']} roofline dots",
            zorder=5,
        )
        if annotate_batches:
            for index in (0, len(points) - 1):
                point = points[index]
                axis.annotate(
                    f"B={point['batch_per_gpu']}",
                    (point["simulated_x"], point["simulated_y"]),
                    xytext=(4, 3 if index == 0 else -9),
                    textcoords="offset points",
                    color=curve["color"],
                    fontsize=6.1,
                )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Per-request decode speed (token/s/request)")
    axis.set_ylabel("Output throughput per GPU (token/s/GPU)")
    axis.grid(True, which="both", color="#d5d9df", linewidth=0.5, alpha=0.8)


def render_deployment_figure(plot: dict[str, Any], output_stem: Path) -> tuple[Path, Path]:
    """Render the new deployment frontier in the frozen 7 by 4.33 contract."""

    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(7.0, 4.33))
    figure.subplots_adjust(left=0.115, right=0.975, bottom=0.29, top=0.78)
    _draw_frontier(axis, plot, annotate_batches=True)
    paired = plot["paired_marker"]
    axis.scatter(
        [paired["x"]],
        [paired["y"]],
        marker="D",
        facecolor="white",
        edgecolor="black",
        linewidth=0.9,
        s=40,
        label=paired["label"],
        zorder=7,
    )
    anchor = plot["y_only_anchor"]
    axis.axhline(
        anchor["y"],
        color="#444444",
        linestyle="--",
        linewidth=1.1,
        label=f"{anchor['label']} y-only anchor",
        zorder=2,
    )
    axis.set_title("DeepSeek decode deployment frontier")
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.2),
        mode="expand",
        ncol=2,
        borderaxespad=0,
        frameon=False,
        handlelength=2.3,
    )
    figure.text(
        0.115,
        0.12,
        "Analytical lines are floor-style step-time bounds: real and simulated points "
        "sit on or below them. Dots use RooflineProvider with kernel simulation off.\n"
        "Intra-node timing uses the cross-architecture A100 NVLink3 candidate; it is "
        "not H100 or B100 measurement evidence.",
        ha="left",
        va="bottom",
        fontsize=6.8,
    )
    return _save(figure, output_stem)


def render_bottleneck_figure(plot: dict[str, Any], output_stem: Path) -> tuple[Path, Path]:
    """Render curves plus the per-point stacked two-network attribution."""

    plt = _matplotlib()
    figure, (frontier_axis, stack_axis) = plt.subplots(
        2,
        1,
        figsize=(7.0, 6.0),
        gridspec_kw={"height_ratios": (1.22, 1.0)},
    )
    figure.subplots_adjust(left=0.105, right=0.98, bottom=0.21, top=0.87, hspace=0.43)
    _draw_frontier(frontier_axis, plot, annotate_batches=False)
    frontier_axis.set_title("Analytical frontier and roofline-simulation dots")
    frontier_axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01, 1.0, 0.2),
        mode="expand",
        ncol=3,
        borderaxespad=0,
        frameon=False,
        handlelength=1.8,
    )

    x_positions = []
    inter_values = []
    intra_values = []
    tick_labels = []
    colors = []
    classifications = []
    index = 0
    for curve in plot["curves"]:
        for point in curve["points"]:
            x_positions.append(index)
            inter_values.append(point["inter_node_attributed_ps"] / 1_000_000_000)
            intra_values.append(point["intra_node_attributed_ps"] / 1_000_000_000)
            tick_labels.append(f"{curve['id'].split('-')[0]}\nB{point['batch_per_gpu']}")
            colors.append(curve["color"])
            classifications.append(point["bottleneck"])
            index += 1
    stack_axis.bar(
        x_positions,
        inter_values,
        color="#8f3f71",
        width=0.72,
        label="Inter-node elapsed attribution",
    )
    stack_axis.bar(
        x_positions,
        intra_values,
        bottom=inter_values,
        color="#4b84c4",
        width=0.72,
        label="Intra-node elapsed attribution",
    )
    for x, inter, intra, classification, color in zip(
        x_positions,
        inter_values,
        intra_values,
        classifications,
        colors,
        strict=True,
    ):
        stack_axis.annotate(
            {"neither": "roof", "inter-node": "fabric", "intra-node": "intra"}.get(
                classification,
                "tie",
            ),
            (x, inter + intra),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
            color=color,
            fontsize=5.8,
        )
    stack_axis.set_xticks(x_positions, tick_labels, rotation=0)
    stack_axis.set_ylabel("Attributed step-time deviation (ms)")
    stack_axis.set_title("Frozen inter-then-intra elapsed attribution")
    stack_axis.grid(True, axis="y", color="#d5d9df", linewidth=0.5, alpha=0.8)
    stack_axis.legend(loc="upper left", frameon=False, ncol=2)
    figure.text(
        0.105,
        0.065,
        "Each stack telescopes exactly from the analytical floor to the simulated step. "
        "Raw off-critical network excess remains in the result table.\n"
        "The intra-node profile is an A100 NVLink3 candidate used across architecture, "
        "not H100 or B100 measurement evidence.",
        ha="left",
        va="bottom",
        fontsize=6.7,
    )
    return _save(figure, output_stem)


def _save(figure: Any, output_stem: Path) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches=None)
    figure.savefig(png_path, dpi=180, bbox_inches=None)
    figure.clear()
    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    plot = prepare_plot(result)
    deployment = render_deployment_figure(plot, args.output_dir / "deployment-frontier")
    bottleneck = render_bottleneck_figure(
        plot,
        args.output_dir / "two-network-bottleneck",
    )
    for path in (*deployment, *bottleneck):
        print(path.as_posix())


if __name__ == "__main__":
    main()
