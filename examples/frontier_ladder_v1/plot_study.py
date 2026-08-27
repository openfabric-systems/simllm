#!/usr/bin/env python3
"""Render the frozen three-rung frontier and mechanism envelope."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from simllm.deploy import (
    FrontierRung,
    frontier_ladder_record_from_json,
    ladder_pareto_front,
)

STUDY_DIR = Path(__file__).resolve().parent
RESULT_SCHEMA = "simllm-frontier-ladder-study-v1"
PLOT_SCHEMA = "simllm-frontier-ladder-plot-v1"
FIXED_TIMESTAMP = datetime(2026, 8, 28, tzinfo=UTC)

COLORS = {
    "b100-one-node-intra": "#2563a6",
    "h100-two-node-serialized": "#d17c0f",
    "h100-nine-node-incast": "#27845c",
}
FALLBACK_COLORS = ("#7656a3", "#8f3f71", "#4b84c4")
CURVE_STYLES = {
    "b100-one-node-intra": {
        "line_style": "-",
        "line_width": 1.35,
        "line_alpha": 0.82,
        "line_zorder": 3,
        "draw_order": 0,
    },
    "h100-nine-node-incast": {
        "line_style": "-",
        "line_width": 3.0,
        "line_alpha": 0.55,
        "line_zorder": 2,
        "draw_order": 1,
    },
    "h100-two-node-serialized": {
        "line_style": "--",
        "line_width": 1.4,
        "line_alpha": 1.0,
        "line_zorder": 4,
        "draw_order": 2,
    },
}
RUNG_STYLES = {
    FrontierRung.ESTIMATE.value: {
        "marker": "o",
        "size": 45,
        "face": "none",
        "edge": "configuration",
        "line_width": 1.05,
        "zorder": 5,
    },
    FrontierRung.LOGGOPSIM_IDEAL.value: {
        "marker": "^",
        "size": 28,
        "face": "white",
        "edge": "configuration",
        "line_width": 0.8,
        "zorder": 6,
    },
    FrontierRung.PACKET.value: {
        "marker": "o",
        "size": 12,
        "face": "configuration",
        "edge": "configuration",
        "line_width": 0.4,
        "zorder": 7,
    },
}


def prepare_plot_data(result: dict[str, Any]) -> dict[str, Any]:
    """Project the strict ladder record into renderer-independent data."""

    if result.get("schema") != RESULT_SCHEMA:
        raise ValueError(f"unexpected ladder result schema: {result.get('schema')!r}")
    record = frontier_ladder_record_from_json(result["ladder_record"])
    grouped: dict[str, list[Any]] = {}
    for point in record.points:
        grouped.setdefault(point.configuration_id, []).append(point)
    curves = []
    for curve_index, (configuration_id, points) in enumerate(grouped.items()):
        points.sort(key=lambda point: point.batch_per_gpu)
        curves.append(
            {
                "configuration_id": configuration_id,
                "configuration_label": points[0].configuration_label,
                "color": COLORS.get(
                    configuration_id,
                    FALLBACK_COLORS[curve_index % len(FALLBACK_COLORS)],
                ),
                **CURVE_STYLES.get(
                    configuration_id,
                    {
                        "line_style": "-",
                        "line_width": 1.35,
                        "line_alpha": 0.82,
                        "line_zorder": 3,
                        "draw_order": curve_index,
                    },
                ),
                "points": [
                    {
                        "batch_per_gpu": point.batch_per_gpu,
                        "rungs": {
                            rung.rung.value: {
                                "point_class": rung.point_class.value,
                                "authority": rung.provenance.authority,
                                "step_ps": rung.step_ps,
                                "fabric_leg_ps": rung.fabric_leg_ps,
                                "x": float(rung.x_tokens_per_second_per_request),
                                "y": float(rung.y_tokens_per_second_per_gpu),
                            }
                            for rung in point.rungs
                        },
                    }
                    for point in points
                ],
            }
        )
    packet_front = ladder_pareto_front(record, FrontierRung.PACKET)
    paired = next((anchor for anchor in record.anchors if not anchor.y_only), None)
    y_only = next((anchor for anchor in record.anchors if anchor.y_only), None)
    envelope = [
        {
            "family": row["family"],
            "batch_per_gpu": row["batch_per_gpu"],
            "quotient": row["quotient"]["decimal"],
            "numerator": row["quotient"]["numerator"],
            "denominator": row["quotient"]["denominator"],
        }
        for row in result["fabric_leg_envelope"]
    ]
    return {
        "schema": PLOT_SCHEMA,
        "axes": {
            "x": {
                "quantity": "per-request decode speed",
                "units": "token/s/request",
                "scale": "log",
                "direction": "rightward",
            },
            "y": {
                "quantity": "output throughput per GPU",
                "units": "token/s/GPU",
                "scale": "log",
                "direction": "upward",
            },
            "optimal_corner": "upper-right",
        },
        "rung_styles": RUNG_STYLES,
        "curves": curves,
        "packet_pareto": [
            {
                "configuration_id": point.configuration_id,
                "batch_per_gpu": point.batch_per_gpu,
                "x": float(
                    point.rung(
                        FrontierRung.PACKET
                    ).x_tokens_per_second_per_request
                ),
                "y": float(
                    point.rung(FrontierRung.PACKET).y_tokens_per_second_per_gpu
                ),
            }
            for point in packet_front
        ],
        "paired_anchor": (
            None
            if paired is None
            else {
                "label": paired.label,
                "x": float(paired.x_tokens_per_second_per_request),
                "y": float(paired.y_tokens_per_second_per_gpu),
            }
        ),
        "y_only_anchor": (
            None
            if y_only is None
            else {
                "label": y_only.label,
                "y": float(y_only.y_tokens_per_second_per_gpu),
            }
        ),
        "envelope": envelope,
    }


def _matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.0,
            "legend.fontsize": 6.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _frontier_panel(axis: Any, plot: dict[str, Any]) -> None:
    from matplotlib.lines import Line2D

    curves = sorted(plot["curves"], key=lambda curve: curve["draw_order"])
    for curve in curves:
        estimate = [point["rungs"][FrontierRung.ESTIMATE.value] for point in curve["points"]]
        axis.plot(
            [point["x"] for point in estimate],
            [point["y"] for point in estimate],
            color=curve["color"],
            linestyle=curve["line_style"],
            linewidth=curve["line_width"],
            alpha=curve["line_alpha"],
            zorder=curve["line_zorder"],
        )
        for rung_id, style in plot["rung_styles"].items():
            values = [point["rungs"][rung_id] for point in curve["points"]]
            face = curve["color"] if style["face"] == "configuration" else style["face"]
            edge = curve["color"] if style["edge"] == "configuration" else style["edge"]
            axis.scatter(
                [point["x"] for point in values],
                [point["y"] for point in values],
                marker=style["marker"],
                s=style["size"],
                facecolors=face,
                edgecolors=edge,
                linewidths=style["line_width"],
                zorder=style["zorder"],
            )
        annotated_points = (
            (curve["points"][0], curve["points"][-1])
            if curve["configuration_id"] == "b100-one-node-intra"
            else ()
        )
        for point in annotated_points:
            packet = point["rungs"][FrontierRung.PACKET.value]
            axis.annotate(
                f"B={point['batch_per_gpu']}",
                (packet["x"], packet["y"]),
                xytext=(4, 4 if point["batch_per_gpu"] == 1 else -8),
                textcoords="offset points",
                color=curve["color"],
                fontsize=5.8,
            )

    front = sorted(plot["packet_pareto"], key=lambda point: point["batch_per_gpu"])
    axis.plot(
        [point["x"] for point in front],
        [point["y"] for point in front],
        color="#111111",
        linewidth=2.5,
        zorder=4,
    )
    axis.scatter(
        [point["x"] for point in front],
        [point["y"] for point in front],
        facecolors="none",
        edgecolors="#111111",
        linewidths=1.2,
        s=68,
        zorder=8,
    )

    paired = plot["paired_anchor"]
    if paired is not None:
        axis.scatter(
            [paired["x"]],
            [paired["y"]],
            marker="D",
            facecolor="white",
            edgecolor="#111111",
            linewidth=0.9,
            s=36,
            zorder=9,
        )
    y_only = plot["y_only_anchor"]
    if y_only is not None:
        axis.axhline(
            y_only["y"],
            color="#4a4a4a",
            linestyle="--",
            linewidth=1.0,
            zorder=2,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Per-request decode speed (token/s/request)")
    axis.set_ylabel("Output throughput per GPU (token/s/GPU)")
    axis.set_title("a. Three-rung deployment frontier", loc="left")
    axis.grid(True, which="both", color="#d5d9df", linewidth=0.45, alpha=0.75)
    configuration_labels = {
        "b100-one-node-intra": "B100 1N",
        "h100-two-node-serialized": "H100 2N",
        "h100-nine-node-incast": "H100 9N",
    }
    axis.legend(
        [
            Line2D(
                [0],
                [0],
                color=curve["color"],
                linestyle=curve["line_style"],
                linewidth=curve["line_width"],
            )
            for curve in plot["curves"]
        ],
        [
            configuration_labels.get(
                curve["configuration_id"],
                curve["configuration_label"],
            )
            for curve in plot["curves"]
        ],
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        handlelength=1.7,
    )


def _envelope_panel(axis: Any, plot: dict[str, Any]) -> None:
    styles = {
        "M-1": ("Serialized packet / ideal", "#d17c0f", "o"),
        "M-2": ("Incast packet / ideal", "#8f3f71", "s"),
        "M-3": ("Isolated packet / ideal", "#27845c", "^"),
    }
    for family, (label, color, marker) in styles.items():
        rows = sorted(
            (row for row in plot["envelope"] if row["family"] == family),
            key=lambda row: row["batch_per_gpu"],
        )
        axis.plot(
            [row["batch_per_gpu"] for row in rows],
            [row["quotient"] for row in rows],
            color=color,
            marker=marker,
            markersize=4.0,
            linewidth=1.4,
            label=label,
            zorder=4,
        )
    m2 = next(
        row
        for row in plot["envelope"]
        if row["family"] == "M-2" and row["batch_per_gpu"] == 32
    )
    axis.annotate(
        f"M-2 = {m2['quotient']:.2f}x",
        (m2["batch_per_gpu"], m2["quotient"]),
        xytext=(-10, -18),
        textcoords="offset points",
        ha="right",
        color="#8f3f71",
        fontsize=7.0,
        arrowprops={"arrowstyle": "-", "color": "#8f3f71", "linewidth": 0.6},
    )
    axis.axhline(1.0, color="#777777", linestyle=":", linewidth=0.8, zorder=2)
    axis.set_xscale("log", base=2)
    batches = [1, 2, 4, 8, 16, 32]
    axis.set_xticks(batches, [str(batch) for batch in batches])
    axis.set_ylim(0.75, 8.65)
    axis.set_xlabel("Batch per GPU")
    axis.set_ylabel("Packet fabric leg / ideal fabric leg")
    axis.set_title("b. Mechanism envelope", loc="left")
    axis.grid(True, axis="y", color="#d5d9df", linewidth=0.45, alpha=0.75)
    axis.legend(
        loc="center right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        handlelength=2.3,
    )


def render(plot: dict[str, Any], output_stem: Path) -> tuple[Path, Path]:
    """Render one two-column figure through the Agg backend."""

    if plot.get("schema") != PLOT_SCHEMA:
        raise ValueError("unexpected frontier ladder plot schema")
    plt = _matplotlib()
    from matplotlib.lines import Line2D

    figure, (frontier_axis, envelope_axis) = plt.subplots(
        1,
        2,
        figsize=(7.0, 4.6),
        gridspec_kw={"width_ratios": (1.22, 1.0)},
    )
    figure.subplots_adjust(left=0.095, right=0.985, bottom=0.25, top=0.78, wspace=0.34)
    _frontier_panel(frontier_axis, plot)
    _envelope_panel(envelope_axis, plot)

    semantic_handles = (
        Line2D([0], [0], color="#555555", linewidth=1.4),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=5.5,
            markerfacecolor="none",
            markeredgecolor="#555555",
            linestyle="none",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            markersize=5.0,
            markerfacecolor="white",
            markeredgecolor="#555555",
            linestyle="none",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=3.5,
            markerfacecolor="#555555",
            markeredgecolor="#555555",
            linestyle="none",
        ),
        Line2D([0], [0], color="#111111", linewidth=2.5),
        Line2D(
            [0],
            [0],
            marker="D",
            markersize=4.5,
            markerfacecolor="white",
            markeredgecolor="#111111",
            linestyle="none",
        ),
    )
    figure.suptitle("Deployment frontier ladder and ideal-level validity envelope", y=0.955)
    figure.legend(
        semantic_handles,
        ("Closed-form line", "ESTIMATE", "Ideal rung", "Packet SIMULATED", "Packet Pareto", "Published paired"),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=6,
        frameon=False,
        columnspacing=1.15,
        handlelength=1.8,
    )
    figure.text(
        0.095,
        0.055,
        "The ideal level stays within about 1.6% for serialized point-to-point traffic.\n"
        "It is about 8x optimistic for eight-into-one incast because it does not "
        "serialize the shared receiver ingress.\n"
        "At step level the H100 kernel masks both fabric differences; only the pinned "
        "B100 batch-32 intra-node packet excess moves a point.\n"
        "H100 2N and 9N coincide at step level in the pinned record; dashed 2N "
        "over the wider 9N line makes both visible.",
        ha="left",
        va="bottom",
        fontsize=6.45,
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
        metadata={"Software": "simllm frontier_ladder_v1"},
    )
    plt.close(figure)
    return pdf, png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=STUDY_DIR / "result.json")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=STUDY_DIR / "figures" / "frontier-ladder",
    )
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    pdf, png = render(prepare_plot_data(result), args.output_stem)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
