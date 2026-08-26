"""Render the CORE-54 flagship figure contract from curve records and anchors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from curve_tools import as_fraction, load_json

PALETTE = ("#2a78d6", "#eb6834", "#1baf7a")
LINE_STYLES = ("-", "--", "-.", ":")


def _float_fraction(value: object, name: str) -> float:
    return float(as_fraction(value, name))


def prepare_plot_data(
    result: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    """Build a matplotlib-independent plot description."""

    curves = []
    for index, curve in enumerate(result["curves"]):
        if curve.get("schema") != "simllm-deployment-curve-v1":
            raise ValueError(f"curves[{index}] uses an unexpected schema")
        points = []
        for point_index, point in enumerate(curve["points"]):
            interval = point["uncertainty"]
            x_interval = interval["aggregated_output_throughput_tokens_per_second"]
            y_interval = interval["inverse_per_token_request_delay_tokens_per_second"]
            points.append(
                {
                    "offered_load": _float_fraction(
                        point["offered_load_requests_per_second"],
                        f"curves[{index}].points[{point_index}].offered_load",
                    ),
                    "x": _float_fraction(x_interval["point"], "x.point"),
                    "x_lower": _float_fraction(x_interval["lower"], "x.lower"),
                    "x_upper": _float_fraction(x_interval["upper"], "x.upper"),
                    "y": _float_fraction(y_interval["point"], "y.point"),
                    "y_lower": _float_fraction(y_interval["lower"], "y.lower"),
                    "y_upper": _float_fraction(y_interval["upper"], "y.upper"),
                }
            )
        curves.append(
            {
                "configuration_id": curve["configuration_id"],
                "label": curve["configuration_label"],
                "points": points,
            }
        )

    anchors = {anchor["id"]: anchor for anchor in freeze["anchors"]}
    latency_ms = anchors["sglang_inter_token_latency"]["value"]
    context_y = 1000.0 / latency_ms
    disclosure_points = []
    for anchor_id, label in (
        ("sglang_decode_standard", "SGLang standard decode"),
        ("sglang_decode_simulated_mtp", "SGLang simulated MTP"),
    ):
        anchor = anchors[anchor_id]
        disclosure_points.append(
            {
                "anchor_id": anchor_id,
                "label": f"{label}, shared ~100 ms context",
                "x": float(anchor["aggregated_value"]),
                "y": context_y,
                "role": anchor["role"],
                "paired_measurement": False,
            }
        )
    deepseek = anchors["deepseek_production_decode"]
    return {
        "classification": result["classification"],
        "curves": curves,
        "disclosure_points": disclosure_points,
        "vertical_references": [
            {
                "anchor_id": deepseek["id"],
                "label": "DeepSeek H800 decode average, delay undisclosed",
                "x": float(deepseek["aggregated_value"]),
            }
        ],
        "axis": freeze["axis_contract"],
    }


def render_figure(
    plot_data: dict[str, Any],
    output_stem: Path,
) -> tuple[Path, Path]:
    """Render publication-sized PDF and PNG outputs."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(7.0, 4.33), constrained_layout=True)
    curve_handles = []
    all_x = []
    all_y = []
    for index, curve in enumerate(plot_data["curves"]):
        color = PALETTE[index % len(PALETTE)]
        style = LINE_STYLES[(index // len(PALETTE)) % len(LINE_STYLES)]
        points = curve["points"]
        x = [point["x"] for point in points]
        y = [point["y"] for point in points]
        x_lower = [point["x_lower"] for point in points]
        x_upper = [point["x_upper"] for point in points]
        y_lower = [point["y_lower"] for point in points]
        y_upper = [point["y_upper"] for point in points]
        axis.fill_between(x, y_lower, y_upper, color=color, alpha=0.16, linewidth=0)
        line = axis.plot(
            x,
            y,
            linestyle=style,
            color=color,
            marker="o",
            markersize=4.2,
            linewidth=1.8,
            label=curve["label"],
            zorder=4,
        )[0]
        axis.errorbar(
            x,
            y,
            xerr=(
                [point - lower for point, lower in zip(x, x_lower, strict=True)],
                [upper - point for point, upper in zip(x, x_upper, strict=True)],
            ),
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            capsize=2,
            alpha=0.65,
            zorder=3,
        )
        curve_handles.append(line)
        all_x.extend(x_lower + x_upper)
        all_y.extend(y_lower + y_upper)

    disclosure_handles = []
    disclosure_markers = ("D", "s")
    for index, anchor in enumerate(plot_data["disclosure_points"]):
        handle = axis.scatter(
            [anchor["x"]],
            [anchor["y"]],
            s=42,
            marker=disclosure_markers[index % len(disclosure_markers)],
            facecolors="white",
            edgecolors="#111111",
            linewidths=1.2,
            zorder=6,
            label=anchor["label"],
        )
        disclosure_handles.append(handle)
        all_x.append(anchor["x"])
        all_y.append(anchor["y"])
    for reference in plot_data["vertical_references"]:
        axis.axvline(
            reference["x"],
            color="#555555",
            linestyle=":",
            linewidth=1.2,
            zorder=1,
        )
        disclosure_handles.append(
            Line2D(
                [0],
                [0],
                color="#555555",
                linestyle=":",
                linewidth=1.2,
                label=reference["label"],
            )
        )
        all_x.append(reference["x"])

    axis.set_xlabel("Aggregated output throughput (tokens/s), rightward is better")
    axis.set_ylabel("Inverse per-token request delay (tokens/s), upward is better")
    title = "DeepSeek-V3 deployment curve scaffold"
    if plot_data["classification"] == "dry-run":
        title += " (DRY RUN)"
    axis.set_title(title, loc="left", fontweight="bold")
    axis.text(
        0.985,
        0.965,
        "UPPER-RIGHT\nOPTIMAL",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#333333",
        fontweight="bold",
    )
    if plot_data["classification"] == "dry-run":
        axis.text(
            0.5,
            0.52,
            "DRY RUN\nGranite + bootstrap pricing",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=15,
            color="#777777",
            alpha=0.22,
            fontweight="bold",
            rotation=10,
            zorder=0,
        )
    axis.grid(True, color="#d8d8d8", linewidth=0.55, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    if all_x:
        x_span = max(all_x) - min(all_x)
        x_pad = max(0.06 * x_span, 1.0)
        axis.set_xlim(max(0.0, min(all_x) - x_pad), max(all_x) + x_pad)
    if all_y:
        y_span = max(all_y) - min(all_y)
        y_pad = max(0.08 * y_span, 0.1)
        axis.set_ylim(max(0.0, min(all_y) - y_pad), max(all_y) + y_pad)

    configurations = axis.legend(
        handles=curve_handles,
        title="Simulated configurations",
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="#cccccc",
    )
    axis.add_artist(configurations)
    axis.legend(
        handles=disclosure_handles,
        title="Published disclosure context",
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="#cccccc",
    )
    axis.text(
        0.01,
        0.01,
        "SGLang markers share the approximate 100 ms headline latency; "
        "they are not paired score rows.",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color="#444444",
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=220)
    plt.close(figure)
    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--anchors", required=True, type=Path)
    parser.add_argument("--output-stem", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_data = prepare_plot_data(load_json(args.result), load_json(args.anchors))
    pdf_path, png_path = render_figure(plot_data, args.output_stem)
    print(f"wrote {pdf_path.as_posix()} and {png_path.as_posix()}")


if __name__ == "__main__":
    main()
