#!/usr/bin/env python3
"""Render the fourth CORE-54 flagship figure with the scored MTP anchor."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from curve_tools import as_fraction, load_json
from plot_flagship_run3 import prepare_flagship_plot as prepare_run3_plot

BLUE = "#1f5aa6"
RED = "#bd2c2c"
GOLD = "#c6961a"
GREEN = "#27845c"

LAYER_STYLES = (
    ("physics_only", "Physics only", BLUE, ":", 1.4),
    ("physics_plus_boundary", "Physics + boundary", RED, "-", 1.5),
    (
        "physics_plus_boundary_plus_attenuation",
        "Physics + boundary + attenuation",
        GOLD,
        "--",
        1.8,
    ),
)


def _number(value: object, name: str) -> float:
    return float(as_fraction(value, name))


def _percent(value: object, name: str) -> float:
    return 100 * _number(value, name)


def prepare_flagship_plot(result: dict[str, Any]) -> dict[str, Any]:
    """Build a matplotlib-independent three-panel plot description."""

    run3 = result["run3_carry_forward"]
    base = prepare_run3_plot(
        {
            "curves": run3["curves"],
            "anchor_predictions": run3["anchor_predictions"],
            "held_out_score": run3["held_out_score"],
            "decode_calibration_miss": run3["decode_calibration_miss"],
            "second_legend": run3["second_legend"],
        }
    )
    score = result["mtp_score"]
    layers = []
    for layer_name, label, color, style, width in LAYER_STYLES:
        comparison = score["layers"][layer_name]
        prediction = comparison["prediction"]
        layers.append(
            {
                "name": layer_name,
                "label": label,
                "color": color,
                "style": style,
                "width": width,
                "point": _number(prediction["point"], f"mtp.{layer_name}.point"),
                "lower": _number(prediction["lower"], f"mtp.{layer_name}.lower"),
                "upper": _number(prediction["upper"], f"mtp.{layer_name}.upper"),
                "signed_error_percent": _percent(
                    comparison["signed_relative_error"],
                    f"mtp.{layer_name}.error",
                ),
                "status": comparison["status"],
            }
        )
    base.update(
        {
            "mtp": {
                "published": _number(score["published"], "mtp.published"),
                "layers": layers,
                "status": score["status"],
                "attenuation_applied": score["attenuation_applied"],
                "attempt_count": score["score_attempt_count"],
            },
            "combined_verdict": result["verdict"],
            "frontier_status": result["deployment_frontier"]["status"],
        }
    )
    return base


def _render_curve_panel(axis: Any, plot: dict[str, Any]) -> None:
    from matplotlib.lines import Line2D

    handles = []
    all_x = []
    all_y = []
    for curve in plot["curves"]:
        points = curve["points"]
        x = [point["x"] for point in points]
        y = [point["y"] for point in points]
        x_lower = [point["x_lower"] for point in points]
        x_upper = [point["x_upper"] for point in points]
        y_lower = [point["y_lower"] for point in points]
        y_upper = [point["y_upper"] for point in points]
        is_what_if = curve["evidence_class"] == "declared-node-linear-what-if"
        color = GREEN if is_what_if else BLUE
        style = "--" if is_what_if else "-"
        axis.fill_between(x, y_lower, y_upper, color=color, alpha=0.13, linewidth=0)
        handles.append(
            axis.plot(
                x,
                y,
                color=color,
                linestyle=style,
                marker="o",
                markersize=3.2,
                linewidth=1.5,
                label=curve["label"],
                zorder=4,
            )[0]
        )
        axis.errorbar(
            x,
            y,
            xerr=(
                [point - lower for point, lower in zip(x, x_lower, strict=True)],
                [upper - point for point, upper in zip(x, x_upper, strict=True)],
            ),
            fmt="none",
            ecolor=color,
            elinewidth=0.7,
            capsize=1.4,
            alpha=0.65,
            zorder=3,
        )
        all_x.extend(x_lower + x_upper)
        all_y.extend(y_lower + y_upper)

    disclosure = plot["standard_decode_disclosure"]
    axis.scatter(
        [disclosure["x"]],
        [disclosure["y"]],
        marker="D",
        s=32,
        facecolors="white",
        edgecolors="#111111",
        linewidths=1.0,
        zorder=6,
    )
    handles.append(
        Line2D(
            [0],
            [0],
            color="#111111",
            marker="D",
            markerfacecolor="white",
            linestyle="none",
            label=disclosure["label"],
        )
    )
    axis.axvline(plot["h800_x"], color="#555555", linestyle=":", linewidth=1.1)
    handles.append(
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=":",
            linewidth=1.1,
            label="DeepSeek H800 production",
        )
    )
    all_x.append(disclosure["x"])
    all_y.append(disclosure["y"])
    x_span = max(all_x) - min(all_x)
    axis.set_xlim(max(0.0, min(all_x) - 0.05 * x_span), max(all_x) + 0.05 * x_span)
    axis.set_ylim(min(all_y) / 1.22, max(all_y) * 1.22)
    axis.set_yscale("log")
    axis.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    axis.set_xlabel("Aggregated output throughput (tokens/s) →")
    axis.set_ylabel("Inverse per-token request delay (tokens/s) ↑")
    axis.set_title("a  Ordered deployment curves", loc="left", fontweight="bold")
    axis.legend(
        handles=handles,
        loc="upper right",
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="#cccccc",
    )


def _render_prefill_panel(axis: Any, plot: dict[str, Any]) -> None:
    prompt = [row["prompt_k"] for row in plot["prefill"]]
    published = [row["published"] for row in plot["prefill"]]
    for layer_name, label, color, style, width in LAYER_STYLES:
        point = [row["layers"][layer_name]["point"] for row in plot["prefill"]]
        lower = [row["layers"][layer_name]["lower"] for row in plot["prefill"]]
        upper = [row["layers"][layer_name]["upper"] for row in plot["prefill"]]
        axis.fill_between(prompt, lower, upper, color=color, alpha=0.10, linewidth=0)
        axis.plot(
            prompt,
            point,
            color=color,
            linestyle=style,
            marker="o",
            markersize=3.0,
            linewidth=width,
            label=label,
        )
    axis.scatter(
        prompt,
        published,
        marker="D",
        s=32,
        facecolors="white",
        edgecolors="#111111",
        linewidths=1.0,
        label="Published SGLang",
        zorder=6,
    )
    for x, value in zip(prompt[1:], published[1:], strict=True):
        axis.vlines(x, value * 0.95, value * 1.05, color="#777777", linewidth=5, alpha=0.22)
    axis.set_xticks(prompt, ["1K\ncal", "2K\nheld out", "4K\nheld out"])
    axis.set_xlabel("Input length (tokens)")
    axis.set_ylabel("Prefill throughput/node (tokens/s)")
    axis.set_title("b  Run-3 prefill rows, unchanged", loc="left", fontweight="bold")
    axis.legend(
        loc="lower left",
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="#cccccc",
    )


def _render_mtp_panel(axis: Any, plot: dict[str, Any]) -> None:
    mtp = plot["mtp"]
    published = mtp["published"]
    axis.axhspan(published * 0.95, published * 1.05, color="#777777", alpha=0.18)
    axis.axhline(
        published,
        color="#111111",
        linestyle="-",
        linewidth=1.2,
        label="Published SGLang",
    )
    for index, layer in enumerate(mtp["layers"]):
        axis.errorbar(
            [index],
            [layer["point"]],
            yerr=(
                [layer["point"] - layer["lower"]],
                [layer["upper"] - layer["point"]],
            ),
            color=layer["color"],
            marker="o",
            markersize=5.0,
            linestyle="none",
            capsize=2.0,
            label=layer["label"],
            zorder=5,
        )
    axis.set_xlim(-0.45, 2.45)
    axis.set_ylim(min(layer["lower"] for layer in mtp["layers"]) * 0.82, published * 1.12)
    axis.set_xticks([0, 1, 2], ["Physics", "+ boundary", "+ attenuation"])
    axis.set_ylabel("MTP decode throughput/node (tokens/s)")
    axis.set_title("c  MTP held-out score", loc="left", fontweight="bold")
    axis.text(
        0.5,
        0.06,
        f"All layers {mtp['layers'][0]['signed_error_percent']:.2f}% low\n"
        "zero-width bands, no attenuation",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=RED,
        fontweight="bold",
    )
    axis.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="#cccccc",
    )


def render_flagship_figure(
    plot: dict[str, Any],
    output_stem: Path,
) -> tuple[Path, Path]:
    """Render the fourth-run publication PDF and PNG."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.labelsize": 7.6,
            "axes.titlesize": 8.5,
            "legend.fontsize": 5.8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(7.0, 6.7))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.25, 1.0))
    curve_axis = figure.add_subplot(grid[0, :])
    prefill_axis = figure.add_subplot(grid[1, 0])
    mtp_axis = figure.add_subplot(grid[1, 1])
    figure.subplots_adjust(
        left=0.095,
        right=0.975,
        bottom=0.225,
        top=0.900,
        hspace=0.47,
        wspace=0.31,
    )
    _render_curve_panel(curve_axis, plot)
    _render_prefill_panel(prefill_axis, plot)
    _render_mtp_panel(mtp_axis, plot)
    for axis in (curve_axis, prefill_axis, mtp_axis):
        axis.grid(True, color="#dddddd", linewidth=0.5, alpha=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    mtp = plot["mtp"]
    figure.suptitle(
        "DeepSeek-V3 fourth scored run: MTP "
        f"{mtp['status']} ({abs(mtp['layers'][0]['signed_error_percent']):.2f}% low)\n"
        "Run-3 2K and 4K prefill PASS rows carried forward unchanged",
        fontsize=8.6,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.118,
        "MTP arithmetic: 128 requests/node × 2 emitted tokens/request; "
        "2.033951 ms × 61/4 = 31.01775275 ms; prediction 8,253.338 tokens/s/node.",
        ha="center",
        va="bottom",
        fontsize=6.15,
        color="#333333",
    )
    figure.text(
        0.5,
        0.073,
        "Scored once against 17,373 tokens/s/node at batch 128 and KV 4,000. "
        "No admissible EP72 decode attenuation; COMP-74 propagation remains open.",
        ha="center",
        va="bottom",
        fontsize=6.1,
        color="#333333",
    )
    figure.text(
        0.5,
        0.028,
        "Prefill and decode are separate experiments. The frontier figure stays "
        "byte-locked because its v2 contract has no MTP marker slot.",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color="#444444",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=240)
    plt.close(figure)
    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot = prepare_flagship_plot(load_json(args.result))
    pdf, png = render_flagship_figure(plot, args.output_stem)
    print(f"wrote {pdf.as_posix()} and {png.as_posix()}")


if __name__ == "__main__":
    main()
