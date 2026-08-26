#!/usr/bin/env python3
"""Render the third CORE-54 publication figure from its scored result."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from curve_tools import as_fraction, load_json

BLUE = "#1f5aa6"
RED = "#bd2c2c"
GOLD = "#c6961a"
GREEN = "#27845c"


def _number(value: object, name: str) -> float:
    return float(as_fraction(value, name))


def _percent(value: object, name: str) -> float:
    return 100 * _number(value, name)


def prepare_flagship_plot(result: dict[str, Any]) -> dict[str, Any]:
    """Build a matplotlib-independent two-panel plot description."""

    curves = []
    for curve in result["curves"]:
        points = []
        for point in curve["points"]:
            interval = point["uncertainty"]
            x = interval["aggregated_output_throughput_tokens_per_second"]
            y = interval["inverse_per_token_request_delay_tokens_per_second"]
            points.append(
                {
                    "x": _number(x["point"], "curve.x.point"),
                    "x_lower": _number(x["lower"], "curve.x.lower"),
                    "x_upper": _number(x["upper"], "curve.x.upper"),
                    "y": _number(y["point"], "curve.y.point"),
                    "y_lower": _number(y["lower"], "curve.y.lower"),
                    "y_upper": _number(y["upper"], "curve.y.upper"),
                }
            )
        curves.append(
            {
                "id": curve["configuration_id"],
                "label": (
                    "PLACE-5 16P+40D what-if"
                    if curve["evidence_class"] == "declared-node-linear-what-if"
                    else "simLLM H100 EP72 decode"
                ),
                "evidence_class": curve["evidence_class"],
                "points": points,
            }
        )

    prediction_map = {
        row["anchor_id"]: row
        for row in result["anchor_predictions"]
        if row["status"] == "PREDICTED"
    }
    prefill = []
    for prompt_k, anchor_id in (
        (1, "sglang_prefill_1k"),
        (2, "sglang_prefill_2k"),
        (4, "sglang_prefill_4k"),
    ):
        row = prediction_map[anchor_id]
        layers = {}
        for layer_name, comparison in row["layers"].items():
            prediction = comparison["prediction"]
            layers[layer_name] = {
                "point": _number(prediction["point"], f"{anchor_id}.{layer_name}.point"),
                "lower": _number(prediction["lower"], f"{anchor_id}.{layer_name}.lower"),
                "upper": _number(prediction["upper"], f"{anchor_id}.{layer_name}.upper"),
                "signed_error_percent": _percent(
                    comparison["signed_relative_error"],
                    f"{anchor_id}.{layer_name}.error",
                ),
            }
        prefill.append(
            {
                "prompt_k": prompt_k,
                "anchor_id": anchor_id,
                "published": _number(row["published"], f"{anchor_id}.published"),
                "layers": layers,
            }
        )

    score = result["held_out_score"]
    decode = result["decode_calibration_miss"]
    standard_decode = prediction_map["sglang_decode_standard"]
    standard_published = _number(
        standard_decode["published"], "standard_decode.published"
    )
    declared_ms = decode["declared_step_ps"] / 1_000_000_000
    implied_ms = _number(
        decode["published_throughput_implied_step_ps"], "decode.implied_step"
    ) / 1_000_000_000
    return {
        "curves": curves,
        "prefill": prefill,
        "standard_decode_disclosure": {
            "label": "Published SGLang decode",
            "x": standard_published * 9,
            "y": 10.0,
        },
        "h800_x": float(
            result["second_legend"]["deepseek_h800_production_decode_tokens_per_second"]
        ),
        "verdict": score["status"],
        "unattenuated_verdict": score["unattenuated_status"],
        "scope": score["scope"],
        "maximum_attenuated_error_percent": _percent(
            score["maximum_attenuated_absolute_relative_error"],
            "score.max_attenuated",
        ),
        "maximum_unattenuated_error_percent": _percent(
            score["maximum_unattenuated_absolute_relative_error"],
            "score.max_unattenuated",
        ),
        "mtp_dependency": score["blocked_rows"][0]["dependency"],
        "decode_error_percent": _percent(
            decode["absolute_relative_error"], "decode.error"
        ),
        "decode_declared_step_ms": declared_ms,
        "decode_implied_step_ms": implied_ms,
        "attenuation_label": (
            "expert-balance correction a=0.90835, two-SE uncertainty"
        ),
    }


def render_flagship_figure(
    plot: dict[str, Any],
    output_stem: Path,
) -> tuple[Path, Path]:
    """Render one publication-sized PDF and PNG from the third scored data."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.8,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.0,
            "legend.fontsize": 6.2,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, (curve_axis, prefill_axis) = plt.subplots(
        1,
        2,
        figsize=(7.0, 4.33),
        gridspec_kw={"width_ratios": (1.45, 1.15)},
        constrained_layout=False,
    )
    figure.subplots_adjust(left=0.095, right=0.975, bottom=0.245, top=0.76, wspace=0.29)

    simulated_handles = []
    context_handles = []
    all_x = []
    all_y = []
    for index, curve in enumerate(plot["curves"]):
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
        curve_axis.fill_between(x, y_lower, y_upper, color=color, alpha=0.13, linewidth=0)
        handle = curve_axis.plot(
            x,
            y,
            color=color,
            linestyle=style,
            marker="o",
            markersize=3.4,
            linewidth=1.6,
            label=curve["label"],
            zorder=4,
        )[0]
        curve_axis.errorbar(
            x,
            y,
            xerr=(
                [point - lower for point, lower in zip(x, x_lower, strict=True)],
                [upper - point for point, upper in zip(x, x_upper, strict=True)],
            ),
            fmt="none",
            ecolor=color,
            elinewidth=0.7,
            capsize=1.5,
            alpha=0.65,
            zorder=3,
        )
        (context_handles if is_what_if else simulated_handles).append(handle)
        all_x.extend(x_lower + x_upper)
        all_y.extend(y_lower + y_upper)

    disclosure = plot["standard_decode_disclosure"]
    curve_axis.scatter(
        [disclosure["x"]],
        [disclosure["y"]],
        marker="D",
        s=34,
        facecolors="white",
        edgecolors="#111111",
        linewidths=1.0,
        zorder=6,
    )
    all_x.append(disclosure["x"])
    all_y.append(disclosure["y"])
    simulated_handles.append(
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
    curve_axis.axvline(plot["h800_x"], color="#555555", linestyle=":", linewidth=1.1)
    context_handles.insert(
        0,
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=":",
            linewidth=1.1,
            label="DeepSeek H800 production",
        ),
    )
    curve_axis.set_xlabel("Aggregated output throughput (tokens/s) →")
    curve_axis.set_ylabel("Inverse per-token request delay (tokens/s) ↑")
    curve_axis.set_yscale("log")
    curve_axis.set_title("a  Ordered deployment curves", loc="left", fontweight="bold")
    curve_axis.grid(True, color="#dddddd", linewidth=0.5, alpha=0.7)
    curve_axis.spines["top"].set_visible(False)
    curve_axis.spines["right"].set_visible(False)
    curve_axis.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    x_span = max(all_x) - min(all_x)
    curve_axis.set_xlim(max(0.0, min(all_x) - 0.06 * x_span), max(all_x) + 0.06 * x_span)
    curve_axis.set_ylim(min(all_y) / 1.25, max(all_y) * 1.25)
    first_legend = figure.legend(
        handles=simulated_handles,
        loc="center",
        bbox_to_anchor=(0.30, 0.835),
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="#cccccc",
    )
    figure.add_artist(first_legend)
    figure.legend(
        handles=context_handles,
        loc="center",
        bbox_to_anchor=(0.74, 0.835),
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="#cccccc",
    )

    prompt = [row["prompt_k"] for row in plot["prefill"]]
    published = [row["published"] for row in plot["prefill"]]
    layer_styles = (
        ("physics_only", "Physics only", BLUE, ":", 1.4, 0.08, 5),
        (
            "physics_plus_boundary",
            "Physics + boundary",
            RED,
            "-",
            1.5,
            0.10,
            4,
        ),
        (
            "physics_plus_boundary_plus_attenuation",
            "Scored attenuated",
            GOLD,
            "--",
            1.8,
            0.13,
            6,
        ),
    )
    for layer_name, label, color, style, width, alpha, zorder in layer_styles:
        point = [row["layers"][layer_name]["point"] for row in plot["prefill"]]
        lower = [row["layers"][layer_name]["lower"] for row in plot["prefill"]]
        upper = [row["layers"][layer_name]["upper"] for row in plot["prefill"]]
        prefill_axis.fill_between(prompt, lower, upper, color=color, alpha=alpha, linewidth=0)
        prefill_axis.plot(
            prompt,
            point,
            color=color,
            linestyle=style,
            marker="o",
            markersize=3.2,
            linewidth=width,
            label=label,
            zorder=zorder,
        )
    prefill_axis.scatter(
        prompt,
        published,
        marker="D",
        s=36,
        facecolors="white",
        edgecolors="#111111",
        linewidths=1.0,
        label="Published SGLang",
        zorder=6,
    )
    for x, value in zip(prompt[1:], published[1:], strict=True):
        prefill_axis.vlines(
            x,
            value * 0.95,
            value * 1.05,
            color="#777777",
            linewidth=5.0,
            alpha=0.22,
            zorder=1,
        )
    prefill_axis.set_xticks(prompt, ["1K\ncal", "2K\nheld out", "4K\nheld out"])
    prefill_axis.set_xlabel("Input length (tokens)")
    prefill_axis.set_ylabel("Prefill throughput/node (tokens/s)")
    prefill_axis.set_title("b  Three-layer prefill score", loc="left", fontweight="bold")
    prefill_axis.grid(True, color="#dddddd", linewidth=0.5, alpha=0.7)
    prefill_axis.spines["top"].set_visible(False)
    prefill_axis.spines["right"].set_visible(False)
    prefill_axis.legend(
        loc="lower left",
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="#cccccc",
    )

    figure.suptitle(
        "DeepSeek-V3 third scored run: attenuated prefill "
        f"{plot['verdict']} (max {plot['maximum_attenuated_error_percent']:.2f}%)\n"
        f"Unattenuated {plot['unattenuated_verdict']} "
        f"({plot['maximum_unattenuated_error_percent']:.2f}%); "
        f"decode calibration {plot['decode_error_percent']:.2f}% low",
        fontsize=8.3,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.080,
        "Scored scope: 2K and 4K prefill under the declared benchmark-bias model. "
        f"{plot['attenuation_label']}.",
        ha="center",
        va="bottom",
        fontsize=6.35,
        color="#333333",
    )
    figure.text(
        0.5,
        0.046,
        "MTP BLOCKED on COMP-72, numeric anchor unread. Decode remains unattenuated "
        f"({plot['decode_declared_step_ms']:.1f} ms declared vs "
        f"{plot['decode_implied_step_ms']:.1f} ms implied).",
        ha="center",
        va="bottom",
        fontsize=6.25,
        color="#333333",
    )
    figure.text(
        0.5,
        0.015,
        "Prefill and decode are separate experiments. Bands propagate record, "
        "constant, overlap-boundary, attenuation, and registered distribution terms.",
        ha="center",
        va="bottom",
        fontsize=6.1,
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
