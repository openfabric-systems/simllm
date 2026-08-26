#!/usr/bin/env python3
"""Render the second CORE-54 publication figure from its scored result."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from curve_tools import as_fraction, load_json

COLORS = ("#2a78d6", "#1baf7a")


def _number(value: object, name: str) -> float:
    return float(as_fraction(value, name))


def prepare_flagship_plot(
    result: dict[str, Any],
    anchors: dict[str, Any],
) -> dict[str, Any]:
    """Build a matplotlib-independent two-panel plot description."""

    anchor_map = {anchor["id"]: anchor for anchor in anchors["anchors"]}
    curves = []
    for curve in result["curves"]:
        points = []
        for point in curve["points"]:
            interval = point["uncertainty"]
            x = interval["aggregated_output_throughput_tokens_per_second"]
            y = interval["inverse_per_token_request_delay_tokens_per_second"]
            points.append(
                {
                    "x": _number(x["point"], "x.point"),
                    "x_lower": _number(x["lower"], "x.lower"),
                    "x_upper": _number(x["upper"], "x.upper"),
                    "y": _number(y["point"], "y.point"),
                    "y_lower": _number(y["lower"], "y.lower"),
                    "y_upper": _number(y["upper"], "y.upper"),
                }
            )
        curves.append(
            {
                "id": curve["configuration_id"],
                "label": curve["configuration_label"],
                "evidence_class": curve["evidence_class"],
                "points": points,
            }
        )
    prediction_map = {
        row["anchor_id"]: row for row in result["anchor_predictions"]
    }
    prefill = []
    for prompt_k, anchor_id in (
        (1, "sglang_prefill_1k"),
        (2, "sglang_prefill_2k"),
        (4, "sglang_prefill_4k"),
    ):
        prediction = prediction_map[anchor_id]["prediction"]
        prefill.append(
            {
                "prompt_k": prompt_k,
                "anchor_id": anchor_id,
                "role": anchor_map[anchor_id]["role"],
                "published": float(anchor_map[anchor_id]["value"]),
                "predicted": _number(prediction["point"], "prefill.point"),
                "lower": _number(prediction["lower"], "prefill.lower"),
                "upper": _number(prediction["upper"], "prefill.upper"),
            }
        )
    score = result["held_out_score"]
    decode = result["decode_calibration_miss"]
    maximum = 100 * _number(
        score["maximum_absolute_relative_error"],
        "maximum_absolute_relative_error",
    )
    decode_error = 100 * _number(
        decode["absolute_relative_error"],
        "decode.absolute_relative_error",
    )
    declared_ms = decode["declared_step_ps"] / 1_000_000_000
    implied_ms = 1_000 * _number(
        decode["published_throughput_implied_step_ps"],
        "decode.published_throughput_implied_step_ps",
    ) / 1_000_000_000_000
    return {
        "curves": curves,
        "prefill": prefill,
        "decode_disclosures": [
            {
                "label": "Published SGLang standard decode, ~100 ms context",
                "x": float(anchor_map["sglang_decode_standard"]["aggregated_value"]),
                "y": 10.0,
                "marker": "D",
            },
            {
                "label": "Published simulated MTP, pricing blocked",
                "x": float(
                    anchor_map["sglang_decode_simulated_mtp"]["aggregated_value"]
                ),
                "y": 10.0,
                "marker": "s",
            },
        ],
        "h800_x": float(anchor_map["deepseek_production_decode"]["aggregated_value"]),
        "verdict": score["status"],
        "scope": score["scope"],
        "maximum_error_percent": maximum,
        "mtp_dependency": score["blocked_rows"][0]["dependency"],
        "decode_error_percent": decode_error,
        "decode_declared_step_ms": declared_ms,
        "decode_implied_step_ms": implied_ms,
    }


def render_flagship_figure(
    plot: dict[str, Any],
    output_stem: Path,
) -> tuple[Path, Path]:
    """Render one publication-sized PDF and PNG from the second scored data."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.2,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, (curve_axis, prefill_axis) = plt.subplots(
        1,
        2,
        figsize=(7.0, 4.33),
        gridspec_kw={"width_ratios": (1.8, 1.0)},
        constrained_layout=False,
    )
    figure.subplots_adjust(left=0.10, right=0.97, bottom=0.23, top=0.81, wspace=0.30)
    simulated_handles = []
    disclosure_handles = []
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
        color = COLORS[index % len(COLORS)]
        style = "--" if is_what_if else "-"
        curve_axis.fill_between(
            x,
            y_lower,
            y_upper,
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        handle = curve_axis.plot(
            x,
            y,
            color=color,
            linestyle=style,
            marker="o",
            markersize=3.7,
            linewidth=1.7,
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
            capsize=1.7,
            alpha=0.65,
            zorder=3,
        )
        if is_what_if:
            context_handles.append(handle)
        else:
            simulated_handles.append(handle)
        all_x.extend(x_lower + x_upper)
        all_y.extend(y_lower + y_upper)
    for disclosure in plot["decode_disclosures"]:
        curve_axis.scatter(
            [disclosure["x"]],
            [disclosure["y"]],
            marker=disclosure["marker"],
            s=34,
            facecolors="white",
            edgecolors="#111111",
            linewidths=1.0,
            zorder=6,
        )
        all_x.append(disclosure["x"])
        all_y.append(disclosure["y"])
        disclosure_handles.append(
            Line2D(
                [0],
                [0],
                color="#111111",
                marker=disclosure["marker"],
                markerfacecolor="white",
                linestyle="none",
                label=disclosure["label"],
            )
        )
    curve_axis.axvline(
        plot["h800_x"],
        color="#555555",
        linestyle=":",
        linewidth=1.1,
        zorder=1,
    )
    context_handles.insert(
        0,
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=":",
            linewidth=1.1,
            label="DeepSeek H800 production, delay undisclosed",
        ),
    )
    curve_axis.set_xlabel("Aggregated output throughput (tokens/s) →")
    curve_axis.set_ylabel("Inverse per-token request delay (tokens/s) ↑")
    curve_axis.set_yscale("log")
    curve_axis.set_title("a  Output deployment curves", loc="left", fontweight="bold")
    curve_axis.grid(True, color="#dddddd", linewidth=0.5, alpha=0.7)
    curve_axis.spines["top"].set_visible(False)
    curve_axis.spines["right"].set_visible(False)
    curve_axis.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    x_span = max(all_x) - min(all_x)
    curve_axis.set_xlim(max(0.0, min(all_x) - 0.06 * x_span), max(all_x) + 0.06 * x_span)
    curve_axis.set_ylim(min(all_y) / 1.25, max(all_y) * 1.25)
    first_legend = curve_axis.legend(
        handles=simulated_handles + disclosure_handles,
        title="SGLang configuration and disclosures",
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        edgecolor="#cccccc",
    )
    curve_axis.add_artist(first_legend)
    curve_axis.legend(
        handles=context_handles,
        title="Other deployment configurations",
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        edgecolor="#cccccc",
    )
    curve_axis.text(
        0.02,
        0.04,
        "Decode calibration miss\n"
        f"{plot['decode_declared_step_ms']:.1f} ms declared vs "
        f"{plot['decode_implied_step_ms']:.1f} ms implied\n"
        f"throughput prediction {plot['decode_error_percent']:.2f}% low",
        transform=curve_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#bbbbbb",
            "alpha": 0.94,
        },
        zorder=8,
    )

    prompt = [row["prompt_k"] for row in plot["prefill"]]
    published = [row["published"] for row in plot["prefill"]]
    predicted = [row["predicted"] for row in plot["prefill"]]
    lower = [row["lower"] for row in plot["prefill"]]
    upper = [row["upper"] for row in plot["prefill"]]
    prefill_axis.plot(
        prompt,
        predicted,
        color=COLORS[0],
        marker="o",
        linewidth=1.7,
        label="simLLM composed projection",
    )
    prefill_axis.fill_between(prompt, lower, upper, color=COLORS[0], alpha=0.18)
    prefill_axis.scatter(
        prompt,
        published,
        marker="D",
        s=38,
        facecolors="white",
        edgecolors="#111111",
        linewidths=1.0,
        label="published SGLang",
        zorder=5,
    )
    prefill_axis.axhspan(
        published[1] * 0.95,
        published[1] * 1.05,
        color="#d9d9d9",
        alpha=0.22,
        linewidth=0,
    )
    prefill_axis.axhspan(
        published[2] * 0.95,
        published[2] * 1.05,
        color="#d9d9d9",
        alpha=0.22,
        linewidth=0,
    )
    prefill_axis.set_xticks(prompt, ["1K\ncal", "2K\nheld out", "4K\nheld out"])
    prefill_axis.set_xlabel("Input length (tokens)")
    prefill_axis.set_ylabel("Prefill throughput/node (tokens/s)")
    prefill_axis.set_title("b  Published prefill anchors", loc="left", fontweight="bold")
    prefill_axis.grid(True, color="#dddddd", linewidth=0.5, alpha=0.7)
    prefill_axis.spines["top"].set_visible(False)
    prefill_axis.spines["right"].set_visible(False)
    prefill_axis.legend(
        loc="lower left",
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        edgecolor="#cccccc",
    )
    figure.suptitle(
        "DeepSeek-V3 second scored run: "
        f"held-out prefill {plot['verdict']} "
        f"(max {plot['maximum_error_percent']:.2f}%); "
        "MTP BLOCKED (COMP-72)",
        fontsize=9.2,
        fontweight="bold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.065,
        "Verdict scope: priced 2K and 4K prefill anchors only. "
        "The MTP anchor is blocked and decode is a disclosed calibration miss.",
        ha="center",
        va="bottom",
        fontsize=6.6,
        color="#333333",
    )
    figure.text(
        0.5,
        0.025,
        "Prefill and decode are separate disclosure experiments. Bands carry "
        "record, inherited-constant and zero-width COMP-74 distribution terms.",
        ha="center",
        va="bottom",
        fontsize=6.4,
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
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot = prepare_flagship_plot(load_json(args.result), load_json(args.anchors))
    pdf, png = render_flagship_figure(plot, args.output_stem)
    print(f"wrote {pdf.as_posix()} and {png.as_posix()}")


if __name__ == "__main__":
    main()
