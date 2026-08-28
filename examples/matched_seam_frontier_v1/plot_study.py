#!/usr/bin/env python3
"""Render the matched-seam frontier in the published external grammar."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import textwrap
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA = "simllm-matched-seam-frontier-record-v2"
LEGACY_SCHEMA = "simllm-matched-seam-frontier-record-v1"


def _matplotlib() -> Any:
    if "MPLCONFIGDIR" not in os.environ:
        cache = Path(tempfile.gettempdir()) / "simllm-matplotlib-cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = os.fspath(cache)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.4,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _point_xy(point: dict[str, Any]) -> tuple[float, float]:
    return (
        float(point["x_tokens_per_second_per_user"]["decimal"]),
        float(point["y_tokens_per_second_per_gpu"]["decimal"]),
    )


def _fraction(value: dict[str, Any]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def _scored_row(record: dict[str, Any], row_id: str) -> dict[str, Any]:
    return next(row for row in record["rows"] if row["id"] == row_id)


def prepare_plot_data(record: dict[str, Any]) -> dict[str, Any]:
    """Project the strict record into renderer-independent plot data."""

    if record.get("schema") not in {SCHEMA, LEGACY_SCHEMA}:
        raise ValueError(f"unexpected record schema {record.get('schema')!r}")
    families = record["families"]
    external = families["external_curves"]
    series = [
        {
            "id": "external-agg",
            "label": "Their aggregate, MEASURED-EXTERNAL",
            "evidence_class": "MEASURED-EXTERNAL",
            "color": "#6f7378",
            "marker": "o",
            "markerfacecolor": "#6f7378",
            "linestyle": "--",
            "linewidth": 1.15,
            "zorder": 2,
            "points": [
                {
                    "x": float(row["x_tokens_per_second_per_user"]),
                    "y": float(row["y_tokens_per_second_per_gpu"]),
                    "row": row["row"],
                }
                for row in external["agg"]
            ],
        },
        {
            "id": "external-disagg",
            "label": "Their disaggregated, MEASURED-EXTERNAL",
            "evidence_class": "MEASURED-EXTERNAL",
            "color": "#c57b00",
            "marker": "D",
            "markerfacecolor": "white",
            "linestyle": (0, (1.2, 1.2)),
            "linewidth": 2.5,
            "zorder": 3,
            "points": [
                {
                    "x": float(row["x_tokens_per_second_per_user"]),
                    "y": float(row["y_tokens_per_second_per_gpu"]),
                    "row": row["row"],
                }
                for row in external["disagg"]
            ],
        },
        {
            "id": "simllm-ideal",
            "label": "Our unpriced network (zero service), MEASURED-EXTERNAL",
            "evidence_class": "MEASURED-EXTERNAL",
            "color": "#1d6f42",
            "marker": "o",
            "markerfacecolor": "white",
            "linestyle": "-",
            "linewidth": 1.35,
            "zorder": 5,
            "points": [
                {
                    "x": _point_xy(point)[0],
                    "y": _point_xy(point)[1],
                    "row": point["row"],
                    "candidate_id": point["candidate_id"],
                }
                for point in families["F"]["ideal_frontier"]
            ],
        },
        {
            "id": "simllm-packet",
            "label": (
                "Our packet-priced network, "
                "MEASURED-EXTERNAL + SIM-DERIVED"
            ),
            "evidence_class": "MEASURED-EXTERNAL + SIM-DERIVED",
            "color": "#245d9c",
            "marker": "^",
            "markerfacecolor": "#245d9c",
            "linestyle": "-.",
            "linewidth": 1.65,
            "zorder": 4,
            "points": [
                {
                    "x": _point_xy(point)[0],
                    "y": _point_xy(point)[1],
                    "row": point["row"],
                    "candidate_id": point["candidate_id"],
                }
                for point in families["F"]["packet_frontier"]
            ],
        },
    ]
    all_x = [point["x"] for arm in series for point in arm["points"]]
    all_y = [point["y"] for arm in series for point in arm["points"]]

    r_rows = families["R"]["rows"]
    low_row = min(r_rows, key=lambda row: _fraction(row["quotient"]))
    high_row = max(r_rows, key=lambda row: _fraction(row["quotient"]))
    frozen_band = json.loads(r_rows[0]["expected"])

    maximum = families["M"].get(
        "maximum_packet_priced_to_unpriced_network_quotient",
        families["M"].get("maximum_quotient"),
    )
    maximum_fraction = _fraction(maximum)
    maximum_rows = [
        row for row in families["M"]["rows"] if _fraction(row["quotient"]) == maximum_fraction
    ]
    ideal_by_id = {
        point["candidate_id"]: point for point in families["F"]["ideal_frontier"]
    }
    packet_by_id = {
        point["candidate_id"]: point for point in families["F"]["packet_frontier"]
    }
    arrow_row = next(
        row
        for row in maximum_rows
        if row["candidate_id"] in ideal_by_id and row["candidate_id"] in packet_by_id
    )
    arrow_ideal = ideal_by_id[arrow_row["candidate_id"]]
    arrow_packet = packet_by_id[arrow_row["candidate_id"]]
    arrow_x, arrow_ideal_y = _point_xy(arrow_ideal)
    _, arrow_packet_y = _point_xy(arrow_packet)
    m2 = _scored_row(record, "M-2")
    maximum_display = str(m2["observed"]).split("=", 1)[1]
    mechanism_label = (
        "Their planner class prices no network cost.\n"
        "Our unpriced-network arm charges zero network service.\n"
        "This workload: packet-priced / unpriced-network\n"
        f"= {maximum_display}.\n"
        f"Unpriced: {arrow_ideal['evidence_class']}.\n"
        f"Packet-priced: {arrow_packet['evidence_class']}."
    )

    f209 = next(row for row in families["F"]["bracket_rows"] if row["id"] == "F-2-09")
    caption = (
        "Shared timing base: their aggregate and disaggregated series and our "
        "unpriced-network series all price the same imported MEASURED-EXTERNAL "
        "operation database, so their curve differences are composition rather "
        "than kernel timing; our packet-priced series keeps that base and adds "
        "SIM-DERIVED network service. The arrow compares that complete network "
        f"charge with zero network service at this workload, ratio {maximum_display}. "
        "The optional LogGOPSim-priced third arm did not run, so this figure does "
        "not isolate receiver-side serialization. "
        "The separate eight-into-one fan-in envelope from frontier_ladder_v1 and "
        "loggopsim_acceptance_v1 is a different schedule regime and is not plotted "
        f"on these curves. F-2-09 remains a published rounded-axis refutation at "
        f"frontier / external = {f209['observed']}."
    )
    return {
        "axes": {
            "x": {
                "label": "tokens/s/user",
                "scale": "log",
                "limits": [min(all_x) * 0.86, max(all_x) * 1.13],
            },
            "y": {
                "label": "tokens/s/gpu",
                "scale": "log",
                "limits": [min(all_y) * 0.82, max(all_y) * 1.18],
            },
            "optimal_corner": "upper-right",
        },
        "series": series,
        "agreement": {
            "rows": [
                {"row": row["row"], "quotient": float(_fraction(row["quotient"]))}
                for row in r_rows
            ],
            "frozen_band": [float(frozen_band[0]), float(frozen_band[1])],
            "minimum": low_row["observed"],
            "maximum": high_row["observed"],
        },
        "mechanism": {
            "arrow_enabled": bool(m2["passed"]),
            "candidate_rows": [row["row"] for row in maximum_rows],
            "selected_row": arrow_row["row"],
            "x": arrow_x,
            "ideal_y": arrow_ideal_y,
            "packet_y": arrow_packet_y,
            "quotient": maximum_display,
            "label": mechanism_label,
        },
        "f209": {"quotient": f209["observed"], "passed": f209["passed"]},
        "caption": caption,
    }


def _draw_frontier_series(axis: Any, series: dict[str, Any], *, zoom: bool = False) -> Any:
    points = series["points"]
    return axis.plot(
        [point["x"] for point in points],
        [point["y"] for point in points],
        color=series["color"],
        linestyle=series["linestyle"],
        linewidth=series["linewidth"] + (0.25 if zoom else 0.0),
        marker=series["marker"],
        markersize=5.2 if zoom else 3.9,
        markerfacecolor=series["markerfacecolor"],
        markeredgecolor=series["color"],
        markeredgewidth=0.9,
        label=series["label"],
        zorder=series["zorder"],
    )[0]


def _configure_frontier_axis(axis: Any, plot: dict[str, Any]) -> None:
    axis.set_xscale(plot["axes"]["x"]["scale"])
    axis.set_yscale(plot["axes"]["y"]["scale"])
    axis.set_xlim(*plot["axes"]["x"]["limits"])
    axis.set_ylim(*plot["axes"]["y"]["limits"])
    axis.set_xlabel(plot["axes"]["x"]["label"])
    axis.set_ylabel(plot["axes"]["y"]["label"])
    axis.grid(True, which="both", color="#d7dce0", linewidth=0.5, alpha=0.72)


def render(
    record: dict[str, Any],
    *,
    pdf_path: Path,
    png_path: Path,
) -> dict[str, Any]:
    """Render PDF and PNG outputs and return their record projection."""

    plot = prepare_plot_data(record)
    plt = _matplotlib()
    figure = plt.figure(figsize=(8.3, 7.15))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.70, 1.10),
        height_ratios=(3.0, 1.12),
        left=0.125,
        right=0.975,
        top=0.82,
        bottom=0.205,
        hspace=0.44,
        wspace=0.28,
    )
    frontier_axis = figure.add_subplot(grid[0, 0])
    right_grid = grid[0, 1].subgridspec(2, 1, height_ratios=(1.42, 1.0), hspace=0.28)
    zoom_axis = figure.add_subplot(right_grid[0])
    arrow_label_axis = figure.add_subplot(right_grid[1])
    agreement_axis = figure.add_subplot(grid[1, :])

    handles = [
        _draw_frontier_series(frontier_axis, series) for series in plot["series"]
    ]
    _configure_frontier_axis(frontier_axis, plot)
    frontier_axis.set_title("(a) Published deployment frontier", loc="left", pad=7)
    frontier_axis.annotate(
        "better: up and right",
        xy=(0.96, 0.96),
        xytext=(0.68, 0.91),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color="#1d6f42",
        arrowprops={"arrowstyle": "->", "color": "#1d6f42", "linewidth": 1.0},
    )

    disagg = next(series for series in plot["series"] if series["id"] == "external-disagg")
    row_offsets = {
        1: (4, 5),
        2: (4, 5),
        3: (5, 5),
        4: (5, 5),
        5: (6, 4),
        6: (6, 4),
        7: (6, 3),
        8: (6, 4),
        9: (5, 4),
        10: (5, 4),
    }
    for point in disagg["points"]:
        frontier_axis.annotate(
            str(point["row"]),
            (point["x"], point["y"]),
            xytext=row_offsets[point["row"]],
            textcoords="offset points",
            fontsize=6.0,
            color="#714700",
            zorder=8,
        )

    for series in plot["series"]:
        if series["id"] != "external-agg":
            _draw_frontier_series(zoom_axis, series, zoom=True)
    mechanism = plot["mechanism"]
    zoom_axis.set_xscale("log")
    zoom_axis.set_yscale("log")
    zoom_axis.set_xlim(mechanism["x"] * 0.972, mechanism["x"] * 1.033)
    zoom_axis.set_ylim(mechanism["packet_y"] * 0.985, mechanism["ideal_y"] * 1.018)
    zoom_axis.set_xticks([82, 84, 86])
    zoom_axis.set_xticklabels(["82", "84", "86"])
    zoom_axis.set_yticks([610, 630, 650])
    zoom_axis.set_yticklabels(["610", "630", "650"])
    zoom_axis.minorticks_off()
    zoom_axis.grid(True, color="#d7dce0", linewidth=0.5, alpha=0.72)
    zoom_axis.set_title(
        f"(b) Network pricing, external row {mechanism['selected_row']}",
        loc="left",
        fontsize=8.0,
        pad=5,
    )
    zoom_axis.set_xlabel("tokens/s/user", fontsize=7.2, labelpad=2)
    zoom_axis.set_ylabel("tokens/s/gpu", fontsize=7.2, labelpad=2)
    if mechanism["arrow_enabled"]:
        zoom_axis.annotate(
            "",
            xy=(mechanism["x"], mechanism["packet_y"]),
            xytext=(mechanism["x"], mechanism["ideal_y"]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#111111",
                "linewidth": 1.45,
                "mutation_scale": 11,
            },
            zorder=12,
        )
        zoom_axis.scatter(
            [mechanism["x"], mechanism["x"]],
            [mechanism["ideal_y"], mechanism["packet_y"]],
            s=24,
            facecolors="none",
            edgecolors="#111111",
            linewidths=0.8,
            zorder=11,
        )

    arrow_label_axis.axis("off")
    arrow_label_axis.text(
        0.0,
        0.92,
        mechanism["label"],
        ha="left",
        va="top",
        fontsize=5.7,
        linespacing=1.18,
        color="#111111",
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": "#f7f7f5",
            "edgecolor": "#555555",
            "linewidth": 0.7,
        },
    )

    agreement = plot["agreement"]
    residuals = [row["quotient"] - 1.0 for row in agreement["rows"]]
    row_indices = [row["row"] for row in agreement["rows"]]
    lower_residual = agreement["frozen_band"][0] - 1.0
    upper_residual = agreement["frozen_band"][1] - 1.0
    agreement_axis.axhspan(
        lower_residual,
        upper_residual,
        color="#dfe7df",
        alpha=0.8,
        zorder=0,
    )
    agreement_axis.axhline(lower_residual, color="#526052", linestyle="--", linewidth=0.8)
    agreement_axis.axhline(upper_residual, color="#526052", linestyle="--", linewidth=0.8)
    agreement_axis.axhline(0.0, color="#111111", linewidth=0.8, zorder=2)
    agreement_axis.plot(
        row_indices,
        residuals,
        color="#1d6f42",
        linestyle="-",
        linewidth=1.0,
        marker="s",
        markersize=3.8,
        markerfacecolor="white",
        markeredgewidth=0.9,
        zorder=4,
    )
    agreement_axis.set_yscale("symlog", linthresh=1e-5, linscale=1.0, base=10)
    agreement_axis.set_ylim(-0.022, 0.022)
    quotient_ticks = [-0.02, -0.0001, 0.0, 0.0001, 0.02]
    agreement_axis.set_yticks(quotient_ticks)
    agreement_axis.set_yticklabels(
        ["0.980", "0.9999", "1.0000", "1.0001", "1.020"]
    )
    agreement_axis.set_xlim(0.6, 10.4)
    agreement_axis.set_xticks(row_indices)
    agreement_axis.set_xlabel("External disaggregated row index")
    agreement_axis.set_ylabel("Our / published\ndecode-step quotient")
    agreement_axis.set_title(
        "(c) Matched decode step, symmetric-log distance about 1",
        loc="left",
        fontsize=8.4,
        pad=5,
    )
    agreement_axis.grid(True, axis="x", color="#d7dce0", linewidth=0.45, alpha=0.7)
    agreement_axis.text(
        0.01,
        0.94,
        (
            f"Family R agreement: {agreement['minimum']} to {agreement['maximum']}\n"
            f"Frozen band: [{agreement['frozen_band'][0]:.2f}, "
            f"{agreement['frozen_band'][1]:.2f}]"
        ),
        transform=agreement_axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        fontweight="bold",
        color="#26352a",
    )

    figure.suptitle(
        "Qwen3-32B FP8 matched-seam deployment frontier",
        x=0.5,
        y=0.975,
        fontsize=12.0,
    )
    figure.legend(
        handles,
        [series["label"] for series in plot["series"]],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=True,
        framealpha=0.98,
        edgecolor="#b8b8b8",
        columnspacing=1.0,
        handlelength=3.1,
    )
    figure.text(
        0.125,
        0.025,
        textwrap.fill(plot["caption"], width=136),
        ha="left",
        va="bottom",
        fontsize=6.2,
        linespacing=1.22,
        color="#30363b",
    )
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        pdf_path,
        dpi=300,
        metadata={"Creator": "SimLLM", "CreationDate": None, "ModDate": None},
    )
    figure.savefig(png_path, dpi=240, metadata={"Software": "SimLLM"})
    plt.close(figure)
    return plot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    render(record, pdf_path=args.pdf, png_path=args.png)


if __name__ == "__main__":
    main()
