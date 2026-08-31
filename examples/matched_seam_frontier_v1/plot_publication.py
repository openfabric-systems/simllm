#!/usr/bin/env python3
"""Render the two-panel matched-seam publication figure from record.json."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "simllm-matched-seam-frontier-record-v2"
MECHANISM_ROW = 3
CAPTION = (
    "Same imported measured DB; composition/network differ "
    "(MEASURED-EXTERNAL + SIM-DERIVED)."
)

SERIES_STYLES = {
    "external-aggregate": {
        "label": "AIConfigurator, aggregate",
        "color": "#666666",
        "linestyle": (0, (4.0, 2.0)),
        "linewidth": 1.0,
        "marker": "o",
        "markerfacecolor": "#666666",
        "zorder": 2,
    },
    "external-disaggregated": {
        "label": "AIConfigurator, disaggregated",
        "color": "#d55e00",
        "linestyle": (0, (1.0, 1.3)),
        "linewidth": 1.65,
        "marker": "D",
        "markerfacecolor": "white",
        "zorder": 3,
    },
    "simllm-unpriced": {
        "label": "SimLLM, network unpriced",
        "color": "#009e73",
        "linestyle": "-",
        "linewidth": 1.15,
        "marker": "s",
        "markerfacecolor": "white",
        "zorder": 5,
    },
    "simllm-packet": {
        "label": "SimLLM, network priced (packet)",
        "color": "#0072b2",
        "linestyle": "-.",
        "linewidth": 1.35,
        "marker": "^",
        "markerfacecolor": "#0072b2",
        "zorder": 4,
    },
}


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
            "font.size": 6.8,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.6,
            "legend.fontsize": 5.8,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _external_points(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    return [
        (
            float(row["x_tokens_per_second_per_user"]),
            float(row["y_tokens_per_second_per_gpu"]),
        )
        for row in rows
    ]


def _simllm_points(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    return [
        (
            float(row["x_tokens_per_second_per_user"]["decimal"]),
            float(row["y_tokens_per_second_per_gpu"]["decimal"]),
        )
        for row in rows
    ]


def prepare_publication_data(record: dict[str, Any]) -> dict[str, Any]:
    """Read the publication projection without deriving study quantities."""

    if record.get("schema") != SCHEMA:
        raise ValueError(f"unexpected record schema {record.get('schema')!r}")

    families = record["families"]
    external = families["external_curves"]
    ideal_frontier = families["F"]["ideal_frontier"]
    packet_frontier = families["F"]["packet_frontier"]
    mechanism = families["M"]
    if mechanism["unpriced_network_service_ps"] != 0:
        raise ValueError("publication figure requires zero unpriced-network service")
    if mechanism["third_loggopsim_priced_arm"]["ran"]:
        raise ValueError("publication figure is defined for the two-arm mechanism run")

    series = [
        {
            **SERIES_STYLES["external-aggregate"],
            "points": _external_points(external["agg"]),
        },
        {
            **SERIES_STYLES["external-disaggregated"],
            "points": _external_points(external["disagg"]),
        },
        {
            **SERIES_STYLES["simllm-unpriced"],
            "points": _simllm_points(ideal_frontier),
        },
        {
            **SERIES_STYLES["simllm-packet"],
            "points": _simllm_points(packet_frontier),
        },
    ]

    m2 = next(row for row in record["rows"] if row["id"] == "M-2")
    ratio_text = str(m2["observed"]).removeprefix("maximum=")
    if len(ratio_text) < 6 or ratio_text[1] != ".":
        raise ValueError(f"unexpected M-2 display value {ratio_text!r}")
    maximum = mechanism["maximum_packet_priced_to_unpriced_network_quotient"]
    mechanism_row = next(
        row for row in mechanism["rows"] if row["row"] == MECHANISM_ROW
    )
    if mechanism_row["quotient"] != maximum:
        raise ValueError("selected mechanism row no longer carries the recorded maximum")

    ideal_row = next(row for row in ideal_frontier if row["row"] == MECHANISM_ROW)
    packet_row = next(row for row in packet_frontier if row["row"] == MECHANISM_ROW)
    if mechanism_row["ideal_y"] != ideal_row["y_tokens_per_second_per_gpu"]:
        raise ValueError("mechanism and frontier ideal values disagree")
    if mechanism_row["packet_y"] != packet_row["y_tokens_per_second_per_gpu"]:
        raise ValueError("mechanism and frontier packet values disagree")

    return {
        "series": series,
        "mechanism": {
            "x": float(ideal_row["x_tokens_per_second_per_user"]["decimal"]),
            "unpriced_y": float(mechanism_row["ideal_y"]["decimal"]),
            "packet_y": float(mechanism_row["packet_y"]["decimal"]),
            "label": (
                "network cost AIConfigurator does not price\n"
                f"packet / unpriced = {ratio_text[:6]} (this workload)"
            ),
        },
        "caption": CAPTION,
    }


def _draw_series(axis: Any, series: dict[str, Any], *, zoom: bool = False) -> Any:
    x_values = [point[0] for point in series["points"]]
    y_values = [point[1] for point in series["points"]]
    return axis.plot(
        x_values,
        y_values,
        color=series["color"],
        linestyle=series["linestyle"],
        linewidth=series["linewidth"] + (0.15 if zoom else 0.0),
        marker=series["marker"],
        markersize=4.0 if zoom else 3.1,
        markerfacecolor=series["markerfacecolor"],
        markeredgecolor=series["color"],
        markeredgewidth=0.75,
        label=series["label"],
        zorder=series["zorder"],
    )[0]


def _style_axis(axis: Any) -> None:
    axis.tick_params(which="both", direction="out", width=0.55, length=2.5)
    axis.grid(True, which="major", color="#d2d2d2", linewidth=0.45, alpha=0.75)
    axis.minorticks_off()


def render(record: dict[str, Any], *, pdf_path: Path, png_path: Path) -> dict[str, Any]:
    """Render the publication PDF and PNG, then return the record projection."""

    plot = prepare_publication_data(record)
    plt = _matplotlib()
    figure = plt.figure(figsize=(3.5, 5.2))
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.45, 1.0),
        left=0.175,
        right=0.975,
        top=0.855,
        bottom=0.095,
        hspace=0.48,
    )
    frontier_axis = figure.add_subplot(grid[0])
    zoom_axis = figure.add_subplot(grid[1])

    handles = [_draw_series(frontier_axis, series) for series in plot["series"]]
    frontier_axis.set_xscale("log")
    frontier_axis.set_yscale("log")
    frontier_axis.set_xlim(49, 240)
    frontier_axis.set_ylim(68, 850)
    frontier_axis.set_xticks([50, 100, 200])
    frontier_axis.set_xticklabels(["50", "100", "200"])
    frontier_axis.set_yticks([100, 200, 400, 800])
    frontier_axis.set_yticklabels(["100", "200", "400", "800"])
    frontier_axis.set_xlabel("tokens/s/user", labelpad=2)
    frontier_axis.set_ylabel("tokens/s/gpu", labelpad=3)
    frontier_axis.set_title("(a) Published frontier", loc="left", pad=4)
    _style_axis(frontier_axis)

    for series in plot["series"][1:]:
        _draw_series(zoom_axis, series, zoom=True)
    mechanism = plot["mechanism"]
    zoom_axis.set_xscale("log")
    zoom_axis.set_yscale("log")
    zoom_axis.set_xlim(81.5, 86.5)
    zoom_axis.set_ylim(610, 665)
    zoom_axis.set_xticks([82, 84, 86])
    zoom_axis.set_xticklabels(["82", "84", "86"])
    zoom_axis.set_yticks([620, 640, 660])
    zoom_axis.set_yticklabels(["620", "640", "660"])
    zoom_axis.set_xlabel("tokens/s/user", labelpad=2)
    zoom_axis.set_ylabel("tokens/s/gpu", labelpad=3)
    zoom_axis.set_title("(b) Network mechanism zoom, row 3", loc="left", pad=4)
    _style_axis(zoom_axis)
    zoom_axis.annotate(
        "",
        xy=(mechanism["x"], mechanism["packet_y"]),
        xytext=(mechanism["x"], mechanism["unpriced_y"]),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#111111",
            "linewidth": 1.05,
            "mutation_scale": 8,
        },
        zorder=10,
    )
    zoom_axis.text(
        0.025,
        0.965,
        mechanism["label"],
        transform=zoom_axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        linespacing=1.12,
        color="#111111",
        backgroundcolor="white",
        zorder=11,
    )

    figure.legend(
        handles,
        [series["label"] for series in plot["series"]],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=2,
        frameon=False,
        columnspacing=0.9,
        handlelength=2.4,
        handletextpad=0.5,
    )
    figure.text(
        0.5,
        0.018,
        plot["caption"],
        ha="center",
        va="bottom",
        fontsize=4.7,
        fontfamily="DejaVu Sans",
        color="#222222",
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        pdf_path,
        dpi=300,
        metadata={"Creator": "SimLLM", "CreationDate": None, "ModDate": None},
    )
    figure.savefig(png_path, dpi=300, metadata={"Software": "SimLLM"})
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
