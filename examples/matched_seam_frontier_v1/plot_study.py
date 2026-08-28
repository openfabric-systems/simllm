#!/usr/bin/env python3
"""Render the matched-seam frontier in the published external grammar."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Any

SCHEMA = "simllm-matched-seam-frontier-record-v1"


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
            "font.size": 8.3,
            "axes.labelsize": 9.0,
            "axes.titlesize": 10.5,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
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


def prepare_plot_data(record: dict[str, Any]) -> dict[str, Any]:
    """Project the strict record into renderer-independent plot data."""

    if record.get("schema") != SCHEMA:
        raise ValueError(f"unexpected record schema {record.get('schema')!r}")
    families = record["families"]
    external = families["external_curves"]
    series = [
        {
            "id": "external-agg",
            "label": "External agg, MEASURED-EXTERNAL",
            "evidence_class": "MEASURED-EXTERNAL",
            "color": "#6f7378",
            "marker": "o",
            "linestyle": "--",
            "linewidth": 1.15,
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
            "label": "External disagg, MEASURED-EXTERNAL",
            "evidence_class": "MEASURED-EXTERNAL",
            "color": "#d48a17",
            "marker": "D",
            "linestyle": "-",
            "linewidth": 1.45,
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
            "label": "SimLLM ideal seam, MEASURED-EXTERNAL",
            "evidence_class": "MEASURED-EXTERNAL",
            "color": "#1f7a3a",
            "marker": "o",
            "linestyle": "-",
            "linewidth": 2.15,
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
            "label": "SimLLM packet seam, MEASURED-EXTERNAL + SIM-DERIVED",
            "evidence_class": "MEASURED-EXTERNAL + SIM-DERIVED",
            "color": "#2563a6",
            "marker": "^",
            "linestyle": "-",
            "linewidth": 1.8,
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
    return {
        "axes": {
            "x": {
                "label": "Per-user output speed (tokens/s/user)",
                "scale": "log",
                "limits": [min(all_x) * 0.86, max(all_x) * 1.13],
            },
            "y": {
                "label": "Output throughput (tokens/s/GPU)",
                "scale": "log",
                "limits": [min(all_y) * 0.82, max(all_y) * 1.18],
            },
            "optimal_corner": "upper-right",
        },
        "series": series,
        "caption": record["figure"]["caption"],
    }


def render(
    record: dict[str, Any],
    *,
    pdf_path: Path,
    png_path: Path,
) -> dict[str, Any]:
    """Render PDF and PNG outputs and return their plot contract."""

    plot = prepare_plot_data(record)
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(7.4, 5.4), constrained_layout=False)
    figure.subplots_adjust(left=0.12, right=0.98, top=0.89, bottom=0.28)

    for series in plot["series"]:
        points = series["points"]
        axis.plot(
            [point["x"] for point in points],
            [point["y"] for point in points],
            color=series["color"],
            linestyle=series["linestyle"],
            linewidth=series["linewidth"],
            marker=series["marker"],
            markersize=4.0,
            markerfacecolor=("white" if series["id"] == "simllm-ideal" else series["color"]),
            markeredgecolor=series["color"],
            markeredgewidth=0.8,
            label=series["label"],
            zorder=4 if series["id"].startswith("simllm") else 3,
        )

    disagg = next(series for series in plot["series"] if series["id"] == "external-disagg")
    for point in disagg["points"]:
        axis.annotate(
            str(point["row"]),
            (point["x"], point["y"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.4,
            color="#6d4c12",
        )

    axis.set_xscale(plot["axes"]["x"]["scale"])
    axis.set_yscale(plot["axes"]["y"]["scale"])
    axis.set_xlim(*plot["axes"]["x"]["limits"])
    axis.set_ylim(*plot["axes"]["y"]["limits"])
    axis.set_xlabel(plot["axes"]["x"]["label"])
    axis.set_ylabel(plot["axes"]["y"]["label"])
    axis.set_title("Qwen3-32B FP8 matched-seam deployment frontier")
    axis.grid(True, which="both", color="#d7dce0", linewidth=0.55, alpha=0.75)
    axis.annotate(
        "better: up and right",
        xy=(0.97, 0.95),
        xytext=(0.76, 0.76),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color="#1f7a3a",
        arrowprops={"arrowstyle": "->", "color": "#1f7a3a", "linewidth": 1.0},
    )
    axis.legend(loc="lower left", frameon=True, framealpha=0.96)
    figure.text(
        0.12,
        0.055,
        textwrap.fill(plot["caption"], width=125),
        ha="left",
        va="bottom",
        fontsize=6.8,
        color="#3c4248",
    )
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        pdf_path,
        dpi=300,
        metadata={"Creator": "SimLLM", "CreationDate": None, "ModDate": None},
    )
    figure.savefig(png_path, dpi=220, metadata={"Software": "SimLLM"})
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
