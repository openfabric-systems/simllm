#!/usr/bin/env python3
"""Render the frozen NV-style frontier comparison overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RESULT_SCHEMA = "simllm-frontier-comparison-study-v1"
STUDY_DIR = Path(__file__).resolve().parent

ARM_STYLES = {
    "0.6": {"color": "#a8d46f", "label": "SimLLM ESTIMATE, e=0.6"},
    "0.8": {"color": "#5fad56", "label": "SimLLM ESTIMATE, e=0.8"},
    "1.0": {"color": "#1f7a3a", "label": "SimLLM ESTIMATE, e=1.0"},
}


def _matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.3,
            "axes.labelsize": 9.0,
            "axes.titlesize": 10.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _decimal(point: dict[str, Any], field: str) -> float:
    return float(point[field]["decimal"])


def prepare_plot_data(result: dict[str, Any]) -> dict[str, Any]:
    """Project the strict record into renderer-independent plot data."""

    if result.get("schema") != RESULT_SCHEMA:
        raise ValueError(f"unexpected result schema {result.get('schema')!r}")
    families = result["families"]
    arms = []
    for arm in ("0.6", "0.8", "1.0"):
        frontier = families["X3"]["arms"][arm]["frontier"]
        arms.append(
            {
                "efficiency": arm,
                "point_class": "ESTIMATE",
                **ARM_STYLES[arm],
                "points": [
                    {
                        "x": _decimal(point, "x_tokens_per_second_per_user"),
                        "y": _decimal(point, "y_tokens_per_second_per_gpu"),
                        "candidate_id": point["candidate_id"],
                    }
                    for point in frontier
                ],
            }
        )
    external = [
        {
            "row": row["row"],
            "x": float(row["x_tokens_per_second_per_user"]),
            "y": float(row["y_tokens_per_second_per_gpu"]),
            "point_class": row["evidence_class"],
            "x3c_pass": row["x3c"]["passed"],
        }
        for row in families["X3"]["external_rows"]
    ]
    return {
        "axes": {
            "x": {
                "label": "Per-user output speed (tokens/s/user)",
                "scale": "log",
                "limits": [45.0, 1300.0],
            },
            "y": {
                "label": "Output throughput (tokens/s/GPU)",
                "scale": "log",
                "limits": [50.0, 1500.0],
            },
            "optimal_corner": "upper-right",
        },
        "arms": arms,
        "external": external,
        "caption": (
            "All SimLLM curves are ESTIMATE classes. External diamonds are "
            "MEASURED-EXTERNAL display points from aiconfigurator 0.11.0 and "
            "its h200_sxm TensorRT-LLM 1.3.0rc10 database. The X4 ladder "
            "regime applies here: contention-free point-to-point legs differ "
            "by about 1.6 percent between ideal and packet rungs, while the "
            "frozen eight-into-one fan-in cell is about 8x and is not claimed "
            "for this workload. Source: frontier_ladder_v1."
        ),
    }


def render(
    result: dict[str, Any],
    *,
    pdf_path: Path,
    png_path: Path,
) -> dict[str, Any]:
    """Render PDF and PNG outputs and return the plot contract."""

    plot = prepare_plot_data(result)
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(7.4, 5.4), constrained_layout=False)
    figure.subplots_adjust(left=0.12, right=0.98, top=0.89, bottom=0.27)

    for arm in plot["arms"]:
        points = arm["points"]
        axis.plot(
            [point["x"] for point in points],
            [point["y"] for point in points],
            color=arm["color"],
            linewidth=2.0 if arm["efficiency"] == "1.0" else 1.55,
            marker="o",
            markersize=3.5,
            markerfacecolor="white",
            markeredgewidth=0.8,
            label=arm["label"],
            zorder=3,
        )

    external = plot["external"]
    axis.plot(
        [point["x"] for point in external],
        [point["y"] for point in external],
        color="#353a40",
        linewidth=1.15,
        alpha=0.72,
        zorder=2,
    )
    axis.scatter(
        [point["x"] for point in external],
        [point["y"] for point in external],
        marker="D",
        s=34,
        facecolors="#f4b942",
        edgecolors="#353a40",
        linewidths=0.7,
        label="aiconfigurator 0.11.0 MEASURED-EXTERNAL",
        zorder=5,
    )
    for point in external:
        offset = {1: (-5, 8), 2: (6, 5)}.get(point["row"], (4, 4))
        axis.annotate(
            str(point["row"]),
            (point["x"], point["y"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.7,
            color="#30343b",
        )
    miss = next(point for point in external if not point["x3c_pass"])
    axis.annotate(
        "X3c miss: row 10 below e=0.6",
        (miss["x"], miss["y"]),
        xytext=(-112, -25),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#a33a2b", "lw": 0.8},
        color="#a33a2b",
        fontsize=7.2,
        zorder=7,
    )

    axis.set_xscale(plot["axes"]["x"]["scale"])
    axis.set_yscale(plot["axes"]["y"]["scale"])
    axis.set_xlim(*plot["axes"]["x"]["limits"])
    axis.set_ylim(*plot["axes"]["y"]["limits"])
    axis.set_xlabel(plot["axes"]["x"]["label"])
    axis.set_ylabel(plot["axes"]["y"]["label"])
    axis.set_title("Qwen3-32B FP8, 32-GPU deployment frontier comparison")
    axis.grid(True, which="both", color="#d7dce0", linewidth=0.55, alpha=0.75)
    axis.text(
        0.985,
        0.965,
        "better",
        transform=axis.transAxes,
        ha="right",
        va="top",
        color="#1f7a3a",
        fontsize=7.2,
        fontweight="bold",
    )
    axis.legend(loc="lower left", frameon=True, framealpha=0.96)
    figure.text(
        0.12,
        0.055,
        plot["caption"],
        ha="left",
        va="bottom",
        fontsize=6.8,
        color="#3c4248",
        wrap=True,
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
    parser.add_argument("result", type=Path)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    render(result, pdf_path=args.pdf, png_path=args.png)


if __name__ == "__main__":
    main()
