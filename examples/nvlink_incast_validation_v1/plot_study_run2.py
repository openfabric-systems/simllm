#!/usr/bin/env python3
"""Render the TRAF-74 second hardware and simulated-goodput comparison."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "results_run2.json"
FIGURE_STEM = "nvlink-incast-hardware-simulation-run2"

INK = "#202124"
MUTED = "#666666"
GRID = "#deded9"
SURFACE = "#fcfcfa"
HARDWARE = "#2878b5"
SIMULATION = "#d1495b"
ACCEPTANCE = "#e9c46a"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    return parser.parse_args(argv)


def _matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "axes.edgecolor": INK,
            "axes.labelcolor": MUTED,
            "axes.linewidth": 0.9,
            "figure.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "pdf.fonttype": 42,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    return plt


def load_result(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8", newline="") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError("TRAF-74 run-two result must be a JSON object")
    if value.get("schema") != "simllm-nvlink-incast-validation-score-v2":
        raise RuntimeError("unexpected TRAF-74 run-two result schema")
    return value


def render(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    plt = _matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = result["comparisons"]
    sizes = sorted({int(row["size_bytes"]) for row in comparisons})
    if sizes != [4 << 20, 8 << 20]:
        raise RuntimeError("the run-two result does not contain both frozen sizes")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.9), sharey=True)
    for axis, size_bytes in zip(axes, sizes, strict=True):
        rows = sorted(
            (row for row in comparisons if row["size_bytes"] == size_bytes),
            key=lambda row: row["degree"],
        )
        if len(rows) != 3:
            raise RuntimeError(f"flow size {size_bytes} does not have three degrees")
        degrees = [row["degree"] for row in rows]
        hardware = [row["hardware_aggregate_gbps"] for row in rows]
        simulation = [row["simulation_aggregate_gbps"] for row in rows]
        accepted_low = [value / 1.16 for value in simulation]
        accepted_high = [value / 0.84 for value in simulation]
        axis.fill_between(
            degrees,
            accepted_low,
            accepted_high,
            color=ACCEPTANCE,
            alpha=0.22,
            linewidth=0,
            label="Frozen plus or minus 16% band",
        )
        axis.plot(
            degrees,
            simulation,
            color=SIMULATION,
            linestyle="--",
            marker="s",
            linewidth=1.8,
            markersize=5.0,
            label="Scored simulation",
        )
        axis.plot(
            degrees,
            hardware,
            color=HARDWARE,
            linestyle="-",
            marker="o",
            linewidth=1.8,
            markersize=5.2,
            label="Measured hardware",
        )
        axis.set_title(f"{size_bytes // (1 << 20)} MiB per sender", fontsize=10.0, pad=7)
        axis.set_xlabel("Incast degree (senders into GPU 0)")
        axis.set_xticks(degrees)
        axis.set_xlim(0.8, 3.2)
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        misses = [row for row in rows if row["verdict"] == "MISS"]
        miss_parameters = {row["responsible_parameter"] for row in misses}
        if misses and len(miss_parameters) == 1:
            parameter = next(iter(miss_parameters)).replace("_", " ")
            axis.text(
                0.03,
                0.96,
                f"All cells miss: {parameter}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7.2,
                color=MUTED,
            )
        else:
            for row in misses:
                axis.annotate(
                    row["responsible_parameter"].replace("_", " "),
                    (row["degree"], row["hardware_aggregate_gbps"]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.9,
                    color=MUTED,
                )
    axes[0].set_ylabel("Aggregate receiver payload goodput (GB/s)")
    axes[0].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        labels[::-1],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.84),
        ncol=3,
        frameon=False,
        fontsize=8.2,
    )
    fig.suptitle(
        "NV4 long-flow incast run two: measured hardware against simulation",
        fontsize=10.5,
        y=0.98,
    )
    if result["measurement_validity"] == "VOID_FATAL_GUARD":
        fig.text(
            0.5,
            0.91,
            "Diagnostic only: a frozen fatal guard voided the run",
            ha="center",
            va="center",
            fontsize=8.2,
            color=SIMULATION,
        )
    fig.text(
        0.5,
        0.015,
        (
            "Aggregate payload goodput is total delivered payload divided by the "
            "latest source completion. Hardware degrees 1 to 3 only."
        ),
        ha="center",
        va="bottom",
        fontsize=7.6,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.20, top=0.68, wspace=0.14)
    pdf = output_dir / f"{FIGURE_STEM}.pdf"
    png = output_dir / f"{FIGURE_STEM}.png"
    fig.savefig(
        pdf,
        metadata={
            "Creator": "SimLLM TRAF-74",
            "Title": "NV4 long-flow incast run two hardware against simulation",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        png,
        dpi=180,
        metadata={"Software": "SimLLM TRAF-74"},
    )
    plt.close(fig)
    return pdf, png


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = load_result(args.result)
    paths = render(result, args.output_dir)
    for path in paths:
        print(path.relative_to(HERE).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
