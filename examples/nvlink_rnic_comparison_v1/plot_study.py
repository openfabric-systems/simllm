#!/usr/bin/env python3
"""Render the frozen TRAF-71 CDF and dispersion figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXPECTATIONS_PATH = HERE / "expectations.json"

INK = "#202124"
MUTED = "#666666"
GRID = "#deded9"
SURFACE = "#fcfcfa"
DEGREE_COLORS = {1: "#2878b5", 2: "#e76f51", 3: "#2a9d8f"}
TRANSPORT_STYLES = {"nvlink-credit": "-", "rnic-nn": "--"}
TRANSPORT_LABELS = {
    "nvlink-credit": "NVLink credit domain",
    "rnic-nn": "rnic-nn max-min slots",
}
SIZE_LABELS = {
    256: "256 B",
    1024: "1 KiB",
    4096: "4 KiB",
    16384: "16 KiB",
    65536: "64 KiB",
    262144: "256 KiB",
    524288: "512 KiB",
}


def _matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "axes.edgecolor": INK,
            "axes.labelcolor": MUTED,
            "axes.linewidth": 1.0,
            "figure.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    return plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _style(axis: Any, title: str, ylabel: str | None = None) -> None:
    axis.set_title(title, fontsize=12, pad=7)
    axis.grid(True, color=GRID, linewidth=0.8, alpha=0.85)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if ylabel is not None:
        axis.set_ylabel(ylabel)


def _save(figure: Any, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        path = out_dir / f"{stem}.{suffix}"
        options: dict[str, object] = {"dpi": 180} if suffix == "png" else {}
        if suffix == "pdf":
            options["metadata"] = {
                "Creator": "simllm",
                "CreationDate": None,
                "ModDate": None,
            }
        figure.savefig(path, **options)
        paths.append(path)
    return paths


def render_cdf(run_dir: Path, out_dir: Path, frozen: dict[str, Any]) -> list[Path]:
    plt = _matplotlib()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter

    rows = _read_csv(run_dir / "fct-cdf.csv")
    grouped: dict[tuple[int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["size_bytes"]), int(row["degree"]), row["transport"])].append(
            row
        )

    figure, axes = plt.subplots(4, 2, figsize=(8.2, 10.0))
    flat = list(axes.flat)
    sizes = frozen["workload"]["flow_sizes_bytes"]
    for axis, size_bytes in zip(flat, sizes, strict=False):
        _style(axis, SIZE_LABELS[size_bytes], "Empirical CDF")
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(
            LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=5)
        )
        axis.xaxis.set_major_formatter(
            LogFormatterSciNotation(
                base=10,
                labelOnlyBase=False,
                minor_thresholds=(float("inf"), float("inf")),
            )
        )
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_ylim(0, 1.03)
        axis.set_xlabel("FCT (us, log)")
        axis.tick_params(axis="x", labelsize=8.4)
        for degree in (1, 2, 3):
            for transport in ("nvlink-credit", "rnic-nn"):
                curve = grouped[(size_bytes, degree, transport)]
                x = [int(row["fct_ps"]) / 1_000_000 for row in curve]
                mean = [float(row["cdf_mean"]) for row in curve]
                low = [float(row["cdf_min"]) for row in curve]
                high = [float(row["cdf_max"]) for row in curve]
                color = DEGREE_COLORS[degree]
                axis.fill_between(
                    x,
                    low,
                    high,
                    step="post",
                    color=color,
                    alpha=0.055 if transport == "rnic-nn" else 0.09,
                    linewidth=0,
                )
                axis.step(
                    x,
                    mean,
                    where="post",
                    color=color,
                    linestyle=TRANSPORT_STYLES[transport],
                    linewidth=1.35,
                )
    flat[-1].axis("off")
    degree_handles = [
        Line2D([0], [0], color=DEGREE_COLORS[degree], linewidth=2, label=f"incast {degree}")
        for degree in (1, 2, 3)
    ]
    transport_handles = [
        Line2D(
            [0],
            [0],
            color=INK,
            linestyle=TRANSPORT_STYLES[transport],
            linewidth=1.6,
            label=TRANSPORT_LABELS[transport],
        )
        for transport in ("nvlink-credit", "rnic-nn")
    ]
    flat[-1].legend(
        handles=[*degree_handles, *transport_handles],
        loc="center",
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        "Same physical mapping and releases: mean FCT CDF with seed bands",
        fontsize=14,
        y=0.985,
    )
    figure.text(
        0.5,
        0.035,
        "9 frozen seeds; every shade is the pointwise seed min-max.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color=MUTED,
    )
    figure.text(
        0.5,
        0.021,
        "Solid: NVLink credit. Dashed: pinned rnic-nn max-min packet slots; no ACK pacing.",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color=MUTED,
    )
    figure.text(
        0.5,
        0.007,
        "Homogeneous rnic-nn capacity is exact at full incast but can bias transient "
        "rnic-nn service left.",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color=MUTED,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.945,
        bottom=0.085,
        hspace=0.54,
        wspace=0.27,
    )
    paths = _save(figure, out_dir, frozen["plot_contract"]["cdf_stem"])
    plt.close(figure)
    return paths


def render_dispersion(run_dir: Path, out_dir: Path, frozen: dict[str, Any]) -> list[Path]:
    plt = _matplotlib()
    rows = _read_csv(run_dir / "cell-summary.csv")
    by_key = {
        (row["transport"], int(row["degree"]), int(row["size_bytes"])): row
        for row in rows
    }
    sizes = frozen["workload"]["flow_sizes_bytes"]
    figure, axes = plt.subplots(1, 3, figsize=(8.2, 4.8), sharey=True)
    x = list(range(len(sizes)))
    width = 0.36
    for axis, degree in zip(axes, (1, 2, 3), strict=True):
        _style(axis, f"Incast degree {degree}", "Seed p50 width / median (%)" if degree == 1 else None)
        axis.set_yscale("symlog", linthresh=0.1, linscale=0.7)
        axis.set_yticks([0, 0.1, 1, 10, 100])
        values = {
            transport: [
                100 * float(by_key[(transport, degree, size)]["dispersion_ratio"])
                for size in sizes
            ]
            for transport in ("nvlink-credit", "rnic-nn")
        }
        axis.bar(
            [position - width / 2 for position in x],
            values["nvlink-credit"],
            width,
            color="#2878b5",
            alpha=0.82,
            label="NVLink credit",
        )
        axis.bar(
            [position + width / 2 for position in x],
            values["rnic-nn"],
            width,
            color="#e76f51",
            alpha=0.82,
            hatch="//",
            label="rnic-nn",
        )
        axis.set_xticks(x, [SIZE_LABELS[size] for size in sizes], rotation=52, ha="right")
        axis.margins(x=0.025)
        if degree == 1:
            axis.legend(loc="upper right", frameon=False, fontsize=8)
    figure.suptitle(
        "Cross-seed FCT dispersion on the identical stagger schedule",
        fontsize=13.5,
        y=0.98,
    )
    figure.text(
        0.5,
        0.055,
        "D = (max seed p50 - min seed p50) / median seed p50. Lower is tighter.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color=MUTED,
    )
    figure.text(
        0.5,
        0.036,
        "Symmetric-log y scale; 0 to 0.1% is linear. 256 B is one packet: "
        "NVLink 12.194 ns; rnic-nn is 5.440 / 2.720 / 2.628 ns.",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=MUTED,
    )
    figure.text(
        0.5,
        0.017,
        "One NVLink credit round is 64 KiB payload and returns in 200 ns.",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=MUTED,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.99,
        top=0.87,
        bottom=0.29,
        wspace=0.13,
    )
    paths = _save(figure, out_dir, frozen["plot_contract"]["dispersion_stem"])
    plt.close(figure)
    return paths


def render(run_dir: Path, out_dir: Path) -> list[Path]:
    frozen = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    return [
        *render_cdf(run_dir, out_dir, frozen),
        *render_dispersion(run_dir, out_dir, frozen),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    arguments = parser.parse_args()
    paths = render(arguments.run_dir, arguments.out_dir)
    for path in paths:
        print(path.as_posix())


if __name__ == "__main__":
    main()
