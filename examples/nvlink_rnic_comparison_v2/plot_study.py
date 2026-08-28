#!/usr/bin/env python3
"""Render the frozen TRAF-72 CDF, tail, fairness, and audit figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXPECTATIONS_PATH = HERE / "expectations.json"
LEGACY_RESULT_PATH = HERE.parent / "nvlink_rnic_comparison_v1" / "results.json"

INK = "#202124"
MUTED = "#666666"
GRID = "#deded9"
SURFACE = "#fcfcfa"
TRANSPORT_COLORS = {
    "nvlink-credit": "#2878b5",
    "rnic-nn": "#e76f51",
    "rnic-nn-fluid": "#2a9d8f",
}
TRANSPORT_STYLES = {
    "nvlink-credit": "-",
    "rnic-nn": "--",
    "rnic-nn-fluid": ":",
}
TRANSPORT_LABELS = {
    "nvlink-credit": "NVLink credit",
    "rnic-nn": "rnic-nn packet",
    "rnic-nn-fluid": "rnic-nn fluid",
}
DEGREE_COLORS = {
    1: "#264653",
    2: "#2878b5",
    3: "#2a9d8f",
    4: "#e9c46a",
    8: "#f4a261",
    16: "#e76f51",
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
EVIDENCE_LINE = (
    "Evidence: SIMULATED FCT; MEASURED endpoint plateaus; DECLARED packet and "
    "credit constants; STRUCTURAL NV4 switch identity."
)


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
            "font.size": 9.2,
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
    axis.set_title(title, fontsize=11, pad=6)
    axis.grid(True, color=GRID, linewidth=0.75, alpha=0.85)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if ylabel is not None:
        axis.set_ylabel(ylabel)


def _save(figure: Any, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        path = out_dir / f"{stem}.{suffix}"
        options: dict[str, object] = {"dpi": 190} if suffix == "png" else {}
        if suffix == "pdf":
            options["metadata"] = {
                "Creator": "simllm",
                "CreationDate": None,
                "ModDate": None,
            }
        figure.savefig(path, **options)
        paths.append(path)
    return paths


def _footer(
    figure: Any,
    frozen: dict[str, Any],
    *,
    mesh: bool,
    small_flow: bool,
    start_y: float = 0.054,
) -> None:
    lines = [EVIDENCE_LINE]
    if mesh:
        lines.append(frozen["topology"]["required_figure_disclosure"])
    if small_flow:
        lines.append(frozen["measurement_caveat"]["required_figure_disclosure"])
    for index, line in enumerate(lines):
        figure.text(
            0.5,
            start_y - index * 0.015,
            line,
            ha="center",
            va="bottom",
            fontsize=6.6,
            color=MUTED,
        )


def render_cdf(
    run_dir: Path,
    out_dir: Path,
    frozen: dict[str, Any],
    degrees: tuple[int, int, int],
    stem: str,
    title: str,
    *,
    mesh: bool,
) -> list[Path]:
    plt = _matplotlib()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter

    rows = _read_csv(run_dir / "fct-cdf.csv")
    grouped: dict[tuple[int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["size_bytes"]), int(row["degree"]), row["transport"])].append(
            row
        )
    figure, axes = plt.subplots(4, 2, figsize=(8.3, 10.5))
    flat = list(axes.flat)
    sizes = frozen["workload"]["flow_sizes_bytes"]
    for axis, size_bytes in zip(flat, sizes, strict=False):
        _style(axis, SIZE_LABELS[size_bytes], "Empirical CDF")
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(
            LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=6)
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
        axis.tick_params(axis="x", labelsize=8)
        for degree in degrees:
            for transport in TRANSPORT_LABELS:
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
                    alpha=0.035,
                    linewidth=0,
                )
                axis.step(
                    x,
                    mean,
                    where="post",
                    color=color,
                    linestyle=TRANSPORT_STYLES[transport],
                    linewidth=1.25,
                )
    flat[-1].axis("off")
    degree_handles = [
        Line2D(
            [0],
            [0],
            color=DEGREE_COLORS[degree],
            linewidth=2,
            label=f"incast {degree}",
        )
        for degree in degrees
    ]
    transport_handles = [
        Line2D(
            [0],
            [0],
            color=INK,
            linestyle=TRANSPORT_STYLES[transport],
            linewidth=1.6,
            label=label,
        )
        for transport, label in TRANSPORT_LABELS.items()
    ]
    flat[-1].legend(
        handles=[*degree_handles, *transport_handles],
        loc="center",
        frameon=False,
        fontsize=8.5,
    )
    figure.suptitle(title, fontsize=13.5, y=0.986)
    figure.text(
        0.5,
        0.071,
        "Nine frozen seeds; every shade is the pointwise seed min-max band.",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=MUTED,
    )
    _footer(figure, frozen, mesh=mesh, small_flow=True)
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.95,
        bottom=0.112 if mesh else 0.098,
        hspace=0.53,
        wspace=0.27,
    )
    paths = _save(figure, out_dir, stem)
    plt.close(figure)
    return paths


def render_tail(run_dir: Path, out_dir: Path, frozen: dict[str, Any]) -> list[Path]:
    plt = _matplotlib()
    from matplotlib.lines import Line2D

    rows = _read_csv(run_dir / "tail-metrics.csv")
    by_key = {
        (row["transport"], int(row["degree"]), int(row["size_bytes"])): row
        for row in rows
    }
    sizes = frozen["workload"]["flow_sizes_bytes"]
    degrees = frozen["workload"]["degrees"]
    x = list(range(len(degrees)))
    figure, axes = plt.subplots(4, 2, figsize=(8.3, 10.4))
    flat = list(axes.flat)
    for axis, size_bytes in zip(flat, sizes, strict=False):
        _style(axis, SIZE_LABELS[size_bytes], "FCT (us, log)")
        axis.set_yscale("log")
        axis.set_xticks(x, [str(degree) for degree in degrees])
        axis.set_xlabel("Incast degree")
        for transport in TRANSPORT_LABELS:
            color = TRANSPORT_COLORS[transport]
            for metric, style, marker in (
                ("p99", "-", "o"),
                ("worst", "--", "s"),
            ):
                mean = [
                    float(by_key[(transport, degree, size_bytes)][f"{metric}_seed_mean_ps"])
                    / 1_000_000
                    for degree in degrees
                ]
                low = [
                    float(by_key[(transport, degree, size_bytes)][f"{metric}_seed_min_ps"])
                    / 1_000_000
                    for degree in degrees
                ]
                high = [
                    float(by_key[(transport, degree, size_bytes)][f"{metric}_seed_max_ps"])
                    / 1_000_000
                    for degree in degrees
                ]
                axis.fill_between(x, low, high, color=color, alpha=0.055, linewidth=0)
                axis.plot(
                    x,
                    mean,
                    color=color,
                    linestyle=style,
                    marker=marker,
                    markersize=3.2,
                    linewidth=1.25,
                )
    flat[-1].axis("off")
    transport_handles = [
        Line2D([0], [0], color=color, linewidth=2, label=TRANSPORT_LABELS[transport])
        for transport, color in TRANSPORT_COLORS.items()
    ]
    metric_handles = [
        Line2D([0], [0], color=INK, linestyle="-", marker="o", label="p99"),
        Line2D([0], [0], color=INK, linestyle="--", marker="s", label="worst flow"),
    ]
    flat[-1].legend(
        handles=[*transport_handles, *metric_handles],
        loc="center",
        frameon=False,
        fontsize=8.5,
    )
    figure.suptitle("Tail FCT versus incast degree", fontsize=13.5, y=0.986)
    figure.text(
        0.5,
        0.071,
        "Nearest-rank p99 and per-seed maximum; lines are seed means and shades are min-max.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=MUTED,
    )
    _footer(figure, frozen, mesh=True, small_flow=True)
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.95,
        bottom=0.112,
        hspace=0.53,
        wspace=0.27,
    )
    paths = _save(figure, out_dir, frozen["plot_contract"]["tail_stem"])
    plt.close(figure)
    return paths


def render_fairness(run_dir: Path, out_dir: Path, frozen: dict[str, Any]) -> list[Path]:
    plt = _matplotlib()

    rows = _read_csv(run_dir / "fairness.csv")
    by_key = {
        (row["transport"], int(row["degree"]), int(row["size_bytes"])): row
        for row in rows
    }
    sizes = frozen["workload"]["flow_sizes_bytes"]
    degrees = frozen["workload"]["degrees"]
    x = list(range(len(degrees)))
    figure, axes = plt.subplots(4, 2, figsize=(8.3, 10.4))
    flat = list(axes.flat)
    for axis, size_bytes in zip(flat, sizes, strict=False):
        _style(axis, SIZE_LABELS[size_bytes], "Jain fairness")
        axis.set_ylim(0.82, 1.005)
        axis.set_xticks(x, [str(degree) for degree in degrees])
        axis.set_xlabel("Incast degree")
        for transport, label in TRANSPORT_LABELS.items():
            mean = [
                float(by_key[(transport, degree, size_bytes)]["jain_seed_mean"])
                for degree in degrees
            ]
            low = [
                float(by_key[(transport, degree, size_bytes)]["jain_seed_min"])
                for degree in degrees
            ]
            high = [
                float(by_key[(transport, degree, size_bytes)]["jain_seed_max"])
                for degree in degrees
            ]
            color = TRANSPORT_COLORS[transport]
            axis.fill_between(x, low, high, color=color, alpha=0.07, linewidth=0)
            axis.plot(
                x,
                mean,
                color=color,
                linestyle=TRANSPORT_STYLES[transport],
                marker="o",
                markersize=3.2,
                linewidth=1.3,
                label=label,
            )
    flat[-1].axis("off")
    handles, labels = flat[0].get_legend_handles_labels()
    flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=8.5)
    figure.suptitle("Fairness across concurrently released senders", fontsize=13.5, y=0.986)
    figure.text(
        0.5,
        0.071,
        "Jain index of per-flow goodput within each release wave; lines are wave and seed means.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=MUTED,
    )
    _footer(figure, frozen, mesh=True, small_flow=True)
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.95,
        bottom=0.112,
        hspace=0.53,
        wspace=0.27,
    )
    paths = _save(figure, out_dir, frozen["plot_contract"]["fairness_stem"])
    plt.close(figure)
    return paths


def render_mapping_audit(
    run_dir: Path, out_dir: Path, frozen: dict[str, Any]
) -> list[Path]:
    plt = _matplotlib()

    legacy = json.loads(LEGACY_RESULT_PATH.read_text(encoding="utf-8"))
    current = _read_csv(run_dir / "tail-metrics.csv")
    sizes = frozen["workload"]["flow_sizes_bytes"]
    x = list(range(len(sizes)))
    legacy_by_key = {
        (row["transport"], int(row["size_bytes"])): row
        for row in legacy["cell_summaries"]
        if int(row["degree"]) == 3
    }
    current_by_key = {
        (row["transport"], int(row["size_bytes"])): row
        for row in current
        if int(row["degree"]) == 3
    }
    figure, axes = plt.subplots(1, 2, figsize=(8.3, 4.9))
    left, right = axes
    _style(left, "Degree-3 p50 before and after", "Mean seed p50 (us, log)")
    left.set_yscale("log")
    left.set_xticks(x, [SIZE_LABELS[size] for size in sizes], rotation=45, ha="right")
    for transport, label, color in (
        ("nvlink-credit", "TRAF-71 NVLink", "#8bb8d8"),
        ("rnic-nn", "TRAF-71 rnic-nn per transfer", "#d9a18f"),
    ):
        values = [
            float(legacy_by_key[(transport, size)]["p50_seed_mean_ps"]) / 1_000_000
            for size in sizes
        ]
        left.plot(x, values, color=color, linestyle="--", marker="x", label=label)
    for transport, transport_label in TRANSPORT_LABELS.items():
        values = [
            float(current_by_key[(transport, size)]["p50_seed_mean_ps"]) / 1_000_000
            for size in sizes
        ]
        left.plot(
            x,
            values,
            color=TRANSPORT_COLORS[transport],
            linestyle=TRANSPORT_STYLES[transport],
            marker="o",
            markersize=3.5,
            label=f"TRAF-72 {transport_label}",
        )
    left.legend(loc="upper left", frameon=False, fontsize=6.9)

    _style(right, "Legacy rnic-nn / corrected rnic-nn", "p50 ratio")
    ratios = [
        float(legacy_by_key[("rnic-nn", size)]["p50_seed_mean_ps"])
        / float(current_by_key[("rnic-nn", size)]["p50_seed_mean_ps"])
        for size in sizes
    ]
    right.plot(x, ratios, color=TRANSPORT_COLORS["rnic-nn"], marker="o", linewidth=1.5)
    right.axhline(
        frozen["mapping_audit"]["normalized_queue_arithmetic"][
            "predicted_ratio_decimal"
        ],
        color=INK,
        linestyle=":",
        linewidth=1.1,
        label="frozen 601/360 prediction",
    )
    right.axhline(1, color=GRID, linewidth=1)
    right.set_xticks(x, [SIZE_LABELS[size] for size in sizes], rotation=45, ha="right")
    right.legend(loc="best", frameon=False, fontsize=7.2)
    figure.suptitle(
        "Mapping audit: same capacity, corrected fair-share entity",
        fontsize=13.5,
        y=0.98,
    )
    figure.text(
        0.5,
        0.075,
        "Degree-3 receiver capacity is 207.101921876 GB/s in both studies; only the active entity changes.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=MUTED,
    )
    _footer(figure, frozen, mesh=False, small_flow=True, start_y=0.054)
    figure.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.27, wspace=0.28)
    paths = _save(figure, out_dir, frozen["plot_contract"]["mapping_audit_stem"])
    plt.close(figure)
    return paths


def render(run_dir: Path, out_dir: Path) -> list[Path]:
    frozen = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    return [
        *render_cdf(
            run_dir,
            out_dir,
            frozen,
            (1, 2, 3),
            frozen["plot_contract"]["cdf_physical_stem"],
            "Physical NV4 degrees: corrected mapping and fluid null",
            mesh=False,
        ),
        *render_cdf(
            run_dir,
            out_dir,
            frozen,
            (4, 8, 16),
            frozen["plot_contract"]["cdf_mesh_stem"],
            "Simulated incast mesh: corrected mapping and fluid null",
            mesh=True,
        ),
        *render_tail(run_dir, out_dir, frozen),
        *render_fairness(run_dir, out_dir, frozen),
        *render_mapping_audit(run_dir, out_dir, frozen),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    arguments = parser.parse_args()
    for path in render(arguments.run_dir, arguments.out_dir):
        print(path.as_posix())


if __name__ == "__main__":
    main()
