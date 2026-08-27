#!/usr/bin/env python3
"""Render the frozen TRAF-69 figures from one external run directory."""

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
FLOW_COLORS = ("#2878b5", "#e76f51", "#2a9d8f")
SIZE_COLORS = ("#264653", "#2a9d8f", "#72a93b", "#e9c46a", "#f4a261", "#e76f51", "#8f5da2")
DISCLOSURE = (
    "Mixed evidence: TX/RX plateaus measured; packet, link, bond, credit, buffer and "
    "queue terms candidates; NV4 switch structural."
)


def _matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _style(axis: Any, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, color=INK, pad=5)
    axis.set_xlabel(xlabel, color=MUTED)
    axis.set_ylabel(ylabel, color=MUTED)
    axis.set_axisbelow(True)
    axis.grid(color=GRID, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(colors=MUTED)


def _footer(figure: Any, text: str, *, y: float = 0.012) -> None:
    figure.text(0.5, y, text, ha="center", va="bottom", fontsize=7.2, color=MUTED)


def _group_rates(rows: list[dict[str, str]]) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["flow_id"]].append(
            (float(row["bin_start_ps"]), float(row["payload_gbps"]))
        )
    return grouped


def _plot_rate_series(
    axis: Any,
    rows: list[dict[str, str]],
    *,
    origin_ps: float,
    unit_ps: float,
    labels: dict[str, str] | None = None,
) -> None:
    for index, (flow_id, points) in enumerate(sorted(_group_rates(rows).items())):
        ordered = sorted(points)
        axis.step(
            [(time_ps - origin_ps) / unit_ps for time_ps, _ in ordered],
            [rate for _, rate in ordered],
            where="post",
            color=FLOW_COLORS[index % len(FLOW_COLORS)],
            linewidth=1.15,
            label=(labels or {}).get(flow_id, flow_id),
        )


def render_flow_dynamics(
    *,
    run_dir: Path,
    out_dir: Path,
    frozen: dict[str, Any],
    result: dict[str, Any],
) -> list[Path]:
    plt = _matplotlib()
    overall = _read_csv(run_dir / "overall-rate.csv")
    convergence = _read_csv(run_dir / "convergence-rate.csv")
    divergence = _read_csv(run_dir / "divergence-rate.csv")
    figure, axes = plt.subplots(3, 1, figsize=(7.0, 8.35))
    figure.patch.set_facecolor(SURFACE)
    for axis in axes:
        axis.set_facecolor(SURFACE)

    _plot_rate_series(axes[0], overall, origin_ps=0, unit_ps=1_000_000)
    for release_ps in frozen["flow_schedule"]["release_ps"][1:]:
        axes[0].axvline(release_ps / 1_000_000, color=MUTED, linestyle=":", linewidth=0.8)
    for active, center in ((1, 94.117647), (2, 47.058824), (3, 31.372549)):
        axes[0].axhline(center, color=GRID, linestyle="--", linewidth=0.65)
        axes[0].text(
            axes[0].get_xlim()[1],
            center,
            f" {active}-flow share",
            va="center",
            fontsize=6.8,
            color=MUTED,
        )
    _style(
        axes[0],
        "Three flows join on a fixed stagger; reverse targets make flow A finish last",
        "receiver time (us)",
        "payload GB/s",
    )
    axes[0].legend(frameon=False, loc="upper right", ncol=3)

    join_ps = frozen["convergence_1_to_2"]["join_ps"]
    _plot_rate_series(
        axes[1],
        convergence,
        origin_ps=join_ps,
        unit_ps=1000,
        labels={"incumbent": "incumbent", "joiner": "joiner"},
    )
    observed_open = result["convergence_1_to_2"]["observed_open_ps"]
    axes[1].axvline(0, color=MUTED, linestyle=":", linewidth=0.8)
    axes[1].axvline(observed_open / 1000, color=INK, linestyle="--", linewidth=0.9)
    axes[1].axhline(47.058824, color=GRID, linestyle="--", linewidth=0.7)
    _style(
        axes[1],
        f"1 to 2: incumbent steps down; open at {observed_open / 1000:.3f} ns, exact",
        "time from join (ns)",
        "raw-bin payload GB/s",
    )
    axes[1].legend(frameon=False, loc="upper right")

    departure_ps = result["divergence_2_to_1"]["departing_completion_ps"]
    _plot_rate_series(
        axes[2],
        divergence,
        origin_ps=departure_ps,
        unit_ps=1000,
        labels={"remaining": "remaining", "departing": "departing"},
    )
    observed_target = result["divergence_2_to_1"]["observed_time_to_target_ps"]
    axes[2].axvline(0, color=MUTED, linestyle=":", linewidth=0.8)
    axes[2].axvline(observed_target / 1000, color=INK, linestyle="--", linewidth=0.9)
    axes[2].axhline(94.117647, color=GRID, linestyle="--", linewidth=0.7)
    _style(
        axes[2],
        f"2 to 1: remaining flow returns to solo cadence at {observed_target / 1000:.3f} ns",
        "time from other flow completion (ns)",
        "raw-bin payload GB/s",
    )
    axes[2].legend(frameon=False, loc="upper right")
    figure.suptitle(
        "NV4 packet flow dynamics through TX, pass-through switch and RX",
        fontsize=11,
        color=INK,
        y=0.985,
    )
    _footer(
        figure,
        "Raw fixed bins, no smoothing. " + DISCLOSURE,
    )
    figure.tight_layout(rect=(0.04, 0.045, 0.98, 0.965))
    paths = []
    for suffix in ("pdf", "png"):
        path = out_dir / f"nvlink-flow-dynamics.{suffix}"
        figure.savefig(path, dpi=220 if suffix == "png" else None)
        paths.append(path)
    plt.close(figure)
    return paths


def _cdf_groups(
    rows: list[dict[str, str]],
) -> dict[tuple[int, int], list[tuple[float, float, float, float]]]:
    grouped: dict[tuple[int, int], list[tuple[float, float, float, float]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["degree"]), int(row["size_bytes"]))].append(
            (
                float(row["fct_ps"]),
                float(row["cdf_mean"]),
                float(row["cdf_min"]),
                float(row["cdf_max"]),
            )
        )
    return grouped


def render_fct_ladder(
    *, run_dir: Path, out_dir: Path, frozen: dict[str, Any]
) -> list[Path]:
    plt = _matplotlib()
    grouped = _cdf_groups(_read_csv(run_dir / "fct-cdf.csv"))
    sizes = frozen["fct_cdf"]["flow_sizes_bytes"]
    figure, axes = plt.subplots(4, 2, figsize=(7.0, 8.8))
    figure.patch.set_facecolor(SURFACE)
    flat = list(axes.flat)
    for index, size_bytes in enumerate(sizes):
        axis = flat[index]
        axis.set_facecolor(SURFACE)
        for degree, color in zip((1, 2, 3), FLOW_COLORS, strict=True):
            points = sorted(grouped[(degree, size_bytes)])
            x = [point[0] / 1_000_000 for point in points]
            axis.fill_between(
                x,
                [point[2] for point in points],
                [point[3] for point in points],
                color=color,
                alpha=0.16,
                linewidth=0,
            )
            axis.plot(
                x,
                [point[1] for point in points],
                color=color,
                linewidth=1.25,
                label=f"incast {degree}",
            )
        axis.set_xscale("log")
        label = f"{size_bytes // 1024} KiB" if size_bytes >= 1024 else f"{size_bytes} B"
        _style(axis, label, "FCT (us, log)", "empirical CDF")
        axis.set_ylim(0, 1.03)
        if index == 0:
            axis.legend(frameon=False, loc="lower right")
    flat[-1].axis("off")
    figure.suptitle(
        "NV4 flow-completion-time ladder, mean empirical CDF with seed jitter",
        fontsize=11,
        color=INK,
        y=0.985,
    )
    _footer(
        figure,
        "9 frozen seeds; shaded band is pointwise seed min-max. " + DISCLOSURE,
    )
    figure.tight_layout(rect=(0.03, 0.045, 0.99, 0.96))
    paths = []
    for suffix in ("pdf", "png"):
        path = out_dir / f"nvlink-fct-cdf.{suffix}"
        figure.savefig(path, dpi=220 if suffix == "png" else None)
        paths.append(path)
    plt.close(figure)
    return paths


def render_incast(
    *,
    degree: int,
    run_dir: Path,
    out_dir: Path,
    frozen: dict[str, Any],
    result: dict[str, Any],
) -> list[Path]:
    plt = _matplotlib()
    rates = _read_csv(run_dir / f"incast-degree-{degree}-rate.csv")
    grouped = _cdf_groups(_read_csv(run_dir / "fct-cdf.csv"))
    figure, axes = plt.subplots(2, 1, figsize=(7.0, 6.2))
    figure.patch.set_facecolor(SURFACE)
    for axis in axes:
        axis.set_facecolor(SURFACE)
    _plot_rate_series(axes[0], rates, origin_ps=0, unit_ps=1_000_000)
    incast_row = next(row for row in result["incast"] if row["degree"] == degree)
    per_flow_ceiling = incast_row["payload_ceiling_gbps"] / degree
    axes[0].axhline(per_flow_ceiling, color=INK, linestyle="--", linewidth=0.8)
    _style(
        axes[0],
        (
            f"Degree {degree} schedule: aggregate {incast_row['simulated_payload_gbps']:.3f} "
            f"vs ceiling {incast_row['payload_ceiling_gbps']:.3f} GB/s"
        ),
        "receiver time (us)",
        "raw-bin payload GB/s per flow",
    )
    axes[0].legend(frameon=False, loc="upper right", ncol=min(degree, 3))

    for index, size_bytes in enumerate(frozen["fct_cdf"]["flow_sizes_bytes"]):
        points = sorted(grouped[(degree, size_bytes)])
        x = [point[0] / 1_000_000 for point in points]
        color = SIZE_COLORS[index]
        axes[1].fill_between(
            x,
            [point[2] for point in points],
            [point[3] for point in points],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        label = f"{size_bytes // 1024} KiB" if size_bytes >= 1024 else f"{size_bytes} B"
        axes[1].plot(
            x,
            [point[1] for point in points],
            color=color,
            linewidth=1.15,
            label=label,
        )
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1.03)
    _style(
        axes[1],
        f"Degree {degree} FCT CDFs across the frozen size ladder",
        "FCT (us, log)",
        "empirical CDF",
    )
    axes[1].legend(frameon=False, loc="lower right", ncol=2)
    binding = "measured RX ingress" if degree == 3 else "candidate ordered-pair links"
    figure.suptitle(
        f"NV4 incast degree {degree}, ceiling owner: {binding}",
        fontsize=11,
        color=INK,
        y=0.985,
    )
    _footer(
        figure,
        "Raw fixed bins, no smoothing. 9 seeds; CDF shade is pointwise min-max. "
        + DISCLOSURE,
    )
    figure.tight_layout(rect=(0.03, 0.055, 0.99, 0.955))
    paths = []
    for suffix in ("pdf", "png"):
        path = out_dir / f"nvlink-incast-degree-{degree}.{suffix}"
        figure.savefig(path, dpi=220 if suffix == "png" else None)
        paths.append(path)
    plt.close(figure)
    return paths


def render(run_dir: Path, out_dir: Path) -> list[Path]:
    frozen = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    paths.extend(
        render_flow_dynamics(
            run_dir=run_dir,
            out_dir=out_dir,
            frozen=frozen,
            result=result,
        )
    )
    paths.extend(render_fct_ladder(run_dir=run_dir, out_dir=out_dir, frozen=frozen))
    for degree in (1, 2, 3):
        paths.extend(
            render_incast(
                degree=degree,
                run_dir=run_dir,
                out_dir=out_dir,
                frozen=frozen,
                result=result,
            )
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    paths = render(arguments.run_dir, arguments.out)
    for path in paths:
        print(path.as_posix())


if __name__ == "__main__":
    main()
