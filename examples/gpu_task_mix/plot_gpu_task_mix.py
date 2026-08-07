#!/usr/bin/env python3
"""Render the GPU task-mix bound and concurrency figures from reviewed CSVs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.csv"
NCCL_CONVERGENCE = HERE / "nccl_convergence.csv"
DIAGNOSTICS = HERE / "diagnostics.csv"

COMPUTE = "#D55E00"
MEMORY = "#0072B2"
NETWORK = "#009E73"
PURPLE = "#7B4AB5"
GOLD = "#E69F00"
DARK = "#252525"
LIGHT = "#D7DCE2"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _result(rows: list[dict[str, str]], check: str, case: str) -> dict[str, str]:
    matches = [row for row in rows if row["check"] == check and row["case"] == case]
    if len(matches) != 1:
        raise ValueError(f"expected one {check}/{case} result, found {len(matches)}")
    return matches[0]


def _diagnostic(rows: list[dict[str, str]], name: str) -> int:
    matches = [row for row in rows if row["diagnostic"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one {name!r} diagnostic, found {len(matches)}")
    return int(matches[0]["duration_cycles"])


def _values_by_integer_parameter(
    rows: list[dict[str, str]], check: str, prefix: str
) -> dict[int, tuple[int, int]]:
    values: dict[int, tuple[int, int]] = {}
    pattern = re.compile(rf"^{re.escape(prefix)}=(\d+)$")
    for row in rows:
        if row["check"] != check:
            continue
        match = pattern.fullmatch(row["case"])
        if match:
            parameter = int(match.group(1))
            if parameter in values:
                raise ValueError(f"duplicate {check}/{prefix}={parameter} result")
            values[parameter] = (int(row["expected"]), int(row["measured"]))
    if not values:
        raise ValueError(f"no integer-parameter rows found for {check}/{prefix}")
    return values


def _memory_sweep(
    rows: list[dict[str, str]],
) -> dict[tuple[int, int, int], tuple[int, int]]:
    pattern = re.compile(r"^bytes=(\d+),bw=(\d+),sms=(\d+)$")
    values: dict[tuple[int, int, int], tuple[int, int]] = {}
    for row in rows:
        if row["check"] not in {"B1", "B2"}:
            continue
        match = pattern.fullmatch(row["case"])
        if match is None:
            raise ValueError(f"malformed memory case {row['case']!r}")
        key = tuple(int(value) for value in match.groups())
        if key in values:
            raise ValueError(f"duplicate memory sweep key {key}")
        values[key] = (int(row["expected"]), int(row["measured"]))
    if not values:
        raise ValueError("memory sweep is empty")
    return values


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "legend.title_fontsize": 7.0,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axis(axis: Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color=LIGHT, linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def _save_figure(figure: Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        out / f"{stem}.png",
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "simllm"},
    )
    figure.savefig(
        out / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"Creator": "simllm", "CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def _plot_bound_sweeps(
    results: list[dict[str, str]], convergence: list[dict[str, str]], out: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.85), constrained_layout=True)

    independent = _values_by_integer_parameter(results, "A1", "interval")
    dependent = _values_by_integer_parameter(results, "A2", "interval")
    axis = axes[0]
    x = sorted(independent)
    axis.plot(
        x,
        [independent[value][1] for value in x],
        color=COMPUTE,
        marker="o",
        label="independent, 1 SM",
    )
    x = sorted(dependent)
    dependent_expected = [dependent[value][0] for value in x]
    dependent_measured = [dependent[value][1] for value in x]
    axis.plot(
        x,
        dependent_expected,
        color=DARK,
        linestyle="--",
        label="dependency prediction",
    )
    axis.scatter(
        x,
        dependent_measured,
        color=DARK,
        marker="s",
        facecolor="white",
        zorder=3,
    )
    a3_rows = [row for row in results if row["check"] == "A3"]
    if len(a3_rows) != 1:
        raise ValueError(f"expected one A3 result, found {len(a3_rows)}")
    a3 = a3_rows[0]
    a3_case = re.fullmatch(r"interval=(\d+),sm_count=2", a3["case"])
    if a3_case is None:
        raise ValueError(f"malformed A3 case {a3['case']!r}")
    a3_interval = int(a3_case.group(1))
    two_sm_cycles = int(a3["measured"])
    axis.scatter([a3_interval], [two_sm_cycles], color=COMPUTE, marker="^", zorder=4)
    axis.annotate(
        f"2 SMs: {two_sm_cycles} cycles",
        xy=(a3_interval, two_sm_cycles),
        xytext=(7, -2),
        textcoords="offset points",
        color=COMPUTE,
        fontsize=7.0,
    )
    axis.set_title("(a) Compute-bound", loc="left", fontweight="bold")
    axis.set_xlabel("ALU initiation interval (cycles/instruction)")
    axis.set_ylabel("task makespan (cycles)")
    axis.set_xticks(sorted(set(independent) | set(dependent)))
    axis.set_ylim(0, 36)
    axis.legend(frameon=False, loc="lower right")
    _style_axis(axis)

    memory = _memory_sweep(results)
    sm_counts = sorted({key[2] for key in memory})
    if sm_counts != [1, 2]:
        raise ValueError(f"expected 1-SM and 2-SM memory sweeps, got {sm_counts}")
    axis = axes[1]
    by_service: dict[int, list[tuple[int, int, int, int]]] = {}
    for (transaction, bandwidth, sm_count), (expected, measured) in memory.items():
        service_cycles = (transaction + bandwidth - 1) // bandwidth
        by_service.setdefault(service_cycles, []).append((sm_count, expected, measured, transaction))
    services = sorted(by_service)
    predictions: list[int] = []
    one_sm: list[int] = []
    two_sm: list[int] = []
    for service in services:
        entries = by_service[service]
        expected_values = {entry[1] for entry in entries}
        if len(expected_values) != 1:
            raise ValueError(f"service={service} has inconsistent predictions: {entries}")
        predictions.append(expected_values.pop())
        measured_by_sm = {
            sm_count: {entry[2] for entry in entries if entry[0] == sm_count}
            for sm_count in sm_counts
        }
        if any(len(values) != 1 for values in measured_by_sm.values()):
            raise ValueError(f"service={service} has inconsistent replay values: {entries}")
        one_sm.append(measured_by_sm[1].pop())
        two_sm.append(measured_by_sm[2].pop())
    if one_sm != two_sm or one_sm != predictions:
        raise ValueError("memory replay, SM-count controls, and predictions no longer overlap")
    axis.plot(
        services,
        predictions,
        color=DARK,
        linestyle="--",
        label="closed-form prediction",
        zorder=1,
    )
    axis.scatter(
        services,
        two_sm,
        marker="s",
        s=54,
        facecolor="white",
        edgecolor=PURPLE,
        linewidth=1.5,
        label="2 SM replay",
        zorder=2,
    )
    axis.scatter(
        services,
        one_sm,
        marker="o",
        s=24,
        color=MEMORY,
        label="1 SM replay",
        zorder=3,
    )
    parameter_pairs_by_service = {
        service: {
            (transaction, bandwidth)
            for transaction, bandwidth, _ in memory
            if (transaction + bandwidth - 1) // bandwidth == service
        }
        for service in services
    }
    duplicate_services = [
        service for service, pairs in parameter_pairs_by_service.items() if len(pairs) > 1
    ]
    for service in duplicate_services:
        axis.annotate(
            f"{len(parameter_pairs_by_service[service])} parameter pairs",
            (service, one_sm[services.index(service)]),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=6.7,
            color=MEMORY,
        )
    axis.set_title("(b) Memory-bound", loc="left", fontweight="bold")
    axis.set_xlabel("HBM transaction service (cycles)")
    axis.set_ylabel("task makespan (cycles)")
    axis.set_xticks(services)
    axis.set_ylim(0, max(predictions) * 1.13)
    axis.legend(frameon=False, loc="lower right")
    _style_axis(axis)

    convergence_by_warps: dict[int, dict[str, str]] = {}
    for row in convergence:
        warp_count = int(row["warps_per_channel"])
        if warp_count in convergence_by_warps:
            raise ValueError(f"duplicate NCCL convergence row for {warp_count} warps")
        convergence_by_warps[warp_count] = row
    warps = sorted(convergence_by_warps)
    convergence = [convergence_by_warps[warp_count] for warp_count in warps]
    durations = [int(row["duration_cycles"]) for row in convergence]
    bounds = [
        int(row["duration_cycles"]) - int(row["excess_over_egress_bound"])
        for row in convergence
    ]
    if len(set(bounds)) != 1:
        raise ValueError(f"NCCL convergence rows imply multiple egress bounds: {bounds}")
    egress_bound = bounds[0]
    efficiency = [100.0 * egress_bound / duration for duration in durations]
    axis = axes[2]
    axis.plot(warps, efficiency, color=NETWORK, marker="o", label="ring replay")
    axis.axhline(100.0, color=DARK, linestyle="--", linewidth=1.3, label="ideal egress")
    for index, (warp_count, percent, duration) in enumerate(
        zip(warps, efficiency, durations, strict=True)
    ):
        axis.annotate(
            f"{duration:,}",
            (warp_count, percent),
            xytext=(0, -12 if index == len(warps) - 1 else 5),
            textcoords="offset points",
            ha="center",
            fontsize=6.7,
            color=NETWORK,
        )
    axis.set_xscale("log", base=2)
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.set_xticks(warps)
    axis.set_title("(c) Network-bound", loc="left", fontweight="bold")
    axis.set_xlabel("warps per NCCL channel")
    axis.set_ylabel("NVLink egress efficiency (%)")
    axis.set_ylim(0, 108)
    axis.legend(frameon=False, loc="lower right")
    axis.text(
        0.03,
        0.86,
        f"bound = {egress_bound:,} cycles\npoint labels: replay cycles",
        transform=axis.transAxes,
        va="top",
        fontsize=6.7,
        color=DARK,
    )
    _style_axis(axis)

    figure.suptitle(
        "GPU task-mix model: bound-specific sensitivity", fontsize=10.0, fontweight="bold"
    )
    _save_figure(figure, out, "gpu_task_mix_bounds")


def _plot_concurrency_mechanisms(
    results: list[dict[str, str]], diagnostics: list[dict[str, str]], out: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), constrained_layout=True)

    d2 = _result(results, "D2", "memory beside network")
    axis = axes[0]
    d3 = _result(results, "D3", "compute beside memory,half-SM demand")
    ring_cycles = _diagnostic(diagnostics, "E ring isolated")
    d2_memory = int(_result(results, "B2", "bytes=64,bw=64,sms=2")["measured"])
    d2_network = int(_result(results, "C2", "chunk=64,bw=16")["measured"])
    d2_control = d2_network
    overlap_cases = (
        ("D2 issue\nsharing", d2_memory, d2_network, int(d2["measured"])),
        (
            "D3 half-SM\ndemand",
            _diagnostic(diagnostics, "compute isolated, half-SM shared-memory demand"),
            _diagnostic(diagnostics, "memory isolated, half-SM shared-memory demand"),
            int(d3["measured"]),
        ),
        (
            "D3 no SMEM\ndemand",
            _diagnostic(diagnostics, "compute isolated, no shared-memory pressure"),
            _diagnostic(diagnostics, "memory isolated, no shared-memory pressure"),
            _diagnostic(diagnostics, "compute beside memory, no shared-memory pressure"),
        ),
        (
            "E ring +\nmemory",
            _diagnostic(diagnostics, "E memory isolated"),
            ring_cycles,
            _diagnostic(diagnostics, "E ring beside memory"),
        ),
    )
    hidden_percent = [
        100.0 * (task_a + task_b - mixed) / min(task_a, task_b)
        for _, task_a, task_b, mixed in overlap_cases
    ]
    if any(percent < 0.0 or percent > 100.0 for percent in hidden_percent):
        raise ValueError(f"shorter-task hidden fractions are out of range: {hidden_percent}")
    axis.bar(
        range(len(overlap_cases)),
        hidden_percent,
        color=[GOLD, PURPLE, "#B99ADC", NETWORK],
        width=0.62,
    )
    axis.axhline(100.0, color=DARK, linestyle="--", linewidth=1.0, label="fully hidden")
    for index, (_, task_a, task_b, mixed) in enumerate(overlap_cases):
        percent = hidden_percent[index]
        if percent == 0.0:
            axis.text(
                index,
                4.0,
                f"0.0%\n{task_a:,}+{task_b:,} → {mixed:,} cycles",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=DARK,
            )
            continue
        axis.annotate(
            f"{percent:.1f}%",
            (index, percent),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )
        axis.text(
            index,
            percent * 0.52,
            f"{task_a:,}+{task_b:,}\n→ {mixed:,}\ncycles",
            ha="center",
            va="center",
            fontsize=6.2,
            color="white",
        )
    axis.set_title("(a) Shorter-task work hidden", loc="left", fontweight="bold")
    axis.set_ylabel("hidden fraction, H (%)")
    axis.set_xticks(range(len(overlap_cases)), [label for label, *_ in overlap_cases])
    axis.set_ylim(0, 110)
    axis.legend(frameon=False, loc="lower right")
    _style_axis(axis)

    issue_cases = (
        ("memory first", int(d2["measured"])),
        ("network first", _diagnostic(diagnostics, "network submitted first, memory second")),
        ("2x sched.", _diagnostic(diagnostics, "D2 scheduler budget doubled")),
        ("2x LD/ST", _diagnostic(diagnostics, "D2 load-store lanes doubled")),
        (
            "both 2x",
            _diagnostic(diagnostics, "D2 scheduler budget and load-store lanes doubled"),
        ),
    )
    issue_delays = [duration - d2_control for _, duration in issue_cases]
    if any(delay < 0 for delay in issue_delays):
        raise ValueError(f"D2 controls fell below the registered control: {issue_delays}")
    axis = axes[1]
    positions = list(range(len(issue_cases)))
    axis.vlines(positions, 0, issue_delays, color=GOLD, linewidth=2.0)
    axis.scatter(
        positions,
        issue_delays,
        s=42,
        color=[GOLD if delay else LIGHT for delay in issue_delays],
        edgecolor=DARK,
        linewidth=0.7,
        zorder=3,
    )
    for position, delay in zip(positions, issue_delays, strict=True):
        axis.annotate(
            str(delay),
            (position, delay),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=7.0,
        )
    one_cycle_percent = 100.0 / d2_control
    axis.text(
        0.98,
        0.95,
        f"historical registration = {d2_control} cycles\n"
        f"zoomed excess: 1 cycle = {one_cycle_percent:.3f}%",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color=DARK,
    )
    axis.set_title("(b) D2 shared issue path", loc="left", fontweight="bold")
    axis.set_ylabel("excess over historical registration (cycles)")
    axis.set_xticks(positions, [label for label, _ in issue_cases], rotation=22, ha="right")
    axis.set_ylim(0, max(issue_delays) + 0.35)
    axis.set_yticks(sorted(set(issue_delays)))
    _style_axis(axis)

    figure.suptitle(
        "GPU task-mix model: concurrency mechanisms", fontsize=10.0, fontweight="bold"
    )
    _save_figure(figure, out, "gpu_task_mix_concurrency")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "plots")
    args = parser.parse_args()

    _configure_matplotlib()
    results = _read_rows(RESULTS)
    convergence = _read_rows(NCCL_CONVERGENCE)
    diagnostics = _read_rows(DIAGNOSTICS)
    _plot_bound_sweeps(results, convergence, args.out)
    _plot_concurrency_mechanisms(results, diagnostics, args.out)
    print(f"wrote two figures as PNG and PDF to {args.out}")


if __name__ == "__main__":
    main()
