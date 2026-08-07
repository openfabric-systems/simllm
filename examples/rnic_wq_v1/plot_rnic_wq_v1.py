#!/usr/bin/env python3
"""Render selected outcomes from the native RNIC work-queue regression sweeps."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.csv"
PLOTS = HERE / "plots"

MEASURED_VS_EXPECTED = (
    ("doorbells", "expected_doorbells"),
    ("cqes", "expected_cqes"),
    ("network_busy", "expected_network_busy"),
    ("jct_ps", "expected_jct_ps"),
)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
FROZEN = "#202020"


def _read_rows() -> list[dict[str, str]]:
    with RESULTS.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _frozen_line(axis, x, y, label: str = "frozen closed form") -> None:
    axis.plot(x, y, linestyle="--", color=FROZEN, linewidth=1.4, alpha=0.85, zorder=5, label=label)


def _plot_jct_vs_batch(axis, rows: list[dict[str, str]]) -> None:
    by_signal: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_signal[int(row["signal_every"])].append(row)

    # Nested marker sizes so three exactly coincident curves stay visible.
    styles = ((BLUE, "o", 13), (ORANGE, "s", 8), (GREEN, "^", 4))
    for (color, marker, size), (signal, values) in zip(
        styles, sorted(by_signal.items()), strict=True
    ):
        values.sort(key=lambda row: int(row["doorbell_batch"]))
        axis.plot(
            [int(row["doorbell_batch"]) for row in values],
            [int(row["jct_ps"]) for row in values],
            marker=marker,
            markersize=size,
            color=color,
            linewidth=1.0,
            alpha=0.9,
            label=f"signal every {signal}",
        )

    reference = sorted(by_signal[1], key=lambda row: int(row["doorbell_batch"]))
    _frozen_line(
        axis,
        [int(row["doorbell_batch"]) for row in reference],
        [int(row["expected_jct_ps"]) for row in reference],
    )

    spread = max(
        max(int(row["jct_ps"]) for row in rows if int(row["doorbell_batch"]) == batch)
        - min(int(row["jct_ps"]) for row in rows if int(row["doorbell_batch"]) == batch)
        for batch in {int(row["doorbell_batch"]) for row in rows}
    )
    axis.set_title("JCT is set by doorbell batching alone")
    axis.set_xlabel("doorbell batch size B")
    axis.set_ylabel("JCT (ps)")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=10)
    axis.set_xticks([1, 4, 16], labels=["1", "4", "16"])
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    axis.annotate(
        f"all three signaling curves coincide\n(spread across S = {spread} ps)",
        xy=(0.04, 0.06),
        xycoords="axes fraction",
        fontsize=8,
        color="#404040",
    )


def _plot_separation(axis, rows: list[dict[str, str]]) -> None:
    """CQE traffic follows the signaling interval and ignores batching."""

    batches = sorted({int(row["doorbell_batch"]) for row in rows})
    signals = sorted({int(row["signal_every"]) for row in rows})
    cqes = {
        (int(row["doorbell_batch"]), int(row["signal_every"])): int(row["cqes"]) for row in rows
    }

    width = 0.26
    for offset, (color, signal) in enumerate(zip((BLUE, ORANGE, GREEN), signals, strict=True)):
        positions = [index + (offset - 1) * width for index in range(len(batches))]
        heights = [cqes[(batch, signal)] for batch in batches]
        axis.bar(
            positions,
            heights,
            width=width,
            color=color,
            alpha=0.88,
            label=f"S = {signal}",
        )
        for position, height in zip(positions, heights, strict=True):
            axis.annotate(
                str(height),
                (position, height),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=7,
            )

    axis.set_title("CQE traffic follows signaling, not batching")
    axis.set_xlabel("doorbell batch size B")
    axis.set_ylabel("CQEs over 32 WQEs")
    axis.set_xticks(range(len(batches)), labels=[str(batch) for batch in batches])
    axis.margins(y=0.2)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8, ncols=3, loc="upper center")


def _plot_backpressure(axis, rows: list[dict[str, str]], column: str, expected: str, title: str,
                       y_label: str, color: str) -> None:
    rows = sorted(rows, key=lambda row: int(row["network_capacity"]))
    labels = [row["network_capacity"] for row in rows]
    positions = range(len(rows))
    measured = [int(row[column]) for row in rows]
    frozen = [int(row[expected]) for row in rows]

    axis.bar(positions, measured, width=0.55, color=color, alpha=0.85, label="measured")
    axis.plot(
        positions,
        frozen,
        linestyle="none",
        marker="_",
        markersize=34,
        markeredgewidth=2.0,
        color=FROZEN,
        label="frozen closed form",
    )
    for position, value in zip(positions, measured, strict=True):
        axis.annotate(
            f"{value:,}",
            (position, value),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
        )

    axis.set_title(title)
    axis.set_xlabel("network in-flight capacity C")
    axis.set_ylabel(y_label)
    axis.set_xticks(list(positions), labels=labels)
    axis.margins(y=0.24)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8, loc="upper right", handlelength=1.1)


def main() -> None:
    rows = _read_rows()
    sweep_a = [row for row in rows if row["sweep"] == "doorbell_signaling"]
    sweep_b = [row for row in rows if row["sweep"] == "network_backpressure"]
    if not sweep_a or not sweep_b:
        raise SystemExit("results.csv is missing one of the two registered sweeps")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
    _plot_jct_vs_batch(axes[0][0], sweep_a)
    _plot_separation(axes[0][1], sweep_a)
    _plot_backpressure(
        axes[1][0],
        sweep_b,
        "jct_ps",
        "expected_jct_ps",
        "Backpressure: JCT follows the credit equation",
        "JCT (ps)",
        BLUE,
    )
    _plot_backpressure(
        axes[1][1],
        sweep_b,
        "network_busy",
        "expected_network_busy",
        "Backpressure: busy attempts are N - C",
        "network-busy attempts",
        ORANGE,
    )

    residual = max(
        abs(int(row[measured]) - int(row[expected]))
        for row in rows
        for measured, expected in MEASURED_VS_EXPECTED
    )
    failures = sum(1 for row in rows if row["check"] != "PASS")
    fig.suptitle(
        "Native RNIC work queue v1: selected outcomes, "
        f"{len(rows)} ledger cells, {failures} failing, ledger max |residual| = {residual}"
    )
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / "rnic_wq_v1_sweeps.png", dpi=180)
    fig.savefig(PLOTS / "rnic_wq_v1_sweeps.pdf")


if __name__ == "__main__":
    main()
