#!/usr/bin/env python3
"""Render the post-specified GPU service-model structural sweeps."""

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

PANELS = (
    ("sm_waves", "CTA waves", "grid CTAs", "kernel cycles", "SMs"),
    ("warp_issue", "Warp issue", "ready warps", "kernel cycles", "schedulers/SM"),
    ("hbm", "HBM service", "transaction bytes", "kernel cycles", "bytes/cycle"),
    ("copy", "Copy service", "descriptor bytes", "service cycles", "bytes/cycle"),
)


def _read_rows() -> list[dict[str, str]]:
    with RESULTS.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = _read_rows()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)

    for axis, (check, title, x_label, y_label, legend_title) in zip(axes.flat, PANELS, strict=True):
        groups: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            if row["check"] == check:
                groups[int(row["parameter_b"])].append(row)

        for group, values in sorted(groups.items()):
            values.sort(key=lambda row: int(row["parameter_a"]))
            x = [int(row["parameter_a"]) for row in values]
            measured = [int(row["measured_value"]) for row in values]
            expected = [int(row["expected_value"]) for row in values]
            line = axis.plot(x, measured, marker="o", label=str(group))[0]
            axis.plot(
                x,
                expected,
                linestyle="--",
                color="#202020",
                linewidth=1.4,
                alpha=0.8,
                zorder=line.get_zorder() + 1,
            )

        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.25)
        if check in {"hbm", "copy"}:
            axis.set_xscale("log", base=2)
        axis.legend(title=legend_title, frameon=False)

    max_residual = max(abs(int(row["residual"])) for row in rows)
    fig.suptitle(
        "GPU service model: replay (colored) vs closed form (black dashed), "
        f"max |residual| = {max_residual}"
    )
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / "gpu_service_structural_checks.png", dpi=180)
    fig.savefig(PLOTS / "gpu_service_structural_checks.pdf")


if __name__ == "__main__":
    main()
