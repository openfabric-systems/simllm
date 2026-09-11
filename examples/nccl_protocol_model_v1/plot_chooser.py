"""Compare NCCL's recorded selection heuristic with fixed-protocol timings."""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simllm.traffic import NcclRingProtocolModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--known-captures", type=Path, required=True)
    parser.add_argument("--known-summary", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = {
        m.architecture: m
        for m in map(NcclRingProtocolModel.from_json, json.loads(args.models.read_text()))
        if m.width == 2
    }
    with args.known_summary.open() as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.8))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.71, bottom=0.23, wspace=0.23)
    fig.text(
        0.09,
        0.95,
        "Why NCCL switches before Simple becomes faster",
        fontsize=12,
        weight="bold",
        color="#25313b",
    )
    fig.text(
        0.09,
        0.89,
        "Two GPUs | Fixed-protocol controls | Software chooser costs are not hardware measurements",
        fontsize=8,
        color="#657380",
    )
    colors = {"LL": "#754b98", "SIMPLE": "#008b84"}
    statistics = []
    handles = [
        Line2D([], [], color=colors["LL"], lw=1.4, label="LL"),
        Line2D([], [], color=colors["SIMPLE"], lw=1.4, label="Simple"),
        Line2D([], [], color="#4b5662", marker="o", ms=3, ls="none", mfc="white", label="Measured"),
        Line2D([], [], color="#4b5662", lw=1.3, label="New work model"),
        Line2D([], [], color="#4b5662", lw=1, ls="--", label="NCCL chooser estimate"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.08, 0.85),
        ncol=3,
        frameon=False,
        fontsize=7.5,
    )
    for ax, (arch, job) in zip(axes, [("a100", "204703"), ("gh200", "204704")], strict=True):
        logfile = next(
            (args.known_captures / arch / job / "raw").glob("*diagnostic_auto_w2_r0.nccl.*.log")
        )
        line = next(
            line
            for line in logfile.read_text().splitlines()
            if re.search(r"INFO\s+AllReduce\s*\|", line)
        )
        pairs = re.findall(r"(-?\d+\.\d+)\s*/\s*(-?\d+\.\d+)", line)[3:6]
        costs = dict(
            zip(("LL", "LL128", "SIMPLE"), [(float(a), float(b)) for a, b in pairs], strict=True)
        )
        measured = {}
        for protocol, arm in [("LL", "Ring_LL"), ("SIMPLE", "Ring_Simple")]:
            selected = sorted(
                [
                    r
                    for r in rows
                    if r["architecture"] == arch
                    and r["width"] == "2"
                    and r["lane"] == "timing"
                    and r["arm"] == arm
                ],
                key=lambda r: int(r["bytes"]),
            )
            measured[protocol] = {int(r["bytes"]): float(r["median_us"]) for r in selected}
            sizes = sorted(measured[protocol])
            x = np.asarray(sizes) / 2**20
            ax.plot(
                x,
                [
                    models[arch].predict(size, protocol=protocol).reference_ps / 1e6
                    for size in sizes
                ],
                color=colors[protocol],
                lw=1.3,
            )
            ax.plot(
                x,
                [measured[protocol][size] for size in sizes],
                color=colors[protocol],
                marker="o",
                mfc="white",
                ms=2.8,
                mew=0.65,
                ls="none",
            )
            latency, bandwidth = costs[protocol]
            ax.plot(
                x,
                latency + np.asarray(sizes) / (1000 * bandwidth),
                color=colors[protocol],
                lw=1,
                ls="--",
            )
        cross = (costs["SIMPLE"][0] - costs["LL"][0]) / (
            1 / (1000 * costs["LL"][1]) - 1 / (1000 * costs["SIMPLE"][1])
        )
        actual = next(
            size
            for size in sorted(measured["LL"])
            if measured["LL"][size] >= measured["SIMPLE"][size]
        )
        ax.axvspan(cross / 2**20, actual / 2**20, color="#efd58c", alpha=0.18, zorder=0)
        ax.axvline(cross / 2**20, color="#9c7d35", ls=":", lw=1)
        ax.set_title(arch.upper(), loc="left", fontsize=10)
        ax.set_xlabel("Payload per GPU (MiB)")
        ax.set_ylabel("All-reduce time (µs)")
        ax.set_xlim(0.46, 2.55)
        ax.set_xticks([0.5, 1, 1.5, 2, 2.5])
        ax.grid(axis="y", color="#e4e8eb", lw=0.6)
        statistics.append(
            {
                "architecture": arch,
                "recorded_chooser_costs": costs,
                "heuristic_crossing_bytes_approximate": cross,
                "first_fixed_control_simple_no_slower_bytes": actual,
                "ll_at_1mib_us": measured["LL"][1048576],
                "simple_at_1mib_us": measured["SIMPLE"][1048576],
            }
        )
    fig.text(
        0.09,
        0.11,
        "Shading: Simple is selected even though the fixed-LL control is still faster at the sampled payloads.",
        fontsize=7.5,
        color="#657380",
    )
    fig.text(
        0.09,
        0.065,
        "Chooser lines use its recorded rounded latency/bandwidth table. The model parameters remain unchanged.",
        fontsize=7.5,
        color="#657380",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output.with_suffix(suffix))
    args.output.with_suffix(".json").write_text(json.dumps(statistics, indent=2) + "\n")
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
