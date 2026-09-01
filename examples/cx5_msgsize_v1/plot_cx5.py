"""Plot the CX-5 message-size calibration study into plots/.

Reads msg.csv, mtu.csv and incast.csv from the runs directory and writes
message_size_vs_goodput.png, mtu_and_latency.png and incast_tax.png next to
this file.

Usage:
    SIMLLM_DATA_ROOT=... python examples/cx5_msgsize_v1/plot_cx5.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simllm._local_config import path_from_env

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"

# Measured ConnectX-5 depth-1 WRITE rows, medians of the two directions.
MEASURED_Q1 = {
    4096: 7.04,
    16384: 21.95,
    65536: 49.63,
    262144: 79.23,
    1048576: 73.95,
    4194304: 77.03,
}
MEASURED_INCAST = {"goodput": 73.9, "wire": 99.4}


def read(path: Path) -> list[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def plot_msg(rows: list[dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, profile in zip(axes, ("cx5_100g", "cx7_400g")):
        for depth, style in ((1, "o-"), (16, "s--")):
            series = sorted(
                (int(row["size_bytes"]), float(row["goodput_gbps"]))
                for row in rows
                if row["profile"] == profile and int(row["q"]) == depth
            )
            axis.plot([size for size, _ in series], [value for _, value in series],
                      style, label=f"model Q={depth}")
        if profile == "cx5_100g":
            axis.plot(sorted(MEASURED_Q1), [MEASURED_Q1[size] for size in sorted(MEASURED_Q1)],
                      "^:", color="black", label="measured depth 1")
        axis.set_xscale("log", base=2)
        axis.set_xlabel("message size (bytes)")
        axis.set_ylabel("goodput (Gbit/s)")
        axis.set_title(profile)
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(PLOTS / "message_size_vs_goodput.png", dpi=140)
    plt.close(figure)


def plot_mtu(rows: list[dict]) -> None:
    figure, axis = plt.subplots(figsize=(5.2, 4.0))
    labels = [f"MTU {row['mtu_bytes']}" for row in rows]
    values = [float(row["goodput_gbps"]) for row in rows]
    axis.bar(labels, values, color=["#4c72b0", "#dd8452"])
    for index, value in enumerate(values):
        axis.text(index, value, f"{value:.2f}", ha="center", va="bottom")
    tax = 100 * (1 - values[1] / values[0])
    axis.set_ylabel("goodput at 1 MiB (Gbit/s)")
    axis.set_title(f"MTU tax {tax:.2f} pp (measured 5.6)")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(PLOTS / "mtu_tax.png", dpi=140)
    plt.close(figure)


def plot_incast(rows: list[dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    buffers = sorted({int(row["buffer_bytes"]) for row in rows}, reverse=True)
    width = 0.35
    for axis, pattern in zip(axes, ("incast", "fanout")):
        selected = [row for row in rows if row["pattern"] == pattern]
        positions = range(len(buffers))

        def best(key: str, buffer: int, cells: list[dict] = selected) -> float:
            return max(float(row[key]) for row in cells
                       if int(row["buffer_bytes"]) == buffer)

        goodput = [best("goodput_gbps", buffer) for buffer in buffers]
        wire = [best("wire_gbps", buffer) for buffer in buffers]
        makespans = [best("makespan_us", buffer) / 1000.0 for buffer in buffers]
        axis.bar([p - width / 2 for p in positions], wire, width, label="wire")
        axis.bar([p + width / 2 for p in positions], goodput, width, label="goodput")
        for position, span in zip(positions, makespans):
            axis.text(position, max(goodput + wire) * 0.06, f"{span:.1f} ms",
                      ha="center", fontsize=8)
        if pattern == "incast":
            axis.axhline(MEASURED_INCAST["goodput"], color="black", linestyle=":",
                         label="measured goodput 73.9")
            axis.axhline(MEASURED_INCAST["wire"], color="grey", linestyle="--",
                         label="measured wire 99.4")
        axis.set_xticks(list(positions))
        axis.set_xticklabels([
            f"buffer {buffer}\n{'registered' if buffer >= 1 << 20 else 'post-specified'}"
            for buffer in buffers
        ])
        axis.set_ylabel("Gbit/s at the receiver")
        axis.set_title(f"{pattern} at 1 MiB, makespan annotated")
        axis.grid(axis="y", alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(PLOTS / "incast_tax.png", dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path)
    args = parser.parse_args()
    if args.runs is None:
        data_root = path_from_env("SIMLLM_DATA_ROOT")
        if data_root is None:
            parser.error("--runs is required when SIMLLM_DATA_ROOT is not set")
        args.runs = data_root / "cx5_msgsize_v1"
    PLOTS.mkdir(exist_ok=True)
    plot_msg(read(args.runs / "msg.csv"))
    plot_mtu(read(args.runs / "mtu.csv"))
    plot_incast(read(args.runs / "incast.csv"))
    print(f"plots -> {PLOTS}")


if __name__ == "__main__":
    main()
