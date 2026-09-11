"""Render all dense curves and the two-GPU protocol intervention evidence."""

import argparse
import csv
import json
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

COLORS = {
    "auto": "#146b90",
    "legacy": "#a54b78",
    "Ring_LL": "#8b7aaf",
    "Ring_LL128": "#258574",
    "Ring_Simple": "#cc793c",
}
MIB = 1024**2


def read_csv(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    numeric = {
        "width",
        "bytes",
        "median_us",
        "min_us",
        "max_us",
        "q1_us",
        "q3_us",
        "relative_iqr",
        "repetitions",
        "repeat",
        "sequence",
        "#channels",
        "#warps",
    }
    for row in rows:
        for key in numeric.intersection(row):
            try:
                row[key] = float(row[key])
            except ValueError:
                pass
    return rows


def select(rows, arch, width, arm="auto", lane=None):
    return sorted(
        [
            r
            for r in rows
            if r["architecture"] == arch
            and r["width"] == width
            and r["arm"] == arm
            and (lane is None or r.get("lane") == lane)
        ],
        key=lambda r: r["bytes"],
    )


def draw_curve(ax, rows, color, label, marker=False):
    if not rows:
        return
    x = np.array([r["bytes"] / MIB for r in rows])
    y = np.array([r["median_us"] for r in rows])
    ax.fill_between(
        x,
        [r["q1_us"] for r in rows],
        [r["q3_us"] for r in rows],
        color=color,
        alpha=0.12,
        linewidth=0,
    )
    ax.plot(
        x,
        y,
        color=color,
        lw=1.15,
        label=label,
        marker="." if marker else None,
        markersize=1.6,
        zorder=3,
    )


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e0e5e9", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#53616c", length=3)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#8d98a2")


def protocol_boundaries(rows):
    return [(a, b) for a, b in pairwise(rows) if (a["algo"], a["proto"]) != (b["algo"], b["proto"])]


def save(fig, output, stem):
    fig.savefig(output / (stem + ".png"), dpi=300, facecolor="white")
    fig.savefig(output / (stem + ".pdf"), facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    points = read_csv(args.data / "timing_summary.csv")
    tuning = read_csv(args.data / "tuning.csv")
    audit = json.loads((args.data / "analysis.json").read_text())
    valid = {a["architecture"] for a in audit["audits"] if a["status"] == "valid"}
    models = json.loads(
        (args.repo / "examples/collective_regime_curve_v1/validation.json").read_text()
    )["curves"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7, 7.0), sharex=True)
    fig.subplots_adjust(left=0.095, right=0.98, top=0.75, bottom=0.20, wspace=0.24, hspace=0.31)
    fig.text(0.055, 0.96, "Dense all-reduce measurements around 1 MiB", fontsize=14, weight="bold")
    fig.text(
        0.055,
        0.916,
        "Merlin | NCCL 2.31.2 | 241 sizes per curve | five process repeats",
        color="#53616c",
    )
    handles = [
        Line2D([], [], color=COLORS["legacy"], lw=1.5, label="Original timing method"),
        Line2D([], [], color=COLORS["auto"], lw=1.5, label="NVIDIA benchmark"),
        Line2D([], [], color="#7e858b", ls="--", lw=1.2, label="Unchanged model"),
        Line2D(
            [],
            [],
            color="#414950",
            marker="o",
            mfc="white",
            ls="",
            ms=4,
            label="Earlier measurements",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.048, 0.89),
        frameon=False,
        ncol=2,
        fontsize=8,
        columnspacing=1.9,
        handlelength=2.6,
    )
    for col, arch in enumerate(("a100", "gh200")):
        old = json.loads(
            (
                args.repo / f"examples/{arch}_hardware_envelope_v1/measurements/lane_b_result.json"
            ).read_text()
        )
        for row, width in enumerate((2, 4)):
            ax = axes[row, col]
            style(ax)
            ax.set_title(
                f"{chr(97 + 2 * row + col)}  {arch.upper()}, {width} GPUs", loc="left", pad=7
            )
            if arch not in valid:
                ax.text(
                    0.5, 0.5, "New capture unavailable or void", ha="center", transform=ax.transAxes
                )
                continue
            for lane in ("legacy", "timing"):
                draw_curve(
                    ax,
                    select(points, arch, width, lane=lane),
                    COLORS["legacy" if lane == "legacy" else "auto"],
                    lane,
                    marker=True,
                )
            model = next(m for m in models if m["machine"] == arch and m["width"] == width)
            x = np.linspace(0.25, 4, 500)
            endpoint = x * MIB * 2 * (width - 1) / width
            anchors = np.array(model["anchors"])
            bandwidth = np.exp(
                np.interp(np.log(endpoint), np.log(anchors[:, 0]), np.log(anchors[:, 1]))
            )
            predicted = model["floor_us"] + endpoint / bandwidth * 1e6
            ax.plot(x, predicted, "--", color="#7e858b", lw=1.15, zorder=2)
            historical = [
                r
                for r in old["collectives"]
                if r["op"] == "allreduce"
                and r["width"] == width
                and 0.25 * MIB <= r["bytes"] <= 4 * MIB
            ]
            ax.scatter(
                [r["bytes"] / MIB for r in historical],
                [r["time_us"] for r in historical],
                s=18,
                facecolors="white",
                edgecolors="#414950",
                linewidths=0.8,
                zorder=5,
            )
            for _, boundary in protocol_boundaries(select(tuning, arch, width)):
                ax.axvline(boundary["bytes"] / MIB, color="#dca977", lw=0.75, ls=":", zorder=1)
            ax.set_xlim(0.2, 4.05)
            ax.set_xticks([0.25, 1, 2, 3, 4], ["0.25", "1", "2", "3", "4"])
            ax.set_ylim(bottom=0)
            if col == 0:
                ax.set_ylabel("Time per all-reduce (µs)")
            if row == 1:
                ax.set_xlabel("Payload per GPU (MiB)")
    fig.text(
        0.055,
        0.105,
        "16 KiB spacing. Shading: repeat interquartile range. Dotted lines: protocol changes.",
        fontsize=8,
        color="#53616c",
    )
    fig.text(
        0.055,
        0.070,
        "Timing runs have no tuning instrumentation. Model anchors and earlier data are unchanged.",
        fontsize=8,
        color="#53616c",
    )
    fig.text(
        0.055,
        0.035,
        "Component evidence for TRAF-43; this figure does not validate end-to-end token latency.",
        fontsize=8,
        color="#53616c",
    )
    save(fig, args.output, "dense-collective-timing")

    fig, axes = plt.subplots(
        2, 2, figsize=(7, 7.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.79, bottom=0.20, wspace=0.26, hspace=0.36)
    fig.text(0.055, 0.96, "What changes inside NCCL on two GPUs?", fontsize=14, weight="bold")
    fig.text(
        0.055,
        0.915,
        "NCCL 2.31.2 | automatic selection versus fixed Ring protocols | 65 control sizes",
        color="#53616c",
    )
    labels = {
        "auto": "Automatic",
        "Ring_LL": "Ring / LL",
        "Ring_LL128": "Ring / LL128",
        "Ring_Simple": "Ring / Simple",
    }
    fig.legend(
        handles=[Line2D([], [], color=COLORS[a], lw=1.5, label=labels[a]) for a in labels],
        loc="upper left",
        bbox_to_anchor=(0.05, 0.885),
        ncol=4,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.2,
        fontsize=8,
    )
    for col, arch in enumerate(("a100", "gh200")):
        ax, channel_ax = axes[:, col]
        for panel in (ax, channel_ax):
            style(panel)
            panel.set_xlim(0.49, 2.51)
            panel.set_xticks([0.5, 1, 1.5, 2, 2.5])
        ax.set_title(f"{chr(97 + col)}  {arch.upper()} latency", loc="left", pad=7)
        channel_ax.set_title(f"{chr(99 + col)}  Automatic channel count", loc="left", pad=7)
        if arch not in valid:
            ax.text(0.5, 0.5, "Capture pending", ha="center", transform=ax.transAxes)
            continue
        for arm, arm_label in labels.items():
            rows = [
                r
                for r in select(points, arch, 2, arm, "timing")
                if 0.5 * MIB <= r["bytes"] <= 2.5 * MIB
            ]
            draw_curve(ax, rows, COLORS[arm], arm_label)
            chosen = [
                r for r in select(tuning, arch, 2, arm) if 0.5 * MIB <= r["bytes"] <= 2.5 * MIB
            ]
            if (
                arm == "auto"
                and chosen
                and all(isinstance(r["#channels"], (int, float)) for r in chosen)
            ):
                channel_ax.step(
                    [r["bytes"] / MIB for r in chosen],
                    [r["#channels"] for r in chosen],
                    where="post",
                    color=COLORS[arm],
                    lw=1.2,
                )
        for before, after in protocol_boundaries(select(tuning, arch, 2)):
            x = after["bytes"] / MIB
            if 0.5 <= x <= 2.5:
                ax.axvline(x, color="#53616c", ls=":", lw=0.9)
                channel_ax.axvline(x, color="#53616c", ls=":", lw=0.9)
                ax.text(
                    x + 0.035,
                    0.96,
                    f"{before['proto']} → {after['proto']}\n{after['bytes'] / 1024:g} KiB",
                    transform=ax.get_xaxis_transform(),
                    va="top",
                    fontsize=7.5,
                    bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 2},
                )
        ax.set_ylim(bottom=0)
        channel_ax.set_ylim(-0.2, 9)
        channel_ax.set_yticks([0, 4, 8])
        channel_ax.set_xlabel("Payload per GPU (MiB)")
        if col == 0:
            ax.set_ylabel("Time per all-reduce (µs)")
            channel_ax.set_ylabel("Channels used")
    fig.text(
        0.055,
        0.105,
        "Latency: uninstrumented process medians with interquartile shading.",
        fontsize=8,
        color="#53616c",
    )
    fig.text(
        0.055,
        0.070,
        "Selections: separate profiler runs. Channels divide GPU work among parallel communication paths.",
        fontsize=7.8,
        color="#53616c",
    )
    fig.text(
        0.055,
        0.035,
        "A threshold is attributed only when recorded choices and the fixed-choice intervention agree.",
        fontsize=7.8,
        color="#53616c",
    )
    save(fig, args.output, "nccl-protocol-controls")

    errors = read_csv(args.data / "model_residuals.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.105, right=0.98, top=0.77, bottom=0.19, wspace=0.20, hspace=0.45)
    fig.text(
        0.055, 0.96, "Dense sampling exposes additional model error", fontsize=14, weight="bold"
    )
    fig.text(
        0.055,
        0.916,
        "Unchanged five-anchor model | descriptive follow-up comparison",
        color="#53616c",
    )
    fig.legend(
        handles=handles[:2],
        loc="upper left",
        bbox_to_anchor=(0.05, 0.88),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    summaries = audit["unchanged_model_descriptive_residuals"]["curves"]
    for col, arch in enumerate(("a100", "gh200")):
        for row, width in enumerate((2, 4)):
            ax = axes[row, col]
            style(ax)
            ax.axhspan(-15, 15, color="#e9f1e6", zorder=0)
            ax.axhline(0, color="#94a18c", lw=0.8)
            ax.axhline(-15, color="#b6c9ac", lw=0.6)
            ax.axhline(15, color="#b6c9ac", lw=0.6)
            worst = {}
            for lane in ("legacy", "timing"):
                curve = select(errors, arch, width, lane=lane)
                ax.plot(
                    [r["bytes"] / MIB for r in curve],
                    [float(r["error_pct"]) for r in curve],
                    lw=1.05,
                    color=COLORS["legacy" if lane == "legacy" else "auto"],
                )
                record = next(
                    s
                    for s in summaries
                    if s["architecture"] == arch and s["width"] == width and s["lane"] == lane
                )
                worst[lane] = abs(record["worst_signed_error_pct"])
            ax.set_title(
                f"{chr(97 + 2 * row + col)}  {arch.upper()}, {width} GPUs\n"
                f"Maximum |error|: {worst['legacy']:.1f}% / {worst['timing']:.1f}%",
                loc="left",
                fontsize=9,
                pad=8,
            )
            ax.set_xlim(0.2, 4.05)
            ax.set_ylim(-53, 24)
            ax.set_xticks([0.25, 1, 2, 3, 4], ["0.25", "1", "2", "3", "4"])
            ax.set_yticks([-45, -30, -15, 0, 15])
            if col == 0:
                ax.set_ylabel("Signed prediction error (%)")
            if row == 1:
                ax.set_xlabel("Payload per GPU (MiB)")
    fig.text(
        0.055,
        0.105,
        "Error = 100 × (model / measured median − 1). Green band: the existing ±15% target.",
        fontsize=8,
        color="#53616c",
    )
    fig.text(
        0.055,
        0.070,
        "Panel maxima list original timing method first, NVIDIA benchmark second.",
        fontsize=8,
        color="#53616c",
    )
    fig.text(
        0.055,
        0.035,
        "Residuals were inspected after capture; they are not an additional preregistered relation family.",
        fontsize=7.8,
        color="#53616c",
    )
    save(fig, args.output, "dense-model-residuals")


if __name__ == "__main__":
    main()
