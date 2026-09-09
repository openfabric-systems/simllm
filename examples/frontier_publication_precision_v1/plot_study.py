"""Plot exact retained interval segments at the two unresolved frontier rows."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent


def rational(value):
    return Fraction(value["numerator"], value["denominator"])


def render(record, output_prefix):
    rows = {row["external_row"]: row for row in record["historical_agreement"]}
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7, 4.25), sharey=True)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.29, top=0.76, wspace=0.19)
    blue, gray, green = "#235d9b", "#686868", "#e3efdf"
    for axis, index in zip(axes, (9, 10), strict=True):
        row = rows[index]
        comparison = row["comparison"]
        center = Fraction(comparison["threshold"]["published_text"])
        declared = comparison["threshold"]["interval"]
        lo, hi = ((rational(declared[key]) - center) * 1000 for key in ("lower", "upper"))
        axis.axhspan(0.75, 1.35, color=green, zorder=0)
        axis.axvspan(float(lo), float(hi), color="#dddddd", alpha=0.40, zorder=1)
        for segment in comparison["segments"]:
            interval = segment["interval"]
            left, right = ((rational(interval[key]) - center) * 1000 for key in ("lower", "upper"))
            choice = segment["selection"]
            quotient = (
                Fraction()
                if choice is None
                else rational(choice["y"]) / rational(comparison["reference_y"])
            )
            color = gray if choice is None else blue
            axis.plot(
                [float(left), float(right)],
                [float(quotient)] * 2,
                color=color,
                linestyle="--" if choice is None else "-",
                linewidth=1.8,
                zorder=3,
            )
            for x, closed in ((left, interval["lower_closed"]), (right, interval["upper_closed"])):
                axis.plot(
                    float(x),
                    float(quotient),
                    "o",
                    color=color,
                    markerfacecolor=color if closed else "white",
                    markersize=4.5,
                    zorder=4,
                )
            if choice is None:
                axis.text(
                    float((left + right) / 2),
                    0.10,
                    "No feasible\ncandidate",
                    ha="center",
                    color=gray,
                    fontsize=8,
                )
            else:
                number = record["point_identity_map"][choice["point_id"]]["row"]
                vertical = 0.055 if float(quotient) > 0.75 else -0.13
                axis.text(
                    float((left + right) / 2),
                    float(quotient) + vertical,
                    f"Config. {number}",
                    ha="center",
                    color=blue,
                    fontsize=8,
                )
        axis.set_xlim(-0.68, 0.68)
        axis.set_ylim(-0.08, 1.43)
        axis.set_xticks([-0.5, 0, 0.5])
        axis.set_yticks([0, 0.4, 0.75, 1, 1.35])
        axis.set_title(f"Row {index}: indeterminate", loc="left", pad=9)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", linewidth=0.45, alpha=0.35, zorder=0)
        axis.set_xlabel("Threshold minus published x\n($10^{-3}$ tokens/s/request)")
    axes[0].set_ylabel("Selected / published\nthroughput per GPU")
    fig.suptitle("Publication precision changes frontier selection", y=0.985, fontsize=12)
    fig.legend(
        handles=[
            Patch(facecolor=green, label="Unchanged quotient band [0.75, 1.35]"),
            Patch(facecolor="#dddddd", label="Conditional x enclosure"),
            Line2D([0], [0], color=blue, lw=1.8, label="Feasible selection"),
            Line2D([0], [0], color=gray, ls="--", lw=1.8, label="No answer (quotient 0)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.925),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.text(
        0.11,
        0.045,
        "Retained Qwen3-32B H200 model output; historical export profile is unverified.\n"
        "Filled endpoints are included; open endpoints are excluded. No service is repriced.",
        fontsize=8,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=180)
    fig.savefig(output_prefix.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=HERE / "results.json")
    parser.add_argument(
        "--output-prefix", type=Path, default=HERE / "figures/publication-boundaries"
    )
    args = parser.parse_args()
    render(json.loads(args.record.read_text(encoding="utf-8")), args.output_prefix)


if __name__ == "__main__":
    main()
