"""Render the request-time breakdown figures from the study CSVs.

Usage:
    SIMLLM_DATA_ROOT=... python examples/breakdown/plot_breakdown.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

from simllm._local_config import path_from_env

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Validated categorical palette (all-pairs colorblind-safe in this order);
# the aqua slot is below 3:1 contrast on light surfaces, so every figure
# ships direct value labels and RESULTS.md carries the full tables.
COLORS = {"compute": "#2a78d6", "memory": "#eb6834", "network": "#1baf7a"}
INK, MUTED, SURFACE = "#222222", "#666666", "#fcfcfb"
MS = 1e-9  # ps to ms


def load_components(path: Path) -> dict[tuple[str, str, int], dict[str, int]]:
    table = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            table[(row["profile"], row["link"], int(row["tp"]))] = {
                key: int(row[f"{key}_ps"])
                for key in ("compute", "memory", "network", "total")
            }
    return table


def stacked_bar(ax, x, comp, width, hatch=None):
    bottom = 0.0
    for key in ("compute", "memory", "network"):
        value = comp[key] * MS
        ax.bar(x, value, width, bottom=bottom, color=COLORS[key],
               edgecolor=SURFACE, linewidth=1.5, hatch=hatch,
               alpha=0.75 if hatch else 1.0)
        bottom += value
    return bottom


def style(ax, title):
    ax.set_title(title, fontsize=11, color=INK, pad=10)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors=MUTED, labelsize=9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--out", default=str(Path(__file__).parent / "plots"))
    args = parser.parse_args()
    if args.runs is None:
        data_root = path_from_env("SIMLLM_DATA_ROOT")
        if data_root is None:
            parser.error("--runs is required when SIMLLM_DATA_ROOT is not set")
        args.runs = data_root / "breakdown"
    table = load_components(args.runs / "components.csv")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    legend = [Patch(facecolor=COLORS[k], edgecolor=SURFACE, label=k)
              for k in ("compute", "memory", "network")]
    legend_exp = legend + [
        Patch(facecolor="#ffffff", edgecolor=MUTED, hatch="//", label="expected"),
        Patch(facecolor="#dddddd", edgecolor=SURFACE, label="actual"),
    ]

    # Figure 1: expected vs actual on the fluid profile, link speed x TP.
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    # Frozen expected components (expectations.md check F), ps.
    expected = {
        (2, "400G"): (11_781_312_000, 10_205_472_000, 23_596_236_800),
        (2, "100G"): (11_781_312_000, 10_205_472_000, 88_240_947_200),
        (4, "400G"): (5_891_072_000, 5_759_360_000, 38_466_355_200),
        (4, "100G"): (5_891_072_000, 5_759_360_000, 135_433_420_800),
        (8, "400G"): (2_945_952_000, 3_536_288_000, 52_045_414_400),
        (8, "100G"): (2_945_952_000, 3_536_288_000, 165_173_657_600),
    }
    for ax, link in zip(axes, ("400G", "100G")):
        ax.set_facecolor(SURFACE)
        for i, tp in enumerate((2, 4, 8)):
            exp = dict(zip(("compute", "memory", "network"), expected[(tp, link)]))
            act = table[("rnic-nn-fluid", link, tp)]
            stacked_bar(ax, i - 0.21, exp, 0.38, hatch="//")
            top = stacked_bar(ax, i + 0.21, act, 0.38)
            ax.text(i + 0.21, top + 1.5, f"{top:.0f}", ha="center",
                    fontsize=8.5, color=INK)
            share = act["network"] / act["total"]
            ax.text(i - 0.21, top + 1.5, f"net {share:.0%}", ha="center",
                    fontsize=8.5, color=MUTED)
        ax.set_xticks(range(3), [f"TP={tp}" for tp in (2, 4, 8)])
        style(ax, f"{link} links (rnic-nn-fluid)")
    axes[0].set_ylabel("request latency, ms (2048-token prefill + 7 decodes)",
                       fontsize=9, color=MUTED)
    axes[1].legend(handles=legend_exp, frameon=False, fontsize=8.5,
                   loc="upper left")
    fig.suptitle("Where one request's time goes: expected vs actual "
                 "(expected hatched; residual 0 ps, bars coincide)",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "expected_vs_actual_fluid.png", dpi=160)
    plt.close(fig)

    # Figure 2: profile/topology comparison at 400G (measured).
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    profiles = [("rnic-nn-fluid", "fluid"),
                ("rnic-nn", "nn"),
                ("rnic-cn", "cn")]
    for i, tp in enumerate((2, 4, 8)):
        for j, (profile, short) in enumerate(profiles):
            x = i + (j - 1) * 0.27
            comp = table[(profile, "400G", tp)]
            top = stacked_bar(ax, x, comp, 0.24)
            ax.text(x, top + 1.5, f"{comp['network'] / comp['total']:.0%}",
                    ha="center", fontsize=8, color=MUTED)
            ax.text(x, -7.5, short, ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(range(3), [f"\nTP={tp}" for tp in (2, 4, 8)])
    style(ax, "400G; fluid and nn on the ideal manifold, cn on the 2-tier "
              "Clos; bar labels give the network share")
    ax.set_ylabel("request latency, ms", fontsize=9, color=MUTED)
    ax.legend(handles=legend, frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Network component vs fidelity profile and topology",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "profiles_400g.png", dpi=160)
    plt.close(fig)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
