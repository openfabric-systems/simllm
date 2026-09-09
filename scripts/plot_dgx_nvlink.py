"""Render the public DGX topology and the declared queue-service inset.

The topology is a schematic, with no numerical axes. Colored routing bundles
represent independent point-to-point NVLinks, not shared electrical buses.
Counts come from the same preset as the model. SVG is for the README; PDF is
for vector export and PNG is the inspected web rendering.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simllm.placement.dgx import DGX_NVLINK_BUNDLES

plt.rcParams["svg.hashsalt"] = "simllm-dgx-nvlink-v1"
plt.rcParams["svg.fonttype"] = "none"

INK = "#203246"
MUTED = "#65778A"
GREEN = "#5E9127"
TEAL = "#238C8A"
PURPLE = "#8872B3"
ORANGE = "#CE8C36"
COLORS = ("#B979AA", "#D3765D", "#548AC2", "#8881B7", "#CE9A51", "#91AA51")


def box(ax, x, y, w, h, label, color, *, fontsize=11, fill=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                               lw=1.2, edgecolor=color, facecolor=fill or "white", zorder=5))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color=INK,
            fontsize=fontsize, linespacing=1.4, zorder=6)


def line(ax, points, color, *, lw=1, style="-", zorder=2):
    ax.plot(*zip(*points), color=color, lw=lw, linestyle=style, zorder=zorder)


def arrow(ax, start, end, color, *, dashed=False, bidir=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="<->" if bidir else "-|>",
                                 mutation_scale=11, lw=1.2, color=color,
                                 linestyle="--" if dashed else "-", zorder=4))


def queues(ax, x, y, n=3, color=TEAL):
    for row in range(n):
        for cell in range(4):
            ax.add_patch(Rectangle((x + cell * .12, y + row * .14), .10, .10,
                                   facecolor=color if cell < 2 else "white",
                                   edgecolor=color, lw=.55, zorder=7))


def render(generation, output):
    bundles = DGX_NVLINK_BUNDLES[generation]
    switches, total = len(bundles), sum(bundles)
    fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
    ax = fig.add_axes((.025, .025, .95, .95))
    ax.set(xlim=(0, 12), ylim=(0, 9))
    ax.axis("off")
    ax.text(.15, 8.78, f"DGX {generation.upper()}  |  one NVLink switching tier",
            fontsize=21, fontweight="bold", color=INK, va="top")
    ax.text(.15, 8.30, f"8 GPUs  /  {switches} independent NVSwitch chips  /  {total} links per GPU",
            fontsize=12, color=MUTED)
    # Context follows the vendor's paired-GPU PCIe arrangement. Optional
    # storage and switch management are omitted to keep the peer path legible.
    for cpu, x in enumerate((2.2, 8.0)):
        box(ax, x, 7.50, 1.7, .43, f"CPU {cpu}", "#7797B7", fontsize=11)
    arrow(ax, (3.94, 7.72), (7.96, 7.72), "#AAC0D6", bidir=True)
    ax.text(5.95, 7.85, "host interconnect", fontsize=9, color=MUTED, ha="center")
    gpu_x = [.27 + i * 1.45 for i in range(8)]
    if generation == "a100":
        for i in range(4):
            cx = .995 + i * 2.90
            box(ax, cx - .52, 6.61, 1.35, .39, "PCIe switch", PURPLE, fontsize=10)
            cpu_x = 3.05 if i < 2 else 8.85
            line(ax, [(cpu_x, 7.50), (cpu_x, 7.27 - .1 * (i % 2)),
                      (cx + .15, 7.27 - .1 * (i % 2)), (cx + .15, 7.02)], PURPLE)
            box(ax, cx + .93, 6.65, .87, .32, "2 NICs", ORANGE, fontsize=9)
            line(ax, [(cx + .84, 6.8), (cx + .91, 6.8)], ORANGE)
            for g in (2 * i, 2 * i + 1):
                line(ax, [(cx + .15, 6.59), (cx + .15, 6.36),
                          (gpu_x[g] + .51, 6.36), (gpu_x[g] + .51, 6.17)], PURPLE)
    else:
        for i, gx in enumerate(gpu_x):
            cpu_x = 3.05 if i < 4 else 8.85
            box(ax, gx, 6.58, 1.02, .48, "ConnectX-7\nmodule", ORANGE, fontsize=8.8)
            line(ax, [(cpu_x, 7.50), (cpu_x, 7.26), (gx + .51, 7.26),
                      (gx + .51, 7.09)], PURPLE)
            line(ax, [(gx + .51, 6.56), (gx + .51, 6.17)], PURPLE)
    for i, x in enumerate(gpu_x):
        box(ax, x, 5.49, 1.02, .66, f"GPU {i}\n{generation.upper()}", GREEN,
            fontsize=11, fill="#F5FAEF")
    sw_x = [.56 + i * (10.15 / (switches - 1)) for i in range(switches)]
    # Each colored lane is a drawing bundle for eight separate GPU attachments.
    # Mark only the true branch endpoint; crossings carry no connection dot.
    for sw, (x, width) in enumerate(zip(sw_x, bundles)):
        color = COLORS[sw]
        y = 4.96 - sw * .155
        taps = [gx + .16 + sw * (.70 / (switches - 1)) for gx in gpu_x]
        for tap in taps:
            line(ax, [(tap, 5.46), (tap, y)], color, lw=1.0)
        line(ax, [(min(taps), y), (max(taps), y)], color, lw=1.05)
        line(ax, [(x + .4, y), (x + .4, 3.78)], color, lw=1.5)
        box(ax, x - .10, 3.13, 1.22, .60, f"NVSwitch {sw + 1}\n{8 * width} active ports",
            TEAL, fontsize=9.4, fill="#F0F9F8")
        ax.text(x + .51, 2.93, f"{width} links / GPU", fontsize=10, color=color, ha="center")
    ax.text(.15, 5.12, "", color=MUTED)
    ax.text(.15, 2.57,
            f"Each link: 25 GB/s per direction.  Each GPU: {total * 25} GB/s send + {total * 25} GB/s receive.",
            fontsize=12, color=INK)
    ax.text(.15, 2.30, "Color groups independent point-to-point links; it does not denote a shared bus.",
            fontsize=10, color=MUTED)
    # A causal path through one of the independent switch chips.
    ax.add_patch(Rectangle((.1, .24), 11.8, 1.81, facecolor="#F6F8FB", edgecolor="none", zorder=0))
    ax.text(.28, 1.84, "One packet through one switch: model queue boundaries", fontsize=12,
            fontweight="bold", color=INK)
    box(ax, .35, .77, 2.00, .68, "GPU egress\nstaging + link grant", GREEN, fontsize=10)
    box(ax, 2.93, .66, 2.36, .86, "", TEAL)
    queues(ax, 3.10, .82)
    ax.text(3.70, 1.12, "Switch input", fontsize=10, color=INK, zorder=8)
    ax.text(3.70, .87, "finite buffer + VOQs", fontsize=8.4, color=INK, zorder=8)
    box(ax, 5.84, .77, 2.41, .68, "Crossbar + output link\none grant per input / output",
        TEAL, fontsize=9.6)
    box(ax, 8.82, .77, 2.76, .68, "GPU ingress\nfinite buffer + ordered visibility", "#548AC2", fontsize=9.6)
    for start, end in (((2.4, 1.10), (2.88, 1.10)), ((5.34, 1.10), (5.79, 1.10)),
                       ((8.30, 1.10), (8.77, 1.10))):
        arrow(ax, start, end, INK)
    line(ax, [(3.3, .65), (3.3, .49), (1.33, .49), (1.33, .75)], MUTED, style="--")
    arrow(ax, (1.33, .49), (1.33, .75), MUTED, dashed=True)
    line(ax, [(10.2, .75), (10.2, .49), (7.1, .49), (7.1, .75)], MUTED, style="--")
    arrow(ax, (7.1, .49), (7.1, .75), MUTED, dashed=True)
    ax.text(4.80, .34, "Space released downstream returns credit upstream", fontsize=9,
            color=MUTED, ha="center")
    ax.text(.15, .04, "Public wiring; declared queue sizes, grant policy and forwarding. CPU/PCIe context simplified; storage and management omitted.",
            fontsize=8.9, color=MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".svg", ".pdf", ".png"):
        metadata = {"Creator": "SimLLM"}
        if suffix == ".svg":
            metadata["Date"] = None
        elif suffix == ".pdf":
            metadata.update(CreationDate=None, ModDate=None)
        target = output.with_suffix(suffix)
        fig.savefig(target, dpi=220, metadata=metadata)
        if suffix == ".svg":
            target.write_text("\n".join(row.rstrip() for row in target.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "resources/figures")
    args = parser.parse_args()
    render("a100", args.output / "nvlink-domain-model")
    render("h100", args.output / "dgx-h100-nvlink")
