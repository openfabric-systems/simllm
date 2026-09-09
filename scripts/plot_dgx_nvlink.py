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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

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


def arrow(ax, start, end, color, *, dashed=False, bidir=False, style=None):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="<->" if bidir else "-|>",
                                 mutation_scale=11, lw=1.2, color=color,
                                 linestyle=style or ("--" if dashed else "-"), zorder=4))


def fifo(ax, x, y, w=.90, h=.20, *, occupied=3, color=TEAL):
    """Symbolic packet slots, with the queue head at the right-hand exit."""
    for cell in range(5):
        filled = cell >= 5 - occupied
        ax.add_patch(Rectangle((x + cell * w / 5, y - h / 2), w / 5, h,
                               edgecolor="white" if filled else color, facecolor=color if filled else "white",
                               lw=.70, zorder=7))
    ax.add_patch(Rectangle((x, y - h / 2), w, h, edgecolor=color,
                           facecolor="none", lw=.85, zorder=8))


def packet(ax, x, y, *, w=.42, h=.25, color=ORANGE):
    ax.add_patch(Rectangle((x, y - h / 2), w, h, edgecolor=color,
                           facecolor="#FFF5E8", lw=1.1, zorder=7))
    line(ax, [(x + w * .35, y - h / 2), (x + w * .35, y + h / 2)], color, zorder=8)
    for label, offset in (("H", .175), ("D", .675)):
        ax.text(x + w * offset, y, label, fontsize=6.8, color=INK,
                ha="center", va="center", zorder=9)


def mux(ax, x, y, *, h=1.65, w=.23, color=TEAL):
    ax.add_patch(Polygon(((x, y - h / 2), (x + w, y - h * .30),
                          (x + w, y + h * .30), (x, y + h / 2)),
                         edgecolor=color, facecolor="white", lw=1.2, zorder=7))


def crossbar(ax, x, y, *, w=1.12, h=1.16):
    ax.add_patch(Rectangle((x, y - h / 2), w, h, edgecolor=TEAL,
                           facecolor="white", lw=1.2, zorder=5))
    levels = [y - .40, y, y + .40]
    for source in levels:
        for destination in levels:
            line(ax, [(x, source), (x + w, destination)], "#CBDDDC", lw=.7, zorder=6)
    line(ax, [(x, y), (x + w, y)], TEAL, lw=2.3, zorder=8)
    ax.plot(x + w / 2, y, "o", color=TEAL, markersize=4, zorder=9)
    for level in (levels[0], levels[2]):
        line(ax, [(x - .13, level), (x, level)], MUTED, style=":")
        line(ax, [(x + w, level), (x + w + .13, level)], MUTED, style=":")


def queue_panel(ax, total):
    aggregate = total * 25
    y = -.60
    ax.add_patch(Rectangle((.10, -3.26), 11.80, 5.31, facecolor="#F6F8FB",
                           edgecolor="none", zorder=0))
    ax.text(.28, 1.78, "Packet datapath  |  one selected NVLink path", fontsize=12,
            fontweight="bold", color=INK)
    ax.text(.28, 1.44, "Dotted stubs: other data ports. Dashed paths: returned credits. Rates are per direction.",
            fontsize=9, color=MUTED)
    for x, width, color, title in ((.25, 2.30, GREEN, "GPU egress"),
                                   (3.42, 1.84, TEAL, "Switch input"),
                                   (5.67, 1.96, TEAL, "Crossbar + output link"),
                                   (9.17, 2.41, "#548AC2", "GPU ingress")):
        box(ax, x, -1.91, width, 3.08, "", color, fill="#FFFFFF00")
        ax.text(x + width / 2, .96, title, ha="center", color=INK,
                fontsize=9.2, fontweight="bold", zorder=8)
    ax.text(1.40, .57, f"Feed F = {aggregate} GB/s total*", fontsize=8.6,
            ha="center", color=GREEN, zorder=8)
    ax.text(4.34, .57, "Per-output FIFOs", fontsize=8.6,
            ha="center", color=TEAL, zorder=8)
    ax.text(6.65, .57, "X = 25 GB/s / connection*", fontsize=8.1,
            ha="center", color=TEAL, zorder=8)
    ax.text(10.38, .57, f"Drain D = {aggregate} GB/s total*", fontsize=8.6,
            ha="center", color="#548AC2", zorder=8)
    # Staging FIFOs feed packetization and striping across all GPU links.
    for row, (level, occupied) in enumerate(((.0, 2), (y, 4), (-1.20, 1))):
        fifo(ax, .43, level, occupied=occupied, color=GREEN)
        ax.text(.88, level + .18, f"dst {row + 1}", fontsize=7.1,
                ha="center", color=MUTED, zorder=8)
        line(ax, [(1.34, level), (1.66, level)], GREEN, zorder=6)
    mux(ax, 1.67, y, color=GREEN)
    arrow(ax, (1.92, y), (2.11, y), GREEN)
    packet(ax, 2.12, y, w=.31, color=GREEN)
    ax.text(2.01, -.30, "F", fontsize=8.2, ha="center", color=GREEN, zorder=8)
    ax.text(1.40, -1.61, f"packetize / stripe over {total} links", fontsize=7.7,
            ha="center", color=MUTED, zorder=8)
    # One highlighted physical input link. Additional branches are omitted.
    arrow(ax, (2.47, y), (3.36, y), INK)
    ax.text(2.96, -.16, "25 GB/s\nper link", fontsize=8.2, ha="center", color=INK)
    for level in (.18, -1.20):
        line(ax, [(2.44, y), (2.59, y), (2.59, level), (2.99, level)],
             MUTED, style=":", lw=.8)
    # One switch input port shares finite storage across its virtual queues.
    line(ax, [(3.47, .0), (3.47, -1.20)], TEAL, zorder=6)
    for row, (level, occupied) in enumerate(((.0, 3), (y, 4), (-1.20, 1))):
        line(ax, [(3.47, level), (3.61, level)], TEAL, zorder=6)
        fifo(ax, 3.62, level, w=.88, occupied=occupied)
        ax.text(4.06, level + .18, f"out {chr(65 + row)}", fontsize=7.1,
                ha="center", color=MUTED, zorder=8)
        line(ax, [(4.52, level), (4.81, level)], TEAL, zorder=6)
    mux(ax, 4.82, y)
    ax.text(4.34, -1.61, "B_sw = 64 KiB / input*", fontsize=8.0,
            ha="center", color=MUTED, zorder=8)
    arrow(ax, (5.09, y), (5.85, y), TEAL)
    ax.text(5.47, -.20, "grant", fontsize=8.0, ha="center", color=TEAL)
    crossbar(ax, 5.89, y)
    ax.text(6.65, -1.49, "one grant per\ninput / output", fontsize=8.3,
            ha="center", va="center", color=MUTED, zorder=8)
    # Crossbar service and the selected output serializer form one grant.
    arrow(ax, (7.04, y), (7.94, y), TEAL)
    packet(ax, 7.96, y)
    arrow(ax, (8.42, y), (9.29, y), INK)
    ax.text(8.30, -.16, "25 GB/s\nper link", fontsize=8.2, ha="center", color=INK)
    ax.text(8.29, -1.01, "packet in flight", fontsize=7.6, ha="center", color=MUTED)
    # All destination links share the GPU receive-storage and drain budget.
    for level in (.0, -1.20):
        arrow(ax, (8.83, level), (9.30, level), MUTED, style=":")
    mux(ax, 9.32, y, color="#548AC2")
    line(ax, [(9.57, y), (9.72, y)], "#548AC2", zorder=6)
    fifo(ax, 9.73, y, w=.94, occupied=3, color="#548AC2")
    arrow(ax, (10.70, y), (11.00, y), "#548AC2")
    ax.text(10.85, -.30, "D", fontsize=8.2, ha="center", color="#548AC2", zorder=8)
    ax.add_patch(Rectangle((11.02, -.93), .38, .66, edgecolor="#548AC2",
                           facecolor="#F0F5FB", lw=1.0, zorder=7))
    for level in (-.72, -.50):
        line(ax, [(11.03, level), (11.39, level)], "#548AC2", lw=.7, zorder=8)
    ax.text(10.38, -1.29, "B_rx = 64 KiB / GPU*", fontsize=8.0,
            ha="center", color=MUTED, zorder=8)
    ax.text(10.38, -1.64, "shared ingress / ordered delivery", fontsize=7.7,
            ha="center", color=MUTED, zorder=8)
    # Credits return to the actual upstream injection and output grant gates.
    for points, label, x in (([(3.76, -1.93), (3.76, -2.18), (.31, -2.18), (.31, -1.39),
                               (1.82, -1.39), (1.82, -1.22)],
                              "switch-space credit", 2.72),
                             ([(10.13, -1.93), (10.13, -2.18), (5.21, -2.18), (5.21, -1.39),
                               (4.94, -1.39), (4.94, -1.28)],
                              "receiver-space credit", 7.74)):
        line(ax, points, MUTED, style="--", lw=1)
        arrow(ax, points[-2], points[-1], MUTED, dashed=True)
        ax.text(x, -2.43, label, fontsize=8.6, ha="center", color=MUTED)
    ax.text(.28, -2.78, "Filled FIFO cells: queued packets. Empty cells: free space. H | D: header + data (symbolic).",
            fontsize=8.6, color=INK)
    ax.text(.28, -3.08, "* Declared study settings, not measured silicon limits. FIFO slot counts and packet sizes are schematic.",
            fontsize=8.5, color=MUTED)


def render(generation, output):
    bundles = DGX_NVLINK_BUNDLES[generation]
    switches, total = len(bundles), sum(bundles)
    fig = plt.figure(figsize=(11.7, 11.62), facecolor="white")
    ax = fig.add_axes((.025, .025, .95, .95))
    ax.set(xlim=(0, 12), ylim=(-3.6, 9))
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
    queue_panel(ax, total)
    ax.text(.15, -3.48, "Public wiring; declared queue sizes, grant policy and forwarding. CPU/PCIe context simplified; storage and management omitted.",
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
