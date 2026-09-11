"""Render source work and declared resource mechanisms before calibration."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

COLORS = ["#176ba0", "#b73846", "#688848", "#9c742c", "#7662a0", "#4b9390"]


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def configure():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.labelsize": 9, "axes.titlesize": 10,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#83909b", "text.color": "#25313b",
                         "axes.labelcolor": "#25313b", "xtick.color": "#61707b",
                         "ytick.color": "#61707b", "pdf.fonttype": 42,
                         "savefig.dpi": 240})


def decorate(axes):
    for ax in np.asarray(axes).flat:
        ax.grid(axis="y", color="#e5e9ec", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.margins(x=0.04, y=0.08)


def mechanism(data):
    fig, axes = plt.subplots(2, 2, figsize=(7, 6.7))
    fig.subplots_adjust(left=0.105, right=0.98, top=0.82, bottom=0.14, hspace=0.45, wspace=0.30)
    fig.text(0.105, 0.955, "Channel work, shared resources, and finite slots", fontsize=13, weight="bold")
    fig.text(0.105, 0.922, "Controlled model checks with declared costs; no hardware calibration", fontsize=9, color="#61707b")
    residency = read(data / "residency.csv")
    ax = axes[0, 0]
    for index, sms in enumerate([1, 2, 4, 8]):
        rows = [r for r in residency if int(r["sms"]) == sms]
        x = [int(r["channels"]) for r in rows]
        y = np.array([float(r["completion_ps"]) / 1e6 for r in rows])
        dense_x = np.arange(1, 33)
        dense_y = np.ceil(dense_x / sms)
        ax.step(dense_x, dense_y, where="mid", color=COLORS[index], lw=1.1,
                label=f"{sms} SM" if sms == 1 else f"{sms} SMs")
        ax.plot(x, y, "o", ms=2.5, color=COLORS[index])
        ax.fill_between(dense_x, 0.8 * dense_y, 1.2 * dense_y, step="mid",
                        color=COLORS[index], alpha=0.08)
    ax.set(title="a  Blocks retain finite GPU capacity", xlabel="Channel blocks", ylabel="Completion time (µs)")
    ax.set_xticks([1, 8, 16, 24, 32])
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.09, 0.893),
               ncol=4, fontsize=7.5, frameon=False)
    ax.text(0.04, 0.76, "1 µs per block\nOne block per SM\nBand: ±20% service", transform=ax.transAxes, fontsize=7.3)
    ax = axes[0, 1]
    rows = [r for r in residency if int(r["sms"]) == 32]
    x = np.array([int(r["channels"]) for r in rows])
    y = np.array([int(r["issued_cycles"]) for r in rows]) / 1000
    ax.plot(x, y, "o-", color=COLORS[0], ms=3, label="Total issued work")
    ax.plot(x, np.ones_like(x), "--", color=COLORS[1], label="Largest channel only")
    ax.set(title="b  Shared work grows with channels", xlabel="Channels at fixed work per channel",
           ylabel="Work / one-channel work")
    ax.set_xticks([1, 8, 16, 24, 32])
    ax.legend(loc="upper left", fontsize=7, framealpha=1)
    ax = axes[1, 0]
    window = read(data / "window.csv")
    for proto, color in [("LL128", COLORS[0]), ("SIMPLE", COLORS[1])]:
        rows = [r for r in window if r["protocol"] == proto and int(r["count"]) == 17
                and int(r["reuse_delay_ps"]) == 2000000]
        ax.step([int(r["reservation"]) for r in rows], [float(r["issued_at_ps"]) / 1e6 for r in rows],
                where="post", color=color, lw=1.6,
                label="LL / LL128: 1 step" if proto == "LL128" else "Simple: 2 steps")
    ax.set(title="c  Reuse waits for the returned head", xlabel="Reservation number", ylabel="Eligible reservation time (µs)")
    ax.set_xticks([1, 4, 8, 9, 13, 17])
    ax.legend(fontsize=7, loc="upper left", framealpha=1)
    ax.text(0.06, 0.63, "8 slots; 2 µs reuse delay\nIdealized data transport", transform=ax.transAxes, fontsize=7.2)
    ax = axes[1, 1]
    stripe_rows = read(data / "stripe_geometry.csv")
    for proto, color in [("LL", COLORS[2]), ("LL128", COLORS[0]), ("SIMPLE", COLORS[1])]:
        rows = [r for r in stripe_rows if r["protocol"] == proto]
        ax.plot([int(r["useful_bytes"]) / 1024 for r in rows],
                [int(r["encoded_bytes"]) / 1024 for r in rows], color=color, label=proto)
    ax.set(title="d  LL128 sends full warp stripes", xlabel="Useful bytes in one primitive (KiB)",
           ylabel="Protocol bytes sent (KiB)")
    ax.legend(fontsize=7, loc="upper left", framealpha=1)
    ax.set_xticks([0, 1, 1.875, 3.75], ["0", "1", "1.875", "3.75"])
    decorate(axes)
    fig.text(0.105, 0.061, "Exact residency and window oracles are separate from protocol-source byte checks.", fontsize=8, color="#61707b")
    fig.text(0.105, 0.035, "LL128: each positive 1,920-byte warp stripe emits 2,048 bytes, including its partial tail.", fontsize=8, color="#61707b")
    return fig


def retrospective(data):
    rows = read(data / "a100_retrospective.csv")
    rows.sort(key=lambda row: int(row["payload_bytes"]))
    x = np.array([int(r["payload_bytes"]) / 1048576 for r in rows])
    measured = np.array([float(r["measured_us"]) for r in rows])
    old = np.array([float(r["old_model_us"]) for r in rows])
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.9), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.81, bottom=0.21, hspace=0.30)
    fig.text(0.12, 0.954, "A100, four GPUs: the missing shared work", fontsize=13, weight="bold")
    fig.text(0.12, 0.919, "Retained LL128 measurements and new source-derived work counts", fontsize=9, color="#61707b")
    axes[0].errorbar(x, measured, yerr=[measured - [float(r["q1_us"]) for r in rows],
                                      [float(r["q3_us"]) for r in rows] - measured],
                     fmt="o-", ms=4, color=COLORS[0], label="Measured benchmark median and IQR")
    axes[0].plot(x, old, "--", color=COLORS[1], lw=1.6, label="Previous model")
    axes[0].set(ylabel="Completion time (µs)", title="a  Measured time rises while the previous prediction stays flat")
    axes[0].legend(loc="upper left", fontsize=7.5, framealpha=1)
    for value, record in zip(x, rows, strict=True):
        axes[0].annotate(f"C={record['channels']}", (value, float(record["measured_us"])),
                         textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7, color="#61707b")
    encoded = np.array([float(r["source_encoded_bytes_per_rank"]) for r in rows])
    maxwork = np.array([float(r["max_channel_encoded_bytes"]) for r in rows])
    local = np.array([float(r["source_input_load_bytes_per_rank"]) + float(r["source_output_bytes_per_rank"]) for r in rows])
    axes[1].plot(x, encoded / encoded[0], "o-", color=COLORS[0], ms=3, label="Total encoded peer bytes / rank")
    axes[1].plot(x, local / local[0], "s-", color=COLORS[2], ms=3, label="Local input + output bytes / rank")
    axes[1].plot(x, maxwork / maxwork[0], "--", color=COLORS[1], label="Previous largest-channel work")
    axes[1].set(ylabel="Work / first payload's work", xlabel="Collective payload per GPU (MiB)",
                title="b  The source adds aggregate work and partially filled channels")
    axes[1].legend(loc="upper left", fontsize=7.5, framealpha=1)
    decorate(axes)
    fig.text(0.12, 0.118, "These work counts repair an omitted mechanism; they do not yet identify its GPU service cost.", fontsize=8, color="#61707b")
    fig.text(0.12, 0.088, "All points are retrospective inputs for the next model. No new fitted curve is shown here.", fontsize=8, color="#61707b")
    fig.text(0.12, 0.058, "IQR: interquartile range across five independent processes. C: active channels.", fontsize=8, color="#61707b")
    return fig


def protocol_curves(data):
    rows = read(data / "protocols.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7, 3.9), sharey=True)
    fig.subplots_adjust(left=0.105, right=0.98, top=0.67, bottom=0.25, wspace=0.18)
    fig.text(0.105, 0.94, "GPU resources and links limit different parts of the same graph", fontsize=11.5, weight="bold")
    fig.text(0.105, 0.892, "Four ranks; 16 KiB per GPU; deterministic declared service inputs", fontsize=8.5, color="#61707b")
    handles = []
    for ax, proto in zip(axes, ["LL", "LL128", "SIMPLE"], strict=True):
        for index, (channels, sms) in enumerate([(1, 1), (4, 1), (4, 4)]):
            selected = sorted([r for r in rows if int(r["width"]) == 4 and int(r["payload_bytes"]) == 16384 and r["protocol"] == proto
                               and int(r["channels"]) == channels and int(r["sms"]) == sms],
                              key=lambda r: float(r["link_rate_multiplier"]))
            line, = ax.plot([float(r["link_rate_multiplier"]) for r in selected],
                           [float(r["duration_ps"]) / 1e6 for r in selected], "o-", color=COLORS[index], ms=3,
                           label=f"C={channels}, SMs={sms}")
            if proto == "LL":
                handles.append(line)
        ax.set(title=proto.title() if proto == "SIMPLE" else proto, xlabel="Peer link rate / baseline", xticks=[0.5, 1, 2])
    axes[0].set_ylabel("Collective completion (µs)")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.09, 0.842), ncol=3, frameon=False, fontsize=8)
    decorate(axes)
    fig.text(0.105, 0.112, "The physical calendar carries payload, protocol flags and separate progress stores.", fontsize=8, color="#61707b")
    fig.text(0.105, 0.069, "This figure checks mechanisms. These are not calibrated A100 or GH200 predictions.", fontsize=8, color="#61707b")
    return fig


def sensitivity(data):
    result = json.loads((data / "sensitivity.json").read_text())
    labels = ["Entry", "Block setup", "Warp work", "Barrier", "Publication", "Poll issue", "Poll cadence", "Memory rate", "Shared rate"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=0.155, right=0.965, top=0.78, bottom=0.29, wspace=0.42)
    fig.text(0.115, 0.944, "Which cost terms can these interventions distinguish?", fontsize=12, weight="bold")
    fig.text(0.115, 0.902, "Model sensitivity diagnostic, before any fit to new hardware observations", fontsize=8.5, color="#61707b")
    values = np.array(result["column_cosines"])
    im = axes[0].imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest", aspect="equal")
    axes[0].set_xticks(range(len(labels)), labels, rotation=65, ha="right", fontsize=7)
    axes[0].set_yticks(range(len(labels)), labels, fontsize=7)
    axes[0].set_title("a  Similarity of parameter responses")
    fig.colorbar(im, ax=axes[0], fraction=0.045, pad=0.025, ticks=[-1, 0, 1])
    singular = np.asarray(result["singular_values"])
    axes[1].semilogy(range(1, len(labels) + 1), singular, "o-", color=COLORS[0], ms=4)
    axes[1].set(title="b  Sensitivity singular values", xlabel="Independent response direction",
                ylabel="Singular value", xticks=range(1, len(labels) + 1))
    axes[1].text(0.06, 0.10, f"Numerical rank: {result['numerical_rank']} / {len(labels)}\nCondition number: {singular[0] / singular[-1]:.1f}",
                 transform=axes[1].transAxes, fontsize=8)
    decorate([axes[1]])
    fig.text(0.115, 0.121, "Rows vary protocol, two/four ranks, one/four channels and one/four available SMs.", fontsize=8, color="#61707b")
    fig.text(0.115, 0.083, "Full numerical rank does not establish physical identification. Similar responses amplify noise.", fontsize=8, color="#61707b")
    fig.text(0.115, 0.045, "Instruction groups remain declared units; ready-peer and delayed-peer primitive probes are still needed.", fontsize=8, color="#61707b")
    return fig


def timeline(data):
    trace = json.loads((data / "example_trace.json").read_text())
    block_ids = [row[0] for row in trace["block_completions"] if row[0].endswith(":r0")]
    block_ids.sort()
    fig, ax = plt.subplots(figsize=(7, 3.9))
    fig.subplots_adjust(left=0.14, right=0.98, top=0.73, bottom=0.24)
    fig.text(0.14, 0.941, "One SM: waiting channels retain their block resources", fontsize=12, weight="bold")
    fig.text(0.14, 0.896, "GPU 0, four LL128 channels; all events use the physical transport calendar", fontsize=8.5, color="#61707b")
    for index, block in enumerate(block_ids):
        events = [row for row in trace["residency_events"] if row["block_id"] == block]
        start = next(row["at_ps"] for row in events if row["kind"] == "admit") / 1e6
        end = next(row["at_ps"] for row in events if row["kind"] == "release") / 1e6
        ax.broken_barh([(start, end - start)], (index - 0.29, 0.58), facecolors="#e4e9ed")
        for visit in trace["resource_visits"]:
            if visit["block_id"] != block:
                continue
            if "poll" in visit["kind"]:
                color = COLORS[1]
            elif "load" in visit["kind"] or "store" in visit["kind"]:
                color = COLORS[0]
            else:
                color = COLORS[2]
            begin, finish = visit["started_at_ps"] / 1e6, visit["finished_at_ps"] / 1e6
            ax.broken_barh([(begin, finish - begin)], (index - 0.22, 0.44), facecolors=color)
    ax.set_yticks(range(len(block_ids)), [f"Channel {i}" for i in range(len(block_ids))])
    ax.set_xlabel("Elapsed collective time (µs)")
    ax.set_ylim(-0.6, len(block_ids) - 0.4)
    ax.set_xlim(0, trace["completed_at_ps"] / 1e6 * 1.04)
    fig.legend(handles=[Line2D([], [], lw=5, color=color, label=label) for color, label in
                        [("#e4e9ed", "Resident block"), (COLORS[0], "Memory work"), (COLORS[1], "Polling issue"),
                         (COLORS[2], "Other GPU work")]], loc="upper left", bbox_to_anchor=(0.12, 0.85),
               ncol=4, fontsize=7, frameon=False, columnspacing=1.0)
    fig.text(0.14, 0.102, "Gray intervals include waiting. Colored spans show executed service, not an additive latency stack.", fontsize=8, color="#61707b")
    fig.text(0.14, 0.060, "GPU cycles are declared inputs. Network packets and returned capacity continue while a block waits.", fontsize=8, color="#61707b")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure()
    with PdfPages(args.output / "channel-fifo-review.pdf") as pdf:
        for name, draw in [("mechanisms", mechanism), ("a100-shared-work", retrospective),
                           ("protocol-resources", protocol_curves), ("parameter-sensitivity", sensitivity),
                           ("channel-timeline", timeline)]:
            fig = draw(args.data)
            fig.savefig(args.output / f"{name}.png", dpi=240)
            fig.savefig(args.output / f"{name}.pdf")
            pdf.savefig(fig)
            plt.close(fig)
    print(args.output / "channel-fifo-review.pdf")


if __name__ == "__main__":
    main()
