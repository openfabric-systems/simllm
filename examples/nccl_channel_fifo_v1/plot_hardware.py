"""Plot realized controls and ordinary timings before fitting GPU service."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_capture import geometry
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from plot_results import COLORS, configure, decorate, read

PROTOCOLS = ["LL", "LL128", "SIMPLE"]


def selection_rows(root, captures):
    rows = []
    for name in captures:
        architecture, capture = name.split(":")
        raw = root / architecture / capture / "raw"
        for cell in json.loads((raw / "conditions.json").read_text()):
            for size, selected in geometry(raw, cell).items():
                rows.append({"architecture": architecture, "family": cell["family"], "width": cell["width"],
                             "bytes": size, "protocol": selected["protocol"], "channels": selected["channels"],
                             "requested_channels": cell["channels"]})
    return rows


def frame(architectures, title, subtitle, *, height=3.5):
    fig, axes = plt.subplots(len(architectures), 3, figsize=(7.2, height * len(architectures) + 1), squeeze=False)
    fig.subplots_adjust(left=0.105, right=0.98, top=1 - 0.95 / fig.get_figheight(),
                        bottom=0.85 / fig.get_figheight(), hspace=0.45, wspace=0.32)
    fig.text(0.105, 1 - 0.23 / fig.get_figheight(), title, fontsize=12, weight="bold")
    fig.text(0.105, 1 - 0.48 / fig.get_figheight(), subtitle, fontsize=8.2, color="#61707b")
    return fig, axes


def realized_channels(rows, architectures):
    fig, axes = frame(architectures, "A channel request is a budget, not the executed channel count",
                      "Separate selection observers; four GPUs; no timing values used", height=2.35)
    handles = []
    for index, architecture in enumerate(architectures):
        for ax, protocol in zip(axes[index], PROTOCOLS, strict=True):
            for color, size in zip(COLORS[:3], [262144, 1048576, 3145728], strict=True):
                subset = sorted([r for r in rows if r["architecture"] == architecture and r["family"] == "channels"
                                 and int(r["width"]) == 4 and r["protocol"] == protocol and int(r["bytes"]) == size],
                                key=lambda r: int(r["requested_channels"]))
                line, = ax.plot([int(r["requested_channels"]) for r in subset], [int(r["channels"]) for r in subset],
                                "o-", color=color, ms=3, lw=1.1, label=f"{size / 1048576:g} MiB")
                if index == 0 and protocol == "LL":
                    handles.append(line)
            ax.plot([0, 32], [0, 32], ":", color="#9aa3aa", lw=0.8)
            ax.set(title=f"{architecture.upper()}  {protocol.title() if protocol == 'SIMPLE' else protocol}",
                   xlabel="Requested channels", xticks=[1, 8, 16, 24, 32], yticks=[0, 8, 16, 24, 32])
            ax.set_xlim(-0.5, 33.5)
            ax.set_ylim(-1, 34)
        axes[index, 0].set_ylabel("Active channels")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.09, 1 - 0.56 / fig.get_figheight()),
               ncol=3, fontsize=8, frameon=False)
    decorate(axes)
    fig.text(0.105, 0.10 / fig.get_figheight(), "NCCL partitions the payload into work cells. Fewer active channels invalidate a fixed-request causal contrast.",
             fontsize=7.8, color="#61707b")
    return fig


def error_curve(ax, rows, xvalues, color, label, metric="event_us", mark_unqualified=True):
    y = np.array([float(r[metric]) for r in rows])
    lo, hi = (np.array([float(r[metric + suffix]) for r in rows]) for suffix in ("_q1", "_q3"))
    line = ax.errorbar(xvalues, y, yerr=[y - lo, hi - y], fmt="o-", color=color, ms=3,
                       lw=1.1, capsize=2, label=label)
    if mark_unqualified:
        invalid = np.array([str(r["qualified_control"]).lower() != "true" for r in rows])
        ax.scatter(np.asarray(xvalues)[invalid], y[invalid], marker="x", color="#25313b", s=23, zorder=5)
    return line


def channel_timings(rows, architectures):
    fig, axes = frame(architectures, "Channel count changes overlap and shared work",
                      "Ordinary GPU-event timing; 1 MiB per GPU; medians and interquartile ranges", height=2.7)
    handles = []
    for index, architecture in enumerate(architectures):
        for ax, protocol in zip(axes[index], PROTOCOLS, strict=True):
            for width, color in [(2, COLORS[1]), (4, COLORS[0])]:
                subset = sorted([r for r in rows if r["architecture"] == architecture and r["family"] == "channels"
                                 and int(r["width"]) == width and r["protocol"] == protocol and int(r["bytes"]) == 1048576],
                                key=lambda r: int(r["requested_channels"]))
                line = error_curve(ax, subset, [int(r["requested_channels"]) for r in subset], color, f"{width} GPUs")
                if index == 0 and protocol == "LL":
                    handles.append(line)
            ax.set(title=f"{architecture.upper()}  {protocol.title() if protocol == 'SIMPLE' else protocol}",
                   xlabel="Requested channels", xticks=[1, 8, 16, 24, 32])
        axes[index, 0].set_ylabel("Completion time (µs)")
    handles.append(Line2D([], [], marker="x", linestyle="none", color="#25313b", label="Request not realized"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.09, 1 - 0.56 / fig.get_figheight()), ncol=3, fontsize=8, frameon=False)
    decorate(axes)
    fig.text(0.105, 0.36 / fig.get_figheight(), "Five independent processes per point. Crosses retain observations whose requested control was not realized.",
             fontsize=7.8, color="#61707b")
    fig.text(0.105, 0.14 / fig.get_figheight(), "Identification data only. Each allocation is also summarized separately; no model curve or fitted band is shown.",
             fontsize=7.8, color="#61707b")
    return fig


def resource_timings(rows, architectures):
    fig, axes = frame(architectures, "Restricting GPU execution leaves the link hardware unchanged",
                      "Four GPUs; 1 MiB per GPU; source-valid protocol and verified stream context", height=2.7)
    handles = []
    for index, architecture in enumerate(architectures):
        for ax, protocol in zip(axes[index], PROTOCOLS, strict=True):
            for channels, color in zip([8, 24, 32], COLORS, strict=False):
                subset = sorted([r for r in rows if r["architecture"] == architecture and r["family"] == "resources"
                                 and int(r["width"]) == 4 and r["protocol"] == protocol and int(r["bytes"]) == 1048576
                                 and int(r["requested_channels"]) == channels], key=lambda r: int(r["granted_sms"]))
                line = error_curve(ax, subset, list(range(len(subset))), color, f"Requested C={channels}")
                if index == 0 and protocol == "LL":
                    handles.append(line)
                labels = [str(r["granted_sms"]) if int(r["requested_sms"]) else f"Full\n({r['granted_sms']})" for r in subset]
                ax.set_xticks(range(len(subset)), labels)
            ax.set(title=f"{architecture.upper()}  {protocol.title() if protocol == 'SIMPLE' else protocol}",
                   xlabel="Granted SMs per GPU")
        axes[index, 0].set_ylabel("Completion time (µs)")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.09, 1 - 0.56 / fig.get_figheight()), ncol=3, fontsize=7.7, frameon=False)
    decorate(axes)
    fig.text(0.105, 0.36 / fig.get_figheight(), "8/16/32 SMs use CUDA green contexts; Full uses an ordinary stream. Bars show five-process interquartile ranges.",
             fontsize=7.6, color="#61707b")
    fig.text(0.105, 0.14 / fig.get_figheight(), "SM: streaming multiprocessor. Crosses mark unrealized channel requests; no pure SM-cost attribution is implied.",
             fontsize=7.7, color="#61707b")
    return fig


def residual_timings(rows, architectures):
    fig, axes = plt.subplots(len(architectures), 2, figsize=(7.2, 2.7 * len(architectures) + 1.6), squeeze=False)
    h = fig.get_figheight()
    fig.subplots_adjust(left=0.105, right=0.98, top=1 - 1.05 / h, bottom=0.85 / h, hspace=0.5, wspace=0.30)
    fig.text(0.105, 1 - 0.25 / h, "A denser look at the four-GPU residual window", fontsize=12, weight="bold")
    fig.text(0.105, 1 - 0.51 / h, "33 payloads at 16-KiB spacing; five ordinary timing processes per point", fontsize=8.4, color="#61707b")
    handles = []
    for index, architecture in enumerate(architectures):
        for channels, color in zip([0, 22, 23, 24], COLORS, strict=False):
            subset = sorted([r for r in rows if r["architecture"] == architecture and r["family"] == "residual"
                             and int(r["width"]) == 4 and int(r["requested_channels"]) == channels], key=lambda r: int(r["bytes"]))
            x = [int(r["bytes"]) / 1048576 for r in subset]
            label = "Default selection" if channels == 0 else f"LL128, request {channels}"
            line = error_curve(axes[index, 0], subset, x, color, label)
            axes[index, 1].step(x, [int(r["channels"]) for r in subset], where="mid", color=color, lw=1.2)
            if index == 0:
                handles.append(line)
        axes[index, 0].set(title=f"{architecture.upper()}  Measured latency", ylabel="GPU-event time (µs)")
        axes[index, 1].set(title=f"{architecture.upper()}  Observed selection", ylabel="Active channels")
        for ax in axes[index]:
            ax.set(xlabel="Payload per GPU (MiB)", xticks=[3, 3.125, 3.25, 3.375, 3.5])
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.09, 1 - 0.63 / h), ncol=4, fontsize=7.1, frameon=False)
    decorate(axes)
    fig.text(0.105, 0.35 / h, "Bars are process interquartile ranges. Crosses mark unrealized requested channels, retained as observations.", fontsize=7.8, color="#61707b")
    fig.text(0.105, 0.13 / h, "Source work and actual channel partitions must explain these curves before any new hardware accuracy claim.", fontsize=7.8, color="#61707b")
    return fig


def paired_timers(rows, architectures):
    fig, axes = frame(architectures, "Timer boundaries are paired on the same invocation",
                      "Four GPUs; 1 MiB per GPU; five independent processes per condition", height=2.7)
    handles = []
    combinations = [(5, 20), (5, 100), (20, 20), (20, 100)]
    for index, architecture in enumerate(architectures):
        for ax, protocol in zip(axes[index], PROTOCOLS, strict=True):
            for rotate, color in [(0, COLORS[0]), (1, COLORS[1])]:
                subset = sorted([r for r in rows if r["architecture"] == architecture and r["family"] == "timers"
                                 and int(r["width"]) == 4 and r["protocol"] == protocol and int(r["bytes"]) == 1048576
                                 and int(r["rotate"]) == rotate], key=lambda r: combinations.index((int(r["warmup"]), int(r["iterations"]))))
                line = error_curve(ax, subset, range(len(subset)), color, "Rotating offsets" if rotate else "Fixed offset", metric="paired_gap_us")
                if index == 0 and protocol == "LL":
                    handles.append(line)
            ax.axhline(0, color="#83909b", lw=0.7)
            ax.set(title=f"{architecture.upper()}  {protocol.title() if protocol == 'SIMPLE' else protocol}",
                   xlabel="Warmup / timed iterations", xticks=range(4), xticklabels=[f"{w}/{t}" for w, t in combinations])
        axes[index, 0].set_ylabel("Host wall minus GPU event (µs)")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.09, 1 - 0.56 / fig.get_figheight()), ncol=2, fontsize=8, frameon=False)
    decorate(axes)
    fig.text(0.105, 0.36 / fig.get_figheight(), "Both timers include their own boundaries around the same launch loop. GPU events may include host issue gaps.", fontsize=7.6, color="#61707b")
    fig.text(0.105, 0.14 / fig.get_figheight(), "A paired difference is a measurement-boundary effect; it is not an isolated kernel-entry or host-service constant.", fontsize=7.7, color="#61707b")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--capture-metadata", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure()
    if args.data:
        summary = json.loads((args.data / "summary.json").read_text())
        if summary["status"] != "complete" or summary["global_guards"] != "valid":
            raise RuntimeError("only a complete audited timing summary may be plotted")
        rows = read(args.data / "measurements.csv")
        draws = [("realized-channels", realized_channels), ("channel-timings", channel_timings),
                 ("resource-timings", resource_timings), ("residual-window", residual_timings),
                 ("paired-timers", paired_timers)]
    else:
        if not args.root or not args.capture_metadata:
            parser.error("provide audited --data or --root and --capture-metadata")
        rows = selection_rows(args.root, args.capture_metadata)
        draws = [("realized-channels", realized_channels)]
    architectures = sorted({r["architecture"] for r in rows})
    with PdfPages(args.output / "hardware-controls-review.pdf") as pdf:
        for name, draw in draws:
            fig = draw(rows, architectures)
            fig.savefig(args.output / f"{name}.png", dpi=240)
            fig.savefig(args.output / f"{name}.pdf")
            pdf.savefig(fig)
            plt.close(fig)
    print(args.output / "hardware-controls-review.pdf")


if __name__ == "__main__":
    main()
