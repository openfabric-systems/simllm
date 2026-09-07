"""Controlled receiver-service discrimination for BACK-68."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from simllm.backends.fct import (
    earliest_completion_byte_floors,
    normalized_fct,
    normalized_phase_makespan,
)
from simllm.backends.htsim_rnic import (
    HtsimRnicConfig,
    build_htsim_rnic_command,
    parse_completion_csv,
    run_htsim_rnic,
)
from simllm.goal import GoalTrace, to_binary

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FREEZE = "7078cc5"
TOPOLOGY = REPO / "examples/m1/topologies/clos_64_400g.topo"
PROFILES = ("rnic-nn", "rnic-cn")
RATES = (400, 200)


@dataclass(frozen=True)
class Cell:
    kind: str
    fan_in: int
    payload: int
    placement: str
    rate: int
    profile: str

    @property
    def name(self):
        return (f"{self.kind}-f{self.fan_in}-b{self.payload}-{self.placement}"
                f"-{self.rate}g-{self.profile}")

    @property
    def pair_key(self):
        return self.name.rsplit("-rnic-", 1)[0]

    @property
    def sources(self):
        if self.placement == "spread":
            order = [leaf * 8 + rail for rail in range(8) for leaf in range(8)]
            return [r for r in order if r != 0][:self.fan_in]
        first = 1 if self.placement == "local" else 8
        return list(range(first, first + self.fan_in))

    @property
    def messages(self):
        return [(s, 0, 1000, self.payload * (4 if self.kind == "pair" and i else 1))
                for i, s in enumerate(self.sources)]

    def propagation(self, source):
        return 2_000_000 if self.profile == "rnic-nn" or source // 8 == 0 else 4_000_000


def cells():
    for rate in RATES:
        for profile in PROFILES:
            for placement in ("local", "remote"):
                for size in (262144, 1048576):
                    yield Cell("pair", 2, size, placement, rate, profile)
                for size in (65536, 1048576):
                    yield Cell("isolated", 1, size, placement, rate, profile)
            for fan_in in (8, 16, 32):
                for size in (65536, 1048576):
                    yield Cell("incast", fan_in, size, "spread", rate, profile)


def text_bytes(path, text):
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def write_json(path, content):
    text_bytes(path, json.dumps(content, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    text_bytes(path, stream.getvalue())


def digest(path, normalize=False):
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if normalize else raw).hexdigest()


def check(family, ok, **evidence):
    return {"family": family, "ok": bool(ok), **evidence}


def build_trace(cell):
    trace = GoalTrace(64)
    for source, destination, tag, size in cell.messages:
        trace.rank(source).send(size, destination, tag)
        trace.rank(destination).recv(size, source, tag)
    return trace


def limits(cell):
    quantum = 8000 // cell.rate
    total = sum(m[3] for m in cell.messages)
    propagation = min(cell.propagation(s) for s in cell.sources)
    fluid = total * quantum + 2_000_000
    return {
        "ps_per_byte": quantum,
        "common_propagation_ps": propagation,
        "phase_byte_floor_ps": total * quantum + propagation,
        "fluid_phase_ps": fluid,
        "fluid_small_ps": 2 * cell.payload * quantum + 2_000_000
        if cell.kind == "pair" else fluid,
        "phase_ceiling_ps": (total // 4096 + 2) * 4160 * quantum + 2_000_000
        if cell.profile == "rnic-nn" else None,
        "individual_floors_ps": {str(s): size * quantum + cell.propagation(s)
                                 for s, _, _, size in cell.messages},
    }


def prefix_floors(flows, rate_bps, propagation_ps):
    """Project the reusable receiver floor into reviewable CSV/JSON rows."""
    return [{**asdict(p), "slack_ps": p.slack_ps, "ok": p.ok} for p in
            earliest_completion_byte_floors(flows, link_rate_bps=rate_bps,
                                           propagation_ps=propagation_ps)]


def phase_span(flows):
    if not flows:
        raise ValueError("phase must contain completions")
    return max(f.completion_time_ps for f in flows) - min(f.start_time_ps for f in flows)


def measure(cell, flows, quiescent):
    bound = limits(cell)
    floors = prefix_floors(flows, cell.rate * 1_000_000_000, bound["common_propagation_ps"])
    guards = [
        check("quiescence", quiescent),
        check("identity", Counter((f.source, f.destination, f.tag, f.payload_bytes)
                                  for f in flows) == Counter(cell.messages)),
        check("flow_id", len({f.flow_id for f in flows}) == len(flows)),
        check("aligned_starts", all(f.start_time_ps == 0 for f in flows)),
        check("timestamps", all(f.fct_ps > 0 and
                                f.completion_time_ps - f.start_time_ps == f.fct_ps for f in flows)),
        check("individual_floor", all(f.fct_ps >= f.payload_bytes * bound["ps_per_byte"]
                                      + cell.propagation(f.source) for f in flows)),
        check("prefix_floor", all(p["ok"] for p in floors),
              failed_rows=[p for p in floors if not p["ok"]]),
        check("phase_byte_floor", phase_span(flows) >= bound["phase_byte_floor_ps"]),
    ]
    ordered = sorted(flows, key=lambda f: f.fct_ps)
    median = ordered[math.ceil(.5 * len(ordered)) - 1]
    return {
        "bounds": bound, "guards": guards, "flow_count": len(flows),
        "phase_makespan_ps": phase_span(flows), "p50_ps": median.fct_ps,
        "p50_flow_floor_ps": median.payload_bytes * bound["ps_per_byte"]
        + cell.propagation(median.source),
        "minimum_prefix_slack_ps": min(p["slack_ps"] for p in floors),
        "prefix_rows": floors,
    }


def compare(physical, ideal):
    ratios = normalized_fct(physical, ideal)
    rows = [{**asdict(r), "slowdown": r.slowdown} for r in ratios]
    return {
        "flow_ratios": rows,
        "phase_ratio": normalized_phase_makespan(physical, ideal).slowdown,
        "phase_baseline_guard": check("phase_baseline_floor", phase_span(physical) >= phase_span(ideal),
                                      physical_ps=phase_span(physical), ideal_ps=phase_span(ideal)),
        "below_one": sum(r.slowdown < 1 for r in ratios),
        "min_flow_ratio": min(r.slowdown for r in ratios),
    }


def decide(rows, expected=56):
    violations = [{"cell": r["name"], **g} for r in rows
                  for g in r.get("guards", []) if not g["ok"]]
    deciding = [v for v in violations if v["family"] in
                ("prefix_floor", "phase_byte_floor", "phase_baseline_floor")]
    errors = [r["name"] for r in rows if r["status"] != "complete"]
    sub_one = [r for r in rows if r.get("comparison", {}).get("below_one", 0)]
    isolated = [r["name"] for r in sub_one if r["fan_in"] == 1]
    complete = len(rows) == expected and not errors
    decision = ("H-credit" if deciding else "void" if violations else "incomplete"
                if not complete else "inconclusive" if isolated or not sub_one else "H-sched")
    return {"decision": decision, "status": "void" if violations else
            "complete" if complete else "incomplete", "fatal_violations": violations,
            "errors": errors, "sub_one_cells": [r["name"] for r in sub_one],
            "isolated_sub_one_cells": isolated,
            "behavioral_score_interpretable": complete and not violations}


def analyze(rows, out):
    for row in rows:
        if row["status"] == "complete" and row["profile"] == "rnic-nn":
            row["fluid_reference_residuals"] = [
                {"source": f["source"], "packet_fct_ps": f["fct_ps"],
                 "fluid_reference_ps": row["bounds"]["fluid_small_ps"]
                 if row["kind"] == "pair" and f["payload_bytes"] == row["payload"]
                 else row["bounds"]["fluid_phase_ps"],
                 "residual_ps": f["fct_ps"] - (row["bounds"]["fluid_small_ps"]
                 if row["kind"] == "pair" and f["payload_bytes"] == row["payload"]
                 else row["bounds"]["fluid_phase_ps"])} for f in row["flows"]]
    by_name = {r["name"]: r for r in rows}
    for row in rows:
        cell = Cell(**{k: row[k] for k in Cell.__dataclass_fields__})
        if cell.profile != "rnic-cn" or row["status"] != "complete":
            continue
        base_name = cell.name.replace("rnic-cn", "rnic-nn")
        base = by_name[base_name]
        if base["status"] != "complete":
            row["guards"].append(check("baseline_available", False))
            continue
        row["guards"].append(check("identical_goal", row["goal_sha256"] == base["goal_sha256"]))
        physical = parse_completion_csv(out / cell.name / "completion.csv")
        ideal = parse_completion_csv(out / base_name / "completion.csv")
        comparison = compare(physical, ideal)
        row["guards"].append(comparison["phase_baseline_guard"])
        row["comparison"] = comparison
        write_csv(out / cell.name / "normalization.csv", comparison["flow_ratios"])
    behavioral = []
    for row in rows:
        if row["status"] != "complete":
            continue
        if row["profile"] == "rnic-nn":
            behavioral.append(check("ideal_envelope", row["phase_makespan_ps"] <=
                                    row["bounds"]["phase_ceiling_ps"], cell=row["name"]))
        if row["rate"] == 400:
            slower = by_name[row["name"].replace("-400g-", "-200g-")]
            if slower["status"] == "complete":
                behavioral.append(check("rate_direction", slower["phase_makespan_ps"] >=
                                        row["phase_makespan_ps"], cell=row["name"]))
        if (row["kind"], row["payload"]) in (("pair", 262144), ("incast", 65536)):
            larger = by_name[row["name"].replace(f"-b{row['payload']}-", "-b1048576-")]
            if larger["status"] == "complete":
                behavioral.append(check("payload_direction", larger["phase_makespan_ps"] >
                                        row["phase_makespan_ps"], cell=row["name"]))
        if row["kind"] == "incast" and row["fan_in"] < 32:
            wider = by_name[row["name"].replace(f"-f{row['fan_in']}-", f"-f{2 * row['fan_in']}-")]
            if wider["status"] == "complete":
                behavioral.append(check("fan_in_direction", wider["phase_makespan_ps"] >
                                        row["phase_makespan_ps"], cell=row["name"]))
    return behavioral


def plot(report, destination):
    """Per-flow and phase ratios against senders, plus the byte-floor prefix curve.

    The left panel carries every physical cell: isolated controls at one
    sender, the two-flow pairs at two, and the many-to-one cells at 8 to 32.
    Only shared-receiver cells may dip below the unit line, which is the
    BACK-68 finding; the right panel shows the fatal completion-prefix floor
    for the cell with the smallest per-flow ratio.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in report["configurations"] if "comparison" in r]
    incast = [r for r in rows if r["kind"] == "incast"]
    if not incast:
        return
    selected = min(incast, key=lambda r: r["comparison"]["min_flow_ratio"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), layout="constrained",
                             gridspec_kw={"width_ratios": (1.25, 1)})
    placement_markers = {"local": "o", "remote": "^", "spread": "o"}
    for payload in (65536, 262144, 1048576):
        for rate in RATES:
            series = sorted([r for r in rows if r["payload"] == payload and r["rate"] == rate],
                            key=lambda r: (r["fan_in"], r["placement"]))
            if not series:
                continue
            subset = [r for r in series if r["kind"] == "incast"]
            label = f"{payload // 1024} KiB, {rate}G"
            if subset:
                line, = axes[0].plot([r["fan_in"] for r in subset],
                                    [r["comparison"]["min_flow_ratio"] for r in subset], "o-",
                                    label=label)
                color = line.get_color()
                axes[0].plot([r["fan_in"] for r in subset],
                             [r["comparison"]["phase_ratio"] for r in subset], "s--",
                             color=color)
            else:
                color = None
            for r in series:
                if r["kind"] == "incast":
                    continue
                marker = placement_markers.get(r["placement"], "o")
                handle, = axes[0].plot([r["fan_in"]], [r["comparison"]["min_flow_ratio"]],
                                       marker=marker, linestyle="none", markersize=6,
                                       markerfacecolor="white", color=color,
                                       label=None if color else label)
                if color is None:
                    color = handle.get_color()
    axes[0].axhline(1, color="black", linewidth=.8)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks((1, 2, 8, 16, 32))
    axes[0].get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].get_xaxis().set_minor_formatter(plt.NullFormatter())
    axes[0].set(xlabel="Senders sharing the receiver link (count)",
                ylabel="Physical / ideal completion (ratio)",
                title="Per-flow minimum (solid) and phase makespan (dashed)")
    axes[0].text(1.02, 0.74, "unit line: a per-flow lower bound only\nwhen the receiver "
                 "link is unshared", fontsize=7, va="bottom")
    axes[0].text(0.36, 0.60, "hollow: isolated (1) and two-flow (2) controls,\n"
                 "circle local leaf, triangle remote leaf; none below 1",
                 transform=axes[0].transAxes, fontsize=7, ha="left", va="bottom", color="0.3")
    axes[0].legend(fontsize=7, loc="upper right", ncol=2)
    prefixes = selected["prefix_rows"]
    axes[1].plot([p["k"] for p in prefixes], [p["elapsed_ps"] / p["floor_ps"] for p in prefixes],
                 "o-", markersize=4)
    axes[1].axhline(1, color="black", linewidth=.8)
    axes[1].text(0.98, 0.9, "fatal floor: every k at or above 1", transform=axes[1].transAxes,
                 fontsize=7, ha="right", va="top", color="0.3")
    axes[1].set(xlabel="Earliest completions k (count)",
                ylabel="Elapsed / cumulative byte floor (ratio)",
                title=f"Completion-prefix floor, F={selected['fan_in']}, "
                      f"{selected['payload'] // 1024} KiB, {selected['rate']}G")
    axes[1].set_ylim(bottom=0)
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"aligned_baseline.{suffix}", dpi=180,
                    metadata={"CreationDate": None} if suffix == "pdf" else None)
    plt.close(fig)


def report_markdown(report):
    rows = report["configurations"]
    physical = [r for r in rows if "comparison" in r]
    prefixes = [p for r in rows for p in r.get("prefix_rows", [])]
    ratios = [n for r in physical for n in r["comparison"]["flow_ratios"]]
    verdict = report["verdict"]
    physical_prefixes = [p for r in physical for p in r["prefix_rows"]]
    minimum_row = min(physical, key=lambda r: r["comparison"]["min_flow_ratio"])
    minimum_flow = min(minimum_row["comparison"]["flow_ratios"], key=lambda f: f["slowdown"])
    lines = ["# Aligned baseline: controlled receiver service", "",
             "## What ran", "",
             f"Expectations-only commit: `{report['provenance']['expectations_commit']}`.",
             "Pinned htsim `617ce20`: 16 two-flow cells, 24 many-to-one cells and",
             "16 isolated controls, at 400/200 Gbit/s with rnic-nn and rnic-cn.",
             "All 56 attempts use fresh external bulk storage and identical GOAL phases.", "",
             "## What came out", "",
             "Before these observations: each flow has floor bytes*8000/rate_Gbps",
             "+ path propagation in ps. Propagation is 2,000,000 ps for ideal or",
             "same-leaf traffic, 4,000,000 ps for remote physical traffic. The",
             "physical ceiling is unbounded. The frozen ideal packet envelope is",
             "(total_payload/4096+2)*4160*8000/rate_Gbps+2,000,000 ps.", "",
             f"Decision under the frozen rule: **{verdict['decision']}**; status **{verdict['status']}**.",
             f"Completion-prefix rows: {len(prefixes)}; failures: {sum(not p['ok'] for p in prefixes)}.",
             (f"Physical prefixes: {len(physical_prefixes)}, with minimum slack "
             f"{min(p['slack_ps'] for p in physical_prefixes)} ps."),
             f"Fatal guard violations: {len(verdict['fatal_violations'])}.",
             f"Physical per-flow ratios below one: {sum(r['slowdown'] < 1 for r in ratios)}/{len(ratios)}.",
             f"Minimum per-flow ratio: {min(r['slowdown'] for r in ratios):.9f}.",
             f"Minimum phase ratio: {min(r['comparison']['phase_ratio'] for r in physical):.9f}.",
             f"Isolated controls with a below-one ratio: {len(verdict['isolated_sub_one_cells'])}.", "",
             "Numbers below are ps except dimensionless ratios. p50 is nearest rank",
             "within each cell, with the selected median flow's own floor beside it.",
             "For two unequal flows, p50 is the earlier completion, not an average.", "",
             "| Cell | NN p50 / floor | CN p50 / floor | CN phase / NN | Minimum flow ratio |",
             "|---|---|---|---|---|"]
    by_name = {r["name"]: r for r in rows}
    for r in physical:
        b = by_name[r["name"].replace("rnic-cn", "rnic-nn")]
        label = r["name"].removesuffix("-rnic-cn")
        lines.append(f"| {label} | {b['p50_ps']} / {b['p50_flow_floor_ps']} | "
                     f"{r['p50_ps']} / {r['p50_flow_floor_ps']} | "
                     f"{r['comparison']['phase_ratio']:.6f} | {r['comparison']['min_flow_ratio']:.6f} |")
    lines += ["", "## Deciding rows and bounds", "",
              "`results.json` is the authority for every per-flow FCT and ratio, every",
              "k-prefix byte count, elapsed time, floor and slack, and each phase bound.",
              "Its command argv uses SIMLLM_* substitutions; the bulk command.json",
              "retains the exact executed argv. Manifests verify physical quiescence.", ""]
    lines += [
        (f"The minimum ratio belongs to `{minimum_row['name']}`, source "
        f"{minimum_flow['source']} to 0, tag 1000: physical {minimum_flow['fct_ps']} ps / "
        f"ideal {minimum_flow['baseline_fct_ps']} ps. Its phase ratio is "
        f"{minimum_row['comparison']['phase_ratio']:.9f}."),
        "All sub-one rows occur in three F=32 incasts (5, 2 and 3 flows).",
        "Every two-flow small ratio is above one, so the pair experiment alone",
        "is nondiscriminating. The larger shared matrix separates the predictions.",
        "No isolated or F=8/16 flow beats its ideal completion.", "",
        "The two-flow ideal uses packet reservations, not the exact fluid FCT.",
        "The table exposes both sizes; fluid points remain the frozen reference",
        "and per-flow packet-minus-fluid residuals are retained in JSON.", "",
        "| S bytes | Rate Gbit/s | Fluid small / large ps | Packet NN small / large ps | CN local small / large ratio | CN remote small / large ratio |",
        "|---|---|---|---|---|---|"]
    for size in (262144, 1048576):
        for rate in RATES:
            local = by_name[Cell("pair", 2, size, "local", rate, "rnic-cn").name]
            remote = by_name[Cell("pair", 2, size, "remote", rate, "rnic-cn").name]
            base = by_name[Cell("pair", 2, size, "local", rate, "rnic-nn").name]
            nn = sorted(base["flows"], key=lambda f: f["payload_bytes"])
            lr = sorted(local["comparison"]["flow_ratios"], key=lambda f: f["payload_bytes"])
            rr = sorted(remote["comparison"]["flow_ratios"], key=lambda f: f["payload_bytes"])
            lines.append(f"| {size} | {rate} | {base['bounds']['fluid_small_ps']} / "
                         f"{base['bounds']['fluid_phase_ps']} | {nn[0]['fct_ps']} / "
                         f"{nn[1]['fct_ps']} | {lr[0]['slowdown']:.6f} / {lr[1]['slowdown']:.6f} | "
                         f"{rr[0]['slowdown']:.6f} / {rr[1]['slowdown']:.6f} |")
    lines += ["", (f"Behavioral relations: {len(report['behavioral_relations'])} instances, "
              f"{sum(not c['ok'] for c in report['behavioral_relations'])} misses. "
              "These envelope and direction checks are separate from fatal guards.")]
    for v in verdict["fatal_violations"]:
        lines += [f"- `{v['cell']}`: `{json.dumps(v, sort_keys=True)}`"]
    lines += ["", "## What it changes", "",
              "BACK-68's shared-flow lower-bound refutation is resolved by H-sched",
              "under the registered decision rule. simllm/backends/fct.py supplies",
              "normalized_phase_makespan and earliest_completion_byte_floors, used",
              "by this report and the fresh width-tail rerun. normalized_fct retains",
              "its existing behavior; unshared aligned handoffs retain their bound.",
              "Only orchestrator integration closure remains in BACK-68: remove its",
              "entry while regenerating the protected task-progress projection.",
              "The width-tail guard amendment is frozen separately at `3d818f7`.",
              "Its [rerun report](../collective_width_tail_v1/RESULTS.md) records the",
              "unchanged original ideal values and the two surviving void cells.", "",
              "## What it does not change", "",
              "A finite set of endpoint completion floors cannot exclude every hidden",
              "credit defect or establish packet-level causality. This experiment tests",
              "the two registered explanations on the specified cells. No native backend or",
              "third_party source is changed. No production request-tail calibration,",
              "COMP-9 closure, BACK-38 state preservation or HTSIM-40 fix is claimed.", "",
              "## Reproduction", "", "```bash", ". ./.env.local.sh",
              ".venv/bin/python examples/aligned_baseline_v1/run_study.py --publish", "```", "",
              "Set SIMLLM_DATA_ROOT, SIMLLM_HTSIM_BUILD, SIMLLM_HTSIM_RNIC and",
              "SIMLLM_TXT2BIN first. The runner rejects an existing evidence directory.",
              "Use --summarize-only to project retained raw evidence without simulation.", "",
              "![Controlled receiver-service ratios](figures/aligned_baseline.png)", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    root = os.environ.get("SIMLLM_DATA_ROOT")
    if args.out is None and not root:
        parser.error("set SIMLLM_DATA_ROOT or --out")
    out = (args.out or Path(root) / "aligned_baseline_v1").resolve()
    if out == REPO or REPO in out.parents:
        parser.error("bulk output must be external to the worktree")
    if out.exists() and not args.summarize_only:
        parser.error("fresh evidence directory required")
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        provenance = json.loads((out / "provenance.json").read_text())
        if provenance["expectations_sha256"] not in {
            digest(HERE / "expectations.md"), digest(HERE / "expectations.md", True)
        }:
            parser.error("expectations changed since simulation")
    else:
        names = ("SIMLLM_HTSIM_BUILD", "SIMLLM_HTSIM_RNIC", "SIMLLM_TXT2BIN")
        if any(not os.environ.get(n) or not Path(os.environ[n]).exists() for n in names):
            parser.error("configure SIMLLM_HTSIM_BUILD, SIMLLM_HTSIM_RNIC and SIMLLM_TXT2BIN")
        provenance = {
            "expectations_commit": subprocess.check_output(
                ["git", "rev-parse", FREEZE], cwd=REPO, text=True).strip(),
            "expectations_sha256": digest(HERE / "expectations.md", True),
            "simulation_script_sha256": digest(Path(__file__), True),
            "htsim_pin": subprocess.check_output(
                ["git", "rev-parse", "HEAD:third_party/htsim"], cwd=REPO, text=True).strip(),
            "binaries_sha256": {n: digest(Path(os.environ[n])) for n in names[1:]},
            "topology_sha256": digest(TOPOLOGY, True),
        }
        write_json(out / "provenance.json", provenance)
    rows = []
    for cell in cells():
        directory = out / cell.name
        if args.summarize_only:
            row = json.loads((directory / "cell.json").read_text())
            rows.append(row)
            continue
        directory.mkdir()
        goal = directory / "phase.goal"
        text_bytes(goal, build_trace(cell).render())
        topology = out / f"clos_64_{cell.rate}g.topo"
        text_bytes(topology, TOPOLOGY.read_text().replace(
            "Downlink_speed_Gbps 400", f"Downlink_speed_Gbps {cell.rate}"))
        row = {**asdict(cell), "name": cell.name, "bounds": limits(cell),
               "goal_sha256": digest(goal), "status": "error"}
        # Persist physical bounds before reading any backend completion.
        write_json(directory / "bounds.json", row["bounds"])
        try:
            cfg = HtsimRnicConfig(goal_bin=to_binary(goal), profile=cell.profile,
                                  linkspeed_bps=cell.rate * 1_000_000_000,
                                  topology=topology if cell.profile == "rnic-cn" else None,
                                  completion_csv=directory / "completion.csv")
            argv = build_htsim_rnic_command(Path(os.environ["SIMLLM_HTSIM_RNIC"]), cfg)
            write_json(directory / "command.json", argv)
            row["command_argv"] = [a.replace(os.environ["SIMLLM_HTSIM_RNIC"],
                                             "${SIMLLM_HTSIM_RNIC}").replace(
                                                 str(out), "${SIMLLM_STUDY_OUT}") for a in argv]
            run = run_htsim_rnic(cfg)
            row.update(measure(cell, run.flows, run.quiescent))
            row["flows"] = [{k: getattr(f, k) for k in (
                "flow_id", "source", "destination", "tag", "payload_bytes",
                "start_time_ps", "completion_time_ps", "fct_ps")} for f in run.flows]
            row["manifest"] = [s.replace(str(out), "${SIMLLM_STUDY_OUT}") for s in run.manifest]
            write_csv(directory / "prefix_floors.csv", row["prefix_rows"])
            row["status"] = "complete"
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
            text_bytes(directory / "error.txt", str(exc) + "\n")
            row["error"] = type(exc).__name__
        write_json(directory / "cell.json", row)
        rows.append(row)
        print(f"{cell.name}: {row['status']}", flush=True)
    behavioral = analyze(rows, out)
    report = {"provenance": provenance, "analysis_script_sha256": digest(Path(__file__), True),
              "configurations": rows, "behavioral_relations": behavioral, "verdict": decide(rows)}
    write_json(out / "results.json", report)
    text_bytes(out / "RESULTS.md", report_markdown(report))
    plot(report, out / "figures")
    if args.publish:
        write_json(HERE / "results.json", report)
        text_bytes(HERE / "RESULTS.md", report_markdown(report))
        plot(report, HERE / "figures")
    print(json.dumps(report["verdict"], indent=2), flush=True)
    if report["verdict"]["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
