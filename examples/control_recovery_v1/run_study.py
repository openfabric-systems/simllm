"""Replay frozen physical schedules with bounded control admission protection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from simllm.backends._child_process import run_owned_process
from simllm.backends.htsim_rnic import (
    HtsimRnicConfig,
    _parse_goal_completion_time_ps,
    build_htsim_rnic_command,
    parse_completion_csv,
    parse_control_recovery_manifest,
    prepare_htsim_child_lifetime,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FREEZE = "e936f5d"
HTSIM_FREEZE = "10f66f5"
HTSIM_INDEX_FREEZE = "0adaf7e"
KINDS = ("declare", "accept", "grant_update", "gap_nack", "gap_resolved", "retire", "nflow_update")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_digests(path):
    data = Path(path).read_bytes()
    return sorted({hashlib.sha256(data).hexdigest(),
                   hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()})


def compatible_locks(saved, current):
    digest_fields = {"expectations_sha256", "goal_text_sha256", "topology_sha256"}
    return saved.keys() == current.keys() and all(
        bool(set(saved[key]) & set(current[key])) if key in digest_fields
        else saved[key] == current[key] for key in saved)


def write_json(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def git(*args, cwd=REPO):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


@dataclass(frozen=True)
class Cell:
    source: str
    reference: Path
    width: int
    rate: int
    pattern: str
    topology: Path
    goal_stem: str
    seed: str
    formerly_failed: bool

    @property
    def name(self):
        return f"{self.source}-{self.reference.name}"

    @property
    def main(self):
        return self.source == "collective" and self.pattern == "all-to-all"

    @property
    def ideal(self):
        return self.reference.with_name(self.reference.name.replace("rnic-cn", "rnic-nn"))

    @property
    def goal(self):
        return self.reference / f"{self.goal_stem}.goal"


def discover(collective_root, pipeline_root):
    cells = []
    for path in sorted(collective_root.glob("collective-*-rnic-cn")):
        match = re.fullmatch(r"collective-(ring|all-to-all)-w(\d+)-(\d+)g-rnic-cn", path.name)
        if match:
            pattern, width, rate = match.groups()
            cells.append(Cell("collective", path, int(width), int(rate), pattern,
                              collective_root / f"clos_64_{rate}g.topo", "collective",
                              "1311768467463790320", int(width) == 64 and pattern == "all-to-all"))
    for path in sorted(pipeline_root.glob("*-rnic-cn")):
        match = re.fullmatch(r"(rail|node-local)-s(2|8)-pp(2|4|8)-ep(0|8|32)-rnic-cn", path.name)
        if match:
            _, spines, _, width = match.groups()
            cells.append(Cell("pipeline", path, int(width), 400, "pipeline", path / "clos.topo",
                              "step", "1", spines == "2" and width == "32"))
    if len(cells) != 52 or sum(c.main for c in cells) != 8 or sum(c.formerly_failed for c in cells) != 8:
        raise ValueError("reference roots must contain 16 collective and 36 pipeline physical inputs")
    for cell in cells:
        for path in (cell.goal, cell.reference / f"{cell.goal_stem}.bin", cell.topology,
                     cell.ideal / "completion.csv"):
            if not path.is_file():
                raise FileNotFoundError(f"missing frozen reference input: {path}")
        if not cell.formerly_failed and not (cell.reference / "completion.csv").is_file():
            raise FileNotFoundError(f"missing completed reference CSV: {cell.name}")
    return cells


def messages(goal):
    rank = None
    result = Counter()
    for line in Path(goal).read_text().splitlines():
        if match := re.fullmatch(r"rank (\d+) \{", line.strip()):
            rank = int(match[1])
        if match := re.fullmatch(r"\S+: send (\d+)b to (\d+) tag (\d+)", line.strip()):
            size, dest, tag = map(int, match.groups())
            if rank is None:
                raise ValueError("send outside GOAL rank block")
            result[(rank, dest, tag, size)] += 1
    if not result:
        raise ValueError("reference GOAL has no sends")
    return result


def pre_run_bounds(cell):
    offered = messages(cell.goal)
    receiver_bytes = Counter()
    for (_, dest, _, size), count in offered.items():
        receiver_bytes[dest] += size * count
    # Two endpoints on the same leaf need two propagation hops; across
    # leaves they need four. All collective-width traffic crosses leaves.
    minimum_prop = 4_000_000 if cell.source == "collective" else 2_000_000
    phase_floor = max(receiver_bytes.values()) * 8000 // cell.rate + minimum_prop
    if cell.pattern == "ring":
        phase_floor = 2 * (cell.width - 1) * (1_048_576 // cell.width * 8000 // cell.rate + 4_000_000)
    if cell.source == "pipeline":
        saved = json.loads((cell.reference / "pre_run_bounds.json").read_text())
        phase_floor = max(phase_floor, saved["ep_phase_floor_ps"])
    return {"phase_floor_ps": phase_floor, "phase_ceiling_ps": None,
            "flow_payload_floor_ps": min(key[3] for key in offered) * 8000 // cell.rate,
            "flow_ceiling_ps": None, "phase_ratio_floor": 1 if cell.main else None,
            "phase_ratio_ceiling": None, "control_egress_reserve_bytes": 131072,
            "control_admitted_fan_in": 64, "expected_flow_count": sum(offered.values())}


def nearest_rank(values, quantile):
    if not values or not 0 < quantile <= 1:
        raise ValueError("nearest-rank quantile requires values and a quantile in (0,1]")
    return sorted(values)[math.ceil(quantile * len(values)) - 1]


def phase(flows):
    return max(f.completion_time_ps for f in flows) - min(f.start_time_ps for f in flows)


def receiver_prefix_findings(flows, rate, propagation=4_000_000):
    receivers = defaultdict(list)
    for flow in flows:
        receivers[flow.destination].append(flow)
    findings = []
    minimum_ratio = None
    for receiver, values in receivers.items():
        if any(f.start_time_ps != 0 for f in values):
            raise ValueError("receiver prefix floors require aligned zero starts")
        delivered = 0
        for k, flow in enumerate(sorted(values, key=lambda f: (f.completion_time_ps, f.flow_id)), 1):
            delivered += flow.payload_bytes
            floor = delivered * 8000 // rate + propagation
            ratio = flow.completion_time_ps / floor
            minimum_ratio = ratio if minimum_ratio is None else min(minimum_ratio, ratio)
            if flow.completion_time_ps < floor:
                findings.append(f"receiver {receiver} prefix {k} beats byte floor")
    return findings, minimum_ratio


def analyze(cell, output, stdout, returncode, mode, arm, bounds):
    manifest = [line for line in stdout.splitlines() if line.startswith("[RNIC manifest]")]
    record = parse_control_recovery_manifest(manifest)
    expected_failure = cell.formerly_failed and (mode == "none" or arm == "pin")
    control_loss = returncode == 2 and "fabric dropped control lifecycle" in stdout
    row = {"cell": cell.name, "source": cell.source, "pattern": cell.pattern,
           "width": cell.width, "rate_gbps": cell.rate, "mode": mode, "arm": arm,
           "formerly_failed": cell.formerly_failed, "returncode": returncode,
           "status": "expected-control-loss" if expected_failure and control_loss else "complete",
           "bounds": bounds, "fatal_findings": [], "control_recovery": record,
           "drop_lines": [line for line in manifest if "ns_tm3_ingress_drops" in line],
           "phase_makespan_ps": None, "fct_p50_ps": None, "fct_p99_ps": None,
           "cn_nn_phase_ratio": None, "flow_count": 0, "completion_sha256": None,
           "compatibility_oracle": None, "job_completion_ps": None}
    fatal = row["fatal_findings"]
    if (arm == "candidate" and mode == "none" and
            (record.get("recovery") != "none" or record.get("headroom_admissions") != 0)):
        fatal.append("off mode changed control admission policy")
    if expected_failure:
        if not control_loss:
            fatal.append("expected control-loss identity exit missing")
        return row
    if returncode != 0 or "physical_quiescence=verified" not in stdout:
        row["status"] = "failed"
        fatal.append("enabled or previously completed cell did not quiesce")
        row["failure_kind"] = ("wall-clock-timeout" if returncode == 124 else
                               "control-loss" if control_loss else
                               "data-retry-exhaustion" if "deterministic retransmission exhausted" in stdout
                               else "backend-exit")
        row["failure_line"] = next((line for line in stdout.splitlines()
                                    if line.startswith("htsim_rnic:") and "Usage:" not in line), "")
        return row
    csv_path = output / "completion.csv"
    if not csv_path.is_file():
        fatal.append("successful exit has no completion CSV")
        return row
    flows = parse_completion_csv(csv_path)
    expected = messages(cell.goal)
    actual = Counter((f.source, f.destination, f.tag, f.payload_bytes) for f in flows)
    if actual != expected or len({f.flow_id for f in flows}) != len(flows):
        fatal.append("flow identity loss or duplication")
    if any(f.start_time_ps < 0 or f.fct_ps <= 0 or
           f.completion_time_ps - f.start_time_ps != f.fct_ps for f in flows):
        fatal.append("inconsistent timestamps")
    if not flows:
        return row
    makespan = phase(flows)
    if makespan < bounds["phase_floor_ps"]:
        fatal.append("phase beats physical byte or causal floor")
    for f in flows:
        prop = 2_000_000 if f.source // 8 == f.destination // 8 else 4_000_000
        if f.fct_ps < f.payload_bytes * 8000 // cell.rate + prop:
            fatal.append("flow beats serialization plus propagation floor")
            break
    ideal = parse_completion_csv(cell.ideal / "completion.csv")
    if Counter((f.source, f.destination, f.tag, f.payload_bytes) for f in ideal) != expected:
        fatal.append("ideal reference has different flow identities")
    ratio = makespan / phase(ideal)
    if cell.main:
        if ratio < 1:
            fatal.append("physical phase beats ideal phase")
        findings, minimum = receiver_prefix_findings(flows, cell.rate)
        fatal.extend(findings)
        row["minimum_receiver_prefix_ratio"] = minimum
    if arm != "pin":
        if record.get("recovery") != mode:
            fatal.append("native recovery selection disagrees with typed config")
        count = record.get("headroom_admissions")
        if count != sum(record.get(f"headroom_{kind}", 0) for kind in KINDS):
            fatal.append("per-kind headroom counts disagree with total")
        if record.get("headroom_remaining_bytes") != 0:
            fatal.append("reserved storage remains at quiescence")
        if record.get("headroom_peak_egress_bytes", 0) > (131072 if mode == "headroom" else 0):
            fatal.append("reserve exceeds configured storage")
        if mode == "none" and count != 0:
            fatal.append("off mode used control reserve")
    if not cell.formerly_failed:
        expected_bytes = (cell.reference / "completion.csv").read_bytes()
        identical = csv_path.read_bytes() == expected_bytes
        row["compatibility_oracle"] = {"byte_identical": identical,
                                       "reference_sha256": hashlib.sha256(expected_bytes).hexdigest()}
        if not identical:
            fatal.append("completion CSV differs from fresh reference bytes")
    row.update(phase_makespan_ps=makespan, fct_p50_ps=nearest_rank([f.fct_ps for f in flows], .5),
               fct_p99_ps=nearest_rank([f.fct_ps for f in flows], .99),
               cn_nn_phase_ratio=ratio, ideal_phase_ps=phase(ideal), flow_count=len(flows),
               completion_sha256=digest(csv_path),
               job_completion_ps=_parse_goal_completion_time_ps(stdout))
    return row


def run_cell(cell, mode, arm, binary, output_root, provenance, resume=False, timeout_s=3600,
             prior_candidate=None):
    output = output_root / cell.name / f"{arm}-{mode}"
    output.mkdir(parents=True, exist_ok=True)
    lock = {"expectations_sha256": provenance["expectations_sha256"],
            "binary_sha256": digest(binary), "arm": arm, "mode": mode,
            "goal_sha256": digest(cell.reference / f"{cell.goal_stem}.bin"),
            "goal_text_sha256": text_digests(cell.goal), "topology_sha256": text_digests(cell.topology),
            "reference_completion_sha256": digest(cell.reference / "completion.csv")
            if not cell.formerly_failed else None,
            "ideal_completion_sha256": digest(cell.ideal / "completion.csv"),
            "bounds": pre_run_bounds(cell), "seed": cell.seed, "rate_gbps": cell.rate}
    lock_path = output / "inputs.json"
    execution = None
    if lock_path.exists():
        if not resume:
            raise FileExistsError(f"existing cell requires --resume or a fresh --out: {output}")
        if not compatible_locks(json.loads(lock_path.read_text()), lock):
            raise ValueError(f"resume input mismatch: {cell.name}")
        if (output / "execution.json").is_file():
            execution = json.loads((output / "execution.json").read_text())
        if execution is None or (execution["returncode"] == 124 and
                                 execution.get("timeout_s", 0) < timeout_s):
            archive = output / f"incomplete-attempt-{len(list(output.glob('incomplete-attempt-*'))) + 1}"
            archive.mkdir()
            for path in output.iterdir():
                if path.is_file():
                    path.rename(archive / path.name)
            execution = None
    if execution is None:
        write_json(lock_path, lock)
        shutil.copyfile(cell.reference / f"{cell.goal_stem}.bin", output / "input.bin")
        (output / "input.goal").write_bytes(cell.goal.read_bytes().replace(b"\r\n", b"\n"))
        (output / "clos.topo").write_bytes(cell.topology.read_bytes().replace(b"\r\n", b"\n"))
        cfg = HtsimRnicConfig(output / "input.bin", "rnic-cn", cell.rate * 10**9,
                              completion_csv=output / "completion.csv", topology=output / "clos.topo",
                              extra_flags={"-rnic_cn_prbs_seed": cell.seed}, control_recovery=mode)
        command = build_htsim_rnic_command(binary, cfg)
        try:
            result = run_owned_process(command, timeout_s=timeout_s)
            stdout = result.stdout + "\n" + result.stderr
            returncode = result.returncode
        except subprocess.TimeoutExpired as error:
            def decoded(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
            stdout = decoded(error.output) + "\n" + decoded(error.stderr)
            stdout += f"\nStudy wall-clock timeout after {timeout_s} seconds\n"
            returncode = 124
        (output / "run.log").write_bytes(stdout.encode())
        execution = {"returncode": returncode, "command": command, "timeout_s": timeout_s,
                     "script_sha256": provenance["script_sha256"], "htsim_commit": provenance["htsim_commit"]}
        write_json(output / "execution.json", execution)
    stdout = (output / "run.log").read_text()
    row = analyze(cell, output, stdout, execution["returncode"], mode, arm, lock["bounds"])
    row["inputs"] = lock
    row["execution_script_sha256"] = execution["script_sha256"]
    row["interrupted_prior_attempts"] = len(list(output.glob("incomplete-attempt-*")))
    if prior_candidate is not None and arm == "candidate":
        prior = prior_candidate / cell.name / f"candidate-{mode}"
        if (prior / "cell.json").is_file():
            old = json.loads((prior / "cell.json").read_text())
            if old["phase_makespan_ps"] is not None:
                identical = (output / "completion.csv").is_file() and (
                    (output / "completion.csv").read_bytes() == (prior / "completion.csv").read_bytes())
                row["timeout_index_oracle"] = {"byte_identical": identical,
                                               "prior_sha256": digest(prior / "completion.csv")}
                if not identical:
                    row["fatal_findings"].append("timeout index changed prior candidate completion bytes")
            elif old["returncode"] == 2:
                same = (old["returncode"], old["status"], old.get("failure_kind"),
                        old.get("failure_line")) == (
                            row["returncode"], row["status"], row.get("failure_kind"),
                            row.get("failure_line"))
                row["timeout_index_failure_identity"] = same
                if not same:
                    row["fatal_findings"].append("timeout index changed prior candidate failure class")
    write_json(output / "cell.json", row)
    print(f"{cell.name} {arm}/{mode}: {row['status']} {row['fatal_findings']}", flush=True)
    return row


def summarize(rows, provenance):
    fatal = [{"cell": r["cell"], "arm": r["arm"], "mode": r["mode"], "finding": finding}
             for r in rows for finding in r["fatal_findings"]]
    active = {(r["width"], r["rate_gbps"]): r for r in rows
              if r["source"] == "collective" and r["pattern"] == "all-to-all"
              and r["arm"] == "candidate" and r["mode"] == "headroom"
              and r["phase_makespan_ps"] is not None}
    behavior = []
    for rate in (400, 200):
        for small, large in pairwise((8, 16, 32, 64)):
            if (small, rate) in active and (large, rate) in active:
                ratio = active[(large, rate)]["phase_makespan_ps"] / active[(small, rate)]["phase_makespan_ps"]
                behavior.append({"family": "width_growth", "rate_gbps": rate, "width": large,
                                 "value": ratio, "within_band": ratio > 1})
        if (64, rate) in active:
            ratio = active[(64, rate)]["cn_nn_phase_ratio"]
            behavior.append({"family": "width64_phase_ratio", "rate_gbps": rate,
                             "value": ratio, "within_band": 1.5 <= ratio <= 3})
    for width in (8, 16, 32, 64):
        if (width, 200) in active and (width, 400) in active:
            ratio = active[(width, 200)]["phase_makespan_ps"] / active[(width, 400)]["phase_makespan_ps"]
            behavior.append({"family": "rate_scaling", "width": width,
                             "value": ratio, "within_band": 1.6 <= ratio <= 2.2})
    return {"schema": "control-recovery-v1", "verdict": "void" if fatal else "valid",
            "provenance": provenance, "run_configurations": len(rows),
            "completed_configurations": sum(r["phase_makespan_ps"] is not None for r in rows),
            "expected_control_loss_configurations": sum(r["status"] == "expected-control-loss" for r in rows),
            "compatibility_oracles": sum(r["compatibility_oracle"] is not None for r in rows),
            "fatal_findings": fatal, "behavioral_relations": behavior, "cells": rows}


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    main = [r for r in result["cells"] if r["arm"] == "candidate" and r["source"] == "collective"
            and r["pattern"] == "all-to-all"]
    for rate, color in ((200, "tab:orange"), (400, "tab:blue")):
        for mode, marker, linestyle in (("headroom", "o", "-"), ("none", "x", ":")):
            rows = sorted((r for r in main if r["rate_gbps"] == rate and r["mode"] == mode),
                          key=lambda r: r["width"])
            widths = [r["width"] for r in rows]
            for ax, field, scale in ((axes[0], "phase_makespan_ps", 1e6),
                                      (axes[1], "cn_nn_phase_ratio", 1)):
                values = [float("nan") if r[field] is None else r[field] / scale for r in rows]
                ax.plot(widths, values, marker=marker, linestyle=linestyle, color=color,
                        label=f"{rate} Gbit/s, {mode}")
            if mode == "headroom":
                for row in rows:
                    if row["phase_makespan_ps"] is not None:
                        count = row["control_recovery"].get("headroom_admissions", 0)
                        axes[0].annotate(str(count), (row["width"], row["phase_makespan_ps"] / 1e6),
                                         xytext=(0, -16 if row["width"] == 64 and rate == 200 else 6),
                                         textcoords="offset points", ha="center", fontsize=8)
    ideal = sorted(((r["width"], r["phase_makespan_ps"] / r["cn_nn_phase_ratio"] / 1e6)
                    for r in main if r["mode"] == "headroom" and r["rate_gbps"] == 400
                    and r["phase_makespan_ps"] is not None and r["cn_nn_phase_ratio"]),
                   key=lambda item: item[0])
    axes[0].plot([w for w, _ in ideal], [v for _, v in ideal], color="gray", linestyle="--",
                 linewidth=.9, label="ideal rnic-nn, 400 Gbit/s")
    axes[0].axhline(50_000, color="black", linewidth=.8, linestyle="-.")
    axes[0].annotate("50 ms data retransmission timeout:\nthe width-64 phase ends one RTO\n"
                     "after a tail loss, not on fabric time",
                     (16, 50_000), xytext=(0, -34), textcoords="offset points", fontsize=7.5,
                     ha="left", va="top")
    for row in main:
        if row["mode"] == "headroom" and row["width"] == 64 and row["cn_nn_phase_ratio"]:
            axes[1].annotate(f"{row['cn_nn_phase_ratio']:.0f}x",
                             (64, row["cn_nn_phase_ratio"]),
                             xytext=(-8, 0), textcoords="offset points", ha="right",
                             va="center", fontsize=8)
    axes[0].set_ylabel("Physical phase makespan (µs)")
    axes[1].set_ylabel("Physical / ideal phase makespan")
    axes[1].axhline(1, color="gray", linewidth=.8, label="Physical phase floor")
    axes[0].legend(fontsize=7.5, loc="center left", bbox_to_anchor=(0.02, 0.5))
    for ax in axes:
        ax.set_yscale("log")
        ax.plot([64], [.03], marker="x", color="black", transform=ax.get_xaxis_transform())
    if result["verdict"] == "void":
        fig.suptitle("All-to-all on the physical Clos with control headroom: "
                     "void study, diagnostic values", fontsize=11)
    for ax in axes:
        ax.set_xlabel("All-to-all width (ranks)")
        ax.set_xticks([8, 16, 32, 64])
        ax.margins(x=.08, y=.18)
    fig.supxlabel("Numbers: headroom admissions per cell (0 below width 64). "
                  "Bottom crosses: width-64 with recovery none exits on control loss (no valid latency).",
                  fontsize=8)
    handles, labels = axes[1].get_legend_handles_labels()
    order = [2, 3, 0, 1, 4]
    axes[1].legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left", fontsize=8)
    fig.savefig(output / "control_recovery.png", dpi=160)
    fig.savefig(output / "control_recovery.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collective-reference", type=Path, default=os.getenv("SIMLLM_CONTROL_COLLECTIVE_REFERENCE"))
    parser.add_argument("--pipeline-reference", type=Path, default=os.getenv("SIMLLM_CONTROL_PIPELINE_REFERENCE"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--prior-candidate", type=Path,
                        default=os.getenv("SIMLLM_CONTROL_PRIOR_CANDIDATE"))
    args = parser.parse_args()
    if args.workers < 1 or args.timeout_s < 1:
        parser.error("workers and timeout must be positive")
    if args.out is None:
        if not os.getenv("SIMLLM_DATA_ROOT"):
            parser.error("configure SIMLLM_DATA_ROOT or --out")
        args.out = Path(os.environ["SIMLLM_DATA_ROOT"]) / "control_recovery_v1"
    if args.out.resolve().is_relative_to(REPO):
        parser.error("--out must be outside the repository")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        result = json.loads((args.out / "results.json").read_text())
    else:
        if args.collective_reference is None or args.pipeline_reference is None:
            parser.error("configure both reference roots with options or SIMLLM_CONTROL_*_REFERENCE")
        binaries = [os.getenv("SIMLLM_HTSIM_RNIC"), os.getenv("SIMLLM_HTSIM_RNIC_BASE")]
        source = os.getenv("SIMLLM_HTSIM_SOURCE")
        if not all(binaries) or not source:
            parser.error("configure SIMLLM_HTSIM_RNIC, SIMLLM_HTSIM_RNIC_BASE and SIMLLM_HTSIM_SOURCE")
        cells = discover(args.collective_reference, args.pipeline_reference)
        provenance = {"expectations_commit": git("rev-parse", FREEZE),
                      "expectations_sha256": text_digests(HERE / "expectations.md"),
                      "htsim_expectations_commit": git("rev-parse", HTSIM_FREEZE, cwd=source),
                      "htsim_commit": git("rev-parse", "HEAD", cwd=source),
                      "htsim_timeout_index_expectations_commit":
                          git("rev-parse", HTSIM_INDEX_FREEZE, cwd=source),
                      "htsim_pin": "617ce203bb8e4d60343be6b7c5f1bd0eff4f053f",
                      "binary_sha256": digest(binaries[0]), "pin_binary_sha256": digest(binaries[1]),
                      "script_sha256": digest(__file__),
                      "collective_reference_name": args.collective_reference.name,
                      "pipeline_reference_name": args.pipeline_reference.name}
        if git("status", "--porcelain", cwd=source):
            parser.error("backend source must be committed before recording binary provenance")
        # Freeze all bounds before executing or reading any completion rows.
        write_json(args.out / "pre_run_bounds.json", {c.name: pre_run_bounds(c) for c in cells})
        write_json(args.out / "provenance.json", provenance)
        snapshots = args.out / "runner-snapshots"
        snapshots.mkdir(exist_ok=True)
        (snapshots / f"{provenance['script_sha256']}.py").write_bytes(Path(__file__).read_bytes())
        jobs = [(c, mode, "candidate", Path(binaries[0])) for c in cells for mode in ("none", "headroom")]
        jobs += [(c, "none", "pin", Path(binaries[1])) for c in cells
                 if c.main or (c.source == "pipeline" and c.formerly_failed)]
        # New recovery cells run first so their loss evidence is available while
        # the independent compatibility replays finish. Every job is retained.
        jobs.sort(key=lambda job: (not job[0].formerly_failed, job[0].name, job[2], job[1]))
        prepare_htsim_child_lifetime()
        def execute(job):
            c, mode, arm, binary = job
            return run_cell(c, mode, arm, binary, args.out, provenance, args.resume, args.timeout_s,
                            args.prior_candidate)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(execute, jobs))
        result = summarize(rows, provenance)
        write_json(args.out / "results.json", result)
    result["provenance"]["report_script_sha256"] = digest(__file__)
    write_json(args.out / "results.json", result)
    plot(result, args.out / "figures")
    if args.publish:
        write_json(HERE / "results.json", result)
        (HERE / "figures").mkdir(exist_ok=True)
        for suffix in ("png", "pdf"):
            shutil.copyfile(args.out / "figures" / f"control_recovery.{suffix}",
                            HERE / "figures" / f"control_recovery.{suffix}")
    print(f"verdict={result['verdict']} configurations={result['run_configurations']} "
          f"fatal_findings={len(result['fatal_findings'])}")
    return 2 if result["fatal_findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
