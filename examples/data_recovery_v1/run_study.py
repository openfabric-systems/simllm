from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from examples.control_recovery_v1.run_study import (
    KINDS,
    Cell,
    compatible_locks,
    digest,
    discover,
    git,
    messages,
    nearest_rank,
    phase,
    receiver_prefix_findings,
    text_digests,
    write_json,
)
from simllm.backends._child_process import run_owned_process
from simllm.backends.fct import earliest_completion_byte_floors, normalized_fct
from simllm.backends.htsim_rnic import (
    HtsimRnicConfig,
    _parse_goal_completion_time_ps,
    build_htsim_rnic_command,
    parse_completion_csv,
    parse_control_recovery_manifest,
    parse_data_recovery_manifest,
    prepare_htsim_child_lifetime,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FREEZE = "7904e8f09a4174f7320a0107ac676660f3dfcf02"
BACKOFF_FREEZE = "fc8614599e51a05d23535f566382368639f52043"
PROBE_POLICIES = ("constant", "exponential")
ARMS = ("disabled", "recovery-only", "window-only", "combined")
BUFFER_BYTES = 1 << 20
MAX_WIRE_BYTES = 4160
CONTROL_DEADLINE_PS = 10_000_000
PROBE_WINDOWS = 4
MAXIMUM_RETRANSMISSIONS = 8


def probe_schedule(probe_policy="constant"):
    if probe_policy not in PROBE_POLICIES:
        raise ValueError(f"unknown probe policy: {probe_policy}")
    base = PROBE_WINDOWS * CONTROL_DEADLINE_PS
    intervals = [base * (2**index if probe_policy == "exponential" else 1)
                 for index in range(MAXIMUM_RETRANSMISSIONS - 1)]
    return {"probe_policy": probe_policy, "probe_base_interval_ps": base,
            "probe_interval_rule": "P*2^(attempt-1)" if probe_policy == "exponential" else "P",
            "probe_intervals_ps": intervals,
            "cumulative_probe_timer_allowance_ps": sum(intervals),
            "terminal_retry": "legacy-timeout"}


def expectation_contract(probe_policy="constant"):
    probe_schedule(probe_policy)
    return ((BACKOFF_FREEZE, HERE / "expectations_backoff.md") if probe_policy == "exponential"
            else (FREEZE, HERE / "expectations.md"))


def selections(arm, bounds, probe_policy="constant"):
    probe_schedule(probe_policy)
    if bounds.get("probe_policy", probe_policy) != probe_policy:
        raise ValueError("probe policy disagrees with declared bounds")
    if arm not in (*ARMS, "legacy-none"):
        raise ValueError(f"unknown arm: {arm}")
    bounded = arm in ("window-only", "combined")
    return {
        "control_recovery": "none" if arm == "legacy-none" else "headroom",
        "data_recovery": ("exponential" if probe_policy == "exponential" else "deadline")
        if arm in ("recovery-only", "combined") else "none",
        "retry_probe_windows": PROBE_WINDOWS,
        "initial_window_bytes": bounds["initial_window_bytes"] if bounded else None,
        "initial_window_fan_in": bounds["declared_fan_in"] if bounded else None,
    }


def planned_jobs(cells):
    jobs = [(cell, "legacy-none") for cell in cells]
    jobs += [(cell, arm) for cell in cells if cell.main for arm in ARMS]
    jobs += [(cell, arm) for cell in cells if cell.source == "pipeline" and cell.formerly_failed
             for arm in ("recovery-only", "combined")]
    return sorted(jobs, key=lambda job: (job[1] != "combined", not job[0].formerly_failed,
                                         job[0].name, job[1]))


def topology_inputs(path):
    values = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        parts = line.split("#", 1)[0].split()
        if len(parts) == 2:
            values[parts[0]].append(int(parts[1]))
    if values["Nodes"] != [64] or values["Tiers"] != [2] or values["Radix_Down"] != [8, 8]:
        raise ValueError("frozen references require 64 hosts on an eight-host-leaf two-tier Clos")
    rates = values["Downlink_speed_Gbps"]
    delays = values["Downlink_Latency_ns"]
    if len(rates) != 2 or min(rates) <= 0 or delays != [1000, 1000]:
        raise ValueError("unexpected reference link rate or propagation")
    return {"hosts_per_leaf": 8, "slowest_data_link_gbps": min(rates),
            "hop_propagation_ps": 1_000_000}


def pre_run_bounds(cell: Cell, probe_policy="constant"):
    schedule = probe_schedule(probe_policy)
    topology = topology_inputs(cell.topology)
    offered = messages(cell.goal)
    receiver_bytes, leaf_flows = Counter(), Counter()
    for (_, destination, _, payload), count in offered.items():
        receiver_bytes[destination] += payload * count
        leaf_flows[destination // topology["hosts_per_leaf"]] += count
    fan_in = max(leaf_flows.values())
    receiver_service = max(receiver_bytes.values()) * 8000 // cell.rate
    minimum_propagation = 4_000_000 if cell.source == "collective" else 2_000_000
    phase_floor = receiver_service + minimum_propagation
    service = receiver_service
    saved = {}
    if cell.pattern == "ring":
        phase_floor = 2 * (cell.width - 1) * (
            1_048_576 // cell.width * 8000 // cell.rate + 4_000_000)
    if cell.source == "pipeline":
        saved = json.loads((cell.reference / "pre_run_bounds.json").read_text())
        phase_floor = max(phase_floor, saved["ep_phase_floor_ps"])
        service = max(service, saved["ep_effective_serialization_floor_ps"],
                      saved["pp_step_data_floor_ps"])
    queue_allowance = (
        3 * BUFFER_BYTES * 8000 // topology["slowest_data_link_gbps"]
        + 8 * topology["hop_propagation_ps"] + MAX_WIRE_BYTES * 8000 // cell.rate)
    budget = 9 * service + 8 * (PROBE_WINDOWS * CONTROL_DEADLINE_PS + queue_allowance)
    return {
        **topology, **schedule, "expected_flow_count": sum(offered.values()),
        "receiver_bytes": {str(key): value for key, value in sorted(receiver_bytes.items())},
        "leaf_input_send_counts": {str(key): value for key, value in sorted(leaf_flows.items())},
        "declared_fan_in": fan_in, "initial_window_bytes": BUFFER_BYTES // fan_in,
        "initial_sizing_assumption": "all input sends targeting one leaf, regardless of overlap",
        "shared_buffer_bytes": BUFFER_BYTES, "max_wire_packet_bytes": MAX_WIRE_BYTES,
        "control_deadline_ps": CONTROL_DEADLINE_PS, "retry_probe_windows": PROBE_WINDOWS,
        "maximum_retransmissions": MAXIMUM_RETRANSMISSIONS,
        "receiver_serialization_floor_ps": receiver_service,
        "phase_floor_ps": phase_floor, "phase_ceiling_ps": None,
        "flow_ceiling_ps": None, "minimum_path_propagation_ps": minimum_propagation,
        "saved_pipeline_bounds": saved, "budget_service_ps": service,
        "budget_queue_allowance_ps": queue_allowance, "engineering_budget_ps": budget,
        "width64_engineering_budget_ps": 1_000_000_000 if cell.main and cell.width == 64 else None,
    }


def flow_identities(flows):
    return Counter((f.source, f.destination, f.tag, f.payload_bytes) for f in flows)


def ideal_reference(cell):
    flows = parse_completion_csv(cell.ideal / "completion.csv")
    if flow_identities(flows) != messages(cell.goal):
        raise ValueError(f"ideal reference identity mismatch: {cell.name}")
    if len({f.flow_id for f in flows}) != len(flows) or any(
            f.start_time_ps < 0 or f.fct_ps <= 0
            or f.completion_time_ps - f.start_time_ps != f.fct_ps for f in flows):
        raise ValueError(f"invalid ideal reference timestamps or duplicate identities: {cell.name}")
    return phase(flows)


def failure_kind(stdout, returncode):
    if returncode == 124:
        return "wall-clock-timeout"
    if "fabric dropped control lifecycle" in stdout:
        return "control-loss"
    if "deterministic retransmission exhausted" in stdout:
        return "data-retry-exhaustion"
    return "backend-exit"


def is_diagnostic_failure(cell, arm, kind):
    eligible = arm == "window-only" or (cell.formerly_failed and arm in ("legacy-none", "disabled"))
    return eligible and kind in ("wall-clock-timeout", "control-loss", "data-retry-exhaustion")


def observe(cell, output, stdout, returncode, arm, bounds):
    manifest = [line for line in stdout.splitlines() if line.startswith("[RNIC manifest]")]
    row = {
        "cell": cell.name, "source": cell.source, "pattern": cell.pattern,
        "probe_policy": bounds.get("probe_policy", "constant"),
        "width": cell.width, "rate_gbps": cell.rate, "arm": arm,
        "formerly_failed": cell.formerly_failed, "returncode": returncode,
        "status": "failed", "bounds": bounds, "fatal_findings": [],
        "manifest": manifest, "flow_count": 0, "raw_completion_row_count": 0,
        "phase_makespan_ps": None, "job_completion_ps": None, "fct_p50_ps": None,
        "fct_p99_ps": None, "cn_nn_phase_ratio": None,
        "completion_sha256": None, "compatibility_oracle": None,
        "behavioral_relations": [],
    }
    if returncode != 0 or not any("physical_quiescence=verified" in line for line in manifest):
        kind = failure_kind(stdout, returncode)
        row["failure_kind"] = kind
        row["failure_line"] = next((line for line in stdout.splitlines()
                                    if line.startswith("htsim_rnic:") and "Usage:" not in line), "")
        if is_diagnostic_failure(cell, arm, kind):
            row["status"] = "diagnostic-failure"
        else:
            row["fatal_findings"].append("required cell failed to complete and quiesce")
        return row
    csv_path = output / "completion.csv"
    try:
        flows = parse_completion_csv(csv_path)
        row["completion_sha256"] = digest(csv_path)
    except (OSError, ValueError, KeyError) as error:
        row["fatal_findings"].append(f"unreadable completion CSV: {error}")
        return row
    row["raw_completion_row_count"] = len(flows)
    if flow_identities(flows) != messages(cell.goal) or len({f.flow_id for f in flows}) != len(flows):
        row["fatal_findings"].append("flow identity loss or duplication")
        return row
    if any(f.profile != "rnic-cn" or f.start_time_ps < 0 or f.fct_ps <= 0
           or f.completion_time_ps - f.start_time_ps != f.fct_ps for f in flows):
        row["fatal_findings"].append("invalid profile or causal timestamps")
        return row
    makespan = phase(flows)
    row.update(status="complete", flow_count=len(flows), phase_makespan_ps=makespan,
               job_completion_ps=_parse_goal_completion_time_ps(stdout),
               fct_p50_ps=nearest_rank([f.fct_ps for f in flows], .5),
               fct_p99_ps=nearest_rank([f.fct_ps for f in flows], .99),
               cn_nn_phase_ratio=makespan / bounds["ideal_phase_ps"])
    if arm == "combined":
        row["behavioral_relations"].append({
            "family": "combined_engineering_budget", "value_ps": makespan,
            "upper_bound_ps": bounds["engineering_budget_ps"],
            "within_band": makespan < bounds["engineering_budget_ps"],
        })
        if cell.main and cell.width == 64:
            row["behavioral_relations"].append({
                "family": "width64_submillisecond", "value_ps": makespan,
                "upper_bound_ps": 1_000_000_000, "within_band": makespan < 1_000_000_000,
            })
    return row


def manifest_findings(row):
    findings = []
    try:
        control = parse_control_recovery_manifest(row["manifest"])
        data = parse_data_recovery_manifest(row["manifest"])
    except (ValueError, TypeError) as error:
        return [f"invalid native manifest: {error}"]
    row["control_recovery"], row["data_recovery"] = control, data
    expected = selections(row["arm"], row["bounds"], row.get("probe_policy", "constant"))
    if control.get("recovery") != expected["control_recovery"]:
        findings.append("control recovery manifest disagrees with typed selection")
    bounded = expected["initial_window_bytes"] is not None
    wanted = {
        "data_recovery": expected["data_recovery"], "retry_probe_windows": PROBE_WINDOWS,
        "initial_window": "bounded" if bounded else "none",
        "initial_window_bytes": expected["initial_window_bytes"] if bounded else 0,
        "initial_window_fan_in": expected["initial_window_fan_in"] if bounded else 0,
        "initial_buffer_bytes": BUFFER_BYTES, "initial_sizing": "F-times-U-at-most-B",
        "probe_epoch": "physical-retry-serialization-end", "recovery_release": "actual-arrival-tick",
        "probe_backoff": {"none": "none", "deadline": "constant", "exponential": "exponential"}[
            expected["data_recovery"]], "terminal_retry": "legacy-timeout",
    }
    for key, value in wanted.items():
        if data.get(key) != value:
            findings.append(f"data recovery manifest disagrees with config: {key}")
    if row["status"] != "complete":
        return findings
    for field in ("tail_probes", "tail_probe_wire_bytes", "late_retry_admissions",
                  "initial_window_holds", "initial_grants_dispatched",
                  "deterministic_retransmissions", "deterministic_retransmission_wire_bytes"):
        if field not in data:
            findings.append(f"missing final data recovery counter: {field}")
    for name, packets, wire in (
            ("probe", "tail_probes", "tail_probe_wire_bytes"),
            ("retransmission", "deterministic_retransmissions",
             "deterministic_retransmission_wire_bytes")):
        if packets in data and wire in data and not (
                data[packets] <= data[wire] <= data[packets] * MAX_WIRE_BYTES):
            findings.append(f"physical {name} packet and wire-byte counts disagree")
    for probes, retries in (("tail_probes", "deterministic_retransmissions"),
                            ("tail_probe_wire_bytes", "deterministic_retransmission_wire_bytes")):
        if probes in data and retries in data and data[probes] > data[retries]:
            findings.append(f"physical probe counter exceeds all retransmissions: {probes}")
    if expected["data_recovery"] == "none" and any(
            data.get(key, 0) != 0 for key in (
                "tail_probes", "tail_probe_wire_bytes", "late_retry_admissions")):
        findings.append("disabled recovery used probes or late retry admission")
    if not bounded and any(data.get(key, 0) != 0 for key in (
            "initial_window_holds", "initial_grants_dispatched")):
        findings.append("disabled initial window changed pre-grant admission")
    count = control.get("headroom_admissions")
    if count != sum(control.get(f"headroom_{kind}", 0) for kind in KINDS):
        findings.append("control headroom per-kind conservation failed")
    if control.get("headroom_remaining_bytes") != 0:
        findings.append("control reserve remains at quiescence")
    reserve = 131072 if expected["control_recovery"] == "headroom" else 0
    if control.get("headroom_peak_egress_bytes", 0) > reserve:
        findings.append("control reserve exceeded finite capacity")
    if not reserve and count != 0:
        findings.append("disabled control protection used reserved storage")
    raw = dict(token.split("=", 1) for line in row["manifest"] for token in line.split()
               if "=" in token)
    for key, value in (("maximum_retransmissions", "8"), ("control_deadline_ps", "10000000"),
                       ("shared_buffer_bytes", str(BUFFER_BYTES)),
                       ("max_wire_packet_bytes", str(MAX_WIRE_BYTES))):
        if raw.get(key) != value:
            findings.append(f"native physical configuration changed: {key}")
    return findings


def apply_guards(cell, output, row):
    fatal = row["fatal_findings"]
    fatal.extend(manifest_findings(row))
    if row["status"] != "complete":
        return row
    flows = parse_completion_csv(output / "completion.csv")
    bounds = row["bounds"]
    if row["phase_makespan_ps"] < bounds["phase_floor_ps"]:
        fatal.append("phase beats receiver byte or saved cut floor")
    if row["cn_nn_phase_ratio"] < 1:
        fatal.append("physical complete phase beats identical ideal phase")
    for f in flows:
        propagation = 2_000_000 if f.source // 8 == f.destination // 8 else 4_000_000
        if f.fct_ps < f.payload_bytes * 8000 // cell.rate + propagation:
            fatal.append("flow beats serialization plus path propagation floor")
            break
    if cell.main:
        try:
            findings, minimum = receiver_prefix_findings(flows, cell.rate)
            fatal.extend(findings)
            row["minimum_receiver_prefix_ratio"] = minimum
        except ValueError as error:
            fatal.append(str(error))
    if cell.source == "pipeline":
        job_floor = bounds["saved_pipeline_bounds"]["pp_step_data_floor_ps"]
        if row["job_completion_ps"] is None or row["job_completion_ps"] < job_floor:
            fatal.append("pipeline schedule beats saved causal floor or lacks schedule completion")
        for receiver in sorted({f.destination for f in flows}):
            values = [f for f in flows if f.destination == receiver]
            propagation = min(2_000_000 if f.source // 8 == receiver // 8 else 4_000_000
                              for f in values)
            if any(not bound.ok for bound in earliest_completion_byte_floors(
                    values, link_rate_bps=cell.rate * 10**9, propagation_ps=propagation)):
                fatal.append(f"pipeline receiver {receiver} completion prefix beats byte floor")
    if row["arm"] == "legacy-none" and not cell.formerly_failed:
        reference = cell.reference / "completion.csv"
        identical = (output / "completion.csv").read_bytes() == reference.read_bytes()
        row["compatibility_oracle"] = {"family": "legacy_disabled_csv",
                                       "byte_identical": identical,
                                       "reference_sha256": digest(reference)}
        if not identical:
            fatal.append("protected legacy completion CSV changed")
    return row


def write_once(path, value):
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"immutable run record differs: {path}")
    else:
        path.write_bytes(encoded)


def verify_execution(output):
    execution = json.loads((output / "execution.json").read_text())
    for name, key in (("run.log", "run_log_sha256"), ("completion.csv", "completion_sha256")):
        path = output / name
        actual = digest(path) if path.exists() else None
        if actual != execution[key]:
            raise ValueError(f"retained raw output digest changed: {name}")
    return execution


def write_normalization(cell, output):
    flows = parse_completion_csv(output / "completion.csv")
    ideal = parse_completion_csv(cell.ideal / "completion.csv")
    with (output / "per_flow_normalization.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("source", "destination", "tag", "payload_bytes", "fct_ps",
                         "baseline_fct_ps", "normalized_fct", "interpretation"))
        for f in normalized_fct(flows, ideal):
            writer.writerow((f.source, f.destination, f.tag, f.payload_bytes, f.fct_ps,
                             f.baseline_fct_ps, f.slowdown, "diagnostic-shared-or-dynamic-phase"))


def run_cell(cell, arm, binary, root, provenance, bounds, resume=False, timeout_s=3600,
             probe_policy="constant"):
    schedule = probe_schedule(probe_policy)
    freeze, expectations = expectation_contract(probe_policy)
    for field, wanted in (("probe_policy", probe_policy), ("expectations_commit", freeze),
                          ("expectations_file", expectations.name)):
        if provenance.get(field, wanted) != wanted:
            raise ValueError(f"probe policy disagrees with run provenance: {field}")
    selected = selections(arm, bounds, probe_policy)
    output = root / cell.name / arm
    output.mkdir(parents=True, exist_ok=True)
    lock = {
        "expectations_sha256": provenance["expectations_sha256"],
        "expectations_commit": provenance.get("expectations_commit", freeze),
        "expectations_file": provenance.get("expectations_file", expectations.name),
        "binary_sha256": provenance["binary_sha256"], "script_sha256": provenance["script_sha256"],
        "helper_sha256": provenance["helper_sha256"], "wrapper_sha256": provenance["wrapper_sha256"],
        "goal_sha256": digest(cell.reference / f"{cell.goal_stem}.bin"),
        "goal_text_sha256": text_digests(cell.goal), "topology_sha256": text_digests(cell.topology),
        "ideal_completion_sha256": digest(cell.ideal / "completion.csv"),
        "reference_completion_sha256": digest(cell.reference / "completion.csv")
        if not cell.formerly_failed else None,
        "seed": cell.seed, "rate_gbps": cell.rate, "arm": arm, "bounds": bounds,
        "selections": selected, "timeout_s": timeout_s, **schedule,
    }
    lock_path = output / "inputs.json"
    if lock_path.exists():
        if not resume:
            raise FileExistsError(f"existing cell requires --resume: {output}")
        if not compatible_locks(json.loads(lock_path.read_text()), json.loads(json.dumps(lock))):
            raise ValueError(f"resume input mismatch: {cell.name}/{arm}")
        if (output / "cell.json").exists():
            verify_execution(output)
            return json.loads((output / "cell.json").read_text())
    else:
        write_json(lock_path, lock)
    execution_path = output / "execution.json"
    if not execution_path.exists():
        if (output / "command.json").exists():
            raise ValueError(f"interrupted execution retained; use a fresh --out: {output}")
        shutil.copyfile(cell.reference / f"{cell.goal_stem}.bin", output / "input.bin")
        (output / "input.goal").write_bytes(cell.goal.read_bytes().replace(b"\r\n", b"\n"))
        (output / "clos.topo").write_bytes(cell.topology.read_bytes().replace(b"\r\n", b"\n"))
        cfg = HtsimRnicConfig(output / "input.bin", "rnic-cn", cell.rate * 10**9,
                              completion_csv=output / "completion.csv", topology=output / "clos.topo",
                              extra_flags={"-rnic_cn_prbs_seed": cell.seed}, **lock["selections"])
        command = build_htsim_rnic_command(binary, cfg)
        write_json(output / "command.json", {"argv": command})
        try:
            result = run_owned_process(command, timeout_s=timeout_s)
            stdout, returncode = result.stdout + "\n" + result.stderr, result.returncode
        except subprocess.TimeoutExpired as error:
            def decoded(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
            stdout = decoded(error.output) + "\n" + decoded(error.stderr)
            stdout += f"\nStudy wall-clock timeout after {timeout_s} seconds\n"
            returncode = 124
        (output / "run.log").write_bytes(stdout.encode())
        write_json(execution_path, {"returncode": returncode,
                                   "run_log_sha256": digest(output / "run.log"),
                                   "completion_sha256": digest(output / "completion.csv")
                                   if (output / "completion.csv").exists() else None})
    execution = verify_execution(output)
    stdout = (output / "run.log").read_text()
    row = observe(cell, output, stdout, execution["returncode"], arm, bounds)
    row["inputs"] = lock
    write_once(output / "observations.json", row)
    if row["status"] == "complete":
        write_normalization(cell, output)
    row = apply_guards(cell, output, row)
    write_once(output / "cell.json", row)
    print(f"{cell.name} {arm}: {row['status']} phase_ps={row['phase_makespan_ps']} "
          f"fatal_findings={row['fatal_findings']}", flush=True)
    return row


def summarize(rows, provenance, expected_jobs=None):
    behavior = [{"cell": row["cell"], "arm": row["arm"], **relation}
                for row in rows for relation in row["behavioral_relations"]]
    indexed = {(row["cell"], row["arm"]): row for row in rows}
    contrasts = []
    for row in rows:
        baseline = indexed.get((row["cell"], "disabled"))
        if row["arm"] != "disabled" and baseline and row["phase_makespan_ps"] is not None:
            contrasts.append({"family": "arm_contrast", "cell": row["cell"], "arm": row["arm"],
                              "phase_ratio_to_disabled": row["phase_makespan_ps"] /
                              baseline["phase_makespan_ps"]
                              if baseline["phase_makespan_ps"] is not None else None})
    combined = {(r["width"], r["rate_gbps"]): r for r in rows
                if r["arm"] == "combined" and r["source"] == "collective"}
    for width in (8, 16, 32, 64):
        slow, fast = combined.get((width, 200)), combined.get((width, 400))
        if slow and fast and all(r["phase_makespan_ps"] is not None for r in (slow, fast)):
            contrasts.append({"family": "rate_contrast", "width": width,
                              "phase_ratio_200_over_400": slow["phase_makespan_ps"] /
                              fast["phase_makespan_ps"], "expected_serialization_ratio": 2,
                              **probe_schedule(slow.get("probe_policy", "constant"))})
    fatal = [{"cell": row["cell"], "arm": row["arm"], "finding": finding}
             for row in rows for finding in row["fatal_findings"]]
    if expected_jobs is not None and (set(indexed) != set(expected_jobs) or len(indexed) != len(rows)):
        fatal.append({"cell": None, "arm": None, "finding": "consumer matrix is missing or duplicated"})
    dormant = []
    for rate in (200, 400):
        name = f"collective-collective-all-to-all-w8-{rate}g-rnic-cn"
        off, enabled = indexed.get((name, "disabled")), indexed.get((name, "recovery-only"))
        if off is None or enabled is None:
            if expected_jobs is not None:
                fatal.append({"cell": name, "arm": "recovery-only",
                              "finding": "missing predeclared dormant recovery fixture"})
            continue
        identical = (off["status"] == enabled["status"] == "complete" and
                     off["completion_sha256"] == enabled["completion_sha256"])
        inactive = all(enabled.get("data_recovery", {}).get(key) == 0
                       for key in ("tail_probes", "tail_probe_wire_bytes", "late_retry_admissions"))
        dormant.append({"cell": name, "family": "dormant_recovery_csv",
                        "byte_identical": identical, "mechanism_inactive": inactive})
        if not identical or not inactive:
            fatal.append({"cell": name, "arm": "recovery-only",
                          "finding": "dormant recovery identity failed"})
    findings = [relation for relation in behavior if not relation["within_band"]]
    return {
        "schema": "data-recovery-v1", "verdict": "void" if fatal else "outside-budget"
        if findings else "consumer-valid", "scope": "consumer evidence; native and Python gates required separately",
        "provenance": provenance, "run_configurations": len(rows),
        "completed_configurations": sum(row["status"] == "complete" for row in rows),
        "diagnostic_failure_configurations": sum(row["status"] == "diagnostic-failure" for row in rows),
        "exact_oracles": {"legacy_disabled_csv": [r["compatibility_oracle"] for r in rows
                                                 if r["compatibility_oracle"] is not None],
                          "dormant_recovery_csv": dormant},
        "behavioral_relation_families": sorted({r["family"] for r in behavior}),
        "behavioral_relations": behavior, "behavioral_findings": findings,
        "diagnostic_contrasts": contrasts, "fatal_findings": fatal, "cells": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Replay the frozen bounded DATA recovery consumer matrix")
    parser.add_argument("--collective-reference", type=Path,
                        default=os.getenv("SIMLLM_CONTROL_COLLECTIVE_REFERENCE"))
    parser.add_argument("--pipeline-reference", type=Path,
                        default=os.getenv("SIMLLM_CONTROL_PIPELINE_REFERENCE"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--probe-policy", choices=PROBE_POLICIES, default="constant")
    args = parser.parse_args()
    if args.workers < 1 or args.timeout_s < 1:
        parser.error("workers and timeout must be positive")
    if args.out is None:
        if not os.getenv("SIMLLM_DATA_ROOT"):
            parser.error("configure SIMLLM_DATA_ROOT or --out")
        args.out = Path(os.environ["SIMLLM_DATA_ROOT"]) / (
            "data_recovery_backoff_v1" if args.probe_policy == "exponential" else "data_recovery_v1")
    if args.out.resolve().is_relative_to(REPO):
        parser.error("--out must be outside the repository")
    if args.collective_reference is None or args.pipeline_reference is None:
        parser.error("configure both saved reference roots")
    binary, baseline, source = (os.getenv(name) for name in (
        "SIMLLM_HTSIM_RNIC", "SIMLLM_HTSIM_RNIC_BASE", "SIMLLM_HTSIM_SOURCE"))
    if not all((binary, baseline, source)):
        parser.error("configure SIMLLM_HTSIM_RNIC, SIMLLM_HTSIM_RNIC_BASE and SIMLLM_HTSIM_SOURCE")
    if git("status", "--porcelain", cwd=source):
        parser.error("backend source must be committed before recording implementation provenance")
    cells = discover(args.collective_reference, args.pipeline_reference)
    args.out.mkdir(parents=True, exist_ok=True)
    bounds = {cell.name: pre_run_bounds(cell, args.probe_policy) for cell in cells}
    write_once(args.out / "physical_bounds.json", bounds)
    for cell in cells:
        bounds[cell.name]["ideal_phase_ps"] = ideal_reference(cell)
        bounds[cell.name]["engineering_budget_over_ideal"] = (
            bounds[cell.name]["engineering_budget_ps"] / bounds[cell.name]["ideal_phase_ps"])
    write_once(args.out / "pre_run_bounds.json", bounds)
    freeze, expectations = expectation_contract(args.probe_policy)
    provenance = {
        **probe_schedule(args.probe_policy),
        "expectations_commit": git("rev-parse", freeze),
        "expectations_file": expectations.name,
        "expectations_sha256": text_digests(expectations),
        "base_expectations_commit": FREEZE,
        "base_expectations_sha256": text_digests(HERE / "expectations.md"),
        "terminal_retry_expectations_commit": BACKOFF_FREEZE,
        "simllm_commit": git("rev-parse", "HEAD"), "htsim_commit": git("rev-parse", "HEAD", cwd=source),
        "binary_sha256": digest(binary), "baseline_binary_sha256": digest(baseline),
        "script_sha256": digest(__file__),
        "helper_sha256": digest(REPO / "examples/control_recovery_v1/run_study.py"),
        "wrapper_sha256": digest(REPO / "simllm/backends/htsim_rnic.py"),
        "collective_reference_name": args.collective_reference.name,
        "pipeline_reference_name": args.pipeline_reference.name,
    }
    write_once(args.out / "provenance.json", provenance)
    snapshots = args.out / "runner-snapshots"
    snapshots.mkdir(exist_ok=True)
    for path in (Path(__file__), REPO / "examples/control_recovery_v1/run_study.py",
                 REPO / "simllm/backends/htsim_rnic.py", HERE / "expectations.md", expectations):
        target = snapshots / f"{digest(path)}{path.suffix}"
        if not target.exists():
            target.write_bytes(path.read_bytes())
    jobs = planned_jobs(cells)
    prepare_htsim_child_lifetime()

    def execute(job):
        cell, arm = job
        return run_cell(cell, arm, Path(binary), args.out, provenance, bounds[cell.name],
                        args.resume, args.timeout_s, args.probe_policy)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(execute, jobs))
    result = summarize(rows, provenance, [(cell.name, arm) for cell, arm in jobs])
    write_once(args.out / "results.json", result)
    print(f"verdict={result['verdict']} configurations={len(rows)} "
          f"fatal_findings={len(result['fatal_findings'])} "
          f"behavioral_findings={len(result['behavioral_findings'])}", flush=True)
    return 2 if result["verdict"] != "consumer-valid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
