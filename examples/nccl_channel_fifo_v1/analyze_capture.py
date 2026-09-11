"""Audit the frozen identification campaign and summarize ordinary timings."""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from run_capture import conditions


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def empty_inventory(path):
    return path.exists() and len(path.read_text().strip().splitlines()) == 1


def audit_allocation(directory, cfg, architecture):
    if (directory / "library.sha256").read_text().split()[0] != cfg["library_sha256"][architecture]:
        raise RuntimeError("shared-library identity changed")
    topology = [line.split() for line in (directory / "topology.txt").read_text().splitlines()
                if re.match(r"^GPU[0-3]\s", line)]
    expected_link = "NV4" if architecture == "a100" else "NV6"
    if (len(topology) != 4 or any(row[0] != f"GPU{rank}" or len(row) < 5
            or any(value != ("X" if peer == rank else expected_link) for peer, value in enumerate(row[1:5]))
            for rank, row in enumerate(topology))):
        raise RuntimeError("captured topology is not the frozen four-GPU direct mesh")
    if not empty_inventory(directory / "processes_before.csv"):
        raise RuntimeError("allocation begins with a foreign GPU process")
    checkpoint = directory / "checkpoint.json"
    if checkpoint.exists():
        status = json.loads(checkpoint.read_text())
        if (status["status"] != "intentional_quiescent_checkpoint" or status["gpu_processes"]
                or status["live_children"] or not empty_inventory(directory / "checkpoint_processes_after.csv")):
            raise RuntimeError("invalid allocation checkpoint")
        termination = "intentional_quiescent_checkpoint"
    else:
        if not (directory / "completed.txt").exists() or not empty_inventory(directory / "processes_after.csv"):
            raise RuntimeError("allocation completion or final process inventory is missing")
        termination = "completed"
    clocks = []
    samples = directory / "state_samples.csv"
    if samples.exists():
        for row in read_csv(samples):
            for key, value in row.items():
                if key and "clocks.current.sm" in key:
                    try:
                        clocks.append(float(value.split()[0]))
                    except (ValueError, AttributeError):
                        pass
    return {"capture": directory.name, "termination": termination,
            "probe_sha256": digest(directory / "bin/channel_probe"),
            "observer_sha256": digest(directory / "bin/libnccl-channel-observer.so"),
            "sampled_sm_clock_mhz_min": min(clocks) if clocks else None,
            "sampled_sm_clock_mhz_max": max(clocks) if clocks else None,
            "clock_scope": "allocation samples include idle periods; not per-kernel clocks"}


def geometry(raw, cell):
    record = json.loads((raw / f"observer-{cell['condition_id']}.record.json").read_text())
    if record["status"] != 0 or not record["launch_stream_verified"]:
        raise RuntimeError("selection observer did not verify the launch stream")
    by_size = defaultdict(list)
    for rank, size, algo, protocol, channels, warps in record["selections"]:
        by_size[size].append((rank, algo.upper(), protocol.upper(), channels, warps))
    if set(by_size) != set(cell["sizes"]):
        raise RuntimeError("selection payload inventory differs")
    result = {}
    for size, rows in by_size.items():
        if sorted(row[0] for row in rows) != list(range(cell["width"])):
            raise RuntimeError("selection rank inventory differs")
        choices = {row[1:] for row in rows}
        if len(choices) != 1:
            raise RuntimeError("ranks disagree on selected geometry")
        algo, protocol, channels, warps = choices.pop()
        reasons = []
        if algo != "RING":
            reasons.append("algorithm_unrealized")
        if cell["protocol"] != "AUTO" and protocol != cell["protocol"]:
            reasons.append("protocol_unrealized")
        if cell["channels"] and channels != cell["channels"]:
            reasons.append("channels_unrealized")
        if cell["threads"]:
            expected = cell["threads"] // 32 + (protocol == "SIMPLE")
            if warps != expected:
                reasons.append("threads_unrealized")
        for resource in record["resources"]:
            if (resource["requested"] != cell["sms"] or resource["granted"] < cell["sms"]
                    or not 0 < resource["observed"] <= resource["granted"]):
                reasons.append("resource_unrealized")
        result[size] = {"protocol": protocol, "channels": channels, "warps": warps,
                        "granted_sms": min(r["granted"] for r in record["resources"]),
                        "reasons": sorted(set(reasons))}
    return result


def timing_rows(raw, record, allocation_mib):
    cell = record["cell"]
    expected_env = {"NCCL_ALGO": "Ring"}
    if cell["protocol"] != "AUTO":
        expected_env["NCCL_PROTO"] = cell["protocol"]
    if cell["channels"]:
        expected_env.update(NCCL_MIN_NCHANNELS=str(cell["channels"]), NCCL_MAX_NCHANNELS=str(cell["channels"]),
                            NCCL_THREAD_THRESHOLDS="0 0 0 0 0 0")
    if cell["threads"]:
        expected_env["NCCL_LL128_NTHREADS" if cell["protocol"] == "LL128" else "NCCL_NTHREADS"] = str(cell["threads"])
    if record["environment_overrides"] != expected_env:
        raise RuntimeError("ordinary timing environment differs from its frozen condition")
    if (sorted(r["rank"] for r in record["resources"]) != list(range(cell["width"]))
            or len({r["cpu"] for r in record["resources"]}) != cell["width"]):
        raise RuntimeError("worker rank or CPU affinity inventory differs")
    table = read_csv(raw / f"{record['tag']}.csv")
    if len(table) != len(cell["sizes"]) * (cell["width"] + 1):
        raise RuntimeError("timed rank or payload inventory differs")
    grouped = defaultdict(list)
    for row in table:
        if (int(row["mismatching_ranks"]) or int(row["width"]) != cell["width"]
                or int(row["warmup"]) != cell["warm"] or int(row["timed"]) != cell["timed"]
                or int(row["rotate"]) != cell["rotate"] or int(row["persistent"]) != 1
                or int(row["requested_sms"]) != cell["sms"]
                or int(row["allocation_bytes"]) != allocation_mib * 1048576):
            raise RuntimeError("correctness or timed-control guard failed")
        if any(not math.isfinite(float(row[key])) or float(row[key]) <= 0 for key in ("event_us", "wall_us")):
            raise RuntimeError("invalid timer value")
        rank = int(row["rank"])
        expected_granted = min(r["granted"] for r in record["resources"]) if rank == -1 else next(
            r["granted"] for r in record["resources"] if r["rank"] == rank)
        if int(row["granted_sms"]) != expected_granted:
            raise RuntimeError("timed SM grant differs from the process resource record")
        grouped[int(row["bytes"])].append(row)
    if set(grouped) != set(cell["sizes"]):
        raise RuntimeError("timed payload inventory differs")
    result = {}
    for size, rows in grouped.items():
        if sorted(int(row["rank"]) for row in rows) != list(range(-1, cell["width"])):
            raise RuntimeError("timed rank identity differs")
        combined = next(row for row in rows if row["rank"] == "-1")
        for key in ("event_us", "wall_us"):
            expected = max(float(row[key]) for row in rows if row["rank"] != "-1")
            if not math.isclose(float(combined[key]), expected, rel_tol=1e-10, abs_tol=1e-8):
                raise RuntimeError("combined timer is not the maximum across ranks")
        result[size] = combined
    return result


def write_table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, architecture, capture, cfg):
    current = root / architecture / capture
    raw = current / "raw"
    manifest = json.loads((raw / "manifest.json").read_text())
    planned = conditions(cfg)
    if json.loads((raw / "conditions.json").read_text()) != planned:
        raise RuntimeError("condition manifest differs from the frozen generator")
    timed = [r for r in manifest if r.get("observer") is False]
    if len(timed) != len(planned) * cfg["process_repetitions"]:
        raise RuntimeError("campaign is incomplete")
    sources = {r.get("source_capture", capture) for r in timed} | {capture}
    allocations = {name: audit_allocation(root / architecture / name, cfg, architecture) for name in sources}
    for name in sources:
        for artifact in ("gpu_identity.csv", "topology.txt", "library.sha256", "nvcc_version.txt"):
            if (root / architecture / name / artifact).read_bytes() != (current / artifact).read_bytes():
                raise RuntimeError(f"allocation identity differs: {artifact}")
        if allocations[name]["probe_sha256"] != allocations[capture]["probe_sha256"]:
            raise RuntimeError("compiled probe identity differs")
        if allocations[name]["observer_sha256"] != allocations[capture]["observer_sha256"]:
            raise RuntimeError("selection observer identity differs")
    observed = {(name, cell["condition_id"]): geometry(root / architecture / name / "raw", cell)
                for name in sources for cell in planned}
    samples = defaultdict(list)
    by_id = {cell["condition_id"]: cell for cell in planned}
    identities = set()
    for record in timed:
        cell, repeat = record["cell"], record["repeat"]
        if cell != by_id[cell["condition_id"]] or record["tag"] != f"timed-r{repeat}-{cell['condition_id']}":
            raise RuntimeError("timed condition or tag differs from the frozen schedule")
        key = (cell["condition_id"], repeat)
        if record["status"] != 0 or key in identities or repeat not in range(cfg["process_repetitions"]):
            raise RuntimeError("timed process identity or status is invalid")
        identities.add(key)
        source = record.get("source_capture", capture)
        source_raw = root / architecture / source / "raw"
        if source != capture and (
                digest(source_raw / f"{record['tag']}.record.json") != record["record_sha256"]
                or digest(source_raw / f"{record['tag']}.csv") != record["csv_sha256"]):
            raise RuntimeError("reused timing bytes changed")
        for size, row in timing_rows(source_raw, record, cfg["allocation_mib"]).items():
            selected = observed[(source, cell["condition_id"])][size]
            reasons = list(selected["reasons"])
            if any(observed[(name, cell["condition_id"])][size] != selected for name in sources):
                reasons.append("geometry_changed_across_allocations")
            # Floor for the declared uncompressed direct meshes, before reading
            # accuracy: at least 2(n-1)S/n useful bytes per rank on average.
            capacity = {"a100": {2: 100e9, 4: 300e9}, "gh200": {2: 150e9, 4: 450e9}}[architecture][cell["width"]]
            floor_us = 2 * (cell["width"] - 1) * size / cell["width"] / capacity * 1e6
            event, wall = float(row["event_us"]), float(row["wall_us"])
            if min(event, wall) + 1e-8 < floor_us:
                raise RuntimeError("timing is below the declared uncompressed physical floor")
            samples[(cell["condition_id"], size)].append({
                "capture": source, "repeat": repeat, "event_us": event, "wall_us": wall,
                "paired_gap_us": wall - event, "floor_us": floor_us,
                "selection": selected, "reasons": sorted(set(reasons)),
            })
    summaries, by_allocation = [], []
    for cell in planned:
        for size in cell["sizes"]:
            rows = samples[(cell["condition_id"], size)]
            if sorted(r["repeat"] for r in rows) != list(range(cfg["process_repetitions"])):
                raise RuntimeError("five independent repetitions are required")
            reasons = sorted({reason for row in rows for reason in row["reasons"]})
            selected = rows[0]["selection"]
            base = {"architecture": architecture, "condition_id": cell["condition_id"],
                    "family": cell["family"], "width": cell["width"], "bytes": size,
                    "requested_protocol": cell["protocol"], "protocol": selected["protocol"],
                    "requested_channels": cell["channels"], "channels": selected["channels"],
                    "requested_sms": cell["sms"], "granted_sms": selected["granted_sms"],
                    "requested_threads": cell["threads"], "warps": selected["warps"],
                    "warmup": cell["warm"], "iterations": cell["timed"], "rotate": cell["rotate"],
                    "qualified_control": not reasons, "unqualified_reasons": ";".join(reasons),
                    "physical_floor_us": rows[0]["floor_us"]}
            for origin in [None, *sorted({r["capture"] for r in rows})]:
                subset = rows if origin is None else [r for r in rows if r["capture"] == origin]
                result = {**base, "captures": ";".join(sorted({r["capture"] for r in subset})),
                          "repetitions": len(subset)}
                for metric in ("event_us", "wall_us", "paired_gap_us"):
                    q1, median, q3 = np.quantile([r[metric] for r in subset], [0.25, 0.5, 0.75])
                    result.update({metric: float(median), f"{metric}_q1": float(q1), f"{metric}_q3": float(q3)})
                (summaries if origin is None else by_allocation).append(result)
    return summaries, by_allocation, list(allocations.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--capture", action="append", required=True, help="architecture:capture_id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((Path(__file__).with_name("hardware_expectations.json")).read_text())
    pooled, separate, allocations = [], [], []
    try:
        for value in args.capture:
            architecture, capture = value.split(":")
            rows, per_allocation, provenance = analyze(args.root, architecture, capture, cfg)
            pooled.extend(rows)
            separate.extend(per_allocation)
            allocations.extend({"architecture": architecture, **p} for p in provenance)
    except (RuntimeError, KeyError, OSError, ValueError) as error:
        (args.output / "summary.json").write_text(json.dumps({
            "status": "void_or_incomplete", "finding": str(error), "evidence_use": cfg["evidence_use"]
        }, indent=2) + "\n")
        raise
    write_table(args.output / "measurements.csv", pooled)
    write_table(args.output / "per_allocation.csv", separate)
    summary = {"status": "complete", "global_guards": "valid", "expectations_commit": "c5c9a963",
               "continuation_expectations_commit": "3615669d", "evidence_use": cfg["evidence_use"],
               "measurement_conditions": len(pooled),
               "unqualified_control_points": sum(not r["qualified_control"] for r in pooled),
               "allocations": allocations}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
