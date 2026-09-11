"""Capture channel peers and FIFO placement separately from ordinary timings."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
from pathlib import Path


def parse_connections(text, width):
    rings, totals, connections = {}, set(), {}
    for line in text.splitlines():
        match = re.search(r"Channel (\d+)/(\d+) :\s*((?:\d+\s*)+)$", line)
        if match:
            channel, total = map(int, match.group(1, 2))
            order = list(map(int, match[3].split()))
            if sorted(order) != list(range(width)) or (channel in rings and rings[channel] != order):
                raise RuntimeError("invalid or conflicting Ring permutation")
            rings[channel] = order
            totals.add(total)
        match = re.search(r"Channel (\d+)/(\d+) : (\d+)\[\d+\] -> (\d+)\[\d+\] via (.+)$", line)
        if match:
            channel, index, source, destination = map(int, match.group(1, 2, 3, 4))
            key = (channel, index, source, destination)
            path = match[5].strip()
            if key in connections and connections[key] != path:
                raise RuntimeError("conflicting directed connection placement")
            connections[key] = path
    reasons = []
    if len(totals) != 1 or set(rings) != set(range(next(iter(totals), 0))):
        reasons.append("incomplete_channel_ring_inventory")
    result = []
    for channel, order in sorted(rings.items()):
        for index, source in enumerate(order):
            destination = order[(index + 1) % width]
            path = connections.get((channel, 0, source, destination))
            if path is None:
                reasons.append("missing_ring_connection")
            elif not path.startswith(("P2P/direct pointer", "P2P/CUMEM", "P2P/IPC")) or "/CE" in path:
                reasons.append("unsupported_transport")
            result.append({"channel": channel, "source": source, "destination": destination,
                           "transport": path, "read_capable": path is not None and "/read" in path})
    return {"rings": rings, "connections": result, "unqualified_reasons": sorted(set(reasons))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=("a100", "gh200"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cells = []
    for width, channels, protocol in itertools.product([2, 4], [0, 1, 8, 22, 23, 24, 32], ["LL", "LL128", "SIMPLE"]):
        for read in ([None, 0, 1] if protocol == "SIMPLE" else [None]):
            cells.append({"width": width, "channels": channels, "protocol": protocol, "read_enable": read})
    plan = (json.dumps(cells, indent=2) + "\n").encode()
    (args.output / "conditions.json").write_bytes(plan)
    (args.output / "conditions.sha256").write_text(hashlib.sha256(plan).hexdigest() + "\n")
    records = []
    for index, cell in enumerate(cells):
        stem = args.output / f"transport-{index:03d}"
        stem.with_suffix(".sizes").write_text("1048576\n3407872\n")
        env = {key: value for key, value in os.environ.items() if not key.startswith("NCCL_")}
        env.update(NCCL_ALGO="Ring", NCCL_PROTO=cell["protocol"], NCCL_DEBUG="INFO",
                   NCCL_DEBUG_SUBSYS="INIT,GRAPH,P2P", NCCL_DEBUG_FILE=str(stem.with_suffix(".debug.log")),
                   NCCL_PROFILER_PLUGIN=str(args.stage / "libnccl-channel-observer.so"),
                   NCCL_TRANSITION_SELECTION_LOG=str(stem.with_suffix(".selection.jsonl")))
        if cell["channels"]:
            env.update(NCCL_MIN_NCHANNELS=str(cell["channels"]), NCCL_MAX_NCHANNELS=str(cell["channels"]),
                       NCCL_THREAD_THRESHOLDS="0 0 0 0 0 0")
        if cell["read_enable"] is not None:
            env["NCCL_P2P_READ_ENABLE"] = str(cell["read_enable"])
        cmd = [str(args.stage / "channel_probe"), str(stem.with_suffix(".sizes")),
               str(stem.with_suffix(".csv")), str(cell["width"]), "1", "1", "1", "0", "64", "0"]
        with stem.with_suffix(".stdout").open("w") as stdout, stem.with_suffix(".stderr").open("w") as stderr:
            result = subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr, timeout=120, check=False)
        record = {"cell": cell, "returncode": result.returncode,
                  "environment_overrides": {k: v for k, v in env.items() if k.startswith("NCCL_")}}
        if result.returncode:
            stem.with_suffix(".record.json").write_text(json.dumps(record, indent=2) + "\n")
            raise RuntimeError("diagnostic process failed")
        with stem.with_suffix(".csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if (len(rows) != 2 * (cell["width"] + 1)
                or {(int(r["bytes"]), int(r["rank"])) for r in rows}
                != set(itertools.product([1048576, 3407872], range(-1, cell["width"])))
                or any(int(r["mismatching_ranks"]) or any(
                    not math.isfinite(float(r[key])) or float(r[key]) <= 0 for key in ("event_us", "wall_us")) for r in rows)):
            raise RuntimeError("diagnostic correctness, rank or timer guard failed")
        record.update(parse_connections(stem.with_suffix(".debug.log").read_text(), cell["width"]))
        selections = [json.loads(line) for line in stem.with_suffix(".selection.jsonl").read_text().splitlines()]
        choices = sorted({(r["rank"], r["count"] * 4, r["algo"], r["proto"], r["#channels"], r["#warps"])
                          for r in selections if r["kind"] == "collective"})
        record["selections"] = choices
        if ({(r[0], r[1]) for r in choices} != set(itertools.product(range(cell["width"]), [1048576, 3407872]))
                or any(r[2].upper() != "RING" or r[3].upper() != cell["protocol"] for r in choices)):
            record["unqualified_reasons"].append("selection_inventory_or_protocol")
        expected_read = args.architecture == "a100" if cell["read_enable"] is None else bool(cell["read_enable"])
        if any(r["read_capable"] != expected_read for r in record["connections"]):
            record["unqualified_reasons"].append("source_predicted_placement_not_observed")
        placements = {"sender_read" if r["read_capable"] and cell["protocol"] == "SIMPLE" else "receiver_write"
                      for r in record["connections"] if r["transport"] is not None}
        record["payload_fifo_placement"] = next(iter(placements)) if len(placements) == 1 else "mixed_or_unobserved"
        record["qualified"] = not record["unqualified_reasons"]
        stem.with_suffix(".record.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print(json.dumps({"completed": index + 1, "planned": len(cells), "qualified": record["qualified"]}), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps({"expectations_commit": "34e8170f",
        "status": "complete", "evidence_use": "transport_diagnostics_not_timing_calibration",
        "configurations": len(records), "unqualified_configurations": sum(not r["qualified"] for r in records)
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
