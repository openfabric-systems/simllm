"""Execute the frozen channel/SM identification manifest with paired timers."""

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
from pathlib import Path


def conditions(cfg):
    result = []

    def add(family, width, protocol, channels, sizes, **extra):
        result.append({"family": family, "width": width, "protocol": protocol,
                       "channels": channels, "sizes": sorted(set(sizes)), "sms": 0,
                       "threads": 0, "warm": cfg["primary_warmup"],
                       "timed": cfg["primary_timed_iterations"], "rotate": 0, **extra})

    for width, proto, channels in itertools.product(cfg["widths"], cfg["protocols"], cfg["channels"]):
        add("channels", width, proto, channels,
            [1024 * size for size in cfg["fixed_payload_kib"]]
            + [1024 * channels * size for size in cfg["per_channel_payload_kib"]])
    residual = [cfg["residual_payload_start"] + index * cfg["residual_payload_step"]
                for index in range(cfg["residual_payload_count"])]
    for width, channels in itertools.product(cfg["widths"], cfg["residual_channels"]):
        add("residual", width, "AUTO" if channels == 0 else "LL128", channels, residual)
    for width, proto, channels, sms in itertools.product(
            cfg["widths"], cfg["protocols"], cfg["resource_channels"], cfg["resource_sms"]):
        add("resources", width, proto, channels,
            [size * 1024 for size in cfg["resource_payload_kib"]], sms=sms)
    for width, proto, channels in itertools.product(cfg["widths"], cfg["protocols"], cfg["thread_channels"]):
        for threads in cfg["thread_requests"][proto]:
            add("threads", width, proto, channels,
                [size * 1024 for size in cfg["thread_payload_kib"]], threads=threads)
    for width, proto, warm, timed, rotate in itertools.product(
            cfg["widths"], cfg["protocols"], cfg["warmups"], cfg["timed_iterations"], cfg["rotation"]):
        add("timers", width, proto, cfg["method_channels"],
            [size * 1024 for size in cfg["method_payload_kib"]], warm=warm, timed=timed, rotate=rotate)
    for index, row in enumerate(result):
        row["condition_id"] = f"c{index:04d}"
    return result


def invoke(stage, output, cfg, cell, *, tag, observe=False, seed=0):
    stem = output / tag
    sizes = list(cell["sizes"])
    random.Random(seed).shuffle(sizes)
    stem.with_suffix(".sizes").write_text("\n".join(map(str, sizes)) + "\n")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("NCCL_")}
    environment["NCCL_ALGO"] = "Ring"
    if cell["protocol"] != "AUTO":
        environment["NCCL_PROTO"] = cell["protocol"]
    if cell["channels"]:
        environment.update(NCCL_MIN_NCHANNELS=str(cell["channels"]), NCCL_MAX_NCHANNELS=str(cell["channels"]),
                           NCCL_THREAD_THRESHOLDS="0 0 0 0 0 0")
    if cell["threads"]:
        environment["NCCL_LL128_NTHREADS" if cell["protocol"] == "LL128" else "NCCL_NTHREADS"] = str(cell["threads"])
    if observe:
        environment["NCCL_PROFILER_PLUGIN"] = str(stage / "libnccl-channel-observer.so")
        environment["NCCL_TRANSITION_SELECTION_LOG"] = str(stem.with_suffix(".selection.jsonl"))
    command = [str(stage / "channel_probe"), str(stem.with_suffix(".sizes")), str(stem.with_suffix(".csv")),
               str(cell["width"]), str(cell["warm"]), str(cell["timed"]), "1", str(cell["rotate"]),
               str(cfg["allocation_mib"]), str(cell["sms"])]
    with stem.with_suffix(".stdout").open("w") as stdout, stem.with_suffix(".stderr").open("w") as stderr:
        try:
            result = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr,
                                    timeout=cfg["process_timeout_seconds"], check=False)
            status = result.returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
    record = {"tag": tag, "cell": cell, "observer": observe, "status": status,
              "environment_overrides": {k: v for k, v in environment.items() if k.startswith("NCCL_")}}
    if status == 0:
        rows = list(csv.DictReader(stem.with_suffix(".csv").open()))
        if (len(rows) != len(sizes) * (cell["width"] + 1)
                or any(int(row["mismatching_ranks"]) or float(row["event_us"]) <= 0
                       or float(row["wall_us"]) <= 0 for row in rows)):
            raise RuntimeError("fatal correctness or timer/rank inventory failure")
        metadata = re.findall(r"META rank=(\d+) requested_sms=(\d+) granted_sms=(\d+) observed_sms=(\d+) stream=(\S+) cpu=(\d+)",
                              stem.with_suffix(".stdout").read_text())
        if len(metadata) != cell["width"]:
            raise RuntimeError("missing resource metadata")
        record["resources"] = [{"rank": int(r), "requested": int(q), "granted": int(g),
                                "observed": int(o), "stream": s, "cpu": int(c)} for r, q, g, o, s, c in metadata]
        if observe:
            path = stem.with_suffix(".selection.jsonl")
            observed = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            launches = [row for row in observed if row["kind"] == "kernel_launch"]
            streams = {row["rank"]: row["stream"] for row in record["resources"]}
            record["launch_stream_verified"] = bool(launches) and all(
                row["stream"] == streams[row["rank"]] for row in launches)
            selections = sorted({(row["rank"], row["count"] * 4, row["algo"], row["proto"],
                                  row["#channels"], row["#warps"]) for row in observed if row["kind"] == "collective"})
            record["selections"] = selections
            # Public descriptor spelling is retained; compare case-insensitively.
            if not selections or any(row[2].upper() != "RING" or row[3].upper() not in ("LL", "LL128", "SIMPLE") for row in selections):
                raise RuntimeError("missing or unsupported source selection")
    stem.with_suffix(".record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((args.stage / "hardware_expectations.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    planned = conditions(cfg)
    plan_bytes = (json.dumps(planned, indent=2) + "\n").encode()
    (args.output / "conditions.json").write_bytes(plan_bytes)
    (args.output / "conditions.sha256").write_text(hashlib.sha256(plan_bytes).hexdigest() + "\n")
    qualified = {0}
    pilots = []
    for sms in cfg["resource_sms"]:
        cell = {"family": "pilot", "width": 4, "protocol": cfg["pilot_protocol"],
                "channels": cfg["pilot_channels"], "sizes": [cfg["pilot_payload_bytes"]],
                "sms": sms, "threads": 0, "warm": cfg["pilot_warmup"],
                "timed": cfg["pilot_iterations"], "rotate": 0}
        pilot = invoke(args.stage, args.output, cfg, cell, tag=f"pilot-sm{sms}", observe=True)
        pilots.append(pilot)
        if pilot["status"] == 0 and pilot.get("launch_stream_verified"):
            qualified.add(sms)
        elif sms == 0:
            raise RuntimeError("ordinary-stream capability pilot failed")
    (args.output / "pilots.json").write_text(json.dumps(pilots, indent=2) + "\n")
    records = []
    for cell in planned:
        if cell["sms"] not in qualified:
            records.append({"cell": cell, "status": "unsupported_resource_pilot"})
            continue
        result = invoke(args.stage, args.output, cfg, cell, tag=f"observer-{cell['condition_id']}",
                        observe=True, seed=cfg["seed"])
        records.append(result)
        if result["status"] != 0:
            if cell["sms"]:
                continue
            raise RuntimeError("ordinary observer failed; retain void evidence")
        if cell["sms"] and not result.get("launch_stream_verified"):
            continue
    eligible = [row["cell"] for row in records if row.get("observer") and row["status"] == 0
                and (row["cell"]["sms"] == 0 or row.get("launch_stream_verified"))]
    schedule = [(repeat, cell) for repeat in range(cfg["process_repetitions"]) for cell in eligible]
    random.Random(cfg["seed"]).shuffle(schedule)
    (args.output / "timed_schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
    for index, (repeat, cell) in enumerate(schedule):
        result = invoke(args.stage, args.output, cfg, cell, tag=f"timed-r{repeat}-{cell['condition_id']}",
                        seed=cfg["seed"] + index)
        result["repeat"] = repeat
        records.append(result)
        if result["status"] != 0:
            raise RuntimeError("timed cell failed; retain void evidence")
        if index % 20 == 0:
            print(json.dumps({"completed": index + 1, "planned": len(schedule)}), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"status": "capture_complete", "conditions": len(planned),
                      "timed_processes": len(schedule), "qualified_requested_sms": sorted(qualified)}), flush=True)


if __name__ == "__main__":
    main()
