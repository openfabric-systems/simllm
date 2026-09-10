"""Capture each selected collective descriptor, independent of summary readiness."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from run_campaign import command_for, execute, make_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expectations", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--implementation", required=True)
    args = parser.parse_args()
    config = json.loads(args.expectations.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "observer.json").exists():
        raise SystemExit("Refusing to overwrite an observer campaign")
    plan = [r for r in make_plan(config) if r["lane"] == "diagnostic"]
    plan.sort(key=lambda r: (r["arm"] != "auto", r["width"], r["arm"]))
    plan.append({"lane": "legacy", "arm": "auto", "width": 0, "repeat": 0,
                 "grid": "dense_grid", "env": {}, "graph": 0})
    manifest = {"architecture": args.architecture, "implementation": args.implementation,
                "amendment": "5cb2bc0c", "plugin_sha256": hashlib.sha256(args.plugin.read_bytes()).hexdigest(),
                "runs": []}
    for index, run in enumerate(plan):
        stem = f"{index:03d}_{run['lane']}_{run['arm']}_w{run['width']}"
        events = args.output / (stem + ".events.jsonl")
        result = args.output / (stem + ".json")
        command = command_for(run, config, args.binary, args.legacy, result)
        if "-U" in command:
            command[command.index("-U") + 1] = "0"
        env = {k: v for k, v in os.environ.items() if not k.startswith("NCCL_")}
        env.update(run["env"])
        env.update(NCCL_PROFILER_PLUGIN=str(args.plugin),
                   NCCL_TRANSITION_SELECTION_LOG=str(events), NCCL_DEBUG="INFO",
                   NCCL_DEBUG_SUBSYS="INIT,GRAPH,TUNING",
                   NCCL_DEBUG_FILE=str(args.output / (stem + ".nccl.%h.%p.log")))
        print(f"[{index + 1}/{len(plan)}] {stem}", flush=True)
        code, elapsed = execute(command, env, args.output / (stem + ".log"),
                                config["process_timeout_seconds"])
        record = dict(run, command=command, returncode=code, elapsed_seconds=elapsed,
                      events=events.name, output=result.name)
        if events.exists():
            record["events_sha256"] = hashlib.sha256(events.read_bytes()).hexdigest()
            record["collective_callbacks"] = sum(
                json.loads(line)["kind"] == "collective" for line in events.read_text().splitlines())
        else:
            record["collective_callbacks"] = 0
        manifest["runs"].append(record)
        (args.output / "observer.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if code != 0 or record["collective_callbacks"] == 0:
            raise SystemExit("Direct observer qualification failed: " + stem)
    manifest["complete"] = True
    (args.output / "observer.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Direct observer complete", flush=True)


if __name__ == "__main__":
    main()
