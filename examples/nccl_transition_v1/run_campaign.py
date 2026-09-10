"""Run the frozen NCCL capture, keeping diagnostic timings in their own lane."""

import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import time
from pathlib import Path


def make_plan(config):
    """Interleave controls and repetitions with dense automatic bookends."""
    auto = []
    controls = []
    diagnostics = []
    for width in config["widths"]:
        for repeat in range(config["auto_repeats"]):
            auto.append(
                {
                    "lane": "timing",
                    "arm": "auto",
                    "width": width,
                    "repeat": repeat,
                    "grid": "dense_grid",
                    "env": {},
                    "graph": 0,
                }
            )
        arms = []
        for algo in config["algorithms"]:
            for proto in config["protocols"]:
                arms.append((algo + "_" + proto, {"NCCL_ALGO": algo, "NCCL_PROTO": proto}, 0))
        for count in config["fixed_channels"]:
            arms.append(
                (
                    f"Ring_LL128_cta{count}",
                    {
                        "NCCL_ALGO": "Ring",
                        "NCCL_PROTO": "LL128",
                        "NCCL_MIN_CTAS": str(count),
                        "NCCL_MAX_CTAS": str(count),
                    },
                    0,
                )
            )
        arms.append(("auto_graph", {}, config["graph_replays"]))
        for arm, env, graph in arms:
            for repeat in range(config["control_repeats"]):
                controls.append(
                    {
                        "lane": "timing",
                        "arm": arm,
                        "width": width,
                        "repeat": repeat,
                        "grid": "control_grid",
                        "env": env,
                        "graph": graph,
                    }
                )
            diagnostics.append(
                {
                    "lane": "diagnostic",
                    "arm": arm,
                    "width": width,
                    "repeat": 0,
                    "grid": "control_grid",
                    "env": env,
                    "graph": graph,
                }
            )
        diagnostics.append(
            {
                "lane": "diagnostic",
                "arm": "auto",
                "width": width,
                "repeat": 0,
                "grid": "dense_grid",
                "env": {},
                "graph": 0,
            }
        )
    legacy = [
        {
            "lane": "legacy",
            "arm": "auto",
            "width": 0,
            "repeat": r,
            "grid": "dense_grid",
            "env": {},
            "graph": 0,
        }
        for r in range(config["auto_repeats"])
    ]
    middle = [r for r in auto if r["repeat"] not in (0, config["auto_repeats"] - 1)]
    middle += controls + legacy[1:-1]
    random.Random(config["random_seed"]).shuffle(middle)
    return (
        [legacy[0]]
        + [r for r in auto if r["repeat"] == 0]
        + middle
        + [r for r in auto if r["repeat"] == config["auto_repeats"] - 1]
        + [legacy[-1]]
        + diagnostics
    )


def command_for(run, config, binary, legacy, output):
    if run["lane"] == "legacy":
        return [str(legacy), "--out", str(output)]
    diagnostic = run["lane"] == "diagnostic"
    grid = config[run["grid"]]
    return [
        str(binary),
        "-t",
        str(run["width"]),
        "-g",
        "1",
        "-b",
        str(grid["start_bytes"]),
        "-e",
        str(grid["end_bytes"]),
        "-i",
        str(grid["step_bytes"]),
        "-d",
        "float",
        "-o",
        "sum",
        "-w",
        str(config["diagnostic_warmup" if diagnostic else "timing_warmup"]),
        "-n",
        str(config["diagnostic_iterations" if diagnostic else "timing_iterations"]),
        "-N",
        "1",
        "-m",
        "1",
        "-a",
        "3",
        "-z",
        "0",
        "-c",
        "1",
        "-G",
        str(run["graph"]),
        "-U",
        "1" if diagnostic else "0",
        "-J",
        str(output),
    ]


def execute(command, env, logfile, timeout):
    started = time.time()
    with logfile.open("w") as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = 124
    return code, time.time() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expectations", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=("a100", "gh200"), required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--implementation", required=True)
    args = parser.parse_args()
    config = json.loads(args.expectations.read_text())
    args.output.mkdir(exist_ok=True, parents=True)
    if (args.output / "campaign.json").exists():
        raise SystemExit("Refusing to overwrite an existing campaign")
    base_env = {k: v for k, v in os.environ.items() if not k.startswith("NCCL_")}
    base_env["NCCL_DEBUG"] = "WARN"
    manifest = {
        "schema": "simllm-nccl-transition-campaign-v1",
        "architecture": args.architecture,
        "freeze": args.freeze,
        "implementation": args.implementation,
        "expectations_sha256": hashlib.sha256(args.expectations.read_bytes()).hexdigest(),
        "inherited_nccl_environment": {
            k: v for k, v in os.environ.items() if k.startswith("NCCL_")
        },
        "runs": [],
    }
    path = args.output / "campaign.json"
    plan = make_plan(config)
    for sequence, run in enumerate(plan):
        stem = (f"{sequence:03d}_{run['lane']}_{run['arm']}_"
                f"w{run['width']}_r{run['repeat']}")
        output = args.output / (stem + ".json")
        log = args.output / (stem + ".log")
        env = dict(base_env, **run["env"])
        if run["lane"] == "diagnostic":
            env.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,GRAPH,TUNING")
            env["NCCL_DEBUG_FILE"] = str(args.output / (stem + ".nccl.%h.%p.log"))
        command = command_for(run, config, args.binary, args.legacy, output)
        record = dict(
            run,
            sequence=sequence,
            command=command,
            output=output.name,
            log=log.name,
            started_unix=time.time(),
        )
        print(f"[{sequence + 1}/{len(plan)}] {stem}", flush=True)
        code, elapsed = execute(command, env, log, config["process_timeout_seconds"])
        record.update(
            returncode=code,
            elapsed_seconds=elapsed,
            output_sha256=(
                hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None
            ),
        )
        manifest["runs"].append(record)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        # A failed primary capture is fatal; a failed forced control is retained.
        if code and run["arm"] == "auto" and run["lane"] != "diagnostic":
            raise SystemExit("Fatal primary capture failure: " + stem)
    manifest["completed_unix"] = time.time()
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Campaign complete: {len(plan)} process records", flush=True)


if __name__ == "__main__":
    main()
