"""Capture a locked model's independent validation and timing controls."""

import argparse
import hashlib
import itertools
import json
import os
import random
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.stage / "expectations.json").read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    sizes = args.stage / "method_sizes.txt"
    sizes.write_text("\n".join(map(str, config["method_payloads_bytes"])) + "\n")
    plan = []
    for repeat in range(config["process_repeats"]):
        plan.append({"kind": "fresh_legacy", "repeat": repeat})
        plan.append({"kind": "visibility", "repeat": repeat})
        for width in config["widths"]:
            plan.append({"kind": "fresh_reference", "width": width, "repeat": repeat})
            for warm, timed in config["method_counts"]:
                # All method cells use the same 1-GiB allocation. Rotation is
                # an isolated access-pattern control, not a claim of matching
                # nccl-tests' smaller allocation and data initialization.
                for persistent, rotate in itertools.product((0, 1), repeat=2):
                    plan.append(
                        {
                            "kind": "method",
                            "width": width,
                            "repeat": repeat,
                            "warm": warm,
                            "timed": timed,
                            "persistent": persistent,
                            "rotate": rotate,
                        }
                    )
                for grid in ((8, 1024, 1016), (524288, 4194304, 262144)):
                    plan.append(
                        {
                            "kind": "method_reference",
                            "width": width,
                            "repeat": repeat,
                            "warm": warm,
                            "timed": timed,
                            "grid": grid,
                        }
                    )
    if config.get("fresh_only", False):
        plan = [cell for cell in plan if cell["kind"] in ("fresh_legacy", "fresh_reference")]
    random.Random(config["seed"]).shuffle(plan)
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    environment = {k: v for k, v in os.environ.items() if not k.startswith("NCCL_")}
    manifest = []
    for index, cell in enumerate(plan):
        stem = out / f"{index:03d}_{cell['kind']}"
        kind = cell["kind"]
        target = stem.with_suffix(".csv" if kind in ("method", "visibility") else ".json")
        if kind == "fresh_legacy":
            command = [str(args.stage / "legacy_fresh"), "--out", str(target)]
        elif kind == "visibility":
            command = [str(args.stage / "visibility_probe")]
        elif kind == "method":
            command = [str(args.stage / "method_probe"), str(sizes), str(target)]
            command += [str(cell[k]) for k in ("width", "warm", "timed", "persistent", "rotate")]
            command += ["1024"]
        else:
            if kind == "fresh_reference":
                grid = (
                    config["fresh_payloads_bytes"][0],
                    config["fresh_payloads_bytes"][-1],
                    65536,
                )
                warm, timed = config["reference_warmup"], config["reference_iterations"]
            else:
                grid = cell["grid"]
                warm, timed = cell["warm"], cell["timed"]
            command = [
                str(args.stage / "all_reduce_perf"),
                "-t",
                str(cell["width"]),
                "-g",
                "1",
                "-b",
                str(grid[0]),
                "-e",
                str(grid[1]),
                "-i",
                str(grid[2]),
                "-d",
                "float",
                "-o",
                "sum",
                "-w",
                str(warm),
                "-n",
                str(timed),
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
                "0",
                "-U",
                "0",
                "-J",
                str(target),
            ]
        start = time.time()
        log = target if kind == "visibility" else stem.with_suffix(".log")
        with log.open("w") as stream:
            result = subprocess.run(
                command,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=120,
                check=False,
            )
        entry = {
            **cell,
            "index": index,
            "result": target.name,
            "command": command,
            "returncode": result.returncode,
            "elapsed_seconds": time.time() - start,
        }
        if result.returncode or not target.is_file():
            entry["fatal"] = "capture failure or missing output"
        else:
            entry["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest.append(entry)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{index + 1}/{len(plan)} {kind} {result.returncode}", flush=True)
        if "fatal" in entry:
            raise RuntimeError(entry["fatal"])


if __name__ == "__main__":
    main()
