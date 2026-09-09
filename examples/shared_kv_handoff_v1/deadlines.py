"""Run the prospectively frozen host-I/O grid with fresh, lifetime-owned children."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from examples.publication_snapshot_v1.common import git, packages, receipts, sha, write
from simllm.backends import _child_process
from simllm.backends._child_process import OwnedBinaryProcess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREEZE = "0e0e71aa586b2ccd2d786534d6b686d888e300b0"
PAYLOAD = b"\x00\xff\x80wire"
CHILD = (
    "import os\n"
    "os.write(1,b'R')\n"
    "for _ in range(2):\n"
    "    data=bytearray()\n"
    "    while len(data)<7:\n"
    "        block=os.read(0,7-len(data))\n"
    "        if not block: raise RuntimeError('early request EOF')\n"
    "        data.extend(block)\n"
    "    os.write(1,data)\n"
    "assert os.read(0,1)==b''\n"
)


def live(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def run_cell(path, allowance, idle, lifetime, enabled):
    path.mkdir()
    markers = path / "markers"
    markers.mkdir()
    environment = dict(os.environ, SIMLLM_CHILD_LIFETIME_MARKER_DIR=str(markers),
                       SIMLLM_CHILD_LIFETIME_RUN_NONCE=path.name)
    row = {"exchange_seconds": allowance, "idle_seconds": idle, "lifetime_seconds": lifetime,
           "exchange_enabled": enabled, "pid": None, "rounds": [], "exit_code": None, "failure": None}
    process = None
    started = time.monotonic()
    try:
        with OwnedBinaryProcess((sys.executable, "-c", CHILD), timeout_s=lifetime,
                                environment=environment) as process:
            row["pid"] = process.pid
            row["ready"] = process.read_exact(1).hex()
            lifetime_deadline = process._deadline
            for index in range(2):
                if index:
                    time.sleep(idle)
                start = time.monotonic()
                same_child = process.pid == row["pid"] and live(process.pid)
                context = process.io_deadline(allowance) if enabled else nullcontext()
                with context:
                    process.write(PAYLOAD)
                    response = process.read_exact(len(PAYLOAD))
                (path / f"response-{index}.bin").write_bytes(response)
                row["rounds"].append({"response_hex": response.hex(), "same_live_child": same_child,
                                      "host_seconds": time.monotonic() - start,
                                      "lifetime_unchanged": process._deadline == lifetime_deadline})
                write(path / "partial.json", row)
            context = process.io_deadline(allowance) if enabled else nullcontext()
            with context:
                row["exit_code"] = process.finish()
    except BaseException as error:
        row["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        row["host_seconds"] = time.monotonic() - started
        row["reaped"] = process is not None and not live(process.pid)
        if process is not None:
            (path / "stderr.bin").write_bytes(process.stderr)
        write(path / "result.json", row)
    return row


def execute(output):
    output = output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("raw evidence must be outside the source repository")
    if os.name != "posix":
        raise ValueError("this native evidence runner requires POSIX process identity checks")
    output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((HERE / "deadline-expectations.json").read_bytes())
    first, rows, findings = {}, [], []
    admitted, pairs = 0, 0
    source = packages(ROOT)
    identity = {"source_commit": git(ROOT, "rev-parse", "HEAD"), "package_sha256": source,
                "freeze_commit": FREEZE, "freeze_sha256": sha(HERE / "deadline-expectations.json"),
                "runner_sha256": sha(Path(__file__)), "interpreter": sys.executable,
                "interpreter_sha256": sha(Path(sys.executable).resolve()),
                "stream_origin": str(Path(_child_process.__file__).resolve())}
    write(output / "identity.json", identity)
    try:
        if git(ROOT, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError("executing source must be clean")
        original = git(ROOT, "show", FREEZE + ":examples/shared_kv_handoff_v1/deadline-expectations.json")
        if json.loads(original) != frozen:
            raise ValueError("deadline freeze changed")
        grid = frozen["native_echo_grid"]
        for allowance in grid["exchange_seconds"]:
            for idle in grid["idle_seconds"]:
                pair = []
                for enabled in (False, True):
                    name = f"exchange-{allowance}-idle-{idle}-{'enabled' if enabled else 'legacy'}"
                    path = output / name
                    try:
                        row = run_cell(path, allowance, idle, grid["lifetime_seconds"], enabled)
                    finally:
                        if path.exists():
                            first[name] = receipts(path)
                            write(output / (name + "-first-receipts.json"), first[name])
                    rows.append({"id": name, **row})
                    if (row["failure"] is not None or row["exit_code"] != 0 or not row["reaped"]
                            or row["ready"] != "52" or not idle <= row["host_seconds"] < grid["lifetime_seconds"]
                            or len(row["rounds"]) != 2 or not all(
                                item["same_live_child"] and item["lifetime_unchanged"]
                                and item["response_hex"] == PAYLOAD.hex() for item in row["rounds"])):
                        raise ValueError("host-I/O conformance failed: " + name)
                    markers = list((path / "markers").glob("*.json"))
                    if len(markers) != 1:
                        raise ValueError("unexpected native child inventory")
                    marker = json.loads(markers[0].read_bytes())
                    if marker["child_pid"] != row["pid"] or marker["owner_pid"] != os.getpid():
                        raise ValueError("native child identity changed")
                    pair.append([item["response_hex"] for item in row["rounds"]])
                    admitted += 1
                    print("Qualified " + name, flush=True)
                if pair[0] != pair[1]:
                    raise ValueError("optional exchange changed complete wire responses")
                pairs += 1
        if packages(ROOT) != source:
            raise ValueError("executing package changed")
        if len({row["pid"] for row in rows}) != 8:
            raise ValueError("fresh native child identities were reused")
    except BaseException as error:  # noqa: BLE001, retain the first failure as a void run.
        findings.append({"type": type(error).__name__, "message": str(error)})
    finally:
        final = {name: receipts(output / name) for name in first}
        if final != first:
            findings.append({"type": "ValueError", "message": "first-exit file receipts changed"})
        result = {"schema": "simllm-shared-handoff-deadline-results-v1", "identity": identity,
                  "verdict": "VOID" if findings else "PASS", "behavioral_score": None,
                  "captured_native_configurations": len(rows), "admitted_native_configurations": admitted,
                  "completed_component_grid_pairs": pairs, "required_component_grid_pairs": 4,
                  "behavioral_instances": 0, "findings": findings, "cells": rows,
                  "first_receipts": first, "final_receipts": final}
        write(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = execute(args.output_root)
    print(result["verdict"], flush=True)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
