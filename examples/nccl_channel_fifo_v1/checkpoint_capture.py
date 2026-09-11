"""Quiesce only this capture coordinator at an ordinary-process boundary."""

import argparse
import datetime
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    target = str(args.capture / "bin/run_capture.py")
    found = []
    for line in subprocess.check_output(["ps", "-u", str(os.getuid()), "-o", "pid=,args="], universal_newlines=True).splitlines():
        pid, command = line.strip().split(None, 1)
        if command.startswith(f"python3 -u {target} "):
            found.append(int(pid))
    if len(found) != 1:
        raise RuntimeError(f"expected one owned coordinator, found {found}")
    parent = found[0]
    executable = os.readlink(f"/proc/{parent}/exe")
    os.kill(parent, signal.SIGSTOP)
    stopped = True
    try:
        deadline = time.monotonic() + 100
        while True:
            inventory = subprocess.check_output([
                "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv"], universal_newlines=True)
            children = subprocess.run(["ps", "--ppid", str(parent), "-o", "stat="],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False).stdout.splitlines()
            live = [state for state in children if not state.strip().startswith("Z")]
            if len(inventory.strip().splitlines()) == 1 and not live:
                break
            if time.monotonic() > deadline:
                raise RuntimeError("capture did not reach an empty process boundary")
            time.sleep(0.5)
        (args.capture / "checkpoint_processes_after.csv").write_text(inventory)
        record = {
            "status": "intentional_quiescent_checkpoint",
            "expectations_commit": "3615669d",
            "coordinator_pid": parent,
            "python_executable": executable,
            "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "gpu_processes": 0,
            "live_children": 0,
            "successful_timed_records": len(list((args.capture / "raw").glob("timed*.record.json"))),
        }
        (args.capture / "checkpoint.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        os.kill(parent, signal.SIGTERM)
        os.kill(parent, signal.SIGCONT)
        stopped = False
    finally:
        if stopped:
            os.kill(parent, signal.SIGCONT)


if __name__ == "__main__":
    main()
