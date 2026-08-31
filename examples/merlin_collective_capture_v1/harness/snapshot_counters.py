#!/usr/bin/env python3
"""Snapshot all unprivileged host and Cassini interface counters."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


def _read_scalar(path: Path) -> int | str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    try:
        return int(value, 0)
    except ValueError:
        return value


def _read_tree(root: Path) -> dict[str, int | str]:
    values: dict[str, int | str] = {}
    if not root.is_dir():
        return values
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        value = _read_scalar(path)
        if value is not None:
            values[path.relative_to(root).as_posix()] = value
    return values


def _run_json(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        return {"available": False, "error": str(exc)}
    result: dict[str, Any] = {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
    }
    if completed.returncode == 0:
        try:
            result["value"] = json.loads(completed.stdout)
        except json.JSONDecodeError:
            result["value"] = completed.stdout.strip()
    else:
        result["error"] = completed.stderr.strip()
    return result


def _ethtool_stats(interface: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["ethtool", "-S", interface], check=False, capture_output=True, text=True
        )
    except OSError as exc:
        return {"available": False, "error": str(exc), "values": {}}
    values: dict[str, int] = {}
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            name, separator, raw = line.strip().partition(":")
            if separator and raw.strip().isdigit():
                values[name] = int(raw.strip())
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "error": completed.stderr.strip(),
        "values": values,
    }


def snapshot(interface: str, tag: str, net_class_root: Path) -> dict[str, Any]:
    interface_root = net_class_root / interface
    clock_id = getattr(time, "CLOCK_MONOTONIC_RAW", time.CLOCK_MONOTONIC)
    return {
        "schema": "simllm-merlin-interface-counter-snapshot-v1",
        "evidence_class": os.environ.get("TRAF77_EVIDENCE_CLASS", "hardware-capture"),
        "tag": tag,
        "node": socket.gethostname(),
        "interface": interface,
        "clock_monotonic_raw_ns": time.clock_gettime_ns(clock_id),
        "clock_realtime_ns": time.time_ns(),
        "sources": {
            "sysfs_statistics": _read_tree(interface_root / "statistics"),
            "sysfs_queues": _read_tree(interface_root / "queues"),
            "ethtool_statistics": _ethtool_stats(interface),
            "traffic_control_qdisc": _run_json(
                ["tc", "-s", "-j", "qdisc", "show", "dev", interface]
            ),
            "ip_link_statistics": _run_json(
                ["ip", "-s", "-j", "link", "show", "dev", interface]
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--net-class-root", type=Path, required=True)
    parser.add_argument("--interfaces", nargs="+", required=True)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / f"{socket.gethostname()}.jsonl"
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        for interface in args.interfaces:
            json.dump(
                snapshot(interface, args.tag, args.net_class_root),
                stream,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
