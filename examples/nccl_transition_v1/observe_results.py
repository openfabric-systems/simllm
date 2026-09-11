"""Qualify direct callback records and join choices by payload and placement."""

import csv
import hashlib
import json
from collections import defaultdict

from run_campaign import make_plan

CHOICES = ("algo", "proto", "#channels", "#warps", "kernelVariant")


def callback_choices(events, widths, sizes, legacy=False):
    grouped = defaultdict(list)
    for event in events:
        if event["kind"] != "collective":
            continue
        if event["function"] != "AllReduce" or event["datatype"] != "ncclFloat32":
            raise ValueError("Observer: unexpected collective or datatype")
        if event["width"] not in widths or event["count"] * 4 not in sizes:
            raise ValueError("Observer: callback shape outside the frozen grid")
        if event["rank"] not in range(event["width"]):
            raise ValueError("Observer: rank outside the communicator")
        if (
            not event["algo"]
            or not event["proto"]
            or event["#channels"] <= 0
            or event["#warps"] <= 0
        ):
            raise ValueError("Observer: selected choice fields are empty")
        key = (event["width"], event["count"] * 4, event["in_place"])
        grouped[key].append(event)
    placements = [False] if legacy else [False, True]
    expected = {(w, b, p) for w in widths for b in sizes for p in placements}
    if set(grouped) != expected:
        raise ValueError("Observer: missing or unexpected payload/placement cells")
    rows, conflicts = [], []
    for (width, size, in_place), calls in sorted(grouped.items()):
        if {e["rank"] for e in calls} != set(range(width)):
            raise ValueError("Observer: a payload is missing a participant rank")
        tuples = {tuple(e[k] for k in CHOICES) for e in calls}
        if len(tuples) != 1:
            conflicts.append(
                {
                    "width": width,
                    "bytes": size,
                    "in_place": in_place,
                    "choices": [dict(zip(CHOICES, t)) for t in sorted(tuples)],
                }
            )
            continue
        rows.append(
            {
                "width": width,
                "bytes": size,
                "in_place": in_place,
                "callbacks": len(calls),
                "ranks": width,
                **dict(zip(CHOICES, next(iter(tuples)))),
            }
        )
    return rows, conflicts


def gpu_mesh(path):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 5 and fields[0] in ("GPU0", "GPU1", "GPU2", "GPU3"):
            rows.append(fields[:5])
    if len(rows) != 4:
        raise ValueError("Observer: cannot identify the four-GPU NVLink matrix")
    return rows


def audit_observer(directory, timing_directory, config):
    manifest = json.loads((directory / "observer/observer.json").read_text())
    arch = manifest["architecture"]
    failures, conflicts, choices = [], [], []
    expected = {
        (r["lane"], r["arm"], r["width"]) for r in make_plan(config) if r["lane"] == "diagnostic"
    } | {("legacy", "auto", 0)}
    actual = [(r["lane"], r["arm"], r["width"]) for r in manifest["runs"]]
    if set(actual) != expected or len(set(actual)) != len(actual) or not manifest.get("complete"):
        failures.append("Observer process inventory incomplete or duplicated")
    if not (directory / "completed.txt").exists():
        failures.append("Observer allocation completion marker absent")
    for phase in ("before", "after"):
        p = directory / f"processes_{phase}.csv"
        if not p.exists() or any(line.strip() for line in p.read_text().splitlines()[1:]):
            failures.append("Observer foreign GPU process guard failed: " + phase)
    if (directory / "library.sha256").read_text().split()[0] != (
        timing_directory / "library.sha256"
    ).read_text().split()[0]:
        failures.append("Observer uses a different NCCL library")
    if (directory / "nccl_tests_commit.txt").read_text().strip() != config["nccl_tests_commit"]:
        failures.append("Observer upstream benchmark pin differs")
    if gpu_mesh(directory / "topology.txt") != gpu_mesh(timing_directory / "topology.txt"):
        failures.append("Observer NVLink participant mesh differs")
    identities = list(csv.DictReader((directory / "gpu_identity.csv").open()))
    timing_ids = list(csv.DictReader((timing_directory / "gpu_identity.csv").open()))
    if len(identities) != 4 or any(arch.upper() not in str(row).upper() for row in identities):
        failures.append("Observer GPU architecture differs")
    callbacks = 0
    for run in manifest["runs"]:
        try:
            if run["returncode"] != 0:
                raise ValueError("Observer process failed")
            payload = (directory / "observer" / run["events"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != run["events_sha256"]:
                raise ValueError("Observer callback checksum differs")
            events = [json.loads(line) for line in payload.splitlines()]
            count = sum(e["kind"] == "collective" for e in events)
            if count != run["collective_callbacks"]:
                raise ValueError("Observer callback inventory differs")
            callbacks += count
            legacy = run["lane"] == "legacy"
            grid = config[run["grid"]]
            sizes = range(grid["start_bytes"], grid["end_bytes"] + 1, grid["step_bytes"])
            rows, issues = callback_choices(
                events, config["widths"] if legacy else [run["width"]], sizes, legacy
            )
            result = json.loads((directory / "observer" / run["output"]).read_text())
            if result["nccl_version"] != 23102:
                raise ValueError("Observer runtime NCCL version differs")
            if legacy:
                if any(r["allreduce_mismatching_ranks"] for r in result["collectives"]):
                    raise ValueError("Observer inherited correctness probe failed")
            else:
                if result["out_of_bounds"]["count"] or any(e.strip() for e in result["errors"]):
                    raise ValueError("Observer full validation failed")
                if any(
                    r[p]["nwrong"] for r in result["results"] for p in ("out_of_place", "in_place")
                ):
                    raise ValueError("Observer full output validation failed")
            for row in rows:
                choices.append(
                    {
                        "architecture": arch,
                        "lane": "legacy" if legacy else "diagnostic",
                        "arm": run["arm"],
                        "repeat": 0,
                        **row,
                    }
                )
            conflicts.extend({"arm": run["arm"], "lane": run["lane"], **row} for row in issues)
        except (ValueError, KeyError, OSError) as error:
            failures.append(run["events"] + ": " + str(error))
    audit = {
        "architecture": arch,
        "job_id": directory.name,
        "status": "void" if failures else "valid",
        "fatal_findings": failures,
        "ambiguous_cells": conflicts,
        "collective_callbacks": callbacks,
        "qualified_cells": len(choices),
        "implementation": manifest["implementation"],
        "amendment": manifest["amendment"],
        "plugin_sha256": manifest["plugin_sha256"],
        "same_gpu_identities_as_timing": identities == timing_ids,
        "same_nvlink_mesh_as_timing": gpu_mesh(directory / "topology.txt")
        == gpu_mesh(timing_directory / "topology.txt"),
    }
    if failures:
        choices = []
    return choices, audit
