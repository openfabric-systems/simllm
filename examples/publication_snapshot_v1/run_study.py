"""Qualify the frozen reader change with immutable source-paired receipts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from examples.publication_snapshot_v1.checks import Evidence, GuardFailure, admit, compare
from examples.publication_snapshot_v1.common import (
    FREEZE,
    HERE,
    ROOT,
    git,
    packages,
    read,
    receipts,
    sha,
    study_origins,
    study_sources,
    write,
)


def launch(arm, root, args, frozen, monitors):
    path = args.output / arm
    path.mkdir()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(root), str(ROOT))),
               PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="")
    command = [sys.executable, str(HERE / "worker.py"), "--repository", str(root), "--output", str(path),
               "--expectations", str(HERE / "expectations.json"), "--arm", arm]
    started = time.monotonic()
    monitor = {"pid": None, "exit_code": None, "rss_kib": 0, "wall_seconds": 0, "failure": None}
    monitors[arm] = monitor
    with (path / "process.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        monitor["pid"] = process.pid
        try:
            while process.poll() is None:
                try:
                    status = Path(f"/proc/{process.pid}/status").read_text().splitlines()
                except FileNotFoundError:
                    status = []
                rss = next((int(row.split()[1]) for row in status if row.startswith("VmRSS:")), 0)
                monitor["rss_kib"] = max(monitor["rss_kib"], rss)
                if rss >= frozen["limits"]["rss_cap_kib"] or time.monotonic() - started >= frozen["limits"]["wall_seconds"]:
                    raise GuardFailure("frozen worker resource limit reached")
                try:
                    process.wait(timeout=frozen["limits"]["sample_seconds"])
                except subprocess.TimeoutExpired:
                    pass
        except BaseException as error:
            monitor["failure"] = str(error)
            process.kill()
            process.wait()
            raise
        finally:
            monitor["exit_code"] = process.returncode
            monitor["wall_seconds"] = time.monotonic() - started
            write(path / "process.json", monitor)
    if process.returncode:
        raise GuardFailure("worker failed: " + arm)


def capture(arm, root, args, frozen, monitors, raw, root_receipts):
    """Take the first receipt even when launch raises after a fatal exit."""
    try:
        launch(arm, root, args, frozen, monitors)
    finally:
        path = args.output / arm
        if path.exists():
            raw[arm] = receipts(path)
            file = args.output / (arm + "-raw-receipts.json")
            write(file, raw[arm])
            root_receipts[file.name] = {"sha256": sha(file), "bytes": file.stat().st_size}


def admit_disk(path, data, frozen, monitor, first, evidence, label):
    names = ["worker.json", "process.log", "process.json"]
    for spec in frozen["cases"]:
        names.extend(spec["id"] + suffix for suffix in ("-before.json", ".json", ".pstats"))
    names.extend(spec["id"] + "-job.json" for spec in frozen["jobs"])
    controls = frozen["common_corruptions"] + (frozen["new_helper_corruptions"] if data["arm"] == "after" else [])
    names.extend(name + suffix for name in controls for suffix in ("-before.json", ".json"))
    evidence.equal(label + ":raw-domain", sorted(first), sorted(names))
    evidence.equal(label + ":raw-bytes", receipts(path), first)
    evidence.equal(label + ":process", read(path / "process.json"), monitor)
    evidence.equal(label + ":process-exit", monitor["exit_code"], 0)
    evidence.equal(label + ":process-failure", monitor["failure"], None)
    evidence.check(label + ":process-time", 0 < monitor["wall_seconds"] < frozen["limits"]["wall_seconds"])
    evidence.check(label + ":process-rss", 0 < monitor["rss_kib"] < frozen["limits"]["rss_cap_kib"])
    evidence.equal(label + ":interpreter", data["interpreter"], sys.executable)
    for row in data["cases"]:
        evidence.equal(label + ":case-disk:" + row["id"], read(path / (row["id"] + ".json")), row)
        before = {key: value for key, value in row.items() if key not in ("profiles", "profile_hook_restored", "after")}
        evidence.equal(label + ":pre-profile-disk:" + row["id"], read(path / (row["id"] + "-before.json")), before)
    for row in data["jobs"]:
        evidence.equal(label + ":job-disk:" + row["id"], read(path / (row["id"] + "-job.json")), row)
    for row in [*data["mutations"], *data["helpers"]]:
        evidence.equal(label + ":mutation-disk:" + row["name"], read(path / (row["name"] + ".json")), row)
        before = {key: row[key] for key in ("name", "delta", "fixture", "pre_trigger", "helper_return")}
        evidence.equal(label + ":pre-trigger-disk:" + row["name"], read(path / (row["name"] + "-before.json")), before)


def corruptions(data, frozen, roots, args, sources, monitors, evidence):
    result = []
    for name in frozen["control_corruptions"]:
        changed = deepcopy(data["after"])
        if name == "disk-only-bytes":
            with TemporaryDirectory(dir=args.output) as directory:
                path = Path(directory)
                write(path / "row.json", {"value": 1})
                initial = receipts(path)
                write(path / "row.json", {"value": 2})
                try:
                    Evidence([]).equal("disk-only-bytes", receipts(path), initial)
                except GuardFailure as error:
                    reason = str(error)
                else:
                    raise GuardFailure("disk mutation was accepted")
        else:
            if name == "source-receipt":
                changed["sources_before"][frozen["allowed_package_change"]] = "0" * 64
            elif name == "profile-calls":
                row = next(row for row in changed["cases"][0]["profiles"] if row["function"][2] == "value_snapshot")
                row["calls"] += 1
            elif name == "count-law":
                changed["cases"][0]["expected"]["state_encoded_visits"] += 1
            elif name == "job-result":
                changed["jobs"][0]["completed_at_ps"] += 1
            elif name == "missing-mutation":
                changed["mutations"].pop()
            else:
                raise GuardFailure("unknown evidence mutation")
            try:
                admit(changed, "after", roots["after"], args.output / "after", frozen,
                      sources["after"], monitors["after"], Evidence([]))
            except GuardFailure as error:
                reason = str(error)
            else:
                raise GuardFailure("evidence mutation was accepted: " + name)
        evidence.check("corruption:" + name, bool(reason))
        result.append({"name": name, "rejected_by": reason})
    evidence.finish("corruption-controls")
    return result


def execute(args):
    args.output = args.output.resolve()
    roots = {"before": args.before_repository.resolve(), "after": ROOT}
    if any(root == args.output or root in args.output.parents for root in roots.values()):
        raise ValueError("raw output must be outside the source repositories")
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((HERE / "expectations.json").read_bytes())
    evidence = Evidence(frozen["required_stages"])
    source = git(ROOT, "rev-parse", "HEAD")
    data, sources, monitors, raw, root_receipts, counts, controls = {}, {}, {}, {}, {}, {}, []
    failure = None
    try:
        git(ROOT, "merge-base", "--is-ancestor", FREEZE, "HEAD")
        for name in ("expectations.md", "expectations.json"):
            committed = subprocess.check_output(["git", "show", FREEZE + ":" + (HERE.relative_to(ROOT) / name).as_posix()], cwd=ROOT)
            evidence.equal("freeze:" + name, (HERE / name).read_bytes().hex(), committed.hex())
        evidence.equal("protocol:before", git(roots["before"], "rev-parse", "HEAD"), frozen["before_commit"])
        for arm, root in roots.items():
            git(root, "diff", "--quiet", "HEAD", "--")
            evidence.equal(arm + ":no-untracked-package", git(root, "ls-files", "--others", "--exclude-standard", "simllm"), "")
            sources[arm] = packages(root)
        evidence.equal("protocol:package-domain", sorted(sources["before"]), sorted(sources["after"]))
        changed = [name for name in sources["before"] if sources["before"][name] != sources["after"][name]]
        evidence.equal("protocol:package-changes", changed, [frozen["allowed_package_change"]])
        evidence.equal("protocol:before-sink", sources["before"][frozen["allowed_package_change"]], frozen["before_sink_sha256"])
        evidence.equal("protocol:snapshot-source", sources["after"]["simllm/core/value_snapshot.py"], frozen["snapshot_source_sha256"])
        for file in HERE.glob("*.py"):
            git(ROOT, "ls-files", "--error-unmatch", file.relative_to(ROOT).as_posix())
        study = {file.name: sha(file) for file in HERE.iterdir() if file.suffix in (".py", ".md", ".json")}
        study_lock = study_sources(ROOT)
        origins = study_origins()
        for name, path in origins.items():
            relative = name.replace(".", "/") + ".py"
            evidence.equal("protocol:helper-origin:" + name, path, str(ROOT / relative))
            evidence.check("protocol:tracked-helper:" + name, relative in study_lock)
            evidence.equal("protocol:helper-bytes:" + name, sha(path), study_lock[relative])
        evidence.equal("protocol:evidence-function", str(Path(Evidence.equal.__code__.co_filename).resolve()),
                       str(ROOT / "examples/pd_session_target_scale_v1/checks.py"))
        write(args.output / "source-manifest.json", {"source_commit": source, "study": study, "packages": sources,
                                                    "study_sources": study_lock, "study_origins": origins})
        write(args.output / "physical-bounds.json", {
            "matrix_bytes_per_rank_per_layer": 16384,
            "step_floor_per_layer_ps": frozen["step_floor_per_layer_ps"],
            "step_ceiling_per_layer_ps": frozen["step_ceiling_per_layer_ps"],
            "publication_counts": {spec["id"]: {"raw": 21 + 3 * spec["history"] * (2 + spec["width"]),
                                               "encoded": 94 + 3 * spec["history"] * (10 + 4 * spec["width"])}
                                   for spec in frozen["cases"]}})
        root_receipts = receipts(args.output)
        evidence.finish("protocol")
        for arm, root in roots.items():
            print("Starting " + arm, flush=True)
            capture(arm, root, args, frozen, monitors, raw, root_receipts)
            evidence.finish(arm + "-capture")
            current = read(args.output / arm / "worker.json")
            admit_disk(args.output / arm, current, frozen, monitors[arm], raw[arm], evidence, arm)
            counts[arm] = admit(current, arm, root, args.output / arm, frozen, sources[arm], monitors[arm], evidence)
            data[arm] = current
            evidence.finish(arm + "-admission")
            print("Qualified " + arm, flush=True)
        compare(data["before"], data["after"], counts, frozen, evidence)
        controls = corruptions(data, frozen, roots, args, sources, monitors, evidence)
        evidence.equal("complete:fresh-workers", len({row["pid"] for row in monitors.values()}), 2)
        evidence.equal("complete:oracles", len(evidence.oracles), frozen["exact_oracles"])
        evidence.equal("complete:relation-domain", [row["name"] for row in evidence.relations], ["paired:" + spec["id"] for spec in frozen["cases"]])
        for arm, root in roots.items():
            evidence.equal(arm + ":final-raw", receipts(args.output / arm), raw[arm])
            evidence.equal(arm + ":final-source", packages(root), sources[arm])
            git(root, "diff", "--quiet", "HEAD", "--")
            evidence.equal(arm + ":final-no-untracked-package", git(root, "ls-files", "--others", "--exclude-standard", "simllm"), "")
        evidence.equal("complete:study", {file.name: sha(file) for file in HERE.iterdir() if file.suffix in (".py", ".md", ".json")}, study)
        evidence.equal("complete:study-sources", study_sources(ROOT), study_lock)
        evidence.equal("complete:study-origins", study_origins(), origins)
        evidence.equal("complete:head", git(ROOT, "rev-parse", "HEAD"), source)
        evidence.equal("complete:before-head", git(roots["before"], "rev-parse", "HEAD"), frozen["before_commit"])
        evidence.equal("complete:root-domain", sorted(file.name for file in args.output.iterdir()), sorted([*root_receipts, "before", "after", "sink-work"]))
        evidence.check("complete:empty-workdir", not list((args.output / "sink-work").iterdir()))
        for name, first in root_receipts.items():
            path = args.output / name
            evidence.equal("complete:root:" + name, {"sha256": sha(path), "bytes": path.stat().st_size}, first)
        evidence.finish("final-receipts")
        evidence.equal("complete:stages", sorted(evidence.finished_stages), sorted(frozen["required_stages"]))
    except Exception as error:  # noqa: BLE001
        failure = str(error)
        (args.output / "exception.txt").write_text(traceback.format_exc())
    summary = {"schema": "simllm-publication-snapshot-summary-v1", "task": "CORE-68", "source_commit": source,
               "expectations_commit": FREEZE, "verdict": "PASS" if failure is None else "VOID", "failure": failure,
               "behavioral_score": 1 if failure is None else None, "behavioral_instances": len(evidence.relations),
               "behavioral_families": len({row["family"] for row in evidence.relations}), "exact_oracles": len(evidence.oracles),
               "unscored_guards": len(evidence.guards), "fatal_guards_violated": [row for row in evidence.guards if not row["passed"]],
               "admitted_sources": list(data), "stages": sorted(evidence.finished_stages), "monitors": monitors,
               "visitor_counts": counts, "corruption_controls": controls, "initial_raw_receipts": raw,
               "root_receipts": root_receipts, "retained_raw_receipts": {arm: receipts(args.output / arm) for arm in roots if (args.output / arm).exists()}}
    write(args.output / "checks.json", {"guards": evidence.guards, "oracles": evidence.oracles, "relations": evidence.relations})
    write(args.output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("verdict", "failure", "admitted_sources")}), flush=True)
    return 0 if failure is None else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return execute(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
