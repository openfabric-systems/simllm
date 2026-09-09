"""Qualify a frozen call-local lookup without fitting host or device time."""

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

from examples.dependency_index_v1.checks import admit, compare, formula, read
from examples.dependency_index_v1.worker import package_sources, sha, write
from examples.pd_session_target_scale_v1.checks import Evidence, GuardFailure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREEZE_COMMIT = "581d41531c42759e1fd1f9b7af3a0f45844f51a2"


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def receipts(path):
    return {item.relative_to(path).as_posix(): {"sha256": sha(item.read_bytes()), "bytes": item.stat().st_size}
            for item in sorted(path.rglob("*")) if item.is_file()}


def reread(path, first, evidence, label):
    evidence.equal(label + ":raw-domain-and-bytes", receipts(path), first)


def launch(arm, root, args, frozen, monitors):
    output = args.output_root / arm
    output.mkdir()
    command = [sys.executable, str(HERE / "worker.py"), "--arm", arm, "--repository-root", str(root),
               "--expectations", str(HERE / "expectations.json"), "--output-root", str(output)]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="")
    monitor = {"pid": None, "exit_code": None, "rss_kib": 0, "stopping_reason": None, "wall_seconds": 0}
    monitors[arm] = monitor
    limits = frozen["limits"]
    started = time.monotonic()
    with (output / "process.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        monitor["pid"] = process.pid
        try:
            while process.poll() is None:
                status = Path(f"/proc/{process.pid}/status")
                try:
                    lines = status.read_text().splitlines()
                except FileNotFoundError:
                    lines = []
                rss = next((int(line.split()[1]) for line in lines if line.startswith("VmRSS:")), 0)
                monitor["rss_kib"] = max(rss, monitor["rss_kib"])
                if rss >= limits["rss_cap_kib"] or time.monotonic() - started >= limits["process_timeout_seconds"]:
                    raise GuardFailure("frozen process resource limit")
                try:
                    process.wait(timeout=limits["rss_sample_seconds"])
                except subprocess.TimeoutExpired:
                    pass
        except BaseException as error:
            monitor["stopping_reason"] = str(error)
            process.kill()
            process.wait()
            raise
        finally:
            monitor["exit_code"] = process.returncode
            monitor["wall_seconds"] = time.monotonic() - started
            write(output / "process.json", monitor)
    if process.returncode:
        raise GuardFailure("source worker failed: " + arm)


def capture_worker(arm, root, args, frozen, monitors, raw, root_receipts):
    """Retain first receipts on every process exit, including fatal exits."""
    try:
        launch(arm, root, args, frozen, monitors)
    finally:
        path = args.output_root / arm
        if path.exists():
            raw[arm] = receipts(path)
            receipt_path = args.output_root / (arm + "-raw-receipts.json")
            write(receipt_path, raw[arm])
            root_receipts[receipt_path.name] = {"sha256": sha(receipt_path.read_bytes()), "bytes": receipt_path.stat().st_size}


def corruption_controls(all_data, frozen, roots, sources, worker_sha, args, monitors, evidence):
    outcomes = []
    for name in frozen["corruption_controls"]:
        if name == "disk-only-bytes":
            with TemporaryDirectory(dir=args.output_root, prefix="corruption-") as directory:
                path = Path(directory)
                write(path / "original.json", {"value": 1})
                initial = receipts(path)
                write(path / "original.json", {"value": 2})
                try:
                    reread(path, initial, Evidence([]), name)
                except GuardFailure as error:
                    reason = str(error)
                else:
                    raise GuardFailure("changed raw bytes were accepted")
        else:
            data = deepcopy(all_data["after"])
            if name == "source-receipt":
                data["package_sources_before"][frozen["allowed_package_change"]] = "0" * 64
                data["package_sources_after"] = deepcopy(data["package_sources_before"])
            elif name == "profile-count":
                row = next(row for row in data["graphs"][0]["profiles"] if row["function"][2] == "_goal_edge")
                row["calls"] += 1
            elif name == "positive-projection":
                data["graphs"][0]["before"]["projection"]["serialized_edges"].pop()
                data["graphs"][0]["after"] = deepcopy(data["graphs"][0]["before"])
            elif name == "sink-result":
                data["sinks"][0]["steps"][0]["result"]["step_latency_ps"] += 1
            else:
                raise GuardFailure("unknown semantic control")
            try:
                admit(data, frozen, "after", roots["after"], args.output_root / "after", sources["after"],
                      worker_sha, monitors["after"], Evidence([]))
            except GuardFailure as error:
                reason = str(error)
            else:
                raise GuardFailure("semantic corruption was accepted: " + name)
        evidence.check("corruption:" + name, bool(reason))
        outcomes.append({"name": name, "rejected_by": reason})
    evidence.finish("corruption-controls")
    return outcomes


def execute(args):
    frozen = json.loads((HERE / "expectations.json").read_bytes())
    roots = {"before": args.before_repository.resolve(), "after": ROOT}
    args.output_root = args.output_root.resolve()
    if any(args.output_root == root or root in args.output_root.parents for root in roots.values()):
        raise ValueError("raw evidence must be outside both source trees")
    args.output_root.mkdir(parents=True, exist_ok=False)
    evidence = Evidence(frozen["required_stages"])
    data, sources, monitors, raw, root_receipts, calls, controls = {}, {}, {}, {}, {}, {}, []
    source = git(ROOT, "rev-parse", "HEAD").decode().strip()
    worker_sha = sha((HERE / "worker.py").read_bytes())
    failure = None
    try:
        git(ROOT, "merge-base", "--is-ancestor", FREEZE_COMMIT, "HEAD")
        for name in ("expectations.md", "expectations.json"):
            evidence.equal("freeze:" + name, sha((HERE / name).read_bytes()),
                           sha(git(ROOT, "show", FREEZE_COMMIT + ":" + (HERE.relative_to(ROOT) / name).as_posix())))
        for root in roots.values():
            git(root, "diff", "--quiet", "HEAD", "--")
        evidence.equal("protocol:baseline-commit", git(roots["before"], "rev-parse", "HEAD").decode().strip(), frozen["as_of_commit"])
        for item in HERE.glob("*.py"):
            git(ROOT, "ls-files", "--error-unmatch", item.relative_to(ROOT).as_posix())
        study_sources = {item.name: sha(item.read_bytes()) for item in HERE.iterdir() if item.suffix in (".py", ".md", ".json")}
        sources = {arm: package_sources(root) for arm, root in roots.items()}
        evidence.equal("protocol:package-domain", sorted(sources["before"]), sorted(sources["after"]))
        changed = [name for name in sources["before"] if sources["before"][name] != sources["after"][name]]
        evidence.equal("protocol:only-package-change", changed, [frozen["allowed_package_change"]])
        evidence.equal("protocol:baseline-verifier", sources["before"][frozen["allowed_package_change"]], frozen["before_verifier_sha256"])
        for arm, root in roots.items():
            evidence.equal("protocol:granite-helper:" + arm, sha((root / "examples/pd_session_v1/run_study.py").read_bytes()), frozen["granite_helper_sha256"])
        write(args.output_root / "source-manifest.json", {"source_commit": source, "study": study_sources, "packages": sources})
        write(args.output_root / "physical-bounds.json", {spec["id"]: formula(spec["depth"], spec["width"]) for spec in frozen["graph_cases"]})
        root_receipts = receipts(args.output_root)
        evidence.finish("protocol")
        for arm, root in roots.items():
            print("Starting " + arm, flush=True)
            capture_worker(arm, root, args, frozen, monitors, raw, root_receipts)
            path = args.output_root / arm
            evidence.finish(arm + "-capture")
            current = read(path / "worker.json")
            reread(path, raw[arm], evidence, arm + ":initial")
            evidence.equal(arm + ":process-receipt", read(path / "process.json"), monitors[arm])
            evidence.equal(arm + ":interpreter", current["interpreter"], sys.executable)
            evidence.equal(arm + ":frozen-input", current["expectations_sha256"], sha((HERE / "expectations.json").read_bytes()))
            for row in current["graphs"]:
                evidence.equal(arm + ":graph-disk:" + row["id"], read(path / (row["id"] + "-graph.json")), row)
            for row in current["sinks"]:
                evidence.equal(arm + ":sink-disk:" + row["id"], read(path / (row["id"] + "-sink.json")), row)
            calls[arm] = admit(current, frozen, arm, root, path, sources[arm], worker_sha, monitors[arm], evidence)
            data[arm] = current
            evidence.finish(arm + "-admission")
            print("Qualified " + arm, flush=True)
        compare(data["before"], data["after"], calls, frozen, evidence)
        controls = corruption_controls(data, frozen, roots, sources, worker_sha, args, monitors, evidence)
        evidence.equal("complete:fresh-processes", len({row["pid"] for row in monitors.values()}), 2)
        evidence.equal("complete:oracle-count", len(evidence.oracles), frozen["exact_oracle_count"])
        evidence.equal("complete:relation-domain", sorted(row["name"] for row in evidence.relations),
                       sorted("pair:" + spec["id"] for spec in frozen["graph_cases"]))
        for arm, root in roots.items():
            reread(args.output_root / arm, raw[arm], evidence, arm + ":final")
            evidence.equal(arm + ":final-source", package_sources(root), sources[arm])
            git(root, "diff", "--quiet", "HEAD", "--")
        evidence.equal("complete:study-source", {item.name: sha(item.read_bytes()) for item in HERE.iterdir() if item.suffix in (".py", ".md", ".json")}, study_sources)
        evidence.equal("complete:head", git(ROOT, "rev-parse", "HEAD").decode().strip(), source)
        evidence.equal("complete:root-domain", sorted(path.name for path in args.output_root.iterdir() if path.is_file()), sorted(root_receipts))
        for name, receipt in root_receipts.items():
            path = args.output_root / name
            evidence.equal("complete:root-bytes:" + name, {"sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}, receipt)
        evidence.finish("final-receipts")
        evidence.equal("complete:stages", sorted(evidence.finished_stages), sorted(frozen["required_stages"]))
    except Exception as error:  # noqa: BLE001
        failure = str(error)
        (args.output_root / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
    result = {"schema": "simllm-dependency-index-summary-v1", "task": "CORE-52", "verdict": "PASS" if failure is None else "VOID",
              "expectations_commit": FREEZE_COMMIT, "source_commit": source, "failure": failure,
              "behavioral_score": 1 if failure is None else None,
              "fatal_guards_violated": [row for row in evidence.guards if not row["passed"]],
              "unscored_guards": len(evidence.guards), "exact_oracles": len(evidence.oracles),
              "behavioral_instances": len(evidence.relations), "behavioral_families": len({row["family"] for row in evidence.relations}),
              "stages": sorted(evidence.finished_stages), "admitted_sources": list(data), "monitors": monitors,
              "conversion_calls": calls, "corruption_controls": controls, "root_receipts": root_receipts,
              "initial_raw_receipts": raw, "retained_raw_receipts": {arm: receipts(args.output_root / arm) for arm in roots if (args.output_root / arm).exists()}}
    write(args.output_root / "checks.json", {"guards": evidence.guards, "oracles": evidence.oracles, "relations": evidence.relations})
    write(args.output_root / "summary.json", result)
    print(json.dumps({key: result[key] for key in ("verdict", "failure", "admitted_sources", "behavioral_score")}), flush=True)
    return 0 if failure is None else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-repository", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return execute(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
