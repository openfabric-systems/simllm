"""Run the frozen source pair with first-exit receipts and fatal admission."""

import argparse
import json
import os
import runpy
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from examples.publication_snapshot_v1.common import (
    git,
    packages,
    read,
    receipts,
    sha,
    study_sources,
    write,
)
from examples.snapshot_dispatch_v1.checks import HERE, ROOT, Evidence, GuardFailure, admit, compare
from examples.snapshot_dispatch_v1.common import runtime_identity

FREEZE = "f08da959e9aca1be75e56a21e4cc5db0fa8b15f9"


def launch(root, output, arm, mode, frozen, monitors):
    output.mkdir()
    monitor = {"pid": None, "exit_code": None, "rss_kib": 0, "wall_seconds": 0, "failure": None}
    monitors[output.name] = monitor
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(root), str(ROOT))),
               PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="")
    command = [sys.executable, str(HERE / "worker.py"), "--repository", str(root), "--output", str(output),
               "--arm", arm, "--mode", mode]
    process, started = None, time.monotonic()
    try:
        with (output / "process.log").open("wb") as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            monitor["pid"] = process.pid
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
            if process.returncode:
                raise GuardFailure("worker failed: " + output.name)
    except BaseException as error:
        monitor["failure"] = str(error)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        monitor["exit_code"] = process.returncode if process is not None else None
        monitor["wall_seconds"] = time.monotonic() - started
        write(output / "process.json", monitor)


def capture(root, path, arm, mode, frozen, monitors, raw, root_receipts):
    try:
        launch(root, path, arm, mode, frozen, monitors)
    finally:
        if path.exists():
            raw[path.name] = receipts(path)
            receipt = path.parent / (path.name + "-raw-receipts.json")
            write(receipt, raw[path.name])
            root_receipts[receipt.name] = {"sha256": sha(receipt), "bytes": receipt.stat().st_size}


def admit_process(path, data, root, arm, mode, frozen, old, source, monitor, first, study, runtime, evidence):
    key = path.name
    evidence.equal(key + ":first-receipt", receipts(path), first)
    evidence.equal(key + ":monitor", read(path / "process.json"), monitor)
    evidence.equal(key + ":successful-process", [monitor["exit_code"], monitor["failure"]], [0, None])
    evidence.check(key + ":resources", 0 < monitor["rss_kib"] < frozen["limits"]["rss_cap_kib"]
                   and 0 < monitor["wall_seconds"] < frozen["limits"]["wall_seconds"])
    for name, expected in (("pid", monitor["pid"]), ("arm", arm), ("mode", mode), ("interpreter", sys.executable),
                           ("repository", str(root)), ("worker_path", str(HERE / "worker.py")),
                           ("worker_sha256", sha(HERE / "worker.py")), ("sources_before", source), ("sources_after", source)):
        evidence.equal(key + ":identity:" + name, data[name], expected)
    evidence.equal(key + ":runtime-before", data["runtime_before"], runtime)
    evidence.equal(key + ":runtime-after", data["runtime_after"], runtime)
    required = {"simllm", "simllm.core.value_snapshot", "simllm.backends.step_sink", "simllm.core.engine_steps"}
    evidence.check(key + ":origin-domain", required <= data["origins"].keys())
    for name, path_string in data["origins"].items():
        base = root / name.replace(".", "/")
        expected = base / "__init__.py" if (base / "__init__.py").is_file() else base.with_suffix(".py")
        evidence.equal(key + ":origin:" + name, path_string, str(expected))
        evidence.equal(key + ":origin-bytes:" + name, sha(expected), source[expected.relative_to(root).as_posix()])
    evidence.check(key + ":helper-domain", {"examples.snapshot_dispatch_v1.semantics", "examples.publication_snapshot_v1.worker"}
                   <= data["study_origins"].keys())
    for name, path_string in data["study_origins"].items():
        relative = name.replace(".", "/") + ".py"
        expected = root / relative if (root / relative).is_file() else ROOT / relative
        evidence.equal(key + ":helper-origin:" + name, path_string, str(expected))
        evidence.equal(key + ":helper-bytes:" + name, sha(expected), study[relative])
    names = ["worker.json", "process.json", "process.log"]
    if mode == "main":
        for group, suffix in (("primitives", "primitive"), ("readers", "reader")):
            for spec in frozen["cases"]:
                names.extend(spec["id"] + "-" + suffix + ending for ending in ("-before.json", ".json", ".pstats"))
            for row in data[group]:
                evidence.equal(key + ":disk:" + suffix + ":" + row["id"], read(path / (row["id"] + "-" + suffix + ".json")), row)
                evidence.equal(key + ":before-disk:" + suffix + ":" + row["id"], read(path / (row["id"] + "-" + suffix + "-before.json")), row["before"])
        for spec in old["jobs"]:
            names.extend(spec["id"] + "-job" + ending for ending in (".json", ".pstats"))
        for row in data["jobs"]:
            evidence.equal(key + ":job-disk:" + row["job"]["id"], read(path / (row["job"]["id"] + "-job.json")), row["job"])
        names.extend(name + "-semantic.json" for name in frozen["semantics"])
        for row in data["semantics"]:
            evidence.equal(key + ":semantic-disk:" + row["name"], read(path / (row["name"] + "-semantic.json")), row)
        names.extend(name + ending for name in old["common_corruptions"] + old["new_helper_corruptions"]
                     for ending in ("-before.json", ".json"))
        for row in data["mutations"] + data["helpers"]:
            evidence.equal(key + ":mutation-disk:" + row["name"], read(path / (row["name"] + ".json")), row)
            evidence.equal(key + ":mutation-before:" + row["name"], read(path / (row["name"] + "-before.json")),
                           {name: row[name] for name in ("name", "delta", "fixture", "pre_trigger", "helper_return")})
    else:
        expected = {"name": mode, "calls": [{"error": {"type": "AttributeError", "message": "'list' object has no attribute 'numerator'"}}],
                    "trace": ["register-list"] if mode == "fraction-during" else [], "identity_witnesses": []}
        evidence.equal(key + ":registered-fraction", data["semantics"], [expected])
    evidence.equal(key + ":raw-domain", sorted(first), sorted(names))


def corruptions(data, roots, output, frozen, old, sources, reference, evidence):
    result = []
    for name in frozen["control_corruptions"]:
        changed = deepcopy(data["after"])
        try:
            if name == "disk-only-bytes":
                with TemporaryDirectory(dir=output) as temporary:
                    path = Path(temporary)
                    write(path / "row.json", {"value": 1})
                    first = receipts(path)
                    write(path / "row.json", {"value": 2})
                    Evidence([]).equal("disk-only-bytes", receipts(path), first)
            else:
                if name == "source-receipt":
                    changed["sources_before"][frozen["allowed_package_change"]] = "0" * 64
                elif name == "profile-calls":
                    next(row for row in changed["primitives"][0]["profiles"] if row["function"][2] == "visit")["calls"] += 1
                elif name == "missing-semantic":
                    changed["semantics"].pop()
                elif name == "job-result":
                    changed["jobs"][0]["job"]["completed_at_ps"] += 1
                elif name == "missing-mutation":
                    changed["mutations"].pop()
                else:
                    raise ValueError("unknown evidence corruption")
                admit(changed, "after", roots["after"], output / "after", frozen, old, sources["after"], reference, Evidence([]))
        except GuardFailure as error:
            reason = str(error)
        else:
            raise GuardFailure("damaged evidence admitted: " + name)
        evidence.check("corruption:" + name, bool(reason))
        result.append({"name": name, "rejected_by": reason})
    evidence.finish("corruption-controls")
    return result


def execute(args):
    output = args.output.resolve()
    roots = {"before": args.before_repository.resolve(), "after": ROOT}
    if any(output == root or root in output.parents for root in roots.values()):
        raise ValueError("raw output must be outside the source repositories")
    output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((HERE / "expectations.json").read_bytes())
    old_path = HERE.parent / "publication_snapshot_v1/expectations.json"
    old = json.loads(old_path.read_bytes())
    evidence = Evidence(frozen["required_stages"])
    head = git(ROOT, "rev-parse", "HEAD")
    sources, monitors, raw, root_receipts, data, vectors, controls = {}, {}, {}, {}, {}, {}, []
    failure = None
    try:
        git(ROOT, "merge-base", "--is-ancestor", FREEZE, "HEAD")
        for name in ("expectations.md", "expectations.json"):
            committed = subprocess.check_output(["git", "show", FREEZE + ":" + (HERE.relative_to(ROOT) / name).as_posix()], cwd=ROOT)
            evidence.equal("freeze:" + name, (HERE / name).read_bytes().hex(), committed.hex())
        evidence.equal("protocol:before-head", git(roots["before"], "rev-parse", "HEAD"), frozen["before_commit"])
        evidence.equal("protocol:helper-input", sha(old_path), frozen["locked_helper_expectations_sha256"])
        for arm, root in roots.items():
            git(root, "diff", "--quiet", "HEAD", "--")
            evidence.equal(arm + ":untracked-package", git(root, "ls-files", "--others", "--exclude-standard", "simllm"), "")
            sources[arm] = packages(root)
        evidence.equal("protocol:source-domain", sorted(sources["before"]), sorted(sources["after"]))
        evidence.equal("protocol:source-change", [name for name in sources["before"] if sources["before"][name] != sources["after"][name]], [frozen["allowed_package_change"]])
        evidence.equal("protocol:old-reader", sources["before"][frozen["allowed_package_change"]], frozen["before_snapshot_sha256"])
        relative = frozen["allowed_package_change"]
        original = (roots["before"] / relative).read_text()
        updated = (ROOT / relative).read_text()
        marker = "        tag = (kind.__module__, kind.__qualname__)"
        evidence.equal("protocol:unchanged-fallback", updated[updated.index(marker):], original[original.index(marker):])
        evidence.check("protocol:unchanged-prefix", updated.startswith(original[:original.index(marker)]))
        runtime = runtime_identity()
        study = study_sources(ROOT)
        for file in HERE.glob("*.py"):
            evidence.check("protocol:tracked:" + file.name, file.relative_to(ROOT).as_posix() in study)
        for name, value in study_sources(roots["before"]).items():
            evidence.equal("protocol:locked-helper:" + name, study[name], value)
        reference = runpy.run_path(str(roots["before"] / relative))["value_snapshot"]
        write(output / "source-manifest.json", {"source_commit": head, "packages": sources, "study_sources": study, "runtime": runtime})
        write(output / "physical-bounds.json", {"step_floor_per_layer_ps": old["step_floor_per_layer_ps"],
              "step_ceiling_per_layer_ps": old["step_ceiling_per_layer_ps"], "cases": [dict(spec,
              primitive_visits=6*spec["history"]*spec["width"], total_visits=1+spec["history"]+6*spec["history"]*spec["width"])
              for spec in frozen["cases"]]})
        root_receipts = receipts(output)
        evidence.finish("protocol")
        for arm, root in roots.items():
            print("Starting " + arm, flush=True)
            capture(root, output / arm, arm, "main", frozen, monitors, raw, root_receipts)
            evidence.finish(arm + "-capture")
            current = read(output / arm / "worker.json")
            admit_process(output / arm, current, root, arm, "main", frozen, old, sources[arm], monitors[arm], raw[arm], study, runtime, evidence)
            vectors[arm] = admit(current, arm, root, output / arm, frozen, old, sources[arm], reference, evidence)
            data[arm] = current
            evidence.finish(arm + "-admission")
            print("Qualified " + arm, flush=True)
        for arm, root in roots.items():
            for mode in frozen["isolated_semantics"]:
                path = output / (arm + "-" + mode)
                capture(root, path, arm, mode, frozen, monitors, raw, root_receipts)
                current = read(path / "worker.json")
                admit_process(path, current, root, arm, mode, frozen, old, sources[arm], monitors[path.name], raw[path.name], study, runtime, evidence)
        evidence.finish("isolated-semantics")
        compare(data, roots, frozen, evidence)
        controls = corruptions(data, roots, output, frozen, old, sources, reference, evidence)
        evidence.equal("complete:processes", len({monitor["pid"] for monitor in monitors.values()}), 6)
        evidence.equal("complete:oracles", len(evidence.oracles), frozen["exact_oracles"])
        evidence.equal("complete:instances", len(evidence.relations), frozen["behavioral_instances"])
        evidence.equal("complete:families", len({row["family"] for row in evidence.relations}), frozen["behavioral_families"])
        for name, first in raw.items():
            evidence.equal("complete:raw:" + name, receipts(output / name), first)
        for name, first in root_receipts.items():
            path = output / name
            evidence.equal("complete:root:" + name, {"sha256": sha(path), "bytes": path.stat().st_size}, first)
        for arm, root in roots.items():
            evidence.equal("complete:source:" + arm, packages(root), sources[arm])
            git(root, "diff", "--quiet", "HEAD", "--")
            evidence.equal("complete:untracked:" + arm, git(root, "ls-files", "--others", "--exclude-standard", "simllm"), "")
        evidence.equal("complete:runtime", runtime_identity(), runtime)
        evidence.equal("complete:study", study_sources(ROOT), study)
        evidence.equal("complete:head", git(ROOT, "rev-parse", "HEAD"), head)
        evidence.equal("complete:before-head", git(roots["before"], "rev-parse", "HEAD"), frozen["before_commit"])
        evidence.equal("complete:root-domain", sorted(path.name for path in output.iterdir()), sorted([*raw, *root_receipts, "sink-work"]))
        evidence.check("complete:empty-workdir", (output / "sink-work").is_dir() and not list((output / "sink-work").iterdir()))
        evidence.finish("final-receipts")
        evidence.equal("complete:stages", sorted(evidence.finished_stages), sorted(frozen["required_stages"]))
    except Exception as error:  # noqa: BLE001
        failure = str(error)
        (output / "exception.txt").write_text(traceback.format_exc())
    summary = {"schema": "simllm-snapshot-dispatch-summary-v1", "task": "CORE-68", "source_commit": head,
               "expectations_commit": FREEZE, "verdict": "PASS" if failure is None else "VOID", "failure": failure,
               "behavioral_score": 1 if failure is None else None, "behavioral_instances": len(evidence.relations),
               "behavioral_families": len({row["family"] for row in evidence.relations}), "exact_oracles": len(evidence.oracles),
               "unscored_guards": len(evidence.guards), "fatal_guards_violated": [row for row in evidence.guards if not row["passed"]],
               "admitted_sources": list(data), "stages": sorted(evidence.finished_stages), "monitors": monitors,
               "dispatch_counts": vectors, "corruption_controls": controls, "initial_raw_receipts": raw,
               "root_receipts": root_receipts, "retained_raw_receipts": {name: receipts(output / name) for name in monitors}}
    write(output / "checks.json", {"guards": evidence.guards, "oracles": evidence.oracles, "relations": evidence.relations})
    write(output / "summary.json", summary)
    print(json.dumps({name: summary[name] for name in ("verdict", "failure", "admitted_sources")}), flush=True)
    return 0 if failure is None else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
