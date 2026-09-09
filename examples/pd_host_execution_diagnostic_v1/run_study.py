"""Coordinate the frozen host diagnostic and retain incomplete attempts."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from examples.pd_host_execution_diagnostic_v1.checks import (
    admit,
    check_profile,
    compare_pair,
    expected_selections,
    fresh_evidence,
)
from examples.pd_host_execution_diagnostic_v1.timing import profile_rows
from examples.pd_session_identity_v1.run_study import exact_json_bytes, sha, write
from examples.pd_session_target_scale_v1.checks import Evidence, GuardFailure, integer
from examples.pd_session_v1 import run_study as baseline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREEZE_COMMIT = "8f06648d26bbfb2bfff4751d29a2e2bf232b440d"


def parse(raw, label):
    value = json.loads(raw)
    if exact_json_bytes(value) != raw:
        raise GuardFailure("noncanonical or duplicate-member JSON: " + label)

    def finite(item):
        if isinstance(item, dict):
            return all(finite(v) for v in item.values())
        if isinstance(item, list):
            return all(finite(v) for v in item)
        return type(item) is not float or math.isfinite(item)

    if not finite(value):
        raise GuardFailure("nonfinite JSON value: " + label)
    return value


def file_locks(args, frozen, evidence, label):
    for category, directory in (("source_sha256", ROOT), ("historical_files_sha256", ROOT),
                                ("native_source_sha256", args.vllm_source.parent)):
        for name, expected in frozen[category].items():
            evidence.equal(label + ":" + category + ":" + name, sha((directory / name).read_bytes()), expected)
    evidence.equal(label + ":prompt-fixture", sha(baseline.TRACE_PATH.read_bytes()), frozen["frontend"]["fixture_sha256"])
    config = (args.hf_hub_cache / ("models--" + frozen["frontend"]["model_id"].replace("/", "--"))
              / "snapshots" / frozen["frontend"]["model_revision"] / "config.json")
    evidence.equal(label + ":model-config", sha(config.read_bytes()), frozen["frontend"]["model_config_sha256"])
    return config.resolve()


def physical_protocol(frozen, evidence):
    """Bound the declared surrogate before reading any new native output."""
    bounds = frozen["physical_bounds"]
    floor = bounds["per_rank_resident_compatibility_weight_bytes"] * 10**12 // bounds["b100_hbm_bytes_per_second"]
    evidence.equal("physics:conditional-streaming-floor", floor, bounds["compatibility_resident_streaming_floor_ps"])
    for spec in (*frozen["historical_control_specs"], *frozen["cells"]):
        name, prompt = spec["id"], str(spec["prompt_tokens"])
        payload_bits = spec["prompt_tokens"] * bounds["kv_bytes_per_prompt_token"] * 8
        handoff_floor = payload_bits * 10**12 // 400_000_000_000
        handoff_ceiling = payload_bits * 10**12 // 10_000_000_000 + 50_000_000
        evidence.equal("physics:handoff-bounds:" + name, bounds["handoff_bounds_ps"][prompt],
                       {"floor": handoff_floor, "ceiling": handoff_ceiling})
        evidence.check("physics:handoff:" + name, handoff_floor <= spec["handoff_ps"] <= handoff_ceiling)
        service = frozen["known_services_ps"][prompt]
        evidence.check("physics:surrogate-service:" + name,
                       all(max(floor, bounds["nonempty_step_service_ps"]["floor"]) <= value
                           <= bounds["nonempty_step_service_ps"]["ceiling"]
                           for value in [service["prefill"], *service["decode"]]))


def source_receipts(data, frozen, package, evidence):
    expected = {"repository": frozen["source_sha256"], "native": frozen["native_source_sha256"],
                "origins": {name: str((ROOT / name).resolve()) for name in frozen["source_sha256"]},
                "native_origins": {name: str((package.parent / name).resolve()) for name in frozen["native_source_sha256"]}}
    for when in ("before", "after"):
        evidence.equal("identity:source-" + when, data["identity"]["sources_" + when], expected)


def reread_tree(path, receipts, evidence, label):
    names = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
    evidence.equal(label + ":domain", sorted(names), sorted(receipts))
    for name, expected in receipts.items():
        evidence.equal(label + ":" + name, sha((path / name).read_bytes()), expected)


def launch(args, arm, frozen):
    path = args.output_root / arm
    path.mkdir()
    env = os.environ.copy()
    for name in ("VLLM_CACHE_ROOT", "XDG_CACHE_HOME", "TMPDIR"):
        cache = args.output_root / "process-caches" / arm / name.lower()
        cache.mkdir(parents=True)
        env[name] = str(cache)
    env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_HUB_CACHE=str(args.hf_hub_cache),
               VLLM_ENABLE_V1_MULTIPROCESSING="0", SIMLLM_VLLM_WORKER_MODE="skeleton",
               CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="")
    command = [str(args.native_python), "-m", "examples.pd_host_execution_diagnostic_v1.native",
               "--arm", arm, "--vllm-source", str(args.vllm_source), "--output-root", str(path)]
    monitor = {"command": command, "pid": None, "sampled_max_current_rss_kib": 0,
               "exit_code": None, "stopping_reason": None}
    started = time.monotonic()
    with (path / "native.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        monitor["pid"] = process.pid
        try:
            while process.poll() is None:
                try:
                    rows = Path(f"/proc/{process.pid}/status").read_text().splitlines()
                except FileNotFoundError:
                    rows = []
                current = next((int(row.split()[1]) for row in rows if row.startswith("VmRSS:")), 0)
                monitor["sampled_max_current_rss_kib"] = max(current, monitor["sampled_max_current_rss_kib"])
                if current >= frozen["limits"]["rss_cap_kib"]:
                    raise GuardFailure("frozen resident-memory stop reached")
                if time.monotonic() - started >= frozen["limits"]["timeout_seconds"]:
                    raise GuardFailure("frozen native timeout reached")
                try:
                    process.wait(timeout=frozen["limits"]["sample_seconds"])
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
            write(path / "process.json", monitor)
    if process.returncode:
        raise GuardFailure("native process failed: " + arm)
    return monitor


def admit_identity(data, args, frozen, monitor, config, evidence):
    identity = data["identity"]
    evidence.fields("identity:fields", identity,
                    "pid interpreter package version config_path config_sha256 sources_before sources_after selections_before selections_after")
    evidence.equal("identity:pid", identity["pid"], monitor["pid"])
    evidence.check("identity:pid-type", integer(identity["pid"], 1))
    evidence.equal("identity:interpreter", str(Path(identity["interpreter"]).resolve()), str(args.native_python.resolve()))
    evidence.equal("identity:package", identity["package"], str(args.vllm_source.resolve()))
    evidence.equal("identity:version", identity["version"], frozen["frontend"]["version"])
    evidence.equal("identity:config-path", identity["config_path"], str(config))
    evidence.equal("identity:config-hash", identity["config_sha256"], frozen["frontend"]["model_config_sha256"])
    source_receipts(data, frozen, args.vllm_source, evidence)
    for when in ("before", "after"):
        evidence.equal("identity:selections-" + when, identity["selections_" + when], expected_selections(frozen))
    evidence.check("process:host-limit", monitor["exit_code"] == 0 and monitor["stopping_reason"] is None
                   and 0 < monitor["wall_seconds"] < frozen["limits"]["timeout_seconds"]
                   and 0 < monitor["sampled_max_current_rss_kib"] < frozen["limits"]["rss_cap_kib"])


def probe_source_admission(data, args, frozen, evidence):
    allowed = {str((ROOT / name).resolve()): name for name in frozen["source_sha256"]}
    allowed.update({str((args.vllm_source.parent / name).resolve()): name
                    for name in frozen["native_source_sha256"]})
    parsed = {}
    for index, row in enumerate(data["probe_restoration"]):
        name = "probe-source:" + str(index)
        source = row["source"]
        evidence.fields(name + ":fields", source, "module qualname filename first_line")
        path = source["filename"]
        evidence.check(name + ":path", type(path) is str and path in allowed)
        evidence.equal(name + ":module", source["module"], allowed[path].removesuffix(".py").replace("/", "."))
        if path not in parsed:
            definitions = set()

            def visit(node, prefix=(), definitions=definitions):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        full = (*prefix, child.name)
                        line = min([child.lineno, *(item.lineno for item in child.decorator_list)])
                        if not isinstance(child, ast.ClassDef):
                            definitions.add((".".join(full), line))
                        visit(child, full)
                    else:
                        visit(child, prefix)

            visit(ast.parse(Path(path).read_text(encoding="utf-8")))
            parsed[path] = definitions
        evidence.check(name + ":definition", (source["qualname"], source["first_line"]) in parsed[path])
        evidence.equal(name + ":function", source["qualname"].split(".")[-1], row["attribute"])


def profile_admission(data, exports, frozen, evidence):
    expected = ([spec["id"] for spec in frozen["cells"] if spec["profile_in_instrumented_arm"]]
                if data["arm"] == "instrumented" else [])
    evidence.equal("profiles:domain", [row["cell_id"] for row in data["profiles"]], expected)
    evidence.equal("profiles:exports", sorted(exports), sorted(expected))
    for entry, spec in zip(data["cells"], frozen["cells"], strict=True):
        selected = spec["id"] in expected
        profile = entry["profile"]
        if not selected:
            evidence.equal("profile:off:" + spec["id"], profile, None)
            continue
        name = "profile:" + spec["id"]
        evidence.fields(name + ":receipt-fields", profile,
                        "cell_id raw_path raw_sha256 export_path export_sha256 functions")
        evidence.equal(name + ":joined-receipt", profile, next(row for row in data["profiles"] if row["cell_id"] == spec["id"]))
        evidence.equal(name + ":raw-path", profile["raw_path"], "profiles/" + spec["id"] + ".pstats")
        evidence.equal(name + ":export-path", profile["export_path"], "profiles/" + spec["id"] + ".json")
        document = exports[spec["id"]]
        check_profile(document, evidence, "profile:" + spec["id"])
        evidence.equal(name + ":cell", document["cell_id"], spec["id"])
        evidence.equal(name + ":function-count", profile["functions"], len(document["functions"]))


def raw_admission(path, data, frozen, evidence, admitted_bytes):
    expected_requests = [*data["baseline_controls"],
                         *(row for entry in data["cells"] for row in entry["cell"]["requests"])]
    progress = [parse(line, "request progress") for line in admitted_bytes["request-progress.jsonl"].splitlines()]
    evidence.equal("raw:complete-progress", progress, expected_requests)
    construction = [parse(line, "construction progress")
                    for line in admitted_bytes["construction-progress.jsonl"].splitlines()]
    evidence.equal("raw:construction", construction, data["construction"])
    exports = {}
    expected_profiles = {row[field + "_path"] for row in data["profiles"] for field in ("raw", "export")}
    evidence.equal("raw:profile-file-domain", sorted(name for name in admitted_bytes if name.startswith("profiles/")),
                   sorted(expected_profiles))
    for entry in data["cells"]:
        name = entry["cell"]["id"] + ".json"
        evidence.equal("raw:cell:" + name, parse(admitted_bytes[name], name), entry["cell"])
    for row in data["profiles"]:
        for field in ("raw", "export"):
            name = row[field + "_path"]
            evidence.check("profile:relative-path:" + name, name in admitted_bytes and (path / name).resolve().is_relative_to(path.resolve()))
            evidence.equal("profile:raw-receipt:" + name, sha(admitted_bytes[name]), row[field + "_sha256"])
        document = parse(admitted_bytes[row["export_path"]], row["export_path"])
        raw_path = path / row["raw_path"]
        evidence.equal("profile:disk-before:" + row["cell_id"], sha(raw_path.read_bytes()), row["raw_sha256"])
        evidence.equal("profile:complete-export:" + row["cell_id"], document["functions"], profile_rows(str(raw_path)))
        evidence.equal("profile:disk-after:" + row["cell_id"], sha(raw_path.read_bytes()), row["raw_sha256"])
        exports[row["cell_id"]] = document
    for role in ("prefill", "decode"):
        engine = "simllm-" + role + "-0"
        name = "engine-work/" + engine + "/step-records.jsonl"
        records = [parse(line, name) for line in admitted_bytes[name].splitlines()]
        evidence.equal("raw:steps:" + role, records, [row["record"] for row in data["unique_steps"] if row["engine_id"] == engine])
    if data["arm"] == "instrumented":
        evidence.equal("raw:restoration", parse(admitted_bytes["probe-restoration.json"], "restoration"), data["probe_restoration"])
    profile_admission(data, exports, frozen, evidence)
    return exports


def mutations(first, second, exports, frozen, output_root, package):
    controls = []
    for name in frozen["semantic_mutations"]:
        changed, trial = deepcopy(second), fresh_evidence()
        altered_exports = deepcopy(exports)
        target = ""
        try:
            if name == "changed-nonopaque-request-member":
                changed["baseline_controls"][0]["result"]["bootstrap_token_id"] += 1
                target = "pair:requests"
                compare_pair(first, changed, frozen, trial)
            elif name == "missing-native-step":
                changed["unique_steps"].pop()
                target = "steps:complete"
                admit(changed, frozen, trial)
            elif name in ("foreign-normalized-step-id", "same-public-request-opposite-role-id"):
                changed["unique_steps"][0]["record"]["scheduled"][0]["request_id"] = (
                    "foreign" if name == "foreign-normalized-step-id"
                    else changed["baseline_controls"][0]["result"]["decode_internal_request_id"])
                target = "foreign engine-qualified"
                compare_pair(first, changed, frozen, trial)
            elif name == "changed-sink-outcome":
                changed["sink_outcomes"][0]["collections"]["outcomes"][0]["makespan_ps"] += 1
                target = "pair:sinks"
                compare_pair(first, changed, frozen, trial)
            elif name == "incoherent-phase-time":
                changed["phase_counters"]["sink-plan"]["exclusive_wall_ns"] += 1
                target = "total-counters:sink-plan:wall"
                admit(changed, frozen, trial)
            elif name == "missing-profile-function-table":
                altered_exports[next(iter(altered_exports))]["functions"] = []
                target = "function-table"
                profile_admission(changed, altered_exports, frozen, trial)
            elif name == "unrestored-wrapper":
                changed["probe_restoration"][0]["restored"] = False
                target = "probe:0:restored"
                admit(changed, frozen, trial)
            elif name == "wrong-source-receipt":
                changed["identity"]["sources_after"]["repository"][next(iter(frozen["source_sha256"]))] = "0" * 64
                target = "identity:source-after"
                source_receipts(changed, frozen, package, trial)
            elif name == "disk-only-raw-change":
                original = exact_json_bytes(changed)
                changed["baseline_controls"][0]["result"]["ttft_ps"] += 1
                target = "mutation:raw-reread:native.json"
                with TemporaryDirectory(prefix="raw-corruption-", dir=output_root) as directory:
                    path = Path(directory)
                    (path / "native.json").write_bytes(exact_json_bytes(changed))
                    reread_tree(path, {"native.json": sha(original)}, trial, "mutation:raw-reread")
            else:
                raise RuntimeError("unknown frozen mutation: " + name)
        except GuardFailure as error:
            finding = str(error)
        else:
            finding = ""
        if not target or target not in finding:
            raise GuardFailure("corruption did not reach its declared guard: " + name + ":" + finding)
        controls.append({"name": name, "guard": finding, "rejected": True})
    return controls


def execute(args):
    frozen = json.loads((HERE / "expectations.json").read_bytes())
    args.output_root.mkdir(parents=True, exist_ok=False)
    evidence = Evidence(frozen["required_stages"])
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result = {"schema": "simllm-pd-host-execution-diagnostic-summary-v1", "verdict": "VOID",
              "behavioral_score": None, "source_commit": source, "freeze_commit": FREEZE_COMMIT,
              "arms": [], "corruption_controls": [], "raw_receipts": {}}
    try:
        subprocess.run(["git", "merge-base", "--is-ancestor", FREEZE_COMMIT, "HEAD"], cwd=ROOT, check=True)
        subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT, check=True)
        for name in ("expectations.md", "expectations.json"):
            original = subprocess.check_output(["git", "show", f"{FREEZE_COMMIT}:{HERE.relative_to(ROOT) / name}"], cwd=ROOT)
            evidence.equal("freeze:" + name, sha((HERE / name).read_bytes()), sha(original))
        harness = {path.name: sha(path.read_bytes()) for path in HERE.iterdir()
                   if path.suffix == ".py" or path.name in ("expectations.md", "expectations.json")}
        for name in harness:
            subprocess.run(["git", "ls-files", "--error-unmatch", str(HERE.relative_to(ROOT) / name)],
                           cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        config = file_locks(args, frozen, evidence, "before")
        physical_protocol(frozen, evidence)
        evidence.finish("protocol")
        accepted, raw_receipts, profiles = [], result["raw_receipts"], {}
        for arm in frozen["arms"]:
            print("Starting " + arm, flush=True)
            monitor = launch(args, arm, frozen)
            path = args.output_root / arm
            admitted_bytes = {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()}
            raw_receipts[arm] = {name: sha(raw) for name, raw in admitted_bytes.items()}
            write(args.output_root / "first-raw-receipts.json", raw_receipts)
            data = parse(admitted_bytes["native.json"], "native.json")
            receipt = parse(admitted_bytes["receipt.json"], "receipt.json")
            evidence.equal(arm + ":receipt", receipt, {"pid": monitor["pid"], "native_sha256": sha(admitted_bytes["native.json"])})
            evidence.equal(arm + ":selected", data["arm"], arm)
            evidence.finish(arm + ":capture")
            trial = fresh_evidence()
            try:
                admit_identity(data, args, frozen, monitor, config, trial)
                admit(data, frozen, trial)
                probe_source_admission(data, args, frozen, trial)
                profiles[arm] = raw_admission(path, data, frozen, trial, admitted_bytes)
            finally:
                for row in trial.guards:
                    evidence.guards.append({**row, "name": arm + ":" + row["name"]})
                for row in trial.oracles:
                    evidence.oracles.append({**row, "name": arm + ":" + row["name"]})
            evidence.finish(arm + ":admission")
            accepted.append(data)
            result["arms"].append({"arm": arm, "pid": monitor["pid"],
                                   "native_sha256": receipt["native_sha256"], "host_process": monitor,
                                   "baseline_host_call": data["baseline_host_call"],
                                   "cells": [{"id": entry["cell"]["id"], "host_call": entry["host_call"],
                                              "request_wall_ns": entry["cell"]["wall_time_ns"],
                                              "phase_counters": entry["phase_counters"], "profile": entry["profile"]}
                                             for entry in data["cells"]], "phase_counters": data["phase_counters"]})
            print("Admitted " + arm, flush=True)
        compare_pair(*accepted, frozen, evidence)
        evidence.finish("pair-comparison")
        evidence.check("fresh-processes", len({data["identity"]["pid"] for data in accepted}) == 2)
        result["profile_exports"] = [{"cell": name, "function_count": len(profile["functions"])}
                                     for name, profile in profiles["instrumented"].items()]
        evidence.finish("profile-attribution")
        result["corruption_controls"] = mutations(*accepted, profiles["instrumented"], frozen, args.output_root, args.vllm_source)
        evidence.finish("semantic-corruptions")
        for arm, receipts in raw_receipts.items():
            reread_tree(args.output_root / arm, receipts, evidence, "reread:" + arm)
        file_locks(args, frozen, evidence, "after")
        for name, expected in harness.items():
            evidence.equal("harness:" + name, sha((HERE / name).read_bytes()), expected)
        subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT, check=True)
        evidence.equal("source:final", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), source)
        evidence.finish("final-identities")
        evidence.equal("stages:complete", sorted(evidence.finished_stages), sorted(frozen["required_stages"]))
        result.update(verdict="COMPLETE", failure=None, raw_receipts=raw_receipts)
    except Exception as error:  # noqa: BLE001
        (args.output_root / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        result["failure"] = {"type": type(error).__name__, "reason": str(error)}
    result["completed_stages"] = sorted(evidence.finished_stages)
    result["evidence"] = {"unscored_guards": evidence.guards, "exact_oracles": evidence.oracles,
                          "behavioral_relations": []}
    write(args.output_root / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-python", "vllm-source", "hf-hub-cache", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = execute(parser.parse_args())
    print(result["verdict"], flush=True)
    return 0 if result["verdict"] == "COMPLETE" else 2


if __name__ == "__main__":
    sys.exit(main())
