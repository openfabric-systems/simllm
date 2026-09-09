"""Qualify retained native engines with a closed, prospective evidence protocol."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

from examples.pd_session_identity_v1.run_study import exact_json_bytes, sha, write
from examples.pd_session_target_scale_v1.checks import (
    Evidence,
    GuardFailure,
    check_accounting,
    check_baseline,
    check_cells,
    check_relations,
    check_retention,
    integer,
)
from examples.pd_session_v1 import run_study as baseline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREEZE_COMMIT = "416fe5ba0386f2770ccd9fc4d0b1ac09a098ef0c"


def parse(raw, label):
    value = json.loads(raw)
    if raw != exact_json_bytes(value):
        raise GuardFailure("noncanonical or duplicate-member JSON: " + label)
    return value


def read(path):
    return parse(path.read_bytes(), path.name)


def file_locks(frozen, package, evidence, label):
    for category, directory in (("current_source_sha256", ROOT),
                                ("historical_files_sha256", ROOT),
                                ("native_source_sha256", package.parent)):
        for name, expected in frozen[category].items():
            evidence.equal(f"{label}:{category}:{name}", sha((directory / name).read_bytes()), expected)
    evidence.equal(label + ":prompt-fixture", sha(baseline.TRACE_PATH.read_bytes()),
                   frozen["frontend"]["fixture_sha256"])


def protocol(args, frozen, evidence):
    relative = HERE.relative_to(ROOT)
    subprocess.run(["git", "merge-base", "--is-ancestor", FREEZE_COMMIT, "HEAD"],
                   cwd=ROOT, check=True)
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT, check=True)
    for name in ("native.py", "checks.py", "run_study.py"):
        subprocess.run(["git", "ls-files", "--error-unmatch", str(relative / name)],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    for name in ("expectations.md", "expectations.json"):
        expected = subprocess.check_output(["git", "show", f"{FREEZE_COMMIT}:{relative / name}"], cwd=ROOT)
        evidence.equal("protocol:freeze:" + name, sha((HERE / name).read_bytes()), sha(expected))
    file_locks(frozen, args.vllm_source, evidence, "before")
    config = (args.hf_hub_cache / ("models--" + frozen["frontend"]["model_id"].replace("/", "--"))
              / "snapshots" / frozen["frontend"]["model_revision"] / "config.json")
    evidence.equal("protocol:model-config", sha(config.read_bytes()), frozen["frontend"]["model_config_sha256"])
    bounds = frozen["physical_bounds"]
    for prompt in (8, 16):
        nbytes = bounds["kv_bytes_per_prompt_token"] * prompt
        evidence.equal(f"protocol:handoff-floor:{prompt}", bounds["handoff_bounds_ps"][str(prompt)]["floor"],
                       nbytes * 8 * 10**12 // (400 * 10**9))
        evidence.equal(f"protocol:handoff-ceiling:{prompt}", bounds["handoff_bounds_ps"][str(prompt)]["ceiling"],
                       nbytes * 8 * 10**12 // (10 * 10**9) + 50000000)
        evidence.check(f"protocol:declared-handoff-bounds:{prompt}", all(
            bounds["handoff_bounds_ps"][str(prompt)]["floor"] <= handoff
            <= bounds["handoff_bounds_ps"][str(prompt)]["ceiling"] for handoff in (100000000, 200000000)))
    evidence.equal("protocol:conditional-resident-floor", bounds["compatibility_resident_streaming_floor_ps"],
                   bounds["per_rank_resident_compatibility_weight_bytes"] * 10**12
                   // bounds["b100_hbm_bytes_per_second"])
    evidence.finish("protocol")
    return config.resolve()


def manifests(output, frozen, evidence):
    from simllm.placement import disaggregated_manifests

    expected = {}
    for scale in frozen["scales"]:
        path = output / "manifest-controls" / scale["id"]
        path.mkdir(parents=True)
        pair = disaggregated_manifests(prefill_nodes=scale["prefill_engines"],
                                      decode_nodes=scale["decode_engines"], render_physical_topology=False)
        pair.placement.save(path / "placement.json")
        pair.fabric.save(path / "fabric.json")
        evidence.check("manifest:off:" + scale["id"], not pair.fabric.physical_rendering_enabled
                       and not pair.fabric.links and not pair.fabric.switches)
        expected[scale["id"]] = {name: sha((path / name).read_bytes())
                                  for name in ("placement.json", "fabric.json")}
        reference = {"p1-d1": "one_plus_one", "p16-d40": "target"}.get(scale["id"])
        if reference is not None:
            raw = (path / "placement.json").read_bytes()
            evidence.equal("manifest:accepted:" + scale["id"], {"sha256": sha(raw), "bytes": len(raw)},
                           frozen["manifest_only_off_path"][reference])
    evidence.finish("manifest-off-path")
    return expected


def launch(args, scale, frozen):
    path = args.output_root / scale["id"]
    path.mkdir()
    env = os.environ.copy()
    for name in ("VLLM_CACHE_ROOT", "XDG_CACHE_HOME", "TMPDIR"):
        cache = args.output_root / "process-caches" / scale["id"] / name.lower()
        cache.mkdir(parents=True)
        env[name] = str(cache)
    env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_HUB_CACHE=str(args.hf_hub_cache),
               VLLM_ENABLE_V1_MULTIPROCESSING="0", SIMLLM_VLLM_WORKER_MODE="skeleton",
               CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="")
    command = [str(args.native_python), "-m", "examples.pd_session_target_scale_v1.native",
               "--scale", scale["id"], "--vllm-source", str(args.vllm_source), "--output-root", str(path)]
    started = time.monotonic()
    monitor = {"command": command, "pid": None, "sampled_max_current_rss_kib": 0,
               "exit_code": None, "stopping_reason": None}
    with (path / "native.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        monitor["pid"] = process.pid
        try:
            while process.poll() is None:
                status = Path(f"/proc/{process.pid}/status")
                if status.exists():
                    try:
                        rows = status.read_text().splitlines()
                    except FileNotFoundError:
                        rows = []
                    current = next((int(row.split()[1]) for row in rows if row.startswith("VmRSS:")), 0)
                    monitor["sampled_max_current_rss_kib"] = max(monitor["sampled_max_current_rss_kib"], current)
                    if current >= frozen["construction"]["stop_rss_kib"]:
                        raise GuardFailure("frozen resident-memory stop reached")
                if time.monotonic() - started >= frozen["construction"]["timeout_seconds_per_scale"]:
                    raise GuardFailure("frozen native timeout reached")
                try:
                    process.wait(timeout=0.25)
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
    if process.returncode != 0:
        raise GuardFailure("native process failed: " + scale["id"])
    return monitor


def admit(data, scale, frozen, args, config, monitor, expected_manifests, evidence):
    prefix, identity = scale["id"], data["native_identity"]
    evidence.fields(prefix + ":identity-fields", identity,
                    "pid interpreter package version config_sha256 config_path freeze_sha256 "
                    "cuda_initialized_before cuda_initialized_after clock_object_id packet_backend_runs "
                    "native_origins simllm_origins simllm_source_before simllm_source_after "
                    "unfinished_requests_after collective_arms observer_weak_references_only all_observed_objects_alive")
    evidence.equal(prefix + ":components", sorted(data), sorted(
        [*frozen["evidence_domains"]["required_scale_components"], "schema", "scale_id", "baseline_controls"]))
    evidence.equal(prefix + ":schema", data["schema"], "simllm-pd-session-target-scale-native-v1")
    evidence.equal(prefix + ":scale", data["scale_id"], prefix)
    evidence.equal(prefix + ":native-pid", identity["pid"], monitor["pid"])
    evidence.check(prefix + ":native-process", integer(identity["pid"], 1) and identity["pid"] != os.getpid())
    evidence.equal(prefix + ":interpreter", str(Path(identity["interpreter"]).absolute()), str(args.native_python))
    evidence.equal(prefix + ":package", identity["package"], str(args.vllm_source))
    evidence.equal(prefix + ":version", identity["version"], frozen["frontend"]["version"])
    evidence.equal(prefix + ":config-path", identity["config_path"], str(config))
    evidence.equal(prefix + ":config", identity["config_sha256"], sha(config.read_bytes()))
    evidence.equal(prefix + ":freeze", identity["freeze_sha256"], sha((HERE / "expectations.json").read_bytes()))
    evidence.equal(prefix + ":gpu-before", identity["cuda_initialized_before"], False)
    evidence.equal(prefix + ":gpu-after", identity["cuda_initialized_after"], False)
    evidence.equal(prefix + ":packet-backend", identity["packet_backend_runs"], 0)
    evidence.equal(prefix + ":collective-arm", identity["collective_arms"],
                   [[frozen["deployment"]["collective_envelope"], frozen["deployment"]["collective_arm"]]])
    evidence.equal(prefix + ":all-finished", identity["unfinished_requests_after"], [False] * scale["engines"])
    for field, category, root in (("native_origins", "native_source_sha256", args.vllm_source.parent),
                                  ("simllm_origins", "current_source_sha256", ROOT)):
        evidence.equal(prefix + ":origin-domain:" + field, sorted(identity[field]), sorted(frozen[category]))
        for name, path in identity[field].items():
            evidence.equal(prefix + ":origin:" + name, path, str((root / name).resolve()))
            evidence.equal(prefix + ":actual-source:" + name, sha(Path(path).read_bytes()), frozen[category][name])
    for field in ("source_before", "source_after"):
        evidence.equal(prefix + ":" + field, data[field], frozen["native_source_sha256"])
    for field in ("simllm_source_before", "simllm_source_after"):
        evidence.equal(prefix + ":" + field, identity[field], frozen["current_source_sha256"])
    if prefix != frozen["baseline_control"]["scale"]:
        evidence.equal(prefix + ":no-extra-baseline", data["baseline_controls"], [])
    for name, expected in expected_manifests.items():
        evidence.equal(prefix + ":manifest:" + name, sha((args.output_root / prefix / name).read_bytes()), expected)
    host = data["host_measurements"]
    evidence.fields(prefix + ":host-fields", host,
                    "current_rss_before_kib peak_rss_before_kib current_rss_constructed_kib "
                    "current_rss_after_requests_kib peak_rss_after_requests_kib construction_wall_time_ns "
                    "request_wall_time_ns request_count final_clock_ps")
    for key, value in host.items():
        evidence.check(prefix + ":host-type:" + key, integer(value, 1))
        if key.startswith("current_rss"):
            evidence.check(prefix + ":rss-cap:" + key, value < frozen["construction"]["stop_rss_kib"])
    evidence.check(prefix + ":memory-high-water", host["peak_rss_before_kib"] >= host["current_rss_before_kib"]
                   and host["peak_rss_after_requests_kib"] >= host["current_rss_after_requests_kib"])
    evidence.equal(prefix + ":host-request-count", host["request_count"], 400)
    evidence.equal(prefix + ":host-time-sum", host["request_wall_time_ns"],
                   sum(row["wall_time_ns"] for row in [*data["serial_cells"], data["burst_cell"]]))
    evidence.check(prefix + ":construction-time", host["construction_wall_time_ns"]
                   >= data["construction_rows"][-1]["elapsed_ns"])


def analyze(data, scale, frozen, evidence):
    check_retention(data, scale, frozen, evidence)
    requests, metrics = check_cells(data, scale, frozen, evidence)
    accounting = check_accounting(data, scale, frozen, requests, evidence)
    check_relations(scale, frozen, metrics, data["serial_cells"], evidence)
    if scale["id"] == frozen["baseline_control"]["scale"]:
        check_baseline(data, frozen, evidence)
    return accounting


def completion(evidence, frozen):
    evidence.equal("complete:stages", sorted(evidence.finished_stages), sorted(frozen["evidence_domains"]["stages"]))
    relations = {f"{scale['id']}:handoff-response:{prompt}": "handoff-response"
                 for scale in frozen["scales"] for prompt in (8, 16)}
    relations.update({f"{scale['id']}:prompt-response:{handoff}": "prompt-response"
                      for scale in frozen["scales"] for handoff in (100000000, 200000000)})
    evidence.equal("complete:relations", {row["name"]: row["family"] for row in evidence.relations}, relations)
    oracles = [f"{scale['id']}:{cell['id']}:historical-cell"
               for scale in frozen["scales"] for cell in frozen["serial_cells"]]
    oracles.extend("baseline:accepted:" + row["cell"] for row in frozen["baseline_control"]["comparisons"])
    evidence.equal("complete:oracles", sorted(row["name"] for row in evidence.oracles), sorted(oracles))


def mutations(original, frozen, evidence):
    scale = frozen["scales"][0]
    targets = {
        "missing-engine": "p1-d1:retention-order",
        "duplicate-worker": "p1-d1:distinct-workers",
        "wrong-global-rank": "p1-d1:global-rank-domain",
        "foreign-scheduled-request": ":member:",
        "duplicate-step-identity": ":unique-step-domain",
        "shift-step-completion": ":result",
        "drop-clock-advance": "p1-d1:clock:",
        "drop-terminal-request": ":request-domain",
        "alter-baseline-member": "baseline:accepted:",
        "missing-stage": "complete:stages",
    }
    evidence.equal("controls:domain", list(targets), frozen["corruption_controls"])
    rows = []
    for name, target in targets.items():
        changed = deepcopy(original)
        trial = Evidence(frozen["evidence_domains"]["stages"])
        if name == "missing-engine":
            changed["retention_before"].pop()
        elif name == "duplicate-worker":
            duplicate = changed["retention_before"][0]["workers"][0]["object_id"]
            old = changed["retention_before"][1]["workers"][0]["object_id"]
            for collection in (changed["retention_before"], changed["retention_after"],
                               [row["identity"] for row in changed["construction_rows"]]):
                for engine in collection:
                    for field in ("workers", "executor_workers"):
                        for worker in engine[field]:
                            if worker["object_id"] == old:
                                worker["object_id"] = duplicate
            for row in changed["worker_manifest_projection"]:
                if row["worker_object_id"] == old:
                    row["worker_object_id"] = duplicate
        elif name == "wrong-global-rank":
            changed["worker_manifest_projection"][0]["global_rank"] += 1
        elif name == "foreign-scheduled-request":
            next(row for row in changed["unique_steps"] if row["record"]["scheduled"])["record"]["scheduled"][0]["request_id"] = "foreign"
        elif name == "duplicate-step-identity":
            changed["unique_steps"].append(deepcopy(changed["unique_steps"][0]))
        elif name == "shift-step-completion":
            changed["unique_steps"][0]["result"]["completed_at_ps"] += 1
        elif name == "drop-clock-advance":
            index = next(i for i, row in enumerate(changed["clock_advances"])
                         if row["after_ps"] > row["before_ps"])
            del changed["clock_advances"][index]
        elif name == "drop-terminal-request":
            changed["serial_cells"][0]["requests"].pop()
        elif name == "alter-baseline-member":
            row = changed["baseline_controls"][0]
            row["result"]["bootstrap_token_id"] += 1
            row["comparison"]["request"]["bootstrap_token_id"] += 1
            row["comparison_sha256"] = sha(exact_json_bytes(row["comparison"]))
        try:
            if name == "missing-stage":
                trial.finished_stages = set(trial.expected_stages) - {"p16-d40:retention"}
                completion(trial, frozen)
            elif name == "alter-baseline-member":
                check_baseline(changed, frozen, trial)
            else:
                analyze(changed, scale, frozen, trial)
        except GuardFailure as error:
            caught = str(error)
        else:
            caught = ""
        evidence.check("controls:" + name, target in caught)
        rows.append({"name": name, "guard": caught, "discriminated": True})
    return rows


def rational(numerator, denominator):
    value = Fraction(numerator, denominator)
    return {"numerator": value.numerator, "denominator": value.denominator}


def summary(evidence, frozen, **extra):
    valid = extra.pop("valid", False)
    return {
        "schema": "simllm-pd-session-target-scale-summary-v1", "verdict": "PASS" if valid else "VOID",
        "freeze_commit": FREEZE_COMMIT, "freeze_sha256": sha((HERE / "expectations.json").read_bytes()),
        "behavioral_score": {"instances": len(evidence.relations), "families": len(frozen["behavioral_families"])} if valid else None,
        "completed_stages": sorted(evidence.finished_stages),
        "evidence": {"unscored_guards": evidence.guards, "exact_oracles": evidence.oracles,
                     "behavioral_relations": evidence.relations}, **extra,
    }


def execute(args):
    frozen = json.loads((HERE / "expectations.json").read_bytes())
    evidence = Evidence(frozen["evidence_domains"]["stages"])
    args.output_root.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    accepted, reports, controls, monitors = {}, [], [], []
    try:
        config = protocol(args, frozen, evidence)
        implementation = {name: sha((HERE / name).read_bytes()) for name in (
            "native.py", "checks.py", "run_study.py", "expectations.md", "expectations.json")}
        expected_manifests = manifests(args.output_root, frozen, evidence)
        for scale in frozen["scales"]:
            print("Starting " + scale["id"], flush=True)
            monitor = launch(args, scale, frozen)
            monitors.append(monitor)
            path = args.output_root / scale["id"]
            raw = (path / "native.json").read_bytes()
            receipt_raw = (path / "receipt.json").read_bytes()
            data, receipt = parse(raw, "native.json"), parse(receipt_raw, "receipt.json")
            accepted[scale["id"]] = {
                "native.json": sha(raw), "receipt.json": sha(receipt_raw),
                "process.json": sha(exact_json_bytes(monitor)), **expected_manifests[scale["id"]],
            }
            evidence.equal(scale["id"] + ":receipt", receipt, {"pid": monitor["pid"], "native_sha256": sha(raw)})
            admit(data, scale, frozen, args, config, monitor, expected_manifests[scale["id"]], evidence)
            accounting = analyze(data, scale, frozen, evidence)
            if scale["id"] == frozen["baseline_control"]["scale"]:
                controls = mutations(data, frozen, evidence)
            host = data["host_measurements"]
            cells = []
            for cell in [*data["serial_cells"], data["burst_cell"]]:
                duration = cell["end_ps"] - cell["start_ps"]
                cells.append({"id": cell["id"], "virtual_makespan_ps": duration,
                              "arrival_mode": cell["arrival_mode"], "wall_time_ns": cell["wall_time_ns"],
                              "completed_requests_per_second": rational(80 * 10**12, duration),
                              "completed_tokens_per_second": rational(320 * 10**12, duration)})
            reports.append({**scale, "pid": monitor["pid"], "native_sha256": sha(raw),
                            "host_measurements": host, "accounting": accounting, "cells": cells,
                            "host_requests_per_second": rational(400 * 10**9, host["request_wall_time_ns"])})
            print("Qualified " + scale["id"], flush=True)
        evidence.check("complete:fresh-processes", len({row["pid"] for row in monitors}) == len(frozen["scales"]))
        for scale_id, members in accepted.items():
            for name, expected in members.items():
                evidence.equal(f"reread:{scale_id}:{name}", sha((args.output_root / scale_id / name).read_bytes()), expected)
        file_locks(frozen, args.vllm_source, evidence, "after")
        for name, expected in implementation.items():
            evidence.equal("after:harness:" + name, sha((HERE / name).read_bytes()), expected)
        subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT, check=True)
        evidence.equal("after:source-commit", subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), source)
        completion(evidence, frozen)
        result = summary(evidence, frozen, valid=True, source_commit=source,
                         scales=reports, corruption_controls=controls, failure=None)
    except Exception as error:  # noqa: BLE001
        (args.output_root / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        result = summary(evidence, frozen, source_commit=source, scales=reports,
                         corruption_controls=controls, failure={"type": type(error).__name__, "reason": str(error)})
    write(args.output_root / "summary.json", result)
    print(result["verdict"], flush=True)
    return 0 if result["verdict"] == "PASS" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-python", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--hf-hub-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.native_python = args.native_python.absolute()
    args.vllm_source = args.vllm_source.resolve()
    args.hf_hub_cache = args.hf_hub_cache.resolve()
    args.output_root = args.output_root.resolve()
    if ROOT == args.output_root or ROOT in args.output_root.parents:
        parser.error("--output-root must be outside the repository")
    return execute(args)


if __name__ == "__main__":
    sys.exit(main())
