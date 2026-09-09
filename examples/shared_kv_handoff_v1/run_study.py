"""Run the frozen shared packet campaign, retaining first receipts on failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from examples.bridge_lifecycle_v1.run_study import _snapshot
from examples.independent_engine_completion_v1.checks import request_times
from examples.independent_engine_completion_v1.common import step_stream
from examples.independent_engine_completion_v1.run_study import progress_rows
from examples.pd_session_target_scale_v1.checks import Evidence, GuardFailure

from . import components, protocol
from .aliases import request_rows
from .checks import admit, check_relations
from .common import FREEZE, ROOT, plain, read, receipts, sha, write
from .network_checks import geometry
from .wire import check_wire


def launch(args, spec, frozen, monitors):
    path = args.output_root / spec["id"]
    path.mkdir()
    (path / "markers").mkdir()
    overrides = protocol.environment(args, spec)
    for name in ("VLLM_CACHE_ROOT", "XDG_CACHE_HOME", "TMPDIR"):
        Path(overrides[name]).mkdir(parents=True)
    monitor = {"command": protocol.command(args, spec), "cwd": str(protocol.selected_root(args, spec)),
               "environment_overrides": overrides, "pid": None, "sampled_max_current_rss_kib": 0,
               "exit_code": None, "stopping_reason": None}
    monitors[spec["id"]] = monitor
    write(path / "process-start.json", monitor)
    start, process = time.monotonic(), None
    try:
        with (path / "native.log").open("wb") as log:
            process = subprocess.Popen(monitor["command"], cwd=monitor["cwd"], env=dict(os.environ, **overrides),
                                       stdout=log, stderr=subprocess.STDOUT)
            monitor["pid"] = process.pid
            write(path / "process-start.json", monitor)
            while process.poll() is None:
                try:
                    lines = (Path("/proc") / str(process.pid) / "status").read_text().splitlines()
                except FileNotFoundError:
                    lines = []
                rss = next((int(row.split()[1]) for row in lines if row.startswith("VmRSS:")), 0)
                monitor["sampled_max_current_rss_kib"] = max(rss, monitor["sampled_max_current_rss_kib"])
                if rss * 1024 >= frozen["limits"]["max_current_rss_bytes"]:
                    raise GuardFailure("frozen native resident-memory limit reached")
                if time.monotonic() - start >= frozen["limits"]["wall_seconds_per_process"]:
                    raise GuardFailure("frozen native wall-time limit reached")
                try:
                    process.wait(timeout=frozen["limits"]["rss_poll_seconds"])
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode != 0:
                raise GuardFailure("native process failed: " + spec["id"])
    except BaseException as error:
        monitor["stopping_reason"] = {"type": type(error).__name__, "message": str(error)}
        try:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
        except BaseException as cleanup:  # noqa: BLE001, preserve the first native failure.
            monitor["cleanup_failure"] = {"type": type(cleanup).__name__, "message": str(cleanup)}
        raise
    finally:
        primary = sys.exc_info()[1]
        monitor["wall_seconds"] = time.monotonic() - start
        monitor["exit_code"] = None if process is None else process.returncode
        try:
            markers = [read(file) for file in sorted((path / "markers").glob("*.json"))]
            monitor["markers"] = markers
            monitor["child_states_after_exit"] = [_snapshot(marker) for marker in markers]
        except BaseException as error:  # noqa: BLE001, keep a failed worker's original error.
            monitor["marker_capture_error"] = {"type": type(error).__name__, "message": str(error)}
        try:
            write(path / "process.json", monitor)
        except BaseException as error:
            monitor["process_write_failure"] = {"type": type(error).__name__, "message": str(error)}
            if primary is None:
                raise


def capture(args, spec, frozen, monitors, raw, root_receipts):
    try:
        launch(args, spec, frozen, monitors)
    finally:
        primary = sys.exc_info()[1]
        path = args.output_root / spec["id"]
        if path.exists():
            try:
                raw[spec["id"]] = receipts(path)
                receipt = args.output_root / (spec["id"] + "-first-receipts.json")
                write(receipt, raw[spec["id"]])
                root_receipts[receipt.name] = {"sha256": sha(receipt), "bytes": receipt.stat().st_size}
            except BaseException as error:
                monitors.setdefault(spec["id"], {})["receipt_failure"] = {"type": type(error).__name__, "message": str(error)}
                if primary is None:
                    raise


def capture_protocol(args, frozen, deadline, aliases, dependency, evidence, root_receipts):
    try:
        return protocol.freeze(args, frozen, deadline, aliases, dependency, evidence)
    finally:
        primary = sys.exc_info()[1]
        try:
            root_receipts.update(receipts(args.output_root))
            file = args.output_root / "protocol-first-receipts.json"
            write(file, dict(root_receipts))
            root_receipts[file.name] = {"sha256": sha(file), "bytes": file.stat().st_size}
        except BaseException as error:
            if primary is None:
                raise
            primary.__dict__["receipt_failure"] = {"type": type(error).__name__, "message": str(error)}


def admit_disk(path, data, spec, frozen, deadline, monitor, initial, evidence):
    label = spec["id"] + ":disk"
    evidence.equal(label + ":first-receipts", receipts(path), initial)
    evidence.equal(label + ":native", read(path / "native.json"), data)
    evidence.equal(label + ":receipt", read(path / "receipt.json"), {"pid": monitor["pid"], "native_sha256": initial["native.json"]["sha256"]})
    evidence.equal(label + ":process", read(path / "process.json"), monitor)
    evidence.equal(label + ":process-start", read(path / "process-start.json"), {
        "command": monitor["command"], "cwd": monitor["cwd"],
        "environment_overrides": monitor["environment_overrides"], "pid": monitor["pid"],
        "sampled_max_current_rss_kib": 0, "exit_code": None, "stopping_reason": None})
    admit_step_streams(path, data, label, evidence)
    for name, key in (("construction-progress", "construction"), ("engine-evidence", "engine_evidence"),
                      ("runtime-evidence", "runtime_evidence"), ("observation-order", "observation_order")):
        evidence.equal(label + ":" + name, progress_rows(path / (name + ".jsonl")), data[key])
    evidence.equal(label + ":request-progress", progress_rows(path / "request-progress.jsonl"), request_rows(data))
    evidence.equal(label + ":source-before", read(path / "sources-before.json"), data["sources_before"])
    for index, cell in enumerate(data["cells"]):
        evidence.equal(label + f":batch:{index}", read(path / f"batch-{index}.json"), cell)
        evidence.equal(label + f":batch-input:{index}", read(path / f"batch-{index}-inputs.json"), cell["inputs"])
    after_fields = {"sources_after", "network_after", "network_observations", "clock_after_close_ps", "clock_owner_released"}
    preclose = {key: value for key, value in data.items() if key not in after_fields}
    if spec["kind"] != "compatibility":
        preclose["observation_order"] = data["observation_order"][:-1]
    evidence.equal(label + ":before-close", read(path / "pre-close-native.json"), preclose)
    summary = read(path / "partial-evidence.json")
    evidence.equal(label + ":completed-prefix-summary", summary, {
        "at_ps": data["final_clock_ps"], "runtime_failure": None, "capture_failures": [],
        "engine_positions": {owner: next(row["stops"] for row in reversed(data["engine_evidence"]) if row["engine_id"] == owner) for owner in data["sinks"]},
        "runtime_positions": data["runtime_evidence"][-1]["stops"]})
    for key, filename in (("checkpoints", "native-checkpoints.jsonl"), ("serialized_observations", "serialized-observations.jsonl"),
                          ("network_observations", "network-observations.jsonl")):
        if data[key]:
            evidence.equal(label + ":" + key, progress_rows(path / filename), data[key])
        else:
            evidence.check(label + ":absent:" + key, not (path / filename).exists())
    if data["serialized_bindings"]:
        evidence.equal(label + ":bindings", read(path / "serialized-bindings.json"), data["serialized_bindings"])
    else:
        evidence.check(label + ":bindings-off", not (path / "serialized-bindings.json").exists())
    for name in ("failure.json", "capture-failures.jsonl"):
        evidence.check(label + ":no-failure:" + name, not (path / name).exists())
    markers = [read(file) for file in sorted((path / "markers").glob("*.json"))]
    evidence.equal(label + ":markers", markers, monitor["markers"])
    expected_count = 0 if spec["kind"] == "compatibility" else 1
    evidence.equal(label + ":native-child-count", len(markers), expected_count)
    evidence.equal(label + ":child-state-count", len(monitor["child_states_after_exit"]), expected_count)
    evidence.check(label + ":all-native-children-reaped", all(row["exists"] is False for row in monitor["child_states_after_exit"]))
    for index, marker in enumerate(markers):
        evidence.fields(label + ":marker-fields:" + str(index), marker, "schema child_pid owner_pid run_nonce command_sha256 start_time_token")
        evidence.equal(label + ":marker-schema:" + str(index), marker["schema"], "simllm-child-lifetime-marker-v1")
        evidence.equal(label + ":marker-owner:" + str(index), marker["owner_pid"], monitor["pid"])
        evidence.equal(label + ":marker-nonce:" + str(index), marker["run_nonce"], spec["id"])
        evidence.equal(label + ":marker-child:" + str(index), marker["child_pid"], data["network_after"]["child_pid"])
        binary = monitor["command"][monitor["command"].index("--htsim-rnic") + 1]
        evidence.equal(label + ":marker-command:" + str(index), marker["command_sha256"], hashlib.sha256((binary + "\0--flow-session").encode()).hexdigest())
    wire = check_wire(path / "network-transcript", data, spec, frozen, deadline, evidence)
    evidence.equal(label + ":final-receipts", receipts(path), initial)
    return wire


def admit_step_streams(path, data, label, evidence):
    engines = [row["engine_id"] for row in data["retained_before"]]
    expected = ["engine-work/" + engine + "/step-records.jsonl" for engine in engines]
    evidence.equal(label + ":stream-domain", sorted(item.relative_to(path).as_posix()
                   for item in path.rglob("step-records.jsonl")), sorted(expected))
    for engine, relative in zip(engines, expected, strict=True):
        evidence.equal(label + ":native-step-stream:" + engine, step_stream(path / relative),
                       [step["record"] for step in data["steps"] if step["engine_id"] == engine])


def compare_complete(campaign, admitted, frozen, evidence):
    for timing in ("serialized", "independent"):
        for arm in ("off", "constant"):
            suffix = timing + "-" + arm
            evidence.equal("compatibility:" + suffix, admitted["after-" + suffix]["comparison"],
                           admitted["before-" + suffix]["comparison"], kind="oracles")
    for spec in frozen["native_processes"]:
        if len(spec["admission_times_ps"]) != 2:
            continue
        data = campaign[spec["id"]]
        vectors, events = [], []
        for index, cell in enumerate(data["cells"]):
            start = cell["admission_ps"]
            times = []
            for row in cell["requests"]:
                value = request_times(row["result"])
                for key in ("admitted_at_ps", "prefill_eligible_at_ps", "prefill_completed_at_ps", "decode_eligible_at_ps"):
                    value[key] -= start
                value["decode_token_completed_at_ps"] = [at - start for at in value["decode_token_completed_at_ps"]]
                times.append(value)
            vectors.append(times)
            events.append([[row["sequence"] - 16 * index, row["kind"], row["timestamp_ps"] - start]
                for row in data["network_after"]["events"] if 16 * index < row["sequence"] <= 16 * (index + 1)])
        evidence.equal(spec["id"] + ":persistent-relative-requests", vectors[1], vectors[0])
        evidence.equal(spec["id"] + ":persistent-relative-packets", events[1], events[0])
    inventory = dict.fromkeys(frozen["primary_structural_inventory"], 0)
    for spec in frozen["native_processes"]:
        if spec["kind"] != "shared":
            continue
        data = campaign[spec["id"]]
        _, _, count, _ = geometry(spec, frozen)
        network = data["network_after"]
        inventory["cells"] += len(data["cells"])
        inventory["requests"] += len(request_rows(data))
        inventory["flows"] += len(network["rows"])
        inventory["lifecycle_events"] += len(network["events"])
        inventory["data_packets"] += count * len(network["rows"])
        inventory["wire_bytes"] += count * len(network["rows"]) * frozen["backend"]["max_wire_packet_bytes"]
    evidence.equal("complete:primary-derived-packet-inventory", inventory, frozen["primary_structural_inventory"])
    check_relations(campaign, frozen, evidence)


def finalize(output, frozen, stages, evidence, monitors, first, root_receipts, campaign, catalog, failure):
    """Retain independent reporting surfaces without replacing the first failure."""
    reporting_failures, final, final_root = [], {}, {}

    def attempt(surface, callback):
        nonlocal failure
        try:
            return callback()
        except BaseException as error:  # noqa: BLE001, preserve the primary campaign failure.
            record = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            reporting_failures.append({"surface": surface, **record})
            failure = failure or record
            return None

    for name in first:
        retained = attempt("final-receipts:" + name, lambda name=name: receipts(output / name))
        if retained is not None:
            final[name] = retained
            attempt("final-integrity:" + name, lambda name=name, retained=retained:
                    evidence.equal("final:raw:" + name, retained, first[name]))
    for name, expected in root_receipts.items():
        file = output / name
        retained = attempt("final-root-receipt:" + name, lambda file=file:
                           {"sha256": sha(file), "bytes": file.stat().st_size})
        if retained is not None:
            final_root[name] = retained
            attempt("final-root-integrity:" + name, lambda name=name, retained=retained, expected=expected:
                    evidence.equal("final:root:" + name, retained, expected))
    checks = {kind: getattr(evidence, kind) for kind in ("guards", "oracles", "relations")}
    attempt("checks-write", lambda: write(output / "checks.json", checks))
    checks_sha = attempt("checks-digest", lambda: sha(output / "checks.json"))
    source_commit = attempt("source-commit", lambda: protocol.git(ROOT, "rev-parse", "HEAD"))
    manifest_sha = (attempt("source-manifest-digest", lambda: sha(output / "source-manifest.json"))
                    if (output / "source-manifest.json").is_file() else None)
    summary = {"schema": "simllm-shared-kv-handoff-study-v1", "verdict": "VOID" if failure else "PASS",
        "failure": failure, "reporting_failures": reporting_failures, "task": "CORE-71",
        "freeze_commit": FREEZE, "source_commit": source_commit, "source_manifest_sha256": manifest_sha,
        "amendment_commits": {"deadline": protocol.DEADLINE_FREEZE, "identity": protocol.ALIAS_FREEZE,
                              "service_vector": protocol.SERVICE_VECTOR_FREEZE},
        "finished_stages": sorted(evidence.finished_stages), "required_stages": stages,
        "monitors": monitors, "first_receipts": first, "final_receipts": final,
        "root_receipts": root_receipts, "final_root_receipts": final_root,
        "admitted_processes": list(campaign), "admitted_requests": sum(len(request_rows(data)) for data in campaign.values()),
        "component_catalog": catalog, "fatal_guards": len(evidence.guards), "exact_oracles": len(evidence.oracles),
        "behavioral_instances": len(evidence.relations),
        "behavioral_score": None if failure else {"passed": len(evidence.relations), "total": frozen["behavioral_instance_total"]},
        "checks_sha256": checks_sha}
    attempt("summary-write", lambda: write(output / "summary.json", plain(summary)))
    if failure is not None:
        summary.update(verdict="VOID", failure=failure, behavioral_score=None)
    print(json.dumps({key: summary[key] for key in (
        "verdict", "failure", "reporting_failures", "admitted_processes", "admitted_requests")}), flush=True)
    return summary


def execute(args):
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.absolute())
    output = args.output_root
    if output.is_relative_to(ROOT) or output.is_relative_to(args.before_repository):
        raise ValueError("raw campaign output must be outside the source repositories")
    output.mkdir(parents=True, exist_ok=False)
    frozen, deadline, aliases, dependency = protocol.inputs()
    specs = frozen["native_processes"]
    stages = ["protocol", "components", *[row["id"] for row in specs], "comparisons", "final-integrity"]
    evidence = Evidence(stages)
    sources, monitors, first, root_receipts, campaign, admitted = {}, {}, {}, {}, {}, {}
    failure, catalog = None, None
    try:
        sources = capture_protocol(args, frozen, deadline, aliases, dependency, evidence, root_receipts)
        evidence.finish("protocol")
        try:
            components.capture(output / "components")
        finally:
            file = output / "components-first-receipts.json"
            if file.is_file():
                first["components"] = read(file)
                root_receipts[file.name] = {"sha256": sha(file), "bytes": file.stat().st_size}
        catalog = components.admit(output / "components", frozen, aliases, evidence)
        file = output / "component-catalog.json"
        write(file, catalog)
        root_receipts[file.name] = {"sha256": sha(file), "bytes": file.stat().st_size}
        evidence.finish("components")
        for spec in specs:
            print("Starting " + spec["id"], flush=True)
            capture(args, spec, frozen, monitors, first, root_receipts)
            data = read(output / spec["id"] / "native.json")
            protocol.admit_sources(data, spec, args, frozen, sources, monitors[spec["id"]], evidence)
            wire = admit_disk(output / spec["id"], data, spec, frozen, deadline, monitors[spec["id"]], first[spec["id"]], evidence)
            admitted[spec["id"]] = admit(data, spec, frozen, dependency, deadline, evidence, wire)
            campaign[spec["id"]] = data
            evidence.finish(spec["id"])
            print("Qualified " + spec["id"], flush=True)
        compare_complete(campaign, admitted, frozen, evidence)
        evidence.finish("comparisons")
        evidence.equal("complete:process-count", len(monitors), frozen["native_process_count"])
        evidence.equal("complete:fresh-processes", len({row["pid"] for row in monitors.values()}), frozen["native_process_count"])
        evidence.equal("complete:request-count", sum(len(request_rows(data)) for data in campaign.values()), frozen["native_request_count"])
        evidence.equal("complete:oracle-vectors", len(evidence.oracles), sum(frozen["exact_oracle_vectors"].values()))
        evidence.equal("complete:behavioral-instances", len(evidence.relations), frozen["behavioral_instance_total"])
        protocol.locks(args, frozen, sources, evidence, "post-run")
        for name, values in first.items():
            evidence.equal("complete:raw:" + name, receipts(output / name), values)
        for name, expected in root_receipts.items():
            file = output / name
            evidence.equal("complete:root:" + name, {"sha256": sha(file), "bytes": file.stat().st_size}, expected)
        evidence.equal("complete:root-file-domain", sorted(path.name for path in output.iterdir() if path.is_file()), sorted(root_receipts))
        evidence.finish("final-integrity")
        evidence.equal("complete:stages", sorted(evidence.finished_stages), sorted(evidence.expected_stages))
    except BaseException as error:  # noqa: BLE001, preserve failed attempts and their complete receipts.
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if hasattr(error, "receipt_failure"):
            failure["receipt_failure"] = error.receipt_failure
    finally:
        summary = finalize(output, frozen, stages, evidence, monitors, first, root_receipts, campaign, catalog, failure)
    return int(summary["verdict"] != "PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before-repository", "native-python", "vllm-source", "hf-hub-cache", "htsim-rnic", "backend-repository",
                 "known-native-receipt", "known-backend-receipt", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    raise SystemExit(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
