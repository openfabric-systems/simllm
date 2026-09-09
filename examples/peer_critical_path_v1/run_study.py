"""Admit original packet and request evidence against the frozen relations."""

# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from fractions import Fraction
from itertools import product
from pathlib import Path

from examples.local_peer_packet_runtime_v1.run_study import session_findings
from examples.peer_critical_path_v1.capture import source
from examples.peer_critical_path_v1.inputs import input_record, inputs, record
from examples.routed_compute_v1.run_study import Evidence, primitive
from simllm.backends.packet_breakdown import packet_step_from_json, sum_breakdowns
from simllm.backends.peer_critical_path import validate_packet_projection
from simllm.core.execution_io import execution_graph_from_json, execution_result_from_json
from simllm.core.step import step_record_to_json
from simllm.core.step_io import step_result_from_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREEZE = "2919cf2d"


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON member")
            result[key] = value
        return result
    raw = path.read_bytes()
    def invalid_constant(_):
        raise ValueError("nonfinite JSON")
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
    expected = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if raw != expected:
        raise ValueError("noncanonical captured JSON: " + path.name)
    return value


def receipt(path):
    blob = path.read_bytes()
    return {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}


def seal(evidence, path):
    name = path.relative_to(evidence.root).as_posix()
    row = receipt(path)
    if name in evidence.receipts:
        if row != evidence.receipts[name]:
            raise ValueError("first receipt changed: " + name)
        return
    evidence.receipts[name] = row
    with evidence.journal.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"path": name, **row}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def strip_reporting(value):
    value = copy.deepcopy(value)
    if isinstance(value, dict):
        return {key: strip_reporting(item) for key, item in value.items()
                if key not in {"critical_path", "switch_implementation", "switch_library_sha256"}}
    if isinstance(value, list):
        return [strip_reporting(item) for item in value]
    return value


def capture_process(evidence, stage, command, cwd, *, mutable_build=None):
    evidence.write(stage + "-command.json", {"command": command, "cwd": str(cwd)})
    run = subprocess.run(command, cwd=cwd, capture_output=True, check=False, env={
        **os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
    })
    original = None if run.returncode == 0 else RuntimeError(f"{stage}: process exit {run.returncode}")
    try:
        for name, blob in (("stdout", run.stdout), ("stderr", run.stderr)):
            path = evidence.root / f"{stage}.{name}"
            with path.open("xb") as stream:
                stream.write(blob)
                stream.flush()
                os.fsync(stream.fileno())
            seal(evidence, path)
        # Seal all child bytes before checking exit status or decoding evidence.
        for path in sorted(evidence.root.rglob("*")):
            if path.is_file() and path != evidence.journal and not (
                mutable_build is not None and path.is_relative_to(mutable_build)
            ):
                seal(evidence, path)
    except Exception as error:
        if original is None:
            raise
        original.reporting_failures = [{"type": type(error).__name__, "message": str(error)}]
        raise original from error
    if original is not None:
        raise original
    evidence.guard(stage + ":exit", True)


def native_binding(evidence, build, library):
    files = tuple(build.rglob("CMakeCXXCompiler.cmake"))
    evidence.guard("one-native-compiler-record", len(files) == 1)
    text = files[0].read_text(encoding="utf-8")
    values = {}
    for name in ("CMAKE_CXX_COMPILER", "CMAKE_CXX_COMPILER_ID", "CMAKE_CXX_COMPILER_VERSION"):
        match = re.search(r'set\(' + name + r' "([^"\n]+)"\)', text)
        evidence.guard("native-compiler:" + name, match is not None)
        values[name] = match.group(1)
    compiler = Path(values["CMAKE_CXX_COMPILER"]).resolve()
    evidence.write("native-binding.json", {
        "compiler": {**values, "resolved_binary": str(compiler), **receipt(compiler)},
        "compiler_record": {"path": files[0].relative_to(evidence.root).as_posix(), **receipt(files[0])},
        "library": {"path": library.relative_to(evidence.root).as_posix(), **receipt(library)},
        "configure_command": receipt(evidence.root / "native-configure-command.json"),
        "build_command": receipt(evidence.root / "native-build-command.json"),
        "source_manifest": receipt(evidence.root / "source-before.json"),
    })


def admit_capture(evidence, directory, *, arm, topology, rate, donors, payload,
                  expected_source, code_root):
    captured = read(directory / "capture.json")
    evidence.guard(directory.name + ":captured", captured["verdict"] == "CAPTURED" and captured["failure"] is None)
    first = captured["first_receipts"]
    evidence.guard(directory.name + ":safe-receipt-paths", all(
        isinstance(name, str) and name and not Path(name).is_absolute()
        and ".." not in Path(name).parts and "\\" not in name
        and Path(name).as_posix() == name for name in first
    ))
    journal = [json.loads(line) for line in (directory / "first-receipts.jsonl").read_bytes().splitlines()]
    evidence.guard(directory.name + ":receipt-journal",
                   set(captured["receipt_order"]) == set(first)
                   and len(captured["receipt_order"]) == len(first)
                   and journal == [{"path": name, **first[name]} for name in captured["receipt_order"]])
    for name, row in first.items():
        evidence.guard(str(directory.name) + ":first:" + name, receipt(directory / name) == row)
    evidence.guard(directory.name + ":complete-file-domain", {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
    } == set(first) | {"capture.json", "first-receipts.jsonl"})
    for boundary in ("before", "after"):
        evidence.guard(directory.name + ":source-" + boundary,
                       read(directory / f"source-{boundary}.json") == expected_source)
    origins = read(directory / "loaded-origins.json")
    evidence.guard(directory.name + ":loaded-origin-domain",
                   bool(origins["modules"]) and set(origins["sha256"]) == set(origins["modules"]))
    for name, path in origins["modules"].items():
        path = Path(path)
        evidence.guard(directory.name + ":origin:" + name, path.is_relative_to(code_root))
        evidence.guard(directory.name + ":loaded-source:" + name,
                       expected_source["files"].get(path.relative_to(code_root).as_posix()) == origins["sha256"][name])
    config, profile, binding, options, transfers = inputs(
        topology, rate, donors, payload, directory / "unused", reported=arm.endswith("-on"),
    )
    evidence.guard(directory.name + ":frozen-inputs", read(directory / "inputs.json") == primitive(input_record(
        config, profile, binding, options, transfers, arm=arm,
        topology=topology, rate=rate, donors=donors, payload=payload,
    )))
    q = 272 * 10**12 // rate
    m, hops = payload // 256, 1 if topology == "direct" else 2
    completion = (donors * m + hops) * q + hops * 1000
    observations = [read(directory / name) for name in ("component-visible.json", "component-drained.json")]
    for label, row in zip(("visible", "drained"), observations, strict=True):
        errors = session_findings(row, final=label == "drained")
        evidence.guard(directory.name + ":" + label + ":physical-invariants", not errors)
        selected = arm.endswith("-on")
        evidence.guard(directory.name + ":" + label + ":selected-report", ("critical_path" in row) == selected)
        if selected:
            _, paths = validate_packet_projection(row)
            evidence.guard(directory.name + ":" + label + ":one-component-path", len(paths) == 1)
            evidence.guard(directory.name + ":" + label + ":critical-duration", paths[0].duration_ps == completion)
        actual = max(packet["visible_at_ps"] for packet in row["packets"])
        evidence.guard(directory.name + ":" + label + ":endpoint-floor", actual * rate >= donors * m * 272 * 10**12)
        serial_ceiling = sum(
            (visit["finished_at_ps"] - visit["started_at_ps"]) +
            (visit["completed_at_ps"] - visit["finished_at_ps"])
            for visit in row["resource_visits"]
        )
        evidence.guard(directory.name + ":" + label + ":serial-ceiling", actual <= serial_ceiling)
        if donors == 3 and m == 4:
            evidence.guard(directory.name + ":" + label + ":work-is-not-wall", serial_ceiling > actual)
    reported_steps, breakdowns, outputs, cursor = [], [], [], 0
    for index in range(3):
        output = read(directory / f"step-{index}-output.json")
        result = step_result_from_json(output["result"])
        graph = execution_graph_from_json(output["graph"])
        execution = execution_result_from_json(output["execution_result"])
        evidence.guard(directory.name + f":step-{index}:original-request-boundary",
                       graph.execution_id == execution.execution_id and result.completed_at_ps == execution.completed_at_ps
                       and result.step_latency_ps == result.completed_at_ps - graph.released_at_ps
                       and read(directory / f"step-{index}-input.json") == output["record"]
                       and output["record"] == step_record_to_json(record(index, cursor)))
        evidence.guard(directory.name + f":step-{index}:one-physical-session", len(output["sessions"]) == 1)
        for session in output["sessions"]:
            evidence.guard(directory.name + f":step-{index}:physical-invariants", not session_findings(session, final=False))
            if arm.endswith("-on"):
                _, paths = validate_packet_projection(session)
                own = [path for path in paths if path.execution_id == graph.execution_id]
                evidence.guard(directory.name + f":step-{index}:dispatch-combine-paths",
                               len(own) == 2 and all(path.duration_ps == completion for path in own))
        if arm.endswith("-on"):
            reported, breakdown = packet_step_from_json(read(directory / f"step-{index}-breakdown.json"))
            evidence.guard(directory.name + f":step-{index}:packet-breakdown",
                           reported == result and breakdown is not None)
            breakdowns.append(breakdown.breakdown)
        reported_steps.append(result.step_latency_ps)
        outputs.append(output)
        cursor = result.completed_at_ps
    metrics = read(directory / "metrics.json")
    totals = metrics["request_totals"]
    tpot = Fraction(totals["tpot_ps"]["numerator"], totals["tpot_ps"]["denominator"])
    expected_step = 37000 + 2 * completion
    evidence.guard(directory.name + ":request-reduction",
                   totals["request_id"] == "peer-star" and totals["arrived_at_ps"] == 0
                   and totals["token_count"] == 3 and totals["ttft_ps"] == reported_steps[0]
                   and totals["first_token_at_ps"] == reported_steps[0]
                   and totals["last_token_at_ps"] == metrics["job_completion_ps"] == cursor
                   and tpot == Fraction(sum(reported_steps[1:]), 2))
    if arm.endswith("-on"):
        histories = read(directory / "request-breakdowns.json")
        evidence.guard(directory.name + ":request-breakdown-history", histories == primitive({
            "peer-star": (breakdowns[0], sum_breakdowns(*breakdowns[1:])),
        }))
    if arm == "candidate-on":
        evidence.oracle(directory.name + ":component-completion",
                        max(packet["visible_at_ps"] for packet in observations[0]["packets"]) == completion)
        evidence.oracle(directory.name + ":request-vector",
                        totals["ttft_ps"] == tpot == expected_step and metrics["job_completion_ps"] == 3 * expected_step
                        and reported_steps == [expected_step] * 3)
    for row in read(directory / "serving-drained.json"):
        evidence.guard(directory.name + ":serving-drain", not session_findings(row, final=True))
        if arm.endswith("-on"):
            validate_packet_projection(row)
    return {"component": observations, "steps": outputs, "metrics": metrics,
            "drained": read(directory / "serving-drained.json")}


def compare(evidence, cells):
    def metric(key, field):
        row = cells[key]["metrics"]["request_totals"][field]
        return Fraction(row["numerator"], row["denominator"]) if isinstance(row, dict) else Fraction(row)
    for topology, donors, payload in product(("direct", "switched"), (1, 3), (256, 1024)):
        keys = ((topology, rate, donors, payload) for rate in (12500000000, 25000000000))
        slow, fast = keys
        fixed = 37000 + (2000 if topology == "direct" else 4000)
        evidence.relation("rate", str((topology, donors, payload)),
                          all(metric(slow, field) - fixed == 2 * (metric(fast, field) - fixed)
                              for field in ("ttft_ps", "tpot_ps")))
    for topology, rate, donors in product(("direct", "switched"), (12500000000, 25000000000), (1, 3)):
        small, large = (topology, rate, donors, 256), (topology, rate, donors, 1024)
        q = 272 * 10**12 // rate
        evidence.relation("payload", str((topology, rate, donors)),
                          all(metric(large, field) - metric(small, field) == 6 * donors * q
                              for field in ("ttft_ps", "tpot_ps")))
    for topology, rate, payload in product(("direct", "switched"), (12500000000, 25000000000), (256, 1024)):
        one, three = (topology, rate, 1, payload), (topology, rate, 3, payload)
        q, m = 272 * 10**12 // rate, payload // 256
        evidence.relation("donors", str((topology, rate, payload)),
                          all(metric(three, field) - metric(one, field) == 4 * m * q
                              for field in ("ttft_ps", "tpot_ps")))


def run(output, baseline):
    evidence = Evidence(output)
    failure, cells, stages, reporting = None, {}, [], []
    try:
        before, old = source(ROOT), source(baseline)
        frozen = json.loads((HERE / "expectations.json").read_bytes())
        freeze = subprocess.check_output(["git", "rev-parse", FREEZE], cwd=ROOT).decode().strip()
        subprocess.run(["git", "merge-base", "--is-ancestor", freeze, before["commit"]], cwd=ROOT, check=True)
        for name in ("expectations.md", "expectations.json"):
            blob = subprocess.check_output(["git", "show", f"{freeze}:examples/peer_critical_path_v1/{name}"], cwd=ROOT)
            evidence.guard("frozen:" + name, blob == (HERE / name).read_bytes())
        evidence.guard("baseline-identity", old["commit"] == frozen["baseline_commit"])
        evidence.write("source-before.json", {"candidate": before, "baseline": old, "freeze_commit": freeze})
        evidence.write("expectations.json", frozen)

        def process(stage, command, cwd):
            capture_process(evidence, stage, command, cwd,
                            mutable_build=build if stage == "native-configure" else None)
            stages.append(stage)

        build = output / "native-build"
        process("native-configure", ["cmake", "-S", str(ROOT / "simllm/backends/nvswitch"),
                                    "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"], ROOT)
        # Build files are mutable until compilation finishes; retain configure
        # output separately, then seal the final compiler/library inventory.
        process("native-build", ["cmake", "--build", str(build), "--config", "Release", "--parallel", "2"], ROOT)
        names = {"libsimllm_nvswitch.so", "libsimllm_nvswitch.dylib", "simllm_nvswitch.dll"}
        libraries = [path for path in build.rglob("*") if path.name in names]
        evidence.guard("one-native-library", len(libraries) == 1)
        library = libraries[0]
        native_binding(evidence, build, library)
        for topology, rate, donors, payload in product(
            frozen["topologies"], frozen["rate_bytes_per_second"], frozen["donors"], frozen["payload_bytes"]
        ):
            key = (topology, rate, donors, payload)
            label = f"{topology}-rate-{rate}-donors-{donors}-payload-{payload}"
            arms = [*frozen["python_arms"], *([frozen["native_switched_arm"]] if topology == "switched" else [])]
            baseline_result = None
            for arm in arms:
                stage = label + "-" + arm
                directory = output / stage
                code = baseline if arm == "baseline-off" else ROOT
                commit = old["commit"] if arm == "baseline-off" else before["commit"]
                command = [sys.executable, str(HERE / "capture.py"), "--code-root", str(code),
                           "--expected-commit", commit, "--output", str(directory), "--arm", arm,
                           "--topology", topology, "--rate", str(rate), "--donors", str(donors),
                           "--payload", str(payload)]
                if arm == "candidate-native-on":
                    command.extend(("--native-library", str(library)))
                process(stage, command, code)
                row = admit_capture(evidence, directory, arm=arm, topology=topology,
                                    rate=rate, donors=donors, payload=payload,
                                    expected_source=old if arm == "baseline-off" else before,
                                    code_root=code)
                if baseline_result is None:
                    baseline_result = strip_reporting(row)
                else:
                    evidence.guard(stage + ":exact-predecessor-output", strip_reporting(row) == baseline_result)
                if arm == "candidate-on":
                    cells[key] = row
            print("Qualified " + label, flush=True)
        compare(evidence, cells)
        evidence.guard("required-stages", len(stages) == 58 and len(set(stages)) == 58)
        evidence.guard("exact-oracle-domain", len(evidence.oracles) == frozen["exact_oracle_vectors"])
        evidence.guard("behavioral-domain", len(evidence.relations) == frozen["behavioral_instances"])
        evidence.guard("candidate-source-continuity", source(ROOT) == before)
        evidence.guard("baseline-source-continuity", source(baseline) == old)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        reporting.extend(getattr(error, "reporting_failures", ()))
    try:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path != evidence.journal:
                seal(evidence, path)
        actual = {path.relative_to(output).as_posix(): receipt(path)
                  for path in output.rglob("*") if path.is_file()}
        expected = {**evidence.receipts, evidence.journal.name: receipt(evidence.journal)}
        evidence.guard("complete-final-file-domain", actual == expected)
    except Exception as error:
        if failure is None:
            failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        else:
            reporting.append({"type": type(error).__name__, "message": str(error)})
    summary = {
        "schema": "simllm-peer-critical-path-study-v1", "verdict": "PASS" if failure is None else "VOID",
        "failure": failure, "reporting_failures": reporting, "stages": stages,
        "behavioral_score": {"passed": 24, "total": 24} if failure is None else None,
        "fatal_guard_count": len(evidence.guards), "exact_oracle_count": len(evidence.oracles),
        "source_commit": before["commit"] if "before" in locals() else None,
        "expectations_commit": freeze if "freeze" in locals() else None,
        "raw_file_receipts": {**evidence.receipts, evidence.journal.name: receipt(evidence.journal)},
        "cells": [{"topology": key[0], "rate": key[1], "donors": key[2], "payload": key[3],
                   **row["metrics"]} for key, row in cells.items()],
    }
    for name, value in (("checks.json", {"guards": evidence.guards, "oracles": evidence.oracles, "relations": evidence.relations}),
                        ("summary.json", summary)):
        try:
            evidence.write(name, value, record=False)
            if name == "checks.json":
                summary["checks_sha256"] = receipt(output / name)["sha256"]
        except Exception as error:
            reporting.append({"type": type(error).__name__, "message": str(error)})
            summary["verdict"], summary["behavioral_score"] = "VOID", None
            if summary["failure"] is None:
                summary["failure"] = {"type": type(error).__name__, "message": str(error),
                                      "traceback": traceback.format_exc()}
    print(json.dumps(primitive(summary), sort_keys=True), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output.resolve(), args.baseline_root.resolve())
    raise SystemExit(0 if result["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
