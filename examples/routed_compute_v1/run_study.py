"""Admit the frozen routed-work grid through real device-runtime step calls."""

from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import json
import os
import subprocess
import sys
from fractions import Fraction
from itertools import product
from pathlib import Path

from examples.routed_compute_v1.inputs import GPU, campaign_config, step
from simllm.backends import DeviceRuntimeStepSink
from simllm.core import CoarseDeviceProfile, CoarseDeviceRuntime, VirtualClock
from simllm.core.execution_io import (
    execution_graph_from_json,
    execution_graph_to_json,
    execution_result_from_json,
    execution_result_to_json,
)
from simllm.core.step import step_record_to_json
from simllm.core.step_io import step_result_from_json, step_result_to_json

ROOT = Path(__file__).resolve().parents[2]
FREEZE = "fd97c1e627397aba6605742406a68c543fa6129c"
INPUT_FILES = (
    "examples/routed_compute_v1/expectations.md",
    "examples/routed_compute_v1/expectations.json",
)


def primitive(value):
    if dataclasses.is_dataclass(value):
        return {field.name: primitive(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, (list, tuple)):
        return [primitive(item) for item in value]
    if isinstance(value, dict):
        return {str(key): primitive(item) for key, item in value.items()}
    if value is None or type(value) in (bool, str, int, float):
        return value
    raise TypeError(f"unsupported evidence value {type(value).__name__}")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT).decode().strip()


def source_files():
    return {
        name: sha(ROOT / name)
        for name in git("ls-files").splitlines()
        if name.startswith(("simllm/", "examples/routed_compute_v1/", "tests/test_routed_compute"))
        and (ROOT / name).is_file()
    }


class Evidence:
    def __init__(self, root):
        root.mkdir(parents=True, exist_ok=False)
        self.root = root
        self.receipts = {}
        self.guards = []
        self.oracles = []
        self.relations = []
        self.journal = root / "first-receipts.jsonl"
        self.journal.touch(exist_ok=False)

    def write(self, name, value, *, record=True):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(primitive(value), stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if record:
            receipt = {"sha256": sha(path), "bytes": path.stat().st_size}
            self.receipts[name] = receipt
            with self.journal.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps({"path": name, **receipt}, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def guard(self, name, passed):
        self.guards.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise AssertionError(name)

    def oracle(self, name, passed):
        self.oracles.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise AssertionError(name)

    def relation(self, family, name, passed):
        self.relations.append({"family": family, "name": name, "passed": bool(passed)})
        if not passed:
            raise AssertionError(name)


def expert_service(count, routing, minimum):
    rows = count if routing == "hot" else count // 2
    return sum(max(int(factor * 16 * 32 * rows / 1_000_000_000 * 10**12), minimum) for factor in (4, 2))


def check_step(evidence, key, index, count, routing, minimum, payload):
    graph = payload["graph"]
    result = payload["step_result"]
    runtime = payload["runtime"]
    execution_graph_from_json(graph)
    execution_result_from_json(payload["execution"])
    step_result_from_json(result)
    operations = {row["operation_id"]: row for row in graph["operations"]}
    records = {row["operation_id"]: row for row in runtime["operations"]}
    evidence.guard(f"{key}:{index}:operation-domain", len(records) == len(runtime["operations"]) and set(records) == set(operations))
    evidence.guard(f"{key}:{index}:terminal", result["completed_at_ps"] == max(records[name]["completed_at_ps"] for name in graph["completion_operation_ids"]))
    expected_requests = {f"request-{request}" for request in range(count)}
    evidence.guard(f"{key}:{index}:request-identities", {row["request_id"] for row in result["request_metrics"]} == expected_requests)
    visits = [row for row in runtime["visits"] if row["resource"]["kind"] == "gpu-work-queue"]
    compute = [row for row in operations.values() if row["work"]["kind"] == "compute"]
    evidence.guard(f"{key}:{index}:visit-identities", len({row["operation_id"] for row in visits}) == len(visits))
    evidence.guard(f"{key}:{index}:visit-totality", len(visits) == len(compute))
    for visit in visits:
        operation = operations[visit["operation_id"]]
        evidence.guard(
            f"{key}:{index}:service:{operation['operation_id']}",
            visit["finished_at_ps"] - visit["started_at_ps"] == operation["work"]["nominal_duration_ps"],
        )
        for dependency in operation["depends_on"]:
            evidence.guard(
                f"{key}:{index}:causal:{operation['operation_id']}:{dependency}",
                visit["started_at_ps"] >= records[dependency]["completed_at_ps"],
            )
    owned = [row for row in compute if row["work"]["kernel"] in ("moe_gate_up", "moe_down")]
    ranks = (3,) if routing == "hot" else (0, 3)
    rows = count if routing == "hot" else count // 2
    expected = {
        (layer, rank, kernel): (
            factor * 16 * 32 * rows, weight_factor * 16 * 32 * 2,
            max(int(factor * 16 * 32 * rows / 1_000_000_000 * 10**12), minimum),
        )
        for layer, rank, (kernel, factor, weight_factor) in product(
            range(2), ranks, (("moe_gate_up", 4, 2), ("moe_down", 2, 1))
        )
    }
    actual = {}
    for operation in owned:
        work = operation["work"]
        config = dict(work["config"])
        identity = (operation["correlation"]["layer"], operation["rank"], work["kernel"])
        evidence.guard(f"{key}:{index}:unique:{identity}", identity not in actual)
        evidence.guard(f"{key}:{index}:rows:{identity}", config["routed_rows"] == rows and config["active_experts"] == 1)
        evidence.guard(f"{key}:{index}:epoch:{identity}", operation["placement_epoch"] == 0)
        actual[identity] = (work["flops"], work["hbm_bytes"], work["nominal_duration_ps"])
    evidence.oracle(f"{key}:{index}:expert-vector", actual == expected)
    nonexpert = [row for row in compute if row not in owned]
    evidence.guard(
        f"{key}:{index}:nonexpert",
        len(nonexpert) == 3 and all(row["rank"] == 0 for row in nonexpert)
        and sorted(row["work"]["kernel"] for row in nonexpert) == ["lm_head", "moe_pre_dispatch", "moe_pre_dispatch"],
    )
    critical = expert_service(count, routing, minimum)
    compute_sum = sum(row["work"]["nominal_duration_ps"] for row in compute)
    transfers = [row for row in runtime["visits"] if row["resource"]["kind"] == "nvlink"]
    transfer_bytes = sum(row["service_bytes"] for row in transfers)
    remote_rows = count if routing == "hot" else count // 2
    evidence.guard(f"{key}:{index}:wire", transfer_bytes == 4 * remote_rows * 16 * 2)
    evidence.guard(f"{key}:{index}:floor", result["step_latency_ps"] >= 2 * critical + transfer_bytes)
    evidence.guard(f"{key}:{index}:ceiling", result["step_latency_ps"] <= compute_sum + 2 * transfer_bytes)
    evidence.guard(f"{key}:{index}:request-domain", len(result["request_metrics"]) == count)
    evidence.guard(
        f"{key}:{index}:visibility",
        all(row["completed_at_ps"] == result["completed_at_ps"] for row in result["request_metrics"]),
    )
    return result["request_metrics"]


def compare(evidence, cells):
    def metric(row, name):
        value = row[name]
        return Fraction(value["numerator"], value["denominator"]) if isinstance(value, dict) else Fraction(value)

    for count, minimum in product((8, 16), (0, 5_000_000, 20_000_000)):
        balanced = cells[count, "balanced", minimum]
        hot = cells[count, "hot", minimum]
        delta = 2 * (expert_service(count, "hot", minimum) - expert_service(count, "balanced", minimum))
        extra_wire = 4 * (count // 2) * 16 * 2
        passed = all(delta <= metric(hot, name) - metric(balanced, name) <= delta + 2 * extra_wire for name in ("ttft_ps", "tpot_ps"))
        if delta:
            evidence.relation("load_imbalance", f"load:{count}:{minimum}", passed)
        else:
            evidence.guard(f"minimum-forced-equal-service:{count}:{minimum}", passed)
    pairs = [(8, "balanced", 0, 5_000_000)]
    pairs.extend((count, route, 5_000_000, 20_000_000) for count, route in product((8, 16), ("balanced", "hot")))
    for count, route, before, after in pairs:
        delta = 2 * (expert_service(count, route, after) - expert_service(count, route, before))
        evidence.relation(
            "minimum_service", f"minimum:{count}:{route}:{before}:{after}",
            delta > 0 and all(
                metric(cells[count, route, after], name) - metric(cells[count, route, before], name) == delta
                for name in ("ttft_ps", "tpot_ps")
            ),
        )
    for route, minimum in product(("balanced", "hot"), (0, 5_000_000, 20_000_000)):
        evidence.relation(
            "request_count", f"count:{route}:{minimum}",
            all(
                metric(cells[8, route, minimum], name) < metric(cells[16, route, minimum], name)
                <= 2 * metric(cells[8, route, minimum], name)
                for name in ("ttft_ps", "tpot_ps")
            ),
        )


def run(output):
    evidence = Evidence(output)
    failure = None
    source_commit = None
    cells = {}
    try:
        evidence.guard("clean-source", git("status", "--porcelain") == "")
        source_commit = git("rev-parse", "HEAD")
        subprocess.run(["git", "merge-base", "--is-ancestor", FREEZE, source_commit], cwd=ROOT, check=True)
        for name in INPUT_FILES:
            original = subprocess.check_output(["git", "show", f"{FREEZE}:{name}"], cwd=ROOT)
            evidence.guard(f"frozen:{name}", (ROOT / name).read_bytes() == original)
        sources = source_files()
        frozen = json.loads((ROOT / INPUT_FILES[1]).read_text())
        evidence.write("source-before.json", {"commit": source_commit, "files": sources})
        evidence.write("command.json", {"argv": sys.argv, "python": sys.executable})
        for count, route, minimum in product((8, 16), ("balanced", "hot"), (0, 5_000_000, 20_000_000)):
            key = f"requests-{count}-{route}-minimum-{minimum}"
            config = campaign_config(count, route, minimum)
            evidence.guard(f"{key}:gpu", config.gpu == GPU and config.provider.efficiency == 1)
            evidence.guard(
                f"{key}:geometry",
                all(getattr(config.dims, name) == value for name, value in frozen["model"].items()),
            )
            evidence.guard(f"{key}:selected-minimum", config.routed_compute.gemm_minimum_ps == minimum)
            evidence.write(f"{key}/inputs.json", {
                "dims": config.dims, "gpu": config.gpu, "routed_compute": config.routed_compute,
                "routing": config.routed_moe_supply,
                "routing_evidence_kind": "synthetic",
            })
            sink = DeviceRuntimeStepSink(config, runtime=CoarseDeviceRuntime(
                CoarseDeviceProfile(rnic_rate_bps=8_000_000_000_000, nvlink_rate_bps=8_000_000_000_000)
            ))
            sink.bind_clock(VirtualClock())
            for index in range(3):
                record = step(config, index, sink.clock.now_ps)
                evidence.write(f"{key}/step-{index}-input.json", step_record_to_json(record))
                result = sink(record, None)
                outcome = sink.outcomes[-1]
                payload = {
                    "graph": execution_graph_to_json(outcome.graph),
                    "execution": execution_result_to_json(outcome.execution_result),
                    "runtime": primitive(outcome.runtime_report),
                    "step_result": step_result_to_json(result),
                }
                evidence.write(f"{key}/step-{index}-output.json", payload)
                requests = check_step(evidence, key, index, count, route, minimum, payload)
                evidence.guard(f"{key}:{index}:uniform-metrics", len({json.dumps((r["ttft_ps"], r["tpot_ps"]), sort_keys=True) for r in requests}) == 1)
            cells[count, route, minimum] = {
                "ttft_ps": requests[0]["ttft_ps"], "tpot_ps": requests[0]["tpot_ps"],
                "last_completion_ps": result.completed_at_ps,
            }
        compare(evidence, cells)
        evidence.guard("complete-oracles", len(evidence.oracles) == 36)
        evidence.guard("complete-behavior", len(evidence.relations) == 16)
        evidence.guard("source-continuity", source_files() == sources)
        for name, module in tuple(sys.modules.items()):
            if name == "simllm" or name.startswith("simllm."):
                location = getattr(module, "__file__", None)
                evidence.guard(f"origin:{name}", location is not None and Path(location).resolve().is_relative_to(ROOT / "simllm"))
    except BaseException as error:  # noqa: BLE001 - preserve interrupted or failed campaign evidence
        failure = {"type": type(error).__name__, "message": str(error)}
    first = dict(evidence.receipts)
    try:
        actual_files = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
        evidence.guard("complete-file-domain", actual_files == set(first) | {"first-receipts.jsonl"})
        journal = [json.loads(line) for line in evidence.journal.read_text().splitlines()]
        evidence.guard("first-receipt-journal", journal == [{"path": name, **receipt} for name, receipt in first.items()])
        for name, receipt in first.items():
            path = output / name
            evidence.guard(f"retained:{name}", path.stat().st_size == receipt["bytes"] and sha(path) == receipt["sha256"])
        first["first-receipts.jsonl"] = {
            "sha256": sha(evidence.journal), "bytes": evidence.journal.stat().st_size,
        }
    except BaseException as error:  # noqa: BLE001 - preserve the first failure through final integrity
        failure = failure or {"type": type(error).__name__, "message": str(error)}
    summary = {
        "schema": "simllm-routed-compute-study-v1",
        "verdict": "VOID" if failure else "PASS", "failure": failure,
        "freeze_commit": FREEZE, "source_commit": source_commit,
        "behavioral_score": None if failure else {"passed": len(evidence.relations), "total": 16},
        "exact_vectors": len(evidence.oracles), "fatal_guards": len(evidence.guards),
        "fatal_violations": [row["name"] for row in evidence.guards if not row["passed"]],
        "cells": [{"requests": count, "routing": route, "minimum_ps": minimum, **row} for (count, route, minimum), row in cells.items()],
        "raw_file_receipts": first,
        "hardware_measurement": False,
    }
    summary["reporting_failures"] = []
    try:
        evidence.write("checks.json", {
            "guards": evidence.guards, "oracles": evidence.oracles, "relations": evidence.relations,
        }, record=False)
        summary["checks_sha256"] = sha(output / "checks.json")
    except BaseException as error:  # noqa: BLE001 - attempt the independent summary after a reporting failure
        summary["reporting_failures"].append({"stage": "checks", "type": type(error).__name__, "message": str(error)})
        summary["verdict"] = "VOID"
        summary["behavioral_score"] = None
    try:
        evidence.write("summary.json", summary, record=False)
    except BaseException as error:  # noqa: BLE001 - retain the original failure in the console receipt
        summary["reporting_failures"].append({"stage": "summary", "type": type(error).__name__, "message": str(error)})
        summary["verdict"] = "VOID"
        summary["behavioral_score"] = None
    print(json.dumps({key: summary[key] for key in ("verdict", "failure", "exact_vectors", "behavioral_score")}))
    return 0 if summary["verdict"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.output.resolve()))


if __name__ == "__main__":
    main()
