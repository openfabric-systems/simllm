"""Run the frozen atomic-successor controls and compile independent negative copies."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

FREEZE = "dbb60ad3fc9f864a372188c143b44b282b22530c"
EXPECTATION_SHA = "d505bdd1ba040d47bb7d6a8a56fff620cd480b529b8a24e3e0e3e4131c28be4b"
BASE = "e64e0fdf6afaf2e600ca5cf5ee9f3534f785954b"
RX = "simllm/backends/rnic"
COMMANDS: list[dict] = []


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(command: list[str], log: Path, timeout: int, *, fixture: str = "") -> None:
    if log.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {log}")
    record = {"command": command, "log": str(log), "guard_seconds": timeout}
    COMMANDS.append(record)
    start = time.monotonic()
    with log.open("x") as stream:
        try:
            with subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                  start_new_session=True) as process:
                try:
                    record["exit_code"] = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # The build may own compiler children. The entire isolated
                    # process group must stop before any timeout is reported.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    record["terminated_process_group"] = process.pid
                    record["terminated_exit_code"] = process.wait()
                    record["exit_code"] = "TIMEOUT"
                    raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - start
            write_json(log.with_suffix(log.suffix + ".command.json"), record)
    text = log.read_text()
    if fixture:
        if record["exit_code"] == 0 or f"FAIL: {fixture}:" not in text:
            raise RuntimeError(f"negative copy missed named fixture {fixture}: {log}")
    elif record["exit_code"] != 0:
        raise RuntimeError(f"unexpected exit {record['exit_code']}: {log}")


def configure(source: Path, case: Path, guard: int) -> None:
    run(["cmake", "-S", str(source), "-B", str(case / "build"),
         "-DSIMLLM_RNIC_WARNINGS_AS_ERRORS=ON"], case / "configure.log", guard)


def manifest(source: Path) -> dict[str, str]:
    return {str(p.relative_to(source)): sha(p)
            for p in sorted(source.rglob("*")) if p.is_file()}


def replace_once(text: str, old: str, new: str, name: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"nonunique mutation anchor: {name}")
    return text.replace(old, new)


def mutate(source: Path, name: str) -> str:
    path = source / "src/rnic_rx_pipeline.cpp"
    text = path.read_text()
    if name == "eager_clear":
        text = replace_once(text, "            same_time_credit_ = SameTimeCredit{now_ps, service_credit_};",
                            "            same_time_credit_.reset();", name)
        fixture = "exact_replacement"
    elif name == "stale_candidate":
        old = """    if (same_time_credit_.has_value()
        && now_ps > same_time_credit_->timestamp_ps) {
        // Even an off-grid positive empty interval ends the earned epoch.
        same_time_credit_.reset();
    }
"""
        text = replace_once(text, old, "", name)
        text = replace_once(text, """        if (same_time_credit_.has_value()
            && same_time_credit_->timestamp_ps == now_ps) {""",
                            "        if (same_time_credit_.has_value()) {", name)
        fixture = "positive_off_grid_progress"
    elif name == "duplicate_tick":
        old = "        service_credit_ = 0;\n        clockAfter(now_ps);"
        new = """        service_credit_ = 0;
        if (same_time_credit_.has_value()) next_tick_ps_ = now_ps;
        else clockAfter(now_ps);"""
        text = replace_once(text, old, new, name)
        fixture = "repeat_query_progress"
    elif name == "reject_activates_candidate":
        old = """    drainTo(now_ps);
    if (packet.wire_bytes == 0"""
        new = """    drainTo(now_ps);
    if (same_time_credit_.has_value()) service_credit_ = same_time_credit_->numerator;
    if (packet.wire_bytes == 0"""
        text = replace_once(text, old, new, name)
        fixture = "invalid_wire_reject_only"
    elif name == "tail_pack":
        old = "        clockAfter(*next);"
        new = """        if (bytes < config_.service.beat_bytes && hasPendingService()) {
            const auto packed = std::min({config_.service.beat_bytes - bytes,
                                         service_records_.front(),
                                         service_credit_ / kByteCreditDenominator});
            if (service_credit_ >= packed * kByteCreditDenominator) {
                service_credit_ -= packed * kByteCreditDenominator;
                occupancy_bytes_ -= packed;
                counters_.ingress_occupancy_bytes = occupancy_bytes_;
                service_bytes_ += packed;
                service_records_.front() -= packed;
                if (service_records_.front() == 0) service_records_.pop_front();
            }
        }
        clockAfter(*next);"""
        text = replace_once(text, old, new, name)
        fixture = "already_queued"
    else:
        raise ValueError(name)
    path.write_text(text)
    return fixture


def check_rows(path: Path, expected: dict) -> dict:
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert all(r["verdict"] == "PASS" for r in rows)
    frontends, widths = {"native", "facade"}, {64, 128}
    exact = {(frontend, width, row["name"]): row
             for frontend in frontends for width in widths for row in expected["exact_rows"]}
    fatal = {(frontend, width, row["name"])
             for frontend in frontends for width in widths for row in expected["fatal_controls"]}
    component = {(frontend, row["width_bytes"], row["rate_bps"]): row
                 for frontend in frontends for row in expected["two_parameter_component_rows"]}
    seen: set[tuple] = set()
    deadlines = {}
    for row in rows:
        key = (row["frontend"], int(row["width_bytes"]), row["name"])
        assert (row["class"], key, row["rate_bps"]) not in seen
        seen.add((row["class"], key, row["rate_bps"]))
        if row["class"] == "exact":
            want = exact.pop(key)
            assert [int(v) for v in row["debit_times_ps"].split(";")] == want["debit_times_ps"]
            assert [int(v) for v in row["debit_bytes"].split(";")] == want["debit_bytes"]
            assert int(row["completed_ps"]) == want["debit_times_ps"][-1]
        elif row["class"] == "fatal":
            fatal.remove(key)
        else:
            frontend, width = key[:2]
            rate = int(row["rate_bps"])
            want = component.pop((frontend, width, rate))
            completed = int(row["completed_ps"])
            assert completed == want["successor_completed_ps"]
            assert int(row["service_debits"]) == want["service_debits"]
            # Independent useful-byte affordability and one-clock-per-beat floor.
            refill = rate * 5000
            affordability = (106 * 8_000_000_000_000 + refill - 1) // refill
            geometry = 1 + (85 + width - 1) // width
            assert completed == 2500 + max(affordability, geometry) * 5000
            deadlines[frontend, width, rate] = completed
    assert not exact and not fatal and not component
    for frontend in frontends:
        for width in widths:
            values = [deadlines[frontend, width, rate]
                      for rate in (40_000_000_000, 96_600_000_000, 160_000_000_000)]
            assert values == sorted(values, reverse=True)
        for rate, delta in zip((40_000_000_000, 96_600_000_000, 160_000_000_000), (0, 5000, 5000)):
            assert deadlines[frontend, 64, rate] - deadlines[frontend, 128, rate] == delta
    return {"exact_oracle_rows": 14, "component_exact_rows": 6,
            "fatal_control_families": 14, "frontends": sorted(frontends),
            "behavioral_relation_families": 2, "behavioral_relation_instances": 5,
            "observations": len(rows), "physics_bounds": "PASS", "verdict": "PASS"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mutants", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output == root or root in output.parents or output.exists():
        parser.error("output must be a new directory outside the repository")
    expectation = Path(__file__).with_name("expectations.json")
    working_expectations = expectation.read_bytes()
    frozen_expectations = subprocess.run(
        ["git", "show", f"{FREEZE}:{expectation.relative_to(root)}"], cwd=root,
        check=True, stdout=subprocess.PIPE, timeout=30,
    ).stdout
    if (working_expectations != frozen_expectations
            or hashlib.sha256(frozen_expectations).hexdigest() != EXPECTATION_SHA):
        raise RuntimeError("expectations do not match the exact frozen Git commit blob")
    expected = json.loads(working_expectations)
    guards = expected["guards_seconds"]
    output.mkdir(parents=True)
    write_json(output / "guards.json", {"freeze": FREEZE, "expectation_sha256": EXPECTATION_SHA,
                                       "frozen_commit_blob_verified": True,
                                       "guards_seconds": guards})
    source = root / RX
    pins = manifest(source)
    tools = {tool: subprocess.run([tool, "--version"], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, check=True).stdout
             for tool in ("cmake", "ctest", "c++")}
    write_json(output / "tools.json", tools)
    pins[str(Path(__file__).relative_to(root))] = sha(Path(__file__))
    pins[str(expectation.relative_to(root))] = sha(expectation)
    write_json(output / "sources.json", pins)
    configure(source, output, guards["native_build"])
    run(["cmake", "--build", str(output / "build"), "-j4"], output / "build.log", guards["native_build"])
    run(["ctest", "--test-dir", str(output / "build"), "--output-on-failure"],
        output / "ctest.log", guards["native_ctest"])
    observations = output / "observations"
    run([str(output / "build/simllm_rnic_rx_atomic_successor_probe"), str(observations)],
        output / "rows.csv", guards["atomic_study"])
    result = check_rows(output / "rows.csv", expected)
    assert sha(source / "include/simllm/rnic/rnic_cmodel_c.h") == expected["baseline_source_sha256"][
        RX + "/include/simllm/rnic/rnic_cmodel_c.h"]
    for relative in ("src/rnic_rx_pipeline.cpp", "include/simllm/rnic/rnic_rx_pipeline.h"):
        old = subprocess.run(["git", "show", f"{BASE}:{RX}/{relative}"], cwd=root,
                             check=True, stdout=subprocess.PIPE, text=True).stdout
        current = (source / relative).read_text()
        assert re.findall(r"^#include .*", current, re.MULTILINE) == re.findall(
            r"^#include .*", old, re.MULTILINE)
        assert not re.search(r"\b(?:getenv|fstream|ifstream|socket|DUT|dut)\b", current)
    result["public_input_source_audit"] = "PASS"
    for frontend in ("native", "facade"):
        for width in (64, 128):
            assert (observations / f"{frontend}_fluid1_w{width}.bin").read_bytes() == (
                observations / f"{frontend}_fluid2_w{width}.bin").read_bytes()
            for suffix in (".trace", ".trace.results.bin"):
                assert (observations / f"default_fluid_identity_{frontend}_w{width}_1{suffix}").read_bytes() == (
                    observations / f"default_fluid_identity_{frontend}_w{width}_2{suffix}").read_bytes()
    baseline = output / "baseline"
    baseline.mkdir()
    shutil.copytree(source, baseline / "source")
    baseline_paths = expected["baseline_source_sha256"]
    for relative, pinned in baseline_paths.items():
        if relative.startswith(RX + "/") and not relative.endswith("README.md"):
            blob = subprocess.run(["git", "show", f"{BASE}:{relative}"], cwd=root,
                                  check=True, stdout=subprocess.PIPE).stdout
            assert hashlib.sha256(blob).hexdigest() == pinned
            target = baseline / "source" / Path(relative).relative_to(RX)
            target.write_bytes(blob)
    write_json(baseline / "sources.json", manifest(baseline / "source"))
    # Probe every original C field offset using mechanically emitted source.
    header = (source / "include/simllm/rnic/rnic_cmodel_c.h").read_text()
    layout = ['#include <cstddef>', '#include <iostream>',
              '#include "simllm/rnic/rnic_cmodel_c.h"', 'int main() {']
    for name, body in re.findall(r"typedef struct (rnic_cm_\w+)\s*\{(.*?)\}\s*\1;", header, re.DOTALL):
        clean = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
        layout.append(f'std::cout << "{name} size " << sizeof({name}) << "\\n";')
        for field in re.findall(r"\b(?:uint\d+_t|char)\s+(\w+)(?:\[[^]]+\])?\s*;", clean):
            layout.append(f'std::cout << "{name}.{field} " << offsetof({name}, {field}) << "\\n";')
    layout.append('}')
    layout_source = output / "abi_layout_probe.cpp"
    layout_source.write_text("\n".join(layout) + "\n")
    for label, includes in (("fixed", source / "include"),
                            ("baseline", baseline / "source/include")):
        binary = output / f"abi_layout_{label}"
        run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
             "-I", str(includes), str(layout_source), "-o", str(binary)],
            output / f"abi_layout_{label}.build.log", guards["native_build"])
        run([str(binary)], output / f"abi_layout_{label}.txt", guards["native_ctest"])
    assert (output / "abi_layout_fixed.txt").read_bytes() == (output / "abi_layout_baseline.txt").read_bytes()
    result["abi_field_offsets"] = len((output / "abi_layout_fixed.txt").read_text().splitlines())
    configure(baseline / "source", baseline, guards["native_build"])
    run(["cmake", "--build", str(baseline / "build"), "-j4", "--target",
         "simllm_rnic_rx_atomic_successor_probe"], baseline / "build.log", guards["native_build"])
    run([str(baseline / "build/simllm_rnic_rx_atomic_successor_probe"),
         str(baseline / "observations"), "default_fluid_identity"],
        baseline / "rows.csv", guards["atomic_study"])
    fluid_names = [p.name for p in observations.iterdir()
                   if "fluid" in p.name or p.name == "abi.txt"]
    for name in fluid_names:
        assert (observations / name).read_bytes() == (baseline / "observations" / name).read_bytes(), name
    result["default_fluid_baseline_identity"] = {"base": BASE, "files": len(fluid_names), "verdict": "PASS"}
    result["mutants"] = []
    if args.mutants:
        for name in expected["mutants"]:
            case = output / name
            case.mkdir()
            shutil.copytree(source, case / "source")
            fixture = mutate(case / "source", name)
            original = (source / "src/rnic_rx_pipeline.cpp").read_text()
            mutated = (case / "source/src/rnic_rx_pipeline.cpp").read_text()
            (case / "mutation.patch").write_text("".join(difflib.unified_diff(
                original.splitlines(keepends=True), mutated.splitlines(keepends=True),
                fromfile="original/src/rnic_rx_pipeline.cpp", tofile=f"{name}/src/rnic_rx_pipeline.cpp")))
            write_json(case / "sources.json", manifest(case / "source"))
            guard = guards["each_mutant_build_and_run"]
            started = time.monotonic()
            configure(case / "source", case, guard)
            run(["cmake", "--build", str(case / "build"), "-j4", "--target",
                 "simllm_rnic_rx_atomic_successor_probe"], case / "build.log",
                max(1, int(guard - (time.monotonic() - started))))
            run([str(case / "build/simllm_rnic_rx_atomic_successor_probe"),
                 str(case / "observations"), fixture], case / "verdict.log",
                max(1, int(guard - (time.monotonic() - started))), fixture=fixture)
            result["mutants"].append({"name": name, "fixture": fixture, "compiled": True,
                                      "exit_code": COMMANDS[-1]["exit_code"]})
    for case in (output, baseline, *(output / item["name"] for item in result["mutants"])):
        binaries = {str(p.relative_to(case)): sha(p) for p in sorted((case / "build").rglob("*"))
                    if p.is_file() and (p.name.startswith("simllm_rnic_") or p.suffix in (".so", ".a"))}
        write_json(case / "binary_pins.json", binaries)
    previous = output.parent / "attempt_001/tail_pack"
    if previous.exists() and previous != output / "tail_pack":
        original = (source / "src/rnic_rx_pipeline.cpp").read_text()
        old = previous / "source/src/rnic_rx_pipeline.cpp"
        (output / "superseded_tail_pack.patch").write_text("".join(difflib.unified_diff(
            original.splitlines(keepends=True), old.read_text().splitlines(keepends=True),
            fromfile="original/src/rnic_rx_pipeline.cpp", tofile="superseded/src/rnic_rx_pipeline.cpp")))
        write_json(output / "superseded_tail_pack.json", {
            "source_sha256": sha(old), "verdict_sha256": sha(previous / "verdict.log"),
            "actual_exit_code": json.loads((previous / "verdict.log.command.json").read_text())["exit_code"],
            "classification": "Incomplete discrimination attempt, affordable gate blocked43/64B pack at39.375B remainder.",
            "correction": "The new copy packs min(remaining beat, successor, affordable whole bytes), reaching39B on queued successor."})
    write_json(output / "trace_pins.json", {str(p.relative_to(output)): sha(p)
                                           for p in sorted(output.rglob("*.trace"))})
    write_json(output / "commands.json", COMMANDS)
    write_json(output / "results.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
