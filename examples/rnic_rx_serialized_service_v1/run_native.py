"""Native component evidence, with immutable source and expectation hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def run(command: list[str], log: Path, *, expect_failure: bool = False) -> None:
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=False)
    if (result.returncode != 0) != expect_failure:
        raise RuntimeError(f"unexpected exit {result.returncode}: {log}")
    if expect_failure and "FAIL:" not in log.read_text():
        raise RuntimeError(f"negative copy did not reach a fixture verdict: {log}")


def configure(source: Path, build: Path, output: Path) -> None:
    run(
        ["cmake", "-S", str(source), "-B", str(build),
         "-DSIMLLM_RNIC_WARNINGS_AS_ERRORS=ON"],
        output / "configure.log",
    )


def check_grid(path: Path, expected: dict) -> dict:
    config = expected["configuration"]
    rows = list(csv.DictReader(path.open()))
    combinations = {
        (w, r, length, cap)
        for w in config["beat_bytes"]
        for r in config["rates_bps"]
        for length in config["packet_lengths_bytes"]
        for cap in config["ingress_capacity_bytes"]
    }
    observed = set()
    deadlines = {}
    for row in rows:
        w, rate, length, cap = (int(row[name]) for name in
                               ("width_bytes", "rate_bps", "length_bytes", "capacity_bytes"))
        observed.add((w, rate, length, cap))
        count = min(128, cap // length)
        tick = prefix = beats = 0
        for _ in range(count):
            for offset in range(0, length, w):
                prefix += min(w, length - offset)
                numerator = prefix * 8_000_000_000_000
                refill = rate * config["period_ps"]
                tick = max(tick + 1, (numerator + refill - 1) // refill)
                beats += 1
        deadline = config["phase_ps"] + tick * config["period_ps"]
        geometry = length * 8_000_000_000_000 // (
            config["period_ps"] * ((length + w - 1) // w)
        )
        assert int(row["packets"]) == count
        assert int(row["completed_ps"]) == int(row["expected_ps"]) == deadline
        assert int(row["service_events"]) == beats
        assert int(row["geometric_bps"]) == geometry
        assert int(row["high_water_bytes"]) == count * length <= cap
        assert int(row["meter_drops"]) == 0 and row["verdict"] == "PASS"
        deadlines[w, rate, length, cap] = deadline
    assert len(rows) == len(combinations) and observed == combinations
    for w, rate, length, cap in combinations:
        for wider in config["beat_bytes"]:
            if wider >= w:
                assert deadlines[wider, rate, length, cap] <= deadlines[w, rate, length, cap]
        for faster in config["rates_bps"]:
            if faster >= rate:
                assert deadlines[w, faster, length, cap] <= deadlines[w, rate, length, cap]
    return {"rows": len(rows), "exact_deadlines": True, "monotonicity": True}


def mutate(source: Path, name: str) -> None:
    cpp = source / "src/rnic_rx_pipeline.cpp"
    text = cpp.read_text()
    if name == "idle_credit":
        old = "        service_credit_ = 0;\n        clockAfter(now_ps);"
        new = "        // Negative copy: retain credit across empty intervals.\n        clockAfter(now_ps);"
    elif name == "tail_packing":
        old = "        clockAfter(*next);"
        new = """        if (bytes < config_.service.beat_bytes && hasPendingService()) {
            const auto packed = std::min(config_.service.beat_bytes - bytes,
                                         service_records_.front());
            if (service_credit_ >= packed * kByteCreditDenominator) {
                service_credit_ -= packed * kByteCreditDenominator;
                occupancy_bytes_ -= packed;
                service_bytes_ += packed;
                service_records_.front() -= packed;
                if (service_records_.front() == 0) service_records_.pop_front();
            }
        }
        clockAfter(*next);"""
    elif name == "fluid_admission":
        header = source / "include/simllm/rnic/rnic_rx_pipeline.h"
        header.write_text(header.read_text().replace(
            "    RnicRxPipelineConfig config_;",
            "    std::unique_ptr<RnicRxPipeline> admission_fluid_;\n    RnicRxPipelineConfig config_;",
        ))
        text = text.replace(
            "        next_tick_ps_ = config_.service.phase_ps;",
            """        auto fluid_config = config_;
        fluid_config.service = {};
        admission_fluid_ = std::make_unique<RnicRxPipeline>(fluid_config);
        next_tick_ps_ = config_.service.phase_ps;""",
        )
        old = "    if (bounded && occupancy_bytes_ + packet.wire_bytes > config_.ingress_bytes) {"
        new = """    const bool fluid_reject = admission_fluid_
        && admission_fluid_->onPacket(packet, now_ps).outcome == RnicRxOutcome::DiscardedSilently;
    if (admission_fluid_ ? fluid_reject
        : (bounded && occupancy_bytes_ + packet.wire_bytes > config_.ingress_bytes)) {"""
    else:
        raise ValueError(name)
    if text.count(old) != 1:
        raise RuntimeError(f"mutation anchor is not unique: {name}")
    cpp.write_text(text.replace(old, new))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mutants", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output == root or root in output.parents:
        parser.error("generated outputs must be outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    source = root / "simllm/backends/rnic"
    expectation_path = Path(__file__).with_name("expectations.json")
    expected = json.loads(expectation_path.read_text())
    manifest = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source.rglob("*")) if p.is_file()
    }
    manifest[str(expectation_path.relative_to(root))] = hashlib.sha256(
        expectation_path.read_bytes()
    ).hexdigest()
    manifest[str(Path(__file__).relative_to(root))] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    (output / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    build = output / "build"
    configure(source, build, output)
    run(["cmake", "--build", str(build), "-j4"], output / "build.log")
    run(["ctest", "--test-dir", str(build), "--output-on-failure"], output / "ctest.log")
    run([str(build / "simllm_rnic_rx_serialized_probe")], output / "grid.csv")
    result = check_grid(output / "grid.csv", expected)
    result["mutants_caught"] = []
    if args.mutants:
        for name in ("tail_packing", "idle_credit", "fluid_admission"):
            case = output / name
            case.mkdir()
            copy = case / "source"
            shutil.copytree(source, copy)
            mutate(copy, name)
            configure(copy, case / "build", case)
            run(["cmake", "--build", str(case / "build"), "-j4", "--target",
                 "simllm_rnic_rx_serialized_test"], case / "build.log")
            run([str(case / "build/simllm_rnic_rx_serialized_test"), str(case)],
                case / "verdict.log", expect_failure=True)
            result["mutants_caught"].append(name)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
