"""Dry-run scaffold for the frozen host-step cost study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTATIONS = HERE / "expectations.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--htsim-rnic", type=Path, required=True)
    parser.add_argument("--baseline-cell", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _check(args: argparse.Namespace) -> None:
    values = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))
    if values["schema"] != "simllm-host-step-cost-v1-expectations-v1":
        raise AssertionError("expectation schema drifted")
    if values["representative_step"]["acceptance_decode_multiplier"] != [1.8, 7.1]:
        raise AssertionError("flagship multiplier band drifted")
    if sum(values["scored_live_relations"].values()) != 12:
        raise AssertionError("scored live inventory drifted")
    fixture = values["live_fixture"]
    if fixture["network_profile"] != "rnic-nn-fluid":
        raise AssertionError("live network profile drifted")
    if fixture["linkspeed_bps"] != 400_000_000_000:
        raise AssertionError("live link rate drifted")
    if [row["step_index"] for row in fixture["steps"]] != [0, 1, 2]:
        raise AssertionError("live step sequence drifted")
    ideal = fixture["ideal_step_ps"]
    if ideal["tpot"] * 2 != sum(ideal["decode"]):
        raise AssertionError("ideal TPOT arithmetic drifted")
    representative = values["representative_step"]
    if (
        representative["modeled_compute_ps"]
        + representative["modeled_decode_network_ps"]
        != representative["modeled_decode_makespan_ps"]
    ):
        raise AssertionError("representative step does not conserve")
    for bounds in fixture["network_physical_bounds_ps"].values():
        if len(bounds) != 2 or not 0 < bounds[0] <= bounds[1]:
            raise AssertionError("network physical bounds drifted")
    launches = values["launch_count"]
    capture_bounds = values["capture"]["acceptance_ps"]
    launch_floor_bounds = (
        launches["minimum"] * capture_bounds["graph_replay"][0],
        launches["maximum"] * capture_bounds["eager_host_bound"][1],
    )
    budget = values["mission_budget"]
    ratios = [
        (launch_ps + real_ps) / (launch_ps + budget["modeled_network_ps"])
        for launch_ps in launch_floor_bounds
        for real_ps in budget["plausible_collective_network_ps"]
    ]
    acceptance = budget["acceptance_conditional_turing_residual_optimism"]
    if min(ratios) < acceptance[0] or max(ratios) > acceptance[1]:
        raise AssertionError("conditional budget enclosure drifted")
    if not args.htsim_rnic.is_file() or not os.access(args.htsim_rnic, os.X_OK):
        raise FileNotFoundError("htsim_rnic is missing or not executable")
    converter_value = os.environ.get("SIMLLM_TXT2BIN")
    if not converter_value:
        raise RuntimeError("SIMLLM_TXT2BIN must be configured")
    converter = Path(converter_value)
    if not converter.is_file() or not os.access(converter, os.X_OK):
        raise FileNotFoundError("SIMLLM_TXT2BIN is missing or not executable")
    for name, expected in fixture["input_sha256"].items():
        path = args.baseline_cell / name.replace("_", ".", 1)
        if name == "routed_experts_json":
            path = args.baseline_cell / "routed-experts.json"
        elif name == "steps_jsonl":
            path = args.baseline_cell / "steps.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"live fixture input is missing: {path}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"live fixture input identity drifted: {path.name}")
    compatibility = values["ideal_compatibility"]
    if len(compatibility["aggregate_cell_sha256"]) != 64:
        raise AssertionError("ideal aggregate digest drifted")
    if set(compatibility["step_record_sha256"]) != {
        "a-ep4-400g",
        "a-ep8-100g",
        "a-ep8-200g",
        "a-ep8-400g",
        "b-ep8-400g",
    }:
        raise AssertionError("ideal cell inventory drifted")
    if any(
        len(digest) != 64
        for digest in compatibility["step_record_sha256"].values()
    ):
        raise AssertionError("ideal step digest drifted")
    root = os.environ.get("SIMLLM_WAVE12_RUN_ROOT")
    if not root:
        raise RuntimeError("SIMLLM_WAVE12_RUN_ROOT must be configured")
    args.out.resolve().relative_to(Path(root).resolve())
    if args.out.exists():
        raise FileExistsError(f"output already exists: {args.out}")
    print(
        "check-only: live registry, arithmetic, inputs and tools valid; "
        "htsim not invoked; no output created"
    )


def main() -> int:
    args = _parse_args()
    _check(args)
    if not args.check_only:
        raise RuntimeError("study implementation lands after the expectations freeze")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
