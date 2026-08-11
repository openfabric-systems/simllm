"""Produce raw Tier C ABI-v2 observations through the Tier B live chain."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rnic_live_v1.tier_b_producer import (
    FROZEN_SIMLLM_BASE,
    _publish,
    _required_path,
    _run_structural_cell,
)
from simllm.backends.composed_rnic import (
    ComposedRnicObservations,
    invoke_composed_tier_a_producer,
)

STUDY_DIR = Path(__file__).resolve().parent
TIER_A_EXPECTATIONS = STUDY_DIR / "tier_a_expectations.json"
TIER_C_SCHEMA = "simllm-rnic-tier-c-expectations-v1"
TIER_C_OBSERVATION_SCHEMA = "simllm-rnic-tier-c-observations-v1"


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {name} {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _validate_expectations(value: dict[str, Any], factory: str) -> None:
    if value.get("schema") != TIER_C_SCHEMA:
        raise ValueError("unsupported Tier C expectations schema")
    if value.get("observation_schema") != TIER_C_OBSERVATION_SCHEMA:
        raise ValueError("unsupported Tier C observation schema")
    if factory != "htsim":
        raise ValueError("Tier C requires the composed htsim factory")
    if value.get("network_abi_version") != 2:
        raise ValueError("Tier C requires NetworkPort ABI v2")


def produce(factory: str, expectations_path: Path, observations_path: Path) -> None:
    expectations = _load_json(expectations_path, "Tier C expectations")
    _validate_expectations(expectations, factory)
    if observations_path.exists() or Path(f"{observations_path}.tmp").exists():
        raise FileExistsError("Tier C observations already exist")
    native_producer = _required_path("SIMLLM_RNIC_TIER_A_PRODUCER")
    observations_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".tier-c-producer-",
        dir=observations_path.parent,
    ) as temporary:
        native_path = Path(temporary) / "tier_a_v2_observations.json"
        invoke_composed_tier_a_producer(
            native_producer,
            TIER_A_EXPECTATIONS,
            native_path,
            network_abi_version=2,
        )
        native_raw = _load_json(native_path, "Tier A ABI-v2 observations")
        native = ComposedRnicObservations.from_json(native_raw)
        if native.network_abi_version != 2:
            raise RuntimeError("Tier C native producer did not retain ABI v2")
        structural_single = [
            _run_structural_cell(
                native.single_wqe[key],
                include_packet_timeline=True,
                session_prefix="tier-c",
            )
            for key in sorted(native.single_wqe)
        ]
        structural_fifo = [
            _run_structural_cell(
                native.fifo[key],
                include_packet_timeline=True,
                session_prefix="tier-c",
            )
            for key in sorted(native.fifo)
        ]

    observations = {
        "schema": TIER_C_OBSERVATION_SCHEMA,
        "factory": factory,
        "network_abi_version": 2,
        "simllm_base_commit": FROZEN_SIMLLM_BASE,
        "structural_single_wqe": structural_single,
        "structural_fifo": structural_fifo,
    }
    _publish(observations_path, observations)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--expectations", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    try:
        produce(
            arguments.factory,
            arguments.expectations.resolve(strict=True),
            arguments.observations.resolve(strict=False),
        )
    except BaseException as error:
        print(f"Tier C producer error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
