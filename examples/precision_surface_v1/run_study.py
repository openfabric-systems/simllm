"""Dry-run registry for the frozen CORE-36 precision-surface study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SEAMS = (
    "workload",
    "request_outcome",
    "framework",
    "compute",
    "dependency",
    "locality",
    "network",
    "rnic_hardware",
)
LEVELS = {
    "workload": ("fixed-trace", "poisson-arrivals"),
    "request_outcome": (
        "fabricated",
        "preplay-oracle",
        "framework-cpu-oracle",
    ),
    "framework": ("recorded-steps", "executor-rpc", "model-runner"),
    "compute": ("fixed", "roofline", "profile-table"),
    "dependency": ("serial", "observed-framework-schedule"),
    "locality": ("all-remote", "analytic-nvlink"),
    "network": ("rnic-nn-fluid", "packet-level"),
    "rnic_hardware": ("timing-neutral-bypass", "composed-native"),
}
NETWORK_HARDWARE_MATRIX = (
    ("rnic-nn-fluid", "timing-neutral-bypass", "accept"),
    ("packet-level", "timing-neutral-bypass", "accept"),
    ("packet-level", "composed-native", "accept"),
    ("rnic-nn-fluid", "composed-native", "refuse"),
)
REFUSAL_DIAGNOSTIC = (
    "precision.rnic_hardware='composed-native' is incompatible with "
    "precision.network='rnic-nn-fluid'; select "
    "rnic_hardware='timing-neutral-bypass' or network='packet-level'"
)
PRECISION_SCHEMA = "simllm-precision-config-v1"
RUN_PROVENANCE_SCHEMA = "simllm-run-provenance-v1"
SOURCE_SCHEMA = "atlahs-closed-loop-step-v1"
SOURCE_RECORD = {
    "schema": SOURCE_SCHEMA,
    "step_index": 0,
    "virtual_time_ps": 0,
    "scheduled": [],
    "preempted_request_ids": [],
    "finished_request_ids": [],
}
SOURCE_SHA256 = "499a5aee695b8269b1ffb5263f62fee6a00207416f7d62d1b0af64f543a68dca"
LEGAL_CONFIG = {
    "schema": PRECISION_SCHEMA,
    "workload": "fixed-trace",
    "request_outcome": "fabricated",
    "framework": "recorded-steps",
    "compute": "roofline",
    "dependency": "serial",
    "locality": "all-remote",
    "network": "packet-level",
    "rnic_hardware": "composed-native",
}
LEGAL_PRECISION_SHA256 = (
    "8e65df0c5296334800755254cb73c4c4f9cb2c090a2b8805a6409bdc3fbe7d45"
)
LEGAL_PROVENANCE_BYTES = 515
LEGAL_PROVENANCE_SHA256 = (
    "9eea24bf89de06325ee492cba345a22c0245c3a806bdd14da0fdbbd77871978d"
)
FROZEN_SCORED_FAMILIES = 1
FROZEN_SCORED_INSTANCES = 1


def _canonical(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _legal_provenance() -> dict[str, Any]:
    return {
        "schema": RUN_PROVENANCE_SCHEMA,
        "source_schema": SOURCE_SCHEMA,
        "source_sha256": SOURCE_SHA256,
        "precision": LEGAL_CONFIG,
        "precision_sha256": LEGAL_PRECISION_SHA256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def check_only(args: argparse.Namespace) -> None:
    if tuple(LEVELS) != SEAMS or len(SEAMS) != 8:
        raise AssertionError("eight-seam registry drifted")
    if any(not values or len(values) != len(set(values)) for values in LEVELS.values()):
        raise AssertionError("level registry is empty or duplicated")
    if NETWORK_HARDWARE_MATRIX != (
        ("rnic-nn-fluid", "timing-neutral-bypass", "accept"),
        ("packet-level", "timing-neutral-bypass", "accept"),
        ("packet-level", "composed-native", "accept"),
        ("rnic-nn-fluid", "composed-native", "refuse"),
    ):
        raise AssertionError("network and RNIC hardware matrix drifted")
    if sum(outcome == "refuse" for _, _, outcome in NETWORK_HARDWARE_MATRIX) != 1:
        raise AssertionError("matrix must contain exactly one refusal")
    if REFUSAL_DIAGNOSTIC != (
        "precision.rnic_hardware='composed-native' is incompatible with "
        "precision.network='rnic-nn-fluid'; select "
        "rnic_hardware='timing-neutral-bypass' or network='packet-level'"
    ):
        raise AssertionError("refusal diagnostic drifted")
    if set(LEGAL_CONFIG) != {"schema", *SEAMS}:
        raise AssertionError("legal configuration is not exact")
    for seam in SEAMS:
        if LEGAL_CONFIG[seam] not in LEVELS[seam]:
            raise AssertionError(f"legal configuration has unknown {seam} level")
    source = _canonical(SOURCE_RECORD, newline=True)
    if len(source) != 143 or _sha256(source) != SOURCE_SHA256:
        raise AssertionError("source record canonical identity drifted")
    precision = _canonical(LEGAL_CONFIG)
    if _sha256(precision) != LEGAL_PRECISION_SHA256:
        raise AssertionError("legal precision hash drifted")
    provenance = _canonical(_legal_provenance(), newline=True)
    if (
        len(provenance) != LEGAL_PROVENANCE_BYTES
        or _sha256(provenance) != LEGAL_PROVENANCE_SHA256
    ):
        raise AssertionError("legal provenance canonical identity drifted")
    if (FROZEN_SCORED_FAMILIES, FROZEN_SCORED_INSTANCES) != (1, 1):
        raise AssertionError("evidence denominator drifted")
    if not str(args.out):
        raise AssertionError("output path must be nonempty")
    print(
        "check-only "
        f"out={args.out} seams={len(SEAMS)} matrix={len(NETWORK_HARDWARE_MATRIX)} "
        f"scored={FROZEN_SCORED_FAMILIES}/{FROZEN_SCORED_INSTANCES}"
    )


def main() -> None:
    args = parse_args()
    if args.check_only:
        check_only(args)
        return
    raise SystemExit("result mode is not implemented in the expectations-only freeze")


if __name__ == "__main__":
    main()
