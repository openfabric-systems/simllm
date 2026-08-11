"""Run the frozen BACK-34 partial-final-packet study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rnic_packet_v2 import run_study as packet_study
from simllm.backends.rnic_records import (
    BypassArtifacts,
    assert_bypass_artifact_identity,
    canonical_bypass_parameters,
)

EXPECTATIONS_PATH = Path(__file__).with_name("back34_expectations.json")
SIMLLM_BASE_COMMIT = "90ada43070adb3b1e624b6819aff34d8620e8571"
HTSIM_BASE_COMMIT = "4885c647eecdfdf81479d1df052223c016ad086b"
EXPECTATION_COMMIT = "51af85937d6b1d3c36f6d841c6445d98ef84c2d3"
CORRECTION_COMMIT = "45c9bba1e0fe2a716d37ccbab2a4a246258781f4"
CORRECTION_2_COMMIT = "0a185e8847c4b42c096488a045af2a8e69ea2616"
V1_ARTIFACT_SHA256 = {
    "raw_observations.json": (
        "37a4e9cf88a1b60094409150dfad25599eb77cbf268b3d08bfacf527e493a26a"
    ),
    "summary.json": (
        "00ef7e4f5bdbd38f4eabe9ba42dc75f56de528c8751b93e6eef4a3089fa61004"
    ),
}
V2_RAW_SHA256 = (
    "39059d56663f73869224613c9c7a0de3bee5733a6654469cd2a54c22354cc692"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wave5_root() -> Path:
    configured = os.environ.get("SIMLLM_WAVE5_RUN_ROOT")
    if not configured:
        raise RuntimeError("SIMLLM_WAVE5_RUN_ROOT must be configured")
    return Path(configured).resolve()


def _validate_commit(repo: Path, revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise AssertionError("frozen revision must be a full hash")
    subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _validate_registry(
    htsim_source: Path,
    v1_reference_dir: Path,
    v2_reference_raw: Path,
    out: Path,
) -> None:
    expectations = _load_json(EXPECTATIONS_PATH)
    if expectations["schema"] != "simllm-rnic-back34-expectations-v1":
        raise AssertionError("BACK-34 expectation schema drifted")
    if expectations["simllm_base_commit"] != SIMLLM_BASE_COMMIT:
        raise AssertionError("SimLLM base commit drifted")
    if expectations["htsim_base_commit"] != HTSIM_BASE_COMMIT:
        raise AssertionError("htsim base commit drifted")
    _validate_commit(REPO_ROOT, SIMLLM_BASE_COMMIT)
    _validate_commit(htsim_source, HTSIM_BASE_COMMIT)

    if expectations["payload_bytes"] != 5000:
        raise AssertionError("BACK-34 payload drifted")
    packets = expectations["expected_packets"]
    if [row["payload_bytes"] for row in packets] != [4032, 968]:
        raise AssertionError("BACK-34 payload packetization drifted")
    if [row["wire_bytes"] for row in packets] != [4096, 1032]:
        raise AssertionError("BACK-34 wire packetization drifted")
    if [row["tier_a_tx_started_at_ps"] for row in packets] != [1000, 82920]:
        raise AssertionError("BACK-34 Tier A TX starts drifted")
    if [row["tier_a_tx_finished_at_ps"] for row in packets] != [82920, 103560]:
        raise AssertionError("BACK-34 Tier A TX finishes drifted")
    if [row["composed_tx_started_at_ps"] for row in packets] != [1000, 144200]:
        raise AssertionError("BACK-34 composed TX starts drifted")
    if [row["composed_tx_finished_at_ps"] for row in packets] != [82920, 164840]:
        raise AssertionError("BACK-34 composed TX finishes drifted")
    if expectations["tier_a_boundaries"] != {
        "first_packet_at_ps": 1000,
        "last_packet_at_ps": 82920,
        "first_rx_at_ps": 82920,
        "last_rx_at_ps": 103560,
        "network_outcome_at_ps": 103560,
    }:
        raise AssertionError("BACK-34 Tier A boundaries drifted")
    if expectations["composed_boundaries"] != {
        "first_packet_at_ps": 1000,
        "last_packet_at_ps": 144200,
        "first_rx_at_ps": 82920,
        "last_rx_at_ps": 164840,
        "network_outcome_at_ps": 185480,
    }:
        raise AssertionError("BACK-34 composed boundaries drifted")

    for name, expected in V1_ARTIFACT_SHA256.items():
        actual = _digest(v1_reference_dir / name)
        if actual != expected:
            raise AssertionError(
                f"accepted ABI-v1 reference drifted for {name}: {actual}"
            )
    actual_v2 = _digest(v2_reference_raw)
    if actual_v2 != V2_RAW_SHA256:
        raise AssertionError(f"accepted ABI-v2 reference drifted: {actual_v2}")
    expected_parent = (
        _wave5_root() / "codex" / "htsim1516_control_producers"
    ).resolve()
    try:
        out.resolve().relative_to(expected_parent)
    except ValueError as error:
        raise ValueError(
            "BACK-34 output must remain under the branch wave-5 directory"
        ) from error


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _full_quantum_projection(value: dict[str, Any]) -> dict[str, Any]:
    projected = dict(value)
    projected.pop("partial_final_packet", None)
    return projected


def _projection_artifacts(value: dict[str, Any]) -> BypassArtifacts:
    projected = _full_quantum_projection(value)
    metadata = {
        name: item
        for name, item in projected.items()
        if name not in {"single_wqe", "fifo", "controlled_drop"}
    }
    return BypassArtifacts(
        goal_text=b"tier-a-v2-full-quantum-projection\n",
        goal_binary=b"network-port-abi-v2\n",
        topology=b"tier-a-zero-hop\n",
        profile="htsim",
        seed=0,
        baseline_parameters=canonical_bypass_parameters(
            {"network_abi_version": 2}
        ),
        completion_csv=_canonical(projected["single_wqe"]),
        canonical_completion=_canonical(projected["fifo"]),
        step_results=_canonical(projected["controlled_drop"]),
        replay_summary=_canonical(metadata),
    )


def _run_v2_raw(producer: Path, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    raw = run_dir / "raw_observations.json"
    subprocess.run(
        [
            str(producer),
            "--factory",
            "htsim",
            "--network-abi-version",
            "2",
            "--expectations",
            str(packet_study.TIER_A_EXPECTATIONS),
            "--observations",
            str(raw),
        ],
        check=True,
    )
    return _load_json(raw)


def _run(
    htsim_source: Path,
    v1_reference_dir: Path,
    v2_reference_raw: Path,
    out: Path,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    producer, ctest = packet_study._build(htsim_source, out / "build")

    v1_digests = packet_study._run_v1(producer, out / "abi-v1")
    v1_identity = {
        name: digest == V1_ARTIFACT_SHA256[name]
        for name, digest in v1_digests.items()
    }
    if not all(v1_identity.values()):
        raise AssertionError("BACK-34 changed accepted ABI-v1 bytes")

    generated = _run_v2_raw(producer, out / "abi-v2")
    reference = _load_json(v2_reference_raw)

    # This scored relation is deliberately evaluated from generated bytes
    # before the new tail's exact fatal oracle is invoked below.
    comparison = assert_bypass_artifact_identity(
        _projection_artifacts(reference),
        _projection_artifacts(generated),
    )
    full_quantum_identity = {
        "passed": 1,
        "total": 1,
        "changed_inputs": list(comparison.changed_inputs),
        "changed_artifacts": list(comparison.changed_artifacts),
        "input_matches": dict(comparison.input_matches),
        "behavioral_matches": dict(comparison.behavioral_matches),
    }

    validated_v2 = packet_study._validate_v2_observations(generated)
    report = {
        "schema": "simllm-rnic-back34-results-v1",
        "simllm_revision": packet_study._git_commit(REPO_ROOT),
        "htsim_revision": packet_study._git_commit(htsim_source),
        "expectation_commits": {
            "freeze": EXPECTATION_COMMIT,
            "checker_correction": CORRECTION_COMMIT,
            "checker_correction_2": CORRECTION_2_COMMIT,
        },
        "ctest": ctest,
        "compatibility": {
            "full_quantum_v2_projection": full_quantum_identity,
            "abi_v1_artifact_identity": {
                name: {"passed": int(matched), "total": 1}
                for name, matched in v1_identity.items()
            },
        },
        "partial_final_packet": validated_v2["partial_final_packet"],
        "existing_packet_v2_validation": validated_v2,
        "genuine_risk": {
            "full_quantum_v2_projection": {
                "plausible_failures": 1,
                "relations": 1,
            },
            "abi_v1_identity": {
                "plausible_failures": 2,
                "relations": 2,
            },
            "overall": {"plausible_failures": 3, "relations": 3},
        },
        "entailment_analysis": {
            "evaluation_order": (
                "generated compatibility bytes before the exact partial-tail "
                "oracle"
            ),
            "full_quantum_v2_projection": (
                "can fail after the producer runs without violating the new "
                "tail fields"
            ),
            "abi_v1_identity": (
                "can fail byte identity after the Tier A semantic gate passes"
            ),
            "partial_final_packet": "fatal_unscored exact component oracle",
            "reference_digests": "unscored change-set guards validated at entry",
        },
        "reference_sha256": {
            "abi_v1": {
                name: _digest(v1_reference_dir / name)
                for name in V1_ARTIFACT_SHA256
            },
            "abi_v2_raw": _digest(v2_reference_raw),
        },
    }
    (out / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--htsim-source", type=Path, required=True)
    parser.add_argument("--v1-reference-dir", type=Path, required=True)
    parser.add_argument("--v2-reference-raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    _validate_registry(
        arguments.htsim_source.resolve(),
        arguments.v1_reference_dir.resolve(),
        arguments.v2_reference_raw.resolve(),
        arguments.out.resolve(),
    )
    if not arguments.check_only:
        report = _run(
            arguments.htsim_source.resolve(),
            arguments.v1_reference_dir.resolve(),
            arguments.v2_reference_raw.resolve(),
            arguments.out.resolve(),
        )
        risk = report["genuine_risk"]["overall"]
        print(
            "BACK-34 study passed "
            f"{risk['plausible_failures']}/{risk['relations']} "
            "genuine-risk compatibility relations"
        )
        return
    print("BACK-34 study registry check passed; no artifacts were produced")


if __name__ == "__main__":
    main()
