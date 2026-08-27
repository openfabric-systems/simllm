"""Publish only the A100 profile changes authorized by a complete TRAF-70 score."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import run_study
import score_hardware

STUDY_ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE = STUDY_ROOT.parent / "a100_nvlink_packet_v1" / "candidate-profile.json"
PUBLISHED_STATUS = "scored_mixed_parameter_evidence"
PUBLISHED_EVIDENCE_CLASS = "parameter_specific_evidence_see_traf70_score"
RUNTIME_PARAMETER_PATHS = {
    ("tx", "max_payload_bytes"),
    ("tx", "header_bytes"),
    ("tx", "links_per_peer"),
    ("tx", "per_link_rate_bytes_per_second"),
    ("tx", "endpoint_egress_rate_bytes_per_second"),
    ("tx", "bond_policy"),
    ("tx", "credits_per_destination"),
    ("tx", "credit_unit_bytes"),
    ("rx", "ingress_rate_bytes_per_second"),
    ("rx", "buffer_capacity_bytes"),
    ("rx", "credit_return_latency_ps"),
    ("rx", "reassembly_policy"),
    ("rx", "delivery_order"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    score = read_json(args.score)
    profile = read_json(args.profile)
    published = build_published_profile(
        score,
        profile,
        score_path=args.score,
        score_sha256=sha256(args.score),
        profile_sha256=sha256(args.profile),
    )
    write_json(args.output, published)
    print(f"published {len(score['profile_patch']['changes'])} score-authorized changes")
    return 0


def build_published_profile(
    score: dict[str, Any],
    profile: dict[str, Any],
    *,
    score_path: Path,
    score_sha256: str,
    profile_sha256: str,
) -> dict[str, Any]:
    validate_complete_score(score)
    if profile_sha256 != run_study.PROTECTED_CANDIDATE_SHA256:
        raise RuntimeError("publication input is not the byte-locked A100 candidate")
    if profile.get("status") != "candidate":
        raise RuntimeError("publication input is not the pre-score candidate")
    if profile.get("evidence_class") != "declared_candidate_not_hardware_measurement":
        raise RuntimeError("publication input already claims non-candidate evidence")
    expected_patch = score_hardware.profile_patch_from_results(
        score["module_parameter_identification"]
    )
    if score.get("profile_patch") != expected_patch:
        raise RuntimeError("score profile_patch is not the literal parameter result projection")

    published = json.loads(json.dumps(profile))
    evidence: dict[str, dict[str, Any]] = {}
    runtime_changes = []
    metadata_only_changes = []
    for result in score["module_parameter_identification"]:
        module = str(result["module"])
        parameter = str(result["parameter"])
        evidence.setdefault(module, {})[parameter] = {
            "status": result["status"],
            "value": (
                result["identified_value"]
                if result["status"] in {"IDENTIFIED", "STRUCTURAL"}
                else result["candidate_value"]
            ),
            "candidate_relation": result["candidate_relation"],
            "evidence_class": result["evidence_class"],
            "rule_id": result["rule_id"],
            "reason": result["reason"],
        }
    for change in expected_patch["changes"]:
        key = (str(change["module"]), str(change["parameter"]))
        if key in RUNTIME_PARAMETER_PATHS:
            module, parameter = key
            if parameter not in published[module]:
                raise RuntimeError(f"runtime profile parameter is absent: {module}.{parameter}")
            published[module][parameter] = change["value"]
            runtime_changes.append({**change, "publication_surface": "runtime_profile"})
        elif key == ("tx", "request_response_direction"):
            if change["candidate_relation"] != "CONFIRMED":
                raise RuntimeError(
                    "a refuted direction mapping requires an explicit scored htsim code update"
                )
            metadata_only_changes.append(
                {
                    **change,
                    "publication_surface": "existing_htsim_directional_packetization",
                }
            )
        else:
            raise RuntimeError(f"score authorized an unknown profile parameter: {key}")

    published["status"] = PUBLISHED_STATUS
    published["evidence_class"] = PUBLISHED_EVIDENCE_CLASS
    published["freeze_path"] = "examples/a100_nvlink_packet_v2/expectations.json"
    published["freeze_sha256"] = run_study.FREEZE_SHA256
    published["parameter_evidence"] = evidence
    published["traf70_score_publication"] = {
        "score_path": score_path.as_posix(),
        "score_sha256": score_sha256,
        "score_schema": score["schema"],
        "score_status": score["status"],
        "execution_heads": score["execution_heads"],
        "scheduler_job": score["scheduler_job"],
        "protected_candidate_before_sha256": profile_sha256,
        "runtime_changes": runtime_changes,
        "metadata_only_changes": metadata_only_changes,
        "unchanged_parameter_count": expected_patch["unchanged_parameter_count"],
    }
    handoff = published.get("handoff")
    if isinstance(handoff, dict):
        handoff["measurement_claim"] = bool(expected_patch["changes"])
        handoff["measurement_scope"] = "only parameters listed in parameter_evidence"
    return published


def validate_complete_score(score: dict[str, Any]) -> None:
    required = {
        "schema": score_hardware.SCORE_SCHEMA,
        "status": "COMPLETE_VALID_86_OF_86",
        "measurement_validity": "VALID_FOR_FROZEN_RULES",
        "freeze_sha256": run_study.FREEZE_SHA256,
        "protected_candidate_before_sha256": run_study.PROTECTED_CANDIDATE_SHA256,
    }
    for key, expected in required.items():
        if score.get(key) != expected:
            raise RuntimeError(f"score {key} is not publication-valid")
    coverage = score.get("coverage", {})
    if coverage.get("completed_cell_count") != 86 or coverage.get("pending_indices") != []:
        raise RuntimeError("score does not cover all 86 cells")
    guard_summary = score.get("fatal_guard_verdicts", {})
    guards = guard_summary.get("guards", [])
    if (
        guard_summary.get("status") != "PASS"
        or len(guards) != 10
        or any(guard.get("status") != "PASS" or guard.get("decidable") is not True for guard in guards)
    ):
        raise RuntimeError("score fatal guards are not all decidable passes")
    if score.get("producer_binary_audit", {}).get("status") != "SINGLE_DIGEST":
        raise RuntimeError("score does not bind one producer binary digest")
    if not isinstance(score.get("module_parameter_identification"), list):
        raise TypeError("score parameter results are absent")
    allowed_statuses = {"IDENTIFIED", "INCONCLUSIVE", "STRUCTURAL"}
    if any(
        result.get("status") not in allowed_statuses
        for result in score["module_parameter_identification"]
    ):
        raise RuntimeError("score contains a pending or void parameter result")


def read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8", newline="") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} is not a JSON object")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
