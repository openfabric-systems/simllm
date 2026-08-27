#!/usr/bin/env python3
"""Compile the COMP-72 partial campaign without mutating its predecessor."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_study import _artifact_manifest, _device_payload, _profile_payload

from simllm.calibration.canonical import canonical_loads, sha256_bytes
from simllm.calibration.kernel_cycle_lut import validate_kernel_cycle_lut

STUDY_DIR = Path(__file__).resolve().parent
INPUT_PATH = STUDY_DIR / "campaign_evidence.json"
FREEZE_PATH = STUDY_DIR / "expectations.json"
PREDECESSOR_PATH = STUDY_DIR / "candidate-record.json"
PREDECESSOR_SHA256 = "ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52"
EXPECTED_FREEZE_SHA256 = "9c92756d2deab0a1cf18509b3b5f96770c7a18c96fb4eb14feb1f58af45a20fa"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _verify_sources(
    evidence: dict[str, Any], evidence_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    definitions: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for source in evidence["sources"]:
        name = str(source["name"])
        relative = Path(str(source["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{name}: source path must stay below the evidence root")
        path = evidence_root / relative
        data = path.read_bytes()
        if len(data) != int(source["bytes"]):
            raise ValueError(f"{name}: expected {source['bytes']} bytes, found {len(data)}")
        digest = sha256_bytes(data)
        if digest != source["sha256"]:
            raise ValueError(f"{name}: expected sha256 {source['sha256']}, found {digest}")
        definitions[name] = source
        paths[name] = path
    if list(definitions) != sorted(definitions):
        raise ValueError("campaign source names must be sorted")
    return definitions, paths


def _round_ps(value_ns: str) -> int:
    return int((Decimal(value_ns) * Decimal(1000)).quantize(Decimal(1), ROUND_HALF_UP))


def _csv_row(path: Path, key: str, value: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        matches = [row for row in csv.DictReader(stream) if row[key] == value]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one {key}={value!r} row, found {len(matches)}")
    return matches[0]


def _repeat_service_ps(repeat: dict[str, Any], paths: dict[str, Path]) -> int:
    if repeat["reader"] == "recovery-json-service-ps":
        recovery = _read_json(paths[str(repeat["value_source"])])
        matches = [
            service
            for service in recovery["services"]
            if service["implementation_suffix"] == repeat["implementation_suffix"]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{repeat['implementation_suffix']}: expected one recovered service"
            )
        return int(matches[0]["measured_service_ps"])
    row = _csv_row(
        paths[str(repeat["value_source"])],
        "cell",
        str(repeat["csv_cell"]),
    )
    if repeat["reader"] == "cell-summary-compute-ns":
        return int(row["compute_ns"]) * 1000
    if repeat["reader"] == "decode-summary-rank-step-ns":
        return _round_ps(row["compute_rank_step_ns"])
    raise ValueError(f"unsupported repeat reader {repeat['reader']!r}")


def _entry_by_suffix(record: dict[str, Any], suffix: str) -> list[dict[str, Any]]:
    return [
        entry
        for entry in record["entries"]
        if str(entry["implementation_id"]).endswith(suffix)
    ]


def _published_point(entries: list[dict[str, Any]]) -> int:
    measured = [
        entry
        for entry in entries
        if entry["evidence"]["service_class"] == "MEASURED"
    ]
    if len(measured) != 1:
        raise ValueError("each repeated key must select one measured physical entry")
    return int(measured[0]["measured_service_ps"])


def _record_source(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": source["name"],
        "fixture_sha256": source["sha256"],
        "fixture_bytes": source["bytes"],
        "retained_source_name": source["path"],
        "retained_source_sha256": source["sha256"],
    }


def _build_successor(
    predecessor: dict[str, Any],
    evidence: dict[str, Any],
    definitions: dict[str, dict[str, Any]],
    paths: dict[str, Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    successor = deepcopy(predecessor)
    successor["campaign_id"] = evidence["campaign_id"]
    successor["created"] = evidence["created"]
    successor["sources"].extend(
        _record_source(source)
        for source in definitions.values()
        if source["publish_in_lookup_record"]
    )
    successor["sources"].sort(key=lambda source: source["name"])

    observations = []
    for repeat in evidence["deepseek"]["priced_repeats"]:
        suffix = str(repeat["implementation_suffix"])
        entries = _entry_by_suffix(successor, suffix)
        if len(entries) != 2:
            raise ValueError(f"{suffix}: expected measured and declared entries")
        source_digests = {
            str(definitions[name]["sha256"])
            for name in repeat["source_names"]
        }
        for entry in entries:
            entry["evidence"]["source_sha256s"] = sorted(
                set(entry["evidence"]["source_sha256s"]) | source_digests
            )
        observed_ps = _repeat_service_ps(repeat, paths)
        point_ps = _published_point(entries)
        observations.append(
            {
                "implementation_suffix": suffix,
                "published_point_ps": point_ps,
                "independent_repeat_ps": observed_ps,
                "signed_repeat_minus_point_ps": observed_ps - point_ps,
                "retained_independent_observations": 2,
                "distribution_propagation": "DEFERRED_TO_COMP-74",
            }
        )

    old_by_id = {entry["implementation_id"]: entry for entry in predecessor["entries"]}
    for entry in successor["entries"]:
        old = old_by_id[entry["implementation_id"]]
        if entry["measured_service_ps"] != old["measured_service_ps"]:
            raise ValueError("successor changed a frozen candidate point")
        if entry["distribution"] != old["distribution"]:
            raise ValueError("successor changed distribution before COMP-74")
    return successor, observations


def _score_campaign(
    successor: dict[str, Any],
    evidence: dict[str, Any],
    paths: dict[str, Path],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    ledger: dict[str, dict[str, int]] = {}
    for entry in successor["entries"]:
        family = "deepseek_v3" if entry["implementation_id"].startswith("deepseek") else "granite"
        service_class = str(entry["evidence"]["service_class"])
        ledger.setdefault(family, {}).setdefault(service_class, 0)
        ledger[family][service_class] += 1

    mtp_source = str(evidence["deepseek"]["mtp"]["source_name"])
    mtp = _read_json(paths[mtp_source])
    if mtp["lookup_pricing"] != "FORBIDDEN_BY_FREEZE":
        raise ValueError("MTP recovery must not authorize lookup pricing")
    if any("mtp" in entry["implementation_id"] for entry in successor["entries"]):
        raise ValueError("frozen campaign forbids an MTP lookup entry")
    measured_mtp_ps = int(mtp["measured_service_ps"])
    if measured_mtp_ps != int(evidence["deepseek"]["mtp"]["measured_service_ps"]):
        raise ValueError("MTP service disagrees with the retained campaign ledger")

    plan = canonical_loads(paths[str(evidence["granite"]["plan_source_name"])].read_bytes())
    plan_cells = plan["cells"]
    if len(plan_cells) != 1212:
        raise ValueError("Granite plan no longer contains 1212 registered cells")
    if int(evidence["granite"]["completed_cell_count"]) != 0:
        raise ValueError("this partial result must not claim an unstaged Granite cell")
    if len(observations) != 4 or any(
        row["retained_independent_observations"] < 2 for row in observations
    ):
        raise ValueError("every priced DeepSeek key requires two retained observations")

    return {
        "verdict": "PARTIAL_CAMPAIGN_RECOMPILED",
        "lookup_service_ledger": ledger,
        "requested_physical_cell_ledger": {
            "deepseek_v3": {"MEASURED": 5, "ABSENT": 0},
            "granite_registered_campaign": {"MEASURED": 0, "ABSENT": 1212},
        },
        "component_overlay_ledger": {
            "lookup_entries": {"granite": {"DISCLOSED": 12}, "deepseek_v3": {"DISCLOSED": 8}},
            "measured_unpriced_mtp": {"DISCLOSED": 1},
        },
        "mtp": {
            "measured_service_ps": measured_mtp_ps,
            "evidence_class": "MEASURED",
            "lookup_pricing": "FORBIDDEN_BY_FREEZE",
        },
        "priced_repeat_observations": observations,
        "granite": {
            "registered_cell_count": len(plan_cells),
            "completed_cell_count": 0,
            "first_incomplete_cell": plan_cells[0]["cell_id"],
            "completed_prefix_sha256": evidence["granite"]["completed_prefix_sha256"],
            "blocker": evidence["granite"]["blocker"],
        },
        "core61": evidence["core61"],
        "task_movement": {
            "comp72": "OPEN",
            "comp74_repeat_inputs": "RETAINED_FOR_ALL_FOUR_PRICED_KEYS",
            "core61": "OPEN_TIME_GATED",
            "remainder_owner": "COMP-78",
        },
    }


def compile_campaign(evidence_root: Path, output_dir: Path) -> dict[str, Any]:
    evidence = _read_json(INPUT_PATH)
    if sha256_bytes(FREEZE_PATH.read_bytes()) != EXPECTED_FREEZE_SHA256:
        raise ValueError("expectations changed after the scored campaign cell")
    predecessor_bytes = PREDECESSOR_PATH.read_bytes()
    if sha256_bytes(predecessor_bytes) != PREDECESSOR_SHA256:
        raise ValueError("immutable predecessor digest changed")
    validated_predecessor = validate_kernel_cycle_lut(predecessor_bytes)
    predecessor = canonical_loads(validated_predecessor.canonical)
    definitions, paths = _verify_sources(evidence, evidence_root)
    successor_value, observations = _build_successor(
        predecessor, evidence, definitions, paths
    )
    successor = validate_kernel_cycle_lut(successor_value)

    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / "candidate-record.json"
    record_path.write_bytes(successor.canonical)
    profile_path = output_dir / "profile-table.json"
    _profile_payload(profile_path, successor_value, successor.record_id)
    device_path = output_dir / "device-service-entries.json"
    _write_json(device_path, _device_payload(successor_value, successor.record_id))
    result = {
        "schema": "simllm-hopper-kernel-cycle-campaign-result-v1",
        "acceptance_status": "candidate",
        "lookup_record_sha256": successor.record_id,
        "predecessor_lookup_record_sha256": PREDECESSOR_SHA256,
        "freeze_sha256": EXPECTED_FREEZE_SHA256,
        "campaign_evidence_sha256": sha256_bytes(INPUT_PATH.read_bytes()),
        "score": _score_campaign(successor_value, evidence, paths, observations),
    }
    result_path = output_dir / "result.json"
    _write_json(result_path, result)
    _write_json(
        output_dir / "artifact-manifest.json",
        _artifact_manifest(
            output_dir,
            [record_path.name, profile_path.name, device_path.name, result_path.name],
        ),
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = compile_campaign(args.evidence_root.resolve(), args.output_dir.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
