"""Audit and score the frozen TRAF-65 Merlin hardware cells.

The scorer is intentionally conservative. It verifies every content-addressed
attempt before reading observations, scores only frozen metrics that the row
schema identifies directly, and records unavailable observables instead of
substituting candidate values. It never edits the expectations freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STUDY_ROOT = Path(__file__).resolve().parent
EXPECTATIONS_PATH = STUDY_ROOT / "expectations.json"
FREEZE_SHA256 = "212a7a26f54e444c9b18f1e528bd0d00b5a28e4f9e005b0dc137f477ad642571"
EXECUTION_HEAD = "2ab092f9255d77c00c547446b65534a3b273ec82"
CANDIDATE_PROFILE_SHA256 = (
    "899712c4734f7a6b410d80231291663a404511528d46aab7497b73831e0e354f"
)
IMPLEMENTATION_SHA256 = "af6801d25f105b612dfa5ca475f33d03d1306bf0e3c80c72089310d0de53b643"
PRODUCER_BINARY_SHA256 = (
    "96b4c544de54457d1fbed8e56b0a1cbe61344bcdab02d6445c07a0ab637277a4"
)
CELL_SCHEMA = "simllm-a100-nvlink-packet-cell-v1"
MANIFEST_SCHEMA = "simllm-a100-nvlink-packet-attempt-manifest-v1"
OBSERVATION_SCHEMA = "simllm-a100-nvlink-packet-observation-v1"
SCORE_SCHEMA = "simllm-a100-nvlink-packet-hardware-score-v1"

CORNER_MODULES = {
    "NVPKT": ("TX", "Switch", "RX"),
    "NVBOND": ("TX", "Switch", "RX"),
    "NVINC": ("TX", "Switch", "RX"),
    "NVCRD": ("TX", "Switch", "RX"),
    "NVHOL": ("TX", "Switch", "RX"),
}

PRIMARY_EVALUATORS = {
    "copy_engine_ordered_pair_payload_rate": "all_copy_engine_rates",
    "copy_engine_three_way_fanout_payload_rate": "three_way_fanout",
    "three_way_fanin_payload_rate": "three_way_fanin",
    "aggregate_ingress_rate_over_300GBps": "peak_ingress_ratio",
}

MISSING_GUARD_OBSERVABLES = (
    "per-row raw and data counter deltas for raw-bytes-below-data",
    "per-row counter sequence for monotonicity",
    "per-row replay, recovery, and error deltas",
    "a measured payload checksum or destination-byte comparison",
    "a recorded throttle verdict derived from clocks, power, and temperature",
)


@dataclass(frozen=True, kw_only=True)
class CellSpec:
    index: int
    cell_id: str
    frame: str
    case_names: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class Attempt:
    path: Path
    cell: CellSpec
    rows: tuple[dict[str, Any], ...]
    plan: dict[str, Any]
    environment: dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--profile-out", type=Path)
    parser.add_argument("--scheduler-job", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    score = audit_hardware(args.bulk_root, scheduler_job=args.scheduler_job)
    if args.json_out is not None:
        write_json(args.json_out, score)
    if args.markdown_out is not None:
        write_text(args.markdown_out, render_markdown(score))
    if args.profile_out is not None:
        publish_candidate_profile(score, args.profile_out)
    if args.json_out is None and args.markdown_out is None and args.profile_out is None:
        print(json.dumps(score, indent=2, sort_keys=True))
    return 0


def audit_hardware(bulk_root: Path, *, scheduler_job: str = "") -> dict[str, Any]:
    expectations = load_expectations()
    cells = build_cells(expectations)
    attempts: dict[int, Attempt] = {}
    rejected_attempts: list[dict[str, str]] = []
    for cell in cells:
        cell_root = bulk_root / FREEZE_SHA256 / "cells" / cell.cell_id
        accepted = []
        for attempt_path in sorted(cell_root.glob("attempt-*")):
            try:
                accepted.append(load_attempt(attempt_path, cell))
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                rejected_attempts.append(
                    {"cell_id": cell.cell_id, "path": str(attempt_path), "reason": str(error)}
                )
        if len(accepted) > 1:
            rejected_attempts.append(
                {
                    "cell_id": cell.cell_id,
                    "path": str(cell_root),
                    "reason": "multiple digest-complete attempts match the frozen execution",
                }
            )
        if accepted:
            attempts[cell.index] = accepted[0]

    completed_indices = sorted(attempts)
    prefix_count = 0
    while prefix_count in attempts:
        prefix_count += 1
    pending_indices = [cell.index for cell in cells if cell.index not in attempts]
    binary_digests = sorted(
        {str(attempt.plan["producer_binary_sha256"]) for attempt in attempts.values()}
    )
    result_rows = [row for attempt in attempts.values() for row in attempt.rows]
    protocol_validation_row_count = sum(
        row.get("protocol_scope") == "protocol_validation_only" for row in result_rows
    )
    case_scores = score_cases(expectations, cells, attempts)
    corner_verdicts = score_corners(case_scores)
    validity = "VOID_FATAL_GUARD_COVERAGE_INCOMPLETE" if attempts else "PENDING"
    if prefix_count == len(cells):
        status = "COMPLETE_VOID_86_OF_86"
    elif prefix_count:
        status = f"PARTIAL_VOID_{prefix_count}_OF_86"
    else:
        status = "PENDING_0_OF_86"
    return {
        "schema": SCORE_SCHEMA,
        "study_id": expectations["study_id"],
        "task_id": "TRAF-65",
        "status": status,
        "task_status": "OPEN",
        "measurement_validity": validity,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "scheduler_job": scheduler_job or None,
        "execution_head": EXECUTION_HEAD,
        "freeze_sha256": FREEZE_SHA256,
        "bulk_root": str(bulk_root),
        "coverage": {
            "completed_cell_count": len(completed_indices),
            "completed_indices": completed_indices,
            "completed_prefix_count": prefix_count,
            "completed_prefix_indices": list(range(prefix_count)),
            "pending_indices": pending_indices,
            "pending_array": compress_indices(pending_indices),
            "result_row_count": len(result_rows),
            "protocol_validation_row_count": protocol_validation_row_count,
            "rejected_attempts": rejected_attempts,
        },
        "producer_binary_audit": {
            "resume_compile_check_sha256": PRODUCER_BINARY_SHA256,
            "observed_batch_binary_sha256": binary_digests,
            "status": (
                "MATCH"
                if binary_digests == [PRODUCER_BINARY_SHA256]
                else "BUILD_REPRODUCIBILITY_MISMATCH"
                if binary_digests
                else "PENDING"
            ),
            "decision": (
                "The merged batch entry point pins the binary digest into each plan but does "
                "not require equality to the earlier compile-check digest. A single observed "
                "batch digest is accepted as execution lineage and the mismatch is published."
            ),
        },
        "fatal_guard_audit": {
            "status": validity,
            "passed_by_completed_attempt_construction": [
                "four A100-SXM4-80GB devices",
                "NV4 direct-mesh topology",
                "no competing compute process at cell boundaries",
                "guard commands returned success",
                "row schema reported checksum_ok=true",
            ],
            "unavailable": list(MISSING_GUARD_OBSERVABLES),
            "decision": (
                "The freeze says a fired guard voids a run. The capture does not record enough "
                "information to decide five guards, so completed timings cannot be promoted to "
                "measurement evidence."
            ),
        },
        "capture_contract_audit": {
            "status": "REFUTED_AS_IDENTIFICATION_CAPTURE",
            "candidate_not_observation_fields": [
                "candidate_packet_count",
                "candidate_raw_bytes",
            ],
            "parsed_but_not_applied_hardware_controls": [
                "access_width",
                "lane_mask",
                "stream_count",
                "outstanding",
                "burst_messages",
                "gap_ns",
                "offered_rate_percent",
            ],
            "copy_engine_batch_contract": (
                "REFUTED: run_hardware enqueues one cudaMemcpyPeerAsync operation per "
                "message, contrary to the frozen no-launch-per-message requirement"
            ),
            "payload_conservation_contract": (
                "UNMEASURED: the emitted checksum is a point-id hash and checksum_ok "
                "is never derived from destination bytes"
            ),
            "counter_contract": (
                "UNMEASURED: only cell-boundary text snapshots are retained; no result "
                "row carries raw, data, direction, link, replay, recovery, or error deltas"
            ),
        },
        "corner_verdicts": corner_verdicts,
        "case_scores": case_scores,
        "module_parameter_identification": module_parameter_identification(attempts),
        "candidate_profile_decision": {
            "status": "RETAIN_DECLARED_CANDIDATE_NO_HARDWARE_PROMOTION",
            "evidence_class": "declared_candidate_not_hardware_measurement",
            "parameter_value_changes": [],
            "switch_decision": "PASS_THROUGH_STANDS_STRUCTURALLY_NOT_MEASURED",
            "packet_overhead_decision": "UNIDENTIFIABLE_FROM_CAPTURED_ROW_SCHEMA",
            "copy_engine_coalescing_decision": "UNIDENTIFIABLE_FROM_CAPTURED_ROW_SCHEMA",
            "reason": (
                "No completed timing can satisfy the frozen measurement-validity contract while "
                "the raw/data, replay, recovery, corruption, and throttle observables are absent."
            ),
        },
    }


def load_expectations() -> dict[str, Any]:
    if sha256(EXPECTATIONS_PATH) != FREEZE_SHA256:
        raise RuntimeError("TRAF-65 expectations digest changed")
    payload = load_json(EXPECTATIONS_PATH)
    if payload.get("status") != "expectations_only":
        raise RuntimeError("TRAF-65 expectations status changed")
    if len(payload.get("catalog", [])) != 80:
        raise RuntimeError("TRAF-65 expectations must contain 80 cases")
    return payload


def build_cells(expectations: dict[str, Any]) -> tuple[CellSpec, ...]:
    catalog = expectations["catalog"]
    cells = [
        CellSpec(
            index=index,
            cell_id=f"isolated-{index + 1:03d}",
            frame="isolated",
            case_names=(str(case["stable_name"]),),
        )
        for index, case in enumerate(catalog)
    ]
    corners = list(dict.fromkeys(str(case["corner"]) for case in catalog))
    for corner in corners:
        cells.append(
            CellSpec(
                index=len(cells),
                cell_id=f"corner-frame-{corner.replace('_', '-')}",
                frame="corner_frame",
                case_names=tuple(
                    str(case["stable_name"])
                    for case in catalog
                    if str(case["corner"]) == corner
                ),
            )
        )
    cells.append(
        CellSpec(
            index=len(cells),
            cell_id="all-corners-frame",
            frame="all_corners_frame",
            case_names=tuple(str(case["stable_name"]) for case in catalog),
        )
    )
    if len(cells) != 86:
        raise RuntimeError("TRAF-65 cell catalog is not 86 cells")
    return tuple(cells)


def load_attempt(path: Path, cell: CellSpec) -> Attempt:
    manifest_path = path / "manifest.json"
    complete_path = path / "COMPLETE.json"
    manifest = load_json(manifest_path)
    complete = load_json(complete_path)
    if manifest.get("schema") != MANIFEST_SCHEMA or complete.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError("manifest schema mismatch")
    if manifest.get("cell_id") != cell.cell_id or complete.get("cell_id") != cell.cell_id:
        raise RuntimeError("cell identity mismatch")
    if manifest.get("freeze_sha256") != FREEZE_SHA256:
        raise RuntimeError("attempt freeze mismatch")
    if complete.get("status") != "complete":
        raise RuntimeError("attempt is not complete")
    if complete.get("manifest_sha256") != sha256(manifest_path):
        raise RuntimeError("manifest digest mismatch")
    payload_names = set()
    for payload in manifest.get("payloads", []):
        name = payload["path"]
        if not isinstance(name, str) or Path(name).name != name or name in payload_names:
            raise RuntimeError("unsafe or duplicate manifest payload")
        payload_names.add(name)
        payload_path = path / name
        if payload_path.stat().st_size != payload["bytes"] or sha256(payload_path) != payload["sha256"]:
            raise RuntimeError(f"payload digest mismatch: {name}")
    required = {
        "environment.json",
        "guards_after.txt",
        "guards_before.txt",
        "plan.json",
        "points.tsv",
        "results.jsonl",
        "stderr.txt",
        "stdout.txt",
        "summary.json",
    }
    if payload_names != required:
        raise RuntimeError("manifest payload inventory mismatch")
    plan = load_json(path / "plan.json")
    environment = load_json(path / "environment.json")
    summary = load_json(path / "summary.json")
    if plan.get("schema") != CELL_SCHEMA or environment.get("schema") != CELL_SCHEMA:
        raise RuntimeError("cell schema mismatch")
    expected_plan = {
        "mode": "hardware",
        "freeze_sha256": FREEZE_SHA256,
        "candidate_profile_sha256": CANDIDATE_PROFILE_SHA256,
        "implementation_sha256": IMPLEMENTATION_SHA256,
        "expected_head": EXECUTION_HEAD,
    }
    for key, expected in expected_plan.items():
        if plan.get(key) != expected:
            raise RuntimeError(f"plan {key} mismatch")
    binary_digest = plan.get("producer_binary_sha256")
    if (
        not isinstance(binary_digest, str)
        or len(binary_digest) != 64
        or any(character not in "0123456789abcdef" for character in binary_digest)
    ):
        raise RuntimeError("plan producer binary digest is invalid")
    if plan.get("cell", {}).get("index") != cell.index:
        raise RuntimeError("plan array index mismatch")
    if environment.get("mode") != "hardware" or environment.get("source_head") != EXECUTION_HEAD:
        raise RuntimeError("environment execution identity mismatch")
    if environment.get("slurm_partition") != "a100-hourly":
        raise RuntimeError("cell did not run on a100-hourly")
    if summary.get("status") != "hardware_unscored":
        raise RuntimeError("cell summary is not hardware_unscored")
    rows = read_rows(path / "results.jsonl")
    if len(rows) != plan.get("point_count") or len(rows) != summary.get("row_count"):
        raise RuntimeError("row count mismatch")
    for row in rows:
        if row.get("schema") != OBSERVATION_SCHEMA or row.get("mode") != "hardware":
            raise RuntimeError("observation schema or mode mismatch")
        if row.get("measurement_claim") != "unscored":
            raise RuntimeError("observation was scored before this scoring path")
        if row.get("checksum_ok") is not True:
            raise RuntimeError("observation reports checksum failure")
        if not observation_belongs_to_cell(row, cell):
            raise RuntimeError("observation case is outside its cell")
    return Attempt(path=path, cell=cell, rows=rows, plan=plan, environment=environment)


def observation_belongs_to_cell(row: dict[str, Any], cell: CellSpec) -> bool:
    if row.get("case_name") in cell.case_names:
        return True
    return (
        cell.frame == "all_corners_frame"
        and row.get("case_name") == cell.cell_id
        and row.get("producer") == "nccl_send_receive_validation"
        and row.get("protocol_scope") == "protocol_validation_only"
    )


def score_cases(
    expectations: dict[str, Any],
    cells: tuple[CellSpec, ...],
    attempts: dict[int, Attempt],
) -> list[dict[str, Any]]:
    cells_by_case: dict[str, list[CellSpec]] = {}
    for cell in cells:
        for case_name in cell.case_names:
            cells_by_case.setdefault(case_name, []).append(cell)
    results = []
    for case in expectations["catalog"]:
        name = str(case["stable_name"])
        required_cells = cells_by_case[name]
        present = [cell for cell in required_cells if cell.index in attempts]
        rows = [
            {**row, "_frame": cell.frame}
            for cell in present
            for row in attempts[cell.index].rows
            if row.get("case_name") == name
        ]
        expected_band = case["expected_band"]
        metric = str(expected_band["metric"])
        if not present:
            metric_score = {
                "status": "PENDING",
                "observed": None,
                "reason": "no digest-complete hardware cell for this case",
            }
        elif metric in PRIMARY_EVALUATORS:
            metric_score = evaluate_primary(metric, rows, expected_band)
        else:
            metric_score = {
                "status": "UNSCORABLE",
                "observed": None,
                "reason": missing_metric_reason(metric),
            }
        if present and len(present) < len(required_cells):
            coverage = "PARTIAL"
        elif len(present) == len(required_cells):
            coverage = "COMPLETE"
        else:
            coverage = "PENDING"
        results.append(
            {
                "ordinal": case["ordinal"],
                "case_name": name,
                "corner": case["corner"],
                "coverage": coverage,
                "completed_frames": [cell.frame for cell in present],
                "pending_cells": [cell.index for cell in required_cells if cell.index not in attempts],
                "row_count": len(rows),
                "expected_band": expected_band,
                "metric_score": metric_score,
                "measurement_verdict": (
                    "VOID_FATAL_GUARD_COVERAGE_INCOMPLETE" if present else "PENDING"
                ),
            }
        )
    return results


def evaluate_primary(
    metric: str, rows: list[dict[str, Any]], expected_band: dict[str, Any]
) -> dict[str, Any]:
    low = float(expected_band["low"])
    high = float(expected_band["high"])
    if metric == "copy_engine_ordered_pair_payload_rate":
        values = [
            float(row["payload_rate_gbps"])
            for row in rows
            if row.get("producer") == "copy_engine_reference"
        ]
    elif metric == "copy_engine_three_way_fanout_payload_rate":
        values = [
            float(row["payload_rate_gbps"])
            for row in rows
            if row.get("producer") == "copy_engine_reference"
            and "destinations=3" in str(row.get("point_id"))
        ]
    elif metric == "three_way_fanin_payload_rate":
        values = [
            float(row["payload_rate_gbps"])
            for row in rows
            if row.get("producer") == "copy_engine_reference"
            and "sources=3" in str(row.get("point_id"))
        ]
    elif metric == "aggregate_ingress_rate_over_300GBps":
        by_frame_peak: dict[str, float] = {}
        for row in rows:
            frame = str(row.get("_frame", "combined"))
            by_frame_peak[frame] = max(
                by_frame_peak.get(frame, 0.0), float(row["payload_rate_gbps"]) / 300.0
            )
        values = list(by_frame_peak.values())
    else:
        raise AssertionError(metric)
    if not values:
        return {
            "status": "UNSCORABLE",
            "observed": None,
            "reason": "the required frozen producer or saturation point is absent",
        }
    passed = all(low <= value <= high for value in values)
    return {
        "status": "PASS" if passed else "REFUTED",
        "observed": {"minimum": min(values), "maximum": max(values), "values": values},
        "reason": f"all selected values must be in [{low}, {high}]",
    }


def missing_metric_reason(metric: str) -> str:
    if metric == "candidate_wire_byte_residual":
        return "results omit observed raw/data counter deltas and counter quantum"
    if metric == "max_over_min_per_link_payload_rate":
        return "results omit per-link payload and raw counter deltas"
    if "knee" in metric or "gap_adjusted" in metric:
        return "results do not record an applied outstanding, burst, gap, or offered-rate control"
    if "isolated_control" in metric or "dispersed_region_control" in metric:
        return "results contain no identified latency-flow/control pair or completion-order ledger"
    if metric == "drain_time_over_serialization_floor":
        return "results omit drain time and the post-burst credit/buffer baseline"
    return "the freeze does not define a result-row reduction for this metric"


def score_corners(case_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verdicts = []
    corners = list(dict.fromkeys(str(case["corner"]) for case in case_scores))
    for corner in corners:
        cases = [case for case in case_scores if case["corner"] == corner]
        states = [case["metric_score"]["status"] for case in cases]
        if all(state == "PENDING" for state in states):
            verdict = "PENDING"
        elif "REFUTED" in states:
            verdict = "MEASURED_BAND_REFUTED_BUT_RUN_VOID"
        else:
            verdict = "UNSCORABLE_RUN_VOID"
        verdicts.append(
            {
                "corner": corner,
                "verdict": verdict,
                "case_counts": {
                    state: states.count(state)
                    for state in ("PASS", "REFUTED", "UNSCORABLE", "PENDING")
                },
            }
        )
    return verdicts


def module_parameter_identification(attempts: dict[int, Attempt]) -> list[dict[str, str]]:
    if attempts:
        unmeasured = "UNIDENTIFIABLE_RUN_VOID"
    else:
        unmeasured = "PENDING"
    return [
        {
            "module": "TX",
            "parameters": "maximum payload; header; packet granularity",
            "status": unmeasured,
            "reason": "no per-row observed raw/data counter deltas",
        },
        {
            "module": "TX",
            "parameters": "links per peer; per-link rate; bond policy",
            "status": unmeasured,
            "reason": "no per-row per-link counter deltas",
        },
        {
            "module": "TX",
            "parameters": "request/response direction",
            "status": unmeasured,
            "reason": "no direction-specific request/data counter ledger",
        },
        {
            "module": "TX",
            "parameters": "effective credit unit and destination window",
            "status": unmeasured,
            "reason": "no recorded applied window and no valid first-knee evidence",
        },
        {
            "module": "Switch",
            "parameters": "pass-through mode; zero bytes and time",
            "status": "STANDS_STRUCTURALLY_NOT_MEASURED" if attempts else "PENDING",
            "reason": "mandatory direct-mesh model invariant; hardware rows expose no switch visit",
        },
        {
            "module": "RX",
            "parameters": "ingress rate and buffer capacity",
            "status": unmeasured,
            "reason": "no valid destination counter, occupancy, overflow, or drain ledger",
        },
        {
            "module": "RX",
            "parameters": "credit return latency",
            "status": unmeasured,
            "reason": "no credit-return event or applied recovery-gap observation",
        },
        {
            "module": "RX",
            "parameters": "reassembly and delivery order",
            "status": unmeasured,
            "reason": "checksum_ok is reported without measured destination bytes or an order ledger",
        },
    ]


def render_markdown(score: dict[str, Any]) -> str:
    coverage = score["coverage"]
    lines = [
        "# TRAF-65 A100 NVLink packet hardware score",
        "",
        f"Status: `{score['status']}`.",
        "",
        "## Per-corner verdicts",
        "",
        "| Corner | Verdict | Pass | Refuted | Unscorable | Pending |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for corner in score["corner_verdicts"]:
        counts = corner["case_counts"]
        lines.append(
            f"| {corner['corner']} | {corner['verdict']} | {counts['PASS']} | "
            f"{counts['REFUTED']} | {counts['UNSCORABLE']} | {counts['PENDING']} |"
        )
    lines.extend(
        [
            "",
            "## Module-parameter identification",
            "",
            "| Module | Parameters | Status | Reason |",
            "|---|---|---|---|",
        ]
    )
    for row in score["module_parameter_identification"]:
        lines.append(
            f"| {row['module']} | {row['parameters']} | {row['status']} | {row['reason']} |"
        )
    decision = score["candidate_profile_decision"]
    lines.extend(
        [
            "",
            "## Candidate-profile decision",
            "",
            f"The decision is `{decision['status']}`. " + decision["reason"],
            "",
            (
                f"The switch result is `{decision['switch_decision']}`. The packet-overhead "
                f"result is `{decision['packet_overhead_decision']}`, and the copy-engine "
                f"coalescing result is `{decision['copy_engine_coalescing_decision']}`."
            ),
            "",
            "## Execution coverage and exact remainder",
            "",
            f"- Scheduler job: `{score['scheduler_job']}`.",
            f"- Frozen execution head: `{score['execution_head']}`.",
            f"- Frozen expectations SHA-256: `{score['freeze_sha256']}`.",
            f"- Digest-complete cells: {coverage['completed_cell_count']} of 86.",
            (
                f"- Hardware rows: {coverage['result_row_count']:,}, including "
                f"{coverage['protocol_validation_row_count']} protocol-validation rows."
            ),
            f"- Consecutive completed prefix: indices 0 through "
            f"{coverage['completed_prefix_count'] - 1}."
            if coverage["completed_prefix_count"]
            else "- Consecutive completed prefix: empty.",
            f"- Exact pending array: `{coverage['pending_array'] or 'none'}`.",
            (
                f"- Batch binary audit: `{score['producer_binary_audit']['status']}`; observed "
                f"`{','.join(score['producer_binary_audit']['observed_batch_binary_sha256']) or 'none'}` "
                "against compile-check "
                f"`{score['producer_binary_audit']['resume_compile_check_sha256']}`."
            ),
            "",
            "## Freeze chronology and guard ruling",
            "",
            (
                "The written resume record held submissions until `2026-08-28T06:30` for "
                "maintenance reservation `SD26082026`. On 2026-08-27 the integrator verified "
                "that the reservation lifted early and the A100 partitions were again visible "
                "in mixed and allocated states. That verified node state superseded only the "
                "submission date, not the freeze or occupancy rules."
            ),
            "",
            score["fatal_guard_audit"]["decision"],
            "",
            (
                "The capture contract is `REFUTED_AS_IDENTIFICATION_CAPTURE`: candidate packet "
                "and raw-byte counts are derived fields rather than observations; access width, "
                "lane mask, stream count, outstanding window, burst length, gap, and offered "
                "rate are parsed but not applied by the hardware path; and the copy-engine loop "
                "enqueues one peer copy per message. The emitted checksum is a point-id hash, "
                "not a measured destination checksum."
            ),
            "",
            "The scorer made no expectations amendment and changed no candidate parameter value.",
            "",
            "## Per-case frozen-band score",
            "",
            "| Case | Coverage | Metric status | Measurement verdict |",
            "|---|---|---|---|",
        ]
    )
    for case in score["case_scores"]:
        lines.append(
            f"| `{case['case_name']}` | {case['coverage']} | "
            f"{case['metric_score']['status']} | {case['measurement_verdict']} |"
        )
    lines.append("")
    return "\n".join(lines)


def publish_candidate_profile(score: dict[str, Any], path: Path) -> None:
    if score["coverage"]["completed_cell_count"] == 0:
        raise RuntimeError("refusing to publish a candidate-profile decision without a cell")
    profile_path = STUDY_ROOT / "candidate-profile.json"
    profile = load_json(profile_path)
    prior = profile.pop("hardware_scoring", None)
    base_profile_bytes = (json.dumps(profile, indent=2) + "\n").encode()
    if hashlib.sha256(base_profile_bytes).hexdigest() != CANDIDATE_PROFILE_SHA256:
        raise RuntimeError("candidate profile changed outside the TRAF-65 scoring path")
    if prior is not None and (
        not isinstance(prior, dict)
        or prior.get("schema") != SCORE_SCHEMA
        or prior.get("measurement_claim") is not False
        or prior.get("execution_head") != EXECUTION_HEAD
        or prior.get("freeze_sha256") != FREEZE_SHA256
    ):
        raise RuntimeError("candidate profile changed outside the TRAF-65 scoring path")
    if profile.get("status") != "candidate":
        raise RuntimeError("TRAF-65 profile is no longer a candidate")
    if profile.get("evidence_class") != "declared_candidate_not_hardware_measurement":
        raise RuntimeError("TRAF-65 profile has an unsupported evidence claim")
    expected_parameters = {
        "tx": {
            "max_payload_bytes": 256,
            "header_bytes": 16,
            "links_per_peer": 4,
            "per_link_rate_bytes_per_second": 25_000_000_000,
            "endpoint_egress_rate_bytes_per_second": 300_000_000_000,
            "bond_policy": "earliest_available_packet_striping",
            "credits_per_destination": 256,
            "credit_unit_bytes": 272,
        },
        "switch": {"mode": "pass_through"},
        "rx": {
            "ingress_rate_bytes_per_second": 300_000_000_000,
            "buffer_capacity_bytes": 1_048_576,
            "credit_return_latency_ps": 200_000,
            "reassembly_policy": "extent_sequence",
            "delivery_order": "per_extent",
        },
    }
    for module, expected in expected_parameters.items():
        if profile.get(module) != expected:
            raise RuntimeError(f"candidate {module} values changed outside the scoring path")
    hardware_scoring = {
        "schema": SCORE_SCHEMA,
        "status": score["candidate_profile_decision"]["status"],
        "measurement_claim": False,
        "measurement_validity": score["measurement_validity"],
        "execution_head": score["execution_head"],
        "freeze_sha256": score["freeze_sha256"],
        "completed_cell_count": score["coverage"]["completed_cell_count"],
        "completed_prefix_count": score["coverage"]["completed_prefix_count"],
        "pending_array": score["coverage"]["pending_array"],
        "parameter_value_changes": score["candidate_profile_decision"][
            "parameter_value_changes"
        ],
        "switch_decision": score["candidate_profile_decision"]["switch_decision"],
        "packet_overhead_decision": score["candidate_profile_decision"][
            "packet_overhead_decision"
        ],
        "copy_engine_coalescing_decision": score["candidate_profile_decision"][
            "copy_engine_coalescing_decision"
        ],
    }
    profile["hardware_scoring"] = hardware_scoring
    write_ordered_json(path, profile)


def compress_indices(indices: list[int]) -> str:
    if not indices:
        return ""
    ranges = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def read_rows(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    with open(path, encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid result JSON at line {line_number}") from error
            if not isinstance(row, dict):
                raise TypeError(f"result line {line_number} is not an object")
            rows.append(row)
    return tuple(rows)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8", newline="") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} is not a JSON object")
    return payload


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_ordered_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
