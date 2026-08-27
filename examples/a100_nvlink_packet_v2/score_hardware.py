"""Audit and score the frozen corrected TRAF-70 hardware capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_study

STUDY_ROOT = Path(__file__).resolve().parent
EXPECTATIONS_PATH = STUDY_ROOT / "expectations.json"
FREEZE_SHA256 = run_study.FREEZE_SHA256
PROTECTED_CANDIDATE_SHA256 = run_study.PROTECTED_CANDIDATE_SHA256
OBSERVATION_SCHEMA = "simllm-a100-nvlink-packet-observation-v2"
SCORE_SCHEMA = "simllm-a100-nvlink-packet-hardware-score-v2"
GUARD_IDS = tuple(f"FG{index:02d}" for index in range(1, 11))
CONTROL_FIELDS = (
    "payload_bytes",
    "message_count",
    "source",
    "destination",
    "sources",
    "destinations",
    "source_alignment",
    "destination_alignment",
    "access_width",
    "active_lanes",
    "lane_mask",
    "stride",
    "stream_count",
    "outstanding",
    "burst_messages",
    "gap_ns",
    "offered_rate_percent",
    "pattern",
)
FORBIDDEN_OBSERVATION_FIELDS = {
    "candidate_packet_count",
    "candidate_raw_bytes",
    "candidate_header_bytes",
    "candidate_payload_bytes",
    "predicted_raw_bytes",
}


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
    guards_before: str
    guards_after: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk-root", type=Path, required=True)
    parser.add_argument("--expected-head", default="")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--scheduler-job", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    score = audit_hardware(
        args.bulk_root,
        expected_head=args.expected_head,
        scheduler_job=args.scheduler_job,
    )
    if args.json_out is not None:
        write_json(args.json_out, score)
    if args.markdown_out is not None:
        write_text(args.markdown_out, render_markdown(score))
    if args.json_out is None and args.markdown_out is None:
        print(json.dumps(score, indent=2, sort_keys=True))
    return 0


def audit_hardware(
    bulk_root: Path,
    *,
    expected_head: str = "",
    scheduler_job: str = "",
) -> dict[str, Any]:
    expectations = load_expectations()
    cells = build_cells(expectations)
    attempts: dict[int, Attempt] = {}
    rejected_attempts: list[dict[str, str]] = []
    for cell in cells:
        accepted: list[Attempt] = []
        cell_root = bulk_root / FREEZE_SHA256 / "cells" / cell.cell_id
        for attempt_path in sorted(cell_root.glob("attempt-*")):
            try:
                accepted.append(load_attempt(attempt_path, cell, expected_head=expected_head))
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                rejected_attempts.append(
                    {
                        "cell_id": cell.cell_id,
                        "path": str(attempt_path),
                        "reason": str(error),
                    }
                )
        if len(accepted) > 1:
            rejected_attempts.append(
                {
                    "cell_id": cell.cell_id,
                    "path": str(cell_root),
                    "reason": "multiple matching digest-complete attempts",
                }
            )
        if accepted:
            attempts[cell.index] = accepted[0]

    completed_indices = sorted(attempts)
    pending_indices = [cell.index for cell in cells if cell.index not in attempts]
    prefix_count = 0
    while prefix_count in attempts:
        prefix_count += 1
    rows_with_context = [
        {**row, "_cell_index": index, "_frame": attempt.cell.frame}
        for index, attempt in sorted(attempts.items())
        for row in attempt.rows
    ]
    row_guard_records = [score_row_guards(row) for row in rows_with_context]
    cell_guard_records = [score_cell_guards(attempt) for attempt in attempts.values()]
    cell_guard_records.append(score_scorer_guard())
    guard_summary = summarize_guards(row_guard_records, cell_guard_records)
    all_cells_complete = len(attempts) == len(cells)
    guards_decidable = all(item["decidable"] for item in guard_summary["guards"])
    guards_pass = all(item["status"] == "PASS" for item in guard_summary["guards"])
    measurement_valid = all_cells_complete and guards_decidable and guards_pass
    scoring_audit: dict[str, Any] = {}
    parameter_results = score_parameters(
        expectations,
        rows_with_context,
        all_cells_complete=all_cells_complete,
        measurement_valid=measurement_valid,
        scoring_audit=scoring_audit,
    )
    case_scores = score_cases(expectations, rows_with_context, attempts, cells)
    gate = score_flow_dynamics_gate(
        all_cells_complete=all_cells_complete,
        guards_decidable=guards_decidable,
        parameter_results=parameter_results,
    )
    binary_digests = sorted(
        {str(attempt.plan["producer_binary_sha256"]) for attempt in attempts.values()}
    )
    execution_heads = sorted(
        {str(attempt.plan["expected_head"]) for attempt in attempts.values()}
    )
    if not attempts:
        status = "PENDING_0_OF_86"
    elif not all_cells_complete:
        status = f"PARTIAL_{len(attempts)}_OF_86"
    elif guards_pass:
        status = "COMPLETE_VALID_86_OF_86"
    else:
        status = "COMPLETE_FATAL_86_OF_86"
    profile_patch = profile_patch_from_results(parameter_results)
    return {
        "schema": SCORE_SCHEMA,
        "study_id": expectations["study_id"],
        "task_id": "TRAF-70",
        "status": status,
        "task_status": "OPEN",
        "measurement_validity": (
            "VALID_FOR_FROZEN_RULES" if measurement_valid else
            "PENDING" if not attempts else
            "INVALID_FATAL_GUARD" if all_cells_complete else
            "PARTIAL_UNSCORED"
        ),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "scheduler_job": scheduler_job or None,
        "execution_heads": execution_heads,
        "freeze_sha256": FREEZE_SHA256,
        "protected_candidate_before_sha256": PROTECTED_CANDIDATE_SHA256,
        "bulk_root": str(bulk_root),
        "coverage": {
            "completed_cell_count": len(completed_indices),
            "completed_indices": completed_indices,
            "completed_prefix_count": prefix_count,
            "completed_prefix_indices": list(range(prefix_count)),
            "pending_indices": pending_indices,
            "pending_array": compress_indices(pending_indices),
            "result_row_count": len(rows_with_context),
            "protocol_validation_row_count": sum(
                row.get("protocol_scope") == "protocol_validation_only"
                for row in rows_with_context
            ),
            "rejected_attempts": rejected_attempts,
        },
        "producer_binary_audit": {
            "observed_batch_binary_sha256": binary_digests,
            "status": (
                "SINGLE_DIGEST" if len(binary_digests) == 1 else
                "PENDING" if not binary_digests else
                "MULTIPLE_DIGESTS_FATAL"
            ),
        },
        "fatal_guard_verdicts": guard_summary,
        "module_parameter_identification": parameter_results,
        "scoring_audit": scoring_audit,
        "profile_patch": profile_patch,
        "case_scores": case_scores,
        "flow_dynamics_gate": gate,
    }


def load_expectations() -> dict[str, Any]:
    if sha256(EXPECTATIONS_PATH) != FREEZE_SHA256:
        raise RuntimeError("TRAF-70 expectations digest changed")
    if sha256(run_study.PROTECTED_CANDIDATE_PROFILE_PATH) != PROTECTED_CANDIDATE_SHA256:
        raise RuntimeError("protected A100 candidate changed before score publication")
    payload = load_json(EXPECTATIONS_PATH)
    if payload.get("status") != "expectations_only_frozen_before_harness":
        raise RuntimeError("TRAF-70 expectations status changed")
    if len(payload.get("catalog", [])) != 80:
        raise RuntimeError("TRAF-70 expectations must contain 80 cases")
    return payload


def build_cells(expectations: dict[str, Any]) -> tuple[CellSpec, ...]:
    raw_cells = run_study._cells(expectations)
    return tuple(
        CellSpec(
            index=cell.index,
            cell_id=cell.cell_id,
            frame=cell.frame,
            case_names=cell.case_names,
        )
        for cell in raw_cells
    )


def load_attempt(path: Path, cell: CellSpec, *, expected_head: str) -> Attempt:
    manifest_path = path / "manifest.json"
    complete_path = path / "COMPLETE.json"
    manifest = load_json(manifest_path)
    complete = load_json(complete_path)
    if manifest.get("schema") != run_study.MANIFEST_SCHEMA:
        raise RuntimeError("manifest schema mismatch")
    if complete.get("schema") != run_study.MANIFEST_SCHEMA:
        raise RuntimeError("complete schema mismatch")
    if manifest.get("cell_id") != cell.cell_id or complete.get("cell_id") != cell.cell_id:
        raise RuntimeError("cell identity mismatch")
    if manifest.get("freeze_sha256") != FREEZE_SHA256:
        raise RuntimeError("attempt freeze mismatch")
    if complete.get("status") != "complete":
        raise RuntimeError("attempt is not complete")
    if complete.get("manifest_sha256") != sha256(manifest_path):
        raise RuntimeError("manifest digest mismatch")
    payload_names: set[str] = set()
    for payload in manifest.get("payloads", []):
        name = payload["path"]
        if not isinstance(name, str) or Path(name).name != name or name in payload_names:
            raise RuntimeError("unsafe or duplicate manifest payload")
        payload_names.add(name)
        payload_path = path / name
        if payload_path.stat().st_size != payload["bytes"]:
            raise RuntimeError(f"payload size mismatch: {name}")
        if sha256(payload_path) != payload["sha256"]:
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
    expected_plan = {
        "schema": run_study.CELL_SCHEMA,
        "mode": "hardware",
        "freeze_sha256": FREEZE_SHA256,
        "protected_candidate_profile_sha256": PROTECTED_CANDIDATE_SHA256,
        "implementation_sha256": run_study._implementation_digest(),
    }
    for key, expected in expected_plan.items():
        if plan.get(key) != expected:
            raise RuntimeError(f"plan {key} mismatch")
    if expected_head and plan.get("expected_head") != expected_head:
        raise RuntimeError("plan expected_head mismatch")
    if plan.get("cell", {}).get("index") != cell.index:
        raise RuntimeError("plan array index mismatch")
    if environment.get("mode") != "hardware":
        raise RuntimeError("environment mode mismatch")
    if environment.get("source_head") != plan.get("expected_head"):
        raise RuntimeError("environment execution head mismatch")
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
            raise RuntimeError("observation was scored before the scoring path")
        if not observation_belongs_to_cell(row, cell):
            raise RuntimeError("observation case is outside its cell")
    return Attempt(
        path=path,
        cell=cell,
        rows=rows,
        plan=plan,
        environment=environment,
        guards_before=(path / "guards_before.txt").read_text(encoding="utf-8"),
        guards_after=(path / "guards_after.txt").read_text(encoding="utf-8"),
    )


def observation_belongs_to_cell(row: dict[str, Any], cell: CellSpec) -> bool:
    if row.get("case_name") in cell.case_names:
        return True
    return (
        cell.frame == "all_corners_frame"
        and row.get("case_name") == cell.cell_id
        and row.get("producer") == "nccl_send_receive_validation"
        and row.get("protocol_scope") == "protocol_validation_only"
    )


def score_cell_guards(attempt: Attempt) -> dict[str, Any]:
    before = attempt.guards_before
    after = attempt.guards_after
    topology_ok = all(
        text.count("A100-SXM4-80GB") == 4 and _nv4_row_count(text) == 4
        for text in (before, after)
    )
    processes_clear = all(not _process_section(text).strip() for text in (before, after))
    allocation_ok = (
        attempt.environment.get("slurm_partition") == "a100-hourly"
        and bool(attempt.environment.get("slurm_job_id"))
    )
    return {
        "cell_id": attempt.cell.cell_id,
        "guards": {
            "FG02": {
                "decidable": True,
                "pass": topology_ok,
                "reason": "qualified NV4 boundary snapshots" if topology_ok else "NV4 boundary mismatch",
            },
            "FG07": {
                "decidable": True,
                "pass": processes_clear and allocation_ok,
                "reason": (
                    "exclusive allocation and empty boundary process lists"
                    if processes_clear and allocation_ok
                    else "boundary exclusivity or allocation metadata failed"
                ),
            },
        },
    }


def score_scorer_guard() -> dict[str, Any]:
    payload_values = tuple(range(16, 4097, 16))
    header_values = tuple(range(129))
    complete_grid = (
        payload_values[0] == 16
        and payload_values[-1] == 4096
        and len(payload_values) == 256
        and header_values[0] == 0
        and header_values[-1] == 128
        and len(header_values) == 129
    )
    return {
        "cell_id": "candidate-blind-scorer",
        "guards": {
            "FG09": _guard(
                True,
                complete_grid,
                "full frozen candidate-blind packet search grid",
            )
        },
    }


def _nv4_row_count(text: str) -> int:
    return sum(
        line.startswith(tuple(f"GPU{index}" for index in range(4)))
        and line.split().count("NV4") == 3
        for line in text.splitlines()
    )


def _process_section(text: str) -> str:
    match = re.search(
        r"=== processes returncode=0 ===\n(.*?)(?=\n=== |\Z)",
        text,
        flags=re.DOTALL,
    )
    return "missing-process-section" if match is None else match.group(1)


def score_row_guards(row: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}

    checksum = row.get("destination_checksum")
    order = row.get("ordering_ledger")
    fg01_decidable = isinstance(checksum, dict) and isinstance(order, dict) and all(
        key in checksum for key in ("expected_sha256", "observed_sha256", "matches")
    ) and all(
        key in order
        for key in (
            "expected_sequence_sha256",
            "observed_sequence_sha256",
            "expected_extents",
            "terminal_extents",
            "missing",
            "duplicate",
            "out_of_order",
        )
    )
    fg01_pass = bool(
        fg01_decidable
        and checksum["matches"] is True
        and checksum["expected_sha256"] == checksum["observed_sha256"]
        and order["expected_sequence_sha256"] == order["observed_sequence_sha256"]
        and order["expected_extents"] == order["terminal_extents"]
        and all(order[key] == 0 for key in ("missing", "duplicate", "out_of_order"))
    )
    results["FG01"] = _guard(fg01_decidable, fg01_pass, "destination integrity and order")

    links = _links(row)
    fg02_decidable = bool(links) and all(
        isinstance(link.get("gpu"), int)
        and isinstance(link.get("link"), int)
        and isinstance(link.get("remote_gpu"), int)
        for link in links
    )
    fg02_pass = fg02_decidable and all(0 <= link["remote_gpu"] <= 3 for link in links)
    results["FG02"] = _guard(fg02_decidable, fg02_pass, "per-row direct remote-link map")

    counter_names = ("data_tx", "data_rx", "raw_tx", "raw_rx")
    fg03_decidable = bool(links) and all(
        isinstance(link.get("statuses"), list)
        and len(link["statuses"]) == 9
        and all(f"{name}_kib_{side}" in link for name in counter_names for side in ("before", "after"))
        for link in links
    )
    fg03_pass = fg03_decidable and all(
        all(status == 0 for status in link["statuses"][:4])
        and all(link[f"{name}_kib_after"] >= link[f"{name}_kib_before"] for name in counter_names)
        for link in links
    )
    results["FG03"] = _guard(fg03_decidable, fg03_pass, "counter availability and monotonicity")

    fg04_decidable = fg03_decidable
    fg04_pass = fg04_decidable and all(
        link["raw_tx_kib_delta"] + 1 >= link["data_tx_kib_delta"]
        and link["raw_rx_kib_delta"] + 1 >= link["data_rx_kib_delta"]
        for link in links
    )
    results["FG04"] = _guard(fg04_decidable, fg04_pass, "raw/data consistency")

    error_names = ("replay", "recovery", "crc_flit", "crc_data", "ecc_data")
    fg05_decidable = bool(links) and all(
        isinstance(link.get("errors"), dict)
        and all(
            isinstance(link["errors"].get(name), dict)
            and all(key in link["errors"][name] for key in ("before", "after", "delta"))
            for name in error_names
        )
        and isinstance(link.get("statuses"), list)
        and len(link["statuses"]) == 9
        for link in links
    )
    fg05_pass = fg05_decidable and all(
        all(status == 0 for status in link["statuses"][4:])
        and all(link["errors"][name]["delta"] == 0 for name in error_names)
        for link in links
    )
    results["FG05"] = _guard(fg05_decidable, fg05_pass, "replay, recovery, CRC and ECC")

    elapsed = row.get("elapsed_us")
    fg06_decidable = fg03_decidable and isinstance(elapsed, (int, float)) and elapsed > 0
    fg06_pass = fg06_decidable and _rate_ceilings_pass(links, float(elapsed))
    results["FG06"] = _guard(fg06_decidable, fg06_pass, "physical rate ceilings")

    telemetry = [*row.get("telemetry_before", []), *row.get("telemetry_after", [])]
    fg07_decidable = (
        row.get("throttle_verdict") in {"CLEAR", "FATAL_CLOCK_EVENT"}
        and len(telemetry) == 8
        and all(
            isinstance(value, dict)
            and isinstance(value.get("statuses"), list)
            and len(value["statuses"]) == 5
            for value in telemetry
        )
    )
    fg07_pass = fg07_decidable and row["throttle_verdict"] == "CLEAR" and all(
        all(status == 0 for status in value["statuses"]) for value in telemetry
    )
    results["FG07"] = _guard(fg07_decidable, fg07_pass, "row throttle telemetry")

    controls = row.get("applied_controls")
    fg08_decidable = isinstance(controls, dict) and all(
        field in controls for field in CONTROL_FIELDS
    ) and isinstance(row.get("applied_control_sha256"), str)
    fg08_pass = bool(
        fg08_decidable
        and row["applied_control_sha256"] == applied_control_sha256(row, controls)
        and isinstance(controls.get("effects"), dict)
        and set(controls["effects"]) == set(CONTROL_FIELDS)
        and all(
            isinstance(effect, str) and effect and effect != "parsed_only"
            for effect in controls["effects"].values()
        )
    )
    if row.get("producer") == "copy_engine_reference":
        enqueue = row.get("copy_engine_host_enqueue_count")
        messages = row.get("message_count")
        fg08_decidable = fg08_decidable and isinstance(enqueue, int) and isinstance(messages, int)
        fg08_pass = fg08_pass and (messages <= 1 or enqueue < messages)
    results["FG08"] = _guard(fg08_decidable, fg08_pass, "applied sweep controls and batching")

    fg09_decidable = True
    fg09_pass = not FORBIDDEN_OBSERVATION_FIELDS.intersection(row)
    results["FG09"] = _guard(fg09_decidable, fg09_pass, "observation/hypothesis separation")

    fg10_decidable = all(
        isinstance(row.get(field), (int, float))
        for field in ("elapsed_us", "completion_us", "drain_us")
    ) and isinstance(order, dict)
    fg10_pass = bool(
        fg10_decidable
        and 0 < row["elapsed_us"] <= row["completion_us"] <= row["drain_us"]
        and order["expected_extents"] == order["terminal_extents"]
    )
    results["FG10"] = _guard(fg10_decidable, fg10_pass, "completion and drain")
    return {
        "cell_index": row.get("_cell_index"),
        "point_id": row.get("point_id"),
        "guards": results,
    }


def _guard(decidable: bool, passed: bool, reason: str) -> dict[str, Any]:
    return {
        "decidable": bool(decidable),
        "pass": bool(decidable and passed),
        "reason": reason if decidable else f"missing observable for {reason}",
    }


def _links(row: dict[str, Any]) -> list[dict[str, Any]]:
    counters = row.get("observed_counter_deltas")
    if not isinstance(counters, dict) or counters.get("unit") != "KiB":
        return []
    links = counters.get("per_gpu_per_link_per_direction")
    return links if isinstance(links, list) else []


def _rate_ceilings_pass(links: list[dict[str, Any]], elapsed_us: float) -> bool:
    denominator = elapsed_us * 1.0e3
    pair_rates: defaultdict[tuple[int, int, str], float] = defaultdict(float)
    endpoint_rates: defaultdict[tuple[int, str], float] = defaultdict(float)
    for link in links:
        for direction in ("tx", "rx"):
            rate = float(link[f"raw_{direction}_kib_delta"]) * 1024 / denominator
            if rate > 25.25:
                return False
            remote = int(link["remote_gpu"])
            gpu = int(link["gpu"])
            pair_rates[(gpu, remote, direction)] += rate
            endpoint_rates[(gpu, direction)] += rate
    return max(pair_rates.values(), default=0.0) <= 101.0 and max(
        endpoint_rates.values(), default=0.0
    ) <= 303.0


def applied_control_sha256(row: dict[str, Any], controls: dict[str, Any]) -> str:
    values: list[object] = [row.get("case_name"), row.get("point_id"), row.get("producer")]
    values.extend(controls[field] for field in CONTROL_FIELDS)
    text = "".join(f"{value}\n" for value in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def summarize_guards(
    row_records: list[dict[str, Any]],
    cell_records: list[dict[str, Any]],
) -> dict[str, Any]:
    summaries = []
    for guard_id in GUARD_IDS:
        observations = [
            record["guards"][guard_id]
            for record in row_records
            if guard_id in record["guards"]
        ]
        observations.extend(
            record["guards"][guard_id]
            for record in cell_records
            if guard_id in record["guards"]
        )
        decidable = bool(observations) and all(item["decidable"] for item in observations)
        passed = decidable and all(item["pass"] for item in observations)
        summaries.append(
            {
                "guard_id": guard_id,
                "status": "PASS" if passed else "FATAL",
                "decidable": decidable,
                "observation_count": len(observations),
                "failure_count": sum(not item["pass"] for item in observations),
                "missing_count": sum(not item["decidable"] for item in observations),
            }
        )
    return {
        "status": (
            "PASS" if summaries and all(item["status"] == "PASS" for item in summaries)
            else "FATAL"
        ),
        "guards": summaries,
    }


def score_parameters(
    expectations: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    all_cells_complete: bool,
    measurement_valid: bool,
    scoring_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    candidate = expectations["candidate_reference_not_observation"]
    specs = _parameter_specs(candidate)
    if not all_cells_complete:
        return [_parameter_result(spec, "PENDING", None, "campaign incomplete") for spec in specs]
    if not measurement_valid:
        return [
            _parameter_result(spec, "VOID_FATAL_GUARD", None, "one or more frozen fatal guards fired")
            if spec["parameter"] != "mode"
            else _parameter_result(
                spec,
                "STRUCTURAL",
                "pass_through",
                "direct NV4 mesh invariant, not hardware measurement",
            )
            for spec in specs
        ]
    evidence_rows = [row for row in rows if _ordinal(row) > 0]
    outcomes: dict[tuple[str, str], tuple[str, object | None, str]] = {}
    outcomes.update(_score_packet_fit(evidence_rows, scoring_audit))
    outcomes.update(_score_links_and_rates(evidence_rows, scoring_audit))
    outcomes.update(_score_direction(evidence_rows))
    outcomes.update(_score_credit_and_buffer(evidence_rows, outcomes, scoring_audit))
    outcomes.update(_score_delivery(rows))
    outcomes.update(_score_queue_scope(evidence_rows))
    outcomes[("switch", "mode")] = (
        "STRUCTURAL",
        "pass_through",
        "direct NV4 mesh invariant, not hardware measurement",
    )
    results = []
    for spec in specs:
        status, value, reason = outcomes.get(
            (spec["module"], spec["parameter"]),
            ("INCONCLUSIVE", None, "frozen rule did not obtain a unique fit"),
        )
        results.append(_parameter_result(spec, status, value, reason))
    return results


def _parameter_specs(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    tx = candidate["tx"]
    rx = candidate["rx"]
    return [
        _spec("tx", "max_payload_bytes", candidate["payload_bytes"], "TX_PACKET_PAYLOAD_AND_HEADER"),
        _spec("tx", "header_bytes", candidate["header_bytes"], "TX_PACKET_PAYLOAD_AND_HEADER"),
        _spec("tx", "links_per_peer", tx["links_per_peer"], "TX_LINK_COUNT_RATE_AND_BOND"),
        _spec("tx", "per_link_rate_bytes_per_second", tx["per_link_rate_bytes_per_second"], "TX_LINK_COUNT_RATE_AND_BOND"),
        _spec("tx", "bond_policy", tx["bond_policy"], "TX_LINK_COUNT_RATE_AND_BOND"),
        _spec("tx", "endpoint_egress_rate_bytes_per_second", tx["endpoint_egress_rate_bytes_per_second"], "TX_ENDPOINT_EGRESS_RATE"),
        _spec("tx", "request_response_direction", tx["request_response_direction"], "TX_REQUEST_RESPONSE_DIRECTION"),
        _spec("tx", "credit_unit_bytes", tx["credit_unit_bytes"], "TX_EFFECTIVE_CREDITS"),
        _spec("tx", "credits_per_destination", tx["credits_per_destination"], "TX_EFFECTIVE_CREDITS"),
        _spec("switch", "mode", "pass_through", "SWITCH_DIRECT_MESH_IDENTITY"),
        _spec("rx", "ingress_rate_bytes_per_second", rx["ingress_rate_bytes_per_second"], "RX_INGRESS_RATE"),
        _spec("rx", "buffer_capacity_bytes", rx["buffer_capacity_bytes"], "RX_EFFECTIVE_BUFFER"),
        _spec("rx", "credit_return_latency_ps", rx["credit_return_latency_ps"], "RX_CREDIT_RETURN_LATENCY"),
        _spec("rx", "reassembly_policy", rx["reassembly_policy"], "RX_DELIVERY"),
        _spec("rx", "delivery_order", rx["delivery_order"], "RX_DELIVERY"),
        _spec("tx_rx", "queue_scope", None, "TX_RX_QUEUE_SCOPE"),
    ]


def _spec(module: str, parameter: str, candidate: object, rule: str) -> dict[str, Any]:
    return {"module": module, "parameter": parameter, "candidate_value": candidate, "rule_id": rule}


def _parameter_result(
    spec: dict[str, Any], status: str, value: object | None, reason: str
) -> dict[str, Any]:
    if status == "IDENTIFIED":
        relation = _identified_relation(spec, value)
        evidence = _evidence_class(str(spec["rule_id"]))
    elif status == "STRUCTURAL":
        relation = "RETAINED_STRUCTURAL"
        evidence = "structural_direct_mesh_invariant_not_measurement"
    else:
        relation = "UNCHANGED"
        evidence = "declared_candidate_not_hardware_measurement"
    return {
        **spec,
        "status": status,
        "identified_value": value,
        "candidate_relation": relation,
        "evidence_class": evidence,
        "reason": reason,
    }


def _identified_relation(spec: dict[str, Any], value: object | None) -> str:
    candidate = spec["candidate_value"]
    if spec["parameter"] in {
        "per_link_rate_bytes_per_second",
        "endpoint_egress_rate_bytes_per_second",
        "ingress_rate_bytes_per_second",
    } and isinstance(candidate, (int, float)) and isinstance(value, (int, float)):
        relative_error = abs(float(value) - float(candidate)) / float(candidate)
        return "CONFIRMED" if relative_error <= 0.10 else "REFUTED_AND_REPLACED"
    return "CONFIRMED" if value == candidate else "REFUTED_AND_REPLACED"


def _evidence_class(rule_id: str) -> str:
    return {
        "TX_PACKET_PAYLOAD_AND_HEADER": "measured_effective_nvml_counter_fit",
        "TX_LINK_COUNT_RATE_AND_BOND": "measured_effective_link_counter_plateau",
        "TX_ENDPOINT_EGRESS_RATE": "measured_effective_endpoint_counter_plateau",
        "TX_REQUEST_RESPONSE_DIRECTION": "measured_directional_counter_conformance",
        "TX_EFFECTIVE_CREDITS": "measured_effective_credit_knee_fit",
        "RX_INGRESS_RATE": "measured_effective_ingress_counter_plateau",
        "RX_EFFECTIVE_BUFFER": "measured_effective_buffer_knee_fit",
        "RX_CREDIT_RETURN_LATENCY": "measured_effective_recovery_gap_fit",
        "RX_DELIVERY": "measured_behavioral_delivery_conformance",
        "TX_RX_QUEUE_SCOPE": "measured_effective_queue_scope",
    }.get(rule_id, "declared_candidate_not_hardware_measurement")


def _score_packet_fit(
    rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[tuple[str, str], tuple[str, object | None, str]]:
    producers = ("persistent_sm_peer_write", "dependent_sm_peer_read")
    training = {
        producer: [
            row
            for row in rows
            if 1 <= _ordinal(row) <= 15 and row.get("producer") == producer
        ]
        for producer in producers
    }
    holdout = {
        producer: [
            row
            for row in rows
            if _ordinal(row) == 16 and row.get("producer") == producer
        ]
        for producer in producers
    }
    copy_rows = [
        row
        for row in rows
        if 1 <= _ordinal(row) <= 16 and row.get("producer") == "copy_engine_reference"
    ]
    grid = tuple(
        (payload_limit, header)
        for payload_limit in range(16, 4097, 16)
        for header in range(129)
    )
    if any(not training[producer] or not holdout[producer] for producer in producers):
        reason = "packet training or blind holdout rows are absent"
        audit["packet_fit"] = {
            "status": "INCONCLUSIVE",
            "grid_cardinality": len(grid),
            "max_payload_bytes": {"minimum": 16, "maximum": 4096, "step": 16},
            "header_bytes": {"minimum": 0, "maximum": 128, "step": 1},
            "reason": reason,
        }
        return {
            ("tx", "max_payload_bytes"): ("INCONCLUSIVE", None, reason),
            ("tx", "header_bytes"): ("INCONCLUSIVE", None, reason),
        }
    rankings: dict[str, list[tuple[int, int, int]]] = {}
    for producer in producers:
        ranking = [
            (
                sum(
                    _packet_residual(row, payload_limit, header)
                    for row in training[producer]
                ),
                payload_limit,
                header,
            )
            for payload_limit, header in grid
        ]
        ranking.sort()
        rankings[producer] = ranking
    best_by_producer = {producer: rankings[producer][0] for producer in producers}
    unique_by_producer = {
        producer: rankings[producer][1][0] > rankings[producer][0][0]
        for producer in producers
    }
    fitted_pairs = {
        (best_by_producer[producer][1], best_by_producer[producer][2])
        for producer in producers
    }
    payload_limit, header = next(iter(fitted_pairs)) if len(fitted_pairs) == 1 else (0, 0)
    training_pass = len(fitted_pairs) == 1 and all(
        _packet_residual(row, payload_limit, header) <= _row_quantum_allowance(row)
        for producer in producers
        for row in training[producer]
    )
    holdout_pass = len(fitted_pairs) == 1 and all(
        _packet_residual(row, payload_limit, header) <= _row_quantum_allowance(row)
        for producer in producers
        for row in holdout[producer]
    )
    copy_pass = bool(copy_rows) and len(fitted_pairs) == 1 and all(
        _packet_residual(row, payload_limit, header) <= _row_quantum_allowance(row)
        for row in copy_rows
    )
    audit["packet_fit"] = {
        "status": (
            "IDENTIFIED"
            if all(unique_by_producer.values())
            and len(fitted_pairs) == 1
            and training_pass
            and holdout_pass
            and copy_pass
            else "INCONCLUSIVE"
        ),
        "grid_cardinality": len(grid),
        "max_payload_bytes": {"minimum": 16, "maximum": 4096, "step": 16},
        "header_bytes": {"minimum": 0, "maximum": 128, "step": 1},
        "producer_fits": {
            producer: {
                "best_residual_bytes": best_by_producer[producer][0],
                "best_max_payload_bytes": best_by_producer[producer][1],
                "best_header_bytes": best_by_producer[producer][2],
                "runner_up_residual_bytes": rankings[producer][1][0],
                "unique": unique_by_producer[producer],
                "training_row_count": len(training[producer]),
                "holdout_row_count": len(holdout[producer]),
            }
            for producer in producers
        },
        "same_producer_fit": len(fitted_pairs) == 1,
        "per_training_row_within_quantum": training_pass,
        "blind_holdout_within_quantum": holdout_pass,
        "copy_engine_agrees_without_authority": copy_pass,
        "copy_engine_row_count": len(copy_rows),
    }
    if not (
        all(unique_by_producer.values())
        and len(fitted_pairs) == 1
        and training_pass
        and holdout_pass
        and copy_pass
    ):
        reason = (
            "producer fits are not unique and common, or a training, copy-engine, "
            "or blind-holdout residual exceeds its counter-quantum allowance"
        )
        return {
            ("tx", "max_payload_bytes"): ("INCONCLUSIVE", None, reason),
            ("tx", "header_bytes"): ("INCONCLUSIVE", None, reason),
        }
    aggregate_residual = sum(best_by_producer[producer][0] for producer in producers)
    reason = f"unique common candidate-blind fit with residual {aggregate_residual} bytes"
    return {
        ("tx", "max_payload_bytes"): ("IDENTIFIED", payload_limit, reason),
        ("tx", "header_bytes"): ("IDENTIFIED", header, reason),
    }


def _packet_residual(row: dict[str, Any], payload_limit: int, header: int) -> int:
    observed = max(0, int(row["observed_raw_bytes"]) - int(row["observed_data_bytes"]))
    flow_count = max(1, len(row.get("flow_rate_ledger", [])))
    expected = (
        math.ceil(int(row["payload_bytes"]) / payload_limit)
        * header
        * int(row["message_count"])
        * flow_count
    )
    return abs(observed - expected)


def _row_quantum_allowance(row: dict[str, Any]) -> int:
    return max(1024, len(_links(row)) * 1024)


def _score_links_and_rates(
    rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[tuple[str, str], tuple[str, object | None, str]]:
    pair_samples: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not 17 <= _ordinal(row) <= 32:
            continue
        if int(row["applied_controls"]["offered_rate_percent"]) < 95:
            continue
        for source, destination in _flow_pairs(row):
            rates = _pair_link_rates(row, source, destination, "tx")
            if rates and sum(rate for _, rate in rates) > 0:
                pair_samples[(source, destination)].append(
                    {"row": row, "rates": rates, "total": sum(rate for _, rate in rates)}
                )
    expected_pairs = {(source, destination) for source in range(4) for destination in range(4) if source != destination}
    plateau_samples: list[dict[str, Any]] = []
    pair_counts: dict[str, list[int]] = {}
    balances: list[float] = []
    for pair in sorted(expected_pairs):
        samples = pair_samples[pair]
        if not samples:
            continue
        maximum = max(float(sample["total"]) for sample in samples)
        plateau = [sample for sample in samples if float(sample["total"]) >= 0.90 * maximum]
        counts = []
        for sample in plateau:
            rates = [float(rate) for _, rate in sample["rates"]]
            threshold = 0.05 * max(rates, default=0.0)
            active = [rate for rate in rates if rate > threshold]
            counts.append(len(active))
            if active:
                balances.append(min(active) / max(active))
        pair_counts[f"{pair[0]}->{pair[1]}"] = counts
        plateau_samples.extend(plateau)
    complete_pairs = set(pair_samples) == expected_pairs
    stable_count_values = {
        counts[0]
        for counts in pair_counts.values()
        if counts and len(set(counts)) == 1
    }
    stable_counts = (
        complete_pairs
        and len(pair_counts) == len(expected_pairs)
        and all(counts and len(set(counts)) == 1 for counts in pair_counts.values())
        and len(stable_count_values) == 1
    )
    balance_pass = bool(balances) and min(balances) >= 0.90
    per_physical_link: defaultdict[tuple[int, int, int], list[float]] = defaultdict(list)
    for sample in plateau_samples:
        row = sample["row"]
        for link, rate in sample["rates"]:
            per_physical_link[
                (int(link["gpu"]), int(link["link"]), int(link["remote_gpu"]))
            ].append(float(rate))
    link_top_three = {
        key: sorted(rates, reverse=True)[:3]
        for key, rates in per_physical_link.items()
        if len(rates) >= 3
    }
    repeatable_links = bool(link_top_three) and len(link_top_three) == len(
        per_physical_link
    ) and all(
        values[-1] > 0 and values[0] / values[-1] <= 1.10
        for values in link_top_three.values()
    )
    link_value = (
        round(statistics.median(statistics.median(values) for values in link_top_three.values()))
        if link_top_three
        else None
    )
    link_identified = stable_counts and balance_pass and repeatable_links and link_value is not None

    endpoint_samples = []
    for row in rows:
        if _ordinal(row) not in {24, 26, 27, 28, 29, 30}:
            continue
        if int(row["applied_controls"]["offered_rate_percent"]) < 95:
            continue
        flows = _flow_pairs(row)
        for source in sorted({source for source, _ in flows}):
            destinations = {destination for candidate, destination in flows if candidate == source}
            rate = sum(
                float(link["raw_tx_kib_delta"]) * 1024 / (float(row["elapsed_us"]) * 1.0e-6)
                for link in _links(row)
                if int(link["gpu"]) == source and int(link["remote_gpu"]) in destinations
            )
            if rate > 0:
                endpoint_samples.append(rate)
    endpoint_top_three = sorted(endpoint_samples, reverse=True)[:3]
    endpoint_repeatable = (
        len(endpoint_top_three) == 3
        and endpoint_top_three[-1] > 0
        and endpoint_top_three[0] / endpoint_top_three[-1] <= 1.10
    )
    endpoint_value = (
        round(statistics.median(endpoint_top_three)) if endpoint_repeatable else None
    )
    audit["link_and_endpoint_plateaus"] = {
        "ordered_pair_count": len(pair_samples),
        "required_ordered_pair_count": len(expected_pairs),
        "plateau_sample_count": len(plateau_samples),
        "active_link_counts": pair_counts,
        "minimum_plateau_balance": min(balances) if balances else None,
        "all_link_top_three_repeatable_within_10_percent": repeatable_links,
        "per_link_identified_value": link_value if link_identified else None,
        "endpoint_top_three": endpoint_top_three,
        "endpoint_repeatable_within_10_percent": endpoint_repeatable,
    }
    count = next(iter(stable_count_values)) if link_identified else None
    common_reason = (
        "all ordered pairs have one stable balanced active-link count and repeatable top-three plateaus"
        if link_identified
        else "ordered-pair coverage, stable count, 0.90 balance, or top-three repeatability failed"
    )
    endpoint_reason = (
        "median of the three highest guarded endpoint plateaus repeats within 10 percent"
        if endpoint_repeatable
        else "three repeatable guarded endpoint plateaus were not obtained"
    )
    return {
        ("tx", "links_per_peer"): (
            "IDENTIFIED" if link_identified else "INCONCLUSIVE",
            count,
            common_reason,
        ),
        ("tx", "per_link_rate_bytes_per_second"): (
            "IDENTIFIED" if link_identified else "INCONCLUSIVE",
            link_value if link_identified else None,
            common_reason,
        ),
        ("tx", "bond_policy"): (
            "INCONCLUSIVE",
            None,
            "balanced aggregate counters cannot identify earliest-available scheduling",
        ),
        ("tx", "endpoint_egress_rate_bytes_per_second"): (
            "IDENTIFIED" if endpoint_repeatable else "INCONCLUSIVE",
            endpoint_value,
            endpoint_reason,
        ),
    }


def _flow_pairs(row: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    pairs = []
    for flow in row.get("flow_rate_ledger", []):
        if not isinstance(flow, dict):
            continue
        pair = (int(flow["source"]), int(flow["destination"]))
        if pair not in pairs:
            pairs.append(pair)
    return tuple(pairs)


def _pair_link_rates(
    row: dict[str, Any], source: int, destination: int, direction: str
) -> list[tuple[dict[str, Any], float]]:
    elapsed_seconds = float(row["elapsed_us"]) * 1.0e-6
    if elapsed_seconds <= 0:
        return []
    return [
        (
            link,
            float(link[f"raw_{direction}_kib_delta"]) * 1024 / elapsed_seconds,
        )
        for link in _links(row)
        if int(link["gpu"]) == source and int(link["remote_gpu"]) == destination
    ]


def _score_direction(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], tuple[str, object | None, str]]:
    ordinals = {13, 14, 15, 49, 55, 64, 73, 74, 75, 76}
    writes = [
        row
        for row in rows
        if _ordinal(row) in ordinals and row.get("producer") == "persistent_sm_peer_write"
    ]
    reads = [
        row
        for row in rows
        if _ordinal(row) in ordinals and row.get("producer") == "dependent_sm_peer_read"
    ]
    write_ledgers = []
    for row in writes:
        for source, destination in _flow_pairs(row):
            write_ledgers.append(
                _counter_kib(row, source, destination, "data_tx") > 0
                and _counter_kib(row, destination, source, "data_rx") > 0
            )
    read_ledgers = []
    for row in reads:
        for issuer, target in _flow_pairs(row):
            issuer_request_raw = _counter_kib(row, issuer, target, "raw_tx")
            issuer_request_data = _counter_kib(row, issuer, target, "data_tx")
            target_request_raw = _counter_kib(row, target, issuer, "raw_rx")
            target_request_data = _counter_kib(row, target, issuer, "data_rx")
            read_ledgers.append(
                issuer_request_raw > issuer_request_data
                and target_request_raw > target_request_data
                and _counter_kib(row, target, issuer, "data_tx") > 0
                and _counter_kib(row, issuer, target, "data_rx") > 0
            )
    if write_ledgers and read_ledgers and all(write_ledgers) and all(read_ledgers):
        value = (
            "write payload travels as request; read control travels as request and read "
            "payload travels as response"
        )
        return {
            ("tx", "request_response_direction"): (
                "IDENTIFIED", value, "guarded write and read directional counter ledgers"
            )
        }
    return {
        ("tx", "request_response_direction"): (
            "INCONCLUSIVE", None, "request or response direction is below counter resolution"
        )
    }


def _counter_kib(
    row: dict[str, Any], gpu: int, remote_gpu: int, counter: str
) -> int:
    return sum(
        int(link[f"{counter}_kib_delta"])
        for link in _links(row)
        if int(link["gpu"]) == gpu and int(link["remote_gpu"]) == remote_gpu
    )


def _score_credit_and_buffer(
    rows: list[dict[str, Any]],
    prior: dict[tuple[str, str], tuple[str, object | None, str]],
    audit: dict[str, Any],
) -> dict[tuple[str, str], tuple[str, object | None, str]]:
    credit_rows = [row for row in rows if 50 <= _ordinal(row) <= 64]
    knees: list[int] = []
    knee_trace: dict[str, dict[str, object]] = {}
    outstanding_sweep = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    for ordinal in range(50, 55):
        by_window: defaultdict[int, list[float]] = defaultdict(list)
        for row in credit_rows:
            if _ordinal(row) == ordinal:
                by_window[int(row["applied_controls"]["outstanding"])].append(
                    float(row["payload_rate_gbps"])
                )
        rates = [
            (window, statistics.median(by_window[window]))
            for window in outstanding_sweep
            if by_window[window]
        ]
        if len(rates) != len(outstanding_sweep):
            continue
        plateau = max(rate for _, rate in rates)
        knee = next((window for window, rate in rates if rate >= 0.95 * plateau), None)
        if knee is not None:
            knees.append(knee)
        knee_trace[str(ordinal)] = {
            "median_rate_by_outstanding": [[window, rate] for window, rate in rates],
            "plateau_gbps": plateau,
            "first_95_percent_knee": knee,
        }
    knee_indices = [outstanding_sweep.index(knee) for knee in knees]
    reproducible_knee = (
        len(knees) == 5 and max(knee_indices) - min(knee_indices) <= 1
    )

    recovery_thresholds = []
    recovery_trace: dict[str, object] = {}
    gap_sweep = (0, 20, 50, 100, 200, 500, 1000, 5000)
    for ordinal in (58, 59):
        by_gap: defaultdict[int, list[float]] = defaultdict(list)
        for row in credit_rows:
            if _ordinal(row) == ordinal:
                by_gap[int(row["applied_controls"]["gap_ns"])].append(
                    float(row["payload_rate_gbps"])
                )
        rates = {
            gap: statistics.median(by_gap[gap]) for gap in gap_sweep if by_gap[gap]
        }
        plateau = max(rates.values(), default=0.0)
        threshold = None
        if len(rates) == len(gap_sweep) and rates[0] < 0.95 * plateau:
            threshold = next(
                (gap for gap in gap_sweep[1:] if rates[gap] >= 0.95 * plateau),
                None,
            )
        if threshold is not None:
            recovery_thresholds.append(threshold)
        recovery_trace[str(ordinal)] = {
            "median_rate_by_gap_ns": [[gap, rates.get(gap)] for gap in gap_sweep],
            "plateau_gbps": plateau,
            "first_restored_gap_ns": threshold,
        }
    recovery_indices = [gap_sweep.index(value) for value in recovery_thresholds]
    recovery_repeatable = (
        len(recovery_thresholds) == 2
        and max(recovery_indices) - min(recovery_indices) <= 1
    )
    recovery_value = (
        round(statistics.median(recovery_thresholds)) * 1000
        if recovery_repeatable
        else None
    )
    recovery = (
        (
            "IDENTIFIED",
            recovery_value,
            "depleted baseline and repeated first 95-percent restored-gap threshold",
        )
        if recovery_repeatable
        else (
            "INCONCLUSIVE",
            None,
            "no repeated recovery threshold rises from a depleted zero-gap baseline",
        )
    )

    destination_scope_rates: dict[int, list[float]] = defaultdict(list)
    for row in credit_rows:
        if _ordinal(row) == 63:
            destination_count = len({destination for _, destination in _flow_pairs(row)})
            destination_scope_rates[destination_count].append(float(row["payload_rate_gbps"]))
    destination_scope_ratio = None
    if destination_scope_rates[1] and destination_scope_rates[2]:
        one = statistics.median(destination_scope_rates[1])
        two = statistics.median(destination_scope_rates[2])
        if one > 0:
            destination_scope_ratio = two / one
    destination_scoped = (
        destination_scope_ratio is not None and 1.80 <= destination_scope_ratio <= 2.20
    )
    packet_value = prior.get(("tx", "max_payload_bytes"), ("", None, ""))[1]
    header_value = prior.get(("tx", "header_bytes"), ("", None, ""))[1]
    credits_identified = (
        reproducible_knee
        and recovery_repeatable
        and destination_scoped
        and isinstance(packet_value, int)
        and isinstance(header_value, int)
    )
    if credits_identified:
        unit = packet_value + header_value
        count = round(statistics.median(knees))
        credit_unit = (
            "IDENTIFIED",
            unit,
            "cross-payload knee, recovery and destination-scope checks all pass",
        )
        credit_count = (
            "IDENTIFIED",
            count,
            "cross-payload knee, recovery and destination-scope checks all pass",
        )
    else:
        reason = (
            "credit knee lacks a required packet fit, recovery return, or destination-scope cross-check"
        )
        credit_unit = ("INCONCLUSIVE", None, reason)
        credit_count = ("INCONCLUSIVE", None, reason)
    buffer = (
        "INCONCLUSIVE",
        None,
        "no common localized incast and outstanding loss-free capacity knee predicts drain",
    )

    ingress_rows = [row for row in rows if 25 == _ordinal(row) or 33 <= _ordinal(row) <= 48]
    ingress_rates = [
        _destination_raw_rate(row)
        for row in ingress_rows
        if _destination_raw_rate(row) > 0
    ]
    ingress_top_three = sorted(ingress_rates, reverse=True)[:3]
    ingress_repeatable = (
        len(ingress_top_three) == 3
        and ingress_top_three[-1] > 0
        and ingress_top_three[0] / ingress_top_three[-1] <= 1.10
        and ingress_top_three[0] <= 300.0e9
    )
    ingress = (
        (
            "IDENTIFIED",
            round(statistics.median(ingress_top_three)),
            "median of three highest guarded ingress plateaus repeats within 10 percent",
        )
        if ingress_repeatable
        else (
            "INCONCLUSIVE",
            None,
            "three repeatable destination-ingress plateaus at or below 300 GB/s were not obtained",
        )
    )
    audit["credits_buffer_recovery_and_ingress"] = {
        "credit_knees": knee_trace,
        "cross_payload_knee_repeatable_within_one_point": reproducible_knee,
        "recovery": recovery_trace,
        "recovery_repeatable_within_one_point": recovery_repeatable,
        "destination_scope_rate_ratio": destination_scope_ratio,
        "destination_scope_pass": destination_scoped,
        "buffer_capacity_identified": False,
        "buffer_reason": buffer[2],
        "ingress_top_three": ingress_top_three,
        "ingress_repeatable_within_10_percent_and_below_ceiling": ingress_repeatable,
    }
    return {
        ("tx", "credit_unit_bytes"): credit_unit,
        ("tx", "credits_per_destination"): credit_count,
        ("rx", "buffer_capacity_bytes"): buffer,
        ("rx", "credit_return_latency_ps"): recovery,
        ("rx", "ingress_rate_bytes_per_second"): ingress,
    }


def _destination_raw_rate(row: dict[str, Any]) -> float:
    elapsed = float(row["elapsed_us"]) * 1.0e-6
    if elapsed <= 0:
        return 0.0
    sources_by_destination: defaultdict[int, set[int]] = defaultdict(set)
    for source, destination in _flow_pairs(row):
        sources_by_destination[destination].add(source)
    raw_rx_bytes = sum(
        int(link["raw_rx_kib_delta"]) * 1024
        for link in _links(row)
        if int(link["gpu"]) in sources_by_destination
        and int(link["remote_gpu"]) in sources_by_destination[int(link["gpu"])]
    )
    return raw_rx_bytes / elapsed


def _score_delivery(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], tuple[str, object | None, str]]:
    applicable = [
        row for row in rows
        if _ordinal(row) in {*range(1, 17), *range(33, 49), *range(65, 81)}
    ]
    passed = bool(applicable) and all(row.get("checksum_ok") is True for row in applicable)
    if passed:
        reason = "all applicable isolated and ordered-frame destination ledgers match"
        return {
            ("rx", "reassembly_policy"): ("IDENTIFIED", "extent_sequence", reason),
            ("rx", "delivery_order"): ("IDENTIFIED", "per_extent", reason),
        }
    reason = "delivery ledger is absent or mismatched"
    return {
        ("rx", "reassembly_policy"): ("INCONCLUSIVE", None, reason),
        ("rx", "delivery_order"): ("INCONCLUSIVE", None, reason),
    }


def _score_queue_scope(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], tuple[str, object | None, str]]:
    hol = [row for row in rows if 65 <= _ordinal(row) <= 80]
    patterns = {str(row.get("pattern")) for row in hol}
    required = {"same_pair_bulk", "other_peer_bulk", "remote_incast", "distinct_regions"}
    if required <= patterns:
        return {
            ("tx_rx", "queue_scope"): (
                "INCONCLUSIVE",
                None,
                "scope controls are observable but no unique stable localization is frozen from aggregate completion",
            )
        }
    return {
        ("tx_rx", "queue_scope"): (
            "INCONCLUSIVE", None, "required queue-scope control patterns are absent"
        )
    }


def _ordinal(row: dict[str, Any]) -> int:
    match = re.search(r"_(\d{3})_", str(row.get("case_name", "")))
    return int(match.group(1)) if match else 0


def profile_patch_from_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    changes = []
    for result in results:
        if result["status"] != "IDENTIFIED":
            continue
        if result["candidate_relation"] not in {"CONFIRMED", "REFUTED_AND_REPLACED"}:
            continue
        if result["module"] not in {"tx", "rx"}:
            continue
        changes.append(
            {
                "module": result["module"],
                "parameter": result["parameter"],
                "value": result["identified_value"],
                "candidate_relation": result["candidate_relation"],
                "evidence_class": result["evidence_class"],
                "rule_id": result["rule_id"],
            }
        )
    return {
        "status": "APPLY_EXACTLY_LISTED_CHANGES_AFTER_SCORE_PUBLICATION",
        "changes": changes,
        "unchanged_parameter_count": len(results) - len(changes),
    }


def score_flow_dynamics_gate(
    *,
    all_cells_complete: bool,
    guards_decidable: bool,
    parameter_results: list[dict[str, Any]],
) -> dict[str, Any]:
    required_rules = {
        "TX_LINK_COUNT_RATE_AND_BOND",
        "TX_EFFECTIVE_CREDITS",
        "RX_INGRESS_RATE",
        "RX_EFFECTIVE_BUFFER",
        "TX_RX_QUEUE_SCOPE",
    }
    rule_statuses: defaultdict[str, set[str]] = defaultdict(set)
    for result in parameter_results:
        rule_statuses[str(result["rule_id"])].add(str(result["status"]))
    nonvoid = all(
        rule_statuses[rule]
        and not rule_statuses[rule] <= {"PENDING", "VOID_FATAL_GUARD"}
        for rule in required_rules
    )
    opens = all_cells_complete and guards_decidable and nonvoid
    return {
        "verdict": "OPEN" if opens else "CLOSED",
        "all_cells_complete": all_cells_complete,
        "all_fatal_guards_decidable": guards_decidable,
        "required_rule_statuses": {
            rule: sorted(rule_statuses[rule]) for rule in sorted(required_rules)
        },
        "reason": (
            "all frozen prerequisites have a decidable non-void outcome"
            if opens
            else "one or more frozen completion, guard, or rule prerequisites failed"
        ),
    }


def score_cases(
    expectations: dict[str, Any],
    rows: list[dict[str, Any]],
    attempts: dict[int, Attempt],
    cells: tuple[CellSpec, ...],
) -> list[dict[str, Any]]:
    cells_by_case: defaultdict[str, list[CellSpec]] = defaultdict(list)
    for cell in cells:
        for case_name in cell.case_names:
            cells_by_case[case_name].append(cell)
    results = []
    for case in expectations["catalog"]:
        name = str(case["stable_name"])
        required_cells = cells_by_case[name]
        case_rows = [row for row in rows if row.get("case_name") == name]
        completed = [cell for cell in required_cells if cell.index in attempts]
        if not completed:
            status = "PENDING"
        elif len(completed) < len(required_cells):
            status = "PARTIAL"
        else:
            status = "COMPLETE"
        results.append(
            {
                "ordinal": case["ordinal"],
                "case_name": name,
                "corner": case["corner"],
                "coverage": status,
                "row_count": len(case_rows),
                "expected_band": case["expected_band"],
                "identification_rule_ids": case["identification_rule_ids"],
            }
        )
    return results


def render_markdown(score: dict[str, Any]) -> str:
    coverage = score["coverage"]
    gate = score["flow_dynamics_gate"]
    lines = [
        "# TRAF-70 corrected A100 NVLink packet score",
        "",
        "## Per-module identification",
        "",
        "| Module | Parameter | Status | Identified value | Candidate relation | Evidence class |",
        "|---|---|---|---|---|---|",
    ]
    for row in score["module_parameter_identification"]:
        value = "none" if row["identified_value"] is None else str(row["identified_value"])
        lines.append(
            f"| {row['module']} | `{row['parameter']}` | {row['status']} | {value} | "
            f"{row['candidate_relation']} | `{row['evidence_class']}` |"
        )
    lines.extend(
        [
            "",
            "## Fatal-guard verdicts",
            "",
            "| Guard | Verdict | Decidable | Observations | Failures | Missing |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for guard in score["fatal_guard_verdicts"]["guards"]:
        lines.append(
            f"| `{guard['guard_id']}` | {guard['status']} | "
            f"{'yes' if guard['decidable'] else 'no'} | {guard['observation_count']} | "
            f"{guard['failure_count']} | {guard['missing_count']} |"
        )
    lines.extend(
        [
            "",
            "## Flow-dynamics gate",
            "",
            f"Gate verdict: `{gate['verdict']}`. {gate['reason']}",
            "",
            "## Execution coverage",
            "",
            f"- Status: `{score['status']}`.",
            f"- Measurement validity: `{score['measurement_validity']}`.",
            f"- Scheduler job: `{score['scheduler_job']}`.",
            f"- Expectations SHA-256: `{score['freeze_sha256']}`.",
            f"- Digest-complete cells: {coverage['completed_cell_count']} of 86.",
            f"- Hardware rows: {coverage['result_row_count']:,}.",
            f"- Exact pending array: `{coverage['pending_array'] or 'none'}`.",
            "",
            "## Score-authorized profile changes",
            "",
        ]
    )
    changes = score["profile_patch"]["changes"]
    if changes:
        lines.extend(
            [
                "| Module | Parameter | Value | Candidate relation | Evidence class | Rule |",
                "|---|---|---|---|---|---|",
            ]
        )
        for change in changes:
            lines.append(
                f"| {change['module']} | `{change['parameter']}` | {change['value']} | "
                f"{change['candidate_relation']} | `{change['evidence_class']}` | "
                f"`{change['rule_id']}` |"
            )
    else:
        lines.append("No parameter value or evidence-class change is authorized.")
    lines.extend(
        [
            "",
            "The scorer did not amend the frozen expectations. Candidate-derived packet",
            "fields were not consumed as observations.",
            "",
        ]
    )
    return "\n".join(lines)


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
