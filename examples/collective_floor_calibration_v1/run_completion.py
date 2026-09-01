"""Run and check the TRAF-76 aggregate-completion publication.

The normal invocation writes two fresh-process evaluations to a new
append-only work directory, then writes the deterministic publication beside
this runner. The check invocation performs no study work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from examples.collective_floor_calibration_v1.bypass_fixture import (
    PRE_WAVE_COMMIT,
    produce_bypass_record,
)
from simllm.calibration.external_nccl import ExternalNcclDatabase
from simllm.traffic import (
    COLLECTIVE_COMPLETION_GEOMETRIC_TRANSITION,
    COLLECTIVE_COMPLETION_SYMMETRIC_TRANSITION,
    CollectiveFloorCell,
    CollectiveFloorCurveBoundaries,
    CollectiveFloorSourceIdentity,
    build_collective_completion_calibration,
    fit_collective_floor_calibration,
)

BASE_CONFIG_PATH = STUDY_DIR / "study_config.json"
SECOND_CONFIG_PATH = STUDY_DIR / "study_config_v3.json"
COMPLETION_CONFIG_PATH = STUDY_DIR / "study_config_v4.json"
QUALIFYING_EXPECTATIONS_PATH = STUDY_DIR / "expectations_v6.md"
PRIOR_RECORD_PATH = STUDY_DIR / "record.json"
ATTEMPT_0005_PATH = STUDY_DIR / "ATTEMPT_0005.md"
PRE_WAVE_GOLDEN_PATH = STUDY_DIR / "pre_wave_bypass_golden.json"
MINIMAX_RECORD_PATH = REPOSITORY_ROOT / "examples/minimax_ep_scaling_v1/record.json"
TRACKED_RECORD_PATH = STUDY_DIR / "completion_record.json"
TRACKED_CSV_PATH = STUDY_DIR / "completion_results.csv"
SCHEMA = "simllm-collective-floor-completion-record-v1"
LEGACY_CALIBRATION_ID = "h200-nccl-2.26.2-aggregate-floor-v1"
ATTEMPT_0005_CALIBRATION_ID = "h200-nccl-2.26.2-aggregate-anchor-v2"
ARITHMETIC_CALIBRATION_ID = "h200-nccl-2.26.2-aggregate-anchor-v3"
FINAL_CALIBRATION_ID = "h200-nccl-2.26.2-aggregate-anchor-v4"
LOCAL_BANDWIDTH_BYTES_PER_SECOND = 450_000_000_000
REPRESENTED_LAYERS = 65


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _observed_cell(
    database: ExternalNcclDatabase,
    member: dict[str, Any],
) -> CollectiveFloorCell:
    latency = database.query(
        dtype=member["dtype"],
        operation=member["operation"],
        ranks=member["ranks"],
        message_size=member["source_elements"],
    )
    return CollectiveFloorCell(
        cell_id=member["cell_id"],
        dtype=member["dtype"],
        operation=member["operation"],
        ranks=member["ranks"],
        source_elements=member["source_elements"],
        message_bytes=member["true_bytes"],
        latency_ps=round(latency.latency_ms * 1_000_000_000),
    )


def _authorities():
    base = _load_json(BASE_CONFIG_PATH)
    completion = _load_json(COMPLETION_CONFIG_PATH)
    if len(base["membership"]["training_cells"]) != 63:
        raise RuntimeError("the frozen training membership is not 63 cells")
    if len(base["membership"]["holdout_cells"]) != 63:
        raise RuntimeError("the frozen holdout membership is not 63 cells")
    database = ExternalNcclDatabase.load()
    training = tuple(
        _observed_cell(database, member)
        for member in base["membership"]["training_cells"]
    )
    source = CollectiveFloorSourceIdentity(**base["source"])
    boundaries = tuple(
        CollectiveFloorCurveBoundaries(
            dtype=row["dtype"],
            operation=row["operation"],
            ranks=row["ranks"],
            lower_bounds_of_following_regimes=tuple(
                row["lower_bounds_of_following_regimes"]
            ),
        )
        for row in base["fit"]["regime_boundaries_true_bytes"]
    )
    byte_range = base["fit"]["true_byte_range"]
    fitted_range = (byte_range["minimum"], byte_range["maximum"])
    legacy = fit_collective_floor_calibration(
        calibration_id=LEGACY_CALIBRATION_ID,
        source=source,
        cells=training,
        boundaries=boundaries,
        fitted_byte_range=fitted_range,
    )
    attempt_0005 = build_collective_completion_calibration(
        calibration_id=ATTEMPT_0005_CALIBRATION_ID,
        source=source,
        cells=training,
        fitted_byte_range=fitted_range,
        compatibility_calibration=legacy,
    )
    arithmetic = build_collective_completion_calibration(
        calibration_id=ARITHMETIC_CALIBRATION_ID,
        source=source,
        cells=training,
        fitted_byte_range=fitted_range,
        compatibility_calibration=legacy,
        model_form=COLLECTIVE_COMPLETION_SYMMETRIC_TRANSITION,
    )
    final = build_collective_completion_calibration(
        calibration_id=FINAL_CALIBRATION_ID,
        source=source,
        cells=training,
        fitted_byte_range=fitted_range,
        compatibility_calibration=legacy,
        model_form=COLLECTIVE_COMPLETION_GEOMETRIC_TRANSITION,
    )
    authority_bytes = _canonical_json_bytes(final.as_dict())
    authority_sha256 = _sha256_bytes(authority_bytes)
    if completion["model"]["model_id"] != final.calibration_id:
        raise RuntimeError("the frozen model ID disagrees with the authority")
    return (
        base,
        completion,
        database,
        training,
        legacy,
        attempt_0005,
        arithmetic,
        final,
        authority_sha256,
    )


def _family_h(
    *,
    base: dict[str, Any],
    database: ExternalNcclDatabase,
    attempt_0005,
    arithmetic,
    final,
    authority_sha256: str,
) -> dict[str, Any]:
    prior = _load_json(PRIOR_RECORD_PATH)
    prior_rows = {
        row["cell_id"]: row for row in prior["families"]["H"]["rows"]
    }
    training_ids = {cell.cell_id for cell in final.training_cells}
    holdout_ids = {
        member["cell_id"] for member in base["membership"]["holdout_cells"]
    }
    if training_ids & holdout_ids:
        raise RuntimeError("training and holdout identities overlap")
    rows = []
    for member in base["membership"]["holdout_cells"]:
        measured = _observed_cell(database, member)
        old = prior_rows[member["cell_id"]]
        attempt = attempt_0005.estimate(
            dtype=member["dtype"],
            operation=member["operation"],
            ranks=member["ranks"],
            message_bytes=member["true_bytes"],
        )
        estimate = final.estimate(
            dtype=member["dtype"],
            operation=member["operation"],
            ranks=member["ranks"],
            message_bytes=member["true_bytes"],
        )
        arithmetic_estimate = arithmetic.estimate(
            dtype=member["dtype"],
            operation=member["operation"],
            ranks=member["ranks"],
            message_bytes=member["true_bytes"],
        )
        attempt_error = abs(attempt.completion_ps - measured.latency_ps) / (
            measured.latency_ps
        )
        final_error = abs(estimate.completion_ps - measured.latency_ps) / (
            measured.latency_ps
        )
        arithmetic_error = abs(
            arithmetic_estimate.completion_ps - measured.latency_ps
        ) / measured.latency_ps
        endpoint_bytes = member["true_bytes"] * (member["ranks"] - 1) / member["ranks"]
        physical_floor_ps = endpoint_bytes / LOCAL_BANDWIDTH_BYTES_PER_SECOND * 1e12
        rows.append(
            {
                "cell_id": member["cell_id"],
                "operation": member["operation"],
                "ranks": member["ranks"],
                "true_bytes": member["true_bytes"],
                "measured_ps": measured.latency_ps,
                "physical_floor_ps": physical_floor_ps,
                "attempt_0004_prediction_ps": old["calibrated_ps"],
                "attempt_0004_relative_error": old["after_relative_error"],
                "attempt_0005_prediction_ps": attempt.completion_ps,
                "attempt_0005_relative_error": attempt_error,
                "arithmetic_prediction_ps": arithmetic_estimate.completion_ps,
                "arithmetic_relative_error": arithmetic_error,
                "final_prediction_ps": estimate.completion_ps,
                "final_relative_error": final_error,
                "tolerance": 0.10,
                "passed": final_error <= 0.10,
                "rule": estimate.rule,
                "training_cell_ids": list(estimate.training_cell_ids),
                "above_physical_floor": (
                    measured.latency_ps >= physical_floor_ps
                    and estimate.completion_ps >= physical_floor_ps
                ),
            }
        )
    errors = [row["final_relative_error"] for row in rows]
    return {
        "status": "PASS" if all(row["passed"] for row in rows) else "REFUTED",
        "passed": sum(row["passed"] for row in rows),
        "denominator": len(rows),
        "authority_sha256_before_holdout_load": authority_sha256,
        "median_relative_error": statistics.median(errors),
        "p95_relative_error_nearest_rank": sorted(errors)[
            round(0.95 * (len(errors) - 1))
        ],
        "maximum_relative_error": max(errors),
        "physical_ceiling": "unbounded because the source identifies no algorithm-progress ceiling",
        "rows": rows,
    }


def _family_d8(final) -> dict[str, Any]:
    estimates = {
        operation: final.estimate(
            dtype="half",
            operation=operation,
            ranks=8,
            message_bytes=196_608,
        )
        for operation in ("reduce_scatter", "all_gather")
    }
    modeled_ms = sum(
        estimate.completion_ps for estimate in estimates.values()
    ) * REPRESENTED_LAYERS / 1_000_000_000
    external_ms = 1.922050
    quotient = modeled_ms / external_ms
    physical_floor_ps = 196_608 * 7 / 8 / LOCAL_BANDWIDTH_BYTES_PER_SECOND * 1e12
    return {
        "status": "PASS" if 0.90 <= quotient <= 1.10 else "REFUTED",
        "passed": int(0.90 <= quotient <= 1.10),
        "denominator": 1,
        "operation_buffer_bytes": 196_608,
        "represented_layers": REPRESENTED_LAYERS,
        "external_ms": external_ms,
        "modeled_ms": modeled_ms,
        "quotient": quotient,
        "band": [0.90, 1.10],
        "physical_floor_ps_per_phase": physical_floor_ps,
        "contributions": {
            operation: estimate.as_dict()
            for operation, estimate in estimates.items()
        },
    }


def _bypass_guard(workdir: Path) -> dict[str, Any]:
    golden = _load_json(PRE_WAVE_GOLDEN_PATH)
    observed = produce_bypass_record(
        workdir,
        backend_replay=golden["record"]["backend_invocations"],
    )
    expected_bytes = _canonical_json_bytes(golden["record"])
    observed_bytes = _canonical_json_bytes(observed)
    return {
        "held": expected_bytes == observed_bytes,
        "generating_commit": PRE_WAVE_COMMIT,
        "golden_record_sha256": _sha256_bytes(expected_bytes),
        "observed_record_sha256": _sha256_bytes(observed_bytes),
        "checked_fields": [
            "phase and step timestamps",
            "local and fabric segment tuples",
            "application and wire byte counts",
            "completion order",
            "backend invocation order",
            "random-generator state",
        ],
    }


def _minimax_legacy_guard(legacy) -> dict[str, Any]:
    record = _load_json(MINIMAX_RECORD_PATH)
    checked = 0
    mismatches = []
    for row in record["rows"]:
        for family_key in (
            "family_d_collective_floor_phases",
            "family_s_collective_floor_phases",
        ):
            for phase in row[family_key]:
                estimate = legacy.estimate(
                    dtype=phase["requested_dtype"],
                    operation=phase["requested_operation"],
                    ranks=phase["requested_ranks"],
                    message_bytes=phase["message_bytes"],
                    donor=(
                        phase["donor_dtype"],
                        phase["donor_operation"],
                        phase["donor_ranks"],
                    ),
                )
                checked += 1
                observed = (
                    estimate.floor_charge_ps,
                    estimate.serialization_ps,
                    estimate.evidence_class,
                    estimate.transfer_reason,
                )
                expected = (
                    phase["aggregate_floor_ps"],
                    phase["calibrated_serialization_ps"],
                    phase["evidence_class"],
                    phase["transfer_reason"],
                )
                if observed != expected:
                    mismatches.append(
                        {"expert_parallel": row["expert_parallel"], "phase": phase["phase"]}
                    )
    return {
        "held": not mismatches,
        "checked_queries": checked,
        "mismatches": mismatches,
        "record_sha256": _sha256_file(MINIMAX_RECORD_PATH),
    }


def _packet_disposition() -> dict[str, Any]:
    frozen_cells = [
        {"family": "PZ", "participants": participants, "receiver_fan_in": 0, "payload_bytes": payload}
        for participants in (2, 8)
        for payload in (65_536, 1_048_576)
    ] + [
        {"family": "PN", "participants": participants, "receiver_fan_in": participants - 1, "payload_bytes": payload}
        for participants in (4, 8)
        for payload in (65_536, 1_048_576)
    ]
    return {
        "status": "UNDECIDABLE",
        "denominator": None,
        "cells": frozen_cells,
        "before_after_phase_completion_errors": None,
        "reason": (
            "the H200 source identifies only opaque aggregate completion; the tree "
            "does not contain generation-scoped flits, receiver-owned credits, "
            "explicit traffic classes and virtual channels, replay identity, "
            "receive ordering, NVSwitch input ports and virtual output queues, or "
            "a two-sided crossbar arbitration policy"
        ),
        "guards": [
            {
                "id": "PC-FG-1",
                "decidable": False,
                "finding": "credit, queue, port, switch and arbitration values lack independent H200 evidence",
            },
            {
                "id": "PC-FG-2",
                "decidable": False,
                "finding": "the required packet mechanism structure is absent",
            },
        ],
        "one_authority_proof": (
            "no packet path is enabled, so the opaque aggregate authority remains "
            "the sole timing owner"
        ),
        "no_double_count_proof": (
            "the packet family did not execute and contributes no bytes, service, "
            "queue wait or aggregate charge"
        ),
        "preservation": "covered by the byte-exact pre-wave bypass guard",
    }


def _evaluate(workdir: Path) -> dict[str, Any]:
    started = time.monotonic()
    (
        base,
        completion,
        database,
        training,
        legacy,
        attempt_0005,
        arithmetic,
        final,
        authority_sha256,
    ) = _authorities()
    family_h = _family_h(
        base=base,
        database=database,
        attempt_0005=attempt_0005,
        arithmetic=arithmetic,
        final=final,
        authority_sha256=authority_sha256,
    )
    family_d8 = _family_d8(final)
    bypass = _bypass_guard(workdir / "bypass")
    minimax = _minimax_legacy_guard(legacy)
    packet = _packet_disposition()
    guards = [
        {"id": "A-H-FG-1", "held": all(row["training_cell_ids"] for row in family_h["rows"]), "finding": "every estimate records its rule and anchors"},
        {"id": "A-H-FG-2", "held": len(training) == 63 and family_h["denominator"] == 63, "finding": "the immutable memberships remain disjoint"},
        {"id": "A-H-FG-3", "held": authority_sha256 == family_h["authority_sha256_before_holdout_load"], "finding": "the authority was serialized and hashed before holdout loading"},
        {"id": "A-H-FG-4", "held": minimax["held"], "finding": minimax},
        {"id": "A-H-FG-5", "held": bypass["held"], "finding": bypass},
        {"id": "A-H-FG-7", "held": completion["model"]["model_id"] == final.calibration_id, "finding": "the qualifying-candidate freeze precedes repository implementation"},
        {"id": "ONE-AUTHORITY", "held": all(row["above_physical_floor"] for row in family_h["rows"]), "finding": "each opaque completion is one whole charge with zero exposed serialization"},
    ]
    return {
        "schema": SCHEMA,
        "study": "collective_floor_calibration_v1",
        "wall_time_seconds": time.monotonic() - started,
        "source": base["source"],
        "chronology": {
            "attempt_0004": "interpretable predecessor, H and D8 refuted",
            "attempt_0005": "paired-operation trend ratio refuted at 46 of 63",
            "attempt_0006": "arithmetic candidate, post-specified and superseded input digest",
            "attempt_0007": "arithmetic candidate, post-specified regression only",
            "attempt_0008": "first qualifying geometric evaluation",
            "current_attempt": workdir.parent.name,
            "qualifying_freeze_commit": "6a0be18",
        },
        "inputs": {
            "base_config_sha256": _sha256_file(BASE_CONFIG_PATH),
            "second_config_sha256": _sha256_file(SECOND_CONFIG_PATH),
            "completion_config_sha256": _sha256_file(COMPLETION_CONFIG_PATH),
            "qualifying_expectations_sha256": _sha256_file(
                QUALIFYING_EXPECTATIONS_PATH
            ),
            "prior_record_sha256": _sha256_file(PRIOR_RECORD_PATH),
            "attempt_0005_report_sha256": _sha256_file(ATTEMPT_0005_PATH),
        },
        "authority": final.as_dict(),
        "authority_sha256_before_holdout_load": authority_sha256,
        "families": {"H": family_h, "D8": family_d8, "C": packet},
        "fatal_guards_without_determinism": guards,
        "bypass": bypass,
        "minimax_legacy_queries": minimax,
    }


def _without_wall_time(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_wall_time(item)
            for key, item in value.items()
            if key != "wall_time_seconds"
        }
    if isinstance(value, list):
        return [_without_wall_time(item) for item in value]
    return value


def _csv_bytes(record: dict[str, Any]) -> bytes:
    output = StringIO(newline="")
    columns = (
        "cell_id",
        "operation",
        "ranks",
        "true_bytes",
        "measured_ps",
        "attempt_0004_prediction_ps",
        "attempt_0004_relative_error",
        "attempt_0005_prediction_ps",
        "attempt_0005_relative_error",
        "arithmetic_prediction_ps",
        "arithmetic_relative_error",
        "final_prediction_ps",
        "final_relative_error",
        "passed",
        "rule",
    )
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in record["families"]["H"]["rows"]:
        writer.writerow({column: row[column] for column in columns})
    return output.getvalue().encode("utf-8")


def _coordinator(workdir: Path, *, write_tracked: bool) -> dict[str, Any]:
    if workdir.exists():
        raise SystemExit(f"refusing to overwrite append-only attempt {workdir}")
    workdir.mkdir(parents=True)
    evaluations = []
    for label in ("evaluation-1", "evaluation-2"):
        output = workdir / f"{label}.json"
        completed = subprocess.run(
            [
                sys.executable,
                os.fspath(Path(__file__).resolve()),
                "--internal-output",
                os.fspath(output),
                "--internal-workdir",
                os.fspath(workdir / label),
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"{label} failed: {detail}")
        evaluations.append(_load_json(output))
    first = _canonical_json_bytes(_without_wall_time(evaluations[0]))
    second = _canonical_json_bytes(_without_wall_time(evaluations[1]))
    deterministic = first == second
    record = evaluations[0]
    record["wall_time_seconds"] = max(
        evaluation["wall_time_seconds"] for evaluation in evaluations
    )
    guards = record.pop("fatal_guards_without_determinism")
    guards.insert(
        5,
        {
            "id": "A-H-FG-6",
            "held": deterministic,
            "finding": {
                "fresh_processes": 2,
                "excluded_field": "wall_time_seconds",
                "evaluation_1_sha256": _sha256_bytes(first),
                "evaluation_2_sha256": _sha256_bytes(second),
            },
        },
    )
    record["fatal_guards"] = guards
    record["verdict"] = "INTERPRETABLE" if all(guard["held"] for guard in guards) else "VOID"
    record["family_tallies"] = {
        "H": {key: record["families"]["H"][key] for key in ("status", "passed", "denominator")},
        "D8": {key: record["families"]["D8"][key] for key in ("status", "passed", "denominator")},
        "C": {"status": record["families"]["C"]["status"], "denominator": None},
    }
    record_bytes = _json_bytes(record)
    csv_bytes = _csv_bytes(record)
    (workdir / "record.json").write_bytes(record_bytes)
    (workdir / "results.csv").write_bytes(csv_bytes)
    if write_tracked:
        TRACKED_RECORD_PATH.write_bytes(record_bytes)
        TRACKED_CSV_PATH.write_bytes(csv_bytes)
    return record


def _check() -> None:
    record_bytes = TRACKED_RECORD_PATH.read_bytes()
    csv_bytes = TRACKED_CSV_PATH.read_bytes()
    if b"\r" in record_bytes or b"\r" in csv_bytes:
        raise SystemExit("completion artifacts must use LF line endings")
    record = json.loads(record_bytes)
    if record.get("schema") != SCHEMA or record.get("verdict") != "INTERPRETABLE":
        raise SystemExit("completion_record.json is not an interpretable publication")
    if _csv_bytes(record) != csv_bytes:
        raise SystemExit("completion_results.csv has drifted from its record")
    print(
        "completion record and CSV are current; "
        f"record_sha256={_sha256_bytes(record_bytes)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-write-tracked", action="store_true")
    parser.add_argument("--internal-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--internal-workdir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.internal_output is not None:
        if args.internal_workdir is None:
            parser.error("--internal-output requires --internal-workdir")
        args.internal_workdir.mkdir(parents=True)
        args.internal_output.write_bytes(_json_bytes(_evaluate(args.internal_workdir)))
        return 0
    if args.check:
        _check()
        return 0
    if args.workdir is None:
        parser.error("--workdir is required unless --check is selected")
    record = _coordinator(args.workdir, write_tracked=not args.no_write_tracked)
    print(
        f"verdict={record['verdict']} "
        f"H={record['families']['H']['passed']}/{record['families']['H']['denominator']} "
        f"D8={record['families']['D8']['status']} C={record['families']['C']['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
