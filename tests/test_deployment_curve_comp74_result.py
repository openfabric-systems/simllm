"""Published-result locks for COMP-74 distribution propagation."""

from __future__ import annotations

import csv
import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/deployment_curve_v1"
RESULT = STUDY / "comp74_distribution_result.json"
DIGEST = STUDY / "comp74_distribution_result.sha256"


def _load() -> dict:
    with RESULT.open(encoding="utf-8", newline="") as stream:
        value = json.load(stream)
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_comp74_per_key_intervals_are_exact_and_nonzero() -> None:
    result = _load()
    rows = {row["anchor_id"]: row for row in result["key_intervals"]}

    assert result["status"] == "PASS"
    assert {
        anchor: (
            row["published_point_ps"],
            row["independent_repeat_ps"],
            row["service_interval_ps"]["lower"],
            row["service_interval_ps"]["upper"],
            Fraction(**row["relative_half_width"]),
        )
        for anchor, row in rows.items()
    } == {
        "sglang_prefill_1k": (
            89_393_440_000,
            91_249_600_000,
            87_537_280_000,
            91_249_600_000,
            Fraction(11_601, 558_709),
        ),
        "sglang_prefill_2k": (
            93_134_208_000,
            94_656_736_000,
            91_611_680_000,
            94_656_736_000,
            Fraction(47_579, 2_910_444),
        ),
        "sglang_prefill_4k": (
            104_598_911_000,
            104_294_464_000,
            104_294_464_000,
            104_903_358_000,
            Fraction(304_447, 104_598_911),
        ),
        "sglang_decode_standard": (
            1_875_680_000,
            1_883_392_000,
            1_867_968_000,
            1_883_392_000,
            Fraction(241, 58_615),
        ),
    }
    assert all(row["observation_count"] == 2 for row in rows.values())
    assert all(row["nonzero"] is True for row in rows.values())
    assert all(row["stability_claim"] is False for row in rows.values())


def test_comp74_off_reproduces_every_point_interval_and_curve_exactly() -> None:
    proof = _load()["distribution_off_reproduction"]

    assert proof["status"] == "PASS"
    assert proof["current_point_prediction_count"] == 15
    assert proof["all_points_exact"] is True
    assert proof["all_interval_objects_equal"] is True
    assert proof["all_stored_curves_equal"] is True
    assert all(row["point_exact"] for row in proof["anchor_layers"])
    assert all(row["interval_object_equal"] for row in proof["anchor_layers"])
    assert [row["point_count"] for row in proof["curves"]] == [5, 5]


def test_comp74_band_movement_never_rescores_and_reports_boundary_contact() -> None:
    result = _load()
    movement = result["band_movement"]

    assert movement["rescore_performed"] is False
    assert movement["new_boundary_contact_count"] == 1
    assert movement["new_boundary_contacts"] == [
        {"anchor_id": "sglang_prefill_2k", "layer": "physics_only"}
    ]
    contact = next(row for row in movement["rows"] if row["new_boundary_contact"])
    assert contact["propagated_touches_upper_boundary"] is True
    assert contact["propagated_touches_lower_boundary"] is False
    assert contact["frozen_point_verdict"] == "PASS"
    assert all(row["rescore_performed"] is False for row in movement["rows"])
    mtp = [
        row
        for row in movement["rows"]
        if row["anchor_id"] == "sglang_decode_simulated_mtp"
    ]
    assert len(mtp) == 3
    assert all(row["prior"] == row["propagated"] for row in mtp)
    assert {row["distribution_status"] for row in mtp} == {
        "SINGLE_SEED_NOT_PROPAGATED"
    }


def test_comp74_verdict_evidence_and_closure_are_literal() -> None:
    result = _load()

    assert result["verdicts"] == {
        "run3_prefill": "PASS",
        "run4_mtp": "REFUTED",
        "combined": "ALL_SCORABLE_HELD_OUT_REFUTED",
        "changed": False,
        "rule": "band widening never flips or recomputes a frozen point verdict",
    }
    evidence = result["evidence"]
    assert evidence["acceptance_status_before"] == "candidate"
    assert evidence["acceptance_status_after"] == "candidate"
    assert evidence["candidate_promotion_performed"] is False
    assert evidence["ledger_equal"] is True
    assert evidence["lookup_service_ledger_before"] == evidence[
        "lookup_service_ledger_after"
    ]
    assert result["closure"]["comp74"] == "CLOSED_LITERAL"
    assert result["closure"]["priced_key_count"] == 4
    assert result["closure"]["residuals_registered"] == [
        {
            "id": "COMP-79",
            "scope": "single-seed DeepSeek keys including simulated MTP",
        },
        {
            "id": "COMP-80",
            "scope": "Granite arm repetitions absent from the retained partial campaign",
        },
    ]


def test_comp74_preservation_access_and_content_address_hold() -> None:
    result = _load()

    assert result["preservation_lock"]["status"] == "PASS"
    assert len(result["preservation_lock"]["artifacts"]) == 18
    for artifact in result["preservation_lock"]["artifacts"]:
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    assert result["access"]["whole_record_loaded"] is False
    assert result["access"]["successful_projection_count"] == 3
    assert all(row["whole_record_loaded"] is False for row in result["access"]["rows"])
    assert all(
        row["unselected_values_returned"] is False
        for row in result["access"]["rows"]
    )
    expected = "5970872d6504081e37b825d56946da5dd8d207ee4253993365a4334c46c613bc"
    assert _sha256(RESULT) == expected
    assert DIGEST.read_text(encoding="utf-8") == f"{expected}  {RESULT.name}\n"


def test_comp74_tables_cover_all_keys_and_anchor_layers() -> None:
    with (STUDY / "comp74_per_key_intervals.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        key_rows = list(csv.DictReader(stream))
    with (STUDY / "comp74_band_table.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        band_rows = list(csv.DictReader(stream))

    assert len(key_rows) == 4
    assert {row["candidate_status"] for row in key_rows} == {"candidate"}
    assert len(band_rows) == 15
    assert {row["rescore_performed"] for row in band_rows} == {"False"}


def test_comp74_report_uses_no_em_dash_or_absolute_machine_path() -> None:
    report = (STUDY / "COMP74_RESULTS.md").read_text(encoding="utf-8")

    assert chr(0x2014) not in report
    assert "/data3/" not in report
