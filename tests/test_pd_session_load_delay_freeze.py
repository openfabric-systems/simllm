from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from simllm.calibration.batch_service_surface import BatchServicePoint

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples" / "pd_session_load_delay_v1"


def _model():
    spec = importlib.util.spec_from_file_location(
        "pd_session_load_delay_frozen_queue_model",
        STUDY_DIR / "queue_model.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(name: str) -> dict:
    return json.loads((STUDY_DIR / name).read_text(encoding="utf-8"))


def _points(surface: dict) -> tuple[BatchServicePoint, ...]:
    return tuple(
        BatchServicePoint(
            row["batch_size"],
            row["measured_service_ps"],
            row["trimmed_coefficient_of_variation_ppm"] / 1_000_000,
            row["entry_key_sha256"],
            row["evidence_class"],
            row["split"],
        )
        for row in surface["points"]
    )


def test_freeze_is_pre_run_and_exposure_is_honestly_contaminated() -> None:
    freeze = _json("expectations.json")

    assert freeze["status"] == "EXPECTATIONS_ONLY"
    assert freeze["chronology"]["concurrent_surface_priced_run_existed_before_freeze"] is False
    assert freeze["chronology"]["curve_values_accessed_before_freeze"] is False
    assert freeze["exposure"]["status"] == "CONTAMINATED"
    assert freeze["exposure"]["clean_close_permitted"] is False
    assert freeze["decision_rule"]["task_state"].startswith(
        "VLLM-39 remains open"
    )


def test_surface_selects_exact_measured_granite_batch_keys() -> None:
    surface = _json("surface.json")

    assert surface["record_sha256"] == (
        "ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52"
    )
    assert surface["acceptance_status"] == "candidate"
    assert surface["calibration_claim"] is False
    assert [row["batch_size"] for row in surface["points"]] == [1, 8]
    assert [row["evidence_class"] for row in surface["points"]] == [
        "MEASURED",
        "MEASURED",
    ]
    assert [row["split"] for row in surface["points"]] == [
        "calibration",
        "calibration",
    ]
    assert surface["held_out_batch_32_decoded_or_captured"] is False
    assert surface["record_bytes_consumed"] < surface["record_total_bytes"]


def test_access_and_surface_content_addresses_are_frozen() -> None:
    freeze = _json("expectations.json")

    access = (STUDY_DIR / "access_ledger.jsonl").read_bytes()
    surface = (STUDY_DIR / "surface.json").read_bytes()

    assert hashlib.sha256(access).hexdigest() == freeze["exposure"][
        "access_ledger_sha256"
    ]
    assert hashlib.sha256(surface).hexdigest() == freeze["surface"]["sha256"]
    rows = [json.loads(line) for line in access.splitlines()]
    assert len(rows) == 8
    assert sum(row["status"] == "PASS" for row in rows) == 5
    assert sum(row["status"] == "REJECTED" for row in rows) == 2
    assert rows[0]["status"] == "CONTAMINATED"


def test_every_segment_and_held_out_band_rederive_without_curves() -> None:
    model = _model()
    freeze = _json("expectations.json")
    points = _points(_json("surface.json"))
    configurations = tuple(
        (prefill, decode, prompt)
        for prefill, decode in model.POOL_RATIOS
        for prompt in model.PROMPT_LENGTHS
    )
    segments = [
        row
        for configuration in configurations
        for row in model.predicted_segments(points, configuration)
    ]
    bands = [
        model.predict_point(
            points,
            prefill_engines=prefill,
            decode_engines=decode,
            prompt_tokens=prompt,
            offered_load=load,
        )
        for prefill, decode, prompt in model.HELD_OUT_CONFIGURATIONS
        for load in model.OFFERED_LOADS
    ]

    assert segments == freeze["expected_segments"]
    assert len(segments) == 30
    assert bands == freeze["held_out_prediction_bands"]
    assert len(bands) == 24


def test_preservation_locks_cover_control_comparator_and_flagships() -> None:
    locks = _json("expectations.json")["preservation_locks"]

    assert locks["core51_one_request_control"] == {
        "artifact_count": 6,
        "manifest_sha256": "092d79c35c7632e87427804cda11bc6fe0890d2c98ceec23ff30af2e1143ad4d",
        "total_bytes": 61_248,
    }
    assert locks["deterministic_concurrent_comparator"] == {
        "artifact_count": 9,
        "manifest_sha256": "d09202846afeba9efc019d4f44f881fde6639457c0398f1ed4ae8da1e0c804c3",
        "total_bytes": 56_495,
    }
    assert locks["scored_flagship_artifacts"] == {
        "artifact_count": 17,
        "manifest_sha256": "7630ebdaf91a722ff5004184a03a38fac98bbf11f2adbbfd5e8e32838ff130d5",
        "total_bytes": 1_198_680,
    }
