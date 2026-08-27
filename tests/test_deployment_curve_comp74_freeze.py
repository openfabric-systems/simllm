from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/deployment_curve_v1"
FREEZE_PATH = STUDY / "comp74_expectations.json"


def _freeze() -> dict[str, object]:
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


def _reader():
    sys.path.insert(0, str(STUDY))
    try:
        spec = importlib.util.spec_from_file_location(
            "comp74_field_reader_test",
            STUDY / "comp74_field_reader.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _encoded(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def test_comp74_freeze_precedes_all_additional_repeat_access_and_results() -> None:
    frozen = _freeze()

    assert frozen["schema"] == "simllm-deployment-curve-comp74-expectations-v1"
    assert frozen["status"] == "EXPECTATIONS_ONLY"
    chronology = frozen["chronology"]
    assert chronology["additional_retained_repetition_values_accessed"] is False
    assert chronology["comp74_result_existed_before_freeze"] is False
    assert chronology["comp74_runner_existed_before_freeze"] is False
    assert chronology["distribution_intervals_computed"] is False
    assert chronology["field_reader_must_be_committed_before_record_access"] is True
    assert chronology["published_throughput_values_used"] is False


def test_comp74_statistic_interval_and_no_pooling_rules_are_literal() -> None:
    rule = _freeze()["estimation_rule"]

    assert rule["minimum_independent_observations"] == 2
    assert rule["relative_half_width_formula"] == (
        "max_i(abs(observation_i - published_point_ps)) / published_point_ps"
    )
    assert rule["pooling"] == {
        "across_roles": False,
        "across_prompt_lengths": False,
        "across_mtp_modes": False,
        "across_implementation_suffixes": False,
    }
    assert len(_freeze()["key_mapping"]) == 4


def test_comp74_reader_projects_only_exact_repeat_fields() -> None:
    reader = _reader()
    source = {
        "acceptance_status": "candidate",
        "lookup_record_sha256": reader.SUCCESSOR_SHA256,
        "predecessor_lookup_record_sha256": "f" * 64,
        "score": {
            "lookup_service_ledger": {"deepseek_v3": {"MEASURED": 4}},
            "priced_repeat_observations": [
                {
                    "implementation_suffix": "key-a",
                    "published_point_ps": 100,
                    "independent_repeat_ps": 104,
                    "signed_repeat_minus_point_ps": 4,
                    "retained_independent_observations": 2,
                    "distribution_propagation": "DEFERRED_TO_COMP-74",
                    "secret": 99,
                }
            ],
            "task_movement": {
                "comp74_repeat_inputs": "RETAINED_FOR_ALL_FOUR_PRICED_KEYS",
                "secret": 98,
            },
            "secret": 97,
        },
        "secret": 96,
    }
    raw = _encoded(source)

    value, consumed = reader.extract_successor_repeats(io.BytesIO(raw))

    row = value["score"]["priced_repeat_observations"][0]
    assert set(row) == set(reader.REPEAT_ROW_PROJECTION)
    assert "secret" not in json.dumps(value)
    assert consumed < len(raw)


def test_comp74_reader_path_allowlist_rejects_before_open_and_logs(tmp_path: Path) -> None:
    reader = _reader()
    access_log = tmp_path / "access.jsonl"

    with pytest.raises(ValueError, match="non-allowlisted"):
        reader.read_successor_repeats(tmp_path / "wrong.json", access_log)
    with pytest.raises(ValueError, match="non-allowlisted"):
        reader.read_run4_publication(tmp_path / "wrong.json", access_log)
    with pytest.raises(ValueError, match="non-allowlisted"):
        reader.read_curve_config(tmp_path / "wrong.json", access_log)

    rows = [
        json.loads(line)
        for line in access_log.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["status"] for row in rows] == ["REJECTED"] * 3
    assert all(row["whole_record_loaded"] is False for row in rows)
    assert all(row["unselected_values_returned"] is False for row in rows)


def test_comp74_freeze_preserves_candidate_status_and_registers_residuals() -> None:
    frozen = _freeze()

    assert frozen["evidence_invariants"]["lookup_record_acceptance_status"] == (
        "candidate"
    )
    assert frozen["evidence_invariants"]["candidate_promotion_allowed"] is False
    assert frozen["closure_rule"]["residuals"] == [
        {
            "id": "COMP-79",
            "scope": "single-seed DeepSeek keys including simulated MTP",
        },
        {
            "id": "COMP-80",
            "scope": "Granite arm repetitions absent from the retained partial campaign",
        },
    ]


def test_comp74_freeze_documents_use_no_em_dash() -> None:
    for path in (FREEZE_PATH, STUDY / "comp74_expectations.md"):
        assert chr(0x2014) not in path.read_text(encoding="utf-8")
