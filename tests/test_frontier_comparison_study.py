from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.frontier_comparison_v1 import publish_study, run_study

STUDY_DIR = REPOSITORY_ROOT / "examples" / "frontier_comparison_v1"
RESULT_PATH = STUDY_DIR / "results.json"
RESULT_SHA256 = "c33154071d5e0275bd15d25813102b629a25eca702f98b527949b53b37728017"
ARTIFACT_SHA256 = {
    "RESULTS.md": "cb11842e3e5e9fd7bb5dd35ab56e5c142741cbdabdf1dac43383aa9ef723a31f",
    "figures/frontier-comparison.pdf": (
        "1f9f81fe5b077d37964d29285d722223f01c76f301445a29e2da3399468279e5"
    ),
    "figures/frontier-comparison.png": (
        "4c4058c6cc231f2d8e8768182a5ff828db8d760726afb855f0c8c7b2ebfe36bf"
    ),
    "results.csv": "18845ff21dc2a32180022ea5e56872506bbbca41a4fa9045412fad2251997130",
    "results.json": RESULT_SHA256,
}


def test_frontier_comparison_publication_bytes_are_locked() -> None:
    for relative, expected in ARTIFACT_SHA256.items():
        assert hashlib.sha256((STUDY_DIR / relative).read_bytes()).hexdigest() == (
            expected
        )


def test_frontier_comparison_frozen_inputs_are_unchanged() -> None:
    assert run_study.sha256_file(run_study.EXPECTATIONS_PATH) == (
        run_study.EXPECTATIONS_SHA256
    )
    rows, all_match = run_study._external_hashes()
    assert all_match is True
    assert len(rows) == 4


def test_published_frontier_comparison_record_is_locked() -> None:
    payload = RESULT_PATH.read_bytes()
    result = json.loads(payload)

    assert hashlib.sha256(payload).hexdigest() == RESULT_SHA256
    assert result["schema"] == run_study.RESULT_SCHEMA
    assert result["verdict"] == "MIXED"
    assert result["nonvoid"] is True
    assert all(guard["held"] for guard in result["fatal_guards"])
    assert result["pricing_subprocess_count"] == 0
    assert result["external_rows_entered_pricing"] is False
    assert result["acceptance"] == {
        "nonvoid": True,
        "wall_time_pass": True,
        "x1_pass": True,
        "x2_pass": False,
        "x3a_pass": True,
        "x3b_pass": True,
        "x3c_pass": False,
    }
    assert result["families"]["X3"]["X3b"]["passed"] == 10
    assert [
        row["frontier_answer"]["frontier_index"]
        for row in result["families"]["X3"]["X3b"]["rows"]
    ] == [1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    assert result["families"]["X3"]["X3c"]["passed"] == 3
    assert [
        row["miss_direction"]
        for row in result["families"]["X3"]["X3c"]["rows"]
    ] == [None, None, None] + ["below-0.6"] * 7
    assert result["families"]["X2"]["decode_e_star"] == {
        "decimal": 0.5860677342847804,
        "denominator": 9_179_000_000,
        "numerator": 5_379_515_733,
    }
    assert result["families"]["X2"]["prefill_e_star"] == {
        "decimal": 0.14255218101749795,
        "denominator": 49_105_750_000,
        "numerator": 7_000_131_763,
    }
    assert result["attempt_evidence"]["deterministic_reproduction_matched"] is (
        True
    )


def test_x1_and_x2_decode_direction_rerun_without_processes() -> None:
    inventory = run_study._load_json(run_study.INVENTORY_PATH)
    suite = run_study._load_json(run_study.SUITE_PATH)
    derivation = run_study.derive_work(inventory)

    with run_study.ProcessGuard() as guard:
        x1 = run_study._x1(
            derivation,
            suite["reference_model"]["parameter_count"],
        )
        x2 = run_study._x2(derivation)

    assert guard.attempts == []
    assert x1["passed"] == x1["denominator"] == 3
    assert x1["candidate_key"] == (
        "9981c2a9ebe89ed40002b9e1ce952ca1d742fa2b082e7f306a21a72d3966b9f9"
    )
    decode_direction = next(row for row in x2["rows"] if row["id"] == "X2a")
    assert decode_direction == {
        "id": "X2a",
        "passed": True,
        "predicted_ps": 5_379_515_733,
        "external_ps": 9_179_000_000,
    }


def test_model_work_divides_flops_and_bytes_once_per_tensor_rank() -> None:
    inventory = run_study._load_json(run_study.INVENTORY_PATH)
    derivation = run_study.derive_work(inventory)

    assert (
        derivation.decode_attention_score_flops_per_token_pair
        == 8 * derivation.prefill_attention_score_flops_per_token_pair
        == 2_097_152
    )
    for tensor_parallel in run_study.TP_WIDTHS:
        for phase, total_flops, total_dynamic_bytes in (
            (
                "decode",
                derivation.decode_total_flops_per_batch_item,
                derivation.decode_kv_bytes,
            ),
            (
                "prefill",
                derivation.prefill_total_flops_per_request,
                derivation.prefill_kv_bytes,
            ),
        ):
            work = run_study._model_work(
                derivation,
                phase=phase,
                tensor_parallel=tensor_parallel,
            )
            assert work.flops_per_batch_item == total_flops // tensor_parallel
            assert (
                work.static_logical_hbm_bytes
                == derivation.static_parameter_bytes // tensor_parallel
            )
            assert (
                work.dynamic_hbm_bytes_per_batch_item
                == total_dynamic_bytes // tensor_parallel
            )


def test_attempt_directories_are_append_only_and_record_portable_argv(
    tmp_path: Path,
) -> None:
    first, previous = run_study._begin_attempt(tmp_path)
    assert first.name == "attempt-1"
    assert previous is None
    manifest = json.loads((first / "attempt.json").read_text(encoding="utf-8"))
    assert manifest["portable_argv"] == run_study._portable_argv()
    assert all(str(tmp_path) not in item for item in manifest["portable_argv"])

    with pytest.raises(SystemExit, match="verdict records are missing"):
        run_study._begin_attempt(tmp_path)

    (first / "verdict.json").write_text("{}\n", encoding="utf-8")
    second, previous = run_study._begin_attempt(tmp_path)
    assert second.name == "attempt-2"
    assert previous == first


def test_deterministic_projection_excludes_only_attempt_and_wall_fields() -> None:
    first = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    first["verdict"] = "PASS"
    first["attempt_evidence"] = {"attempt_id": "attempt-1"}
    second = deepcopy(first)
    second["verdict"] = "MIXED"
    second["attempt_evidence"] = {"attempt_id": "attempt-2"}
    second["families"]["X3"]["elapsed_seconds"] += 1.0
    second["families"]["W"]["elapsed_seconds"] += 1.0
    second["families"]["W"]["passed"] = 0
    second["score_classes"]["behavioral_relations"]["W"]["passed"] = 0
    second["acceptance"]["wall_time_pass"] = False

    assert run_study.deterministic_projection(first) == (
        run_study.deterministic_projection(second)
    )


def test_publisher_derives_both_x3c_miss_directions() -> None:
    assert publish_study._x3c_verdict(
        {"passed": False, "miss_direction": "below-0.6"}
    ) == "FAIL, below e=0.6"
    assert publish_study._x3c_verdict(
        {"passed": False, "miss_direction": "above-1.0"}
    ) == "FAIL, above e=1.0"
