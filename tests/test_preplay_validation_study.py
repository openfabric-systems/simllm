"""Regression tests for the PLAY-5 study harness accounting."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

STUDY_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "preplay_validation_v1"
    / "run_study.py"
)
SPEC = importlib.util.spec_from_file_location("preplay_validation_study", STUDY_PATH)
assert SPEC is not None and SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_passed_scored_counts_passing_oracle_rows_not_executed_rows():
    oracle = {
        "status": "executed",
        "executed_scored": 2,
        "rows": [{"passed": True}, {"passed": False}],
    }
    replay = {"passed_scored": 13}
    assert study._count_passed_scored(oracle, replay) == 14


def test_none_stop_reason_is_not_eos_when_tokenizer_has_no_eos_id():
    choice = SimpleNamespace(finish_reason="stop", stop_reason=None, token_ids=(7,))
    assert study._normalized_vllm_stop(choice, None) == "unknown-stop:None"


def test_none_stop_reason_uses_the_scheduler_final_eos_token():
    choice = SimpleNamespace(finish_reason="stop", stop_reason=None, token_ids=(7, 0))
    assert study._normalized_vllm_stop(choice, 0) == "eos"
