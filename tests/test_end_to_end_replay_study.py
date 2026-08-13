"""Study-entrypoint regression checks for the end-to-end per-request replay."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from simllm.core import RequestPhase, ScheduledRequest, StepRecord

REPOSITORY = Path(__file__).resolve().parents[1]
STUDY_PATH = REPOSITORY / "examples/end_to_end_replay_v1/run_study.py"


def _study_module():
    spec = importlib.util.spec_from_file_location("end_to_end_replay_v1", STUDY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_only_refuses_a_missing_model_and_writes_nothing(tmp_path):
    output = tmp_path / "must-not-exist"
    missing = tmp_path / "also-missing"

    result = subprocess.run(
        [
            sys.executable,
            str(STUDY_PATH),
            "--cache-dir",
            str(missing),
            "--htsim-rnic",
            str(missing),
            "--run-dir",
            str(output),
            "--check-only",
        ],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "pinned model snapshot is missing" in result.stderr
    assert not output.exists()


def test_frozen_constants_stay_self_consistent():
    study = _study_module()

    assert study.VECTOR_BYTES == 2048
    assert study.S1_MOE_BYTES_FLOOR == 5_308_416
    assert study.S1_MOE_BYTES_CEILING == 37_158_912
    assert study.BANDWIDTHS_BPS == (100_000_000_000, 200_000_000_000, 400_000_000_000)
    assert study.CASE_REQUEST_IDS["a"] == study.CASE_REQUEST_IDS["b"][:3]
    assert len(study.CASE_REQUEST_IDS["b"]) == 4 * len(study.CASE_REQUEST_IDS["a"])
    assert study.EXPECTED_EXACT_ORACLE_RELATIONS == 13
    assert study.EXPECTED_BEHAVIORAL_RELATIONS == 4
    assert study.ENGINE_RANK == 0
    assert {bandwidth for _, case, ep, bandwidth in study.CELLS if case == "a" and ep == 8} == set(
        study.BANDWIDTHS_BPS
    )


@pytest.mark.parametrize(
    ("oracle", "finish", "stop", "expected"),
    [
        ("length-cap", "length", None, True),
        ("length-cap", "stop", None, False),
        ("eos", "stop", None, True),
        ("eos", "stop", "SIMLLM_STOP", False),
        ("eos", "length", None, False),
        ("stop-string", "stop", "SIMLLM_STOP", True),
        ("stop-string", "stop", None, False),
    ],
)
def test_stop_reason_normalization_is_the_frozen_table(oracle, finish, stop, expected):
    study = _study_module()
    row = {"oracle_stop_reason": oracle, "finish_reason": finish, "stop_reason": stop}
    assert study._stop_reason_agrees(row) is expected


def test_goal_parser_reads_sends_and_refuses_a_duplicate(tmp_path):
    study = _study_module()
    path = tmp_path / "artifact.goal"
    path.write_text(
        "num_ranks 2\n"
        "rank 0 {\n"
        "  l1: send 2048b to 1 tag 1000 cpu 0 nic 0\n"
        "}\n"
        "rank 1 {\n"
        "  l2: send 4096b to 0 tag 1000 cpu 0 nic 0\n"
        "}\n",
        encoding="utf-8",
    )
    assert study._goal_sends(path) == {1000: {(0, 1): 2048, (1, 0): 4096}}

    duplicate = tmp_path / "duplicate.goal"
    duplicate.write_text(
        "rank 0 {\n"
        "  l1: send 2048b to 1 tag 1000 cpu 0 nic 0\n"
        "  l2: send 2048b to 1 tag 1000 cpu 0 nic 0\n"
        "}\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="duplicate GOAL send"):
        study._goal_sends(duplicate)


def test_composition_key_separates_steps_that_share_request_identities():
    study = _study_module()
    first = {
        "scheduled": [
            {"request_id": "r0", "phase": "decode", "num_new_tokens": 1, "context_length": 9}
        ]
    }
    second = {
        "scheduled": [
            {"request_id": "r0", "phase": "decode", "num_new_tokens": 1, "context_length": 10}
        ]
    }
    assert study._composition_key(first) != study._composition_key(second)


#: every expert here is owned by rank 0 at both frozen expert-parallel widths
LOCAL_ONLY = (0, 8, 16, 24, 0, 8, 16, 24)


def _routing(layer_zero: tuple[int, ...]) -> dict[int, tuple[int, ...]]:
    routing = {layer: LOCAL_ONLY for layer in range(24)}
    routing[0] = layer_zero
    return routing


def _capture_fixture() -> dict:
    #: at width 8 the first token reaches ranks 1 and 5; at width 4 both of
    #: those experts collapse onto rank 1. Every other layer stays local, so it
    #: must move no bytes at all.
    return {
        "requests": {
            "r0": {"input_token_ids": [1, 2], "output_token_ids": [3, 4]},
        },
        "forwards": {
            ("r0", "prefill"): [
                {
                    "token_index": 0,
                    "token_id": 1,
                    "routing": _routing((1, 5, 9, 13, 17, 21, 25, 29)),
                },
                {
                    "token_index": 1,
                    "token_id": 2,
                    "routing": _routing((1, 9, 17, 25, 1, 9, 17, 25)),
                },
            ],
        },
    }


def test_independent_recomputation_matches_a_hand_computed_table():
    study = _study_module()
    raw = _capture_fixture()
    record = StepRecord(
        0,
        0,
        [ScheduledRequest("r0", RequestPhase.PREFILL, 2, context_length=2)],
        num_sampled=1,
    )
    tables = study._independent_request_pairs(record, raw, 8)

    # layer 0: token 0 reaches ranks 1 and 5, token 1 reaches rank 1 only
    assert tables["step-0:layer-0:ep-dispatch"] == {
        ("r0", 0, 1): 2 * study.VECTOR_BYTES,
        ("r0", 0, 5): 1 * study.VECTOR_BYTES,
    }
    assert tables["step-0:layer-0:ep-combine"] == {
        ("r0", 1, 0): 2 * study.VECTOR_BYTES,
        ("r0", 5, 0): 1 * study.VECTOR_BYTES,
    }
    # layer 1 keeps every expert on rank 0, so it contributes no operation at all
    assert "step-0:layer-1:ep-dispatch" not in tables


def test_independent_recomputation_follows_the_expert_parallel_width():
    study = _study_module()
    raw = _capture_fixture()
    record = StepRecord(
        0,
        0,
        [ScheduledRequest("r0", RequestPhase.PREFILL, 2, context_length=2)],
        num_sampled=1,
    )
    narrow = study._independent_request_pairs(record, raw, 4)
    wide = study._independent_request_pairs(record, raw, 8)
    narrow_bytes = sum(narrow["step-0:layer-0:ep-dispatch"].values())
    wide_bytes = sum(wide["step-0:layer-0:ep-dispatch"].values())
    assert narrow_bytes < wide_bytes


def test_independent_recomputation_refuses_a_slice_outside_the_capture():
    study = _study_module()
    record = StepRecord(
        0,
        0,
        [ScheduledRequest("r0", RequestPhase.PREFILL, 9, context_length=9)],
        num_sampled=1,
    )
    with pytest.raises(AssertionError, match="outside the capture"):
        study._independent_request_pairs(record, _capture_fixture(), 8)
