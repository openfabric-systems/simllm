"""Study-entrypoint regression checks for the SGLang closed-loop study.

None of these import SGLang or torch: they exercise the frozen constants, the
independent standard-library trace reader, the independent metric
recomputation and the provenance guard that keeps another framework's capture
out of an SGLang result.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
STUDY_PATH = REPOSITORY / "examples/sglang_end_to_end_v1/run_study.py"


def _study_module():
    spec = importlib.util.spec_from_file_location("sglang_end_to_end_v1", STUDY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_trace(path: Path, *, framework: str = "sglang") -> None:
    rows = [
        {
            "row_type": "header",
            "schema": "simllm-preplay-trace-v2",
            "provenance": {
                "schema": "simllm-preplay-trace-v2",
                "framework": framework,
                "routing_source": "observed-dispatch",
                "model_id": "ibm-granite/granite-3.0-1b-a400m-instruct",
                "model_revision": "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445",
                "expert_count": 32,
                "top_k": 8,
                "moe_layer_indices": list(range(24)),
            },
        },
        {
            "row_type": "request",
            "request_id": "p0",
            "input_token_ids": [1000, 1001],
            "output_token_ids": [7, 9],
            "output_length": 2,
        },
        {
            "row_type": "kv-event",
            "request_id": "p0",
            "event_kind": "prefix-hit",
        },
        {
            "row_type": "observed-dispatch",
            "request_id": "p0",
            "phase": "prefill",
            "token_index": 1,
            "token_id": 1001,
            "routing": [{"layer_index": 0, "expert_ids": [3, 4]}],
        },
        {
            "row_type": "observed-dispatch",
            "request_id": "p0",
            "phase": "prefill",
            "token_index": 0,
            "token_id": 1000,
            "routing": [{"layer_index": 0, "expert_ids": [1, 2]}],
        },
        {"row_type": "footer", "request_count": 1},
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_check_only_refuses_a_missing_model_and_writes_nothing(tmp_path):
    missing = tmp_path / "also-missing"
    output = tmp_path / "must-not-exist"

    result = subprocess.run(
        [
            sys.executable,
            str(STUDY_PATH),
            "--cache-dir",
            str(missing),
            "--sglang-python",
            str(missing),
            "--sglang-source",
            str(missing),
            "--routing-trace",
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
    assert study.S1_MOE_BYTES_FLOOR == 786_432
    assert study.S1_MOE_BYTES_CEILING == 5_505_024
    assert study.BANDWIDTHS_BPS == (100_000_000_000, 200_000_000_000, 400_000_000_000)
    assert study.REQUEST_IDS == ("p0", "p1", "p2", "p3")
    assert study.PROMPT_TOKENS == 8
    assert study.MAX_NEW_TOKENS == 12
    assert study.ARRIVAL_SPACING_PS == 1_000_000_000
    assert study.SGLANG_CHUNKED_PREFILL_SIZE == -1
    assert study.ENGINE_RANK == 0
    assert study.EXPECTED_EXACT_RELATIONS == 5
    assert study.EXPECTED_BEHAVIORAL_RELATIONS == 4
    assert {bandwidth for _, ep, bandwidth in study.CELLS if ep == 8} == set(
        study.BANDWIDTHS_BPS
    )
    assert study.prompt_token_ids(2) == tuple(range(1200, 1208))
    assert (
        study.B4_COMPUTE_RATIO_FLOOR
        < 857_217_024 / 555_227_136
        < study.B4_COMPUTE_RATIO_CEILING
    )


def test_the_provenance_guard_refuses_another_framework(tmp_path):
    study = _study_module()
    good = tmp_path / "sglang.jsonl"
    bad = tmp_path / "vllm.jsonl"
    _write_trace(good)
    _write_trace(bad, framework="vllm")

    provenance = study.check_trace_provenance(good)
    assert provenance["framework"] == "sglang"

    with pytest.raises(SystemExit, match="framework"):
        study.check_trace_provenance(bad)


def test_the_raw_reader_sorts_forward_rows_and_keeps_layer_tuples(tmp_path):
    study = _study_module()
    path = tmp_path / "trace.jsonl"
    _write_trace(path)

    raw = study.read_raw_trace(path)

    assert raw["header"]["framework"] == "sglang"
    assert raw["requests"]["p0"]["input_token_ids"] == [1000, 1001]
    rows = raw["forwards"][("p0", "prefill")]
    assert [row["token_index"] for row in rows] == [0, 1]
    assert rows[0]["routing"][0] == (1, 2)
    assert rows[1]["routing"][0] == (3, 4)


def _cell_payload() -> dict:
    """Two requests, three steps, hand-computed conserved metrics."""

    def step(index, virtual, latency, scheduled, kernel, collective):
        return {
            "step_index": index,
            "virtual_time_ps": virtual,
            "step_latency_ps": latency,
            "completed_at_ps": virtual + latency,
            "simulated": True,
            "scheduled": scheduled,
            "composed_phase_service_ps": [kernel, collective],
            "fabric_phase_service_ps": [0, collective],
        }

    prefill = {"request_id": "p0", "phase": "prefill", "num_new_tokens": 8, "context_length": 8}
    decode0 = {"request_id": "p0", "phase": "decode", "num_new_tokens": 1, "context_length": 9}
    decode1 = {"request_id": "p1", "phase": "prefill", "num_new_tokens": 8, "context_length": 8}
    return {
        "simulated": True,
        "requests": [
            {
                "request_id": "p0",
                "arrival_ps": 0,
                "token_count": 2,
                "first_token_at_ps": 100,
                "last_token_at_ps": 400,
                "ttft_ps": 100,
                "tpot_numerator": 300,
                "tpot_denominator": 1,
                "ttft_attribution": {
                    "queue_ps": 0,
                    "kv_ps": 0,
                    "kernel_ps": 60,
                    "dma_ps": 0,
                    "collective_ps": 40,
                    "nic_ps": 0,
                    "control_ps": 0,
                    "total_ps": 100,
                },
                "decode_attribution": {
                    "queue_ps": 100,
                    "kv_ps": 0,
                    "kernel_ps": 120,
                    "dma_ps": 0,
                    "collective_ps": 80,
                    "nic_ps": 0,
                    "control_ps": 0,
                    "total_ps": 300,
                },
            },
            {
                "request_id": "p1",
                "arrival_ps": 150,
                "token_count": 1,
                "first_token_at_ps": 300,
                "last_token_at_ps": 300,
                "ttft_ps": 150,
                "tpot_numerator": None,
                "tpot_denominator": None,
                "ttft_attribution": {
                    "queue_ps": 50,
                    "kv_ps": 0,
                    "kernel_ps": 60,
                    "dma_ps": 0,
                    "collective_ps": 40,
                    "nic_ps": 0,
                    "control_ps": 0,
                    "total_ps": 150,
                },
                "decode_attribution": {
                    "queue_ps": 0,
                    "kv_ps": 0,
                    "kernel_ps": 0,
                    "dma_ps": 0,
                    "collective_ps": 0,
                    "nic_ps": 0,
                    "control_ps": 0,
                    "total_ps": 0,
                },
            },
        ],
        "steps": [
            step(0, 0, 100, [prefill], 60, 40),
            step(1, 200, 100, [decode1], 60, 40),
            step(2, 300, 100, [decode0], 60, 40),
        ],
    }


def test_the_independent_recomputation_reproduces_a_hand_computed_cell():
    study = _study_module()

    metrics = study._independent_request_metrics(_cell_payload())

    assert metrics["p0"]["token_count"] == 2
    assert metrics["p0"]["first_token_at_ps"] == 100
    assert metrics["p0"]["last_token_at_ps"] == 400
    assert metrics["p0"]["ttft_ps"] == 100
    assert metrics["p0"]["tpot"] == Fraction(300, 1)
    assert metrics["p0"]["ttft_attribution"] == {
        "queue_ps": 0,
        "kernel_ps": 60,
        "collective_ps": 40,
    }
    # The 200 ps gap between p0's first completion at 100 and step 2's release
    # at 300 is queue time, so the decode interval is 200 + 60 + 40 = 300 ps.
    assert metrics["p0"]["decode_attribution"] == {
        "queue_ps": 200,
        "kernel_ps": 60,
        "collective_ps": 40,
    }
    assert metrics["p1"]["ttft_ps"] == 150
    assert metrics["p1"]["tpot"] is None


def test_e1_fails_when_a_published_metric_disagrees():
    study = _study_module()
    cell = _cell_payload()

    # The hand-built payload deliberately over-states p0's decode kernel time,
    # so E1 must catch the disagreement rather than absorb it.
    assert study._score_e1({"cell": cell})["passed"] is False

    cell["requests"][0]["decode_attribution"]["queue_ps"] = 200
    cell["requests"][0]["decode_attribution"]["kernel_ps"] = 60
    cell["requests"][0]["decode_attribution"]["collective_ps"] = 40
    assert study._score_e1({"cell": cell})["passed"] is True


def test_composition_keys_separate_steps_by_what_they_schedule():
    study = _study_module()
    cell = _cell_payload()

    keys = [study._composition_key(step) for step in cell["steps"]]

    assert keys[0] == (("p0", "prefill", 8, 8),)
    assert keys[1] == (("p1", "prefill", 8, 8),)
    assert len(set(keys)) == 3
    assert len(study._steps_by_composition(cell)) == 3
