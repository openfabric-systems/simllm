"""Study-entrypoint regression checks for the composed SGLang deployment study.

None of these import SGLang or torch. They exercise the frozen constants, the
deployment resolution, the independent standard-library recomputation of the
medium partition, and the provenance guard that keeps another framework's
capture out of an SGLang result.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
STUDY_PATH = REPOSITORY / "examples/sglang_composed_deployment_v1/run_study.py"


def _study_module():
    spec = importlib.util.spec_from_file_location(
        "sglang_composed_deployment_v1", STUDY_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def study():
    return _study_module()


def _synthetic_cell() -> dict:
    """Two steps, one request, one compute artifact and one fabric artifact.

    Artifact 0 is compute owned: local 60, no fabric, no base. Artifact 1 is a
    collective whose fabric service of 30 beats its zero local service and
    which carries 10 of base, so each step conserves as 60 + 30 + 10 = 100.
    """

    def step(index: int) -> dict:
        return {
            "step_index": index,
            "virtual_time_ps": 100 * index,
            "step_latency_ps": 100,
            "completed_at_ps": 100 * (index + 1),
            "scheduled": [
                {
                    "request_id": "p0",
                    "phase": "prefill" if index == 0 else "decode",
                    "num_new_tokens": 8 if index == 0 else 1,
                    "context_length": 8 + index,
                    "num_cached_tokens": 0,
                }
            ],
            "composed": [60, 40],
            "fabric": [0, 30],
            "local": [60, 0],
            "base": [0, 10],
            "medium": "gn",
        }

    media = {
        "queue_ps": 0,
        "kernel_ps": 60,
        "nvlink_ps": 0,
        "fabric_ps": 30,
        "co_critical_ps": 0,
        "collective_base_ps": 10,
        "control_ps": 0,
        "total_ps": 100,
    }
    return {
        "steps": [step(0), step(1)],
        "requests": [
            {
                "request_id": "p0",
                "arrival_ps": 0,
                "token_count": 2,
                "first_token_at_ps": 100,
                "last_token_at_ps": 200,
                "ttft_ps": 100,
                "tpot_numerator": 100,
                "tpot_denominator": 1,
                "ttft_media": dict(media),
                "decode_media": dict(media),
            }
        ],
    }


def test_frozen_registry_holds(study) -> None:
    study._check_frozen_registry()
    assert len(study.CELLS) == 18
    assert len(study.intra_cells()) == 6
    assert len(study.cross_cells()) == 12
    assert study.INTRA_ARTIFACTS == 408
    assert study.CROSS_ARTIFACTS == 72


def test_arm_constants_match_the_closed_form(study) -> None:
    assert study.arm_base_latency_ps("intra-off-ideal") == 0
    assert study.arm_base_latency_ps("intra-lower-ideal") == 0
    assert study.arm_base_latency_ps("intra-upper-turing") == 30_128_029
    assert study.arm_base_latency_ps("cross400-off-ideal") == 0
    assert study.arm_base_latency_ps("cross400-lower-ideal") == 30_128_029
    assert study.arm_base_latency_ps("cross100-upper-turing") == 49_487_789
    assert study.collective_count("intra-off-ideal") == 72
    assert study.collective_count("cross400-off-ideal") == 48


def test_deployments_avoid_the_back44_shape(study) -> None:
    intra = study._deployment("intra-upper-ideal")
    assert intra["tp_ranks"] == intra["ep_ranks"]
    assert len(set(intra["hostnames"])) == 1
    cross = study._deployment("cross100-lower-turing")
    assert cross["tp_ranks"] == (0,)
    assert len(set(cross["hostnames"])) == 8
    assert cross["link_bps"] == 100_000_000_000


def test_families_are_ordered(study) -> None:
    arms = study._families("arm")
    assert len(arms) == 6
    for names in arms.values():
        assert tuple(study.CELLS[name][2] for name in names) == study.ARMS
    hosts = study._families("host")
    assert len(hosts) == 9
    for names in hosts.values():
        assert tuple(study.CELLS[name][3] for name in names) == ("ideal", "turing")


def test_step_media_partitions_by_owner(study) -> None:
    cell = _synthetic_cell()
    media = study._step_media(cell["steps"][0])
    assert media == {
        "kernel_ps": 60,
        "nvlink_ps": 0,
        "fabric_ps": 30,
        "co_critical_ps": 0,
        "collective_base_ps": 10,
    }


def test_step_media_calls_a_tie_co_critical(study) -> None:
    step = {
        "composed": [30],
        "fabric": [20],
        "local": [20],
        "base": [10],
        "medium": "n",
    }
    assert study._step_media(step)["co_critical_ps"] == 20


def test_step_media_refuses_a_broken_composition(study) -> None:
    step = {"composed": [99], "fabric": [20], "local": [0], "base": [10], "medium": "n"}
    with pytest.raises(AssertionError):
        study._step_media(step)


def test_independent_recomputation_agrees_with_a_conserving_cell(study) -> None:
    result = study._score_e1({"synthetic": _synthetic_cell()})
    assert result["passed"]
    assert result["compared"] == 1


def test_independent_recomputation_catches_a_misassigned_medium(study) -> None:
    cell = _synthetic_cell()
    cell["requests"][0]["ttft_media"]["fabric_ps"] = 0
    cell["requests"][0]["ttft_media"]["nvlink_ps"] = 30
    result = study._score_e1({"synthetic": cell})
    assert not result["passed"]
    fields = result["failures"][0]["fields"]
    assert "ttft_media.fabric_ps" in fields
    assert "ttft_media.nvlink_ps" in fields


def test_provenance_guard_rejects_another_framework(study, tmp_path: Path) -> None:
    header = {
        "row_type": "header",
        "schema": "simllm-preplay-trace-v2",
        "provenance": {
            "schema": "simllm-preplay-trace-v2",
            "framework": "vllm",
            "routing_source": "observed-dispatch",
            "model_id": study.MODEL_ID,
            "model_revision": study.MODEL_REVISION,
            "expert_count": 32,
            "top_k": 8,
            "observed_source": study.SGLANG_PINNED_COMMIT,
            "moe_layer_indices": list(range(24)),
        },
    }
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(header) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        study.check_trace_provenance(path)


def test_provenance_guard_accepts_the_pinned_capture(study, tmp_path: Path) -> None:
    header = {
        "row_type": "header",
        "schema": "simllm-preplay-trace-v2",
        "provenance": {
            "schema": "simllm-preplay-trace-v2",
            "framework": "sglang",
            "routing_source": "observed-dispatch",
            "model_id": study.MODEL_ID,
            "model_revision": study.MODEL_REVISION,
            "expert_count": 32,
            "top_k": 8,
            "observed_source": study.SGLANG_PINNED_COMMIT,
            "moe_layer_indices": list(range(24)),
        },
    }
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(header) + "\n", encoding="utf-8")
    assert study.check_trace_provenance(path)["framework"] == "sglang"


def test_the_largest_representable_step_sits_on_the_profile_boundary(study) -> None:
    assert study.MAX_CRITICAL_ENDPOINT_BYTES == study.PROFILE_ENDPOINT_MAX_BYTES
    assert study.MAX_CRITICAL_ENDPOINT_BYTES == 458_752
