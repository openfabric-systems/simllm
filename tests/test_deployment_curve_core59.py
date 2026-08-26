from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from simllm.compute import ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.placement import (
    RankMapper,
    SglangPoolArrangement,
    sglang_disaggregated_manifests,
)
from simllm.traffic import plan_step_locality

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples/deployment_curve_v1"


def _module():
    name = "deployment_curve_core59_role_mechanisms"
    sys.path.insert(0, str(STUDY_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            name,
            STUDY_DIR / "core59_role_mechanisms.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(STUDY_DIR))


def _json(name: str):
    return json.loads((STUDY_DIR / name).read_text(encoding="utf-8"))


def _calibration_rows():
    visible = {"sglang_prefill_1k", "sglang_decode_standard"}
    return {
        row["anchor_id"]: row
        for row in _json("scored_expectations.json")["pre_tuning_predicted_bands"]
        if row["anchor_id"] in visible
    }


def test_freeze_has_no_fit_knob_and_locks_the_first_scored_run():
    mechanisms = _module()
    expectations = _json("core59_expectations.json")

    mechanisms.validate_expectations(expectations)
    history = mechanisms.verify_historical_refutation(
        expectations,
        REPOSITORY_ROOT,
    )

    assert expectations["constants"]["new_free_constants"] == []
    assert expectations["constants"]["tunable"] == []
    assert expectations["shared_residual_disposition"] == {
        "historical_id": "intra_node_collective_surcharge_ps",
        "next_run_application_count": 0,
        "next_run_selected_ps": None,
        "reason": (
            "The shared positive additive term is not role-identifiable. It "
            "remains in the immutable first-run artifacts and is absent from "
            "the CORE-59 projection."
        ),
    }
    assert history["status"] == "PASS"
    assert history["first_scored_run_mutated"] is False
    assert len(history["checked_artifacts"]) == 9


def test_existing_traffic_and_placement_reconstruct_the_prefill_byte_ledger():
    expectations = _json("core59_expectations.json")
    mechanism = expectations["mechanisms"][0]
    gate = mechanism["shape_gate"]
    traffic = mechanism["traffic_arithmetic"]
    dims = ModelDims(
        num_layers=gate["moe_layers"],
        hidden_size=gate["hidden_size"],
        intermediate_size=18_432,
        num_heads=128,
        num_kv_heads=128,
        head_size=192,
        vocab_size=129_280,
        dtype_bytes=gate["vector_bytes_per_element"],
        num_experts=256,
        top_k=gate["top_k"],
        moe_intermediate_size=2_048,
        local_num_experts=32,
    )
    arrangement = SglangPoolArrangement(
        enable_data_parallel_attention=True,
        attention_data_parallel_size=32,
        dense_data_parallel_size=32,
        expert_parallel_size=32,
    )
    placement = sglang_disaggregated_manifests(
        prefill_nodes=4,
        decode_nodes=1,
        gpus_per_node=8,
        prefill_arrangement=arrangement,
        decode_arrangement=SglangPoolArrangement.identity(),
        framework_version="0.5.19.dev345+gbfeae4e79",
    ).placement
    record = StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                "prefill-calibration",
                RequestPhase.PREFILL,
                num_new_tokens=gate["new_tokens_per_rank"],
                context_length=gate["new_tokens_per_rank"],
            )
        ],
    )

    plan = plan_step_locality(
        record,
        dims,
        (0,),
        ep_ranks=tuple(range(32)),
        rank_mapper=RankMapper(placement),
    )

    assert len(plan.phases) == traffic["application_count"] == 116
    assert {phase.nvlink_bytes for phase in plan.phases} == {traffic["local_bytes_per_phase"]}
    assert {phase.fabric_bytes for phase in plan.phases} == {traffic["fabric_bytes_per_phase"]}
    assert {phase.nvlink_service_ps for phase in plan.phases} == {
        mechanism["physical_service"]["arms"][0]["local_phase_service_ps"]
    }
    assert plan.total_directed_bytes == (
        traffic["total_directed_bytes_per_phase"] * traffic["application_count"]
    )
    assert plan.fabric_bytes + plan.nvlink_bytes == plan.total_directed_bytes


def test_role_and_shape_gates_move_only_prefill():
    mechanisms = _module()
    expectations = _json("core59_expectations.json")
    shapes = mechanisms.calibration_shapes(expectations)
    rows = _calibration_rows()

    prefill = rows["sglang_prefill_1k"]
    prefill_baseline = Fraction(
        prefill["per_node_tokens"] * mechanisms.PS_PER_SECOND,
        prefill["candidate_service_ps"],
    )
    prefill_updated = mechanisms.prediction_at_role_shape_service(
        prefill,
        expectations,
        shapes["sglang_prefill_1k"],
    )
    decode = rows["sglang_decode_standard"]
    decode_baseline = Fraction(
        decode["per_node_tokens"] * mechanisms.PS_PER_SECOND,
        decode["candidate_service_ps"],
    )
    decode_updated = mechanisms.prediction_at_role_shape_service(
        decode,
        expectations,
        shapes["sglang_decode_standard"],
    )

    assert prefill_updated < prefill_baseline
    assert (
        mechanisms.mechanism_service_ps(
            expectations,
            shapes["sglang_prefill_1k"],
        )
        == 3_320_872_128_000
    )
    assert decode_updated == decode_baseline
    assert (
        mechanisms.mechanism_service_ps(
            expectations,
            shapes["sglang_decode_standard"],
        )
        == 0
    )


def test_unknown_shapes_fail_closed_without_interpolation():
    mechanisms = _module()
    expectations = _json("core59_expectations.json")
    shape = mechanisms.calibration_shapes(expectations)["sglang_prefill_1k"]

    with pytest.raises(ValueError, match="outside the frozen CORE-59 gate"):
        mechanisms.mechanism_service_ps(
            expectations,
            {**shape, "new_tokens_per_rank": 16_383},
        )


def test_calibration_fit_reads_only_the_frozen_visible_ids():
    mechanisms = _module()
    expectations = _json("core59_expectations.json")
    result = mechanisms.fit_calibration_only(
        _json("expectations.json"),
        _json("scored_expectations.json"),
        expectations,
    )

    assert result["status"] == "FROZEN"
    assert result["classification"] == "CALIBRATION_ONLY_NOT_SCORED"
    assert result["accessed_anchor_ids"] == [
        "sglang_prefill_1k",
        "sglang_decode_standard",
    ]
    assert result["forbidden_anchor_ids_accessed"] == []
    assert result["held_out_numeric_values_accessed"] is False
    assert result["held_out_score_performed"] is False
    assert result["fitted_parameters"] == []
    movements = {
        row["anchor_id"]: row["signed_movement"]["direction"] for row in result["calibration_rows"]
    }
    assert movements == {
        "sglang_prefill_1k": "decrease",
        "sglang_decode_standard": "unchanged",
    }


def test_published_calibration_result_matches_the_frozen_projection():
    mechanisms = _module()
    expectations = _json("core59_expectations.json")
    computed = mechanisms.fit_calibration_only(
        _json("expectations.json"),
        _json("scored_expectations.json"),
        expectations,
    )
    published = _json("core59_calibration_result.json")
    published_rows = []
    for row in published["calibration_rows"]:
        selected = dict(row)
        selected.pop("decimal_summary")
        published_rows.append(selected)

    assert published_rows == computed["calibration_rows"]
    for name in (
        "schema",
        "status",
        "classification",
        "accessed_anchor_ids",
        "forbidden_anchor_ids_accessed",
        "held_out_numeric_values_accessed",
        "held_out_score_performed",
        "fitted_parameters",
        "shared_collective_surcharge_application_count",
    ):
        assert published[name] == computed[name]
    expectations_bytes = (STUDY_DIR / "core59_expectations.json").read_bytes()
    assert published["expectations_sha256"] == hashlib.sha256(expectations_bytes).hexdigest()
    assert published["historical_refutation_lock"] == {
        "checked_artifact_count": 9,
        "first_scored_run_mutated": False,
        "status": "PASS",
    }
