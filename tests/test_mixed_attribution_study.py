"""Study-entrypoint regression checks for the mixed attribution study."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
STUDY_PATH = REPOSITORY / "examples/mixed_attribution_v1/run_study.py"


def _study_module():
    spec = importlib.util.spec_from_file_location("mixed_attribution_v1", STUDY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_only_validates_the_registry_and_writes_nothing(tmp_path):
    output = tmp_path / "must-not-exist"

    result = subprocess.run(
        [sys.executable, str(STUDY_PATH), "--run-dir", str(output), "--check-only"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "produced no artifacts" in result.stdout
    assert not output.exists()


def test_frozen_constants_stay_self_consistent():
    study = _study_module()

    assert study.VECTOR_BYTES == 2_048
    assert study.NVLINK_RATE_BPS == 2 * study.NVLINK_HALF_RATE_BPS
    assert study.MIXED_NVLINK_TTFT_PS == 120_000
    assert study.MIXED_NVLINK_TTFT_HALF_PS == 240_000
    assert study.MIXED_NVLINK_DELTA_PS == 120_000
    assert set(study.CELLS) == {
        "all-remote-450",
        "all-local-450",
        "all-local-225",
        "mixed-450",
        "mixed-225",
    }
    study._check_frozen_registry()


def test_the_declared_expert_layouts_partition_every_layer():
    study = _study_module()
    projection = study._routed_projection()

    for layout in (study.UNIFORM_LAYOUT, study.NODE_LOCAL_LAYOUT):
        per_layer = study._expert_layout(projection, layout)
        assert set(per_layer) == set(range(study.NUM_LAYERS))
        for owners in per_layer.values():
            experts = [expert for rank in study.EP_RANKS for expert in owners[rank]]
            assert sorted(experts) == list(range(study.NUM_EXPERTS))
            assert all(len(owners[rank]) == 8 for rank in study.EP_RANKS)


def test_the_node_local_layout_puts_even_layer_experts_on_one_node():
    study = _study_module()
    projection = study._routed_projection()
    token = projection.by_request_id(study.REQUEST_ID).prefill_tokens[0]
    per_layer = study._expert_layout(projection, study.NODE_LOCAL_LAYOUT)

    for layer in range(0, study.NUM_LAYERS, 2):
        selected = set(token.layers[layer].expert_ids)
        #: ranks 0 and 1 share the first node of the AABB placement
        assert selected <= set(per_layer[layer][0]) | set(per_layer[layer][1])
    for layer in range(1, study.NUM_LAYERS, 2):
        assert per_layer[layer] == {
            rank: list(range(rank * 8, rank * 8 + 8)) for rank in study.EP_RANKS
        }


def test_the_step_recomputation_resolves_owners_from_the_published_tuples():
    study = _study_module()
    locality = {
        "composed_phase_service_ps": [1_000, 5_000, 9_000, 12_000, 4_000],
        "fabric_phase_service_ps": [0, 0, 9_000, 4_000, 3_000],
        "local_phase_service_ps": [1_000, 5_000, 2_000, 12_000, 0],
        "base_phase_latency_ps": [0, 0, 0, 0, 1_000],
        "local_phase_medium": [
            "gpu-compute",
            "nvlink",
            "nvlink",
            "nvlink",
            "nvlink",
        ],
    }

    recomputed = study._recompute_step(locality)

    assert recomputed["owners"] == ["kernel", "nvlink", "fabric", "nvlink", "fabric"]
    assert recomputed["kernel_ps"] == 1_000
    assert recomputed["nvlink_ps"] == 17_000
    assert recomputed["fabric_ps"] == 12_000
    assert recomputed["collective_base_ps"] == 1_000
    assert recomputed["masked_nvlink_ps"] == 2_000
    assert recomputed["masked_fabric_ps"] == 4_000
    assert recomputed["total_ps"] == sum(locality["composed_phase_service_ps"])
