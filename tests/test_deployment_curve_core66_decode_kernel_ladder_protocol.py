from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/deployment_curve_v1"


def _score_module():
    path = STUDY / "core66_decode_kernel_ladder_score.py"
    spec = importlib.util.spec_from_file_location("core66_kladder_score", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expectations() -> dict:
    return json.loads(
        (STUDY / "core66_decode_kernel_ladder_expectations.json").read_text(
            encoding="utf-8"
        )
    )


def _capture() -> dict:
    scheduler = {
        "is_exact_decode": True,
        "num_requests": 32,
        "request_ids": [f"core66-kladder-{index:02d}" for index in range(32)],
        "num_computed_tokens": [2000] * 32,
        "num_output_tokens": [1] * 32,
    }
    return {
        "evidence": "confirmatory, dummy weights",
        "model": "deepseek-ai/DeepSeek-V3",
        "revision": "frozen-revision",
        "framework": {
            "name": "vllm",
            "version": "0.27.1",
            "torch": "2.8.0+cu129",
            "cuda": "12.9",
            "python": "3.11.13",
            "machine": "aarch64",
        },
        "shape": {
            "batch_size": 32,
            "kv_tokens_per_request": 2000,
            "layers": 4,
            "dense_layers": 3,
            "moe_layers": 1,
            "first_k_dense_replace": 3,
        },
        "installed": {
            "layer_mlp_types": [
                "DeepseekV3MLP",
                "DeepseekV3MLP",
                "DeepseekV3MLP",
                "DeepseekV3MoE",
            ]
        },
        "marker": {
            "active": {
                "enabled": True,
                "batch_size": 32,
                "remote_kv_tokens": 2000,
            },
            "scheduler": scheduler,
        },
        "measurement": {
            "capture_output_shape": {
                "shape": [32, 7168],
                "dtype": "torch.bfloat16",
            },
            "subtraction_method": "frozen device-event subtraction",
        },
    }


def _summary(service_ps: int, repeat: int) -> dict:
    return {
        "repeat": repeat,
        "raw_service_ps": service_ps + 20_000_000,
        "subtracted_service_ps": service_ps + 10_000_000,
        "cold_native_service_ps": service_ps,
        "resident_native_service_ps": service_ps - 10_000_000,
        "cold_native_union_ps": service_ps,
        "resident_native_union_ps": service_ps - 10_000_000,
        "kernel_count_values": [100],
        "expected_kernel_count": 100,
        "native_stream_ids": [7],
        "native_stream_count": 1,
        "cold_segment_service_ps": {},
        "resident_segment_service_ps": {},
        "has_dense_gate_up": True,
        "has_dense_down": True,
        "has_moe": True,
        "has_mla": True,
        "has_embedding": True,
    }


def _score(service_ps: int, *, process_exit_zero: bool = True) -> dict:
    module = _score_module()
    return module.score(
        _capture(),
        _expectations(),
        [_summary(service_ps, 50), _summary(service_ps + 1_000_000, 20)],
        process_exit_zero=process_exit_zero,
        weights_empty_and_identical=True,
        preservation=[{"pass": True}] * 7,
        commit="confirmatory-commit",
    )


def test_score_passes_fatal_and_behavioral_evidence_separately() -> None:
    result = _score(1_300_000_000)

    assert result["status"] == "PASS"
    evidence = result["evidence_classes"]
    assert len(evidence["fatal_structural_guards"]) == 8
    assert all(evidence["fatal_structural_guards"].values())
    assert evidence["behavioral_relations"] == {
        "composition_residual": True,
        "repeat_stability": True,
        "physical_bounds": True,
    }
    assert evidence["preservation_locks"]["pass"] is True


def test_fatal_failure_voids_instead_of_becoming_a_score() -> None:
    result = _score(1_300_000_000, process_exit_zero=False)

    assert result["status"] == "VOID"
    assert result["evidence_classes"]["behavioral_relations"][
        "composition_residual"
    ]
    assert not result["evidence_classes"]["fatal_structural_guards"][
        "process_exit_zero"
    ]


def test_behavioral_miss_fails_only_after_fatal_guards_pass() -> None:
    result = _score(1_500_000_000)

    assert result["status"] == "FAIL"
    assert all(result["evidence_classes"]["fatal_structural_guards"].values())
    assert not result["evidence_classes"]["behavioral_relations"][
        "composition_residual"
    ]


def test_native_interval_union_counts_overlap_once() -> None:
    module = _score_module()
    rows = [
        {"start_ns": 0, "end_ns": 10},
        {"start_ns": 5, "end_ns": 12},
        {"start_ns": 20, "end_ns": 25},
    ]

    assert module.interval_union_ps(rows) == 17_000


def test_confirmation_launcher_uses_only_explicit_site_paths() -> None:
    launcher = (
        STUDY / "core66_decode_kernel_ladder_confirm.sbatch"
    ).read_text(encoding="utf-8")

    assert "KLADDER_MODEL_SNAPSHOT" in launcher
    assert "KLADDER_OUTPUT_ROOT" in launcher
    assert "KLADDER_REPO_ROOT" in launcher
    assert "KLADDER_VENV_DIR" in launcher
    assert "/data3/" not in launcher
    assert "/home/" not in launcher
    assert "~/" not in launcher
