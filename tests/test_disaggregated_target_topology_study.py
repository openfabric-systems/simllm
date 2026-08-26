import builtins
import importlib.util
import json
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = (
    ROOT / "examples" / "disaggregated_target_topology_v1" / "run_study.py"
)
EXPECTATIONS_PATH = STUDY_PATH.with_name("expectations.json")
RESULTS_PATH = STUDY_PATH.with_name("results.json")


def _study():
    spec = importlib.util.spec_from_file_location("place5_topology_study", STUDY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_place5_study_matches_the_frozen_structural_contract(tmp_path):
    study = _study()
    frozen = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    observation = {
        "cells": {
            cell["label"]: study._run_cell(tmp_path, cell)
            for cell in frozen["cells"]
        },
        "expectations_commit": study.EXPECTATIONS_COMMIT,
        "implementation_commit": "test-implementation",
    }

    analysis = study.analyze_observation(observation, frozen)

    assert analysis["status"] == "PASS"
    assert analysis["findings"] == []
    assert analysis["evidence"] == {
        "fatal_guard_count": 9,
        "scored_behavioral_families": 0,
    }


def test_place5_study_voids_a_goal_path_disagreement(tmp_path):
    study = _study()
    frozen = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    observation = {
        "cells": {
            cell["label"]: study._run_cell(tmp_path, cell)
            for cell in frozen["cells"]
        },
        "expectations_commit": study.EXPECTATIONS_COMMIT,
        "implementation_commit": "test-implementation",
    }
    observation["cells"]["target"]["goal"]["path_propagation_ps"] = [3_000_000]

    analysis = study.analyze_observation(observation, frozen)

    assert analysis["status"] == "VOID"
    assert "target: GOAL path propagation" in analysis["findings"]


def test_place5_recorded_result_closes_only_the_structural_scope():
    result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    assert result["status"] == "PASS"
    assert result["evidence"] == {
        "fatal_guard_classes": 9,
        "findings": [],
        "scored_behavioral_families": 0,
    }
    target = result["cells"]["target"]
    assert target["counts"]["rank_count"] == 448
    assert target["counts"]["gpu_count"] == 448
    assert target["counts"]["nic_count"] == 448
    assert target["counts"]["prefill_ranks"] == 128
    assert target["counts"]["decode_ranks"] == 320
    assert target["reachable_endpoints"] == 448
    assert target["goal"] == {
        "message_count": 448,
        "path_bottleneck_rate_bps": 400_000_000_000,
        "path_link_count": 4,
        "path_propagation_ps": 4_000_000,
        "payload_bytes": 1_835_008,
    }
    assert target["placement_disabled_identity"] == {
        "byte_identical_to_enabled": True,
        "bytes": 2_639_042,
        "sha256": (
            "48029d871293762007ab33082d59a7b5a4efb22583394e718c97e733717fd709"
        ),
    }


def test_place5_study_uses_posix_rendering_for_windows_paths():
    study = _study()

    assert study.render_cli_path(PureWindowsPath("C:/runs/place5")) == (
        "C:/runs/place5"
    )


def test_place5_study_imports_without_unix_only_or_python_311_modules(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"resource", "tomllib"}:
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    study = _study()
    assert study.RESULT_SCHEMA == "simllm-disaggregated-target-topology-result-v1"
