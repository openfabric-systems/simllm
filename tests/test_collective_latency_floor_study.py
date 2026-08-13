"""Regression checks for collective-floor evidence classification."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
STUDY_PATH = REPOSITORY / "examples/collective_latency_floor_v1/run_study.py"


def _study_module():
    spec = importlib.util.spec_from_file_location(
        "collective_latency_floor_v1",
        STUDY_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_c3_and_c4_are_guards_and_never_scored() -> None:
    study = _study_module()
    registered = {
        name: {"passed": True} for name in ("C1", "C2", "C3", "C4")
    }

    scored, guards = study._partition_registered_relations(registered)

    assert set(guards) == {"C3", "C4"}
    assert set(scored) == {"C1", "C2"}
    assert set(guards).isdisjoint(scored)
    assert study.EXPECTED_SCORED_FAMILIES == len(scored) == 2
