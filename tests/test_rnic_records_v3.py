from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from simllm.backends import rnic_session_config_from_json

STUDY_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "rnic_records_v3"
    / "run_study.py"
)
SPEC = importlib.util.spec_from_file_location("rnic_records_v3_study", STUDY_PATH)
assert SPEC is not None
assert SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def _bases() -> dict[str, dict[str, object]]:
    return {
        "v2": study._effective_v2(),
        "v3_host": study._effective_v3("host_cpu_driver"),
        "v3_proxy": study._effective_v3("cpu_proxy"),
        "v3_gpu": study._effective_v3("gpu_initiated"),
    }


def _thaw(value: Any) -> Any:
    if isinstance(value, dict) or hasattr(value, "items"):
        return {name: _thaw(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@pytest.mark.parametrize("fixture", tuple(study.EFFECTIVE_HASHES))
def test_native_effective_hardware_fixtures_are_ingested_and_frozen(fixture: str):
    effective = _bases()[fixture]
    assert study._digest(effective) == study.EFFECTIVE_HASHES[fixture]

    record = rnic_session_config_from_json(study._config(effective))

    assert _thaw(record.effective_hardware) == effective
    with pytest.raises(TypeError):
        record.effective_hardware["schema"] = "changed"
    with pytest.raises(AttributeError):
        record.effective_hardware["host_memory"]["allocations"].append({})


@pytest.mark.parametrize(
    ("mutation_id", "base_name"),
    [(row[0], row[1]) for row in study.MUTATIONS],
    ids=[row[0] for row in study.MUTATIONS],
)
def test_frozen_native_rejection_corpus_is_rejected_by_python(
    mutation_id: str,
    base_name: str,
):
    bases = _bases()
    effective, digest = study._mutated_effective(mutation_id, base_name, bases)
    config = study._config(effective)
    config["hardware_config_sha256"] = digest

    with pytest.raises((TypeError, ValueError)):
        rnic_session_config_from_json(config)


def test_non_string_schema_is_rejected_without_dispatch_failure():
    effective = study._effective_v2()
    effective["schema"] = []
    config = study._config(effective)

    with pytest.raises((TypeError, ValueError)):
        rnic_session_config_from_json(config)
