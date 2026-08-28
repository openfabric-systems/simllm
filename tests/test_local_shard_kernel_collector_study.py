from __future__ import annotations

import importlib.util
from pathlib import Path

from simllm.calibration.canonical import canonical_bytes

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "local_shard_kernel_collector_v1"
RUNNER = STUDY / "run_study.py"

SPEC = importlib.util.spec_from_file_location("local_shard_kernel_collector_study", RUNNER)
assert SPEC is not None and SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_study_reproduces_the_committed_result_byte_for_byte() -> None:
    result = study.run_study()

    assert result["status"] == "PASS"
    assert result["scored_cell_count"] == 4
    assert result["passed_cell_count"] == 4
    assert all(result["relations"].values())
    assert canonical_bytes(result) == (STUDY / "result.json").read_bytes()
