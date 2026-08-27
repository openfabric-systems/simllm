from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "examples/hopper_kernel_cycle_candidate_v1"
SCRIPT = SCRIPT_DIR / "recover_deepseek_repeat.py"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("hopper_deepseek_repeat_recovery", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _base_fixture(tmp_path: Path) -> Path:
    profile = {
        "model": MODULE.EXPECTED_MODEL,
        "model_key": "deepseek-v3",
        "tensor_parallel_size": 1,
        "mode": "graph",
        "shape_set": "deepseek",
        "deepseek_suite": "base",
        "reduced_layers": 4,
        "phase": "profile",
        "model_config": {
            "requested_revision": MODULE.EXPECTED_REVISION,
            "resolved_revision": MODULE.EXPECTED_REVISION,
            "config_sha256": MODULE.EXPECTED_CONFIG_SHA256,
            "effective_num_hidden_layers": 4,
        },
        "cases": [
            {
                "cell": cell,
                "batch_size": batch_size,
                "input_len": input_len,
            }
            for cell, batch_size, input_len in (
                ("prefill_r16_l1024_t16384", 16, 1024),
                ("prefill_r8_l2048_t16384", 8, 2048),
                ("prefill_r4_l4096_t16384", 4, 4096),
            )
        ],
    }
    (tmp_path / "profile.json").write_text(
        json.dumps(profile) + "\n", encoding="utf-8", newline="\n"
    )
    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis/ordered-kernels.csv").write_text(
        "cell,duration_ns,is_collective\n"
        "prefill_r16_l1024_t16384,11,False\n"
        "prefill_r16_l1024_t16384,5,True\n"
        "prefill_r8_l2048_t16384,13,False\n"
        "prefill_r4_l4096_t16384,17,False\n",
        encoding="utf-8",
        newline="\n",
    )
    for relative in (
        "profile.sqlite",
        "profile.nsys-rep",
        "harness_sha256.txt",
        "sha256.txt",
        "weight_files.txt",
    ):
        (tmp_path / relative).write_bytes(b"")
    (tmp_path / "analysis_status.txt").write_text(
        "1\n", encoding="utf-8", newline="\n"
    )
    return tmp_path


def test_base_recovery_keeps_each_registered_prefill_cell_separate(
    tmp_path: Path,
) -> None:
    result = MODULE.recover_repeat(_base_fixture(tmp_path), "base")

    services = {row["cell"]: row for row in result["services"]}
    assert services["prefill_r16_l1024_t16384"]["measured_service_ps"] == 11_000
    assert services["prefill_r16_l1024_t16384"]["collective_service_ps"] == 5_000
    assert services["prefill_r8_l2048_t16384"]["measured_service_ps"] == 13_000
    assert services["prefill_r4_l4096_t16384"]["measured_service_ps"] == 17_000
    assert result["distribution_propagation"] == "DEFERRED_TO_COMP-74"
    assert result["original_compact_analysis"]["status"] == "BLOCKED"


def test_base_recovery_rejects_a_changed_shape(tmp_path: Path) -> None:
    run_dir = _base_fixture(tmp_path)
    profile = json.loads((run_dir / "profile.json").read_text(encoding="utf-8"))
    profile["cases"][0]["input_len"] = 512
    (run_dir / "profile.json").write_text(
        json.dumps(profile) + "\n", encoding="utf-8", newline="\n"
    )

    with pytest.raises(ValueError, match="shape changed"):
        MODULE.recover_repeat(run_dir, "base")
