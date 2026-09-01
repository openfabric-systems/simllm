"""Lock the frozen TRAF-81 measurement and scoring protocol."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/collective_floor_extrapolation_v1"
RUNNER = STUDY / "score_expectations.py"
FREEZE = json.loads((STUDY / "expectations.json").read_text(encoding="utf-8"))


def _load_runner():
    name = "collective_floor_extrapolation_score"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCORE = _load_runner()


def _synthetic_measurement(ranks: int) -> dict:
    rank4_scale = {2: 0.82, 4: 1.0, 8: 3.5, 16: 3.75}[ranks]
    measurements = []
    for operation in ("all_gather", "reduce_scatter"):
        floor_ps = 5_000_000 if operation == "all_gather" else 6_000_000
        slope_ps = 20 if operation == "all_gather" else 24
        for index, message_bytes in enumerate(FREEZE["byte_grid"]):
            jitter = 1.0 + 0.002 * ((index % 5) - 2)
            latency_ps = round(
                (floor_ps + slope_ps * message_bytes) * rank4_scale * jitter
            )
            median_us = latency_ps / 1_000_000
            measurements.append(
                {
                    "operation": operation,
                    "operation_buffer_bytes": message_bytes,
                    "median_us": median_us,
                    "max_rank_mismatches": 0,
                    "samples_us": [median_us] * 31,
                }
            )
    return {
        "schema": "simllm-collective-floor-measurement-v1",
        "study": "collective_floor_extrapolation_v1",
        "world": ranks,
        "tasks_per_node": min(ranks, 4),
        "visible_device_count_rank0": min(ranks, 4),
        "nccl_version": 23102,
        "rank0_device_name": "NVIDIA A100-SXM4-80GB",
        "warmup_iterations": 10,
        "timed_repetitions": 31,
        "sample_reduction": "maximum-over-ranks",
        "aggregation": "observed-median",
        "measurements": measurements,
    }


def _load_documents(tmp_path: Path) -> tuple[dict[int, dict], list]:
    documents = {}
    rows = []
    for ranks in (2, 4, 8, 16):
        path = tmp_path / f"rank-{ranks}.json"
        path.write_text(
            json.dumps(_synthetic_measurement(ranks)) + "\n", encoding="utf-8"
        )
        document, loaded = SCORE.load_measurement(path, FREEZE)
        documents[ranks] = document
        rows.extend(loaded)
    return documents, rows


def test_freeze_locks_dense_grid_fit_and_shape_only_scope() -> None:
    assert FREEZE["fit"]["training_ranks"] == [2, 4]
    assert FREEZE["fit"]["donor_rank"] == 4
    assert FREEZE["fit"]["maximum_regimes"] == 3
    assert FREEZE["fit"]["minimum_cells_per_regime"] == 2
    assert FREEZE["harness"]["warmup_iterations"] == 10
    assert FREEZE["harness"]["timed_repetitions"] == 31
    assert len(FREEZE["byte_grid"]) == 24
    dense = [value for value in FREEZE["byte_grid"] if 524_288 <= value <= 2_097_152]
    assert len(dense) == 9
    assert FREEZE["cross_architecture_rule"].startswith("shape-only")


def test_harness_source_matches_the_frozen_coordinate() -> None:
    source = (STUDY / "collective_lane.cu").read_text(encoding="utf-8")
    assert "constexpr int kWarmups = 10;" in source
    assert "constexpr int kSamples = 31;" in source
    assert "ncclAllGather" in source
    assert "ncclReduceScatter" in source
    assert "ncclMax" in source
    assert "first_target_timing_started" in source
    for message_bytes in FREEZE["byte_grid"]:
        assert str(message_bytes) in source


def test_nv4_guard_accepts_the_frozen_visible_submeshes() -> None:
    rank_two = "GPU0 X NV4 PHB\nGPU1 NV4 X PHB\n"
    rank_four = (
        "GPU0 X NV4 NV4 NV4 PHB\n"
        "GPU1 NV4 X NV4 NV4 PHB\n"
        "GPU2 NV4 NV4 X NV4 PHB\n"
        "GPU3 NV4 NV4 NV4 X PHB"
    )
    assert SCORE._nv4_rows(rank_two, 2) == 2
    assert SCORE._nv4_rows(rank_four, 4) == 4
    assert SCORE._nv4_rows(rank_two.replace("NV4", "SYS", 1), 2) == 1


@pytest.mark.parametrize(
    ("ranks", "nodes", "tasks_per_node", "memory"),
    [
        (2, 1, 2, "64G"),
        (4, 1, 4, "128G"),
        (8, 2, 4, "256G"),
        (16, 4, 4, "256G"),
    ],
)
def test_batch_cells_lock_rank_and_node_shape(
    ranks: int, nodes: int, tasks_per_node: int, memory: str
) -> None:
    text = (STUDY / f"measure_r{ranks}.sbatch").read_text(encoding="utf-8")
    assert f"#SBATCH --nodes={nodes}" in text
    assert f"#SBATCH --ntasks-per-node={tasks_per_node}" in text
    assert f"#SBATCH --mem={memory}" in text
    assert "#SBATCH --partition=a100-hourly" in text
    assert "#SBATCH --account=merlin" in text
    assert "#SBATCH --time=01:00:00" in text


def test_synthetic_shape_preserving_transfer_passes_scored_families(
    tmp_path: Path,
) -> None:
    documents, rows = _load_documents(tmp_path)
    result = SCORE.score_measurements(documents, rows, FREEZE)
    assert result["verdict"] == "FIT-SMALL-EXTRAPOLATE-WIDE HOLDS"
    assert result["scores"]["S1"]["held"]
    assert result["scores"]["S2"]["held"]
    assert result["scores"]["S3"]["held"]
    assert len(result["extrapolation_rows"]) == 96
    assert {row["ranks"] for row in result["fits"]} == {2, 4, 8, 16}


def test_missing_wide_cell_is_blocked_without_substitution(tmp_path: Path) -> None:
    documents, rows = _load_documents(tmp_path)
    documents.pop(16)
    rows = [row for row in rows if row.ranks != 16]
    result = SCORE.score_measurements(
        documents,
        rows,
        FREEZE,
        blocked={16: "campaign window expired while pending"},
    )
    assert result["verdict"] == "BLOCKED"
    assert result["blocked_cells"] == [
        {"ranks": 16, "reason": "campaign window expired while pending"}
    ]
    assert not result["scores"]["S3"]["evaluated"]
    assert all(row["ranks"] == 8 for row in result["extrapolation_rows"])


def test_physical_ceiling_is_a_fatal_input_guard(tmp_path: Path) -> None:
    document = _synthetic_measurement(2)
    row = document["measurements"][-1]
    row["median_us"] = 0.001
    row["samples_us"] = [0.001] * 31
    path = tmp_path / "impossible.json"
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    with pytest.raises(SCORE.StudyError, match="above ceiling"):
        SCORE.load_measurement(path, FREEZE)


def test_runner_imports_without_pythonpath() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, os.fspath(RUNNER), "--help"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--measurement" in completed.stdout
