"""Lock the frozen TRAF-81 measurement and scoring protocol."""

from __future__ import annotations

import hashlib
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
RESULT = json.loads((STUDY / "record.json").read_text(encoding="utf-8"))

RESULT_HASHES = {
    "measurements.csv": "25e072df4200caa65b18823f117679af83c4d507e66d227e96bd1454a0f4913d",
    "extrapolation.csv": "2467b920fdf79f263b7d28f58cb8f138b4c1126f08cee884637de39771341e46",
    "fits.csv": "798cc4f243d10fb4e2d648709f357a9cf3c409a6a18b9a8f171b1b2cb7c27f5a",
    "record.json": "501fc85d070bb7b7daf263a2bd81b01396642d64bc8d5b7468f5a4e67d3842a3",
}


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
    assert len(result["measurement_summaries"]) == 192
    assert len(result["physical_sanity"]) == 8
    assert all(row["held"] for row in result["physical_sanity"])
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


def test_compact_medians_reproduce_every_scored_quantity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents, rows = _load_documents(tmp_path)
    result = SCORE.score_measurements(documents, rows, FREEZE)
    output = tmp_path / "tracked"
    SCORE.write_outputs(result, output)
    monkeypatch.setattr(SCORE, "STUDY_ROOT", output)
    SCORE.check_tracked_outputs()


def test_measured_result_is_blocked_only_on_rank_sixteen() -> None:
    assert RESULT["verdict"] == "BLOCKED"
    assert RESULT["measured_ranks"] == [2, 4, 8]
    assert RESULT["blocked_cells"] == [
        {
            "ranks": 16,
            "reason": (
                "job 202442 remained PENDING for 120 minutes with "
                "QOSMaxGRESPerJob; gpu_hourly and gpu_daily cap one job at 8 "
                "GPUs; gpu_general caps one job at 4 GPUs and exposes only three "
                "nodes"
            ),
        }
    ]
    assert RESULT["guard_audit"]["status"] == "HELD"
    assert [guard["id"] for guard in RESULT["guard_audit"]["fatal_guards"]] == [
        f"FG-{number}" for number in range(1, 8)
    ]
    assert all(
        guard["held"] for guard in RESULT["guard_audit"]["fatal_guards"]
    )
    assert len(RESULT["measurement_summaries"]) == 144
    assert len(RESULT["extrapolation_rows"]) == 48
    assert len(RESULT["fits"]) == 16


def test_rank_eight_refutes_both_transfer_families() -> None:
    scores = RESULT["scores"]
    assert not scores["S1"]["held"]
    assert not scores["S2"]["held"]
    assert not scores["S3"]["evaluated"]
    assert not scores["S4"]["held"]

    s1 = {row["operation"]: row for row in scores["S1"]["curves"]}
    assert s1["all_gather"]["median"] == pytest.approx(0.6288907424244802)
    assert s1["all_gather"]["p95_nearest_rank"] == pytest.approx(
        1.5559280169063312
    )
    assert s1["reduce_scatter"]["median"] == pytest.approx(0.6111079100349189)
    assert s1["reduce_scatter"]["p95_nearest_rank"] == pytest.approx(
        1.4454434660994055
    )

    s2 = {row["operation"]: row for row in scores["S2"]["curves"]}
    assert s2["all_gather"]["median"] == pytest.approx(0.23015940697040227)
    assert s2["reduce_scatter"]["median"] == pytest.approx(0.4840110490665882)


def test_measured_rates_stay_below_physical_ceilings() -> None:
    assert len(RESULT["physical_sanity"]) == 6
    assert all(row["held"] for row in RESULT["physical_sanity"])
    by_cell = {
        (row["operation"], row["ranks"]): row
        for row in RESULT["physical_sanity"]
    }
    assert by_cell[("all_gather", 4)]["ceiling_fraction"] == pytest.approx(
        0.45134986347319933
    )
    assert by_cell[("reduce_scatter", 8)]["ceiling_fraction"] == pytest.approx(
        0.052875980085538274
    )


def test_tracked_result_artifacts_are_byte_locked_and_lf_only() -> None:
    for name, expected in RESULT_HASHES.items():
        payload = (STUDY / name).read_bytes()
        assert b"\r" not in payload
        assert hashlib.sha256(payload).hexdigest() == expected


def test_result_report_states_project_consequence_without_absolute_transfer() -> None:
    report = (STUDY / "RESULTS.md").read_text(encoding="utf-8")
    assert "formal verdict is **BLOCKED**" in report
    assert "TRAF-81 stays open only for the blocked rank-16 cell" in report
    assert "TRAF-76's rank-8" in report
    assert "donor transfers at expert-parallel widths 32 and 128" in report
    assert "does not calibrate an H200 absolute" in report
    assert "latency, bandwidth, floor or slope" in report
    assert "TRAF-85 and TRAF-86 remain available" in report
    assert chr(0x2014) not in report


def test_physical_ceiling_is_a_fatal_input_guard(tmp_path: Path) -> None:
    document = _synthetic_measurement(2)
    row = document["measurements"][-1]
    row["median_us"] = 0.001
    row["samples_us"] = [0.001] * 31
    path = tmp_path / "impossible.json"
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    with pytest.raises(SCORE.StudyError, match="above ceiling"):
        SCORE.load_measurement(path, FREEZE)


def test_runner_reconstructs_results_without_pythonpath() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, os.fspath(RUNNER), "--check"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "tracked medians reproduce every fitted and scored quantity" in (
        completed.stdout
    )
