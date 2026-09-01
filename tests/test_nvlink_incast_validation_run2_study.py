import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_incast_validation_v1"


def _load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, STUDY / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_campaign = _load(
    "nvlink_incast_validation_run2_campaign_test", "run_campaign_run2.py"
)
score_study = _load(
    "nvlink_incast_validation_run2_score_test", "score_study_run2.py"
)
plot_study = _load(
    "nvlink_incast_validation_run2_plot_test", "plot_study_run2.py"
)


def _frozen() -> dict[str, object]:
    return json.loads((STUDY / "expectations_run2.json").read_text(encoding="utf-8"))


def _synthetic_rows() -> list[dict[str, object]]:
    frozen = _frozen()
    rows = []
    predictions = {
        (row["degree"], row["size_bytes"]): row
        for row in frozen["simulation_arm"]["predictions"]
    }
    for point in run_campaign.campaign_points(frozen):
        size_bytes = point.payload_bytes * point.message_count
        degree = len(point.sources.split(","))
        prediction = predictions[(degree, size_bytes)]
        ledgers = [
            {
                "source": source,
                "destination": 0,
                "logical_bytes": size_bytes,
                "completion_us": prediction["completion_ps_by_source"][source - 1]
                / 1_000_000,
            }
            for source in range(1, degree + 1)
        ]
        rows.append(
            {
                "point_id": point.point_id,
                "payload_bytes": point.payload_bytes,
                "message_count": point.message_count,
                "applied_controls": {"sources": point.sources},
                "latency_flow_ledger": ledgers[:1],
                "bulk_flow_ledger": ledgers[1:],
            }
        )
    return rows


def test_run2_runner_pins_the_final_expectations_commit_and_digest() -> None:
    assert run_campaign.EXPECTATIONS_COMMIT == (
        "b21ba822707d2d7c80b83ee2d3fb87f4fa93178d"
    )
    assert run_campaign.EXPECTATIONS_SHA256 == (
        "5465271e9909cebc214c153209316a6f266ec142d7e578b3279935b1c6a10a53"
    )
    assert run_campaign.sha256(STUDY / "expectations_run2.json") == (
        run_campaign.EXPECTATIONS_SHA256
    )
    assert run_campaign.load_expectations()["study"]["task_id"] == "TRAF-74"


def test_run2_campaign_expands_one_cell_to_the_exact_matrix() -> None:
    points = run_campaign.campaign_points(_frozen())

    assert len(points) == 42
    assert {point.producer for point in points} == {"persistent_sm_peer_write"}
    assert {point.payload_bytes for point in points} == {256}
    assert {point.payload_bytes * point.message_count for point in points} == {
        4 << 20,
        8 << 20,
    }
    assert {len(point.sources.split(",")) for point in points} == {1, 2, 3}
    assert all(point.destination == 0 and point.destinations == "0" for point in points)
    assert all(point.point_id.startswith("TRAF74_NVINC_RUN2_LONG_D") for point in points)
    keys = {
        score_study.observation_key(
            {
                "point_id": point.point_id,
                "payload_bytes": point.payload_bytes,
                "message_count": point.message_count,
                "applied_controls": {"sources": point.sources},
            }
        )
        for point in points
    }
    assert len(keys) == 42


def test_run2_synthetic_exact_hardware_passes_all_six_cells() -> None:
    frozen = _frozen()
    samples = score_study.summarize_samples(_synthetic_rows(), frozen)
    comparisons = score_study.compare_cells(samples, frozen, measurement_valid=True)

    assert len(samples) == 42
    assert len(comparisons) == 6
    assert all(row["verdict"] == "PASS" for row in comparisons)
    assert all(row["physical_sanity"] == "PASS" for row in comparisons)
    assert all(abs(row["aggregate_signed_relative_error"]) < 1e-15 for row in comparisons)
    assert all(
        all(
            abs(value) < 1e-15
            for value in row["completion_signed_relative_error_by_source"]
        )
        for row in comparisons
    )


def test_run2_void_measurement_never_becomes_a_behavioral_miss() -> None:
    frozen = _frozen()
    samples = score_study.summarize_samples(_synthetic_rows(), frozen)
    comparisons = score_study.compare_cells(
        samples,
        frozen,
        measurement_valid=False,
    )

    assert all(row["verdict"] == "VOID" for row in comparisons)
    assert all(
        row["responsible_parameter"] == "undecidable_under_void_run"
        for row in comparisons
    )


def test_run2_residual_is_restricted_to_the_integrator_assigned_id(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="TRAF-86"):
        score_study.audit_hardware(
            tmp_path,
            expected_head="0" * 40,
            scheduler_job="test",
            residual_task="TRAF-87",
        )


def test_run2_merlin_entry_is_one_short_paced_resumable_cell() -> None:
    text = (STUDY / "run_merlin_cell_run2.sbatch").read_text(encoding="utf-8")

    assert "#SBATCH --partition=a100-hourly" in text
    assert "#SBATCH --gres=gpu:4" in text
    assert "#SBATCH --exclusive" in text
    assert "#SBATCH --time=00:12:00" in text
    assert "#SBATCH --array" not in text
    assert "sleep 2" in text
    assert "trap 'stop_cell HUP' HUP" in text
    assert "trap 'stop_cell TERM' TERM" in text
    assert "expectations_run2.json" in text
    assert "run_campaign_run2.py" in text
    assert "--mode hardware" in text


def test_run2_renderer_defines_metric_evidence_and_scope_before_matplotlib() -> None:
    source = (STUDY / "plot_study_run2.py").read_text(encoding="utf-8")

    assert plot_study.FIGURE_STEM == "nvlink-incast-hardware-simulation-run2"
    assert "Aggregate receiver payload goodput (GB/s)" in source
    assert "Scored simulation" in source
    assert "Measured hardware" in source
    assert "total delivered payload divided by" in source
    tree = ast.parse(source)
    module_imports = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "matplotlib" not in module_imports


def test_run2_post_freeze_python_is_portable_and_all_writers_pin_lf() -> None:
    forbidden = {"fcntl", "pwd", "resource", "termios"}
    for filename in (
        "run_campaign_run2.py",
        "score_study_run2.py",
        "plot_study_run2.py",
    ):
        path = STUDY / filename
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name.split(".", 1)[0]
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").split(".", 1)[0]
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
        }
        assert not imports & forbidden
        assert "typing_extensions" not in imports
        if filename == "score_study_run2.py":
            assert 'newline="\\n"' in (
                STUDY / "score_study.py"
            ).read_text(encoding="utf-8")
        contents = path.read_bytes()
        assert b"\r\n" not in contents
        assert b"+" + b"/-" not in contents
        assert "\u2014".encode() not in contents
        assert b"/" + b"data3" not in contents
        assert b"/" + b"home/" not in contents
