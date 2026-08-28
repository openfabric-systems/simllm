import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_incast_validation_v1"


def _load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, STUDY / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_campaign = _load("nvlink_incast_validation_run_campaign", "run_campaign.py")
score_study = _load("nvlink_incast_validation_score_study", "score_study.py")
plot_study = _load("nvlink_incast_validation_plot_study", "plot_study.py")


def _frozen() -> dict[str, object]:
    return json.loads((STUDY / "expectations.json").read_text(encoding="utf-8"))


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


def test_runner_pins_the_final_expectations_commit_and_digest() -> None:
    assert run_campaign.EXPECTATIONS_COMMIT == (
        "092080e682acaee9d68779c6ebb2195e97d0d6fb"
    )
    assert run_campaign.EXPECTATIONS_SHA256 == (
        "9f50aadba0085a54e78c156d61837e4c7db19a498d8fef9c1aba7b32e0a163b4"
    )
    assert run_campaign.sha256(STUDY / "expectations.json") == (
        run_campaign.EXPECTATIONS_SHA256
    )


def test_reused_traf70_modules_do_not_claim_generic_import_names() -> None:
    assert sys.modules.get("case_matrix") is not run_campaign.traf70_cases
    assert sys.modules.get("run_study") is not run_campaign.traf70_run
    assert sys.modules.get("score_hardware") is not score_study.traf70_score


def test_campaign_expands_one_short_cell_to_the_exact_matrix() -> None:
    points = run_campaign.campaign_points(_frozen())

    assert len(points) == 42
    assert {point.producer for point in points} == {"persistent_sm_peer_write"}
    assert {point.payload_bytes for point in points} == {256}
    assert {point.payload_bytes * point.message_count for point in points} == {
        262144,
        524288,
    }
    assert {len(point.sources.split(",")) for point in points} == {1, 2, 3}
    assert all(point.destination == 0 and point.destinations == "0" for point in points)
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


def test_synthetic_exact_hardware_passes_all_six_behavioral_cells() -> None:
    frozen = _frozen()
    samples = score_study.summarize_samples(_synthetic_rows(), frozen)
    comparisons = score_study.compare_cells(samples, frozen, measurement_valid=True)

    assert len(samples) == 42
    assert len(comparisons) == 6
    assert all(row["verdict"] == "PASS" for row in comparisons)
    assert all(abs(row["aggregate_signed_relative_error"]) < 1e-15 for row in comparisons)
    assert all(
        all(abs(value) < 1e-15 for value in row["completion_signed_relative_error_by_source"])
        for row in comparisons
    )


def test_void_measurement_never_becomes_a_behavioral_miss() -> None:
    frozen = _frozen()
    samples = score_study.summarize_samples(_synthetic_rows(), frozen)
    comparisons = score_study.compare_cells(samples, frozen, measurement_valid=False)

    assert all(row["verdict"] == "VOID" for row in comparisons)
    assert all(
        row["responsible_parameter"] == "undecidable_under_void_run"
        for row in comparisons
    )


def test_frozen_attribution_order_names_each_identifiable_parameter() -> None:
    base = []
    for degree in (1, 2, 3):
        for size, error, hardware, simulation in (
            (262144, -0.40, [100.0] * degree, [3.0] * degree),
            (524288, -0.20, [200.0] * degree, [6.0] * degree),
        ):
            base.append(
                {
                    "degree": degree,
                    "size_bytes": size,
                    "aggregate_signed_relative_error": error,
                    "hardware_completion_us_by_source": hardware,
                    "simulation_completion_us_by_source": simulation,
                }
            )
    assert score_study.attribute_misses(base) == {
        1: "packetization",
        2: "packetization",
        3: "packetization",
    }

    stable = []
    for degree in (1, 2, 3):
        for size, hardware, simulation in (
            (262144, [100.0] * degree, [3.0] * degree),
            (524288, [200.0] * degree, [6.0] * degree),
        ):
            stable.append(
                {
                    "degree": degree,
                    "size_bytes": size,
                    "aggregate_signed_relative_error": -0.40,
                    "hardware_completion_us_by_source": hardware,
                    "simulation_completion_us_by_source": simulation,
                }
            )
    assert score_study.attribute_misses(stable) == {
        1: "tx_egress_plateau",
        2: "tx_egress_plateau",
        3: "rx_ingress_plateau",
    }

    credit = []
    for degree in (1, 2, 3):
        for size, hardware, simulation in (
            (262144, [8.0] * degree, [3.0] * degree),
            (524288, [11.0] * degree, [6.0] * degree),
        ):
            credit.append(
                {
                    "degree": degree,
                    "size_bytes": size,
                    "aggregate_signed_relative_error": -0.40,
                    "hardware_completion_us_by_source": hardware,
                    "simulation_completion_us_by_source": simulation,
                }
            )
    assert score_study.attribute_misses(credit) == {
        1: "credit_round",
        2: "credit_round",
        3: "credit_round",
    }


def test_merlin_entry_point_is_one_short_paced_resumable_cell() -> None:
    text = (STUDY / "run_merlin_cell.sbatch").read_text(encoding="utf-8")

    assert "#SBATCH --partition=a100-hourly" in text
    assert "#SBATCH --gres=gpu:4" in text
    assert "#SBATCH --exclusive" in text
    assert "#SBATCH --time=00:12:00" in text
    assert "#SBATCH --array" not in text
    assert "sleep 2" in text
    assert "trap 'stop_cell HUP' HUP" in text
    assert "trap 'stop_cell TERM' TERM" in text
    assert "STOPPED" not in text
    assert "--mode hardware" in text


def test_renderer_defines_axes_evidence_and_scope_before_importing_matplotlib() -> None:
    source = (STUDY / "plot_study.py").read_text(encoding="utf-8")

    assert plot_study.FIGURE_STEM == "nvlink-incast-hardware-simulation"
    assert "Aggregate receiver payload goodput (GB/s)" in source
    assert "Scored simulation" in source
    assert "Measured hardware" in source
    assert "Degrees 4, 8 and 16 remain declared" in source
    tree = ast.parse(source)
    module_imports = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "matplotlib" not in module_imports


def test_post_freeze_python_is_portable_and_all_writers_pin_lf() -> None:
    forbidden = {"fcntl", "pwd", "resource", "termios"}
    for filename in ("run_campaign.py", "score_study.py", "plot_study.py"):
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
        if filename != "plot_study.py":
            assert 'newline="\\n"' in source
        assert b"\r\n" not in path.read_bytes()
