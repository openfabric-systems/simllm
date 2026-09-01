import ast
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_mechanism_alignment_v1"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "nvlink_mechanism_alignment_run_study",
        STUDY / "run_study.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_study = _load_runner()


@pytest.fixture(scope="module")
def result() -> dict[str, object]:
    return run_study.run_study()


def test_runner_pins_the_final_preimplementation_authority() -> None:
    assert run_study.EXPECTATIONS_COMMIT == (
        "c589abadcfe7d142ffeee3a38db9f9d0a1dc23c8"
    )
    assert run_study.EXPECTATIONS_SHA256 == hashlib.sha256(
        (STUDY / "expectations.json").read_bytes()
    ).hexdigest()


def test_sanity_cells_hit_both_exact_payload_ceiling_oracles(result) -> None:
    rows = {
        (row["packet_flits"], row["per_link_bytes_per_second"]): row
        for row in result["sanity_cells"]
    }

    assert len(rows) == 4
    assert rows[(17, 25_000_000_000)]["payload_rate_gbps"] == pytest.approx(
        94.11764705882354
    )
    assert rows[(18, 25_000_000_000)]["payload_rate_gbps"] == pytest.approx(
        88.88888888888889
    )
    assert rows[(17, 25_000_000_000)]["job_completion_time_ps"] == 11_147_510
    assert rows[(18, 25_000_000_000)]["job_completion_time_ps"] == 11_803_247
    assert rows[(17, 12_500_000_000)]["job_completion_time_ps"] == 22_288_630
    assert rows[(18, 12_500_000_000)]["job_completion_time_ps"] == 23_599_727
    assert all(row["physical_bound_verdict"] == "PASS" for row in rows.values())


def test_serialization_relations_match_exactly(result) -> None:
    optional = [
        row
        for row in result["relation_checks"]
        if row["relation"] == "optional_flit_serialization"
    ]
    inverse_rate = [
        row
        for row in result["relation_checks"]
        if row["relation"] == "inverse_link_rate"
    ]

    assert [row["observed_shift_percent"] for row in optional] == [
        5.882352941176471,
        5.882352941176471,
    ]
    assert [row["signed_jct_shift_ps"] for row in optional] == [1_311_097, 655_737]
    assert [row["observed_ratio"] for row in inverse_rate] == [2.0, 2.0]
    assert all(row["verdict"] == "PASS" for row in result["relation_checks"])


def test_replay_credit_and_identity_guards_pass(result) -> None:
    replay = result["replay_probe"]

    assert replay["error_free_added_wire_bytes"] == 0
    assert replay["error_free_added_time_ps"] == 0
    assert replay["injected_added_wire_bytes"] == 272
    assert replay["injected_added_time_ps"] == 10_980
    assert all(
        row["credit_release_count"] == row["packet_count"] == 4096
        for row in result["sanity_cells"]
    )
    assert all(
        row["receiver_ownership_verdict"] == "PASS"
        and row["pass_through_identity_verdict"] == "PASS"
        and row["arbitration_identity_verdict"] == "PASS"
        and row["random_draw_count"] == 0
        for row in result["sanity_cells"]
    )


def test_every_consumer_and_recursive_preservation_lock_passes(result) -> None:
    preservation = result["preservation"]

    assert preservation["root_pin_count"] == 22
    assert preservation["root_failures"] == 0
    assert preservation["recursive_artifact_count"] == 95
    assert preservation["recursive_failures"] == 0
    assert preservation["historical_source_locks"] == [
        {
            "path": "simllm/backends/htsim_nvlink.py",
            "sha256": "0f16f44d0c6c74a1113ac0df7b9d6250aaa1c1ef20b8edd1d0f7e7449c76440c",
            "verification_commit": "9898a66dc215fd853d10492c6b852009326e376e",
        }
    ]


def test_every_inherited_envelope_publishes_a_signed_zero_shift(result) -> None:
    shifts = result["inherited_envelope_shifts"]

    assert len(shifts) == 6
    assert all(row["signed_shift"].startswith("+0") for row in shifts)
    assert all(row["verdict"] == "PASS" for row in shifts)


def test_runner_is_portable_and_all_writers_pin_lf() -> None:
    path = STUDY / "run_study.py"
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

    assert not imports & {"fcntl", "pwd", "resource", "termios"}
    assert "typing_extensions" not in imports
    assert 'newline="\\n"' in source
    assert b"\r" not in path.read_bytes()
    assert "\N{EM DASH}" not in source
    assert "/data3/" not in source
    assert "/home/" not in source
