import ast
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_incast_validation_v1"
FREEZE_JSON = STUDY / "expectations_run2.json"
FREEZE_MARKDOWN = STUDY / "expectations_run2.md"
MODEL = ROOT / "simllm" / "backends" / "htsim_nvlink.py"
FREEZE_SHA256 = "5465271e9909cebc214c153209316a6f266ec142d7e578b3279935b1c6a10a53"
FREEZE_MARKDOWN_SHA256 = (
    "8bcdb38545aa17b82f8b42c92a3bc9b225dc61a00ce6dab0bf25a6d75dd2428c"
)
FIRST_RUN_SHA256 = {
    "RESULTS.md": "f81971b6113b36be9998339b8f8ddd174a1f45488ed79c4bc46c9c159b1fa0a6",
    "comparison.csv": "874af0453fa673b37575ce8c03ef0fcf28eb5aad1b23161d2ff5cdebd41052bd",
    "expectations.json": "9f50aadba0085a54e78c156d61837e4c7db19a498d8fef9c1aba7b32e0a163b4",
    "expectations.md": "138bc56e9779549d3f2fae3c18d2b46060d21ce6c4ada89c65d842075f852f43",
    "results.json": "114c8f86c6df378f7afbc9e2a2d57cc665d71d9da017283810d30ed970f7ebfd",
}


def _load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, STUDY / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_expectations = _load(
    "nvlink_incast_validation_run2_build_expectations",
    "build_expectations_run2.py",
)


def _frozen() -> dict[str, object]:
    return json.loads(FREEZE_JSON.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_run2_freeze_is_reproducible_without_rediscovering_preserved_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _frozen()

    def fail_current_tree_access(_value: object) -> object:
        pytest.fail("run-two freeze replay rediscovered the preservation tree")

    monkeypatch.setattr(build_expectations, "_tracked_paths", fail_current_tree_access)
    assert build_expectations.build(
        recorded_preservation=frozen["preservation_lock"]
    ) == frozen
    assert frozen["schema"] == "simllm-nvlink-incast-validation-expectations-v2"
    assert frozen["study"]["task_id"] == build_expectations.TASK_ID == "TRAF-74"
    assert frozen["study"]["status"] == "expectations_only"
    assert _sha256(FREEZE_JSON) == FREEZE_SHA256
    assert _sha256(FREEZE_MARKDOWN) == FREEZE_MARKDOWN_SHA256
    encoded = json.dumps(frozen, sort_keys=True).lower()
    assert "second_hardware_result" not in encoded
    assert "run2_cell_verdicts" not in encoded
    assert "observed_run2_hardware" not in encoded


def test_run2_freeze_selects_millisecond_scale_long_flows() -> None:
    frozen = _frozen()
    arm = frozen["hardware_arm"]

    assert arm["degrees"] == [1, 2, 3]
    assert arm["flow_sizes_bytes"] == [4 << 20, 8 << 20]
    assert arm["producer"] == "persistent_sm_peer_write"
    assert arm["producer_payload_bytes"] == 256
    assert arm["repetitions_per_cell"] == 7
    assert frozen["physical_sanity"]["ceiling_ps"] == 5_000_000_000
    assert frozen["scope_limits"]["simulation_only_degrees"] == [4, 8, 16]
    assert frozen["scope_limits"]["flow_scope"] == "long flows only"


def test_run2_launch_skew_has_large_prefrozen_margin() -> None:
    frozen = _frozen()
    launch = frozen["hardware_arm"]["launch_skew"]

    assert launch["per_additional_sender_budget_ps"] == 5_000_000
    assert launch["negligible_fraction_high"] == 0.10
    assert len(launch["rows"]) == 6
    assert all(row["pre_run_negligible"] is True for row in launch["rows"])
    worst = max(row["maximum_launch_skew_fraction"] for row in launch["rows"])
    minimum_margin = min(row["pre_run_margin_to_budget"] for row in launch["rows"])
    assert worst == pytest.approx(0.005998540953289184)
    assert minimum_margin > 0.094
    guard_ids = {guard["id"] for guard in frozen["fatal_guards"]["study_specific"]}
    assert guard_ids == {
        "FG11_LAUNCH_SKEW_NEGLIGIBLE",
        "FG12_COMPLETE_REPETITION_MATRIX",
        "FG13_PRESERVATION_LOCK",
    }


def test_run2_predictions_pin_the_merged_base_domain_and_policy() -> None:
    frozen = _frozen()
    simulation = frozen["simulation_arm"]

    assert simulation["module_version_commit"] == (
        "65593131a0448d2b33f51018d5972c918dad3493"
    )
    assert simulation["flow_policy"] == "release_aware_round_robin"
    assert simulation["release_ps"] == 0
    assert simulation["model_sha256"] == _sha256(MODEL)
    base = subprocess.run(
        (
            "git",
            "show",
            f"{simulation['module_version_commit']}:simllm/backends/htsim_nvlink.py",
        ),
        cwd=ROOT,
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert hashlib.sha256(base.stdout).hexdigest() == simulation["model_sha256"]
    assert len(simulation["predictions"]) == 6
    assert {
        (row["degree"], row["size_bytes"])
        for row in simulation["predictions"]
    } == {
        (degree, size)
        for degree in (1, 2, 3)
        for size in (4 << 20, 8 << 20)
    }
    for row in simulation["predictions"]:
        assert len(row["completion_ps_by_source"]) == row["degree"]
        assert len(row["completion_acceptance_by_source"]) == row["degree"]
        assert row["physical_floor_ps"] <= max(row["completion_ps_by_source"])
        assert row["physical_ceiling_ps"] == 5_000_000_000


def test_run2_acceptance_band_is_composed_from_retained_physical_allowances() -> None:
    frozen = _frozen()
    comparison = frozen["comparison"]
    basis = comparison["physical_justification"]

    assert comparison["signed_relative_error_formula"] == (
        "(simulation - hardware) / hardware"
    )
    assert comparison["acceptance_low"] == -0.16
    assert comparison["acceptance_high"] == 0.16
    expected_sum = (
        basis["traf70_endpoint_repeatability_fraction"]
        + basis["first_512k_source_repetition_deviation_fraction"]
        + basis["run2_worst_prefrozen_launch_skew_fraction"]
    )
    assert basis["unrounded_sum_fraction"] == pytest.approx(expected_sum)
    assert 0.15 < expected_sum < comparison["acceptance_high"]
    assert [
        row["parameter"] for row in comparison["miss_attribution_order"]
    ] == [
        "pass_through_switch_identity",
        "packetization",
        "credit_round",
        "rx_ingress_plateau",
        "tx_egress_plateau",
    ]


def test_run2_lock_preserves_the_complete_first_result_and_inherited_artifacts() -> None:
    frozen = _frozen()
    lock = frozen["preservation_lock"]
    artifacts = lock["artifacts"]

    assert len(artifacts) == lock["artifact_count"] == 71
    paths = {artifact["path"] for artifact in artifacts}
    assert "examples/a100_nvlink_packet_v2/hardware-score.json" in paths
    assert "examples/nvlink_flow_dynamics_v1/results.json" in paths
    assert "examples/nvlink_rnic_comparison_v2/results.json" in paths
    assert "examples/a100_nvlink_packet_v1/candidate-profile.json" in paths
    assert "simllm/backends/htsim_nvlink.py" in paths
    assert "examples/nvlink_incast_validation_v1/RESULTS.md" in paths
    assert "examples/nvlink_incast_validation_v1/results.json" in paths
    assert build_expectations._recorded_preservation_lock(lock) == lock
    assert lock["artifacts_sha256"] == build_expectations._canonical_sha256(artifacts)
    for name, digest in FIRST_RUN_SHA256.items():
        assert _sha256(STUDY / name) == digest


def test_run2_freeze_sources_are_portable_and_pin_lf_writers() -> None:
    source = STUDY / "build_expectations_run2.py"
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text)
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
    assert 'newline="\\n"' in text
    for path in (source, FREEZE_JSON, FREEZE_MARKDOWN):
        contents = path.read_bytes()
        assert b"\r\n" not in contents
        assert b"+/-" not in contents
        assert "\u2014".encode() not in contents
        assert b"/data3" not in contents
        assert b"/home/" not in contents
