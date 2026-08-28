import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from simllm.backends.htsim_nvlink import (
    NvlinkDomainResult,
    NvlinkDomainService,
    NvlinkFlowPolicy,
    NvlinkTransfer,
    load_nvlink_candidate_profile,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_incast_validation_v1"
PROFILE = ROOT / "examples" / "a100_nvlink_packet_v1" / "candidate-profile.json"
TRAFFIC_DOC = ROOT / "docs" / "modules" / "traffic.md"
EXPECTATIONS_SHA256 = "9f50aadba0085a54e78c156d61837e4c7db19a498d8fef9c1aba7b32e0a163b4"
EXPECTATIONS_MARKDOWN_SHA256 = (
    "138bc56e9779549d3f2fae3c18d2b46060d21ce6c4ada89c65d842075f852f43"
)


def _load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, STUDY / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_expectations = _load(
    "nvlink_incast_validation_build_expectations", "build_expectations.py"
)


def _frozen() -> dict[str, object]:
    return json.loads((STUDY / "expectations.json").read_text(encoding="utf-8"))


def test_freeze_is_reproducible_from_its_recorded_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _frozen()

    def fail_current_tree_access(_value: object) -> object:
        pytest.fail("recorded freeze replay accessed the current preservation tree")

    monkeypatch.setattr(build_expectations, "_tracked_paths", fail_current_tree_access)
    monkeypatch.setattr(build_expectations, "_sha256", fail_current_tree_access)
    assert build_expectations.build() == frozen
    assert build_expectations.build(
        recorded_preservation=frozen["preservation_lock"]
    ) == frozen
    assert frozen["schema"] == "simllm-nvlink-incast-validation-expectations-v1"
    assert frozen["study"]["task_id"] == build_expectations.FROZEN_TASK_ID
    assert build_expectations.FROZEN_TASK_ID == "TRAF-73"
    assert build_expectations.REGISTRY_TASK_ID == "TRAF-74"
    assert build_expectations.FROZEN_TASK_ID != build_expectations.REGISTRY_TASK_ID
    assert frozen["study"]["status"] == "expectations_only"
    assert hashlib.sha256((STUDY / "expectations.json").read_bytes()).hexdigest() == (
        EXPECTATIONS_SHA256
    )
    assert hashlib.sha256((STUDY / "expectations.md").read_bytes()).hexdigest() == (
        EXPECTATIONS_MARKDOWN_SHA256
    )
    traffic = TRAFFIC_DOC.read_text(encoding="utf-8")
    assert "- TRAF-74 (Precision; P1; L): validate the scored" in traffic
    assert "frozen artifacts keep the\n  original string" in traffic
    encoded = json.dumps(frozen, sort_keys=True).lower()
    assert "hardware_result" not in encoded
    assert "cell_verdicts" not in encoded
    assert "observed_hardware" not in encoded


def test_freeze_selects_only_capturable_long_flow_cells() -> None:
    frozen = _frozen()
    arm = frozen["hardware_arm"]

    assert arm["degrees"] == [1, 2, 3]
    assert arm["flow_sizes_bytes"] == [262144, 524288]
    assert arm["producer"] == "persistent_sm_peer_write"
    assert arm["producer_payload_bytes"] == 256
    assert arm["repetitions_per_cell"] == 7
    assert frozen["scope_limits"]["simulation_only_degrees"] == [4, 8, 16]
    assert frozen["scope_limits"]["flow_scope"] == "long flows only"


def test_launch_skew_is_frozen_as_a_fatal_negligibility_guard() -> None:
    frozen = _frozen()
    launch = frozen["hardware_arm"]["launch_skew"]

    assert launch["per_additional_sender_budget_ps"] == 5_000_000
    assert launch["negligible_fraction_high"] == 0.10
    assert len(launch["rows"]) == 6
    assert all(row["pre_run_negligible"] is True for row in launch["rows"])
    worst = max(row["maximum_launch_skew_fraction"] for row in launch["rows"])
    assert 0.095 < worst < 0.10
    guard_ids = {
        guard["id"] for guard in frozen["fatal_guards"]["study_specific"]
    }
    assert "FG11_LAUNCH_SKEW_NEGLIGIBLE" in guard_ids


def test_predictions_recompute_through_the_scored_domain() -> None:
    frozen = _frozen()
    profile = load_nvlink_candidate_profile(PROFILE)
    service = NvlinkDomainService(profile)

    for row in frozen["simulation_arm"]["predictions"]:
        transfers = [
            NvlinkTransfer(
                extent_id=f"source-{source}",
                source=source,
                destination=0,
                payload_bytes=row["size_bytes"],
            )
            for source in range(1, row["degree"] + 1)
        ]
        result = service.serve(
            transfers,
            analytic_result=None,
            flow_policy=NvlinkFlowPolicy.RELEASE_AWARE_ROUND_ROBIN,
        )
        assert isinstance(result, NvlinkDomainResult)
        completion = [
            max(
                packet.delivered_at_ps
                for packet in result.packets
                if packet.extent_id == transfer.extent_id
            )
            for transfer in transfers
        ]
        assert completion == row["completion_ps_by_source"]
        expected_aggregate = row["degree"] * row["size_bytes"] * 1000 / max(
            completion
        )
        assert expected_aggregate == row["aggregate_payload_gbps"]


def test_acceptance_band_and_attribution_are_literal() -> None:
    frozen = _frozen()
    comparison = frozen["comparison"]

    assert comparison["signed_relative_error_formula"] == (
        "(simulation - hardware) / hardware"
    )
    assert comparison["acceptance_low"] == -0.15
    assert comparison["acceptance_high"] == 0.15
    assert [row["parameter"] for row in comparison["miss_attribution_order"]] == [
        "pass_through_switch_identity",
        "packetization",
        "credit_round",
        "rx_ingress_plateau",
        "tx_egress_plateau",
    ]


def test_recorded_preservation_lock_is_self_contained_and_not_rediscovered() -> None:
    frozen = _frozen()
    lock = frozen["preservation_lock"]
    artifacts = lock["artifacts"]

    assert len(artifacts) == lock["artifact_count"] == 59
    paths = {artifact["path"] for artifact in artifacts}
    assert "examples/a100_nvlink_packet_v2/hardware-score.json" in paths
    assert "examples/nvlink_flow_dynamics_v1/results.json" in paths
    assert "examples/nvlink_rnic_comparison_v2/results.json" in paths
    assert "examples/a100_nvlink_packet_v1/candidate-profile.json" in paths
    assert "simllm/backends/htsim_nvlink.py" in paths
    assert build_expectations._recorded_preservation_lock(lock) == lock
    assert lock["artifacts_sha256"] == build_expectations._canonical_sha256(artifacts)
    for artifact in artifacts:
        path = Path(artifact["path"])
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert artifact["bytes"] >= 0
        assert len(artifact["sha256"]) == 64


def test_freeze_sources_are_portable_and_use_lf_writers() -> None:
    source = STUDY / "build_expectations.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
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
    assert 'newline="\\n"' in source.read_text(encoding="utf-8")
    for path in STUDY.glob("*"):
        if path.is_file():
            assert b"\r\n" not in path.read_bytes()
