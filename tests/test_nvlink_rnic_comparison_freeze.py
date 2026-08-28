import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_rnic_comparison_v1"
EXPECTATIONS = STUDY / "expectations.json"
SOURCE = ROOT / "examples" / "nvlink_flow_dynamics_v1"


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "nvlink_rnic_comparison_build_expectations",
        STUDY / "build_expectations.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> dict[str, object]:
    return json.loads(EXPECTATIONS.read_text(encoding="utf-8"))


def test_builder_roundtrips_the_expectations_bytes(tmp_path):
    output = tmp_path / "expectations.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(STUDY / "build_expectations.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.read_bytes() == EXPECTATIONS.read_bytes()


def test_zero_fit_mapping_carries_every_declared_physical_constant():
    frozen = _load()
    physical = frozen["physical_constants"]
    mapping = frozen["physical_mapping"]

    assert physical == {
        "credit_return_ps": 200000,
        "credit_round_payload_bytes": 65536,
        "credit_round_wire_bytes": 69632,
        "credit_unit_bytes": 272,
        "credits_per_destination": 256,
        "header_bytes": 16,
        "links_per_ordered_pair": 4,
        "max_payload_bytes": 256,
        "max_wire_bytes": 272,
        "nvlink_no_queue_one_packet_ps": 12194,
        "pair_raw_bytes_per_second": 100000000000,
        "per_link_bytes_per_second": 25000000000,
        "propagation_ps": 0,
        "rx_ingress_bytes_per_second": 207101921876,
        "tx_endpoint_egress_bytes_per_second": 160795737454,
        "wire_header_fraction": 16 / 272,
    }
    assert mapping["zero_fitted_constants"] is True
    assert [row["rnic_link_rate_bps"] for row in mapping["degree_specific_rnic_rates"]] == [
        800000000000,
        1600000000000,
        1656815375008,
    ]
    assert [row["isolated_one_packet_two_serializer_ps"] for row in mapping["degree_specific_rnic_rates"]] == [
        5440,
        2720,
        2628,
    ]
    assert all(row["fitted_constant"] is False for row in mapping["rows"])


def test_workload_is_the_source_ladder_and_all_release_digests_rebuild():
    builder = _load_builder()
    frozen = _load()
    workload = frozen["workload"]

    assert workload["flow_sizes_bytes"] == [
        256,
        1024,
        4096,
        16384,
        65536,
        262144,
        524288,
    ]
    assert workload["degrees"] == [1, 2, 3]
    assert workload["seed_count"] == len(workload["seeds"]) == 9
    assert workload["samples_per_seed_per_sender"] == 12
    assert len(workload["cells"]) == 21
    source = json.loads((SOURCE / "expectations.json").read_text(encoding="utf-8"))
    rebuilt = builder._workload(source)
    assert rebuilt == workload


def test_pinned_rnic_semantics_reject_the_ack_pacing_label():
    frozen = _load()
    authority = frozen["htsim_authority"]

    assert authority["commit"] == "1dcbfec36a33753bf978cf6323bade1a6645fe4f"
    assert authority["profile"] == "rnic-nn"
    assert authority["runtime_class"] == "RnicPacketizedManifoldRuntime"
    assert authority["ack_pacing_claim"] == "NOT_APPLICABLE_EXPECTED"
    assert "no route, queue, loss, backpressure, acknowledgement" in authority[
        "source_semantics"
    ]
    assert any("zero reverse bytes" in row["expectation"] for row in frozen["expected_directions"])


def test_source_result_and_every_direct_preservation_file_are_locked():
    frozen = _load()
    source = frozen["source_authority"]
    lock = frozen["preservation_lock"]

    assert _sha256(ROOT / source["expectations_path"]) == source["expectations_sha256"]
    assert _sha256(ROOT / source["result_path"]) == source["result_sha256"]
    assert lock["inherited_expected_artifacts"] == 60
    assert lock["direct_count"] == len(lock["direct_flow_dynamics_artifacts"]) == 18
    assert len({row["path"] for row in lock["direct_flow_dynamics_artifacts"]}) == 18
    for artifact in lock["direct_flow_dynamics_artifacts"]:
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]


def test_freeze_is_portable_lf_text_and_contains_no_observation_files():
    frozen = _load()
    report = (STUDY / "expectations.md").read_text(encoding="utf-8")

    assert frozen["study"]["status"] == "expectations_only"
    assert frozen["plot_contract"]["path_rendering"] == "POSIX"
    assert "\N{EM DASH}" not in report
    assert "/data3/" not in report
    assert "/home/" not in report
    for path in (
        STUDY / "build_expectations.py",
        STUDY / "expectations.json",
        STUDY / "expectations.md",
        ROOT / "tests" / "test_nvlink_rnic_comparison_freeze.py",
    ):
        assert b"\r" not in path.read_bytes()
    assert not (STUDY / "run_study.py").exists()
    assert not (STUDY / "results.json").exists()
    assert not (STUDY / "RESULTS.md").exists()
