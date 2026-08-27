import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_flow_dynamics_v1"
EXPECTATIONS = STUDY / "expectations.json"


def load() -> dict[str, object]:
    return json.loads(EXPECTATIONS.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_expectation_builder_roundtrips_frozen_bytes(tmp_path):
    copy = tmp_path / "expectations.json"
    completed = subprocess.run(
        [sys.executable, str(STUDY / "build_expectations.py"), "--output", str(copy)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert copy.read_bytes() == EXPECTATIONS.read_bytes()


def test_scored_profile_and_parameter_evidence_are_frozen():
    frozen = load()
    source = frozen["source_profile"]
    profile_path = ROOT / source["path"]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    assert sha256(profile_path) == source["sha256"]
    assert source["score_status"] == "COMPLETE_VALID_86_OF_86"
    assert source["score_sha256"] == sha256(
        ROOT / "examples" / "a100_nvlink_packet_v2" / "hardware-score.json"
    )
    assert source["flow_dynamics_gate"] == "OPEN"
    assert profile["status"] == "scored_mixed_parameter_evidence"
    ledger = frozen["parameter_ledger"]
    assert ledger["candidate_count"] == len(ledger["declared_candidates"]) == 10
    assert ledger["unchanged_parameter_count"] == 11
    assert ledger["measured"] == [
        "rx.delivery_order",
        "rx.ingress_rate_bytes_per_second",
        "rx.reassembly_policy",
        "tx.endpoint_egress_rate_bytes_per_second",
        "tx.request_response_direction",
    ]
    assert ledger["structural"] == ["switch.mode"]


def test_transition_identities_and_rate_arithmetic_are_literal():
    frozen = load()
    packet = frozen["packet_arithmetic"]
    convergence = frozen["convergence_1_to_2"]
    divergence = frozen["divergence_2_to_1"]

    assert packet == {
        "credit_return_ps": 200000,
        "credits_per_destination": 256,
        "endpoint_packet_ps": 1692,
        "header_bytes": 16,
        "link_packet_ps": 10880,
        "links_per_peer": 4,
        "pair_payload_rate_gbps": 94.11764705882354,
        "pair_raw_rate_bytes_per_second": 100000000000,
        "payload_bytes": 256,
        "rx_packet_ps": 1314,
        "wire_bytes": 272,
    }
    assert convergence["terms_ps"] == {
        "credit_wait": 0,
        "link_serialization": 10880,
        "packet_admission": 1692,
        "rx_serialization": 1314,
    }
    assert convergence["expected_open_ps"] == 13886
    assert convergence["exact_tolerance_ps"] == 0
    assert divergence["terms_ps"] == {
        "credit_wait": 0,
        "rx_serialization_difference": 0,
        "three_packet_admissions": -5076,
        "two_link_cadences": 21760,
    }
    assert divergence["expected_time_to_target_ps"] == 16684
    assert divergence["exact_tolerance_ps"] == 0


def test_schedule_cdf_and_incast_matrix_are_frozen():
    frozen = load()
    schedule = frozen["flow_schedule"]
    cdf = frozen["fct_cdf"]
    incast = frozen["incast"]

    assert schedule["flow_ids"] == ["flow-a", "flow-b", "flow-c"]
    assert schedule["target_bytes"] == [4194304, 2097152, 1048576]
    assert schedule["release_ps"] == [0, 11141120, 22282240]
    assert schedule["raw_bin_ps"] == 696320
    assert schedule["smoothing"] == "none"
    assert cdf["flow_sizes_bytes"] == [256, 1024, 4096, 16384, 65536, 262144, 524288]
    assert cdf["seed_count"] == len(cdf["seeds"]) == 9
    assert cdf["samples_per_seed_per_sender"] == 12
    assert cdf["band"] == "pointwise minimum to maximum empirical CDF across seeds"
    assert len(cdf["bands"]) == 21
    assert {row["degree"] for row in cdf["bands"]} == {1, 2, 3}
    assert [row["degree"] for row in incast["degrees"]] == [1, 2, 3]
    assert [row["expected_binding_module"] for row in incast["degrees"]] == [
        "tx_pair_links",
        "tx_pair_links",
        "rx",
    ]
    assert incast["fanout_check"]["expected_verdict"] == "REFUTED"
    assert incast["fanout_check"]["published_payload_gbps"] == 281.65


def test_preservation_class_has_60_current_artifacts():
    frozen = load()
    lock = frozen["preservation_lock"]
    inherited_path = ROOT / lock["inherited"]["path"]
    assert sha256(inherited_path) == lock["inherited"]["sha256"]
    inherited = json.loads(inherited_path.read_text(encoding="utf-8"))["preservation_lock"]
    inherited_artifacts = (
        json.loads((ROOT / inherited["inherited"]["path"]).read_text(encoding="utf-8"))[
            "preservation_lock"
        ]["artifacts"]
        + inherited["additional_artifacts"]
    )
    assert len(inherited_artifacts) == lock["inherited"]["expected_artifacts"] == 43
    artifacts = inherited_artifacts + lock["additional_artifacts"]
    assert len(artifacts) == lock["expected_total_artifacts"] == 60
    for artifact in artifacts:
        assert sha256(ROOT / artifact["path"]) == artifact["sha256"]


def test_freeze_contains_no_result_or_runner_and_pins_posix_rendering():
    frozen = load()
    assert frozen["study"]["status"] == "expectations_only"
    assert frozen["plot_contract"]["path_rendering"] == "POSIX"
    assert frozen["plot_contract"]["raw_rate_style"].endswith("no smoothing")
    assert not (STUDY / "run_study.py").exists()
    assert not (STUDY / "results.json").exists()
    assert not (STUDY / "RESULTS.md").exists()
