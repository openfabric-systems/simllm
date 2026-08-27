import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "a100_nvlink_packet_v2"
PREVIOUS_STUDY = ROOT / "examples" / "a100_nvlink_packet_v1"
FREEZE_SHA256 = "f0ab026e054873a56614af63ab3a7ae3219dc0b045423808cb41522910fa6da6"
PROTECTED_EXPECTATIONS_SHA256 = (
    "212a7a26f54e444c9b18f1e528bd0d00b5a28e4f9e005b0dc137f477ad642571"
)
PROTECTED_CANDIDATE_SHA256 = (
    "899712c4734f7a6b410d80231291663a404511528d46aab7497b73831e0e354f"
)


def run_study(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, str(STUDY / "run_study.py"), *arguments),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_expectations_freeze_is_immutable_and_complete():
    path = STUDY / "expectations.json"
    expectations = json.loads(path.read_text())

    assert hashlib.sha256(path.read_bytes()).hexdigest() == FREEZE_SHA256
    assert expectations["status"] == "expectations_only_frozen_before_harness"
    assert len(expectations["catalog"]) == 80
    assert len(expectations["decision_rules"]) == 11
    assert len(expectations["fatal_guards"]) == 10
    assert [case["ordinal"] for case in expectations["catalog"]] == list(range(1, 81))
    assert all(case["expected_band"] for case in expectations["catalog"])
    assert all(case["required_observables"] for case in expectations["catalog"])
    assert all(case["identification_rule_ids"] for case in expectations["catalog"])
    assert {guard["id"][:4] for guard in expectations["fatal_guards"]} == {
        f"FG{index:02d}" for index in range(1, 11)
    }


def test_traf65_and_candidate_inputs_remain_byte_locked_before_score():
    expectations = PREVIOUS_STUDY / "expectations.json"
    candidate = PREVIOUS_STUDY / "candidate-profile.json"

    assert hashlib.sha256(expectations.read_bytes()).hexdigest() == (
        PROTECTED_EXPECTATIONS_SHA256
    )
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == PROTECTED_CANDIDATE_SHA256


def test_cell_registry_has_80_isolated_5_ordered_and_1_all_frame():
    completed = run_study("--list-cells")

    assert completed.returncode == 0, completed.stderr
    cells = json.loads(completed.stdout)
    assert len(cells) == 86
    assert [cell["index"] for cell in cells] == list(range(86))
    assert sum(cell["frame"] == "isolated" for cell in cells) == 80
    assert sum(cell["frame"] == "corner_frame" for cell in cells) == 5
    assert cells[-1]["frame"] == "all_corners_frame"
    assert len(cells[-1]["case_names"]) == 80


def test_hardware_matrix_keeps_fixed_and_soak_arms_bounded():
    previous = sys.modules.get("case_matrix")
    sys.path.insert(0, str(STUDY))
    try:
        import case_matrix

        expectations = json.loads((STUDY / "expectations.json").read_text())
        fixed_total = case_matrix.points_for_case(expectations["catalog"][9])
        pair_balance = case_matrix.points_for_case(expectations["catalog"][17])
        soak = case_matrix.points_for_case(expectations["catalog"][79])
    finally:
        sys.path.pop(0)
        if previous is None:
            sys.modules.pop("case_matrix", None)
        else:
            sys.modules["case_matrix"] = previous

    assert {point.payload_bytes * point.message_count for point in fixed_total} == {
        16 << 20
    }
    assert {point.destinations for point in pair_balance} == {"1", "2", "3"}
    assert max(point.payload_bytes * point.message_count for point in soak) <= 32 << 20


@pytest.fixture(scope="module")
def mock_binary(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    suffix = ".exe" if os.name == "nt" else ""
    output = tmp_path_factory.mktemp("traf70-build") / f"packet-mock{suffix}"
    completed = subprocess.run(
        (
            compiler,
            "-x",
            "c++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-DSIMLLM_NVLINK_MOCK",
            str(STUDY / "nvlink_packet_lane.cu"),
            "-o",
            str(output),
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    return output


def test_mock_row_has_corrected_observation_contract(tmp_path, mock_binary):
    arguments = (
        "--mode",
        "mock",
        "--binary",
        str(mock_binary),
        "--output-root",
        str(tmp_path),
        "--array-index",
        "14",
    )
    first = run_study(*arguments)
    second = run_study(*arguments)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "already complete and digest verified" in second.stdout
    attempt = next(
        (tmp_path / FREEZE_SHA256 / "cells" / "isolated-015").glob("attempt-*")
    )
    rows = [json.loads(line) for line in (attempt / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert {row["producer"] for row in rows} == {
        "persistent_sm_peer_write",
        "dependent_sm_peer_read",
        "copy_engine_reference",
    }
    required = {
        "observed_data_bytes",
        "observed_raw_bytes",
        "observed_counter_deltas",
        "destination_checksum",
        "ordering_ledger",
        "applied_controls",
        "applied_control_sha256",
        "throttle_verdict",
        "telemetry_before",
        "telemetry_after",
        "replay_recovery_crc_ecc_deltas",
        "latency_flow_ledger",
        "bulk_flow_ledger",
        "drain_time_us",
        "candidate_blind_fit_membership",
    }
    assert all(required <= set(row) for row in rows)
    assert all(row["checksum_ok"] is True for row in rows)
    assert all(set(row["applied_controls"]["effects"]) == {
        "payload_bytes",
        "message_count",
        "source",
        "destination",
        "sources",
        "destinations",
        "source_alignment",
        "destination_alignment",
        "access_width",
        "active_lanes",
        "lane_mask",
        "stride",
        "stream_count",
        "outstanding",
        "burst_messages",
        "gap_ns",
        "offered_rate_percent",
        "pattern",
    } for row in rows)
    assert all(
        not {"candidate_packet_count", "candidate_raw_bytes", "predicted_raw_bytes"}
        & set(row)
        for row in rows
    )
    copy_row = next(row for row in rows if row["producer"] == "copy_engine_reference")
    assert copy_row["copy_engine_host_enqueue_count"] < copy_row["message_count"]
    assert copy_row["copy_engine_batch_mode"] == "single_contiguous_batch_per_flow_stream"


def test_hardware_source_and_submission_pin_required_interfaces():
    source = (STUDY / "nvlink_packet_lane.cu").read_text()
    sbatch = (STUDY / "run_merlin_cell.sbatch").read_text()

    assert "nvmlDeviceGetFieldValues" in source
    assert "NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX" in source
    assert "NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_RX" in source
    assert "nvmlDeviceGetNvLinkErrorCounter" in source
    assert "std::array<std::thread, 4> workers" in source
    assert "nvmlDeviceGetCurrentClocksThrottleReasons" in source
    assert "cudaMemcpyPeerAsync" in source
    assert "ncclSend" in source and "ncclRecv" in source
    assert "candidate_packet_count" not in source
    assert "candidate_raw_bytes" not in source
    assert "#SBATCH --partition=a100-hourly" in sbatch
    assert "#SBATCH --gres=gpu:4" in sbatch
    assert "#SBATCH --exclusive" in sbatch
    assert "#SBATCH --array=0-85%1" in sbatch
    assert FREEZE_SHA256 in sbatch
    assert "-lnvidia-ml" in sbatch


def test_scorer_exposes_all_frozen_guards_and_gate_logic():
    completed = subprocess.run(
        (
            sys.executable,
            str(STUDY / "score_hardware.py"),
            "--bulk-root",
            str(ROOT / "does-not-exist"),
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    score = json.loads(completed.stdout)

    assert score["status"] == "PENDING_0_OF_86"
    assert score["coverage"]["pending_indices"] == list(range(86))
    assert {guard["guard_id"] for guard in score["fatal_guard_verdicts"]["guards"]} == {
        f"FG{index:02d}" for index in range(1, 11)
    }
    assert score["flow_dynamics_gate"]["verdict"] == "CLOSED"
    assert score["profile_patch"]["changes"] == []
    assert score["scoring_audit"] == {}


def test_scorer_refuses_shortcut_promotions_and_uses_frozen_rate_band():
    guard_text = """=== gpu_list returncode=0 ===
GPU 0: NVIDIA A100-SXM4-80GB
GPU 1: NVIDIA A100-SXM4-80GB
GPU 2: NVIDIA A100-SXM4-80GB
GPU 3: NVIDIA A100-SXM4-80GB

=== topology returncode=0 ===
GPU0 X NV4 NV4 NV4
GPU1 NV4 X NV4 NV4
GPU2 NV4 NV4 X NV4
GPU3 NV4 NV4 NV4 X

=== clocks returncode=0 ===
GPU 0: NVIDIA A100-SXM4-80GB
"""
    script = f"""
import json
import sys
sys.path.insert(0, {str(STUDY)!r})
import score_hardware as scorer
guard_text = {guard_text!r}
packet_audit = {{}}
packet = scorer._score_packet_fit([], packet_audit)
link_audit = {{}}
links = scorer._score_links_and_rates([], link_audit)
credit_audit = {{}}
credits = scorer._score_credit_and_buffer([], {{}}, credit_audit)
spec = {{
    "module": "tx",
    "parameter": "per_link_rate_bytes_per_second",
    "candidate_value": 25_000_000_000,
    "rule_id": "TX_LINK_COUNT_RATE_AND_BOND",
}}
print(json.dumps({{
    "packet_statuses": sorted({{value[0] for value in packet.values()}}),
    "packet_grid": packet_audit["packet_fit"]["grid_cardinality"],
    "link_statuses": sorted({{value[0] for value in links.values()}}),
    "credit_statuses": sorted({{value[0] for value in credits.values()}}),
    "confirmed": scorer._identified_relation(spec, 27_250_000_000),
    "refuted": scorer._identified_relation(spec, 27_750_000_001),
    "gpu_list_count": scorer._gpu_list_count(guard_text),
    "nv4_row_count": scorer._nv4_row_count(guard_text),
}}))
"""
    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result == {
        "packet_statuses": ["INCONCLUSIVE"],
        "packet_grid": 33_024,
        "link_statuses": ["INCONCLUSIVE"],
        "credit_statuses": ["INCONCLUSIVE"],
        "confirmed": "CONFIRMED",
        "refuted": "REFUTED_AND_REPLACED",
        "gpu_list_count": 4,
        "nv4_row_count": 4,
    }
