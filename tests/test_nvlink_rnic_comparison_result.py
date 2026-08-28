import csv
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_rnic_comparison_v1"
RESULT_PATH = STUDY / "results.json"
REPORT_PATH = STUDY / "RESULTS.md"
DISPERSION_PATH = STUDY / "dispersion.csv"


def _result():
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_frozen_workload_and_every_fatal_guard_are_published():
    result = _result()

    assert result["schema"] == "simllm-nvlink-rnic-comparison-result-v1"
    assert result["study_verdict"] == "PASS_WITH_HONEST_MISSES"
    assert result["sample_count"] == 9072
    assert result["rnic_adapter_invocations"] == 189
    assert len(result["cell_summaries"]) == 42
    assert len(result["fatal_guards"]) == 16
    assert result["fatal_guard_verdict"] == "PASS"
    assert all(row["verdict"] == "PASS" for row in result["fatal_guards"])
    assert result["authority"]["inherited_artifacts_checked"] == 60
    assert result["authority"]["flow_dynamics_files_checked"] == 18


def test_nvlink_arm_is_byte_identical_to_the_merged_raw_fct_evidence():
    result = _result()

    assert result["nvlink_source_raw_projection_sha256"] == {
        "fct-cdf.csv": "500f4a2a2aa971d33e31b7769199d4107ec7cf29f617d3d36524b1ea8283ab65",
        "fct-samples.csv": "d051bb65d802d4a3e90a65f7dbf3ba573bee32b7d76ea5956c51d172f841bef8",
    }
    assert result["authority"]["expectations_commit"] == (
        "6224d90fea2eed788b8e6ba876787fe7f0e52319"
    )
    assert result["authority"]["expectations_sha256"] == (
        "4b60365d8251b5fd3c7627dbe38c66ad1fc1c096b21fdfada4fc744320a5bdfa"
    )


def test_pinned_rnic_event_ledger_has_no_ack_or_reverse_control_work():
    result = _result()
    provenance = result["adapter_provenance"]
    rnic_rows = [
        row for row in result["cell_summaries"] if row["transport"] == "rnic-nn"
    ]

    assert provenance["htsim_source_commit"] == (
        "1dcbfec36a33753bf978cf6323bade1a6645fe4f"
    )
    assert provenance["runtime_class"] == "RnicPacketizedManifoldRuntime"
    assert provenance["ack_pacing"] == "absent"
    assert len(provenance["executable_sha256"]) == 64
    assert sum(row["ack_events"] for row in rnic_rows) == 0
    assert sum(row["reverse_control_bytes"] for row in rnic_rows) == 0
    assert sum(row["non_data_events"] for row in rnic_rows) == 0


def test_dispersion_comparison_and_honest_misses_are_complete():
    result = _result()
    rows = result["dispersion_comparison"]
    directions = {row["id"]: row for row in result["expected_direction_verdicts"]}

    assert len(rows) == 21
    assert sum(row["tighter_transport"] == "rnic-nn" for row in rows) == 8
    assert sum(row["tighter_transport"] == "nvlink-credit" for row in rows) == 11
    assert sum(row["tighter_transport"] == "tie" for row in rows) == 2
    assert sum(
        row["tighter_transport"] == "rnic-nn"
        for row in rows
        if row["size_bytes"] >= 65536
    ) == 5
    assert result["scored_misses"] == ["E3", "E5"]
    assert directions["E3"]["passed_instances"] == 5
    assert directions["E3"]["required_passes"] == 7
    assert directions["E5"]["passed_instances"] == 1
    assert directions["E5"]["required_passes"] == 4
    assert directions["E4"]["verdict"] == "PASS"
    assert directions["E6"]["verdict"] == "PASS"


def test_credit_and_arbitration_diagnostics_support_the_named_mechanisms():
    result = _result()
    nvlink_rows = [
        row
        for row in result["cell_summaries"]
        if row["transport"] == "nvlink-credit"
    ]

    assert sum(row["credit_wait_packets"] for row in nvlink_rows) == 0
    assert sum(row["credit_wait_ps"] for row in nvlink_rows) == 0
    assert sum(row["rx_wait_packets"] for row in nvlink_rows) == 1_733_130
    assert sum(row["rx_wait_packets"] > 0 for row in nvlink_rows) == 14


def test_compact_dispersion_csv_matches_the_published_json():
    result = _result()
    with DISPERSION_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 21
    assert [int(row["size_bytes"]) for row in rows] == [
        int(row["size_bytes"]) for row in result["dispersion_comparison"]
    ]
    assert [int(row["degree"]) for row in rows] == [
        int(row["degree"]) for row in result["dispersion_comparison"]
    ]
    assert [row["tighter_transport"] for row in rows] == [
        row["tighter_transport"] for row in result["dispersion_comparison"]
    ]


def test_all_published_figures_match_their_manifest_and_are_readable():
    result = _result()

    assert len(result["figure_artifacts"]) == 4
    for artifact in result["figure_artifacts"]:
        path = STUDY / "figures" / artifact["path"]
        payload = path.read_bytes()
        assert len(payload) == artifact["bytes"]
        assert hashlib.sha256(payload).hexdigest() == artifact["sha256"]
        if path.suffix == ".pdf":
            assert payload.startswith(b"%PDF-")
        else:
            assert payload.startswith(b"\x89PNG\r\n\x1a\n")
            width, height = struct.unpack(">II", payload[16:24])
            assert width >= 1400
            assert height >= 800


def test_report_is_portable_names_the_mechanisms_and_closes_traf71():
    report_bytes = REPORT_PATH.read_bytes()
    report = report_bytes.decode("utf-8")
    traffic = (ROOT / "docs" / "modules" / "traffic.md").read_text(
        encoding="utf-8"
    )
    ledger = json.loads((ROOT / "docs" / "task-ledger.json").read_text())

    assert b"\r" not in report_bytes
    assert "\N{EM DASH}" not in report
    assert "/data3/" not in report
    assert "/home/" not in report
    assert "rnic-nn is tighter in\n8/21 rung-degree cells" in report
    assert "zero ACK events and zero reverse bytes" in report
    assert "not ACK pacing" in report
    assert "0/21 NVLink cells" in report
    assert "210.880 ns" in report
    assert "696.320 ns" in report
    assert "finite-sample and stagger-alignment effects" in report
    assert "The common 1 KiB sign" in report
    assert "- TRAF-71 (" not in traffic
    assert "TRAF-71" in ledger["closed"]


def test_every_published_text_file_is_lf_pinned():
    for path in (
        REPORT_PATH,
        RESULT_PATH,
        DISPERSION_PATH,
        STUDY / "plot_study.py",
        STUDY / "publish_study.py",
        ROOT / "tests" / "test_nvlink_rnic_comparison_result.py",
    ):
        assert b"\r" not in path.read_bytes()
