import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_rnic_comparison_v2"
EXPECTATIONS = STUDY / "expectations.json"


def _load() -> dict[str, object]:
    return json.loads(EXPECTATIONS.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builder_roundtrips_the_frozen_bytes(tmp_path):
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


def test_mapping_audit_identifies_the_entity_deficit_with_arithmetic():
    audit = _load()["mapping_audit"]

    assert audit["verdict"] == (
        "MAPPING_DEFICIT_IN_FAIR_SHARE_ENTITY_NOT_CAPACITY_VALUE"
    )
    assert [
        row["legacy_receiver_capacity_bytes_per_second"]
        for row in audit["rows"]
    ] == [100_000_000_000, 200_000_000_000, 207_101_921_876]
    assert [row["binding_capacity"] for row in audit["rows"]] == [
        "ordered-pair class cap",
        "ordered-pair class cap",
        "RX ingress plateau",
    ]
    assert audit["degree_3_aggregate_below_nvlink"] is False
    assert audit["degree_3_aggregate_capacity_ratio"] == 1.0
    arithmetic = audit["normalized_queue_arithmetic"]
    assert arithmetic["predicted_ratio"] == "601/360"
    assert arithmetic["relative_error_from_observed"] < 0.003


def test_corrected_mapping_is_asymmetric_and_pair_class_scoped():
    mapping = _load()["corrected_mapping"]

    assert mapping["zero_fitted_constants"] is True
    assert mapping["source_ordered_pair_capacity_bytes_per_second"] == 100_000_000_000
    assert mapping["destination_rx_ingress_capacity_bytes_per_second"] == 207_101_921_876
    assert mapping["max_active_entities_per_pair"] == 1
    assert [row["aggregate_capacity_bytes_per_second"] for row in mapping["degree_rows"]] == [
        100_000_000_000,
        200_000_000_000,
        207_101_921_876,
        207_101_921_876,
        207_101_921_876,
        207_101_921_876,
    ]


def test_workload_freezes_six_degrees_seven_rungs_and_nine_seeds():
    workload = _load()["workload"]

    assert workload["flow_sizes_bytes"] == [
        256,
        1024,
        4096,
        16384,
        65536,
        262144,
        524288,
    ]
    assert workload["degrees"] == [1, 2, 3, 4, 8, 16]
    assert workload["physical_degrees"] == [1, 2, 3]
    assert workload["simulated_mesh_degrees"] == [4, 8, 16]
    assert workload["seed_count"] == len(workload["seeds"]) == 9
    assert workload["samples_per_seed_per_sender"] == 12
    assert len(workload["cells"]) == 42
    assert sum(row["source_relation"] == "byte-identical TRAF-69 tuples" for row in workload["cells"]) == 21


def test_fluid_hypothesis_and_required_disclosures_are_literal():
    frozen = _load()
    hypotheses = {row["id"]: row for row in frozen["frozen_hypotheses"]}

    assert "within 1 ps" in hypotheses["H2"]["bar"]
    assert "all 42 cells" in hypotheses["H2"]["bar"]
    assert "degrees 4, 8, and 16" in hypotheses["H3"]["claim"]
    assert frozen["measurement_caveat"]["hardware_identification_scope"] == (
        "LONG-FLOW ONLY"
    )
    assert "SIMULATED MESH EXTRAPOLATION" in frozen["topology"][
        "required_figure_disclosure"
    ]
    assert "MODEL PREDICTION" in frozen["measurement_caveat"][
        "required_figure_disclosure"
    ]


def test_legacy_study_is_fully_byte_locked():
    lock = _load()["preservation_lock"]

    assert lock["artifact_count"] == len(lock["artifacts"]) == 16
    for artifact in lock["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert _sha256(path) == artifact["sha256"]


def test_freeze_is_portable_lf_and_contains_no_result_artifact():
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
        ROOT / "tests" / "test_nvlink_rnic_comparison_v2_freeze.py",
    ):
        assert b"\r" not in path.read_bytes()
    assert not (STUDY / "results.json").exists()
    assert not (STUDY / "RESULTS.md").exists()
