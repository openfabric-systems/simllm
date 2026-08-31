import csv
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_rnic_comparison_v2"
LEGACY = ROOT / "examples" / "nvlink_rnic_comparison_v1"


def _load() -> dict[str, object]:
    return json.loads((STUDY / "results.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _by_key(rows: list[dict[str, object]]) -> dict[tuple[str, int, int], dict[str, object]]:
    return {
        (str(row["transport"]), int(row["degree"]), int(row["size_bytes"])): row
        for row in rows
    }


def test_result_is_complete_and_keeps_evidence_classes_separate():
    result = _load()
    hypotheses = {row["id"]: row for row in result["hypothesis_verdicts"]}

    assert result["schema"] == "simllm-nvlink-rnic-comparison-result-v2"
    assert result["study_verdict"] == "PASS_WITH_HONEST_REFUTATIONS"
    assert result["fatal_guard_verdict"] == "PASS"
    assert result["maximum_fluid_oracle_error_ps"] == 0
    assert result["sample_count"] == 77_112
    assert result["adapter_invocations"] == 378
    assert [hypotheses[name]["verdict"] for name in ("H1", "H2", "H3", "H4", "H5")] == [
        "PASS",
        "REFUTED",
        "REFUTED",
        "REFUTED",
        "PASS",
    ]
    assert (hypotheses["H5"]["passed_instances"], hypotheses["H5"]["total_instances"]) == (
        126,
        126,
    )
    assert hypotheses["H5"]["evidence_class"] == "EXACT/FATAL"
    assert result["honest_refutations"] == ["H2", "H3", "H4"]


def test_mapping_audit_explains_the_legacy_degree_three_shift():
    result = _load()
    audit = result["mapping_audit"]
    tail = _by_key(result["tail_metrics"])
    legacy = audit["legacy_degree_3_512k_p50_ps"]

    assert [row["legacy_receiver_capacity_bytes_per_second"] for row in audit["rows"]] == [
        100_000_000_000,
        200_000_000_000,
        207_101_921_876,
    ]
    assert [row["binding_capacity"] for row in audit["rows"]] == [
        "ordered-pair class cap",
        "ordered-pair class cap",
        "RX ingress plateau",
    ]
    assert audit["degree_3_aggregate_below_nvlink"] is False
    assert audit["degree_3_aggregate_capacity_ratio"] == 1.0
    assert legacy["observed_ratio"] == pytest.approx(1.66455305035)
    corrected = float(tail[("rnic-nn", 3, 524_288)]["p50_seed_mean_ps"])
    corrected_nvlink = float(
        tail[("nvlink-credit", 3, 524_288)]["p50_seed_mean_ps"]
    )
    corrected_ratio = float(legacy["rnic_nn"]) / corrected
    assert corrected == pytest.approx(18_120_617.0)
    assert corrected <= corrected_nvlink
    assert corrected_ratio == pytest.approx(601 / 360, rel=0.05)
    assert audit["verdict"] == (
        "MAPPING_DEFICIT_IN_FAIR_SHARE_ENTITY_NOT_CAPACITY_VALUE"
    )


def test_tail_and_fairness_tables_cover_every_frozen_cell():
    result = _load()
    expected = {
        (transport, degree, size_bytes)
        for transport in ("nvlink-credit", "rnic-nn", "rnic-nn-fluid")
        for degree in (1, 2, 3, 4, 8, 16)
        for size_bytes in (256, 1024, 4096, 16_384, 65_536, 262_144, 524_288)
    }
    tail = _by_key(result["tail_metrics"])
    fairness = _by_key(result["fairness_metrics"])

    assert set(tail) == expected
    assert set(fairness) == expected
    assert len(tail) == len(fairness) == 126
    for row in tail.values():
        assert row["p50_seed_mean_ps"] <= row["p99_seed_mean_ps"]
        assert row["p99_seed_mean_ps"] <= row["worst_seed_mean_ps"]
    for row in fairness.values():
        assert 0 <= row["jain_seed_min"] <= row["jain_seed_max"] <= 1


def test_fluid_location_refutation_is_exactly_the_short_packet_slot_cases():
    tail = _by_key(_load()["tail_metrics"])
    misses = []
    for degree in (1, 2, 3, 4, 8, 16):
        for size_bytes in (256, 1024, 4096, 16_384, 65_536, 262_144, 524_288):
            fluid = tail[("rnic-nn-fluid", degree, size_bytes)]
            for reference in ("nvlink-credit", "rnic-nn"):
                other = tail[(reference, degree, size_bytes)]
                for metric in ("p50", "p99", "worst"):
                    if fluid[f"{metric}_seed_mean_ps"] > other[f"{metric}_seed_mean_ps"]:
                        misses.append((degree, size_bytes, reference, metric))

    assert len(misses) == 13
    assert {reference for _, _, reference, _ in misses} == {"rnic-nn"}
    assert {size_bytes for _, size_bytes, _, _ in misses} <= {256, 1024}


def test_mesh_tail_wins_every_cell_but_never_grows_monotonically():
    tail = _by_key(_load()["tail_metrics"])
    strict_left = 0
    monotone = 0
    for size_bytes in (256, 1024, 4096):
        for reference in ("rnic-nn", "rnic-nn-fluid"):
            for metric in ("p99", "worst"):
                advantages = []
                for degree in (4, 8, 16):
                    nvlink = float(
                        tail[("nvlink-credit", degree, size_bytes)][
                            f"{metric}_seed_mean_ps"
                        ]
                    )
                    value = float(
                        tail[(reference, degree, size_bytes)][
                            f"{metric}_seed_mean_ps"
                        ]
                    )
                    strict_left += value < nvlink
                    advantages.append(nvlink / value)
                monotone += all(
                    advantages[index + 1] >= advantages[index] for index in range(2)
                )

    assert strict_left == 36
    assert monotone == 0


def test_mesh_fairness_scores_eleven_of_twenty_four():
    fairness = _by_key(_load()["fairness_metrics"])
    passed = 0
    for size_bytes in (256, 1024, 4096):
        nvlink = [
            float(fairness[("nvlink-credit", degree, size_bytes)]["jain_seed_mean"])
            for degree in (4, 8, 16)
        ]
        for reference in ("rnic-nn", "rnic-nn-fluid"):
            values = [
                float(fairness[(reference, degree, size_bytes)]["jain_seed_mean"])
                for degree in (4, 8, 16)
            ]
            passed += sum(value >= baseline for value, baseline in zip(values, nvlink))
            gaps = [value - baseline for value, baseline in zip(values, nvlink)]
            passed += all(gaps[index + 1] >= gaps[index] for index in range(2))

    assert passed == 11


def test_published_tables_figures_and_external_hashes_are_locked():
    result = _load()
    figure_artifacts = result["figure_artifacts"]

    assert len(figure_artifacts) == 10
    assert {Path(row["path"]).suffix for row in figure_artifacts} == {".pdf", ".png"}
    for row in figure_artifacts:
        path = STUDY / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert _sha256(path) == row["sha256"]
        if path.suffix == ".pdf":
            assert path.read_bytes().startswith(b"%PDF")
        else:
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for name in ("tail-metrics.csv", "fairness.csv"):
        assert _sha256(STUDY / name) == result["external_run_artifacts_sha256"][name]
        with (STUDY / name).open(encoding="utf-8", newline="") as handle:
            assert len(list(csv.DictReader(handle))) == 126


def test_legacy_tree_and_publication_disclosures_remain_locked():
    result = _load()
    report = (STUDY / "RESULTS.md").read_text(encoding="utf-8")
    traffic = (ROOT / "docs" / "modules" / "traffic.md").read_text(encoding="utf-8")
    ledger = json.loads((ROOT / "docs" / "task-ledger.json").read_text())

    assert result["authority"]["legacy_files_checked"] == 16
    for artifact in json.loads(
        (STUDY / "expectations.json").read_text(encoding="utf-8")
    )["preservation_lock"]["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.is_relative_to(LEGACY)
        assert path.stat().st_size == artifact["bytes"]
        assert _sha256(path) == artifact["sha256"]
    for text in (report, traffic):
        assert "LONG-FLOW ONLY" in text or "long-flow only" in text
        assert "NVSwitch-class" in text
        assert "model prediction" in text.lower()
        assert "\N{EM DASH}" not in text
    assert "TRAF-72" in ledger["closed"]
    open_tasks = traffic.partition("## Open tasks")[2]
    assert "- TRAF-72 " not in open_tasks
