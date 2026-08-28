import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "examples" / "nvlink_credit_arbitration_v1" / "run_study.py"


def _runner():
    spec = importlib.util.spec_from_file_location("nvlink_credit_arbitration_study", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("degree", "policy", "expected"),
    [
        (2, "release_aware_round_robin", [100.0, 60.0]),
        (3, "release_aware_round_robin", [87.101921876, 60.0, 60.0]),
        (2, "static_interleave", [60.0, 60.0]),
        (3, "static_interleave", [60.0, 60.0, 60.0]),
        (3, "greedy_capture", [100.0, 53.550960938, 53.550960938]),
    ],
)
def test_frozen_share_oracles_are_derived_before_the_run(degree, policy, expected):
    runner = _runner()

    assert runner.expected_wire_rates_gbps(
        degree=degree,
        policy=runner.NvlinkArbitrationPolicy(policy),
        receiver_rate_gbps=207.101921876,
    ) == pytest.approx(expected)


def test_preservation_digest_covers_all_merged_nvlink_study_files():
    runner = _runner()
    frozen = runner.load_expectations()

    assert runner.preservation_evidence(frozen) == {
        "tracked_file_count": 89,
        "tracked_bytes": 6_429_838,
        "path_content_digest_sha256": (
            "61af15faf7c7080f40a33f8f9d5503b3b0278f15be15997e90c6895cddf85c72"
        ),
        "candidate_profile_sha256": (
            "d33ef5b2c6fa87cc97e1e7b45a43a841a5da45f5462311e3981fbc903c56deb2"
        ),
    }


def test_legacy_identity_is_explicitly_pinned_to_static_interleave():
    runner = _runner()
    frozen = runner.load_expectations()

    assert runner._legacy_identity(ROOT / frozen["candidate"]["profile_path"]) == (
        "2f2af64619ed3c6341b209d877d9f1e6984a67e44b97b5eb176a157294a6c252"
    )


def test_aggregate_quantization_bound_sums_the_per_sender_allowances():
    runner = _runner()
    frozen = runner.load_expectations()
    row, _ = runner.simulation_row(
        frozen,
        degree=16,
        policy=runner.NvlinkArbitrationPolicy.RELEASE_AWARE_ROUND_ROBIN,
    )

    assert row["behavioral_verdict"] == "PASS"
    assert row["aggregate_tolerance_gbps"] == pytest.approx(
        16 * row["per_source_tolerance_gbps"]
    )
