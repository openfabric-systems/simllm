from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rnic_live_v1.tier_b_acceptance import (
    TierBAcceptanceError,
    check_observations,
)
from examples.rnic_live_v1.tier_b_producer import (
    BYPASS_PROFILES,
    FROZEN_SIMLLM_BASE,
    TIER_B_OBSERVATION_SCHEMA,
    _graph,
    _profile,
    _run_structural_cell,
    _write_bypass_topology,
)
from simllm.backends.composed_rnic import (
    ComposedRnicCell,
    ComposedRnicSession,
    ComposedWqeObservation,
)
from simllm.core import CoarseDeviceRuntime, RnicAuthorityMode

EXPECTATIONS = (
    REPO_ROOT
    / "examples"
    / "rnic_live_v1"
    / "tier_b_review_expectations.json"
)


def _bypass_binary_hashes() -> dict[str, tuple[str, str]]:
    return {profile: ("1" * 64, "2" * 64) for profile in BYPASS_PROFILES}


def _cell(
    payload_bytes: int,
    rate_gbps: int,
    doorbell_ps: int,
    wqe_count: int,
) -> ComposedRnicCell:
    service_ps = payload_bytes * 8 * 1000 // rate_gbps
    wqes = tuple(
        ComposedWqeObservation(
            ordinal=ordinal,
            native_wqe_id=ordinal + 1,
            eligible_at_ps=doorbell_ps,
            network_started_at_ps=doorbell_ps + ordinal * service_ps,
            network_finished_at_ps=doorbell_ps + (ordinal + 1) * service_ps,
            completed_at_ps=doorbell_ps + (ordinal + 1) * service_ps,
        )
        for ordinal in range(wqe_count)
    )
    return ComposedRnicCell(
        payload_bytes=payload_bytes,
        link_rate_gbps=rate_gbps,
        doorbell_service_ps=doorbell_ps,
        wqes=wqes,
        jct_ps=doorbell_ps + wqe_count * service_ps,
    )


def _bypass_row(profile: str) -> dict[str, object]:
    artifacts = {
        "completion_csv_hex": "00",
        "canonical_completion_rows": [[profile, 1], ["jct_ps", 1]],
        "step_result_tuples": [[0, 1, 7001], [1, 1, 7002], [2, 1, 7003]],
        "replay_request_summary": [
            "tier-b-request",
            [7001, 7002, 7003],
            1,
            [1, 1],
        ],
    }
    return {
        "profile": profile,
        "hardware_mode": "bypass",
        "authority": "AtlahsWqeLedger",
        "inputs": {
            "goal_text_hex": "00",
            "goal_binary_hex": "00",
            "topology_hex": "00" if profile == "dcqcn" else "",
            "seed": 1,
            "baseline_argv": ["-rnic_profile", profile],
        },
        "reference_artifacts": artifacts,
        "candidate_artifacts": dict(artifacts),
    }


def _tier_b_observations() -> dict[str, object]:
    structural_single = [
        _run_structural_cell(_cell(payload, rate, doorbell, 1))
        for payload in (4096, 1_048_576)
        for rate in (200, 400)
        for doorbell in (0, 1000)
    ]
    structural_fifo = [
        _run_structural_cell(_cell(4096, rate, doorbell, 2))
        for rate in (200, 400)
        for doorbell in (0, 1000)
    ]
    return {
        "schema": TIER_B_OBSERVATION_SCHEMA,
        "factory": "htsim",
        "simllm_base_commit": FROZEN_SIMLLM_BASE,
        "structural_single_wqe": structural_single,
        "structural_fifo": structural_fifo,
        "bypass": [_bypass_row(profile) for profile in BYPASS_PROFILES],
    }
def test_composed_projection_passes_the_complete_frozen_tier_b_checker():
    observations = _tier_b_observations()
    expectations = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))

    report = check_observations(
        observations,
        expectations,
        bypass_binary_hashes=_bypass_binary_hashes(),
    )

    assert report["passed"] is True
    assert report["doorbell_owner"] == "nic_owner"
    assert {
        name: family["genuine_risk_fraction"]
        for name, family in report["behavioral_families"].items()
    } == {
        "single_wqe_d_additivity": "4/4",
        "single_wqe_inverse_rate": "4/4",
        "single_wqe_metric_forms": "8/8",
        "single_wqe_component_rows": "8/8",
        "two_wqe_fifo": "4/4",
        "bypass_artifact_identity": "4/4",
    }
    assert all(report["fatal_unscored_invariants"].values())
    assert all(report["negative_controls"].values())


@pytest.mark.parametrize(
    "artifact,empty",
    [
        ("completion_csv_hex", ""),
        ("canonical_completion_rows", []),
        ("step_result_tuples", []),
        ("replay_request_summary", []),
    ],
)
def test_tier_b_bypass_artifacts_must_be_nonempty(artifact, empty):
    observations = _tier_b_observations()
    for side in ("reference_artifacts", "candidate_artifacts"):
        observations["bypass"][0][side][artifact] = empty
    expectations = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))

    with pytest.raises(TierBAcceptanceError, match="empty|must not be empty"):
        check_observations(
            observations,
            expectations,
            bypass_binary_hashes=_bypass_binary_hashes(),
        )


def test_tier_b_bypass_binary_hashes_must_distinguish_the_executables():
    observations = _tier_b_observations()
    expectations = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))
    hashes = _bypass_binary_hashes()
    hashes["rnic-nn"] = ("3" * 64, "3" * 64)

    with pytest.raises(TierBAcceptanceError, match="binary hashes must differ"):
        check_observations(
            observations,
            expectations,
            bypass_binary_hashes=hashes,
        )


def test_tier_b_fifo_completion_order_is_fatal_and_unscored():
    observations = _tier_b_observations()
    expectations = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))
    wqes = observations["structural_fifo"][0]["steps"][0]["runtime_report"]["wqes"]
    wqes[0]["cq_post_sequence"] = 2
    wqes[1]["cq_post_sequence"] = 1

    with pytest.raises(TierBAcceptanceError, match="FIFO W0 to W1 completion order"):
        check_observations(
            observations,
            expectations,
            bypass_binary_hashes=_bypass_binary_hashes(),
        )


def test_composed_projection_transaction_aborts_without_consuming_native_evidence():
    cell = _cell(4096, 400, 1000, 2)
    session = ComposedRnicSession(cell, session_id="atomic-test")
    runtime = CoarseDeviceRuntime(
        _profile(400),
        authority_mode=RnicAuthorityMode.STRUCTURAL,
        native_session=session,
    )

    with pytest.raises(ValueError, match="consume every WQE"):
        runtime.execute(_graph(0, 7000, 4096, 1))

    assert session.committed_transactions == 0
    assert session.committed_wqes == 0
    assert runtime.last_report is None

    result = runtime.execute(_graph(0, 7000, 4096, 2))

    assert result.completed_at_ps == 7000 + 1000 + 2 * 81_920
    assert session.committed_transactions == 1
    assert session.committed_wqes == 2
    assert runtime.last_report is not None
    assert [wqe.sq_post_sequence for wqe in runtime.last_report.wqes] == [1, 2]
    assert [visit.stage for visit in runtime.last_report.visits].count(
        "native_doorbell"
    ) == 2
    assert [visit.stage for visit in runtime.last_report.visits].count(
        "native_network"
    ) == 2


def test_bypass_topology_retains_the_tiny_shape_at_the_frozen_link_rate(tmp_path):
    topology = _write_bypass_topology(tmp_path)
    text = topology.read_text(encoding="utf-8")

    assert topology.name == "leaf_spine_tiny_400g.topo"
    assert text.count("Downlink_speed_Gbps 400") == 2
    assert "Downlink_speed_Gbps 100" not in text
    assert "Nodes 32" in text
