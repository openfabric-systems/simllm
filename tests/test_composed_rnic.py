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
    TRACKED_BYPASS_TOPOLOGY,
    _graph,
    _profile,
    _run_structural_cell,
    _write_bypass_topology,
)
from examples.rnic_live_v1.tier_c_acceptance import (
    check_observations as check_tier_c_observations,
)
from simllm.backends.composed_rnic import (
    ComposedRnicCell,
    ComposedRnicObservationError,
    ComposedRnicObservations,
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
TIER_C_EXPECTATIONS = EXPECTATIONS.with_name("tier_c_expectations.json")


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


def _packet_cell(
    payload_bytes: int,
    rate_gbps: int,
    doorbell_ps: int,
    wqe_count: int,
) -> ComposedRnicCell:
    service_ps = payload_bytes * 8 * 1000 // rate_gbps
    packet_service_ps = 4096 * 8 * 1000 // rate_gbps
    packet_count = payload_bytes // 4096
    wqes = []
    for ordinal in range(wqe_count):
        first_packet_ps = doorbell_ps + ordinal * service_ps
        packet_starts = tuple(
            first_packet_ps + packet_index * packet_service_ps
            for packet_index in range(packet_count)
        )
        terminal_ps = doorbell_ps + (ordinal + 1) * service_ps
        wqes.append(
            ComposedWqeObservation(
                ordinal=ordinal,
                native_wqe_id=ordinal + 1,
                eligible_at_ps=doorbell_ps,
                network_started_at_ps=first_packet_ps,
                network_finished_at_ps=terminal_ps,
                completed_at_ps=terminal_ps,
                network_accepted_at_ps=first_packet_ps,
                first_packet_at_ps=packet_starts[0],
                last_packet_at_ps=packet_starts[-1],
                packet_tx_started_at_ps=packet_starts,
            )
        )
    return ComposedRnicCell(
        payload_bytes=payload_bytes,
        link_rate_gbps=rate_gbps,
        doorbell_service_ps=doorbell_ps,
        wqes=tuple(wqes),
        jct_ps=doorbell_ps + wqe_count * service_ps,
        network_abi_version=2,
    )


def _native_packet_cell(
    payload_bytes: int,
    rate_gbps: int,
    doorbell_ps: int,
    wqe_count: int,
) -> dict[str, object]:
    cell = _packet_cell(payload_bytes, rate_gbps, doorbell_ps, wqe_count)
    packet_events = []
    wqes = []
    for wqe in cell.wqes:
        for packet_index, started_at_ps in enumerate(wqe.packet_tx_started_at_ps):
            packet_events.append(
                {
                    "attempt_token": 1000 + packet_index,
                    "extent_token": wqe.native_wqe_id,
                    "wqe_id": wqe.native_wqe_id,
                    "event_kind": "packet_tx_started",
                    "event_time_ps": started_at_ps,
                    "extent_index": 0,
                    "packet_index": packet_index,
                    "transmission_attempt": 0,
                    "payload_offset_bytes": packet_index * 4096,
                    "payload_bytes": 4096,
                    "wire_bytes": 4096,
                    "packet_kind": "data",
                }
            )
        wqes.append(
            {
                "ordinal": wqe.ordinal,
                "wqe_id": wqe.native_wqe_id,
                "eligible_at_ps": wqe.eligible_at_ps,
                "network_accepted_at_ps": wqe.network_accepted_at_ps,
                "port_tx_at_ps": wqe.first_packet_at_ps,
                "terminal_kind": "delivered",
                "terminal_at_ps": wqe.network_finished_at_ps,
                "cqe_status": "success",
                "cqe_visible_at_ps": wqe.completed_at_ps,
                "polled_at_ps": wqe.completed_at_ps,
                "first_packet_at_ps": wqe.first_packet_at_ps,
                "last_packet_at_ps": wqe.last_packet_at_ps,
                "first_rx_at_ps": wqe.network_finished_at_ps,
                "last_rx_at_ps": wqe.network_finished_at_ps,
            }
        )
    return {
        "payload_bytes": payload_bytes,
        "link_rate_gbps": rate_gbps,
        "doorbell_service_ps": doorbell_ps,
        "authority": {
            "mode": "structural",
            "native_session_constructed": 1,
            "native_posts": wqe_count,
            "legacy_ledger_constructed": 0,
            "legacy_posts": 0,
            "legacy_mutations": 0,
        },
        "device": {
            "counters": {
                "posted_wqes": wqe_count,
                "network_accepted": wqe_count,
                "network_delivered": wqe_count,
                "network_dropped": 0,
            },
            "has_pending_physical_work": False,
            "occupied_sq_entries": 0,
            "completion_queue_depth": 0,
            "unpublished_wqes": 0,
            "fatal": False,
        },
        "port": {
            "issued": [{} for _ in range(wqe_count)],
            "terminals": [{} for _ in range(wqe_count)],
            "live_tokens": [],
            "packet_events": packet_events,
        },
        "wqes": wqes,
        "cqe_order": list(range(1, wqe_count + 1)),
        "jct_ps": cell.jct_ps,
    }


def _native_packet_observations() -> dict[str, object]:
    return {
        "schema": "simllm-rnic-tier-a-observations-v2",
        "factory": "htsim",
        "network_abi_version": 2,
        "single_wqe": [
            _native_packet_cell(payload, rate, doorbell, 1)
            for payload in (4096, 1_048_576)
            for rate in (200, 400)
            for doorbell in (0, 1000)
        ],
        "fifo": [
            _native_packet_cell(4096, rate, doorbell, 2)
            for rate in (200, 400)
            for doorbell in (0, 1000)
        ],
    }


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


def _tier_c_observations() -> dict[str, object]:
    return {
        "schema": "simllm-rnic-tier-c-observations-v1",
        "factory": "htsim",
        "network_abi_version": 2,
        "simllm_base_commit": FROZEN_SIMLLM_BASE,
        "structural_single_wqe": [
            _run_structural_cell(
                _packet_cell(payload, rate, doorbell, 1),
                include_packet_timeline=True,
                session_prefix="tier-c-test",
            )
            for payload in (4096, 1_048_576)
            for rate in (200, 400)
            for doorbell in (0, 1000)
        ],
        "structural_fifo": [
            _run_structural_cell(
                _packet_cell(4096, rate, doorbell, 2),
                include_packet_timeline=True,
                session_prefix="tier-c-test",
            )
            for rate in (200, 400)
            for doorbell in (0, 1000)
        ],
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


def test_composed_abi_v2_parser_uses_explicit_packet_tx_starts():
    observations = ComposedRnicObservations.from_json(
        _native_packet_observations()
    )

    cell = observations.single_wqe[(1_048_576, 400, 1000)]
    wqe = cell.wqes[0]

    assert observations.network_abi_version == 2
    assert len(wqe.packet_tx_started_at_ps) == 256
    assert wqe.network_accepted_at_ps == 1000
    assert wqe.first_packet_at_ps == 1000
    assert wqe.last_packet_at_ps == 20_890_600
    assert wqe.network_started_at_ps == wqe.first_packet_at_ps
    assert wqe.network_finished_at_ps == 20_972_520


def test_composed_abi_v2_parser_rejects_missing_packet_tx_start():
    raw = _native_packet_observations()
    raw["single_wqe"][0]["port"]["packet_events"] = []

    with pytest.raises(
        ComposedRnicObservationError,
        match="no explicit data TX-start event",
    ):
        ComposedRnicObservations.from_json(raw)


def test_tier_c_packet_timeline_reaches_the_live_request_metrics():
    observations = _tier_c_observations()
    expectations = json.loads(TIER_C_EXPECTATIONS.read_text(encoding="utf-8"))

    report = check_tier_c_observations(
        observations,
        expectations,
        _tier_b_observations(),
        bypass_binary_hashes=_bypass_binary_hashes(),
    )

    assert report["passed"] is True
    assert {
        name: family["genuine_risk_fraction"]
        for name, family in report["behavioral_families"].items()
    } == {
        "doorbell_packet_to_live_chain": "4/4",
        "link_rate_packet_to_live_chain": "4/4",
    }
    assert all(report["fatal_unscored_invariants"].values())
    assert all(report["negative_controls"].values())
    assert report["entailment_analysis"][
        "origin_guard_entails_scored_relations"
    ] is False

    cell = next(
        row
        for row in observations["structural_single_wqe"]
        if (
            row["payload_bytes"],
            row["link_rate_gbps"],
            row["doorbell_service_ps"],
        )
        == (1_048_576, 400, 1000)
    )
    step = cell["steps"][0]
    release = step["execution_graph"]["released_at_ps"]
    wqe = step["runtime_report"]["wqes"][0]
    started = [
        event
        for event in step["completion_events"]
        if event["subject_object_id"] == wqe["wqe_id"]
        and event["phase"] == "started"
    ]

    assert wqe["network_accepted_at_ps"] == release + 1000
    assert wqe["first_packet_at_ps"] == release + 1000
    assert wqe["last_packet_at_ps"] == release + 20_890_600
    assert wqe["network_finished_at_ps"] == release + 20_972_520
    assert started[0]["timestamp_ps"] == wqe["first_packet_at_ps"]
    assert step["step_result"]["step_latency_ps"] == 20_972_520
    assert step["step_result"]["request_metrics"][0]["ttft_ps"] == 20_972_520


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


def test_bypass_topology_accepts_machine_local_source(tmp_path, monkeypatch):
    source = tmp_path / "leaf_spine_tiny.topo"
    source.write_text(
        "Nodes 32\n"
        "Downlink_speed_Gbps 100\n"
        "Downlink_speed_Gbps 100\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SIMLLM_TIER_B_BYPASS_TOPOLOGY", str(source))

    topology = _write_bypass_topology(tmp_path)
    text = topology.read_text(encoding="utf-8")

    assert text.count("Downlink_speed_Gbps 400") == 2
    assert "Downlink_speed_Gbps 100" not in text
    assert "Nodes 32" in text


def test_bypass_topology_retains_the_tiny_shape_at_the_frozen_link_rate(tmp_path):
    if not TRACKED_BYPASS_TOPOLOGY.is_file():
        pytest.skip(
            "tracked bypass topology lives in the private htsim submodule, "
            "which CI checkouts do not initialize"
        )
    topology = _write_bypass_topology(tmp_path)
    text = topology.read_text(encoding="utf-8")

    assert topology.name == "leaf_spine_tiny_400g.topo"
    assert text.count("Downlink_speed_Gbps 400") == 2
    assert "Downlink_speed_Gbps 100" not in text
    assert "Nodes 32" in text
