"""Fabric profile carrier: rendering, the no-marking guard, evidence, constants."""

from __future__ import annotations

import re
from dataclasses import fields, replace
from pathlib import Path

import pytest

from simllm.backends.fabric_profile import (
    ECN_NONE,
    EVIDENCE_CLASSES,
    FABRIC_GAP_TASKS,
    FABRICS,
    HACC_LEAF_4X100G,
    MODEL_FIELDS,
    FabricProfile,
    dcqcn_port_gbps,
    fabric_gap_fields,
    render_dcqcn,
    render_topology,
)
from simllm.backends.nic_profile import CX5_100G, NicProfile

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKENDS_DOC = REPO_ROOT / "docs" / "modules" / "backends.md"
DCQCN_CLI = (
    REPO_ROOT / "third_party" / "htsim" / "htsim" / "sim" / "datacenter" / "dcqcn_atlahs_cli.cpp"
)


def _fabric_kwargs(**overrides) -> dict:
    values = {field.name: getattr(HACC_LEAF_4X100G, field.name) for field in fields(FabricProfile)}
    values["evidence"] = dict(HACC_LEAF_4X100G.evidence)
    values.update(overrides)
    return values


def _structural_lines(topology: str) -> list[str]:
    return [
        line.strip()
        for line in topology.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


# The frozen constants ---------------------------------------------------------


def test_hacc_profile_reproduces_the_measured_constants():
    assert HACC_LEAF_4X100G.name == "hacc_leaf_4x100g"
    assert HACC_LEAF_4X100G.switch_count == 1
    assert HACC_LEAF_4X100G.host_ports == 4
    assert HACC_LEAF_4X100G.port_bps == 100_000_000_000
    assert HACC_LEAF_4X100G.pipe_latency_ps == 515_000
    assert HACC_LEAF_4X100G.pipes_per_path == 4
    assert HACC_LEAF_4X100G.egress_buffer_bytes == 5_200_000
    assert HACC_LEAF_4X100G.ecn == ECN_NONE
    assert HACC_LEAF_4X100G.pfc_enabled is False
    assert HACC_LEAF_4X100G.pause_honoured is False
    assert HACC_LEAF_4X100G.ecmp_paths == 1


def test_measured_buffer_sits_inside_the_campaign_spread():
    # t_drop times excess gave 5.04 to 5.39 MB over 12 runs at three excess
    # rates; the profile carries the round number the record reports.
    assert 5_040_000 <= HACC_LEAF_4X100G.egress_buffer_bytes <= 5_390_000


def test_latency_floor_reproduces_the_measured_2b_anchor():
    # Four pipe traversals plus two store-and-forward serializations in each
    # direction, at the port rate: this is the arithmetic the topology's
    # per-hop latency was chosen to satisfy.
    propagation_ps = HACC_LEAF_4X100G.latency_floor_ps
    assert propagation_ps == 2_060_000
    serialization_ps = round(
        (2 * (66 + 64) * 8) / HACC_LEAF_4X100G.port_bps * 1e12
    )
    floor_us = (propagation_ps + serialization_ps) / 1e6
    assert abs(floor_us - 2.08) / 2.08 <= 0.15
    assert round(floor_us, 4) == 2.0808


def test_shared_pool_is_the_sum_of_the_per_port_pools():
    assert HACC_LEAF_4X100G.shared_buffer_bytes == 4 * 5_200_000
    assert HACC_LEAF_4X100G.shared_buffer_bytes >= HACC_LEAF_4X100G.egress_buffer_bytes


def test_provenance_names_the_campaign_records():
    for record in ("RESULTS-p6-fabric.md", "expectations-p6-fabric.md", "FINDINGS-cx5.md"):
        assert record in HACC_LEAF_4X100G.provenance
    for table in ("buffer.csv", "ecn_fit.txt", "dcqcn_summary.csv", "latency_matrix.csv"):
        assert table in HACC_LEAF_4X100G.provenance


def test_fabrics_registry_is_keyed_by_name():
    assert FABRICS["hacc_leaf_4x100g"] is HACC_LEAF_4X100G
    with pytest.raises(TypeError):
        FABRICS["other"] = HACC_LEAF_4X100G  # type: ignore[index]


# Rendering --------------------------------------------------------------------


def test_hacc_renders_the_measured_flag_vector():
    _, flags = render_dcqcn(CX5_100G, HACC_LEAF_4X100G)
    assert flags == {
        # The fabric owns the link, so the port rate wins over the NIC's
        # goodput asymptote.
        "-link_bps": "100000000000",
        "-max_wire_packet_bytes": "4096",
        "-data_header_bytes": "64",
        "-pfc": "off",
        "-recovery": "gbn",
        "-loss_rate_cut": "on",
        "-silent_rto_us": "67108",
        "-ecn_kmin_bytes": "5199998",
        "-ecn_kmax_bytes": "5199999",
        "-ecn_pmax_ppm": "1",
        "-egress_buffer_bytes": "5200000",
        "-shared_buffer_bytes": "20800000",
    }


def test_hacc_renders_the_one_leaf_topology():
    topology, _ = render_dcqcn(CX5_100G, HACC_LEAF_4X100G)
    assert _structural_lines(topology) == [
        "Nodes 4",
        "Tiers 2",
        "Podsize 4",
        "Tier 0",
        "Downlink_speed_Gbps 100",
        "Radix_Down 4",
        "Radix_Up 4",
        "Downlink_Latency_ns 515",
        "Switch_Latency_ns 0",
        "Tier 1",
        "Downlink_speed_Gbps 100",
        "Radix_Down 1",
        "Downlink_Latency_ns 515",
        "Switch_Latency_ns 0",
    ]
    assert "hacc_leaf_4x100g fabric profile" in topology


def test_rendered_geometry_satisfies_the_loader_constraint_chain():
    topology = render_topology(HACC_LEAF_4X100G)
    header, tier0, tier1 = re.split(r"^Tier \d$", topology, flags=re.MULTILINE)
    header_values = dict(re.findall(r"^(\w+) (\d+)$", header, re.MULTILINE))
    tier0_values = dict(re.findall(r"^(\w+) (\d+)$", tier0, re.MULTILINE))
    tier1_values = dict(re.findall(r"^(\w+) (\d+)$", tier1, re.MULTILINE))
    nodes = int(header_values["Nodes"])
    podsize = int(header_values["Podsize"])
    down = int(tier0_values["Radix_Down"])
    up = int(tier0_values["Radix_Up"])
    assert int(header_values["Tiers"]) == 2
    assert int(tier1_values["Radix_Down"]) == 1
    assert nodes % podsize == 0
    assert podsize % down == 0
    # Oversubscription 1 requires the leaf's two radices to match, and the
    # uplink count then has to equal the spine count times their down radix.
    assert down == up
    pods = nodes // podsize
    tor_uplinks = nodes
    assert tor_uplinks % pods == 0
    assert tor_uplinks // pods == up


def test_rendered_link_rate_matches_the_topology_rate():
    topology, flags = render_dcqcn(CX5_100G, HACC_LEAF_4X100G)
    rates = {
        int(match)
        for match in re.findall(r"^Downlink_speed_Gbps (\d+)", topology, re.MULTILINE)
    }
    assert rates == {dcqcn_port_gbps(HACC_LEAF_4X100G)}
    assert int(flags["-link_bps"]) == dcqcn_port_gbps(HACC_LEAF_4X100G) * 1_000_000_000


def test_renderer_is_pure():
    _, first = render_dcqcn(CX5_100G, HACC_LEAF_4X100G)
    first["-pfc"] = "on"
    _, second = render_dcqcn(CX5_100G, HACC_LEAF_4X100G)
    assert second["-pfc"] == "off"


def test_a_marking_fabric_renders_its_own_thresholds():
    marking = FabricProfile(
        **_fabric_kwargs(
            name="marking_leaf",
            ecn={"kmin_bytes": 102_400, "kmax_bytes": 409_600, "pmax_ppm": 250_000},
        )
    )
    _, flags = render_dcqcn(CX5_100G, marking)
    assert flags["-ecn_kmin_bytes"] == "102400"
    assert flags["-ecn_kmax_bytes"] == "409600"
    assert flags["-ecn_pmax_ppm"] == "250000"


def test_a_smaller_buffer_arm_moves_the_thresholds_with_it():
    smaller = FabricProfile(**_fabric_kwargs(name="hacc_half_buffer", egress_buffer_bytes=2_600_000))
    _, flags = render_dcqcn(CX5_100G, smaller)
    assert flags["-egress_buffer_bytes"] == "2600000"
    assert flags["-shared_buffer_bytes"] == "10400000"
    assert flags["-ecn_kmin_bytes"] == "2599998"
    assert flags["-ecn_kmax_bytes"] == "2599999"


@pytest.mark.skipif(not DCQCN_CLI.is_file(), reason="htsim submodule is not checked out")
def test_every_rendered_flag_is_accepted_by_the_backend_cli():
    accepted = set(re.findall(r'option == "(-[a-z0-9_]+)"', DCQCN_CLI.read_text()))
    assert accepted, "could not read the DCQCN option names from the backend CLI"
    for fabric in FABRICS.values():
        _, flags = render_dcqcn(CX5_100G, fabric)
        assert not sorted(set(flags) - accepted)


# The no-marking realisation ---------------------------------------------------


def test_no_marking_thresholds_clear_the_runtime_configuration_guard():
    # validate_config in the DCQCN runtime refuses a zero Pmax and refuses
    # Kmax at or above the egress buffer, so this is the whole legal chain.
    _, flags = render_dcqcn(CX5_100G, HACC_LEAF_4X100G)
    kmin = int(flags["-ecn_kmin_bytes"])
    kmax = int(flags["-ecn_kmax_bytes"])
    pmax = int(flags["-ecn_pmax_ppm"])
    egress = int(flags["-egress_buffer_bytes"])
    shared = int(flags["-shared_buffer_bytes"])
    assert 0 <= kmin < kmax < egress <= shared
    assert 0 < pmax <= 1_000_000


def test_no_marking_band_is_the_last_two_bytes_at_the_lowest_probability():
    _, flags = render_dcqcn(CX5_100G, HACC_LEAF_4X100G)
    egress = int(flags["-egress_buffer_bytes"])
    # The RED predicate marks nothing at or below Kmin, so the reachable
    # marking band is the occupancy range above Kmin, and it is two bytes wide.
    assert egress - int(flags["-ecn_kmin_bytes"]) == 2
    assert int(flags["-ecn_pmax_ppm"]) == 1


def test_a_marking_threshold_at_or_above_the_buffer_is_refused():
    with pytest.raises(ValueError, match="below egress_buffer_bytes"):
        FabricProfile(
            **_fabric_kwargs(
                ecn={"kmin_bytes": 1024, "kmax_bytes": 5_200_000, "pmax_ppm": 250_000}
            )
        )


def test_a_zero_marking_probability_is_refused():
    with pytest.raises(ValueError, match="pmax_ppm"):
        FabricProfile(
            **_fabric_kwargs(ecn={"kmin_bytes": 1024, "kmax_bytes": 4096, "pmax_ppm": 0})
        )


def test_a_buffer_too_small_for_the_threshold_pair_is_refused():
    tiny = FabricProfile(**_fabric_kwargs(name="tiny", egress_buffer_bytes=2))
    with pytest.raises(ValueError, match="no-marking threshold pair"):
        render_dcqcn(CX5_100G, tiny)


@pytest.mark.parametrize(
    "ecn",
    [
        "off",
        {"kmin_bytes": 1024},
        {"kmin_bytes": 1024, "kmax_bytes": 4096, "pmax_ppm": 1, "extra": 1},
        {"kmin_bytes": 4096, "kmax_bytes": 1024, "pmax_ppm": 1},
        {"kmin_bytes": 1024, "kmax_bytes": 4096, "pmax_ppm": 1_000_001},
        {"kmin_bytes": -1, "kmax_bytes": 4096, "pmax_ppm": 1},
    ],
)
def test_malformed_ecn_records_are_refused(ecn):
    with pytest.raises(ValueError):
        FabricProfile(**_fabric_kwargs(ecn=ecn))


# The gap ledger ---------------------------------------------------------------


def test_a_drop_only_fabric_has_exactly_one_gap():
    assert fabric_gap_fields(HACC_LEAF_4X100G) == ("ecn",)
    assert set(fabric_gap_fields(HACC_LEAF_4X100G)) == set(FABRIC_GAP_TASKS)


def test_a_marking_fabric_has_no_gap():
    marking = FabricProfile(
        **_fabric_kwargs(
            name="marking_leaf",
            ecn={"kmin_bytes": 102_400, "kmax_bytes": 409_600, "pmax_ppm": 250_000},
        )
    )
    assert fabric_gap_fields(marking) == ()


def test_each_gap_task_is_registered_in_the_module_doc():
    doc = BACKENDS_DOC.read_text()
    for field_name, task in sorted(FABRIC_GAP_TASKS.items()):
        assert re.search(rf"^- {re.escape(task)} \(", doc, re.MULTILINE), (
            f"{field_name} points at {task}, which is not a registered task entry"
        )


# Evidence ---------------------------------------------------------------------


def test_every_fabric_field_carries_a_known_evidence_class():
    for fabric in FABRICS.values():
        assert set(fabric.evidence) == set(MODEL_FIELDS)
        assert set(fabric.evidence.values()) <= set(EVIDENCE_CLASSES)


def test_the_measured_fabric_is_inferred_throughout():
    # No switch counter was readable, so nothing here is documented and
    # nothing is merely declared.
    assert set(HACC_LEAF_4X100G.evidence.values()) == {"inferred"}


def test_missing_evidence_entry_is_refused():
    evidence = dict(HACC_LEAF_4X100G.evidence)
    del evidence["egress_buffer_bytes"]
    with pytest.raises(ValueError, match="no evidence class"):
        FabricProfile(**_fabric_kwargs(evidence=evidence))


def test_evidence_for_a_non_model_field_is_refused():
    evidence = dict(HACC_LEAF_4X100G.evidence)
    evidence["provenance"] = "inferred"
    with pytest.raises(ValueError, match="non-model fields"):
        FabricProfile(**_fabric_kwargs(evidence=evidence))


def test_unknown_evidence_class_is_refused():
    evidence = dict(HACC_LEAF_4X100G.evidence)
    evidence["port_bps"] = "measured"
    with pytest.raises(ValueError, match="unknown evidence classes"):
        FabricProfile(**_fabric_kwargs(evidence=evidence))


def test_evidence_mapping_is_frozen():
    with pytest.raises(TypeError):
        HACC_LEAF_4X100G.evidence["port_bps"] = "declared"  # type: ignore[index]


def test_a_declared_sensitivity_arm_keeps_its_own_evidence():
    evidence = dict(HACC_LEAF_4X100G.evidence)
    evidence["egress_buffer_bytes"] = "declared"
    smaller = FabricProfile(
        **_fabric_kwargs(
            name="hacc_half_buffer", egress_buffer_bytes=2_600_000, evidence=evidence
        )
    )
    assert smaller.evidence["egress_buffer_bytes"] == "declared"
    assert smaller.evidence["port_bps"] == "inferred"


# Validation -------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"switch_count": 0},
        {"host_ports": -1},
        {"port_bps": 0},
        {"pipe_latency_ps": 0},
        {"pipes_per_path": 0},
        {"egress_buffer_bytes": 0},
        {"ecmp_paths": 0},
        {"ecmp_paths": True},
        {"name": ""},
        {"provenance": ""},
        {"pfc_enabled": True, "pause_honoured": False},
    ],
)
def test_invalid_fabrics_are_refused(overrides):
    with pytest.raises(ValueError):
        FabricProfile(**_fabric_kwargs(**overrides))


def test_profile_is_frozen_and_replaceable():
    with pytest.raises(AttributeError):
        HACC_LEAF_4X100G.port_bps = 1  # type: ignore[misc]
    smaller = replace(HACC_LEAF_4X100G, name="hacc_half_buffer", egress_buffer_bytes=2_600_000)
    assert smaller.shared_buffer_bytes == 10_400_000
    assert HACC_LEAF_4X100G.egress_buffer_bytes == 5_200_000


def test_a_fractional_gigabit_port_rate_cannot_be_rendered():
    odd = FabricProfile(**_fabric_kwargs(name="odd_rate", port_bps=97_100_000_000))
    with pytest.raises(ValueError, match="whole Gb/s"):
        dcqcn_port_gbps(odd)


def test_a_fractional_nanosecond_pipe_latency_cannot_be_rendered():
    odd = FabricProfile(**_fabric_kwargs(name="odd_latency", pipe_latency_ps=515_500))
    with pytest.raises(ValueError, match="whole nanosecond"):
        render_topology(odd)


def test_a_multi_switch_fabric_does_not_render_onto_the_two_tier_clos():
    multi = FabricProfile(**_fabric_kwargs(name="two_leaves", switch_count=2))
    with pytest.raises(ValueError, match="single-switch fabric"):
        render_topology(multi)


def test_path_diversity_a_one_leaf_geometry_cannot_provide_is_refused():
    multipath = FabricProfile(**_fabric_kwargs(name="multipath", ecmp_paths=4))
    with pytest.raises(ValueError, match="one path"):
        render_topology(multipath)


def test_a_pfc_disagreement_between_the_two_ends_is_refused():
    pfc_nic = NicProfile(
        **{
            **{field.name: getattr(CX5_100G, field.name) for field in fields(CX5_100G)},
            "name": "cx5_pfc_on",
            "pfc_enabled": True,
            "evidence": dict(CX5_100G.evidence),
        }
    )
    with pytest.raises(ValueError, match="one -pfc flag"):
        render_dcqcn(pfc_nic, HACC_LEAF_4X100G)
