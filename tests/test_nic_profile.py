"""NIC profile carrier: rendering, scaling, evidence and the gap ledger."""

from __future__ import annotations

import re
from dataclasses import fields, replace
from pathlib import Path

import pytest

from simllm.backends.nic_profile import (
    CX5_100G,
    CX7_400G,
    EVIDENCE_CLASSES,
    GAP_TASKS,
    MODEL_FIELDS,
    PROFILES,
    SCALED_FIELDS,
    NicProfile,
    dcqcn_flags,
    dcqcn_link_bps,
    gap_fields,
    scale_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKENDS_DOC = REPO_ROOT / "docs" / "modules" / "backends.md"
DCQCN_CLI = (
    REPO_ROOT / "third_party" / "htsim" / "htsim" / "sim" / "datacenter" / "dcqcn_atlahs_cli.cpp"
)


def _profile_kwargs(**overrides) -> dict:
    values = {field.name: getattr(CX5_100G, field.name) for field in fields(CX5_100G)}
    values["evidence"] = dict(CX5_100G.evidence)
    values.update(overrides)
    return values


# Rendering ------------------------------------------------------------------


def test_cx5_flags_render_the_measured_configuration():
    assert dcqcn_flags(CX5_100G) == {
        "-link_bps": "97000000000",
        "-max_wire_packet_bytes": "4096",
        "-data_header_bytes": "64",
        "-pfc": "off",
        "-recovery": "gbn",
        "-loss_rate_cut": "on",
        "-silent_rto_us": "67108",
        "-ecn_kmin_bytes": "102400",
        "-ecn_kmax_bytes": "409600",
        "-ecn_pmax_ppm": "250000",
    }


def test_cx7_flags_differ_only_in_the_scaled_rates():
    cx5 = dcqcn_flags(CX5_100G)
    cx7 = dcqcn_flags(CX7_400G)
    assert cx7["-link_bps"] == "388000000000"
    assert cx7["-ecn_kmin_bytes"] == "409600"
    assert cx7["-ecn_kmax_bytes"] == "1638400"
    unchanged = set(cx5) - {"-link_bps", "-ecn_kmin_bytes", "-ecn_kmax_bytes"}
    assert {flag: cx7[flag] for flag in unchanged} == {flag: cx5[flag] for flag in unchanged}


def test_rendered_rate_is_a_whole_gigabit_per_second():
    # The backend's fat-tree loader parses Downlink_speed_Gbps as a whole
    # number and the runtime rejects a topology whose rate differs from
    # -link_bps, so the rendered rate has to survive that round trip.
    for profile in PROFILES.values():
        rendered = dcqcn_link_bps(profile)
        assert rendered % 1_000_000_000 == 0
        assert int(dcqcn_flags(profile)["-link_bps"]) == rendered
        assert abs(rendered - profile.goodput_bps) <= 500_000_000


def test_flag_renderer_is_pure():
    first = dcqcn_flags(CX5_100G)
    first["-pfc"] = "on"
    assert dcqcn_flags(CX5_100G)["-pfc"] == "off"


@pytest.mark.skipif(not DCQCN_CLI.is_file(), reason="htsim submodule is not checked out")
def test_every_rendered_flag_is_accepted_by_the_backend_cli():
    accepted = set(re.findall(r'option == "(-[a-z0-9_]+)"', DCQCN_CLI.read_text()))
    assert accepted, "could not read the DCQCN option names from the backend CLI"
    for profile in PROFILES.values():
        unknown = sorted(set(dcqcn_flags(profile)) - accepted)
        assert not unknown, f"{profile.name} renders options the backend rejects: {unknown}"


# Scaling --------------------------------------------------------------------


def test_scaling_multiplies_exactly_the_rate_carrying_fields():
    for field_name in SCALED_FIELDS:
        assert getattr(CX7_400G, field_name) == 4 * getattr(CX5_100G, field_name)
    carried = [name for name in MODEL_FIELDS if name not in SCALED_FIELDS]
    for field_name in carried:
        assert getattr(CX7_400G, field_name) == getattr(CX5_100G, field_name)
    # The offset, the packetization and the recovery mode are the fields that
    # make the two profiles "the same architecture".
    assert "t_eff_ps" in carried
    assert "mtu_bytes" in carried
    assert "header_bytes" in carried
    assert "recovery" in carried


def test_scaled_ecn_thresholds_match_the_published_400g_row():
    # docs/papers/msg-size-vs-bandwidth.md, set D3: 400/1600 KB at 400G.
    assert CX7_400G.ecn_kmin_bytes == 400 * 1024
    assert CX7_400G.ecn_kmax_bytes == 1600 * 1024


def test_every_field_of_a_scaled_profile_is_declared():
    assert set(CX7_400G.evidence.values()) == {"declared"}
    assert set(CX7_400G.evidence) == set(MODEL_FIELDS)
    assert CX5_100G.name in CX7_400G.provenance
    assert "scaled by link factor 4" in CX7_400G.provenance


def test_scaling_by_one_changes_only_the_record_fields():
    same = scale_profile(CX5_100G, 1.0, name="cx5_copy")
    for field_name in MODEL_FIELDS:
        assert getattr(same, field_name) == getattr(CX5_100G, field_name)
    assert same.name == "cx5_copy"
    assert set(same.evidence.values()) == {"declared"}


def test_scaling_rejects_a_nonpositive_factor():
    with pytest.raises(ValueError):
        scale_profile(CX5_100G, 0.0)


def test_scaling_rejects_a_factor_that_does_not_give_whole_units():
    with pytest.raises(ValueError, match="whole number"):
        scale_profile(CX5_100G, 1.0 / 3.0)


# Evidence -------------------------------------------------------------------


def test_every_model_field_carries_a_known_evidence_class():
    for profile in PROFILES.values():
        assert set(profile.evidence) == set(MODEL_FIELDS)
        assert set(profile.evidence.values()) <= set(EVIDENCE_CLASSES)


def test_measured_profile_uses_every_measured_class():
    assert set(CX5_100G.evidence.values()) == {
        "documented",
        "driver-inferred",
        "calibrated-opaque",
        "declared",
    }
    # The endpoint's ECN state was unreadable on the measured deployment, so
    # those three are asserted rather than evidenced.
    assert CX5_100G.evidence["ecn_kmin_bytes"] == "declared"
    assert CX5_100G.evidence["ecn_kmax_bytes"] == "declared"
    assert CX5_100G.evidence["ecn_pmax_ppm"] == "declared"
    assert CX5_100G.evidence["goodput_bps"] == "calibrated-opaque"
    assert CX5_100G.evidence["t_eff_ps"] == "calibrated-opaque"


def test_provenance_names_the_campaign_records():
    for record in ("RESULTS-p4-kernels.md", "RESULTS-p5a-incast.md", "FINDINGS-cx5.md"):
        assert record in CX5_100G.provenance


def test_missing_evidence_entry_is_refused():
    evidence = dict(CX5_100G.evidence)
    del evidence["t_eff_ps"]
    with pytest.raises(ValueError, match="no evidence class"):
        NicProfile(**_profile_kwargs(evidence=evidence))


def test_evidence_for_a_non_model_field_is_refused():
    evidence = dict(CX5_100G.evidence)
    evidence["provenance"] = "documented"
    with pytest.raises(ValueError, match="non-model fields"):
        NicProfile(**_profile_kwargs(evidence=evidence))


def test_unknown_evidence_class_is_refused():
    evidence = dict(CX5_100G.evidence)
    evidence["mtu_bytes"] = "measured"
    with pytest.raises(ValueError, match="unknown evidence classes"):
        NicProfile(**_profile_kwargs(evidence=evidence))


def test_evidence_mapping_is_frozen():
    with pytest.raises(TypeError):
        CX5_100G.evidence["mtu_bytes"] = "declared"  # type: ignore[index]


# The gap ledger -------------------------------------------------------------


def test_gap_fields_are_exactly_the_fields_the_registry_tasks_cover():
    for profile in PROFILES.values():
        assert set(gap_fields(profile)) == set(GAP_TASKS)
    # The gap is the exact complement of what the command line carries: each
    # rendered flag carries one model field, and -link_bps carries goodput_bps.
    assert len(gap_fields(CX5_100G)) + len(dcqcn_flags(CX5_100G)) == len(MODEL_FIELDS)


def test_each_gap_task_is_registered_in_the_module_doc():
    doc = BACKENDS_DOC.read_text()
    for field_name, task in sorted(GAP_TASKS.items()):
        assert re.search(rf"^- {re.escape(task)} \(", doc, re.MULTILINE), (
            f"{field_name} points at {task}, which is not a registered task entry"
        )


def test_gap_fields_are_the_named_missing_mechanisms():
    assert set(gap_fields(CX5_100G)) == {
        "link_bps",
        "t_eff_ps",
        "sq_depth",
        "pps_ceiling_per_qp",
        "pps_ceiling_per_nic",
        "rx_ingress_meter_bytes",
    }


# Validation -----------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"link_bps": 0},
        {"link_bps": -1},
        {"mtu_bytes": 64, "header_bytes": 64},
        {"goodput_bps": 200_000_000_000},
        {"ecn_kmin_bytes": 500_000},
        {"ecn_pmax_ppm": 1_000_001},
        {"recovery": "selective"},
        {"name": ""},
        {"provenance": ""},
        {"rx_ingress_meter_bytes": 0},
        {"sq_depth": True},
    ],
)
def test_invalid_profiles_are_refused(overrides):
    with pytest.raises(ValueError):
        NicProfile(**_profile_kwargs(**overrides))


def test_profile_is_frozen_and_replaceable():
    with pytest.raises(AttributeError):
        CX5_100G.mtu_bytes = 1024  # type: ignore[misc]
    smaller = replace(CX5_100G, name="cx5_mtu1024", mtu_bytes=1024)
    assert smaller.mss_bytes == 960
    assert CX5_100G.mss_bytes == 4032


def test_profiles_registry_is_keyed_by_name():
    assert PROFILES["cx5_100g"] is CX5_100G
    assert PROFILES["cx7_400g"] is CX7_400G
    with pytest.raises(TypeError):
        PROFILES["other"] = CX5_100G  # type: ignore[index]
