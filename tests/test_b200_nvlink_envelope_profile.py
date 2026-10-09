"""TRAF-31: the first-party B200 NVLink profile and the freeze's identity guards.

The constants here are the ones the scorer derived from the stage 1 result of
record under ``examples/b200_nvlink_envelope_v1/measurements``. The identity
checks are cell E8 of the freeze: the profile the study did not touch still
resolves to its own constants, and the five PLACE-13 reference manifests still
hash to the digests the freeze recorded before the study ran.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simllm.placement import declared_manifest, declared_pipeline_placement
from simllm.traffic import (
    B200_NCCL_2_27_LOCAL_FIRSTPARTY_PROFILE,
    B200_NCCL_2_27_LOCAL_PROFILE,
    resolve_collective_latency_profile,
)

STUDY = Path(__file__).resolve().parents[1] / "examples/b200_nvlink_envelope_v1"
PROFILE_ID = "b200-nccl-2.27-local-firstparty-v1"
BANDWIDTH_BYTES_PER_SECOND = 76_201_055_302
INTERCEPT_PS = {2: 9_169_338, 4: 12_827_518, 8: 23_092_555}
BAND_PS = {
    2: (8_152_855, 10_837_880),
    4: (10_301_763, 14_078_330),
    8: (16_949_257, 24_958_747),
}
#: the 4 KiB holdout of cell E5 per width, with endpoint bytes 2(W-1)S/W
HOLDOUT_ENDPOINT_BYTES = {2: 4_096, 4: 6_144, 8: 7_168}
SERVICE_AT_4KIB_PS = {2: 9_223_091, 4: 12_908_147, 8: 23_186_622}


def _expectations() -> dict:
    return json.loads((STUDY / "expectations.json").read_text(encoding="utf-8"))


def test_the_first_party_profile_carries_the_measured_constants():
    profile = B200_NCCL_2_27_LOCAL_FIRSTPARTY_PROFILE

    assert profile.profile_id == PROFILE_ID
    assert profile.bandwidth_bytes_per_second == BANDWIDTH_BYTES_PER_SECOND
    assert profile.participant_latency_ps == tuple(sorted(INTERCEPT_PS.items()))
    assert profile.supported_participant_counts == (2, 4, 8)
    assert profile.propagation_reference_ps == 2_000_000
    assert profile.endpoint_byte_bounds(2) == (8, 262_144)
    assert profile.endpoint_byte_bounds(4) == (12, 393_216)
    assert profile.endpoint_byte_bounds(8) == (14, 458_752)
    for width, band in BAND_PS.items():
        assert profile.base_latency_band_ps(width) == band
        assert band[0] <= INTERCEPT_PS[width] <= band[1]
    assert profile.evidence_class == "calibrated"


def test_the_first_party_profile_resolves_by_name():
    assert resolve_collective_latency_profile(PROFILE_ID) is (
        B200_NCCL_2_27_LOCAL_FIRSTPARTY_PROFILE
    )


@pytest.mark.parametrize("width", [3, 6, 16, 64])
def test_the_first_party_profile_refuses_every_width_it_did_not_measure(width):
    with pytest.raises(ValueError) as raised:
        B200_NCCL_2_27_LOCAL_FIRSTPARTY_PROFILE.base_latency_ps(width)
    message = str(raised.value)
    assert PROFILE_ID in message
    assert f"does not support participant count {width}" in message
    assert "supported counts are 2, 4, 8" in message


@pytest.mark.parametrize("width", [2, 4, 8])
def test_the_first_party_profile_charges_the_scored_holdout_service(width):
    profile = B200_NCCL_2_27_LOCAL_FIRSTPARTY_PROFILE
    # The 4 KiB holdout of cell E5, whose endpoint load is 2(W-1)S/W.
    endpoint = HOLDOUT_ENDPOINT_BYTES[width]
    assert endpoint == 2 * (width - 1) * 4_096 // width
    assert profile.total_service_ps(width, endpoint) == SERVICE_AT_4KIB_PS[width]
    assert profile.realized_fixed_cost_ps(width) == INTERCEPT_PS[width] + 2_000_000


def test_the_smallest_payload_costs_its_intercept_plus_one_rounded_slope():
    profile = B200_NCCL_2_27_LOCAL_FIRSTPARTY_PROFILE
    assert profile.total_service_ps(2, 8) == INTERCEPT_PS[2] + 105


def test_the_provenance_names_the_substrate_and_the_band_rule():
    provenance = B200_NCCL_2_27_LOCAL_FIRSTPARTY_PROFILE.require_provenance()

    assert provenance.participant_latency_band_ps == tuple(
        (width, low, high) for width, (low, high) in sorted(BAND_PS.items())
    )
    for fragment in (
        "vast.ai",
        "2026-09-15",
        "595.91.07",
        "2.27.3",
        "2.8.0",
        "CUDA graph",
        "142255",
        "150403",
    ):
        assert fragment in provenance.source
    assert "b200_nvlink_envelope_v1" in provenance.locator
    assert "residuals" in provenance.locator
    assert "holdout" in provenance.locator
    assert "refuses every other" in provenance.transfer


def test_the_public_profile_is_untouched_by_the_first_party_study():
    baseline = _expectations()["baseline"]["profile_constants"]
    profile = resolve_collective_latency_profile("b200-nccl-2.27-local-v1")

    assert profile is B200_NCCL_2_27_LOCAL_PROFILE
    assert profile.bandwidth_bytes_per_second == baseline["bandwidth_bytes_per_second"]
    assert profile.participant_latency_ps == tuple(
        (int(width), latency)
        for width, latency in sorted(
            baseline["participant_latency_ps"].items(), key=lambda item: int(item[0])
        )
    )
    assert [profile.source_payload_bytes_min, profile.source_payload_bytes_max] == (
        baseline["source_payload_bytes"]
    )
    for width, edges in baseline["band_ps"].items():
        assert list(profile.base_latency_band_ps(int(width))) == edges


_DIGEST_BUILDERS = {
    "worked_example_tp4_pp2_dp2": lambda: declared_manifest(tp=4, pp=2, dp=2),
    "m4_tp8": lambda: declared_manifest(tp=8),
    "rail_pp8": lambda: declared_pipeline_placement(8),
    "m5_tp1_dp8": lambda: declared_manifest(tp=1, dp=8),
    "width_tp64": lambda: declared_manifest(tp=64, nodes=8, gpus_per_node=8),
}


@pytest.mark.parametrize("label", sorted(_DIGEST_BUILDERS))
def test_the_reference_manifest_digests_are_unchanged(tmp_path, label):
    record = _expectations()["baseline"]["placement_records"][label]
    payload = _DIGEST_BUILDERS[label]().save(tmp_path / f"{label}.json").read_bytes()

    assert len(payload) == record["bytes"]
    assert hashlib.sha256(payload).hexdigest() == record["sha256"]


def test_the_refit_names_the_stage_each_width_came_from():
    """Width 2 is the pinned pair of stage 1, the wider communicators stage 2's."""

    scored = STUDY / "measurements/scored.json"
    if not scored.is_file():
        pytest.skip("no scored result of record is tracked yet")
    refit = json.loads(scored.read_text(encoding="utf-8"))["refit"]

    assert refit["selection"]["2"]["source_stage"] == "stage1"
    assert refit["selection"]["4"]["source_stage"] == "stage2"
    assert refit["selection"]["8"]["source_stage"] == "stage2"
    assert refit["selection"]["2"]["stages_measuring_this_width"] == ["stage1", "stage2"]
    for width, holdout in refit["holdouts"].items():
        assert holdout["source_stage"] == refit["selection"][width]["source_stage"]


def test_the_scorer_proposes_exactly_the_registered_constants():
    """The tracked profile is what the scorer derived, integer for integer."""

    scored = STUDY / "measurements/scored.json"
    if not scored.is_file():
        pytest.skip("no scored result of record is tracked yet")
    proposed = json.loads(scored.read_text(encoding="utf-8"))["proposed_profile"]
    profile = B200_NCCL_2_27_LOCAL_FIRSTPARTY_PROFILE

    assert proposed["profile_id"] == profile.profile_id
    assert proposed["bandwidth_bytes_per_second"] == profile.bandwidth_bytes_per_second
    assert [tuple(entry) for entry in proposed["participant_latency_ps"]] == list(
        profile.participant_latency_ps
    )
    assert proposed["source_payload_bytes_min"] == profile.source_payload_bytes_min
    assert proposed["source_payload_bytes_max"] == profile.source_payload_bytes_max
    assert proposed["propagation_reference_ps"] == profile.propagation_reference_ps
    assert [
        tuple(entry) for entry in proposed["provenance"]["participant_latency_band_ps"]
    ] == list(profile.require_provenance().participant_latency_band_ps)
    assert proposed["provenance"]["evidence_class"] == profile.evidence_class
