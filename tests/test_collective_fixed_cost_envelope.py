"""The named per-collective fixed-cost envelope, its arms and its provenance."""

from types import SimpleNamespace

import pytest

import simllm.backends.step_sink as step_sink_module
from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.traffic import (
    B200_NCCL_2_27_CROSS_NODE_PROVISIONAL_PROFILE,
    B200_NCCL_2_27_LOCAL_PROFILE,
    COLLECTIVE_FIXED_COST_ARMS,
    COLLECTIVE_FIXED_COST_FLOOR_PROFILE,
    COLLECTIVE_PROPAGATION_REFERENCE_PS,
    CROSS_NODE_COLLECTIVE_FIXED_COST_ENVELOPE,
    FABRIC_RING_STEP_HIGH_PS,
    FABRIC_RING_STEP_LOW_PS,
    FABRIC_RING_STEP_POINT_PS,
    INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE,
    NVLINK_RING_STEP_PS,
    CollectiveFixedCostEnvelope,
    CollectiveLatencyProfile,
    CollectiveLatencyProvenance,
    arm_ratio_envelope,
    resolve_collective_fixed_cost_envelope,
    resolve_collective_latency_profile,
)

NAMED_ENVELOPES = (
    INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE,
    CROSS_NODE_COLLECTIVE_FIXED_COST_ENVELOPE,
)

FABRIC_SERVICE_PS = 2_500_000

SINK_DIMS = ModelDims(
    num_layers=1,
    hidden_size=1_024,
    intermediate_size=2_048,
    num_heads=8,
    num_kv_heads=8,
    head_size=128,
    vocab_size=256,
    dtype_bytes=2,
)


class FixedProvider(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=2_000, bound="measured")


def _provenance(**overrides):
    values = {
        "evidence_class": "provisional-transferred",
        "source": "a source",
        "locator": "a locator",
        "transfer": "a transfer",
        "participant_latency_band_ps": ((2, 1, 3),),
    }
    values.update(overrides)
    return CollectiveLatencyProvenance(**values)


def _profile(**overrides):
    values = {
        "profile_id": "unit-profile",
        "bandwidth_bytes_per_second": 1,
        "participant_latency_ps": ((2, 2),),
        "source_payload_bytes_min": 8,
        "source_payload_bytes_max": 16,
        "propagation_reference_ps": 0,
        "provenance": _provenance(),
    }
    values.update(overrides)
    return CollectiveLatencyProfile(**values)


def _narrower_widths() -> CollectiveLatencyProfile:
    """Return an otherwise matching arm that supports one width fewer."""

    source = B200_NCCL_2_27_LOCAL_PROFILE
    return CollectiveLatencyProfile(
        profile_id="narrower-widths",
        bandwidth_bytes_per_second=source.bandwidth_bytes_per_second,
        participant_latency_ps=source.participant_latency_ps[:2],
        source_payload_bytes_min=source.source_payload_bytes_min,
        source_payload_bytes_max=source.source_payload_bytes_max,
        propagation_reference_ps=source.propagation_reference_ps,
        provenance=CollectiveLatencyProvenance(
            evidence_class="calibrated",
            source="the same capture",
            locator="the same locator",
            transfer="the same transfer",
            participant_latency_band_ps=(
                source.require_provenance().participant_latency_band_ps[:2]
            ),
        ),
    )


def _record(step_index: int = 0) -> StepRecord:
    return StepRecord(
        step_index=step_index,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                f"decode-{step_index}",
                RequestPhase.DECODE,
                num_new_tokens=1,
                context_length=8,
            )
        ],
    )


def _config(workdir, **overrides) -> HtsimStepSinkConfig:
    values = {
        "profile": "rnic-nn-fluid",
        "tp_ranks": (0, 1),
        "dims": SINK_DIMS,
        "workdir": workdir,
        "provider": FixedProvider(),
    }
    values.update(overrides)
    return HtsimStepSinkConfig(**values)


def _stub_backend(monkeypatch):
    monkeypatch.setattr(step_sink_module, "to_binary", lambda path: path)
    monkeypatch.setattr(
        step_sink_module,
        "run_htsim_rnic",
        lambda config: SimpleNamespace(
            job_completion_time_ps=lambda: FABRIC_SERVICE_PS,
            flows=(),
            quiescent=True,
        ),
    )


def _goal_artifacts(workdir):
    return tuple(
        (path.name, path.read_bytes()) for path in sorted(workdir.glob("*.goal"))
    )


@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        ({"evidence_class": "measured"}, "evidence_class must be one of"),
        ({"source": " "}, "source must be a nonblank string"),
        ({"locator": ""}, "locator must be a nonblank string"),
        ({"transfer": ""}, "transfer must be a nonblank string"),
        ({"participant_latency_band_ps": ()}, "must not be empty"),
        ({"participant_latency_band_ps": ((2, 1),)}, "must be a triple"),
        ({"participant_latency_band_ps": ((2, 5, 1),)}, "upper edge is below"),
        (
            {"participant_latency_band_ps": ((4, 1, 3), (2, 1, 3))},
            "unique and increasing",
        ),
    ),
)
def test_provenance_rejects_an_unusable_record(overrides, expected):
    with pytest.raises(ValueError, match=expected):
        _provenance(**overrides)


def test_provenance_fails_closed_on_an_unanchored_width():
    provenance = _provenance(participant_latency_band_ps=((2, 1, 3), (8, 1, 3)))

    assert provenance.banded_participant_counts == (2, 8)
    assert provenance.band_ps(8) == (1, 3)
    with pytest.raises(ValueError, match="no uncertainty band anchors"):
        provenance.band_ps(4)


def test_profile_refuses_provenance_that_does_not_cover_its_own_table():
    with pytest.raises(ValueError, match="provenance anchors widths"):
        _profile(
            participant_latency_ps=((2, 2), (4, 3)),
            provenance=_provenance(participant_latency_band_ps=((2, 1, 3),)),
        )
    with pytest.raises(ValueError, match="outside its own declared band"):
        _profile(participant_latency_ps=((2, 9),))


def test_profile_without_provenance_stays_constructible_but_unattributed():
    bare = CollectiveLatencyProfile(
        profile_id="bare",
        bandwidth_bytes_per_second=1,
        participant_latency_ps=((2, 2),),
        source_payload_bytes_min=8,
        source_payload_bytes_max=16,
        propagation_reference_ps=0,
    )

    assert bare.evidence_class == "unattributed"
    with pytest.raises(ValueError, match="carries no provenance record"):
        bare.require_provenance()
    with pytest.raises(ValueError, match="carries no provenance record"):
        bare.base_latency_band_ps(2)


def test_realized_fixed_cost_adds_the_propagation_the_backend_already_charges():
    profile = B200_NCCL_2_27_LOCAL_PROFILE

    for width in profile.supported_participant_counts:
        assert profile.realized_fixed_cost_ps(width) == (
            profile.base_latency_ps(width) + COLLECTIVE_PROPAGATION_REFERENCE_PS
        )
    assert COLLECTIVE_FIXED_COST_FLOOR_PROFILE.realized_fixed_cost_ps(8) == (
        COLLECTIVE_PROPAGATION_REFERENCE_PS
    )


def test_floor_profile_charges_no_surcharge_and_matches_the_capture_envelope():
    floor = COLLECTIVE_FIXED_COST_FLOOR_PROFILE
    source = B200_NCCL_2_27_LOCAL_PROFILE

    assert floor.evidence_class == "structural-floor"
    assert floor.supported_participant_counts == source.supported_participant_counts
    assert all(latency_ps == 0 for _, latency_ps in floor.participant_latency_ps)
    assert floor.bandwidth_bytes_per_second == source.bandwidth_bytes_per_second
    assert floor.source_payload_bytes_min == source.source_payload_bytes_min
    assert floor.source_payload_bytes_max == source.source_payload_bytes_max
    assert floor.endpoint_byte_bounds(8) == source.endpoint_byte_bounds(8)


def test_cross_node_profile_is_the_declared_ring_step_transfer():
    source = B200_NCCL_2_27_LOCAL_PROFILE
    cross = B200_NCCL_2_27_CROSS_NODE_PROVISIONAL_PROFILE

    assert cross.evidence_class == "provisional-transferred"
    assert "never calibrated" in cross.require_provenance().transfer
    for width, source_ps in source.participant_latency_ps:
        ring_steps = 2 * (width - 1)
        assert cross.base_latency_ps(width) == source_ps + ring_steps * (
            FABRIC_RING_STEP_POINT_PS - NVLINK_RING_STEP_PS
        )
        assert cross.base_latency_band_ps(width) == (
            source_ps + ring_steps * (FABRIC_RING_STEP_LOW_PS - NVLINK_RING_STEP_PS),
            source_ps + ring_steps * (FABRIC_RING_STEP_HIGH_PS - NVLINK_RING_STEP_PS),
        )
    assert cross.participant_latency_ps == (
        (2, 13_487_792),
        (4, 24_042_207),
        (8, 49_487_789),
    )
    assert NVLINK_RING_STEP_PS == -(
        -(source.base_latency_ps(8) - source.base_latency_ps(2)) // 12
    )


def test_named_envelopes_bracket_every_supported_width():
    for envelope in NAMED_ENVELOPES:
        assert envelope.arm_names == COLLECTIVE_FIXED_COST_ARMS
        assert envelope.arm_profile("off") is None
        for width in envelope.supported_participant_counts:
            lower_ps, upper_ps = envelope.bracket_ps(width)
            assert lower_ps < upper_ps
            assert envelope.realized_bracket_ps(width) == (
                lower_ps + COLLECTIVE_PROPAGATION_REFERENCE_PS,
                upper_ps + COLLECTIVE_PROPAGATION_REFERENCE_PS,
            )
        with pytest.raises(ValueError, match="arm must be one of"):
            envelope.arm_profile("lowest")


def test_the_intra_node_ceiling_is_the_cross_node_floor():
    assert INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE.arm_profile("lower") is (
        COLLECTIVE_FIXED_COST_FLOOR_PROFILE
    )
    assert INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE.arm_profile("upper") is (
        B200_NCCL_2_27_LOCAL_PROFILE
    )
    assert CROSS_NODE_COLLECTIVE_FIXED_COST_ENVELOPE.arm_profile("lower") is (
        B200_NCCL_2_27_LOCAL_PROFILE
    )
    assert CROSS_NODE_COLLECTIVE_FIXED_COST_ENVELOPE.arm_profile("upper") is (
        B200_NCCL_2_27_CROSS_NODE_PROVISIONAL_PROFILE
    )


@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        ({"envelope_id": " "}, "envelope_id must be a nonblank string"),
        ({"claim": ""}, "claim must be a nonblank string"),
        (
            {"upper_profile": COLLECTIVE_FIXED_COST_FLOOR_PROFILE},
            "two different profiles",
        ),
        (
            {"upper_profile": _narrower_widths()},
            "must support the same participant counts",
        ),
    ),
)
def test_envelope_rejects_an_unusable_bracket(overrides, expected):
    values = {
        "envelope_id": "unit-envelope",
        "claim": "a claim",
        "lower_profile": COLLECTIVE_FIXED_COST_FLOOR_PROFILE,
        "upper_profile": B200_NCCL_2_27_LOCAL_PROFILE,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=expected):
        CollectiveFixedCostEnvelope(**values)


def test_envelope_refuses_arms_that_do_not_isolate_the_fixed_cost():
    floor = COLLECTIVE_FIXED_COST_FLOOR_PROFILE
    source = B200_NCCL_2_27_LOCAL_PROFILE
    rebanded = CollectiveLatencyProfile(
        profile_id="rebanded",
        bandwidth_bytes_per_second=source.bandwidth_bytes_per_second + 1,
        participant_latency_ps=source.participant_latency_ps,
        source_payload_bytes_min=source.source_payload_bytes_min,
        source_payload_bytes_max=source.source_payload_bytes_max,
        propagation_reference_ps=source.propagation_reference_ps,
        provenance=source.provenance,
    )

    with pytest.raises(ValueError, match="must share bandwidth_bytes_per_second"):
        CollectiveFixedCostEnvelope(
            envelope_id="unit-envelope",
            claim="a claim",
            lower_profile=floor,
            upper_profile=rebanded,
        )
    with pytest.raises(ValueError, match="carries no provenance record"):
        CollectiveFixedCostEnvelope(
            envelope_id="unit-envelope",
            claim="a claim",
            lower_profile=CollectiveLatencyProfile(
                profile_id="bare",
                bandwidth_bytes_per_second=source.bandwidth_bytes_per_second,
                participant_latency_ps=((2, 0), (4, 0), (8, 0)),
                source_payload_bytes_min=source.source_payload_bytes_min,
                source_payload_bytes_max=source.source_payload_bytes_max,
                propagation_reference_ps=source.propagation_reference_ps,
            ),
            upper_profile=source,
        )
    with pytest.raises(ValueError, match="the lower arm is not strictly cheaper"):
        CollectiveFixedCostEnvelope(
            envelope_id="unit-envelope",
            claim="a claim",
            lower_profile=B200_NCCL_2_27_CROSS_NODE_PROVISIONAL_PROFILE,
            upper_profile=source,
        )


def test_resolvers_reach_every_named_profile_and_envelope():
    assert resolve_collective_fixed_cost_envelope(None) is None
    for envelope in NAMED_ENVELOPES:
        assert resolve_collective_fixed_cost_envelope(envelope.envelope_id) is envelope
        assert resolve_collective_fixed_cost_envelope(envelope) is envelope
    for profile in (
        B200_NCCL_2_27_LOCAL_PROFILE,
        COLLECTIVE_FIXED_COST_FLOOR_PROFILE,
        B200_NCCL_2_27_CROSS_NODE_PROVISIONAL_PROFILE,
    ):
        assert resolve_collective_latency_profile(profile.profile_id) is profile
    with pytest.raises(ValueError, match="unknown collective fixed cost envelope"):
        resolve_collective_fixed_cost_envelope("not-an-envelope")
    with pytest.raises(TypeError, match="must be None"):
        resolve_collective_fixed_cost_envelope(1)


def test_arm_ratio_envelope_brackets_a_ratio_and_flags_an_undetermined_sign():
    envelope = arm_ratio_envelope(
        "ep4 over ep8 decode at 400G",
        "ep4-decode",
        "ep8-decode",
        (
            ("off", 260_667_977, 208_707_291),
            ("local", 1_016_435_993, 1_654_852_683),
            ("cross", 1_414_693_913, 2_584_121_163),
        ),
    )

    assert [arm for arm, _ in envelope.arm_ratios] == ["off", "local", "cross"]
    assert envelope.maximum == pytest.approx(1.248964, abs=5e-7)
    assert envelope.minimum == pytest.approx(0.547456, abs=5e-7)
    assert envelope.brackets_unity
    assert envelope.width == pytest.approx(envelope.maximum / envelope.minimum)

    determined = arm_ratio_envelope(
        "a determined ratio",
        "numerator",
        "denominator",
        (("off", 3, 2), ("upper", 5, 2)),
    )

    assert not determined.brackets_unity


def test_arm_ratio_envelope_refuses_malformed_or_degenerate_rows():
    with pytest.raises(ValueError, match="needs at least two arms"):
        arm_ratio_envelope("a", "n", "d", (("off", 1, 1),))
    with pytest.raises(TypeError, match=r"arm_values\[0\] must be a triple"):
        arm_ratio_envelope("a", "n", "d", (("off", 1), ("upper", 2, 1)))
    with pytest.raises(ValueError, match="must be a nonblank string"):
        arm_ratio_envelope("a", "n", "d", ((" ", 1, 1), ("upper", 2, 1)))
    with pytest.raises(ValueError, match="must be at least 1"):
        arm_ratio_envelope("a", "n", "d", (("off", 0, 1), ("upper", 2, 1)))


def test_off_arm_is_byte_identical_to_the_repository_default(tmp_path, monkeypatch):
    _stub_backend(monkeypatch)
    record = _record()
    default = HtsimStepSink(_config(tmp_path / "default"))
    off_arm = HtsimStepSink(
        _config(
            tmp_path / "off",
            collective_fixed_cost_envelope=(
                INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE.envelope_id
            ),
            collective_fixed_cost_arm="off",
        )
    )

    default_result = default(record)
    off_result = off_arm(record)

    assert off_result == default_result
    assert off_arm.outcomes == default.outcomes
    assert off_arm.locality_outcomes == default.locality_outcomes
    assert _goal_artifacts(tmp_path / "off") == _goal_artifacts(tmp_path / "default")
    assert default.collective_timing_outcomes == []
    assert off_arm.collective_timing_outcomes == []
    assert off_arm.config.resolved_collective_latency_profile is None


def test_lower_floor_arm_publishes_the_claim_without_changing_any_timestamp(
    tmp_path,
    monkeypatch,
):
    _stub_backend(monkeypatch)
    record = _record()
    default = HtsimStepSink(_config(tmp_path / "default"))
    floor_arm = HtsimStepSink(
        _config(
            tmp_path / "floor",
            collective_fixed_cost_envelope=(
                INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE
            ),
            collective_fixed_cost_arm="lower",
        )
    )

    default_result = default(record)
    floor_result = floor_arm(record)

    assert floor_result == default_result
    assert floor_arm.outcomes == default.outcomes
    assert _goal_artifacts(tmp_path / "floor") == _goal_artifacts(tmp_path / "default")
    timing = floor_arm.collective_timing_outcomes[0]
    assert timing.profile_id == COLLECTIVE_FIXED_COST_FLOOR_PROFILE.profile_id
    assert timing.envelope_id == "intra-node-fixed-cost-v1"
    assert timing.arm == "lower"
    assert timing.evidence_class == "structural-floor"
    assert timing.propagation_reference_ps == COLLECTIVE_PROPAGATION_REFERENCE_PS
    assert all(row.collective_base_latency_ps == 0 for row in timing.artifacts)


def test_upper_cross_node_arm_charges_its_transferred_surcharge_once(
    tmp_path,
    monkeypatch,
):
    _stub_backend(monkeypatch)
    record = _record()
    default = HtsimStepSink(_config(tmp_path / "default"))
    cross_arm = HtsimStepSink(
        _config(
            tmp_path / "cross",
            collective_fixed_cost_envelope=(
                CROSS_NODE_COLLECTIVE_FIXED_COST_ENVELOPE.envelope_id
            ),
            collective_fixed_cost_arm="upper",
        )
    )

    default_result = default(record)
    cross_result = cross_arm(record)

    timing = cross_arm.collective_timing_outcomes[0]
    collective_rows = tuple(
        row for row in timing.artifacts if row.collective_operation_id is not None
    )
    operation_ids = {row.collective_operation_id for row in collective_rows}
    surcharge_ps = B200_NCCL_2_27_CROSS_NODE_PROVISIONAL_PROFILE.base_latency_ps(2)

    assert timing.envelope_id == "cross-node-fixed-cost-provisional-v1"
    assert timing.arm == "upper"
    assert timing.evidence_class == "provisional-transferred"
    assert sum(row.collective_base_latency_ps for row in timing.artifacts) == (
        len(operation_ids) * surcharge_ps
    )
    assert cross_result.step_latency_ps == default_result.step_latency_ps + (
        len(operation_ids) * surcharge_ps
    )
    assert _goal_artifacts(tmp_path / "cross") == _goal_artifacts(tmp_path / "default")


def test_config_rejects_conflicting_or_unusable_arm_selections(tmp_path):
    with pytest.raises(ValueError, match="collective_fixed_cost_arm must be one of"):
        _config(tmp_path / "bad-arm", collective_fixed_cost_arm="lowest")
    with pytest.raises(ValueError, match="keep the arm 'off'"):
        _config(tmp_path / "arm-without-envelope", collective_fixed_cost_arm="upper")
    with pytest.raises(ValueError, match="use exactly one"):
        _config(
            tmp_path / "both-spellings",
            collective_fixed_cost_envelope=(
                INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE.envelope_id
            ),
            collective_fixed_cost_arm="upper",
            collective_latency_profile=B200_NCCL_2_27_LOCAL_PROFILE,
        )
    with pytest.raises(ValueError, match="unknown collective fixed cost envelope"):
        _config(tmp_path / "unknown-envelope", collective_fixed_cost_envelope="nope")
    with pytest.raises(ValueError, match="requires profile='rnic-nn-fluid'"):
        HtsimStepSinkConfig(
            profile="rnic-nn",
            tp_ranks=(0, 1),
            dims=SINK_DIMS,
            workdir=tmp_path / "wrong-network-profile",
            collective_fixed_cost_envelope=(
                CROSS_NODE_COLLECTIVE_FIXED_COST_ENVELOPE.envelope_id
            ),
            collective_fixed_cost_arm="upper",
        )
    for name in (
        "bad-arm",
        "arm-without-envelope",
        "both-spellings",
        "unknown-envelope",
        "wrong-network-profile",
    ):
        assert not (tmp_path / name).exists()


def test_an_unsupported_width_still_fails_closed_under_every_named_arm(tmp_path):
    for envelope in NAMED_ENVELOPES:
        for arm in ("lower", "upper"):
            profile = envelope.arm_profile(arm)
            with pytest.raises(ValueError, match="does not support participant count"):
                profile.base_latency_ps(3)
            with pytest.raises(ValueError, match="does not support participant count"):
                profile.base_latency_band_ps(3)
    assert not (tmp_path / "unused").exists()
