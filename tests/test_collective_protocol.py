"""Software byte accounting, deterministic uncertainty and live metric changes."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from simllm.backends import HtsimRequestMetricReducer, HtsimStepSink, HtsimStepSinkConfig
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.placement import PlacementManifest, RankPlacement
from simllm.traffic import CollectiveLatencyProfile
from simllm.traffic.collective_protocol import (
    NcclProtocolService,
    NcclRingProtocolModel,
    encoded_bytes,
    ring_geometry,
)

MODELS = (
    Path(__file__).resolve().parents[1] / "examples/nccl_protocol_model_v1/calibration/models.json"
)


def models():
    return tuple(NcclRingProtocolModel.from_json(row) for row in json.loads(MODELS.read_text()))


def profile(model, arm="central"):
    return CollectiveLatencyProfile(
        profile_id=model.model_id + "-" + arm,
        bandwidth_bytes_per_second=model.endpoint_rate_bytes_per_second,
        participant_latency_ps=((model.width, model.startup_ps),),
        source_payload_bytes_min=model.payload_min_bytes,
        source_payload_bytes_max=model.payload_max_bytes,
        propagation_reference_ps=0,
        protocol_models=((model.width, model),),
        protocol_arm=arm,
    )


class FixedProvider(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=2000, bound="measured")


def sink(root, model, arm="central", *, dtype=4, remote=False, enabled=True):
    dims = ModelDims(
        num_layers=1,
        hidden_size=4096,
        intermediate_size=8192,
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
        vocab_size=256,
        dtype_bytes=dtype,
    )
    placement = PlacementManifest(
        ranks=[
            RankPlacement(global_rank=r, hostname=f"node-{r}" if remote else "node", local_rank=r)
            for r in range(model.width)
        ]
    )
    return HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=tuple(range(model.width)),
            dims=dims,
            provider=FixedProvider(),
            workdir=root,
            placement_manifest=placement,
            collective_latency_profile=profile(model, arm) if enabled else None,
        ),
        request_metric_reducer=HtsimRequestMetricReducer({f"r{i}": 0 for i in range(128)}),
    )


def record(phase, tokens):
    scheduled = (
        [ScheduledRequest("r0", phase, num_new_tokens=tokens, context_length=tokens)]
        if phase is RequestPhase.PREFILL
        else [
            ScheduledRequest(f"r{i}", phase, num_new_tokens=1, context_length=32)
            for i in range(tokens)
        ]
    )
    return StepRecord(
        step_index=0, virtual_time_ps=0, scheduled=scheduled, num_sampled=len(scheduled)
    )


@pytest.mark.parametrize(
    "payload,ll,ll128", [(0, 0, 0), (8, 16, 128), (120, 240, 128), (128, 256, 256)]
)
def test_encoded_flag_bytes_have_whole_line_padding(payload, ll, ll128):
    assert encoded_bytes(payload, "LL") == ll
    assert encoded_bytes(payload, "LL128") == ll128
    assert encoded_bytes(payload, "SIMPLE") == payload


@pytest.mark.parametrize(
    "width,protocol,size,channels,warps",
    [
        (2, "LL", 589824, 8, 16),
        (2, "SIMPLE", 606208, 8, 17),
        (4, "LL", 1458176, 24, 16),
        (4, "LL128", 1474560, 23, 20),
    ],
)
def test_observed_source_selection_fixture(width, protocol, size, channels, warps):
    work = ring_geometry(size, width, protocol)
    assert (work.channels, work.warps) == (channels, warps)
    assert sum(work.channel_payload_bytes) == size
    assert work.encoded_endpoint_floor_bytes >= work.application_endpoint_bytes
    assert work.counter_store_bytes_per_rank > 0


def test_simple_keeps_empty_sync_slices_separate_from_fenced_data():
    work = ring_geometry(1 << 20, 2, "SIMPLE")
    assert work.synchronization_slices == 6
    assert work.nonempty_publications == 4
    assert work.inline_flag_floor_bytes == 0
    assert ring_geometry(1 << 20, 2, "LL").inline_flag_floor_bytes == 1 << 20


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_non_integer_service_costs_are_rejected(value):
    with pytest.raises(ValueError):
        NcclProtocolService("LL", value, 0, 0, 0)


def test_protocol_codec_is_strict_and_deterministic():
    for model in models():
        assert NcclRingProtocolModel.from_json(model.to_json()) == model
        payload = model.to_json()
        payload["unknown"] = 1
        with pytest.raises(ValueError, match="strict schema"):
            NcclRingProtocolModel.from_json(payload)
        for size in (262144, 524288, 1048576, 4194304):
            assert model.predict(size) == model.predict(size)
            estimate = model.predict(size)
            assert (
                estimate.physical_floor_ps
                <= estimate.lower_ps
                <= estimate.central_ps
                <= estimate.upper_ps
            )
        with pytest.raises(ValueError, match="outside"):
            model.predict(4194320)


def test_rate_and_rtt_have_separate_nonnegative_ownership():
    for model in models():
        base = model.predict(1048576, protocol="SIMPLE")
        half = replace(
            model,
            peer_rate_bytes_per_second=model.peer_rate_bytes_per_second // 2,
            endpoint_rate_bytes_per_second=model.endpoint_rate_bytes_per_second // 2,
        ).predict(1048576, protocol="SIMPLE")
        assert abs(half.physical_floor_ps - 2 * base.physical_floor_ps) <= 1
        assert half.reference_ps >= base.reference_ps
        for rtt in (0, 250000, 1000000, 100000000):
            changed = replace(model, visibility_rtt_ps=rtt).predict(1048576, protocol="SIMPLE")
            assert changed.attributed_rtt_ps == rtt * base.geometry.nonempty_publications
            assert changed.residual_poll_fence_ps >= 0
            assert changed.reference_ps == base.reference_ps + max(
                0, changed.attributed_rtt_ps - base.composite_publication_ps
            )


def test_boundary_uncertainty_unites_source_resolution_interval_only():
    model = NcclRingProtocolModel.from_json(json.loads(MODELS.with_name("candidate_v4.json").read_text())[0])
    boundary, new = model.protocol_starts[1]
    size = boundary - 8192
    estimate = model.predict(size)
    old = model.predict(size, protocol=model.protocol(size))
    other = model.predict(size, protocol=new)
    assert estimate.lower_ps == min(old.lower_ps, other.lower_ps)
    assert estimate.upper_ps == max(old.upper_ps, other.upper_ps)
    assert model.predict(boundary).geometry.protocol == new


@pytest.mark.parametrize("model", models(), ids=lambda m: m.model_id)
@pytest.mark.parametrize("tokens", [32, 128])
@pytest.mark.parametrize("phase", [RequestPhase.PREFILL, RequestPhase.DECODE])
def test_live_original_graph_changes_token_time_by_collective_service(
    tmp_path, model, tokens, phase
):
    item = record(phase, tokens)
    instances = {arm: sink(tmp_path / arm, model, arm) for arm in ("lower", "central", "upper")}
    results = {arm: instance(item) for arm, instance in instances.items()}
    estimate = model.predict(tokens * 4096 * 4)
    for arm in ("lower", "upper"):
        expected = 2 * (getattr(estimate, arm + "_ps") - estimate.central_ps)
        assert results[arm].step_latency_ps - results["central"].step_latency_ps == expected
        assert results[arm].completed_at_ps - results["central"].completed_at_ps == expected
        # The supported sink supplies the request's sampled token boundary.
        assert results[arm].request_metrics
        for changed, base in zip(
            results[arm].request_metrics, results["central"].request_metrics, strict=True
        ):
            assert changed.completed_at_ps - base.completed_at_ps == expected
    if phase is RequestPhase.DECODE:
        second = {
            arm: instance(replace(item, step_index=1, virtual_time_ps=results[arm].completed_at_ps))
            for arm, instance in instances.items()
        }
        for arm in ("lower", "upper"):
            expected = 2 * (getattr(estimate, arm + "_ps") - estimate.central_ps)
            for changed, base in zip(
                second[arm].request_metrics, second["central"].request_metrics, strict=True
            ):
                assert changed.tpot_ps is not None and base.tpot_ps is not None
                assert changed.tpot_ps - base.tpot_ps == expected
                assert changed.ttft_ps - base.ttft_ps == expected
    else:
        for arm in ("lower", "upper"):
            expected = 2 * (getattr(estimate, arm + "_ps") - estimate.central_ps)
            assert (
                results[arm].request_metrics[0].ttft_ps
                - results["central"].request_metrics[0].ttft_ps
                == expected
            )

    for arm, instance in instances.items():
        rows = instance.collective_timing_outcomes[0].artifacts
        ids = {
            row.collective_operation_id for row in rows if row.collective_operation_id is not None
        }
        assert len(ids) == 2
        for operation in ids:
            selected = [row for row in rows if row.collective_operation_id == operation]
            assert sum(row.composed_service_ps for row in selected) == getattr(
                estimate, arm + "_ps"
            )


def test_unsupported_scope_fails_before_state_publication(tmp_path):
    model = models()[0]
    for label, kwargs in [("dtype", {"dtype": 2}), ("remote", {"remote": True})]:
        instance = sink(tmp_path / label, model, **kwargs)
        with pytest.raises(ValueError, match="float32|fully local"):
            instance(record(RequestPhase.PREFILL, 32))
        assert instance.outcomes == []
        assert instance.collective_timing_outcomes == []


def test_source_chooser_resolves_the_fresh_payload_inside_the_old_bracket():
    model = models()[0]
    assert model.protocol(589824) == "LL"
    assert model.protocol(598016) == "SIMPLE"
    assert model.protocol(606208) == "SIMPLE"
    assert model.center_method == "event"
    estimate = model.predict(1048576)
    assert estimate.central_ps == estimate.reference_ps + estimate.method_allowance_ps
    with pytest.raises(ValueError, match="three protocols"):
        replace(model, choice_costs=(("LL", 1, 1),))
