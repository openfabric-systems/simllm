"""Live sink checks for the calibrated collective latency profile."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest

import simllm.backends.step_sink as step_sink_module
from simllm.backends import (
    HtsimPersistentStepSink,
    HtsimStepSink,
    HtsimStepSinkConfig,
)
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.placement import PlacementManifest, RankPlacement
from simllm.traffic import (
    B200_NCCL_2_27_LOCAL_PROFILE,
    LEGACY_COLLECTIVE_LATENCY_PROFILE,
)

FABRIC_SERVICE_PS = 2_500_000

CALIBRATED_DIMS = ModelDims(
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


def _record(step_index: int = 0, virtual_time_ps: int = 0) -> StepRecord:
    return StepRecord(
        step_index=step_index,
        virtual_time_ps=virtual_time_ps,
        scheduled=[
            ScheduledRequest(
                f"decode-{step_index}",
                RequestPhase.DECODE,
                num_new_tokens=1,
                context_length=8,
            )
        ],
    )


def _manifest(hosts: tuple[str, ...]) -> PlacementManifest:
    return PlacementManifest(
        ranks=[
            RankPlacement(global_rank=rank, hostname=host, local_rank=rank)
            for rank, host in enumerate(hosts)
        ]
    )


def _config(
    workdir,
    *,
    tp_ranks=(0, 1),
    placement_manifest=None,
    collective_latency_profile=None,
    nvlink_bandwidth_bytes_per_second=450_000_000_000,
    dependency_cross_check=None,
) -> HtsimStepSinkConfig:
    return HtsimStepSinkConfig(
        profile="rnic-nn-fluid",
        tp_ranks=tp_ranks,
        dims=CALIBRATED_DIMS,
        workdir=workdir,
        placement_manifest=placement_manifest,
        provider=FixedProvider(),
        collective_latency_profile=collective_latency_profile,
        nvlink_bandwidth_bytes_per_second=nvlink_bandwidth_bytes_per_second,
        dependency_cross_check=dependency_cross_check,
    )


def _stub_backend(monkeypatch, *, service_ps: int = FABRIC_SERVICE_PS):
    calls = []
    monkeypatch.setattr(step_sink_module, "to_binary", lambda path: path)

    def run(config):
        calls.append(config.goal_bin)
        return SimpleNamespace(
            job_completion_time_ps=lambda: service_ps,
            flows=(),
            quiescent=True,
        )

    monkeypatch.setattr(step_sink_module, "run_htsim_rnic", run)
    return calls


def _goal_artifacts(workdir):
    return tuple(
        (path.name, path.read_bytes()) for path in sorted(workdir.glob("*.goal"))
    )


def _without_authority(outcome):
    values = asdict(outcome)
    values.pop("authority")
    return values


def test_none_and_explicit_legacy_are_byte_exact_identity_paths(
    tmp_path,
    monkeypatch,
):
    _stub_backend(monkeypatch)
    record = _record()
    implicit = HtsimStepSink(_config(tmp_path / "implicit"))
    explicit = HtsimStepSink(
        _config(
            tmp_path / "explicit",
            collective_latency_profile=LEGACY_COLLECTIVE_LATENCY_PROFILE,
        )
    )

    implicit_result = implicit(record)
    explicit_result = explicit(record)

    assert explicit_result == implicit_result
    assert explicit.outcomes == implicit.outcomes
    assert explicit.locality_outcomes == implicit.locality_outcomes
    assert _goal_artifacts(tmp_path / "explicit") == _goal_artifacts(
        tmp_path / "implicit"
    )
    assert implicit.collective_timing_outcomes == []
    assert explicit.collective_timing_outcomes == []


def test_active_all_remote_profile_adds_one_base_and_reports_every_term(
    tmp_path,
    monkeypatch,
):
    _stub_backend(monkeypatch)
    record = _record()
    legacy = HtsimStepSink(_config(tmp_path / "legacy"))
    active = HtsimStepSink(
        _config(
            tmp_path / "active",
            collective_latency_profile=B200_NCCL_2_27_LOCAL_PROFILE.profile_id,
        )
    )

    legacy_result = legacy(record)
    active_result = active(record)

    assert legacy_result is not None
    assert active_result is not None
    assert _goal_artifacts(tmp_path / "active") == _goal_artifacts(
        tmp_path / "legacy"
    )
    timing = active.collective_timing_outcomes[0]
    locality = active.locality_outcomes[0]
    network = active.outcomes[0]
    profile = B200_NCCL_2_27_LOCAL_PROFILE
    collective_rows = tuple(
        row for row in timing.artifacts if row.collective_operation_id is not None
    )
    operation_ids = {row.collective_operation_id for row in collective_rows}

    assert timing.step_index == record.step_index
    assert timing.profile_id == profile.profile_id
    assert timing.bandwidth_bytes_per_second == profile.bandwidth_bytes_per_second
    assert timing.participant_latency_ps == profile.participant_latency_ps
    assert timing.propagation_reference_ps == 2_000_000
    assert len(operation_ids) == 2
    for operation_id in operation_ids:
        rows = tuple(
            row
            for row in collective_rows
            if row.collective_operation_id == operation_id
        )
        assert sum(row.collective_base_latency_ps for row in rows) == (
            profile.base_latency_ps(2)
        )
        assert sum(row.collective_base_latency_ps > 0 for row in rows) == 1
    assert tuple(row.operation_ids for row in timing.artifacts) == (
        locality.artifact_operation_ids
    )
    assert tuple(row.fabric_transport_ps for row in timing.artifacts) == (
        locality.fabric_phase_service_ps
    )
    assert tuple(row.composed_service_ps for row in timing.artifacts) == (
        locality.composed_phase_service_ps
    )
    for row in timing.artifacts:
        assert row.composed_service_ps == row.collective_base_latency_ps + max(
            row.local_service_ps,
            row.fabric_transport_ps,
        )
    expected_added_ps = len(operation_ids) * profile.base_latency_ps(2)
    assert active_result.step_latency_ps == legacy_result.step_latency_ps + (
        expected_added_ps
    )
    assert active_result.step_latency_ps == sum(
        row.composed_service_ps for row in timing.artifacts
    )
    assert network.makespan_ps == active_result.step_latency_ps
    assert active_result.completed_at_ps == (
        record.virtual_time_ps + active_result.step_latency_ps
    )


def test_active_all_remote_manifest_is_identity_except_for_authority(
    tmp_path,
    monkeypatch,
):
    _stub_backend(monkeypatch)
    record = _record()
    omitted = HtsimStepSink(
        _config(
            tmp_path / "omitted",
            collective_latency_profile=B200_NCCL_2_27_LOCAL_PROFILE,
        )
    )
    explicit = HtsimStepSink(
        _config(
            tmp_path / "explicit",
            placement_manifest=_manifest(("node-a", "node-b")),
            collective_latency_profile=B200_NCCL_2_27_LOCAL_PROFILE,
        )
    )

    omitted_result = omitted(record)
    explicit_result = explicit(record)

    assert explicit_result == omitted_result
    assert explicit.outcomes == omitted.outcomes
    assert explicit.collective_timing_outcomes == omitted.collective_timing_outcomes
    assert _goal_artifacts(tmp_path / "explicit") == _goal_artifacts(
        tmp_path / "omitted"
    )
    omitted_locality = omitted.locality_outcomes[0]
    explicit_locality = explicit.locality_outcomes[0]
    assert omitted_locality.authority == "compatibility-all-remote"
    assert explicit_locality.authority == "placement-manifest"
    assert omitted_locality.compatibility_fast_path
    assert explicit_locality.compatibility_fast_path
    assert _without_authority(explicit_locality) == _without_authority(
        omitted_locality
    )


def test_active_all_local_profile_owns_bandwidth_and_charges_once_per_operation(
    tmp_path,
    monkeypatch,
):
    backend_calls = _stub_backend(monkeypatch)
    sink = HtsimStepSink(
        _config(
            tmp_path / "local",
            placement_manifest=_manifest(("node", "node")),
            collective_latency_profile=B200_NCCL_2_27_LOCAL_PROFILE,
        )
    )

    result = sink(_record())

    assert result is not None
    assert backend_calls == []
    assert _goal_artifacts(tmp_path / "local") == ()
    profile = B200_NCCL_2_27_LOCAL_PROFILE
    locality = sink.locality_outcomes[0]
    timing = sink.collective_timing_outcomes[0]
    assert locality.nvlink_bandwidth_bytes_per_second == (
        profile.bandwidth_bytes_per_second
    )
    assert locality.fabric_directed_bytes == 0
    grouped = {}
    for row in timing.artifacts:
        if row.collective_operation_id is not None:
            grouped.setdefault(row.collective_operation_id, []).append(row)
    assert len(grouped) == 2
    for rows in grouped.values():
        assert len(rows) == 2
        assert {row.participant_count for row in rows} == {2}
        assert {row.critical_endpoint_bytes for row in rows} == {2_048}
        assert sum(row.collective_base_latency_ps for row in rows) == (
            profile.base_latency_ps(2)
        )
        assert sum(row.collective_base_latency_ps > 0 for row in rows) == 1
        expected_phase_service_ps = (
            (1_024 * 1_000_000_000 + profile.bandwidth_bytes_per_second - 1)
            // profile.bandwidth_bytes_per_second
        ) * 1_000
        assert {row.local_service_ps for row in rows} == {
            expected_phase_service_ps
        }
        assert {row.fabric_transport_ps for row in rows} == {0}
    assert result.step_latency_ps == sum(
        row.composed_service_ps for row in timing.artifacts
    )


def test_active_profile_rejects_unsupported_width_before_artifacts_or_publish(
    tmp_path,
    monkeypatch,
):
    backend_calls = _stub_backend(monkeypatch)
    workdir = tmp_path / "unsupported"
    sink = HtsimStepSink(
        _config(
            workdir,
            tp_ranks=(0, 1, 2),
            collective_latency_profile=B200_NCCL_2_27_LOCAL_PROFILE,
        )
    )

    with pytest.raises(ValueError, match="participant count 3"):
        sink(_record())

    assert backend_calls == []
    assert tuple(workdir.glob("*.goal")) == ()
    assert tuple(workdir.glob("*.csv")) == ()
    assert sink.outcomes == []
    assert sink.locality_outcomes == []
    assert sink.collective_timing_outcomes == []


def test_config_rejects_conflicting_or_unknown_profile_selections(tmp_path):
    with pytest.raises(ValueError, match="owns its bandwidth"):
        _config(
            tmp_path / "bandwidth-conflict",
            collective_latency_profile=B200_NCCL_2_27_LOCAL_PROFILE,
            nvlink_bandwidth_bytes_per_second=123,
        )
    with pytest.raises(ValueError, match="unknown collective latency profile"):
        _config(
            tmp_path / "unknown",
            collective_latency_profile="not-a-profile",
        )
    with pytest.raises(ValueError, match="disable one of the two selections"):
        _config(
            tmp_path / "cross-check",
            collective_latency_profile=B200_NCCL_2_27_LOCAL_PROFILE,
            dependency_cross_check="atlahs-goal",
        )
    with pytest.raises(ValueError, match="requires profile='rnic-nn-fluid'"):
        HtsimStepSinkConfig(
            profile="rnic-nn",
            tp_ranks=(0, 1),
            dims=CALIBRATED_DIMS,
            workdir=tmp_path / "wrong-network-profile",
            collective_latency_profile=B200_NCCL_2_27_LOCAL_PROFILE,
        )
    assert not (tmp_path / "bandwidth-conflict").exists()
    assert not (tmp_path / "unknown").exists()
    assert not (tmp_path / "cross-check").exists()
    assert not (tmp_path / "wrong-network-profile").exists()


def test_persistent_active_timing_matches_diagnostic_in_record_order(
    tmp_path,
    monkeypatch,
):
    _stub_backend(monkeypatch)
    records = (_record(7, 100), _record(3, 200))

    def config(label):
        return _config(
            tmp_path / label,
            collective_latency_profile=B200_NCCL_2_27_LOCAL_PROFILE,
        )

    diagnostic = HtsimStepSink(config("diagnostic"))
    diagnostic_results = tuple(diagnostic(record) for record in records)
    with HtsimPersistentStepSink(config("persistent"), max_workers=2) as persistent:
        persistent.prepare(records)
        persistent_results = tuple(persistent(record) for record in records)

    assert persistent_results == diagnostic_results
    assert persistent.outcomes == diagnostic.outcomes
    assert persistent.locality_outcomes == diagnostic.locality_outcomes
    assert persistent.collective_timing_outcomes == (
        diagnostic.collective_timing_outcomes
    )
    assert [outcome.step_index for outcome in persistent.collective_timing_outcomes] == [
        7,
        3,
    ]
