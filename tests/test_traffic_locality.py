"""Placement-backed collective locality and live step-sink tests."""

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
from simllm.placement import PlacementManifest, RankMapper, RankPlacement
from simllm.traffic import (
    CollectiveCommunicationPhase,
    DirectedCollectiveSegment,
    classify_step_locality,
    plan_step_locality,
    render_fabric_phase_goal,
)

TINY_DIMS = ModelDims(
    num_layers=2,
    hidden_size=4,
    intermediate_size=8,
    num_heads=2,
    num_kv_heads=2,
    head_size=2,
    vocab_size=16,
    dtype_bytes=2,
)

TINY_MOE_DIMS = ModelDims(
    num_layers=2,
    hidden_size=4,
    intermediate_size=8,
    num_heads=2,
    num_kv_heads=2,
    head_size=2,
    vocab_size=16,
    dtype_bytes=2,
    num_experts=8,
    top_k=2,
    moe_intermediate_size=4,
    local_num_experts=2,
)


class FixedProvider(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=2_000, bound="measured")


def _decode_record() -> StepRecord:
    return StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                "decode",
                RequestPhase.DECODE,
                num_new_tokens=3,
                context_length=8,
            )
        ],
    )


def _manifest(hosts: list[str]) -> PlacementManifest:
    return PlacementManifest(
        ranks=[
            RankPlacement(global_rank=rank, hostname=host, local_rank=rank)
            for rank, host in enumerate(hosts)
        ]
    )


def _stub_backend(monkeypatch, *, makespan_ps: int = 123_456) -> None:
    monkeypatch.setattr(step_sink_module, "to_binary", lambda path: path)

    def run(config):
        text = config.goal_bin.read_text()
        return SimpleNamespace(
            job_completion_time_ps=lambda: makespan_ps,
            flows=[object()] * text.count(": send "),
            quiescent=True,
        )

    monkeypatch.setattr(step_sink_module, "run_htsim_rnic", run)


def test_classifier_uses_per_source_local_egress_and_whole_ns_rounding():
    phase = CollectiveCommunicationPhase(
        phase_id="mixed",
        layer=0,
        participants=(0, 1, 2, 3),
        segments=(
            DirectedCollectiveSegment(0, 1, 450, 7),
            DirectedCollectiveSegment(1, 0, 451, 7),
            DirectedCollectiveSegment(2, 3, 901, 7),
            DirectedCollectiveSegment(3, 2, 900, 7),
            DirectedCollectiveSegment(0, 2, 123, 7),
            DirectedCollectiveSegment(3, 1, 321, 7),
        ),
    )

    plan = classify_step_locality(
        (phase,),
        rank_mapper=RankMapper(_manifest(["a", "a", "b", "b"])),
    )
    classified = plan.phases[0]

    assert [segment.payload_bytes for segment in classified.fabric_segments] == [
        123,
        321,
    ]
    assert [segment.payload_bytes for segment in classified.nvlink_segments] == [
        450,
        451,
        901,
        900,
    ]
    assert classified.nvlink_source_service_ns == (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 2),
    )
    assert classified.nvlink_service_ps == 3_000
    assert plan.fabric_bytes == 444
    assert plan.nvlink_bytes == 2_702
    assert plan.total_directed_bytes == 3_146


def test_absent_placement_is_all_remote_without_reordering():
    segments = (
        DirectedCollectiveSegment(2, 0, 17, 11),
        DirectedCollectiveSegment(0, 1, 19, 12),
    )
    phase = CollectiveCommunicationPhase(
        phase_id="identity",
        layer=0,
        participants=(0, 1, 2),
        segments=segments,
    )

    plan = classify_step_locality((phase,), rank_mapper=None)

    assert plan.authority == "compatibility-all-remote"
    assert plan.phases[0].fabric_segments == segments
    assert plan.phases[0].nvlink_segments == ()
    assert plan.fabric_bytes == 36
    assert plan.nvlink_bytes == 0
    assert plan.nvlink_service_ps == 0


@pytest.mark.parametrize("world", range(1, 9))
def test_single_node_tp_widths_one_through_eight_have_zero_fabric(world):
    plan = plan_step_locality(
        _decode_record(),
        TINY_DIMS,
        tuple(range(world)),
        rank_mapper=RankMapper(_manifest(["node"] * world)),
    )

    assert plan.fabric_bytes == 0
    assert plan.fabric_segments == 0
    assert plan.nvlink_bytes == plan.total_directed_bytes
    if world == 1:
        assert plan.phases == ()
    else:
        assert plan.nvlink_bytes > 0


def test_two_node_ring_splits_exactly_at_host_boundaries():
    plan = plan_step_locality(
        _decode_record(),
        TINY_DIMS,
        (0, 1, 2, 3),
        rank_mapper=RankMapper(_manifest(["a", "a", "b", "b"])),
    )

    assert len(plan.phases) == 24
    assert plan.fabric_segments == 48
    assert plan.nvlink_segments == 48
    assert plan.fabric_bytes == 288
    assert plan.nvlink_bytes == 288
    assert plan.nvlink_service_ps == 24_000


def test_two_node_uniform_moe_splits_local_and_remote_pairs():
    plan = plan_step_locality(
        _decode_record(),
        TINY_MOE_DIMS,
        (0,),
        ep_ranks=(0, 1, 2, 3),
        rank_mapper=RankMapper(_manifest(["a", "a", "b", "b"])),
    )

    assert len(plan.phases) == 4
    assert plan.fabric_segments == 32
    assert plan.nvlink_segments == 16
    assert plan.fabric_bytes == 384
    assert plan.nvlink_bytes == 192
    assert plan.total_directed_bytes == 576


def test_phase_renderer_sends_only_cross_node_ring_segments():
    mapper = RankMapper(_manifest(["a", "a", "b", "b"]))
    plan = plan_step_locality(
        _decode_record(),
        TINY_DIMS,
        (0, 1, 2, 3),
        rank_mapper=mapper,
    )
    traces = tuple(
        render_fabric_phase_goal(phase, rank_mapper=mapper)
        for phase in plan.phases
    )
    text = "".join(trace.render() for trace in traces)

    assert text.count(": send 6b") == 48
    assert text.count(": recv 6b") == 48
    assert plan.fabric_segments == 48
    assert plan.nvlink_segments == 48


@pytest.mark.parametrize("bandwidth", [0, -1, True, 1.5])
def test_classifier_rejects_invalid_bandwidth(bandwidth):
    with pytest.raises(ValueError, match="positive integer"):
        classify_step_locality(
            (),
            rank_mapper=None,
            bandwidth_bytes_per_second=bandwidth,
        )


def test_classifier_rejects_unknown_participant_rank():
    phase = CollectiveCommunicationPhase(
        phase_id="unknown",
        layer=0,
        participants=(0, 1),
        segments=(DirectedCollectiveSegment(0, 1, 1, 0),),
    )

    with pytest.raises(KeyError, match="global rank 1"):
        classify_step_locality(
            (phase,),
            rank_mapper=RankMapper(_manifest(["a"])),
        )


def test_live_sink_reports_all_intra_analytic_outcome(tmp_path, monkeypatch):
    _stub_backend(monkeypatch)
    manifest = _manifest(["a", "a"])
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=TINY_DIMS,
            workdir=tmp_path / "local",
            placement_manifest=manifest,
            provider=FixedProvider(),
        )
    )
    manifest.ranks[1].hostname = "mutated-after-sink-construction"

    result = sink(_decode_record())
    outcome = sink.outcomes[0]
    locality = sink.locality_outcomes[0]

    assert result is not None
    assert result.step_latency_ps == 10_000
    assert locality.authority == "placement-manifest"
    assert not locality.compatibility_fast_path
    assert locality.total_directed_bytes == 192
    assert locality.fabric_directed_bytes == 0
    assert locality.nvlink_directed_bytes == 192
    assert locality.fabric_segments == 0
    assert locality.nvlink_segments == 16
    assert locality.nvlink_service_ps == 8_000
    assert locality.compute_service_ps == 2_000
    assert locality.backend_runs == 0
    assert locality.ordering_authority == "execution-graph"
    assert locality.graph_execution_id == "step-0"
    assert locality.artifact_count == 10
    # Post-specified re-acceptance: the unchanged graph wire projects into six
    # causal-level graph artifacts; all-intra collective phases remain analytic.
    assert locality.graph_artifact_count == 6
    assert locality.composed_phase_service_ps == (1_000,) * 10
    assert outcome.num_flows == 0
    assert list((tmp_path / "local").glob("*.goal")) == []


def test_live_sink_executes_mixed_artifacts_in_graph_order(tmp_path, monkeypatch):
    _stub_backend(monkeypatch, makespan_ps=123_456)
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1, 2, 3),
            dims=TINY_DIMS,
            workdir=tmp_path / "mixed",
            placement_manifest=_manifest(["a", "a", "b", "b"]),
            provider=FixedProvider(),
        )
    )

    result = sink(_decode_record())
    locality = sink.locality_outcomes[0]

    assert result is not None
    assert result.step_latency_ps == 2_000 + 24 * 123_456
    assert locality.backend_runs == 24
    assert locality.ordering_authority == "execution-graph"
    assert locality.artifact_count == 26
    # Post-specified re-acceptance: the unchanged graph wire projects into six
    # causal levels while only the 24 fabric phases invoke the backend.
    assert locality.graph_artifact_count == 6
    assert locality.fabric_phase_service_ps == (
        (0,)
        + (123_456,) * 12
        + (0,)
        + (123_456,) * 12
    )
    assert locality.composed_phase_service_ps == (
        (1_000,)
        + (123_456,) * 12
        + (1_000,)
        + (123_456,) * 12
    )
    assert sink.outcomes[0].num_flows == 48
    assert len(
        list((tmp_path / "mixed").glob("step-*.artifact-*.phase-*.goal"))
    ) == 24


def test_persistent_sink_matches_diagnostic_locality_projection(tmp_path, monkeypatch):
    _stub_backend(monkeypatch, makespan_ps=123_456)
    records = (
        _decode_record(),
        StepRecord(
            step_index=1,
            virtual_time_ps=999,
            scheduled=_decode_record().scheduled,
        ),
    )

    def config(label):
        return HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1, 2, 3),
            dims=TINY_DIMS,
            workdir=tmp_path / label,
            placement_manifest=_manifest(["a", "a", "b", "b"]),
            provider=FixedProvider(),
        )

    diagnostic = HtsimStepSink(config("diagnostic"))
    diagnostic_results = tuple(diagnostic(record) for record in records)
    with HtsimPersistentStepSink(config("persistent"), max_workers=2) as persistent:
        persistent.prepare(records)
        persistent_results = tuple(persistent(record) for record in records)

    assert persistent_results == diagnostic_results
    assert persistent.outcomes == diagnostic.outcomes
    assert persistent.locality_outcomes == diagnostic.locality_outcomes


def test_live_sink_explicit_all_remote_matches_omitted_placement(
    tmp_path,
    monkeypatch,
):
    _stub_backend(monkeypatch)
    record = _decode_record()
    omitted = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=TINY_DIMS,
            workdir=tmp_path / "omitted",
        )
    )
    explicit = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=TINY_DIMS,
            workdir=tmp_path / "explicit",
            placement_manifest=_manifest(["a", "b"]),
        )
    )

    assert omitted(record) == explicit(record)
    omitted_artifacts = tuple(
        (path.name, path.read_bytes())
        for path in sorted((tmp_path / "omitted").glob("*.goal"))
    )
    explicit_artifacts = tuple(
        (path.name, path.read_bytes())
        for path in sorted((tmp_path / "explicit").glob("*.goal"))
    )
    assert explicit_artifacts == omitted_artifacts
    assert len(explicit_artifacts) == 4
    omitted_locality = omitted.locality_outcomes[0]
    explicit_locality = explicit.locality_outcomes[0]
    assert omitted_locality.authority == "compatibility-all-remote"
    assert explicit_locality.authority == "placement-manifest"
    assert omitted_locality.compatibility_fast_path
    assert explicit_locality.compatibility_fast_path
    assert explicit_locality.fabric_directed_bytes == 192
    assert explicit_locality.nvlink_directed_bytes == 0
    assert omitted_locality.ordering_authority == "execution-graph"
    assert explicit_locality.ordering_authority == "execution-graph"
    assert (
        explicit_locality.artifact_operation_ids
        == omitted_locality.artifact_operation_ids
    )


def test_live_sink_rejects_incomplete_manifest_before_goal_write(
    tmp_path,
    monkeypatch,
):
    _stub_backend(monkeypatch)
    with pytest.raises(KeyError, match="global rank 1"):
        HtsimStepSink(
            HtsimStepSinkConfig(
                profile="rnic-nn-fluid",
                tp_ranks=(0, 1),
                dims=TINY_DIMS,
                workdir=tmp_path / "incomplete",
                placement_manifest=_manifest(["a"]),
            )
        )
    assert not (tmp_path / "incomplete").exists()


@pytest.mark.parametrize("bandwidth", [0, True, 1.5])
def test_sink_config_rejects_invalid_nvlink_bandwidth(tmp_path, bandwidth):
    with pytest.raises(ValueError, match="positive integer"):
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=TINY_DIMS,
            workdir=tmp_path,
            nvlink_bandwidth_bytes_per_second=bandwidth,
        )


def test_sink_config_rejects_invalid_manifest_type(tmp_path):
    with pytest.raises(TypeError, match="PlacementManifest"):
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=TINY_DIMS,
            workdir=tmp_path,
            placement_manifest=object(),
        )


@pytest.mark.parametrize("malformation", ["schema", "duplicate", "blank-host"])
def test_sink_rejects_malformed_manifest_before_workdir(
    tmp_path,
    malformation,
):
    manifest = _manifest(["a", "a"])
    if malformation == "schema":
        manifest.schema = "wrong-schema"
    elif malformation == "duplicate":
        manifest.ranks[1].global_rank = 0
    else:
        manifest.ranks[1].hostname = " "
    workdir = tmp_path / malformation

    with pytest.raises((ValueError, KeyError)):
        HtsimStepSink(
            HtsimStepSinkConfig(
                profile="rnic-nn-fluid",
                tp_ranks=(0, 1),
                dims=TINY_DIMS,
                workdir=workdir,
                placement_manifest=manifest,
            )
        )
    assert not workdir.exists()
