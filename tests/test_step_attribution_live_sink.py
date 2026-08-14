"""Per-request attribution driven by the live sink under real placements.

These cases run `HtsimStepSink` with a stubbed fabric backend, so they check
the seam the unit tests cannot: that the projection the sink publishes is the
projection the reducer's guards accept, on an all-local placement that runs no
backend at all and on a two-node placement whose every collective artifact
carries both media.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import simllm.backends.step_sink as step_sink_module
from simllm.backends import (
    GPU_COMPUTE_MEDIUM,
    NVLINK_MEDIUM,
    HtsimRequestMetricReducer,
    HtsimStepSink,
    HtsimStepSinkConfig,
    attribute_step_detail,
)
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.placement import PlacementManifest, RankPlacement

DIMS = ModelDims(
    num_layers=2,
    hidden_size=4,
    intermediate_size=8,
    num_heads=2,
    num_kv_heads=2,
    head_size=2,
    vocab_size=16,
    dtype_bytes=2,
)

#: one nanosecond of stubbed fabric service per collective artifact
STUB_FABRIC_PS = 1_000_000


class FixedProvider(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=2_000, bound="measured")


def _stub_backend(monkeypatch) -> None:
    monkeypatch.setattr(step_sink_module, "to_binary", lambda path: path)

    def run(config):
        text = config.goal_bin.read_text()
        return SimpleNamespace(
            job_completion_time_ps=lambda: STUB_FABRIC_PS,
            flows=[object()] * text.count(": send "),
            quiescent=True,
        )

    monkeypatch.setattr(step_sink_module, "run_htsim_rnic", run)


def _manifest(hosts: list[str]) -> PlacementManifest:
    counts: dict[str, int] = {}
    ranks = []
    for global_rank, host in enumerate(hosts):
        local_rank = counts.get(host, 0)
        counts[host] = local_rank + 1
        ranks.append(
            RankPlacement(global_rank=global_rank, hostname=host, local_rank=local_rank)
        )
    return PlacementManifest(ranks=ranks)


def _record(step_index: int, virtual_time_ps: int) -> StepRecord:
    return StepRecord(
        step_index=step_index,
        virtual_time_ps=virtual_time_ps,
        scheduled=[
            ScheduledRequest("r0", RequestPhase.DECODE, 3, context_length=8)
        ],
        num_sampled=1,
    )


def _sink(tmp_path, hosts: list[str], label: str) -> HtsimStepSink:
    return HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=tuple(range(len(hosts))),
            dims=DIMS,
            workdir=tmp_path / label,
            placement_manifest=_manifest(hosts),
            provider=FixedProvider(),
        )
    )


def _replay(sink: HtsimStepSink, steps: int = 2):
    reducer = HtsimRequestMetricReducer({"r0": 0})
    virtual_time_ps = 0
    attributions = []
    for step_index in range(steps):
        record = _record(step_index, virtual_time_ps)
        result = sink(record)
        assert result is not None
        locality = sink.locality_outcomes[step_index]
        attributions.append(attribute_step_detail(result, locality))
        reducer.consume(record, result, locality)
        virtual_time_ps = result.completed_at_ps
    return attributions, reducer.totals()


def test_the_sink_publishes_a_medium_projection_for_every_artifact(
    tmp_path,
    monkeypatch,
):
    _stub_backend(monkeypatch)
    sink = _sink(tmp_path, ["a", "a", "b", "b"], "mixed")
    sink(_record(0, 0))
    locality = sink.locality_outcomes[0]

    assert len(locality.local_phase_service_ps) == locality.artifact_count
    assert len(locality.base_phase_latency_ps) == locality.artifact_count
    assert len(locality.local_phase_medium) == locality.artifact_count
    assert set(locality.local_phase_medium) == {GPU_COMPUTE_MEDIUM, NVLINK_MEDIUM}
    assert sum(
        local
        for local, medium in zip(
            locality.local_phase_service_ps,
            locality.local_phase_medium,
            strict=True,
        )
        if medium == NVLINK_MEDIUM
    ) == locality.nvlink_service_ps
    #: every collective artifact here carries local and fabric work at once,
    #: which is exactly the shape the reducer used to refuse
    assert locality.nvlink_directed_bytes > 0
    assert locality.fabric_directed_bytes > 0


def test_an_all_local_placement_reaches_ttft_with_no_fabric_component(
    tmp_path,
    monkeypatch,
):
    _stub_backend(monkeypatch)
    sink = _sink(tmp_path, ["a", "a"], "local")
    attributions, totals = _replay(sink)

    for step, locality in zip(attributions, sink.locality_outcomes, strict=True):
        assert locality.backend_runs == 0
        assert locality.fabric_directed_bytes == 0
        assert step.media.fabric_ps == 0
        assert step.media.nvlink_ps == locality.nvlink_service_ps
        assert step.media.kernel_ps == locality.compute_service_ps
        assert step.media.total_ps == step.attribution.total_ps

    (row,) = totals
    assert row.ttft_media.fabric_ps == 0
    assert row.ttft_media.nvlink_ps > 0
    assert row.ttft_media.total_ps == row.ttft_ps
    assert row.decode_media.total_ps == row.decode_attribution.total_ps


def test_a_two_node_placement_reaches_ttft_with_a_masked_nvlink_term(
    tmp_path,
    monkeypatch,
):
    _stub_backend(monkeypatch)
    sink = _sink(tmp_path, ["a", "a", "b", "b"], "mixed")
    attributions, totals = _replay(sink)

    locality = sink.locality_outcomes[0]
    assert locality.backend_runs > 0
    step = attributions[0]
    #: the stubbed fabric service dwarfs every local endpoint serializer, so
    #: the fabric owns every collective artifact and the NVLink work is masked
    assert step.media.fabric_ps == locality.backend_runs * STUB_FABRIC_PS
    assert step.media.nvlink_ps == 0
    assert step.masked.nvlink_ps == locality.nvlink_service_ps
    assert step.masked.nvlink_ps > 0
    assert step.media.total_ps == step.attribution.total_ps

    (row,) = totals
    assert row.ttft_media.total_ps == row.ttft_ps
    assert row.ttft_media.fabric_ps > 0
    #: the masked work sum would exceed the step's own wall time if it were
    #: added to the partition, which is why it is kept out of every total
    assert (
        step.media.total_ps + step.masked.nvlink_ps > sink.outcomes[0].makespan_ps
    )


@pytest.mark.parametrize("hosts", [["a", "a"], ["a", "a", "b", "b"]])
def test_every_replayed_step_conserves_its_partition(tmp_path, monkeypatch, hosts):
    _stub_backend(monkeypatch)
    sink = _sink(tmp_path, hosts, "conserve")
    attributions, totals = _replay(sink)

    for index, step in enumerate(attributions):
        result_latency_ps = sink.outcomes[index].makespan_ps
        assert step.attribution.total_ps == result_latency_ps
        assert step.media.total_ps == result_latency_ps
        assert step.media.collective_ps == step.attribution.collective_ps
    (row,) = totals
    assert row.ttft_media.total_ps == row.ttft_attribution.total_ps
    assert row.decode_media.total_ps == row.decode_attribution.total_ps
