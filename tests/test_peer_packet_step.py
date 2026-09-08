"""Original graph and request metrics consume physical peer visibility."""

from dataclasses import replace

import pytest
from test_peer_packet_runtime import engine

from simllm.backends.htsim_rnic import FlowCompletion, RnicRunResult
from simllm.backends.peer_step import PeerPacketConfig
from simllm.backends.step_attribution import HtsimRequestMetricReducer
from simllm.backends.step_sink import HtsimPersistentStepSink, HtsimStepSink, HtsimStepSinkConfig
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.core.execution import EventPhase
from simllm.placement import (
    FabricNodePlacement,
    FabricTopologyManifest,
    GpuFabricPlacement,
    PlacementManifest,
    RankPlacement,
)
from simllm.preplay.routing import RoutedExperts, RoutedLayer, RoutedRequest, RoutedToken
from simllm.preplay.schema import PREPLAY_TRACE_SCHEMA, ForwardPhase
from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply


class FixedProvider(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=37000, bound="compute")


def config(tmp_path, *, donors=3, switched=True, packet=True, rx_rate=25_000_000_000, **kwargs):
    ranks = tuple(range(donors + 1))
    runtime = engine(switched=switched, ranks=len(ranks))
    profile = replace(runtime.profile, rx=replace(runtime.profile.rx, ingress_rate_bytes_per_second=rx_rate))
    placement = PlacementManifest(ranks=[RankPlacement(global_rank=rank, hostname="node-0", local_rank=rank,
                                                      local_expert_ids={0: [] if rank == 0 else [rank - 1]}) for rank in ranks])
    node = FabricNodePlacement("node-0", "serving", tuple(
        GpuFabricPlacement(rank, f"gpu{rank}", "node-0", f"pcie{rank}", f"nic{rank}") for rank in ranks
    ), ())
    topology = FabricTopologyManifest(nodes=[node], peer_fabrics=(runtime.physical.fabric,))
    tokens = tuple(RoutedToken(phase=phase, token_index=index, token_id=token,
                               layers=(RoutedLayer(layer_index=0, expert_ids=tuple(range(donors))),))
                   for phase, index, token in ((ForwardPhase.PREFILL, 0, 10),
                                               (ForwardPhase.DECODE, 0, 20), (ForwardPhase.DECODE, 1, 21)))
    routing = RoutedExperts(trace_schema=PREPLAY_TRACE_SCHEMA, trace_sha256="a" * 64,
                            expert_count=donors, top_k=donors, moe_layer_indices=(0,),
                            requests=(RoutedRequest(request_id="peer-star", prompt_token_count=1,
                                                    output_token_count=3, tokens=tokens),))
    supply = RoutedMoeSupply(engine_rank=0, placements=(ExpertPlacementSnapshot.from_manifest(placement, ranks),),
                             step_placement_epochs=((0, 0), (1, 0), (2, 0)), routed_experts=routing)
    dims = ModelDims(num_layers=1, hidden_size=512, intermediate_size=512, num_heads=8,
                     num_kv_heads=4, head_size=64, vocab_size=49152, dtype_bytes=2,
                     num_experts=donors, top_k=donors, moe_intermediate_size=512)
    return HtsimStepSinkConfig(profile="rnic-nn-fluid", tp_ranks=(0,), ep_ranks=ranks,
                               dims=dims, workdir=tmp_path, placement_manifest=placement,
                               provider=FixedProvider(), routed_moe_supply=supply,
                               peer_packet=PeerPacketConfig(topology, (("peer-domain", profile),)) if packet else None,
                               **kwargs)


def record(index, released):
    return StepRecord(step_index=index, virtual_time_ps=released, num_sampled=1,
                       scheduled=[ScheduledRequest("peer-star", RequestPhase.PREFILL if index == 0 else RequestPhase.DECODE,
                                                   1, context_length=index + 1)])


@pytest.mark.parametrize("switched,communication", ((False, 284880), (True, 308640)))
def test_packet_extents_reach_original_graph_and_three_request_tokens(tmp_path, switched, communication):
    reducer = HtsimRequestMetricReducer({"peer-star": 0})
    sink = HtsimStepSink(config(tmp_path, switched=switched), request_metric_reducer=reducer)
    cursor = 0
    for index in range(3):
        result = sink(record(index, cursor))
        assert result.step_latency_ps == 37000 + communication
        cursor = result.completed_at_ps
        evidence = sink.peer_evidence[-1]
        evidence.validate_result(result)
        operation_completions = [event for event in evidence.execution_result.events
                                 if event.subject_object_id is None and event.phase is EventPhase.COMPLETED]
        assert {row.operation_id for row in operation_completions} == {operation.operation_id for operation in evidence.graph.operations}
        assert sink.locality_outcomes[-1].nvlink_service_ps == communication
    totals, = reducer.totals()
    assert totals.ttft_ps == 37000 + communication
    assert totals.decode_media.nvlink_ps == 2 * communication
    assert cursor == 3 * (37000 + communication)
    drained = sink.close_peer_packets()
    assert all(not row["has_pending_physical_work"] for row in drained)
    assert sink.config.selected_precision_levels["locality"].value == "packet-nvlink"


def test_packet_and_analytic_compositions_keep_compute_identical(tmp_path):
    packet = HtsimStepSink(config(tmp_path / "packet", donors=1, switched=False))
    analytic = HtsimStepSink(config(tmp_path / "analytic", donors=1, switched=False, packet=False))
    observed = packet(record(0, 0))
    baseline = analytic(record(0, 0))
    assert observed.step_latency_ps - baseline.step_latency_ps == 110800 - 6000
    assert packet.locality_outcomes[0].nvlink_directed_bytes == analytic.locality_outcomes[0].nvlink_directed_bytes
    assert packet.outcomes[0].compute_estimate_ps == analytic.outcomes[0].compute_estimate_ps == 37000
    assert not analytic.peer_evidence and analytic.close_peer_packets() == ()
    packet.close_peer_packets()


@pytest.mark.parametrize("field", ("emit_packet_breakdown", "emit_bottleneck_report"))
def test_detailed_packet_reporting_rejects_before_workdir_creation(tmp_path, field):
    path = tmp_path / "not-created"
    with pytest.raises(ValueError, match="BACK-73"):
        config(path, **{field: True})
    assert not path.exists()


def test_parallel_preparation_cannot_clone_retained_packet_state(tmp_path):
    with HtsimPersistentStepSink(config(tmp_path), max_workers=2) as sink, pytest.raises(
        ValueError, match="retained peer packet calendar"
    ):
        sink.prepare((record(0, 0),))


def test_reused_execution_rejects_before_any_new_packet(tmp_path):
    sink = HtsimStepSink(config(tmp_path))
    sink(record(0, 0))
    before = len(sink.peer_evidence)
    with pytest.raises(ValueError, match="same graph twice"):
        sink(record(0, 0))
    assert len(sink.peer_evidence) == before
    sink.close_peer_packets()


def mixed_config(tmp_path):
    cfg = config(tmp_path)
    placement = replace(cfg.placement_manifest, ranks=[
        replace(rank, hostname="node-1", local_rank=0) if rank.global_rank == 3 else rank
        for rank in cfg.placement_manifest.ranks
    ])
    domain = engine(switched=True, ranks=3).physical.fabric
    first = replace(cfg.peer_packet.fabric.nodes[0], gpus=cfg.peer_packet.fabric.nodes[0].gpus[:3])
    last = FabricNodePlacement("node-1", "serving", (replace(cfg.peer_packet.fabric.nodes[0].gpus[3], node_id="node-1"),), ())
    topology = FabricTopologyManifest(nodes=[first, last], peer_fabrics=(domain,))
    supply = replace(cfg.routed_moe_supply, placements=(ExpertPlacementSnapshot.from_manifest(placement, (0, 1, 2, 3)),))
    return replace(cfg, placement_manifest=placement, routed_moe_supply=supply,
                   peer_packet=PeerPacketConfig(topology, cfg.peer_packet.profiles))


@pytest.mark.parametrize("corruption", (None, "missing", "bytes", "time", "not_quiescent"))
def test_mixed_peer_path_joins_every_remote_completion_before_request_metrics(tmp_path, monkeypatch, corruption):
    reducer = HtsimRequestMetricReducer({"peer-star": 0})
    sink = HtsimStepSink(mixed_config(tmp_path), request_metric_reducer=reducer)

    def run(plan, goal_path, completion_csv):
        artifact = next(row for row in plan.artifacts if row.goal_path == goal_path)
        rows = [FlowCompletion(plan.profile, i, message.source_rank, message.destination_rank,
                               message.tag, message.payload_bytes, 0, 1000000, 1000000)
                for i, message in enumerate(artifact.goal_snapshot.messages)]
        if corruption == "missing":
            rows = []
        elif corruption == "bytes":
            rows[0] = replace(rows[0], payload_bytes=rows[0].payload_bytes + 1)
        elif corruption == "time":
            rows[0] = replace(rows[0], fct_ps=1)
        return RnicRunResult(rows, [], corruption != "not_quiescent", 1000000)

    monkeypatch.setattr(sink, "_run_goal", run)
    if corruption:
        with pytest.raises(ValueError, match="remote completion"):
            sink(record(0, 0))
        assert not sink.peer_evidence and not reducer.totals()
        with pytest.raises(RuntimeError, match="invalid after a failed"):
            sink(record(0, 0))
    else:
        result = sink(record(0, 0))
        assert result.step_latency_ps == 2037000
        assert sum(len(row.fabric_completions) for row in sink.peer_evidence[0].artifacts) == 2
        assert reducer.totals()[0].ttft_ps == result.step_latency_ps
        sink.close_peer_packets()


def test_future_phase_overflow_rejects_before_any_calendar_advances(tmp_path):
    from simllm.backends.peer_step import PeerPacketRuntime
    cfg = config(tmp_path)
    sink = HtsimStepSink(cfg)
    sink(record(0, 0))
    phase = next(row.local_phase.phase for row in sink.peer_evidence[0].artifacts if row.local_phase is not None)
    runtime = PeerPacketRuntime(cfg.peer_packet, cfg.placement_manifest)
    before = tuple(session.evidence() for session in runtime._sessions.values())
    with pytest.raises(ValueError, match="horizon"):
        runtime.run_phase("overflow", phase, 2**64 - 1)
    assert tuple(session.evidence() for session in runtime._sessions.values()) == before
    assert runtime.run_phase("valid", phase, 0).visible_at_ps > 0
    runtime.close()
    sink.close_peer_packets()


def test_failed_metric_publication_invalidates_the_advanced_peer_calendar(tmp_path):
    sink = HtsimStepSink(config(tmp_path), request_metric_reducer=HtsimRequestMetricReducer({}))
    with pytest.raises(ValueError, match="arrival"):
        sink(record(0, 0))
    assert not sink.outcomes and not sink.peer_evidence
    with pytest.raises(RuntimeError, match="invalid after a failed"):
        sink(record(1, 1000000))
