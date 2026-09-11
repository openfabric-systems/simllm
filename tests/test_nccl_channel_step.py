"""The structural channel path reaches original-graph request metrics."""

from dataclasses import replace

import pytest
from test_nccl_channel_fifo import gpu
from test_peer_packet_runtime import engine

from simllm.backends.nccl_runtime import NcclExecutionConfig
from simllm.backends.peer_step import PeerPacketConfig
from simllm.backends.step_attribution import HtsimRequestMetricReducer
from simllm.backends.step_sink import HtsimStepSink, HtsimStepSinkConfig
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


class FixedCompute(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=200000, bound="compute")


def config(path, width=2, sms=4, rate=25_000_000_000, protocol="LL128", *, enabled=True):
    physical = engine(ranks=width, rate=rate, feed=100_000_000_000)
    ranks = tuple(range(width))
    placement = PlacementManifest(ranks=[RankPlacement(rank, "node-0", rank) for rank in ranks])
    topology = FabricTopologyManifest(nodes=[FabricNodePlacement("node-0", "serving", tuple(
        GpuFabricPlacement(rank, f"gpu{rank}", "node-0", f"pcie{rank}", f"nic{rank}") for rank in ranks
    ), ())], peer_fabrics=(physical.physical.fabric,))
    selected = NcclExecutionConfig(gpu(sms), protocol=protocol, channels=2,
                                   warps=5 if protocol == "SIMPLE" else 4)
    return HtsimStepSinkConfig(
        profile="rnic-nn-fluid", tp_ranks=ranks,
        dims=ModelDims(num_layers=1, hidden_size=256, intermediate_size=512,
                       num_heads=4, num_kv_heads=4, head_size=64, vocab_size=256, dtype_bytes=4),
        workdir=path, provider=FixedCompute(), placement_manifest=placement,
        peer_packet=PeerPacketConfig(topology, (("peer-domain", physical.profile),), nccl=selected if enabled else None),
    )


def record(index, at):
    return StepRecord(step_index=index, virtual_time_ps=at, num_sampled=1,
                      scheduled=[ScheduledRequest("r", RequestPhase.PREFILL if index == 0 else RequestPhase.DECODE,
                                                  1, context_length=index + 1)])


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("protocol", ["LL", "LL128", "SIMPLE"])
def test_two_allreduces_complete_after_gpu_output_and_reach_ttft_tpot(tmp_path, width, protocol):
    sink = HtsimStepSink(config(tmp_path, width, protocol=protocol),
                         request_metric_reducer=HtsimRequestMetricReducer({"r": 0}))
    at = 0
    for index in range(2):
        result = sink(record(index, at))
        evidence = sink.peer_evidence[-1]
        evidence.validate_result(result)
        phases = [a.local_phase for a in evidence.artifacts if a.local_phase is not None]
        assert len(phases) == 2
        assert all(p.nccl is not None for p in phases)
        communication = sum(p.service_ps for p in phases)
        assert sink.locality_outcomes[-1].nvlink_service_ps == communication
        for phase in phases:
            completed = [e.timestamp_ps for e in evidence.execution_result.events
                         if e.phase is EventPhase.COMPLETED and e.subject_object_id is None
                         and e.operation_id == phase.nccl.operation_id]
            assert completed == [phase.nccl.completed_at_ps]
        assert all(a.fixed_service_ps == 0 for a in evidence.artifacts)
        metric = result.request_metrics[0]
        assert (metric.ttft_ps if index == 0 else metric.tpot_ps) == result.step_latency_ps
        at = result.completed_at_ps
    sink.close_peer_packets()


def test_two_parameter_changes_have_exact_serial_token_delta(tmp_path):
    rows = []
    for sms in [1, 4]:
        for rate in [12_500_000_000, 50_000_000_000]:
            sink = HtsimStepSink(config(tmp_path / f"{sms}-{rate}", sms=sms, rate=rate),
                                 request_metric_reducer=HtsimRequestMetricReducer({"r": 0}))
            result = sink(record(0, 0))
            durations = sum(a.local_phase.service_ps for a in sink.peer_evidence[-1].artifacts
                            if a.local_phase is not None)
            rows.append((result.request_metrics[0].ttft_ps, durations))
            sink.close_peer_packets()
    assert len({ttft - communication for ttft, communication in rows}) == 1
    assert rows[0][0] > rows[1][0]
    assert rows[0][0] > rows[2][0]


def test_explicit_off_keeps_existing_packet_artifacts_identical(tmp_path):
    cfg = config(tmp_path / "one", enabled=False)
    first = HtsimStepSink(cfg)
    second = HtsimStepSink(replace(cfg, workdir=tmp_path / "two",
                                  peer_packet=replace(cfg.peer_packet, nccl=None)))
    assert first(record(0, 0)) == second(record(0, 0))
    assert first.peer_evidence == second.peer_evidence
    assert first.close_peer_packets() == second.close_peer_packets()
