"""Exercise original execution graphs with declared GPU and link resources."""

import itertools
from pathlib import Path

from simllm.backends.nccl_resources import NcclGpuProfile
from simllm.backends.nccl_runtime import NcclExecutionConfig
from simllm.backends.peer_step import PeerPacketConfig
from simllm.backends.step_attribution import HtsimRequestMetricReducer
from simllm.backends.step_sink import HtsimStepSink, HtsimStepSinkConfig
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.placement import (
    FabricNodePlacement,
    FabricTopologyManifest,
    GpuFabricPlacement,
    PlacementManifest,
    RankPlacement,
)


class DeclaredCompute(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=200000, bound="compute")


def run(cfg, output: Path, session_factory):
    rows = []
    for width, protocol, sms, multiplier in itertools.product(
            cfg["widths"], cfg["protocols"], [1, 4], cfg["link_rate_multipliers"]):
        rate = int(25_000_000_000 * multiplier)
        profile, binding = session_factory(width, rate, inputs_only=True)
        ranks = tuple(range(width))
        placement = PlacementManifest(ranks=[RankPlacement(rank, "node-0", rank) for rank in ranks])
        topology = FabricTopologyManifest(nodes=[FabricNodePlacement("node-0", "serving", tuple(
            GpuFabricPlacement(rank, f"gpu{rank}", "node-0", f"pcie{rank}", f"nic{rank}") for rank in ranks
        ), ())], peer_fabrics=(binding.fabric,))
        selected = NcclExecutionConfig(NcclGpuProfile(available_sms=sms, **cfg["synthetic_gpu"]),
                                       protocol=protocol, channels=2, warps=5 if protocol == "SIMPLE" else 4)
        settings = HtsimStepSinkConfig(
            profile="rnic-nn-fluid", tp_ranks=ranks,
            dims=ModelDims(num_layers=1, hidden_size=256, intermediate_size=512,
                           num_heads=4, num_kv_heads=4, head_size=64, vocab_size=256, dtype_bytes=4),
            workdir=output / "graphs" / f"{width}-{protocol}-{sms}-{multiplier}",
            provider=DeclaredCompute(), placement_manifest=placement,
            peer_packet=PeerPacketConfig(topology, (("peer-domain", profile),), nccl=selected),
        )
        sink = HtsimStepSink(settings, request_metric_reducer=HtsimRequestMetricReducer({"r": 0}))
        at = 0
        for index in range(2):
            record = StepRecord(step_index=index, virtual_time_ps=at, num_sampled=1, scheduled=[
                ScheduledRequest("r", RequestPhase.PREFILL if index == 0 else RequestPhase.DECODE,
                                 1, context_length=index + 1)
            ])
            result = sink(record)
            evidence = sink.peer_evidence[-1]
            evidence.validate_result(result)
            phases = [a.local_phase for a in evidence.artifacts if a.local_phase is not None]
            assert len(phases) == 2 and all(p.nccl is not None for p in phases)
            communication = sum(p.service_ps for p in phases)
            metric = result.request_metrics[0]
            token_time = metric.ttft_ps if index == 0 else metric.tpot_ps
            assert token_time == result.step_latency_ps
            assert all(a.fixed_service_ps == 0 for a in evidence.artifacts)
            rows.append({"width": width, "protocol": protocol, "sms": sms,
                         "link_rate_multiplier": multiplier, "phase": "prefill" if index == 0 else "decode",
                         "token_metric_ps": token_time, "communication_ps": communication,
                         "fixed_compute_ps": token_time - communication,
                         "completed_at_ps": result.completed_at_ps})
            at = result.completed_at_ps
        sink.close_peer_packets()
    # The graph is serial and the compute provider is independent of both
    # interventions. This is the oracle for the additive token-time delta.
    assert len({row["fixed_compute_ps"] for row in rows}) == 1
    return rows
