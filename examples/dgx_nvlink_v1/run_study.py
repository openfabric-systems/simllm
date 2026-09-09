"""Frozen product-wiring/native-switch checks through packet and request timing."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from enum import Enum
from fractions import Fraction
from itertools import pairwise, product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.local_peer_packet_runtime_v1.run_study import FrozenCompute, profile_for
from simllm.backends.htsim_nvlink import (
    NvlinkAlignedOptions,
    NvlinkSwitchArbitration,
    NvlinkTransfer,
)
from simllm.backends.nvlink_runtime import NvlinkCausalEngine, NvlinkPhysicalBinding
from simllm.backends.peer_step import PeerPacketConfig
from simllm.backends.step_attribution import HtsimRequestMetricReducer
from simllm.backends.step_sink import HtsimStepSink, HtsimStepSinkConfig
from simllm.compute import ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.placement import (
    FabricNodePlacement,
    FabricTopologyManifest,
    GpuFabricPlacement,
    PlacementManifest,
    RankPlacement,
    dgx_peer_fabric,
)
from simllm.preplay.routing import RoutedExperts, RoutedLayer, RoutedRequest, RoutedToken
from simllm.preplay.schema import PREPLAY_TRACE_SCHEMA, ForwardPhase
from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply

FREEZE = "24f41b8"
HERE = Path(__file__).resolve().parent


def canonical(value):
    def encode(item):
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, Fraction):
            return {"numerator": item.numerator, "denominator": item.denominator}
        raise TypeError(type(item).__name__)
    return (json.dumps(value, sort_keys=True, indent=2, default=encode) + "\n").encode()


def domain_profile(generation, rate, *, capacity=65536, rx_capacity=65536):
    domain = dgx_peer_fabric(generation, node_id="node-0", ranks=tuple(range(8)),
                             propagation_delay_ps=1000, switch_input_buffer_bytes=capacity)
    domain = replace(domain, links=tuple(replace(link, link_rate_bps=8 * rate) for link in domain.links))
    endpoint_rate = (300 if generation == "a100" else 450) * 10**9
    profile = profile_for(True, rate=rate, feed=endpoint_rate, rx_rate=endpoint_rate,
                          capacity=capacity, receiver_capacity=rx_capacity)
    profile = replace(profile, profile_id=f"hgx-{generation}-8-declared-v1",
                      freeze_sha256=hashlib.sha256((HERE / "EXPECTATIONS.md").read_bytes()).hexdigest())
    return domain, profile


def component(generation, rate, payload, pairs, capacity, policy, library, *, processing=0):
    domain, profile = domain_profile(generation, rate, capacity=capacity, rx_capacity=capacity)
    engine = NvlinkCausalEngine(profile, NvlinkAlignedOptions(switch_arbitration=policy),
                                physical=NvlinkPhysicalBinding(domain, processing),
                                native_switch_library=library)
    transfers = tuple(NvlinkTransfer(extent_id=f"flow-{i}", source=a, destination=b,
                                    payload_bytes=payload, topology_endpoint_count=8)
                      for i, (a, b) in enumerate(pairs))
    engine.admit(transfers, include_switch=True)
    result = engine.drain()
    check_physics(engine, result, transfers)
    return result


def check_physics(engine, result, transfers):
    """Independent half-open resource intervals and outstanding byte bounds."""
    packets = {p.packet_id: p for p in result.packets}
    assert len(packets) == len(result.packets), "duplicate packet"
    assert sum(p.payload_bytes for p in packets.values()) == sum(t.payload_bytes for t in transfers)
    intervals = defaultdict(list)
    for observation in engine.physical_paths:
        p, path = packets[observation.packet_id], observation.path
        assert p.tx_started_at_ps >= p.released_at_ps
        assert p.switch_started_at_ps >= p.tx_finished_at_ps + path.input_link.propagation_delay_ps
        assert p.rx_started_at_ps >= p.switch_finished_at_ps + path.output_link.propagation_delay_ps
        assert p.visible_at_ps >= p.rx_finished_at_ps
        intervals[("wire", path.input_resource)].append((p.tx_started_at_ps, p.tx_finished_at_ps))
        intervals[("wire", path.output_resource)].append((p.switch_started_at_ps, p.switch_finished_at_ps))
        intervals[("input", path.switch_input_port_id)].append((p.switch_started_at_ps, p.switch_finished_at_ps))
        intervals[("output", path.switch_output_port_id)].append((p.switch_started_at_ps, p.switch_finished_at_ps))
    for visits in intervals.values():
        assert all(a[1] <= b[0] for a, b in pairwise(sorted(visits))), "overlapping physical port"
    buffers = defaultdict(list)
    for claim in engine.buffer_claims:
        assert claim.returned and claim.credit_available_at_ps >= claim.released_at_ps
        buffers[claim.buffer_id].extend(((claim.reserved_at_ps, claim.wire_bytes, claim.capacity_bytes),
                                        (claim.credit_available_at_ps, -claim.wire_bytes, claim.capacity_bytes)))
    for events in buffers.values():
        outstanding = 0
        for _, delta, capacity in sorted(events):
            outstanding += delta
            assert 0 <= outstanding <= capacity, "buffer byte conservation"
        assert outstanding == 0
    by_source, by_destination = defaultdict(int), defaultdict(int)
    for p in packets.values():
        by_source[p.source] += p.wire_bytes
        by_destination[p.destination] += p.wire_bytes
    for values, endpoint_rate in ((by_source, engine.profile.tx.endpoint_egress_rate_bytes_per_second),
                                   (by_destination, engine.profile.rx.ingress_rate_bytes_per_second)):
        for amount in values.values():
            assert result.completion_time_ps * endpoint_rate >= amount * 10**12


def live_config(path, generation, rate, width, kind, library):
    domain, profile = domain_profile(generation, rate)
    ranks = tuple(range(width))
    donors = width - 1
    expert_owners = {rank: [] for rank in range(8)}
    for expert in range(donors):
        expert_owners[1 if kind == "ep-skewed" else expert + 1].append(expert)
    placement = PlacementManifest(ranks=[RankPlacement(
        global_rank=rank, hostname="node-0", local_rank=rank, local_expert_ids={0: expert_owners[rank]},
    ) for rank in range(8)])
    node = FabricNodePlacement("node-0", "serving", tuple(
        GpuFabricPlacement(rank, f"gpu{rank}", "node-0", f"pcie{rank}", f"nic{rank}") for rank in range(8)
    ), ())
    topology = FabricTopologyManifest(nodes=[node], peer_fabrics=(domain,))
    supply = None
    if kind != "tp":
        tokens = tuple(RoutedToken(phase=phase, token_index=index, token_id=token,
                                   layers=(RoutedLayer(layer_index=0, expert_ids=tuple(range(donors))),))
                       for phase, index, token in ((ForwardPhase.PREFILL, 0, 10),
                                                   (ForwardPhase.DECODE, 0, 20), (ForwardPhase.DECODE, 1, 21)))
        routing = RoutedExperts(trace_schema=PREPLAY_TRACE_SCHEMA, trace_sha256="a" * 64,
                                expert_count=donors, top_k=donors, moe_layer_indices=(0,),
                                requests=(RoutedRequest(request_id="dgx", prompt_token_count=1,
                                                        output_token_count=3, tokens=tokens),))
        supply = RoutedMoeSupply(engine_rank=0,
                                 placements=(ExpertPlacementSnapshot.from_manifest(placement, ranks),),
                                 step_placement_epochs=((0, 0), (1, 0), (2, 0)), routed_experts=routing)
    dims = ModelDims(num_layers=1, hidden_size=512, intermediate_size=512, num_heads=8,
                     num_kv_heads=4, head_size=64, vocab_size=49152, dtype_bytes=2,
                     num_experts=0 if kind == "tp" else donors,
                     top_k=0 if kind == "tp" else donors, moe_intermediate_size=512)
    return HtsimStepSinkConfig(
        profile="rnic-nn-fluid", tp_ranks=ranks if kind == "tp" else (0,),
        ep_ranks=None if kind == "tp" else ranks, dims=dims, workdir=path,
        placement_manifest=placement, provider=FrozenCompute(), routed_moe_supply=supply,
        peer_packet=PeerPacketConfig(topology, ((domain.domain_id, profile),),
                                     native_switch_library=library),
    )


def request(config):
    reducer = HtsimRequestMetricReducer({"dgx": 0})
    sink = HtsimStepSink(config, request_metric_reducer=reducer)
    cursor, steps = 0, []
    for index in range(3):
        result = sink(StepRecord(step_index=index, virtual_time_ps=cursor, num_sampled=1,
                                 scheduled=[ScheduledRequest("dgx", RequestPhase.PREFILL if index == 0
                                                             else RequestPhase.DECODE, 1,
                                                             context_length=index + 1)]))
        sink.peer_evidence[-1].validate_result(result)
        cursor = result.completed_at_ps
        steps.append({"result": asdict(result), "compute_ps": sink.outcomes[-1].compute_estimate_ps})
    observation = sink.close_peer_packets()
    totals, = reducer.totals()
    return {"steps": steps, "observations": observation,
            "ttft_ps": totals.ttft_ps, "tpot_ps": totals.tpot_ps, "jct_ps": cursor}


def comparable(result):
    """Normalize provenance only for comparison; preserve it in raw evidence."""
    view = copy.deepcopy(result)
    for row in view["observations"]:
        row.pop("switch_implementation", None)
        row.pop("switch_library_sha256", None)
    return view


def run(output, library):
    output.mkdir(parents=True, exist_ok=True)
    component_rows, live_rows = [], []
    checks = defaultdict(int)
    patterns = {"pair": ((0, 1),), "bidirectional": ((0, 1), (1, 0)),
                "disjoint": ((0, 1), (2, 3), (4, 5), (6, 7))}
    patterns.update({f"fanin-{n}": tuple((i, 0) for i in range(1, n + 1)) for n in (1, 3, 7)})
    for gen, rate, payload, capacity, policy, pattern in product(
        ("a100", "h100"), (12_500_000_000, 25_000_000_000), (256, 4096),
        (272, 65536), tuple(NvlinkSwitchArbitration), patterns,
    ):
        results = [component(gen, rate, payload, patterns[pattern], capacity, policy, selected)
                   for selected in (None, library)]
        assert results[0] == results[1], "native/Python packet conformance"
        checks["component_conformance"] += 1
        if pattern == "pair" and payload == 256:
            def ps(w, r):
                return (w * 10**12 + r - 1) // r
            endpoint = (300 if gen == "a100" else 450) * 10**9
            expected = max(ps(272, endpoint), ps(272, rate)) + 2000 + max(
                ps(272, 25_000_000_000), ps(272, rate)) + ps(272, endpoint)
            assert results[1].completion_time_ps == expected, "single-packet equation"
            checks["single_packet_equation"] += 1
        component_rows.append({"generation": gen, "rate": rate, "payload": payload, "capacity": capacity,
                               "policy": policy.value, "pattern": pattern,
                               "completion_ps": results[1].completion_time_ps})
    for row in component_rows:
        if row["rate"] == 25_000_000_000 and row["pattern"] in ("pair", "disjoint"):
            slower = next(other for other in component_rows if all(
                other[k] == v for k, v in row.items() if k not in ("rate", "completion_ps")
            ) and other["rate"] == 12_500_000_000)
            assert row["completion_ps"] <= slower["completion_ps"]
            checks["isolated_rate_direction"] += 1
    for gen in ("a100", "h100"):
        fast = component(gen, 25_000_000_000, 4096, ((0, 1),), 272,
                         NvlinkSwitchArbitration.IDENTITY, library)
        delayed = component(gen, 25_000_000_000, 4096, ((0, 1),), 272,
                            NvlinkSwitchArbitration.IDENTITY, library, processing=100000)
        assert delayed.completion_time_ps > fast.completion_time_ps
        checks["credit_backpressure"] += 1
    (output / "components.json").write_bytes(canonical(component_rows))
    for gen, rate, width, kind in product(("a100", "h100"), (12_500_000_000, 25_000_000_000),
                                         (2, 4, 8), ("tp", "ep-balanced", "ep-skewed")):
        name = f"{gen}-{rate}-{width}-{kind}"
        results = [request(live_config(output / name / selected, gen, rate, width, kind,
                                       None if selected == "python" else library))
                   for selected in ("python", "native")]
        assert canonical(comparable(results[0])) == canonical(comparable(results[1])), "native/Python live conformance"
        checks["live_conformance"] += 1
        (output / f"{name}.json").write_bytes(canonical(results[1]))
        live_rows.append({"generation": gen, "rate": rate, "width": width, "pattern": kind,
                          "ttft_ps": results[1]["ttft_ps"], "tpot_ps": results[1]["tpot_ps"],
                          "jct_ps": results[1]["jct_ps"],
                          "compute_ps": [s["compute_ps"] for s in results[1]["steps"]]})
    changed = 0
    for row in live_rows:
        if row["rate"] != 25_000_000_000:
            continue
        slower = next(other for other in live_rows if other["rate"] == 12_500_000_000 and all(
            other[k] == row[k] for k in ("generation", "width", "pattern")))
        assert row["compute_ps"] == slower["compute_ps"]
        changed += row["jct_ps"] != slower["jct_ps"]
    assert changed > 0, "selected transport is unreachable from request timing"
    summary = {"status": "VALID_MODEL_CONFORMANCE", "expectations_commit": FREEZE,
               "evidence_class": "declared_model_not_hardware_measurement", "hardware_captures": 0,
               "fatal_guards": "no violations", "relation_instances_by_family": dict(checks),
               "component_configurations": len(component_rows), "live_configurations": len(live_rows),
               "live_rate_pairs_with_changed_jct": changed,
               "components": component_rows, "live": live_rows}
    (output / "results.json").write_bytes(canonical(summary))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="external evidence directory")
    parser.add_argument("--library", required=True, help="explicit built native switch library")
    parser.add_argument("--check", action="store_true", help="compare with committed results")
    args = parser.parse_args()
    try:
        summary = run(args.output, args.library)
        if args.check:
            assert canonical(summary) == (HERE / "results.json").read_bytes(), "committed results differ"
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "VOID.json").write_bytes(canonical({"status": "VOID", "finding": str(error),
                                                         "expectations_commit": FREEZE}))
        raise
    print(json.dumps({k: v for k, v in summary.items() if k not in ("components", "live")}, indent=2))
