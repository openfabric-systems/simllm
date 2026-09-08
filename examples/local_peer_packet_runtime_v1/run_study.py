"""Check the frozen local packet path through original graph and token metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from enum import Enum
from fractions import Fraction
from itertools import pairwise, product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simllm.backends.htsim_nvlink import (
    NvlinkFifoPlacement,
    NvlinkSwitchConfig,
    NvlinkSwitchMode,
    NvlinkTrafficClass,
    NvlinkTransfer,
    load_nvlink_candidate_profile,
)
from simllm.backends.nvlink_runtime import NvlinkPhysicalBinding
from simllm.backends.peer_step import PeerPacketConfig
from simllm.backends.step_attribution import HtsimRequestMetricReducer
from simllm.backends.step_sink import HtsimStepSink, HtsimStepSinkConfig
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.compute.gpu_packet_port import GpuPeerPacketSession
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.core.execution_io import execution_graph_to_json, execution_result_to_json
from simllm.core.step import step_record_to_json
from simllm.core.step_io import step_result_to_json
from simllm.placement import (
    FabricLink,
    FabricNodePlacement,
    FabricTopologyManifest,
    GpuFabricPlacement,
    PeerFabric,
    PeerPortPlacement,
    PeerRoute,
    PlacementManifest,
    RankPlacement,
)
from simllm.preplay.routing import RoutedExperts, RoutedLayer, RoutedRequest, RoutedToken
from simllm.preplay.schema import PREPLAY_TRACE_SCHEMA, ForwardPhase
from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply

HERE = Path(__file__).resolve().parent
FREEZE = "319cf7f3341ad7d204454810a096981d8cc216c7"
PROFILE = ROOT / "examples/a100_nvlink_packet_v1/candidate-profile-pre-traf70.json"


def canonical(value):
    def encode(row):
        if isinstance(row, Enum):
            return row.value
        if isinstance(row, Path):
            return str(row)
        if isinstance(row, Fraction):
            return {"numerator": row.numerator, "denominator": row.denominator}
        raise TypeError(f"unsupported study evidence type: {type(row).__name__}")
    return (json.dumps(value, indent=2, sort_keys=True, default=encode) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def physical_domain(ranks, switched, *, rate=25_000_000_000, capacity=65536):
    ports, links, routes = [], [], []
    if switched:
        for rank in range(ranks):
            ports.extend((PeerPortPlacement(f"gpu{rank}-p", gpu_rank=rank),
                          PeerPortPlacement(f"switch-p{rank}", switch_id="peer-switch")))
            links.append(FabricLink(f"link{rank}", f"gpu{rank}-p", f"switch-p{rank}", rate * 8, 1000))
        for source, destination in product(range(ranks), repeat=2):
            if source != destination:
                routes.append(PeerRoute(source, destination, ((f"link{source}", f"link{destination}"),)))
    else:
        for source in range(ranks):
            for destination in range(source + 1, ranks):
                a, b, link = f"gpu{source}-to{destination}", f"gpu{destination}-to{source}", f"link{source}-{destination}"
                ports.extend((PeerPortPlacement(a, gpu_rank=source), PeerPortPlacement(b, gpu_rank=destination)))
                links.append(FabricLink(link, a, b, rate * 8, 1000))
                routes.extend((PeerRoute(source, destination, ((link,),)), PeerRoute(destination, source, ((link,),))))
    return PeerFabric("peer-domain", "node-0", tuple(ports), tuple(links), tuple(routes),
                      switch_input_buffer_bytes=capacity if switched else None)


def profile_for(switched, *, rate=25_000_000_000, rx_rate=25_000_000_000,
                feed=25_000_000_000, capacity=65536, receiver_capacity=65536, credits=256):
    base = load_nvlink_candidate_profile(PROFILE)
    return replace(base, tx=replace(base.tx, links_per_peer=1, per_link_rate_bytes_per_second=rate,
                                   endpoint_egress_rate_bytes_per_second=feed, credits_per_destination=credits),
                   rx=replace(base.rx, ingress_rate_bytes_per_second=rx_rate,
                              buffer_capacity_bytes=receiver_capacity, credit_return_latency_ps=0),
                   switch=NvlinkSwitchConfig(mode=NvlinkSwitchMode.QUEUED, fifo_placement=NvlinkFifoPlacement.INPUT,
                                             service_rate_bytes_per_second=25_000_000_000,
                                             buffer_capacity_bytes=capacity, arbitration="fifo", head_of_line_blocking=True)
                   if switched else NvlinkSwitchConfig(mode=NvlinkSwitchMode.PASS_THROUGH))


class FrozenCompute(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=37000, bound="compute")


def live_config(path, donors, rx_rate, switched, *, enabled):
    ranks = tuple(range(donors + 1))
    placement = PlacementManifest(ranks=[RankPlacement(
        global_rank=rank, hostname="node-0", local_rank=rank,
        local_expert_ids={0: [] if rank == 0 else [rank - 1]},
    ) for rank in ranks])
    node = FabricNodePlacement("node-0", "serving", tuple(
        GpuFabricPlacement(rank, f"gpu{rank}", "node-0", f"pcie{rank}", f"nic{rank}") for rank in ranks
    ), ())
    topology = FabricTopologyManifest(nodes=[node], peer_fabrics=(physical_domain(len(ranks), switched),))
    routing = RoutedExperts(
        trace_schema=PREPLAY_TRACE_SCHEMA, trace_sha256="a" * 64, expert_count=donors, top_k=donors,
        moe_layer_indices=(0,), requests=(RoutedRequest(
            request_id="peer-star", prompt_token_count=1, output_token_count=3,
            tokens=tuple(RoutedToken(phase=phase, token_index=index, token_id=token,
                                    layers=(RoutedLayer(layer_index=0, expert_ids=tuple(range(donors))),))
                         for phase, index, token in ((ForwardPhase.PREFILL, 0, 10),
                                                     (ForwardPhase.DECODE, 0, 20), (ForwardPhase.DECODE, 1, 21))),
        ),),
    )
    supply = RoutedMoeSupply(engine_rank=0, placements=(ExpertPlacementSnapshot.from_manifest(placement, ranks),),
                             step_placement_epochs=((0, 0), (1, 0), (2, 0)), routed_experts=routing)
    dims = ModelDims(num_layers=1, hidden_size=512, intermediate_size=512, num_heads=8,
                     num_kv_heads=4, head_size=64, vocab_size=49152, dtype_bytes=2,
                     num_experts=donors, top_k=donors, moe_intermediate_size=512)
    return HtsimStepSinkConfig(
        profile="rnic-nn-fluid", tp_ranks=(0,), ep_ranks=ranks, dims=dims, workdir=path,
        placement_manifest=placement, provider=FrozenCompute(), routed_moe_supply=supply,
        peer_packet=PeerPacketConfig(topology, (("peer-domain", profile_for(switched, rx_rate=rx_rate)),)) if enabled else None,
    )


def run_request(config):
    reducer = HtsimRequestMetricReducer({"peer-star": 0})
    sink = HtsimStepSink(config, request_metric_reducer=reducer)
    steps, cursor = [], 0
    for index in range(3):
        record = StepRecord(step_index=index, virtual_time_ps=cursor, num_sampled=1,
                            scheduled=[ScheduledRequest("peer-star", RequestPhase.PREFILL if index == 0 else RequestPhase.DECODE,
                                                        1, context_length=index + 1)])
        result = sink(record)
        steps.append({"record": step_record_to_json(record), "result": step_result_to_json(result),
                      "locality": asdict(sink.locality_outcomes[-1]), "outcome": asdict(sink.outcomes[-1]),
                      "graph": execution_graph_to_json(sink.peer_evidence[-1].graph) if sink.peer_evidence else None,
                      "execution_result": execution_result_to_json(sink.peer_evidence[-1].execution_result) if sink.peer_evidence else None,
                      "artifacts": [asdict(row) for row in sink.peer_evidence[-1].artifacts] if sink.peer_evidence else [],
                      "sessions": sink.peer_evidence[-1].session_observations if sink.peer_evidence else ()})
        cursor = result.completed_at_ps
    totals, = reducer.totals()
    return {"steps": steps, "request_totals": asdict(totals), "job_completion_ps": cursor,
            "final_sessions": sink.close_peer_packets()}


def session_findings(observation, *, final):
    """Independent identity, capacity and service conservation checks."""
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    packets = observation["packets"]
    extents = observation["extents"]
    events = observation["packet_events"]
    require(len({p["packet_id"] for p in packets}) == len(packets), "unique packet identities")
    require(sum(p["payload_bytes"] for p in packets) == sum(e["transfer"]["payload_bytes"] for e in extents), "payload conservation")
    require([e["event_time_ps"] for e in events] == sorted(e["event_time_ps"] for e in events), "actual callback order")
    tokens = [token for e in extents for token in (e["extent_token"], *e["attempt_tokens"])]
    require(len(set(tokens)) == len(tokens), "session token uniqueness")
    for snapshot in observation["port_snapshots"]:
        source_extents = [e for e in extents if e["transfer"]["source"] == snapshot["gpu_rank"]]
        source_tokens = {t for e in source_extents for t in e["attempt_tokens"]}
        starts = {e["attempt_token"] for e in events if e["event_kind"] == "packet_tx_started" and e["attempt_token"] in source_tokens}
        retired = {e["attempt_token"] for e in events if e["event_kind"] in ("delivered", "dropped") and e["attempt_token"] in source_tokens}
        require(set(snapshot["live_attempt_tokens"]) == starts - retired, "partial live attempt set")
        live_parents = {e["extent_token"] for e in source_extents if not set(e["attempt_tokens"]) <= retired}
        require(set(snapshot["live_extent_tokens"]) == live_parents, "partial live parent set")
    resource_rows = defaultdict(list)
    visit_keys = set()
    for visit in observation["resource_visits"]:
        key = (visit["subject_object_id"], visit["stage"])
        require(key not in visit_keys, "duplicate resource visit")
        visit_keys.add(key)
        require(visit["submitted_at_ps"] <= visit["eligible_at_ps"] <= visit["started_at_ps"]
                <= visit["finished_at_ps"] <= visit["completed_at_ps"] <= observation["now_ps"], "queue visit order")
        resource_rows[visit["resource"]["resource_id"]].append(visit)
    for rows in resource_rows.values():
        rows.sort(key=lambda row: (row["started_at_ps"], row["subject_object_id"]))
        require(all(a["finished_at_ps"] <= b["started_at_ps"] for a, b in pairwise(rows)), "physical resource exclusion")
    ownership, resident = defaultdict(list), defaultdict(list)
    for claim in observation["buffer_claims"]:
        key = (claim["buffer_id"], claim["capacity_bytes"])
        ownership[key].append((claim["reserved_at_ps"], claim["wire_bytes"]))
        if claim["returned"]:
            ownership[key].append((claim["credit_available_at_ps"], -claim["wire_bytes"]))
        if claim["arrived_at_ps"] is not None:
            resident[key].append((claim["arrived_at_ps"], claim["wire_bytes"]))
        if claim["released_at_ps"] is not None:
            resident[key].append((claim["released_at_ps"], -claim["wire_bytes"]))
    for rows in (ownership, resident):
        for (_, capacity), deltas in rows.items():
            occupied = 0
            for _, delta in sorted(deltas):
                occupied += delta
                require(0 <= occupied <= capacity, "finite buffer capacity")
            if final:
                require(occupied == 0, "final buffer release")
    if final:
        require(not observation["has_pending_physical_work"], "final calendar quiescence")
        require(all(not s["live_attempt_tokens"] and not s["live_extent_tokens"] for s in observation["port_snapshots"]), "final token retirement")
        require(all(row["returned"] for row in observation["buffer_claims"]), "final credit return")
        require(observation["drained_result"] is not None, "final reservation history retained")
        expected_visits = (6 if observation["binding"]["fabric"]["switch_input_buffer_bytes"] is not None else 3) * len(packets)
        require(len(visit_keys) == expected_visits, "complete resource visit inventory")
        for packet in packets:
            require(Counter(e["event_kind"] for e in events if e["attempt_token"] == next(
                e["attempt_tokens"][e["packet_ids"].index(packet["packet_id"])] for e in extents if packet["packet_id"] in e["packet_ids"]
            )) == Counter(("packet_tx_started", "packet_tx_finished", "packet_rx_arrived", "delivered")), "one complete attempt lifecycle")
    return sorted(set(errors))


def strip_class(value):
    if isinstance(value, list):
        return [strip_class(row) for row in value]
    if isinstance(value, dict):
        return {key: strip_class(row) for key, row in value.items() if key != "traffic_class"}
    return value


def run_study(output, compatibility):
    git("merge-base", "--is-ancestor", FREEZE, "HEAD")
    for name in ("expectations.md", "expectations.json"):
        path = HERE / name
        if path.read_bytes() != git("show", f"{FREEZE}:{path.relative_to(ROOT).as_posix()}"):
            raise ValueError("frozen expectations changed")
    dirty = git("status", "--porcelain", "--untracked-files=all").decode().splitlines()
    if any(not row[3:].startswith("notes/") for row in dirty):
        raise ValueError("commit all study and implementation sources before running")
    frozen = json.loads((HERE / "expectations.json").read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    findings, guards, oracles, relations, configurations, raw_files = [], [], [], [], [], {}

    def write(name, value):
        data = canonical(value)
        (output / name).write_bytes(data)
        raw_files[name] = digest(data)

    def guard(name, condition):
        guards.append({"id": name, "held": bool(condition)})
        if not condition:
            findings.append(name)

    def oracle(name, actual, expected):
        oracles.append({"id": name, "actual_ps": actual, "expected_ps": expected,
                        "residual_ps": actual - expected, "passed": actual == expected})

    def relation(family, name, actual, expected):
        relations.append({"family": family, "id": name, "actual_ps": actual,
                          "expected_ps": expected, "passed": actual == expected})

    def check_session(name, value, final):
        for finding in session_findings(value, final=final):
            guard(name + ":" + finding, False)
        guard(name + ":identity-capacity-resource-guards", not session_findings(value, final=final))

    observed = {}
    for donors, rx_rate, route in product(frozen["donors"], frozen["rx_bytes_per_second"], frozen["routes"]):
        name = f"k{donors}-rx{rx_rate}-{route}"
        switched = route == "switched"
        n, link, prop = 4, 10880, 1000
        rx = (272 * 10**12 + rx_rate - 1) // rx_rate
        first_transport = (2 if switched else 1) * (link + prop)
        # These independent bounds are fixed before executing either arm.
        floor = first_transport + donors * n * rx
        ceiling = 2 * donors * n * (first_transport + rx + 2 * prop)
        configuration = {"id": name, "donors": donors, "rx_rate": rx_rate, "route": route,
                         "combine_floor_ps": floor, "communication_ceiling_ps": ceiling}
        configurations.append(configuration)
        write(name + "-configuration.json", configuration)
        try:
            packet = run_request(live_config(output / name / "packet", donors, rx_rate, switched, enabled=True))
            analytic = run_request(live_config(output / name / "analytic", donors, rx_rate, switched, enabled=False))
            write(name + "-raw.json", {"packet": packet, "analytic": analytic})
            expected_comm = frozen["oracle"][route + "_communication_ps"][f"{donors},{rx_rate}"]
            expected_analytic = frozen["oracle"]["analytic_communication_ps"][str(donors)]
            steps = packet["steps"]
            expected_dispatch = (donors - 1) * n * link + first_transport + n * rx
            expected_combine = first_transport + donors * n * rx
            for index, (row, off) in enumerate(zip(steps, analytic["steps"], strict=True)):
                phases = [a["local_service_ps"] for a in row["artifacts"] if a["local_phase"] is not None]
                guard(f"{name}-step{index}:two-phase-inventory", len(phases) == 2)
                oracle(f"{name}-step{index}-dispatch", phases[0], expected_dispatch)
                oracle(f"{name}-step{index}-combine", phases[1], expected_combine)
                oracle(f"{name}-step{index}-latency", row["result"]["step_latency_ps"], 37000 + expected_comm)
                oracle(f"{name}-step{index}-analytic", off["result"]["step_latency_ps"], 37000 + expected_analytic)
                guard(f"{name}-step{index}:physical-floors-ceilings", phases[1] >= floor and sum(phases) <= ceiling)
                guard(f"{name}-step{index}:same-compute", row["outcome"]["compute_estimate_ps"] == off["outcome"]["compute_estimate_ps"] == 37000)
                guard(f"{name}-step{index}:same-logical-bytes", row["locality"]["nvlink_directed_bytes"] == off["locality"]["nvlink_directed_bytes"] == 2 * donors * 1024)
                guard(f"{name}-step{index}:original-boundary", row["execution_result"]["completed_at_ps"] == row["result"]["completed_at_ps"])
                relation("packet-versus-analytic-token-latency", f"{name}-token{index}",
                         row["result"]["step_latency_ps"] - off["result"]["step_latency_ps"], expected_comm - expected_analytic)
                for session in row["sessions"]:
                    check_session(f"{name}-step{index}", session, False)
            oracle(f"{name}-TTFT", packet["request_totals"]["ttft_ps"], 37000 + expected_comm)
            oracle(f"{name}-job-completion", packet["job_completion_ps"], 3 * (37000 + expected_comm))
            for session in packet["final_sessions"]:
                check_session(name + "-drain", session, True)
            observed[(donors, rx_rate, route)] = steps[0]["result"]["step_latency_ps"] - 37000
        except (AssertionError, ValueError, TypeError, RuntimeError, KeyError, IndexError) as error:
            write(name + "-failure.json", {"type": type(error).__name__, "message": str(error)})
            guard(name + ":execution-or-evidence-error", False)
    for donors, route in product(frozen["donors"], frozen["routes"]):
        keys = ((donors, 12500000000, route), (donors, 25000000000, route))
        if all(key in observed for key in keys):
            relation("receiver-serialization", f"k{donors}-{route}", observed[keys[0]] - observed[keys[1]], (donors + 1) * 4 * 10880)
    for rx_rate, route in product(frozen["rx_bytes_per_second"], frozen["routes"]):
        keys = ((3, rx_rate, route), (1, rx_rate, route))
        if all(key in observed for key in keys):
            relation("donor-width", f"rx{rx_rate}-{route}", observed[keys[0]] - observed[keys[1]], 8 * 10880 + 8 * (272 * 10**12 // rx_rate))
    for donors, rx_rate in product(frozen["donors"], frozen["rx_bytes_per_second"]):
        keys = ((donors, rx_rate, "switched"), (donors, rx_rate, "direct"))
        if all(key in observed for key in keys):
            relation("hop-transport", f"k{donors}-rx{rx_rate}", observed[keys[0]] - observed[keys[1]], 2 * (10880 + 1000))

    for processing in frozen["retained_credit_control"]["return_processing_ps"]:
        name = f"retained-return{processing}"
        configurations.append({"id": name, "processing_ps": processing})
        session = GpuPeerPacketSession(name, profile_for(False, credits=1, receiver_capacity=272),
                                       NvlinkPhysicalBinding(physical_domain(2, False), processing, 200000))
        session.admit("step0", "dispatch", (NvlinkTransfer(extent_id="first", source=0, destination=1, payload_bytes=256),))
        first = session.advance_until_visible(("first",))
        partial = session.evidence()
        session.admit("step1", "dispatch", (NvlinkTransfer(extent_id="second", source=0, destination=1, payload_bytes=256, released_at_ps=first),))
        second = session.advance_until_visible(("second",))
        result = session.drain()
        final = session.evidence()
        write(name + ".json", {"partial": partial, "final": final})
        oracle(name + "-first-visible", first, 22760)
        oracle(name + "-second-start", session.packets[1].tx_started_at_ps, 23760 + processing)
        oracle(name + "-second-visible", second, 46520 + processing)
        oracle(name + "-physical-drain", result.physical_drain_time_ps, 236640 + processing)
        guard(name + ":retained-ack", partial["has_pending_physical_work"] and bool(partial["port_snapshots"][0]["live_extent_tokens"]))
        check_session(name + "-partial", partial, False)
        check_session(name + "-final", final, True)

    for rate, capacity in product(frozen["attachment_control"]["link_bytes_per_second"], frozen["attachment_control"]["switch_input_buffer_bytes"]):
        name = f"attachment-rate{rate}-cap{capacity}"
        configurations.append({"id": name, "link_rate": rate, "input_capacity": capacity})
        results = []
        for permutation in ("baseline", "reverse", "class"):
            session = GpuPeerPacketSession(name, profile_for(True, rate=rate, feed=100000000000, capacity=capacity),
                                           NvlinkPhysicalBinding(physical_domain(3, True, rate=rate, capacity=capacity)))
            transfers = (NvlinkTransfer(extent_id="one", source=0, destination=1, payload_bytes=1024),
                         NvlinkTransfer(extent_id="two", source=0, destination=2, payload_bytes=1024))
            if permutation == "reverse":
                transfers = tuple(reversed(transfers))
            elif permutation == "class":
                transfers = tuple(replace(t, traffic_class=kind) for t, kind in zip(
                    transfers, (NvlinkTrafficClass.RESPONSE, NvlinkTrafficClass.NON_POSTED_REQUEST), strict=True))
            session.admit("step0", "dispatch", transfers)
            result = session.drain()
            raw = json.loads(canonical(session.evidence()))
            write(name + "-" + permutation + ".json", raw)
            check_session(name + "-" + permutation, raw, True)
            results.append((result, raw))
        first, reverse, classes = results
        guard(name + ":identity-class-permutation", strip_class(first[1]) == strip_class(classes[1]))
        guard(name + ":symmetric-order-makespan", first[0].completion_time_ps == reverse[0].completion_time_ps)
        guard(name + ":reversed-first-donor", min(first[0].packets, key=lambda p: p.visible_at_ps).destination == 1
              and min(reverse[0].packets, key=lambda p: p.visible_at_ps).destination == 2)
        rows = sorted(first[0].packets, key=lambda p: p.tx_started_at_ps)
        guard(name + ":one-attachment", len({(p["path"]["input_link"]["link_id"], p["path"]["source_port_id"]) for p in first[1]["physical_paths"]}) == 1)
        guard(name + ":attachment-byte-floor", rows[-1].tx_finished_at_ps >= sum(p.wire_bytes for p in rows) * 10**12 // rate)

    compat = json.loads(compatibility.read_bytes())
    git("diff", "--quiet", compat["current_commit"], "HEAD", "--", "simllm", "pyproject.toml")
    write("compatibility-input.json", compat)
    guard("compatibility:baseline-identity", compat["baseline_commit"] == frozen["compatibility_baseline_commit"])
    guard("compatibility:family-inventory", set(compat["families"]) == set(frozen["compatibility_families"]))
    for family, value in compat["families"].items():
        for side in ("before", "after"):
            bundle = compatibility.parent / f"{family}-{side}.bin"
            guard(f"compatibility:{family}:{side}-bundle", digest(bundle.read_bytes()) == value[f"{side}_sha256"])
        guard("compatibility:" + family, value["equal"] and value["before_sha256"] == value["after_sha256"])
    guard("timing-family-inventory", {row["family"] for row in relations} == set(frozen["behavioral_relations"]))
    source_paths = git("ls-files", "simllm", "examples/local_peer_packet_runtime_v1", "pyproject.toml").decode().splitlines()
    summary = {"schema": "simllm-local-peer-packet-study-v1", "expectations_commit": FREEZE,
               "source_commit": git("rev-parse", "HEAD").decode().strip(),
               "source_sha256": {p: digest((ROOT / p).read_bytes()) for p in source_paths},
               "evidence_class": "declared_model_not_hardware_measurement", "hardware_measurements": 0,
               "configurations": configurations, "exact_oracles": oracles, "behavioral_relations": relations,
               "structural_guards": guards, "compatibility": compat, "raw_sha256": raw_files,
               "fatal_findings": findings, "verdict": "VOID" if findings else "PASS" if all(r["passed"] for r in (*oracles, *relations)) else "FAIL",
               "behavioral_score": None if findings else {"passed_instances": sum(r["passed"] for r in relations),
                                                          "instances": len(relations), "families": len({r["family"] for r in relations})}}
    write("summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compatibility", type=Path, required=True, help="baseline/current canonical artifact comparison manifest")
    args = parser.parse_args()
    result = run_study(args.output.resolve(), args.compatibility.resolve())
    print(json.dumps({key: result[key] for key in ("verdict", "fatal_findings", "behavioral_score")}, indent=2))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
