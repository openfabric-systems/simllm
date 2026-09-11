"""Run source geometry, finite-resource relations and protocol interventions."""

import argparse
import csv
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import graph_study
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simllm.backends.htsim_nvlink import (
    NvlinkAlignedOptions,
    NvlinkSwitchConfig,
    NvlinkSwitchMode,
    NvlinkTransfer,
    load_nvlink_candidate_profile,
)
from simllm.backends.nccl_resources import NcclGpuProfile, NcclGpuResources
from simllm.backends.nccl_runtime import NcclChannelRuntime, NcclConnection
from simllm.backends.nvlink_runtime import NvlinkPhysicalBinding
from simllm.compute.gpu_packet_port import GpuPeerPacketSession
from simllm.placement import FabricLink, PeerFabric, PeerPortPlacement, PeerRoute
from simllm.traffic.collective_protocol import ring_geometry
from simllm.traffic.nccl_program import NcclRingProgram, balanced_channels, protocol_stripes

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def session(width=2, rate=25_000_000_000, *, inputs_only=False):
    base = load_nvlink_candidate_profile(ROOT / "examples/a100_nvlink_packet_v1/candidate-profile-pre-traf70.json")
    profile = replace(base, tx=replace(base.tx, links_per_peer=1, per_link_rate_bytes_per_second=rate,
                                      endpoint_egress_rate_bytes_per_second=100_000_000_000,
                                      credits_per_destination=256),
                      rx=replace(base.rx, ingress_rate_bytes_per_second=100_000_000_000,
                                 buffer_capacity_bytes=65536, credit_return_latency_ps=0),
                      switch=NvlinkSwitchConfig(mode=NvlinkSwitchMode.PASS_THROUGH))
    ports, links, routes = [], [], []
    for source in range(width):
        for destination in range(source + 1, width):
            a, b, name = f"p{source}-{destination}", f"p{destination}-{source}", f"l{source}-{destination}"
            ports += [PeerPortPlacement(a, gpu_rank=source), PeerPortPlacement(b, gpu_rank=destination)]
            links.append(FabricLink(name, a, b, rate * 8, 1000))
            routes += [PeerRoute(source, destination, ((name,),)), PeerRoute(destination, source, ((name,),))]
    fabric = PeerFabric("peer-domain", "node-0", tuple(ports), tuple(links), tuple(routes), switch_input_buffer_bytes=65536)
    binding = NvlinkPhysicalBinding(fabric)
    if inputs_only:
        return profile, binding
    return GpuPeerPacketSession("source", profile, binding, options=NvlinkAlignedOptions())


def write_table(output, name, rows):
    with (output / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def residency(cfg, output):
    rows = []
    for channels, sms in itertools.product(cfg["channels"], cfg["available_sms"]):
        calendar = session()
        resources = NcclGpuResources(calendar, NcclGpuProfile(available_sms=sms, **cfg["synthetic_gpu"]))
        completions = []
        for index in range(channels):
            block = f"c{index}"

            def job(block=block, completions=completions, calendar=calendar, resources=resources):
                def done():
                    completions.append(calendar.now_ps)
                    resources.release(block)
                resources.service(block, "fixture", 1000, done)

            resources.admit(block, 0, 4, job)
        calendar.advance_until(lambda completions=completions, channels=channels: len(completions) == channels)
        expected = ((channels + sms - 1) // sms) * cfg["residency_job_service_ps"]
        assert max(completions) == expected
        rows.append({"channels": channels, "sms": sms, "completion_ps": max(completions),
                     "oracle_ps": expected, "issued_cycles": sum(v.units for v in resources.visits)})
    write_table(output, "residency.csv", rows)
    return len(rows)


def window_cell(protocol, count, delay):
    calendar = session()
    conn = NcclConnection(("fixture", 0, 0, 1, 0))
    primitive = NcclRingProgram(64, (0, 1), protocol, (64,), 4).primitives(0, 0)[0]
    issued, completed = [], []

    def pump():
        conn.cached_head = conn.returned_head
        while len(issued) < count and conn.can_reserve(primitive.steps):
            reservation = conn.reserve(primitive, protocol, "fixture")
            ordinal = len(issued)
            issued.append(calendar.now_ps)

            def returned(reservation=reservation, ordinal=ordinal):
                reservation.ready = True
                conn.consume(reservation.sequence)
                conn.return_head(reservation.sequence + reservation.steps)
                completed.append(ordinal)
                pump()

            calendar.schedule_callback(calendar.now_ps + delay, returned)

    pump()
    calendar.advance_until(lambda: len(completed) == count)
    capacity = 8 // primitive.steps
    rows = []
    for index, at in enumerate(issued):
        expected = (index // capacity) * delay
        assert at == expected
        rows.append({"protocol": protocol, "count": count, "reuse_delay_ps": delay,
                     "reservation": index + 1, "steps": primitive.steps, "issued_at_ps": at,
                     "oracle_ps": expected})
    return rows


def windows(cfg, output):
    rows = []
    for protocol, count, delay in itertools.product(cfg["protocols"], cfg["reservation_counts"], cfg["receiver_delays_ps"]):
        rows.extend(window_cell(protocol, count, delay))
    write_table(output, "window.csv", rows)
    return len(cfg["protocols"]) * len(cfg["reservation_counts"]) * len(cfg["receiver_delays_ps"])


def isolated_serialization(cfg, output):
    rows = []
    for size, multiplier in itertools.product(cfg["serializer_bytes"], cfg["link_rate_multipliers"]):
        rate = int(cfg["serializer_base_rate_bytes_per_second"] * multiplier)
        physical = session(2, rate)
        transfer = NvlinkTransfer(extent_id="isolated", source=0, destination=1, payload_bytes=size, released_at_ps=0)
        physical.admit("serializer", "one", (transfer,))
        physical.advance_until_visible(("isolated",))
        for packet in physical.packets:
            measured = packet.tx_finished_at_ps - packet.tx_started_at_ps
            oracle = (packet.wire_bytes * 10**12 + rate - 1) // rate
            assert measured == oracle
            rows.append({"payload_bytes": size, "rate_bytes_per_second": rate,
                         "packet": packet.sequence, "wire_bytes": packet.wire_bytes,
                         "service_ps": measured, "oracle_ps": oracle})
        physical.drain()
    write_table(output, "serialization.csv", rows)
    return len(cfg["serializer_bytes"]) * len(cfg["link_rate_multipliers"])


def shared_serialization(staging, cfg, output):
    rows = []
    for size, multiplier, sms in itertools.product(staging["shared_serializer_bytes"],
                                                   staging["shared_rate_multipliers"], [1, 2]):
        calendar = session()
        rate = int(staging["shared_memory_bytes_per_second"] * multiplier)
        profile = NcclGpuProfile(available_sms=sms, **{**cfg["synthetic_gpu"], "shared_memory_bytes_per_second": rate})
        resources = NcclGpuResources(calendar, profile)
        completions = []
        for index in range(2):
            block = f"b{index}"

            def job(block=block, completions=completions, calendar=calendar, resources=resources, size=size):
                def finish():
                    completions.append(calendar.now_ps)
                    resources.release(block)
                resources.service(block, "shared", size, finish, shared=True)

            resources.admit(block, 0, 4, job)
        calendar.advance_until(lambda completions=completions: len(completions) == 2)
        oracle = ((size * 10**12 + rate - 1) // rate) * (2 if sms == 1 else 1)
        assert max(completions) == oracle
        rows.append({"bytes_per_block": size, "sms": sms, "rate_bytes_per_second": rate,
                     "completion_ps": max(completions), "oracle_ps": oracle})
    write_table(output, "shared_serialization.csv", rows)
    return len(rows)


def source_geometry(output):
    stripes = []
    for protocol in ["LL", "LL128", "SIMPLE"]:
        for payload in range(4, 4097, 4):
            items = protocol_stripes(payload, protocol, 4)
            stripes.append({"protocol": protocol, "useful_bytes": payload,
                            "encoded_bytes": sum(s.encoded_bytes for s in items),
                            "local_load_bytes": sum(s.local_load_bytes for s in items),
                            "shared_staging_bytes": sum(s.shared_staging_bytes for s in items),
                            "shared_tail_load_bytes": sum(s.shared_tail_load_bytes for s in items)})
    write_table(output, "stripe_geometry.csv", stripes)
    rows = []
    with (ROOT / "examples/nccl_protocol_model_v1/measurements/independent_predictions.csv").open() as stream:
        for old in csv.DictReader(stream):
            if old["architecture"] != "a100" or old["lane"] != "timing" or int(old["width"]) != 4:
                continue
            payload = int(old["bytes"])
            if not 3 * 1048576 <= payload <= 3.5 * 1048576:
                continue
            geometry = ring_geometry(payload, 4, "LL128")
            program = NcclRingProgram(payload, (0, 1, 2, 3), "LL128", geometry.channel_payload_bytes, geometry.warps)
            primitives = [p for channel in range(len(program.channel_bytes)) for rank in range(4)
                          for p in program.primitives(channel, rank)]
            rows.append({"payload_bytes": payload, "channels": len(program.channel_bytes),
                         "measured_us": float(old["median_us"]), "q1_us": float(old["q1_us"]),
                         "q3_us": float(old["q3_us"]), "old_model_us": float(old["reference_us"]),
                         "max_channel_encoded_bytes": geometry.maximum_channel_encoded_bytes,
                         "source_encoded_bytes_per_rank": program.encoded_network_bytes / 4,
                         "source_input_load_bytes_per_rank": sum(s.local_load_bytes for p in primitives if p.source_load for s in p.stripes) / 4,
                         "source_output_bytes_per_rank": sum(p.useful_bytes for p in primitives if p.output_store) / 4})
    if not rows:
        raise RuntimeError("retrospective A100 reference rows missing")
    write_table(output, "a100_retrospective.csv", rows)


def protocols(cfg, output):
    rows = []
    for width, protocol, channels, sms, multiplier, payload in itertools.product(
            cfg["widths"], cfg["protocols"], [1, 4], [1, 4], cfg["link_rate_multipliers"], cfg["fixed_payload_bytes"]):
        gpu = NcclGpuProfile(available_sms=sms, **cfg["synthetic_gpu"])
        physical = session(width, int(25_000_000_000 * multiplier))
        runtime = NcclChannelRuntime(physical, gpu)
        program = NcclRingProgram(payload, tuple(range(width)), protocol, balanced_channels(payload, channels),
                                  5 if protocol == "SIMPLE" else 4)
        result = runtime.run("study", "reduce", program)
        result.validate()
        rows.append({"width": width, "protocol": protocol, "channels": channels, "sms": sms,
                     "payload_bytes": payload, "link_rate_multiplier": multiplier, "duration_ps": result.duration_ps,
                     "useful_network_bytes": result.useful_network_bytes,
                     "protocol_data_bytes": result.protocol_data_bytes, "counter_bytes": result.counter_bytes,
                     "resource_work_wait_ps": sum(v.wait_ps for v in result.resource_visits),
                     "input_bytes": sum(v.units for v in result.resource_visits if v.kind == "input_load"),
                     "output_bytes": sum(v.units for v in result.resource_visits if v.kind == "output_store")})
        if (width, protocol, channels, sms, multiplier, payload) == (2, "LL128", 4, 1, 1, 16384):
            (output / "example_trace.json").write_text(json.dumps(result.summary(), indent=2) + "\n")
        physical.drain()
        print(f"protocol model cells {len(rows)}", flush=True)
    write_table(output, "protocols.csv", rows)
    return len(rows)


def sensitivities(cfg, output):
    # These finite interventions characterize the model's parameter surface.
    # They are post-specified diagnostics, not scored physical identifiability.
    parameters = ["kernel_entry_cycles", "block_setup_cycles", "warp_issue_cycles",
                  "barrier_cycles_per_warp", "publication_cycles", "poll_issue_cycles",
                  "poll_interval_cycles", "memory_bytes_per_second", "shared_memory_bytes_per_second"]
    rows, matrix = [], []
    for width, proto, channels, sms in itertools.product([2, 4], cfg["protocols"], [1, 4], [1, 4]):
        base = NcclGpuProfile(available_sms=sms, **cfg["synthetic_gpu"])
        program = NcclRingProgram(1024, tuple(range(width)), proto, balanced_channels(1024, channels),
                                  5 if proto == "SIMPLE" else 4)

        def run(profile, width=width, program=program):
            model = NcclChannelRuntime(session(width), profile)
            result = model.run("sensitivity", "o", program)
            model.session.drain()
            return result.duration_ps

        central = run(base)
        derivatives = []
        for name in parameters:
            old = getattr(base, name)
            new = max(old + 1, round(old * 1.05))
            changed = run(replace(base, **{name: new}))
            derivative = ((changed - central) / central) / ((new - old) / old)
            derivatives.append(derivative)
            rows.append({"width": width, "protocol": proto, "channels": channels, "sms": sms,
                         "parameter": name, "baseline_ps": central, "perturbed_ps": changed,
                         "fractional_parameter_change": (new - old) / old, "normalized_sensitivity": derivative})
        matrix.append(derivatives)
    values = np.asarray(matrix)
    singular = np.linalg.svd(values, compute_uv=False)
    result = {"parameters": parameters, "rows": len(matrix), "singular_values": singular.tolist(),
              "numerical_rank": int(np.sum(singular > singular[0] * 1e-8)),
              "column_cosines": (values.T @ values / np.outer(np.linalg.norm(values, axis=0), np.linalg.norm(values, axis=0))).tolist(),
              "evidence_class": "post_specified_model_sensitivity_not_hardware_identification"}
    (output / "sensitivity.json").write_text(json.dumps(result, indent=2) + "\n")
    write_table(output, "sensitivity.csv", rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((HERE / "expectations.json").read_text())
    staging = json.loads((HERE / "expectations_shared_staging.json").read_text())
    cfg["synthetic_gpu"]["shared_memory_bytes_per_second"] = staging["shared_memory_bytes_per_second"]
    summary = {"expectations_commit": "e7afffe9", "staging_correction_commit": "0929a8b8", "evidence_class": cfg["evidence_class"],
               "fatal_guards": "valid", "residency_instances": residency(cfg, args.output),
               "window_instances": windows(cfg, args.output)}
    summary["serialization_instances"] = isolated_serialization(cfg, args.output)
    summary["shared_memory_instances"] = shared_serialization(staging, cfg, args.output)
    graph_rows = graph_study.run(cfg, args.output, session)
    write_table(args.output, "live_graph.csv", graph_rows)
    summary["live_graph_configurations"] = len(graph_rows) // 2
    summary["fixed_compute_ps"] = graph_rows[0]["fixed_compute_ps"]
    source_geometry(args.output)
    summary["protocol_configurations"] = protocols(cfg, args.output)
    if not args.skip_sensitivity:
        summary["sensitivity"] = sensitivities(cfg, args.output)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
