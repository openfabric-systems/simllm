"""Execute the frozen causal NVLink correctness study without GPU hardware."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from itertools import pairwise, product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simllm.backends.htsim_nvlink import (
    NvlinkAlignedOptions,
    NvlinkDomainService,
    NvlinkFifoPlacement,
    NvlinkFlowControlConfig,
    NvlinkOperation,
    NvlinkPacketDirection,
    NvlinkSwitchConfig,
    NvlinkSwitchMode,
    NvlinkTrafficClass,
    NvlinkTransfer,
    load_nvlink_candidate_profile,
)

HERE = Path(__file__).resolve().parent
EXPECTATIONS_COMMIT = "9787af3"
BASE_COMMIT = "043c8cfd53054fb5ec8102d0173730480ab6df1f"
PROFILE_PATH = ROOT / "examples/a100_nvlink_packet_v1/candidate-profile-pre-traf70.json"
SOURCE_PATHS = (
    "simllm/backends/htsim_nvlink.py",
    "simllm/backends/nvlink_runtime.py",
    "examples/nvlink_causal_service_v1/run_study.py",
)


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def make_profile(rate, credits=4, capacity=4, return_ps=0, queued=False, slow_rx=False):
    base = load_nvlink_candidate_profile(PROFILE_PATH)
    return replace(
        base,
        tx=replace(
            base.tx,
            links_per_peer=1,
            credits_per_destination=credits,
            per_link_rate_bytes_per_second=rate,
            endpoint_egress_rate_bytes_per_second=100_000_000_000,
        ),
        rx=replace(
            base.rx,
            buffer_capacity_bytes=272 * capacity,
            ingress_rate_bytes_per_second=(10_000_000_000 if slow_rx else 100_000_000_000),
            credit_return_latency_ps=return_ps,
        ),
        switch=(
            NvlinkSwitchConfig(
                mode=NvlinkSwitchMode.QUEUED,
                fifo_placement=NvlinkFifoPlacement.INPUT,
                service_rate_bytes_per_second=25_000_000_000,
                buffer_capacity_bytes=272,
                arbitration="fifo",
                head_of_line_blocking=True,
            )
            if queued
            else base.switch
        ),
    )


def transfer(name="job", source=0, destination=1, size=1024, release=0, read=False):
    return NvlinkTransfer(
        extent_id=name,
        source=source,
        destination=destination,
        payload_bytes=size,
        released_at_ps=release,
        operation=NvlinkOperation.PEER_READ if read else NvlinkOperation.PEER_WRITE,
    )


def packet_times(result, extent):
    return sorted(
        (
            p.packet_id,
            p.tx_eligible_at_ps,
            p.tx_started_at_ps,
            p.tx_finished_at_ps,
            p.rx_started_at_ps,
            p.rx_finished_at_ps,
            p.visible_at_ps,
        )
        for p in result.packets
        if p.extent_id == extent
    )


def guard_findings(result, profile, inputs, options=None):
    """Join every projection to input identities, physical bounds and its owner."""
    findings = []
    options = options or NvlinkAlignedOptions()
    flow = options.flow_control or NvlinkFlowControlConfig.from_candidate_profile(profile)
    queued = profile.switch.mode is NvlinkSwitchMode.QUEUED

    def require(condition, name):
        if not condition:
            findings.append(name)

    def service(byte_count, rate):
        return (byte_count * 10**12 + rate - 1) // rate

    packets = result.packets
    by_id = {p.packet_id: p for p in packets}
    ids = set(by_id)
    expected = {}
    for transfer_input in inputs:
        maximum = min(
            profile.tx.max_payload_bytes, options.packet_format.maximum_payload_flits * 16
        )
        payloads = [
            min(maximum, transfer_input.payload_bytes - offset)
            for offset in range(0, transfer_input.payload_bytes, maximum)
        ]
        read = transfer_input.operation is NvlinkOperation.PEER_READ
        if read:
            payloads.insert(0, 0)
        offered_bytes = 0
        for sequence, payload in enumerate(payloads):
            response = read and sequence > 0
            source, destination = transfer_input.source, transfer_input.destination
            if response:
                source, destination = destination, source
            wire = 16 * (
                options.packet_format.header_flits
                + transfer_input.address_extension_flits
                + transfer_input.byte_enable_flits
                + (payload + 15) // 16
            )
            released = transfer_input.released_at_ps
            if transfer_input.offered_rate_bytes_per_second is not None:
                released += service(offered_bytes, transfer_input.offered_rate_bytes_per_second)
            expected[f"{transfer_input.extent_id}:packet-{sequence}"] = (
                transfer_input.extent_id,
                sequence,
                source,
                destination,
                NvlinkPacketDirection.RESPONSE if response else NvlinkPacketDirection.REQUEST,
                payload,
                wire,
                released,
                transfer_input.virtual_channel,
                transfer_input.ordering_domain or transfer_input.extent_id,
            )
            offered_bytes += wire
    require(ids == set(expected) and len(ids) == len(packets), "exact input packet identities")
    require(
        sum(p.payload_bytes for p in packets)
        == sum(t.payload_bytes for t in inputs)
        == result.logical_bytes,
        "logical byte conservation",
    )
    require(
        result.total_wire_bytes == sum(p.wire_bytes + p.replay_wire_bytes for p in packets),
        "wire byte conservation",
    )
    require(
        result.request_wire_bytes
        == sum(p.wire_bytes for p in packets if p.direction is NvlinkPacketDirection.REQUEST),
        "request wire projection",
    )
    require(
        result.response_wire_bytes
        == sum(p.wire_bytes for p in packets if p.direction is NvlinkPacketDirection.RESPONSE),
        "response wire projection",
    )
    require(
        result.completion_time_ps == max((p.visible_at_ps or 0 for p in packets), default=0),
        "logical completion projection",
    )
    require(result.physical_drain_time_ps >= result.completion_time_ps, "finite physical drain")
    require(result.fixed_point_iterations == 0, "no iterative timing authority")
    require(result.acknowledgement_count == len(packets), "acknowledgement count")
    visibility = {v.packet_id: v for v in result.visibility_events}
    releases = {r.packet_id: r for r in result.credit_releases}
    grants = {g.packet_id: g for g in result.switch_grants}
    visits = {(v.packet_id, v.buffer_id): v for v in result.buffer_visits}
    require(
        set(visibility) == ids and len(visibility) == len(result.visibility_events),
        "one visibility per packet",
    )
    require(
        set(releases) == ids and len(releases) == len(result.credit_releases),
        "one first-hop credit return",
    )
    require(
        set(grants) == (ids if queued else set()) and len(grants) == len(result.switch_grants),
        "exact switch grant identities",
    )
    expected_visits = {(p.packet_id, f"rx:{p.destination}:{p.virtual_channel}") for p in packets}
    if queued:
        expected_visits |= {
            (p.packet_id, f"switch:{p.source}:{p.destination}:{p.virtual_channel}") for p in packets
        }
    require(
        set(visits) == expected_visits and len(visits) == len(result.buffer_visits),
        "exact buffer reservation identities",
    )
    links, endpoints, receivers, domains = (defaultdict(list) for _ in range(4))
    replay_counts = dict(options.replay_counts)
    for p in packets:
        actual = (
            p.extent_id,
            p.sequence,
            p.source,
            p.destination,
            p.direction,
            p.payload_bytes,
            p.wire_bytes,
            p.released_at_ps,
            p.virtual_channel,
            p.ordering_domain,
        )
        require(actual == expected.get(p.packet_id), f"input packet geometry {p.packet_id}")
        times = (
            p.tx_eligible_at_ps,
            p.tx_started_at_ps,
            p.tx_finished_at_ps,
            p.rx_buffer_accepted_at_ps,
            p.rx_started_at_ps,
            p.rx_finished_at_ps,
            p.rx_buffer_released_at_ps,
            p.visible_at_ps,
        )
        require(all(t is not None for t in times), f"complete timestamps {p.packet_id}")
        if any(t is None for t in times):
            continue
        require(list(times) == sorted(times), f"causal timestamps {p.packet_id}")
        require(p.tx_eligible_at_ps >= p.released_at_ps, f"no future grant {p.packet_id}")
        if p.direction is NvlinkPacketDirection.RESPONSE:
            request = by_id.get(f"{p.extent_id}:packet-0")
            require(
                request is not None and request.direction is NvlinkPacketDirection.REQUEST,
                f"read request present {p.packet_id}",
            )
            if request is not None and request.visible_at_ps is not None:
                require(
                    p.tx_eligible_at_ps == max(p.released_at_ps, request.visible_at_ps),
                    f"read dependency {p.packet_id}",
                )
        else:
            require(p.tx_eligible_at_ps == p.released_at_ps, f"write eligibility {p.packet_id}")
        require(
            p.link_index is not None and 0 <= p.link_index < profile.tx.links_per_peer,
            f"physical link identity {p.packet_id}",
        )
        require(
            p.replay_count == replay_counts.get(p.packet_id, 0), f"replay request {p.packet_id}"
        )
        require(
            p.replay_wire_bytes == p.replay_count * p.wire_bytes,
            f"replay byte conservation {p.packet_id}",
        )
        per_attempt = max(
            service(p.wire_bytes, profile.tx.per_link_rate_bytes_per_second),
            service(p.wire_bytes, profile.tx.endpoint_egress_rate_bytes_per_second),
        )
        require(
            p.tx_finished_at_ps - p.tx_started_at_ps
            >= (1 + p.replay_count) * per_attempt + p.replay_count * options.replay_timeout_ps,
            f"source and link floor {p.packet_id}",
        )
        endpoint_end = (
            p.tx_finished_at_ps
            if p.replay_count
            else p.tx_started_at_ps
            + service(p.wire_bytes, profile.tx.endpoint_egress_rate_bytes_per_second)
        )
        endpoints[p.source].append((p.tx_started_at_ps, endpoint_end))
        links[(p.source, p.destination, p.link_index)].append(
            (p.tx_started_at_ps, p.tx_finished_at_ps)
        )
        receivers[p.destination].append((p.rx_started_at_ps, p.rx_finished_at_ps))
        domains[(p.destination, p.ordering_domain)].append((p.sequence, p.visible_at_ps))
        require(
            p.rx_finished_at_ps - p.rx_started_at_ps
            == service(p.wire_bytes, profile.rx.ingress_rate_bytes_per_second),
            f"receiver service {p.packet_id}",
        )
        require(
            p.acknowledged_at_ps
            == p.tx_finished_at_ps + options.acknowledgement_latency_ps
            == p.replay_buffer_released_at_ps,
            f"acknowledgement projection {p.packet_id}",
        )
        v = visibility.get(p.packet_id)
        if v is not None:
            require(
                (v.ordering_domain, v.sequence, v.rx_finished_at_ps, v.visible_at_ps)
                == (p.ordering_domain, p.sequence, p.rx_finished_at_ps, p.visible_at_ps),
                f"visibility projection {p.packet_id}",
            )
        rx_id = f"rx:{p.destination}:{p.virtual_channel}"
        switch_id = f"switch:{p.source}:{p.destination}:{p.virtual_channel}"
        owners = [
            (
                rx_id,
                profile.rx.buffer_capacity_bytes,
                p.switch_started_at_ps if queued else p.tx_started_at_ps,
                p.switch_finished_at_ps if queued else p.tx_finished_at_ps,
                p.rx_finished_at_ps,
            )
        ]
        if queued:
            owners.append(
                (
                    switch_id,
                    profile.switch.buffer_capacity_bytes,
                    p.tx_started_at_ps,
                    p.tx_finished_at_ps,
                    p.switch_finished_at_ps,
                )
            )
            grant = grants.get(p.packet_id)
            require(
                p.tx_finished_at_ps <= p.switch_started_at_ps <= p.switch_finished_at_ps,
                f"switch causality {p.packet_id}",
            )
            require(
                p.switch_finished_at_ps - p.switch_started_at_ps
                == service(p.wire_bytes, profile.switch.service_rate_bytes_per_second),
                f"switch service {p.packet_id}",
            )
            if grant is not None:
                require(
                    (
                        grant.input_port,
                        grant.output_port,
                        grant.virtual_channel,
                        grant.started_at_ps,
                        grant.finished_at_ps,
                        grant.policy,
                    )
                    == (
                        p.source,
                        p.destination,
                        p.virtual_channel,
                        p.switch_started_at_ps,
                        p.switch_finished_at_ps,
                        options.switch_arbitration,
                    ),
                    f"switch grant projection {p.packet_id}",
                )
        else:
            require(
                p.switch_started_at_ps is None and p.switch_finished_at_ps is None,
                f"direct mesh identity {p.packet_id}",
            )
        for buffer_id, capacity, reserved, arrived, released in owners:
            visit = visits.get((p.packet_id, buffer_id))
            if visit is not None:
                require(
                    (
                        visit.capacity_bytes,
                        visit.wire_bytes,
                        visit.reserved_at_ps,
                        visit.arrived_at_ps,
                        visit.released_at_ps,
                        visit.credit_available_at_ps,
                    )
                    == (
                        capacity,
                        p.wire_bytes,
                        reserved,
                        arrived,
                        released,
                        released + flow.return_transport_latency_ps,
                    ),
                    f"buffer owner projection {p.packet_id} {buffer_id}",
                )
        receive = visits.get((p.packet_id, rx_id))
        if receive is not None:
            require(
                receive.arrived_at_ps == p.rx_buffer_accepted_at_ps
                and receive.released_at_ps == p.rx_buffer_released_at_ps,
                f"receive projection {p.packet_id}",
            )
        release = releases.get(p.packet_id)
        if release is not None:
            owner_id = switch_id if queued else rx_id
            freed = p.switch_finished_at_ps if queued else p.rx_finished_at_ps
            require(
                (
                    release.source,
                    release.destination,
                    release.link_index,
                    release.virtual_channel,
                    release.credit_units,
                    release.buffer_id,
                    release.buffer_released_at_ps,
                    release.credit_available_at_ps,
                )
                == (
                    p.source,
                    p.destination,
                    p.link_index,
                    p.virtual_channel,
                    p.credit_units,
                    owner_id,
                    freed,
                    freed + flow.return_transport_latency_ps,
                ),
                f"credit projection {p.packet_id}",
            )
            require(
                p.credit_available_at_ps == release.credit_available_at_ps,
                f"packet credit projection {p.packet_id}",
            )
    for kind, groups in (("link", links), ("endpoint", endpoints), ("receiver", receivers)):
        for key, intervals in groups.items():
            require(
                all(a[1] <= b[0] for a, b in pairwise(sorted(intervals))),
                f"exclusive {kind} grant {key}",
            )
    for key, rows in domains.items():
        require(all(a[1] <= b[1] for a, b in pairwise(sorted(rows))), f"visibility order {key}")
    pools = defaultdict(list)
    for visit in result.buffer_visits:
        pools[visit.buffer_id].append(visit)
    for key, rows in pools.items():
        capacity = (
            profile.rx.buffer_capacity_bytes
            if key.startswith("rx:")
            else profile.switch.buffer_capacity_bytes
        )
        for start, end in (
            ("reserved_at_ps", "credit_available_at_ps"),
            ("reserved_at_ps", "released_at_ps"),
            ("arrived_at_ps", "released_at_ps"),
        ):
            deltas = sorted(
                [(getattr(v, start), v.wire_bytes) for v in rows]
                + [(getattr(v, end), -v.wire_bytes) for v in rows]
            )
            occupied = 0
            for _, delta in deltas:
                occupied += delta
                require(capacity is not None and 0 <= occupied <= capacity, f"finite {key} {start}")
            require(occupied == 0, f"drained {key} {start}")
    for field in ("input_port", "output_port"):
        groups = defaultdict(list)
        for grant in result.switch_grants:
            groups[getattr(grant, field)].append((grant.started_at_ps, grant.finished_at_ps))
        for port, intervals in groups.items():
            require(
                all(a[1] <= b[0] for a, b in pairwise(sorted(intervals))),
                f"exclusive switch {field} {port}",
            )
    credit_groups = defaultdict(list)
    for packet_id, release in releases.items():
        p = by_id.get(packet_id)
        if p is not None:
            key = (p.source, p.destination, p.link_index, p.virtual_channel)
            credit_groups[key] += [
                (p.tx_started_at_ps, p.credit_units),
                (release.credit_available_at_ps, -p.credit_units),
            ]
    for key, deltas in credit_groups.items():
        outstanding = 0
        for _, delta in sorted(deltas):
            outstanding += delta
            require(0 <= outstanding <= flow.credits_per_pool, f"finite link credit {key}")
        require(outstanding == 0, f"drained link credit {key}")
    return sorted(set(findings))


def historical_module(output_root):
    path = "simllm/backends/htsim_nvlink.py"
    source = git("show", f"{BASE_COMMIT}:{path}")
    name = "_frozen_nvlink_causal_study_base"
    source_path = output_root / "historical_htsim_nvlink.py"
    source_path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the frozen NVLink baseline source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, digest(source)


def controls(output_root):
    old, source_sha = historical_module(output_root)
    old_profile = old.load_nvlink_candidate_profile(PROFILE_PATH)
    new_profile = load_nvlink_candidate_profile(PROFILE_PATH)
    specifications = {
        "write": [{"extent_id": "a", "source": 0, "destination": 1, "payload_bytes": 1024}],
        "read": [
            {
                "extent_id": "r",
                "source": 0,
                "destination": 1,
                "payload_bytes": 1024,
                "operation": "peer_read",
            }
        ],
        "fanin": [
            {"extent_id": f"source-{i}", "source": i, "destination": 3, "payload_bytes": 1024}
            for i in (0, 1, 2)
        ],
        "staggered": [
            {
                "extent_id": "late",
                "source": 0,
                "destination": 1,
                "payload_bytes": 256,
                "released_at_ps": 1_000_000,
            },
            {"extent_id": "early", "source": 0, "destination": 1, "payload_bytes": 256},
        ],
    }
    rows = []
    for name, specs in specifications.items():
        old_inputs, new_inputs = [], []
        for spec in specs:
            spec = dict(spec)
            operation = spec.pop("operation", "peer_write")
            old_inputs.append(old.NvlinkTransfer(**spec, operation=old.NvlinkOperation(operation)))
            new_inputs.append(NvlinkTransfer(**spec, operation=NvlinkOperation(operation)))
        before = old.NvlinkDomainService(old_profile).serve(old_inputs, analytic_result=None)
        after = NvlinkDomainService(new_profile).serve(new_inputs, analytic_result=None)
        rows.append(
            {
                "case": name,
                "equal": before.canonical_json_bytes() == after.canonical_json_bytes(),
                "before_sha256": digest(before.canonical_json_bytes()),
                "after_sha256": digest(after.canonical_json_bytes()),
            }
        )
    sentinel = object()
    bypass = NvlinkDomainService().serve_aligned([], analytic_result=sentinel) is sentinel
    old_future = old.NvlinkDomainService(old_profile).serve_aligned(
        [old.NvlinkTransfer(**spec) for spec in specifications["staggered"]],
        analytic_result=None,
    )
    old_isolated = old.NvlinkDomainService(old_profile).serve_aligned(
        [old.NvlinkTransfer(**specifications["staggered"][1])],
        analytic_result=None,
    )
    old_read = old.NvlinkDomainService(old_profile).serve_aligned(
        [
            old.NvlinkTransfer(
                extent_id="read",
                source=0,
                destination=1,
                payload_bytes=256,
                operation=old.NvlinkOperation.PEER_READ,
            )
        ],
        analytic_result=None,
    )
    request = next(p for p in old_read.packets if p.direction is old.NvlinkPacketDirection.REQUEST)
    response = next(
        p for p in old_read.packets if p.direction is old.NvlinkPacketDirection.RESPONSE
    )
    findings = {
        "evidence_class": "historical_implementation_diagnostic_unscored",
        "base_source_sha256": source_sha,
        "future_insertion_early_visibility_shift_ps": (
            next(p.visible_at_ps for p in old_future.packets if p.extent_id == "early")
            - old_isolated.completion_time_ps
        ),
        "read_response_before_request_visibility_ps": request.visible_at_ps
        - response.tx_started_at_ps,
    }
    return rows, bypass, findings


def run_study(output_root):
    for path in SOURCE_PATHS:
        if (ROOT / path).read_bytes() != git("show", f"HEAD:{path}"):
            raise ValueError(f"commit the study source before execution: {path}")
    git("merge-base", "--is-ancestor", EXPECTATIONS_COMMIT, "HEAD")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    frozen_bytes = (HERE / "expectations.json").read_bytes()
    committed = git(
        "show", f"{EXPECTATIONS_COMMIT}:examples/nvlink_causal_service_v1/expectations.json"
    )
    if frozen_bytes != committed:
        raise ValueError("expectations changed after the final preimplementation freeze")
    frozen = json.loads(frozen_bytes)
    configurations, oracles, relations, findings = [], [], [], []
    structural_checks = []
    write_results = {}

    def relation(family, instance, condition, *, guard=False, observed=None, expected=None):
        guard = guard or family in {"future_insertion", "independent_resources"}
        row = {
            "family": family,
            "instance": instance,
            "verdict": "PASS" if condition else "FAIL",
            "observed_relation_ps": observed,
            "expected_relation_ps": expected,
        }
        if guard:
            structural_checks.append(row)
            if not condition:
                findings.append(f"structural guard: {family}/{instance}")
        else:
            relations.append(row)

    def cell(name, inputs, config, options=None):
        row = {
            "case": name,
            "input_count": len(inputs),
            "parameters": {
                "tx": asdict(config.tx),
                "rx": asdict(config.rx),
                "switch": asdict(config.switch),
            },
            "inputs": [asdict(t) for t in inputs],
            "options": asdict(options or NvlinkAlignedOptions()),
        }
        configurations.append(row)
        try:
            result = NvlinkDomainService(config).serve_aligned(
                inputs, analytic_result=None, options=options
            )
        except (AssertionError, ValueError, TypeError, RuntimeError) as error:
            finding = f"{type(error).__name__}: {error}"
            findings.append(f"{name}: {finding}")
            row.update(fatal_verdict="VOID", finding=finding)
            return None
        data = canonical({"configuration": row, "result": asdict(result)})
        filename = f"{name}.json"
        (output_root / filename).write_bytes(data)
        try:
            cell_findings = guard_findings(result, config, inputs, options)
        except (
            AssertionError,
            ValueError,
            TypeError,
            RuntimeError,
            KeyError,
            IndexError,
            AttributeError,
            StopIteration,
        ) as error:
            cell_findings = [f"checker failure: {type(error).__name__}: {error}"]
        findings.extend(f"{name}: {finding}" for finding in cell_findings)
        row.update(
            {
                "packet_count": len(result.packets),
                "completion_time_ps": result.completion_time_ps,
                "physical_drain_time_ps": result.physical_drain_time_ps,
                "fatal_verdict": "VOID" if cell_findings else "PASS",
                "evidence_file": filename,
                "evidence_sha256": digest(data),
            }
        )
        return None if cell_findings else result

    rates, sizes = frozen["link_rates_bytes_per_second"], frozen["payload_bytes"]
    for rate, size, credits, capacity, delay in product(
        rates,
        sizes,
        frozen["credits_per_pool"],
        frozen["receiver_buffer_packets"],
        frozen["credit_return_latency_ps"],
    ):
        name = f"write-r{rate}-b{size}-c{credits}-m{capacity}-d{delay}"
        result = cell(name, [transfer(size=size)], make_profile(rate, credits, capacity, delay))
        if result is None:
            continue
        n, link, receive = size // 256, (272 * 10**12 + rate - 1) // rate, 2720
        expected = (
            n * (link + receive) + (n - 1) * delay
            if min(credits, capacity) == 1
            else n * link + receive
        )
        floor, ceiling = n * link, n * (link + receive) + (n - 1) * delay
        measured = result.completion_time_ps
        oracles.append(
            {
                "case": name,
                "expected_ps": expected,
                "observed_ps": measured,
                "residual_ps": measured - expected,
                "floor_ps": floor,
                "ceiling_ps": ceiling,
            }
        )
        if not floor <= measured <= ceiling:
            findings.append(f"{name}: physical bounds")
        write_results[(rate, size, credits, capacity, delay)] = result
    for rate, size in product(rates, sizes):
        name = f"read-r{rate}-b{size}"
        config = make_profile(rate)
        result = cell(name, [transfer(size=size, read=True)], config)
        if result is None:
            continue
        n, link = size // 256, (272 * 10**12 + rate - 1) // rate
        request = (16 * 10**12 + rate - 1) // rate + 160
        expected = request + n * link + 2720
        floor, ceiling = request + n * link, request + n * (link + 2720)
        measured = result.completion_time_ps
        oracles.append(
            {
                "case": name,
                "expected_ps": expected,
                "observed_ps": measured,
                "residual_ps": measured - expected,
                "floor_ps": floor,
                "ceiling_ps": ceiling,
            }
        )
        if not floor <= measured <= ceiling:
            findings.append(f"{name}: physical bounds")
        baseline = write_results.get((rate, size, 4, 4, 0))
        if baseline is not None:
            delta = measured - baseline.completion_time_ps
            relation("read_dependency", name, delta == request, observed=delta, expected=request)
        extra = cell(
            name + "-competing",
            [
                transfer("read", size=size, read=True),
                transfer("target-write", source=1, destination=2, size=256),
            ],
            config,
        )
        if extra:
            request_packet = next(
                p
                for p in extra.packets
                if p.extent_id == "read" and p.direction is NvlinkPacketDirection.REQUEST
            )
            relation(
                "read_dependency",
                name + "-competing",
                all(
                    p.tx_started_at_ps >= request_packet.visible_at_ps
                    for p in extra.packets
                    if p.direction is NvlinkPacketDirection.RESPONSE
                ),
                guard=True,
            )
    for size in sizes:
        slow = write_results.get((rates[0], size, 4, 4, 0))
        fast = write_results.get((rates[1], size, 4, 4, 0))
        if slow is not None and fast is not None:
            observed = slow.completion_time_ps - 2720
            expected = 2 * (fast.completion_time_ps - 2720)
            relation(
                "bandwidth_scaling",
                f"payload-{size}",
                observed == expected,
                observed=observed,
                expected=expected,
            )
    for rate in rates:
        small = write_results.get((rate, sizes[0], 4, 4, 0))
        large = write_results.get((rate, sizes[1], 4, 4, 0))
        if small is not None and large is not None:
            observed = large.completion_time_ps - 2720
            expected = 4 * (small.completion_time_ps - 2720)
            relation(
                "payload_scaling",
                f"rate-{rate}",
                observed == expected,
                observed=observed,
                expected=expected,
            )
        for credits, capacity in ((1, 4), (4, 1), (1, 1)):
            immediate = write_results.get((rate, 1024, credits, capacity, 0))
            delayed = write_results.get((rate, 1024, credits, capacity, 10000))
            if immediate is not None and delayed is not None:
                delta = delayed.completion_time_ps - immediate.completion_time_ps
                relation(
                    "backpressure",
                    f"return-r{rate}-c{credits}-m{capacity}",
                    delta == 30000,
                    observed=delta,
                    expected=30000,
                )
        for delay in (0, 10000):
            narrow = write_results.get((rate, 1024, 4, 1, delay))
            wide = write_results.get((rate, 1024, 4, 4, delay))
            if narrow is not None and wide is not None:
                delta = narrow.completion_time_ps - wide.completion_time_ps
                expected = 3 * (2720 + delay)
                relation(
                    "backpressure",
                    f"capacity-r{rate}-d{delay}",
                    delta == expected,
                    observed=delta,
                    expected=expected,
                )
    for rate in rates:
        config = make_profile(rate)
        early = transfer("early", size=256)
        alone = cell(f"early-r{rate}", [early], config)
        for offset, order in product(frozen["future_offsets_ps"], (0, 1)):
            name = f"future-r{rate}-o{offset}-order{order}"
            inputs = [early, transfer("future", size=256, release=offset)]
            result = cell(name, inputs[::-1] if order else inputs, config)
            if result and alone:
                relation(
                    "future_insertion",
                    name,
                    packet_times(result, "early") == packet_times(alone, "early"),
                )
        duplex = cell(
            f"duplex-r{rate}",
            [transfer("a", size=256), transfer("b", source=1, destination=0, size=256)],
            config,
        )
        if duplex:
            relation(
                "independent_resources",
                f"duplex-r{rate}",
                all(p.tx_started_at_ps == 0 for p in duplex.packets),
            )
        for queued in (False, True):
            name = f"disjoint-r{rate}-q{int(queued)}"
            config = make_profile(rate, queued=queued)
            baseline = cell(name + "-alone", [early], config)
            combined = cell(
                name + "-combined", [early, transfer("other", source=2, destination=3)], config
            )
            if baseline and combined:
                relation(
                    "independent_resources",
                    name,
                    packet_times(baseline, "early") == packet_times(combined, "early"),
                )
            clean = cell(name + "-clean", [transfer(size=256)], config)
            if clean:
                replay = cell(
                    name + "-replay",
                    [transfer(size=256)],
                    config,
                    NvlinkAlignedOptions(
                        replay_counts=((clean.packets[0].packet_id, 1),),
                        replay_timeout_ps=100,
                    ),
                )
                if replay:
                    relation(
                        "replay",
                        name,
                        replay.total_wire_bytes == clean.total_wire_bytes + 272
                        and replay.completion_time_ps
                        == clean.completion_time_ps + (272 * 10**12 + rate - 1) // rate + 100,
                    )
            for delay in frozen["credit_return_latency_ps"]:
                config = make_profile(
                    rate, capacity=1, return_ps=delay, queued=queued, slow_rx=True
                )
                inputs = [transfer(f"source-{i}", source=i, destination=3) for i in (0, 1, 2)]
                name = f"fanin-r{rate}-q{int(queued)}-d{delay}"
                result = cell(name, inputs, config)
                relabeled = cell(
                    name + "-classes",
                    [replace(t, traffic_class=NvlinkTrafficClass.RESPONSE) for t in inputs],
                    config,
                )
                if result and relabeled:
                    same = (
                        result.buffer_visits == relabeled.buffer_visits
                        and result.visibility_events == relabeled.visibility_events
                        and result.switch_grants == relabeled.switch_grants
                    )
                    if not same:
                        findings.append(f"{name}: class identity off mode changed")
                    serial_ceiling = 12 * (
                        (272 * 10**12 + rate - 1) // rate
                        + 27200
                        + (10880 if queued else 0)
                        + 2 * delay
                    )
                    relation(
                        "finite_drain",
                        name,
                        result.completion_time_ps <= serial_ceiling,
                        guard=True,
                    )
    control_rows, bypass, historical = controls(output_root)
    if not all(row["equal"] for row in control_rows) or not bypass:
        findings.append("compatibility or analytic identity control failed")
    refuted = any(row["residual_ps"] != 0 for row in oracles) or any(
        row["verdict"] != "PASS" for row in relations
    )
    verdict = "VOID" if findings else "REFUTED" if refuted else "PASS"
    result = {
        "schema": "simllm-nvlink-causal-service-result-v1",
        "verdict": verdict,
        "evidence_class": frozen["evidence_class"],
        "hardware_measurement": False,
        "expectations_commit": git("rev-parse", EXPECTATIONS_COMMIT).decode().strip(),
        "expectations_sha256": digest(frozen_bytes),
        "base_commit": BASE_COMMIT,
        "implementation_commit": git("rev-parse", "HEAD").decode().strip(),
        "source_sha256": {path: digest((ROOT / path).read_bytes()) for path in SOURCE_PATHS},
        "configuration_count": len(configurations),
        "configurations": configurations,
        "exact_oracle_count": len(oracles),
        "exact_oracles": oracles,
        "behavioral_family_count": len({row["family"] for row in relations}),
        "behavioral_instance_count": len(relations),
        "behavioral_score": None
        if findings
        else {
            "passed": sum(row["verdict"] == "PASS" for row in relations),
            "total": len(relations),
        },
        "behavioral_relations": relations,
        "fatal_verdict": "VOID" if findings else "PASS",
        "fatal_findings": findings,
        "structural_checks": structural_checks,
        "evidence_classification": (
            "Future causality, disjoint independence, class identity, finite drain "
            "and compatibility are unscored guards. Exact rows have their own count. "
            "Only nonzero timing relations enter the behavioral denominator."
        ),
        "compatibility_controls": control_rows,
        "analytic_identity": bypass,
        "historical_findings": historical,
        "closure_scope": "TRAF-90 component correctness only",
        "remaining_tasks": ["TRAF-45", "TRAF-73", "TRAF-86"],
    }
    (output_root / "summary.json").write_bytes(canonical(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = run_study(args.output_root)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "verdict",
                    "configuration_count",
                    "exact_oracle_count",
                    "behavioral_family_count",
                    "behavioral_instance_count",
                    "fatal_findings",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
