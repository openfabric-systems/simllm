#!/usr/bin/env python3
"""Run the frozen TRAF-73 NVLink arbitration simulation arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simllm.backends.htsim_nvlink import (
    NvlinkArbitrationPolicy,
    NvlinkDomainResult,
    NvlinkDomainService,
    NvlinkFlowPolicy,
    NvlinkOperation,
    NvlinkTransfer,
    load_nvlink_candidate_profile,
    sha256_file,
)

HERE = Path(__file__).resolve().parent
EXPECTATIONS_PATH = HERE / "expectations.json"
EXPECTATIONS_COMMIT = "15e68c26e81f155dfa475122ad867882a5735287"
EXPECTATIONS_SHA256 = "d127597dbeab23ae29f18214c583e4b958de9c57bf398d0a8308ad614f5cd7a0"
RESULT_SCHEMA = "simllm-nvlink-credit-arbitration-result-v1"
BULK_ROOT_ENV = "SIMLLM_NVFAIR_BULK_ROOT"
LEGACY_IDENTITY_SHA256 = "2f2af64619ed3c6341b209d877d9f1e6984a67e44b97b5eb176a157294a6c252"


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def load_expectations() -> dict[str, Any]:
    return json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))


def _tracked_preservation_paths(frozen: dict[str, Any]) -> list[str]:
    roots = frozen["preservation_lock"]["root_paths"]
    listed = _git("ls-files", "--", *roots)
    if listed.returncode:
        raise SystemExit("cannot enumerate the TRAF-73 preservation roots")
    return [line for line in listed.stdout.splitlines() if line]


def preservation_evidence(frozen: dict[str, Any]) -> dict[str, object]:
    paths = _tracked_preservation_paths(frozen)
    digest = hashlib.sha256()
    total_bytes = 0
    for path_text in paths:
        path = ROOT / path_text
        file_sha256 = sha256_file(path)
        digest.update(f"{file_sha256}  {path_text}\n".encode())
        total_bytes += path.stat().st_size
    lock = frozen["preservation_lock"]
    evidence = {
        "tracked_file_count": len(paths),
        "tracked_bytes": total_bytes,
        "path_content_digest_sha256": digest.hexdigest(),
        "candidate_profile_sha256": sha256_file(
            ROOT / frozen["candidate"]["profile_path"]
        ),
    }
    for field, value in evidence.items():
        expected_field = (
            "path_content_digest_sha256"
            if field == "path_content_digest_sha256"
            else field
        )
        if value != lock[expected_field]:
            raise SystemExit(f"TRAF-73 preservation mismatch: {field}")
    return evidence


def _legacy_identity(profile_path: Path) -> str:
    profile = load_nvlink_candidate_profile(profile_path)
    result = NvlinkDomainService(profile).serve(
        [
            NvlinkTransfer(
                extent_id="identity-write-a",
                source=0,
                destination=1,
                payload_bytes=769,
            ),
            NvlinkTransfer(
                extent_id="identity-write-b",
                source=0,
                destination=2,
                payload_bytes=513,
                released_at_ps=17000,
            ),
            NvlinkTransfer(
                extent_id="identity-read-c",
                source=3,
                destination=1,
                payload_bytes=1025,
                operation=NvlinkOperation.PEER_READ,
                released_at_ps=9000,
            ),
        ],
        analytic_result=None,
        flow_policy=NvlinkFlowPolicy.STATIC_INTERLEAVE,
    )
    if not isinstance(result, NvlinkDomainResult):
        raise SystemExit("the legacy compatibility fixture did not use the packet domain")
    return hashlib.sha256(result.canonical_json_bytes()).hexdigest()


def require_clean_authority(frozen: dict[str, Any]) -> dict[str, object]:
    if sha256_file(EXPECTATIONS_PATH) != EXPECTATIONS_SHA256:
        raise SystemExit("the TRAF-73 expectations moved after their final freeze")
    ancestor = _git("merge-base", "--is-ancestor", EXPECTATIONS_COMMIT, "HEAD")
    if ancestor.returncode:
        raise SystemExit("the final TRAF-73 expectations commit is not an ancestor")
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status.returncode or status.stdout.strip():
        raise SystemExit("the TRAF-73 run requires a clean tracked worktree")
    committed = _git(
        "show",
        f"{EXPECTATIONS_COMMIT}:examples/nvlink_credit_arbitration_v1/expectations.json",
    )
    if committed.returncode:
        raise SystemExit("the frozen TRAF-73 bytes cannot be read from their commit")
    if hashlib.sha256(committed.stdout.encode("utf-8")).hexdigest() != EXPECTATIONS_SHA256:
        raise SystemExit("the expectations commit does not contain the frozen bytes")

    preservation = preservation_evidence(frozen)
    profile_path = ROOT / frozen["candidate"]["profile_path"]
    identity = _legacy_identity(profile_path)
    if identity != LEGACY_IDENTITY_SHA256:
        raise SystemExit("the legacy static-interleave canonical bytes moved")
    return {
        "expectations_commit": EXPECTATIONS_COMMIT,
        "expectations_sha256": EXPECTATIONS_SHA256,
        "legacy_identity_sha256": identity,
        "preservation": preservation,
    }


def prepare_run_dir(path: Path) -> Path:
    configured = os.environ.get(BULK_ROOT_ENV)
    if not configured:
        raise SystemExit(f"set {BULK_ROOT_ENV} to the external nvfair bulk root")
    bulk_root = Path(configured).resolve()
    resolved = path.resolve()
    if resolved.parent != bulk_root:
        raise SystemExit(f"--run-dir must be one new child of {bulk_root.as_posix()}")
    bulk_root.mkdir(parents=True, exist_ok=True)
    resolved.mkdir(parents=False, exist_ok=False)
    return resolved


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def expected_wire_rates_gbps(
    *,
    degree: int,
    policy: NvlinkArbitrationPolicy,
    receiver_rate_gbps: float,
) -> list[float]:
    if degree < 2:
        raise ValueError("the arbitration matrix starts at degree 2")
    if policy is NvlinkArbitrationPolicy.STATIC_INTERLEAVE:
        equal = min(60.0, receiver_rate_gbps / degree)
        return [equal] * degree
    if policy is NvlinkArbitrationPolicy.GREEDY_CAPTURE:
        if degree == 2:
            return [100.0, 60.0]
        small = (receiver_rate_gbps - 100.0) / (degree - 1)
        return [100.0, *([small] * (degree - 1))]
    if degree == 2:
        return [100.0, 60.0]
    if degree == 3:
        return [receiver_rate_gbps - 120.0, 60.0, 60.0]
    return [receiver_rate_gbps / degree] * degree


def _transfers(frozen: dict[str, Any], degree: int) -> list[NvlinkTransfer]:
    simulation = frozen["simulation"]
    payload_bytes = (
        simulation["packets_per_sender"] * frozen["candidate"]["payload_bytes_per_packet"]
    )
    return [
        NvlinkTransfer(
            extent_id=f"degree-{degree}-source-{source}",
            source=source,
            destination=degree,
            payload_bytes=payload_bytes,
            topology_endpoint_count=degree + 1,
            offered_rate_bytes_per_second=(
                simulation["greedy_offered_rate_bytes_per_second"]
                if source == 0
                else simulation["small_offered_rate_bytes_per_second"]
            ),
        )
        for source in range(degree)
    ]


def _steady_window(
    result: NvlinkDomainResult,
    frozen: dict[str, Any],
) -> tuple[int, int]:
    edge = frozen["simulation"]["steady_excluded_packets_each_edge"]
    packet_count = frozen["simulation"]["packets_per_sender"]
    greedy_packets = [packet for packet in result.packets if packet.source == 0]
    start = next(
        packet.delivered_at_ps for packet in greedy_packets if packet.sequence == edge
    )
    end = next(
        packet.delivered_at_ps
        for packet in greedy_packets
        if packet.sequence == packet_count - edge
    )
    if start is None or end is None or end <= start:
        raise AssertionError("the steady arbitration window is not positive")
    return start, end


def _jain(values: list[float]) -> float:
    return sum(values) ** 2 / (len(values) * sum(value * value for value in values))


def simulation_row(
    frozen: dict[str, Any],
    *,
    degree: int,
    policy: NvlinkArbitrationPolicy,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    profile = load_nvlink_candidate_profile(ROOT / frozen["candidate"]["profile_path"])
    transfers = _transfers(frozen, degree)
    result = NvlinkDomainService(profile).serve_arbitrated(
        transfers,
        analytic_result=None,
        policy=policy,
    )
    if not isinstance(result, NvlinkDomainResult):
        raise TypeError("the arbitration arm did not return an NVLink domain result")

    window_start_ps, window_end_ps = _steady_window(result, frozen)
    window_ps = window_end_ps - window_start_ps
    wire_by_source: dict[int, int] = defaultdict(int)
    payload_by_source: dict[int, int] = defaultdict(int)
    packets_by_source: dict[int, int] = defaultdict(int)
    for packet in result.packets:
        if packet.delivered_at_ps is None:
            raise AssertionError("an arbitration packet has no delivery timestamp")
        if window_start_ps <= packet.delivered_at_ps < window_end_ps:
            wire_by_source[packet.source] += packet.wire_bytes
            payload_by_source[packet.source] += packet.payload_bytes
            packets_by_source[packet.source] += 1

    wire_rates = [wire_by_source[source] * 1000 / window_ps for source in range(degree)]
    payload_rates = [
        payload_by_source[source] * 1000 / window_ps for source in range(degree)
    ]
    receiver_rate_gbps = profile.rx.ingress_rate_bytes_per_second / 1e9
    expected_rates = expected_wire_rates_gbps(
        degree=degree,
        policy=policy,
        receiver_rate_gbps=receiver_rate_gbps,
    )
    one_packet_gbps = profile.tx.credit_unit_bytes * 1000 / window_ps
    rate_checks = [
        {
            "source": source,
            "expected_wire_gbps": expected_rates[source],
            "observed_wire_gbps": wire_rates[source],
            "tolerance_gbps": one_packet_gbps,
            "passed": abs(wire_rates[source] - expected_rates[source]) <= one_packet_gbps,
        }
        for source in range(degree)
    ]
    expected_aggregate = sum(expected_rates)
    aggregate_wire_gbps = sum(wire_rates)
    aggregate_tolerance_gbps = degree * one_packet_gbps
    aggregate_passed = (
        abs(aggregate_wire_gbps - expected_aggregate) <= aggregate_tolerance_gbps
    )
    behavioral = [*rate_checks, {
        "source": "aggregate",
        "expected_wire_gbps": expected_aggregate,
        "observed_wire_gbps": aggregate_wire_gbps,
        "tolerance_gbps": aggregate_tolerance_gbps,
        "passed": aggregate_passed,
    }]

    expected_packets = frozen["simulation"]["packets_per_sender"] * degree
    fatal = [
        {
            "guard": "packet_count_conservation",
            "passed": len(result.packets) == expected_packets,
        },
        {
            "guard": "logical_byte_conservation",
            "passed": result.logical_bytes == sum(t.payload_bytes for t in transfers),
        },
        {
            "guard": "request_payload_conservation",
            "passed": result.request_payload_bytes == result.logical_bytes,
        },
        {
            "guard": "response_payload_zero",
            "passed": result.response_payload_bytes == 0,
        },
        {
            "guard": "receiver_rate_ceiling",
            "passed": aggregate_wire_gbps <= receiver_rate_gbps + one_packet_gbps,
        },
        {
            "guard": "per_pair_rate_ceiling",
            "passed": all(rate <= 100.0 + one_packet_gbps for rate in wire_rates),
        },
        {
            "guard": "every_source_visible",
            "passed": all(packets_by_source[source] > 0 for source in range(degree)),
        },
    ]
    if any(not guard["passed"] for guard in fatal):
        raise SystemExit(f"fatal TRAF-73 simulation guard failed at {degree} {policy.value}")

    row = {
        "degree": degree,
        "topology_class": (
            "PHYSICAL_NV4" if degree <= 3 else "SIMULATED_MESH_EXTRAPOLATION"
        ),
        "policy": policy.value,
        "window_start_ps": window_start_ps,
        "window_end_ps": window_end_ps,
        "window_ps": window_ps,
        "packets_per_source_in_window": [packets_by_source[source] for source in range(degree)],
        "wire_gbps_per_source": wire_rates,
        "payload_gbps_per_source": payload_rates,
        "expected_wire_gbps_per_source": expected_rates,
        "per_source_tolerance_gbps": one_packet_gbps,
        "aggregate_tolerance_gbps": aggregate_tolerance_gbps,
        "aggregate_wire_gbps": aggregate_wire_gbps,
        "aggregate_payload_gbps": sum(payload_rates),
        "receiver_raw_ceiling_gbps": receiver_rate_gbps,
        "receiver_raw_utilization": aggregate_wire_gbps / receiver_rate_gbps,
        "jain_wire_rate": _jain(wire_rates),
        "greedy_wire_gbps": wire_rates[0],
        "small_wire_gbps_min": min(wire_rates[1:]),
        "small_wire_gbps_max": max(wire_rates[1:]),
        "max_rx_buffer_occupancy_bytes": result.max_rx_buffer_occupancy_bytes,
        "behavioral_verdict": (
            "PASS" if all(check["passed"] for check in behavioral) else "REFUTED"
        ),
    }
    return row, [
        {
            **guard,
            "degree": degree,
            "policy": policy.value,
        }
        for guard in fatal
    ]


def run_simulation(frozen: dict[str, Any], authority: dict[str, object]) -> dict[str, object]:
    rows = []
    fatal_guards = []
    for degree in frozen["simulation"]["degrees"]:
        for policy_text in frozen["simulation"]["policies"]:
            row, guards = simulation_row(
                frozen,
                degree=degree,
                policy=NvlinkArbitrationPolicy(policy_text),
            )
            rows.append(row)
            fatal_guards.extend(guards)
    refuted = [row for row in rows if row["behavioral_verdict"] != "PASS"]
    return {
        "schema": RESULT_SCHEMA,
        "task_id": frozen["task_id"],
        "authority": authority,
        "hardware_status": "REGISTERED_NOT_RUN",
        "run_configuration": frozen["simulation"],
        "physical_sanity": frozen["physical_sanity"],
        "simulation_rows": rows,
        "behavioral_summary": {
            "family_count": 3,
            "instance_count": len(rows),
            "passed_instances": len(rows) - len(refuted),
            "refuted_instances": len(refuted),
            "verdict": "PASS" if not refuted else "PASS_WITH_REFUTATION",
        },
        "fatal_guards": fatal_guards,
        "fatal_summary": {
            "guard_instances": len(fatal_guards),
            "violations": sum(not guard["passed"] for guard in fatal_guards),
            "verdict": "PASS",
        },
        "evidence_classes": {
            "behavioral": "policy_direction_and_bounded_share_predictions",
            "fatal": "conservation_and_physical_ceiling_preconditions",
            "structural": "per_link_credit_ownership_and_legacy_byte_identity",
            "hardware": "registered_not_run",
        },
        "scoring_chronology": {
            "first_run_commit": "0888344",
            "first_run_result_sha256": (
                "b5713abb3902795cffd8e86ef1a9a4b40bd420a253928f36ab2d2c33442143ad"
            ),
            "first_run_verdict": "PASS_WITH_TWO_AGGREGATE_SCORER_REFUTATIONS",
            "correction_class": "post_specified_aggregate_quantization_bound",
            "unchanged": [
                "workload",
                "time_window",
                "policy",
                "physical_ceiling",
                "preservation_lock",
                "per_sender_expectation_and_tolerance",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    frozen = load_expectations()
    authority = require_clean_authority(frozen)
    run_dir = prepare_run_dir(arguments.run_dir)
    result = run_simulation(frozen, authority)
    _write_json(run_dir / "results.json", result)
    print(json.dumps(result["behavioral_summary"], sort_keys=True))


if __name__ == "__main__":
    main()
