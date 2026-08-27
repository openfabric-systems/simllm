#!/usr/bin/env python3
"""Build the expectations-only TRAF-69 NVLink flow-dynamics freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROFILE_PATH = ROOT / "examples" / "a100_nvlink_packet_v1" / "candidate-profile.json"
INHERITED_PATH = ROOT / "examples" / "deployment_frontier_v1" / "expectations.json"
OUTPUT_PATH = HERE / "expectations.json"
PS_PER_SECOND = 1_000_000_000_000

SEEDS = [1103, 1907, 2801, 3691, 4513, 5381, 6271, 7159, 8053]
FLOW_SIZES = [256, 1024, 4096, 16384, 65536, 262144, 524288]
SAMPLES_PER_SEED_PER_SENDER = 12

ADDITIONAL_PRESERVATION_PATHS = [
    "examples/deployment_frontier_v1/expectations.json",
    "examples/deployment_frontier_v1/expectations.md",
    "examples/deployment_frontier_v1/result.json",
    "examples/deployment_frontier_v1/results.csv",
    "examples/deployment_frontier_v1/RESULTS.md",
    "examples/deployment_frontier_v1/figures/deployment-frontier.pdf",
    "examples/deployment_frontier_v1/figures/deployment-frontier.png",
    "examples/deployment_frontier_v1/figures/two-network-bottleneck.pdf",
    "examples/deployment_frontier_v1/figures/two-network-bottleneck.png",
    "examples/a100_nvlink_packet_v1/candidate-profile-pre-traf70.json",
    "examples/a100_nvlink_packet_v1/candidate-profile.json",
    "examples/a100_nvlink_packet_v2/expectations.json",
    "examples/a100_nvlink_packet_v2/expectations.md",
    "examples/a100_nvlink_packet_v2/hardware-score.json",
    "examples/a100_nvlink_packet_v2/local-validation.json",
    "examples/a100_nvlink_packet_v2/RESUME.md",
    "examples/a100_nvlink_packet_v2/RESULTS.md",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _serialize_ps(byte_count: int, rate_bytes_per_second: int) -> int:
    return _ceil_div(byte_count * PS_PER_SECOND, rate_bytes_per_second)


def _rate_gbps(rate_bytes_per_second: int, payload_bytes: int, wire_bytes: int) -> float:
    return float(Fraction(rate_bytes_per_second * payload_bytes, wire_bytes * 1_000_000_000))


def _single_flow_fct_ps(
    size_bytes: int,
    *,
    payload_bytes: int,
    header_bytes: int,
    links_per_peer: int,
    endpoint_packet_ps: int,
    link_packet_ps: int,
    per_link_rate: int,
    rx_rate: int,
) -> int:
    packet_count = _ceil_div(size_bytes, payload_bytes)
    last_payload = size_bytes - (packet_count - 1) * payload_bytes
    packet_index = packet_count - 1
    last_wire_bytes = last_payload + header_bytes
    start_ps = (packet_index // links_per_peer) * link_packet_ps + (
        packet_index % links_per_peer
    ) * endpoint_packet_ps
    return start_ps + _serialize_ps(last_wire_bytes, per_link_rate) + _serialize_ps(
        last_wire_bytes, rx_rate
    )


def _cdf_bands(
    *,
    payload_bytes: int,
    header_bytes: int,
    links_per_peer: int,
    endpoint_packet_ps: int,
    link_packet_ps: int,
    per_link_rate: int,
    rx_rate: int,
) -> list[dict[str, object]]:
    rows = []
    for degree in (1, 2, 3):
        for size_bytes in FLOW_SIZES:
            packet_count = _ceil_div(size_bytes, payload_bytes)
            wire_bytes_per_flow = size_bytes + packet_count * header_bytes
            single_ps = _single_flow_fct_ps(
                size_bytes,
                payload_bytes=payload_bytes,
                header_bytes=header_bytes,
                links_per_peer=links_per_peer,
                endpoint_packet_ps=endpoint_packet_ps,
                link_packet_ps=link_packet_ps,
                per_link_rate=per_link_rate,
                rx_rate=rx_rate,
            )
            if degree < 3:
                wave_service_ps = single_ps
            else:
                wave_service_ps = link_packet_ps + _serialize_ps(
                    degree * wire_bytes_per_flow, rx_rate
                )
            rows.append(
                {
                    "degree": degree,
                    "size_bytes": size_bytes,
                    "single_flow_physical_floor_ps": single_ps,
                    "wave_service_ps": wave_service_ps,
                    "release_interval_ps": 3 * wave_service_ps // 4,
                    "release_jitter_ps": [-link_packet_ps, link_packet_ps],
                    "p50_band_ps": [single_ps, 4 * wave_service_ps + 2 * link_packet_ps],
                    "p95_band_ps": [
                        single_ps,
                        SAMPLES_PER_SEED_PER_SENDER * wave_service_ps
                        + 2 * link_packet_ps,
                    ],
                }
            )
    return rows


def build() -> dict[str, object]:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    tx = profile["tx"]
    rx = profile["rx"]
    payload_bytes = tx["max_payload_bytes"]
    header_bytes = tx["header_bytes"]
    wire_bytes = payload_bytes + header_bytes
    endpoint_packet_ps = _serialize_ps(wire_bytes, tx["endpoint_egress_rate_bytes_per_second"])
    link_packet_ps = _serialize_ps(wire_bytes, tx["per_link_rate_bytes_per_second"])
    rx_packet_ps = _serialize_ps(wire_bytes, rx["ingress_rate_bytes_per_second"])
    pair_raw_rate = min(
        tx["endpoint_egress_rate_bytes_per_second"],
        tx["links_per_peer"] * tx["per_link_rate_bytes_per_second"],
    )
    pair_payload_gbps = _rate_gbps(pair_raw_rate, payload_bytes, wire_bytes)
    raw_bin_ps = 64 * link_packet_ps
    raw_bin_quantum_gbps = payload_bytes * 1000 / raw_bin_ps

    incast = []
    for degree in (1, 2, 3):
        raw_ceiling = min(degree * pair_raw_rate, rx["ingress_rate_bytes_per_second"])
        incast.append(
            {
                "degree": degree,
                "raw_ceiling_bytes_per_second": raw_ceiling,
                "payload_ceiling_gbps": _rate_gbps(raw_ceiling, payload_bytes, wire_bytes),
                "expected_binding_module": "rx" if degree == 3 else "tx_pair_links",
                "per_flow_payload_gbps": _rate_gbps(
                    raw_ceiling, payload_bytes, wire_bytes
                )
                / degree,
                "steady_band_quantum_gbps": raw_bin_quantum_gbps,
            }
        )

    evidence = profile["parameter_evidence"]
    candidate_parameters = sorted(
        f"{module}.{parameter}"
        for module, values in evidence.items()
        for parameter, record in values.items()
        if record["status"] == "INCONCLUSIVE"
    )
    measured_parameters = sorted(
        f"{module}.{parameter}"
        for module, values in evidence.items()
        for parameter, record in values.items()
        if record["status"] == "IDENTIFIED"
    )
    structural_parameters = sorted(
        f"{module}.{parameter}"
        for module, values in evidence.items()
        for parameter, record in values.items()
        if record["status"] == "STRUCTURAL"
    )

    inherited = json.loads(INHERITED_PATH.read_text(encoding="utf-8"))["preservation_lock"]
    additional = [
        {"path": relative, "sha256": _sha256(ROOT / relative)}
        for relative in ADDITIONAL_PRESERVATION_PATHS
    ]
    transition_open_ps = endpoint_packet_ps + link_packet_ps + rx_packet_ps
    divergence_target_ps = 2 * link_packet_ps - 3 * endpoint_packet_ps
    return {
        "schema": "simllm-nvlink-flow-dynamics-expectations-v1",
        "study": {
            "task": "TRAF-69",
            "date": "2026-08-27",
            "source": "maintainer directive of 2026-08-27",
            "status": "expectations_only",
            "chronology": (
                "Committed before release-aware flow scheduling, the study runner, "
                "any TRAF-69 simulated run, or any result-dependent edit."
            ),
            "preimplementation_amendment": (
                "The first expectations commit used packet admission plus one link "
                "cadence for divergence. Preimplementation phase review corrected the "
                "64 KiB departing flow's phase-3 identity to two link cadences minus "
                "three packet admissions before any target edit or simulated run."
            ),
        },
        "source_profile": {
            "path": PROFILE_PATH.relative_to(ROOT).as_posix(),
            "sha256": _sha256(PROFILE_PATH),
            "profile_id": profile["profile_id"],
            "status": profile["status"],
            "score_sha256": profile["traf70_score_publication"]["score_sha256"],
            "score_status": profile["traf70_score_publication"]["score_status"],
            "flow_dynamics_gate": "OPEN",
        },
        "parameter_ledger": {
            "measured": measured_parameters,
            "declared_candidates": candidate_parameters,
            "structural": structural_parameters,
            "candidate_count": len(candidate_parameters),
            "unchanged_parameter_count": profile["traf70_score_publication"][
                "unchanged_parameter_count"
            ],
            "figure_disclosure": (
                "Every figure states that packet size, header, links, link rate, bond, "
                "credits, RX buffer, return latency and queue scope remain candidates; "
                "the TX and RX plateaus are measured and the switch is structural."
            ),
        },
        "packet_arithmetic": {
            "payload_bytes": payload_bytes,
            "header_bytes": header_bytes,
            "wire_bytes": wire_bytes,
            "links_per_peer": tx["links_per_peer"],
            "endpoint_packet_ps": endpoint_packet_ps,
            "link_packet_ps": link_packet_ps,
            "rx_packet_ps": rx_packet_ps,
            "credit_return_ps": rx["credit_return_latency_ps"],
            "credits_per_destination": tx["credits_per_destination"],
            "pair_raw_rate_bytes_per_second": pair_raw_rate,
            "pair_payload_rate_gbps": pair_payload_gbps,
        },
        "flow_schedule": {
            "topology": "NV4 four-GPU direct mesh, source 0 to receiver 1",
            "flow_ids": ["flow-a", "flow-b", "flow-c"],
            "release_ps": [0, 16 * raw_bin_ps, 32 * raw_bin_ps],
            "target_bytes": [4194304, 2097152, 1048576],
            "reverse_target_rule": "first joiner has the largest target and completes last",
            "raw_bin_ps": raw_bin_ps,
            "raw_bin_basis": "64 candidate maximum-wire-packet per-link serializations",
            "smoothing": "none",
            "steady_bands": [
                {
                    "active_flows": count,
                    "per_flow_center_gbps": pair_payload_gbps / count,
                    "half_width_gbps": raw_bin_quantum_gbps,
                }
                for count in (1, 2, 3)
            ],
        },
        "convergence_1_to_2": {
            "incumbent_target_bytes": 1048576,
            "joiner_target_bytes": 262144,
            "join_after_incumbent_packet": 256,
            "join_ps": 64 * link_packet_ps,
            "boundary_rule": "the incumbent owns the packet-256 grant at the join timestamp",
            "identity": "credit_wait + packet_admission + link_serialization + rx_serialization",
            "terms_ps": {
                "credit_wait": 0,
                "packet_admission": endpoint_packet_ps,
                "link_serialization": link_packet_ps,
                "rx_serialization": rx_packet_ps,
            },
            "expected_open_ps": transition_open_ps,
            "exact_tolerance_ps": 0,
            "credit_inactive_reason": (
                "256 packet credits recycle much earlier than the packet-256 join; "
                "the 200,000 ps declared return constant contributes exactly zero."
            ),
            "incumbent_pre_gbps": pair_payload_gbps,
            "incumbent_post_gbps": pair_payload_gbps / 2,
            "rate_band_half_width_gbps": raw_bin_quantum_gbps,
            "raw_bin_ps": link_packet_ps,
            "smoothing": "none",
        },
        "divergence_2_to_1": {
            "remaining_target_bytes": 1048576,
            "departing_target_bytes": 65536,
            "identity": "two_link_cadences - three_packet_admissions",
            "terms_ps": {
                "credit_wait": 0,
                "two_link_cadences": 2 * link_packet_ps,
                "three_packet_admissions": -3 * endpoint_packet_ps,
                "rx_serialization_difference": 0,
            },
            "target_definition": (
                "the receiver observes the fifth remaining-flow delivery after exit, "
                "which closes the first complete four-link solo cadence"
            ),
            "expected_time_to_target_ps": divergence_target_ps,
            "exact_tolerance_ps": 0,
            "remaining_pre_gbps": pair_payload_gbps / 2,
            "remaining_post_gbps": pair_payload_gbps,
            "rate_band_half_width_gbps": raw_bin_quantum_gbps,
            "raw_bin_ps": link_packet_ps,
            "smoothing": "none",
        },
        "fct_cdf": {
            "flow_sizes_bytes": FLOW_SIZES,
            "largest_supported_basis": (
                "TRAF-70 scored profile published_envelope_validation."
                "composed_validation_extent_bytes_per_destination"
            ),
            "seeds": SEEDS,
            "seed_count": len(SEEDS),
            "samples_per_seed_per_sender": SAMPLES_PER_SEED_PER_SENDER,
            "band": "pointwise minimum to maximum empirical CDF across seeds",
            "mean": "pointwise arithmetic mean empirical CDF across seeds",
            "grid": "sorted union of observed FCT values within each degree and size rung",
            "bands": _cdf_bands(
                payload_bytes=payload_bytes,
                header_bytes=header_bytes,
                links_per_peer=tx["links_per_peer"],
                endpoint_packet_ps=endpoint_packet_ps,
                link_packet_ps=link_packet_ps,
                per_link_rate=tx["per_link_rate_bytes_per_second"],
                rx_rate=rx["ingress_rate_bytes_per_second"],
            ),
        },
        "incast": {
            "degrees": incast,
            "receiver": 3,
            "sources_by_degree": {"1": [0], "2": [0, 1], "3": [0, 1, 2]},
            "schedule_target_bytes_per_flow": FLOW_SIZES[-1],
            "raw_bin_ps": raw_bin_ps,
            "smoothing": "none",
            "fanout_check": {
                "shape": "one sender to three receivers, not incast",
                "published_payload_gbps": 281.65,
                "predicted_formula_payload_gbps": _rate_gbps(
                    min(
                        tx["endpoint_egress_rate_bytes_per_second"],
                        3 * pair_raw_rate,
                    ),
                    payload_bytes,
                    wire_bytes,
                ),
                "relative_error_limit": 0.10,
                "expected_verdict": "REFUTED",
            },
        },
        "plot_contract": {
            "grammar": "rnic-cn join and exit rate presentation",
            "formats": ["pdf", "png"],
            "path_rendering": "POSIX",
            "flow_dynamics_panels": ["overall_schedule", "convergence_1_to_2", "divergence_2_to_1"],
            "fct_panels": "one panel per size rung with mean CDF and shaded min-max band",
            "incast_panels": "one schedule plus one FCT CDF panel per degree",
            "raw_rate_style": "steps from fixed bins with no smoothing",
            "candidate_disclosure_required_on_every_figure": True,
        },
        "preservation_lock": {
            "class": "all-43-prior-flagship-locks-plus-frontier-and-traf70-publication",
            "inherited": {
                "path": INHERITED_PATH.relative_to(ROOT).as_posix(),
                "sha256": _sha256(INHERITED_PATH),
                "json_pointer": "/preservation_lock",
                "expected_artifacts": inherited["expected_total_artifacts"],
            },
            "additional_artifacts": additional,
            "expected_total_artifacts": inherited["expected_total_artifacts"] + len(additional),
        },
        "fatal_guards": [
            "expectations digest or expectations commit is not the run authority",
            "tracked worktree is dirty before the gated run",
            "the scored profile, score status, score digest or flow-dynamics gate changes",
            "the parameter evidence catalog is incomplete or changes evidence class",
            "any inherited or additional preservation artifact changes",
            "a byte, packet, extent, request direction or response direction is lost or duplicated",
            "a per-extent delivery sequence is not strictly increasing",
            "the pass-through switch changes any packet object or timestamp",
            "the default flow-inactive domain result differs from its pre-change canonical bytes",
            "a simulated rate exceeds its physical ceiling",
            "a CDF is not monotone, does not end at one or leaves its frozen quantile band",
            "a transition identity differs by any picosecond",
            "a prior runner, record or figure is executed or rewritten",
        ],
        "verdict_rules": {
            "fatal": "Any fatal-guard failure voids the study and TRAF-69 stays open.",
            "transition": "Both exact timing identities must match with zero-picosecond tolerance.",
            "steady_rate": "Every scored steady window must land in its pre-run quantized band.",
            "cdf": "Each degree and size rung must pass monotonicity, terminal-one, p50 and p95 bands.",
            "honest_miss": "Any nonfatal miss is published as REFUTED with its frozen band unchanged.",
            "closure": (
                "Close TRAF-69 only after all figures render in both formats, every panel and "
                "rung is scored, preservation holds, and all fatal guards pass."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    arguments = parser.parse_args()
    arguments.output.write_text(
        json.dumps(build(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    try:
        rendered = arguments.output.relative_to(ROOT).as_posix()
    except ValueError:
        rendered = arguments.output.as_posix()
    print(rendered)


if __name__ == "__main__":
    main()
