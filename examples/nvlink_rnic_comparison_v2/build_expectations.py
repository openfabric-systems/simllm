#!/usr/bin/env python3
"""Build the expectations-only TRAF-72 transport comparison freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT_PATH = HERE / "expectations.json"
FLOW_STUDY = ROOT / "examples" / "nvlink_flow_dynamics_v1"
LEGACY_STUDY = ROOT / "examples" / "nvlink_rnic_comparison_v1"
PROFILE_PATH = ROOT / "examples" / "a100_nvlink_packet_v1" / "candidate-profile.json"
HTSIM_COMMIT = "1dcbfec36a33753bf978cf6323bade1a6645fe4f"

FLOW_SIZES = (256, 1024, 4096, 16384, 65536, 262144, 524288)
DEGREES = (1, 2, 3, 4, 8, 16)
PHYSICAL_DEGREES = (1, 2, 3)
MESH_DEGREES = (4, 8, 16)
LEGACY_PATHS = (
    "examples/nvlink_rnic_comparison_v1/CMakeLists.txt",
    "examples/nvlink_rnic_comparison_v1/RESULTS.md",
    "examples/nvlink_rnic_comparison_v1/build_expectations.py",
    "examples/nvlink_rnic_comparison_v1/dispersion.csv",
    "examples/nvlink_rnic_comparison_v1/expectations.json",
    "examples/nvlink_rnic_comparison_v1/expectations.md",
    "examples/nvlink_rnic_comparison_v1/figures/nvlink-rnic-dispersion.pdf",
    "examples/nvlink_rnic_comparison_v1/figures/nvlink-rnic-dispersion.png",
    "examples/nvlink_rnic_comparison_v1/figures/nvlink-rnic-fct-cdf.pdf",
    "examples/nvlink_rnic_comparison_v1/figures/nvlink-rnic-fct-cdf.png",
    "examples/nvlink_rnic_comparison_v1/htsim_logged_minimal.cpp",
    "examples/nvlink_rnic_comparison_v1/plot_study.py",
    "examples/nvlink_rnic_comparison_v1/publish_study.py",
    "examples/nvlink_rnic_comparison_v1/results.json",
    "examples/nvlink_rnic_comparison_v1/rnic_nn_schedule.cpp",
    "examples/nvlink_rnic_comparison_v1/run_study.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _serialize_ps(wire_bytes: int, bytes_per_second: int) -> int:
    return _ceil_div(wire_bytes * 1_000_000_000_000, bytes_per_second)


def _sample_releases(
    *,
    seed: int,
    degree: int,
    size_bytes: int,
    release_interval_ps: int,
    jitter_low_ps: int,
    jitter_high_ps: int,
    samples_per_sender: int,
) -> list[tuple[int, int, int]]:
    generator = random.Random(seed * 1_000_003 + degree * 10_007 + size_bytes)
    releases = []
    wave_release = 0
    for wave in range(samples_per_sender):
        if wave:
            wave_release += release_interval_ps + generator.randint(
                jitter_low_ps, jitter_high_ps
            )
        for source in range(degree):
            source_skew = generator.randint(0, jitter_high_ps)
            releases.append((wave, source, wave_release + source_skew))
    return releases


def _single_flow_floor(
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


def _workload(profile: dict[str, Any], source: dict[str, Any]) -> dict[str, object]:
    source_cdf = source["fct_cdf"]
    tx = profile["tx"]
    rx = profile["rx"]
    payload_bytes = int(tx["max_payload_bytes"])
    header_bytes = int(tx["header_bytes"])
    full_wire_bytes = payload_bytes + header_bytes
    endpoint_packet_ps = _serialize_ps(
        full_wire_bytes, int(tx["endpoint_egress_rate_bytes_per_second"])
    )
    link_packet_ps = _serialize_ps(
        full_wire_bytes, int(tx["per_link_rate_bytes_per_second"])
    )
    source_cells = {
        (int(row["degree"]), int(row["size_bytes"])): row
        for row in source_cdf["bands"]
    }
    cells = []
    for degree in DEGREES:
        for size_bytes in FLOW_SIZES:
            packet_count = _ceil_div(size_bytes, payload_bytes)
            wire_bytes_per_flow = size_bytes + packet_count * header_bytes
            single_ps = _single_flow_floor(
                size_bytes,
                payload_bytes=payload_bytes,
                header_bytes=header_bytes,
                links_per_peer=int(tx["links_per_peer"]),
                endpoint_packet_ps=endpoint_packet_ps,
                link_packet_ps=link_packet_ps,
                per_link_rate=int(tx["per_link_rate_bytes_per_second"]),
                rx_rate=int(rx["ingress_rate_bytes_per_second"]),
            )
            wave_service_ps = (
                single_ps
                if degree < 3
                else link_packet_ps
                + _serialize_ps(
                    degree * wire_bytes_per_flow,
                    int(rx["ingress_rate_bytes_per_second"]),
                )
            )
            release_interval_ps = 3 * wave_service_ps // 4
            jitter = [-link_packet_ps, link_packet_ps]
            releases_by_seed = {
                str(seed): _sample_releases(
                    seed=seed,
                    degree=degree,
                    size_bytes=size_bytes,
                    release_interval_ps=release_interval_ps,
                    jitter_low_ps=jitter[0],
                    jitter_high_ps=jitter[1],
                    samples_per_sender=int(source_cdf["samples_per_seed_per_sender"]),
                )
                for seed in source_cdf["seeds"]
            }
            inherited = source_cells.get((degree, size_bytes))
            if inherited is not None:
                for field, expected in (
                    ("wave_service_ps", wave_service_ps),
                    ("release_interval_ps", release_interval_ps),
                    ("release_jitter_ps", jitter),
                ):
                    if inherited[field] != expected:
                        raise RuntimeError(
                            f"TRAF-69 {field} moved for degree {degree}, size {size_bytes}"
                        )
            cells.append(
                {
                    "degree": degree,
                    "destination": max(3, degree),
                    "size_bytes": size_bytes,
                    "single_flow_physical_floor_ps": single_ps,
                    "wave_service_ps": wave_service_ps,
                    "release_interval_ps": release_interval_ps,
                    "release_jitter_ps": jitter,
                    "release_schedule_sha256": hashlib.sha256(
                        _canonical(releases_by_seed)
                    ).hexdigest(),
                    "source_relation": (
                        "byte-identical TRAF-69 tuples"
                        if degree in PHYSICAL_DEGREES
                        else "same frozen generator extrapolated to the declared mesh"
                    ),
                }
            )
    return {
        "flow_sizes_bytes": list(FLOW_SIZES),
        "degrees": list(DEGREES),
        "physical_degrees": list(PHYSICAL_DEGREES),
        "simulated_mesh_degrees": list(MESH_DEGREES),
        "seeds": source_cdf["seeds"],
        "seed_count": len(source_cdf["seeds"]),
        "samples_per_seed_per_sender": source_cdf["samples_per_seed_per_sender"],
        "release_generator": (
            "Random(seed*1000003 + degree*10007 + size_bytes); each later wave "
            "adds release_interval_ps plus randint(jitter_low_ps,jitter_high_ps); "
            "each source adds randint(0,jitter_high_ps)"
        ),
        "offered_load": (
            "release interval is three quarters of the frozen wave-service bound"
        ),
        "cells": cells,
    }


def _legacy_p50(result: dict[str, Any], transport: str) -> float:
    rows = [
        row
        for row in result["cell_summaries"]
        if row["transport"] == transport
        and row["degree"] == 3
        and row["size_bytes"] == 524288
    ]
    if len(rows) != 1:
        raise RuntimeError(f"legacy result has {len(rows)} matching rows")
    return float(rows[0]["p50_seed_mean_ps"])


def _mapping_audit(legacy: dict[str, Any]) -> dict[str, object]:
    pair = 100_000_000_000
    tx = 160_795_737_454
    rx = 207_101_921_876
    rows = []
    for degree in PHYSICAL_DEGREES:
        pair_class = degree * pair
        tx_aggregate = degree * tx
        granted = min(pair_class, tx_aggregate, rx)
        rows.append(
            {
                "degree": degree,
                "ordered_pair_class_cap_bytes_per_second": pair_class,
                "tx_egress_plateau_aggregate_bytes_per_second": tx_aggregate,
                "rx_ingress_plateau_bytes_per_second": rx,
                "legacy_receiver_capacity_bytes_per_second": granted,
                "legacy_receiver_capacity_bps": granted * 8,
                "binding_capacity": (
                    "RX ingress plateau"
                    if granted == rx
                    else "ordered-pair class cap"
                    if granted == pair_class
                    else "TX egress plateau"
                ),
                "full_membership_per_source_share_bytes_per_second": granted
                / degree,
            }
        )
    legacy_nvlink = _legacy_p50(legacy, "nvlink-credit")
    legacy_rnic = _legacy_p50(legacy, "rnic-nn")
    observed_ratio = legacy_rnic / legacy_nvlink
    legacy_ps_units = Fraction(601, 160)
    pair_class_fifo_units = Fraction(9, 4)
    schedule_ratio = legacy_ps_units / pair_class_fifo_units
    return {
        "verdict": "MAPPING_DEFICIT_IN_FAIR_SHARE_ENTITY_NOT_CAPACITY_VALUE",
        "plain_verdict": (
            "mapping deficit, not a genuine packet-transport tail: the aggregate "
            "capacity was correct, but each overlapping transfer was mapped as an "
            "independent max-min flow instead of one queued ordered-pair class"
        ),
        "legacy_formula": "min(degree*pair_raw, degree*tx_plateau, rx_plateau)",
        "rows": rows,
        "degree_3_aggregate_below_nvlink": False,
        "degree_3_nvlink_effective_ingress_bytes_per_second": rx,
        "degree_3_rnic_receiver_grant_bytes_per_second": rows[-1][
            "legacy_receiver_capacity_bytes_per_second"
        ],
        "degree_3_aggregate_capacity_ratio": 1.0,
        "legacy_degree_3_512k_p50_ps": {
            "nvlink_credit": legacy_nvlink,
            "rnic_nn": legacy_rnic,
            "observed_ratio": observed_ratio,
        },
        "normalized_queue_arithmetic": {
            "wave_service_symbol": "S",
            "release_interval": "3S/4",
            "pair_class_fifo_nearest_rank_p50_service_units": "9/4",
            "per_transfer_max_min_nearest_rank_p50_service_units": "601/160",
            "predicted_ratio": "601/360",
            "predicted_ratio_decimal": float(schedule_ratio),
            "relative_error_from_observed": abs(float(schedule_ratio) / observed_ratio - 1),
        },
        "correction": (
            "Keep one active htsim fair-share entity per ordered source-destination "
            "class. Later released transfers wait in that class until its active "
            "transfer drains. Allocate the class against a 100 GB/s source cap and "
            "the shared 207.101921876 GB/s destination cap."
        ),
    }


def _mapping() -> dict[str, object]:
    pair = 100_000_000_000
    tx = 160_795_737_454
    rx = 207_101_921_876
    return {
        "zero_fitted_constants": True,
        "source_ordered_pair_capacity_bytes_per_second": min(pair, tx),
        "source_tx_egress_plateau_bytes_per_second": tx,
        "destination_rx_ingress_capacity_bytes_per_second": rx,
        "active_entity": "one queued ordered source-destination pair class",
        "max_active_entities_per_pair": 1,
        "later_transfer_rule": "wait at the class mapping boundary until the active extent drains",
        "packet_allocator": (
            "htsim RnicMaxMinAllocator grants feed RnicPacketizedSlotCalendar"
        ),
        "fluid_allocator": "htsim RnicFluidManifold with the same asymmetric maps",
        "degree_rows": [
            {
                "degree": degree,
                "aggregate_capacity_bytes_per_second": min(degree * pair, rx),
                "binding_capacity": (
                    "ordered-pair class cap" if degree < 3 else "RX ingress plateau"
                ),
                "full_membership_per_class_bytes_per_second": min(
                    pair, rx / degree
                ),
            }
            for degree in DEGREES
        ],
    }


def _preservation_lock() -> dict[str, object]:
    artifacts = [
        {
            "path": relative_path,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for relative_path in LEGACY_PATHS
        for path in (ROOT / relative_path,)
    ]
    return {
        "legacy_study": "examples/nvlink_rnic_comparison_v1",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "recursive_object_list_sha256": (
            "3b052bca52fd684080b7c419075d27dbea43b0fe5d7d27b5d3ffff1b83cac7bb"
        ),
        "policy": (
            "TRAF-71 stays byte-identical and its degree-3 interpretation is "
            "superseded only by reference from TRAF-72"
        ),
    }


def build() -> dict[str, object]:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    flow_expectations = json.loads(
        (FLOW_STUDY / "expectations.json").read_text(encoding="utf-8")
    )
    legacy_result = json.loads(
        (LEGACY_STUDY / "results.json").read_text(encoding="utf-8")
    )
    workload = _workload(profile, flow_expectations)
    return {
        "schema": "simllm-nvlink-rnic-comparison-expectations-v2",
        "task_id": "TRAF-72",
        "study": {
            "name": "nvlink_rnic_comparison_v2",
            "status": "expectations_only",
            "freeze_date": "2026-08-28",
            "prohibited_writeback": (
                "No TRAF-72 observation may change this freeze. Refutations publish "
                "with the thresholds and direction signs unchanged."
            ),
        },
        "source_authority": {
            "repository_commit": "6c1ea0b",
            "flow_expectations_path": "examples/nvlink_flow_dynamics_v1/expectations.json",
            "flow_expectations_sha256": _sha256(FLOW_STUDY / "expectations.json"),
            "legacy_expectations_path": "examples/nvlink_rnic_comparison_v1/expectations.json",
            "legacy_expectations_sha256": _sha256(LEGACY_STUDY / "expectations.json"),
            "legacy_result_path": "examples/nvlink_rnic_comparison_v1/results.json",
            "legacy_result_sha256": _sha256(LEGACY_STUDY / "results.json"),
            "profile_path": "examples/a100_nvlink_packet_v1/candidate-profile.json",
            "profile_sha256": _sha256(PROFILE_PATH),
            "legacy_freeze_commit": (
                "6224d90fea2eed788b8e6ba876787fe7f0e52319"
            ),
        },
        "htsim_authority": {
            "commit": HTSIM_COMMIT,
            "packet_transport": "rnic-nn",
            "packet_primitives": [
                "RnicMaxMinAllocator",
                "RnicPacketizedSlotCalendar",
            ],
            "fluid_transport": "rnic-nn-fluid",
            "fluid_primitive": "RnicFluidManifold",
            "fluid_semantics": (
                "continuous bytes, no packetization, perfectly fair max-min service, "
                "zero propagation on this mapping"
            ),
        },
        "physical_constants": {
            "max_payload_bytes": 256,
            "header_bytes": 16,
            "max_wire_bytes": 272,
            "per_link_bytes_per_second": 25_000_000_000,
            "links_per_ordered_pair": 4,
            "pair_raw_bytes_per_second": 100_000_000_000,
            "tx_endpoint_egress_bytes_per_second": 160_795_737_454,
            "rx_ingress_bytes_per_second": 207_101_921_876,
            "credits_per_destination": 256,
            "credit_unit_bytes": 272,
            "credit_return_ps": 200_000,
            "propagation_ps": 0,
        },
        "mapping_audit": _mapping_audit(legacy_result),
        "corrected_mapping": _mapping(),
        "workload": workload,
        "topology": {
            "physical": "NV4 four-GPU direct mesh only for degrees 1, 2, and 3",
            "extrapolated": (
                "degrees 4, 8, and 16 are a simulated mesh using the same "
                "per-endpoint scored constants on more endpoints"
            ),
            "hardware_counterpart": (
                "none on NV4; an NVSwitch-class configuration is the physical route "
                "to higher incast degrees"
            ),
            "required_figure_disclosure": (
                "SIMULATED MESH EXTRAPOLATION at degrees 4/8/16; no NV4 hardware "
                "counterpart; NVSwitch-class hardware is the physical route"
            ),
        },
        "measurement_caveat": {
            "hardware_identification_scope": "LONG-FLOW ONLY",
            "reason": (
                "sender launches serialize through sequential PCIe writes, so real "
                "hardware cannot construct nanosecond-scale true-sync small-flow co-arrival"
            ),
            "small_flow_class": (
                "simulated model prediction with no direct hardware check"
            ),
            "required_figure_disclosure": (
                "SMALL-FLOW INCAST IS A MODEL PREDICTION; real true-sync launch is "
                "not constructible through sequential PCIe writes; hardware "
                "identification is long-flow only"
            ),
        },
        "metrics": {
            "tail": (
                "per transport, degree, rung, and seed: nearest-rank p50, p99, "
                "and maximum flow-completion time; publish seed mean and min-max"
            ),
            "fairness": (
                "for each release wave, Jain J=(sum g_i)^2/(n*sum g_i^2), "
                "g_i=payload_bytes/FCT_ps across its concurrently released senders; "
                "publish the mean across waves and seeds with seed min-max"
            ),
            "cdf": (
                "mean empirical CDF across nine seeds with a pointwise seed min-max band"
            ),
            "capacity_bound": (
                "source classes are capped at 100 GB/s and their aggregate at the "
                "207.101921876 GB/s receiver plateau"
            ),
        },
        "frozen_hypotheses": [
            {
                "id": "H1",
                "claim": (
                    "The corrected degree-3 rnic-nn 512 KiB p50 moves left of "
                    "TRAF-71 by the fair-share entity correction and no raw capacity change."
                ),
                "bar": (
                    "legacy/corrected p50 is within 5 percent of 601/360 and the "
                    "corrected p50 is at or left of corrected NVLink"
                ),
            },
            {
                "id": "H2",
                "claim": (
                    "rnic-nn-fluid has no tail beyond its exact capacity-bound "
                    "class-service oracle and sits at or left of both packet transports."
                ),
                "bar": (
                    "every fluid completion is within 1 ps of the analytical class-fluid "
                    "oracle and fluid p50, p99, and worst are no larger in all 42 cells"
                ),
            },
            {
                "id": "H3",
                "claim": (
                    "At degrees 4, 8, and 16, rnic-nn and fluid increasingly beat "
                    "NVLink on small-flow p99 and worst-flow FCT."
                ),
                "bar": (
                    "for 256 B, 1 KiB, and 4 KiB, both references are strictly left "
                    "at every mesh degree and their relative advantages are nondecreasing"
                ),
            },
            {
                "id": "H4",
                "claim": (
                    "Global fair share increasingly improves concurrent-flow fairness "
                    "over the NVLink credit domain at mesh degrees."
                ),
                "bar": (
                    "for the three small-flow rungs, rnic-nn and fluid Jain fairness "
                    "are no lower at degrees 4, 8, and 16 and the gap is nondecreasing"
                ),
            },
            {
                "id": "H5",
                "claim": (
                    "Long-flow service stays within the mapped physical capacity and "
                    "fluid removes only packet overhead, not required queued work."
                ),
                "bar": (
                    "no source or destination service exceeds its cap; packet wire and "
                    "fluid payload ledgers are exact; every flow completes"
                ),
            },
        ],
        "fatal_guards": [
            "the TRAF-71 tree differs by one byte or tracked object",
            "a degree 1 to 3 release tuple differs from TRAF-69",
            "a corrected source class is granted above 100000000000 byte/s",
            "the corrected destination aggregate exceeds 207101921876 byte/s",
            "more than one rnic flow is active for one ordered-pair class",
            "any transport loses, duplicates, or changes a flow release or payload",
            "rnic-nn packet geometry differs from 256 payload plus 16 header bytes",
            "rnic-nn-fluid emits a packet, header, ACK, control byte, or reverse byte",
            "a CDF is nonmonotone or its final pointwise band does not equal one",
            "a reported fairness value is outside [0,1] or degree 1 differs from one",
            "a degree 4, 8, or 16 figure omits the simulated-mesh disclosure",
            "a small-flow figure omits the hardware-identification caveat",
        ],
        "evidence_classes": {
            "run_configuration": "42 rung-degree cells, nine seeds, three transports",
            "exact_oracle": (
                "mapping arithmetic, byte ledgers, capacity ledgers, and fluid class service"
            ),
            "behavioral": "H1 through H4, reported separately from fatal guards",
            "structural": "preservation locks, topology labels, and figure disclosures",
            "simulation": (
                "all FCT, tail, and fairness observations; no new hardware evidence"
            ),
        },
        "plot_contract": {
            "cdf_physical_stem": "nvlink-rnic-fluid-fct-cdf-physical",
            "cdf_mesh_stem": "nvlink-rnic-fluid-fct-cdf-mesh",
            "tail_stem": "nvlink-rnic-fluid-tail",
            "fairness_stem": "nvlink-rnic-fluid-fairness",
            "mapping_audit_stem": "nvlink-rnic-mapping-audit-degree-3",
            "formats": ["pdf", "png"],
            "path_rendering": "POSIX",
            "bands": "pointwise minimum to maximum across nine seeds",
            "transport_styles": {
                "nvlink-credit": "solid",
                "rnic-nn": "dashed",
                "rnic-nn-fluid": "dotted",
            },
            "layout": (
                "split physical and extrapolated CDF figures; one panel per rung; "
                "tail and fairness figures use one panel per rung; mapping audit "
                "directly overlays TRAF-71 and corrected degree-3 curves"
            ),
        },
        "preservation_lock": _preservation_lock(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(build(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
