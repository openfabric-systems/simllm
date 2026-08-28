#!/usr/bin/env python3
"""Build the expectations-only TRAF-71 NVLink versus rnic-nn freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT_PATH = HERE / "expectations.json"
SOURCE_STUDY = ROOT / "examples" / "nvlink_flow_dynamics_v1"
SOURCE_EXPECTATIONS = SOURCE_STUDY / "expectations.json"
SOURCE_RESULTS = SOURCE_STUDY / "results.json"
HTSIM_COMMIT = "1dcbfec36a33753bf978cf6323bade1a6645fe4f"

PRESERVED_FLOW_STUDY_PATHS = (
    "examples/nvlink_flow_dynamics_v1/RESULTS.md",
    "examples/nvlink_flow_dynamics_v1/build_expectations.py",
    "examples/nvlink_flow_dynamics_v1/expectations.json",
    "examples/nvlink_flow_dynamics_v1/expectations.md",
    "examples/nvlink_flow_dynamics_v1/figures/nvlink-fct-cdf.pdf",
    "examples/nvlink_flow_dynamics_v1/figures/nvlink-fct-cdf.png",
    "examples/nvlink_flow_dynamics_v1/figures/nvlink-flow-dynamics.pdf",
    "examples/nvlink_flow_dynamics_v1/figures/nvlink-flow-dynamics.png",
    "examples/nvlink_flow_dynamics_v1/figures/nvlink-incast-degree-1.pdf",
    "examples/nvlink_flow_dynamics_v1/figures/nvlink-incast-degree-1.png",
    "examples/nvlink_flow_dynamics_v1/figures/nvlink-incast-degree-2.pdf",
    "examples/nvlink_flow_dynamics_v1/figures/nvlink-incast-degree-2.png",
    "examples/nvlink_flow_dynamics_v1/figures/nvlink-incast-degree-3.pdf",
    "examples/nvlink_flow_dynamics_v1/figures/nvlink-incast-degree-3.png",
    "examples/nvlink_flow_dynamics_v1/plot_study.py",
    "examples/nvlink_flow_dynamics_v1/publish_study.py",
    "examples/nvlink_flow_dynamics_v1/results.json",
    "examples/nvlink_flow_dynamics_v1/run_study.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


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
    """Reproduce the frozen release generator without running either transport."""

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


def _workload(source: dict[str, Any]) -> dict[str, object]:
    cdf = source["fct_cdf"]
    cells = []
    for source_cell in cdf["bands"]:
        releases_by_seed = {
            str(seed): _sample_releases(
                seed=seed,
                degree=source_cell["degree"],
                size_bytes=source_cell["size_bytes"],
                release_interval_ps=source_cell["release_interval_ps"],
                jitter_low_ps=source_cell["release_jitter_ps"][0],
                jitter_high_ps=source_cell["release_jitter_ps"][1],
                samples_per_sender=cdf["samples_per_seed_per_sender"],
            )
            for seed in cdf["seeds"]
        }
        cells.append(
            {
                "degree": source_cell["degree"],
                "size_bytes": source_cell["size_bytes"],
                "wave_service_ps": source_cell["wave_service_ps"],
                "release_interval_ps": source_cell["release_interval_ps"],
                "release_jitter_ps": source_cell["release_jitter_ps"],
                "release_schedule_sha256": hashlib.sha256(
                    _canonical(releases_by_seed)
                ).hexdigest(),
            }
        )
    return {
        "identity": "byte-for-byte release tuples from nvlink_flow_dynamics_v1",
        "flow_sizes_bytes": cdf["flow_sizes_bytes"],
        "degrees": [1, 2, 3],
        "seeds": cdf["seeds"],
        "seed_count": cdf["seed_count"],
        "samples_per_seed_per_sender": cdf["samples_per_seed_per_sender"],
        "destination": 3,
        "release_generator": (
            "Random(seed*1000003 + degree*10007 + size_bytes); each later wave "
            "adds release_interval_ps plus randint(jitter_low_ps,jitter_high_ps); "
            "each source adds randint(0,jitter_high_ps)"
        ),
        "cells": cells,
    }


def _mapping_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tx_plateau = 160_795_737_454
    rx_plateau = 207_101_921_876
    pair_raw = 4 * 25_000_000_000
    degree_rows = []
    for degree in (1, 2, 3):
        mapped_bytes_per_second = min(degree * pair_raw, degree * tx_plateau, rx_plateau)
        slot_ps = (272 * 8 * 1_000_000_000_000 + mapped_bytes_per_second * 8 - 1) // (
            mapped_bytes_per_second * 8
        )
        degree_rows.append(
            {
                "degree": degree,
                "formula": "min(degree*pair_raw, degree*tx_plateau, rx_plateau)*8",
                "rnic_link_rate_bps": mapped_bytes_per_second * 8,
                "binding_physical_resource": (
                    "rx_ingress" if degree == 3 else "bonded_ordered_pairs"
                ),
                "full_wire_slot_ps": slot_ps,
                "isolated_one_packet_two_serializer_ps": 2 * slot_ps,
                "full_incast_per_flow_raw_share_bytes_per_second": (
                    mapped_bytes_per_second / degree
                ),
            }
        )
    mapping = [
        {
            "physical_field": "TX endpoint egress plateau",
            "nvlink_credit_domain": "160795737454 byte/s shared by one source",
            "rnic_nn": (
                "enters min(degree*pair_raw, degree*tx_plateau, rx_plateau); it is "
                "nonbinding because one source has one 100 GB/s ordered pair"
            ),
            "fitted_constant": False,
            "expected_signed_effect": "none at full incast membership",
        },
        {
            "physical_field": "RX ingress plateau",
            "nvlink_credit_domain": "207101921876 byte/s shared at destination 3",
            "rnic_nn": "caps the degree-specific homogeneous endpoint link rate",
            "fitted_constant": False,
            "expected_signed_effect": "degree 3 is RX limited in both transports",
        },
        {
            "physical_field": "bonded ordered-pair links",
            "nvlink_credit_domain": "four links times 25000000000 byte/s",
            "rnic_nn": "degree times 100000000000 byte/s enters the endpoint minimum",
            "fitted_constant": False,
            "expected_signed_effect": "degrees 1 and 2 are pair limited at full membership",
        },
        {
            "physical_field": "propagation",
            "nvlink_credit_domain": "0 ps explicit transit term in the scored direct-link model",
            "rnic_nn": "rnic-nn propagation_delay_ps=0",
            "fitted_constant": False,
            "expected_signed_effect": "neither transport receives an additive flight delay",
        },
        {
            "physical_field": "packet geometry",
            "nvlink_credit_domain": "256 payload bytes plus 16 header bytes",
            "rnic_nn": "max wire packet 272 bytes and DATA header 16 bytes",
            "fitted_constant": False,
            "expected_signed_effect": "identical 5.882352941 percent wire-header fraction",
        },
        {
            "physical_field": "credit or sender window",
            "nvlink_credit_domain": (
                "256 destination credits of 272 bytes, 200000 ps return, 69632 wire "
                "bytes or 65536 payload bytes per full credit round"
            ),
            "rnic_nn": "no credit, congestion window, or backpressure in the pinned profile",
            "fitted_constant": False,
            "expected_signed_effect": (
                "rnic-nn cannot incur credit-window stalls, biasing its large-rung FCT and "
                "dispersion downward if NVLink exhausts credits"
            ),
        },
        {
            "physical_field": "ACK and reverse direction",
            "nvlink_credit_domain": "credit return is a timestamped resource release, not a packet",
            "rnic_nn": "0 ACK bytes because the pinned profile emits no ACK or control packet",
            "fitted_constant": False,
            "expected_signed_effect": (
                "zero reverse load biases rnic-nn downward relative to an ACK-carrying design; "
                "an ACK-pacing attribution is not applicable to this pinned algorithm"
            ),
        },
        {
            "physical_field": "arbitration",
            "nvlink_credit_domain": (
                "per-source extent round robin at packet boundaries, then stable topology "
                "order for tied RX arrivals"
            ),
            "rnic_nn": "central progressive max-min grants plus deterministic packet-slot calendar",
            "fitted_constant": False,
            "expected_signed_effect": (
                "rnic-nn should remove credit burstiness but can retain deterministic tie-order steps"
            ),
        },
    ]
    return mapping, degree_rows


def build() -> dict[str, object]:
    source = json.loads(SOURCE_EXPECTATIONS.read_text(encoding="utf-8"))
    source_result = json.loads(SOURCE_RESULTS.read_text(encoding="utf-8"))
    mapping, degree_rows = _mapping_rows()
    protected = [
        {
            "path": relative,
            "sha256": _sha256(ROOT / relative),
        }
        for relative in PRESERVED_FLOW_STUDY_PATHS
    ]
    inherited_lock = source["preservation_lock"]
    return {
        "schema": "simllm-nvlink-rnic-comparison-expectations-v1",
        "task": "TRAF-71",
        "study": {
            "name": "nvlink_rnic_comparison_v1",
            "status": "expectations_only",
            "date": "2026-08-28",
            "rule": (
                "Commit this freeze before the comparison adapter, runner, any transport "
                "execution, raw sample, measured dispersion, or result-dependent edit."
            ),
            "prohibited_writeback": (
                "Observed values never change this file. A changed mapping or expected "
                "direction requires a new unscored study version."
            ),
        },
        "source_authority": {
            "expectations_path": SOURCE_EXPECTATIONS.relative_to(ROOT).as_posix(),
            "expectations_sha256": _sha256(SOURCE_EXPECTATIONS),
            "result_path": SOURCE_RESULTS.relative_to(ROOT).as_posix(),
            "result_sha256": _sha256(SOURCE_RESULTS),
            "result_verdict": source_result["study_verdict"],
            "profile_path": source["source_profile"]["path"],
            "profile_sha256": source["source_profile"]["sha256"],
        },
        "htsim_authority": {
            "submodule_path": "third_party/htsim",
            "commit": HTSIM_COMMIT,
            "profile": "rnic-nn",
            "runtime_class": "RnicPacketizedManifoldRuntime",
            "adapter_rule": (
                "Build a study-local adapter from an immutable export of the pinned commit "
                "and call the runtime with exact picosecond releases. Do not edit htsim."
            ),
            "source_semantics": (
                "central max-min table and collision-free full-wire-quantum calendar; no "
                "route, queue, loss, backpressure, acknowledgement, PRBS pacer, or Ring-CAM"
            ),
            "ack_pacing_claim": "NOT_APPLICABLE_EXPECTED",
        },
        "physical_constants": {
            "tx_endpoint_egress_bytes_per_second": 160_795_737_454,
            "rx_ingress_bytes_per_second": 207_101_921_876,
            "links_per_ordered_pair": 4,
            "per_link_bytes_per_second": 25_000_000_000,
            "pair_raw_bytes_per_second": 100_000_000_000,
            "propagation_ps": 0,
            "max_payload_bytes": 256,
            "header_bytes": 16,
            "max_wire_bytes": 272,
            "wire_header_fraction": 16 / 272,
            "credits_per_destination": 256,
            "credit_unit_bytes": 272,
            "credit_return_ps": 200_000,
            "credit_round_wire_bytes": 69_632,
            "credit_round_payload_bytes": 65_536,
            "nvlink_no_queue_one_packet_ps": 12_194,
        },
        "physical_mapping": {
            "zero_fitted_constants": True,
            "rows": mapping,
            "degree_specific_rnic_rates": degree_rows,
            "homogeneous_capacity_limit": (
                "The pinned rnic-nn runtime accepts one symmetric endpoint capacity. The "
                "degree-specific minimum matches full-incast aggregate service exactly, but "
                "when fewer than degree senders are active it permits transient source rate "
                "above one 100 GB/s ordered pair. This can only bias rnic-nn FCT left and its "
                "dispersion downward; the result must quantify and retain this limitation."
            ),
        },
        "workload": _workload(source),
        "metrics": {
            "fct_ps": "completion_time_ps minus the exact frozen release time",
            "cdf": (
                "per transport, degree and rung: mean empirical CDF across nine seeds on "
                "the sorted union of that transport's seed observations"
            ),
            "cdf_band": "pointwise minimum to maximum empirical CDF across the nine seeds",
            "seed_quantile": "nearest-rank quantile ceil(p*n), clamped to one through n",
            "dispersion_formula": (
                "D=(max_seed(q_seed_0.50)-min_seed(q_seed_0.50)) / "
                "median_seed(q_seed_0.50)"
            ),
            "dispersion_units": "dimensionless ratio, plotted as percent",
        },
        "expected_directions": [
            {
                "id": "E1",
                "expectation": (
                    "At 256 bytes, rnic-nn p50 FCT is left of NVLink for every degree "
                    "because its mapped two-serializer one-packet costs are 5440, 2720, "
                    "and 2628 ps versus the NVLink no-queue 12194 ps."
                ),
                "scored": True,
            },
            {
                "id": "E2",
                "expectation": (
                    "For each transport and degree, median-relative seed dispersion at "
                    "512 KiB is below its 1 KiB value because fixed 10880 ps release jitter "
                    "occupies a smaller fraction of the longer FCT."
                ),
                "scored": True,
            },
            {
                "id": "E3",
                "expectation": (
                    "At 64 KiB and above, rnic-nn is no wider than NVLink in at least seven "
                    "of nine rung-degree cells because it has no credit-return stalls."
                ),
                "scored": True,
            },
            {
                "id": "E4",
                "expectation": (
                    "The regenerated NVLink samples reproduce the merged degree-3-left-of-"
                    "degree-1 p50 ordering at 1 KiB through 512 KiB, but not at 256 bytes."
                ),
                "scored": False,
                "classification": "entailed preservation check",
            },
            {
                "id": "E5",
                "expectation": (
                    "rnic-nn reproduces the degree-3-left-of-degree-1 ordering on at least "
                    "four of the six 1 KiB through 512 KiB rungs. A miss points to NVLink "
                    "credit or RX arbitration rather than the shared stagger schedule."
                ),
                "scored": True,
            },
            {
                "id": "E6",
                "expectation": (
                    "The ACK-pacing mechanism claim is not applicable: the pinned rnic-nn "
                    "event ledger contains DATA packets only and zero reverse bytes."
                ),
                "scored": False,
                "classification": "fatal source-semantics check",
            },
        ],
        "diagnosis_decision_rules": {
            "credit_window": (
                "Attribute an NVLink-only widening above 64 KiB to credit-window pressure "
                "only when the packet ledger shows reuse of the 256-credit round and positive "
                "credit or RX admission delay."
            ),
            "max_min_pacing": (
                "Attribute rnic-nn smoothness to central max-min packet-slot pacing, never "
                "to ACK pacing."
            ),
            "packetization": (
                "Both transports carry exactly 16/272 wire-header fraction. A difference in "
                "small-rung intercept comes from serializer composition and slot phase, not "
                "a packet-overhead mismatch."
            ),
            "incast_three": (
                "If both transports retain degree 3 left of degree 1, assign the sign to the "
                "shared release pattern. If only NVLink retains it, assign the difference to "
                "credit and stable RX arbitration. If neither does, publish failure to "
                "reproduce before further interpretation."
            ),
            "homogeneous_capacity": (
                "Any rnic-nn transient above one ordered-pair raw rate is a declared mapping "
                "bias, not an algorithm win."
            ),
        },
        "plot_contract": {
            "cdf_stem": "nvlink-rnic-fct-cdf",
            "dispersion_stem": "nvlink-rnic-dispersion",
            "formats": ["pdf", "png"],
            "cdf_panels": 7,
            "cdf_panel_key": "one panel per size rung, both transports and all degrees",
            "cdf_x_scale": "log FCT in microseconds",
            "transport_styles": {
                "nvlink-credit": "solid",
                "rnic-nn": "dashed",
            },
            "degree_encoding": "one stable color per incast degree",
            "bands": "a separate pointwise nine-seed min-max band for each curve",
            "dispersion_panels": 3,
            "dispersion_panel_key": "one panel per degree with transports side by side",
            "required_annotation": (
                "256 B is one packet: NVLink no-queue 12.194 ns; mapped rnic-nn "
                "two-serializer costs are 5.440, 2.720 and 2.628 ns for degrees 1, "
                "2 and 3. One NVLink credit round spans 64 KiB payload and returns in 200 ns."
            ),
            "path_rendering": "POSIX",
            "inspection": (
                "Inspect final PNGs at publication size for clipping, overlap, readable log "
                "ticks, visible min-max bands, legend crossings and border contact."
            ),
        },
        "preservation_lock": {
            "inherited_flow_dynamics_lock_path": (
                SOURCE_EXPECTATIONS.relative_to(ROOT).as_posix()
            ),
            "inherited_expected_artifacts": inherited_lock["expected_total_artifacts"],
            "direct_flow_dynamics_artifacts": protected,
            "direct_count": len(protected),
        },
        "fatal_guards": [
            "the expectations digest or expectations commit is not the run authority",
            "the tracked worktree is dirty before the gated run",
            "the htsim submodule pin or immutable source export is not commit 1dcbfec",
            "the study adapter is not built from that exact source export",
            "the scored profile, source expectations or source result changes",
            "any inherited preservation artifact changes",
            "any of the 18 flow-dynamics study files changes",
            "the two transports receive different release tuples in any cell",
            "a cell does not contain exactly degree times 12 flows for every seed and transport",
            "a flow, payload byte or packet is lost or duplicated",
            "a completion precedes its exact picosecond release",
            "rnic-nn emits an ACK, reverse byte, control packet, route, queue or loss event",
            "a CDF is nonmonotone or does not terminate at one",
            "the mapped rate, packet geometry or propagation differs from the frozen table",
            "the run writes bulk evidence inside the tracked repository",
            "a prior flow-dynamics runner, result or figure is executed or rewritten",
        ],
        "verdict_rules": {
            "fatal": "Any fatal-guard failure makes the run void and TRAF-71 stays open.",
            "honest_miss": "Every failed expected direction is published without retuning.",
            "closure": (
                "Close TRAF-71 only after every cell and both transports publish, both "
                "figures render in both formats, visual inspection passes, preservation "
                "holds, all fatal guards pass and each expected direction has a verdict."
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
