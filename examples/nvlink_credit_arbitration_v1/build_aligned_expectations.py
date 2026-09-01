#!/usr/bin/env python3
"""Build the expectations-only aligned TRAF-73 hardware freeze."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE_COMMIT = "4073b4e762d2a209e8f9f642e360054290d41465"
PROFILE_PATH = ROOT / "examples/a100_nvlink_packet_v1/candidate-profile.json"
MODULE_PATH = ROOT / "simllm/backends/htsim_nvlink.py"
OUTPUT_JSON = HERE / "aligned_expectations.json"
OUTPUT_MARKDOWN = HERE / "aligned_expectations.md"

H1_PAYLOAD_SIZES = (
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    196608,
    229376,
    245760,
    253952,
    258048,
    260096,
    261120,
    261632,
    261888,
    262144,
    262400,
    262656,
    263168,
    264192,
    266240,
    270336,
    278528,
    294912,
    327680,
    393216,
    524288,
    1048576,
    2097152,
    4194304,
    8388608,
)
POLICIES = (
    "release_aware_round_robin",
    "static_interleave",
    "greedy_capture",
)
DEGREES = (2, 3, 4, 8, 16)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob(path: Path) -> bytes:
    relative = path.relative_to(ROOT).as_posix()
    completed = subprocess.run(
        ("git", "show", f"{BASE_COMMIT}:{relative}"),
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _blob_sha256(path: Path) -> str:
    return hashlib.sha256(_git_blob(path)).hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _expected_rates(
    degree: int,
    policy: str,
    receiver_rate: float,
) -> list[float]:
    if policy == "static_interleave":
        return [min(60.0, receiver_rate / degree)] * degree
    if policy == "greedy_capture":
        if degree == 2:
            return [100.0, 60.0]
        small = (receiver_rate - 100.0) / (degree - 1)
        return [100.0, *([small] * (degree - 1))]
    if policy != "release_aware_round_robin":
        raise ValueError(f"unknown policy: {policy}")
    if degree == 2:
        return [100.0, 60.0]
    if degree == 3:
        return [receiver_rate - 120.0, 60.0, 60.0]
    return [receiver_rate / degree] * degree


def _policy_matrix(receiver_rate: float) -> list[dict[str, object]]:
    rows = []
    for degree in DEGREES:
        for policy in POLICIES:
            rates = _expected_rates(degree, policy, receiver_rate)
            rows.append(
                {
                    "degree": degree,
                    "topology_class": (
                        "PHYSICAL_NV4"
                        if degree in (2, 3)
                        else "SIMULATED_MESH_EXTRAPOLATION"
                    ),
                    "policy": policy,
                    "expected_raw_gbps_per_source": rates,
                    "expected_aggregate_raw_gbps": sum(rates),
                    "prediction_basis": (
                        "17-flit aligned packets paced in raw wire bytes and "
                        "served at the frozen receiver ingress plateau"
                    ),
                }
            )
    return rows


def _pool_discriminator(window_payload_bytes: int) -> list[dict[str, object]]:
    rows = []
    for senders in (1, 2, 3):
        shared_knee = window_payload_bytes / senders
        rows.append(
            {
                "sender_count": senders,
                "per_link_pool": {
                    "ideal_per_sender_knee_payload_bytes": window_payload_bytes,
                    "ideal_aggregate_outstanding_payload_bytes": (
                        senders * window_payload_bytes
                    ),
                },
                "shared_destination_pool": {
                    "ideal_per_sender_knee_payload_bytes": shared_knee,
                    "ideal_aggregate_outstanding_payload_bytes": window_payload_bytes,
                    "registered_sweep_bracket_bytes": (
                        [65536, 131072]
                        if senders == 3
                        else [int(shared_knee), int(shared_knee)]
                    ),
                },
            }
        )
    return rows


def build_expectations() -> dict[str, object]:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    module_blob_sha256 = _blob_sha256(MODULE_PATH)
    if _sha256(MODULE_PATH) != module_blob_sha256:
        raise RuntimeError("the aligned module moved from the registered base commit")

    tx = profile["tx"]
    rx = profile["rx"]
    flit_bytes = 16
    header_flits = 1
    payload_flits = int(tx["max_payload_bytes"]) // flit_bytes
    packet_wire_bytes = (header_flits + payload_flits) * flit_bytes
    if packet_wire_bytes != int(tx["credit_unit_bytes"]):
        raise RuntimeError("the declared packet and credit candidates diverged")
    links = int(tx["links_per_peer"])
    credits = int(tx["credits_per_destination"])
    payload_window = links * credits * int(tx["max_payload_bytes"])
    wire_window = links * credits * packet_wire_bytes
    per_link_window_serialization_ps = (
        credits
        * packet_wire_bytes
        * 1_000_000_000_000
        // int(tx["per_link_rate_bytes_per_second"])
    )
    receiver_rate = int(rx["ingress_rate_bytes_per_second"])

    preservation_paths = (
        HERE / "expectations.json",
        HERE / "expectations.md",
        HERE / "results.json",
        HERE / "RESULTS.md",
        ROOT / "examples/nvlink_mechanism_alignment_v1/results.json",
        PROFILE_PATH,
    )
    return {
        "schema": "simllm-nvlink-credit-arbitration-aligned-expectations-v1",
        "study": {
            "task_id": "TRAF-73",
            "date": "2026-09-01",
            "status": "EXPECTATIONS_ONLY",
            "base_commit": BASE_COMMIT,
            "chronology": (
                "Committed after TRAF-80 alignment and before the producer extension, "
                "the first aligned TRAF-73 simulation check, or any H1, H2 or H3 "
                "hardware observation."
            ),
            "hardware_executed": False,
            "supersedes": (
                "The pre-alignment mechanism predictions and the unobservable H2 "
                "23-size subset. H2 now uses the same full 31-size H1 sweep because "
                "the shared three-sender prediction is below the old subset. This is "
                "a pre-run correction. Historical artifacts, the candidate set, H1, "
                "H3 and the simulation chronology remain preserved."
            ),
        },
        "aligned_authority": {
            "implementation": "simllm-htsim-nvlink-domain-v2",
            "module_path": MODULE_PATH.relative_to(ROOT).as_posix(),
            "module_base_blob_sha256": module_blob_sha256,
            "sole_authority": (
                "Aligned packet, receiver-buffer, credit-release, reliability, "
                "ordering and switch ledgers own each modeled packet."
            ),
            "packet": {
                "flit_bytes": flit_bytes,
                "header_flits": header_flits,
                "maximum_payload_flits": payload_flits,
                "maximum_payload_bytes": int(tx["max_payload_bytes"]),
                "maximum_wire_flits": header_flits + payload_flits,
                "maximum_wire_bytes": packet_wire_bytes,
                "credit_units_per_maximum_packet": 1,
                "virtual_channel": "vc0",
            },
            "declared_flow_control": {
                "credits_per_pool": credits,
                "credit_quantum_bytes": int(tx["credit_unit_bytes"]),
                "pool_scope": "link_destination_virtual_channel",
                "return_transport_latency_ps": int(rx["credit_return_latency_ps"]),
                "evidence_class": "DECLARED_CANDIDATE",
            },
            "candidate_profile": {
                "path": PROFILE_PATH.relative_to(ROOT).as_posix(),
                "sha256": _sha256(PROFILE_PATH),
                "status": profile["status"],
            },
        },
        "candidate_set": {
            "effective_window": {
                "payload_bytes": payload_window,
                "wire_bytes": wire_window,
                "return_latency_ps": int(rx["credit_return_latency_ps"]),
                "status": "DECLARED_CANDIDATE",
            },
            "pool_scope": [
                "per_link_destination_virtual_channel",
                "shared_destination_virtual_channel",
            ],
            "arbitration": list(POLICIES),
            "retention_rule": (
                "Every candidate stays declared until a non-void hardware cell "
                "selects or refutes it. Candidates the cells cannot separate remain "
                "published as unseparated."
            ),
        },
        "physical_sanity": {
            "per_link_rate_bytes_per_second": int(
                tx["per_link_rate_bytes_per_second"]
            ),
            "ordered_pair_raw_ceiling_bytes_per_second": (
                links * int(tx["per_link_rate_bytes_per_second"])
            ),
            "receiver_raw_ceiling_bytes_per_second": receiver_rate,
            "maximum_packet_payload_ceiling_gbps": (
                links
                * int(tx["per_link_rate_bytes_per_second"])
                * int(tx["max_payload_bytes"])
                / packet_wire_bytes
                / 1e9
            ),
            "per_link_window_serialization_ps": per_link_window_serialization_ps,
            "declared_return_latency_ps": int(rx["credit_return_latency_ps"]),
            "return_to_window_serialization_ratio": (
                int(rx["credit_return_latency_ps"])
                / per_link_window_serialization_ps
            ),
            "h1_floor": (
                "Packetized wire bytes divided by the 100 GB/s ordered-pair raw "
                "ceiling; no completion may be faster."
            ),
            "h1_ceiling": (
                "The fully serialized packetized transfer plus one declared return "
                "latency per packet; this deliberately loose bound cannot promote a "
                "candidate."
            ),
            "h3_bounds": (
                "Each ordered pair is at most 100 GB/s raw and the three-source "
                "aggregate is at most 207.101921876 GB/s raw."
            ),
        },
        "h1_credit_window_and_return": {
            "directed_pairs": [
                [source, destination]
                for source in range(4)
                for destination in range(4)
                if source != destination
            ],
            "payload_sizes_bytes": list(H1_PAYLOAD_SIZES),
            "randomization_seed": 7301,
            "warmups_per_pair_and_size": 32,
            "timed_repetitions_per_pair_and_size": 200,
            "configuration_count": 12 * len(H1_PAYLOAD_SIZES),
            "timed_sample_count": 12 * len(H1_PAYLOAD_SIZES) * 200,
            "knee_rule": (
                "Fit a continuous line below each candidate break and a second line "
                "above it. A positive residual must exceed five median absolute "
                "deviations and persist for three consecutive sizes on every repeated "
                "pass of that directed pair."
            ),
            "aligned_candidate_prediction": {
                "outcome": "NO_BREAK_THROUGH_8_MIB",
                "reason": (
                    "The declared 200000 ps return is shorter than one link's "
                    "2785280 ps window serialization, so credit availability overlaps "
                    "slot reuse in the aligned receiver-owned ledger."
                ),
                "interpretation": (
                    "INCONCLUSIVE for both effective window and return. It supplies a "
                    "lower bound or shows that return overlaps serialization; it does "
                    "not confirm the declared values."
                ),
            },
            "hardware_selectors": [
                {
                    "outcome": "repeated knee near 262144 payload bytes",
                    "selects": "supports the declared effective bonded window",
                    "promotion": "none unless the fitted value contradicts the candidate",
                },
                {
                    "outcome": "repeated knee elsewhere",
                    "selects": "refutes the declared effective window",
                    "promotion": "TRAF-85 with the exact directed pair and break cell",
                },
                {
                    "outcome": "no repeated knee through 8388608 payload bytes",
                    "selects": "INCONCLUSIVE; no candidate is promoted",
                    "promotion": "none",
                },
                {
                    "outcome": "inconsistent pair knees or an unexplained boundary",
                    "selects": "INCONCLUSIVE or VOID when a fatal guard caused it",
                    "promotion": "none",
                },
            ],
        },
        "h2_pool_scope": {
            "receiver": 3,
            "source_sets": [[0], [0, 1], [0, 1, 2]],
            "payload_sizes_bytes": list(H1_PAYLOAD_SIZES),
            "sweep_relation": (
                "The aligned re-freeze uses the same complete 31-size H1 sweep. This "
                "keeps the registered shared-pool one-third prediction inside the "
                "observed range before any hardware result exists."
            ),
            "warmups_per_source_count_and_size": 64,
            "timed_repetitions_per_source_count_and_size": 200,
            "configuration_count": 3 * len(H1_PAYLOAD_SIZES),
            "timed_batch_count": 3 * len(H1_PAYLOAD_SIZES) * 200,
            "timed_per_sender_sample_count": (
                (1 + 2 + 3) * len(H1_PAYLOAD_SIZES) * 200
            ),
            "aggregate_outstanding_discriminator": _pool_discriminator(
                payload_window
            ),
            "hardware_selectors": [
                {
                    "outcome": (
                        "per-sender knees stay within one sweep interval of the "
                        "single-sender knee and aggregate outstanding bytes grow with "
                        "sender count"
                    ),
                    "selects": "per_link_destination_virtual_channel",
                },
                {
                    "outcome": (
                        "per-sender knees move to roughly one half and one third while "
                        "aggregate outstanding bytes remain within one sweep interval "
                        "of the single-sender value"
                    ),
                    "selects": "shared_destination_virtual_channel",
                },
                {
                    "outcome": "missing or inconsistent knees",
                    "selects": "INCONCLUSIVE; architecture background remains background",
                },
            ],
        },
        "h3_arbitration": {
            "receiver": 3,
            "greedy_role_sources": [0, 1, 2],
            "offered_raw_bytes_per_second_by_role": {
                "greedy": 100_000_000_000,
                "small": 60_000_000_000,
            },
            "chunk_bytes": 8 * 1024 * 1024,
            "warmup_ms": 50,
            "measurement_ms": 500,
            "drain_ms": 50,
            "steady_window": (
                "The common device measurement window opens only after all three "
                "streams are active. Completed bytes before and after that window do "
                "not enter achieved rate."
            ),
            "long_flow_only": True,
            "aligned_policy_predictions": _policy_matrix(receiver_rate / 1e9),
            "physical_nv4_selectors": [
                {
                    "policy": "release_aware_round_robin",
                    "condition": (
                        "both small senders are 57 to 63 GB/s and the greedy sender "
                        "receives the remaining work-conserving service"
                    ),
                    "center_raw_gbps_by_role": {
                        "greedy": receiver_rate / 1e9 - 120.0,
                        "small": 60.0,
                    },
                },
                {
                    "policy": "greedy_capture",
                    "condition": (
                        "the greedy sender is at least 95 GB/s and at least one small "
                        "sender is below 57 GB/s"
                    ),
                    "center_raw_gbps_by_role": {
                        "greedy": 100.0,
                        "small": (receiver_rate / 1e9 - 100.0) / 2.0,
                    },
                },
                {
                    "policy": "static_interleave",
                    "condition": (
                        "all three senders are 57 to 63 GB/s and aggregate achieved "
                        "rate is at most 189 GB/s"
                    ),
                    "center_raw_gbps_by_role": {"greedy": 60.0, "small": 60.0},
                },
                {
                    "policy": "mixed_or_inconclusive",
                    "condition": "any other non-void shape",
                    "center_raw_gbps_by_role": None,
                },
            ],
        },
        "producer_contract": {
            "lineage": "corrected_TRAF_70_nvlink_packet_lane",
            "source": _artifact(
                ROOT / "examples/a100_nvlink_packet_v2/nvlink_packet_lane.cu"
            ),
            "reuse_rule": (
                "Extend the corrected producer in place with an explicit identity off "
                "mode. Do not create another CUDA capture harness."
            ),
            "required_new_observables": [
                "per-repetition device completion for H1 and H2",
                "per-source offered rate for H3",
                "all-streams-active device timestamp",
                "common-window start and finish device timestamps",
                "completed bytes per source inside the common window",
            ],
            "inherited_observables": [
                "logical bytes and terminal extents",
                "destination checksum and order ledger",
                "per-link data and raw TX and RX counters",
                "replay, recovery, CRC and ECC counters",
                "clock, power, temperature and throttle state",
                "qualified NV4 topology and competing-process guards",
            ],
            "forbidden_evidence": "ip_link_stats64_is_not_an_nvlink_wire_authority",
        },
        "fatal_guards": [
            "the committed aligned expectations blob or its commit is not the run authority",
            "the aligned module or candidate profile differs from its frozen base identity",
            "the corrected TRAF-70 lineage or inherited fatal observables are unavailable",
            "the allocation is not one exclusive four-GPU A100 NV4 node",
            "a directed H1 pair, H2 source set or rotated H3 greedy role is missing or duplicated",
            "a logical byte, terminal extent or completed-window byte is lost or duplicated",
            "a destination checksum or order ledger disagrees",
            "a required NVLink counter, replay, recovery, CRC or ECC field is undecidable",
            "a replay, recovery, CRC or ECC counter increases",
            "per-link data and raw TX or RX counters disagree above their frozen quantization allowance",
            "a clock throttle, competing process or foreign NVLink transfer contaminates a scored cell",
            "an H1 or H2 device time is nonpositive or its randomized order differs from seed 7301",
            "the H3 common window is not exactly 500 ms after every stream is active",
            "an H3 per-source achieved rate exceeds its offered rate beyond one chunk of quantization",
            "an ordered-pair raw rate exceeds 100 GB/s or aggregate raw rate exceeds 207.101921876 GB/s beyond one chunk of quantization",
            "a live ip link stats64 value enters an NVLink wire, byte, rate or classification decision",
            "a README, module default, candidate profile or inherited result changes before classification",
        ],
        "void_rule": (
            "Any fatal-guard violation makes the full TRAF-73 hardware result VOID. "
            "H1, H2 and H3 findings may be retained diagnostically, but no candidate "
            "is selected, no task closes and no behavioral pass fraction is reported."
        ),
        "promotion_rule": {
            "task_id": "TRAF-85",
            "free_on_base_commit": True,
            "scope": (
                "Only promote an identified model value or policy that contradicts a "
                "declared aligned-module or candidate-profile value. Name the exact "
                "deciding hardware cell. Do not edit the module or profile in this wave."
            ),
        },
        "preservation": {
            "recorded_artifacts": [_artifact(path) for path in preservation_paths],
            "test_rule": (
                "Freeze tests validate the evidence recorded inside this study. They "
                "do not pin current live-tree hashes for files outside the study."
            ),
        },
    }


def render_markdown(frozen: dict[str, Any]) -> str:
    physical = frozen["physical_sanity"]
    h1 = frozen["h1_credit_window_and_return"]
    h2 = frozen["h2_pool_scope"]
    lines = [
        "# TRAF-73 aligned NVLink identification freeze",
        "",
        "## Expectations-only status",
        "",
        "This record is committed after TRAF-80 aligned the mechanism and before",
        "the producer extension, the aligned policy check and every H1, H2 or H3",
        "hardware observation. It replaces only the pre-alignment mechanism",
        "predictions. The original workloads, candidates and simulation chronology",
        "remain preserved except for one pre-run H2 sampling correction described",
        "below. No cluster time has been requested.",
        "",
        "Every candidate stays declared until a non-void hardware cell decides it.",
        "A candidate the data cannot separate is published as unseparated. The",
        "module and candidate profile remain unchanged during identification.",
        "",
        "## Aligned physical basis",
        "",
        "The aligned authority packetizes the candidate maximum payload into sixteen",
        "16-byte payload flits plus one 16-byte header flit. One maximum packet is",
        "therefore 256 payload bytes, 272 wire bytes and one declared credit unit.",
        "Credits are returned only after the receiver releases the owning buffer.",
        "The declared scope is one pool per link, destination and virtual channel; a",
        "shared destination pool remains the H2 alternative rather than a fact.",
        "",
        f"The ordered-pair raw ceiling is {physical['ordered_pair_raw_ceiling_bytes_per_second'] / 1e9:.0f} GB/s.",
        f"The measured receiver raw ceiling carried by the candidate profile is {physical['receiver_raw_ceiling_bytes_per_second'] / 1e9:.12f} GB/s.",
        f"The 17-flit payload ceiling is {physical['maximum_packet_payload_ceiling_gbps']:.12f} GB/s per ordered pair.",
        "For H1, the floor is packetized wire bytes divided by the ordered-pair",
        "ceiling. The deliberately loose ceiling is fully serialized packet service",
        "plus one declared return latency per packet. A value outside those bounds is",
        "a defect before any knee fit is interpreted.",
        "",
        "## H1: credit window and return",
        "",
        f"H1 runs all {len(h1['directed_pairs'])} directed pairs over the registered {len(h1['payload_sizes_bytes'])} payload sizes from 4 KiB through 8 MiB.",
        f"Each pair and size has {h1['warmups_per_pair_and_size']} warmups and {h1['timed_repetitions_per_pair_and_size']} device-timed repetitions in seed-7301 order.",
        "A break must exceed five median absolute deviations and persist across three",
        "consecutive sizes on every repeated pass of that directed pair.",
        "",
        f"The aligned declared candidate predicts **no break**: its {physical['declared_return_latency_ps']:,} ps return is shorter than one link's {physical['per_link_window_serialization_ps']:,} ps window serialization.",
        "No repeated break through 8 MiB is therefore INCONCLUSIVE for both window",
        "and return. It never confirms the declared values. A repeated break near",
        "262,144 payload bytes supports the effective bonded window. A repeated break",
        "elsewhere refutes that candidate and assigns its exact pair and break cell to",
        "TRAF-85 for later promotion.",
        "",
        "## H2: pool scope",
        "",
        "H2 uses receiver 3, source sets {0}, {0,1} and {0,1,2}, and the same full",
        "31-size H1 sweep. Using the full sweep before hardware keeps the shared",
        "three-sender prediction, about 87,381 payload bytes per sender, inside the",
        "sampled range. Each source-count and size has 64 warmups and 200 timed",
        "repetitions. The H1 knee rule is applied per sender.",
        "",
        "| Senders | Per-link knee per sender, B | Per-link aggregate, B | Shared knee per sender, B | Shared aggregate, B |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in h2["aggregate_outstanding_discriminator"]:
        per_link = row["per_link_pool"]
        shared = row["shared_destination_pool"]
        lines.append(
            "| {sender_count} | {per_link_knee} | {per_link_aggregate} | "
            "{shared_knee:.6f} | {shared_aggregate} |".format(
                sender_count=row["sender_count"],
                per_link_knee=per_link["ideal_per_sender_knee_payload_bytes"],
                per_link_aggregate=per_link[
                    "ideal_aggregate_outstanding_payload_bytes"
                ],
                shared_knee=shared["ideal_per_sender_knee_payload_bytes"],
                shared_aggregate=shared[
                    "ideal_aggregate_outstanding_payload_bytes"
                ],
            )
        )
    lines.extend(
        [
            "",
            "Stable per-sender knees with aggregate outstanding bytes growing with",
            "sender count select per-link pools. Knees near one half and one third with",
            "constant aggregate outstanding bytes select a shared destination pool.",
            "Missing or inconsistent knees are INCONCLUSIVE and promote no scope.",
            "",
            "## H3: downstream arbitration",
            "",
            "H3 rotates the greedy role across sources 0, 1 and 2. The greedy stream",
            "offers 100 GB/s raw and each small stream offers 60 GB/s raw. Each stream",
            "cycles through an 8 MiB ring for 50 ms warmup, one common 500 ms device",
            "measurement window and 50 ms drain. The window opens only after every",
            "stream is active, so sequential PCIe launch skew is outside the score.",
            "",
            "| Policy | Greedy center, GB/s | Small center, GB/s each | Aggregate center, GB/s | Hardware selector |",
            "|---|---:|---:|---:|---|",
            f"| release-aware round robin | {physical['receiver_raw_ceiling_bytes_per_second'] / 1e9 - 120.0:.12f} | 60.000000000000 | {physical['receiver_raw_ceiling_bytes_per_second'] / 1e9:.12f} | both small senders 57 to 63 GB/s; greedy gets the remainder |",
            f"| greedy capture | 100.000000000000 | {(physical['receiver_raw_ceiling_bytes_per_second'] / 1e9 - 100.0) / 2.0:.12f} | {physical['receiver_raw_ceiling_bytes_per_second'] / 1e9:.12f} | greedy at least 95 GB/s; at least one small sender below 57 GB/s |",
            "| static interleave | 60.000000000000 | 60.000000000000 | 180.000000000000 | every sender 57 to 63 GB/s; aggregate at most 189 GB/s |",
            "",
            "Any other non-void shape is mixed or inconclusive and promotes no policy.",
            "Degrees 4, 8 and 16 in the JSON matrix are SIMULATED MESH EXTRAPOLATION",
            "with no NV4 hardware counterpart.",
            "",
            "## Producer lineage and fatal guards",
            "",
            "The hardware cells extend the corrected TRAF-70 producer in place and",
            "retain its checksum, ordering, byte, counter, replay, recovery, clock,",
            "throttle, topology and competing-process observables. No new CUDA capture",
            "harness is allowed. H1 and H2 add per-repetition device completions. H3",
            "adds per-source offered rates and completed bytes inside the common device",
            "window. No `ip link stats64` field is an NVLink wire authority.",
            "",
            "Every fatal guard in `aligned_expectations.json` must be decidable and",
            "pass. One violation makes the complete TRAF-73 hardware result VOID. A",
            "void run reports findings, keeps TRAF-73 open, selects no candidate and",
            "does not publish a behavioral pass fraction.",
            "",
            "## Promotion boundary",
            "",
            "TRAF-85 is free at the base commit. It is used only when a non-void cell",
            "identifies a value or policy that contradicts the declared aligned module",
            "or candidate profile. The residual names the exact deciding cell. This",
            "identification wave does not edit the module, profile or any README.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_text(path: Path, value: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def main() -> None:
    frozen = build_expectations()
    _write_text(
        OUTPUT_JSON,
        json.dumps(frozen, indent=2, sort_keys=True) + "\n",
    )
    _write_text(OUTPUT_MARKDOWN, render_markdown(frozen))


if __name__ == "__main__":
    main()
