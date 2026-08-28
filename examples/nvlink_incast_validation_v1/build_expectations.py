#!/usr/bin/env python3
"""Build the expectations-only NV4 incast freeze registered as TRAF-74."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simllm.backends.htsim_nvlink import (
    NvlinkDomainResult,
    NvlinkDomainService,
    NvlinkFlowPolicy,
    NvlinkTransfer,
    load_nvlink_candidate_profile,
)

HERE = Path(__file__).resolve().parent
EXPECTATIONS_JSON = HERE / "expectations.json"
EXPECTATIONS_MARKDOWN = HERE / "expectations.md"
PROFILE_PATH = ROOT / "examples" / "a100_nvlink_packet_v1" / "candidate-profile.json"
MODEL_PATH = ROOT / "simllm" / "backends" / "htsim_nvlink.py"
PRODUCER_PATH = ROOT / "examples" / "a100_nvlink_packet_v2" / "nvlink_packet_lane.cu"
PRODUCER_SHA_HEADER = ROOT / "examples" / "a100_nvlink_packet_v2" / "sha256.h"
TRAF70_SCORE_PATH = ROOT / "examples" / "a100_nvlink_packet_v2" / "hardware-score.json"

FROZEN_TASK_ID = "TRAF-73"
REGISTRY_TASK_ID = "TRAF-74"
SCHEMA = "simllm-nvlink-incast-validation-expectations-v1"
FLOW_SIZES_BYTES = (262_144, 524_288)
DEGREES = (1, 2, 3)
REPETITIONS = 7
PRODUCER = "persistent_sm_peer_write"
PRODUCER_PAYLOAD_BYTES = 256
ACCEPTANCE_RELATIVE_ERROR = 0.15
LAUNCH_SKEW_PER_ADDITIONAL_SENDER_PS = 5_000_000
LAUNCH_SKEW_MAX_FRACTION = 0.10
PRIOR_ONE_MIB_COMPLETION_PS = 416_768_014
PRIOR_ONE_MIB_SOURCE = (
    "accepted TRAF-70 isolated CORNER_NVINC_033 persistent peer-write row"
)

PRESERVED_ROOTS = (
    "examples/a100_nvlink_packet_v2",
    "examples/nvlink_flow_dynamics_v1",
    "examples/nvlink_rnic_comparison_v2",
)
PRESERVED_FILES = (
    "examples/a100_nvlink_packet_v1/candidate-profile.json",
    "simllm/backends/htsim_nvlink.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tracked_paths(root: str) -> list[str]:
    completed = subprocess.run(
        ("git", "ls-files", "--", root),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"git ls-files failed for {root}: {completed.stderr.strip()}")
    paths = [line for line in completed.stdout.splitlines() if line]
    if not paths:
        raise RuntimeError(f"preservation root has no tracked files: {root}")
    return paths


def _recorded_preservation_lock(
    recorded: dict[str, object],
) -> dict[str, object]:
    artifacts = recorded.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError("the recorded preservation lock has no artifact list")
    if recorded.get("artifact_count") != len(artifacts):
        raise RuntimeError("the recorded preservation inventory count changed")
    if recorded.get("artifacts_sha256") != _canonical_sha256(artifacts):
        raise RuntimeError("the recorded preservation inventory digest changed")
    paths = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise TypeError("the recorded preservation artifact is not an object")
        path = artifact.get("path")
        digest = artifact.get("sha256")
        size = artifact.get("bytes")
        if (
            not isinstance(path, str)
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise ValueError("the recorded preservation artifact path is not portable")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("the recorded preservation artifact digest is malformed")
        if not isinstance(size, int) or size < 0:
            raise ValueError("the recorded preservation artifact size is malformed")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise RuntimeError("the recorded preservation inventory has duplicate paths")
    return json.loads(json.dumps(recorded))


def _preservation_lock(
    recorded: dict[str, object] | None = None,
) -> dict[str, object]:
    if recorded is not None:
        return _recorded_preservation_lock(recorded)
    paths: list[str] = []
    for root in PRESERVED_ROOTS:
        paths.extend(_tracked_paths(root))
    paths.extend(PRESERVED_FILES)
    unique = sorted(set(paths))
    artifacts = [
        {
            "path": path,
            "sha256": _sha256(ROOT / path),
            "bytes": (ROOT / path).stat().st_size,
        }
        for path in unique
    ]
    return {
        "rule": (
            "every listed merged study, corrected capture and scored runtime "
            "artifact remains byte-identical"
        ),
        "artifact_count": len(artifacts),
        "artifacts_sha256": _canonical_sha256(artifacts),
        "artifacts": artifacts,
    }


def _recorded_or_current_sha256(
    path: Path, preservation: dict[str, object]
) -> str:
    relative = path.relative_to(ROOT).as_posix()
    artifacts = preservation["artifacts"]
    assert isinstance(artifacts, list)
    for artifact in artifacts:
        assert isinstance(artifact, dict)
        if artifact.get("path") == relative:
            digest = artifact.get("sha256")
            if not isinstance(digest, str):
                raise TypeError(f"the recorded digest for {relative} is malformed")
            return digest
    return _sha256(path)


def _flow_completion_ps(
    result: NvlinkDomainResult, transfers: list[NvlinkTransfer]
) -> list[int]:
    values = []
    for transfer in transfers:
        delivered = [
            packet.delivered_at_ps
            for packet in result.packets
            if packet.extent_id == transfer.extent_id
        ]
        if not delivered or any(value is None for value in delivered):
            raise RuntimeError(f"simulation did not complete {transfer.extent_id}")
        values.append(max(int(value) for value in delivered) - transfer.released_at_ps)
    return values


def _acceptance_time_band(predicted_ps: int) -> dict[str, int]:
    return {
        "hardware_low_ps": int(predicted_ps / (1.0 + ACCEPTANCE_RELATIVE_ERROR)),
        "hardware_high_ps": int(
            predicted_ps / (1.0 - ACCEPTANCE_RELATIVE_ERROR) + 0.999999
        ),
    }


def _acceptance_rate_band(predicted_gbps: float) -> dict[str, float]:
    return {
        "hardware_low_gbps": predicted_gbps / (1.0 + ACCEPTANCE_RELATIVE_ERROR),
        "hardware_high_gbps": predicted_gbps / (1.0 - ACCEPTANCE_RELATIVE_ERROR),
    }


def _predictions(profile: dict[str, Any]) -> list[dict[str, object]]:
    parsed = load_nvlink_candidate_profile(PROFILE_PATH)
    service = NvlinkDomainService(parsed)
    rows = []
    packet_payload = int(profile["tx"]["max_payload_bytes"])
    header_bytes = int(profile["tx"]["header_bytes"])
    pair_raw_rate = (
        int(profile["tx"]["links_per_peer"])
        * int(profile["tx"]["per_link_rate_bytes_per_second"])
    )
    rx_rate = int(profile["rx"]["ingress_rate_bytes_per_second"])
    for size_bytes in FLOW_SIZES_BYTES:
        packet_count = (size_bytes + packet_payload - 1) // packet_payload
        wire_bytes_per_flow = size_bytes + packet_count * header_bytes
        for degree in DEGREES:
            transfers = [
                NvlinkTransfer(
                    extent_id=f"size-{size_bytes}-degree-{degree}-source-{source}",
                    source=source,
                    destination=0,
                    payload_bytes=size_bytes,
                    released_at_ps=0,
                )
                for source in range(1, degree + 1)
            ]
            result = service.serve(
                transfers,
                analytic_result=None,
                flow_policy=NvlinkFlowPolicy.RELEASE_AWARE_ROUND_ROBIN,
            )
            if not isinstance(result, NvlinkDomainResult):
                raise TypeError("the scored NVLink profile did not select the domain")
            completion_ps = _flow_completion_ps(result, transfers)
            makespan_ps = max(completion_ps)
            aggregate_gbps = degree * size_bytes * 1000 / makespan_ps
            link_floor_ps = (wire_bytes_per_flow * 1_000_000_000_000 + pair_raw_rate - 1) // (
                pair_raw_rate
            )
            receiver_floor_ps = (
                degree * wire_bytes_per_flow * 1_000_000_000_000 + rx_rate - 1
            ) // rx_rate
            physical_floor_ps = max(link_floor_ps, receiver_floor_ps)
            rows.append(
                {
                    "cell_id": f"d{degree}-s{size_bytes}",
                    "degree": degree,
                    "size_bytes": size_bytes,
                    "packet_count_per_flow": packet_count,
                    "wire_bytes_per_flow": wire_bytes_per_flow,
                    "completion_ps_by_source": completion_ps,
                    "completion_acceptance_by_source": [
                        _acceptance_time_band(value) for value in completion_ps
                    ],
                    "makespan_ps": makespan_ps,
                    "aggregate_payload_gbps": aggregate_gbps,
                    "aggregate_acceptance": _acceptance_rate_band(aggregate_gbps),
                    "physical_floor_ps": physical_floor_ps,
                    "physical_ceiling_ps": 1_000_000_000,
                    "binding_parameter": (
                        "rx_ingress_plateau" if degree == 3 else "tx_egress_plateau"
                    ),
                    "switch_mode": profile["switch"]["mode"],
                    "max_rx_buffer_occupancy_bytes": result.max_rx_buffer_occupancy_bytes,
                }
            )
    return rows


def _launch_skew_arithmetic() -> dict[str, object]:
    rows = []
    for size_bytes in FLOW_SIZES_BYTES:
        scaled_prior_ps = PRIOR_ONE_MIB_COMPLETION_PS * size_bytes // (1 << 20)
        for degree in DEGREES:
            maximum_skew_ps = (degree - 1) * LAUNCH_SKEW_PER_ADDITIONAL_SENDER_PS
            fraction = maximum_skew_ps / scaled_prior_ps
            rows.append(
                {
                    "degree": degree,
                    "size_bytes": size_bytes,
                    "prior_scaled_completion_ps": scaled_prior_ps,
                    "maximum_launch_skew_ps": maximum_skew_ps,
                    "maximum_launch_skew_fraction": fraction,
                    "pre_run_negligible": fraction <= LAUNCH_SKEW_MAX_FRACTION,
                }
            )
    if not all(bool(row["pre_run_negligible"]) for row in rows):
        raise RuntimeError("selected upper-rung flow size fails launch-skew arithmetic")
    return {
        "source": PRIOR_ONE_MIB_SOURCE,
        "prior_one_mib_completion_ps": PRIOR_ONE_MIB_COMPLETION_PS,
        "scaling_rule": (
            "scale the accepted one-source 1 MiB completion linearly by bytes; "
            "fixed producer work makes this conservative for the smaller flow"
        ),
        "per_additional_sender_budget_ps": LAUNCH_SKEW_PER_ADDITIONAL_SENDER_PS,
        "negligible_fraction_high": LAUNCH_SKEW_MAX_FRACTION,
        "post_run_fatal_guard": (
            "(degree-1)*5000000 / minimum per-flow hardware completion must be "
            "no larger than 0.10"
        ),
        "rows": rows,
    }


def build(
    *, recorded_preservation: dict[str, object] | None = None
) -> dict[str, object]:
    if recorded_preservation is None and EXPECTATIONS_JSON.is_file():
        with open(EXPECTATIONS_JSON, encoding="utf-8", newline="") as handle:
            existing = json.load(handle)
        value = existing.get("preservation_lock")
        if not isinstance(value, dict):
            raise TypeError("the existing freeze has no recorded preservation lock")
        recorded_preservation = value
    with open(PROFILE_PATH, encoding="utf-8", newline="") as handle:
        profile = json.load(handle)
    with open(TRAF70_SCORE_PATH, encoding="utf-8", newline="") as handle:
        traf70_score = json.load(handle)
    if profile.get("status") != "scored_mixed_parameter_evidence":
        raise RuntimeError(f"{REGISTRY_TASK_ID} requires the scored TRAF-70 profile")
    if traf70_score.get("status") != "COMPLETE_VALID_86_OF_86":
        raise RuntimeError(
            f"{REGISTRY_TASK_ID} requires the complete valid TRAF-70 score"
        )
    predictions = _predictions(profile)
    preservation = _preservation_lock(recorded_preservation)
    payload = {
        "schema": SCHEMA,
        "study": {
            "task_id": FROZEN_TASK_ID,
            "status": "expectations_only",
            "date": "2026-08-28",
            "title": "NV4 long-flow incast hardware validation",
            "chronology": (
                "these bytes and all predictions precede the first TRAF-73 hardware cell"
            ),
            "expectations_commit_rule": (
                "the first commit containing these exact bytes is the final pre-run freeze"
            ),
        },
        "hardware_arm": {
            "node_class": "one four-A100-SXM4-80GB NV4 node",
            "partition": "a100-hourly",
            "allocation": "one exclusive short cell, submitted once and paced",
            "producer": PRODUCER,
            "producer_path": PRODUCER_PATH.relative_to(ROOT).as_posix(),
            "producer_sha256": _recorded_or_current_sha256(
                PRODUCER_PATH, preservation
            ),
            "producer_sha_header_path": PRODUCER_SHA_HEADER.relative_to(ROOT).as_posix(),
            "producer_sha_header_sha256": _recorded_or_current_sha256(
                PRODUCER_SHA_HEADER, preservation
            ),
            "producer_payload_bytes": PRODUCER_PAYLOAD_BYTES,
            "degrees": list(DEGREES),
            "flow_sizes_bytes": list(FLOW_SIZES_BYTES),
            "repetitions_per_cell": REPETITIONS,
            "receiver": 0,
            "senders_by_degree": {
                str(degree): list(range(1, degree + 1)) for degree in DEGREES
            },
            "required_observables": [
                "per-flow receiver completion time",
                "per-flow receiver payload goodput",
                "aggregate receiver payload goodput",
                "TRAF-70 destination checksum and ordering ledger",
                "TRAF-70 per-link data and raw counters",
                "TRAF-70 replay, recovery, CRC and ECC deltas",
                "TRAF-70 clock, throttle, topology and competing-process records",
            ],
            "flow_size_encoding": (
                "256-byte producer payload times size_bytes/256 messages per flow"
            ),
            "launch_skew": _launch_skew_arithmetic(),
        },
        "simulation_arm": {
            "implementation": "simllm-htsim-nvlink-domain-v1",
            "flow_policy": "release_aware_round_robin",
            "release_ps": 0,
            "profile_path": PROFILE_PATH.relative_to(ROOT).as_posix(),
            "profile_sha256": _recorded_or_current_sha256(
                PROFILE_PATH, preservation
            ),
            "model_path": MODEL_PATH.relative_to(ROOT).as_posix(),
            "model_sha256": _recorded_or_current_sha256(MODEL_PATH, preservation),
            "profile_status": profile["status"],
            "profile_evidence_class": profile["evidence_class"],
            "parameter_snapshot": {
                "tx": profile["tx"],
                "rx": profile["rx"],
                "switch": profile["switch"],
            },
            "predictions": predictions,
        },
        "comparison": {
            "signed_relative_error_formula": "(simulation - hardware) / hardware",
            "acceptance_low": -ACCEPTANCE_RELATIVE_ERROR,
            "acceptance_high": ACCEPTANCE_RELATIVE_ERROR,
            "physical_justification": (
                "ten percentage points cover the frozen worst launch-skew fraction; "
                "five more cover guarded run-to-run endpoint variation without "
                "changing a module parameter after observation"
            ),
            "cell_verdict": (
                "PASS only when aggregate and every per-flow median signed relative "
                "error are inside the frozen band and all fatal guards pass"
            ),
            "miss_attribution_order": [
                {
                    "parameter": "pass_through_switch_identity",
                    "when": (
                        "the qualified direct NV4 topology or unchanged pass-through "
                        "identity fails; the run is void rather than scored"
                    ),
                },
                {
                    "parameter": "packetization",
                    "when": (
                        "the two size errors at one degree differ by more than 0.05 "
                        "and the larger flow is closer"
                    ),
                },
                {
                    "parameter": "credit_round",
                    "when": (
                        "the hardware-minus-simulation completion residual is equal "
                        "across sizes within 1000000 ps"
                    ),
                },
                {
                    "parameter": "rx_ingress_plateau",
                    "when": "degree 3 misses without a packetization or credit signature",
                },
                {
                    "parameter": "tx_egress_plateau",
                    "when": "degree 1 or 2 misses without a packetization or credit signature",
                },
            ],
        },
        "fatal_guards": {
            "inherited": {
                "source_path": "examples/a100_nvlink_packet_v2/expectations.json",
                "source_sha256": _recorded_or_current_sha256(
                    ROOT / "examples" / "a100_nvlink_packet_v2" / "expectations.json",
                    preservation,
                ),
                "ids": [guard["id"] for guard in json.loads(
                    (
                        ROOT / "examples" / "a100_nvlink_packet_v2" / "expectations.json"
                    ).read_text(encoding="utf-8")
                )["fatal_guards"]],
            },
            "study_specific": [
                {
                    "id": "FG11_LAUNCH_SKEW_NEGLIGIBLE",
                    "fatal_when": (
                        "the frozen sequential launch-skew budget exceeds ten percent "
                        "of the minimum per-flow hardware completion"
                    ),
                },
                {
                    "id": "FG12_COMPLETE_REPETITION_MATRIX",
                    "fatal_when": (
                        "any degree, size, repetition or source completion is missing "
                        "or duplicated"
                    ),
                },
                {
                    "id": "FG13_PRESERVATION_LOCK",
                    "fatal_when": "any merged or scored artifact digest changes",
                },
            ],
            "fatal_semantics": (
                "one violated or undecidable guard voids the run and leaves TRAF-73 open"
            ),
        },
        "physical_sanity": {
            "floor": (
                "max(one flow's wire bytes / 100 GB/s ordered-pair raw capacity, "
                "all flow wire bytes / 207.101921876 GB/s RX ingress)"
            ),
            "ceiling": "one millisecond per flow for this upper-rung short cell",
            "scaling_checks": [
                "doubling 256 KiB to 512 KiB approximately doubles serialization service",
                "degree 2 cannot exceed twice the degree-1 aggregate",
                "degree 3 cannot exceed the measured RX ingress plateau after packet overhead",
            ],
        },
        "scope_limits": {
            "hardware_degrees": [1, 2, 3],
            "simulation_only_degrees": [4, 8, 16],
            "flow_scope": "long flows only",
            "higher_degree_statement": (
                "agreement at degrees 1 to 3 supports but does not prove the "
                "simulation-only degree 4, 8 and 16 extrapolation"
            ),
            "small_flow_statement": (
                "true-sync small-flow incast is not constructible with sequential "
                "sender launch writes and remains a model prediction"
            ),
        },
        "preservation_lock": preservation,
        "publication": {
            "table_keys": ["degree", "size_bytes"],
            "required_columns": [
                "hardware per-flow completion and goodput",
                "hardware aggregate goodput",
                "simulation per-flow completion",
                "simulation aggregate goodput",
                "signed relative errors",
                "cell verdict",
                "responsible parameter on a miss",
            ],
            "figure_axes": {
                "x": "incast degree",
                "y": "aggregate receiver payload goodput in GB/s",
                "series": "measured hardware and scored simulation, one panel per flow size",
            },
            "formats": ["pdf", "png"],
        },
    }
    return payload


def _fmt_us(ps: int) -> str:
    return f"{ps / 1_000_000:.6f}"


def _render_markdown(payload: dict[str, object]) -> str:
    simulation = payload["simulation_arm"]
    assert isinstance(simulation, dict)
    rows = simulation["predictions"]
    assert isinstance(rows, list)
    launch = payload["hardware_arm"]
    assert isinstance(launch, dict)
    launch_data = launch["launch_skew"]
    assert isinstance(launch_data, dict)
    lines = [
        "# TRAF-73 NV4 long-flow incast validation freeze",
        "",
        "## Expectations-only status",
        "",
        "This record freezes the hardware matrix, simulator predictions, physical",
        "bounds, acceptance band and miss attribution before the first TRAF-73",
        "hardware cell. It contains no TRAF-73 hardware observation or result.",
        "A miss remains a published model finding and never widens this freeze.",
        "",
        "## Physical mechanism and long-flow choice",
        "",
        "Each sender writes one long byte stream into GPU 0. Sender launch writes",
        "are issued sequentially over PCIe, so their starts cannot be truly",
        "simultaneous at nanosecond scale. The corrected TRAF-70 persistent",
        "peer-write producer is reused unchanged. Its accepted one-source 1 MiB",
        f"completion was {PRIOR_ONE_MIB_COMPLETION_PS / 1_000_000:.6f} us. Scaling",
        "that duration linearly gives conservative 104.192004 us and 208.384007 us",
        "transfer times for 256 KiB and 512 KiB. The frozen budget is 5 us for",
        "each later sender, or at most 10 us at degree 3. The worst ratios are",
        "10 / 104.192004 = 9.598 percent and 10 / 208.384007 = 4.799 percent.",
        "Both are below the frozen 10 percent negligibility ceiling. The scored",
        "run recomputes the ratio from its own minimum per-flow completion; a",
        "larger ratio is fatal and voids the comparison.",
        "",
        "The physical floor is the larger of one flow's wire bytes divided by",
        "the 100 GB/s ordered-pair raw capacity and all receiver wire bytes divided",
        "by the measured 207.101921876 GB/s RX ingress plateau. One millisecond is",
        "the conservative per-flow ceiling inherited from the accepted producer",
        "scale. A point outside these bounds is a defect finding before precision",
        "is discussed.",
        "",
        "## Frozen simulation predictions",
        "",
        "The simulator uses the scored three-module NVLink domain, simultaneous",
        "release at 0 ps, the measured TX and RX endpoint plateaus, all declared",
        "candidate internals unchanged, and the structural pass-through switch.",
        "",
        "| Degree | Flow size | Per-flow completion us by source | Aggregate GB/s | Physical floor us | Binding parameter |",
        "|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        assert isinstance(row, dict)
        completions = ", ".join(_fmt_us(int(value)) for value in row["completion_ps_by_source"])
        lines.append(
            f"| {row['degree']} | {int(row['size_bytes']) // 1024} KiB | {completions} | "
            f"{float(row['aggregate_payload_gbps']):.6f} | "
            f"{_fmt_us(int(row['physical_floor_ps']))} | `{row['binding_parameter']}` |"
        )
    lines += [
        "",
        "For every per-flow completion and aggregate goodput, signed relative",
        "error is `(simulation - hardware) / hardware`. The frozen acceptance",
        "band is [-0.15, +0.15]. Ten percentage points cover the maximum allowed",
        "launch-skew fraction and five cover guarded endpoint repeatability. A cell",
        "passes only when its aggregate and every source median are inside the band",
        "and every fatal guard passes.",
        "",
        "## Frozen attribution of a miss",
        "",
        "A topology or pass-through identity failure names the pass-through switch",
        "identity and voids the run. Otherwise, a size-dependent miss that shrinks",
        "by more than five percentage points at 512 KiB names packetization. A",
        "size-independent additive completion residual within 1 us names the credit",
        "round. Remaining degree-3 misses name the RX ingress plateau; remaining",
        "degree-1 or degree-2 misses name the TX egress plateau. These rules are",
        "applied in that order and are not edited after hardware is observed.",
        "",
        "## Scope and preservation",
        "",
        "Only degrees 1, 2 and 3 have a hardware arm, and only for these long",
        "flows. Degrees 4, 8 and 16 remain DECLARED SIMULATION with no hardware",
        "counterpart on this node class. Agreement at degrees 1 to 3 supports but",
        "does not prove that higher-degree extrapolation. True-sync small-flow",
        "incast remains a model prediction.",
        "",
        f"All {payload['preservation_lock']['artifact_count']} frozen artifacts are",
        "locked. They cover TRAF-69, TRAF-70 and TRAF-72 plus the scored profile",
        "and runtime source. They must remain byte-identical. Raw hardware evidence",
        "stays outside Git and this study publishes its own compact records.",
        "",
        "## Evidence classes",
        "",
        "Run configuration, model predictions, measured hardware rows, behavioral",
        "comparisons, structural invariants and fatal guards remain separate. A",
        "fatal failure makes the result void; it is never counted as a lost point.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = build()
    with open(EXPECTATIONS_JSON, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(EXPECTATIONS_MARKDOWN, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(_render_markdown(payload))
    print(_sha256(EXPECTATIONS_JSON))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
