#!/usr/bin/env python3
"""Build the expectations-only second NV4 incast freeze for TRAF-74."""

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
EXPECTATIONS_JSON = HERE / "expectations_run2.json"
EXPECTATIONS_MARKDOWN = HERE / "expectations_run2.md"
PROFILE_PATH = ROOT / "examples" / "a100_nvlink_packet_v1" / "candidate-profile.json"
MODEL_PATH = ROOT / "simllm" / "backends" / "htsim_nvlink.py"
PRODUCER_PATH = ROOT / "examples" / "a100_nvlink_packet_v2" / "nvlink_packet_lane.cu"
PRODUCER_SHA_HEADER = ROOT / "examples" / "a100_nvlink_packet_v2" / "sha256.h"
TRAF70_SCORE_PATH = ROOT / "examples" / "a100_nvlink_packet_v2" / "hardware-score.json"
FIRST_RESULT_PATH = HERE / "results.json"

TASK_ID = "TRAF-74"
SCHEMA = "simllm-nvlink-incast-validation-expectations-v2"
MODEL_BASE_COMMIT = "65593131a0448d2b33f51018d5972c918dad3493"
FLOW_SIZES_BYTES = (4_194_304, 8_388_608)
DEGREES = (1, 2, 3)
REPETITIONS = 7
PRODUCER = "persistent_sm_peer_write"
PRODUCER_PAYLOAD_BYTES = 256
FLOW_POLICY = NvlinkFlowPolicy.RELEASE_AWARE_ROUND_ROBIN
ACCEPTANCE_RELATIVE_ERROR = 0.16
LAUNCH_SKEW_PER_ADDITIONAL_SENDER_PS = 5_000_000
LAUNCH_SKEW_MAX_FRACTION = 0.10
PRIOR_ONE_MIB_COMPLETION_PS = 416_768_014
PRIOR_ONE_MIB_SOURCE = (
    "accepted TRAF-70 isolated CORNER_NVINC_033 persistent peer-write row"
)
TRAF70_REPEATABILITY_FRACTION = 0.10
FIRST_LONG_RUNG_MAX_REPETITION_DEVIATION = 0.05263150421734001
WORST_PREFROZEN_SKEW_FRACTION = (
    2 * LAUNCH_SKEW_PER_ADDITIONAL_SENDER_PS
    / (PRIOR_ONE_MIB_COMPLETION_PS * FLOW_SIZES_BYTES[0] / (1 << 20))
)
PHYSICAL_CEILING_PS = 5_000_000_000

PRESERVED_ROOTS = (
    "examples/a100_nvlink_packet_v2",
    "examples/nvlink_flow_dynamics_v1",
    "examples/nvlink_rnic_comparison_v2",
    "examples/nvlink_incast_validation_v1",
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


def _recorded_preservation_lock(recorded: dict[str, object]) -> dict[str, object]:
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
            "every listed merged TRAF-69, TRAF-70 and TRAF-72 artifact and "
            "every retained first-capture record remains byte-identical"
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
                    extent_id=(
                        f"run2-size-{size_bytes}-degree-{degree}-source-{source}"
                    ),
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
                flow_policy=FLOW_POLICY,
            )
            if not isinstance(result, NvlinkDomainResult):
                raise TypeError("the scored NVLink profile did not select the domain")
            completion_ps = _flow_completion_ps(result, transfers)
            makespan_ps = max(completion_ps)
            aggregate_gbps = degree * size_bytes * 1000 / makespan_ps
            link_floor_ps = (
                wire_bytes_per_flow * 1_000_000_000_000 + pair_raw_rate - 1
            ) // pair_raw_rate
            receiver_floor_ps = (
                degree * wire_bytes_per_flow * 1_000_000_000_000 + rx_rate - 1
            ) // rx_rate
            rows.append(
                {
                    "cell_id": f"run2-d{degree}-s{size_bytes}",
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
                    "physical_floor_ps": max(link_floor_ps, receiver_floor_ps),
                    "physical_ceiling_ps": PHYSICAL_CEILING_PS,
                    "binding_parameter": (
                        "rx_ingress_plateau" if degree == 3 else "tx_egress_plateau"
                    ),
                    "switch_mode": profile["switch"]["mode"],
                    "max_rx_buffer_occupancy_bytes": (
                        result.max_rx_buffer_occupancy_bytes
                    ),
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
                    "pre_run_margin_to_budget": LAUNCH_SKEW_MAX_FRACTION - fraction,
                    "pre_run_negligible": fraction <= LAUNCH_SKEW_MAX_FRACTION,
                }
            )
    if not all(bool(row["pre_run_negligible"]) for row in rows):
        raise RuntimeError("selected run-two flow size fails launch-skew arithmetic")
    return {
        "source": PRIOR_ONE_MIB_SOURCE,
        "prior_one_mib_completion_ps": PRIOR_ONE_MIB_COMPLETION_PS,
        "scaling_rule": (
            "scale the accepted one-source 1 MiB completion linearly by bytes; "
            "fixed producer work makes this conservative for larger flows"
        ),
        "per_additional_sender_budget_ps": LAUNCH_SKEW_PER_ADDITIONAL_SENDER_PS,
        "negligible_fraction_high": LAUNCH_SKEW_MAX_FRACTION,
        "post_run_fatal_guard": (
            "(degree-1)*5000000 / minimum per-flow hardware completion must be "
            "no larger than 0.10"
        ),
        "rows": rows,
    }


def _base_model_sha256() -> str:
    relative = MODEL_PATH.relative_to(ROOT).as_posix()
    completed = subprocess.run(
        ("git", "show", f"{MODEL_BASE_COMMIT}:{relative}"),
        cwd=ROOT,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("the frozen model base commit is unavailable")
    return hashlib.sha256(completed.stdout).hexdigest()


def _verify_model_base() -> None:
    if _base_model_sha256() != _sha256(MODEL_PATH):
        raise RuntimeError("the current NVLink module differs from the frozen base")


def build(
    *, recorded_preservation: dict[str, object] | None = None
) -> dict[str, object]:
    if recorded_preservation is None and EXPECTATIONS_JSON.is_file():
        with open(EXPECTATIONS_JSON, encoding="utf-8", newline="") as handle:
            existing = json.load(handle)
        value = existing.get("preservation_lock")
        if not isinstance(value, dict):
            raise TypeError("the existing run-two freeze has no preservation lock")
        recorded_preservation = value
    if recorded_preservation is None:
        _verify_model_base()
    with open(PROFILE_PATH, encoding="utf-8", newline="") as handle:
        profile = json.load(handle)
    with open(TRAF70_SCORE_PATH, encoding="utf-8", newline="") as handle:
        traf70_score = json.load(handle)
    with open(FIRST_RESULT_PATH, encoding="utf-8", newline="") as handle:
        first_result = json.load(handle)
    if profile.get("status") != "scored_mixed_parameter_evidence":
        raise RuntimeError(f"{TASK_ID} requires the scored TRAF-70 profile")
    if traf70_score.get("status") != "COMPLETE_VALID_86_OF_86":
        raise RuntimeError(f"{TASK_ID} requires the complete valid TRAF-70 score")
    if first_result.get("status") != "VOID_FATAL_GUARD":
        raise RuntimeError(f"{TASK_ID} requires the retained first result")
    preservation = _preservation_lock(recorded_preservation)
    predictions = _predictions(profile)
    return {
        "schema": SCHEMA,
        "study": {
            "task_id": TASK_ID,
            "status": "expectations_only",
            "date": "2026-09-01",
            "title": "NV4 long-flow incast hardware validation second capture",
            "chronology": (
                "these bytes and every prediction precede the second hardware cell"
            ),
            "expectations_commit_rule": (
                "the first commit containing these exact bytes is the final run-two "
                "pre-run freeze"
            ),
            "first_result_status": first_result["status"],
            "first_result_scheduler_job": first_result["scheduler_job"],
            "first_result_sha256": _recorded_or_current_sha256(
                FIRST_RESULT_PATH, preservation
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
            "producer_sha_header_path": PRODUCER_SHA_HEADER.relative_to(
                ROOT
            ).as_posix(),
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
            "module_version_commit": MODEL_BASE_COMMIT,
            "flow_policy": FLOW_POLICY.value,
            "release_ps": 0,
            "profile_path": PROFILE_PATH.relative_to(ROOT).as_posix(),
            "profile_sha256": _recorded_or_current_sha256(
                PROFILE_PATH, preservation
            ),
            "model_path": MODEL_PATH.relative_to(ROOT).as_posix(),
            "model_sha256": _base_model_sha256(),
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
            "physical_justification": {
                "traf70_endpoint_repeatability_fraction": (
                    TRAF70_REPEATABILITY_FRACTION
                ),
                "first_512k_source_repetition_deviation_fraction": (
                    FIRST_LONG_RUNG_MAX_REPETITION_DEVIATION
                ),
                "run2_worst_prefrozen_launch_skew_fraction": (
                    WORST_PREFROZEN_SKEW_FRACTION
                ),
                "unrounded_sum_fraction": (
                    TRAF70_REPEATABILITY_FRACTION
                    + FIRST_LONG_RUNG_MAX_REPETITION_DEVIATION
                    + WORST_PREFROZEN_SKEW_FRACTION
                ),
                "rule": (
                    "round the independent retained endpoint-repeatability, prior "
                    "long-rung source spread and run-two launch-skew allowances "
                    "outward to plus or minus 16 percent"
                ),
            },
            "cell_verdict": (
                "PASS only when aggregate and every per-source median signed relative "
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
                    "when": (
                        "degree 3 misses without a packetization or credit signature"
                    ),
                },
                {
                    "parameter": "tx_egress_plateau",
                    "when": (
                        "degree 1 or 2 misses without a packetization or credit signature"
                    ),
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
                "ids": [
                    guard["id"]
                    for guard in json.loads(
                        (
                            ROOT
                            / "examples"
                            / "a100_nvlink_packet_v2"
                            / "expectations.json"
                        ).read_text(encoding="utf-8")
                    )["fatal_guards"]
                ],
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
                    "fatal_when": "any frozen artifact digest changes",
                },
            ],
            "fatal_semantics": (
                "one violated or undecidable guard voids the run and leaves TRAF-74 open"
            ),
        },
        "physical_sanity": {
            "floor": (
                "max(one flow wire bytes / 100 GB/s ordered-pair raw capacity, "
                "all flow wire bytes / 207.101921876 GB/s RX ingress)"
            ),
            "ceiling_ps": PHYSICAL_CEILING_PS,
            "ceiling_basis": (
                "8 MiB divided by the first capture 2.2260869 GB/s degree-one "
                "apparent goodput is 3.768 ms; five ms rounds outward beyond that "
                "retained producer scale"
            ),
            "scaling_checks": [
                "doubling 4 MiB to 8 MiB approximately doubles serialization service",
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
                "launch-skew fraction and budget",
                "signed relative errors",
                "cell verdict",
                "responsible parameter on a miss",
            ],
            "figure_axes": {
                "x": "incast degree",
                "y": "aggregate receiver payload goodput in GB/s",
                "series": (
                    "measured hardware and scored simulation, one panel per flow size"
                ),
            },
            "formats": ["pdf", "png"],
        },
    }


def _fmt_us(ps: int) -> str:
    return f"{ps / 1_000_000:.6f}"


def _render_markdown(payload: dict[str, Any]) -> str:
    rows = payload["simulation_arm"]["predictions"]
    launch_rows = payload["hardware_arm"]["launch_skew"]["rows"]
    lines = [
        "# TRAF-74 NV4 long-flow incast second-capture freeze",
        "",
        "## Expectations-only status",
        "",
        "This record freezes the second hardware matrix, simulator predictions,",
        "physical bounds, acceptance band and miss attribution before the second",
        "TRAF-74 hardware cell. It contains no second-capture hardware observation",
        "or result. The retained first result remains byte-identical and void.",
        "",
        "## Physical mechanism and long-flow choice",
        "",
        "Each sender writes one long byte stream into GPU 0. The unchanged TRAF-70",
        "producer starts sender work through sequential PCIe launch writes, so the",
        "starts cannot be simultaneous at nanosecond scale. The first capture showed",
        "that 256 KiB to 512 KiB flows still measured launch overhead: per-source",
        "apparent goodput ranged from about 2.2 to 3.5 GB/s against 94.117647 GB/s",
        "of packetized wire payload. The second capture therefore uses 4 MiB and",
        "8 MiB flows so observed completion is on the millisecond scale.",
        "",
        "The retained TRAF-70 one-source 1 MiB completion was 416.768014 us.",
        "Linear byte scaling gives conservative 1667.072056 us and 3334.144112 us",
        "for 4 MiB and 8 MiB. At degree 3, the maximum 10 us sequential launch",
        "offset is 0.600 percent and 0.300 percent of those values. The respective",
        "margins below the ten percent fatal ceiling are 9.400 and 9.700 percentage",
        "points. The scored run recomputes the fraction from every observed minimum",
        "per-source completion; a value above ten percent voids the entire run.",
        "",
        "The physical floor is the larger of one flow's packetized wire bytes divided",
        "by the 100 GB/s ordered-pair raw capacity and all receiver wire bytes divided",
        "by the measured 207.101921876 GB/s RX ingress plateau. The five millisecond",
        "ceiling rounds outward from 8 MiB divided by the retained slow 2.2260869 GB/s",
        "apparent producer goodput, which is 3.768 milliseconds. A point outside the",
        "frozen range is a defect finding before precision is discussed.",
        "",
        "## Frozen launch-skew margins",
        "",
        "| Degree | Flow size | Scaled completion us | Launch offset us | Fraction | Margin to budget |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in launch_rows:
        lines.append(
            f"| {row['degree']} | {int(row['size_bytes']) // (1 << 20)} MiB | "
            f"{_fmt_us(int(row['prior_scaled_completion_ps']))} | "
            f"{_fmt_us(int(row['maximum_launch_skew_ps']))} | "
            f"{100 * float(row['maximum_launch_skew_fraction']):.3f}% | "
            f"{100 * float(row['pre_run_margin_to_budget']):.3f} percentage points |"
        )
    lines += [
        "",
        "## Frozen simulation predictions",
        "",
        "The simulator is exactly `simllm-htsim-nvlink-domain-v1` from base commit",
        f"`{MODEL_BASE_COMMIT}`. Every source releases at 0 ps. The flow policy is",
        "explicitly `release_aware_round_robin`. The scored TX egress and RX ingress",
        "plateaus, declared packetization and credit round, and structural",
        "pass-through switch identity are unchanged.",
        "",
        "| Degree | Flow size | Per-source completion us | Aggregate GB/s | Physical floor us | Binding parameter |",
        "|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        completions = ", ".join(
            _fmt_us(int(value)) for value in row["completion_ps_by_source"]
        )
        lines.append(
            f"| {row['degree']} | {int(row['size_bytes']) // (1 << 20)} MiB | "
            f"{completions} | {float(row['aggregate_payload_gbps']):.6f} | "
            f"{_fmt_us(int(row['physical_floor_ps']))} | "
            f"`{row['binding_parameter']}` |"
        )
    lines += [
        "",
        "For every per-source completion median and aggregate receiver goodput,",
        "signed relative error is `(simulation - hardware) / hardware`. The band is",
        "plus or minus 16 percent. It is fixed from the retained ten percent TRAF-70",
        "endpoint-repeatability allowance, 5.263 percent maximum per-source spread",
        "on the first capture's 512 KiB rung, and 0.600 percent worst pre-run skew",
        "fraction, whose 15.863 percent sum is rounded outward. A cell passes only",
        "when its aggregate and every source median are inside the band and every",
        "fatal guard passes.",
        "",
        "## Frozen attribution of a miss",
        "",
        "A topology or pass-through identity failure names the pass-through switch",
        "identity and voids the run. Otherwise, a size-dependent miss that shrinks",
        "by more than five percentage points at 8 MiB names packetization. A",
        "size-independent additive completion residual within 1 us names the credit",
        "round. Remaining degree-3 misses name the RX ingress plateau; remaining",
        "degree-1 or degree-2 misses name the TX egress plateau. These rules are",
        "applied in that order and are not edited after hardware is observed.",
        "",
        "## Scope and preservation",
        "",
        "Only degrees 1, 2 and 3 have a hardware arm, and only for these long flows.",
        "Degrees 4, 8 and 16 remain DECLARED SIMULATION with no hardware counterpart",
        "on this node class. Agreement at degrees 1 to 3 supports but does not prove",
        "that higher-degree extrapolation. True-sync small-flow incast remains a",
        "model prediction.",
        "",
        f"All {payload['preservation_lock']['artifact_count']} frozen artifacts are",
        "locked. They cover TRAF-69, TRAF-70 and TRAF-72, the scored model and profile,",
        "and every first-capture record. They must remain byte-identical. Raw hardware",
        "evidence stays outside Git and the second capture publishes separate records.",
        "",
        "## Evidence classes",
        "",
        "Run configuration, frozen model predictions, measured hardware rows,",
        "behavioral comparisons, structural invariants and fatal guards remain",
        "separate. A fatal failure makes the result void; it is never counted as a",
        "lost behavioral point.",
    ]
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: object) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def main() -> int:
    payload = build()
    write_json(EXPECTATIONS_JSON, payload)
    write_text(EXPECTATIONS_MARKDOWN, _render_markdown(payload))
    print(_sha256(EXPECTATIONS_JSON))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
