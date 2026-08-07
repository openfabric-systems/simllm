"""Build and run the pre-registered native RNIC PCIe v1 sweeps."""

from __future__ import annotations

import argparse
import csv
import heapq
import io
import re
import subprocess
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "simllm" / "backends" / "rnic"
DEFAULT_BUILD_DIR = REPO_ROOT / "build" / "rnic_pcie_v1"
RESULTS = Path(__file__).with_name("results.csv")

RATE = {
    1: (2500, 8, 10),
    2: (5000, 8, 10),
    3: (8000, 128, 130),
    4: (16000, 128, 130),
    5: (32000, 128, 130),
}

MASK64 = (1 << 64) - 1
PROBABILITY_SCALE_PPM = 1_000_000
GAUSSIAN_SCALE_Q20 = 1 << 20
GAUSSIAN_QUANTILES_Q20 = (
    -2534994, -2083969, -1847245, -1678779, -1545043, -1432569,
    -1334521, -1246929, -1167269, -1093831, -1025400, -961079,
    -900186, -842187, -786658, -733252, -681684, -631714,
    -583140, -535786, -489502, -444152, -399618, -355794,
    -312583, -269897, -227653, -185776, -144193, -102836,
    -61638, -20536, 20536, 61638, 102836, 144193, 185776,
    227653, 269897, 312583, 355794, 399618, 444152, 489502,
    535786, 583140, 631714, 681684, 733252, 786658, 842187,
    900186, 961079, 1025400, 1093831, 1167269, 1246929,
    1334521, 1432569, 1545043, 1678779, 1847245, 2083969,
    2534994,
)


def _ceil(value: Fraction) -> int:
    return value.numerator // value.denominator + (value.numerator % value.denominator != 0)


def _native_executable(build_dir: Path, name: str) -> Path:
    candidates = (
        build_dir / name,
        build_dir / f"{name}.exe",
        build_dir / "Release" / name,
        build_dir / "Release" / f"{name}.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    joined = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"native executable not found; checked {joined}")


def _build(build_dir: Path) -> Path:
    subprocess.run(
        [
            "cmake",
            "-S",
            str(SOURCE_DIR),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSIMLLM_RNIC_BUILD_TESTS=ON",
            "-DSIMLLM_RNIC_BUILD_TOOLS=ON",
            "-DSIMLLM_RNIC_WARNINGS_AS_ERRORS=ON",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--parallel",
        ],
        check=True,
    )
    listed = subprocess.run(
        ["ctest", "--test-dir", str(build_dir), "-C", "Release", "-N"],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"Total Tests:\s+(\d+)", listed.stdout)
    if match is None or int(match.group(1)) < 3:
        raise RuntimeError("CTest did not discover all three RNIC native checks")
    subprocess.run(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "-C",
            "Release",
            "--output-on-failure",
        ],
        check=True,
    )
    return _native_executable(build_dir, "simllm_rnic_pcie_probe")


def _probe(probe: Path, **parameters: int | str) -> dict[str, int | str]:
    command = [str(probe)]
    for name, value in parameters.items():
        command.extend([f"--{name.replace('_', '-')}", str(value)])
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"probe returned {len(rows)} rows, expected one")
    return {
        name: value
        if name
        in {"operation", "direction", "service_class", "path_endpoint", "numa_profile"}
        else int(value)
        for name, value in rows[0].items()
    }


def _split_mix_64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def _analytical_word(seed: int, draw_index: int, stream: int) -> int:
    key = seed
    key ^= (0x9E3779B97F4A7C15 * (2 + 1)) & MASK64
    key ^= (0xD1B54A32D192ED03 * (0 + 1)) & MASK64
    key ^= (0x94D049BB133111EB * (draw_index + 1)) & MASK64
    key ^= (0xBF58476D1CE4E5B9 * (stream + 1)) & MASK64
    return _split_mix_64(key & MASK64)


def _probability_event(probability_ppm: int, word: int) -> bool:
    if probability_ppm == 0:
        return False
    if probability_ppm == PROBABILITY_SCALE_PPM:
        return True
    bucket = (((word >> 32) * PROBABILITY_SCALE_PPM) >> 32)
    return bucket < probability_ppm


def _gaussian_sample(mean_ps: int, standard_deviation_ps: int, word: int) -> int:
    quantile = GAUSSIAN_QUANTILES_Q20[word & 63]
    magnitude = (
        standard_deviation_ps * abs(quantile) + GAUSSIAN_SCALE_Q20 // 2
    ) // GAUSSIAN_SCALE_Q20
    if quantile < 0:
        return max(0, mean_ps - magnitude)
    return mean_ps + magnitude


def _analytical_samples(
    *,
    transactions: int,
    seed: int,
    profile: str,
    incidence_probability_ppm: int,
    mean_ps: int,
    standard_deviation_ps: int,
    tail_probability_ppm: int = 0,
    tail_mean_ps: int = 0,
    tail_standard_deviation_ps: int = 0,
) -> tuple[list[int], int, int]:
    samples = []
    occurrences = 0
    tails = 0
    for draw_index in range(transactions):
        if not _probability_event(
            incidence_probability_ppm,
            _analytical_word(seed, draw_index, 0),
        ):
            samples.append(0)
            continue
        occurrences += 1
        if profile == "fixed":
            samples.append(mean_ps)
        elif profile == "gaussian":
            samples.append(
                _gaussian_sample(
                    mean_ps,
                    standard_deviation_ps,
                    _analytical_word(seed, draw_index, 2),
                )
            )
        elif profile == "gaussian_tail_mixture":
            tail = _probability_event(
                tail_probability_ppm,
                _analytical_word(seed, draw_index, 1),
            )
            tails += int(tail)
            samples.append(
                _gaussian_sample(
                    tail_mean_ps if tail else mean_ps,
                    tail_standard_deviation_ps if tail else standard_deviation_ps,
                    _analytical_word(seed, draw_index, 3 if tail else 2),
                )
            )
        else:
            raise ValueError(f"unsupported analytical profile {profile}")
    return samples, occurrences, tails


def _serialization_ps(bytes_: int, generation: int, lanes: int) -> int:
    rate, encoding_numerator, encoding_denominator = RATE[generation]
    return _ceil(
        Fraction(
            8 * bytes_ * encoding_denominator * 1_000_000,
            rate * lanes * encoding_numerator,
        )
    )


def _aligned_read_counts(bytes_: int, mps: int, mrrs: int) -> tuple[int, int]:
    request_tlps = (bytes_ + mrrs - 1) // mrrs
    completion_tlps = 0
    remaining = bytes_
    while remaining:
        request_bytes = min(remaining, mrrs)
        completion_tlps += (request_bytes + mps - 1) // mps
        remaining -= request_bytes
    return request_tlps, completion_tlps


class _RationalLink:
    def __init__(self, generation: int, lanes: int) -> None:
        rate, encoding_numerator, encoding_denominator = RATE[generation]
        self._per_byte = Fraction(
            8 * encoding_denominator * 1_000_000,
            rate * lanes * encoding_numerator,
        )
        self.cursor = Fraction(0)

    def ready(self, not_before_ps: int) -> int:
        return _ceil(max(self.cursor, Fraction(not_before_ps)))

    def reserve(self, not_before_ps: int, bytes_: int) -> tuple[int, int]:
        start = max(self.cursor, Fraction(not_before_ps))
        self.cursor = start + bytes_ * self._per_byte
        return _ceil(start), _ceil(self.cursor)


def _read_oracle(
    transactions: int,
    bytes_: int,
    mps: int,
    mrrs: int,
    slots: int,
    response_latency_ps: int,
) -> tuple[int, int]:
    request_link = _RationalLink(5, 16)
    completion_link = _RationalLink(5, 16)
    releases: list[int] = []
    outstanding_wait = 0
    final_completion = 0

    for _ in range(transactions):
        remaining = bytes_
        while remaining:
            request_bytes = min(remaining, mrrs)
            initial_link_ready = request_link.ready(0)
            issue_at = initial_link_ready
            while releases and releases[0] <= issue_at:
                heapq.heappop(releases)
            if len(releases) == slots:
                issue_at = max(issue_at, releases[0])
                while releases and releases[0] <= issue_at:
                    heapq.heappop(releases)
            outstanding_wait += issue_at - initial_link_ready
            request_not_before = issue_at if issue_at > initial_link_ready else 0
            _, request_finished = request_link.reserve(request_not_before, 24)
            response_ready = request_finished + response_latency_ps
            final_completion = response_ready
            completion_remaining = request_bytes
            while completion_remaining:
                payload = min(completion_remaining, mps)
                _, final_completion = completion_link.reserve(response_ready, payload + 20)
                completion_remaining -= payload
            heapq.heappush(releases, final_completion)
            remaining -= request_bytes
    return final_completion, outstanding_wait


def _base_row(
    sweep: str,
    measured: dict[str, int | str],
    expected_request_tlps: int,
    expected_completion_tlps: int,
    expected_link_bytes: int,
    expected_jct_ps: int,
    expected_outstanding_wait_ps: int = 0,
    expected_numa_ps: int = 0,
    expected_iommu_ps: int = 0,
    expected_numa_occurrences: int | None = None,
    expected_numa_tail_draws: int = 0,
    expected_minimum_completion_ps: int | None = None,
) -> dict[str, int | str]:
    failures = []
    checks = {
        "request_tlps": expected_request_tlps,
        "completion_tlps": expected_completion_tlps,
        "expected_link_bytes": expected_link_bytes,
        "jct_ps": expected_jct_ps,
        "outstanding_wait_ps": expected_outstanding_wait_ps,
        "numa_attributed_ps": expected_numa_ps,
        "iommu_attributed_ps": expected_iommu_ps,
    }
    actual_link_bytes = int(measured["h2d_link_bytes"]) + int(measured["d2h_link_bytes"])
    actual = dict(measured)
    actual["expected_link_bytes"] = actual_link_bytes
    for name, expected in checks.items():
        if int(actual[name]) != expected:
            failures.append(f"{name}={actual[name]} expected {expected}")
    for prefix in ("h2d", "d2h"):
        payload = int(measured[f"{prefix}_payload_bytes"])
        overhead = int(measured[f"{prefix}_overhead_bytes"])
        link = int(measured[f"{prefix}_link_bytes"])
        if payload + overhead != link:
            failures.append(f"{prefix} conservation={payload}+{overhead} != {link}")
    requested_transactions = int(measured["requested_transactions"])
    accounted_transactions = int(measured["accounted_transactions"])
    if accounted_transactions != requested_transactions:
        failures.append(
            f"accounted_transactions={accounted_transactions} expected {requested_transactions}"
        )
    if int(measured["analytical_profile_version"]) != 1:
        failures.append(
            "analytical_profile_version="
            f"{measured['analytical_profile_version']} expected 1"
        )
    expected_useful = requested_transactions * int(measured["useful_bytes"])
    expected_transferred = requested_transactions * int(measured["bytes"])
    if int(measured["accounted_useful_bytes"]) != expected_useful:
        failures.append(
            "accounted_useful_bytes="
            f"{measured['accounted_useful_bytes']} expected {expected_useful}"
        )
    if int(measured["accounted_transferred_bytes"]) != expected_transferred:
        failures.append(
            "accounted_transferred_bytes="
            f"{measured['accounted_transferred_bytes']} "
            f"expected {expected_transferred}"
        )
    operation = str(measured["operation"])
    request_prefix = "h2d" if measured["direction"] == "host_to_device" else "d2h"
    completion_prefix = "d2h" if request_prefix == "h2d" else "h2d"
    directional = {
        "h2d": {"tlps": 0, "payload_bytes": 0, "overhead_bytes": 0},
        "d2h": {"tlps": 0, "payload_bytes": 0, "overhead_bytes": 0},
    }
    if operation == "posted_write":
        directional[request_prefix] = {
            "tlps": expected_request_tlps,
            "payload_bytes": expected_transferred,
            "overhead_bytes": expected_request_tlps * int(measured["posted_write_overhead_bytes"]),
        }
    elif operation == "nonposted_read":
        directional[request_prefix] = {
            "tlps": expected_request_tlps,
            "payload_bytes": 0,
            "overhead_bytes": expected_request_tlps * int(measured["read_request_overhead_bytes"]),
        }
        directional[completion_prefix] = {
            "tlps": expected_completion_tlps,
            "payload_bytes": expected_transferred,
            "overhead_bytes": expected_completion_tlps * int(measured["completion_overhead_bytes"]),
        }
    elif operation != "host_store":
        failures.append(f"unsupported directional oracle operation {operation}")
    for prefix, expected_direction in directional.items():
        for field, expected in expected_direction.items():
            measured_field = int(measured[f"{prefix}_{field}"])
            if measured_field != expected:
                failures.append(f"{prefix}_{field}={measured_field} expected {expected}")
        expected_direction_link = (
            expected_direction["payload_bytes"] + expected_direction["overhead_bytes"]
        )
        measured_link = int(measured[f"{prefix}_link_bytes"])
        if measured_link != expected_direction_link:
            failures.append(
                f"{prefix}_link_bytes={measured_link} expected {expected_direction_link}"
            )
    total_directional_tlps = int(measured["h2d_tlps"]) + int(measured["d2h_tlps"])
    expected_total_tlps = expected_request_tlps + expected_completion_tlps
    if total_directional_tlps != expected_total_tlps:
        failures.append(f"directional_tlps={total_directional_tlps} expected {expected_total_tlps}")
    expected_payload = 0 if operation == "host_store" else expected_transferred
    actual_payload = int(measured["h2d_payload_bytes"]) + int(measured["d2h_payload_bytes"])
    if actual_payload != expected_payload:
        failures.append(f"directional payload={actual_payload} expected {expected_payload}")

    expected_service_class = {
        "host_store": "doorbell_record",
        "posted_write": "payload_write",
        "nonposted_read": "payload_read",
    }[operation]
    if measured["service_class"] != expected_service_class:
        failures.append(
            f"service_class={measured['service_class']} expected {expected_service_class}"
        )
    expected_samples = (
        requested_transactions if operation == "host_store" else expected_request_tlps
    )
    if int(measured["latency_samples"]) != expected_samples:
        failures.append(
            f"latency_samples={measured['latency_samples']} expected {expected_samples}"
        )
    expected_host_store_bytes = expected_transferred if operation == "host_store" else 0
    if int(measured["host_store_bytes"]) != expected_host_store_bytes:
        failures.append(
            f"host_store_bytes={measured['host_store_bytes']} expected {expected_host_store_bytes}"
        )
    service_expectations = {
        "host_store_service_ps": (
            expected_samples * int(measured["host_store_latency_ps"])
            if operation == "host_store"
            else 0
        ),
        "posted_visibility_service_ps": (
            expected_samples * int(measured["posted_visibility_ps"])
            if operation == "posted_write"
            else 0
        ),
        "read_completion_service_ps": (
            expected_samples * int(measured["read_completion_ps"])
            if operation == "nonposted_read"
            else 0
        ),
    }
    for field, expected in service_expectations.items():
        if int(measured[field]) != expected:
            failures.append(f"{field}={measured[field]} expected {expected}")
    path_fields = {
        "base_path_attributed_ps": expected_samples * int(measured["base_path_ps"]),
        "numa_attributed_ps": expected_numa_ps,
        "iommu_attributed_ps": expected_iommu_ps,
        "acs_attributed_ps": expected_samples * int(measured["acs_ps"]),
        "switch_attributed_ps": expected_samples * int(measured["switch_ps"]),
        "ddio_attributed_ps": expected_samples * int(measured["ddio_ps"]),
        "gpu_direct_attributed_ps": expected_samples * int(measured["gpu_direct_ps"]),
    }
    for attributed_field, expected in path_fields.items():
        if int(measured[attributed_field]) != expected:
            failures.append(f"{attributed_field}={measured[attributed_field]} expected {expected}")
    if expected_numa_occurrences is None:
        expected_numa_occurrences = (
            expected_samples if int(measured["numa_mean_ps"]) != 0 else 0
        )
    profile_expectations = {
        "numa_profile_draws": expected_samples,
        "numa_profile_occurrences": expected_numa_occurrences,
        "numa_profile_tail_draws": expected_numa_tail_draws,
    }
    fixed_components = {
        "iommu": "iommu_ps",
        "acs": "acs_ps",
        "switch": "switch_ps",
        "ddio": "ddio_ps",
        "gpu_direct": "gpu_direct_ps",
    }
    for component, configured_field in fixed_components.items():
        profile_expectations[f"{component}_profile_draws"] = expected_samples
        profile_expectations[f"{component}_profile_occurrences"] = (
            expected_samples if int(measured[configured_field]) != 0 else 0
        )
        profile_expectations[f"{component}_profile_tail_draws"] = 0
    for field, expected in profile_expectations.items():
        if int(measured[field]) != expected:
            failures.append(f"{field}={measured[field]} expected {expected}")
    if (
        expected_minimum_completion_ps is not None
        and int(measured["minimum_completion_ps"]) != expected_minimum_completion_ps
    ):
        failures.append(
            "minimum_completion_ps="
            f"{measured['minimum_completion_ps']} expected "
            f"{expected_minimum_completion_ps}"
        )
    for zero_wait in (
        "ordering_wait_ps",
        "completion_buffer_wait_ps",
        "credit_wait_ps",
    ):
        if int(measured[zero_wait]) != 0:
            failures.append(f"{zero_wait}={measured[zero_wait]} expected 0")
    return {
        "sweep": sweep,
        **measured,
        "expected_request_tlps": expected_request_tlps,
        "expected_completion_tlps": expected_completion_tlps,
        "expected_link_bytes": expected_link_bytes,
        "expected_jct_ps": expected_jct_ps,
        "expected_outstanding_wait_ps": expected_outstanding_wait_ps,
        "expected_numa_ps": expected_numa_ps,
        "expected_iommu_ps": expected_iommu_ps,
        "expected_numa_occurrences": expected_numa_occurrences,
        "expected_numa_tail_draws": expected_numa_tail_draws,
        "expected_minimum_completion_ps": (
            "" if expected_minimum_completion_ps is None else expected_minimum_completion_ps
        ),
        "check": "PASS" if not failures else "; ".join(failures),
    }


def _sweep_bytes(probe: Path) -> list[dict[str, int | str]]:
    rows = []
    for mrrs in (128, 256, 512):
        for mps in (128, 256, 512):
            measured = _probe(
                probe,
                operation="nonposted_read",
                direction="device_to_host",
                bytes=512,
                mps=mps,
                mrrs=mrrs,
            )
            requests, completions = _aligned_read_counts(512, mps, mrrs)
            expected_bytes = requests * 24 + 512 + completions * 20
            rows.append(
                _base_row(
                    "mps_mrrs_bytes",
                    measured,
                    requests,
                    completions,
                    expected_bytes,
                    _read_oracle(1, 512, mps, mrrs, 64, 0)[0],
                )
            )
    return rows


def _sweep_link(probe: Path) -> list[dict[str, int | str]]:
    rows = []
    for generation in (4, 5):
        for lanes in (8, 16):
            for mps in (128, 256, 512):
                measured = _probe(
                    probe,
                    operation="posted_write",
                    direction="device_to_host",
                    bytes=4096,
                    generation=generation,
                    lanes=lanes,
                    mps=mps,
                )
                tlps = 4096 // mps
                link_bytes = 4096 + tlps * 24
                rows.append(
                    _base_row(
                        "generation_width_mps",
                        measured,
                        tlps,
                        0,
                        link_bytes,
                        _serialization_ps(link_bytes, generation, lanes),
                    )
                )
    return rows


def _sweep_read_window(probe: Path) -> list[dict[str, int | str]]:
    rows = []
    for slots in (1, 4):
        for mps in (128, 512):
            measured = _probe(
                probe,
                operation="nonposted_read",
                direction="device_to_host",
                transactions=16,
                bytes=512,
                mps=mps,
                mrrs=512,
                outstanding_reads=slots,
                read_completion_ps=1_000_000,
            )
            expected_jct, expected_wait = _read_oracle(16, 512, mps, 512, slots, 1_000_000)
            completion_tlps = 16 * (512 // mps)
            link_bytes = 16 * 24 + 16 * 512 + completion_tlps * 20
            rows.append(
                _base_row(
                    "read_window",
                    measured,
                    16,
                    completion_tlps,
                    link_bytes,
                    expected_jct,
                    expected_wait,
                )
            )
    return rows


def _sweep_path(probe: Path) -> list[dict[str, int | str]]:
    rows = []
    for numa_ps, iommu_ps in (
        (0, 0),
        (100_000, 0),
        (0, 200_000),
        (100_000, 200_000),
    ):
        measured = _probe(
            probe,
            operation="nonposted_read",
            direction="device_to_host",
            bytes=512,
            mps=512,
            mrrs=512,
            numa_mean_ps=numa_ps,
            iommu_ps=iommu_ps,
        )
        local_serialization = _read_oracle(1, 512, 512, 512, 64, 0)[0]
        rows.append(
            _base_row(
                "path_attribution",
                measured,
                1,
                1,
                556,
                local_serialization + numa_ps + iommu_ps,
                expected_numa_ps=numa_ps,
                expected_iommu_ps=iommu_ps,
            )
        )
    return rows


def _sweep_analytical_profiles(probe: Path) -> list[dict[str, int | str]]:
    transactions = 4096
    seed = 0xC0FFEE
    profiles = (
        {
            "name": "fixed",
            "profile": "fixed",
            "incidence_probability_ppm": 1_000_000,
            "mean_ps": 100_000,
            "standard_deviation_ps": 0,
        },
        {
            "name": "gaussian_narrow",
            "profile": "gaussian",
            "incidence_probability_ppm": 1_000_000,
            "mean_ps": 100_000,
            "standard_deviation_ps": 10_000,
        },
        {
            "name": "gaussian_wide",
            "profile": "gaussian",
            "incidence_probability_ppm": 1_000_000,
            "mean_ps": 100_000,
            "standard_deviation_ps": 40_000,
        },
        {
            "name": "tail_rare",
            "profile": "gaussian_tail_mixture",
            "incidence_probability_ppm": 1_000_000,
            "mean_ps": 100_000,
            "standard_deviation_ps": 10_000,
            "tail_probability_ppm": 10_000,
            "tail_mean_ps": 500_000,
            "tail_standard_deviation_ps": 50_000,
        },
        {
            "name": "tail_frequent",
            "profile": "gaussian_tail_mixture",
            "incidence_probability_ppm": 1_000_000,
            "mean_ps": 100_000,
            "standard_deviation_ps": 10_000,
            "tail_probability_ppm": 100_000,
            "tail_mean_ps": 500_000,
            "tail_standard_deviation_ps": 50_000,
        },
        {
            "name": "intermittent",
            "profile": "gaussian",
            "incidence_probability_ppm": 250_000,
            "mean_ps": 100_000,
            "standard_deviation_ps": 10_000,
        },
    )
    rows = []
    for parameters in profiles:
        tail_probability_ppm = int(parameters.get("tail_probability_ppm", 0))
        tail_mean_ps = int(parameters.get("tail_mean_ps", 0))
        tail_standard_deviation_ps = int(
            parameters.get("tail_standard_deviation_ps", 0)
        )
        samples, occurrences, tails = _analytical_samples(
            transactions=transactions,
            seed=seed,
            profile=str(parameters["profile"]),
            incidence_probability_ppm=int(parameters["incidence_probability_ppm"]),
            mean_ps=int(parameters["mean_ps"]),
            standard_deviation_ps=int(parameters["standard_deviation_ps"]),
            tail_probability_ppm=tail_probability_ppm,
            tail_mean_ps=tail_mean_ps,
            tail_standard_deviation_ps=tail_standard_deviation_ps,
        )
        measured = _probe(
            probe,
            operation="host_store",
            direction="host_to_device",
            transactions=transactions,
            bytes=8,
            analytical_seed=seed,
            numa_profile=str(parameters["profile"]),
            numa_incidence_ppm=int(parameters["incidence_probability_ppm"]),
            numa_mean_ps=int(parameters["mean_ps"]),
            numa_standard_deviation_ps=int(parameters["standard_deviation_ps"]),
            numa_tail_probability_ppm=tail_probability_ppm,
            numa_tail_mean_ps=tail_mean_ps,
            numa_tail_standard_deviation_ps=tail_standard_deviation_ps,
        )
        row = _base_row(
            f"analytical_profile_{parameters['name']}",
            measured,
            0,
            0,
            0,
            max(samples),
            expected_numa_ps=sum(samples),
            expected_numa_occurrences=occurrences,
            expected_numa_tail_draws=tails,
            expected_minimum_completion_ps=min(samples),
        )
        rows.append(row)
    return rows


def _cross_checks(rows: list[dict[str, int | str]]) -> tuple[int, list[str]]:
    failures = []
    link_rows = [row for row in rows if row["sweep"] == "generation_width_mps"]
    for mps in (128, 256, 512):
        lookup = {
            (int(row["generation"]), int(row["lanes"])): int(row["jct_ps"])
            for row in link_rows
            if int(row["mps"]) == mps
        }
        if lookup[(5, 8)] != lookup[(4, 16)]:
            failures.append(f"MPS {mps}: Gen5 x8 != Gen4 x16")
        if lookup[(5, 8)] not in {
            2 * lookup[(5, 16)],
            2 * lookup[(5, 16)] - 1,
        }:
            failures.append(f"MPS {mps}: x8 does not scale as 2x x16")

    window_rows = [row for row in rows if row["sweep"] == "read_window"]
    for mps in (128, 512):
        jct = {
            int(row["outstanding_reads"]): int(row["jct_ps"])
            for row in window_rows
            if int(row["mps"]) == mps
        }
        if jct[4] >= jct[1]:
            failures.append(f"MPS {mps}: four read slots did not reduce JCT")
    for slots in (1, 4):
        jct = {
            int(row["mps"]): int(row["jct_ps"])
            for row in window_rows
            if int(row["outstanding_reads"]) == slots
        }
        if jct[512] > jct[128]:
            failures.append(f"read slots {slots}: MPS 512 is slower than MPS 128")

    profile_rows = {
        str(row["sweep"]): row
        for row in rows
        if str(row["sweep"]).startswith("analytical_profile_")
    }
    narrow = profile_rows["analytical_profile_gaussian_narrow"]
    wide = profile_rows["analytical_profile_gaussian_wide"]
    for name, row in (("narrow", narrow), ("wide", wide)):
        if not (
            int(row["minimum_completion_ps"]) < 100_000
            and int(row["jct_ps"]) > 100_000
        ):
            failures.append(f"{name} Gaussian did not span both sides of its mean")
    narrow_range = int(narrow["jct_ps"]) - int(narrow["minimum_completion_ps"])
    wide_range = int(wide["jct_ps"]) - int(wide["minimum_completion_ps"])
    if wide_range <= narrow_range:
        failures.append("wider Gaussian sigma did not increase the observed range")
    rare = profile_rows["analytical_profile_tail_rare"]
    frequent = profile_rows["analytical_profile_tail_frequent"]
    for name, row in (("rare", rare), ("frequent", frequent)):
        if int(row["numa_profile_tail_draws"]) == 0:
            failures.append(f"{name} tail mixture selected no tail samples")
    if int(frequent["numa_profile_tail_draws"]) <= int(
        rare["numa_profile_tail_draws"]
    ):
        failures.append("higher tail probability did not increase tail selections")
    if int(frequent["numa_attributed_ps"]) <= int(rare["numa_attributed_ps"]):
        failures.append("higher tail probability did not increase aggregate delay")
    intermittent = profile_rows["analytical_profile_intermittent"]
    if not (
        int(intermittent["numa_profile_occurrences"]) < 4096
        and int(intermittent["minimum_completion_ps"]) == 0
    ):
        failures.append("intermittent incidence did not produce absent penalties")
    return 18, failures


def _render_csv(rows: list[dict[str, int | str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help="CMake build directory (default: build/rnic_pcie_v1)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if measured CSV differs from the tracked results",
    )
    arguments = parser.parse_args()

    probe = _build(arguments.build_dir.resolve())
    rows = (
        _sweep_bytes(probe)
        + _sweep_link(probe)
        + _sweep_read_window(probe)
        + _sweep_path(probe)
        + _sweep_analytical_profiles(probe)
    )
    failed = [row for row in rows if row["check"] != "PASS"]
    cross_total, cross_failures = _cross_checks(rows)
    rendered = _render_csv(rows)
    if arguments.check:
        if not RESULTS.is_file() or RESULTS.read_bytes() != rendered:
            raise SystemExit(f"measured RNIC PCIe rows differ from tracked {RESULTS}")
        print(f"tracked results match {len(rows)} measured rows")
    else:
        RESULTS.write_bytes(rendered)
        print(f"wrote {len(rows)} rows to {RESULTS}")
    print(f"row checks: {len(rows) - len(failed)}/{len(rows)} PASS")
    print(f"cross checks: {cross_total - len(cross_failures)}/{cross_total} PASS")
    if failed or cross_failures:
        for row in failed:
            print(row["check"])
        for failure in cross_failures:
            print(failure)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
