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
        if name in {"operation", "direction", "service_class", "path_endpoint"}
        else int(value)
        for name, value in rows[0].items()
    }


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
        "base_path_attributed_ps": "base_path_ps",
        "numa_attributed_ps": "numa_ps",
        "iommu_attributed_ps": "iommu_ps",
        "acs_attributed_ps": "acs_ps",
        "switch_attributed_ps": "switch_ps",
        "ddio_attributed_ps": "ddio_ps",
        "gpu_direct_attributed_ps": "gpu_direct_ps",
    }
    for attributed_field, configured_field in path_fields.items():
        expected = expected_samples * int(measured[configured_field])
        if int(measured[attributed_field]) != expected:
            failures.append(f"{attributed_field}={measured[attributed_field]} expected {expected}")
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
            numa_ps=numa_ps,
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
    return 10, failures


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
    rows = _sweep_bytes(probe) + _sweep_link(probe) + _sweep_read_window(probe) + _sweep_path(probe)
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
