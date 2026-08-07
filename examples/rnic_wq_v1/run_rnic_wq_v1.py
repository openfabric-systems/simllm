"""Build and run the pre-registered native RNIC WQ v1 sweeps."""

from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "simllm" / "backends" / "rnic"
DEFAULT_BUILD_DIR = REPO_ROOT / "build" / "rnic_wq_v1"
RESULTS = Path(__file__).with_name("results.csv")


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
    if match is None or int(match.group(1)) == 0:
        raise RuntimeError("CTest did not discover the RNIC native test")
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
    return _native_executable(build_dir, "simllm_rnic_wq_probe")


def _probe(probe: Path, **parameters: int) -> dict[str, int]:
    option_names = {
        "wqes": "--wqes",
        "doorbell_batch": "--doorbell-batch",
        "signal_every": "--signal-every",
        "network_capacity": "--network-capacity",
        "network_latency_ps": "--network-latency-ps",
        "doorbell_service_ps": "--doorbell-service-ps",
        "fetch_service_ps": "--fetch-service-ps",
        "sq_depth": "--sq-depth",
        "cq_depth": "--cq-depth",
    }
    command = [str(probe)]
    for name, value in parameters.items():
        command.extend([option_names[name], str(value)])
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"probe returned {len(rows)} rows, expected one")
    return {name: int(value) for name, value in rows[0].items()}


def _common_checks(row: dict[str, int], wqes: int) -> list[str]:
    failures = []
    for name in ("posted", "delivered", "reclaimed"):
        if row[name] != wqes:
            failures.append(f"{name}={row[name]} expected {wqes}")
    if row["sq_high_watermark"] != wqes:
        failures.append(
            f"sq_high_watermark={row['sq_high_watermark']} expected {wqes}"
        )
    if row["evidence_count"] != 0:
        failures.append(f"evidence_count={row['evidence_count']} expected 0")
    return failures


def _sweep_a(probe: Path) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    wqes = 32
    doorbell_service_ps = 1000
    fetch_service_ps = 10
    for batch in (1, 4, 16):
        for signal_every in (1, 4, 16):
            measured = _probe(
                probe,
                wqes=wqes,
                doorbell_batch=batch,
                signal_every=signal_every,
                network_capacity=32,
                network_latency_ps=0,
                doorbell_service_ps=doorbell_service_ps,
                fetch_service_ps=fetch_service_ps,
                sq_depth=64,
                cq_depth=64,
            )
            expected_doorbells = wqes // batch
            expected_cqes = wqes // signal_every
            expected_jct_ps = expected_doorbells * doorbell_service_ps + (
                batch * fetch_service_ps
            )
            failures = _common_checks(measured, wqes)
            for name, expected in (
                ("doorbells", expected_doorbells),
                ("cqes", expected_cqes),
                ("network_busy", 0),
                ("jct_ps", expected_jct_ps),
            ):
                if measured[name] != expected:
                    failures.append(f"{name}={measured[name]} expected {expected}")
            rows.append(
                {
                    "sweep": "doorbell_signaling",
                    **measured,
                    "expected_doorbells": expected_doorbells,
                    "expected_cqes": expected_cqes,
                    "expected_network_busy": 0,
                    "expected_jct_ps": expected_jct_ps,
                    "check": "PASS" if not failures else "; ".join(failures),
                }
            )
    return rows


def _sweep_b(probe: Path) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    wqes = 16
    doorbell_service_ps = 100
    fetch_service_ps = 10
    network_latency_ps = 1000
    for capacity in (1, 4):
        measured = _probe(
            probe,
            wqes=wqes,
            doorbell_batch=16,
            signal_every=1,
            network_capacity=capacity,
            network_latency_ps=network_latency_ps,
            doorbell_service_ps=doorbell_service_ps,
            fetch_service_ps=fetch_service_ps,
            sq_depth=32,
            cq_depth=32,
        )
        expected_doorbells = 1
        expected_cqes = 16
        expected_network_busy = wqes - capacity
        expected_jct_ps = (
            doorbell_service_ps
            + capacity * fetch_service_ps
            + (wqes // capacity) * network_latency_ps
        )
        failures = _common_checks(measured, wqes)
        for name, expected in (
            ("doorbells", expected_doorbells),
            ("cqes", expected_cqes),
            ("network_busy", expected_network_busy),
            ("jct_ps", expected_jct_ps),
        ):
            if measured[name] != expected:
                failures.append(f"{name}={measured[name]} expected {expected}")
        rows.append(
            {
                "sweep": "network_backpressure",
                **measured,
                "expected_doorbells": expected_doorbells,
                "expected_cqes": expected_cqes,
                "expected_network_busy": expected_network_busy,
                "expected_jct_ps": expected_jct_ps,
                "check": "PASS" if not failures else "; ".join(failures),
            }
        )
    return rows


def _render_csv(rows: list[dict[str, int | str]]) -> bytes:
    stream = io.StringIO(newline="")
    fieldnames = list(rows[0])
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help="CMake build directory (default: build/rnic_wq_v1)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if measured CSV differs from the tracked results",
    )
    arguments = parser.parse_args()

    probe = _build(arguments.build_dir.resolve())
    rows = _sweep_a(probe) + _sweep_b(probe)
    rendered = _render_csv(rows)
    failed = [row for row in rows if row["check"] != "PASS"]
    if arguments.check:
        if not RESULTS.is_file() or RESULTS.read_bytes() != rendered:
            raise SystemExit(
                f"measured RNIC rows differ from tracked {RESULTS}"
            )
        print(f"tracked results match {len(rows)} measured rows")
    else:
        RESULTS.write_bytes(rendered)
        print(f"wrote {len(rows)} rows to {RESULTS}")
    print(f"checks: {len(rows) - len(failed)}/{len(rows)} PASS")
    if failed:
        for row in failed:
            print(row["check"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
