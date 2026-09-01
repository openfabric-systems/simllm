"""Run the RNIC golden model's slice-B transmit study through its C facade.

The sweep, the closed forms and the bands are frozen in expectations.md. This
script only executes them: it builds the native gate, drives the facade probe
over the grid, fits the depth-1 law per profile, and writes one summary row
per registered check.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "simllm" / "backends" / "rnic"
DEFAULT_BUILD_DIR = REPO_ROOT / "build" / "rnic_cmodel_v1"
SUMMARY = Path(__file__).with_name("summary.csv")
CURVES = Path(__file__).with_name("curves.csv")

# Frozen grid (expectations.md).
SIZES = (4096, 8192, 16384, 65536, 262144, 1048576)
DEPTHS = (1, 16, 1024)
PROFILES = ("cx5_100g", "cx7_400g")
CALIBRATION_MTU = 4096
SMALL_MESSAGE_BYTES = 1024
TAX_MESSAGE_BYTES = 1048576
TAX_MTU = 1024
REPLAY_CELL = ("cx5_100g", 65536, 16, CALIBRATION_MTU)

# Frozen constants (expectations.md, from the mlx5 campaign).
T_EFF_S = 4.48e-6
GOODPUT_BPS = {"cx5_100g": 97.1e9, "cx7_400g": 388.4e9}
LINK_FACTOR = {"cx5_100g": 1, "cx7_400g": 4}
MESSAGE_RATE_PPS = {"cx5_100g": 3.87e6, "cx7_400g": 15.48e6}
MEASURED_DEPTH_RATIO = {8192: 5.9, 65536: 1.57}
MEASURED_MTU_TAX_PP = 5.6

# Frozen message-count rule.
TARGET_PAYLOAD_BYTES = 33554432
PACKET_BUDGET = 262144


def _messages(size: int, depth: int, mtu: int) -> int:
    per_message = max(1, math.ceil(size / mtu))
    count = max(4 * depth, TARGET_PAYLOAD_BYTES // size)
    if count * per_message > PACKET_BUDGET:
        count = max(1, PACKET_BUDGET // per_message)
    return count


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
        ["cmake", "--build", str(build_dir), "--config", "Release", "--parallel"],
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
        raise RuntimeError("CTest did not discover the RNIC native tests")
    subprocess.run(
        ["ctest", "--test-dir", str(build_dir), "-C", "Release", "--output-on-failure"],
        check=True,
    )
    return _native_executable(build_dir, "simllm_rnic_cmodel_probe")


def _probe(
    probe: Path,
    profile: str,
    size: int,
    depth: int,
    mtu: int,
    replay: bool,
    trace_prefix: Path | None,
) -> dict[str, int | str]:
    command = [
        str(probe),
        "--profile",
        profile,
        "--size-bytes",
        str(size),
        "--depth",
        str(depth),
        "--mtu-bytes",
        str(0 if mtu == CALIBRATION_MTU else mtu),
        "--messages",
        str(_messages(size, depth, mtu)),
    ]
    if replay:
        command.append("--replay")
    if trace_prefix is not None:
        command.extend(["--trace-prefix", str(trace_prefix)])
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"probe returned {len(rows)} rows, expected one")
    row: dict[str, int | str] = {}
    for name, value in rows[0].items():
        row[name] = value if name == "profile" else int(value)
    row["mtu_bytes"] = mtu
    return row


def _goodput_bps(row: dict[str, int | str]) -> float:
    payload = float(row["payload_bytes"])
    seconds = float(row["last_completion_ps"]) / 1e12
    return payload * 8.0 / seconds


def _packet_rate_pps(row: dict[str, int | str]) -> float:
    return float(row["packets"]) / (float(row["last_completion_ps"]) / 1e12)


def _law_bps(size: int, goodput_bps: float) -> float:
    bits = float(size) * 8.0
    return bits / (T_EFF_S + bits / goodput_bps)


def _fit_law(rows: list[dict[str, int | str]]) -> tuple[float, float]:
    """Least-squares fit of time-per-message against message bits."""
    points = [
        (float(row["size_bytes"]) * 8.0, float(row["last_completion_ps"]) / 1e12 / float(row["messages"]))
        for row in rows
    ]
    count = float(len(points))
    sum_x = sum(x for x, _ in points)
    sum_y = sum(y for _, y in points)
    sum_xx = sum(x * x for x, _ in points)
    sum_xy = sum(x * y for x, y in points)
    denominator = count * sum_xx - sum_x * sum_x
    slope = (count * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / count
    return intercept, 1.0 / slope


def _check(
    name: str,
    cell: str,
    measured: float,
    reference: float,
    band: str,
    passed: bool,
    note: str = "",
) -> dict[str, object]:
    return {
        "check": name,
        "cell": cell,
        "measured": f"{measured:.6g}",
        "reference": f"{reference:.6g}",
        "band": band,
        "verdict": "PASS" if passed else "FAIL",
        "note": note,
    }


def _within(measured: float, reference: float, fraction: float) -> bool:
    return abs(measured - reference) <= abs(reference) * fraction


def _run_grid(probe: Path, raw_dir: Path | None) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for profile in PROFILES:
        for size in SIZES:
            for depth in DEPTHS:
                rows.append(_probe(probe, profile, size, depth, CALIBRATION_MTU, False, None))
        rows.append(_probe(probe, profile, TAX_MESSAGE_BYTES, 1024, TAX_MTU, False, None))
        rows.append(
            _probe(probe, profile, SMALL_MESSAGE_BYTES, 1024, CALIBRATION_MTU, False, None)
        )
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "cells.csv").write_bytes(_render_csv(rows))
    return rows


def _cell(rows: list[dict[str, int | str]], profile: str, size: int, depth: int, mtu: int):
    for row in rows:
        if (
            row["profile"] == profile
            and row["size_bytes"] == size
            and row["depth"] == depth
            and row["mtu_bytes"] == mtu
        ):
            return row
    raise KeyError(f"missing cell {profile} {size} {depth} {mtu}")


def _fatal_guards(rows: list[dict[str, int | str]], replay_row: dict[str, int | str]) -> list[str]:
    problems: list[str] = []
    for row in rows:
        label = f"{row['profile']}/{row['size_bytes']}/{row['depth']}/{row['mtu_bytes']}"
        if row["completions"] != row["messages"] or row["errors"] != 0:
            problems.append(f"{label}: {row['completions']} completions of {row['messages']}")
        if not row["posted"] == row["delivered"] == row["reclaimed"] == row["messages"]:
            problems.append(f"{label}: posted/delivered/reclaimed disagree")
        if row["cq_overruns"] != 0:
            problems.append(f"{label}: {row['cq_overruns']} CQ overruns")
        if row["late_releases"] != 0:
            problems.append(f"{label}: {row['late_releases']} late packet releases")
        if row["payload_bytes"] != row["messages"] * row["size_bytes"]:
            problems.append(f"{label}: packetizer lost payload bytes")
        expected_packets = row["messages"] * max(
            1, math.ceil(row["size_bytes"] / row["mtu_bytes"])
        )
        if row["packets"] != expected_packets:
            problems.append(f"{label}: {row['packets']} packets, expected {expected_packets}")
        header = row["wire_bytes"] - row["payload_bytes"]
        if header != row["packets"] * 64:
            problems.append(f"{label}: wire header bytes are not 64 per packet")
    if replay_row["replay_identical"] != 1:
        problems.append("deterministic replay identity failed on the replay cell")
    return problems


def _evaluate(rows: list[dict[str, int | str]]) -> tuple[list[dict[str, object]], dict[str, tuple[float, float]]]:
    checks: list[dict[str, object]] = []
    fits: dict[str, tuple[float, float]] = {}

    for profile in PROFILES:
        depth1 = [_cell(rows, profile, size, 1, CALIBRATION_MTU) for size in SIZES]
        fits[profile] = _fit_law(depth1)
        for size, row in zip(SIZES, depth1):
            measured = _goodput_bps(row)
            reference = _law_bps(size, GOODPUT_BPS[profile])
            checks.append(
                _check(
                    "depth1_law",
                    f"{profile}/{size}",
                    measured / 1e9,
                    reference / 1e9,
                    "15 percent",
                    _within(measured, reference, 0.15),
                )
            )

    for profile in PROFILES:
        for size in (8192, 65536):
            shallow = _goodput_bps(_cell(rows, profile, size, 1, CALIBRATION_MTU))
            deep = _goodput_bps(_cell(rows, profile, size, 1024, CALIBRATION_MTU))
            ratio = deep / shallow
            reference = GOODPUT_BPS[profile] / _law_bps(size, GOODPUT_BPS[profile])
            checks.append(
                _check(
                    "depth_ratio_law",
                    f"{profile}/{size}",
                    ratio,
                    reference,
                    "2 percent",
                    _within(ratio, reference, 0.02),
                    "the lossless pipeline must saturate at the goodput ceiling",
                )
            )
            if profile == "cx5_100g":
                measured_reference = MEASURED_DEPTH_RATIO[size]
                passed = _within(ratio, measured_reference, 0.20)
                note = "" if passed else "residual owned by the ingress meter (BACK-56)"
                checks.append(
                    _check(
                        "depth_ratio_measured",
                        f"{profile}/{size}",
                        ratio,
                        measured_reference,
                        "20 percent",
                        passed,
                        note,
                    )
                )

    for profile in PROFILES:
        big = _goodput_bps(_cell(rows, profile, TAX_MESSAGE_BYTES, 1024, CALIBRATION_MTU))
        small = _goodput_bps(_cell(rows, profile, TAX_MESSAGE_BYTES, 1024, TAX_MTU))
        tax_pp = (1.0 - small / big) * 100.0
        checks.append(
            _check(
                "mtu_tax",
                f"{profile}/1 MiB",
                tax_pp,
                MEASURED_MTU_TAX_PP,
                "2 percentage points",
                abs(tax_pp - MEASURED_MTU_TAX_PP) <= 2.0,
            )
        )

    for size in SIZES:
        measured = _goodput_bps(_cell(rows, "cx7_400g", size, 1, CALIBRATION_MTU))
        reference = _law_bps(size, GOODPUT_BPS["cx5_100g"] * LINK_FACTOR["cx7_400g"])
        checks.append(
            _check(
                "cx7_scaling",
                f"cx7_400g/{size}",
                measured / 1e9,
                reference / 1e9,
                "1 percent",
                _within(measured, reference, 0.01),
            )
        )

    for profile in PROFILES:
        row = _cell(rows, profile, SMALL_MESSAGE_BYTES, 1024, CALIBRATION_MTU)
        rate = _packet_rate_pps(row)
        capped = _goodput_bps(row) < GOODPUT_BPS[profile]
        checks.append(
            _check(
                "pps_ceiling",
                f"{profile}/1 KiB",
                rate / 1e6,
                MESSAGE_RATE_PPS[profile] / 1e6,
                "5 percent",
                _within(rate, MESSAGE_RATE_PPS[profile], 0.05) and capped,
            )
        )

    for profile in PROFILES:
        for size in SIZES:
            curve = [
                _goodput_bps(_cell(rows, profile, size, depth, CALIBRATION_MTU))
                for depth in DEPTHS
            ]
            monotone = curve[0] <= curve[1] <= curve[2]
            checks.append(
                _check(
                    "depth_monotone",
                    f"{profile}/{size}",
                    curve[2] / 1e9,
                    curve[0] / 1e9,
                    "non-decreasing",
                    monotone,
                )
            )

    for profile in PROFILES:
        worst = max(
            _goodput_bps(row) for row in rows if row["profile"] == profile
        )
        checks.append(
            _check(
                "ceiling_bound",
                profile,
                worst / 1e9,
                GOODPUT_BPS[profile] / 1e9,
                "at or below the ceiling",
                worst <= GOODPUT_BPS[profile] * 1.001,
            )
        )

    return checks, fits


def _render_csv(rows: list[dict[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _curve_rows(rows: list[dict[str, int | str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        out.append(
            {
                "profile": row["profile"],
                "size_bytes": row["size_bytes"],
                "depth": row["depth"],
                "mtu_bytes": row["mtu_bytes"],
                "messages": row["messages"],
                "packets": row["packets"],
                "payload_bytes": row["payload_bytes"],
                "wire_bytes": row["wire_bytes"],
                "last_completion_ps": row["last_completion_ps"],
                "goodput_gbps": f"{_goodput_bps(row) / 1e9:.6f}",
                "packet_rate_mpps": f"{_packet_rate_pps(row) / 1e6:.6f}",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the measured CSVs differ from the tracked ones",
    )
    arguments = parser.parse_args()

    probe = _build(arguments.build_dir.resolve())
    data_root = os.environ.get("SIMLLM_DATA_ROOT")
    raw_dir = Path(data_root) / "rnic_cmodel_v1" if data_root else None
    if raw_dir is None:
        print("SIMLLM_DATA_ROOT is unset; raw per-cell output is not archived")

    rows = _run_grid(probe, raw_dir)
    replay_profile, replay_size, replay_depth, replay_mtu = REPLAY_CELL
    trace_prefix = (raw_dir / "replay") if raw_dir is not None else None
    replay_row = _probe(
        probe, replay_profile, replay_size, replay_depth, replay_mtu, True, trace_prefix
    )

    problems = _fatal_guards(rows, replay_row)
    if problems:
        print("FATAL GUARD: the run is void, not scored")
        for problem in problems:
            print(f"  {problem}")
        raise SystemExit(2)

    checks, fits = _evaluate(rows)
    curves = _render_csv(_curve_rows(rows))
    summary = _render_csv(checks)

    if arguments.check:
        for path, rendered in ((CURVES, curves), (SUMMARY, summary)):
            if not path.is_file() or path.read_bytes() != rendered:
                raise SystemExit(f"measured rows differ from tracked {path}")
        print("tracked results match the measured rows")
    else:
        CURVES.write_bytes(curves)
        SUMMARY.write_bytes(summary)
        print(f"wrote {len(curves.splitlines()) - 1} cells and {len(checks)} checks")

    for profile, (t_eff, goodput) in fits.items():
        print(
            f"fit {profile}: T_eff {t_eff * 1e6:.3f} us, C {goodput / 1e9:.3f} Gb/s "
            f"(frozen {T_EFF_S * 1e6:.2f} us, {GOODPUT_BPS[profile] / 1e9:.1f} Gb/s)"
        )
    failed = [check for check in checks if check["verdict"] != "PASS"]
    print(f"checks: {len(checks) - len(failed)}/{len(checks)} PASS")
    for check in failed:
        print(f"  FAIL {check['check']} {check['cell']}: {check['measured']} vs {check['reference']}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
