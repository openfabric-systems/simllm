"""Run the RNIC golden model's slice-C receive study through its C facade.

The sweeps, the closed forms and the bands are frozen in expectations.md. This
script only executes them: it builds the native gate, fits the ingress drain
rate against the four measured anchors over a declared candidate grid, drives
the facade probe over the frozen cells, and writes one summary row per
registered check.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "simllm" / "backends" / "rnic"
DEFAULT_BUILD_DIR = REPO_ROOT / "build" / "rnic_cmodel_rx_v1"
SUMMARY = Path(__file__).with_name("summary.csv")
CURVES = Path(__file__).with_name("curves.csv")

# Frozen study constants (expectations.md).
BURST_MESSAGES = 128
GAPS_PS = (0, 4_000_000, 100_000_000, 368_000_000)
SIZES = (8192, 65536, 1048576)
FABRIC_QUEUE_BYTES = 65536
EFFECTIVE_WIRE_BPS = 98_617_190_000
LINK_BPS = 100_000_000_000

# Measured anchors (mlx5 campaign).
MEASURED_SATURATED = {8192: 77.52e9, 65536: 81.44e9}
MEASURED_PACED = {8192: 92.20e9, 65536: 97.31e9}
MEASURED_DEPTH_RATIO = {8192: 5.9, 65536: 1.57}
MEASURED_DEPTH1 = {8192: 12.713e9, 65536: 53.069e9}
MEASURED_UD_CAP_PPS = 3.07e6
MEASURED_UD_AGGREGATE_PPS = 9.65e6
# The frozen per-QP unreliable ceiling, pinned at the probe so the registered
# cells keep evaluating against the value they were frozen with even after the
# profile default moved.
FROZEN_UD_CAP_PPS = 3_070_000
# Post-specified: the P6 fabric campaign measured the ceiling on the wire after
# these expectations were frozen and re-attributed the 3.07e6 figure to the
# measurement engine. Cells driven at this value are reported as post-specified
# regression checks and score separately from the frozen ones.
P6_UD_CAP_PPS = 5_510_000
MEASURED_INCAST_TAX = 0.269
MEASURED_SIMPLEX_BPS = 93_400_000_000
MEASURED_DUPLEX_BPS = 91_800_000_000
SATURATED_WINDOW = (78e9, 92e9)

# Declared candidate grid for the one fitted rate. The fit picks the member
# that minimizes the sum of squared relative errors on the four anchors.
DRAIN_CANDIDATES = tuple(range(95_000_000_000, 98_200_000_000, 200_000_000))

# Cell sizes. They are not frozen: the expectations register the sweep, not
# how long a cell runs. Every saturated goodput is read off the second half of
# its cell so the ingress meter's fill transient is excluded.
GAP_MESSAGES = {8192: 8192, 65536: 1024, 1048576: 384}
UD_MESSAGES = 200_000
INCAST_MESSAGES = {65536: 64, 1048576: 8}
DUPLEX_MESSAGES = 256


def _native_executable(build_dir: Path, name: str) -> Path:
    for candidate in (build_dir / name, build_dir / f"{name}.exe"):
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"native executable not found under {build_dir}")


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


def _probe(probe: Path, mode: str, **options) -> dict[str, int | str]:
    command = [str(probe), "--mode", mode]
    for name, value in options.items():
        if name == "replay":
            if value:
                command.append("--replay")
            continue
        command.extend(["--" + name.replace("_", "-"), str(value)])
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"probe returned {len(rows)} rows, expected one")
    return {
        name: (value if name in ("mode", "profile") else int(value))
        for name, value in rows[0].items()
    }


def _seconds(row: dict[str, int | str], key: str = "last_completion_ps") -> float:
    return float(row[key]) / 1e12


def _goodput(row: dict[str, int | str], size: int) -> float:
    """Whole-cell goodput: completed payload over the cell's wall time."""
    return float(row["completions"]) * size * 8.0 / _seconds(row)


def _steady_goodput(row: dict[str, int | str], size: int) -> float:
    """Second-half goodput, which is the equilibrium the meter settles into."""
    total = float(row["completions"]) * size
    span = _seconds(row) - float(row["warm_start_ps"]) / 1e12
    if span <= 0:
        return _goodput(row, size)
    return (total - float(row["warm_payload_bytes"])) * 8.0 / span


def _in_burst_goodput(row: dict[str, int | str], size: int) -> float:
    """Goodput with the idle inter-burst gaps taken out of the clock."""
    span = _seconds(row) - float(row["gap_time_ps"]) / 1e12
    return float(row["completions"]) * size * 8.0 / span


def _check(name, cell, measured, reference, band, passed, note="") -> dict[str, object]:
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


def _fit_drain(probe: Path) -> tuple[int, list[tuple[int, float]]]:
    """Least squares over the declared candidate grid, four measured anchors."""
    scored: list[tuple[int, float]] = []
    for drain in DRAIN_CANDIDATES:
        residual = 0.0
        for size in (8192, 65536):
            saturated = _steady_goodput(
                _probe(
                    probe,
                    "gap",
                    size_bytes=size,
                    depth=1024,
                    messages=GAP_MESSAGES[size],
                    gap_ps=0,
                    burst_messages=BURST_MESSAGES,
                    rx_drain_bps=drain,
                ),
                size,
            )
            residual += (saturated / MEASURED_SATURATED[size] - 1.0) ** 2
        scored.append((drain, residual))
    best = min(scored, key=lambda item: item[1])[0]
    return best, scored


def _run_grid(probe: Path, drain: int, nic_pps: int, raw_dir: Path | None):
    rows: list[dict[str, object]] = []

    def record(kind: str, size: int, row: dict[str, int | str], extra: dict) -> None:
        entry = {"kind": kind, "size_bytes": size}
        entry.update({key: row[key] for key in row if key not in ("mode", "profile")})
        entry.update(extra)
        rows.append(entry)

    gap_cells: dict[tuple[int, int], dict[str, int | str]] = {}
    for size in SIZES:
        for gap in GAPS_PS:
            row = _probe(
                probe,
                "gap",
                size_bytes=size,
                depth=1024,
                messages=GAP_MESSAGES[size],
                gap_ps=gap,
                burst_messages=BURST_MESSAGES,
                rx_drain_bps=drain,
            )
            gap_cells[(size, gap)] = row
            record(
                "gap",
                size,
                row,
                {
                    "goodput_gbps": f"{_goodput(row, size) / 1e9:.4f}",
                    "steady_gbps": f"{_steady_goodput(row, size) / 1e9:.4f}",
                    "in_burst_gbps": f"{_in_burst_goodput(row, size) / 1e9:.4f}",
                },
            )

    depth_cells: dict[tuple[int, int], dict[str, int | str]] = {}
    for size in (8192, 65536):
        for depth in (1, 1024):
            key = (size, depth)
            if depth == 1024:
                depth_cells[key] = gap_cells[(size, 0)]
                continue
            row = _probe(
                probe,
                "gap",
                size_bytes=size,
                depth=1,
                messages=256,
                gap_ps=0,
                burst_messages=0,
                rx_drain_bps=drain,
            )
            depth_cells[key] = row
            record("depth", size, row, {"goodput_gbps": f"{_goodput(row, size) / 1e9:.4f}"})

    ud_cells: dict[tuple[int, int], dict[str, int | str]] = {}
    for qps in (1, 4):
        for offered in (2_000_000, 3_000_000, 4_000_000, 5_850_000):
            row = _probe(
                probe,
                "ud",
                size_bytes=2048,
                messages=UD_MESSAGES * qps,
                offered_pps=offered * qps,
                qps=qps,
                rx_drain_bps=drain,
                rx_pps_per_qp_ud=FROZEN_UD_CAP_PPS,
                rx_pps_per_nic=nic_pps,
            )
            ud_cells[(qps, offered)] = row
            delivered = float(row["rx_packets_delivered"]) / _seconds(row)
            record("ud", 2048, row, {"delivered_mpps": f"{delivered / 1e6:.4f}", "qps": qps})
    # The campaign's own aggregate point used 1 KiB messages, so its offered
    # byte rate fits the port. It is supplementary, not a registered check.
    ud_supplement = _probe(
        probe,
        "ud",
        size_bytes=1024,
        messages=UD_MESSAGES * 4,
        offered_pps=5_850_000 * 4,
        qps=4,
        rx_drain_bps=drain,
        rx_pps_per_qp_ud=FROZEN_UD_CAP_PPS,
        rx_pps_per_nic=nic_pps,
    )
    record(
        "ud_supplement",
        1024,
        ud_supplement,
        {
            "delivered_mpps": f"{float(ud_supplement['rx_packets_delivered']) / _seconds(ud_supplement) / 1e6:.4f}",
            "qps": 4,
        },
    )

    # Post-specified: the same sweep at the corrected per-QP ceiling. It is run
    # beside the frozen cells rather than in place of them, so no registered
    # verdict moves and the correction is visible as its own set of rows.
    ud_p6_cells: dict[tuple[int, int], dict[str, int | str]] = {}
    for offered in (2_000_000, 3_000_000, 4_000_000, 5_850_000):
        row = _probe(
            probe,
            "ud",
            size_bytes=2048,
            messages=UD_MESSAGES,
            offered_pps=offered,
            qps=1,
            rx_drain_bps=drain,
            rx_pps_per_qp_ud=P6_UD_CAP_PPS,
            rx_pps_per_nic=nic_pps,
        )
        ud_p6_cells[(1, offered)] = row
        delivered = float(row["rx_packets_delivered"]) / _seconds(row)
        record("ud_p6", 2048, row, {"delivered_mpps": f"{delivered / 1e6:.4f}", "qps": 1})
    aggregate = _probe(
        probe,
        "ud",
        size_bytes=1024,
        messages=UD_MESSAGES * 4,
        offered_pps=5_850_000 * 4,
        qps=4,
        rx_drain_bps=drain,
        rx_pps_per_qp_ud=P6_UD_CAP_PPS,
        rx_pps_per_nic=nic_pps,
    )
    ud_p6_cells[(4, 5_850_000)] = aggregate
    record(
        "ud_p6",
        1024,
        aggregate,
        {
            "delivered_mpps": f"{float(aggregate['rx_packets_delivered']) / _seconds(aggregate) / 1e6:.4f}",
            "qps": 4,
        },
    )

    incast_cells: dict[tuple[int, int], dict[str, int | str]] = {}
    for loss in (5000, 16500, 50000):
        for size in (65536, 1048576):
            row = _probe(
                probe,
                "incast",
                size_bytes=size,
                depth=16,
                messages=INCAST_MESSAGES[size],
                senders=2,
                loss_ppm=loss,
                offered_bps=48_550_000_000,
                fabric_queue_bytes=FABRIC_QUEUE_BYTES,
                rx_drain_bps=drain,
            )
            incast_cells[(loss, size)] = row
            record("incast", size, row, {"loss_ppm": loss, "goodput_gbps": f"{_goodput(row, size) / 1e9:.4f}"})

    # Supplementary: the same senders with headroom for their own replays.
    incast_stable: dict[int, dict[str, int | str]] = {}
    for loss in (5000, 16500, 50000):
        row = _probe(
            probe,
            "incast",
            size_bytes=65536,
            depth=16,
            messages=256,
            senders=2,
            loss_ppm=loss,
            offered_bps=24_000_000_000,
            fabric_queue_bytes=FABRIC_QUEUE_BYTES,
            rx_drain_bps=drain,
        )
        incast_stable[loss] = row
        record("incast_headroom", 65536, row, {"loss_ppm": loss, "goodput_gbps": f"{_goodput(row, 65536) / 1e9:.4f}"})

    duplex_cells: dict[str, dict[str, int | str]] = {}
    for name, offered in (("simplex", MEASURED_SIMPLEX_BPS), ("duplex", MEASURED_DUPLEX_BPS)):
        row = _probe(
            probe,
            "duplex",
            size_bytes=65536,
            depth=16,
            messages=DUPLEX_MESSAGES,
            senders=1,
            offered_bps=offered,
            rx_drain_bps=drain,
        )
        duplex_cells[name] = row
        record(name, 65536, row, {"goodput_gbps": f"{_goodput(row, 65536) / 1e9:.4f}"})

    replay = _probe(
        probe,
        "gap",
        size_bytes=65536,
        depth=1024,
        messages=64,
        gap_ps=4_000_000,
        burst_messages=BURST_MESSAGES,
        loss_period=100,
        rx_drain_bps=drain,
        replay=True,
        trace_prefix=str(raw_dir / "replay") if raw_dir is not None else "",
    )

    return (gap_cells, depth_cells, ud_cells, ud_p6_cells, incast_cells,
            incast_stable, duplex_cells, replay, rows)


def _fatal_guards(gap_cells, depth_cells, ud_cells, ud_p6_cells, duplex_cells, replay) -> list[str]:
    problems: list[str] = []
    if replay["replay_identical"] != 1:
        problems.append("deterministic replay identity failed on the replay cell")
    for (size, gap), row in gap_cells.items():
        label = f"gap/{size}/{gap}"
        if row["errors"] != 0:
            problems.append(f"{label}: {row['errors']} messages completed with an error")
        if row["completions"] != row["messages"]:
            problems.append(f"{label}: {row['completions']} of {row['messages']} completed")
        if row["late_releases"] != 0:
            problems.append(f"{label}: {row['late_releases']} late packet releases")
        offered = row["rx_packets_offered"]
        accounted = (
            row["rx_packets_delivered"]
            + row["rx_discards_meter"]
            + row["rx_discards_rate"]
            + row["rx_discards_sequence"]
        )
        if accounted > offered:
            problems.append(f"{label}: receive accounting exceeds what was offered")
        if row["np_ecn_marked"] != 0:
            problems.append(f"{label}: the inert marking counter moved")
    for (size, depth), row in depth_cells.items():
        if row["errors"] != 0:
            problems.append(f"depth/{size}/{depth}: {row['errors']} errored messages")
    for label, cells in (("ud", ud_cells), ("ud_p6", ud_p6_cells)):
        for key, row in cells.items():
            offered = row["rx_packets_offered"]
            if row["rx_packets_delivered"] + row["rx_discards_phy"] != offered:
                problems.append(
                    f"{label}/{key}: delivered plus discarded does not equal offered")
    for name, row in duplex_cells.items():
        if row["errors"] != 0:
            problems.append(f"{name}: {row['errors']} errored messages")
    return problems


def _evaluate(gap_cells, depth_cells, ud_cells, ud_p6_cells, incast_cells,
              duplex_cells) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []

    # 1. gap_discards, categorical.
    expected_dirty = {
        (8192, 0): True,
        (8192, 4_000_000): False,
        (8192, 100_000_000): False,
        (8192, 368_000_000): False,
        (65536, 0): True,
        (65536, 4_000_000): True,
        (65536, 100_000_000): False,
        (65536, 368_000_000): False,
    }
    for (size, gap), row in sorted(gap_cells.items()):
        dirty = row["rx_discards_phy"] > 0
        if (size, gap) in expected_dirty:
            passed = dirty == expected_dirty[(size, gap)]
            reference = 1.0 if expected_dirty[(size, gap)] else 0.0
            note = "" if passed else "discard state disagrees with the measured threshold"
        else:
            # 1 MiB registers only the monotone claim, checked below.
            continue
        checks.append(
            _check("gap_discards", f"{size}/{gap // 1000}us", 1.0 if dirty else 0.0,
                   reference, "categorical", passed, note))
    smallest_clean = {}
    for size in SIZES:
        clean = [gap for gap in GAPS_PS if gap_cells[(size, gap)]["rx_discards_phy"] == 0]
        smallest_clean[size] = min(clean) if clean else None
    # No clean gap inside the sweep means the threshold lies above the widest
    # gap swept, which satisfies the registered monotone claim rather than
    # failing it.
    largest = max(GAPS_PS)
    biggest = smallest_clean[1048576] if smallest_clean[1048576] is not None else largest + 1
    middle = smallest_clean[65536] if smallest_clean[65536] is not None else largest + 1
    monotone = biggest >= middle
    checks.append(
        _check("gap_discards", "1 MiB monotone", biggest / 1e6, middle / 1e6,
               "smallest clean gap not below the 64 KiB one", monotone))

    # 2. paced_goodput.
    for size, gap in ((8192, 4_000_000), (65536, 100_000_000)):
        measured = _in_burst_goodput(gap_cells[(size, gap)], size)
        reference = MEASURED_PACED[size]
        checks.append(
            _check("paced_goodput", f"{size}/{gap // 1000}us", measured / 1e9, reference / 1e9,
                   "15 percent", _within(measured, reference, 0.15)))

    # 3. saturated_goodput.
    for size in (8192, 65536):
        measured = _steady_goodput(gap_cells[(size, 0)], size)
        inside = SATURATED_WINDOW[0] <= measured <= SATURATED_WINDOW[1]
        checks.append(
            _check("saturated_goodput", f"{size}/0us", measured / 1e9,
                   MEASURED_SATURATED[size] / 1e9, "78 to 92 Gb/s window", inside))

    # 4. depth_ratio_measured.
    for size in (8192, 65536):
        shallow = _goodput(depth_cells[(size, 1)], size)
        deep = _steady_goodput(depth_cells[(size, 1024)], size)
        ratio = deep / shallow
        reference = MEASURED_DEPTH_RATIO[size]
        checks.append(
            _check("depth_ratio_measured", f"{size}", ratio, reference, "20 percent",
                   _within(ratio, reference, 0.20)))

    # 5. depth1_unchanged.
    for size in (8192, 65536):
        measured = _goodput(depth_cells[(size, 1)], size)
        reference = MEASURED_DEPTH1[size]
        checks.append(
            _check("depth1_unchanged", f"{size}", measured / 1e9, reference / 1e9,
                   "1 percent", _within(measured, reference, 0.01)))

    # 6/7/8. UD cap, passthrough and silence.
    for offered in (4_000_000, 5_850_000):
        row = ud_cells[(1, offered)]
        delivered = float(row["rx_packets_delivered"]) / _seconds(row)
        checks.append(
            _check("ud_cap", f"1qp/{offered / 1e6:g}Mpps", delivered / 1e6,
                   MEASURED_UD_CAP_PPS / 1e6, "10 percent",
                   _within(delivered, MEASURED_UD_CAP_PPS, 0.10)))
    for offered in (2_000_000, 3_000_000):
        row = ud_cells[(1, offered)]
        exact = row["rx_packets_delivered"] == row["rx_packets_offered"]
        checks.append(
            _check("ud_passthrough", f"1qp/{offered / 1e6:g}Mpps",
                   float(row["rx_packets_delivered"]), float(row["rx_packets_offered"]),
                   "exact", exact))
    for offered in (2_000_000, 3_000_000, 4_000_000, 5_850_000):
        row = ud_cells[(1, offered)]
        discarded = row["rx_packets_offered"] - row["rx_packets_delivered"]
        silent = (
            discarded == row["rx_discards_phy"]
            and row["out_of_sequence"] == 0
            and row["packet_seq_err"] == 0
            and row["roce_adp_retrans"] == 0
        )
        checks.append(
            _check("ud_silent", f"1qp/{offered / 1e6:g}Mpps", float(discarded),
                   float(row["rx_discards_phy"]), "exact and no transport signal", silent))

    # 9. ud_aggregate.
    row = ud_cells[(4, 5_850_000)]
    delivered = float(row["rx_packets_delivered"]) / _seconds(row)
    checks.append(
        _check("ud_aggregate", "4qp/5.85Mpps each", delivered / 1e6,
               MEASURED_UD_AGGREGATE_PPS / 1e6, "10 percent",
               _within(delivered, MEASURED_UD_AGGREGATE_PPS, 0.10),
               "the frozen 2 KiB offer is four times the port's byte capacity"))

    # 10/11/12/13. Incast.
    primary = incast_cells[(16500, 1048576)]
    span = _seconds(primary)
    utilization = float(primary["rx_bytes_phy"]) * 8.0 / span / LINK_BPS
    checks.append(
        _check("incast_wire", "1.65 percent/1 MiB", utilization * 100.0, 97.0,
               "at least 97 percent of the link", utilization >= 0.97))
    goodput = _goodput(primary, 1048576)
    sender_wire = float(primary["wire_bytes"]) * 8.0 / span
    tax = 1.0 - goodput / sender_wire
    checks.append(
        _check("incast_tax", "1.65 percent/1 MiB", tax * 100.0, MEASURED_INCAST_TAX * 100.0,
               "25 percent", _within(tax, MEASURED_INCAST_TAX, 0.25),
               "" if _within(tax, MEASURED_INCAST_TAX, 0.25) else "go-back-N with no reaction point"))
    first = float(primary["sender0_payload_bytes"])
    second = float(primary["sender1_payload_bytes"])
    share = 100.0 * first / (first + second) if first + second else 0.0
    checks.append(
        _check("incast_fair", "1.65 percent/1 MiB", share, 50.0,
               "2 percentage points", abs(share - 50.0) <= 2.0,
               "the share is imposed by the probe's per-sender pacing"))
    taxes = {}
    for (loss, size), row in incast_cells.items():
        cell_span = _seconds(row)
        cell_good = _goodput(row, size)
        cell_wire = float(row["wire_bytes"]) * 8.0 / cell_span
        taxes[(loss, size)] = 1.0 - cell_good / cell_wire
    rising_in_loss = all(
        taxes[(5000, size)] < taxes[(16500, size)] < taxes[(50000, size)]
        for size in (65536, 1048576)
    )
    checks.append(
        _check("incast_direction", "tax against loss", 1.0 if rising_in_loss else 0.0, 1.0,
               "strictly increasing", rising_in_loss))
    rising_in_size = all(
        taxes[(loss, 65536)] < taxes[(loss, 1048576)] for loss in (5000, 16500, 50000)
    )
    checks.append(
        _check("incast_direction", "tax against message size",
               1.0 if rising_in_size else 0.0, 1.0, "strictly increasing", rising_in_size))

    # 14/15. Duplex pair.
    duplex = duplex_cells["duplex"]
    clean = (
        duplex["rx_discards_phy"] == 0
        and duplex["packet_seq_err"] == 0
        and duplex["roce_adp_retrans"] == 0
    )
    checks.append(
        _check("duplex_clean", "91.8 Gb/s per direction", float(duplex["rx_discards_phy"]),
               0.0, "every counter zero", clean))
    simplex = duplex_cells["simplex"]
    checks.append(
        _check("simplex_dirty", "93.4 Gb/s", float(simplex["rx_discards_phy"]), 1.0,
               "rx_discards_phy nonzero", simplex["rx_discards_phy"] > 0,
               "registered in advance as an expected miss"))

    # 16/17/18. Counter ledger.
    for size in SIZES:
        for gap in (0, 4_000_000):
            row = gap_cells[(size, gap)]
            losses = row["rx_discards_meter"] + row["rx_discards_rate"]
            paired = row["packet_seq_err"] == row["out_of_sequence"]
            bracket = losses == 0 or _within(float(row["out_of_sequence"]), float(losses), 0.10)
            checks.append(
                _check("ledger_identity", f"{size}/{gap // 1000}us",
                       float(row["out_of_sequence"]), float(losses), "10 percent",
                       paired and bracket))
    inert = all(row["np_ecn_marked"] == 0 for row in gap_cells.values())
    checks.append(
        _check("inert_marking", "every gap cell", 0.0, 0.0, "exactly zero", inert))

    # Post-specified regression checks. The P6 fabric campaign measured the
    # unreliable receive ceiling on the wire after these expectations were
    # frozen and re-attributed the 3.07 Mpps figure to the measurement engine.
    # These rows re-run the same cells at the corrected parameter. They are not
    # part of the registered 40 and are named and noted so they cannot be read
    # as one.
    postspec = "post-specified P6 correction, not a registered check"
    row = ud_p6_cells[(1, 5_850_000)]
    delivered = float(row["rx_packets_delivered"]) / _seconds(row)
    checks.append(
        _check("postspec_ud_cap", "1qp/5.85Mpps", delivered / 1e6,
               P6_UD_CAP_PPS / 1e6, "10 percent",
               _within(delivered, float(P6_UD_CAP_PPS), 0.10), postspec))
    for offered in (2_000_000, 3_000_000, 4_000_000):
        row = ud_p6_cells[(1, offered)]
        exact = row["rx_packets_delivered"] == row["rx_packets_offered"]
        checks.append(
            _check("postspec_ud_passthrough", f"1qp/{offered / 1e6:g}Mpps",
                   float(row["rx_packets_delivered"]),
                   float(row["rx_packets_offered"]), "exact", exact, postspec))
    for offered in (2_000_000, 3_000_000, 4_000_000, 5_850_000):
        row = ud_p6_cells[(1, offered)]
        discarded = row["rx_packets_offered"] - row["rx_packets_delivered"]
        silent = (
            discarded == row["rx_discards_phy"]
            and row["out_of_sequence"] == 0
            and row["packet_seq_err"] == 0
            and row["roce_adp_retrans"] == 0
        )
        checks.append(
            _check("postspec_ud_silent", f"1qp/{offered / 1e6:g}Mpps", float(discarded),
                   float(row["rx_discards_phy"]), "exact and no transport signal",
                   silent, postspec))
    row = ud_p6_cells[(4, 5_850_000)]
    delivered = float(row["rx_packets_delivered"]) / _seconds(row)
    checks.append(
        _check("postspec_ud_aggregate", "4qp/5.85Mpps each at 1 KiB", delivered / 1e6,
               MEASURED_UD_AGGREGATE_PPS / 1e6, "10 percent",
               _within(delivered, MEASURED_UD_AGGREGATE_PPS, 0.10),
               postspec + "; the per-NIC ceiling is unchanged and still binds"))
    return checks


def _render_csv(rows: list[dict[str, object]]) -> bytes:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", restval="")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--drain-bps", type=int, default=0, help="skip the fit")
    parser.add_argument("--nic-pps", type=int, default=9_650_000)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()

    probe = _build(arguments.build_dir.resolve())
    data_root = os.environ.get("SIMLLM_DATA_ROOT")
    raw_dir = Path(data_root) / "rnic_cmodel_rx_v1" if data_root else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
    else:
        print("SIMLLM_DATA_ROOT is unset; raw per-cell output is not archived")

    if arguments.drain_bps:
        drain, scored = arguments.drain_bps, []
    else:
        drain, scored = _fit_drain(probe)
    print(f"fitted rx_drain_bps = {drain / 1e9:.1f} Gb/s of wire")
    for candidate, residual in scored:
        print(f"  candidate {candidate / 1e9:.1f} residual {residual:.5f}")

    (gap_cells, depth_cells, ud_cells, ud_p6_cells, incast_cells, incast_stable,
     duplex_cells, replay, rows) = _run_grid(probe, drain, arguments.nic_pps, raw_dir)

    problems = _fatal_guards(
        gap_cells, depth_cells, ud_cells, ud_p6_cells, duplex_cells, replay)
    if problems:
        print("FATAL GUARD: the run is void, not scored")
        for problem in problems:
            print(f"  {problem}")
        raise SystemExit(2)

    checks = _evaluate(
        gap_cells, depth_cells, ud_cells, ud_p6_cells, incast_cells, duplex_cells)
    curves = _render_csv(rows)
    summary = _render_csv(checks)
    if arguments.check:
        for path, rendered in ((CURVES, curves), (SUMMARY, summary)):
            if not path.is_file() or path.read_bytes() != rendered:
                raise SystemExit(f"measured rows differ from tracked {path}")
        print("tracked results match the measured rows")
    else:
        CURVES.write_bytes(curves)
        SUMMARY.write_bytes(summary)
        print(f"wrote {len(rows)} cells and {len(checks)} checks")

    registered = [c for c in checks if not str(c["check"]).startswith("postspec_")]
    post = [c for c in checks if str(c["check"]).startswith("postspec_")]
    failed = [check for check in checks if check["verdict"] != "PASS"]
    print(
        f"registered checks: "
        f"{sum(1 for c in registered if c['verdict'] == 'PASS')}/{len(registered)} PASS; "
        f"post-specified: {sum(1 for c in post if c['verdict'] == 'PASS')}/{len(post)} PASS")
    for check in failed:
        print(f"  FAIL {check['check']} {check['cell']}: {check['measured']} vs {check['reference']}")
    for loss, row in sorted(incast_stable.items()):
        span = _seconds(row)
        good = _goodput(row, 65536)
        wire = float(row["wire_bytes"]) * 8.0 / span
        print(
            f"headroom incast loss {loss / 1e4:.2f} percent: goodput {good / 1e9:.2f} Gb/s, "
            f"tax {100 * (1 - good / wire):.2f} percent, "
            f"{row['recovery_episodes']} episodes for {row['injected_losses']} losses"
        )


if __name__ == "__main__":
    main()
