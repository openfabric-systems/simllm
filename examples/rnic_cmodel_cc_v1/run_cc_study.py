"""Run the RNIC golden model's slice-D congestion-control study.

The sweeps, the closed forms and the bands are frozen in expectations.md. This
script only executes them: it builds the native gate, fits the notification and
reaction parameters over the declared candidate grids, drives the facade probe
over the frozen cells, and writes one summary row per registered check.

Every cell is one probe process, so cells are independent and the pool below
changes wall time and nothing else.
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "simllm" / "backends" / "rnic"
DEFAULT_BUILD_DIR = REPO_ROOT / "build" / "rnic_cmodel_cc_v1"
SUMMARY = Path(__file__).with_name("summary.csv")
CURVES = Path(__file__).with_name("curves.csv")

# Frozen fabric and hardware constants (expectations.md).
LINK_BPS = 100_000_000_000
EFFECTIVE_WIRE_BPS = 98_617_190_000
EGRESS_BUFFER_BYTES = 5_200_000
HALF_EGRESS_BUFFER_BYTES = 2_600_000
INGRESS_BYTES = 262_016
MTU = 4096
WIRE_HEADER = 64

# Measured anchors (mlx5 campaign P5a and P6).
MEASURED_CNP_PER_QP = 283.0
MEASURED_INCAST_GOODPUT_BPS = 73.89e9
MEASURED_INCAST_TAX = (0.21, 0.27)
MEASURED_CUT_MS = (3.0, 39.0)
MEASURED_FAIR_MS = (5.0, 2300.0)
MEASURED_RECOVERY_MS = (337.0, 557.0)
MEASURED_SENDER_SLOPE_BPS_PER_MS = 0.1e9
MEASURED_FANOUT_FRACTION = 0.978

# Declared candidate grids (expectations.md).
THRESHOLD_CANDIDATES = (INGRESS_BYTES // 4, INGRESS_BYTES // 2, 3 * INGRESS_BYTES // 4)
CNP_INTERVAL_CANDIDATES = (2_000_000_000, 3_530_000_000, 6_000_000_000)
ALPHA_INIT_CANDIDATES = (250_000, 350_000, 500_000)
ALPHA_GAIN_PPM = 3906
ALPHA_UPDATE_PS = 50_000_000
RATE_STEP_CANDIDATES = (22_500_000, 25_000_000, 27_500_000)
RATE_INTERVAL_PS = 1_000_000_000
RATE_FLOOR_BPS = 1_000_000_000

# Cell sizes. They are not frozen: the expectations register the sweep, not how
# long a cell runs. Every steady quantity is read off the second half of its
# cell so the startup transient is excluded, exactly as slice C reads its
# equilibria.
GRID_RUN_PS = 250_000_000_000
INCAST_RUN_PS = 400_000_000_000
CONTROL_RUN_PS = 200_000_000_000
SAMPLE_PS = 5_000_000_000
DYNAMICS_START_PS = 200_000_000_000
DYNAMICS_STOP_PS = 1_600_000_000_000
DYNAMICS_RUN_PS = 2_400_000_000_000
DYNAMICS_SAMPLE_PS = 1_000_000_000
IDENTITY_MESSAGES = 8
QPS = 4
# The dynamics cell needs a competitor that stops when its window closes. A
# send queue of 1024 messages of 1 MiB is a gigabyte of outstanding work per
# queue pair, which keeps draining long after the sender has stopped posting
# and hides the recovery leg entirely. Thirty-two messages is still far above
# the bandwidth-delay product at this rate, so the cell saturates exactly as it
# did and the stop becomes observable inside the tail.
DYNAMICS_DEPTH = 32
# The post-specified alpha sensitivity only has to show how long a cut takes,
# which happens in the first tenth of the overlap, so it runs on a shorter
# window than the registered dynamics cell and reports only the cut time.
ALPHA_START_PS = 200_000_000_000
ALPHA_STOP_PS = 700_000_000_000
ALPHA_RUN_PS = 900_000_000_000


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


class Samples:
    """The per-sample trace of one cell, and the rates read off it."""

    def __init__(self, path: Path, hosts: int):
        self.rows = list(csv.DictReader(path.open()))
        self.hosts = hosts

    def _rate(self, first: dict, last: dict, key: str, bits: bool = True) -> float:
        span = (int(last["t_ps"]) - int(first["t_ps"])) / 1e12
        if span <= 0:
            return 0.0
        delta = int(last[key]) - int(first[key])
        return delta * (8.0 if bits else 1.0) / span

    def window(self, start_ps: int, stop_ps: int) -> tuple[dict, dict] | None:
        inside = [r for r in self.rows if start_ps <= int(r["t_ps"]) <= stop_ps]
        if len(inside) < 2:
            return None
        return inside[0], inside[-1]

    def rate(self, key: str, start_ps: int, stop_ps: int, bits: bool = True) -> float:
        pair = self.window(start_ps, stop_ps)
        if pair is None:
            return 0.0
        return self._rate(pair[0], pair[1], key, bits)

    def host_rate(self, host: int, start_ps: int, stop_ps: int) -> float:
        return self.rate(f"host{host}_wire_bytes", start_ps, stop_ps)

    def series(self, key: str, bits: bool = True) -> list[tuple[int, float]]:
        """Instantaneous rate at each sample, from the sample before it."""
        out: list[tuple[int, float]] = []
        for earlier, later in itertools.pairwise(self.rows):
            out.append((int(later["t_ps"]), self._rate(earlier, later, key, bits)))
        return out

    def smoothed(self, key: str, window: int = 5) -> list[tuple[int, float]]:
        """The campaign's instrument: a boxcar over the per-sample rates."""
        raw = self.series(key)
        out: list[tuple[int, float]] = []
        for index in range(len(raw)):
            lo = max(0, index - window + 1)
            chunk = [value for _, value in raw[lo : index + 1]]
            out.append((raw[index][0], sum(chunk) / len(chunk)))
        return out


def _cell(probe: Path, scratch: Path, name: str, mode: str, **options):
    path = scratch / f"{name}.csv"
    row = _probe(probe, mode, samples_path=str(path), **options)
    return row, path


def _base(threshold: int, interval: int, step: int, alpha_init: int) -> dict:
    return {
        "size_bytes": 1048576,
        "depth": 1024,
        "messages": 100_000_000,
        "congestion_control": 1,
        "queue_pairs_per_port": QPS,
        "qps": QPS,
        "egress_buffer_bytes": EGRESS_BUFFER_BYTES,
        "np_threshold_bytes": threshold,
        "cnp_min_interval_ps": interval,
        "alpha_init_ppm": alpha_init,
        "alpha_gain_ppm": ALPHA_GAIN_PPM,
        "alpha_update_ps": ALPHA_UPDATE_PS,
        "rate_step_bps": step,
        "rate_interval_ps": RATE_INTERVAL_PS,
        "rate_floor_bps": RATE_FLOOR_BPS,
    }


def _steady(samples: Samples, run_ps: int, hosts: int) -> dict[str, float]:
    """Rates over the second half of a cell, which is its equilibrium."""
    start = run_ps // 2
    goodput = samples.rate("rx_payload_bytes", start, run_ps)
    wire = samples.rate("rx_wire_bytes", start, run_ps)
    offered = sum(samples.host_rate(h, start, run_ps) for h in range(hosts))
    cnps = samples.rate("np_cnp_sent", start, run_ps, bits=False)
    return {
        "steady_goodput_bps": goodput,
        "steady_wire_bps": wire,
        "steady_offered_bps": offered,
        # The campaign's own definition: what the receiver's application got
        # over what the receiver's wire carried.
        "steady_tax": 1.0 - goodput / wire if wire > 0 else 0.0,
        "steady_cnp_per_s": cnps,
    }


def _run_np_grid(probe: Path, scratch: Path, jobs: int) -> tuple[dict, list[dict]]:
    """Sweep (a): the notification threshold against the limiter interval."""
    cells = [
        (threshold, interval)
        for threshold in THRESHOLD_CANDIDATES
        for interval in CNP_INTERVAL_CANDIDATES
    ]

    def one(cell):
        threshold, interval = cell
        name = f"np_{threshold}_{interval}"
        row, path = _cell(
            probe,
            scratch,
            name,
            "cc",
            senders=2,
            run_ps=GRID_RUN_PS,
            sample_interval_ps=SAMPLE_PS,
            **_base(threshold, interval, RATE_STEP_CANDIDATES[1], ALPHA_INIT_CANDIDATES[1]),
        )
        samples = Samples(path, 2)
        derived = _steady(samples, GRID_RUN_PS, 2)
        row.update(derived)
        row["kind"] = "np_grid"
        row["np_threshold_bytes"] = threshold
        row["cnp_min_interval_ps"] = interval
        row["cnp_per_qp_per_s"] = derived["steady_cnp_per_s"] / (2 * QPS)
        return row

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(one, cells))
    best = min(rows, key=lambda r: abs(r["cnp_per_qp_per_s"] - MEASURED_CNP_PER_QP))
    return best, rows


def _run_rp_grid(probe: Path, scratch: Path, jobs: int, threshold: int, interval: int):
    """Sweep (c): the reaction point's alpha start against its additive step."""
    cells = [
        (alpha_init, step)
        for alpha_init in ALPHA_INIT_CANDIDATES
        for step in RATE_STEP_CANDIDATES
    ]

    def one(cell):
        alpha_init, step = cell
        name = f"rp_{alpha_init}_{step}"
        row, path = _cell(
            probe,
            scratch,
            name,
            "dynamics",
            senders=2,
            run_ps=DYNAMICS_RUN_PS,
            sample_interval_ps=DYNAMICS_SAMPLE_PS,
            competitor_start_ps=DYNAMICS_START_PS,
            competitor_stop_ps=DYNAMICS_STOP_PS,
            **(
                _base(threshold, interval, step, alpha_init)
                | {"depth": DYNAMICS_DEPTH}
            ),
        )
        samples = Samples(path, 2)
        row.update(_dynamics_metrics(samples))
        row["kind"] = "rp_grid"
        row["alpha_init_ppm"] = alpha_init
        row["rate_step_bps"] = step
        return row

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(one, cells))

    def score(row):
        penalty = 0.0
        for key, band in (
            ("cut30_ms", MEASURED_CUT_MS),
            ("fair_ms", MEASURED_FAIR_MS),
            ("recovery_ms", MEASURED_RECOVERY_MS),
        ):
            value = row[key]
            if value < 0:
                penalty += 100.0
            elif value < band[0]:
                penalty += (band[0] - value) / band[0]
            elif value > band[1]:
                penalty += (value - band[1]) / band[1]
        return penalty

    best = min(rows, key=score)
    return best, rows


def _dynamics_metrics(
    samples: Samples,
    start_ps: int = DYNAMICS_START_PS,
    stop_ps: int = DYNAMICS_STOP_PS,
) -> dict[str, float]:
    """The campaign's own readings, taken off the incumbent's smoothed rate."""
    trace = samples.smoothed("host0_wire_bytes")
    pre = [v for t, v in trace if start_ps // 2 <= t < start_ps]
    pre_rate = sum(pre) / len(pre) if pre else 0.0
    overlap = [(t, v) for t, v in trace if start_ps <= t < stop_ps]
    after = [(t, v) for t, v in trace if t >= stop_ps]

    cut_ms = -1.0
    for t, v in overlap:
        if pre_rate > 0 and v <= 0.7 * pre_rate:
            cut_ms = (t - start_ps) / 1e9
            break

    fair_ms = -1.0
    competitor = dict(samples.smoothed("host1_wire_bytes"))
    for t, v in overlap:
        other = competitor.get(t, 0.0)
        total = v + other
        if total > 0 and abs(v / total - 0.5) <= 0.05:
            fair_ms = (t - start_ps) / 1e9
            break

    recovery_ms = -1.0
    for t, v in after:
        if pre_rate > 0 and v >= 0.95 * pre_rate:
            recovery_ms = (t - stop_ps) / 1e9
            break

    # The additive slope, fitted over the recovery leg the way the campaign
    # fitted it: a least-squares line through the smoothed rate between the
    # competitor stopping and the rate reaching its pre-competitor level.
    leg = [
        (t, v)
        for t, v in after
        if recovery_ms < 0 or t - stop_ps <= recovery_ms * 1e9
    ]
    slope = 0.0
    if len(leg) >= 3:
        xs = [(t - stop_ps) / 1e9 for t, _ in leg]
        ys = [v for _, v in leg]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denominator = sum((x - mean_x) ** 2 for x in xs)
        if denominator > 0:
            slope = sum(
                (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
            ) / denominator

    steady_start = start_ps + (stop_ps - start_ps) // 2
    incumbent = samples.host_rate(0, steady_start, stop_ps)
    challenger = samples.host_rate(1, steady_start, stop_ps)
    total = incumbent + challenger
    goodput = samples.rate("rx_payload_bytes", steady_start, stop_ps)
    received = samples.rate("rx_wire_bytes", steady_start, stop_ps)
    return {
        "pre_bps": pre_rate,
        "cut30_ms": cut_ms,
        "fair_ms": fair_ms,
        "recovery_ms": recovery_ms,
        "slope_bps_per_ms": slope,
        "overlap_share": incumbent / total if total > 0 else 0.0,
        "overlap_offered_bps": total,
        "overlap_wire_bps": received,
        "overlap_goodput_bps": goodput,
    }


def _run_incast(probe: Path, scratch: Path, jobs: int, fit: dict) -> list[dict]:
    """Sweep (d): buffer against message size against sender count."""
    cells = [
        (buffer_bytes, size, senders)
        for buffer_bytes in (EGRESS_BUFFER_BYTES, HALF_EGRESS_BUFFER_BYTES)
        for size in (65536, 1048576)
        for senders in (2, 3)
    ]

    def one(cell):
        buffer_bytes, size, senders = cell
        name = f"incast_{buffer_bytes}_{size}_{senders}"
        options = _base(
            fit["threshold"], fit["interval"], fit["step"], fit["alpha_init"]
        )
        options["size_bytes"] = size
        options["egress_buffer_bytes"] = buffer_bytes
        row, path = _cell(
            probe,
            scratch,
            name,
            "cc",
            senders=senders,
            run_ps=INCAST_RUN_PS,
            sample_interval_ps=SAMPLE_PS,
            **options,
        )
        samples = Samples(path, senders)
        row.update(_steady(samples, INCAST_RUN_PS, senders))
        row["kind"] = "incast"
        row["egress_buffer_bytes"] = buffer_bytes
        row["cnp_per_qp_per_s"] = row["steady_cnp_per_s"] / (senders * QPS)
        shares = [
            row[f"sender{index}_payload_bytes"] for index in range(senders)
        ]
        total = sum(shares) or 1
        row["max_share"] = max(shares) / total
        row["min_share"] = min(shares) / total
        lost = row["egress_dropped"] + row["rx_discards_phy"]
        row["packets_lost"] = lost
        row["burst_ratio"] = lost / row["packet_seq_err"] if row["packet_seq_err"] else 0.0
        row["loss_rate"] = lost / row["packets_issued"] if row["packets_issued"] else 0.0
        row["amplification"] = (
            row["steady_tax"] / row["loss_rate"] if row["loss_rate"] > 0 else 0.0
        )
        return row

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(one, cells))


def _run_drain_sensitivity(probe: Path, scratch: Path, jobs: int, fit: dict):
    """Post-specified. Not registered in expectations.md and run only after the
    frozen incast verdict was recorded.

    The notification point sits at the receiver's ingress meter, so the control
    loop settles where the meter's drain rate is, not where the switch's egress
    rate is. Slice C fitted that drain from the lone-flow anchors. The incast
    measurement is a different anchor for the same latent, and this sweep asks
    what the incast cell does across the range between them, against the alpha
    start that sets how deep the opening cuts are.
    """
    cells = [
        (drain, alpha_init)
        for drain in (96_600_000_000, 98_600_000_000, 99_400_000_000)
        for alpha_init in (250_000, 350_000, 500_000)
    ]

    def one(cell):
        drain, alpha_init = cell
        options = _base(
            fit["threshold"], fit["interval"], fit["step"], alpha_init
        )
        options["rx_drain_bps"] = drain
        row, path = _cell(
            probe,
            scratch,
            f"drain_{drain}_{alpha_init}",
            "cc",
            senders=2,
            run_ps=INCAST_RUN_PS,
            sample_interval_ps=SAMPLE_PS,
            **options,
        )
        samples = Samples(path, 2)
        row.update(_steady(samples, INCAST_RUN_PS, 2))
        row["kind"] = "drain_sensitivity"
        row["rx_drain_bps"] = drain
        row["alpha_init_ppm"] = alpha_init
        row["cnp_per_qp_per_s"] = row["steady_cnp_per_s"] / (2 * QPS)
        shares = [row["sender0_payload_bytes"], row["sender1_payload_bytes"]]
        total = sum(shares) or 1
        row["max_share"] = max(shares) / total
        return row

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(one, cells))


def _run_alpha_sensitivity(probe: Path, scratch: Path, jobs: int, fit: dict):
    """Post-specified. Not registered in expectations.md and run only after the
    frozen reaction-point verdict was recorded.

    One alpha controls two things the measurement wants pulled apart: how deep
    a queue pair cuts on a notification, and where the loop settles. This sweep
    asks what the gain and the decay interval do to the cut time, over the
    candidate grids the frozen latent table already declared.
    """
    cells = [
        (gain, update)
        for gain in (3906, 15625, 62500)
        for update in (50_000_000, 100_000_000, 200_000_000)
    ]

    def one(cell):
        gain, update = cell
        options = _base(
            fit["threshold"], fit["interval"], fit["step"], fit["alpha_init"]
        )
        options["depth"] = DYNAMICS_DEPTH
        options["alpha_gain_ppm"] = gain
        options["alpha_update_ps"] = update
        row, path = _cell(
            probe,
            scratch,
            f"alpha_{gain}_{update}",
            "dynamics",
            senders=2,
            run_ps=ALPHA_RUN_PS,
            sample_interval_ps=DYNAMICS_SAMPLE_PS,
            competitor_start_ps=ALPHA_START_PS,
            competitor_stop_ps=ALPHA_STOP_PS,
            **options,
        )
        samples = Samples(path, 2)
        row.update(
            _dynamics_metrics(samples, ALPHA_START_PS, ALPHA_STOP_PS)
        )
        row["kind"] = "alpha_sensitivity"
        row["alpha_gain_ppm"] = gain
        row["alpha_update_ps"] = update
        return row

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(one, cells))


def _run_lone(probe: Path, scratch: Path, jobs: int, fit: dict) -> list[dict]:
    """Sweep (b): a lone flow paced below saturation raises nothing."""
    cells = [(offered, qps) for offered in (80e9, 90e9) for qps in (1, 4)]

    def one(cell):
        offered, qps = cell
        options = _base(
            fit["threshold"], fit["interval"], fit["step"], fit["alpha_init"]
        )
        options["qps"] = qps
        options["queue_pairs_per_port"] = qps
        options["size_bytes"] = 65536
        row, path = _cell(
            probe,
            scratch,
            f"lone_{int(offered)}_{qps}",
            "lone",
            senders=1,
            offered_bps=int(offered / qps),
            run_ps=CONTROL_RUN_PS,
            sample_interval_ps=SAMPLE_PS,
            **options,
        )
        samples = Samples(path, 1)
        row.update(_steady(samples, CONTROL_RUN_PS, 1))
        row["kind"] = "lone"
        row["offered_target_bps"] = offered
        row["qps"] = qps
        return row

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(one, cells))


def _run_fanout(probe: Path, scratch: Path, jobs: int, fit: dict) -> list[dict]:
    """Sweep (e): one sender into two receivers pays nothing."""
    cells = [(size, qps) for size in (65536, 1048576) for qps in (2, 4)]

    def one(cell):
        size, qps = cell
        options = _base(
            fit["threshold"], fit["interval"], fit["step"], fit["alpha_init"]
        )
        options["size_bytes"] = size
        options["qps"] = qps
        options["queue_pairs_per_port"] = qps
        row, path = _cell(
            probe,
            scratch,
            f"fanout_{size}_{qps}",
            "fanout",
            senders=1,
            receivers=2,
            run_ps=CONTROL_RUN_PS,
            sample_interval_ps=SAMPLE_PS,
            **options,
        )
        samples = Samples(path, 1)
        row.update(_steady(samples, CONTROL_RUN_PS, 1))
        row["kind"] = "fanout"
        row["qps"] = qps
        first = row["rx_payload_bytes"] - row["rx_payload_bytes_receiver1"]
        second = row["rx_payload_bytes_receiver1"]
        total = first + second or 1
        row["receiver_split"] = first / total
        row["delivered_fraction"] = row["steady_wire_bps"] / LINK_BPS
        return row

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(one, cells))


def _run_identity(probe: Path, jobs: int) -> list[dict]:
    """Sweep (f): with the block off the slice-C code path comes back exactly."""
    cells = [(size, senders) for size in (65536, 1048576) for senders in (1, 2, 3)]
    def one(cell):
        size, senders = cell
        common = {
            "size_bytes": size,
            "depth": 1024,
            "messages": IDENTITY_MESSAGES,
            "senders": senders,
            "qps": 1,
            "fabric_queue_bytes": 65536,
        }
        slice_c = _probe(probe, "incast", **common)
        slice_d = _probe(
            probe,
            "cc",
            congestion_control=0,
            egress_buffer_bytes=0,
            queue_pairs_per_port=0,
            **common,
        )
        differences = [
            key
            for key, value in slice_c.items()
            if key not in ("mode",) and slice_d.get(key) != value
        ]
        row = dict(slice_d)
        row["kind"] = "identity"
        row["identity_differences"] = len(differences)
        row["identity_difference_names"] = ";".join(differences)
        return row

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(one, cells))


def _run_collapse(probe: Path, scratch: Path, fit: dict) -> dict:
    """The slice-C behaviour on the slice-D fabric: no reaction point, collapse."""
    options = _base(fit["threshold"], fit["interval"], fit["step"], fit["alpha_init"])
    options["congestion_control"] = 0
    options["messages"] = 6
    row, _ = _cell(
        probe,
        scratch,
        "collapse",
        "cc",
        senders=2,
        **options,
    )
    row["kind"] = "collapse"
    span = row["last_completion_ps"] / 1e12
    row["steady_goodput_bps"] = row["rx_payload_bytes"] * 8 / span if span else 0.0
    offered = row["wire_bytes"] * 8 / span if span else 0.0
    row["steady_offered_bps"] = offered
    row["steady_tax"] = 1.0 - row["steady_goodput_bps"] / offered if offered else 0.0
    return row


def _run_replay(probe: Path, scratch: Path, fit: dict) -> dict:
    options = _base(fit["threshold"], fit["interval"], fit["step"], fit["alpha_init"])
    options["messages"] = 12
    row, _ = _cell(
        probe,
        scratch,
        "replay",
        "cc",
        senders=2,
        replay=True,
        trace_prefix=str(scratch / "replay"),
        **options,
    )
    row["kind"] = "replay"
    return row


def _check(name, cell, measured, reference, band, verdict, note=""):
    return {
        "check": name,
        "cell": cell,
        "measured": measured,
        "reference": reference,
        "band": band,
        "verdict": verdict,
        "note": note,
    }


def _within(value: float, reference: float, fraction: float) -> bool:
    return abs(value - reference) <= abs(reference) * fraction


def _render_csv(rows: list[dict]) -> bytes:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", restval="")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--jobs", type=int, default=9)
    parser.add_argument("--scratch", type=Path, default=None)
    arguments = parser.parse_args()

    probe = _build(arguments.build_dir)
    holder = None
    if arguments.scratch is None:
        holder = tempfile.TemporaryDirectory()
        scratch = Path(holder.name)
    else:
        scratch = arguments.scratch
        scratch.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    checks: list[dict] = []

    best_np, np_rows = _run_np_grid(probe, scratch, arguments.jobs)
    rows.extend(np_rows)
    threshold = best_np["np_threshold_bytes"]
    interval = best_np["cnp_min_interval_ps"]

    best_rp, rp_rows = _run_rp_grid(
        probe, scratch, arguments.jobs, threshold, interval
    )
    rows.extend(rp_rows)
    fit = {
        "threshold": threshold,
        "interval": interval,
        "step": best_rp["rate_step_bps"],
        "alpha_init": best_rp["alpha_init_ppm"],
    }

    lone_rows = _run_lone(probe, scratch, arguments.jobs, fit)
    rows.extend(lone_rows)
    incast_rows = _run_incast(probe, scratch, arguments.jobs, fit)
    rows.extend(incast_rows)
    fanout_rows = _run_fanout(probe, scratch, arguments.jobs, fit)
    rows.extend(fanout_rows)
    identity_rows = _run_identity(probe, arguments.jobs)
    rows.extend(identity_rows)
    collapse_row = _run_collapse(probe, scratch, fit)
    rows.append(collapse_row)
    replay_row = _run_replay(probe, scratch, fit)
    rows.append(replay_row)
    drain_rows = _run_drain_sensitivity(probe, scratch, arguments.jobs, fit)
    rows.extend(drain_rows)
    alpha_rows = _run_alpha_sensitivity(probe, scratch, arguments.jobs, fit)
    rows.extend(alpha_rows)

    # ---- Sweep (a): the notification point -------------------------------
    rate = best_np["cnp_per_qp_per_s"]
    checks.append(
        _check(
            "np_rate",
            f"{threshold}B/{interval / 1e9:.2f}ms",
            f"{rate:.1f}",
            f"{MEASURED_CNP_PER_QP:.0f}",
            "30 percent",
            "PASS" if _within(rate, MEASURED_CNP_PER_QP, 0.30) else "FAIL",
        )
    )
    for threshold_value in THRESHOLD_CANDIDATES:
        series = sorted(
            (r for r in np_rows if r["np_threshold_bytes"] == threshold_value),
            key=lambda r: r["cnp_min_interval_ps"],
        )
        values = [r["cnp_per_qp_per_s"] for r in series]
        monotone = all(
            later <= earlier * 1.001
            for earlier, later in itertools.pairwise(values)
        )
        checks.append(
            _check(
                "np_grid_direction",
                f"threshold {threshold_value}B",
                ";".join(f"{v:.0f}" for v in values),
                "non-increasing in the limiter interval",
                "categorical",
                "PASS" if monotone else "FAIL",
            )
        )

    # ---- Sweep (b): the lone flow ----------------------------------------
    for row in sorted(lone_rows, key=lambda r: (r["offered_target_bps"], r["qps"])):
        cell = f"{row['offered_target_bps'] / 1e9:.0f}Gb/s x{row['qps']}"
        checks.append(
            _check(
                "lone_quiet",
                cell,
                row["np_cnp_sent"],
                0,
                "exact",
                "PASS" if row["np_cnp_sent"] == 0 else "FAIL",
            )
        )
        delivered = row["steady_goodput_bps"]
        checks.append(
            _check(
                "lone_rate_intact",
                cell,
                f"{delivered / 1e9:.2f}",
                f"{row['offered_target_bps'] / 1e9:.2f}",
                "2 percent",
                "PASS" if _within(delivered, row["offered_target_bps"], 0.02) else "FAIL",
            )
        )

    # ---- Sweep (c): the reaction point -----------------------------------
    cell = f"alpha {best_rp['alpha_init_ppm']}ppm step {best_rp['rate_step_bps'] / 1e6:.1f}Mb"
    for name, key, band in (
        ("rp_cut", "cut30_ms", MEASURED_CUT_MS),
        ("rp_fair_time", "fair_ms", MEASURED_FAIR_MS),
        ("rp_recovery", "recovery_ms", MEASURED_RECOVERY_MS),
    ):
        value = best_rp[key]
        ok = value >= 0 and band[0] <= value <= band[1]
        checks.append(
            _check(
                name,
                cell,
                f"{value:.1f}" if value >= 0 else "never",
                f"{band[0]:.0f} to {band[1]:.0f} ms",
                "measured range",
                "PASS" if ok else "FAIL",
            )
        )
    slope = best_rp["slope_bps_per_ms"]
    checks.append(
        _check(
            "rp_slope",
            cell,
            f"{slope / 1e9:.4f}",
            f"{MEASURED_SENDER_SLOPE_BPS_PER_MS / 1e9:.2f}",
            "25 percent",
            "PASS" if _within(slope, MEASURED_SENDER_SLOPE_BPS_PER_MS, 0.25) else "FAIL",
        )
    )
    share = best_rp["overlap_share"]
    checks.append(
        _check(
            "rp_steady",
            cell + " split",
            f"{100 * share:.2f}",
            "50",
            "2 points",
            "PASS" if abs(share - 0.5) <= 0.02 else "FAIL",
        )
    )
    wire = best_rp["overlap_wire_bps"]
    checks.append(
        _check(
            "rp_steady",
            cell + " wire",
            f"{100 * wire / EFFECTIVE_WIRE_BPS:.2f}",
            "97",
            "at least",
            "PASS" if wire >= 0.97 * EFFECTIVE_WIRE_BPS else "FAIL",
        )
    )
    checks.append(
        _check(
            "rp_persistent",
            cell,
            best_rp["rp_rate_persistence_breaks"],
            0,
            "exact",
            "PASS" if best_rp["rp_rate_persistence_breaks"] == 0 else "FAIL",
        )
    )

    # ---- Sweep (d): the incast -------------------------------------------
    primary = next(
        r
        for r in incast_rows
        if r["egress_buffer_bytes"] == EGRESS_BUFFER_BYTES
        and r["size_bytes"] == 1048576
        and r["senders"] == 2
    )
    wire = primary["steady_wire_bps"]
    checks.append(
        _check(
            "incast_wire",
            "5.2MB/1MiB/2",
            f"{100 * wire / EFFECTIVE_WIRE_BPS:.2f}",
            "97",
            "at least",
            "PASS" if wire >= 0.97 * EFFECTIVE_WIRE_BPS else "FAIL",
        )
    )
    goodput = primary["steady_goodput_bps"]
    checks.append(
        _check(
            "incast_goodput",
            "5.2MB/1MiB/2 goodput",
            f"{goodput / 1e9:.2f}",
            f"{MEASURED_INCAST_GOODPUT_BPS / 1e9:.2f}",
            "15 percent",
            "PASS" if _within(goodput, MEASURED_INCAST_GOODPUT_BPS, 0.15) else "FAIL",
        )
    )
    tax = primary["steady_tax"]
    checks.append(
        _check(
            "incast_goodput",
            "5.2MB/1MiB/2 tax",
            f"{100 * tax:.2f}",
            "21 to 27",
            "window",
            "PASS" if MEASURED_INCAST_TAX[0] <= tax <= MEASURED_INCAST_TAX[1] else "FAIL",
        )
    )
    checks.append(
        _check(
            "incast_fair",
            "5.2MB/1MiB/2",
            f"{100 * primary['max_share']:.2f}",
            "50",
            "2 points",
            "PASS" if abs(primary["max_share"] - 0.5) <= 0.02 else "FAIL",
        )
    )
    for row in incast_rows:
        cell = (
            f"{row['egress_buffer_bytes'] // 100000 / 10:.1f}MB/"
            f"{row['size_bytes'] // 1024}KiB/{row['senders']}"
        )
        ok = row["packet_seq_err"] > 0 and row["burst_ratio"] >= 2.0
        checks.append(
            _check(
                "incast_seq_err_bursts",
                cell,
                f"{row['burst_ratio']:.2f}",
                "at least 2",
                "ratio",
                "PASS" if ok else "FAIL",
                "packets lost over packet_seq_err",
            )
        )
    index = {
        (r["egress_buffer_bytes"], r["size_bytes"], r["senders"]): r
        for r in incast_rows
    }
    for buffer_bytes in (EGRESS_BUFFER_BYTES, HALF_EGRESS_BUFFER_BYTES):
        for senders in (2, 3):
            small = index[(buffer_bytes, 65536, senders)]["steady_tax"]
            large = index[(buffer_bytes, 1048576, senders)]["steady_tax"]
            checks.append(
                _check(
                    "incast_direction_size",
                    f"{buffer_bytes // 100000 / 10:.1f}MB/{senders}",
                    f"{100 * small:.2f} then {100 * large:.2f}",
                    "strictly increasing",
                    "categorical",
                    "PASS" if large > small else "FAIL",
                )
            )
        for size in (65536, 1048576):
            two = index[(buffer_bytes, size, 2)]["steady_tax"]
            three = index[(buffer_bytes, size, 3)]["steady_tax"]
            checks.append(
                _check(
                    "incast_direction_senders",
                    f"{buffer_bytes // 100000 / 10:.1f}MB/{size // 1024}KiB",
                    f"{100 * two:.2f} then {100 * three:.2f}",
                    "strictly increasing",
                    "categorical",
                    "PASS" if three > two else "FAIL",
                )
            )
    for size in (65536, 1048576):
        for senders in (2, 3):
            big = index[(EGRESS_BUFFER_BYTES, size, senders)]["steady_tax"]
            small_buffer = index[(HALF_EGRESS_BUFFER_BYTES, size, senders)]["steady_tax"]
            checks.append(
                _check(
                    "incast_buffer_direction",
                    f"{size // 1024}KiB/{senders}",
                    f"{100 * big:.2f} then {100 * small_buffer:.2f}",
                    "not smaller at 2.6 MB",
                    "categorical",
                    "PASS" if small_buffer >= big else "FAIL",
                )
            )

    # ---- Sweep (e): the fan-out control ----------------------------------
    for row in fanout_rows:
        cell = f"{row['size_bytes'] // 1024}KiB x{row['qps']}"
        fraction = row["delivered_fraction"]
        checks.append(
            _check(
                "fanout_rate",
                cell,
                f"{100 * fraction:.2f}",
                f"{100 * MEASURED_FANOUT_FRACTION:.1f}",
                "3 points",
                "PASS" if abs(fraction - MEASURED_FANOUT_FRACTION) <= 0.03 else "FAIL",
            )
        )
        checks.append(
            _check(
                "fanout_split",
                cell,
                f"{100 * row['receiver_split']:.3f}",
                "50",
                "0.5 points",
                "PASS" if abs(row["receiver_split"] - 0.5) <= 0.005 else "FAIL",
            )
        )
        clean = (
            row["egress_dropped"] == 0
            and row["rx_discards_phy"] == 0
            and row["np_cnp_sent"] == 0
        )
        checks.append(
            _check(
                "fanout_clean",
                cell,
                f"{row['egress_dropped']}/{row['rx_discards_phy']}/{row['np_cnp_sent']}",
                "0/0/0",
                "exact",
                "PASS" if clean else "FAIL",
                "egress drops, PHY discards, notifications",
            )
        )

    # ---- Sweep (f): identity off -----------------------------------------
    for row in identity_rows:
        cell = f"{row['size_bytes'] // 1024}KiB/{row['senders']}"
        checks.append(
            _check(
                "identity_off",
                cell,
                row["identity_differences"],
                0,
                "exact",
                "PASS" if row["identity_differences"] == 0 else "FAIL",
                row["identity_difference_names"],
            )
        )
        zeroed = (
            row["np_cnp_sent"] == 0
            and row["rp_cnp_handled"] == 0
            and row["rp_cnp_ignored"] == 0
        )
        checks.append(
            _check(
                "identity_counters",
                cell,
                f"{row['np_cnp_sent']}/{row['rp_cnp_handled']}/{row['rp_cnp_ignored']}",
                "0/0/0",
                "exact",
                "PASS" if zeroed else "FAIL",
            )
        )
    checks.append(
        _check(
            "identity_off",
            "collapse without a reaction point",
            f"{100 * collapse_row['steady_tax']:.2f}",
            "far above the measured 26.9",
            "categorical",
            "PASS" if collapse_row["steady_tax"] > 0.5 else "FAIL",
            "the tax the same transport pays with no rate control",
        )
    )

    # ---- Sweep (g): the counter ledger -----------------------------------
    controlled = [r for r in rows if r.get("congestion_control") == 1]
    ignored = sum(r["rp_cnp_ignored"] for r in controlled)
    checks.append(
        _check(
            "cnp_ignored_zero",
            f"{len(controlled)} cells",
            ignored,
            0,
            "exact",
            "PASS" if ignored == 0 else "FAIL",
        )
    )
    marked = sum(r["np_ecn_marked"] for r in rows)
    notified = sum(r["np_cnp_sent"] for r in controlled)
    checks.append(
        _check(
            "inert_marking",
            f"{len(rows)} cells",
            marked,
            0,
            "exact",
            "PASS" if marked == 0 and notified > 0 else "FAIL",
            f"{notified} notifications generated with the marking counter inert",
        )
    )
    ledger = [
        r for r in controlled if r["np_cnp_sent"] != r["rp_cnp_handled"]
    ]
    checks.append(
        _check(
            "cnp_ledger",
            f"{len(controlled)} cells",
            len(ledger),
            0,
            "exact",
            "PASS" if not ledger else "FAIL",
            "cells whose sender and receiver notification counts disagree",
        )
    )

    # ---- Fatal guards ----------------------------------------------------
    guards = []
    guards.append(("deterministic replay", replay_row["replay_identical"] == 1))
    switched = [r for r in rows if r.get("egress_buffer_bytes", 0) != 0]
    conserved = all(
        r["wire_bytes"] == r["egress_offered_bytes"]
        and r["rx_bytes_phy"] <= r["egress_admitted_bytes"]
        for r in switched
    )
    guards.append(("byte conservation", conserved))
    completions = all(
        r["completions"] * r["size_bytes"] <= r["rx_payload_bytes"] for r in rows
    )
    guards.append(("no completion before delivery", completions))
    guards.append(("pacing integrity", all(r["late_releases"] == 0 for r in rows)))
    guards.append(
        (
            "rate bounds",
            all(
                r["rp_min_rate_bps"] >= RATE_FLOOR_BPS
                for r in controlled
                if r["rp_min_rate_bps"] > 0
            ),
        )
    )
    guards.append(
        (
            "egress-queue conservation",
            all(
                r["egress_offered_bytes"]
                == r["egress_admitted_bytes"] + r["egress_dropped_bytes"]
                for r in rows
            ),
        )
    )
    guards.append(
        (
            "identity off",
            all(r["identity_differences"] == 0 for r in identity_rows),
        )
    )
    for name, ok in guards:
        checks.append(
            _check(
                "fatal_guard",
                name,
                "held" if ok else "violated",
                "held",
                "guard",
                "PASS" if ok else "VOID",
            )
        )

    # ---- Post-specified: the meter's drain rate under fan-in ---------------
    # Registered nowhere, run after the frozen incast verdict above was
    # recorded, and reported as a sensitivity rather than as a score.
    for row in sorted(
        drain_rows, key=lambda r: (r["rx_drain_bps"], r["alpha_init_ppm"])
    ):
        cell = (
            f"drain {row['rx_drain_bps'] / 1e9:.1f} alpha "
            f"{row['alpha_init_ppm']}ppm"
        )
        inside = (
            MEASURED_INCAST_TAX[0] <= row["steady_tax"] <= MEASURED_INCAST_TAX[1]
            and _within(row["steady_goodput_bps"], MEASURED_INCAST_GOODPUT_BPS, 0.15)
            and abs(row["max_share"] - 0.5) <= 0.02
            and row["steady_offered_bps"] >= 0.97 * EFFECTIVE_WIRE_BPS
        )
        checks.append(
            _check(
                "post_incast_drain",
                cell,
                f"tax {100 * row['steady_tax']:.2f} goodput "
                f"{row['steady_goodput_bps'] / 1e9:.2f} share "
                f"{100 * row['max_share']:.2f} cnp/qp "
                f"{row['cnp_per_qp_per_s']:.0f}",
                "tax 21 to 27, goodput 73.89 within 15 percent, share 50 plus "
                "or minus 2",
                "post-specified",
                "INFO",
                "all four incast bars met" if inside else "at least one bar missed",
            )
        )

    for row in sorted(
        alpha_rows, key=lambda r: (r["alpha_gain_ppm"], r["alpha_update_ps"])
    ):
        cell = (
            f"gain {row['alpha_gain_ppm']}ppm update "
            f"{row['alpha_update_ps'] / 1e6:.0f}us"
        )
        checks.append(
            _check(
                "post_rp_alpha",
                cell,
                f"cut {row['cut30_ms']:.1f} fair {row['fair_ms']:.1f} recovery "
                f"{row['recovery_ms']:.1f} split "
                f"{100 * row['overlap_share']:.1f}",
                "cut 3 to 39, fair 5 to 2300, recovery 337 to 557 ms",
                "post-specified",
                "INFO",
            )
        )

    checks.append(
        _check(
            "fit",
            "notification threshold",
            threshold,
            "candidate grid",
            "fitted",
            "INFO",
            f"{THRESHOLD_CANDIDATES}",
        )
    )
    checks.append(
        _check(
            "fit",
            "notification interval",
            interval,
            "candidate grid",
            "fitted",
            "INFO",
            f"{CNP_INTERVAL_CANDIDATES}",
        )
    )
    checks.append(
        _check(
            "fit",
            "alpha start",
            fit["alpha_init"],
            "candidate grid",
            "fitted",
            "INFO",
            f"{ALPHA_INIT_CANDIDATES}",
        )
    )
    checks.append(
        _check(
            "fit",
            "additive step per queue pair",
            fit["step"],
            "candidate grid",
            "fitted",
            "INFO",
            f"{RATE_STEP_CANDIDATES}",
        )
    )

    CURVES.write_bytes(_render_csv(rows))
    SUMMARY.write_bytes(_render_csv(checks))
    failures = [c for c in checks if c["verdict"] == "FAIL"]
    voids = [c for c in checks if c["verdict"] == "VOID"]
    informational = [c for c in checks if c["verdict"] == "INFO"]
    scored = len(checks) - len(informational)
    for entry in checks:
        if entry["verdict"] in ("FAIL", "VOID"):
            print(
                f"  {entry['verdict']} {entry['check']} {entry['cell']}: "
                f"{entry['measured']} vs {entry['reference']}"
            )
    print(
        f"slice D: {scored - len(failures) - len(voids)} of {scored} scored "
        f"checks passed, {len(voids)} guards voided, "
        f"{len(informational)} informational rows"
    )
    if holder is not None:
        holder.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
