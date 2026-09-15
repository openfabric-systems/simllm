"""Checks for the B200 NVLink envelope benchmark lane and its scorer.

The benchmark half runs the lane in ``--mock`` mode under
``torch.distributed.run`` with two ranks, so the control flow of all three
lanes is exercised without a GPU. The scorer half feeds a synthetic stage 1
result built from the frozen profile predictions and a planted slope, then
asserts that the fits recover the planted constants and that the holdout logic
passes and fails when it should.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
STUDY = REPOSITORY / "examples/b200_nvlink_envelope_v1"
BENCH_PATH = STUDY / "bench_nvlink.py"
SCORER_PATH = STUDY / "score_expectations.py"
EXPECTATIONS_PATH = STUDY / "expectations.json"

MOCK_PAYLOAD_CAP_BYTES = 1_048_576
MOCK_TIMED_ITERATIONS = 2

# The planted constants of the synthetic result.
PLANTED_P1 = {"0->1": (5.0e-6, 700_000_000_000.0), "1->0": (5.5e-6, 650_000_000_000.0)}
PLANTED_P2 = (12.0e-6, 600_000_000_000.0)
PLANTED_INTERCEPT_PS = 10_722_112
PLANTED_BANDWIDTH = 70_027_079_100


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The scorer defines dataclasses under postponed annotations, which resolve
    # against the module entry rather than the local namespace.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _scorer():
    return _load("b200_nvlink_envelope_scorer", SCORER_PATH)


def _expectations() -> dict:
    return json.loads(EXPECTATIONS_PATH.read_text())


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


@pytest.fixture(scope="module")
def mock_result(tmp_path_factory) -> dict:
    pytest.importorskip("torch")
    output = tmp_path_factory.mktemp("b200_mock")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(BENCH_PATH),
            "--stage",
            "1",
            "--output",
            str(output),
            "--mock",
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    path = output / "stage1_result.json"
    assert path.is_file(), completed.stdout
    return json.loads(path.read_text())


def test_mock_header_records_the_cap_and_the_frozen_sweep(mock_result: dict) -> None:
    frozen = _expectations()["lanes"]
    header = mock_result["header"]
    assert mock_result["stage"] == 1
    assert header["hostname"] == "redacted"
    assert header["mock"] is True
    assert header["backend"] == "gloo"
    assert header["mock_payload_cap_bytes"] == MOCK_PAYLOAD_CAP_BYTES
    assert header["timed_iterations"] == MOCK_TIMED_ITERATIONS
    assert header["frozen_payload_bytes"] == frozen["payload_bytes"]
    assert header["payload_bytes"] == [
        size for size in frozen["payload_bytes"] if size <= MOCK_PAYLOAD_CAP_BYTES
    ]
    assert header["started_utc"] and header["finished_utc"]
    for key in ("torch_version", "cuda_version", "driver_version", "visible_device_count"):
        assert key in header


def test_mock_lanes_cover_every_frozen_cell(mock_result: dict) -> None:
    sweep = mock_result["header"]["payload_bytes"]

    p1_directions = {"0->1", "1->0", "bidirectional"}
    for direction in p1_directions:
        rows = [row for row in mock_result["p1"] if row["direction"] == direction]
        assert [row["requested_bytes"] for row in rows] == sweep

    for cell in ("unidirectional", "bidirectional"):
        rows = [row for row in mock_result["p2"] if row["cell"] == cell]
        assert [row["requested_bytes"] for row in rows] == sweep
    assert all(row["token_bytes"] == 8 for row in mock_result["p2"] if "token_bytes" in row)

    for op in ("all_reduce", "all_gather", "reduce_scatter"):
        rows = [row for row in mock_result["p3"] if row["op"] == op and row["width"] == 2]
        assert [row["requested_bytes"] for row in rows] == sweep
    assert all(
        row["all_reduce_correct"] is True
        for row in mock_result["p3"]
        if row["op"] == "all_reduce" and row["status"] == "measured"
    )

    for lane in ("p1", "p2", "p3"):
        for row in mock_result[lane]:
            if row["status"] != "measured":
                continue
            assert row["bytes"] <= MOCK_PAYLOAD_CAP_BYTES
            assert row["iterations"] == MOCK_TIMED_ITERATIONS
            assert row["time_ns"] > 0.0
            assert row["bytes_per_second"] > 0.0
    for row in mock_result["p3"]:
        if row["status"] != "measured":
            continue
        assert row["algbw_bytes_per_second"] == row["bytes_per_second"]
        expected = 2.0 * (row["width"] - 1) / row["width"]
        if row["op"] != "all_reduce":
            expected = (row["width"] - 1) / row["width"]
        assert row["busbw_factor"] == pytest.approx(expected)
        assert row["busbw_bytes_per_second"] == pytest.approx(
            row["algbw_bytes_per_second"] * expected
        )


def _copy_row(cell: str, direction: str, size: int, seconds: float) -> dict:
    pairs = [[0, 1], [1, 0]] if cell == "bidirectional" else [[0, 1]]
    if direction == "1->0":
        pairs = [[1, 0]]
    return {
        "cell": cell,
        "pairs": pairs,
        "src": pairs[0][0],
        "dst": pairs[0][1],
        "direction": direction,
        "requested_bytes": size,
        "bytes": size,
        "iterations": 20,
        "status": "measured",
        "time_ns": seconds * 1e9,
        "bytes_per_second": size / seconds,
        "aggregate_bytes": size * len(pairs),
        "aggregate_bytes_per_second": size * len(pairs) / seconds,
    }


def _synthetic_result(expectations: dict) -> dict:
    sweep = expectations["lanes"]["payload_bytes"]
    p1: list[dict] = []
    for direction, (alpha, beta) in PLANTED_P1.items():
        for size in sweep:
            p1.append(_copy_row("unidirectional", direction, size, alpha + size / beta))
    for size in sweep:
        alpha, beta = PLANTED_P1["0->1"]
        p1.append(_copy_row("bidirectional", "bidirectional", size, alpha + size / beta))

    alpha, beta = PLANTED_P2
    p2 = [
        {
            "cell": "unidirectional",
            "src": 0,
            "dst": 1,
            "direction": "0->1",
            "requested_bytes": size,
            "bytes": size,
            "iterations": 20,
            "status": "measured",
            "time_ns": (alpha + size / beta) * 1e9,
            "bytes_per_second": size / (alpha + size / beta),
            "token_bytes": 8,
        }
        for size in sweep
    ]

    p3 = []
    for size in sweep:
        picoseconds = PLANTED_INTERCEPT_PS + _ceil_div(size * 10**12, PLANTED_BANDWIDTH)
        seconds = picoseconds * 1e-12
        p3.append(
            {
                "op": "all_reduce",
                "width": 2,
                "requested_bytes": size,
                "bytes": size,
                "iterations": 20,
                "status": "measured",
                "time_ns": picoseconds / 1_000.0,
                "bytes_per_second": size / seconds,
                "algbw_bytes_per_second": size / seconds,
                "busbw_bytes_per_second": size / seconds,
                "busbw_factor": 1.0,
                "all_reduce_correct": True,
            }
        )
    return {
        "schema": "simllm-b200-nvlink-envelope-result-v1",
        "study": "b200_nvlink_envelope_v1",
        "stage": 1,
        "header": {
            "hostname": "redacted",
            "started_utc": "2026-09-15T00:00:00Z",
            "finished_utc": "2026-09-15T00:10:00Z",
            "torch_version": "2.8.0+cu128",
            "cuda_version": "12.8",
            "nccl_version": [2, 27, 3],
            "driver_version": "595.91.07",
            "visible_device_count": 2,
            "mock": False,
        },
        "peer_access": [{"src": 0, "dst": 1, "can_access": True}],
        "p1": p1,
        "p2": p2,
        "p3": p3,
    }


def _score(result: dict, tmp_path: Path) -> dict:
    scorer = _scorer()
    measurements = tmp_path / "measurements"
    measurements.mkdir(parents=True, exist_ok=True)
    (measurements / "stage1_result.json").write_text(json.dumps(result))
    exit_code = scorer.main(["--measurements", str(measurements)])
    assert exit_code == 0
    return json.loads((measurements / "scored.json").read_text())


def _outcome(report: dict, ident: str) -> dict:
    for entry in report["expectations"]:
        if entry["id"] == ident:
            return entry
    raise AssertionError(f"no outcome {ident} in {[e['id'] for e in report['expectations']]}")


def test_scorer_recovers_the_planted_constants(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    expectations = _expectations()
    report = _score(_synthetic_result(expectations), tmp_path)

    assert report["void"] is False
    assert report["fatal"] == []

    for direction, (alpha, beta) in PLANTED_P1.items():
        fit = _outcome(report, f"E2-{direction}-fit")
        assert fit["passed"] is True
        assert fit["observed"]["beta_bytes_per_second"] == pytest.approx(beta, rel=1e-6)
        assert fit["observed"]["alpha_us"] == pytest.approx(alpha * 1e6, abs=1e-3)
        assert fit["observed"]["r_squared"] > 0.999999
        holdout = _outcome(report, f"E2-{direction}-holdout")
        assert holdout["passed"] is True
        assert holdout["observed"]["holdout_error_fraction"] < 1e-9

    e3 = _outcome(report, "E3-fit")
    assert e3["passed"] is True
    assert e3["observed"]["beta_bytes_per_second"] == pytest.approx(PLANTED_P2[1], rel=1e-6)
    assert _outcome(report, "E3-holdout")["passed"] is True

    before = _outcome(report, "E4-w2")
    assert before["passed"] is True
    for row in before["observed"]["rows"]:
        assert abs(row["delta_ps"]) < 1.0

    refit = _outcome(report, "E5-fit")["observed"]
    assert refit["widths"] == [2]
    assert refit["bandwidth_bytes_per_second"] == pytest.approx(PLANTED_BANDWIDTH, rel=1e-6)
    assert refit["intercept_ps"]["2"] == pytest.approx(PLANTED_INTERCEPT_PS, abs=100)
    assert _outcome(report, "E5-holdout-w2")["passed"] is True

    profile = report["proposed_profile"]
    assert profile["profile_id"] == "b200-nccl-2.27-local-firstparty-v1"
    assert profile["participant_latency_ps"] == [[2, refit["intercept_ps"]["2"]]]
    assert profile["bandwidth_bytes_per_second"] == refit["bandwidth_bytes_per_second"]
    assert profile["provenance"]["evidence_class"] == "calibrated"
    low, high = profile["provenance"]["participant_latency_band_ps"][0][1:]
    assert low <= profile["participant_latency_ps"][0][1] <= high

    assert report["scored_total"] == 4
    assert report["scored_passed"] == 4
    assert _outcome(report, "E6")["passed"] is not None
    assert _outcome(report, "E7")["passed"] is None
    assert report["identity_guards_pending"]
    for ident in report["identity_guards_pending"]:
        assert _outcome(report, ident)["passed"] is None


def test_scorer_fails_a_holdout_that_leaves_its_band(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    expectations = _expectations()

    moved = _synthetic_result(expectations)
    for row in moved["p3"]:
        if row["requested_bytes"] == 4_096:
            row["time_ns"] += 10_000.0
    report = _score(moved, tmp_path / "refit")
    assert _outcome(report, "E5-holdout-w2")["passed"] is False
    assert _outcome(report, "E4-w2")["passed"] is False
    assert report["scored_passed"] == 3

    stretched = _synthetic_result(expectations)
    for row in stretched["p1"]:
        if row["requested_bytes"] == 67_108_864 and row["direction"] == "0->1":
            row["time_ns"] *= 1.4
    report = _score(stretched, tmp_path / "asymptote")
    holdout = _outcome(report, "E2-0->1-holdout")
    assert holdout["passed"] is False
    assert holdout["observed"]["holdout_error_fraction"] > 0.10
    assert _outcome(report, "E2-1->0-holdout")["passed"] is True


def test_the_benchmark_fails_closed_without_peer_access(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    bench = _load("b200_nvlink_envelope_bench", BENCH_PATH)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda src, dst: False)
    with pytest.raises(SystemExit) as raised:
        bench.check_peer_access([0, 1], False)
    assert raised.value.code == bench.EXIT_NO_PEER_ACCESS

    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda src, dst: True)
    rows = bench.check_peer_access([0, 1], False)
    assert [(row["src"], row["dst"]) for row in rows] == [(0, 1), (1, 0)]


def test_the_benchmark_constants_match_the_freeze() -> None:
    pytest.importorskip("torch")
    bench = _load("b200_nvlink_envelope_bench", BENCH_PATH)
    assert bench.PAYLOAD_BYTES == tuple(_expectations()["lanes"]["payload_bytes"])
    assert bench.busbw_factor("all_reduce", 8) == pytest.approx(1.75)
    assert bench.busbw_factor("all_gather", 8) == pytest.approx(0.875)
    assert bench._iterations(64 * 1024 * 1024, False) == (5, 20)
    assert bench._iterations(128 * 1024 * 1024, False) == (5, 10)
    assert math.isfinite(bench._rate(1024, 1e-6))


class _FakeCudaDevices:
    """Track the process device the way ``torch.cuda`` does, without a GPU."""

    def __init__(self, current: int) -> None:
        self.current = current
        self.entered: list[int] = []
        self.set_device_calls: list[int] = []


class _FakeDeviceContext:
    def __init__(self, state: _FakeCudaDevices, index: int) -> None:
        self.state = state
        self.index = index
        self.previous = index

    def __enter__(self):
        self.previous = self.state.current
        self.state.current = self.index
        self.state.entered.append(self.index)
        return self

    def __exit__(self, *exc_info) -> bool:
        self.state.current = self.previous
        return False


class _FakeStream:
    def __init__(self, device=None) -> None:
        self.device = device


class _FakeStreamContext:
    def __init__(self, stream) -> None:
        self.stream = stream

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


class _FakeEvent:
    def __init__(self, enable_timing: bool = False) -> None:
        self.enable_timing = enable_timing

    def record(self, stream=None) -> None:
        return None

    def elapsed_time(self, other) -> float:
        return 1.0


def _fake_cuda(monkeypatch, bench, state: _FakeCudaDevices) -> None:
    torch = bench.torch
    monkeypatch.setattr(bench, "_device", lambda index, mock: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda index: _FakeDeviceContext(state, index))
    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "stream", _FakeStreamContext)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda index=None: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: state.current)

    def _set_device(index: int) -> None:
        state.current = index
        state.set_device_calls.append(index)

    monkeypatch.setattr(torch.cuda, "set_device", _set_device)


def test_lane_p1_leaves_the_process_on_its_own_device(monkeypatch) -> None:
    """The stage 1 hang: P1 walked both devices and left the process on one.

    A communicator created afterwards binds to the current device, so rank 0
    and rank 1 claimed the same GPU and the next init never completed.
    """

    pytest.importorskip("torch")
    bench = _load("b200_nvlink_envelope_bench_devices", BENCH_PATH)
    monkeypatch.setattr(bench, "PAYLOAD_BYTES", (8, 1_024))
    state = _FakeCudaDevices(current=0)
    _fake_cuda(monkeypatch, bench, state)

    rows = bench.run_lane_p1(1, [0, 1], False, 0)

    assert rows, "the lane produced no rows"
    assert state.entered, "the lane never entered a device context"
    assert 1 in state.entered, "the lane never drove the second device"
    assert state.current == 0, "lane P1 left the process on another rank's device"
    assert state.set_device_calls == [0], "only the deliberate restore may set the device"


def test_lane_p1_restores_the_device_even_when_a_cell_raises(monkeypatch) -> None:
    pytest.importorskip("torch")
    bench = _load("b200_nvlink_envelope_bench_raises", BENCH_PATH)
    monkeypatch.setattr(bench, "PAYLOAD_BYTES", (8,))
    state = _FakeCudaDevices(current=0)
    _fake_cuda(monkeypatch, bench, state)

    def _boom(*args, **kwargs):
        state.current = 1
        raise RuntimeError("cell failed")

    monkeypatch.setattr(bench, "_time_transfers", _boom)
    with pytest.raises(RuntimeError):
        bench.run_lane_p1(1, [0, 1], False, 0)
    assert state.current == 0


def test_the_cuda_path_pins_the_communicator_device(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    bench = _load("b200_nvlink_envelope_bench_init", BENCH_PATH)
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        bench.dist,
        "init_process_group",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    bench._init_process_group(False, 1, 120)
    args, kwargs = calls[-1]
    assert args[0] == "nccl"
    assert kwargs["device_id"] == torch.device("cuda", 1)
    assert kwargs["timeout"].total_seconds() == 120

    bench._init_process_group(True, 0, 45)
    args, kwargs = calls[-1]
    assert args[0] == "gloo"
    assert kwargs["timeout"].total_seconds() == 45
    assert "device_id" not in kwargs


def test_the_default_timeout_is_bounded() -> None:
    pytest.importorskip("torch")
    bench = _load("b200_nvlink_envelope_bench_timeout", BENCH_PATH)
    assert 0 < bench.PG_TIMEOUT_SECONDS <= 600
    args = bench.parse_args(["--stage", "1", "--output", "out"])
    assert args.pg_timeout_seconds == bench.PG_TIMEOUT_SECONDS
