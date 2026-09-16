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

    def wait_event(self, event) -> None:
        return None

    def wait_stream(self, other) -> None:
        return None


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


def _method_row(size: int, method: str, time_ns: float) -> dict:
    return {
        "cell": "unidirectional",
        "method": method,
        "of_record": method == ("graph" if size <= 1_048_576 else "eager"),
        "direction": "0->1",
        "requested_bytes": size,
        "bytes": size,
        "iterations": 200 if size <= 1_048_576 else 20,
        "status": "measured",
        "time_ns": time_ns,
        "bytes_per_second": size / (time_ns * 1e-9),
    }


def test_the_scorer_selects_the_amendments_row_of_record() -> None:
    """Graph rows at or below 1 MiB, eager rows above, fallbacks disclosed."""

    pytest.importorskip("numpy")
    scorer = _scorer()
    rows = [
        _method_row(8, "graph", 5_000.0),
        _method_row(8, "eager", 80_000.0),
        _method_row(1_048_576, "graph", 9_000.0),
        _method_row(1_048_576, "eager", 43_000.0),
        _method_row(2_097_152, "graph", 12_000.0),
        _method_row(2_097_152, "eager", 11_000.0),
        _method_row(4_194_304, "eager", 20_000.0),
    ]
    selected, record = scorer.rows_of_record(rows)
    chosen = {row["requested_bytes"]: scorer.row_method(row) for row in selected}
    assert chosen == {
        8: "graph",
        1_048_576: "graph",
        2_097_152: "eager",
        4_194_304: "eager",
    }
    assert record["fallback_payload_bytes"] == []
    assert record["methods_present"] == ["eager", "graph"]

    eager_only = [row for row in rows if scorer.row_method(row) == "eager"]
    selected, record = scorer.rows_of_record(eager_only)
    assert record["fallback_payload_bytes"] == [8, 1_048_576]
    assert all(scorer.row_method(row) == "eager" for row in selected)

    assert scorer.method_of_record(1_048_576) == "graph"
    assert scorer.method_of_record(1_048_577) == "eager"
    assert scorer.row_method({"bytes": 8}) == "eager"


def test_the_scorer_reports_a_degenerate_fit_instead_of_crashing(tmp_path: Path) -> None:
    """The stage 1 shape: a flat window whose fitted slope is not positive."""

    pytest.importorskip("numpy")
    expectations = _expectations()
    result = _synthetic_result(expectations)
    for row in result["p3"]:
        size = row["requested_bytes"]
        if size <= 262_144:
            row["time_ns"] = 70_000.0 + (8 - size) * 1e-6
            row["method"] = "graph"
    report = _score(result, tmp_path)

    fit = _outcome(report, "E5-fit")
    assert fit["passed"] is False
    assert fit["observed"]["fit_degenerate"] is True
    assert "not positive" in fit["observed"]["fit_reason"]
    holdout = _outcome(report, "E5-holdout-w2")
    assert holdout["passed"] is False
    assert holdout["observed"]["fit_degenerate"] is True
    assert report["proposed_profile"] is None
    assert report["void"] is False

    raw = (tmp_path / "measurements" / "scored.json").read_text()
    assert "Infinity" not in raw and "NaN" not in raw
    json.loads(raw)


def test_a_degenerate_fit_is_structured_not_an_exception() -> None:
    pytest.importorskip("numpy")
    scorer = _scorer()
    flat = scorer.ols_fit([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])
    assert flat.degenerate is True
    assert flat.beta_bytes_per_second == float("inf")
    too_few = scorer.ols_fit([1.0], [5.0])
    assert too_few.degenerate is True
    assert "at least 3" in too_few.reason
    no_spread = scorer.ols_fit([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])
    assert no_spread.degenerate is True
    good = scorer.ols_fit([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    assert good.degenerate is False
    assert good.beta_bytes_per_second == pytest.approx(1.0)
    assert scorer._finite(float("inf")) is None
    assert scorer._sanitize({"a": [float("nan"), 1.0]}) == {"a": [None, 1.0]}


def test_the_graph_path_is_requested_only_in_cuda_mode() -> None:
    pytest.importorskip("torch")
    bench = _load("b200_nvlink_envelope_bench_graph", BENCH_PATH)

    assert bench._graph_supported(True) is False
    assert bench.GRAPH_ROW_MAX_BYTES == 1_048_576
    assert bench._iterations(1_048_576, False) == (5, 200)
    assert bench._iterations(2_097_152, False) == (5, 20)
    assert bench._of_record("graph", 1_048_576) is True
    assert bench._of_record("eager", 1_048_576) is False
    assert bench._of_record("eager", 2_097_152) is True

    enabled, reason = bench.probe_graph_support(True, bench.torch.device("cpu"))
    assert enabled is False
    assert "mock" in reason

    seconds, skip = bench._time_block(
        True, bench.torch.device("cpu"), 2, lambda: None, bench.METHOD_GRAPH
    )
    assert seconds == 0.0
    assert "cannot be captured" in skip

    pool = bench.BufferPool(True)
    seconds, skip = bench._time_transfers(
        [(0, 1)], 8, 1, 2, pool, True, bench.METHOD_GRAPH
    )
    assert seconds == []
    assert "cannot be captured" in skip
    seconds, skip = bench._time_transfers([(0, 1)], 8, 1, 2, pool, True, bench.METHOD_EAGER)
    assert len(seconds) == 1 and skip == ""


class _CaptureLog:
    def __init__(self) -> None:
        self.calls: list[tuple] = []


class _LoggingStream:
    def __init__(self, log: _CaptureLog, name: str) -> None:
        self.log = log
        self.name = name
        self.device = name

    def wait_event(self, event) -> None:
        self.log.calls.append(("stream_wait_event", self.name, event.name))

    def wait_stream(self, other) -> None:
        self.log.calls.append(("stream_wait_stream", self.name, other.name))


class _LoggingEvent:
    counter = 0

    def __init__(self, log: _CaptureLog, enable_timing: bool = False) -> None:
        _LoggingEvent.counter += 1
        self.log = log
        self.name = f"event{_LoggingEvent.counter}"
        self.enable_timing = enable_timing

    def record(self, stream=None) -> None:
        self.log.calls.append(("event_record", self.name, getattr(stream, "name", None)))

    def elapsed_time(self, other) -> float:
        return 1.0


class _LoggingGraph:
    def __init__(self, log: _CaptureLog) -> None:
        self.log = log
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1
        self.log.calls.append(("replay",))


def _capture_fakes(monkeypatch, bench, log: _CaptureLog):
    """Patch the CUDA surface the capture path uses, recording the call shape."""

    torch = bench.torch
    state = _FakeCudaDevices(current=0)
    _fake_cuda(monkeypatch, bench, state)
    streams: dict[str, _LoggingStream] = {}

    def _stream_factory(device=None):
        name = f"stream{len(streams)}@{device}"
        stream = _LoggingStream(log, name)
        streams[name] = stream
        return stream

    class _StreamContext:
        def __init__(self, stream) -> None:
            self.stream = stream

        def __enter__(self):
            log.calls.append(("stream_enter", self.stream.name))
            return self

        def __exit__(self, *exc_info) -> bool:
            log.calls.append(("stream_exit", self.stream.name))
            return False

    class _GraphContext:
        def __init__(self, graph, stream=None, capture_error_mode=None) -> None:
            self.stream = stream
            log.calls.append(("capture_begin", getattr(stream, "name", None), capture_error_mode))

        def __enter__(self):
            return self

        def __exit__(self, *exc_info) -> bool:
            log.calls.append(("capture_end",))
            return False

    graphs: list[_LoggingGraph] = []

    def _graph_factory():
        graph = _LoggingGraph(log)
        graphs.append(graph)
        return graph

    def _synchronize(device=None):
        index = device if isinstance(device, int) else getattr(device, "index", device)
        log.calls.append(("synchronize", index))

    monkeypatch.setattr(torch.cuda, "synchronize", _synchronize)
    monkeypatch.setattr(torch.cuda, "Stream", _stream_factory)
    monkeypatch.setattr(torch.cuda, "stream", _StreamContext)
    monkeypatch.setattr(torch.cuda, "Event", lambda enable_timing=False: _LoggingEvent(log))
    monkeypatch.setattr(torch.cuda, "CUDAGraph", _graph_factory)
    monkeypatch.setattr(torch.cuda, "graph", _GraphContext)
    return state, graphs


def test_the_peer_copy_capture_forks_and_joins_the_destination_stream(monkeypatch) -> None:
    """The shape the first board run was missing.

    A cross-device copy touches the destination device's stream, so that stream
    has to join the capture from the capture stream and rejoin it before the
    capture ends. Capturing only the source stream is what failed on the board
    with "operation failed due to a previous error during capture".
    """

    pytest.importorskip("torch")
    bench = _load("b200_nvlink_envelope_bench_capture", BENCH_PATH)
    log = _CaptureLog()
    state, _ = _capture_fakes(monkeypatch, bench, log)

    pool = bench.BufferPool(False)
    source = pool.get(0, "src0", 8)
    destination = pool.get(1, "dst0", 8)
    bench._capture_transfer(source, destination, 0, 1, 3)

    kinds = [call[0] for call in log.calls]
    assert kinds.index("capture_begin") < kinds.index("event_record")
    fork_record = log.calls[kinds.index("event_record")]
    capture_begin = log.calls[kinds.index("capture_begin")]
    assert fork_record[2] == capture_begin[1], "the fork event is recorded on the capture stream"
    assert capture_begin[2] == "thread_local"

    wait = log.calls[kinds.index("stream_wait_event")]
    fork_name = fork_record[1]
    assert wait[2] == fork_name, "the destination stream waits on the fork event"
    destination_stream = wait[1]
    assert destination_stream != capture_begin[1]

    assert ("stream_enter", destination_stream) in log.calls
    join_index = len(kinds) - 1 - kinds[::-1].index("event_record")
    join_record = log.calls[join_index]
    assert join_record[2] == destination_stream, "the join event is recorded on the destination"
    last_wait = log.calls[len(kinds) - 1 - kinds[::-1].index("stream_wait_event")]
    assert last_wait[1] == capture_begin[1], "the capture stream waits on the join event"
    assert last_wait[2] == join_record[1]
    assert kinds.index("capture_end") == len(kinds) - 1
    assert state.current == 0, "capture must not leave the process on another device"


def test_every_graph_row_throws_away_one_replay(monkeypatch) -> None:
    pytest.importorskip("torch")
    bench = _load("b200_nvlink_envelope_bench_replay", BENCH_PATH)
    assert bench.UNTIMED_REPLAYS == 1
    log = _CaptureLog()
    _, graphs = _capture_fakes(monkeypatch, bench, log)

    pool = bench.BufferPool(False)
    seconds, skip = bench._time_transfers(
        [(0, 1)], 8, 1, 4, pool, False, bench.METHOD_GRAPH
    )
    assert skip == ""
    assert len(seconds) == 1
    assert len(graphs) == 1
    assert graphs[0].replays == bench.UNTIMED_REPLAYS + 1, "one untimed replay, then the timed one"

    timed_index = [index for index, call in enumerate(log.calls) if call[0] == "replay"]
    records = [index for index, call in enumerate(log.calls) if call[0] == "event_record"]
    assert timed_index[0] < max(records), "the untimed replay precedes the timing events"


def test_the_eager_copy_runs_first_and_is_timed_on_the_destination(monkeypatch) -> None:
    """The stage 1 attempt 5 defect: eager rows polluted by the capture.

    Every eager 0 to 1 row read a flat 2.2 ms at every payload while 1 to 0 was
    physical, and the only asymmetry was the capture that ran first in the same
    cell. The eager control is now measured before anything captures, both
    devices are synchronized before the timed block, streams and events are
    fresh, and the interval is bracketed on the destination device with an
    explicit completion event from the source stream.
    """

    pytest.importorskip("torch")
    bench = _load("b200_nvlink_envelope_bench_order", BENCH_PATH)
    log = _CaptureLog()
    state, _ = _capture_fakes(monkeypatch, bench, log)

    pool = bench.BufferPool(False)
    rows = bench._copy_cell_rows("unidirectional", [(0, 1)], 8, pool, False, bench.METHODS)

    assert [row["method"] for row in rows] == ["eager", "graph"]
    assert rows[0]["timed_on"] == "destination"
    assert rows[1]["timed_on"] == "source"

    kinds = [call[0] for call in log.calls]
    first_capture = kinds.index("capture_begin")
    first_record = kinds.index("event_record")
    assert first_record < first_capture, "the eager row is measured before any capture"

    synchronized = {
        call[1] for call in log.calls[:first_record] if call[0] == "synchronize"
    }
    assert {0, 1} <= synchronized, "both devices are synchronized before the timed block"

    eager_calls = log.calls[:first_capture]
    records = [call for call in eager_calls if call[0] == "event_record"]
    assert len(records) == 3
    assert records[0][2].endswith("@1"), "the start event sits on the destination stream"
    assert records[1][2].endswith("@0"), "the completion event sits on the source stream"
    assert records[2][2].endswith("@1"), "the stop event sits on the destination stream"

    waits = [call for call in eager_calls if call[0] == "stream_wait_event"]
    assert waits, "the destination stream waits for the copies to complete"
    assert waits[-1][1].endswith("@1")
    assert waits[-1][2] == records[1][1], "it waits on the source side completion event"

    entered = [call[1] for call in eager_calls if call[0] == "stream_enter"]
    assert any(name.endswith("@0") for name in entered)
    assert any(name.endswith("@1") for name in entered)
    assert state.current == 0, "the cell must not leave the process on another device"


def test_a_missing_scored_holdout_fails_and_keeps_the_denominator(tmp_path: Path) -> None:
    """A scored row that was never measured is not a row that passed."""

    pytest.importorskip("numpy")
    expectations = _expectations()
    result = _synthetic_result(expectations)
    for lane, cell in (("p1", "unidirectional"), ("p2", "unidirectional")):
        result[lane] = [
            row
            for row in result[lane]
            if not (row["cell"] == cell and row["requested_bytes"] == 67_108_864)
        ]
    result["p3"] = [row for row in result["p3"] if row["requested_bytes"] != 4_096]
    report = _score(result, tmp_path)

    assert report["scored_total"] == 4
    assert report["scored_passed"] == 0
    for ident in ("E2-0->1-holdout", "E2-1->0-holdout", "E3-holdout", "E5-holdout-w2"):
        outcome = _outcome(report, ident)
        assert outcome["passed"] is False, ident
        assert "missing" in outcome["detail"] or "has no" in outcome["detail"]


def test_an_all_reduce_row_without_a_correctness_result_is_fatal(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    expectations = _expectations()
    result = _synthetic_result(expectations)
    for row in result["p3"]:
        if row["requested_bytes"] == 65_536:
            del row["all_reduce_correct"]
    scorer = _scorer()
    measurements = tmp_path / "measurements"
    measurements.mkdir(parents=True, exist_ok=True)
    (measurements / "stage1_result.json").write_text(json.dumps(result))
    assert scorer.main(["--measurements", str(measurements)]) == 1

    report = json.loads((measurements / "scored.json").read_text())
    assert report["void"] is True
    assert any("carries no correctness result" in entry["detail"] for entry in report["fatal"])
    assert "E1" in report["void_scope"] and "E8" in report["void_scope"]


def test_the_e6_asymptote_fits_the_whole_window(tmp_path: Path) -> None:
    """E6 reports rather than scores, so it keeps the 64 MiB row."""

    pytest.importorskip("numpy")
    expectations = _expectations()
    report = _score(_synthetic_result(expectations), tmp_path)

    window = [
        size
        for size in expectations["lanes"]["payload_bytes"]
        if 1_048_576 <= size <= 1_073_741_824
    ]
    asymptote = _outcome(report, "E6")["observed"]["all_reduce_asymptote"]
    assert asymptote["fit_points"] == len(window)
    holdout_fit = _outcome(report, "E2-0->1-fit")["observed"]
    assert holdout_fit["fit_points"] == len(window) - 1


class _FakeDist:
    """Enough of torch.distributed to record how the lane calls it."""

    def __init__(self, new_group_error: Exception | None = None) -> None:
        self.calls: list[tuple] = []
        self.new_group_error = new_group_error

    def new_group(self, *args, **kwargs):
        self.calls.append(("new_group", args, kwargs))
        if self.new_group_error is not None:
            raise self.new_group_error
        return f"group:{kwargs.get('backend', 'default')}"

    def barrier(self, *args, **kwargs):
        self.calls.append(("barrier", args, kwargs))


def test_the_inter_lane_barriers_wait_in_a_host_group(monkeypatch) -> None:
    """Amendment c: an idle rank must not spin on a device another rank times."""

    pytest.importorskip("torch")
    bench = _load("b200_nvlink_envelope_bench_barrier", BENCH_PATH)
    fake = _FakeDist()
    monkeypatch.setattr(bench, "dist", fake)

    group, backend = bench.make_barrier_group(False)
    assert backend == "gloo"
    assert group == "group:gloo"
    assert fake.calls == [("new_group", (), {"backend": "gloo"})]

    # Mock mode already runs the world group on gloo, so it needs no second one.
    fake.calls.clear()
    group, backend = bench.make_barrier_group(True)
    assert (group, backend) == (None, "gloo")
    assert fake.calls == []


def test_a_host_group_that_cannot_be_built_falls_back_and_says_so(monkeypatch) -> None:
    pytest.importorskip("torch")
    bench = _load("b200_nvlink_envelope_bench_fallback", BENCH_PATH)
    monkeypatch.setattr(bench, "dist", _FakeDist(RuntimeError("no gloo in this build")))

    group, backend = bench.make_barrier_group(False)
    assert group is None
    assert backend == "nccl"


def test_every_barrier_in_the_lane_names_its_group() -> None:
    """The NCCL world group is never waited in outside a timed lane."""

    source = BENCH_PATH.read_text(encoding="utf-8")
    bare = source.count("dist.barrier()")
    assert bare == 0, "a barrier without a group waits in the NCCL world group"
    assert source.count("dist.barrier(group=") >= 4


def _placement_row(cell: str, method: str, size: int, time_ns: float, **extra) -> dict:
    row = {
        "cell": cell,
        "method": method,
        "of_record": method == "graph",
        "direction": extra.pop("direction", "0->1"),
        "requested_bytes": size,
        "bytes": size,
        "iterations": 20,
        "status": "measured",
        "time_ns": time_ns,
        "bytes_per_second": size / (time_ns * 1e-9),
        "aggregate_bytes_per_second": extra.pop("aggregate", size / (time_ns * 1e-9)),
    }
    row.update(extra)
    return row


def test_the_placement_cells_read_graph_rows_and_mark_the_spinning_peer(tmp_path) -> None:
    pytest.importorskip("numpy")
    scorer = _scorer()
    rows = [
        _placement_row("all_pairs", "graph", 16_777_216, 27_500.0),
        _placement_row("all_pairs", "eager", 16_777_216, 2_330_000.0),
        _placement_row("fanin", "graph", 67_108_864, 594_000.0, donors=7),
        _placement_row("fanin", "eager", 67_108_864, 595_000.0, donors=7),
    ]
    selected, record = scorer.placement_rows_of_record(rows)
    assert [scorer.row_method(row) for row in selected] == ["graph", "graph"]
    assert record["method_of_record"] == "graph"
    assert record["fallback_payload_bytes"] == []

    marked = scorer.mark_not_meaningful(rows)
    assert marked["not_meaningful_eager_rows"] == 1
    assert marked["rows"][0]["cell"] == "all_pairs"
    assert "spinning" in marked["reason"]
    # The clean fan-in control is under 2 ms and keeps its meaning.
    assert all(entry["cell"] != "fanin" for entry in marked["rows"])


def test_the_fanin_floor_is_the_nameplate_not_the_inverted_fraction() -> None:
    """The frozen rule asked for 636 us where the nameplate allows 522 us."""

    pytest.importorskip("numpy")
    scorer = _scorer()
    stage2 = {
        "p1": [
            _placement_row("unidirectional", "graph", 67_108_864, 85_900.0),
            _placement_row("fanin", "graph", 67_108_864, 594_000.0, donors=7, aggregate=790.9e9),
        ]
    }
    outcomes = scorer.score_e7(stage2, {"0->1": {"beta_bytes_per_second": 781e9}})
    fanin = next(outcome for outcome in outcomes if outcome.ident == "E7-fanin")

    floor_ns = 7 * 67_108_864 / 900e9 * 1e9
    assert floor_ns == pytest.approx(521_962.0, abs=100)
    assert fanin.observed["floor_ns"] == pytest.approx(floor_ns)
    assert fanin.passed is True
    assert fanin.observed["fraction_of_nameplate"] == pytest.approx(0.879, abs=0.001)


def test_the_eager_agreement_check_reports_the_worst_gap() -> None:
    pytest.importorskip("numpy")
    scorer = _scorer()
    agree = {
        "stage2": {
            "p1": [
                _placement_row("all_pairs", "graph", 16_777_216, 27_500.0),
                _placement_row("all_pairs", "eager", 16_777_216, 28_000.0),
            ]
        }
    }
    outcome = scorer.score_eager_agreement(agree)
    assert outcome.passed is True
    assert outcome.observed["outside"] == 0

    spinning = {
        "stage2": {
            "p1": [
                _placement_row("all_pairs", "graph", 16_777_216, 27_500.0),
                _placement_row("all_pairs", "eager", 16_777_216, 2_330_000.0),
            ]
        }
    }
    outcome = scorer.score_eager_agreement(spinning)
    assert outcome.passed is False
    assert outcome.observed["outside"] == 1
    assert outcome.cls == "structural"
