import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_credit_arbitration_v1"
PRODUCER = ROOT / "examples" / "a100_nvlink_packet_v2" / "nvlink_packet_lane.cu"
PRODUCER_EXTENSION = STUDY / "producer_traf73.patch"
RUNNER_PATH = STUDY / "run_hardware_campaign.py"
SCORER_PATH = STUDY / "score_hardware_identification.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_module("test_traf73_campaign_runner", RUNNER_PATH)


@pytest.fixture(scope="module")
def scorer():
    return _load_module("test_traf73_hardware_scorer", SCORER_PATH)


@pytest.fixture(scope="module")
def frozen(runner):
    return runner.load_expectations()


@pytest.fixture(scope="module")
def mock_binary(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    suffix = ".exe" if os.name == "nt" else ""
    build_root = tmp_path_factory.mktemp("traf73-producer")
    source = build_root / "examples/a100_nvlink_packet_v2/nvlink_packet_lane.cu"
    source.parent.mkdir(parents=True)
    shutil.copyfile(PRODUCER, source)
    applied = subprocess.run(
        (
            "git",
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.eol=lf",
            "-C",
            str(build_root),
            "apply",
            "--whitespace=nowarn",
            str(PRODUCER_EXTENSION),
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert applied.returncode == 0, applied.stderr
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "3e4b24382314f5f0dd84f4b54d126c5777e6c92a71c3644f8835b2f5cd3a4694"
    )
    output = build_root / f"producer{suffix}"
    completed = subprocess.run(
        (
            compiler,
            "-x",
            "c++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-DSIMLLM_NVLINK_MOCK",
            f"-I{PRODUCER.parent}",
            str(source),
            "-o",
            str(output),
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    return output


def test_campaign_pins_final_aligned_freeze(runner, frozen):
    assert runner.EXPECTATIONS_COMMIT == "f3f2624e7a96efe3ad67eac5940fee8746e40b98"
    assert runner.EXPECTATIONS_SHA256 == (
        "a17b9e298d11a4a6ba92b382121c15a5a48f8100b6f343893260419b1d3382f6"
    )
    assert frozen["study"]["hardware_executed"] is False
    runner.verify_frozen_authority(frozen)


def test_archive_stage_uses_the_frozen_blob_without_remote_git(
    runner, tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    frozen = runner.load_expectations()
    assert frozen["study"]["status"] == "EXPECTATIONS_ONLY"


def test_h1_expands_all_pairs_and_the_seeded_31_size_order(runner, frozen):
    points = runner.h1_points(frozen)
    assert len(points) == 372
    by_pair = {}
    for point in points:
        pair = (point["source"], point["destination"])
        by_pair.setdefault(pair, []).append(point["payload_bytes"])
        assert point["pattern"] == "traf73_latency_batch"
        assert point["traf73_warmup_repetitions"] == 32
        assert point["traf73_timed_repetitions"] == 200
    assert len(by_pair) == 12
    orders = list(by_pair.values())
    assert all(order == orders[0] for order in orders)
    assert sorted(orders[0]) == frozen["h1_credit_window_and_return"][
        "payload_sizes_bytes"
    ]
    assert orders[0] != sorted(orders[0])


def test_h2_uses_full_sweep_and_records_steady_exclusion(runner, frozen):
    points = runner.h2_points(frozen)
    assert len(points) == 93
    assert {point["sources"] for point in points} == {"0", "0,1", "0,1,2"}
    assert {point["traf73_excluded_repetitions_each_edge"] for point in points} == {
        20
    }
    assert {point["traf73_warmup_repetitions"] for point in points} == {64}
    assert {point["payload_bytes"] for point in points} == set(
        frozen["h1_credit_window_and_return"]["payload_sizes_bytes"]
    )


def test_h3_rotates_greedy_role_and_freezes_common_window(runner, frozen):
    points = runner.h3_points(frozen)
    assert [point["source"] for point in points] == [0, 1, 2]
    assert [point["sources"].split(",")[0] for point in points] == ["0", "1", "2"]
    assert {point["traf73_flow_offered_rate_percents"] for point in points} == {
        "100,60,60"
    }
    assert {point["traf73_window_warmup_ms"] for point in points} == {50}
    assert {point["traf73_window_measurement_ms"] for point in points} == {500}
    assert {point["traf73_window_drain_ms"] for point in points} == {50}
    assert {point["traf73_ring_bytes"] for point in points} == {8 * 1024 * 1024}


def _run_family(tmp_path, mock_binary, family):
    completed = subprocess.run(
        (
            sys.executable,
            str(RUNNER_PATH),
            "--family",
            family,
            "--mode",
            "mock",
            "--binary",
            str(mock_binary),
            "--output-root",
            str(tmp_path),
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed


def test_mock_campaign_is_resumable_and_emits_each_new_observable(
    tmp_path, mock_binary, runner
):
    for family in ("h1", "h2", "h3"):
        first = _run_family(tmp_path, mock_binary, family)
        second = _run_family(tmp_path, mock_binary, family)
        assert "complete" in first.stdout
        assert "already complete and digest verified" in second.stdout
        attempt = next(
            runner.cell_root(tmp_path, family).glob("attempt-*")
        )
        rows = [
            json.loads(line)
            for line in (attempt / "results.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert rows
        assert all(row["checksum_ok"] is True for row in rows)
        assert all(row["measurement_claim"] is False for row in rows)
        if family in {"h1", "h2"}:
            assert all(
                len(flow) == 200
                for row in rows
                for flow in row["traf73"]["repetition_completion_us_by_flow"]
            )
        else:
            assert len(rows) == 3
            assert all(
                row["traf73"]["flow_offered_rate_percents"] == [100, 60, 60]
                for row in rows
            )
            assert all(
                row["traf73"]["window_device_us_by_flow"]
                == [500_000.0, 500_000.0, 500_000.0]
                for row in rows
            )


def test_legacy_control_digest_and_effect_shape_remain_identity_off(
    tmp_path, mock_binary
):
    plan = tmp_path / "legacy.tsv"
    output = tmp_path / "legacy.jsonl"
    fields = (
        "case_name",
        "point_id",
        "producer",
        "payload_bytes",
        "message_count",
        "source",
        "destination",
        "sources",
        "destinations",
        "source_alignment",
        "destination_alignment",
        "access_width",
        "active_lanes",
        "lane_mask",
        "stride",
        "stream_count",
        "outstanding",
        "burst_messages",
        "gap_ns",
        "offered_rate_percent",
        "pattern",
    )
    values = (
        "LEGACY",
        "LEGACY:point",
        "persistent_sm_peer_write",
        "256",
        "32",
        "0",
        "1",
        "0",
        "1",
        "0",
        "0",
        "16",
        "32",
        "contiguous",
        "1",
        "1",
        "256",
        "256",
        "0",
        "100",
        "unidirectional",
    )
    plan.write_text(
        "\t".join(fields) + "\n" + "\t".join(values) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    completed = subprocess.run(
        (
            str(mock_binary),
            "--points",
            str(plan),
            "--output",
            str(output),
            "--mode",
            "mock",
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    row = json.loads(output.read_text(encoding="utf-8"))
    canonical = "\n".join(values) + "\n"
    assert row["applied_control_sha256"] == hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    assert "traf73" not in row
    assert set(row["applied_controls"]["effects"]) == {
        "payload_bytes",
        "message_count",
        "source",
        "destination",
        "sources",
        "destinations",
        "source_alignment",
        "destination_alignment",
        "access_width",
        "active_lanes",
        "lane_mask",
        "stride",
        "stream_count",
        "outstanding",
        "burst_messages",
        "gap_ns",
        "offered_rate_percent",
        "pattern",
    }


def test_knee_detector_applies_positive_five_mad_and_three_point_rule(
    scorer, frozen
):
    sizes = frozen["h1_credit_window_and_return"]["payload_sizes_bytes"]
    completion = [2.0 + size / 100_000 for size in sizes]
    break_index = sizes.index(262_144)
    completion = [
        value + (8.0 if index >= break_index else 0.0)
        for index, value in enumerate(completion)
    ]
    result = scorer.detect_knee(sizes, completion)
    assert result["status"] == "REPEATED_PERSISTENT_BREAK"
    assert result["payload_bytes"] == 262_144
    assert result["return_delay_ps"] == pytest.approx(8_000_000)

    no_break = scorer.detect_knee(
        sizes, [2.0 + size / 100_000 for size in sizes]
    )
    assert no_break["status"] == "NO_REPEATED_PERSISTENT_BREAK"
    assert no_break["payload_bytes"] is None


def test_repeated_knee_requires_the_same_break_on_every_timed_pass(
    scorer, frozen
):
    sizes = frozen["h1_credit_window_and_return"]["payload_sizes_bytes"]
    break_index = sizes.index(262_144)
    samples = [
        [2.0 + size / 100_000 + (8.0 if index >= break_index else 0.0)] * 4
        for index, size in enumerate(sizes)
    ]
    repeated = scorer.detect_repeated_knee(sizes, samples)
    assert repeated["payload_bytes"] == 262_144
    assert repeated["matching_repetition_count"] == 4

    samples[break_index][3] -= 8.0
    missing_pass = scorer.detect_repeated_knee(sizes, samples)
    assert missing_pass["status"] == "NO_REPEATED_PERSISTENT_BREAK"
    assert missing_pass["payload_bytes"] is None


def _h3_row(greedy, rates):
    sources = [greedy, *sorted({0, 1, 2} - {greedy})]
    links = []
    for source, rate in zip(sources, rates, strict=True):
        kib = round(rate * 1e9 * 0.5 / 1024)
        links.append(
            {
                "gpu": source,
                "remote_gpu": 3,
                "raw_tx_kib_delta": kib,
                "raw_rx_kib_delta": 0,
            }
        )
    return {
        "applied_controls": {
            "source": greedy,
            "destination": 3,
            "sources": ",".join(str(value) for value in sources),
            "traf73_window_measurement_ms": 500,
        },
        "traf73": {
            "window_counter_deltas": {
                "per_gpu_per_link_per_direction": links,
            },
            "window_completed_bytes_by_flow": [1, 1, 1],
        },
    }


@pytest.mark.parametrize(
    ("rates", "policy"),
    [
        ([87.101921876, 60.0, 60.0], "release_aware_round_robin"),
        ([100.0, 53.550960938, 53.550960938], "greedy_capture"),
        ([60.0, 60.0, 60.0], "static_interleave"),
    ],
)
def test_h3_classifier_selects_each_frozen_policy(scorer, frozen, rates, policy):
    rows = [_h3_row(greedy, rates) for greedy in (0, 1, 2)]
    result = scorer.classify_h3(rows, frozen, valid=True)
    assert result["identified_policy"] == policy
    assert result["verdict"] == f"IDENTIFIED_{policy.upper()}"


def test_submission_is_one_family_at_a_time_and_uses_no_new_cuda_harness():
    batch = (STUDY / "run_merlin_hypothesis.sbatch").read_text(encoding="utf-8")
    tracked_cuda = subprocess.run(
        ("git", "ls-files", "examples/nvlink_credit_arbitration_v1/*.cu"),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    ).stdout
    assert "#SBATCH --array" not in batch
    assert "TRAF73_FAMILY" in batch
    assert "Submit H1, H2 and H3 separately" in batch
    assert "examples/a100_nvlink_packet_v2/nvlink_packet_lane.cu" in batch
    assert "producer_traf73.patch" in batch
    assert tracked_cuda == ""
    assert "+/-" not in batch
    assert "\N{EM DASH}" not in batch
