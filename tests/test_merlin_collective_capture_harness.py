"""Offline tests for the TRAF-77 Merlin collective capture harness.

Every generated capture row in this module is labeled synthetic. These tests
exercise schemas, arithmetic and command entry points only. They make no claim
about Merlin hardware behavior.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "merlin_collective_capture_v1"
HARNESS = STUDY / "harness"
ANALYZER_PATH = HARNESS / "analyze_capture.py"
HASHER_PATH = HARNESS / "hash_manifest.py"
SNAPSHOT_PATH = HARNESS / "snapshot_counters.py"
CONFIG = json.loads((HARNESS / "study_config.json").read_text(encoding="utf-8"))


def _git_index_mode(path: Path) -> str:
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True, cwd=os.fspath(path.parent),
    ).stdout.strip()
    line = subprocess.run(
        ["git", "ls-files", "-s", "--", os.fspath(path)],
        capture_output=True, text=True, check=True, cwd=root,
    ).stdout.strip()
    return line.split()[0] if line else ""


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analyzer = _load_module("traf77_capture_analyzer", ANALYZER_PATH)
hasher = _load_module("traf77_capture_hasher", HASHER_PATH)
snapshotter = _load_module("traf77_counter_snapshotter", SNAPSHOT_PATH)

EXPECTED_PAYLOADS = [
    8,
    64,
    512,
    4096,
    16384,
    65536,
    131072,
    196608,
    262144,
    393216,
    524288,
    786432,
    1048576,
    1572864,
    2097152,
    3145728,
    4194304,
    8388608,
    16777216,
    33554432,
    67108864,
    134217728,
]


def _synthetic_duration_ns(
    width: int,
    concentration: str,
    operation: str,
    payload: int,
) -> int:
    if width == 2 and concentration == "four-port":
        duration = 40_000 + payload // 400
    elif width == 2:
        duration = 45_000 + payload // 150
    elif concentration == "four-port":
        duration = 60_000 + payload // 200
    else:
        four_port = 60_000 + payload // 200
        duration = int(four_port * (2.2 if payload >= 33_554_432 else 1.5))

    anchors = {
        (2, "four-port", "all_reduce", 8): 40_141,
        (2, "one-port", "all_reduce", 8): 45_000,
        (8, "four-port", "all_reduce", 8): 50_790,
        (8, "one-port", "all_reduce", 8): 80_000,
        (8, "four-port", "all_to_allv", 8): 89_805,
        (8, "one-port", "all_to_allv", 8): 120_000,
    }
    return anchors.get((width, concentration, operation, payload), duration)


def _synthetic_cell(
    width: int,
    concentration: str,
    operation: str,
    payload: int,
) -> dict:
    duration = _synthetic_duration_ns(width, concentration, operation, payload)
    chunks = (payload + CONFIG["chunk_limit_bytes"] - 1) // CONFIG["chunk_limit_bytes"]
    samples = []
    for repeat in range(CONFIG["measured_repeats"]):
        release = 1_000_000_000 + repeat * 1_000_000
        completions = [
            {
                "chunk_index": index,
                "payload_bytes": min(
                    CONFIG["chunk_limit_bytes"],
                    payload - index * CONFIG["chunk_limit_bytes"],
                ),
                "completion_monotonic_raw_ns": release
                + int(duration * (index + 1) / chunks),
                "elapsed_ns": int(duration * (index + 1) / chunks),
            }
            for index in range(chunks)
        ]
        samples.append(
            {
                "repeat": repeat,
                "max_rank_duration_ns": duration,
                "ranks": [
                    {
                        "rank": 0,
                        "host": "synthetic-node-a",
                        "release_monotonic_raw_ns": release,
                        "completion_monotonic_raw_ns": release + duration,
                        "duration_ns": duration,
                        "chunk_completions": completions,
                    }
                ],
            }
        )

    if concentration == "one-port":
        traffic = {"hsn0": 10_000_000, "hsn1": 10_000, "hsn2": 10_000, "hsn3": 10_000}
    else:
        traffic = {interface: 3_000_000 for interface in CONFIG["interfaces"]}
    rank_counters = [
        {
            "rank": 0,
            "host": "synthetic-node-a",
            "local_rank": 0,
            "before_monotonic_raw_ns": 10,
            "after_monotonic_raw_ns": 20,
            "ports": [
                {
                    "interface": interface,
                    "before_rx_bytes": 100,
                    "before_tx_bytes": 200,
                    "after_rx_bytes": 100 + delta // 2,
                    "after_tx_bytes": 200 + delta // 2,
                }
                for interface, delta in traffic.items()
            ],
        }
    ]
    row = {
        "schema": "simllm-merlin-collective-cell-v1",
        "evidence_class": "synthetic",
        "study": "merlin_collective_capture_v1",
        "attempt_id": f"synthetic-w{width}-{concentration}",
        "slurm_job_id": "synthetic-job",
        "slurm_nodelist": "synthetic-nodes",
        "submitted_script": f"capture_w{width}_{concentration.replace('-', '_')}.sbatch",
        "submitted_script_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "concentration": concentration,
        "operation": operation,
        "payload_semantics": CONFIG["payload_semantics"][operation],
        "clock": "CLOCK_MONOTONIC_RAW",
        "clock_epoch_scope": "rank-local",
        "width": width,
        "tasks_per_node": 1 if width == 2 else 4,
        "payload_bytes": payload,
        "measured_repeats": CONFIG["measured_repeats"],
        "excluded_warmups": CONFIG["excluded_warmups"],
        "chunk_limit_bytes": CONFIG["chunk_limit_bytes"],
        "chunk_count": chunks,
        "nccl_version": 23102,
        "max_rank_mismatches": 0,
        "samples": samples,
        "rank_counters": rank_counters,
    }
    anchor = next(
        (
            item
            for item in CONFIG["anchors_ps"]
            if item["width"] == width
            and item["operation"] == operation
            and item["payload_bytes"] == payload
        ),
        None,
    )
    if anchor is not None:
        ratio = duration * 1000 / anchor["completion_ps"]
        row.update(
            {
                "fg4_anchor_ps": anchor["completion_ps"],
                "fg4_anchor_ratio": ratio,
                "fg4_anchor_held": 0.5 <= ratio <= 2.0,
            }
        )
    return row


def _counter_snapshot(
    node: str,
    interface: str,
    concentration: str,
    tag: str,
) -> dict:
    if concentration == "one-port":
        delta = 20_000_000 if interface == "hsn0" else 10_000
    else:
        delta = 5_000_000
    before = tag == "before"
    value = 100 if before else 100 + delta
    return {
        "schema": "simllm-merlin-interface-counter-snapshot-v1",
        "evidence_class": "synthetic",
        "tag": tag,
        "node": node,
        "interface": interface,
        "clock_monotonic_raw_ns": 1 if before else 2,
        "clock_realtime_ns": 1 if before else 2,
        "sources": {
            "sysfs_statistics": {"rx_bytes": value, "tx_bytes": value},
            "sysfs_queues": {"tx-0/byte_queue_limits/hold_time": value},
            "ethtool_statistics": {
                "available": True,
                "returncode": 0,
                "error": "",
                "values": {"tx_packets": value},
            },
            "traffic_control_qdisc": {"available": True, "returncode": 0},
            "ip_link_statistics": {"available": True, "returncode": 0},
        },
    }


def _write_synthetic_attempt(
    root: Path,
    width: int,
    concentration: str,
    full: bool = True,
) -> Path:
    attempt = root / "attempts" / f"synthetic-w{width}-{concentration}"
    attempt.mkdir(parents=True)
    rows = []
    operations = CONFIG["operations"] if full else ["all_reduce"]
    payloads = CONFIG["payload_bytes"] if full else [8]
    for operation in operations:
        for payload in payloads:
            rows.append(_synthetic_cell(width, concentration, operation, payload))
    (attempt / "capture.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    counter_dir = attempt / "counter_snapshots"
    counter_dir.mkdir()
    for node in ("synthetic-node-a", "synthetic-node-b"):
        counter_rows = [
            _counter_snapshot(node, interface, concentration, tag)
            for tag in ("before", "after")
            for interface in CONFIG["interfaces"]
        ]
        (counter_dir / f"{node}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in counter_rows),
            encoding="utf-8",
        )
    (attempt / "nccl_selection.txt").write_text(
        "SYNTHETIC FIXTURE NCCL INFO Using network Socket\n"
        "SYNTHETIC FIXTURE NCCL INFO NET/Socket : Using [0]hsn0\n"
        "SYNTHETIC FIXTURE NCCL INFO 4 coll channels\n"
        "SYNTHETIC FIXTURE NCCL INFO TUNING Algo RING Proto SIMPLE GDR 0\n",
        encoding="utf-8",
    )
    return attempt


@pytest.fixture
def synthetic_capture(tmp_path: Path) -> Path:
    root = tmp_path / "synthetic-evidence"
    for width in CONFIG["widths"]:
        for concentration in CONFIG["concentrations"]:
            _write_synthetic_attempt(root, width, concentration)
    return root


def test_config_and_batch_scripts_lock_every_frozen_cell() -> None:
    assert CONFIG["payload_bytes"] == EXPECTED_PAYLOADS
    assert CONFIG["operations"] == [
        "all_gather",
        "reduce_scatter",
        "all_reduce",
        "all_to_allv",
    ]
    assert CONFIG["measured_repeats"] == 25
    assert CONFIG["excluded_warmups"] == 5
    assert CONFIG["chunk_limit_bytes"] == 8_388_608

    for width in (2, 8):
        for concentration in ("one_port", "four_port"):
            path = HARNESS / f"capture_w{width}_{concentration}.sbatch"
            text = path.read_text(encoding="utf-8")
            assert f"export TRAF77_WIDTH={width}" in text
            assert "all four operations at all\n# 22 payloads, for 88 frozen cells" in text
            expected_tasks = 1 if width == 2 else 4
            assert f"#SBATCH --ntasks-per-node={expected_tasks}" in text
            if os.name == "posix":
                assert _git_index_mode(path) == "100755"


@pytest.mark.skipif(
    os.name != "posix", reason="the capture scripts target the Linux cluster runtime"
)
def test_shell_entry_points_are_syntactically_valid_and_executable() -> None:
    scripts = [
        HARNESS / "run_capture.sh",
        HARNESS / "capture_rank_identity.sh",
        *sorted(HARNESS.glob("*.sbatch")),
    ]
    for path in scripts:
        completed = subprocess.run(
            ["bash", "-n", str(path)], check=False, capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stderr
        assert _git_index_mode(path) == "100755"


def test_lane_source_locks_raw_wall_timing_and_anchor_first_order() -> None:
    source = (HARNESS / "merlin_collective_lane.cu").read_text(encoding="utf-8")
    assert "clock_gettime(CLOCK_MONOTONIC_RAW" in source
    assert "local_timing[base] = now_raw_ns()" in source
    assert "local_timing[base + 1] = now_raw_ns()" in source
    assert "chunk_completion - release" in source
    assert source.index("Operation::AllReduce, 8") < source.index(
        "const Operation operations[]"
    )
    assert '"FATAL FG-4 anchor miss' in source
    assert "kPayloadCount" in source
    payload_block = re.search(
        r"constexpr size_t kPayloads\[\] = \{(.*?)\};", source, re.DOTALL
    )
    assert payload_block is not None
    assert [int(value) for value in re.findall(r"\d+", payload_block.group(1))] == (
        EXPECTED_PAYLOADS
    )


def test_runbook_locks_the_remote_command_sequence_and_hash_checks() -> None:
    text = (STUDY / "RUNBOOK.md").read_text(encoding="utf-8")
    assert text.index("ssh merlin") < text.index("rsync -av --checksum")
    assert "scp merlin:simllm-stage" in text
    assert text.count("sbatch -M gmerlin7 --account=merlin") == 4
    assert "squeue -M gmerlin7" in text
    assert (
        'LOCAL_EVIDENCE_ROOT="${SIMLLM_TRAF77_EVIDENCE_ROOT:?set '
        'SIMLLM_TRAF77_EVIDENCE_ROOT}"' in text
    )
    assert "cuda/12.2.2" in text
    assert "nvidia_nccl_cu12" in text
    assert 'ln -s libnccl.so.2 "${NCCL_ROOT}/lib/libnccl.so"' in text
    assert "submitted_scripts.remote.pre_submit.sha256" in text
    assert "submitted_scripts.remote.post_run.sha256" in text


def test_cell_identity_round_trip() -> None:
    raw = {
        "width": 8,
        "concentration": "one-port",
        "operation": "reduce_scatter",
        "payload_bytes": 1_048_576,
    }
    identity = analyzer.CellIdentity.from_row(raw)
    assert identity.to_dict() == raw
    assert identity.cell_id == "w8/one-port/reduce_scatter/1048576"


def test_counter_delta_handles_increment_wrap_and_nested_sources() -> None:
    assert analyzer.counter_delta(10, 25) == 15
    assert analyzer.counter_delta((1 << 64) - 4, 3) == 7
    with pytest.raises(ValueError, match="nonnegative"):
        analyzer.counter_delta(-1, 2)
    rows = analyzer.diff_numeric_counters(
        {"port": {"tx_bytes": 100, "label": "synthetic"}},
        {"port": {"tx_bytes": 145, "label": "synthetic"}},
    )
    assert rows == [
        {
            "counter": "port.tx_bytes",
            "before": 100,
            "after": 145,
            "delta": 45,
            "wrapped": False,
        }
    ]


def test_routing_observation_keeps_tx_and_rx_fractions_separate() -> None:
    nodes = {
        "synthetic-node": {
            "ports": {
                "hsn0": {"tx_delta_bytes": 1_000, "rx_delta_bytes": 9_000_000},
                "hsn1": {"tx_delta_bytes": 1_000, "rx_delta_bytes": 1_000},
                "hsn2": {"tx_delta_bytes": 9_000_000, "rx_delta_bytes": 1_000},
                "hsn3": {"tx_delta_bytes": 1_000, "rx_delta_bytes": 1_000},
            }
        }
    }

    observation = analyzer._routing_observation("one-port", nodes, CONFIG)

    node = observation["nodes"]["synthetic-node"]
    assert observation["status"] == "contradicted"
    assert node["directions"]["tx"]["status"] == "contradicted"
    assert node["directions"]["rx"]["status"] == "proven"


def test_routing_observation_retains_the_pooled_one_mib_signal_gate() -> None:
    nodes = {
        "synthetic-node": {
            "ports": {
                port: {"tx_delta_bytes": 10_000, "rx_delta_bytes": 10_000}
                for port in CONFIG["interfaces"]
            }
        }
    }

    observation = analyzer._routing_observation("one-port", nodes, CONFIG)

    node = observation["nodes"]["synthetic-node"]
    assert observation["status"] == "insufficient-signal"
    assert {row["status"] for row in node["directions"].values()} == {
        "insufficient-signal"
    }


@pytest.mark.skipif(
    os.name != "posix", reason="the snapshot tool needs the POSIX monotonic raw clock"
)
def test_snapshot_parser_reads_synthetic_sysfs_and_names_sources(tmp_path: Path) -> None:
    net_root = tmp_path / "synthetic-sysfs"
    statistics = net_root / "hsn0" / "statistics"
    queue = net_root / "hsn0" / "queues" / "tx-0"
    statistics.mkdir(parents=True)
    queue.mkdir(parents=True)
    (statistics / "rx_bytes").write_text("123\n", encoding="utf-8")
    (statistics / "tx_bytes").write_text("456\n", encoding="utf-8")
    (queue / "synthetic_counter").write_text("7\n", encoding="utf-8")

    row = snapshotter.snapshot("hsn0", "synthetic-before", net_root)
    assert row["sources"]["sysfs_statistics"] == {"rx_bytes": 123, "tx_bytes": 456}
    assert row["sources"]["sysfs_queues"] == {"tx-0/synthetic_counter": 7}
    assert set(row["sources"]) == {
        "sysfs_statistics",
        "sysfs_queues",
        "ethtool_statistics",
        "traffic_control_qdisc",
        "ip_link_statistics",
    }


def test_submitted_source_manifest_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    manifest = hasher.manifest_text(HARNESS)
    assert [line[66:] for line in manifest.splitlines()] == list(hasher.SUBMITTED_FILES)
    copied = tmp_path / "synthetic-staged-copy"
    copied.mkdir()
    for name in hasher.SUBMITTED_FILES:
        shutil.copyfile(HARNESS / name, copied / name)
    manifest_path = copied / "submitted_scripts.sha256"
    manifest_path.write_text(manifest, encoding="utf-8")
    assert hasher.check_manifest(copied, manifest_path) == []
    (copied / "study_config.json").write_bytes(
        (copied / "study_config.json").read_bytes() + b"\n"
    )
    assert hasher.check_manifest(copied, manifest_path) == ["study_config.json"]


def test_synthetic_capture_normalizes_and_exercises_fg4_and_e1_to_e4(
    synthetic_capture: Path,
) -> None:
    result = analyzer.analyze(synthetic_capture)
    assert result["evidence_classes"] == ["synthetic"]
    assert not result["t2b_scored"]
    assert result["coverage"] == {
        "expected_cells": 352,
        "observed_cells": 352,
        "missing_cell_ids": [],
        "unexpected_cell_ids": [],
        "duplicate_cell_ids": [],
        "complete": True,
    }
    assert result["fatal_guards"][0]["id"] == "FG-2"
    assert result["fatal_guards"][0]["held"]
    assert result["fatal_guards"][1]["id"] == "FG-4"
    assert result["fatal_guards"][1]["held"]
    assert [row["status"] for row in result["directional_evaluations"]] == [
        "PASS",
        "PASS",
        "PASS",
        "PASS",
    ]
    first_attempt = result["attempts"][0]
    assert first_attempt["nccl"]["algorithms"] == ["RING"]
    assert first_attempt["nccl"]["protocols"] == ["SIMPLE"]
    assert first_attempt["nccl"]["collective_channel_counts"] == [4]
    assert first_attempt["routing_proof"]["held"]
    assert first_attempt["counter_snapshots"]["evidence_classes"] == ["synthetic"]
    first_row = result["normalized_rows"][0]
    assert first_row["nccl_selection"]["scope"] == "attempt"
    assert first_row["attempt_routing_proof"]["held"]
    assert first_row["per_cell_routing_observation"]["status"] == "proven"


def _without_pythonpath() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    return environment


@pytest.mark.skipif(
    os.name != "posix", reason="the harness entry points target the Linux cluster runtime"
)
def test_python_entry_points_run_without_pythonpath(
    synthetic_capture: Path,
    tmp_path: Path,
) -> None:
    first = tmp_path / "normalized.first.json"
    second = tmp_path / "normalized.second.json"
    for output in (first, second):
        completed = subprocess.run(
            [
                sys.executable,
                str(ANALYZER_PATH),
                "--capture-root",
                str(synthetic_capture),
                "--output",
                str(output),
            ],
            cwd=tmp_path,
            env=_without_pythonpath(),
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
    assert first.read_bytes() == second.read_bytes()

    manifest = tmp_path / "submitted.sha256"
    completed = subprocess.run(
        [
            sys.executable,
            str(HASHER_PATH),
            "--root",
            str(HARNESS),
            "--output",
            str(manifest),
        ],
        cwd=tmp_path,
        env=_without_pythonpath(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert manifest.read_text(encoding="utf-8") == hasher.manifest_text(HARNESS)

    net_root = tmp_path / "synthetic-net-class"
    statistics = net_root / "hsn0" / "statistics"
    statistics.mkdir(parents=True)
    (statistics / "rx_bytes").write_text("1\n", encoding="utf-8")
    (statistics / "tx_bytes").write_text("2\n", encoding="utf-8")
    snapshot_output = tmp_path / "synthetic-counter-output"
    completed = subprocess.run(
        [
            sys.executable,
            str(SNAPSHOT_PATH),
            "--out-dir",
            str(snapshot_output),
            "--tag",
            "synthetic",
            "--net-class-root",
            str(net_root),
            "--interfaces",
            "hsn0",
        ],
        cwd=tmp_path,
        env={**_without_pythonpath(), "TRAF77_EVIDENCE_CLASS": "synthetic"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    snapshot_rows = list(snapshot_output.glob("*.jsonl"))
    assert len(snapshot_rows) == 1
    assert json.loads(snapshot_rows[0].read_text(encoding="utf-8"))[
        "evidence_class"
    ] == "synthetic"
