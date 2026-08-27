"""Byte locks and a fast native subset for the LogGOPSim ideal study."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import pytest

from simllm.backends import LogGopsimConfig, build_loggopsim_command
from simllm.goal import find_txt2bin, to_binary

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STUDY_DIR = _REPO_ROOT / "examples" / "loggopsim_ideal_v1"
_PINNED_BINARY_SHA256 = (
    "7e0f13ee3c87a20e9d2e94dbbd74c46075fd03df2f1b04d1ed9739c43ee0a2bf"
)
_ARTIFACT_SHA256 = {
    "RESULTS.md": "d12f2c0033d713415c3d58978a5a5c7f474a23cb4f00ba84c4b8717b2c9be7e5",
    "results.csv": "18b8f9f20e40d497478dae26313d24f464bcd183026a0e1bec45351da1358dc1",
    "results.json": "02a7985de910059c44766b4e2057a81f180eed014bfc8f4bf3c2a465329dfac3",
    "run_study.py": "f30f043cc37ace75352bae1c92d5836fb400ac195bab49e4f03754321beb7b2c",
}
_MAX_FINISH_RE = re.compile(
    rb"^Maximum finishing time at host \d+: (\d+)", re.MULTILINE
)
_HOST_TIME_RE = re.compile(rb"^Host \d+: (\d+)\s*$", re.MULTILINE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name, expected", sorted(_ARTIFACT_SHA256.items()))
def test_study_artifact_bytes(name: str, expected: str) -> None:
    assert _sha256(_STUDY_DIR / name) == expected


def test_study_record_keeps_evidence_classes_and_fatal_guards_separate() -> None:
    record = json.loads((_STUDY_DIR / "results.json").read_text(encoding="utf-8"))
    assert record["verdict"] == "PASS"
    assert record["findings"] == []
    assert record["exact_oracles"]["passed"] == 28
    assert record["exact_oracles"]["denominator"] == 28
    assert record["live_chain"]["passed"] == 3
    assert record["live_chain"]["denominator"] == 3
    assert record["wall_time"]["passed"] == 3
    assert record["wall_time"]["denominator"] == 3
    assert [row["id"] for row in record["fatal_guards"]] == [
        "FG-1",
        "FG-2",
        "FG-3",
        "FG-4",
        "FG-5",
        "FG-6",
    ]
    assert all(row["held"] for row in record["fatal_guards"])
    assert all(row["mutation_negative_control"] for row in record["fatal_guards"])


def _maximum_host_finish(stdout: bytes) -> int:
    maximum = _MAX_FINISH_RE.search(stdout)
    if maximum is not None:
        return int(maximum.group(1))
    hosts = [int(value) for value in _HOST_TIME_RE.findall(stdout)]
    assert hosts, "LogGOPSim stdout has no host finishing time"
    return max(hosts)


def _run_twice(argv: list[str]) -> tuple[bytes, int]:
    while time.time() % 1.0 > 0.02:
        time.sleep(0.002)
    outputs: list[bytes] = []
    for _ in range(2):
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr.decode(errors="replace")
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1]
    return outputs[0], _maximum_host_finish(outputs[0])


def test_fast_native_quantization_and_ring_subset(tmp_path: Path) -> None:
    configured_binary = os.environ.get("SIMLLM_LOGGOPSIM")
    if not configured_binary:
        pytest.skip("SIMLLM_LOGGOPSIM is unset; CI does not install pinned LogGOPSim")
    binary = Path(configured_binary)
    assert binary.is_file(), f"SIMLLM_LOGGOPSIM is not a file: {binary}"
    assert _sha256(binary) == _PINNED_BINARY_SHA256

    txt2bin = find_txt2bin()
    if txt2bin is None:
        pytest.skip("SIMLLM_TXT2BIN is unavailable; the native subset needs txt2bin")

    cells = (
        (
            "pair-51",
            LogGopsimConfig(
                goal_bin=tmp_path / "pair-51.bin",
                latency_ns=0,
                overhead_ns=0,
                message_gap_ns=0,
                byte_gap_ns=0.02,
                byte_gap_ns_string="0.02",
                byte_overhead_ns=0,
                rendezvous_threshold_bytes=10_000_000,
            ),
            1,
        ),
        (
            "ring-allreduce-p4-s17",
            LogGopsimConfig(
                goal_bin=tmp_path / "ring-allreduce-p4-s17.bin",
                latency_ns=100,
                overhead_ns=10,
                message_gap_ns=7,
                byte_gap_ns=3.0,
                byte_gap_ns_string="3.0",
                byte_overhead_ns=0,
                rendezvous_threshold_bytes=50,
            ),
            1050,
        ),
    )
    for goal_name, config, expected_ns in cells:
        to_binary(
            _STUDY_DIR / "goals" / f"{goal_name}.goal",
            config.goal_bin,
            tool=txt2bin,
        )
        argv = build_loggopsim_command(binary, config)
        stdout, observed_ns = _run_twice(argv)
        assert stdout
        assert argv[argv.index("-G") + 1] == config.byte_gap_ns_string
        assert observed_ns == expected_ns
