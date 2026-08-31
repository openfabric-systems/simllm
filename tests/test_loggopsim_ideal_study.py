"""Byte locks and a fast native subset for the LogGOPSim ideal study."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from argparse import Namespace
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
    "RESULTS.md": "e285d9e4d637f5b425671d1ec5da7f8a9c0caf706f8a687365aaf68160575c96",
    "results.csv": "32674c5afa0186176f9e6cf95578818fb6ef4425c07a9da97f23dc69a7d9fdb1",
    "results.json": "8d84d4d8ebc2a46953d271a8ef92e3509495a5c3c7953a8a87632ce4b62759a1",
    "run_study.py": "9fcfec3cc0972c30bc44b3218688a283a97b4d2fbf341ca3902661de4dacd6e3",
}
_FROZEN_EXPECTATIONS_SHA256 = (
    "934ee355e4d5a376d1eecdb1d0e62f6e4f7acfd9ada93def5ba1bcf8fa8508ff"
)
_MAX_FINISH_RE = re.compile(
    rb"^Maximum finishing time at host \d+: (\d+)", re.MULTILINE
)
_HOST_TIME_RE = re.compile(rb"^Host \d+: (\d+)\s*$", re.MULTILINE)


def _load_study_module():
    spec = importlib.util.spec_from_file_location(
        "loggopsim_ideal_run_study", _STUDY_DIR / "run_study.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


study = _load_study_module()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name, expected", sorted(_ARTIFACT_SHA256.items()))
def test_study_artifact_bytes(name: str, expected: str) -> None:
    assert _sha256(_STUDY_DIR / name) == expected


def test_frozen_expectations_bytes_are_unchanged() -> None:
    assert _sha256(_STUDY_DIR / "expectations.md") == _FROZEN_EXPECTATIONS_SHA256


def test_study_record_keeps_evidence_classes_and_fatal_guards_separate() -> None:
    record = json.loads((_STUDY_DIR / "results.json").read_text(encoding="utf-8"))
    assert record["verdict"] == "PASS"
    assert record["findings"] == []
    assert record["exact_oracles"]["passed"] == 30
    assert record["exact_oracles"]["denominator"] == 30
    assert record["exact_oracles"]["families"]["E1"]["passed"] == 11
    assert record["exact_oracles"]["families"]["E1"]["denominator"] == 11
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
    assert all(
        row["mutation_control"]["kind"] == "predicate-exercised"
        and row["mutation_control"]["rejected"]
        for row in record["fatal_guards"]
    )
    boundaries = {
        row["id"]: row
        for row in record["exact_oracles"]["rows"]
        if row["id"] in {"E1-S50", "E1-S51"}
    }
    assert boundaries["E1-S50"]["observed_host_finish_ns"] == {
        "0": 277,
        "1": 277,
    }
    assert boundaries["E1-S51"]["observed_host_finish_ns"] == {
        "0": 10,
        "1": 277,
    }
    assert record["run_history"]["prior_void_stdout_bytes_retained"] is False
    assert record["run_history"]["legacy_attempts_without_own_verdict"] == 5
    assert record["attempt_evidence"]["verdict"] == "verdict.json"
    serialized = json.dumps(record)
    assert "/data3/" not in serialized
    assert "/home/" not in serialized


def test_attempts_are_append_only_and_require_a_predecessor_verdict(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "attempts"
    args = Namespace(
        binary=tmp_path / "LogGOPSim",
        txt2bin=tmp_path / "txt2bin",
        results_json=None,
        results_csv=None,
    )
    first = study._begin_attempt(run_root, args)
    assert first.name == "attempt-1"
    with pytest.raises(SystemExit, match="verdict records are missing: attempt-1"):
        study._begin_attempt(run_root, args)
    study._write_json(first / "verdict.json", {"verdict": "ERROR"})
    second = study._begin_attempt(run_root, args)
    assert second.name == "attempt-2"
    assert first.is_dir()


def test_flag_only_invocation_reaches_live_sink_and_retains_attempt_evidence(
    tmp_path: Path,
) -> None:
    configured_binary = os.environ.get("SIMLLM_LOGGOPSIM")
    if not configured_binary:
        pytest.skip("SIMLLM_LOGGOPSIM is unset; CI does not install pinned LogGOPSim")
    binary = Path(configured_binary)
    txt2bin = find_txt2bin()
    if txt2bin is None:
        pytest.skip("SIMLLM_TXT2BIN is unavailable; the native study needs txt2bin")
    assert _sha256(binary) == _PINNED_BINARY_SHA256

    run_root = tmp_path / "flag-only"
    environment = os.environ.copy()
    environment.pop("SIMLLM_LOGGOPSIM", None)
    environment.pop("SIMLLM_TXT2BIN", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(_STUDY_DIR / "run_study.py"),
            "--run-dir",
            str(run_root),
            "--binary",
            str(binary),
            "--txt2bin",
            str(txt2bin),
        ],
        cwd=_REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    attempt = run_root / "attempt-1"
    record = json.loads((attempt / "verdict.json").read_text(encoding="utf-8"))
    assert record["verdict"] == "PASS"
    assert record["exact_oracles"]["passed"] == 30
    assert record["exact_oracles"]["denominator"] == 30
    boundaries = {
        row["id"]: row
        for row in record["exact_oracles"]["rows"]
        if row["id"] in {"E1-S50", "E1-S51"}
    }
    assert boundaries["E1-S50"]["observed_host_finish_ns"] == {
        "0": 277,
        "1": 277,
    }
    assert boundaries["E1-S51"]["observed_host_finish_ns"] == {
        "0": 10,
        "1": 277,
    }
    for guard in record["fatal_guards"]:
        assert guard["mutation_control"]["kind"] == "predicate-exercised"
        assert guard["mutation_control"]["rejected"] is True
    assert len(list((attempt / "native").rglob("*.stdout"))) == 85
    assert len(list((attempt / "native").rglob("*.json"))) == 85
    for row in record["exact_oracles"]["rows"] + record["wall_time"]["rows"]:
        assert row["argv"][0] == "LogGOPSim"
        assert not Path(row["argv"][row["argv"].index("-f") + 1]).is_absolute()
    live_argv = record["live_chain"]["invocation"]["argv"]
    assert live_argv[0] == "LogGOPSim"
    assert not Path(live_argv[live_argv.index("-f") + 1]).is_absolute()


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
