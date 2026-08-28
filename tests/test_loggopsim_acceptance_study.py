"""Locks and CI subsets for the LogGOPSim acceptance study."""

from __future__ import annotations

import hashlib
import json
import os
from argparse import Namespace
from pathlib import Path

import pytest

from examples.loggopsim_acceptance_v1 import run_study as study

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples" / "loggopsim_acceptance_v1"
FROZEN_EXPECTATIONS_SHA256 = (
    "ae189d42cd5889152a101d63feb86cb44004d67700c102e9a72a0deacbecd832"
)
PINNED_LOGGOPSIM_SHA256 = (
    "7e0f13ee3c87a20e9d2e94dbbd74c46075fd03df2f1b04d1ed9739c43ee0a2bf"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_expectations_bytes_are_unchanged() -> None:
    assert _sha256(STUDY_DIR / "expectations.md") == FROZEN_EXPECTATIONS_SHA256


def test_runner_imports_no_other_example_module() -> None:
    source = (STUDY_DIR / "run_study.py").read_text(encoding="utf-8")
    assert "from examples" not in source
    assert "import examples" not in source


def test_pinned_cells_include_the_mixed_incast_partition() -> None:
    cells = study._load_cells()
    assert len(cells) == 12
    batch32 = next(cell for cell in cells if cell.cell_id == "incast-b32")
    assert batch32.payloads == (
        47_302_429,
        47_302_429,
        47_302_429,
        47_302_429,
        47_302_428,
        47_302_428,
        47_302_428,
        47_302_428,
    )
    assert sum(batch32.payloads) == 378_419_428


def test_family_c_refusal_cell_is_binary_free_and_stops_before_execution() -> None:
    incast = next(cell for cell in study._load_cells() if cell.cell_id == "incast-b1")

    row = study._run_refusal_cell(study._render_goal_text(incast))

    assert row["id"] == "C-1"
    assert row["passed"] is True
    assert row["observed"]["refused"] is True
    assert row["observed"]["execution_calls"] == 0
    assert "receiver per-byte gap is unmodeled" in row["observed"]["diagnostic"]
    assert "examples/frontier_ladder_v1/RESULTS.md" in row["observed"][
        "diagnostic"
    ]
    mutation = row["mutation_control"]
    assert mutation["mutant_acknowledge_fan_in"] is True
    assert mutation["native_boundary_reached"] is True


def test_attempts_are_append_only_and_require_prior_verdicts(tmp_path: Path) -> None:
    args = Namespace(
        htsim_rnic=tmp_path / "htsim_rnic",
        loggopsim=tmp_path / "LogGOPSim",
        txt2bin=tmp_path / "txt2bin",
    )
    run_root = tmp_path / "attempts"
    first = study._begin_attempt(run_root, args)
    assert first.name == "attempt-1"
    with pytest.raises(SystemExit, match="verdict records are missing: attempt-1"):
        study._begin_attempt(run_root, args)
    study._write_json(first / "verdict.json", {"verdict": "ERROR"})
    second = study._begin_attempt(run_root, args)
    assert second.name == "attempt-2"
    assert first.is_dir()


def _native_family_c_tools() -> tuple[Path, Path]:
    configured_loggopsim = os.environ.get("SIMLLM_LOGGOPSIM")
    if not configured_loggopsim:
        pytest.skip(
            "SIMLLM_LOGGOPSIM is unset; the executed Family C subset needs "
            "the pinned LogGOPSim binary"
        )
    loggopsim = Path(configured_loggopsim)
    if not loggopsim.is_file():
        pytest.skip(
            f"SIMLLM_LOGGOPSIM is unavailable; executed Family C skipped: {loggopsim}"
        )
    configured_txt2bin = os.environ.get("SIMLLM_TXT2BIN")
    if not configured_txt2bin:
        pytest.skip(
            "SIMLLM_TXT2BIN is unset; the executed Family C subset needs txt2bin"
        )
    txt2bin = Path(configured_txt2bin)
    if not txt2bin.is_file():
        pytest.skip(
            f"SIMLLM_TXT2BIN is unavailable; executed Family C skipped: {txt2bin}"
        )
    assert _sha256(loggopsim) == PINNED_LOGGOPSIM_SHA256
    return loggopsim, txt2bin


def test_executed_family_c_subset_when_native_tools_are_available(
    tmp_path: Path,
) -> None:
    loggopsim, txt2bin = _native_family_c_tools()
    cells = study._load_cells()
    selected = tuple(
        cell for cell in cells if cell.cell_id in {"serialized-b1", "incast-b1"}
    )
    prepared = study._prepare_goals(selected, tmp_path, txt2bin)
    recorder = study.NativeEvidenceRecorder(tmp_path)

    family = study._run_family_c(
        prepared,
        binary=loggopsim,
        recorder=recorder,
        timeout_s=60,
    )

    assert (family["passed"], family["denominator"]) == (3, 3)
    rows = {row["id"]: row for row in family["rows"]}
    assert rows["C-1"]["observed"]["execution_calls"] == 0
    assert rows["C-2"]["observed"]["fan_in"]["fan_in_detected"] is True
    assert rows["C-2"]["observed"]["fan_in"]["acknowledged"] is True
    assert rows["C-3"]["observed"]["differing_fields"] == [
        "acknowledge_fan_in_option"
    ]
    assert len(list((tmp_path / "native").rglob("*.stdout"))) == 3
    assert len(list((tmp_path / "native").rglob("*.json"))) == 3
    serialized = json.dumps(family)
    assert "/data3/" not in serialized
    assert "/home/" not in serialized
