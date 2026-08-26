from __future__ import annotations

import importlib.util
import json
from pathlib import Path, PureWindowsPath

STUDY = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "pd_session_kernel_cycle_v1"
)


def _runner():
    spec = importlib.util.spec_from_file_location(
        "pd_session_kernel_cycle_study",
        STUDY / "run_study.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_study_pins_the_expectation_and_implementation_commits() -> None:
    runner = _runner()

    assert runner.EXPECTATION_COMMIT == (
        "fda6eed557aef037bf1794da1c1d8556a10a1ee0"
    )
    assert runner.IMPLEMENTATION_COMMIT == (
        "6817019376d153be2a4b6cdd972bbec36dfa23e6"
    )


def test_frozen_grid_has_the_exact_signed_movement_oracle() -> None:
    freeze = json.loads((STUDY / "expectations.json").read_text(encoding="utf-8"))
    rows = freeze["movement_oracle"]["cells"]

    assert len(rows) == 4
    assert [row["signed_ttft_delta_ps"] for row in rows] == [
        0,
        0,
        1_972_200_000,
        1_972_200_000,
    ]
    assert [row["signed_tpot_delta_ps"] for row in rows] == [0, 0, 0, 0]


def test_study_renders_command_paths_with_posix_separators() -> None:
    runner = _runner()

    assert runner.render_cli_path(PureWindowsPath("C:/study/run")) == "C:/study/run"
