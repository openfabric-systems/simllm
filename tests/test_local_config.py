import subprocess
import sys
from pathlib import Path

import pytest

from simllm._local_config import path_from_env

VARIABLE = "SIMLLM_TEST_DATA_ROOT"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RNIC_STUDY = (
    REPOSITORY_ROOT / "examples" / "rnic_session_records_v1" / "run_study.py"
)


def test_path_from_env_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv(VARIABLE, raising=False)

    assert path_from_env(VARIABLE) is None


@pytest.mark.parametrize("value", ["", "   "])
def test_path_from_env_returns_none_when_blank(value, monkeypatch):
    monkeypatch.setenv(VARIABLE, value)

    assert path_from_env(VARIABLE) is None


def test_path_from_env_resolves_an_absolute_path(tmp_path, monkeypatch):
    configured = tmp_path / "nested" / "missing"
    monkeypatch.setenv(VARIABLE, f"  {configured}  ")

    assert path_from_env(VARIABLE) == configured.resolve(strict=False)


def test_path_from_env_expands_user_syntax(tmp_path, monkeypatch):
    if sys.platform == "win32":
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.delenv("HOMEDRIVE", raising=False)
        monkeypatch.delenv("HOMEPATH", raising=False)
    else:
        monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(VARIABLE, str(Path("~") / "raw-traces"))

    assert path_from_env(VARIABLE) == (tmp_path / "raw-traces").resolve(
        strict=False
    )


def test_path_from_env_rejects_a_relative_path(monkeypatch):
    monkeypatch.setenv(VARIABLE, str(Path("runs") / "study"))

    with pytest.raises(ValueError, match=rf"^{VARIABLE} must be an absolute path"):
        path_from_env(VARIABLE)


def test_explicit_runner_path_bypasses_invalid_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMLLM_DATA_ROOT", "relative-data-root")

    completed = subprocess.run(
        [
            sys.executable,
            str(RNIC_STUDY),
            "--out",
            str(tmp_path / "explicit-output"),
            "--check-only",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "registry check passed" in completed.stdout
