from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPOSITORY_ROOT / "pyproject.toml"


def test_default_import_does_not_load_calibration_or_change_public_identity() -> None:
    source = """
import json
import sys
import simllm
print(json.dumps({
    "module": simllm.__name__,
    "version": simllm.__version__,
    "calibration_modules": sorted(
        name for name in sys.modules if name.startswith("simllm.calibration")
    ),
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "calibration_modules": [],
        "module": "simllm",
        "version": "0.0.1",
    }


def test_calibration_package_attributes_are_lazy() -> None:
    source = """
import json
import sys
import simllm.calibration
print(json.dumps(sorted(
    name for name in sys.modules if name.startswith("simllm.calibration")
)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == ["simllm.calibration"]


def test_module_entry_point_exposes_the_stable_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "simllm.calibration", "--help"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "{doctor,extract,run,validate,pack,submit}" in completed.stdout


def test_console_script_is_declared() -> None:
    configuration = PYPROJECT.read_text(encoding="utf-8")

    assert (
        "[project.scripts]\n"
        'simllm-calibrate = "simllm.calibration.cli:main"\n'
    ) in configuration


def test_wheel_contains_the_lazy_package_and_no_checkout_only_roots(
    tmp_path: Path,
) -> None:
    pip = shutil.which("pip")
    if pip is None:
        pytest.skip("wheel test requires a pip frontend")
    wheel_directory = tmp_path / "wheel"
    completed = subprocess.run(
        [
            pip,
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_directory),
            str(REPOSITORY_ROOT),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheels = tuple(wheel_directory.glob("simllm-*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
        entry_points = archive.read(
            next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        ).decode("utf-8")

    required_modules = {
        "simllm/calibration/__init__.py",
        "simllm/calibration/__main__.py",
        "simllm/calibration/bindings.py",
        "simllm/calibration/canonical.py",
        "simllm/calibration/cli.py",
        "simllm/calibration/doctor.py",
        "simllm/calibration/identity.py",
        "simllm/calibration/manifests.py",
        "simllm/calibration/protocols.py",
        "simllm/calibration/record_types.py",
        "simllm/calibration/registry.py",
        "simllm/calibration/splits.py",
        "simllm/calibration/store.py",
        "simllm/calibration/validation.py",
        "simllm/compute/device_model.py",
        "simllm/compute/device_model_io.py",
    }
    assert required_modules <= names
    assert not any(name.startswith("offline/") for name in names)
    assert not any(name.startswith("devices/") for name in names)
    assert not any(name.startswith("third_party/") for name in names)
    assert "simllm-calibrate = simllm.calibration.cli:main" in entry_points
