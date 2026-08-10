import importlib.util
import tempfile
from pathlib import Path

STUDY_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "rnic_device_v1"
    / "run_rnic_device_v1.py"
)
SPEC = importlib.util.spec_from_file_location("run_rnic_device_v1", STUDY_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_rnic_device_v1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_rnic_device_v1)


def test_device_study_build_dir_override_is_used_exactly(
    monkeypatch,
) -> None:
    override = "relative/device-build"
    monkeypatch.setenv(run_rnic_device_v1.BUILD_DIR_ENV, override)

    assert run_rnic_device_v1._default_build_dir(Path("ignored")) == Path(
        override
    )


def test_device_study_default_build_dir_is_stable_and_worktree_specific(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(run_rnic_device_v1.BUILD_DIR_ENV, raising=False)
    first_output = tmp_path / "first" / "examples" / "rnic_device_v1"
    second_output = tmp_path / "second" / "examples" / "rnic_device_v1"

    first = run_rnic_device_v1._default_build_dir(first_output)
    repeated = run_rnic_device_v1._default_build_dir(first_output)
    second = run_rnic_device_v1._default_build_dir(second_output)

    assert first == repeated
    assert first != second
    assert first.is_relative_to(Path(tempfile.gettempdir()))
