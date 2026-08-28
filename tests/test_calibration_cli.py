from __future__ import annotations

import io
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import simllm.calibration.cli as cli_module
from simllm.calibration.cli import build_parser, main
from simllm.calibration.doctor import DoctorRecord, DoctorState
from simllm.calibration.record_types import RecordObject

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_parser_exposes_exact_stable_command_names() -> None:
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if action.dest == "command"
    )

    assert tuple(subparsers.choices) == (
        "doctor",
        "extract",
        "run",
        "validate",
        "pack",
        "submit",
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["doctor", "--help"],
        ["extract", "--help"],
        ["run", "--help"],
        ["validate", "--help"],
        ["pack", "--help"],
        ["submit", "--help"],
    ],
)
def test_help_needs_no_hardware_or_simulator_package(arguments: list[str]) -> None:
    source = (
        "import sys; "
        "from simllm.calibration.cli import main; "
        f"argv={arguments!r}; "
        "\ntry:\n main(argv)\nexcept SystemExit as error:\n "
        "assert error.code == 0\n"
        "forbidden=('torch','cupy','cuda','rocm','rocprofiler','accel_sim'); "
        "assert not any(name == prefix or name.startswith(prefix + '.') "
        "for name in sys.modules for prefix in forbidden)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def test_inert_doctor_emits_one_deterministic_typed_record() -> None:
    first = io.StringIO()
    second = io.StringIO()

    assert main(["doctor"], stdout=first) == 0
    assert main(["doctor"], stdout=second) == 0

    assert first.getvalue() == second.getvalue()
    record = json.loads(first.getvalue())
    assert DoctorRecord.from_obj(record).state is DoctorState.BLOCKED
    assert record == {
        "capabilities": [],
        "reason": (
            "This installation provides backend-neutral protocols only; concrete "
            "CUDA, ROCm and offline simulator probes are not installed."
        ),
        "reason_code": "no-concrete-backends",
        "schema": "simllm-calibration-doctor-v1",
        "state": "blocked",
    }


def test_inert_doctor_loads_no_hardware_or_simulator_runtime() -> None:
    source = (
        "import sys; from simllm.calibration.cli import main; "
        "assert main(['doctor']) == 0; "
        "forbidden=('torch','cupy','cuda','rocm','rocprofiler','accel_sim'); "
        "assert not any(name == prefix or name.startswith(prefix + '.') "
        "for name in sys.modules for prefix in forbidden)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["state"] == "blocked"


def test_extract_lazily_resolves_suite_and_writes_one_content_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_root = tmp_path / "calibration"
    suite_file = suite_root / "suites" / "transformer-dag-v1" / "suite.json"
    suite_file.parent.mkdir(parents=True)
    suite_file.write_bytes(b"suite bytes")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output_root = tmp_path / "objects"
    steps = tmp_path / "steps.jsonl"
    calls: list[dict[str, object]] = []
    module = types.ModuleType("simllm.adapters.vllm.extraction")
    record = RecordObject.from_value({"schema": "test-inventory-v1", "value": 1})

    def extract(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            framework=SimpleNamespace(framework_id="vllm"),
            model=SimpleNamespace(name="unit/model"),
            cases=(object(),),
            record=record,
        )

    module.extract = extract
    real_import = cli_module.importlib.import_module

    def import_with_driver(name: str):
        if name == module.__name__:
            return module
        return real_import(name)

    monkeypatch.setattr(cli_module.importlib, "import_module", import_with_driver)
    output = io.StringIO()
    status = main(
        [
            "extract",
            "--framework",
            "vllm",
            "--suite-root",
            str(suite_root),
            "--checkpoint-root",
            str(checkpoint),
            "--step-records",
            str(steps),
            "--output-root",
            str(output_root),
        ],
        stdout=output,
    )

    assert status == 0
    assert calls == [
        {
            "suite_raw": b"suite bytes",
            "checkpoint_root": checkpoint,
            "step_records_path": steps,
        }
    ]
    assert (output_root / f"{record.record_id}.json").read_bytes() == record.canonical
    assert json.loads(output.getvalue()) == {
        "schema": "simllm-model-kernel-inventory-extraction-v1",
        "framework": "vllm",
        "model": "unit/model",
        "case_count": 1,
        "record_schema": "test-inventory-v1",
        "record_sha256": record.record_id,
        "size_bytes": len(record.canonical),
    }


def test_extract_rejection_writes_no_content_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_root = tmp_path / "calibration"
    suite_file = suite_root / "suites" / "transformer-dag-v1" / "suite.json"
    suite_file.parent.mkdir(parents=True)
    suite_file.write_bytes(b"suite bytes")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output_root = tmp_path / "objects"
    module = types.ModuleType("simllm.adapters.vllm.extraction")

    def reject(**kwargs: object) -> None:
        raise RuntimeError("partial capture")

    module.extract = reject
    real_import = cli_module.importlib.import_module
    monkeypatch.setattr(
        cli_module.importlib,
        "import_module",
        lambda name: module if name == module.__name__ else real_import(name),
    )
    errors = io.StringIO()
    status = main(
        [
            "extract",
            "--framework",
            "vllm",
            "--suite-root",
            str(suite_root),
            "--checkpoint-root",
            str(checkpoint),
            "--step-records",
            str(tmp_path / "steps.jsonl"),
            "--output-root",
            str(output_root),
        ],
        stderr=errors,
    )

    assert status == 2
    assert "partial capture" in errors.getvalue()
    assert not output_root.exists()


def test_ordinary_import_and_extract_help_load_no_framework_runtime() -> None:
    source = (
        "import sys; import simllm; "
        "from simllm.calibration.cli import main; "
        "\ntry:\n main(['extract', '--help'])\nexcept SystemExit as error:\n "
        "assert error.code == 0\n"
        "assert 'vllm' not in sys.modules; assert 'sglang' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("command", ["pack", "submit"])
def test_unimplemented_commands_fail_without_side_effects(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = tuple(tmp_path.iterdir())
    output = io.StringIO()
    errors = io.StringIO()
    root_option = "--suite-root" if command == "run" else "--registry-root"

    status = main(
        [command, root_option, str(tmp_path / "must-not-be-created")],
        stdout=output,
        stderr=errors,
    )

    assert status == 2
    assert output.getvalue() == ""
    assert errors.getvalue().startswith(f"simllm-calibrate: {command}:")
    assert tuple(tmp_path.iterdir()) == before


def test_run_requires_explicit_request_target_and_output_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = tuple(tmp_path.iterdir())
    errors = io.StringIO()

    status = main(["run", "--suite-root", str(tmp_path / "suite")], stderr=errors)

    assert status == 2
    assert "requires --request, --target, --output-root" in errors.getvalue()
    assert tuple(tmp_path.iterdir()) == before


def test_run_lazily_calls_external_local_shard_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = tmp_path / "request.json"
    request.write_bytes(b"request bytes")
    output_root = tmp_path / "capture"
    calls: list[dict[str, object]] = []
    module = types.ModuleType("simllm.calibration.local_shard")

    class Run:
        def to_obj(self) -> dict[str, object]:
            return {
                "schema": "simllm-local-shard-kernel-capture-run-v1",
                "request_sha256": "1" * 64,
                "result_sha256": "2" * 64,
                "kernel_count": 3,
            }

    def run_local_shard_capture(raw: bytes, **kwargs: object) -> Run:
        calls.append({"raw": raw, **kwargs})
        return Run()

    module.run_local_shard_capture = run_local_shard_capture
    monkeypatch.setitem(sys.modules, module.__name__, module)
    output = io.StringIO()

    status = main(
        [
            "run",
            "--request",
            str(request),
            "--target",
            "framework-target",
            "--target-arg",
            "target-script.py",
            "--output-root",
            str(output_root),
        ],
        stdout=output,
    )

    assert status == 0
    assert calls == [
        {
            "raw": b"request bytes",
            "target": "framework-target",
            "target_args": ("target-script.py",),
            "output_root": output_root,
        }
    ]
    assert json.loads(output.getvalue())["kernel_count"] == 3


def test_validate_fails_cleanly_when_validator_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = cli_module.importlib.import_module

    def import_without_validator(name: str):
        if name == "simllm.calibration.validation":
            raise ModuleNotFoundError("validation module is absent", name=name)
        return real_import(name)

    monkeypatch.setattr(cli_module.importlib, "import_module", import_without_validator)
    output = io.StringIO()
    errors = io.StringIO()

    status = main(
        ["validate", str(tmp_path / "object.json")],
        stdout=output,
        stderr=errors,
    )

    assert status == 2
    assert output.getvalue() == ""
    assert "object validator is unavailable" in errors.getvalue()


def test_validate_lazily_calls_the_object_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "object.json"
    calls: list[Path] = []
    module = types.ModuleType("simllm.calibration.validation")

    class Report:
        def to_obj(self) -> dict[str, object]:
            return {"schema": "test-validation-v1", "valid": True}

    def validate_path(path: Path) -> Report:
        calls.append(path)
        return Report()

    module.validate_path = validate_path
    monkeypatch.setitem(sys.modules, module.__name__, module)
    output = io.StringIO()

    assert main(["validate", str(target)], stdout=output) == 0
    assert calls == [target]
    assert json.loads(output.getvalue()) == {
        "schema": "test-validation-v1",
        "valid": True,
    }


def test_validate_reports_reader_errors_without_writing_an_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "bad.json"
    module = types.ModuleType("simllm.calibration.validation")

    def validate_path(path: Path) -> None:
        raise ValueError(f"invalid object at {path.name}")

    module.validate_path = validate_path
    monkeypatch.setitem(sys.modules, module.__name__, module)
    errors = io.StringIO()

    status = main(["validate", str(target)], stderr=errors)

    assert status == 2
    assert "validation failed: invalid object at bad.json" in errors.getvalue()
    assert not target.exists()
