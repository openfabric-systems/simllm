"""Stable offline calibration command surface."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

_UNIMPLEMENTED = {
    "run": (
        "no collector or simulator backend is installed; select a build with "
        "a declared local backend"
    ),
    "pack": "package creation is not implemented in this build",
    "submit": (
        "authenticated data-only submission is not implemented in this build"
    ),
}


class CommandUnavailable(RuntimeError):
    """Raised when a stable command has no enabled implementation."""


def build_parser() -> argparse.ArgumentParser:
    """Build the hardware-independent command parser."""

    parser = argparse.ArgumentParser(
        prog="simllm-calibrate",
        description="Build and validate SimLLM device calibration evidence offline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "doctor",
        help="report locally available collector and simulator capabilities",
    )

    extract_parser = subparsers.add_parser(
        "extract",
        help="extract a model kernel inventory through a pinned framework",
    )
    extract_parser.add_argument("--framework", choices=("vllm", "sglang"), required=True)
    extract_parser.add_argument("--suite", default="transformer-dag-v1")
    extract_parser.add_argument("--suite-root", type=Path)
    extract_parser.add_argument("--checkpoint-root", type=Path, required=True)
    extract_parser.add_argument("--step-records", type=Path, required=True)
    extract_parser.add_argument("--output-root", type=Path, required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="run a declared local collector or offline simulator",
    )
    run_parser.add_argument("--suite-root", type=Path)

    validate_parser = subparsers.add_parser(
        "validate",
        help="validate a calibration object or release without hardware",
    )
    validate_parser.add_argument("path", type=Path)

    pack_parser = subparsers.add_parser(
        "pack",
        help="build a deterministic data-only contribution archive",
    )
    pack_parser.add_argument("--registry-root", type=Path)

    submit_parser = subparsers.add_parser(
        "submit",
        help="review and explicitly open a data-only contribution",
    )
    submit_parser.add_argument("--registry-root", type=Path)

    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one command and return a process exit status."""

    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    arguments = build_parser().parse_args(argv)
    try:
        return _dispatch(arguments, output)
    except CommandUnavailable as error:
        print(f"simllm-calibrate: {arguments.command}: {error}", file=errors)
        return 2


def _dispatch(arguments: argparse.Namespace, output: TextIO) -> int:
    if arguments.command == "doctor":
        _write_json(_inert_doctor_record(), output)
        return 0
    if arguments.command == "extract":
        _write_json(_call_extractor(arguments), output)
        return 0
    if arguments.command == "validate":
        result = _call_validator(arguments.path)
        if result is not None:
            _write_json(result, output)
        return 0
    try:
        reason = _UNIMPLEMENTED[arguments.command]
    except KeyError as error:
        raise AssertionError(f"unhandled command: {arguments.command}") from error
    raise CommandUnavailable(reason)


def _inert_doctor_record() -> dict[str, Any]:
    from .doctor import DoctorRecord

    return DoctorRecord.blocked(
        reason=(
            "This installation provides backend-neutral protocols only; concrete "
            "CUDA, ROCm and offline simulator probes are not installed."
        )
    ).to_obj()


def _call_validator(path: Path) -> Any:
    validator = _load_validator()
    try:
        result = validator(path)
    except (OSError, ValueError) as error:
        raise CommandUnavailable(f"validation failed: {error}") from error
    to_obj = getattr(result, "to_obj", None)
    return to_obj() if callable(to_obj) else result


def _call_extractor(arguments: argparse.Namespace) -> dict[str, Any]:
    if (
        not isinstance(arguments.suite, str)
        or not arguments.suite
        or arguments.suite in {".", ".."}
        or "/" in arguments.suite
        or "\\" in arguments.suite
    ):
        raise CommandUnavailable("suite must be one local directory name")
    try:
        from .registry import resolve_suite_root
        from .store import ObjectStore

        selection = resolve_suite_root(arguments.suite_root)
        suite_file = selection.root.joinpath(
            "suites", arguments.suite, "suite.json"
        )
        if not suite_file.is_file():
            raise ValueError(f"suite {arguments.suite!r} has no suite.json")
        module = importlib.import_module(
            f"simllm.adapters.{arguments.framework}.extraction"
        )
        extractor = getattr(module, "extract", None)
        if extractor is None or not callable(extractor):
            raise ValueError(
                f"framework {arguments.framework!r} has no extraction driver"
            )
        inventory = extractor(
            suite_raw=suite_file.read_bytes(),
            checkpoint_root=arguments.checkpoint_root,
            step_records_path=arguments.step_records,
        )
        record = ObjectStore(arguments.output_root).write(inventory.record)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise CommandUnavailable(f"extraction failed: {error}") from error
    return {
        "schema": "simllm-model-kernel-inventory-extraction-v1",
        "framework": inventory.framework.framework_id,
        "model": inventory.model.name,
        "case_count": len(inventory.cases),
        "record_schema": record.schema,
        "record_sha256": record.record_id,
        "size_bytes": len(record.canonical),
    }


def _load_validator() -> Callable[[Path], Any]:
    try:
        module = importlib.import_module("simllm.calibration.validation")
    except ModuleNotFoundError as error:
        if error.name != "simllm.calibration.validation":
            raise
        raise CommandUnavailable(
            "the object validator is unavailable in this installation"
        ) from error
    validator = getattr(module, "validate_path", None)
    if validator is None or not callable(validator):
        raise CommandUnavailable(
            "simllm.calibration.validation.validate_path is unavailable"
        )
    return validator


def _write_json(value: Any, output: TextIO) -> None:
    from .canonical import canonical_bytes

    output.write(canonical_bytes(value).decode("utf-8"))
    output.write("\n")
