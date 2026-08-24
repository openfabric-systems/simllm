"""Strict, hardware-independent validation for calibration record files."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import CanonicalError, canonical_bytes, canonical_sha256
from .record_types import RecordObject
from .store import DEFAULT_MAX_OBJECT_BYTES


class CalibrationValidationError(ValueError):
    """A record path, canonical object, or typed payload failed validation."""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Small data-valued result returned by the offline validator."""

    record_schema: str
    record_sha256: str
    size_bytes: int

    def to_obj(self) -> dict[str, Any]:
        """Return the stable CLI projection without embedding local paths."""

        return {
            "valid": True,
            "record_schema": self.record_schema,
            "record_sha256": self.record_sha256,
            "size_bytes": self.size_bytes,
        }


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path) -> None:
    absolute = _absolute_without_resolving(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise CalibrationValidationError(
                f"validation path traverses a symbolic link: {current}"
            )
        if not current.exists():
            break


def _read_regular_file(path: Path, *, max_record_bytes: int) -> bytes:
    if type(max_record_bytes) is not int or max_record_bytes <= 0:
        raise ValueError("max_record_bytes must be a positive integer")

    absolute = _absolute_without_resolving(path)
    _reject_symlink_components(absolute)
    try:
        before = absolute.lstat()
    except FileNotFoundError as error:
        raise CalibrationValidationError("validation path does not exist") from error
    if stat.S_ISLNK(before.st_mode):
        raise CalibrationValidationError("validation path must not be a symbolic link")
    if not stat.S_ISREG(before.st_mode):
        raise CalibrationValidationError("validation path must be one regular record file")
    if before.st_size > max_record_bytes:
        raise CalibrationValidationError(
            f"record has {before.st_size} bytes; limit is {max_record_bytes}"
        )

    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise CalibrationValidationError(f"cannot open validation record: {error}") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise CalibrationValidationError(
                "validation path changed to a non-regular file"
            )
        if opened.st_size > max_record_bytes:
            raise CalibrationValidationError(
                f"record has {opened.st_size} bytes; limit is {max_record_bytes}"
            )
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise CalibrationValidationError("record changed while it was read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CalibrationValidationError("record grew while it was read")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    identity_before = (before.st_dev, before.st_ino)
    identity_opened = (opened.st_dev, opened.st_ino)
    if identity_before != identity_opened or opened.st_size != after.st_size:
        raise CalibrationValidationError("record identity changed while it was read")
    return b"".join(chunks)


def _typed_handlers() -> dict[str, Callable[[object], object]]:
    from simllm.compute.device_model_io import (
        TYPED_RECORD_READERS as DEVICE_MODEL_READERS,
    )

    from .bindings import TYPED_RECORD_READERS as BINDING_READERS
    from .doctor import TYPED_RECORD_READERS as DOCTOR_READERS
    from .model_inventory import TYPED_RECORD_READERS as INVENTORY_READERS

    handlers: dict[str, Callable[[object], object]] = {}
    for source in (
        DEVICE_MODEL_READERS,
        BINDING_READERS,
        DOCTOR_READERS,
        INVENTORY_READERS,
    ):
        overlap = handlers.keys() & source.keys()
        if overlap:
            raise RuntimeError(f"duplicate typed record readers for {sorted(overlap)!r}")
        handlers.update(source)
    return handlers


def validate_typed_record(value: Mapping[str, Any]) -> object:
    """Strictly decode one known schema-bearing record without hardware imports."""

    schema = value.get("schema")
    if not isinstance(schema, str):
        raise CalibrationValidationError("record.schema must be a string")
    handler = _typed_handlers().get(schema)
    if handler is None:
        raise CalibrationValidationError(f"unsupported calibration record schema {schema!r}")
    try:
        typed = handler(value)
    except (TypeError, ValueError) as error:
        raise CalibrationValidationError(str(error)) from error
    to_obj = getattr(typed, "to_obj", None)
    if not callable(to_obj):
        raise CalibrationValidationError(
            f"typed reader for {schema!r} returned no object projection"
        )
    if canonical_bytes(to_obj()) != canonical_bytes(value):
        raise CalibrationValidationError(
            f"typed reader for {schema!r} did not preserve the exact record"
        )
    from simllm.compute.device_model import DeviceModel

    if isinstance(typed, DeviceModel):
        try:
            typed.validate_registry_sha256(
                canonical_sha256(typed.resource_registry.to_obj())
            )
        except (TypeError, ValueError) as error:
            raise CalibrationValidationError(str(error)) from error
    return typed


def validate_path(
    path: str | os.PathLike[str],
    *,
    max_record_bytes: int = DEFAULT_MAX_OBJECT_BYTES,
) -> ValidationResult:
    """Validate one canonical typed record from a safe regular file path."""

    raw = _read_regular_file(Path(path), max_record_bytes=max_record_bytes)
    try:
        record = RecordObject.from_bytes(raw)
        validate_typed_record(record.value)
    except (CanonicalError, CalibrationValidationError, RuntimeError) as error:
        raise CalibrationValidationError(f"invalid calibration record: {error}") from error
    return ValidationResult(
        record_schema=record.schema,
        record_sha256=record.record_id,
        size_bytes=len(record.canonical),
    )


__all__ = [
    "CalibrationValidationError",
    "ValidationResult",
    "validate_path",
    "validate_typed_record",
]
