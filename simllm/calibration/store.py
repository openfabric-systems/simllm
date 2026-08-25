"""Path-safe local storage for immutable calibration record objects."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .canonical import CanonicalError, validate_sha256
from .record_types import RecordObject

DEFAULT_MAX_OBJECT_BYTES = 64 * 1024 * 1024


class ObjectStoreError(ValueError):
    """A local object store violates identity, layout or path safety."""


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path) -> None:
    absolute = _absolute_without_resolving(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ObjectStoreError(f"object-store path traverses symlink {current}")
        if not current.exists():
            break


class ObjectStore:
    """A directory of canonical ``<sha256>.json`` record objects."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES,
    ) -> None:
        if not isinstance(max_object_bytes, int) or isinstance(max_object_bytes, bool):
            raise TypeError("max_object_bytes must be an integer")
        if max_object_bytes <= 0:
            raise ValueError("max_object_bytes must be positive")
        self._root = _absolute_without_resolving(Path(root))
        self._max_object_bytes = max_object_bytes
        _reject_symlink_components(self._root)
        if self._root.exists() and not self._root.is_dir():
            raise ObjectStoreError(f"object-store root is not a directory: {self._root}")

    @property
    def root(self) -> Path:
        """Return the absolute store root without following symlinks."""

        return self._root

    @property
    def max_object_bytes(self) -> int:
        """Return the admission limit for one canonical record."""

        return self._max_object_bytes

    def path_for(self, record_id: str) -> Path:
        """Return the only local path for a validated record identity."""

        digest = validate_sha256(record_id, "record_id")
        return self._root / f"{digest}.json"

    def _ensure_write_directory(self, record_id: str) -> Path:
        _reject_symlink_components(self._root)
        self._root.mkdir(parents=True, exist_ok=True)
        if self._root.is_symlink() or not self._root.is_dir():
            raise ObjectStoreError(f"unsafe object-store root: {self._root}")
        return self._root

    def _check_size(self, size: int, record_id: str) -> None:
        if size > self._max_object_bytes:
            raise ObjectStoreError(
                f"object {record_id} has {size} bytes; limit is {self._max_object_bytes}"
            )

    def write(self, value: RecordObject | Mapping[str, Any]) -> RecordObject:
        """Create one immutable object, reusing an identical existing object."""

        record = value if isinstance(value, RecordObject) else RecordObject.from_value(value)
        self._check_size(len(record.canonical), record.record_id)
        self._ensure_write_directory(record.record_id)
        path = self.path_for(record.record_id)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        try:
            descriptor = os.open(path, flags, 0o644)
        except FileExistsError:
            existing = self.read(record.record_id)
            if existing.canonical != record.canonical:  # hash collision or concurrent corruption
                raise ObjectStoreError(
                    f"existing object {record.record_id} does not contain the expected bytes"
                )
            return existing

        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(record.canonical)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            try:
                path.unlink(missing_ok=True)
            finally:
                raise
        return record

    def write_bytes(self, raw: str | bytes | bytearray | memoryview) -> RecordObject:
        """Validate canonical record bytes before storing them."""

        return self.write(RecordObject.from_bytes(raw))

    def read(self, record_id: str, *, expected_schema: str | None = None) -> RecordObject:
        """Load an object and verify path, canonical bytes and external hash."""

        digest = validate_sha256(record_id, "record_id")
        _reject_symlink_components(self._root)
        path = self.path_for(digest)
        if path.parent.is_symlink() or path.is_symlink():
            raise ObjectStoreError(f"object path is a symlink: {path}")
        try:
            info = path.stat()
        except FileNotFoundError as error:
            raise ObjectStoreError(f"missing object {digest}") from error
        if not stat.S_ISREG(info.st_mode):
            raise ObjectStoreError(f"object path is not a regular file: {path}")
        self._check_size(info.st_size, digest)
        raw = path.read_bytes()
        if len(raw) != info.st_size:
            raise ObjectStoreError(f"object {digest} changed while it was read")
        try:
            record = RecordObject.from_bytes(raw, expected_schema=expected_schema)
        except CanonicalError as error:
            raise ObjectStoreError(f"object {digest} is invalid: {error}") from error
        if record.record_id != digest:
            raise ObjectStoreError(
                f"object hash mismatch: path names {digest}, bytes hash to {record.record_id}"
            )
        return record

    def contains(self, record_id: str) -> bool:
        """Return whether a safe regular object exists at the exact digest path."""

        digest = validate_sha256(record_id, "record_id")
        _reject_symlink_components(self._root)
        path = self.path_for(digest)
        if path.parent.is_symlink() or path.is_symlink():
            raise ObjectStoreError(f"object path is a symlink: {path}")
        return path.is_file()


__all__ = [
    "DEFAULT_MAX_OBJECT_BYTES",
    "ObjectStore",
    "ObjectStoreError",
]
