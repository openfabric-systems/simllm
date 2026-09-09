"""Keep full runtime byte receipts across file-backed and built-in libraries."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.machinery
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.snapshot_dispatch_v1 import common


@pytest.mark.parametrize("kind", ["file-backed", "built-in"])
@pytest.mark.parametrize("content", [b"first runtime bytes", b"second runtime bytes"])
def test_frozen_library_kind_and_backing_bytes_grid(tmp_path, monkeypatch, kind, content):
    path = tmp_path / "runtime-image.bin"
    path.write_bytes(content)
    monkeypatch.setattr(common, "_windows_runtime_path", lambda: path)
    module = SimpleNamespace(__file__=str(path)) if kind == "file-backed" else sys
    expected = {"path": str(path.resolve()), "sha256": hashlib.sha256(content).hexdigest()}
    if kind == "built-in":
        expected["origin"] = "built-in"
    observed = common.library_identity(module)
    assert json.dumps(observed, sort_keys=True) == json.dumps(expected, sort_keys=True)
    path.write_bytes(content + b"changed")
    assert common.library_identity(module)["sha256"] != observed["sha256"]


def test_missing_library_image_rejects(tmp_path):
    with pytest.raises(FileNotFoundError):
        common.library_identity(SimpleNamespace(__file__=str(tmp_path / "missing")))


@pytest.mark.parametrize("kind", ["unknown-module", "unknown-origin", "forged-module", "unknown-loader"])
def test_unverified_builtin_provenance_rejects(monkeypatch, kind):
    module = SimpleNamespace(__name__="sys", __spec__=SimpleNamespace(
        origin="built-in", loader=importlib.machinery.BuiltinImporter))
    if kind == "unknown-module":
        module.__name__ = "unregistered-module"
    elif kind == "unknown-origin":
        module.__spec__.origin = "frozen"
    elif kind == "unknown-loader":
        module.__spec__.loader = object()
    if kind != "forged-module":
        monkeypatch.setattr(common, "sys", SimpleNamespace(
            builtin_module_names=sys.builtin_module_names, modules={module.__name__: module}))
    with pytest.raises(ValueError, match="verified built-in"):
        common.library_identity(module)


class PathLookup:
    def __init__(self, path, mode="success"):
        self.path, self.mode, self.calls = path, mode, []

    def __call__(self, handle, buffer, capacity):
        self.calls.append((handle, capacity))
        if self.mode == "failure":
            return 0
        length = len(str(self.path).encode("utf-16-le")) // 2
        if self.mode == "truncated" or capacity <= length:
            return capacity
        buffer.value = str(self.path)
        return length


def install_lookup(monkeypatch, lookup, handle=12345):
    monkeypatch.setattr(sys, "dllhandle", handle, raising=False)
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, **kwargs: SimpleNamespace(GetModuleFileNameW=lookup), raising=False)


@pytest.mark.parametrize("filename", ["actual-runtime.dll", "runtime-\U0001f600.dll"])
def test_windows_lookup_uses_actual_runtime_handle_and_complete_path(tmp_path, monkeypatch, filename):
    path = tmp_path / filename
    path.write_bytes(b"actual library")
    lookup = PathLookup(path)
    install_lookup(monkeypatch, lookup)
    assert common._windows_runtime_path() == path.resolve()
    assert all(handle == 12345 for handle, _ in lookup.calls)
    assert lookup.calls[-1][1] > len(str(path))
    assert len(lookup.argtypes) == 3


@pytest.mark.parametrize("handle", [None, 0, -1, True, "12345", 2**128])
def test_invalid_runtime_handles_reject_before_lookup(monkeypatch, handle):
    lookup = PathLookup("unused")
    install_lookup(monkeypatch, lookup, handle)
    with pytest.raises(ValueError, match="DLL handle"):
        common._windows_runtime_path()
    assert not lookup.calls


@pytest.mark.parametrize("mode,error", [("failure", OSError), ("truncated", ValueError)])
def test_runtime_lookup_failure_or_truncation_is_fatal(monkeypatch, mode, error):
    lookup = PathLookup("unused", mode)
    install_lookup(monkeypatch, lookup)
    with pytest.raises(error):
        common._windows_runtime_path()


@pytest.mark.skipif(sys.platform != "win32", reason="actual Windows loaded-module lookup")
def test_actual_windows_builtin_math_has_runtime_image_bytes():
    value = common.library_identity(math)
    assert value["origin"] == "built-in"
    assert Path(value["path"]).is_file()
    assert value["sha256"] == hashlib.sha256(Path(value["path"]).read_bytes()).hexdigest()
