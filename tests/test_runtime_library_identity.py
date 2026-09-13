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
    monkeypatch.setattr(common, "_posix_runtime_path", lambda: path)
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


def maps_line(start, end, name=""):
    """Spell one kernel memory-map row: bounds, permissions, offset, device, inode, path."""
    return f"{start:012x}-{end:012x} r-xp 00000000 00:1b 4242 {name}".rstrip()


def install_maps(monkeypatch, text):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(common, "_process_maps", lambda: text)


def test_posix_lookup_uses_the_mapping_that_contains_the_core_symbol(tmp_path, monkeypatch):
    path = tmp_path / "actual-runtime.so"
    path.write_bytes(b"actual library")
    address = common._core_symbol_address()
    install_maps(monkeypatch, "\n".join([
        maps_line(address - 0x3000, address - 0x2000, "[vdso]"),
        maps_line(address - 0x1000, address),
        maps_line(address, address + 0x1000, str(path)),
        maps_line(address + 0x1000, address + 0x2000, str(tmp_path / "later-runtime.so")),
    ]))
    assert common._posix_runtime_path() == path.resolve()


def test_posix_builtin_identity_hashes_the_mapped_image(tmp_path, monkeypatch):
    path = tmp_path / "actual-runtime.so"
    path.write_bytes(b"actual library")
    address = common._core_symbol_address()
    install_maps(monkeypatch, maps_line(address, address + 0x1000, str(path)))
    expected = {"path": str(path.resolve()), "sha256": hashlib.sha256(b"actual library").hexdigest(),
                "origin": "built-in"}
    observed = common.library_identity(sys)
    assert json.dumps(observed, sort_keys=True) == json.dumps(expected, sort_keys=True)
    path.write_bytes(b"actual library changed")
    assert common.library_identity(sys)["sha256"] != observed["sha256"]


@pytest.mark.parametrize("api", [None, SimpleNamespace()])
def test_missing_core_symbol_rejects(monkeypatch, api):
    monkeypatch.setattr(ctypes, "pythonapi", api, raising=False)
    with pytest.raises(ValueError, match="interpreter core symbol"):
        common._core_symbol_address()


def test_core_symbol_without_an_address_rejects(monkeypatch):
    monkeypatch.setattr(ctypes, "pythonapi", SimpleNamespace(Py_Initialize=ctypes.c_void_p(0)), raising=False)
    with pytest.raises(ValueError, match="no usable address"):
        common._core_symbol_address()


@pytest.mark.parametrize("name,match", [
    ("", "anonymous or pseudo"),
    ("[vdso]", "anonymous or pseudo"),
    ("[heap]", "anonymous or pseudo"),
    ("relative/runtime.so", "not absolute"),
])
def test_posix_mapping_without_a_real_image_rejects(monkeypatch, name, match):
    address = common._core_symbol_address()
    install_maps(monkeypatch, maps_line(address, address + 0x1000, name))
    with pytest.raises(ValueError, match=match):
        common._posix_runtime_path()


def test_posix_missing_mapped_image_file_rejects(tmp_path, monkeypatch):
    address = common._core_symbol_address()
    install_maps(monkeypatch, maps_line(address, address + 0x1000, str(tmp_path / "missing.so")))
    with pytest.raises(FileNotFoundError):
        common._posix_runtime_path()


@pytest.mark.parametrize("text", ["", "not a mapping row", "zzzz-zzzz r-xp 00000000 00:1b 4242 image.so"])
def test_posix_address_outside_every_mapping_rejects(tmp_path, monkeypatch, text):
    address = common._core_symbol_address()
    rows = [text, maps_line(address + 0x2000, address + 0x3000, str(tmp_path / "elsewhere.so"))]
    install_maps(monkeypatch, "\n".join(rows))
    with pytest.raises(ValueError, match="no process mapping"):
        common._posix_runtime_path()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX dynamic loader lookup")
def test_non_linux_posix_asks_the_dynamic_loader(tmp_path, monkeypatch):
    path = tmp_path / "loader-runtime.so"
    path.write_bytes(b"loader library")
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(common, "_process_maps", lambda: pytest.fail("no kernel map off Linux"))
    monkeypatch.setattr(common, "_loader_image_path", lambda address: path.resolve())
    assert common._posix_runtime_path() == path.resolve()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX dynamic loader lookup")
def test_dynamic_loader_lookup_failure_is_fatal():
    with pytest.raises(OSError, match="dladdr"):
        common._loader_image_path(1)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux process memory map")
def test_kernel_map_and_dynamic_loader_agree_on_the_loaded_image():
    address = common._core_symbol_address()
    image = common._mapped_image_path(address, common._process_maps())
    assert image.is_absolute() and image.is_file()
    assert image == common._loader_image_path(address)


@pytest.mark.skipif(sys.platform == "win32" or hasattr(math, "__file__"),
                    reason="actual POSIX built-in module lookup")
def test_actual_posix_builtin_math_has_runtime_image_bytes():
    value = common.library_identity(math)
    assert value["origin"] == "built-in"
    assert Path(value["path"]).is_file()
    assert value["sha256"] == hashlib.sha256(Path(value["path"]).read_bytes()).hexdigest()
    assert value["sha256"]
    before = common.runtime_identity()
    assert before["libraries"]["math"] == value
    assert json.dumps(before, sort_keys=True) == json.dumps(common.runtime_identity(), sort_keys=True)
