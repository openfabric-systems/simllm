"""Runtime byte identity and source-bound callable witnesses for raw admission."""

import cProfile
import dataclasses
import enum
import fractions
import hashlib
import importlib.machinery
import json
import math
import pathlib
import platform
import pstats
import struct
import sys
from types import CodeType

from examples.publication_snapshot_v1.common import sha


def code_hash(code, filename):
    """Hash code values without marshal's reference-count and interning flags."""
    def constant(item):
        if isinstance(item, CodeType):
            return ["code", structure(item)]
        kind = type(item)
        if item is None:
            return ["none"]
        if item is Ellipsis:
            return ["ellipsis"]
        if kind is bool:
            return ["bool", item]
        if kind is int:
            return ["int", str(item)]
        if kind is str:
            return ["str", item]
        if kind is bytes:
            return ["bytes", item.hex()]
        if kind is float:
            return ["float", struct.pack("!d", item).hex()]
        if kind is complex:
            return ["complex", struct.pack("!dd", item.real, item.imag).hex()]
        if kind is tuple:
            return ["tuple", [constant(value) for value in item]]
        if kind is frozenset:
            return ["frozenset", sorted((constant(value) for value in item), key=canonical)]
        raise TypeError("unsupported code constant: " + kind.__name__)

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def structure(item):
        return {"filename": filename, "name": item.co_name, "qualname": getattr(item, "co_qualname", item.co_name),
                "argcount": item.co_argcount, "posonlyargcount": item.co_posonlyargcount,
                "kwonlyargcount": item.co_kwonlyargcount, "nlocals": item.co_nlocals,
                "stacksize": item.co_stacksize, "flags": item.co_flags, "code": item.co_code.hex(),
                "constants": [constant(value) for value in item.co_consts],
                "names": list(item.co_names), "varnames": list(item.co_varnames),
                "freevars": list(item.co_freevars), "cellvars": list(item.co_cellvars),
                "firstlineno": item.co_firstlineno, "lnotab": item.co_lnotab.hex(),
                "linetable": getattr(item, "co_linetable", b"").hex(),
                "exceptiontable": getattr(item, "co_exceptiontable", b"").hex()}
    return hashlib.sha256(canonical(structure(code)).encode()).hexdigest()


def _windows_runtime_path():
    """Resolve the loaded Python image, never a guessed DLL or launcher path."""
    import ctypes
    from ctypes import wintypes

    handle = getattr(sys, "dllhandle", None)
    maximum = 2 ** (8 * ctypes.sizeof(ctypes.c_void_p)) - 1
    if type(handle) is not int or not 0 < handle <= maximum or not hasattr(ctypes, "WinDLL"):
        raise ValueError("built-in module provenance requires a loaded Windows Python DLL handle")
    lookup = ctypes.WinDLL("kernel32", use_last_error=True).GetModuleFileNameW
    lookup.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    lookup.restype = wintypes.DWORD
    for capacity in (260, 520, 1040, 2080, 4160, 8320, 16640, 32768):
        buffer = ctypes.create_unicode_buffer(capacity)
        length = lookup(handle, buffer, capacity)
        if length == 0:
            raise OSError("GetModuleFileNameW failed for the loaded Python runtime")
        if length < capacity:
            if length != len(buffer.value.encode("utf-16-le")) // 2:
                raise ValueError("loaded runtime path length disagrees with the operating system")
            return pathlib.Path(buffer.value).resolve(strict=True)
    raise ValueError("loaded runtime path remains truncated")


def library_identity(module):
    """Keep file-backed receipts exact and identify real built-in module bytes."""
    filename = getattr(module, "__file__", None)
    if filename is not None:
        return {"path": str(pathlib.Path(filename).resolve()), "sha256": sha(filename)}
    spec = getattr(module, "__spec__", None)
    if (module.__name__ not in sys.builtin_module_names or sys.modules.get(module.__name__) is not module
            or spec is None or spec.origin != "built-in" or spec.loader is not importlib.machinery.BuiltinImporter):
        raise ValueError("module has neither a source file nor verified built-in provenance")
    path = _windows_runtime_path()
    return {"path": str(path), "sha256": sha(path), "origin": "built-in"}


def runtime_identity():
    executable = pathlib.Path(sys.executable).resolve()
    modules = (dataclasses, enum, fractions, pathlib, math, cProfile, pstats)
    return {"executable": str(executable), "executable_sha256": sha(executable),
            "implementation": platform.python_implementation(), "version": sys.version,
            "version_info": list(sys.version_info), "cache_tag": sys.implementation.cache_tag,
            "libraries": {module.__name__: library_identity(module) for module in modules},
            "dataclass_functions": dataclass_functions()}


def dataclass_functions():
    return {name: {"identity": [function.__code__.co_filename, function.__code__.co_firstlineno, name],
                   "code_sha256": code_hash(function.__code__, "dataclasses.py")}
            for name, function in (("fields", dataclasses.fields), ("is_dataclass", dataclasses.is_dataclass))}


def binding_witnesses(sink, names, root):
    entries = sink._deferred_bindings()
    if len(entries) != len(names):
        raise ValueError("binding witness domain changed")
    result = []
    for name, (owner, function, code) in zip(names, entries, strict=True):
        if owner is not sink or function is not getattr(type(sink), name) or code is not function.__code__:
            raise ValueError("binding witness is not the selected implementation")
        file = pathlib.Path(code.co_filename).resolve()
        relative = file.relative_to(root).as_posix()
        result.append({"name": name, "owner_id": id(owner), "function_id": id(function), "code_id": id(code),
                       "module": function.__module__, "qualname": function.__qualname__, "source": relative,
                       "source_sha256": sha(file), "line": code.co_firstlineno, "code_sha256": code_hash(code, relative)})
    return result
