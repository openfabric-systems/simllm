"""Strict receipts and a structural count independent of the snapshot visitor."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import pstats
import subprocess
import sys
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path, PurePath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREEZE = "5efe20b9df1c8156cf33409a9f5db6086a209fba"


@dataclass(frozen=True)
class InvalidObservation:
    """An explicit marker at a deliberately nonserializable mutation location."""

    kind: str


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_bytes((json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())


def read(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    raw = Path(path).read_bytes()
    value = json.loads(raw, object_pairs_hook=pairs)
    expected = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if raw != expected:
        raise ValueError("noncanonical JSON: " + Path(path).name)
    return value


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root).decode().strip()


def packages(root):
    return {name: sha(root / name) for name in git(root, "ls-files", "simllm").splitlines()}


def study_sources(root):
    return {name: sha(root / name) for name in git(root, "ls-files", "examples").splitlines() if name.endswith(".py")}


def study_origins():
    return {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
            if name.startswith("examples.") and getattr(module, "__file__", None)}


def receipts(root):
    return {path.relative_to(root).as_posix(): {"sha256": sha(path), "bytes": path.stat().st_size}
            for path in sorted(root.rglob("*")) if path.is_file()}


def pack(value):
    """Encode an already typed snapshot, preserving its bytes and set bodies."""
    if type(value) is tuple:
        return [pack(child) for child in value]
    if type(value) is frozenset:
        return {"frozenset": sorted((pack(child) for child in value), key=lambda child: json.dumps(child, sort_keys=True))}
    if type(value) is bytes:
        return {"bytes": value.hex()}
    if type(value) in (str, int, bool, type(None)):
        return value
    raise TypeError("snapshot wire contains an unexpected type")


def unpack(value):
    if type(value) is list:
        return tuple(unpack(child) for child in value)
    if type(value) is dict:
        if set(value) == {"frozenset"}:
            return frozenset(unpack(child) for child in value["frozenset"])
        if set(value) == {"bytes"}:
            return bytes.fromhex(value["bytes"])
        raise TypeError("snapshot wire contains an unknown object")
    if type(value) not in (str, int, bool, type(None)):
        raise TypeError("snapshot wire contains an invalid scalar")
    return value


def nodes(value):
    """Count source input visits without invoking the production serializer."""
    if isinstance(value, Enum):
        return 1 + nodes(value.value)
    if type(value) in (str, int, bool, bytes, type(None)) or isinstance(value, (Fraction, PurePath)):
        return 1
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite structural count")
        return 1
    if is_dataclass(value) and not isinstance(value, type):
        return 1 + sum(nodes(getattr(value, field.name)) for field in fields(value))
    if type(value) is dict:
        return 1 + sum(nodes(key) + nodes(child) for key, child in value.items())
    if type(value) in (tuple, list, set, frozenset):
        return 1 + sum(nodes(child) for child in value)
    raise TypeError("unsupported structural count type")


def profile_rows(path):
    result = []
    for function, (primitive, calls, own, cumulative, callers) in sorted(pstats.Stats(str(path)).stats.items()):
        result.append({"function": list(function), "primitive": primitive, "calls": calls,
                       "own_seconds": own, "cumulative_seconds": cumulative,
                       "callers": [[list(caller), list(values)] for caller, values in sorted(callers.items())]})
    return result


def original_nodes(snapshot):
    """Count original values from a complete typed capture, checking its shape."""
    if type(snapshot) is not tuple or len(snapshot) < 3:
        raise ValueError("invalid typed snapshot node")
    module, name, body = snapshot[:3]
    if type(module) is not str or type(name) is not str:
        raise ValueError("invalid typed snapshot tag")
    if module == "builtins":
        if len(snapshot) != 3:
            raise ValueError("invalid typed snapshot arity")
        scalars = {"str": str, "int": int, "bool": bool, "bytes": bytes, "NoneType": type(None)}
        if name in scalars:
            if type(body) is not scalars[name]:
                raise ValueError("typed scalar body disagrees with its tag")
            return 1
        if name == "float":
            if type(body) is not str:
                raise ValueError("typed float requires a canonical finite hex string")
            try:
                value = float.fromhex(body)
            except (ValueError, OverflowError) as error:
                raise ValueError("invalid typed float") from error
            if not math.isfinite(value) or value.hex() != body:
                raise ValueError("typed float requires a canonical finite hex string")
            return 1
        if type(body) is not (frozenset if name in ("set", "frozenset") else tuple):
            raise ValueError("typed container body disagrees with its tag")
        if name == "dict":
            if any(type(pair) is not tuple or len(pair) != 2 for pair in body):
                raise ValueError("invalid typed dictionary pair")
            return 1 + sum(original_nodes(key) + original_nodes(value) for key, value in body)
        if name in ("tuple", "list", "set", "frozenset"):
            return 1 + sum(original_nodes(value) for value in body)
        raise ValueError("unknown builtin snapshot type")
    owner = sys.modules.get(module)
    kind = getattr(owner, name, None)
    if not isinstance(kind, type):
        raise TypeError("snapshot type is not in the loaded source domain")
    if len(snapshot) != (4 if issubclass(kind, Fraction) else 3):
        raise ValueError("invalid typed snapshot arity")
    if issubclass(kind, Enum):
        return 1 + original_nodes(body)
    if issubclass(kind, Fraction):
        if (len(snapshot) != 4 or type(body) is not int or type(snapshot[3]) is not int or snapshot[3] <= 0
                or (Fraction(body, snapshot[3]).numerator, Fraction(body, snapshot[3]).denominator) != snapshot[2:]):
            raise ValueError("invalid typed fraction")
        return 1
    if issubclass(kind, PurePath):
        if type(body) is not str or str(kind(body)) != body:
            raise ValueError("invalid typed path")
        return 1
    if is_dataclass(kind):
        if type(body) is not tuple or any(type(pair) is not tuple or len(pair) != 2 for pair in body):
            raise ValueError("invalid typed dataclass body")
        if [field for field, _ in body] != [field.name for field in fields(kind)]:
            raise ValueError("snapshot dataclass field inventory changed")
        return 1 + sum(original_nodes(value) for _, value in body)
    raise ValueError("unsupported original snapshot type")


def typed_fields(snapshot, kind):
    """Require an exact loaded dataclass and its complete ordered field domain."""
    if snapshot[:2] != (kind.__module__, kind.__qualname__) or len(snapshot) != 3:
        raise ValueError("typed row class disagrees: " + kind.__name__)
    original_nodes(snapshot)
    return dict(snapshot[2])


def plain(snapshot):
    """Project a checked typed capture to JSON values without constructing objects."""
    original_nodes(snapshot)
    module, name, body = snapshot[:3]
    if module == "builtins":
        if name == "float":
            return float.fromhex(body)
        if name in ("str", "int", "bool", "NoneType"):
            return body
        if name in ("tuple", "list"):
            return [plain(value) for value in body]
        if name == "dict":
            return {plain(key): plain(value) for key, value in body}
    kind = getattr(sys.modules.get(module), name, None)
    if isinstance(kind, type) and is_dataclass(kind):
        return {key: plain(value) for key, value in body}
    raise TypeError("typed row is outside the publication projection domain")


def snapshot_functions(root):
    path = root / "simllm/core/value_snapshot.py"
    tree = ast.parse(path.read_text())
    return {node.name: [str(path), node.lineno, node.name] for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in {"value_snapshot", "visit"}}
