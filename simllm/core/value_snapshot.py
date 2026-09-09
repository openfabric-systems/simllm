"""Immutable typed value snapshots for owned deferred work."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from enum import Enum
from fractions import Fraction
from pathlib import PurePath


def value_snapshot(value: object, *, project: Callable[[object], object] | None = None) -> tuple:
    """Include every dataclass field, including fields excluded from equality.

    Unknown objects require an explicit caller projection. Object identity and
    value state are separate checks; neither substitutes for the other.
    """
    active: set[int] = set()

    def visit(item: object) -> tuple:
        kind = type(item)
        # Exact builtins have immutable type tags. Keep all other dispatch in
        # the original order, including virtual subclasses and custom types.
        if kind is str:
            return ("builtins", "str", item)
        if kind is int:
            return ("builtins", "int", item)
        if kind is bool:
            return ("builtins", "bool", item)
        if kind is bytes:
            return ("builtins", "bytes", item)
        if item is None:
            return ("builtins", "NoneType", item)
        if kind is float:
            if not math.isfinite(item):
                raise ValueError("deferred values must be finite")
            return ("builtins", "float", item.hex())
        tag = (kind.__module__, kind.__qualname__)
        if isinstance(item, Enum):
            return (*tag, visit(item.value))
        if kind in (str, int, bool, bytes, type(None)):
            return (*tag, item)
        if kind is float:
            if not math.isfinite(item):
                raise ValueError("deferred values must be finite")
            return (*tag, item.hex())
        if isinstance(item, Fraction):
            return (*tag, item.numerator, item.denominator)
        if isinstance(item, PurePath):
            return (*tag, str(item))
        identity = id(item)
        if identity in active:
            raise ValueError("deferred value state contains a cycle")
        active.add(identity)
        try:
            if is_dataclass(item) and not isinstance(item, type):
                return (*tag, tuple((field.name, visit(getattr(item, field.name)))
                                    for field in fields(item)))
            if kind is dict:
                return (*tag, tuple((visit(key), visit(child)) for key, child in item.items()))
            if kind in (tuple, list):
                return (*tag, tuple(visit(child) for child in item))
            if kind in (set, frozenset):
                return (*tag, frozenset(visit(child) for child in item))
            if project is not None:
                projected = project(item)
                if projected is not NotImplemented:
                    return (*tag, visit(projected))
            raise TypeError("unsupported deferred value type: " + ".".join(tag))
        finally:
            active.remove(identity)

    return visit(value)
