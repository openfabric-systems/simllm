"""Shared strict-JSON helpers for the versioned core wire contracts.

One helper family and one ``OperationCorrelation`` codec serve every core
schema (execution graph, completion event, execution result and request
bookkeeping), so a wire object embedded in several schemas cannot drift
between parsers.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from typing import Any, NoReturn, TypeVar

from simllm.core.execution import OperationCorrelation

_EnumT = TypeVar("_EnumT", bound=enum.Enum)


def _fail(path: str, message: str) -> NoReturn:
    raise ValueError(f"{path}: {message}")


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    if any(not isinstance(key, str) for key in value):
        _fail(path, "object keys must be strings")
    return value


def _fields(
    value: Mapping[str, Any],
    path: str,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = sorted(required - value.keys())
    if missing:
        _fail(path, f"missing fields {missing}")
    unknown = sorted(value.keys() - required - optional)
    if unknown:
        _fail(path, f"unknown fields {unknown}")


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, "expected an array")
    return value


def _string(value: Any, path: str, *, nonblank: bool = True) -> str:
    if not isinstance(value, str):
        _fail(path, "expected a string")
    if nonblank and not value.strip():
        _fail(path, "must not be blank")
    return value


def _optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _integer(
    value: Any,
    path: str,
    *,
    nonnegative: bool = False,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        _fail(path, "expected an integer")
    if nonnegative and value < 0:
        _fail(path, "must be nonnegative")
    if minimum is not None and value < minimum:
        _fail(path, f"must be at least {minimum}")
    return value


def _optional_integer(
    value: Any,
    path: str,
    *,
    nonnegative: bool = False,
    minimum: int | None = None,
) -> int | None:
    if value is None:
        return None
    return _integer(value, path, nonnegative=nonnegative, minimum=minimum)


def _number(
    value: Any,
    path: str,
    *,
    nonnegative: bool = False,
    minimum: float | None = None,
) -> float:
    if type(value) not in (int, float):
        _fail(path, "expected a finite number")
    try:
        result = float(value)
    except OverflowError:
        _fail(path, "must be finite")
    if not math.isfinite(result):
        _fail(path, "must be finite")
    if nonnegative and result < 0:
        _fail(path, "must be nonnegative")
    if minimum is not None and result < minimum:
        _fail(path, f"must be at least {minimum}")
    return result


def _optional_number(
    value: Any,
    path: str,
    *,
    nonnegative: bool = False,
    minimum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _number(value, path, nonnegative=nonnegative, minimum=minimum)


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        _fail(path, "expected a boolean")
    return value


def _enum_value(cls: type[_EnumT], value: Any, path: str) -> _EnumT:
    raw = _string(value, path)
    try:
        return cls(raw)
    except ValueError as exc:
        choices = [member.value for member in cls]
        _fail(path, f"unknown value {raw!r}; expected one of {choices}")
        raise AssertionError from exc


def _scalar(value: Any, path: str) -> str | int | float | bool:
    if isinstance(value, bool) or type(value) is int or isinstance(value, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    _fail(path, "expected a finite JSON scalar")


def _string_tuple(value: Any, path: str, *, nonblank: bool = True) -> tuple[str, ...]:
    return tuple(
        _string(entry, f"{path}[{index}]", nonblank=nonblank)
        for index, entry in enumerate(_array(value, path))
    )


def _int_tuple(value: Any, path: str) -> tuple[int, ...]:
    return tuple(
        _integer(entry, f"{path}[{index}]", nonnegative=True)
        for index, entry in enumerate(_array(value, path))
    )


def _require_tuple(value: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        _fail(path, "in-memory contract requires a tuple")
    return value


def _validate_unique(values: tuple[Any, ...], path: str) -> None:
    if len(values) != len(set(values)):
        _fail(path, "contains duplicate values")


def _validate_correlation(correlation: OperationCorrelation, path: str) -> None:
    if not isinstance(correlation, OperationCorrelation):
        _fail(path, "expected OperationCorrelation")
    request_ids = _require_tuple(correlation.request_ids, f"{path}.request_ids")
    for index, request_id in enumerate(request_ids):
        _string(request_id, f"{path}.request_ids[{index}]")
    _validate_unique(request_ids, f"{path}.request_ids")
    for field_name in ("batch_id",):
        value = getattr(correlation, field_name)
        if value is not None:
            _string(value, f"{path}.{field_name}")
    for field_name in ("layer", "microbatch", "iteration"):
        value = getattr(correlation, field_name)
        if value is not None:
            _integer(value, f"{path}.{field_name}", nonnegative=True)


def _correlation_to_json(correlation: OperationCorrelation) -> dict[str, Any]:
    return {
        "request_ids": list(correlation.request_ids),
        "batch_id": correlation.batch_id,
        "layer": correlation.layer,
        "microbatch": correlation.microbatch,
        "iteration": correlation.iteration,
    }


def _correlation_from_json(value: Any, path: str) -> OperationCorrelation:
    """Parse the correlation wire object with every field optional.

    The graph schema's leniency is the single contract: writers emit all
    five keys, readers accept any subset with defined defaults. The
    bookkeeping schema consumes this same codec so the two wire forms
    cannot disagree on the embedded correlation object.
    """
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required=set(),
        optional={"request_ids", "batch_id", "layer", "microbatch", "iteration"},
    )
    correlation = OperationCorrelation(
        request_ids=_string_tuple(payload.get("request_ids", []), f"{path}.request_ids"),
        batch_id=_optional_string(payload.get("batch_id"), f"{path}.batch_id"),
        layer=_optional_integer(payload.get("layer"), f"{path}.layer", nonnegative=True),
        microbatch=_optional_integer(
            payload.get("microbatch"), f"{path}.microbatch", nonnegative=True
        ),
        iteration=_optional_integer(
            payload.get("iteration"), f"{path}.iteration", nonnegative=True
        ),
    )
    _validate_correlation(correlation, path)
    return correlation
