"""Strict three-rung deployment frontier records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from pathlib import PurePosixPath

from simllm.core._wire import (
    _array,
    _enum_value,
    _fields,
    _integer,
    _object,
    _require_tuple,
    _string,
)
from simllm.deploy.estimator import PICOSECONDS_PER_SECOND
from simllm.deploy.frontier import ExternalAnchor, PointClass

FRONTIER_LADDER_RECORD_SCHEMA = "simllm-deployment-frontier-ladder-record-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class FrontierRung(str, Enum):
    """Stable authority identity for one ladder coordinate."""

    ESTIMATE = "estimate"
    LOGGOPSIM_IDEAL = "loggopsim-ideal"
    PACKET = "packet"


class RungAuthorityClass(str, Enum):
    """Whether a rung is priced by an estimator or a simulator level."""

    ESTIMATOR = "estimator"
    LEVEL = "level"


def _sha256(value: object, path: str) -> str:
    digest = _string(value, path)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{path}: expected 64 lowercase hexadecimal digits")
    return digest


def _portable_path(value: object, path: str) -> str:
    text = _string(value, path)
    pure = PurePosixPath(text)
    if pure.is_absolute() or pure == PurePosixPath(".") or ".." in pure.parts:
        raise ValueError(f"{path}: expected a portable relative POSIX path")
    if "\\" in text:
        raise ValueError(f"{path}: expected a portable relative POSIX path")
    return text


def _fraction(value: object, path: str) -> Fraction:
    if not isinstance(value, Fraction):
        raise TypeError(f"{path}: expected Fraction")
    if value <= 0:
        raise ValueError(f"{path}: must be positive")
    return value


def _fraction_to_json(value: Fraction) -> dict[str, int]:
    _fraction(value, "fraction")
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction_from_json(value: object, path: str) -> Fraction:
    payload = _object(value, path)
    _fields(payload, path, required={"numerator", "denominator"})
    return Fraction(
        _integer(payload["numerator"], f"{path}.numerator", minimum=1),
        _integer(payload["denominator"], f"{path}.denominator", minimum=1),
    )


@dataclass(frozen=True, slots=True)
class RungProvenance:
    """Evidence authority and optional executed-level invocation."""

    authority_class: RungAuthorityClass
    authority: str
    source_path: str
    source_sha256: str
    binary_sha256: str | None = None
    goal_sha256: str | None = None
    argv: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.authority_class, RungAuthorityClass):
            raise TypeError("provenance.authority_class: expected RungAuthorityClass")
        _string(self.authority, "provenance.authority")
        _portable_path(self.source_path, "provenance.source_path")
        _sha256(self.source_sha256, "provenance.source_sha256")
        if self.binary_sha256 is not None:
            _sha256(self.binary_sha256, "provenance.binary_sha256")
        if self.goal_sha256 is not None:
            _sha256(self.goal_sha256, "provenance.goal_sha256")
        argv = _require_tuple(self.argv, "provenance.argv")
        for index, value in enumerate(argv):
            _string(value, f"provenance.argv[{index}]")


@dataclass(frozen=True, slots=True)
class FrontierRungPoint:
    """One exact point on one authority rung."""

    rung: FrontierRung
    point_class: PointClass
    step_ps: int
    fabric_leg_ps: int
    x_tokens_per_second_per_request: Fraction
    y_tokens_per_second_per_gpu: Fraction
    provenance: RungProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.rung, FrontierRung):
            raise TypeError("rung_point.rung: expected FrontierRung")
        if not isinstance(self.point_class, PointClass):
            raise TypeError("rung_point.point_class: expected PointClass")
        step_ps = _integer(self.step_ps, "rung_point.step_ps", minimum=1)
        _integer(self.fabric_leg_ps, "rung_point.fabric_leg_ps", nonnegative=True)
        x_value = _fraction(
            self.x_tokens_per_second_per_request,
            "rung_point.x_tokens_per_second_per_request",
        )
        _fraction(
            self.y_tokens_per_second_per_gpu,
            "rung_point.y_tokens_per_second_per_gpu",
        )
        if x_value != Fraction(PICOSECONDS_PER_SECOND, step_ps):
            raise ValueError(
                "rung_point.x_tokens_per_second_per_request: does not match step_ps"
            )
        if not isinstance(self.provenance, RungProvenance):
            raise TypeError("rung_point.provenance: expected RungProvenance")
        self.provenance.__post_init__()


@dataclass(frozen=True, slots=True)
class FrontierLadderPoint:
    """One deployment point with estimate, ideal and packet rungs."""

    configuration_id: str
    configuration_label: str
    batch_per_gpu: int
    rungs: tuple[FrontierRungPoint, ...]

    def __post_init__(self) -> None:
        _string(self.configuration_id, "ladder_point.configuration_id")
        _string(self.configuration_label, "ladder_point.configuration_label")
        batch = _integer(
            self.batch_per_gpu,
            "ladder_point.batch_per_gpu",
            minimum=1,
        )
        rungs = _require_tuple(self.rungs, "ladder_point.rungs")
        for index, rung in enumerate(rungs):
            if not isinstance(rung, FrontierRungPoint):
                raise TypeError(
                    f"ladder_point.rungs[{index}]: expected FrontierRungPoint"
                )
            rung.__post_init__()
            if rung.y_tokens_per_second_per_gpu != (
                batch * rung.x_tokens_per_second_per_request
            ):
                raise ValueError(
                    f"ladder_point.rungs[{index}].y_tokens_per_second_per_gpu: "
                    "does not match batch times x"
                )
        expected = tuple(FrontierRung)
        if tuple(rung.rung for rung in rungs) != expected:
            raise ValueError(
                "ladder_point.rungs: expected estimate, loggopsim-ideal and packet"
            )
        self._validate_authority(rungs[0], PointClass.ESTIMATE, "closed-form", False)
        if rungs[1].fabric_leg_ps > 0:
            self._validate_authority(
                rungs[1],
                PointClass.SIMULATED,
                "loggopsim-ideal",
                True,
            )
        else:
            self._validate_authority(
                rungs[1],
                PointClass.ESTIMATE,
                "closed-form",
                False,
            )
        self._validate_authority(rungs[2], PointClass.SIMULATED, "rnic-nn", False)
        if rungs[1].provenance == rungs[2].provenance:
            raise ValueError(
                "ladder_point.rungs: ideal and packet authorities must stay distinct"
            )

    @staticmethod
    def _validate_authority(
        point: FrontierRungPoint,
        point_class: PointClass,
        authority: str,
        executed: bool,
    ) -> None:
        if point.point_class is not point_class:
            raise ValueError(
                f"ladder_point.{point.rung.value}: point class does not match authority"
            )
        expected_authority_class = (
            RungAuthorityClass.ESTIMATOR
            if point_class is PointClass.ESTIMATE
            else RungAuthorityClass.LEVEL
        )
        provenance = point.provenance
        if provenance.authority_class is not expected_authority_class:
            raise ValueError(
                f"ladder_point.{point.rung.value}: authority class does not match rung"
            )
        if provenance.authority != authority:
            raise ValueError(
                f"ladder_point.{point.rung.value}: authority identity does not match rung"
            )
        has_execution = bool(
            provenance.argv and provenance.binary_sha256 and provenance.goal_sha256
        )
        if has_execution != executed:
            raise ValueError(
                f"ladder_point.{point.rung.value}: execution provenance is incomplete"
            )
        if not executed and (
            provenance.argv or provenance.goal_sha256 is not None
        ):
            raise ValueError(
                f"ladder_point.{point.rung.value}: unexpected execution provenance"
            )

    def rung(self, rung: FrontierRung) -> FrontierRungPoint:
        """Return one rung by stable identity."""

        return next(point for point in self.rungs if point.rung is rung)


@dataclass(frozen=True, slots=True)
class FrontierLadderRecord:
    """Strict three-rung frontier with measured context anchors."""

    points: tuple[FrontierLadderPoint, ...]
    anchors: tuple[ExternalAnchor, ...] = ()

    def __post_init__(self) -> None:
        points = _require_tuple(self.points, "ladder.points")
        if not points:
            raise ValueError("ladder.points: must not be empty")
        identities: list[tuple[str, int]] = []
        labels: dict[str, str] = {}
        for index, point in enumerate(points):
            if not isinstance(point, FrontierLadderPoint):
                raise TypeError(
                    f"ladder.points[{index}]: expected FrontierLadderPoint"
                )
            point.__post_init__()
            identities.append((point.configuration_id, point.batch_per_gpu))
            previous = labels.setdefault(
                point.configuration_id,
                point.configuration_label,
            )
            if previous != point.configuration_label:
                raise ValueError("ladder.points: configuration labels disagree")
        if len(identities) != len(set(identities)):
            raise ValueError("ladder.points: duplicate configuration and batch")
        anchors = _require_tuple(self.anchors, "ladder.anchors")
        anchor_ids = []
        for index, anchor in enumerate(anchors):
            if not isinstance(anchor, ExternalAnchor):
                raise TypeError(f"ladder.anchors[{index}]: expected ExternalAnchor")
            anchor.__post_init__()
            anchor_ids.append(anchor.anchor_id)
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("ladder.anchors: duplicate anchor id")

    @property
    def schema(self) -> str:
        return FRONTIER_LADDER_RECORD_SCHEMA


def _provenance_to_json(provenance: RungProvenance) -> dict[str, object]:
    provenance.__post_init__()
    return {
        "authority_class": provenance.authority_class.value,
        "authority": provenance.authority,
        "source_path": provenance.source_path,
        "source_sha256": provenance.source_sha256,
        "binary_sha256": provenance.binary_sha256,
        "goal_sha256": provenance.goal_sha256,
        "argv": list(provenance.argv),
    }


def _provenance_from_json(value: object, path: str) -> RungProvenance:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={
            "authority_class",
            "authority",
            "source_path",
            "source_sha256",
            "binary_sha256",
            "goal_sha256",
            "argv",
        },
    )
    raw_binary = payload["binary_sha256"]
    raw_goal = payload["goal_sha256"]
    return RungProvenance(
        authority_class=_enum_value(
            RungAuthorityClass,
            payload["authority_class"],
            f"{path}.authority_class",
        ),
        authority=_string(payload["authority"], f"{path}.authority"),
        source_path=_portable_path(payload["source_path"], f"{path}.source_path"),
        source_sha256=_sha256(payload["source_sha256"], f"{path}.source_sha256"),
        binary_sha256=(
            None if raw_binary is None else _sha256(raw_binary, f"{path}.binary_sha256")
        ),
        goal_sha256=(
            None if raw_goal is None else _sha256(raw_goal, f"{path}.goal_sha256")
        ),
        argv=tuple(
            _string(item, f"{path}.argv[{index}]")
            for index, item in enumerate(_array(payload["argv"], f"{path}.argv"))
        ),
    )


def _rung_to_json(point: FrontierRungPoint) -> dict[str, object]:
    point.__post_init__()
    return {
        "rung": point.rung.value,
        "point_class": point.point_class.value,
        "step_ps": point.step_ps,
        "fabric_leg_ps": point.fabric_leg_ps,
        "x_tokens_per_second_per_request": _fraction_to_json(
            point.x_tokens_per_second_per_request
        ),
        "y_tokens_per_second_per_gpu": _fraction_to_json(
            point.y_tokens_per_second_per_gpu
        ),
        "provenance": _provenance_to_json(point.provenance),
    }


def _rung_from_json(value: object, path: str) -> FrontierRungPoint:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={
            "rung",
            "point_class",
            "step_ps",
            "fabric_leg_ps",
            "x_tokens_per_second_per_request",
            "y_tokens_per_second_per_gpu",
            "provenance",
        },
    )
    rung = _enum_value(FrontierRung, payload["rung"], f"{path}.rung")
    point_class = _enum_value(
        PointClass,
        payload["point_class"],
        f"{path}.point_class",
    )
    fabric_leg_ps = _integer(
        payload["fabric_leg_ps"],
        f"{path}.fabric_leg_ps",
        nonnegative=True,
    )
    provenance = _provenance_from_json(
        payload["provenance"],
        f"{path}.provenance",
    )
    if (
        rung is FrontierRung.LOGGOPSIM_IDEAL
        and fabric_leg_ps == 0
        and point_class is PointClass.SIMULATED
        and provenance.authority_class is RungAuthorityClass.LEVEL
        and provenance.authority == "loggopsim-ideal"
        and provenance.goal_sha256 is None
        and not provenance.argv
    ):
        point_class = PointClass.ESTIMATE
        provenance = RungProvenance(
            authority_class=RungAuthorityClass.ESTIMATOR,
            authority="closed-form",
            source_path=provenance.source_path,
            source_sha256=provenance.source_sha256,
        )
    return FrontierRungPoint(
        rung=rung,
        point_class=point_class,
        step_ps=_integer(payload["step_ps"], f"{path}.step_ps", minimum=1),
        fabric_leg_ps=fabric_leg_ps,
        x_tokens_per_second_per_request=_fraction_from_json(
            payload["x_tokens_per_second_per_request"],
            f"{path}.x_tokens_per_second_per_request",
        ),
        y_tokens_per_second_per_gpu=_fraction_from_json(
            payload["y_tokens_per_second_per_gpu"],
            f"{path}.y_tokens_per_second_per_gpu",
        ),
        provenance=provenance,
    )


def _anchor_to_json(anchor: ExternalAnchor) -> dict[str, object]:
    anchor.__post_init__()
    x_value = anchor.x_tokens_per_second_per_request
    return {
        "anchor_id": anchor.anchor_id,
        "label": anchor.label,
        "x_tokens_per_second_per_request": (
            None if x_value is None else _fraction_to_json(x_value)
        ),
        "y_tokens_per_second_per_gpu": _fraction_to_json(
            anchor.y_tokens_per_second_per_gpu
        ),
        "point_class": anchor.point_class.value,
    }


def _anchor_from_json(value: object, path: str) -> ExternalAnchor:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={
            "anchor_id",
            "label",
            "x_tokens_per_second_per_request",
            "y_tokens_per_second_per_gpu",
            "point_class",
        },
    )
    raw_x = payload["x_tokens_per_second_per_request"]
    return ExternalAnchor(
        anchor_id=_string(payload["anchor_id"], f"{path}.anchor_id"),
        label=_string(payload["label"], f"{path}.label"),
        x_tokens_per_second_per_request=(
            None
            if raw_x is None
            else _fraction_from_json(raw_x, f"{path}.x_tokens_per_second_per_request")
        ),
        y_tokens_per_second_per_gpu=_fraction_from_json(
            payload["y_tokens_per_second_per_gpu"],
            f"{path}.y_tokens_per_second_per_gpu",
        ),
        point_class=_enum_value(
            PointClass,
            payload["point_class"],
            f"{path}.point_class",
        ),
    )


def frontier_ladder_record_to_json(record: FrontierLadderRecord) -> dict[str, object]:
    """Serialize one strict sibling ladder record."""

    if not isinstance(record, FrontierLadderRecord):
        raise TypeError("ladder: expected FrontierLadderRecord")
    record.__post_init__()
    return {
        "schema": FRONTIER_LADDER_RECORD_SCHEMA,
        "points": [
            {
                "configuration_id": point.configuration_id,
                "configuration_label": point.configuration_label,
                "batch_per_gpu": point.batch_per_gpu,
                "rungs": [_rung_to_json(rung) for rung in point.rungs],
            }
            for point in record.points
        ],
        "anchors": [_anchor_to_json(anchor) for anchor in record.anchors],
    }


def frontier_ladder_record_from_json(value: object) -> FrontierLadderRecord:
    """Parse a strict ladder record after checking its sibling schema first."""

    payload = _object(value, "ladder")
    schema = _string(payload.get("schema"), "ladder.schema")
    if schema != FRONTIER_LADDER_RECORD_SCHEMA:
        raise ValueError(
            "ladder.schema: unsupported schema "
            f"{schema!r}; expected {FRONTIER_LADDER_RECORD_SCHEMA!r}"
        )
    _fields(payload, "ladder", required={"schema", "points", "anchors"})
    points = []
    for index, raw_point in enumerate(_array(payload["points"], "ladder.points")):
        path = f"ladder.points[{index}]"
        point = _object(raw_point, path)
        _fields(
            point,
            path,
            required={
                "configuration_id",
                "configuration_label",
                "batch_per_gpu",
                "rungs",
            },
        )
        points.append(
            FrontierLadderPoint(
                configuration_id=_string(
                    point["configuration_id"],
                    f"{path}.configuration_id",
                ),
                configuration_label=_string(
                    point["configuration_label"],
                    f"{path}.configuration_label",
                ),
                batch_per_gpu=_integer(
                    point["batch_per_gpu"],
                    f"{path}.batch_per_gpu",
                    minimum=1,
                ),
                rungs=tuple(
                    _rung_from_json(rung, f"{path}.rungs[{rung_index}]")
                    for rung_index, rung in enumerate(
                        _array(point["rungs"], f"{path}.rungs")
                    )
                ),
            )
        )
    anchors = tuple(
        _anchor_from_json(anchor, f"ladder.anchors[{index}]")
        for index, anchor in enumerate(_array(payload["anchors"], "ladder.anchors"))
    )
    return FrontierLadderRecord(points=tuple(points), anchors=anchors)


def ladder_pareto_front(
    record: FrontierLadderRecord,
    rung: FrontierRung,
) -> tuple[FrontierLadderPoint, ...]:
    """Return exact non-dominated ladder points for one selected rung."""

    if not isinstance(record, FrontierLadderRecord):
        raise TypeError("record: expected FrontierLadderRecord")
    if not isinstance(rung, FrontierRung):
        raise TypeError("rung: expected FrontierRung")
    record.__post_init__()
    front = []
    for candidate in record.points:
        candidate_rung = candidate.rung(rung)
        dominated = any(
            other.rung(rung).x_tokens_per_second_per_request
            >= candidate_rung.x_tokens_per_second_per_request
            and other.rung(rung).y_tokens_per_second_per_gpu
            >= candidate_rung.y_tokens_per_second_per_gpu
            and (
                other.rung(rung).x_tokens_per_second_per_request
                > candidate_rung.x_tokens_per_second_per_request
                or other.rung(rung).y_tokens_per_second_per_gpu
                > candidate_rung.y_tokens_per_second_per_gpu
            )
            for other in record.points
        )
        if not dominated:
            front.append(candidate)
    return tuple(
        sorted(front, key=lambda point: (point.configuration_id, point.batch_per_gpu))
    )


__all__ = [
    "FRONTIER_LADDER_RECORD_SCHEMA",
    "FrontierLadderPoint",
    "FrontierLadderRecord",
    "FrontierRung",
    "FrontierRungPoint",
    "RungAuthorityClass",
    "RungProvenance",
    "frontier_ladder_record_from_json",
    "frontier_ladder_record_to_json",
    "ladder_pareto_front",
]
