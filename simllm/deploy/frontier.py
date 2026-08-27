"""Strict backend-free deployment frontiers and plot-data preparation."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, TypeAlias

from simllm.core._wire import (
    _array,
    _boolean,
    _enum_value,
    _fields,
    _integer,
    _object,
    _require_tuple,
    _string,
)
from simllm.deploy.candidate import (
    DeploymentCandidate,
    candidate_key,
    check_feasibility,
)
from simllm.deploy.estimator import (
    PICOSECONDS_PER_SECOND,
    EstimateStamp,
    EstimatorInputs,
    estimate_decode_step,
    estimate_stamp_from_json,
    estimate_stamp_to_json,
)

FRONTIER_RECORD_SCHEMA = "simllm-deployment-frontier-record-v1"
PLOT_CONTRACT_V3_SCHEMA = "simllm-deployment-frontier-plot-contract-v3"

_LEGACY_RESULT_SCHEMA = "simllm-deployment-frontier-result-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_LEGACY_COLORS = {
    "b100-one-node-intra": "#2563a6",
    "h100-two-node-serialized": "#d17c0f",
    "h100-nine-node-incast": "#27845c",
}
_FALLBACK_COLORS = (
    "#2563a6",
    "#d17c0f",
    "#27845c",
    "#8f3f71",
    "#7656a3",
    "#4b84c4",
)


class PointClass(str, Enum):
    """Authority class represented by one frontier coordinate."""

    ESTIMATE = "ESTIMATE"
    SIMULATED = "SIMULATED"
    MEASURED = "MEASURED"


def _sha256(value: object, path: str) -> str:
    digest = _string(value, path)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{path}: expected 64 lowercase hexadecimal digits")
    return digest


def _fraction(value: object, path: str, *, positive: bool = True) -> Fraction:
    if not isinstance(value, Fraction):
        raise TypeError(f"{path}: expected Fraction")
    if positive and value <= 0:
        raise ValueError(f"{path}: must be positive")
    return value


def _fraction_to_json(value: Fraction) -> dict[str, int]:
    _fraction(value, "fraction")
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction_from_json(value: object, path: str) -> Fraction:
    payload = _object(value, path)
    _fields(payload, path, required={"numerator", "denominator"})
    numerator = _integer(payload["numerator"], f"{path}.numerator", minimum=1)
    denominator = _integer(payload["denominator"], f"{path}.denominator", minimum=1)
    return Fraction(numerator, denominator)


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """One exact operating point priced by the installed estimator."""

    candidate_key: str
    configuration_id: str
    batch_per_gpu: int
    x_tokens_per_second_per_request: Fraction
    y_tokens_per_second_per_gpu: Fraction
    point_class: PointClass
    step_ps: int
    stamp: EstimateStamp

    def __post_init__(self) -> None:
        key = _sha256(self.candidate_key, "point.candidate_key")
        _string(self.configuration_id, "point.configuration_id")
        batch = _integer(self.batch_per_gpu, "point.batch_per_gpu", minimum=1)
        step = _integer(self.step_ps, "point.step_ps", minimum=1)
        x_value = _fraction(
            self.x_tokens_per_second_per_request,
            "point.x_tokens_per_second_per_request",
        )
        y_value = _fraction(
            self.y_tokens_per_second_per_gpu,
            "point.y_tokens_per_second_per_gpu",
        )
        if x_value != Fraction(PICOSECONDS_PER_SECOND, step):
            raise ValueError(
                "point.x_tokens_per_second_per_request: does not match step_ps"
            )
        if y_value != batch * x_value:
            raise ValueError(
                "point.y_tokens_per_second_per_gpu: does not match batch times x"
            )
        if not isinstance(self.point_class, PointClass):
            raise TypeError("point.point_class: expected PointClass")
        if self.point_class is PointClass.MEASURED:
            raise ValueError(
                "point.point_class: MEASURED is reserved for external anchors"
            )
        if not isinstance(self.stamp, EstimateStamp):
            raise TypeError("point.stamp: expected EstimateStamp")
        self.stamp.__post_init__()
        if self.stamp.candidate_key != key:
            raise ValueError("point.stamp: candidate_key does not match point")
        term_durations = {
            term.name: term.estimate.duration_ps for term in self.stamp.terms
        }
        required_terms = {"kernel_floor", "fabric_floor", "intra_floor"}
        missing_terms = sorted(required_terms - term_durations.keys())
        if missing_terms:
            raise ValueError(f"point.stamp: missing required terms {missing_terms}")
        analytical_step = max(term_durations[name] for name in required_terms)
        if analytical_step <= 0:
            raise ValueError("point.stamp: analytical floor must be positive")
        stamped_step = max(
            term_durations["kernel_floor"],
            term_durations["fabric_floor"] + term_durations.get("fabric_excess", 0),
            term_durations["intra_floor"] + term_durations.get("intra_excess", 0),
        )
        if stamped_step != step:
            raise ValueError("point.step_ps: does not match stamped floor composition")
        expected_class = (
            PointClass.SIMULATED
            if self.stamp.consumes_sim_derived
            else PointClass.ESTIMATE
        )
        if self.point_class is not expected_class:
            raise ValueError(
                "point.point_class: must reflect whether the stamp consumes "
                "SIM-DERIVED evidence"
            )


@dataclass(frozen=True, slots=True)
class ExternalAnchor:
    """Measured context carried as a paired point or a y-only anchor line."""

    anchor_id: str
    label: str
    y_tokens_per_second_per_gpu: Fraction
    x_tokens_per_second_per_request: Fraction | None = None
    point_class: PointClass = PointClass.MEASURED

    def __post_init__(self) -> None:
        _string(self.anchor_id, "anchor.anchor_id")
        _string(self.label, "anchor.label")
        _fraction(
            self.y_tokens_per_second_per_gpu,
            "anchor.y_tokens_per_second_per_gpu",
        )
        if self.x_tokens_per_second_per_request is not None:
            _fraction(
                self.x_tokens_per_second_per_request,
                "anchor.x_tokens_per_second_per_request",
            )
        if self.point_class is not PointClass.MEASURED:
            raise ValueError("anchor.point_class: external anchors must be MEASURED")

    @property
    def y_only(self) -> bool:
        """Return whether this anchor intentionally omits an x coordinate."""

        return self.x_tokens_per_second_per_request is None


@dataclass(frozen=True, slots=True)
class CandidateFrontier:
    """Feasibility decision and zero or more points for one candidate."""

    candidate_key: str
    configuration_id: str
    configuration_label: str
    accepted: bool
    rejection_reasons: tuple[str, ...]
    points: tuple[FrontierPoint, ...]

    def __post_init__(self) -> None:
        key = _sha256(self.candidate_key, "candidate_frontier.candidate_key")
        configuration_id = _string(
            self.configuration_id,
            "candidate_frontier.configuration_id",
        )
        _string(self.configuration_label, "candidate_frontier.configuration_label")
        if type(self.accepted) is not bool:
            raise TypeError("candidate_frontier.accepted: expected a boolean")
        reasons = _require_tuple(
            self.rejection_reasons,
            "candidate_frontier.rejection_reasons",
        )
        for index, reason in enumerate(reasons):
            _string(reason, f"candidate_frontier.rejection_reasons[{index}]")
        if len(reasons) != len(set(reasons)):
            raise ValueError(
                "candidate_frontier.rejection_reasons: contains duplicate values"
            )
        if self.accepted != (not reasons):
            raise ValueError(
                "candidate_frontier.accepted: must be true exactly when reasons are empty"
            )
        points = _require_tuple(self.points, "candidate_frontier.points")
        if self.accepted and not points:
            raise ValueError("candidate_frontier.points: accepted candidate has no points")
        if not self.accepted and points:
            raise ValueError("candidate_frontier.points: rejected candidate must have no points")
        for index, point in enumerate(points):
            if not isinstance(point, FrontierPoint):
                raise TypeError(
                    f"candidate_frontier.points[{index}]: expected FrontierPoint"
                )
            point.__post_init__()
            if point.candidate_key != key:
                raise ValueError(
                    f"candidate_frontier.points[{index}]: candidate_key mismatch"
                )
            if point.configuration_id != configuration_id:
                raise ValueError(
                    f"candidate_frontier.points[{index}]: configuration_id mismatch"
                )


@dataclass(frozen=True, slots=True)
class FrontierRecord:
    """Strict scan result with nested candidate decisions and measured anchors."""

    candidates: tuple[CandidateFrontier, ...]
    anchors: tuple[ExternalAnchor, ...] = ()

    def __post_init__(self) -> None:
        candidates = _require_tuple(self.candidates, "frontier.candidates")
        if not candidates:
            raise ValueError("frontier.candidates: must not be empty")
        ids: list[str] = []
        keys: list[str] = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, CandidateFrontier):
                raise TypeError(f"frontier.candidates[{index}]: expected CandidateFrontier")
            candidate.__post_init__()
            ids.append(candidate.configuration_id)
            keys.append(candidate.candidate_key)
        if len(ids) != len(set(ids)):
            raise ValueError("frontier.candidates: contains duplicate configuration ids")
        if len(keys) != len(set(keys)):
            raise ValueError("frontier.candidates: contains duplicate candidate keys")
        anchors = _require_tuple(self.anchors, "frontier.anchors")
        anchor_ids: list[str] = []
        for index, anchor in enumerate(anchors):
            if not isinstance(anchor, ExternalAnchor):
                raise TypeError(f"frontier.anchors[{index}]: expected ExternalAnchor")
            anchor.__post_init__()
            anchor_ids.append(anchor.anchor_id)
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("frontier.anchors: contains duplicate anchor ids")

    @property
    def schema(self) -> str:
        return FRONTIER_RECORD_SCHEMA

    @property
    def points(self) -> tuple[FrontierPoint, ...]:
        """Return all accepted points in candidate declaration order."""

        return tuple(point for candidate in self.candidates for point in candidate.points)


EstimatorInputResolver: TypeAlias = Callable[
    [DeploymentCandidate, int],
    EstimatorInputs,
]
StaticRankBytesResolver: TypeAlias = Callable[
    [DeploymentCandidate],
    Mapping[str, int],
]


@dataclass(slots=True)
class ScanInputs:
    """Estimator, feasibility and optional anchor inputs for one scan."""

    estimator_inputs: EstimatorInputs | EstimatorInputResolver
    static_rank_bytes_per_pool: Mapping[str, int] | StaticRankBytesResolver
    device_hbm_capacity_bytes: Mapping[str, int]
    anchors: tuple[ExternalAnchor, ...] = ()
    configuration_labels: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.estimator_inputs, EstimatorInputs) and not callable(
            self.estimator_inputs
        ):
            raise TypeError(
                "inputs.estimator_inputs: expected EstimatorInputs or callable"
            )
        if isinstance(self.estimator_inputs, EstimatorInputs):
            self.estimator_inputs.__post_init__()
        if not isinstance(self.static_rank_bytes_per_pool, Mapping) and not callable(
            self.static_rank_bytes_per_pool
        ):
            raise TypeError(
                "inputs.static_rank_bytes_per_pool: expected mapping or callable"
            )
        if isinstance(self.static_rank_bytes_per_pool, Mapping):
            _validate_integer_mapping(
                self.static_rank_bytes_per_pool,
                "inputs.static_rank_bytes_per_pool",
                minimum=0,
            )
        _validate_integer_mapping(
            self.device_hbm_capacity_bytes,
            "inputs.device_hbm_capacity_bytes",
            minimum=1,
        )
        anchors = _require_tuple(self.anchors, "inputs.anchors")
        for index, anchor in enumerate(anchors):
            if not isinstance(anchor, ExternalAnchor):
                raise TypeError(f"inputs.anchors[{index}]: expected ExternalAnchor")
            anchor.__post_init__()
        if self.configuration_labels is not None:
            if not isinstance(self.configuration_labels, Mapping):
                raise TypeError("inputs.configuration_labels: expected a mapping")
            for key, label in self.configuration_labels.items():
                _string(key, "inputs.configuration_labels key")
                _string(label, f"inputs.configuration_labels[{key!r}]")

    def resolve_estimator(
        self,
        candidate: DeploymentCandidate,
        batch_per_gpu: int,
    ) -> EstimatorInputs:
        """Resolve and validate the T2 inputs for one candidate and batch."""

        selected = self.estimator_inputs
        resolved = selected(candidate, batch_per_gpu) if callable(selected) else selected
        if not isinstance(resolved, EstimatorInputs):
            raise TypeError("inputs.estimator_inputs resolver: expected EstimatorInputs")
        resolved.__post_init__()
        return resolved

    def resolve_static_rank_bytes(
        self,
        candidate: DeploymentCandidate,
    ) -> Mapping[str, int]:
        """Resolve and validate per-role static rank bytes for one candidate."""

        selected = self.static_rank_bytes_per_pool
        resolved = selected(candidate) if callable(selected) else selected
        return _validate_integer_mapping(
            resolved,
            "inputs.static_rank_bytes_per_pool resolver",
            minimum=0,
        )

    def configuration_label(self, candidate: DeploymentCandidate) -> str:
        """Return the optional display label without changing stable identity."""

        if self.configuration_labels is None:
            return candidate.candidate_id
        return self.configuration_labels.get(candidate.candidate_id, candidate.candidate_id)


def _validate_integer_mapping(
    value: object,
    path: str,
    *,
    minimum: int,
) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected a mapping")
    for key, item in value.items():
        _string(key, f"{path} key")
        _integer(item, f"{path}[{key!r}]", minimum=minimum)
    return value


def _point_to_json(point: FrontierPoint) -> dict[str, object]:
    point.__post_init__()
    return {
        "candidate_key": point.candidate_key,
        "configuration_id": point.configuration_id,
        "batch_per_gpu": point.batch_per_gpu,
        "x_tokens_per_second_per_request": _fraction_to_json(
            point.x_tokens_per_second_per_request
        ),
        "y_tokens_per_second_per_gpu": _fraction_to_json(
            point.y_tokens_per_second_per_gpu
        ),
        "point_class": point.point_class.value,
        "step_ps": point.step_ps,
        "stamp": estimate_stamp_to_json(point.stamp),
    }


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


def frontier_record_to_json(record: FrontierRecord) -> dict[str, object]:
    """Return the strict schema-tagged JSON object for one frontier record."""

    if not isinstance(record, FrontierRecord):
        raise TypeError("frontier: expected FrontierRecord")
    record.__post_init__()
    return {
        "schema": FRONTIER_RECORD_SCHEMA,
        "candidates": [
            {
                "candidate_key": candidate.candidate_key,
                "configuration_id": candidate.configuration_id,
                "configuration_label": candidate.configuration_label,
                "accepted": candidate.accepted,
                "rejection_reasons": list(candidate.rejection_reasons),
                "points": [_point_to_json(point) for point in candidate.points],
            }
            for candidate in record.candidates
        ],
        "anchors": [_anchor_to_json(anchor) for anchor in record.anchors],
    }


def _point_from_json(value: object, path: str) -> FrontierPoint:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={
            "candidate_key",
            "configuration_id",
            "batch_per_gpu",
            "x_tokens_per_second_per_request",
            "y_tokens_per_second_per_gpu",
            "point_class",
            "step_ps",
            "stamp",
        },
    )
    return FrontierPoint(
        candidate_key=_sha256(payload["candidate_key"], f"{path}.candidate_key"),
        configuration_id=_string(
            payload["configuration_id"],
            f"{path}.configuration_id",
        ),
        batch_per_gpu=_integer(
            payload["batch_per_gpu"],
            f"{path}.batch_per_gpu",
            minimum=1,
        ),
        x_tokens_per_second_per_request=_fraction_from_json(
            payload["x_tokens_per_second_per_request"],
            f"{path}.x_tokens_per_second_per_request",
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
        step_ps=_integer(payload["step_ps"], f"{path}.step_ps", minimum=1),
        stamp=estimate_stamp_from_json(payload["stamp"]),
    )


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


def frontier_record_from_json(value: object) -> FrontierRecord:
    """Parse a strict frontier record after checking its schema tag first."""

    payload = _object(value, "frontier")
    schema = _string(payload.get("schema"), "frontier.schema")
    if schema != FRONTIER_RECORD_SCHEMA:
        raise ValueError(
            "frontier.schema: unsupported schema "
            f"{schema!r}; expected {FRONTIER_RECORD_SCHEMA!r}"
        )
    _fields(payload, "frontier", required={"schema", "candidates", "anchors"})
    candidates: list[CandidateFrontier] = []
    for index, raw_candidate in enumerate(
        _array(payload["candidates"], "frontier.candidates")
    ):
        path = f"frontier.candidates[{index}]"
        candidate_payload = _object(raw_candidate, path)
        _fields(
            candidate_payload,
            path,
            required={
                "candidate_key",
                "configuration_id",
                "configuration_label",
                "accepted",
                "rejection_reasons",
                "points",
            },
        )
        reasons = tuple(
            _string(reason, f"{path}.rejection_reasons[{reason_index}]")
            for reason_index, reason in enumerate(
                _array(candidate_payload["rejection_reasons"], f"{path}.rejection_reasons")
            )
        )
        points = tuple(
            _point_from_json(point, f"{path}.points[{point_index}]")
            for point_index, point in enumerate(
                _array(candidate_payload["points"], f"{path}.points")
            )
        )
        candidates.append(
            CandidateFrontier(
                candidate_key=_sha256(
                    candidate_payload["candidate_key"],
                    f"{path}.candidate_key",
                ),
                configuration_id=_string(
                    candidate_payload["configuration_id"],
                    f"{path}.configuration_id",
                ),
                configuration_label=_string(
                    candidate_payload["configuration_label"],
                    f"{path}.configuration_label",
                ),
                accepted=_boolean(candidate_payload["accepted"], f"{path}.accepted"),
                rejection_reasons=reasons,
                points=points,
            )
        )
    anchors = tuple(
        _anchor_from_json(anchor, f"frontier.anchors[{index}]")
        for index, anchor in enumerate(_array(payload["anchors"], "frontier.anchors"))
    )
    return FrontierRecord(candidates=tuple(candidates), anchors=anchors)


def scan(
    candidates: Iterable[DeploymentCandidate],
    batches: Iterable[int],
    inputs: ScanInputs,
) -> FrontierRecord:
    """Price candidates and batches in declaration order without a backend."""

    candidate_values = tuple(candidates)
    batch_values = tuple(batches)
    if not candidate_values:
        raise ValueError("candidates: must not be empty")
    if not batch_values:
        raise ValueError("batches: must not be empty")
    if not isinstance(inputs, ScanInputs):
        raise TypeError("inputs: expected ScanInputs")
    inputs.__post_init__()
    configuration_ids: list[str] = []
    for index, candidate in enumerate(candidate_values):
        if not isinstance(candidate, DeploymentCandidate):
            raise TypeError(f"candidates[{index}]: expected DeploymentCandidate")
        candidate.__post_init__()
        configuration_ids.append(candidate.candidate_id)
    if len(configuration_ids) != len(set(configuration_ids)):
        raise ValueError("candidates: contains duplicate candidate ids")
    for index, batch in enumerate(batch_values):
        _integer(batch, f"batches[{index}]", minimum=1)

    candidate_frontiers: list[CandidateFrontier] = []
    for candidate in candidate_values:
        key = candidate_key(candidate)
        feasibility = check_feasibility(
            candidate,
            static_rank_bytes_per_pool=inputs.resolve_static_rank_bytes(candidate),
            device_hbm_capacity_bytes=inputs.device_hbm_capacity_bytes,
        )
        points: list[FrontierPoint] = []
        if feasibility.accepted:
            for batch in batch_values:
                estimate = estimate_decode_step(
                    candidate,
                    batch,
                    inputs.resolve_estimator(candidate, batch),
                )
                if estimate.step_ps <= 0:
                    raise ValueError("estimate.step_ps: frontier coordinates require positivity")
                point_class = (
                    PointClass.SIMULATED
                    if estimate.stamp.consumes_sim_derived
                    else PointClass.ESTIMATE
                )
                x_value = Fraction(PICOSECONDS_PER_SECOND, estimate.step_ps)
                points.append(
                    FrontierPoint(
                        candidate_key=key,
                        configuration_id=candidate.candidate_id,
                        batch_per_gpu=batch,
                        x_tokens_per_second_per_request=x_value,
                        y_tokens_per_second_per_gpu=batch * x_value,
                        point_class=point_class,
                        step_ps=estimate.step_ps,
                        stamp=estimate.stamp,
                    )
                )
        candidate_frontiers.append(
            CandidateFrontier(
                candidate_key=key,
                configuration_id=candidate.candidate_id,
                configuration_label=inputs.configuration_label(candidate),
                accepted=feasibility.accepted,
                rejection_reasons=feasibility.reasons,
                points=tuple(points),
            )
        )
    return FrontierRecord(
        candidates=tuple(candidate_frontiers),
        anchors=inputs.anchors,
    )


def _point_sort_key(point: FrontierPoint) -> tuple[object, ...]:
    evidence_key = tuple(
        (
            term.name,
            term.estimate.duration_ps,
            term.estimate.evidence.value,
            term.estimate.source,
        )
        for term in point.stamp.terms
    )
    return (
        point.configuration_id,
        point.batch_per_gpu,
        point.candidate_key,
        point.point_class.value,
        point.step_ps,
        evidence_key,
    )


def pareto_front(points: Iterable[FrontierPoint]) -> tuple[FrontierPoint, ...]:
    """Return the exact non-dominated points in canonical deterministic order."""

    values = tuple(points)
    for index, point in enumerate(values):
        if not isinstance(point, FrontierPoint):
            raise TypeError(f"points[{index}]: expected FrontierPoint")
        point.__post_init__()
    front = []
    for candidate in values:
        dominated = any(
            other.x_tokens_per_second_per_request
            >= candidate.x_tokens_per_second_per_request
            and other.y_tokens_per_second_per_gpu
            >= candidate.y_tokens_per_second_per_gpu
            and (
                other.x_tokens_per_second_per_request
                > candidate.x_tokens_per_second_per_request
                or other.y_tokens_per_second_per_gpu
                > candidate.y_tokens_per_second_per_gpu
            )
            for other in values
        )
        if not dominated:
            front.append(candidate)
    return tuple(sorted(front, key=_point_sort_key))


def _term_duration(stamp: EstimateStamp, name: str, *, default: int | None = None) -> int:
    matching = [term.estimate.duration_ps for term in stamp.terms if term.name == name]
    if matching:
        return matching[0]
    if default is not None:
        return default
    raise ValueError(f"point.stamp: missing required term {name!r}")


def _new_record_plot(record: FrontierRecord) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    curves: list[dict[str, Any]] = []
    point_classes: list[dict[str, Any]] = []
    for curve_index, candidate in enumerate(record.candidates):
        if not candidate.accepted:
            continue
        curve_points: list[dict[str, Any]] = []
        for point in sorted(candidate.points, key=lambda value: value.batch_per_gpu):
            kernel = _term_duration(point.stamp, "kernel_floor")
            fabric = _term_duration(point.stamp, "fabric_floor")
            intra = _term_duration(point.stamp, "intra_floor")
            fabric_excess = _term_duration(point.stamp, "fabric_excess", default=0)
            intra_excess = _term_duration(point.stamp, "intra_excess", default=0)
            analytical = max(kernel, fabric, intra)
            after_inter = max(kernel, fabric + fabric_excess, intra)
            inter_attributed = after_inter - analytical
            intra_attributed = point.step_ps - after_inter
            if min(inter_attributed, intra_attributed) < 0:
                raise ValueError("point.stamp: network excess terms do not telescope")
            services = {
                "roofline": kernel,
                "inter-node": fabric + fabric_excess,
                "intra-node": intra + intra_excess,
            }
            maximum = max(services.values())
            owners = [name for name, value in services.items() if value == maximum]
            if len(owners) > 1:
                bottleneck = "co-critical"
            elif owners[0] == "roofline":
                bottleneck = "neither"
            else:
                bottleneck = owners[0]
            curve_points.append(
                {
                    "batch_per_gpu": point.batch_per_gpu,
                    "analytical_x": float(Fraction(PICOSECONDS_PER_SECOND, analytical)),
                    "analytical_y": float(
                        Fraction(point.batch_per_gpu * PICOSECONDS_PER_SECOND, analytical)
                    ),
                    "simulated_x": float(point.x_tokens_per_second_per_request),
                    "simulated_y": float(point.y_tokens_per_second_per_gpu),
                    "inter_node_attributed_ps": inter_attributed,
                    "intra_node_attributed_ps": intra_attributed,
                    "fabric_raw_excess_ps": fabric_excess,
                    "bottleneck": bottleneck,
                }
            )
            point_classes.append(
                {
                    "configuration_id": point.configuration_id,
                    "batch_per_gpu": point.batch_per_gpu,
                    "point_class": point.point_class.value,
                }
            )
        curves.append(
            {
                "id": candidate.configuration_id,
                "label": candidate.configuration_label,
                "color": _LEGACY_COLORS.get(
                    candidate.configuration_id,
                    _FALLBACK_COLORS[curve_index % len(_FALLBACK_COLORS)],
                ),
                "points": curve_points,
            }
        )

    paired = next((anchor for anchor in record.anchors if not anchor.y_only), None)
    y_only = next((anchor for anchor in record.anchors if anchor.y_only), None)
    base = {
        "curves": curves,
        "paired_marker": (
            None
            if paired is None
            else {
                "label": paired.label,
                "x": float(paired.x_tokens_per_second_per_request),
                "y": float(paired.y_tokens_per_second_per_gpu),
            }
        ),
        "y_only_anchor": (
            None
            if y_only is None
            else {"label": y_only.label, "y": float(y_only.y_tokens_per_second_per_gpu)}
        ),
        "status": "deployment estimate",
        "candidate_disclosure": "in-process closed-form deployment estimator",
    }
    return base, point_classes


def _legacy_coordinate(point: Mapping[str, Any], kind: str, axis: str) -> float:
    return float(point[f"{kind}_operating_point"][axis]["decimal"])


def _legacy_plot(result: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw_point in result["points"]:
        point = dict(raw_point)
        grouped.setdefault(point["configuration_id"], []).append(point)
    curves = []
    point_classes = []
    for configuration_id, points in grouped.items():
        points.sort(key=lambda point: point["batch_per_gpu"])
        curves.append(
            {
                "id": configuration_id,
                "label": points[0]["configuration_label"],
                "color": _LEGACY_COLORS[configuration_id],
                "points": [
                    {
                        "batch_per_gpu": point["batch_per_gpu"],
                        "analytical_x": _legacy_coordinate(
                            point,
                            "analytical",
                            "x_tokens_per_second_per_request",
                        ),
                        "analytical_y": _legacy_coordinate(
                            point,
                            "analytical",
                            "y_tokens_per_second_per_gpu",
                        ),
                        "simulated_x": _legacy_coordinate(
                            point,
                            "simulated",
                            "x_tokens_per_second_per_request",
                        ),
                        "simulated_y": _legacy_coordinate(
                            point,
                            "simulated",
                            "y_tokens_per_second_per_gpu",
                        ),
                        "inter_node_attributed_ps": point["accounting"][
                            "inter_node_attributed_ps"
                        ],
                        "intra_node_attributed_ps": point["accounting"][
                            "intra_node_attributed_ps"
                        ],
                        "fabric_raw_excess_ps": point["fabric_attribution"][
                            "raw_excess_ps"
                        ],
                        "bottleneck": point["bottleneck"]["classification"],
                    }
                    for point in points
                ],
            }
        )
        point_classes.extend(
            {
                "configuration_id": configuration_id,
                "batch_per_gpu": point["batch_per_gpu"],
                "point_class": PointClass.SIMULATED.value,
            }
            for point in points
        )
    contract = result["plot_contract"]
    if contract["x"]["scale"] != "log" or contract["y"]["scale"] != "log":
        raise ValueError("the frozen frontier axes must both be logarithmic")
    paired = result["published_context"]["paired"][0]
    y_only = result["published_context"]["y_only"][0]
    base = {
        "curves": curves,
        "paired_marker": {
            "label": paired["label"],
            "x": paired["tokens_per_second_per_node"] / paired["batch_per_node"],
            "y": paired["tokens_per_second_per_node"] / paired["gpus_per_node"],
        },
        "y_only_anchor": {
            "label": y_only["label"],
            "y": y_only["tokens_per_second_per_node"] / y_only["gpus_per_node"],
        },
        "status": result["status"],
        "candidate_disclosure": result["intra_node_candidate_disclosure"],
    }
    return base, point_classes


def _legacy_pareto_refs(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    coordinates: list[tuple[str, int, Fraction, Fraction]] = []
    for point in result["points"]:
        operating_point = point["simulated_operating_point"]
        x_record = operating_point["x_tokens_per_second_per_request"]
        y_record = operating_point["y_tokens_per_second_per_gpu"]
        coordinates.append(
            (
                str(point["configuration_id"]),
                int(point["batch_per_gpu"]),
                Fraction(x_record["numerator"], x_record["denominator"]),
                Fraction(y_record["numerator"], y_record["denominator"]),
            )
        )
    front = []
    for candidate in coordinates:
        dominated = any(
            other[2] >= candidate[2]
            and other[3] >= candidate[3]
            and (other[2] > candidate[2] or other[3] > candidate[3])
            for other in coordinates
        )
        if not dominated:
            front.append(candidate)
    return [
        {"configuration_id": item[0], "batch_per_gpu": item[1]}
        for item in sorted(front, key=lambda item: (item[0], item[1]))
    ]


def prepare_plot_v3(record: FrontierRecord | Mapping[str, Any]) -> dict[str, Any]:
    """Build plot-contract v3 data while preserving every v2 series field."""

    if isinstance(record, FrontierRecord):
        record.__post_init__()
        base, point_classes = _new_record_plot(record)
        front_points = pareto_front(record.points)
        pareto_refs = [
            {
                "configuration_id": point.configuration_id,
                "batch_per_gpu": point.batch_per_gpu,
            }
            for point in front_points
        ]
    elif isinstance(record, Mapping):
        schema = record.get("schema")
        if schema == FRONTIER_RECORD_SCHEMA:
            parsed = frontier_record_from_json(record)
            return prepare_plot_v3(parsed)
        if schema != _LEGACY_RESULT_SCHEMA:
            raise ValueError(
                "record.schema: unsupported schema "
                f"{schema!r}; expected {FRONTIER_RECORD_SCHEMA!r} or "
                f"{_LEGACY_RESULT_SCHEMA!r}"
            )
        base, point_classes = _legacy_plot(record)
        pareto_refs = _legacy_pareto_refs(record)
    else:
        raise TypeError("record: expected FrontierRecord or mapping")

    return {
        **base,
        "schema": PLOT_CONTRACT_V3_SCHEMA,
        "axes": {"x_scale": "log", "y_scale": "log", "optimal_corner": "upper-right"},
        "series_styles": {
            "analytical": {"kind": "line", "style": "solid"},
            "estimated": {"kind": "marker", "marker": "circle", "fill": "none"},
            "simulated": {"kind": "marker", "marker": "circle", "fill": "solid"},
            "measured_paired": {
                "kind": "marker",
                "marker": "diamond",
                "fill": "white",
                "edge": "black",
            },
            "measured_y_only": {"kind": "horizontal-line", "style": "dashed"},
        },
        "point_classes": point_classes,
        "pareto_emphasis": {
            "points": pareto_refs,
            "style": {"line_width": "strong", "marker_edge": "strong"},
        },
    }


__all__ = [
    "FRONTIER_RECORD_SCHEMA",
    "PLOT_CONTRACT_V3_SCHEMA",
    "CandidateFrontier",
    "ExternalAnchor",
    "FrontierPoint",
    "FrontierRecord",
    "PointClass",
    "ScanInputs",
    "frontier_record_from_json",
    "frontier_record_to_json",
    "pareto_front",
    "prepare_plot_v3",
    "scan",
]
