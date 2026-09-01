"""Aggregate collective-completion floors fitted on a physical byte axis.

The calibrated object in this module is one timing authority. It represents a
complete measured collective as exactly two fitted quantities inside one
piecewise-linear regime: a zero-byte floor and a slope in picoseconds per
byte. It does not name launch, synchronization, algorithm-selection, packet,
credit, switch or arbitration terms because the source completion table does
not identify them separately.

Callers provide physical bytes. Source-table element counts are converted at
the import boundary before a :class:`CollectiveFloorCell` exists. A fitted
value is ``calibrated`` only for the source operation, dtype, rank count and
byte envelope. Explicit transfer through another fitted curve, or
extrapolation outside that envelope, is marked ``transferred-at-use``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, pairwise

from simllm.core.step import (
    CollectiveServiceEnvironment,
    CollectiveServiceInvocation,
)

COLLECTIVE_FLOOR_CALIBRATED = "calibrated"
COLLECTIVE_FLOOR_TRANSFERRED = "transferred-at-use"
COLLECTIVE_FLOOR_EVIDENCE_CLASSES = (
    COLLECTIVE_FLOOR_CALIBRATED,
    COLLECTIVE_FLOOR_TRANSFERRED,
)

COLLECTIVE_FLOOR_DTYPE_WIDTH_BYTES: Mapping[str, int] = {
    "half": 2,
    "int8": 1,
}


class CollectiveFloorEnvironmentMismatchError(ValueError):
    """Raised when in-situ service and floor environments differ by default."""


class CollectiveServiceFloorTransferError(ValueError):
    """Raised when an in-situ comparison would consume a transferred floor."""


def _require_nonblank(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _ceil_fraction(value: Fraction) -> int:
    return (value.numerator + value.denominator - 1) // value.denominator


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def source_elements_for_bytes(dtype: str, message_bytes: int) -> int:
    """Convert physical bytes to the pinned table's element coordinate."""

    _require_nonblank("dtype", dtype)
    message_bytes = _require_int("message_bytes", message_bytes, minimum=1)
    try:
        width = COLLECTIVE_FLOOR_DTYPE_WIDTH_BYTES[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported collective-floor dtype {dtype!r}") from error
    elements, remainder = divmod(message_bytes, width)
    if remainder:
        raise ValueError(f"message_bytes {message_bytes} is not divisible by {dtype} width {width}")
    return elements


@dataclass(frozen=True)
class CollectiveFloorSourceIdentity:
    """Complete identity of the external table behind a fitted value."""

    artifact_sha256: str
    tool: str
    aiconfigurator_version: str
    aiconfigurator_core_version: str
    system: str
    backend: str
    database_version: str
    row_version: str
    duplicate_resolution: str

    def __post_init__(self) -> None:
        if len(self.artifact_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.artifact_sha256
        ):
            raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")
        for name in (
            "tool",
            "aiconfigurator_version",
            "aiconfigurator_core_version",
            "system",
            "backend",
            "database_version",
            "row_version",
            "duplicate_resolution",
        ):
            _require_nonblank(name, getattr(self, name))

    def as_dict(self) -> dict[str, str]:
        """Return a stable JSON-ready identity record."""

        return {
            "artifact_sha256": self.artifact_sha256,
            "tool": self.tool,
            "aiconfigurator_version": self.aiconfigurator_version,
            "aiconfigurator_core_version": self.aiconfigurator_core_version,
            "system": self.system,
            "backend": self.backend,
            "database_version": self.database_version,
            "row_version": self.row_version,
            "duplicate_resolution": self.duplicate_resolution,
        }


@dataclass(frozen=True)
class CollectiveFloorCell:
    """One training observation after element-to-byte conversion."""

    cell_id: str
    dtype: str
    operation: str
    ranks: int
    source_elements: int
    message_bytes: int
    latency_ps: int

    def __post_init__(self) -> None:
        _require_nonblank("cell_id", self.cell_id)
        _require_nonblank("dtype", self.dtype)
        _require_nonblank("operation", self.operation)
        _require_int("ranks", self.ranks, minimum=2)
        _require_int("source_elements", self.source_elements, minimum=1)
        _require_int("message_bytes", self.message_bytes, minimum=1)
        _require_int("latency_ps", self.latency_ps, minimum=1)
        try:
            width = COLLECTIVE_FLOOR_DTYPE_WIDTH_BYTES[self.dtype]
        except KeyError as error:
            raise ValueError(f"unsupported collective-floor dtype {self.dtype!r}") from error
        if self.message_bytes != self.source_elements * width:
            raise ValueError("message_bytes disagrees with source elements and dtype")

    @property
    def curve_key(self) -> tuple[str, str, int]:
        """Return ``(dtype, operation, ranks)`` for grouping."""

        return self.dtype, self.operation, self.ranks


@dataclass(frozen=True)
class CollectiveFloorCurveBoundaries:
    """Training-cell byte values that start later regimes on one curve."""

    dtype: str
    operation: str
    ranks: int
    lower_bounds_of_following_regimes: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_nonblank("dtype", self.dtype)
        _require_nonblank("operation", self.operation)
        _require_int("ranks", self.ranks, minimum=2)
        object.__setattr__(
            self,
            "lower_bounds_of_following_regimes",
            tuple(self.lower_bounds_of_following_regimes),
        )
        bounds = self.lower_bounds_of_following_regimes
        if len(bounds) > 2:
            raise ValueError("a collective-floor curve supports at most three regimes")
        if tuple(sorted(bounds)) != bounds or len(set(bounds)) != len(bounds):
            raise ValueError("regime boundaries must be unique and increasing")
        for index, boundary in enumerate(bounds):
            _require_int(f"regime boundary {index}", boundary, minimum=1)

    @property
    def curve_key(self) -> tuple[str, str, int]:
        return self.dtype, self.operation, self.ranks


@dataclass(frozen=True)
class CollectiveFloorRegime:
    """One positive fitted line and the training cells that identify it."""

    dtype: str
    operation: str
    ranks: int
    regime_index: int
    lower_bytes: int
    upper_bytes: int
    floor_ps: Fraction
    slope_ps_per_byte: Fraction
    training_cell_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonblank("dtype", self.dtype)
        _require_nonblank("operation", self.operation)
        _require_int("ranks", self.ranks, minimum=2)
        _require_int("regime_index", self.regime_index)
        _require_int("lower_bytes", self.lower_bytes, minimum=1)
        _require_int("upper_bytes", self.upper_bytes, minimum=self.lower_bytes)
        if not isinstance(self.floor_ps, Fraction) or self.floor_ps <= 0:
            raise ValueError("floor_ps must be a positive Fraction")
        if not isinstance(self.slope_ps_per_byte, Fraction) or self.slope_ps_per_byte <= 0:
            raise ValueError("slope_ps_per_byte must be a positive Fraction")
        object.__setattr__(self, "training_cell_ids", tuple(self.training_cell_ids))
        if len(self.training_cell_ids) < 2:
            raise ValueError("a fitted regime needs at least two training cells")
        if len(set(self.training_cell_ids)) != len(self.training_cell_ids):
            raise ValueError("training_cell_ids must be unique")
        for cell_id in self.training_cell_ids:
            _require_nonblank("training cell ID", cell_id)

    @property
    def curve_key(self) -> tuple[str, str, int]:
        return self.dtype, self.operation, self.ranks

    @property
    def effective_bandwidth_bytes_per_second(self) -> Fraction:
        """Return the slope's exact effective bytes per second."""

        return Fraction(1_000_000_000_000, 1) / self.slope_ps_per_byte

    def as_dict(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "operation": self.operation,
            "ranks": self.ranks,
            "regime_index": self.regime_index,
            "lower_bytes": self.lower_bytes,
            "upper_bytes": self.upper_bytes,
            "floor_ps": _fraction_text(self.floor_ps),
            "slope_ps_per_byte": _fraction_text(self.slope_ps_per_byte),
            "effective_bandwidth_bytes_per_second": _fraction_text(
                self.effective_bandwidth_bytes_per_second
            ),
            "training_cell_ids": list(self.training_cell_ids),
        }


@dataclass(frozen=True)
class CollectiveFloorEstimate:
    """One served aggregate completion with its complete evidence identity."""

    calibration_id: str
    requested_dtype: str
    requested_operation: str
    requested_ranks: int
    message_bytes: int
    completion_ps: int
    floor_charge_ps: int
    serialization_ps: int
    evidence_class: str
    transfer_reason: str | None
    source: CollectiveFloorSourceIdentity
    regime: CollectiveFloorRegime

    def __post_init__(self) -> None:
        _require_nonblank("calibration_id", self.calibration_id)
        _require_nonblank("requested_dtype", self.requested_dtype)
        _require_nonblank("requested_operation", self.requested_operation)
        _require_int("requested_ranks", self.requested_ranks, minimum=2)
        _require_int("message_bytes", self.message_bytes, minimum=1)
        _require_int("completion_ps", self.completion_ps, minimum=1)
        _require_int("floor_charge_ps", self.floor_charge_ps, minimum=1)
        _require_int("serialization_ps", self.serialization_ps)
        if self.completion_ps != self.floor_charge_ps + self.serialization_ps:
            raise ValueError("completion does not equal floor plus serialization")
        if self.evidence_class not in COLLECTIVE_FLOOR_EVIDENCE_CLASSES:
            raise ValueError("unsupported collective-floor evidence class")
        if self.evidence_class == COLLECTIVE_FLOOR_CALIBRATED:
            if self.transfer_reason is not None:
                raise ValueError("a calibrated estimate cannot carry a transfer reason")
        else:
            _require_nonblank("transfer_reason", self.transfer_reason)
        if not isinstance(self.source, CollectiveFloorSourceIdentity):
            raise TypeError("source must be a CollectiveFloorSourceIdentity")
        if not isinstance(self.regime, CollectiveFloorRegime):
            raise TypeError("regime must be a CollectiveFloorRegime")

    def as_dict(self) -> dict[str, object]:
        return {
            "calibration_id": self.calibration_id,
            "requested_dtype": self.requested_dtype,
            "requested_operation": self.requested_operation,
            "requested_ranks": self.requested_ranks,
            "message_bytes": self.message_bytes,
            "completion_ps": self.completion_ps,
            "floor_charge_ps": self.floor_charge_ps,
            "serialization_ps": self.serialization_ps,
            "evidence_class": self.evidence_class,
            "transfer_reason": self.transfer_reason,
            "source": self.source.as_dict(),
            "regime": self.regime.as_dict(),
        }


@dataclass(frozen=True)
class CollectiveServiceFloorComparison:
    """One exact-coordinate comparison of observed service with its floor."""

    invocation: CollectiveServiceInvocation
    environment: CollectiveServiceEnvironment
    estimate: CollectiveFloorEstimate
    residual_ps: int
    observed_to_floor_ratio: Fraction
    cross_environment_acknowledged: bool
    transferred_at_use_acknowledged: bool

    def __post_init__(self) -> None:
        if not isinstance(self.invocation, CollectiveServiceInvocation):
            raise TypeError("invocation must be a CollectiveServiceInvocation")
        if not isinstance(self.environment, CollectiveServiceEnvironment):
            raise TypeError("environment must be a CollectiveServiceEnvironment")
        if not isinstance(self.estimate, CollectiveFloorEstimate):
            raise TypeError("estimate must be a CollectiveFloorEstimate")
        if type(self.residual_ps) is not int:
            raise TypeError("residual_ps must be an integer")
        if not isinstance(self.observed_to_floor_ratio, Fraction):
            raise TypeError("observed_to_floor_ratio must be a Fraction")
        for name in (
            "cross_environment_acknowledged",
            "transferred_at_use_acknowledged",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if self.invocation.kind != self.estimate.requested_operation:
            raise ValueError("collective kind disagrees with floor operation")
        if self.invocation.payload_bytes != self.estimate.message_bytes:
            raise ValueError("collective payload disagrees with floor byte coordinate")
        if self.invocation.world_size != self.estimate.requested_ranks:
            raise ValueError("collective world size disagrees with floor rank coordinate")
        if self.residual_ps != (self.invocation.service_ps - self.estimate.completion_ps):
            raise ValueError("residual_ps disagrees with observation and floor")
        expected_ratio = Fraction(self.invocation.service_ps, self.estimate.completion_ps)
        if self.observed_to_floor_ratio != expected_ratio:
            raise ValueError("observed_to_floor_ratio is inconsistent")
        environment_differs = (
            self.environment.system,
            self.environment.backend,
        ) != (self.estimate.source.system, self.estimate.source.backend)
        if self.cross_environment_acknowledged != environment_differs:
            raise ValueError("cross-environment use requires an acknowledgement stamp")
        estimate_transferred = self.estimate.evidence_class == COLLECTIVE_FLOOR_TRANSFERRED
        if self.transferred_at_use_acknowledged != estimate_transferred:
            raise ValueError("transferred-at-use floor requires an acknowledgement stamp")

    def as_dict(self) -> dict[str, object]:
        """Return the score and both acknowledgement stamps as JSON-ready data."""

        return {
            "sequence": self.invocation.sequence,
            "kind": self.invocation.kind,
            "payload_bytes": self.invocation.payload_bytes,
            "world_size": self.invocation.world_size,
            "observed_service_ps": self.invocation.service_ps,
            "floor_completion_ps": self.estimate.completion_ps,
            "residual_ps": self.residual_ps,
            "observed_to_floor_ratio": _fraction_text(self.observed_to_floor_ratio),
            "capture_environment": {
                "system": self.environment.system,
                "backend": self.environment.backend,
            },
            "floor_environment": {
                "system": self.estimate.source.system,
                "backend": self.estimate.source.backend,
            },
            "cross_environment_acknowledged": (self.cross_environment_acknowledged),
            "transferred_at_use_acknowledged": (self.transferred_at_use_acknowledged),
            "estimate": self.estimate.as_dict(),
        }


@dataclass(frozen=True)
class CollectiveFloorCalibration:
    """Immutable piecewise aggregate completion authority."""

    calibration_id: str
    source: CollectiveFloorSourceIdentity
    fitted_byte_range: tuple[int, int]
    regimes: tuple[CollectiveFloorRegime, ...]
    input_surface: tuple[str, ...] = (
        "external_nccl_training_cells",
        "element_to_byte_width",
        "training_only_regime_boundaries",
        "source_identity",
    )

    def __post_init__(self) -> None:
        _require_nonblank("calibration_id", self.calibration_id)
        if not isinstance(self.source, CollectiveFloorSourceIdentity):
            raise TypeError("source must be a CollectiveFloorSourceIdentity")
        if not isinstance(self.fitted_byte_range, tuple) or len(self.fitted_byte_range) != 2:
            raise TypeError("fitted_byte_range must be a two-item tuple")
        minimum = _require_int("fitted_byte_range[0]", self.fitted_byte_range[0], minimum=1)
        _require_int("fitted_byte_range[1]", self.fitted_byte_range[1], minimum=minimum)
        object.__setattr__(self, "regimes", tuple(self.regimes))
        object.__setattr__(self, "input_surface", tuple(self.input_surface))
        if not self.regimes:
            raise ValueError("a calibration needs at least one fitted regime")
        if any(not isinstance(regime, CollectiveFloorRegime) for regime in self.regimes):
            raise TypeError("regimes must contain CollectiveFloorRegime values")
        if len(set(self.input_surface)) != len(self.input_surface):
            raise ValueError("input_surface must contain unique names")
        for name in self.input_surface:
            _require_nonblank("input surface name", name)

        by_curve: dict[tuple[str, str, int], list[CollectiveFloorRegime]] = defaultdict(list)
        for regime in self.regimes:
            by_curve[regime.curve_key].append(regime)
        for key, curve in by_curve.items():
            ordered = sorted(curve, key=lambda regime: regime.regime_index)
            if [regime.regime_index for regime in ordered] != list(range(len(ordered))):
                raise ValueError(f"curve {key!r} regime indexes are not contiguous")
            if ordered[0].lower_bytes != minimum:
                raise ValueError(f"curve {key!r} does not start at fitted byte minimum")
            if ordered[-1].upper_bytes != self.fitted_byte_range[1]:
                raise ValueError(f"curve {key!r} does not end at fitted byte maximum")
            for previous, following in pairwise(ordered):
                if previous.upper_bytes + 1 != following.lower_bytes:
                    raise ValueError(f"curve {key!r} regimes have a gap or overlap")

    @property
    def curve_keys(self) -> tuple[tuple[str, str, int], ...]:
        return tuple(sorted({regime.curve_key for regime in self.regimes}))

    def _curve(self, key: tuple[str, str, int]) -> tuple[CollectiveFloorRegime, ...]:
        curve = tuple(
            sorted(
                (regime for regime in self.regimes if regime.curve_key == key),
                key=lambda regime: regime.regime_index,
            )
        )
        if not curve:
            raise ValueError(f"collective-floor donor curve {key!r} is not fitted")
        return curve

    def estimate(
        self,
        *,
        dtype: str,
        operation: str,
        ranks: int,
        message_bytes: int,
        donor: tuple[str, str, int] | None = None,
    ) -> CollectiveFloorEstimate:
        """Serve one value, downgrading every explicit or range transfer."""

        _require_nonblank("dtype", dtype)
        _require_nonblank("operation", operation)
        _require_int("ranks", ranks, minimum=2)
        message_bytes = _require_int("message_bytes", message_bytes, minimum=1)
        requested_key = (dtype, operation, ranks)
        donor_key = requested_key if donor is None else donor
        if not isinstance(donor_key, tuple) or len(donor_key) != 3:
            raise TypeError("donor must be a (dtype, operation, ranks) tuple")
        donor_dtype, donor_operation, donor_ranks = donor_key
        _require_nonblank("donor dtype", donor_dtype)
        _require_nonblank("donor operation", donor_operation)
        _require_int("donor ranks", donor_ranks, minimum=2)
        curve = self._curve((donor_dtype, donor_operation, donor_ranks))
        regime = next(
            (
                candidate
                for candidate in curve
                if candidate.lower_bytes <= message_bytes <= candidate.upper_bytes
            ),
            curve[0] if message_bytes < self.fitted_byte_range[0] else curve[-1],
        )
        total = regime.floor_ps + regime.slope_ps_per_byte * message_bytes
        floor_charge_ps = _ceil_fraction(regime.floor_ps)
        completion_ps = _ceil_fraction(total)

        transfer_reasons = []
        if requested_key != donor_key:
            transfer_reasons.append(f"requested curve {requested_key!r} uses donor {donor_key!r}")
        minimum, maximum = self.fitted_byte_range
        if not minimum <= message_bytes <= maximum:
            transfer_reasons.append(
                f"message_bytes {message_bytes} is outside fitted range [{minimum}, {maximum}]"
            )
        evidence_class = (
            COLLECTIVE_FLOOR_TRANSFERRED if transfer_reasons else COLLECTIVE_FLOOR_CALIBRATED
        )
        return CollectiveFloorEstimate(
            calibration_id=self.calibration_id,
            requested_dtype=dtype,
            requested_operation=operation,
            requested_ranks=ranks,
            message_bytes=message_bytes,
            completion_ps=completion_ps,
            floor_charge_ps=floor_charge_ps,
            serialization_ps=completion_ps - floor_charge_ps,
            evidence_class=evidence_class,
            transfer_reason=("; ".join(transfer_reasons) or None),
            source=self.source,
            regime=regime,
        )


def compare_collective_service_to_floor(
    *,
    invocation: CollectiveServiceInvocation,
    environment: CollectiveServiceEnvironment,
    calibration: CollectiveFloorCalibration,
    floor_dtype: str,
    acknowledge_cross_environment: bool = False,
    acknowledge_floor_transfer: bool = False,
) -> CollectiveServiceFloorComparison:
    """Score one observation at matching kind, byte and rank coordinates.

    System/backend disagreement and the aggregate authority's existing
    transferred-at-use state are separate refusal surfaces. Each deliberate
    override is stamped into the returned comparison record.
    """

    if not isinstance(invocation, CollectiveServiceInvocation):
        raise TypeError("invocation must be a CollectiveServiceInvocation")
    if not isinstance(environment, CollectiveServiceEnvironment):
        raise TypeError("environment must be a CollectiveServiceEnvironment")
    if not isinstance(calibration, CollectiveFloorCalibration):
        raise TypeError("calibration must be a CollectiveFloorCalibration")
    _require_nonblank("floor_dtype", floor_dtype)
    if type(acknowledge_cross_environment) is not bool:
        raise TypeError("acknowledge_cross_environment must be a boolean")
    if type(acknowledge_floor_transfer) is not bool:
        raise TypeError("acknowledge_floor_transfer must be a boolean")

    environment_differs = (environment.system, environment.backend) != (
        calibration.source.system,
        calibration.source.backend,
    )
    if environment_differs and not acknowledge_cross_environment:
        raise CollectiveFloorEnvironmentMismatchError(
            "in-situ collective comparison refuses capture environment "
            f"{(environment.system, environment.backend)!r} against floor "
            f"environment {(calibration.source.system, calibration.source.backend)!r}. "
            "Pass acknowledge_cross_environment=True only for a deliberate "
            "cross-environment diagnostic."
        )

    estimate = calibration.estimate(
        dtype=floor_dtype,
        operation=invocation.kind,
        ranks=invocation.world_size,
        message_bytes=invocation.payload_bytes,
    )
    estimate_transferred = estimate.evidence_class == COLLECTIVE_FLOOR_TRANSFERRED
    if estimate_transferred and not acknowledge_floor_transfer:
        raise CollectiveServiceFloorTransferError(
            "in-situ collective comparison refuses transferred-at-use floor "
            f"estimate: {estimate.transfer_reason}. Pass "
            "acknowledge_floor_transfer=True only for a deliberate transferred run."
        )
    return CollectiveServiceFloorComparison(
        invocation=invocation,
        environment=environment,
        estimate=estimate,
        residual_ps=invocation.service_ps - estimate.completion_ps,
        observed_to_floor_ratio=Fraction(invocation.service_ps, estimate.completion_ps),
        cross_environment_acknowledged=environment_differs,
        transferred_at_use_acknowledged=estimate_transferred,
    )


def _weighted_relative_fit(
    cells: Sequence[CollectiveFloorCell],
) -> tuple[Fraction, Fraction, Fraction]:
    """Return floor, slope and relative SSE using exact rational arithmetic."""

    if len(cells) < 2:
        raise ValueError("a fitted regime needs at least two training cells")
    weights = tuple(Fraction(1, cell.latency_ps**2) for cell in cells)
    sum_w = sum(weights, Fraction())
    sum_wx = sum(
        (weight * cell.message_bytes for weight, cell in zip(weights, cells)),
        Fraction(),
    )
    sum_wy = sum(
        (weight * cell.latency_ps for weight, cell in zip(weights, cells)),
        Fraction(),
    )
    sum_wxx = sum(
        (weight * cell.message_bytes * cell.message_bytes for weight, cell in zip(weights, cells)),
        Fraction(),
    )
    sum_wxy = sum(
        (weight * cell.message_bytes * cell.latency_ps for weight, cell in zip(weights, cells)),
        Fraction(),
    )
    denominator = sum_w * sum_wxx - sum_wx * sum_wx
    if denominator == 0:
        raise ValueError("training cells do not span two distinct byte sizes")
    slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denominator
    floor = (sum_wy - slope * sum_wx) / sum_w
    relative_sse = sum(
        ((floor + slope * cell.message_bytes - cell.latency_ps) / cell.latency_ps) ** 2
        for cell in cells
    )
    return floor, slope, relative_sse


def choose_collective_floor_boundaries(
    cells: Sequence[CollectiveFloorCell],
    *,
    maximum_regimes: int = 3,
    minimum_cells_per_regime: int = 2,
) -> tuple[int, ...]:
    """Choose positive fitted regimes by the frozen training-only BIC rule."""

    maximum_regimes = _require_int("maximum_regimes", maximum_regimes, minimum=1)
    minimum_cells_per_regime = _require_int(
        "minimum_cells_per_regime", minimum_cells_per_regime, minimum=2
    )
    ordered = tuple(sorted(cells, key=lambda cell: cell.message_bytes))
    if len({cell.curve_key for cell in ordered}) != 1:
        raise ValueError("boundary selection accepts exactly one curve")
    if len({cell.message_bytes for cell in ordered}) != len(ordered):
        raise ValueError("training curve contains duplicate byte sizes")
    candidates: list[tuple[float, int, tuple[int, ...]]] = []
    count = len(ordered)
    for regime_count in range(1, maximum_regimes + 1):
        for cuts in combinations(range(1, count), regime_count - 1):
            starts = (0, *cuts)
            ends = (*cuts, count)
            if any(end - start < minimum_cells_per_regime for start, end in zip(starts, ends)):
                continue
            relative_sse = Fraction()
            positive = True
            for start, end in zip(starts, ends):
                floor, slope, segment_sse = _weighted_relative_fit(ordered[start:end])
                if floor <= 0 or slope <= 0:
                    positive = False
                    break
                relative_sse += segment_sse
            if not positive or relative_sse <= 0:
                continue
            parameter_count = 3 * regime_count - 1
            bic = count * math.log(float(relative_sse) / count) + (
                parameter_count * math.log(count)
            )
            boundaries = tuple(ordered[index].message_bytes for index in cuts)
            candidates.append((bic, regime_count, boundaries))
    if not candidates:
        raise ValueError("no positive collective-floor segmentation is available")
    _, _, boundaries = min(candidates)
    return boundaries


def fit_collective_floor_calibration(
    *,
    calibration_id: str,
    source: CollectiveFloorSourceIdentity,
    cells: Sequence[CollectiveFloorCell],
    boundaries: Sequence[CollectiveFloorCurveBoundaries],
    fitted_byte_range: tuple[int, int],
) -> CollectiveFloorCalibration:
    """Fit every frozen curve without consulting a holdout cell."""

    _require_nonblank("calibration_id", calibration_id)
    if not isinstance(source, CollectiveFloorSourceIdentity):
        raise TypeError("source must be a CollectiveFloorSourceIdentity")
    minimum, maximum = fitted_byte_range
    _require_int("fitted_byte_range[0]", minimum, minimum=1)
    _require_int("fitted_byte_range[1]", maximum, minimum=minimum)
    grouped: dict[tuple[str, str, int], list[CollectiveFloorCell]] = defaultdict(list)
    cell_ids = set()
    for cell in cells:
        if not isinstance(cell, CollectiveFloorCell):
            raise TypeError("cells must contain CollectiveFloorCell values")
        if cell.cell_id in cell_ids:
            raise ValueError(f"duplicate training cell ID {cell.cell_id!r}")
        cell_ids.add(cell.cell_id)
        grouped[cell.curve_key].append(cell)
    boundary_by_key: dict[tuple[str, str, int], CollectiveFloorCurveBoundaries] = {}
    for boundary in boundaries:
        if not isinstance(boundary, CollectiveFloorCurveBoundaries):
            raise TypeError("boundaries must contain CollectiveFloorCurveBoundaries values")
        if boundary.curve_key in boundary_by_key:
            raise ValueError(f"duplicate boundary curve {boundary.curve_key!r}")
        boundary_by_key[boundary.curve_key] = boundary
    if set(grouped) != set(boundary_by_key):
        raise ValueError("training curves and frozen boundary curves disagree")

    regimes = []
    for key in sorted(grouped):
        curve = tuple(sorted(grouped[key], key=lambda cell: cell.message_bytes))
        if len({cell.message_bytes for cell in curve}) != len(curve):
            raise ValueError(f"training curve {key!r} contains duplicate byte sizes")
        frozen = boundary_by_key[key].lower_bounds_of_following_regimes
        training_bytes = {cell.message_bytes for cell in curve}
        if any(boundary not in training_bytes for boundary in frozen):
            raise ValueError(f"curve {key!r} boundary is not a training byte size")
        split_indexes = tuple(
            next(index for index, cell in enumerate(curve) if cell.message_bytes == value)
            for value in frozen
        )
        starts = (0, *split_indexes)
        ends = (*split_indexes, len(curve))
        regime_lowers = (minimum, *frozen)
        regime_uppers = tuple(value - 1 for value in frozen) + (maximum,)
        for regime_index, (start, end, lower, upper) in enumerate(
            zip(starts, ends, regime_lowers, regime_uppers, strict=True)
        ):
            training = curve[start:end]
            floor, slope, _ = _weighted_relative_fit(training)
            if floor <= 0 or slope <= 0:
                raise ValueError(f"curve {key!r} regime {regime_index} does not fit positive terms")
            regimes.append(
                CollectiveFloorRegime(
                    dtype=key[0],
                    operation=key[1],
                    ranks=key[2],
                    regime_index=regime_index,
                    lower_bytes=lower,
                    upper_bytes=upper,
                    floor_ps=floor,
                    slope_ps_per_byte=slope,
                    training_cell_ids=tuple(cell.cell_id for cell in training),
                )
            )
    return CollectiveFloorCalibration(
        calibration_id=calibration_id,
        source=source,
        fitted_byte_range=fitted_byte_range,
        regimes=tuple(regimes),
    )


def distribute_collective_serialization_ps(
    serialization_ps: int,
    phase_count: int,
) -> tuple[int, ...]:
    """Split an aggregate slope term exactly over ordered collective phases."""

    serialization_ps = _require_int("serialization_ps", serialization_ps)
    phase_count = _require_int("phase_count", phase_count, minimum=1)
    quotient, remainder = divmod(serialization_ps, phase_count)
    return tuple(quotient + (1 if index < remainder else 0) for index in range(phase_count))


__all__ = [
    "COLLECTIVE_FLOOR_CALIBRATED",
    "COLLECTIVE_FLOOR_DTYPE_WIDTH_BYTES",
    "COLLECTIVE_FLOOR_EVIDENCE_CLASSES",
    "COLLECTIVE_FLOOR_TRANSFERRED",
    "CollectiveFloorCalibration",
    "CollectiveFloorCell",
    "CollectiveFloorCurveBoundaries",
    "CollectiveFloorEstimate",
    "CollectiveFloorRegime",
    "CollectiveFloorSourceIdentity",
    "choose_collective_floor_boundaries",
    "distribute_collective_serialization_ps",
    "fit_collective_floor_calibration",
    "source_elements_for_bytes",
]
