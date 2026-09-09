"""Exact publication intervals over existing deployment frontier coordinates."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from simllm.deploy.frontier import (
    _boolean,
    _fraction,
    _ParetoValue,
    _require_tuple,
    _sha256,
    _string,
)


class PublicationBasis(str, Enum):
    """What the source evidence establishes about a published threshold."""

    SOURCE_EXACT = "SOURCE-EXACT"
    CONDITIONAL = "CONDITIONAL"
    DECLARED = "DECLARED"


@dataclass(frozen=True, slots=True)
class ThresholdInterval:
    """An exact nonempty positive interval with explicit endpoint ownership."""

    lower: Fraction
    upper: Fraction
    lower_closed: bool = True
    upper_closed: bool = True

    def __post_init__(self) -> None:
        _fraction(self.lower, "interval.lower")
        _fraction(self.upper, "interval.upper")
        _boolean(self.lower_closed, "interval.lower_closed")
        _boolean(self.upper_closed, "interval.upper_closed")
        if self.lower > self.upper or (
            self.lower == self.upper and not (self.lower_closed and self.upper_closed)
        ):
            raise ValueError("interval must be nonempty")

    def contains(self, value: Fraction) -> bool:
        """Test exact membership, including singleton endpoints."""

        _fraction(value, "interval member")
        return (self.lower < value or (self.lower_closed and self.lower == value)) and (
            value < self.upper or (self.upper_closed and value == self.upper)
        )


@dataclass(frozen=True, slots=True)
class PublishedThreshold:
    """Source evidence for a tokens/s/request threshold, supplied by its reader.

    This projection validates the declaration, not the referenced file. The
    reader owns source verification and any numeric export assumptions. Display
    text never implicitly becomes an exact source value or a rounding interval.
    """

    published_text: str
    source_sha256: str
    row_id: str
    interval: ThresholdInterval
    basis: PublicationBasis
    assumptions: str

    def __post_init__(self) -> None:
        text = _string(self.published_text, "publication.published_text")
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text) is None or Fraction(text) <= 0:
            raise ValueError("publication.published_text: expected a positive decimal")
        _sha256(self.source_sha256, "publication.source_sha256")
        _string(self.row_id, "publication.row_id")
        if not isinstance(self.interval, ThresholdInterval):
            raise TypeError("publication.interval: expected ThresholdInterval")
        self.interval.__post_init__()
        if not isinstance(self.basis, PublicationBasis):
            raise TypeError("publication.basis: expected PublicationBasis")
        _string(self.assumptions, "publication.assumptions")
        if self.basis is PublicationBasis.SOURCE_EXACT and (
            self.interval.lower != self.interval.upper
        ):
            raise ValueError("exact source evidence requires a singleton interval")

    @property
    def axis(self) -> str:
        return "x_tokens_per_second_per_request"

    @property
    def units(self) -> str:
        return "tokens/s/request"


@dataclass(frozen=True, slots=True)
class FrontierSelection:
    """An immutable coordinate projection joined by the original point ID."""

    point_id: str
    x: Fraction
    y: Fraction

    def __post_init__(self) -> None:
        _string(self.point_id, "selection.point_id")
        _fraction(self.x, "selection.x")
        _fraction(self.y, "selection.y")


@dataclass(frozen=True, slots=True)
class FrontierSelectionSegment:
    """A maximal constant-selection region, including explicit infeasibility."""

    interval: ThresholdInterval
    selection: FrontierSelection | None

    def __post_init__(self) -> None:
        if not isinstance(self.interval, ThresholdInterval):
            raise TypeError("segment.interval: expected ThresholdInterval")
        self.interval.__post_init__()
        if self.selection is not None:
            if not isinstance(self.selection, FrontierSelection):
                raise TypeError("segment.selection: expected FrontierSelection or None")
            self.selection.__post_init__()
            if self.selection.x < self.interval.upper:
                raise ValueError("segment selection is not feasible throughout interval")


@dataclass(frozen=True, slots=True, init=False)
class FrontierPublicationComparison:
    """Complete interval choices; agreement and source qualification stay separate."""

    threshold: PublishedThreshold
    reference_y: Fraction
    quotient_band: tuple[Fraction, Fraction]
    segments: tuple[FrontierSelectionSegment, ...]

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError("comparison results are issued by compare_published_frontier")

    @classmethod
    def _issue(cls, threshold, reference_y, quotient_band, segments):
        result = object.__new__(cls)
        for name, value in (
            ("threshold", threshold), ("reference_y", reference_y),
            ("quotient_band", quotient_band), ("segments", segments),
        ):
            object.__setattr__(result, name, value)
        result.__post_init__()
        return result

    def __post_init__(self) -> None:
        if not isinstance(self.threshold, PublishedThreshold):
            raise TypeError("comparison.threshold: expected PublishedThreshold")
        self.threshold.__post_init__()
        _fraction(self.reference_y, "comparison.reference_y")
        if not isinstance(self.quotient_band, tuple) or len(self.quotient_band) != 2:
            raise TypeError("comparison.quotient_band: expected two-tuple")
        lo, hi = self.quotient_band
        _fraction(lo, "comparison.quotient_band lower")
        _fraction(hi, "comparison.quotient_band upper")
        if lo > hi:
            raise ValueError("comparison.quotient_band: lower exceeds upper")
        _require_tuple(self.segments, "comparison.segments")
        if not self.segments:
            raise ValueError("comparison.segments: must cover the threshold interval")
        for segment in self.segments:
            if not isinstance(segment, FrontierSelectionSegment):
                raise TypeError("comparison.segments: expected FrontierSelectionSegment")
            segment.__post_init__()
        declared = self.threshold.interval
        first, last = self.segments[0].interval, self.segments[-1].interval
        if (first.lower, first.lower_closed, last.upper, last.upper_closed) != (
            declared.lower,
            declared.lower_closed,
            declared.upper,
            declared.upper_closed,
        ):
            raise ValueError("comparison.segments: boundary coverage differs")
        for left, right in zip(self.segments, self.segments[1:]):
            a, b = left.interval, right.interval
            if a.upper != b.lower or a.upper_closed == b.lower_closed:
                raise ValueError("comparison.segments: gap or overlapping endpoint")
            if left.selection == right.selection:
                raise ValueError("comparison.segments: adjacent choices must be merged")

    @property
    def quotients(self) -> tuple[Fraction, ...]:
        """Return the discrete possible values, never filling their hull."""

        return tuple(
            sorted(
                {
                    Fraction() if row.selection is None else row.selection.y / self.reference_y
                    for row in self.segments
                }
            )
        )

    @property
    def quotient_bounds(self) -> tuple[Fraction, Fraction]:
        values = self.quotients
        return values[0], values[-1]

    @property
    def possibly_infeasible(self) -> bool:
        return any(row.selection is None for row in self.segments)

    @property
    def verdict(self) -> str:
        """Classify the whole discrete answer under the unchanged supplied band."""

        lo, hi = self.quotient_band
        within = [lo <= value <= hi for value in self.quotients]
        if all(within):
            return "PASS"
        return "INDETERMINATE" if any(within) else "FAIL"


def _selection_snapshot(
    values: Iterable[_ParetoValue],
    coordinate: Callable[[_ParetoValue], tuple[Fraction, Fraction]],
    identity: Callable[[_ParetoValue], str],
) -> tuple[FrontierSelection, ...]:
    selections = []
    seen = set()
    for index, value in enumerate(values):
        point = coordinate(value)
        if not isinstance(point, tuple) or len(point) != 2:
            raise TypeError(f"values[{index}] coordinate: expected a two-tuple")
        selection = FrontierSelection(identity(value), *point)
        if selection.point_id in seen:
            raise ValueError("frontier contains duplicate stable identities")
        seen.add(selection.point_id)
        selections.append(selection)
    return tuple(selections)


def _select_threshold(
    values: tuple[FrontierSelection, ...], threshold: Fraction
) -> FrontierSelection | None:
    return max(
        (value for value in values if value.x >= threshold),
        key=lambda value: (value.y, value.point_id),
        default=None,
    )


def frontier_at_threshold(
    values: Iterable[_ParetoValue],
    threshold: Fraction,
    *,
    coordinate: Callable[[_ParetoValue], tuple[Fraction, Fraction]],
    identity: Callable[[_ParetoValue], str],
) -> FrontierSelection | None:
    """Keep exact feasibility and the deterministic greatest-(y, ID) tie rule."""

    _fraction(threshold, "threshold")
    return _select_threshold(_selection_snapshot(values, coordinate, identity), threshold)


def compare_published_frontier(
    values: Iterable[_ParetoValue],
    threshold: PublishedThreshold,
    reference_y: Fraction,
    *,
    coordinate: Callable[[_ParetoValue], tuple[Fraction, Fraction]],
    identity: Callable[[_ParetoValue], str],
    quotient_band: tuple[Fraction, Fraction] = (Fraction(3, 4), Fraction(27, 20)),
) -> FrontierPublicationComparison:
    """Enumerate all selections over the declared interval without repricing.

    Open regions and singleton boundaries are evaluated separately using exact
    rational representatives. They are then merged only when the full choice
    agrees. No favorable endpoint or midpoint represents an entire interval.
    """

    if not isinstance(threshold, PublishedThreshold):
        raise TypeError("threshold: expected PublishedThreshold")
    threshold.__post_init__()
    choices = _selection_snapshot(values, coordinate, identity)
    interval = threshold.interval
    boundaries = sorted(
        {
            interval.lower,
            interval.upper,
            *(value.x for value in choices if interval.lower <= value.x <= interval.upper),
        }
    )
    atoms = []
    for index, boundary in enumerate(boundaries):
        if interval.contains(boundary):
            atoms.append(
                FrontierSelectionSegment(
                    ThresholdInterval(boundary, boundary),
                    _select_threshold(choices, boundary),
                )
            )
        if index + 1 < len(boundaries):
            end = boundaries[index + 1]
            atoms.append(
                FrontierSelectionSegment(
                    ThresholdInterval(boundary, end, False, False),
                    _select_threshold(choices, (boundary + end) / 2),
                )
            )
    segments: list[FrontierSelectionSegment] = []
    for atom in atoms:
        if segments and segments[-1].selection == atom.selection:
            prior = segments.pop()
            segments.append(
                FrontierSelectionSegment(
                    ThresholdInterval(
                        prior.interval.lower,
                        atom.interval.upper,
                        prior.interval.lower_closed,
                        atom.interval.upper_closed,
                    ),
                    atom.selection,
                )
            )
        else:
            segments.append(atom)
    return FrontierPublicationComparison._issue(threshold, reference_y, quotient_band, tuple(segments))


__all__ = [
    "FrontierPublicationComparison",
    "FrontierSelection",
    "FrontierSelectionSegment",
    "PublicationBasis",
    "PublishedThreshold",
    "ThresholdInterval",
    "compare_published_frontier",
    "frontier_at_threshold",
]
