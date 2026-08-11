"""Causal divergence accounting for independent pre-play oracle checks.

This module does not run either framework. It compares two already captured
observations and admits only the PLAY-5 taxonomy frozen by the validation
study. In particular, an unclassified difference remains a hard failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class DivergenceKind(StrEnum):
    """Root classifications admitted by the PLAY-5 study."""

    SAMPLER_DIFFERENCE = "sampler-difference"
    NUMERICS_NEAR_TIE_FLIP = "numerics-near-tie-flip"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True, kw_only=True)
class DecisionBoundary:
    """Selected identities and the measured decision-boundary margin."""

    selected_ids: tuple[int, ...]
    boundary_ids: tuple[int, int]
    margin: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_ids", tuple(self.selected_ids))
        object.__setattr__(self, "boundary_ids", tuple(self.boundary_ids))
        if not self.selected_ids:
            raise ValueError("selected_ids must not be empty")
        if any(type(value) is not int or value < 0 for value in self.selected_ids):
            raise ValueError("selected_ids must contain nonnegative integers")
        if len(set(self.selected_ids)) != len(self.selected_ids):
            raise ValueError("selected_ids must be unique")
        if len(self.boundary_ids) != 2 or any(
            type(value) is not int or value < 0 for value in self.boundary_ids
        ):
            raise ValueError("boundary_ids must contain two nonnegative integers")
        if self.boundary_ids[0] == self.boundary_ids[1]:
            raise ValueError("boundary_ids must be distinct")
        if not isinstance(self.margin, int | float) or not math.isfinite(self.margin):
            raise ValueError("margin must be finite")
        if self.margin < 0:
            raise ValueError("margin must be nonnegative")


@dataclass(frozen=True, kw_only=True)
class RoutedDecision:
    """One forwarded input token's expert selection at one layer."""

    phase: str
    token_index: int
    token_id: int
    layer_index: int
    boundary: DecisionBoundary

    def __post_init__(self) -> None:
        if self.phase not in ("prefill", "decode"):
            raise ValueError("phase must be prefill or decode")
        for name, value in (
            ("token_index", self.token_index),
            ("token_id", self.token_id),
            ("layer_index", self.layer_index),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.boundary, DecisionBoundary):
            raise TypeError("boundary must be a DecisionBoundary")

    @property
    def key(self) -> tuple[str, int, int]:
        return self.phase, self.token_index, self.layer_index


@dataclass(frozen=True, kw_only=True)
class OracleRequestObservation:
    """Framework-neutral request result plus decision diagnostics."""

    request_id: str
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    stop_reason: str
    token_boundaries: tuple[DecisionBoundary, ...]
    routed_decisions: tuple[RoutedDecision, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_token_ids", tuple(self.input_token_ids))
        object.__setattr__(self, "output_token_ids", tuple(self.output_token_ids))
        object.__setattr__(self, "token_boundaries", tuple(self.token_boundaries))
        object.__setattr__(self, "routed_decisions", tuple(self.routed_decisions))
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a nonempty string")
        if not self.input_token_ids or any(
            type(value) is not int or value < 0 for value in self.input_token_ids
        ):
            raise ValueError("input_token_ids must contain nonnegative integers")
        if not self.output_token_ids or any(
            type(value) is not int or value < 0 for value in self.output_token_ids
        ):
            raise ValueError("output_token_ids must contain nonnegative integers")
        if not isinstance(self.stop_reason, str) or not self.stop_reason:
            raise ValueError("stop_reason must be a nonempty string")
        if len(self.token_boundaries) != len(self.output_token_ids):
            raise ValueError("token boundary count must equal output token count")
        if any(not isinstance(boundary, DecisionBoundary) for boundary in self.token_boundaries):
            raise TypeError("token_boundaries must contain DecisionBoundary values")
        keys = []
        for decision in self.routed_decisions:
            if not isinstance(decision, RoutedDecision):
                raise TypeError("routed_decisions must contain RoutedDecision values")
            keys.append(decision.key)
        if len(keys) != len(set(keys)):
            raise ValueError("routed_decisions contain a duplicate phase/token/layer key")


@dataclass(frozen=True, kw_only=True)
class OracleDivergence:
    """One exact field difference and its causal classification."""

    field: str
    location: str
    classification: DivergenceKind
    cascade: bool
    detail: str


@dataclass(frozen=True, kw_only=True)
class OracleComparison:
    """Complete comparison for one request and sampling mode."""

    request_id: str
    sampling_mode: str
    divergences: tuple[OracleDivergence, ...]
    input_tokens_exact: bool
    output_tokens_exact: bool
    output_length_exact: bool
    stop_reason_exact: bool
    routing_exact: bool

    @property
    def unclassified_count(self) -> int:
        return sum(
            divergence.classification is DivergenceKind.UNCLASSIFIED
            for divergence in self.divergences
        )

    @property
    def passed(self) -> bool:
        return self.unclassified_count == 0


def _token_near_tie(
    left: DecisionBoundary,
    right: DecisionBoundary,
    left_token: int,
    right_token: int,
    threshold: float,
) -> bool:
    changed = {left_token, right_token}
    candidates = set(left.boundary_ids) | set(right.boundary_ids)
    return changed <= candidates and min(left.margin, right.margin) <= threshold


def _routing_near_tie(
    left: DecisionBoundary,
    right: DecisionBoundary,
    threshold: float,
) -> bool:
    left_only = set(left.selected_ids) - set(right.selected_ids)
    right_only = set(right.selected_ids) - set(left.selected_ids)
    if len(left_only) != 1 or len(right_only) != 1:
        return False
    left_expert = next(iter(left_only))
    right_expert = next(iter(right_only))
    return (
        left.boundary_ids == (left_expert, right_expert)
        and right.boundary_ids == (right_expert, left_expert)
        and min(left.margin, right.margin) <= threshold
    )


def compare_oracle_requests(
    left: OracleRequestObservation,
    right: OracleRequestObservation,
    *,
    sampling_mode: str,
    near_tie_abs_logit: float,
) -> OracleComparison:
    """Compare one request without silently accepting an unknown difference."""

    if left.request_id != right.request_id:
        raise ValueError("request identities differ")
    if sampling_mode not in ("greedy", "seeded-sampling"):
        raise ValueError("sampling_mode must be greedy or seeded-sampling")
    if (
        not isinstance(near_tie_abs_logit, int | float)
        or not math.isfinite(near_tie_abs_logit)
        or near_tie_abs_logit <= 0
    ):
        raise ValueError("near_tie_abs_logit must be positive and finite")

    divergences: list[OracleDivergence] = []
    root: DivergenceKind | None = None
    first_token_difference: int | None = None

    input_tokens_exact = left.input_token_ids == right.input_token_ids
    if not input_tokens_exact:
        divergences.append(
            OracleDivergence(
                field="input_token_ids",
                location="prompt",
                classification=DivergenceKind.UNCLASSIFIED,
                cascade=False,
                detail="the two runners did not consume identical prompt token IDs",
            )
        )

    common_outputs = min(len(left.output_token_ids), len(right.output_token_ids))
    for index in range(common_outputs):
        left_token = left.output_token_ids[index]
        right_token = right.output_token_ids[index]
        if left_token == right_token:
            continue
        if first_token_difference is None:
            first_token_difference = index
            if sampling_mode == "seeded-sampling" and input_tokens_exact:
                root = DivergenceKind.SAMPLER_DIFFERENCE
            elif _token_near_tie(
                left.token_boundaries[index],
                right.token_boundaries[index],
                left_token,
                right_token,
                float(near_tie_abs_logit),
            ):
                root = DivergenceKind.NUMERICS_NEAR_TIE_FLIP
            else:
                root = DivergenceKind.UNCLASSIFIED
        divergences.append(
            OracleDivergence(
                field="output_token_id",
                location=f"output[{index}]",
                classification=root,
                cascade=index != first_token_difference,
                detail=f"left={left_token}, right={right_token}",
            )
        )

    output_length_exact = len(left.output_token_ids) == len(right.output_token_ids)
    if not output_length_exact:
        divergences.append(
            OracleDivergence(
                field="output_length",
                location="request",
                classification=root or DivergenceKind.UNCLASSIFIED,
                cascade=root is not None,
                detail=(f"left={len(left.output_token_ids)}, right={len(right.output_token_ids)}"),
            )
        )

    stop_reason_exact = left.stop_reason == right.stop_reason
    if not stop_reason_exact:
        divergences.append(
            OracleDivergence(
                field="stop_reason",
                location="request",
                classification=root or DivergenceKind.UNCLASSIFIED,
                cascade=root is not None,
                detail=f"left={left.stop_reason}, right={right.stop_reason}",
            )
        )

    left_routes = {decision.key: decision for decision in left.routed_decisions}
    right_routes = {decision.key: decision for decision in right.routed_decisions}
    for key in sorted(set(left_routes) | set(right_routes)):
        left_route = left_routes.get(key)
        right_route = right_routes.get(key)
        phase, token_index, layer_index = key
        location = f"{phase}[{token_index}].layer[{layer_index}]"
        context_diverged = (
            phase == "decode"
            and first_token_difference is not None
            and token_index >= first_token_difference
        )
        if left_route is None or right_route is None:
            classification = (
                root if context_diverged and root is not None else DivergenceKind.UNCLASSIFIED
            )
            divergences.append(
                OracleDivergence(
                    field="routing_presence",
                    location=location,
                    classification=classification,
                    cascade=context_diverged and root is not None,
                    detail="routing decision exists in only one observation",
                )
            )
            continue
        token_exact = left_route.token_id == right_route.token_id
        experts_exact = left_route.boundary.selected_ids == right_route.boundary.selected_ids
        if token_exact and experts_exact:
            continue
        if context_diverged and root is not None:
            classification = root
            cascade = True
        elif token_exact and _routing_near_tie(
            left_route.boundary,
            right_route.boundary,
            float(near_tie_abs_logit),
        ):
            classification = DivergenceKind.NUMERICS_NEAR_TIE_FLIP
            cascade = False
        else:
            classification = DivergenceKind.UNCLASSIFIED
            cascade = False
        divergences.append(
            OracleDivergence(
                field="routing",
                location=location,
                classification=classification,
                cascade=cascade,
                detail=(
                    f"left_token={left_route.token_id}, right_token={right_route.token_id}, "
                    f"left_experts={left_route.boundary.selected_ids}, "
                    f"right_experts={right_route.boundary.selected_ids}"
                ),
            )
        )

    return OracleComparison(
        request_id=left.request_id,
        sampling_mode=sampling_mode,
        divergences=tuple(divergences),
        input_tokens_exact=input_tokens_exact,
        output_tokens_exact=left.output_token_ids == right.output_token_ids,
        output_length_exact=output_length_exact,
        stop_reason_exact=stop_reason_exact,
        routing_exact=all(
            divergence.field not in ("routing", "routing_presence") for divergence in divergences
        ),
    )
