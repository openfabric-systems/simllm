"""Independent direct signer for the COMP-75 visible 1K movement."""

from __future__ import annotations

from fractions import Fraction
from typing import Any


def _fraction_json(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _movement(updated: Fraction, reference: Fraction) -> dict[str, Any]:
    delta = updated - reference
    direction = "decrease" if delta < 0 else "increase" if delta > 0 else "unchanged"
    return {
        "direction": direction,
        "absolute_tokens_per_second_per_node": _fraction_json(delta),
        "relative_to_reference": _fraction_json(delta / reference),
    }


def sign_visible_movement(
    *,
    per_node_tokens: int,
    candidate_service_ps: int,
    core59_total_service_ps: int,
    comp75_total_service_ps: int,
    published_tokens_per_second_per_node: int,
) -> dict[str, Any]:
    """Reconstruct every sign directly from raw integer inputs."""

    values = (
        per_node_tokens,
        candidate_service_ps,
        core59_total_service_ps,
        comp75_total_service_ps,
        published_tokens_per_second_per_node,
    )
    if any(isinstance(value, bool) or type(value) is not int or value <= 0 for value in values):
        raise ValueError("independent signer inputs must be positive integers")
    ps_per_second = 1_000_000_000_000
    candidate = Fraction(per_node_tokens * ps_per_second, candidate_service_ps)
    core59 = Fraction(per_node_tokens * ps_per_second, core59_total_service_ps)
    updated = Fraction(per_node_tokens * ps_per_second, comp75_total_service_ps)
    published = Fraction(published_tokens_per_second_per_node)
    return {
        "method": "direct Fraction reconstruction from raw tokens and integer services",
        "candidate_only": _fraction_json(candidate),
        "core59": _fraction_json(core59),
        "comp75": _fraction_json(updated),
        "movement_from_candidate_only": _movement(updated, candidate),
        "movement_from_core59": _movement(updated, core59),
        "absolute_remaining_error_tokens_per_second_per_node": _fraction_json(
            updated - published
        ),
        "signed_relative_remaining_error": _fraction_json(
            (updated - published) / published
        ),
    }


__all__ = ["sign_visible_movement"]
