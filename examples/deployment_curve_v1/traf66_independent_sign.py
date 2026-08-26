"""Independent sign reconstruction for the TRAF-66 visible movement."""

from __future__ import annotations

from fractions import Fraction

PS_PER_SECOND = 1_000_000_000_000


def sign_visible_movement(
    *,
    per_node_tokens: int,
    published_numerator: int,
    published_denominator: int,
    compute_service_ps: int,
    packet_service_ps: int,
    children: int,
) -> dict[str, object]:
    """Rebuild the movement directly, without importing the boundary module."""

    values = (
        per_node_tokens,
        published_numerator,
        published_denominator,
        compute_service_ps,
        packet_service_ps,
        children,
    )
    if any(isinstance(value, bool) or type(value) is not int for value in values):
        raise TypeError("independent sign inputs must be integers")
    if any(value <= 0 for value in values):
        raise ValueError("independent sign inputs must be positive")
    if children != 2:
        raise ValueError("TRAF-66 is exactly a two-child schedule")

    target = Fraction(published_numerator, published_denominator)
    previous_service = Fraction(max(compute_service_ps, packet_service_ps))
    boundary_service = Fraction(min(compute_service_ps, packet_service_ps), children)
    updated_service = previous_service + boundary_service
    previous = Fraction(per_node_tokens * PS_PER_SECOND, previous_service)
    updated = Fraction(per_node_tokens * PS_PER_SECOND, updated_service)
    previous_error = previous / target - 1
    updated_error = updated / target - 1
    movement = updated - previous
    return {
        "method": "direct Fraction reconstruction from child count and raw services",
        "children": children,
        "boundary_service_ps": {
            "numerator": boundary_service.numerator,
            "denominator": boundary_service.denominator,
        },
        "comp75": {"numerator": previous.numerator, "denominator": previous.denominator},
        "traf66": {"numerator": updated.numerator, "denominator": updated.denominator},
        "movement": {
            "numerator": movement.numerator,
            "denominator": movement.denominator,
            "direction": "decrease" if movement < 0 else "increase" if movement > 0 else "unchanged",
        },
        "signed_relative_error_before": {
            "numerator": previous_error.numerator,
            "denominator": previous_error.denominator,
        },
        "signed_relative_error_after": {
            "numerator": updated_error.numerator,
            "denominator": updated_error.denominator,
        },
        "signed_residual_movement": {
            "numerator": (updated_error - previous_error).numerator,
            "denominator": (updated_error - previous_error).denominator,
            "direction": "more_negative",
        },
    }
