"""Independent symbolic sign check for the CORE-63 residency correction."""

from __future__ import annotations

from fractions import Fraction


def residency_sign(
    *,
    retained_service_ps: int,
    routed_service_ps: int,
    fixed_service_ps: int,
    routed_scale: Fraction,
) -> dict[str, str]:
    """Prove the correction's sign without using an anchor or retained value."""

    if min(retained_service_ps, fixed_service_ps) < 0:
        raise ValueError("retained and fixed service must be nonnegative")
    if routed_service_ps <= 0:
        raise ValueError("routed service must be positive")
    if not 0 < routed_scale < 1:
        raise ValueError("routed scale must be strictly between zero and one")
    before = Fraction(retained_service_ps + routed_service_ps + fixed_service_ps)
    after = Fraction(retained_service_ps + fixed_service_ps) + (
        routed_scale * routed_service_ps
    )
    if not after < before:
        raise AssertionError("residency correction did not reduce service")
    return {
        "corrected_step": "decrease",
        "predicted_throughput": "increase",
        "signed_residual": "less_negative_before_any_possible_crossing",
    }


__all__ = ["residency_sign"]
