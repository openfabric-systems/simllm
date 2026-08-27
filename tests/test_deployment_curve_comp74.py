from __future__ import annotations

import importlib.util
import sys
from fractions import Fraction
from pathlib import Path, PureWindowsPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/deployment_curve_v1"


def _module(filename: str, name: str):
    sys.path.insert(0, str(STUDY))
    try:
        spec = importlib.util.spec_from_file_location(name, STUDY / filename)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _freeze() -> dict:
    import json

    return json.loads((STUDY / "comp74_expectations.json").read_text(encoding="utf-8"))


def _successor(rows: list[dict]) -> dict:
    return {
        "acceptance_status": "candidate",
        "lookup_record_sha256": "d" * 64,
        "predecessor_lookup_record_sha256": "f" * 64,
        "score": {
            "lookup_service_ledger": {
                "deepseek_v3": {"DECLARED": 4, "MEASURED": 4},
                "granite": {"MEASURED": 12},
            },
            "priced_repeat_observations": rows,
            "task_movement": {
                "comp74_repeat_inputs": "RETAINED_FOR_ALL_FOUR_PRICED_KEYS"
            },
        },
    }


def _repeat_rows(frozen: dict, *, delta: int = 4) -> list[dict]:
    return [
        {
            "implementation_suffix": mapping["implementation_suffix"],
            "published_point_ps": 100,
            "independent_repeat_ps": 100 + delta,
            "signed_repeat_minus_point_ps": delta,
            "retained_independent_observations": 2,
            "distribution_propagation": "DEFERRED_TO_COMP-74",
        }
        for mapping in frozen["key_mapping"]
    ]


def test_comp74_estimates_each_key_without_pooling_and_contains_repeat() -> None:
    module = _module("comp74_distribution.py", "comp74_distribution_estimate_test")
    frozen = _freeze()

    rows = module.estimate_key_intervals(frozen, _successor(_repeat_rows(frozen)))

    assert len(rows) == 4
    assert {row["anchor_id"] for row in rows} == {
        "sglang_prefill_1k",
        "sglang_prefill_2k",
        "sglang_prefill_4k",
        "sglang_decode_standard",
    }
    assert all(Fraction(**row["relative_half_width"]) == Fraction(1, 25) for row in rows)
    assert all(row["service_interval_ps"] == {"lower": 96, "point": 100, "upper": 104} for row in rows)
    assert all(row["nonzero"] is True for row in rows)


def test_comp74_refuses_a_missing_or_duplicate_priced_key() -> None:
    module = _module("comp74_distribution.py", "comp74_distribution_key_test")
    frozen = _freeze()
    rows = _repeat_rows(frozen)

    with pytest.raises(ValueError, match="key set differs"):
        module.estimate_key_intervals(frozen, _successor(rows[:-1]))
    with pytest.raises(ValueError, match="key set differs"):
        module.estimate_key_intervals(frozen, _successor([*rows, rows[0]]))


def test_distribution_off_is_exact_and_on_adds_minkowski_spread() -> None:
    module = _module("comp74_distribution.py", "comp74_distribution_toggle_test")
    interval = {
        "lower": {"numerator": 90, "denominator": 1},
        "point": {"numerator": 100, "denominator": 1},
        "upper": {"numerator": 120, "denominator": 1},
        "contributions": [
            {
                "source_kind": "distribution",
                "source_id": module.ZERO_WIDTH_SOURCE,
                "relative_half_width": 0,
            }
        ],
    }
    estimate = {
        "anchor_id": "anchor",
        "implementation_suffix": "key",
        "relative_half_width": {"numerator": 1, "denominator": 25},
        "observation_count": 2,
        "service_interval_ps": {"lower": 96, "point": 100, "upper": 104},
    }

    off = module.propagate_prediction_interval(interval, estimate, enabled=False)
    on = module.propagate_prediction_interval(interval, estimate, enabled=True)

    assert off == interval
    assert Fraction(**on["lower"]) == 86
    assert Fraction(**on["point"]) == 100
    assert Fraction(**on["upper"]) == 124
    assert len(on["contributions"]) == 1
    assert on["contributions"][0]["source_id"] == "comp74-repeat-envelope:key"
    assert on["contributions"][0]["stability_claim"] is False


def test_distribution_refuses_a_nonphysical_lower_bound() -> None:
    module = _module("comp74_distribution.py", "comp74_distribution_zero_test")
    estimate = {
        "anchor_id": "anchor",
        "implementation_suffix": "key",
        "relative_half_width": {"numerator": 1, "denominator": 1},
        "observation_count": 2,
        "service_interval_ps": {"lower": 1, "point": 2, "upper": 3},
    }

    with pytest.raises(ValueError, match="zero"):
        module.propagate_prediction_interval(
            {"lower": 90, "point": 100, "upper": 110},
            estimate,
            enabled=True,
        )


def test_runner_uses_posix_paths_without_unix_only_imports() -> None:
    runner = _module("run_comp74_distribution.py", "comp74_distribution_runner_test")

    assert runner.render_cli_path(PureWindowsPath("C:/runs/result.json")) == (
        "C:/runs/result.json"
    )
