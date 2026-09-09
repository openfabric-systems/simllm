"""Publication uncertainty preserves exact threshold feasibility and provenance."""

from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
from pathlib import Path

import pytest

from simllm.deploy import (
    FrontierPublicationComparison,
    FrontierSelection,
    FrontierSelectionSegment,
    PublicationBasis,
    PublishedThreshold,
    ThresholdInterval,
    compare_published_frontier,
    frontier_at_threshold,
)

ROOT = Path(__file__).resolve().parents[1]
FREEZE = json.loads(
    (ROOT / "examples/frontier_publication_precision_v1/expectations.json").read_text()
)
POINTS = (("a", F(2), F(100)), ("b", F(3), F(40)))


def source(interval, *, basis=PublicationBasis.DECLARED):
    return PublishedThreshold(
        "2.000", "a" * 64, "source-row-1", interval, basis, "declared test interval"
    )


def compare(points=POINTS, interval=None, **kwargs):
    return compare_published_frontier(
        points,
        source(interval or ThresholdInterval(F(3, 2), F(5, 2))),
        F(100),
        coordinate=lambda p: (p[1], p[2]),
        identity=lambda p: p[0],
        **kwargs,
    )


def exact(points, value):
    return frontier_at_threshold(
        points, value, coordinate=lambda p: (p[1], p[2]), identity=lambda p: p[0]
    )


@pytest.mark.parametrize("case", FREEZE["boundary_oracles"], ids=lambda row: row["id"])
def test_frozen_boundary_choices(case):
    points = tuple((name, F(x), F(y)) for name, x, y in case["points"])
    interval = ThresholdInterval(
        F(case["lower"]), F(case["upper"]), case["lower_closed"], case["upper_closed"]
    )
    result = compare(points, interval)
    assert [s.selection.point_id if s.selection else None for s in result.segments] == (
        case["expected_choice_sequence"]
    )
    assert result.verdict == case["expected_verdict"]
    assert result.threshold.interval == interval


def test_exact_endpoint_is_a_distinct_singleton():
    result = compare(interval=ThresholdInterval(F(2), F(5, 2)))
    assert result.segments == (
        FrontierSelectionSegment(
            ThresholdInterval(F(2), F(2)), FrontierSelection("a", F(2), F(100))
        ),
        FrontierSelectionSegment(
            ThresholdInterval(F(2), F(5, 2), False, True), FrontierSelection("b", F(3), F(40))
        ),
    )
    assert result.quotients == (F(2, 5), F(1))
    assert result.quotient_bounds == (F(2, 5), F(1))
    assert not result.possibly_infeasible
    assert result.verdict == "INDETERMINATE"


def test_disconnected_failed_values_do_not_fill_their_hull():
    result = compare((("a", F(2), F(200)), ("b", F(3), F(40))))
    assert result.quotients == (F(2, 5), F(2))
    assert result.quotient_bounds[0] < 1 < result.quotient_bounds[1]
    assert result.verdict == "FAIL"


def test_infeasibility_is_preserved_alongside_zero_quotient():
    result = compare((("a", F(2), F(100)),), ThresholdInterval(F(2), F(3)))
    assert result.possibly_infeasible
    assert result.segments[-1].selection is None
    assert result.quotients == (F(0), F(1))
    assert result.verdict == "INDETERMINATE"


@pytest.mark.parametrize("point", (F(1), F(2), F(5, 2), F(3), F(4)))
def test_singleton_preserves_exact_lookup(point):
    result = compare(interval=ThresholdInterval(point, point))
    assert len(result.segments) == 1
    assert result.segments[0].selection == exact(POINTS, point)


def test_source_qualification_is_not_an_agreement_verdict():
    threshold = source(ThresholdInterval(F(1), F(1)), basis=PublicationBasis.CONDITIONAL)
    result = compare_published_frontier(
        POINTS, threshold, F(100), coordinate=lambda p: (p[1], p[2]), identity=lambda p: p[0]
    )
    assert result.verdict == "PASS"
    assert result.threshold.basis is PublicationBasis.CONDITIONAL
    assert result.threshold.axis == "x_tokens_per_second_per_request"
    assert result.threshold.units == "tokens/s/request"
    assert result.threshold.published_text == "2.000"


def test_callbacks_are_snapshotted_once_and_input_changes_do_not_rewrite_result():
    points = [["a", F(2), F(100)], ["b", F(3), F(40)]]
    calls = {"coordinate": 0, "identity": 0}

    def coordinate(row):
        calls["coordinate"] += 1
        return row[1], row[2]

    def identity(row):
        calls["identity"] += 1
        return row[0]

    before = copy.deepcopy(points)
    result = compare_published_frontier(
        iter(points),
        source(ThresholdInterval(F(1), F(3))),
        F(100),
        coordinate=coordinate,
        identity=identity,
    )
    assert calls == {"coordinate": 2, "identity": 2}
    assert points == before
    points[0][2] = F(500)
    assert result.segments[0].selection.y == 100
    with pytest.raises(FrozenInstanceError):
        result.segments[0].selection.y = F(500)


def test_input_permutation_does_not_change_any_projected_field():
    assert compare(POINTS) == compare(tuple(reversed(POINTS)))


@pytest.mark.parametrize(
    "lower,upper,lc,uc",
    [
        (F(2), F(1), True, True),
        (F(2), F(2), False, True),
        (F(2), F(2), True, False),
        (F(0), F(2), True, True),
        (1.0, F(2), True, True),
        (F(1), F(2), 1, True),
    ],
)
def test_malformed_interval_rejects(lower, upper, lc, uc):
    with pytest.raises((TypeError, ValueError)):
        ThresholdInterval(lower, upper, lc, uc)


@pytest.mark.parametrize(
    "field,value",
    [
        ("published_text", "nan"),
        ("published_text", "1e3"),
        ("published_text", "0"),
        ("source_sha256", "A" * 64),
        ("row_id", ""),
        ("assumptions", ""),
        ("basis", "CONDITIONAL"),
        ("interval", (F(1), F(2))),
        ("basis", PublicationBasis.SOURCE_EXACT),
    ],
)
def test_malformed_source_declaration_rejects(field, value):
    with pytest.raises((TypeError, ValueError)):
        replace(source(ThresholdInterval(F(1), F(2))), **{field: value})


@pytest.mark.parametrize(
    "points",
    [
        (("a", F(2), F(100)), ("a", F(3), F(40))),
        (("a", 2.0, F(100)),),
        (("a", F(2), 100.0),),
        (("a", F(2), F(0)),),
        (("", F(2), F(100)),),
    ],
)
def test_invalid_frontier_rejects_before_selection(points):
    with pytest.raises((TypeError, ValueError)):
        compare(points)
    with pytest.raises((TypeError, ValueError)):
        exact(points, F(9))


@pytest.mark.parametrize("band", [(F(2), F(1)), [F(1), F(2)], (F(0), F(1)), (0.75, F(2))])
def test_invalid_quotient_band_rejects(band):
    with pytest.raises((TypeError, ValueError)):
        compare(quotient_band=band)


def test_result_rejects_gap_overlap_and_unmerged_segments():
    threshold = source(ThresholdInterval(F(1), F(3)))
    choice = FrontierSelection("a", F(4), F(100))
    first = FrontierSelectionSegment(ThresholdInterval(F(1), F(2)), choice)
    for second in [
        FrontierSelectionSegment(ThresholdInterval(F(5, 2), F(3)), None),
        FrontierSelectionSegment(ThresholdInterval(F(2), F(3)), None),
        FrontierSelectionSegment(ThresholdInterval(F(2), F(3), False, True), choice),
    ]:
        with pytest.raises(ValueError):
            FrontierPublicationComparison._issue(
                threshold, F(100), (F(3, 4), F(27, 20)), (first, second)
            )


def test_result_rejects_infeasible_selection_and_empty_coverage():
    with pytest.raises(ValueError):
        FrontierSelectionSegment(
            ThresholdInterval(F(1), F(3)), FrontierSelection("a", F(2), F(100))
        )
    with pytest.raises(ValueError):
        FrontierPublicationComparison._issue(
            source(ThresholdInterval(F(1), F(2))), F(100), (F(3, 4), F(27, 20)), ()
        )


def test_historical_point_and_stamp_objects_are_unchanged():
    record = json.loads((ROOT / "examples/matched_seam_frontier_v1/record.json").read_text())
    points = record["families"]["F"]["ideal_frontier"]
    before = copy.deepcopy(points)

    def coordinate(row):
        return tuple(
            F(row[key]["numerator"], row[key]["denominator"])
            for key in ("x_tokens_per_second_per_user", "y_tokens_per_second_per_gpu")
        )

    result = compare_published_frontier(
        points,
        source(ThresholdInterval(F(168), F(169))),
        F(100),
        coordinate=coordinate,
        identity=lambda row: row["candidate_key"],
    )
    assert points == before
    keys = {row["candidate_key"] for row in points}
    assert all(row.selection is None or row.selection.point_id in keys for row in result.segments)


@pytest.mark.parametrize(
    "forgery", ["suboptimal", "false-none", "false-coordinate", "source", "band"]
)
def test_coherent_public_result_replacement_rejects(forgery):
    result = compare()
    whole = result.threshold.interval
    altered = {
        "suboptimal": {
            "segments": (FrontierSelectionSegment(whole, FrontierSelection("b", F(3), F(40))),)
        },
        "false-none": {"segments": (FrontierSelectionSegment(whole, None),)},
        "false-coordinate": {
            "segments": (FrontierSelectionSegment(whole, FrontierSelection("a", F(3), F(100))),)
        },
        "source": {"threshold": replace(result.threshold, source_sha256="b" * 64)},
        "band": {"quotient_band": (F(1, 10), F(10))},
    }[forgery]
    with pytest.raises(TypeError, match="issued by compare_published_frontier"):
        replace(result, **altered)
    with pytest.raises(TypeError, match="issued by compare_published_frontier"):
        FrontierPublicationComparison()


def test_study_fatal_completeness_voids_the_behavioral_score():
    from examples.frontier_publication_precision_v1.run_study import Evaluation

    evaluation = Evaluation()
    result = evaluation.finish()
    assert result["state"] == "VOID"
    assert result["behavioral_score"] is None
    assert "complete:synthetic_cells" in result["fatal_findings"]
    assert "complete:boundary" in result["fatal_findings"]


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"schema": "wrong"},
        {"schema": "simllm-frontier-publication-precision-evaluation-v1", "fatal_guards": []},
    ],
)
def test_malformed_study_worker_cannot_be_qualified(value):
    from examples.frontier_publication_precision_v1.run_study import valid_worker

    assert valid_worker(value) is False


def test_void_worker_requires_explicit_null_score_and_typed_rows():
    from examples.frontier_publication_precision_v1.run_study import SCHEMA, valid_worker

    value = {
        "schema": SCHEMA,
        "fatal_guards": [{"id": "bad", "passed": False}],
        "exact_oracles": [{"id": "x", "passed": True}],
        "behavioral_relations": [{"id": "y", "passed": True}],
        "state": "VOID",
        "fatal_findings": ["bad"],
        "behavioral_score": None,
        "historical_agreement_is_scored_for_study": False,
    }
    assert valid_worker(value)
    missing = dict(value)
    del missing["behavioral_score"]
    assert not valid_worker(missing)
    assert not valid_worker({**value, "behavioral_score": {"passed": 1, "instances": 1}})
    assert not valid_worker({**value, "fatal_guards": [{"id": "bad", "passed": 0}]})
    assert not valid_worker({**value, "state": "PASS"})


@pytest.mark.parametrize("key", ["boundary_cells", "historical_agreement"])
def test_study_retained_identity_replacement_voids_even_with_correct_count(key):
    from examples.frontier_publication_precision_v1.run_study import Evaluation

    evaluation = Evaluation()
    ids = (
        [r["id"] for r in FREEZE["boundary_oracles"]]
        if key == "boundary_cells"
        else [f"historical-row-{i:02d}" for i in FREEZE["historical"]["rows"]]
    )
    evaluation.data[key] = [{"id": name} for name in ids[:-1]] + [{"id": "unrelated-unique-id"}]
    result = evaluation.finish()
    assert f"complete:{key}" in result["fatal_findings"]
    assert result["behavioral_score"] is None


def test_publication_extension_preserves_the_frozen_exact_frontier_source():
    from examples.decode_hbm_crossover_v1.run_study import frozen_findings

    frozen = json.loads((ROOT / "examples/decode_hbm_crossover_v1/expectations.json").read_bytes())
    assert frozen_findings(frozen) == []
