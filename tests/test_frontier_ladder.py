from __future__ import annotations

import hashlib
from copy import deepcopy
from fractions import Fraction

import pytest

from examples.frontier_ladder_v1.plot_study import PLOT_SCHEMA, prepare_plot_data
from simllm.deploy import (
    FRONTIER_LADDER_RECORD_SCHEMA,
    ExternalAnchor,
    FrontierLadderPoint,
    FrontierLadderRecord,
    FrontierRung,
    FrontierRungPoint,
    PointClass,
    RungAuthorityClass,
    RungProvenance,
    frontier_ladder_record_from_json,
    frontier_ladder_record_to_json,
    ladder_pareto_front,
)

PICOSECONDS_PER_SECOND = 1_000_000_000_000


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _rung(
    rung: FrontierRung,
    point_class: PointClass,
    step_ps: int,
    batch: int,
    fabric_leg_ps: int = 10,
) -> FrontierRungPoint:
    if rung is FrontierRung.ESTIMATE:
        provenance = RungProvenance(
            authority_class=RungAuthorityClass.ESTIMATOR,
            authority="closed-form",
            source_path="examples/source.json",
            source_sha256=_digest("estimate"),
        )
    elif rung is FrontierRung.LOGGOPSIM_IDEAL:
        if point_class is PointClass.ESTIMATE:
            provenance = RungProvenance(
                authority_class=RungAuthorityClass.ESTIMATOR,
                authority="closed-form",
                source_path="examples/source.json",
                source_sha256=_digest("estimate"),
            )
        else:
            provenance = RungProvenance(
                authority_class=RungAuthorityClass.LEVEL,
                authority="loggopsim-ideal",
                source_path="goals/cell.goal",
                source_sha256=_digest("goal-text"),
                binary_sha256=_digest("binary"),
                goal_sha256=_digest("goal-binary"),
                argv=(
                    "LogGOPSim",
                    "-f",
                    "goals/cell.bin",
                    "-L",
                    "2000",
                    "-o",
                    "0",
                    "-g",
                    "0",
                    "-G",
                    "0.02",
                    "-O",
                    "0",
                    "-S",
                    "1001",
                    "-n",
                    "LogGP",
                ),
            )
    else:
        provenance = RungProvenance(
            authority_class=RungAuthorityClass.LEVEL,
            authority="rnic-nn",
            source_path="examples/source.json",
            source_sha256=_digest("packet"),
            binary_sha256=_digest("packet-binary"),
        )
    x_value = Fraction(PICOSECONDS_PER_SECOND, step_ps)
    return FrontierRungPoint(
        rung=rung,
        point_class=point_class,
        step_ps=step_ps,
        fabric_leg_ps=fabric_leg_ps,
        x_tokens_per_second_per_request=x_value,
        y_tokens_per_second_per_gpu=batch * x_value,
        provenance=provenance,
    )


def _point(
    configuration_id: str = "first",
    batch: int = 1,
    steps: tuple[int, int, int] = (100, 101, 102),
) -> FrontierLadderPoint:
    return FrontierLadderPoint(
        configuration_id=configuration_id,
        configuration_label=configuration_id.title(),
        batch_per_gpu=batch,
        rungs=(
            _rung(FrontierRung.ESTIMATE, PointClass.ESTIMATE, steps[0], batch),
            _rung(
                FrontierRung.LOGGOPSIM_IDEAL,
                PointClass.SIMULATED,
                steps[1],
                batch,
            ),
            _rung(FrontierRung.PACKET, PointClass.SIMULATED, steps[2], batch),
        ),
    )


def _record() -> FrontierLadderRecord:
    return FrontierLadderRecord(
        points=(_point(),),
        anchors=(
            ExternalAnchor(
                anchor_id="paired",
                label="Paired",
                x_tokens_per_second_per_request=Fraction(10),
                y_tokens_per_second_per_gpu=Fraction(20),
            ),
            ExternalAnchor(
                anchor_id="y-only",
                label="Y only",
                y_tokens_per_second_per_gpu=Fraction(15),
            ),
        ),
    )


def test_ladder_record_strict_round_trip_keeps_three_authorities() -> None:
    record = _record()

    rendered = frontier_ladder_record_to_json(record)

    assert rendered["schema"] == FRONTIER_LADDER_RECORD_SCHEMA
    assert frontier_ladder_record_from_json(rendered) == record
    rungs = rendered["points"][0]["rungs"]
    assert [rung["point_class"] for rung in rungs] == [
        "ESTIMATE",
        "SIMULATED",
        "SIMULATED",
    ]
    assert [rung["provenance"]["authority"] for rung in rungs] == [
        "closed-form",
        "loggopsim-ideal",
        "rnic-nn",
    ]


@pytest.mark.parametrize(
    "path",
    [(), ("points", 0), ("points", 0, "rungs", 0), ("anchors", 0)],
)
def test_ladder_record_rejects_unknown_fields(path: tuple[str | int, ...]) -> None:
    payload = deepcopy(frontier_ladder_record_to_json(_record()))
    target: object = payload
    for component in path:
        target = target[component]  # type: ignore[index]
    target["unexpected"] = 1  # type: ignore[index]

    with pytest.raises(ValueError, match="unknown fields"):
        frontier_ladder_record_from_json(payload)


def test_ladder_record_checks_schema_before_unknown_fields() -> None:
    payload = frontier_ladder_record_to_json(_record())
    payload["schema"] = "simllm-deployment-frontier-ladder-record-v0"
    payload["unexpected"] = 1

    with pytest.raises(ValueError, match="unsupported schema") as error:
        frontier_ladder_record_from_json(payload)

    assert "unknown fields" not in str(error.value)


def test_ladder_record_rejects_a_packet_point_relabelled_as_estimate() -> None:
    payload = frontier_ladder_record_to_json(_record())
    payload["points"][0]["rungs"][2]["point_class"] = "ESTIMATE"

    with pytest.raises(ValueError, match="point class does not match authority"):
        frontier_ladder_record_from_json(payload)


def test_ladder_record_requires_ideal_execution_for_a_nonzero_fabric_leg() -> None:
    payload = frontier_ladder_record_to_json(_record())
    ideal = payload["points"][0]["rungs"][1]
    ideal["provenance"]["argv"] = []
    ideal["provenance"]["goal_sha256"] = None

    with pytest.raises(ValueError, match="execution provenance is incomplete"):
        frontier_ladder_record_from_json(payload)


def test_zero_fabric_ideal_rung_is_a_nonexecuted_estimate() -> None:
    point = FrontierLadderPoint(
        configuration_id="local",
        configuration_label="Local",
        batch_per_gpu=1,
        rungs=(
            _rung(FrontierRung.ESTIMATE, PointClass.ESTIMATE, 100, 1),
            _rung(
                FrontierRung.LOGGOPSIM_IDEAL,
                PointClass.ESTIMATE,
                100,
                1,
                fabric_leg_ps=0,
            ),
            _rung(FrontierRung.PACKET, PointClass.SIMULATED, 100, 1),
        ),
    )

    rendered = frontier_ladder_record_to_json(FrontierLadderRecord(points=(point,)))
    ideal = rendered["points"][0]["rungs"][1]

    assert ideal["point_class"] == "ESTIMATE"
    assert ideal["provenance"]["authority_class"] == "estimator"
    assert ideal["provenance"]["authority"] == "closed-form"
    assert ideal["provenance"]["binary_sha256"] is None
    assert ideal["provenance"]["goal_sha256"] is None
    assert ideal["provenance"]["argv"] == []


def test_published_v1_nonexecuted_ideal_level_is_corrected_on_read() -> None:
    point = FrontierLadderPoint(
        configuration_id="local",
        configuration_label="Local",
        batch_per_gpu=1,
        rungs=(
            _rung(FrontierRung.ESTIMATE, PointClass.ESTIMATE, 100, 1),
            _rung(
                FrontierRung.LOGGOPSIM_IDEAL,
                PointClass.ESTIMATE,
                100,
                1,
                fabric_leg_ps=0,
            ),
            _rung(FrontierRung.PACKET, PointClass.SIMULATED, 100, 1),
        ),
    )
    payload = frontier_ladder_record_to_json(FrontierLadderRecord(points=(point,)))
    legacy_ideal = payload["points"][0]["rungs"][1]
    legacy_ideal["point_class"] = "SIMULATED"
    legacy_ideal["provenance"]["authority_class"] = "level"
    legacy_ideal["provenance"]["authority"] = "loggopsim-ideal"
    legacy_ideal["provenance"]["binary_sha256"] = _digest("binary")

    corrected = frontier_ladder_record_to_json(
        frontier_ladder_record_from_json(payload)
    )
    ideal = corrected["points"][0]["rungs"][1]

    assert ideal["point_class"] == "ESTIMATE"
    assert ideal["provenance"]["authority_class"] == "estimator"
    assert ideal["provenance"]["authority"] == "closed-form"
    assert ideal["provenance"]["binary_sha256"] is None


def test_packet_pareto_uses_only_the_selected_rung() -> None:
    dominant = _point("dominant", 1, (1_000, 1_000, 1_000))
    dominated = _point("dominated", 2, (900, 900, 2_000))
    record = FrontierLadderRecord(points=(dominant, dominated))

    assert ladder_pareto_front(record, FrontierRung.PACKET) == (dominant,)


def test_plot_projection_keeps_rung_styles_and_y_only_anchor_unpaired() -> None:
    result = {
        "schema": "simllm-frontier-ladder-study-v1",
        "ladder_record": frontier_ladder_record_to_json(_record()),
        "fabric_leg_envelope": [],
    }

    plot = prepare_plot_data(result)

    assert plot["schema"] == PLOT_SCHEMA
    assert set(plot["rung_styles"]) == {rung.value for rung in FrontierRung}
    assert plot["y_only_anchor"] == {"label": "Y only", "y": 15.0}
    assert "x" not in plot["y_only_anchor"]
