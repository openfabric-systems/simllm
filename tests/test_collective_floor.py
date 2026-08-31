import hashlib
import json
import random
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

import simllm.backends.step_sink as step_sink_module
from simllm.backends import (
    HtsimRequestMetricReducer,
    HtsimStepSink,
    HtsimStepSinkConfig,
    attribute_step_detail,
)
from simllm.calibration.external_nccl import ExternalNcclDatabase
from simllm.compute import (
    ComputeProvider,
    DurationEstimate,
    HostInitiationModel,
    ModelDims,
)
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.placement import PlacementManifest, RankPlacement
from simllm.traffic import (
    B200_NCCL_2_27_LOCAL_PROFILE,
    COLLECTIVE_FLOOR_CALIBRATED,
    COLLECTIVE_FLOOR_TRANSFERRED,
    CollectiveFloorCell,
    CollectiveFloorCurveBoundaries,
    CollectiveFloorSourceIdentity,
    choose_collective_floor_boundaries,
    distribute_collective_serialization_ps,
    fit_collective_floor_calibration,
    source_elements_for_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (
        REPO_ROOT
        / "examples"
        / "collective_floor_calibration_v1"
        / "study_config.json"
    ).read_text(encoding="utf-8")
)

SEAM_DIMS = ModelDims(
    num_layers=1,
    hidden_size=1_024,
    intermediate_size=2_048,
    num_heads=8,
    num_kv_heads=8,
    head_size=128,
    vocab_size=256,
    dtype_bytes=2,
)


class _FixedProvider(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=2_000, bound="measured")


def _manifest(hosts: tuple[str, ...]) -> PlacementManifest:
    return PlacementManifest(
        ranks=[
            RankPlacement(global_rank=rank, hostname=host, local_rank=rank)
            for rank, host in enumerate(hosts)
        ]
    )


def _record(step_index: int = 0, virtual_time_ps: int = 0) -> StepRecord:
    return StepRecord(
        step_index=step_index,
        virtual_time_ps=virtual_time_ps,
        scheduled=[
            ScheduledRequest(
                "r0",
                RequestPhase.DECODE,
                num_new_tokens=1,
                context_length=8,
            )
        ],
        num_sampled=1,
    )


def _sink_config(workdir, hosts: tuple[str, ...], **kwargs) -> HtsimStepSinkConfig:
    return HtsimStepSinkConfig(
        profile="rnic-nn-fluid",
        tp_ranks=tuple(range(len(hosts))),
        dims=SEAM_DIMS,
        workdir=workdir,
        placement_manifest=_manifest(hosts),
        provider=_FixedProvider(),
        **kwargs,
    )


def _stub_backend(monkeypatch, calls: list[str]) -> None:
    monkeypatch.setattr(step_sink_module, "to_binary", lambda path: path)

    def run(config):
        calls.append(config.goal_bin.name)
        return SimpleNamespace(
            job_completion_time_ps=lambda: 1_000_000,
            flows=(),
            quiescent=True,
        )

    monkeypatch.setattr(step_sink_module, "run_htsim_rnic", run)


def _goal_manifest(workdir: Path) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (
            path.name,
            len(path.read_bytes()),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(workdir.rglob("*.goal"))
    )


def _json_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _source() -> CollectiveFloorSourceIdentity:
    source = CONFIG["source"]
    return CollectiveFloorSourceIdentity(
        artifact_sha256=source["artifact_sha256"],
        tool=source["tool"],
        aiconfigurator_version=source["aiconfigurator_version"],
        aiconfigurator_core_version=source["aiconfigurator_core_version"],
        system=source["system"],
        backend=source["backend"],
        database_version=source["database_version"],
        row_version=source["row_version"],
        duplicate_resolution=source["duplicate_resolution"],
    )


def _training_cells() -> tuple[CollectiveFloorCell, ...]:
    database = ExternalNcclDatabase.load()
    cells = []
    for member in CONFIG["membership"]["training_cells"]:
        latency = database.query(
            dtype=member["dtype"],
            operation=member["operation"],
            ranks=member["ranks"],
            message_size=member["source_elements"],
        )
        cells.append(
            CollectiveFloorCell(
                cell_id=member["cell_id"],
                dtype=member["dtype"],
                operation=member["operation"],
                ranks=member["ranks"],
                source_elements=member["source_elements"],
                message_bytes=member["true_bytes"],
                latency_ps=round(latency.latency_ms * 1_000_000_000),
            )
        )
    return tuple(cells)


def _boundaries() -> tuple[CollectiveFloorCurveBoundaries, ...]:
    return tuple(
        CollectiveFloorCurveBoundaries(
            dtype=row["dtype"],
            operation=row["operation"],
            ranks=row["ranks"],
            lower_bounds_of_following_regimes=tuple(
                row["lower_bounds_of_following_regimes"]
            ),
        )
        for row in CONFIG["fit"]["regime_boundaries_true_bytes"]
    )


@pytest.fixture(scope="module")
def calibration():
    byte_range = CONFIG["fit"]["true_byte_range"]
    return fit_collective_floor_calibration(
        calibration_id="h200-nccl-2.26.2-aggregate-floor-v1",
        source=_source(),
        cells=_training_cells(),
        boundaries=_boundaries(),
        fitted_byte_range=(byte_range["minimum"], byte_range["maximum"]),
    )


def test_equal_byte_queries_resolve_distinct_source_cells():
    guard = CONFIG["axis"]["equal_byte_guard"]
    database = ExternalNcclDatabase.load()
    observed = {}
    for dtype in ("half", "int8"):
        member = guard[f"dtype_{dtype}"]
        source_elements = source_elements_for_bytes(dtype, guard["true_bytes"])
        assert source_elements == member["source_elements"]
        observed[dtype] = database.query(
            dtype=dtype,
            operation=member["operation"],
            ranks=member["ranks"],
            message_size=source_elements,
        ).latency_ms
        assert format(observed[dtype], ".5f") == member["measured_latency_ms"]
    assert observed["half"] != observed["int8"]


def test_frozen_boundaries_are_reproduced_from_training_only():
    by_key = {}
    for cell in _training_cells():
        by_key.setdefault(cell.curve_key, []).append(cell)
    for frozen in _boundaries():
        assert choose_collective_floor_boundaries(by_key[frozen.curve_key]) == (
            frozen.lower_bounds_of_following_regimes
        )


def test_fit_carries_positive_byte_slopes_and_complete_identity(calibration):
    assert len(calibration.regimes) == 18
    assert len({cell.cell_id for cell in _training_cells()}) == 63
    assert {
        cell_id
        for regime in calibration.regimes
        for cell_id in regime.training_cell_ids
    } == {cell.cell_id for cell in _training_cells()}
    assert all(regime.floor_ps > 0 for regime in calibration.regimes)
    assert all(regime.slope_ps_per_byte > 0 for regime in calibration.regimes)
    assert all(
        regime.effective_bandwidth_bytes_per_second > 0
        for regime in calibration.regimes
    )
    assert calibration.source.as_dict() == CONFIG["source"]


def test_exact_domain_is_calibrated_and_every_transfer_is_downgraded(calibration):
    exact = calibration.estimate(
        dtype="half",
        operation="all_gather",
        ranks=8,
        message_bytes=344_064,
    )
    assert exact.evidence_class == COLLECTIVE_FLOOR_CALIBRATED
    assert exact.transfer_reason is None
    assert exact.completion_ps == exact.floor_charge_ps + exact.serialization_ps
    assert exact.regime.training_cell_ids
    assert exact.as_dict()["source"] == CONFIG["source"]

    changed_operation = calibration.estimate(
        dtype="half",
        operation="all_reduce",
        ranks=8,
        message_bytes=344_064,
        donor=("half", "all_gather", 8),
    )
    changed_dtype = calibration.estimate(
        dtype="int8",
        operation="all_gather",
        ranks=8,
        message_bytes=344_064,
        donor=("half", "all_gather", 8),
    )
    changed_rank = calibration.estimate(
        dtype="half",
        operation="all_gather",
        ranks=16,
        message_bytes=344_064,
        donor=("half", "all_gather", 8),
    )
    changed_range = calibration.estimate(
        dtype="half",
        operation="all_gather",
        ranks=8,
        message_bytes=CONFIG["fit"]["true_byte_range"]["maximum"] + 1,
    )
    for transferred in (
        changed_operation,
        changed_dtype,
        changed_rank,
        changed_range,
    ):
        assert transferred.evidence_class == COLLECTIVE_FLOOR_TRANSFERRED
        assert transferred.transfer_reason


def test_unfitted_curve_requires_an_explicit_donor(calibration):
    with pytest.raises(ValueError, match="donor curve"):
        calibration.estimate(
            dtype="int8",
            operation="all_gather",
            ranks=8,
            message_bytes=512,
        )


def test_calibration_input_surface_fences_packet_candidates(calibration):
    prohibited = {
        "packet_geometry",
        "credits",
        "link_count",
        "link_rate",
        "switch_buffer",
        "arbitration",
        "a100_candidate_profile",
    }
    assert prohibited.isdisjoint(calibration.input_surface)
    assert calibration.input_surface == (
        "external_nccl_training_cells",
        "element_to_byte_width",
        "training_only_regime_boundaries",
        "source_identity",
    )


@pytest.mark.parametrize(
    ("serialization_ps", "phase_count", "expected"),
    [
        (0, 3, (0, 0, 0)),
        (10, 3, (4, 3, 3)),
        (Fraction(5, 1).numerator, 2, (3, 2)),
    ],
)
def test_serialization_distribution_is_exact(
    serialization_ps,
    phase_count,
    expected,
):
    assert distribute_collective_serialization_ps(
        serialization_ps, phase_count
    ) == expected
    assert sum(expected) == serialization_ps


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        (
            {"collective_latency_profile": B200_NCCL_2_27_LOCAL_PROFILE},
            "complete measured fixed floor",
        ),
        (
            {"collective_registration": "nccl-channel-registration-v1"},
            "timed collective setup",
        ),
        (
            {"host_model": HostInitiationModel(initiation_delay_ps=800)},
            "launch service",
        ),
        (
            {"nvlink_bandwidth_bytes_per_second": 900_000_000_000},
            "owns local service",
        ),
    ],
)
def test_aggregate_authority_rejects_every_constructed_double_charge(
    tmp_path,
    calibration,
    selection,
    message,
):
    with pytest.raises(ValueError, match=message):
        _sink_config(
            tmp_path,
            ("node",) * 8,
            collective_floor_calibration=calibration,
            collective_floor_dtype="half",
            **selection,
        )


def test_live_seam_charges_each_aggregate_half_once_outside_composition(
    tmp_path,
    calibration,
):
    off = HtsimStepSink(_sink_config(tmp_path / "off", ("node",) * 8))
    on = HtsimStepSink(
        _sink_config(
            tmp_path / "on",
            ("node",) * 8,
            collective_floor_calibration=calibration,
            collective_floor_dtype="half",
        )
    )
    record = _record()

    off_result = off(record)
    on_result = on(record)

    assert off_result is not None
    assert on_result is not None
    assert off.collective_floor_timing_outcomes == []
    (timing,) = on.collective_floor_timing_outcomes
    on_locality = on.locality_outcomes[0]
    off_locality = off.locality_outcomes[0]
    assert timing.host_launch_floor_ps == 0
    assert on.config.resolved_collective_latency_profile is None
    assert not on.collective_registration_ledger.enabled
    assert on_locality.base_phase_latency_ps == (0,) * on_locality.artifact_count
    assert on_locality.registration_phase_cost_ps == ()
    assert on_locality.backend_runs == 0

    grouped = {}
    for artifact in timing.artifacts:
        identity = (
            artifact.collective_operation_id,
            artifact.semantic_collective,
        )
        grouped.setdefault(identity, []).append(artifact)
        assert artifact.collective_base_latency_ps == 0
        assert artifact.registration_cost_ps == 0
        assert artifact.fabric_transport_ps == 0
        assert artifact.composed_service_ps == (
            artifact.aggregate_floor_ps + artifact.local_service_ps
        )
    assert len(grouped) == 4
    for phases in grouped.values():
        estimate = phases[0].estimate
        assert sum(phase.aggregate_floor_ps for phase in phases) == (
            estimate.floor_charge_ps
        )
        assert sum(phase.local_service_ps for phase in phases) == (
            estimate.serialization_ps
        )
        assert sum(phase.composed_service_ps for phase in phases) == (
            estimate.completion_ps
        )

    expected_delta = sum(on_locality.composed_phase_service_ps) - sum(
        off_locality.composed_phase_service_ps
    )
    assert on_result.step_latency_ps - off_result.step_latency_ps == expected_delta
    attribution = attribute_step_detail(on_result, on_locality)
    assert attribution.media.collective_floor_ps == sum(
        on_locality.collective_floor_phase_ps
    )
    assert attribution.media.collective_base_ps == 0
    assert attribution.media.collective_registration_ps == 0
    assert attribution.media.total_ps == on_result.step_latency_ps


def test_calibration_off_is_a_byte_exact_bypass_of_every_pinned_field(
    tmp_path,
    monkeypatch,
):
    implicit_calls: list[str] = []
    explicit_calls: list[str] = []
    record = _record()
    random.seed(76)
    initial_random_state = random.getstate()

    _stub_backend(monkeypatch, implicit_calls)
    implicit = HtsimStepSink(
        _sink_config(tmp_path / "implicit", ("node-a", "node-b"))
    )
    implicit_plan = implicit._plan_step(record)
    implicit_result = implicit(record)
    implicit_random_state = random.getstate()

    random.setstate(initial_random_state)
    _stub_backend(monkeypatch, explicit_calls)
    explicit = HtsimStepSink(
        _sink_config(
            tmp_path / "explicit",
            ("node-a", "node-b"),
            collective_floor_calibration=None,
            collective_floor_dtype=None,
        )
    )
    explicit_plan = explicit._plan_step(record)
    explicit_result = explicit(record)
    explicit_random_state = random.getstate()

    assert implicit_plan is not None
    assert explicit_plan is not None
    implicit_segments = asdict(implicit_plan.locality)
    explicit_segments = asdict(explicit_plan.locality)
    assert _json_bytes(explicit_segments) == _json_bytes(implicit_segments)
    assert explicit_result == implicit_result
    assert _json_bytes(asdict(explicit.outcomes[0])) == _json_bytes(
        asdict(implicit.outcomes[0])
    )
    assert _json_bytes(asdict(explicit.locality_outcomes[0])) == _json_bytes(
        asdict(implicit.locality_outcomes[0])
    )
    assert _goal_manifest(tmp_path / "explicit") == _goal_manifest(
        tmp_path / "implicit"
    )
    assert explicit_calls == implicit_calls
    assert explicit_random_state == implicit_random_state == initial_random_state
    assert implicit.collective_floor_timing_outcomes == []
    assert explicit.collective_floor_timing_outcomes == []
    assert implicit.locality_outcomes[0].collective_floor_phase_ps == ()
    assert explicit.locality_outcomes[0].collective_floor_phase_ps == ()


def test_metric_chain_reports_the_fitted_floor_in_ttft_and_tpot(
    tmp_path,
    calibration,
):
    sinks = {
        "off": HtsimStepSink(_sink_config(tmp_path / "metric-off", ("node",) * 8)),
        "on": HtsimStepSink(
            _sink_config(
                tmp_path / "metric-on",
                ("node",) * 8,
                collective_floor_calibration=calibration,
                collective_floor_dtype="half",
            )
        ),
    }
    totals = {}
    for arm, selected_sink in sinks.items():
        reducer = HtsimRequestMetricReducer({"r0": 0})
        virtual_time_ps = 0
        for step_index in range(2):
            record = _record(step_index, virtual_time_ps)
            result = selected_sink(record)
            assert result is not None
            reducer.consume(
                record,
                result,
                selected_sink.locality_outcomes[step_index],
            )
            virtual_time_ps = result.completed_at_ps
        (totals[arm],) = reducer.totals()

    assert totals["on"].ttft_ps > totals["off"].ttft_ps
    assert totals["on"].tpot_ps > totals["off"].tpot_ps
    assert totals["on"].ttft_media.collective_floor_ps > 0
    assert totals["on"].decode_media.collective_floor_ps > 0
    assert totals["off"].ttft_media.collective_floor_ps == 0
    assert totals["off"].decode_media.collective_floor_ps == 0
