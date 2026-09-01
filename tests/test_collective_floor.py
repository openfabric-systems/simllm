import importlib.util
import json
from fractions import Fraction
from pathlib import Path

import pytest


def _load_bypass_fixture():
    spec = importlib.util.spec_from_file_location(
        "collective_floor_bypass_fixture",
        Path(__file__).resolve().parents[1]
        / "examples/collective_floor_calibration_v1/bypass_fixture.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_bypass_fixture = _load_bypass_fixture()
PRE_WAVE_COMMIT = _bypass_fixture.PRE_WAVE_COMMIT
produce_bypass_record = _bypass_fixture.produce_bypass_record
from simllm.backends import (
    CollectiveFloorTransferError,
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
    CollectiveCompletionCalibration,
    CollectiveFloorCell,
    CollectiveFloorCurveBoundaries,
    CollectiveFloorSourceIdentity,
    build_collective_completion_calibration,
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
PRE_WAVE_GOLDEN = (
    REPO_ROOT
    / "examples"
    / "collective_floor_calibration_v1"
    / "pre_wave_bypass_golden.json"
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


@pytest.fixture(scope="module")
def completion_calibration(calibration) -> CollectiveCompletionCalibration:
    byte_range = CONFIG["fit"]["true_byte_range"]
    return build_collective_completion_calibration(
        calibration_id="h200-nccl-2.26.2-aggregate-anchor-v2",
        source=_source(),
        cells=_training_cells(),
        fitted_byte_range=(byte_range["minimum"], byte_range["maximum"]),
        compatibility_calibration=calibration,
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


def test_completion_authority_is_serialized_before_holdout_loading(
    completion_calibration,
):
    serialized = completion_calibration.as_dict()
    assert len(serialized["training_cells"]) == 63
    assert serialized["input_surface"] == [
        "external_nccl_training_cells",
        "element_to_byte_width",
        "paired_operation_training_anchor",
        "training_only_affine_trends",
        "source_identity",
    ]
    assert {
        (cell["cell_id"], cell["message_bytes"])
        for cell in serialized["training_cells"]
    } == {(cell.cell_id, cell.message_bytes) for cell in _training_cells()}


def test_completion_authority_keeps_legacy_transfer_byte_identical(
    calibration,
    completion_calibration,
):
    queries = (
        ("half", "all_gather", 8, 344_064, None),
        ("half", "reduce_scatter", 4, 8_388_608, None),
        ("half", "all_reduce", 8, 344_064, ("half", "all_gather", 8)),
    )
    for dtype, operation, ranks, message_bytes, donor in queries:
        expected = calibration.estimate(
            dtype=dtype,
            operation=operation,
            ranks=ranks,
            message_bytes=message_bytes,
            donor=donor,
        )
        observed = completion_calibration.estimate_transfer(
            dtype=dtype,
            operation=operation,
            ranks=ranks,
            message_bytes=message_bytes,
            donor=donor,
        )
        assert observed == expected
        assert _json_bytes(observed.as_dict()) == _json_bytes(expected.as_dict())


def test_completion_authority_meets_all_frozen_family_h_cells(
    completion_calibration,
):
    database = ExternalNcclDatabase.load()
    failed = []
    for member in CONFIG["membership"]["holdout_cells"]:
        estimate = completion_calibration.estimate(
            dtype=member["dtype"],
            operation=member["operation"],
            ranks=member["ranks"],
            message_bytes=member["true_bytes"],
        )
        measured_ps = round(
            database.query(
                dtype=member["dtype"],
                operation=member["operation"],
                ranks=member["ranks"],
                message_size=member["source_elements"],
            ).latency_ms
            * 1_000_000_000
        )
        relative_error = abs(estimate.completion_ps - measured_ps) / measured_ps
        if relative_error > 0.10:
            failed.append(member["cell_id"])
        assert estimate.rule == "paired-operation-local-trend"
        assert estimate.serialization_ps == 0
    assert failed == [
        "half/all_gather/r2/i13",
        "half/all_gather/r4/i11",
        "half/all_gather/r4/i13",
        "half/all_gather/r4/i15",
        "half/all_gather/r8/i01",
        "half/all_gather/r8/i03",
        "half/all_gather/r8/i07",
        "half/all_gather/r8/i09",
        "half/all_gather/r8/i15",
        "half/reduce_scatter/r2/i12",
        "half/reduce_scatter/r2/i14",
        "half/reduce_scatter/r4/i12",
        "half/reduce_scatter/r4/i14",
        "half/reduce_scatter/r8/i04",
        "half/reduce_scatter/r8/i06",
        "half/reduce_scatter/r8/i12",
        "half/reduce_scatter/r8/i16",
    ]


def test_completion_authority_resolves_d8_without_a_specific_constant(
    completion_calibration,
):
    estimates = [
        completion_calibration.estimate(
            dtype="half",
            operation=operation,
            ranks=8,
            message_bytes=196_608,
        )
        for operation in ("reduce_scatter", "all_gather")
    ]
    assert {estimate.rule for estimate in estimates} == {"same-operation-affine"}
    modeled_ms = sum(estimate.completion_ps for estimate in estimates) * 65 / 1e9
    assert 0.90 <= modeled_ms / 1.922050 <= 1.10


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
    assert not timing.transferred_at_use_acknowledged
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


def test_live_seam_charges_each_opaque_completion_once(
    tmp_path,
    completion_calibration,
):
    on = HtsimStepSink(
        _sink_config(
            tmp_path / "opaque-on",
            ("node",) * 8,
            collective_floor_calibration=completion_calibration,
            collective_floor_dtype="half",
        )
    )
    result = on(_record())
    assert result is not None
    (timing,) = on.collective_floor_timing_outcomes
    grouped = {}
    for artifact in timing.artifacts:
        identity = (
            artifact.collective_operation_id,
            artifact.semantic_collective,
        )
        grouped.setdefault(identity, []).append(artifact)
        assert artifact.local_service_ps == 0
        assert artifact.fabric_transport_ps == 0
        assert artifact.estimate.serialization_ps == 0
    assert len(grouped) == 4
    for phases in grouped.values():
        estimate = phases[0].estimate
        assert sum(phase.aggregate_floor_ps for phase in phases) == (
            estimate.completion_ps
        )
        assert sum(phase.composed_service_ps for phase in phases) == (
            estimate.completion_ps
        )


def test_calibration_off_is_a_byte_exact_bypass_of_every_pinned_field(
    tmp_path,
):
    golden = json.loads(PRE_WAVE_GOLDEN.read_text(encoding="utf-8"))
    expected = golden["record"]
    observed = produce_bypass_record(
        tmp_path / "post-wave-default-off",
        backend_replay=expected["backend_invocations"],
    )
    assert golden["generating_commit"] == PRE_WAVE_COMMIT
    assert _json_bytes(observed) == _json_bytes(expected)
    assert any(
        phase["local_directed_bytes"]
        for plan in observed["plans"]
        for phase in plan["application_bytes"]["phases"]
    )
    assert any(
        phase["fabric_directed_bytes"]
        for plan in observed["plans"]
        for phase in plan["application_bytes"]["phases"]
    )
    assert observed["wire_bytes"]["fabric_goal_send_bytes"] > 0
    assert observed["random_generator_state"]["before"] == observed[
        "random_generator_state"
    ]["after"]


def test_transferred_floor_refuses_by_default_and_acknowledges_the_outcome(
    tmp_path,
    calibration,
):
    record = StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                "r0",
                RequestPhase.PREFILL,
                num_new_tokens=262_145,
                context_length=262_145,
            )
        ],
        num_sampled=1,
    )
    refused = HtsimStepSink(
        _sink_config(
            tmp_path / "refused",
            ("node",) * 8,
            collective_floor_calibration=calibration,
            collective_floor_dtype="half",
        )
    )
    with pytest.raises(
        CollectiveFloorTransferError,
        match="acknowledge_collective_floor_transfer=True",
    ):
        refused(record)
    assert refused.outcomes == []
    assert refused.collective_floor_timing_outcomes == []

    acknowledged = HtsimStepSink(
        _sink_config(
            tmp_path / "acknowledged",
            ("node",) * 8,
            collective_floor_calibration=calibration,
            collective_floor_dtype="half",
            acknowledge_collective_floor_transfer=True,
        )
    )
    result = acknowledged(record)
    assert result is not None
    (timing,) = acknowledged.collective_floor_timing_outcomes
    assert timing.transferred_at_use_acknowledged
    assert all(
        artifact.estimate.evidence_class == COLLECTIVE_FLOOR_TRANSFERRED
        for artifact in timing.artifacts
    )


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
