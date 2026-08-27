from __future__ import annotations

import subprocess
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import pytest

from simllm.calibration import BatchServicePoint
from simllm.deploy import (
    DEPLOYMENT_ESTIMATE_SCHEMA,
    BudgetSpec,
    DeploymentCandidate,
    EnvelopeSpec,
    EstimatorClass,
    EstimatorInputs,
    EvidenceClass,
    FabricSpec,
    ModelRef,
    ModelWork,
    PoolSpec,
    SimDerivedTerms,
    SlaSpec,
    TermEstimate,
    WorkloadPoint,
    decode_capacity_requests_per_second,
    estimate_decode_step,
    estimate_prefill_request,
    estimate_stamp_from_json,
    estimate_stamp_to_json,
    match_pools,
    queue_delay_ps,
    queue_occupancy,
)

INVENTORY_SHA256 = "a" * 64
RECORD_SHA256 = "b" * 64


def _pool(
    role: str,
    *,
    engines: int = 1,
    width: int = 8,
    device: str = "b100",
) -> PoolSpec:
    return PoolSpec(
        role=role,
        engines=engines,
        gpus_per_engine=width,
        tensor_parallel=min(width, 8),
        pipeline_parallel=1,
        expert_parallel=max(1, width // 8),
        data_parallel=1,
        device=device,
    )


def _candidate(
    *,
    pools: tuple[PoolSpec, ...] | None = None,
    arrival_rate_rps: int | None = 100,
    output_tokens: int = 4,
    fabric_bps: int = 400_000_000_000,
    intra_bytes_per_second: int = 100_000_000_000,
    tpot_target_ps: int | None = None,
    ttft_target_ps: int | None = None,
) -> DeploymentCandidate:
    if pools is None:
        pools = (_pool("decode"),)
    return DeploymentCandidate(
        candidate_id="synthetic-capacity-cell",
        model=ModelRef(
            framework="synthetic",
            model_id="synthetic/model",
            inventory_sha256=INVENTORY_SHA256,
        ),
        pools=pools,
        fabric=FabricSpec(
            inter_node_bits_per_second=fabric_bps,
            intra_node_bytes_per_second=intra_bytes_per_second,
        ),
        workload=WorkloadPoint(
            arrival_rate_rps=arrival_rate_rps,
            prompt_tokens=16,
            output_tokens=output_tokens,
            kv_context_tokens=2_000,
        ),
        sla=SlaSpec(
            tpot_target_ps=tpot_target_ps,
            ttft_target_ps=ttft_target_ps,
        ),
        budget=BudgetSpec(max_gpus=None, max_nodes=None),
    )


def _model_work(
    *,
    flops_per_batch_item: int = 8_000_000_000_000,
    static_logical_hbm_bytes: int = 10_000_000_000,
    dynamic_hbm_bytes_per_batch_item: int = 0,
    collective_bytes_per_item: int = 0,
) -> ModelWork:
    return ModelWork(
        kernel_name="synthetic-decode-rank",
        flops_per_batch_item=flops_per_batch_item,
        static_logical_hbm_bytes=static_logical_hbm_bytes,
        dynamic_hbm_bytes_per_batch_item=dynamic_hbm_bytes_per_batch_item,
        logical_collective_bytes_per_gpu_per_batch_item=collective_bytes_per_item,
        inventory_sha256=INVENTORY_SHA256,
        source="synthetic model-work declaration",
    )


def _envelope(
    *,
    device: str = "b100",
    peak_flops_per_second: int = 8_000_000_000_000_000,
    hbm_bytes_per_second: int = 8_000_000_000_000,
) -> EnvelopeSpec:
    return EnvelopeSpec(
        device=device,
        peak_flops_per_second=peak_flops_per_second,
        hbm_bytes_per_second=hbm_bytes_per_second,
        efficiency=1.0,
        source="synthetic declared GPU envelope",
    )


def _inputs(
    *,
    model_work: ModelWork | None = None,
    envelope: EnvelopeSpec | None = None,
    surfaces: tuple[BatchServicePoint, ...] | None = None,
    surface_evidence: EvidenceClass = EvidenceClass.MEASURED,
    surface_source: str | None = None,
    sim_derived: SimDerivedTerms | None = None,
    prefill_service: TermEstimate | None = None,
    handoff_ps: int | None = None,
    handoff_source: str | None = None,
) -> EstimatorInputs:
    selected_envelope = _envelope() if envelope is None else envelope
    return EstimatorInputs(
        model_work=_model_work() if model_work is None else model_work,
        envelopes={selected_envelope.device: selected_envelope},
        surfaces=surfaces,
        surface_evidence=surface_evidence,
        surface_source=surface_source,
        sim_derived=sim_derived,
        prefill_service=prefill_service,
        handoff_ps=handoff_ps,
        handoff_source=handoff_source,
    )


def _surface() -> tuple[BatchServicePoint, ...]:
    return (
        BatchServicePoint(
            batch_size=2,
            duration_ps=200_000_000,
            uncertainty_fraction=0.0,
            entry_key_sha256="1" * 64,
        ),
        BatchServicePoint(
            batch_size=8,
            duration_ps=800_000_000,
            uncertainty_fraction=0.0,
            entry_key_sha256="2" * 64,
        ),
    )


def _frozen_model_work() -> ModelWork:
    return _model_work(
        flops_per_batch_item=112_322_823_926,
        static_logical_hbm_bytes=27_446_643_040,
        dynamic_hbm_bytes_per_batch_item=140_544_000,
        collective_bytes_per_item=13_303_808,
    )


def test_e1_roofline_literal_is_memory_bound() -> None:
    estimate = estimate_decode_step(_candidate(), 1, _inputs())

    assert estimate.kernel_floor.duration_ps == 1_250_000_000
    assert estimate.kernel_floor.evidence is EvidenceClass.ROOFLINE
    assert estimate.analytical_step_ps == 1_250_000_000
    assert estimate.step_ps == 1_250_000_000


def test_e2_fabric_floor_uses_largest_single_flow_and_floor_division() -> None:
    candidate = _candidate(pools=(_pool("decode", width=16),))
    inputs = _inputs(
        model_work=_model_work(
            flops_per_batch_item=1,
            static_logical_hbm_bytes=0,
            collective_bytes_per_item=1_000_000_000,
        )
    )

    estimate = estimate_decode_step(candidate, 1, inputs)

    assert estimate.fabric_floor.duration_ps == 10_000_000_000
    assert estimate.fabric_floor.evidence is EvidenceClass.DECLARED


def test_e3_intra_floor_uses_logical_bytes() -> None:
    candidate = _candidate(
        intra_bytes_per_second=450_000_000_000,
    )
    inputs = _inputs(
        model_work=_model_work(
            flops_per_batch_item=1,
            static_logical_hbm_bytes=0,
            collective_bytes_per_item=900_000_000,
        )
    )

    estimate = estimate_decode_step(candidate, 1, inputs)

    assert estimate.intra_floor.duration_ps == 2_000_000_000
    assert estimate.intra_floor.evidence is EvidenceClass.DECLARED


def test_e4_calls_the_landed_surface_interpolator() -> None:
    estimate = estimate_decode_step(
        _candidate(),
        4,
        _inputs(
            surfaces=_surface(),
            surface_evidence=EvidenceClass.DECLARED,
            surface_source="deployment scan E4 synthetic declaration",
        ),
    )

    assert estimate.batch_service == TermEstimate(
        duration_ps=400_000_000,
        evidence=EvidenceClass.DECLARED,
        source="deployment scan E4 synthetic declaration",
    )


def test_e5_queue_capacity_occupancy_and_overload_wait_literals() -> None:
    points = _surface()

    assert decode_capacity_requests_per_second(
        points,
        output_tokens=4,
        max_batch=8,
        decode_engines=1,
    ) == Fraction(2_500)
    assert queue_occupancy(
        points,
        offered_load_rps=500,
        output_tokens=4,
        max_batch=8,
        decode_engines=1,
    ) == 2
    assert queue_occupancy(
        points,
        offered_load_rps=2_000,
        output_tokens=4,
        max_batch=8,
        decode_engines=1,
    ) == 7
    assert queue_occupancy(
        points,
        offered_load_rps=4_000,
        output_tokens=4,
        max_batch=8,
        decode_engines=1,
    ) == 8
    assert queue_delay_ps(
        points,
        offered_load_rps=500,
        output_tokens=4,
        max_batch=8,
        decode_engines=1,
        cell_requests=64,
    ) == 0
    assert queue_delay_ps(
        points,
        offered_load_rps=2_000,
        output_tokens=4,
        max_batch=8,
        decode_engines=1,
        cell_requests=64,
    ) == 0
    assert queue_delay_ps(
        points,
        offered_load_rps=4_000,
        output_tokens=4,
        max_batch=8,
        decode_engines=1,
        cell_requests=64,
    ) == Fraction(4_725_000_000)


def test_e6_rate_match_literals_and_exact_utilizations() -> None:
    candidate = _candidate(
        pools=(
            _pool("prefill", engines=5),
            _pool("decode", engines=1),
        ),
        tpot_target_ps=400_000_000,
        ttft_target_ps=50_000_000_000,
    )

    report = match_pools(
        candidate,
        prefill_request_ps=50_000_000_000,
        decode_step_ps=400_000_000,
        batch_per_gpu=8,
    )

    assert report.required_prefill_engines == 5
    assert report.required_decode_engines == 1
    assert report.decode_capacity_requests_per_second == Fraction(5_000)
    assert report.prefill_utilization == 1
    assert report.decode_utilization == Fraction(1, 50)
    assert report.sla_pass


def test_rate_match_combined_pool_accounts_for_both_services() -> None:
    candidate = _candidate(pools=(_pool("combined", engines=6),))

    report = match_pools(
        candidate,
        prefill_request_ps=50_000_000_000,
        decode_step_ps=400_000_000,
        batch_per_gpu=8,
    )

    assert report.pool_matches[0].role == "combined"
    assert report.pool_matches[0].required_engines == 6
    assert report.pool_matches[0].utilization == Fraction(251, 300)
    assert report.sla_pass


def test_rate_match_sla_checks_latency_and_capacity() -> None:
    candidate = _candidate(
        pools=(
            _pool("prefill", engines=4),
            _pool("decode", engines=1),
        ),
        tpot_target_ps=399_999_999,
        ttft_target_ps=49_999_999_999,
    )

    report = match_pools(
        candidate,
        prefill_request_ps=50_000_000_000,
        decode_step_ps=400_000_000,
        batch_per_gpu=8,
    )

    assert not report.sla_pass


def test_prefill_request_keeps_handoff_additive_and_stamped() -> None:
    candidate = _candidate(
        pools=(
            _pool("prefill"),
            _pool("decode"),
        )
    )
    prefill = TermEstimate(
        95_424_000,
        EvidenceClass.DECLARED,
        "declared prefill service",
    )

    estimate = estimate_prefill_request(
        candidate,
        _inputs(
            prefill_service=prefill,
            handoff_ps=100_000_000,
            handoff_source="declared P/D handoff",
        ),
    )

    assert estimate.step_ps == 95_424_000
    assert estimate.request_ps == 195_424_000
    assert estimate.handoff == TermEstimate(
        100_000_000,
        EvidenceClass.DECLARED,
        "declared P/D handoff",
    )
    assert {term.name for term in estimate.stamp.terms} == {
        "prefill_service",
        "fabric_floor",
        "intra_floor",
        "handoff",
    }


def test_frozen_b100_batch_32_analytical_and_simulated_spot_literals() -> None:
    candidate = _candidate()
    inputs = _inputs(
        model_work=_frozen_model_work(),
        envelope=_envelope(
            peak_flops_per_second=1_800_000_000_000_000,
            hbm_bytes_per_second=8_000_000_000_000,
        ),
        sim_derived=SimDerivedTerms(
            fabric_excess_ps=0,
            intra_excess_ps=266_079_788,
            record_path="examples/deployment_frontier_v1/result.json",
            record_sha256=RECORD_SHA256,
        ),
    )

    estimate = estimate_decode_step(candidate, 32, inputs)

    assert estimate.analytical_step_ps == 4_257_218_560
    assert estimate.step_ps == 4_523_298_348
    assert estimate.stamp.consumes_sim_derived
    sim_terms = [
        term.estimate
        for term in estimate.stamp.terms
        if term.estimate.evidence is EvidenceClass.SIM_DERIVED
    ]
    assert len(sim_terms) == 2
    assert all("examples/deployment_frontier_v1/result.json" in term.source for term in sim_terms)
    assert all(RECORD_SHA256 in term.source for term in sim_terms)


@pytest.mark.parametrize(
    ("width", "device", "fabric_floor_ps", "intra_floor_ps"),
    [
        (8, "b100", 0, 133_038_080),
        (16, "h100", 133_038_080, 66_519_040),
        (72, "h100", 29_564_020, 14_782_000),
    ],
)
def test_three_frozen_byte_partition_shapes(
    width: int,
    device: str,
    fabric_floor_ps: int,
    intra_floor_ps: int,
) -> None:
    candidate = _candidate(pools=(_pool("decode", width=width, device=device),))
    envelope = _envelope(
        device=device,
        peak_flops_per_second=(
            1_800_000_000_000_000 if device == "b100" else 989_500_000_000_000
        ),
        hbm_bytes_per_second=(
            8_000_000_000_000 if device == "b100" else 3_350_000_000_000
        ),
    )

    estimate = estimate_decode_step(
        candidate,
        1,
        _inputs(model_work=_frozen_model_work(), envelope=envelope),
    )

    assert estimate.fabric_floor.duration_ps == fabric_floor_ps
    assert estimate.intra_floor.duration_ps == intra_floor_ps


def test_fabric_floor_scales_from_physical_bytes_and_declared_rate() -> None:
    work = _model_work(
        flops_per_batch_item=1,
        static_logical_hbm_bytes=0,
        collective_bytes_per_item=1_000_000_000,
    )
    inputs = _inputs(model_work=work)
    at_400g = estimate_decode_step(
        _candidate(pools=(_pool("decode", width=16),), fabric_bps=400_000_000_000),
        1,
        inputs,
    )
    at_200g = estimate_decode_step(
        _candidate(pools=(_pool("decode", width=16),), fabric_bps=200_000_000_000),
        1,
        inputs,
    )
    batch_two = estimate_decode_step(
        _candidate(pools=(_pool("decode", width=16),), fabric_bps=400_000_000_000),
        2,
        inputs,
    )

    assert at_400g.fabric_floor.duration_ps == 10_000_000_000
    assert at_200g.fabric_floor.duration_ps == 20_000_000_000
    assert batch_two.fabric_floor.duration_ps == 20_000_000_000


def test_measured_surface_stamp_names_selected_entry_keys() -> None:
    estimate = estimate_decode_step(
        _candidate(),
        4,
        _inputs(surfaces=_surface()),
    )

    assert estimate.batch_service is not None
    assert estimate.batch_service.evidence is EvidenceClass.MEASURED
    assert "1" * 64 in estimate.batch_service.source
    assert "2" * 64 in estimate.batch_service.source


def test_estimate_stamp_strict_round_trip() -> None:
    stamp = estimate_decode_step(_candidate(), 1, _inputs()).stamp

    rendered = estimate_stamp_to_json(stamp)

    assert rendered["schema"] == DEPLOYMENT_ESTIMATE_SCHEMA
    assert rendered["estimator_class"] == EstimatorClass.ESTIMATE.value
    assert estimate_stamp_from_json(rendered) == stamp


def test_estimate_stamp_checks_schema_before_unknown_fields() -> None:
    payload = estimate_stamp_to_json(
        estimate_decode_step(_candidate(), 1, _inputs()).stamp
    )
    payload["schema"] = "simllm-deployment-estimate-v0"
    payload["unexpected"] = 1

    with pytest.raises(ValueError, match="unsupported schema") as error:
        estimate_stamp_from_json(payload)

    assert "unknown fields" not in str(error.value)


@pytest.mark.parametrize(
    "path",
    [(), ("terms", 0)],
)
def test_estimate_stamp_rejects_unknown_fields(path: tuple[str | int, ...]) -> None:
    payload = deepcopy(
        estimate_stamp_to_json(estimate_decode_step(_candidate(), 1, _inputs()).stamp)
    )
    target: object = payload
    for component in path:
        target = target[component]  # type: ignore[index]
    target["unexpected"] = 1  # type: ignore[index]

    with pytest.raises(ValueError, match="unknown fields"):
        estimate_stamp_from_json(payload)


def test_estimator_does_not_create_a_process(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"process creation attempted: {args!r} {kwargs!r}")

    monkeypatch.setattr(subprocess, "Popen", forbidden)

    estimate = estimate_decode_step(_candidate(), 1, _inputs())

    assert estimate.step_ps == 1_250_000_000


def test_missing_envelope_fails_closed() -> None:
    inputs = _inputs()
    inputs.envelopes = {}

    with pytest.raises(ValueError, match="missing sourced envelope"):
        estimate_decode_step(_candidate(), 1, inputs)


def test_inventory_mismatch_fails_closed() -> None:
    candidate = replace(
        _candidate(),
        model=replace(_candidate().model, inventory_sha256="f" * 64),
    )

    with pytest.raises(ValueError, match="does not match candidate.model"):
        estimate_decode_step(candidate, 1, _inputs())


def test_unpriced_width_fails_closed_with_registry_owner() -> None:
    candidate = _candidate(pools=(_pool("decode", width=24),))

    with pytest.raises(ValueError, match="DEPLOY-4"):
        estimate_decode_step(candidate, 1, _inputs())


def test_surface_outside_measured_span_fails_closed() -> None:
    with pytest.raises(ValueError, match="outside the measured surface"):
        estimate_decode_step(
            _candidate(),
            16,
            _inputs(surfaces=_surface()),
        )


def test_declared_surface_requires_a_source() -> None:
    with pytest.raises(ValueError, match="surface_source"):
        _inputs(
            surfaces=_surface(),
            surface_evidence=EvidenceClass.DECLARED,
        )


def test_prefill_service_and_handoff_each_fail_closed_when_missing() -> None:
    candidate = _candidate(
        pools=(
            _pool("prefill"),
            _pool("decode"),
        )
    )
    with pytest.raises(ValueError, match="prefill_service"):
        estimate_prefill_request(candidate, _inputs())

    with pytest.raises(ValueError, match="handoff"):
        estimate_prefill_request(
            candidate,
            _inputs(
                prefill_service=TermEstimate(
                    1,
                    EvidenceClass.DECLARED,
                    "declared prefill service",
                )
            ),
        )


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: TermEstimate(1, EvidenceClass.DECLARED, ""),
        lambda: ModelWork(
            kernel_name="kernel",
            flops_per_batch_item=1,
            static_logical_hbm_bytes=0,
            dynamic_hbm_bytes_per_batch_item=0,
            logical_collective_bytes_per_gpu_per_batch_item=0,
            inventory_sha256=INVENTORY_SHA256,
            source="",
        ),
        lambda: EnvelopeSpec(
            device="b100",
            peak_flops_per_second=1,
            hbm_bytes_per_second=1,
            efficiency=1.0,
            source="",
        ),
        lambda: SimDerivedTerms(
            fabric_excess_ps=0,
            intra_excess_ps=0,
            record_path="",
            record_sha256=RECORD_SHA256,
        ),
        lambda: SimDerivedTerms(
            fabric_excess_ps=0,
            intra_excess_ps=0,
            record_path="result.json",
            record_sha256="not-a-sha",
        ),
    ],
)
def test_missing_evidence_sources_fail_closed(constructor: object) -> None:
    with pytest.raises(ValueError):
        constructor()  # type: ignore[operator]


def test_rate_match_requires_a_workload_rate() -> None:
    candidate = _candidate(
        pools=(_pool("prefill"), _pool("decode")),
        arrival_rate_rps=None,
    )

    with pytest.raises(ValueError, match="arrival_rate_rps"):
        match_pools(
            candidate,
            prefill_request_ps=1,
            decode_step_ps=1,
            batch_per_gpu=1,
        )
