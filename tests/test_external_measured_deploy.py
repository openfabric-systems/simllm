from __future__ import annotations

import json
from pathlib import Path

import pytest

from simllm.calibration.external_db import (
    ExternalOperationDatabase,
    default_artifact_dir,
)
from simllm.deploy import (
    BudgetSpec,
    DeploymentCandidate,
    EnvelopeSpec,
    EstimatorInputs,
    EvidenceClass,
    ExternalQwen32BDeploymentBinding,
    FabricSpec,
    ModelRef,
    ModelWork,
    PoolSpec,
    SlaSpec,
    TermEstimate,
    WorkloadPoint,
    estimate_decode_step,
    validate_external_scored_stamp,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "examples/matched_seam_frontier_v1/study_config.json").read_text(
        encoding="utf-8"
    )
)
INVENTORY_SHA256 = "a" * 64


@pytest.fixture(scope="module")
def binding() -> ExternalQwen32BDeploymentBinding:
    return ExternalQwen32BDeploymentBinding(
        ExternalOperationDatabase.load(default_artifact_dir())
    )


@pytest.mark.parametrize("oracle", CONFIG["oracles"]["decode"])
def test_decode_binding_matches_frozen_external_sdk_hex(
    binding: ExternalQwen32BDeploymentBinding,
    oracle: dict[str, object],
) -> None:
    value = binding.decode_service(
        tensor_parallel=int(oracle["tensor_parallel"]),
        batch_size=int(oracle["batch_size"]),
        isl=int(oracle["isl"]),
        osl=int(oracle["osl"]),
        prefix=int(oracle["prefix"]),
        stride=int(oracle["stride"]),
        latency_correction_scale=float.fromhex(
            CONFIG["composition"]["decode_latency_correction_hex"]
        ),
    )

    assert value.total_ms_hex == oracle["expected_total_ms_hex"]
    assert value.service_ms_hex == oracle["expected_step_ms_hex"]
    assert value.evidence_class == "MEASURED-EXTERNAL"


@pytest.mark.parametrize("oracle", CONFIG["oracles"]["prefill"])
def test_prefill_binding_matches_frozen_external_sdk_hex(
    binding: ExternalQwen32BDeploymentBinding,
    oracle: dict[str, object],
) -> None:
    value = binding.prefill_service(
        tensor_parallel=int(oracle["tensor_parallel"]),
        batch_size=int(oracle["batch_size"]),
        isl=int(oracle["isl"]),
        prefix=int(oracle["prefix"]),
        latency_correction_scale=float.fromhex(
            CONFIG["composition"]["prefill_latency_correction_hex"]
        ),
    )

    assert value.service_ms_hex == oracle["expected_service_ms_hex"]
    assert value.evidence_class == "MEASURED-EXTERNAL"


def _candidate() -> DeploymentCandidate:
    return DeploymentCandidate(
        candidate_id="external-scored-tp4",
        model=ModelRef("external", "Qwen/Qwen3-32B-FP8", INVENTORY_SHA256),
        pools=(PoolSpec("decode", 1, 4, 4, 1, 1, 1, "h200"),),
        fabric=FabricSpec(400_000_000_000, 900_000_000_000),
        workload=WorkloadPoint(None, 4000, 500, 4250),
        sla=SlaSpec(None, None),
        budget=BudgetSpec(None, None),
    )


def _inputs(*, surfaces: tuple) -> EstimatorInputs:
    return EstimatorInputs(
        model_work=ModelWork(
            "unused-external-surface-placeholder",
            1,
            0,
            0,
            0,
            INVENTORY_SHA256,
            "not consumed when MEASURED-EXTERNAL owns kernel service",
        ),
        envelopes={
            "h200": EnvelopeSpec(
                "h200",
                1,
                1,
                1.0,
                "not consumed when MEASURED-EXTERNAL owns kernel service",
            )
        },
        surfaces=surfaces,
        surface_evidence=EvidenceClass.MEASURED_EXTERNAL,
        surface_source="pinned external pass surface",
    )


def test_external_surface_owns_the_scored_kernel_without_roofline(
    binding: ExternalQwen32BDeploymentBinding,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import simllm.deploy.estimator as estimator_module

    class UnexpectedRooflineProvider:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("RooflineProvider must be bypassed by the external surface")

    monkeypatch.setattr(estimator_module, "RooflineProvider", UnexpectedRooflineProvider)
    values = binding.decode_surface(
        tensor_parallel=4,
        batch_sizes=(56, 64),
        isl=4000,
        osl=500,
        prefix=500,
        stride=32,
        latency_correction_scale=float.fromhex(
            CONFIG["composition"]["decode_latency_correction_hex"]
        ),
    )
    points = tuple(value.as_batch_service_point() for value in values)

    estimate = estimate_decode_step(_candidate(), 64, _inputs(surfaces=points))

    assert estimate.step_ps == values[1].service_ps
    assert estimate.kernel_floor.evidence is EvidenceClass.MEASURED_EXTERNAL
    assert all(
        term.estimate.evidence is not EvidenceClass.ROOFLINE
        for term in estimate.stamp.terms
    )
    validate_external_scored_stamp(estimate.stamp)


@pytest.mark.parametrize(
    "evidence",
    [EvidenceClass.ROOFLINE, EvidenceClass.DECLARED],
)
def test_external_scored_stamp_rejects_roofline_and_fitted_terms(
    evidence: EvidenceClass,
) -> None:
    estimate = estimate_decode_step(_candidate(), 1, EstimatorInputs(
        model_work=ModelWork(
            "fallback",
            1,
            0,
            0,
            0,
            INVENTORY_SHA256,
            "fallback source",
        ),
        envelopes={"h200": EnvelopeSpec("h200", 1, 1, 1.0, "fallback")},
    ))
    terms = list(estimate.stamp.terms)
    terms[0] = type(terms[0])(
        terms[0].name,
        TermEstimate(terms[0].estimate.duration_ps, evidence, "disallowed term"),
    )
    mutant = type(estimate.stamp)(estimate.stamp.candidate_key, tuple(terms))

    with pytest.raises(ValueError, match="non-MEASURED-EXTERNAL"):
        validate_external_scored_stamp(mutant)


def test_external_surface_refuses_locally_measured_point_relabelling() -> None:
    from simllm.calibration.batch_service_surface import BatchServicePoint

    point = BatchServicePoint(1, 1, 0.0, "b" * 64)

    with pytest.raises(ValueError, match="every batch-service point"):
        _inputs(surfaces=(point, point))
