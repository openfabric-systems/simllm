from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from simllm.deploy import (
    BUDGET_GPUS_EXCEEDED,
    BUDGET_NODES_EXCEEDED,
    DEPLOYMENT_CANDIDATE_SCHEMA,
    HBM_CAPACITY_EXCEEDED,
    PIPELINE_PARALLEL_UNPRICED,
    BudgetSpec,
    DeploymentCandidate,
    FabricSpec,
    ModelRef,
    PoolSpec,
    SlaSpec,
    WorkloadPoint,
    candidate_from_json,
    candidate_key,
    candidate_to_json,
    check_feasibility,
    from_json,
    to_json,
)


def _candidate() -> DeploymentCandidate:
    return DeploymentCandidate(
        candidate_id="decode-b100",
        model=ModelRef(
            framework="sglang",
            model_id="deepseek-ai/DeepSeek-V3",
            inventory_sha256="0" * 64,
        ),
        pools=(
            PoolSpec(
                role="decode",
                engines=2,
                gpus_per_engine=8,
                tensor_parallel=8,
                pipeline_parallel=1,
                expert_parallel=8,
                data_parallel=1,
                device="b100",
            ),
        ),
        fabric=FabricSpec(
            inter_node_bits_per_second=400_000_000_000,
            intra_node_bytes_per_second=450_000_000_000,
        ),
        workload=WorkloadPoint(
            arrival_rate_rps=100,
            prompt_tokens=1024,
            output_tokens=256,
            kv_context_tokens=2000,
        ),
        sla=SlaSpec(
            tpot_target_ps=4_000_000_000,
            ttft_target_ps=100_000_000_000,
        ),
        budget=BudgetSpec(max_gpus=16, max_nodes=2),
    )


def _feasibility(
    candidate: DeploymentCandidate,
    *,
    static_bytes: int = 79_000_000_000,
    capacity_bytes: int = 192_000_000_000,
):
    return check_feasibility(
        candidate,
        static_rank_bytes_per_pool={"decode": static_bytes},
        device_hbm_capacity_bytes={"b100": capacity_bytes},
    )


def test_candidate_strict_round_trip() -> None:
    candidate = _candidate()

    rendered = to_json(candidate)

    assert rendered["schema"] == DEPLOYMENT_CANDIDATE_SCHEMA
    assert from_json(rendered) == candidate
    assert candidate_to_json(candidate) == rendered
    assert candidate_from_json(rendered) == candidate


def test_candidate_key_is_stable_golden_literal() -> None:
    assert candidate_key(_candidate()) == (
        "fe5adaed6a45d80306ce2eecade7ddec219cf98868ff4666ce9657a4f4884a1b"
    )


@pytest.mark.parametrize(
    ("path", "field"),
    [
        ((), "unexpected"),
        (("model",), "unexpected"),
        (("pools", 0), "unexpected"),
        (("fabric",), "unexpected"),
        (("workload",), "unexpected"),
        (("sla",), "unexpected"),
        (("budget",), "unexpected"),
    ],
)
def test_candidate_rejects_unknown_fields(
    path: tuple[str | int, ...],
    field: str,
) -> None:
    payload = deepcopy(to_json(_candidate()))
    target: Any = payload
    for component in path:
        target = target[component]
    target[field] = 1

    with pytest.raises(ValueError, match="unknown fields"):
        from_json(payload)


def test_candidate_checks_schema_before_unknown_fields() -> None:
    payload = to_json(_candidate())
    payload["schema"] = "simllm-deployment-candidate-v0"
    payload["unexpected"] = 1

    with pytest.raises(ValueError, match="unsupported schema") as error:
        from_json(payload)

    assert "unknown fields" not in str(error.value)


@pytest.mark.parametrize(
    ("path", "field"),
    [
        (("pools", 0), "engines"),
        (("fabric",), "inter_node_bits_per_second"),
        (("workload",), "prompt_tokens"),
        (("sla",), "tpot_target_ps"),
        (("budget",), "max_gpus"),
    ],
)
def test_candidate_rejects_boolean_masquerading_as_integer(
    path: tuple[str | int, ...],
    field: str,
) -> None:
    payload = deepcopy(to_json(_candidate()))
    target: Any = payload
    for component in path:
        target = target[component]
    target[field] = True

    with pytest.raises(ValueError, match="expected an integer"):
        from_json(payload)


def test_candidate_rejects_duplicate_pool_roles() -> None:
    candidate = _candidate()

    with pytest.raises(ValueError, match="duplicate pool roles"):
        replace(candidate, pools=(candidate.pools[0], candidate.pools[0]))


@pytest.mark.parametrize(
    "field",
    [
        "engines",
        "gpus_per_engine",
        "tensor_parallel",
        "pipeline_parallel",
        "expert_parallel",
        "data_parallel",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_pool_rejects_nonpositive_widths(field: str, value: int) -> None:
    values = {
        "role": "decode",
        "engines": 1,
        "gpus_per_engine": 8,
        "tensor_parallel": 8,
        "pipeline_parallel": 1,
        "expert_parallel": 8,
        "data_parallel": 1,
        "device": "b100",
    }
    values[field] = value

    with pytest.raises(ValueError, match="must be at least 1"):
        PoolSpec(**values)  # type: ignore[arg-type]


def test_pipeline_parallel_reason_has_accepted_and_rejected_boundaries() -> None:
    candidate = _candidate()
    assert _feasibility(candidate).accepted

    rejected = replace(
        candidate,
        pools=(replace(candidate.pools[0], pipeline_parallel=2),),
    )

    report = _feasibility(rejected)
    assert not report.accepted
    assert report.reasons == (PIPELINE_PARALLEL_UNPRICED,)


def test_hbm_capacity_reason_has_accepted_and_rejected_boundaries() -> None:
    candidate = _candidate()
    assert _feasibility(candidate, static_bytes=191_999_999_999).accepted

    report = _feasibility(candidate, static_bytes=192_000_000_000)

    assert not report.accepted
    assert report.reasons == (HBM_CAPACITY_EXCEEDED,)


def test_gpu_budget_reason_has_accepted_and_rejected_boundaries() -> None:
    candidate = _candidate()
    assert _feasibility(candidate).accepted

    report = _feasibility(replace(candidate, budget=replace(candidate.budget, max_gpus=15)))

    assert not report.accepted
    assert report.reasons == (BUDGET_GPUS_EXCEEDED,)


def test_node_budget_reason_has_accepted_and_rejected_boundaries() -> None:
    candidate = _candidate()
    assert _feasibility(candidate).accepted

    report = _feasibility(replace(candidate, budget=replace(candidate.budget, max_nodes=1)))

    assert not report.accepted
    assert report.reasons == (BUDGET_NODES_EXCEEDED,)


def test_feasibility_reports_all_reasons_once_in_stable_order() -> None:
    candidate = _candidate()
    rejected = replace(
        candidate,
        pools=(replace(candidate.pools[0], pipeline_parallel=2),),
        budget=BudgetSpec(max_gpus=1, max_nodes=1),
    )

    report = _feasibility(rejected, static_bytes=192_000_000_000)

    assert not report.accepted
    assert report.reasons == (
        PIPELINE_PARALLEL_UNPRICED,
        HBM_CAPACITY_EXCEEDED,
        BUDGET_GPUS_EXCEEDED,
        BUDGET_NODES_EXCEEDED,
    )


def test_feasibility_requires_sizing_inputs_for_every_pool_and_device() -> None:
    candidate = _candidate()

    with pytest.raises(ValueError, match="missing candidate pool role 'decode'"):
        check_feasibility(
            candidate,
            static_rank_bytes_per_pool={},
            device_hbm_capacity_bytes={"b100": 1},
        )
    with pytest.raises(ValueError, match="missing candidate device 'b100'"):
        check_feasibility(
            candidate,
            static_rank_bytes_per_pool={"decode": 0},
            device_hbm_capacity_bytes={},
        )


def test_non_ascii_candidate_string_is_rejected_everywhere() -> None:
    with pytest.raises(ValueError, match="model.model_id: v1 candidate strings must be ASCII"):
        replace(
            _candidate().model,
            model_id="Qwen/通义千问-72B",
        ).__post_init__()
    payload = to_json(_candidate())
    payload = deepcopy(payload)
    payload["candidate_id"] = "decode-é"
    with pytest.raises(ValueError, match="candidate.candidate_id: v1 candidate strings must be ASCII"):
        from_json(payload)


def test_unpaired_surrogate_candidate_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="candidate.candidate_id: v1 candidate strings must be ASCII"):
        replace(_candidate(), candidate_id="bad-\ud800").__post_init__()


def test_candidate_key_is_total_over_valid_candidates() -> None:
    first = candidate_key(_candidate())
    second = candidate_key(_candidate())
    assert first == second
    assert len(first) == 64


def test_pool_parse_errors_carry_indexed_paths() -> None:
    payload = deepcopy(to_json(_candidate()))
    prefill = deepcopy(payload["pools"][0])
    prefill["role"] = "prefill"
    payload["pools"] = [prefill, deepcopy(payload["pools"][0])]
    payload["pools"][1]["role"] = "chat"
    with pytest.raises(ValueError, match=r"candidate\.pools\[1\]\.role: unknown value 'chat'"):
        from_json(payload)
    payload["pools"][1]["role"] = "decode"
    payload["pools"][1]["device"] = "tpu9"
    with pytest.raises(ValueError, match=r"candidate\.pools\[1\]\.device: unknown GPU envelope 'tpu9'"):
        from_json(payload)
