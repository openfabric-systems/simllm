"""SGLang disaggregated-session mechanics without importing SGLang."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from simllm.adapters.sglang import pd_session
from simllm.adapters.sglang.pd_session import (
    DEPLOYMENT_CURVE_POINT_SCHEMA,
    DEPLOYMENT_CURVE_SCHEMA,
    SGLANG_PD_JOIN_MODE,
    SglangDisaggregatedSession,
    SglangPdCurveRecord,
    SglangPdRequest,
    SglangPdSessionConfig,
)
from simllm.compute import ModelDims
from simllm.core import (
    DeclaredKvHandoffPolicy,
    KvHandoffGeometry,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
)
from simllm.placement import (
    SglangPoolArrangement,
    sglang_disaggregated_manifests,
)


def _dims() -> ModelDims:
    return ModelDims(
        num_layers=24,
        hidden_size=2048,
        intermediate_size=8192,
        num_heads=16,
        num_kv_heads=8,
        head_size=128,
        vocab_size=49_159,
    )


def _geometry() -> KvHandoffGeometry:
    return KvHandoffGeometry(
        num_layers=24,
        num_kv_heads=8,
        head_size=64,
        element_bytes=2,
    )


def _arrangement(size: int) -> SglangPoolArrangement:
    return SglangPoolArrangement(
        enable_data_parallel_attention=True,
        attention_data_parallel_size=size,
        dense_data_parallel_size=size,
        expert_parallel_size=size,
    )


def _config(
    tmp_path: Path,
    *,
    handoff_ps: int = 100,
    prefill_engines: int = 1,
    decode_engines: int = 1,
) -> SglangPdSessionConfig:
    return SglangPdSessionConfig(
        model_path=tmp_path / "model",
        workdir=tmp_path / "run",
        dims=_dims(),
        handoff_geometry=_geometry(),
        handoff_policy=DeclaredKvHandoffPolicy(handoff_ps),
        prefill_arrangement=_arrangement(prefill_engines * 8),
        decode_arrangement=_arrangement(decode_engines * 8),
        prefill_engines=prefill_engines,
        decode_engines=decode_engines,
    )


class _FakePoolEngine:
    """Deterministic scheduler-shaped test double at the process RPC seam."""

    launches: ClassVar[list] = []

    def __init__(self, config, *, timeout_s):
        del timeout_s
        self.config = config
        self.role = config.role
        self.ordinal = config.ordinal
        self.engine_id = config.engine_id
        self.ranks = config.ranks
        self.attention_data_parallel_ranks = (
            config.attention_data_parallel_ranks
        )
        self.dense_data_parallel_ranks = config.dense_data_parallel_ranks
        self.expert_parallel_ranks = config.expert_parallel_ranks
        self.process_id = len(type(self).launches) + 100
        self.scheduler_type = "Scheduler"
        self.worker_type = "SimTpModelWorker"
        self.records = []
        self._unfinished = {}
        self._step_index = 0
        type(self).launches.append(config)

    @property
    def has_unfinished_requests(self):
        return bool(self._unfinished)

    def submit(self, *, request_id, input_token_ids, max_new_tokens):
        del input_token_ids
        self._unfinished[request_id] = max_new_tokens

    def step(self, now_ps):
        request_ids = list(self._unfinished)
        sampled = list(request_ids)
        phase = (
            RequestPhase.PREFILL
            if self.role.value == "prefill"
            else RequestPhase.DECODE
        )
        record = StepRecord(
            step_index=self._step_index,
            virtual_time_ps=now_ps,
            scheduled=[
                ScheduledRequest(request_id, phase, 1, context_length=1)
                for request_id in request_ids
            ],
            num_sampled=len(sampled),
            sampled_request_ids=sampled,
        )
        self._step_index += 1
        self.records.append(record)
        completions = []
        for request_id in request_ids:
            remaining = self._unfinished[request_id] - 1
            if remaining:
                self._unfinished[request_id] = remaining
            else:
                del self._unfinished[request_id]
                completions.append(
                    SimpleNamespace(
                        request_id=request_id,
                        output_token_count=(
                            1 if self.role.value == "prefill" else 4
                        ),
                    )
                )
        return {
            "completed_at_ps": now_ps + 10,
            "record": record,
            "completions": tuple(completions),
            "token_id": 512,
        }

    def close(self):
        return None


@pytest.fixture
def fake_pool_engines(monkeypatch):
    _FakePoolEngine.launches = []
    monkeypatch.setattr(pd_session, "_ProcessPoolEngine", _FakePoolEngine)
    return _FakePoolEngine


def test_structural_arrangements_project_role_groups_and_flagship_width():
    manifests = sglang_disaggregated_manifests(
        prefill_nodes=4,
        decode_nodes=9,
        gpus_per_node=8,
        prefill_arrangement=_arrangement(32),
        decode_arrangement=SglangPoolArrangement(
            enable_data_parallel_attention=True,
            attention_data_parallel_size=72,
            dense_data_parallel_size=1,
            expert_parallel_size=72,
        ),
        framework_version="0.5.19.dev345+gbfeae4e79",
    )

    assert len(manifests.placement.ranks) == 104
    assert len(tuple(manifests.fabric.nodes)) == 13
    assert manifests.placement.group_ranks(0, "attn_dp") == list(range(32))
    assert manifests.placement.group_ranks(0, "dense_dp") == list(range(32))
    assert manifests.placement.group_ranks(0, "ep") == list(range(32))
    assert manifests.placement.group_ranks(32, "attn_dp") == list(range(32, 104))
    assert manifests.placement.group_ranks(32, "dense_dp") == [32]
    assert manifests.placement.group_ranks(32, "ep") == list(range(32, 104))


def test_arrangement_rejects_disabled_attention_with_parallel_width():
    with pytest.raises(ValueError, match="disabled data-parallel attention"):
        SglangPoolArrangement(False, 2, 1, 1)


def test_session_rejects_arrangement_that_does_not_divide_role(tmp_path):
    with pytest.raises(ValueError, match="does not divide simulated role width"):
        SglangPdSessionConfig(
            model_path=tmp_path / "model",
            workdir=tmp_path / "run",
            dims=_dims(),
            handoff_geometry=_geometry(),
            handoff_policy=DeclaredKvHandoffPolicy(100),
            prefill_arrangement=SglangPoolArrangement(True, 3, 1, 1),
            decode_arrangement=_arrangement(8),
        )


def test_session_conserves_stable_identities_and_scheduler_batches(
    tmp_path,
    fake_pool_engines,
):
    requests = (
        SglangPdRequest("request-a", (1, 2), 4, 0),
        SglangPdRequest("request-b", (3, 4), 4, 0),
    )

    with SglangDisaggregatedSession(_config(tmp_path)) as session:
        result = session.run_requests(requests)

    assert [row.timeline.request_id for row in result.requests] == [
        "request-a",
        "request-b",
    ]
    assert len(session.handoffs) == 2
    assert result.maximum_prefill_batch_size == 2
    assert result.maximum_decode_batch_size == 2
    assert all(len(row.decode_token_ids) == 4 for row in result.requests)
    assert all(row.timeline.decomposition_total_ps == row.timeline.ttft_ps for row in result.requests)
    assert all(row.timeline.tpot_ps == Fraction(10) for row in result.requests)
    assert all(row.join_metadata["join_mode"] == SGLANG_PD_JOIN_MODE for row in result.requests)
    assert all(
        row.join_metadata["prefill_process_id"]
        != row.join_metadata["decode_process_id"]
        for row in result.requests
    )
    assert all(
        row.prefill_internal_request_id != row.decode_internal_request_id
        for row in result.requests
    )
    assert len(fake_pool_engines.launches) == 2
    assert fake_pool_engines.launches[0].expert_parallel_ranks == tuple(range(8))
    assert fake_pool_engines.launches[1].expert_parallel_ranks == tuple(range(8, 16))


def test_handoff_constant_moves_ttft_alone(tmp_path, fake_pool_engines):
    request = SglangPdRequest("request", (1, 2), 4, 0)
    with SglangDisaggregatedSession(
        _config(tmp_path / "low", handoff_ps=100)
    ) as low_session:
        low = low_session.run_requests((request,)).requests[0].timeline
    with SglangDisaggregatedSession(
        _config(tmp_path / "high", handoff_ps=200)
    ) as high_session:
        high = high_session.run_requests((request,)).requests[0].timeline

    assert high.ttft_ps - low.ttft_ps == 100
    assert high.tpot_ps == low.tpot_ps
    assert high.prefill_service_ps == low.prefill_service_ps
    assert high.decode_first_token_service_ps == low.decode_first_token_service_ps


def test_curve_records_use_the_framework_shared_exact_schema(
    tmp_path,
    fake_pool_engines,
):
    requests = (
        SglangPdRequest("request-a", (1, 2), 4, 0),
        SglangPdRequest("request-b", (3, 4), 4, 5),
    )
    with SglangDisaggregatedSession(_config(tmp_path)) as session:
        result = session.run_requests(requests)
    points = tuple(result.curve_point(Fraction(load)) for load in (8, 16, 32))
    rendered = SglangPdCurveRecord(
        "p1-d1-prompt2",
        1,
        1,
        2,
        points,
    ).to_json()

    assert rendered["schema"] == DEPLOYMENT_CURVE_SCHEMA
    assert rendered["orientation"] == {
        "x": "aggregated-output-throughput-rightward",
        "y": "inverse-per-token-request-delay-upward",
    }
    assert all(
        point["schema"] == DEPLOYMENT_CURVE_POINT_SCHEMA
        for point in rendered["points"]
    )
    assert set(rendered["points"][0]) == {
        "schema",
        "offered_load_requests_per_second",
        "aggregated_output_throughput_tokens_per_second",
        "per_token_request_delay_ps",
        "request_count",
        "output_token_count",
        "first_admitted_at_ps",
        "last_completed_at_ps",
    }
