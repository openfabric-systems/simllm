"""SGLang disaggregated-session mechanics without importing SGLang."""

from __future__ import annotations

import hashlib
import json
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
from simllm.calibration.kernel_cycle_lut import compile_session_profile_provider
from simllm.compute import (
    ComputeProvider,
    DurationEstimate,
    KernelSpec,
    ModelDims,
    step_kernel,
)
from simllm.core import (
    DeclaredKvHandoffPolicy,
    KvHandoffGeometry,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    StepResult,
)
from simllm.placement import (
    SglangPoolArrangement,
    sglang_disaggregated_manifests,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTATIONS_PATH = (
    REPOSITORY_ROOT / "examples/sglang_decode_shape_v1/expectations.json"
)
CANDIDATE_PATH = (
    REPOSITORY_ROOT
    / "examples/hopper_kernel_cycle_candidate_v1/candidate-record.json"
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
    project_remote_kv_length: bool = False,
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
        project_remote_kv_length=project_remote_kv_length,
    )


class _FakePoolEngine:
    """Deterministic scheduler-shaped test double at the process RPC seam."""

    launches: ClassVar[list] = []
    instances: ClassVar[list] = []

    def __init__(self, config, *, timeout_s):
        del timeout_s
        self.config = config
        self.role = config.role
        self.ordinal = config.ordinal
        self.engine_id = config.engine_id
        self.ranks = config.ranks
        self.tensor_parallel_ranks = config.tensor_parallel_ranks
        self.attention_data_parallel_ranks = (
            config.attention_data_parallel_ranks
        )
        self.dense_data_parallel_ranks = config.dense_data_parallel_ranks
        self.expert_parallel_ranks = config.expert_parallel_ranks
        self.process_id = len(type(self).launches) + 100
        self.scheduler_type = "Scheduler"
        self.worker_type = "SimTpModelWorker"
        self.records = []
        self.results = []
        self.submissions = []
        self.remote_prefix_tokens = {}
        self._unfinished = {}
        self._output_targets = {}
        self._step_index = 0
        type(self).launches.append(config)
        type(self).instances.append(self)

    @property
    def has_unfinished_requests(self):
        return bool(self._unfinished)

    def submit(self, *, request_id, input_token_ids, max_new_tokens, **kwargs):
        del input_token_ids
        self.submissions.append(dict(kwargs))
        remote_prefix_tokens = kwargs.pop("remote_prefix_tokens", None)
        if kwargs:
            raise TypeError(f"unexpected submit arguments: {sorted(kwargs)}")
        if remote_prefix_tokens is not None:
            self.remote_prefix_tokens[request_id] = remote_prefix_tokens
        self._unfinished[request_id] = max_new_tokens
        self._output_targets[request_id] = max_new_tokens

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
                ScheduledRequest(
                    request_id,
                    phase,
                    1,
                    context_length=(
                        1 + self.remote_prefix_tokens.get(request_id, 0)
                    ),
                )
                for request_id in request_ids
            ],
            num_sampled=len(sampled),
            sampled_request_ids=sampled,
        )
        self._step_index += 1
        self.records.append(record)
        result = StepResult(
            step_index=record.step_index,
            step_latency_ps=10,
            completed_at_ps=now_ps + 10,
        )
        self.results.append(result)
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
                        output_token_count=self._output_targets.pop(request_id),
                    )
                )
        return {
            "completed_at_ps": now_ps + 10,
            "record": record,
            "result": result,
            "completions": tuple(completions),
            "token_id": 512,
        }

    def close(self):
        return None

    def pricing_provenance(self):
        return self.config.provider.pricing_provenance()


class _ProvenanceProvider(ComputeProvider):
    def __init__(self, label):
        self.label = label

    def estimate(self, kernel: KernelSpec, gpu):
        del kernel, gpu
        return DurationEstimate(1, "test")

    def pricing_provenance(self):
        return {"label": self.label}


class _ShapePricingFakePoolEngine(_FakePoolEngine):
    """Fake stock scheduler that sends its authored shape to the provider."""

    kernels: ClassVar[list[KernelSpec]] = []

    def step(self, now_ps):
        response = super().step(now_ps)
        if self.role.value == "decode":
            record = response["record"]
            kernel = step_kernel(self.config.dims, record, record.num_sampled)
            type(self).kernels.append(kernel)
            self.config.provider.estimate(kernel, self.config.gpu)
        return response


@pytest.fixture
def fake_pool_engines(monkeypatch):
    _FakePoolEngine.launches = []
    _FakePoolEngine.instances = []
    monkeypatch.setattr(pd_session, "_ProcessPoolEngine", _FakePoolEngine)
    return _FakePoolEngine


@pytest.fixture
def shape_pricing_fake_pool_engines(monkeypatch):
    _ShapePricingFakePoolEngine.launches = []
    _ShapePricingFakePoolEngine.instances = []
    _ShapePricingFakePoolEngine.kernels = []
    monkeypatch.setattr(
        pd_session,
        "_ProcessPoolEngine",
        _ShapePricingFakePoolEngine,
    )
    return _ShapePricingFakePoolEngine


def _expectations():
    return json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))


def _stable_projection(row):
    value = row.to_json()
    stable = {
        name: value[name]
        for name in (
            "request_id",
            "admitted_at_ps",
            "prefill_eligible_at_ps",
            "prefill_completed_at_ps",
            "handoff",
            "decode_eligible_at_ps",
            "decode_token_completed_at_ps",
            "prefill_engine_id",
            "decode_engine_id",
            "bootstrap_token_id",
            "decode_token_ids",
            "prefill_step_count",
            "decode_step_count",
        )
    }
    join = dict(value["join_metadata"])
    join.pop("prefill_process_id")
    join.pop("decode_process_id")
    stable["join_metadata"] = join
    if "compute_pricing" in value:
        stable["compute_pricing"] = value["compute_pricing"]
    return stable


def _disabled_byte_projection(requests, result):
    rows = []
    for request, row in zip(requests, result.requests, strict=True):
        timeline = row.timeline
        rows.append(
            {
                "request_id": request.request_id,
                "prompt_token_ids": list(request.prompt_token_ids),
                "handoff": {
                    "request_id": timeline.handoff.request_id,
                    "authority": timeline.handoff.authority,
                    "pricing_arm": timeline.handoff.pricing_arm,
                    "kv_bytes": timeline.handoff.kv_bytes,
                    "submitted_at_ps": timeline.handoff.submitted_at_ps,
                    "eligible_at_ps": timeline.handoff.eligible_at_ps,
                    "started_at_ps": timeline.handoff.started_at_ps,
                    "finished_at_ps": timeline.handoff.finished_at_ps,
                    "completed_at_ps": timeline.handoff.completed_at_ps,
                },
                "bootstrap_token_id": row.bootstrap_token_id,
                "decode_token_ids": list(row.decode_token_ids),
                "timestamps": {
                    "admitted_at_ps": timeline.admitted_at_ps,
                    "prefill_eligible_at_ps": timeline.prefill_eligible_at_ps,
                    "prefill_completed_at_ps": timeline.prefill_completed_at_ps,
                    "decode_eligible_at_ps": timeline.decode_eligible_at_ps,
                    "decode_token_completed_at_ps": list(
                        timeline.decode_token_completed_at_ps
                    ),
                },
            }
        )
    return json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()


def _run_selected_shape(tmp_path):
    expected = _expectations()["selected_key_acceptance"]
    provider = compile_session_profile_provider(
        CANDIDATE_PATH.read_bytes(),
        expected_sha256=expected["candidate_record_sha256"],
        pool="decode",
        selection_entry_index=expected["candidate_entry_index"],
    )
    base = _config(tmp_path, project_remote_kv_length=True)
    config = SglangPdSessionConfig(
        **{
            **base.__dict__,
            "decode_provider": provider,
            "max_running_requests": 64,
        }
    )
    requests = tuple(
        SglangPdRequest(
            f"standard-decode-{index:02d}",
            tuple(1_000 + (index + token) % 97 for token in range(2_000)),
            1,
            0,
        )
        for index in range(32)
    )
    with SglangDisaggregatedSession(config) as session:
        result = session.run_requests(requests)
    return result


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
    assert manifests.placement.group_ranks(0, "tp") == [0]
    assert manifests.placement.group_ranks(0, "dense_dp") == list(range(32))
    assert manifests.placement.group_ranks(0, "ep") == list(range(32))
    assert manifests.placement.group_ranks(32, "attn_dp") == list(range(32, 104))
    assert manifests.placement.group_ranks(32, "tp") == [32]
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


def test_remote_kv_shape_flag_requires_a_boolean(tmp_path):
    base = _config(tmp_path)

    with pytest.raises(TypeError, match="must be a boolean"):
        SglangPdSessionConfig(
            **{
                **base.__dict__,
                "project_remote_kv_length": 1,
            }
        )


def test_remote_kv_shape_selects_exact_candidate_once_and_is_stable(
    tmp_path,
    shape_pricing_fake_pool_engines,
):
    expected = _expectations()["selected_key_acceptance"]

    first = _run_selected_shape(tmp_path / "first")
    second = _run_selected_shape(tmp_path / "second")

    expected_batch = tuple(f"standard-decode-{index:02d}" for index in range(32))
    assert first.decode_batches == (expected_batch,)
    assert second.decode_batches == first.decode_batches
    assert first.maximum_decode_batch_size == expected["request_count"]
    assert len(shape_pricing_fake_pool_engines.kernels) == 2
    for kernel in shape_pricing_fake_pool_engines.kernels:
        assert len(kernel.request_shapes) == expected["request_count"]
        assert {
            shape.num_new_tokens for shape in kernel.request_shapes
        } == {expected["num_new_tokens_per_request"]}
        assert {
            shape.context_length for shape in kernel.request_shapes
        } == {expected["projected_context_length_per_request"]}
        assert {
            shape.prior_context_tokens for shape in kernel.request_shapes
        } == {expected["remote_prefix_tokens_per_request"]}

    for result in (first, second):
        pricing = result.requests[-1].compute_pricing["decode"]
        assert pricing["record_sha256"] == expected["candidate_record_sha256"]
        assert pricing["lookup_hits"] == expected["lookup_hits"]
        assert pricing["lookup_misses"] == expected["lookup_misses"]
        assert pricing["selected_entry_key_sha256"] == expected[
            "selected_entry_key_sha256"
        ]
        assert pricing["selected_entry_key_sha256s"] == expected[
            "selected_entry_key_sha256s"
        ]

    assert [_stable_projection(row) for row in first.requests] == [
        _stable_projection(row) for row in second.requests
    ]


def test_remote_kv_shape_disabled_preserves_frozen_session_bytes(
    tmp_path,
    fake_pool_engines,
):
    frozen = _expectations()["feature_disabled_byte_identity"]
    prompts = frozen["fixture_prompt_token_ids"]
    requests = tuple(
        SglangPdRequest(
            f"request-{index:02d}",
            tuple(prompt),
            frozen["fixture_decode_output_tokens"],
            0,
        )
        for index, prompt in enumerate(prompts)
    )

    with SglangDisaggregatedSession(_config(tmp_path)) as session:
        result = session.run_requests(requests)

    decode_engine = next(
        engine
        for engine in fake_pool_engines.instances
        if engine.role.value == "decode"
    )
    assert decode_engine.submissions == [{} for _ in requests]
    payload = _disabled_byte_projection(requests, result)
    assert hashlib.sha256(payload).hexdigest() == frozen["canonical_json_sha256"]


def test_session_selects_pool_specific_providers_and_surfaces_provenance(
    tmp_path,
    fake_pool_engines,
):
    config = _config(tmp_path)
    config = SglangPdSessionConfig(
        **{
            **config.__dict__,
            "prefill_provider": _ProvenanceProvider("prefill"),
            "decode_provider": _ProvenanceProvider("decode"),
        }
    )
    request = SglangPdRequest("request", (1, 2), 4, 0)

    with SglangDisaggregatedSession(config) as session:
        result = session.run_requests((request,)).requests[0]

    assert [launch.provider.label for launch in fake_pool_engines.launches] == [
        "prefill",
        "decode",
    ]
    assert result.compute_pricing == {
        "prefill": {"label": "prefill"},
        "decode": {"label": "decode"},
    }
    assert result.to_json()["compute_pricing"] == result.compute_pricing


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
    assert all(
        [step.step_latency_ps for step in row.decode_results] == [10, 10, 10, 10]
        for row in result.requests
    )
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
    assert fake_pool_engines.launches[0].tensor_parallel_ranks == (0,)
    assert fake_pool_engines.launches[1].tensor_parallel_ranks == (8,)


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
