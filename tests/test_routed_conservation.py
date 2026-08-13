"""Independent routed-MoE byte conservation and vLLM MoE geometry."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from simllm.compute import ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.traffic import (
    ALWAYS_APPLIED_RULES,
    CAPTURED_ONLY_RULES,
    OBSERVED_ABSENT_BYTE_EVIDENCE,
    OBSERVED_NO_BYTE_EVIDENCE,
    OBSERVED_PAIR_TABLE_EVIDENCE,
    ROUTED_EVIDENCE_CAPTURED,
    ROUTED_EVIDENCE_UNIFORM,
    RoutedPhaseTable,
    RoutedTokenOwnership,
    observed_routed_byte_evidence,
    routed_moe_conservation,
    routed_moe_conservation_report,
)

VECTOR_BYTES = 8


def _ownership(tokens: int = 4, *, top_k: int = 4, layers: int = 2):
    return RoutedTokenOwnership(
        engine_rank=0,
        request_token_counts=(("alpha", tokens),),
        num_layers=layers,
        top_k=top_k,
        vector_bytes=VECTOR_BYTES,
    )


def _layer_tables(rows, layer=0):
    dispatch = tuple(sorted(rows))
    combine = tuple(
        sorted((destination, source, size) for source, destination, size in rows)
    )
    return (
        RoutedPhaseTable(layer=layer, phase="dispatch", pair_payload_bytes=dispatch),
        RoutedPhaseTable(layer=layer, phase="combine", pair_payload_bytes=combine),
    )


def test_owner_attributed_tables_conserve_and_report_both_endpoints():
    tables = _layer_tables(((0, 1, 16), (0, 2, 8)))
    report = routed_moe_conservation_report(
        tables,
        _ownership(),
        (0, 1, 2),
        evidence_mode=ROUTED_EVIDENCE_CAPTURED,
    )
    assert report.conserved
    assert report.checked_rules == ALWAYS_APPLIED_RULES + CAPTURED_ONLY_RULES
    assert report.total_directed_bytes == 48
    assert report.owner_egress_bytes == 24
    assert report.owner_ingress_bytes == 24
    assert report.emitted_hops == 6
    # T * top_k * layers * 2 hidden-vector hops
    assert report.step_hop_bound == 4 * 4 * 2 * 2
    report.require_conserved()


def test_source_replication_is_detected_by_both_owner_rules():
    tables = _layer_tables(((0, 1, 8), (0, 2, 8)))
    replicated = _layer_tables(
        ((0, 1, 8), (0, 2, 8), (1, 0, 8), (1, 2, 8), (2, 0, 8), (2, 1, 8))
    )
    assert routed_moe_conservation_report(
        tables, _ownership(), (0, 1, 2), evidence_mode=ROUTED_EVIDENCE_CAPTURED
    ).conserved
    report = routed_moe_conservation_report(
        replicated,
        _ownership(),
        (0, 1, 2),
        evidence_mode=ROUTED_EVIDENCE_CAPTURED,
    )
    assert "source-attribution" in report.violations
    assert "owner-egress" in report.violations
    with pytest.raises(ValueError, match="conservation failed"):
        report.require_conserved()


def test_step_hop_bound_fires_only_when_the_world_leaves_no_slack():
    # top_k 4 with a 2-rank world: one remote owner per token-layer at most, so
    # a doubling still fits inside T * top_k * layers * 2
    narrow = _layer_tables(((0, 1, 4 * VECTOR_BYTES),))
    doubled = _layer_tables(((0, 1, 8 * VECTOR_BYTES),))
    assert "step-hop-bound" not in routed_moe_conservation_report(
        doubled, _ownership(), (0, 1), evidence_mode=ROUTED_EVIDENCE_UNIFORM
    ).violations
    assert routed_moe_conservation_report(
        narrow, _ownership(), (0, 1), evidence_mode=ROUTED_EVIDENCE_UNIFORM
    ).conserved
    # the same doubling in a wide world clears the bound
    wide = _layer_tables(
        tuple((0, peer, 30 * VECTOR_BYTES) for peer in range(1, 8))
    )
    assert "step-hop-bound" in routed_moe_conservation_report(
        wide, _ownership(), tuple(range(8)), evidence_mode=ROUTED_EVIDENCE_UNIFORM
    ).violations


def test_captured_only_rules_are_not_applied_to_the_uniform_approximation():
    # top_k 4 over a 2-rank world exceeds min(top_k, W - 1) per token by
    # construction, because the uniform model never merges experts that share
    # a destination
    tables = _layer_tables(((0, 1, 8 * VECTOR_BYTES),))
    uniform = routed_moe_conservation_report(
        tables, _ownership(), (0, 1), evidence_mode=ROUTED_EVIDENCE_UNIFORM
    )
    assert uniform.conserved
    captured = routed_moe_conservation_report(
        tables, _ownership(), (0, 1), evidence_mode=ROUTED_EVIDENCE_CAPTURED
    )
    assert "per-layer-hop-bound" in captured.violations


def test_vector_granularity_and_request_identity_need_captured_routing():
    tables = (
        RoutedPhaseTable(
            layer=0,
            phase="dispatch",
            pair_payload_bytes=((0, 1, 7),),
            request_pair_payload_bytes=(("ghost", 0, 1, 7),),
        ),
        RoutedPhaseTable(
            layer=0,
            phase="combine",
            pair_payload_bytes=((1, 0, 7),),
            request_pair_payload_bytes=(("ghost", 1, 0, 7),),
        ),
    )
    report = routed_moe_conservation_report(
        tables, _ownership(), (0, 1), evidence_mode=ROUTED_EVIDENCE_CAPTURED
    )
    assert "vector-granularity" in report.violations
    assert "request-identity" in report.violations


def test_transpose_symmetry_needs_both_phases():
    only_dispatch = (
        RoutedPhaseTable(layer=0, phase="dispatch", pair_payload_bytes=((0, 1, 8),)),
    )
    report = routed_moe_conservation_report(
        only_dispatch, _ownership(), (0, 1), evidence_mode=ROUTED_EVIDENCE_CAPTURED
    )
    assert "transpose-symmetry" in report.violations


def test_conservation_entry_point_skips_a_step_without_moe_traffic():
    dims = ModelDims(
        num_layers=2,
        hidden_size=4,
        intermediate_size=8,
        num_heads=2,
        num_kv_heads=2,
        head_size=2,
        vocab_size=16,
    )
    record = StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[ScheduledRequest("alpha", RequestPhase.DECODE, 1, context_length=4)],
        num_sampled=1,
    )
    assert routed_moe_conservation(record, dims, (0, 1), None) is None
    assert routed_moe_conservation(record, dims, None, None) is None


def test_observed_routed_byte_evidence_names_the_marker_path():
    from simllm.core import (
        CollectiveWork,
        ExecutionObservations,
        ExecutionOperation,
        OperationCorrelation,
    )

    def _operation(operation_id: str, **work_fields):
        return ExecutionOperation(
            operation_id=operation_id,
            rank=0,
            logical_queue="cuda:0:comm:ep",
            work=CollectiveWork(
                collective="all-to-allv",
                ranks=(0, 1),
                channel_hint="dispatch",
                **work_fields,
            ),
            correlation=OperationCorrelation(request_ids=("alpha",), layer=0),
        )

    empty = ExecutionObservations(operations=(), completion_operation_ids=())
    assert observed_routed_byte_evidence(empty) == OBSERVED_ABSENT_BYTE_EVIDENCE

    marker = ExecutionObservations(
        operations=(_operation("marker", payload_bytes=0),),
        completion_operation_ids=(),
    )
    assert observed_routed_byte_evidence(marker) == OBSERVED_NO_BYTE_EVIDENCE

    with_table = ExecutionObservations(
        operations=(
            _operation("table", payload_bytes=0, pair_payload_bytes=((0, 1, 8),)),
        ),
        completion_operation_ids=(),
    )
    assert observed_routed_byte_evidence(with_table) == OBSERVED_PAIR_TABLE_EVIDENCE


# --- vLLM MoE geometry (VLLM-6) ---------------------------------------------


def _model_config(num_local_experts):
    text = SimpleNamespace(
        model_type="granitemoe" if num_local_experts else "llama",
        architectures=["GraniteMoeForCausalLM"] if num_local_experts else ["Llama"],
        intermediate_size=512,
    )
    if num_local_experts is not None:
        text.num_local_experts = num_local_experts
        text.num_experts_per_tok = 8

    class _ModelConfig:
        dtype = SimpleNamespace(itemsize=2)
        hf_text_config = text
        is_moe = bool(num_local_experts)

        @staticmethod
        def get_hidden_size():
            return 1024

        @staticmethod
        def get_num_layers(parallel_config):
            return 24

        @staticmethod
        def get_num_attention_heads(parallel_config):
            return 16

        @staticmethod
        def get_num_kv_heads(parallel_config):
            return 8

        @staticmethod
        def get_head_size():
            return 64

        @staticmethod
        def get_vocab_size():
            return 49_155

    return _ModelConfig()


def _vllm_config(
    *,
    dp=1,
    pcp=1,
    tp=1,
    pp=1,
    rank=0,
    enable_ep=False,
    experts=32,
    redundant=0,
):
    return SimpleNamespace(
        model_config=_model_config(experts),
        cache_config=SimpleNamespace(block_size=16, cache_dtype="auto"),
        parallel_config=SimpleNamespace(
            data_parallel_size=dp,
            prefill_context_parallel_size=pcp,
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            rank=rank,
            enable_expert_parallel=enable_ep,
            eplb_config=SimpleNamespace(num_redundant_experts=redundant),
        ),
        quant_config=None,
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, (32, 8, 512, 32)),
        ({"tp": 2, "rank": 1}, (32, 8, 256, 32)),
        ({"dp": 2, "tp": 2, "rank": 3}, (32, 8, 128, 32)),
        ({"enable_ep": True}, (32, 8, 512, 32)),
        ({"dp": 8, "rank": 5, "enable_ep": True}, (32, 8, 512, 4)),
        ({"dp": 2, "tp": 2, "rank": 3, "enable_ep": True}, (32, 8, 512, 8)),
        ({"dp": 8, "rank": 0, "enable_ep": True, "experts": 30}, (30, 8, 512, 4)),
        ({"dp": 8, "rank": 7, "enable_ep": True, "experts": 30}, (30, 8, 512, 3)),
        ({"dp": 2, "rank": 1, "enable_ep": True, "redundant": 2}, (34, 8, 512, 17)),
        ({"experts": None}, (0, 0, None, 0)),
        ({"experts": None, "tp": 2, "rank": 1}, (0, 0, None, 0)),
    ],
)
def test_moe_geometry_matches_the_pinned_vllm_mapping(kwargs, expected):
    from simllm.adapters.vllm.executor import model_dims_from_vllm_config

    dims = model_dims_from_vllm_config(_vllm_config(**kwargs))
    observed = (
        dims.num_experts,
        dims.top_k,
        dims.moe_intermediate_size,
        dims.local_num_experts,
    )
    assert observed == expected


@pytest.mark.parametrize(
    ("kwargs", "expected_ranks", "expected_index"),
    [
        ({"dp": 2, "tp": 2, "rank": 3}, (0, 1, 2, 3), 3),
        ({"dp": 2, "pp": 2, "tp": 2, "rank": 6}, (2, 3, 6, 7), 2),
        ({"dp": 2, "pp": 2, "tp": 2, "rank": 1}, (0, 1, 4, 5), 1),
        ({"dp": 2, "pcp": 2, "tp": 2, "rank": 5}, tuple(range(8)), 5),
        ({"dp": 2, "tp": 2, "rank": 5}, (4, 5, 6, 7), 1),
    ],
)
def test_expert_group_follows_the_vllm_rank_layout(
    kwargs, expected_ranks, expected_index
):
    from simllm.adapters.vllm.executor import expert_parallel_geometry

    geometry = expert_parallel_geometry(_vllm_config(enable_ep=True, **kwargs))
    assert geometry.ep_ranks == expected_ranks
    assert geometry.ep_rank == expected_index


def test_dense_model_reports_no_expert_group_and_moe_model_does():
    from simllm.adapters.vllm.executor import expert_group_ranks

    assert expert_group_ranks(_vllm_config(experts=None, tp=2, rank=1)) is None
    assert expert_group_ranks(_vllm_config(dp=8, rank=5, enable_ep=True)) == tuple(
        range(8)
    )


def test_expert_world_wider_than_the_expert_count_is_refused():
    from simllm.adapters.vllm.executor import model_dims_from_vllm_config

    with pytest.raises(ValueError, match="exceeds"):
        model_dims_from_vllm_config(
            _vllm_config(dp=8, rank=0, enable_ep=True, experts=4)
        )


def test_device_sink_binds_and_rejects_a_disagreeing_expert_group():
    from simllm.backends import DeviceRuntimeStepSink, SerialStepLowererConfig

    dims = ModelDims(
        num_layers=1,
        hidden_size=4,
        intermediate_size=8,
        num_heads=2,
        num_kv_heads=2,
        head_size=2,
        vocab_size=16,
    )
    config = SerialStepLowererConfig(dims=dims, tp_ranks=(0,))
    sink = DeviceRuntimeStepSink(config)
    sink.bind_expert_group((0, 1, 2, 3))
    assert sink.lowerer.config.ep_ranks == (0, 1, 2, 3)
    sink.bind_expert_group((0, 1, 2, 3))
    with pytest.raises(RuntimeError, match="disagrees"):
        sink.bind_expert_group((0, 1))
