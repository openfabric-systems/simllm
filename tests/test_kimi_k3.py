"""Independent structural, storage and causal guards for Kimi K3."""

from dataclasses import replace

import pytest

from simllm.backends.kimi_k3_lowerer import (
    KimiK3Lowerer,
    KimiK3LowererConfig,
    bind_synthetic_kimi_k3_graph,
    kimi_k3_step_shape,
)
from simllm.calibration.graph_identity import unbound_execution_graph_record
from simllm.compute.kimi_k3 import KdaSpec, KimiK3Spec, LatentMoeSpec, MlaSpec
from simllm.core import CoarseDeviceRuntime
from simllm.core.execution import ComputeWork, ExecutionGraph, ExecutionOperation
from simllm.core.execution_io import execution_graph_from_json, execution_graph_to_json
from simllm.core.step import RequestPhase, ScheduledRequest, StepRecord


def spec():
    return KimiK3Spec(
        hidden_size=7168, vocab_size=163840, dense_intermediate_size=33792,
        dense_prefix_layers=1,
        layer_types=tuple("mla" if (i + 1) % 4 == 0 or i == 92 else "kda" for i in range(93)),
        residual_block_size=12,
        kda=KdaSpec(96, 128, 4, True, "-5", "float32", "bfloat16"),
        mla=MlaSpec(96, 1536, 512, 128, 64, 128, True, True, "bfloat16"),
        moe=LatentMoeSpec(896, 16, 3584, 3072, 2, True, "sigmoid", "noaux_tc",
                         True, 1, 1, "1", "situ", "4", "25"),
        rms_norm_epsilon="1/100000", activation_dtype="bfloat16", tied_embeddings=False,
        max_context=1048576, checkpoint_quantization="mxfp4-e2m1-group32-e8m0",
    )


def step(*, prefill=True, tokens=1, prior=0, batch=1, index=0, released=0):
    return StepRecord(index, released, scheduled=[
        ScheduledRequest(f"request-{i}", RequestPhase.PREFILL if prefill else RequestPhase.DECODE,
                         tokens, context_length=prior + tokens) for i in range(batch)
    ], num_sampled=batch)


def lower(record, framework="vllm"):
    return KimiK3Lowerer(KimiK3LowererConfig(spec(), framework)).lower(record)


def depth(graph, service=lambda operation: 1):
    finished = {}
    for operation in graph.operations:
        finished[operation.operation_id] = max((finished[name] for name in operation.depends_on), default=0) + service(operation)
    return max(finished[name] for name in graph.completion_operation_ids)


def test_exact_heterogeneous_geometry_and_storage_have_separate_names():
    model = spec()
    assert KimiK3Spec.from_obj(model.to_obj()) == model
    assert len(model.layers) == 93
    assert model.layer_types.count("kda") == 69
    assert model.layer_types.count("mla") == 24
    assert model.layers[92].attention == "mla"
    assert sum(layer.feed_forward == "dense" for layer in model.layers) == 1
    assert [layer.index for layer in model.layers if layer.writes_residual_snapshot] == list(range(0, 85, 12))
    state = model.retained_state(sequences=1, cached_tokens=1, current_tokens=1)
    assert state == {"kda_recurrent_bytes": 434110464, "kda_convolution_history_bytes": 15261696,
                     "mla_cached_history_bytes": 27648, "mla_new_token_state_bytes": 27648,
                     "residual_snapshot_capacity_bytes": 114688, "residual_current_stream_bytes": 14336}
    assert model.moe.expert_parameters == 33030144
    assert model.moe.expert_encoding_floor_bytes == 17547264
    weights = model.weight_shapes()
    routed = [w for w in weights if w.name in {"moe.expert_gate", "moe.expert_up", "moe.expert_down"}]
    assert sum(w.parameters for w in routed) == 2722740830208
    shared = [w for w in weights if w.name in {"moe.shared_gate", "moe.shared_up", "moe.shared_down"}]
    assert 2 * sum(w.parameters for w in shared) == 24310185984
    assert sum(w.parameters for w in weights if w.name.startswith("dense.")) == 726663168
    assert all(w.to_obj()["scope"] == "logical-weight-shape" for w in weights)


@pytest.mark.parametrize("prefill,visits,critical,slow_shared", [(True, 3244, 1812, 3744000), (False, 3268, 1860, 3792000)])
def test_complete_original_graph_and_shared_branch_join(prefill, visits, critical, slow_shared):
    graph = lower(step(prefill=prefill))
    assert len(graph.operations) == visits
    assert depth(graph) == critical
    # Post-specified correction: the frozen arithmetic omitted 92 ordinary
    # shared/routed additions. Preserve those joins; the original study is void.
    assert depth(graph, lambda op: 10000 if op.work.kernel.startswith("kimi_k3.moe.shared_")
                 and op.work.kernel != "kimi_k3.moe.shared_routed_add" else 1000) == slow_shared
    assert {op.correlation.layer for op in graph.operations if op.correlation.layer is not None} == set(range(93))
    assert all(op.work.scope == "logical-operator" and op.work.hbm_bytes is None
               and op.work.nominal_duration_ps is None for op in graph.operations)
    assert sum(dict(op.work.config).get("source_vectors", 0) for op in graph.operations) == 1002
    assert execution_graph_from_json(execution_graph_to_json(graph)) == graph
    # Prefill's cache commit is a state frontier, not an extra attention edge.
    operations = {op.operation_id: op for op in graph.operations}
    caches = [op for op in graph.operations if op.work.kernel == "kimi_k3.mla.compressed_cache_commit"]
    assert len(caches) == 24
    assert all(op.operation_id in graph.completion_operation_ids for op in caches)
    scores = [op for op in graph.operations if op.work.kernel == "kimi_k3.mla.query_key_product"]
    for score in scores:
        parents = [operations[name].work.kernel for name in score.depends_on]
        assert ("kimi_k3.mla.compressed_cache_commit" in parents) is not prefill


@pytest.mark.parametrize("tokens,prior,pairs", [(1, 0, 1), (4, 0, 10), (16, 0, 136), (1, 16, 17), (1, 256, 257)])
def test_inclusive_causal_pairs_and_attention_algorithm_coefficients(tokens, prior, pairs):
    record = step(prefill=prior == 0, tokens=tokens, prior=prior, batch=3)
    assert kimi_k3_step_shape(record, spec()).causal_pairs == 3 * pairs
    graph = lower(record)
    score = next(op.work for op in graph.operations if op.work.kernel == "kimi_k3.mla.query_key_product")
    value = next(op.work for op in graph.operations if op.work.kernel == "kimi_k3.mla.probability_value_product")
    assert score.flops + value.flops == 3 * pairs * (61440 if prior == 0 else 208896)


def test_final_norm_token_axis_preserves_the_native_caller_difference():
    record = step(tokens=4, batch=3)
    for framework, expected in (("vllm", 3), ("sglang", 12)):
        graph = lower(record, framework)
        work = next(op.work for op in graph.operations if op.work.kernel == "kimi_k3.final_norm")
        assert dict(work.config)["tokens"] == expected


@pytest.mark.parametrize("field,value", [("head_dim", True), ("convolution_width", 0),
                                        ("gate_lower_bound", "-4"), ("full_rank_output_gate", False)])
def test_unsupported_geometry_rejects(field, value):
    with pytest.raises((TypeError, ValueError)):
        replace(spec().kda, **{field: value})


@pytest.mark.parametrize("record", [step(tokens=2, prior=1), step(prefill=False, tokens=2),
                                    step(prefill=False, prior=1048576)])
def test_undeclared_dispatch_and_context_reject(record):
    with pytest.raises(ValueError):
        lower(record)


def test_unbound_or_logical_demand_never_advances_runtime_even_with_duration():
    for work in (ComputeWork("logical", flops=None, hbm_bytes=None, scope="logical-operator"),
                 ComputeWork("logical", flops=0, hbm_bytes=0, nominal_duration_ps=1, scope="logical-operator")):
        graph = ExecutionGraph("invalid", 0, 0, (ExecutionOperation("op", 0, "queue", work),))
        runtime = CoarseDeviceRuntime()
        events = []
        with pytest.raises(ValueError, match="logical-operator"):
            runtime.execute(graph, on_event=events.append)
        assert runtime.last_report is None and events == []
        valid = replace(graph, operations=(replace(graph.operations[0], work=ComputeWork("valid", nominal_duration_ps=7)),))
        assert runtime.execute(valid).completed_at_ps == 7


def test_synthetic_binding_preserves_logical_topology_with_distinct_identity():
    original = lower(step())
    bound = bind_synthetic_kimi_k3_graph(original, 1000)
    assert bound.execution_id != original.execution_id
    assert unbound_execution_graph_record(bound).record_id != unbound_execution_graph_record(original).record_id
    assert bound.completion_operation_ids == original.completion_operation_ids
    for before, after in zip(original.operations, bound.operations, strict=True):
        assert (before.operation_id, before.logical_queue, before.depends_on) == (after.operation_id, after.logical_queue, after.depends_on)
        assert after.work.scope == "synthetic-operator"
        assert after.work.hbm_bytes == 0
        assert dict(after.work.config)["synthetic_memory_arbitration"] == "zero"


def test_serial_compute_selection_and_default_identity_are_explicit():
    graph = ExecutionGraph("two-branches", 0, 0, (
        ExecutionOperation("a", 0, "a", ComputeWork("a", nominal_duration_ps=7)),
        ExecutionOperation("b", 0, "b", ComputeWork("b", nominal_duration_ps=11)),
    ))
    baseline = CoarseDeviceRuntime().execute(graph)
    assert baseline == CoarseDeviceRuntime(serial_compute=False).execute(graph)
    assert baseline.completed_at_ps == 11
    serial = CoarseDeviceRuntime(serial_compute=True)
    assert serial.execute(graph).completed_at_ps == 18
    with pytest.raises(TypeError):
        CoarseDeviceRuntime(serial_compute=1)
