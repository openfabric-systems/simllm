"""Original-index Kimi K3 logical operations behind ExecutionLowerer.

The graph declares mathematical dependency and complete invocation shapes.
It does not reconstruct a GPU launch schedule, choose a device implementation
or price checkpoint bytes as memory traffic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from simllm.compute.kimi_k3 import KimiK3Spec
from simllm.core.execution import (
    ComputeWork,
    ExecutionGraph,
    ExecutionObservations,
    ExecutionOperation,
    OperationCorrelation,
)
from simllm.core.execution_io import execution_graph_to_json, validate_execution_graph
from simllm.core.step import RequestPhase, StepRecord


@dataclass(frozen=True, slots=True)
class KimiK3LowererConfig:
    spec: KimiK3Spec
    framework: str
    prefill_variant: str = "expanded-new-token"
    decode_variant: str = "absorbed-latent"

    def __post_init__(self) -> None:
        if not isinstance(self.spec, KimiK3Spec):
            raise TypeError("K3 lowering requires a typed heterogeneous specification")
        if self.framework not in {"vllm", "sglang"}:
            raise ValueError("K3 lowering requires a pinned native framework projection")
        if (self.prefill_variant, self.decode_variant) != ("expanded-new-token", "absorbed-latent"):
            raise ValueError("undeclared MLA algorithm dispatch")


@dataclass(frozen=True, slots=True)
class KimiK3StepShape:
    phase: str
    sequences: int
    new_tokens: int
    cached_tokens: int
    causal_pairs: int
    sampled_tokens: int


def kimi_k3_step_shape(record: StepRecord, spec: KimiK3Spec) -> KimiK3StepShape:
    if not isinstance(record, StepRecord) or not record.scheduled:
        raise ValueError("K3 structure requires a nonempty model step")
    if type(record.step_index) is not int or record.step_index < 0:
        raise ValueError("K3 step index must be a nonnegative integer")
    if type(record.virtual_time_ps) is not int or record.virtual_time_ps < 0:
        raise ValueError("K3 step time must be a nonnegative integer")
    phases = {request.phase for request in record.scheduled}
    if len(phases) != 1 or not phases.issubset({RequestPhase.PREFILL, RequestPhase.DECODE}):
        raise ValueError("mixed-phase K3 batches require a separately declared dispatch")
    phase = next(iter(phases))
    ids = [request.request_id for request in record.scheduled]
    if len(ids) != len(set(ids)) or any(not isinstance(name, str) or not name for name in ids):
        raise ValueError("K3 request identities must be unique nonblank strings")
    new = cached = pairs = 0
    for request in record.scheduled:
        n, context = request.num_new_tokens, request.context_length
        if type(n) is not int or n <= 0 or type(context) is not int or context < n:
            raise ValueError("K3 context must include every positive newly computed token")
        if context > spec.max_context:
            raise ValueError("K3 context exceeds the checkpoint envelope")
        if type(request.num_cached_tokens) is not int or request.num_cached_tokens != 0:
            raise ValueError("prefix-cache K3 dispatch is outside this envelope")
        prior = context - n
        if phase is RequestPhase.PREFILL and prior != 0:
            raise ValueError("K3 prefill requires an explicitly cold no-prefix batch")
        if phase is RequestPhase.DECODE and n != 1:
            raise ValueError("K3 speculative or multi-token decode requires a declared dispatch")
        new += n
        cached += prior
        pairs += n * prior + n * (n + 1) // 2
    sampled = record.num_sampled
    if type(sampled) is not int or sampled != len(ids):
        raise ValueError("K3 structure requires exactly one sampled row per request")
    if record.num_tokens_after_padding is not None and (
        type(record.num_tokens_after_padding) is not int or record.num_tokens_after_padding != new
    ):
        raise ValueError("K3 padded token work requires a separate dispatch envelope")
    if record.sampled_request_ids is not None and tuple(record.sampled_request_ids) != tuple(ids):
        raise ValueError("K3 sampled rows do not match the complete ordered request set")
    return KimiK3StepShape(phase.value, len(ids), new, cached, pairs, sampled)


class _GraphBuilder:
    def __init__(self, record: StepRecord, config: KimiK3LowererConfig):
        self.record = record
        self.config = config
        self.spec = config.spec
        self.shape = kimi_k3_step_shape(record, config.spec)
        self.operations: list[ExecutionOperation] = []
        self.state_commits: list[str] = []
        self.execution_id = f"kimi-k3-step-{record.step_index}"

    def region(self, name: str, parents: tuple[str, ...], *, layer: int | None = None,
               flops: int | None = None, tokens: int | None = None,
               **axes: int | str | bool) -> str:
        tokens = self.shape.new_tokens if tokens is None else tokens
        site = "outer" if layer is None else f"layer-{layer:03d}"
        identity = f"{self.execution_id}:{site}:{name}"
        values = {"tokens": tokens, **axes}
        work = ComputeWork(
            kernel=f"kimi_k3.{name}",
            config=tuple(sorted(values.items())),
            flops=flops,
            hbm_bytes=None,
            scope="logical-operator",
        )
        self.operations.append(ExecutionOperation(
            identity, 0, f"logical:{site}:{name}", work, depends_on=parents,
            correlation=OperationCorrelation(
                request_ids=tuple(request.request_id for request in self.record.scheduled),
                layer=layer,
            ),
        ))
        return identity

    def matrix(self, name: str, parent: str, inputs: int, outputs: int, *, layer: int | None = None,
               tokens: int | None = None, groups: int = 1) -> str:
        n = self.shape.new_tokens if tokens is None else tokens
        return self.region(name, (parent,), layer=layer, tokens=n,
                           input_width=inputs, output_width=outputs, groups=groups,
                           flops=2 * n * inputs * outputs * groups)

    def kda(self, parent: str, layer: int) -> str:
        spec, h = self.spec.kda, self.spec.hidden_size
        width = spec.projection_width
        projected = {name: self.matrix(f"kda.{name}_projection", parent, h, width, layer=layer)
                     for name in ("q", "k", "v")}
        convolved = {name: self.region(f"kda.{name}_convolution_silu", (projected[name],),
                                      layer=layer, channels=width, history=spec.convolution_width - 1,
                                      convolution_width=spec.convolution_width,
                                      sequences=self.shape.sequences)
                     for name in ("q", "k", "v")}
        normalized = {name: self.region(f"kda.{name}_l2_norm", (convolved[name],), layer=layer,
                                       heads=spec.heads, head_dim=spec.head_dim, epsilon="1/1000000")
                      for name in ("q", "k")}
        forget_a = self.matrix("kda.forget_a_projection", parent, h, spec.head_dim, layer=layer)
        forget_b = self.matrix("kda.forget_b_projection", forget_a, spec.head_dim, width, layer=layer)
        forget = self.region("kda.forget_gate", (forget_b,), layer=layer, heads=spec.heads,
                             head_dim=spec.head_dim, lower_bound=spec.gate_lower_bound,
                             formula="bound*sigmoid(exp(A_log)*(raw+dt_bias))")
        beta = self.matrix("kda.beta_projection", parent, h, spec.heads, layer=layer)
        beta_gate = self.region("kda.beta_sigmoid", (beta,), layer=layer, heads=spec.heads)
        output_gate = self.matrix("kda.output_gate_projection", parent, h, width, layer=layer)
        update = self.region("kda.state_update_and_query",
                             (normalized["q"], normalized["k"], convolved["v"], forget, beta_gate),
                             layer=layer, heads=spec.heads, head_dim=spec.head_dim,
                             sequences=self.shape.sequences, recurrent_dtype=spec.recurrent_dtype,
                             algorithm="delta-recurrence-semantics")
        gated = self.region("kda.gated_rms_norm", (update, output_gate), layer=layer,
                            heads=spec.heads, head_dim=spec.head_dim, gate="sigmoid")
        return self.matrix("kda.output_projection", gated, width, h, layer=layer)

    def mla(self, parent: str, layer: int) -> str:
        spec, h, shape = self.spec.mla, self.spec.hidden_size, self.shape
        q_a = self.matrix("mla.q_a_projection", parent, h, spec.query_rank, layer=layer)
        q_norm = self.region("mla.q_norm", (q_a,), layer=layer, width=spec.query_rank)
        q_b = self.matrix("mla.q_b_projection", q_norm, spec.query_rank,
                          spec.heads * spec.expanded_score_dim, layer=layer)
        kv_a = self.matrix("mla.kv_a_projection", parent, h, spec.compressed_width, layer=layer)
        kv_norm = self.region("mla.kv_norm", (kv_a,), layer=layer, width=spec.kv_rank,
                              extra_width=spec.score_extra_dim)
        cache = self.region("mla.compressed_cache_commit", (kv_a, kv_norm), layer=layer,
                            cached_tokens=shape.cached_tokens, width=spec.compressed_width,
                            cache_dtype=spec.cache_dtype, flops=0)
        self.state_commits.append(cache)
        if shape.phase == "prefill":
            kv_b = self.matrix("mla.kv_b_projection", kv_norm, spec.kv_rank,
                               spec.heads * (spec.score_nope_dim + spec.value_dim), layer=layer)
            score_parents = (q_b, kv_b)
            score_width, value_width = spec.expanded_score_dim, spec.value_dim
            variant = self.config.prefill_variant
        else:
            absorbed = self.matrix("mla.query_absorption", q_b, spec.score_nope_dim,
                                   spec.kv_rank, layer=layer, groups=spec.heads)
            score_parents = (absorbed, cache)
            score_width, value_width = spec.compressed_width, spec.kv_rank
            variant = self.config.decode_variant
        score = self.region("mla.query_key_product", score_parents, layer=layer,
                            heads=spec.heads, width=score_width, pairs=shape.causal_pairs,
                            cached_tokens=shape.cached_tokens, variant=variant,
                            flops=2 * shape.causal_pairs * spec.heads * score_width)
        softmax = self.region("mla.softmax", (score,), layer=layer, heads=spec.heads,
                              pairs=shape.causal_pairs, variant=variant)
        value = self.region("mla.probability_value_product", (softmax,), layer=layer,
                            heads=spec.heads, width=value_width, pairs=shape.causal_pairs,
                            variant=variant, flops=2 * shape.causal_pairs * spec.heads * value_width)
        if shape.phase == "decode":
            value = self.matrix("mla.value_reexpansion", value, spec.kv_rank, spec.value_dim,
                                layer=layer, groups=spec.heads)
        gate = self.matrix("mla.output_gate_projection", parent, h, spec.heads * spec.value_dim, layer=layer)
        gated = self.region("mla.sigmoid_multiply", (value, gate), layer=layer,
                            heads=spec.heads, width=spec.value_dim)
        return self.matrix("mla.output_projection", gated, spec.heads * spec.value_dim, h, layer=layer)

    def dense(self, parent: str, layer: int) -> str:
        h, width = self.spec.hidden_size, self.spec.dense_intermediate_size
        gate = self.matrix("dense.gate_projection", parent, h, width, layer=layer)
        up = self.matrix("dense.up_projection", parent, h, width, layer=layer)
        active = self.region("dense.situ", (gate, up), layer=layer, width=width,
                             beta=self.spec.moe.situ_beta, linear_beta=self.spec.moe.situ_linear_beta)
        return self.matrix("dense.down_projection", active, width, h, layer=layer)

    def moe(self, parent: str, layer: int) -> str:
        spec, h = self.spec.moe, self.spec.hidden_size
        e, width, visits = spec.latent_width, spec.intermediate_size, self.shape.new_tokens * spec.selected_experts
        router = self.matrix("moe.router_projection", parent, h, spec.experts, layer=layer)
        select = self.region("moe.expert_selection", (router,), layer=layer,
                             resident_experts=spec.experts, selected_experts=spec.selected_experts,
                             correction_bias=True, renormalize=spec.renormalize,
                             activation=spec.router_activation, topk_method=spec.topk_method)
        latent = self.matrix("moe.latent_down_projection", parent, h, e, layer=layer)
        dispatch = self.region("moe.dispatch", (select, latent), layer=layer, latent_width=e,
                               selected_experts=spec.selected_experts, expert_token_visits=visits,
                               routing_scope="logical-selections-without-rank-placement")
        gate = self.matrix("moe.expert_gate_projection", dispatch, e, width, layer=layer, tokens=visits)
        up = self.matrix("moe.expert_up_projection", dispatch, e, width, layer=layer, tokens=visits)
        active = self.region("moe.expert_situ", (gate, up), layer=layer, tokens=visits, width=width,
                             beta=spec.situ_beta, linear_beta=spec.situ_linear_beta)
        down = self.matrix("moe.expert_down_projection", active, width, e, layer=layer, tokens=visits)
        combined = self.region("moe.weighted_combine", (down, select), layer=layer,
                               selected_experts=spec.selected_experts, width=e)
        normalized = self.region("moe.latent_rms_norm", (combined,), layer=layer, width=e)
        routed = self.matrix("moe.latent_up_projection", normalized, e, h, layer=layer)
        shared_width = width * spec.shared_experts
        shared_gate = self.matrix("moe.shared_gate_projection", parent, h, shared_width, layer=layer)
        shared_up = self.matrix("moe.shared_up_projection", parent, h, shared_width, layer=layer)
        shared_active = self.region("moe.shared_situ", (shared_gate, shared_up), layer=layer,
                                    width=shared_width, beta=spec.situ_beta, linear_beta=spec.situ_linear_beta)
        shared = self.matrix("moe.shared_down_projection", shared_active, shared_width, h, layer=layer)
        return self.region("moe.shared_routed_add", (routed, shared), layer=layer, width=h)

    def build(self) -> ExecutionGraph:
        spec, shape = self.spec, self.shape
        parent = self.region("embedding", (), width=spec.hidden_size, vocabulary=spec.vocab_size, flops=0)
        for layer in spec.layers:
            before = layer.residual_bank_before
            sources = 0 if layer.index == 0 else before + 1
            mixed = self.region(f"residual.attention_sources_{sources}", (parent,), layer=layer.index,
                                width=spec.hidden_size, bank_before=before, source_vectors=sources,
                                snapshot=layer.writes_residual_snapshot, rms_epsilon=spec.rms_norm_epsilon)
            attention = self.kda(mixed, layer.index) if layer.attention == "kda" else self.mla(mixed, layer.index)
            bank = before + int(layer.writes_residual_snapshot)
            post = self.region(f"residual.mlp_sources_{bank + 1}", (attention, mixed), layer=layer.index,
                               width=spec.hidden_size, bank_after=bank, source_vectors=bank + 1,
                               reset_prefix=layer.writes_residual_snapshot, rms_epsilon=spec.rms_norm_epsilon)
            feed_forward = self.dense(post, layer.index) if layer.feed_forward == "dense" else self.moe(post, layer.index)
            parent = self.region("residual.ffn_delta_prefix_add", (feed_forward, post),
                                 layer=layer.index, width=spec.hidden_size)
        bank = (spec.num_layers + spec.residual_block_size - 1) // spec.residual_block_size
        mixed = self.region("final_residual_mixture", (parent,), width=spec.hidden_size, source_vectors=bank + 1)
        norm_tokens = shape.sampled_tokens if self.config.framework == "vllm" else shape.new_tokens
        norm = self.region("final_norm", (mixed,), width=spec.hidden_size, tokens=norm_tokens,
                           token_scope="sampled-rows" if self.config.framework == "vllm" else "forward-rows")
        head = self.matrix("lm_head", norm, spec.hidden_size, spec.vocab_size, tokens=shape.sampled_tokens)
        graph = ExecutionGraph(self.execution_id, self.record.step_index, self.record.virtual_time_ps,
                               operations=tuple(self.operations),
                               completion_operation_ids=(head, *self.state_commits))
        validate_execution_graph(graph)
        return graph


class KimiK3Lowerer:
    """Lower an explicit unsharded K3 logical plan, without uniform ModelDims."""

    def __init__(self, config: KimiK3LowererConfig):
        if not isinstance(config, KimiK3LowererConfig):
            raise TypeError("K3 lowerer requires KimiK3LowererConfig")
        self.config = config

    def lower(self, record: StepRecord, observations: ExecutionObservations | None = None) -> ExecutionGraph:
        if observations is not None:
            raise ValueError("native physical observations require their own total implementation binding")
        return _GraphBuilder(record, self.config).build()


def bind_synthetic_kimi_k3_graph(graph: ExecutionGraph, service_ps_per_token: int) -> ExecutionGraph:
    """Return distinct diagnostic work, retaining all logical edges and queues."""
    from simllm.calibration.canonical import canonical_sha256

    validate_execution_graph(graph)
    if type(service_ps_per_token) is not int or service_ps_per_token <= 0:
        raise ValueError("synthetic service must be a positive integer")
    source = canonical_sha256(execution_graph_to_json(graph))
    # Every region gets the same per-step token scale. Expert-token visits and
    # sampled rows remain logical axes, not multipliers of this diagnostic.
    embeddings = [op for op in graph.operations if isinstance(op.work, ComputeWork)
                  and op.work.kernel == "kimi_k3.embedding"]
    if len(embeddings) != 1:
        raise ValueError("synthetic K3 binding requires one total embedding region")
    tokens = dict(embeddings[0].work.config)["tokens"]
    operations = []
    for operation in graph.operations:
        work = operation.work
        if not isinstance(work, ComputeWork) or work.scope != "logical-operator":
            raise ValueError("synthetic K3 binding requires every region to be unbound logical work")
        config = dict(work.config)
        config.update(binding="uniform-logical-service-v1", source_graph_sha256=source,
                      synthetic_memory_arbitration="zero", service_ps_per_token=service_ps_per_token)
        operations.append(replace(operation, work=replace(
            work, config=tuple(sorted(config.items())), flops=0, hbm_bytes=0,
            nominal_duration_ps=tokens * service_ps_per_token, scope="synthetic-operator",
        )))
    bound = replace(graph, execution_id=f"{graph.execution_id}:synthetic:{service_ps_per_token}",
                    operations=tuple(operations))
    validate_execution_graph(bound)
    return bound
