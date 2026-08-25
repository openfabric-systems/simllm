"""SGLang adapter tests. Nothing here imports SGLang: the batch observation,
translation, geometry and config logic are plain Python that runs (and is
checked) without a GPU stack installed."""

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from simllm.adapters.sglang import (
    BatchRow,
    SglReplayTokenSource,
    SglStepTranslator,
    SimWorkerConfig,
    configure,
    model_dims_from_sglang,
    observe_schedule_batch,
    reset_configuration,
)
from simllm.adapters.sglang.plugin import ENABLE_ENV, enabled
from simllm.adapters.sglang.replay import sample_adapter_tokens
from simllm.adapters.sglang.worker import _sglang_moe_parallel_sizes
from simllm.core import RequestPhase, StepRecord, sampled_request_ids, step_record_to_json
from simllm.preplay import (
    ForwardPhase,
    RequestArrival,
    join_preplay_arrivals,
    read_preplay_trace,
    write_preplay_replay_run,
    write_preplay_trace,
)

SGLANG_INSTALLED = importlib.util.find_spec("sglang") is not None
GOLDEN_TRACE = Path(__file__).parents[1] / "examples/preplay_trace_v1/writer_golden.jsonl"


# ScheduleBatch stubs: only the source-backed attribute names, nothing else


class FakeForwardMode:
    def __init__(self, mode: str) -> None:
        self._mode = mode

    def is_extend(self) -> bool:
        return self._mode in ("extend", "mixed")

    def is_decode(self) -> bool:
        return self._mode == "decode"

    def is_idle(self) -> bool:
        return self._mode == "idle"


class FakeSpecAlgorithm:
    def __init__(self, none: bool = True) -> None:
        self._none = none

    def is_none(self) -> bool:
        return self._none


@dataclass
class FakeReq:
    """Shaped like ``sglang.srt.managers.schedule_batch.Req``."""

    rid: str
    kv_committed_len: int = 0
    cached_tokens: int = 0
    origin_input_ids: list = field(default_factory=list)
    output_ids: list = field(default_factory=list)
    #: positive while this request's prefill is still being chunked; SGLang
    #: increments it before the batch is built and discards the row's token
    inflight_middle_chunks: int = 0
    is_retracted: bool = False
    finished_reason: object | None = None
    sampling_params: object | None = None
    grammar: object | None = None
    eos_token_ids: set | None = None
    vocab_size: int | None = None
    tokenizer: object | None = None

    def finished(self) -> bool:
        return self.finished_reason is not None


@dataclass
class FakeScheduleBatch:
    """Shaped like ``sglang.srt.managers.schedule_batch.ScheduleBatch``."""

    reqs: list = field(default_factory=list)
    forward_mode: FakeForwardMode = field(default_factory=lambda: FakeForwardMode("extend"))
    seq_lens_cpu: list = field(default_factory=list)
    extend_lens: list = field(default_factory=list)
    prefix_lens: list = field(default_factory=list)
    decoding_reqs: list | None = None
    return_logprob: bool = False
    device: str = "cpu"
    spec_algorithm: FakeSpecAlgorithm = field(default_factory=FakeSpecAlgorithm)


# Import surface

@pytest.mark.skipif(SGLANG_INSTALLED, reason="the worker's guarded import pulls SGLang in")
def test_package_imports_without_sglang():
    assert "sglang" not in sys.modules
    import simllm.adapters.sglang.worker as worker_module

    assert worker_module.PINNED_SGLANG_COMMIT == "bfeae4e"
    assert "sglang" not in sys.modules
    assert worker_module.sglang_is_available() is False


@pytest.mark.skipif(SGLANG_INSTALLED, reason="construction should succeed with SGLang")
def test_construction_without_sglang_raises_a_clear_error():
    from simllm.adapters.sglang import SimTpModelWorker

    with pytest.raises(ImportError) as excinfo:
        SimTpModelWorker(server_args=object(), gpu_id=0, ps=object(), nccl_port=0)
    message = str(excinfo.value)
    assert "bfeae4e" in message
    assert "plugin" in message


def test_plugin_gate_reads_the_enable_env():
    assert enabled({}) is False
    assert enabled({ENABLE_ENV: "0"}) is False
    assert enabled({ENABLE_ENV: "1"}) is True


# Batch observation

def test_extend_batch_observation():
    batch = FakeScheduleBatch(
        reqs=[FakeReq("a", cached_tokens=128), FakeReq("b")],
        forward_mode=FakeForwardMode("extend"),
        seq_lens_cpu=[512, 40],
        extend_lens=[384, 40],
        prefix_lens=[128, 0],
    )
    rows = observe_schedule_batch(batch)
    assert rows == [
        BatchRow(rid="a", is_decode=False, num_new_tokens=384, context_length=512,
                 cached_tokens=128),
        BatchRow(rid="b", is_decode=False, num_new_tokens=40, context_length=40,
                 cached_tokens=0),
    ]


def test_decode_batch_ignores_stale_extend_lens():
    # A reused batch object carries stale extend_lens from a prior extend;
    # decode rows are always exactly one token.
    batch = FakeScheduleBatch(
        reqs=[FakeReq("a"), FakeReq("b")],
        forward_mode=FakeForwardMode("decode"),
        seq_lens_cpu=[513, 41],
        extend_lens=[384, 40],
    )
    rows = observe_schedule_batch(batch)
    assert [(row.rid, row.is_decode, row.num_new_tokens, row.context_length) for row in rows] == [
        ("a", True, 1, 513),
        ("b", True, 1, 41),
    ]


def test_mixed_batch_marks_decoding_reqs_as_decode_rows():
    # mix_with_running appends the running decode requests after the
    # prefills, each with an extend_lens entry of 1 and a synthetic
    # prefix_lens entry that must never be read as a radix hit.
    running = FakeReq("old", cached_tokens=999)
    batch = FakeScheduleBatch(
        reqs=[FakeReq("new", cached_tokens=16), running],
        forward_mode=FakeForwardMode("mixed"),
        seq_lens_cpu=[64, 201],
        extend_lens=[64, 1],
        prefix_lens=[16, 200],
        decoding_reqs=[running],
    )
    rows = observe_schedule_batch(batch)
    assert rows[0] == BatchRow(
        rid="new", is_decode=False, num_new_tokens=64, context_length=64, cached_tokens=16
    )
    assert rows[1] == BatchRow(
        rid="old", is_decode=True, num_new_tokens=1, context_length=201, cached_tokens=0
    )


def test_idle_and_empty_batches_produce_no_rows():
    assert observe_schedule_batch(FakeScheduleBatch(forward_mode=FakeForwardMode("idle"))) == []
    assert observe_schedule_batch(FakeScheduleBatch(reqs=[])) == []


def test_context_length_falls_back_without_the_batch_tensor():
    batch = FakeScheduleBatch(
        reqs=[FakeReq("a", kv_committed_len=77)],
        forward_mode=FakeForwardMode("decode"),
        seq_lens_cpu=[],
    )
    assert observe_schedule_batch(batch)[0].context_length == 77


# Translation

def test_translation_reports_the_radix_hit_exactly_once():
    translator = SglStepTranslator()
    chunk1 = [BatchRow(rid="r", is_decode=False, num_new_tokens=512,
                       context_length=640, cached_tokens=128)]
    record0 = translator.translate(step_index=0, virtual_time_ps=0, rows=chunk1)
    assert record0.scheduled[0].phase is RequestPhase.PREFILL
    assert record0.scheduled[0].num_cached_tokens == 128

    # The second chunk carries the same cumulative Req.cached_tokens, but
    # the record is per step, not cumulative.
    chunk2 = [BatchRow(rid="r", is_decode=False, num_new_tokens=360,
                       context_length=1000, cached_tokens=128)]
    record1 = translator.translate(step_index=1, virtual_time_ps=1_000, rows=chunk2)
    assert record1.scheduled[0].num_cached_tokens == 0

    decode = [BatchRow(rid="r", is_decode=True, num_new_tokens=1, context_length=1001)]
    record2 = translator.translate(step_index=2, virtual_time_ps=2_000, rows=decode)
    assert record2.scheduled[0].phase is RequestPhase.DECODE
    assert record2.finished_request_ids == []
    assert record2.total_new_tokens == 1


def test_mixed_decode_row_never_reports_a_cache_hit():
    translator = SglStepTranslator()
    rows = [
        BatchRow(rid="p", is_decode=False, num_new_tokens=8, context_length=8,
                 cached_tokens=4),
        BatchRow(rid="d", is_decode=True, num_new_tokens=1, context_length=100),
    ]
    record = translator.translate(step_index=0, virtual_time_ps=0, rows=rows)
    by_id = {req.request_id: req for req in record.scheduled}
    assert by_id["p"].num_cached_tokens == 4
    assert by_id["d"].num_cached_tokens == 0
    assert by_id["d"].phase is RequestPhase.DECODE
    # A decode row must not consume the request's one-time cache report: if
    # "d" later resumes as a prefill (retraction), its hit is still reportable.
    resumed = [BatchRow(rid="d", is_decode=False, num_new_tokens=50,
                        context_length=150, cached_tokens=60)]
    record2 = translator.translate(step_index=1, virtual_time_ps=1, rows=resumed)
    assert record2.scheduled[0].num_cached_tokens == 60


# Sampled-row identity (SGL-12)

def test_mid_prompt_chunk_row_consumes_no_generated_token():
    # SGLang increments inflight_middle_chunks on the chunked request before
    # building the batch, and process_batch_result_prefill then decrements it
    # instead of appending the sampled token.
    batch = FakeScheduleBatch(
        reqs=[
            FakeReq("chunked", inflight_middle_chunks=1, origin_input_ids=[0] * 1000),
            FakeReq("whole", origin_input_ids=[0] * 40),
        ],
        seq_lens_cpu=[600, 40],
        extend_lens=[600, 40],
    )
    rows = observe_schedule_batch(batch)
    assert [(row.rid, row.produces_token) for row in rows] == [
        ("chunked", False),
        ("whole", True),
    ]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("is_retracted", True), ("finished_reason", "length")],
)
def test_retracted_and_finished_extend_rows_consume_no_generated_token(field_name, value):
    req = FakeReq("r", origin_input_ids=[0] * 8)
    setattr(req, field_name, value)
    batch = FakeScheduleBatch(reqs=[req], seq_lens_cpu=[8], extend_lens=[8])
    assert observe_schedule_batch(batch)[0].produces_token is False


def test_resumed_prefill_after_retraction_consumes_a_token_again():
    # prepare_for_extend clears is_retracted for every admitted row, so the
    # resumed prefill is a token-producing row and keeps its output index.
    req = FakeReq("r", origin_input_ids=[0] * 8, output_ids=[7, 9], cached_tokens=4)
    batch = FakeScheduleBatch(reqs=[req], seq_lens_cpu=[10], extend_lens=[6])
    row = observe_schedule_batch(batch)[0]
    assert row.produces_token is True
    assert row.num_output_tokens == 2


def test_decode_rows_always_consume_a_generated_token():
    batch = FakeScheduleBatch(
        reqs=[FakeReq("a", output_ids=[1, 2]), FakeReq("b", output_ids=[3])],
        forward_mode=FakeForwardMode("decode"),
        seq_lens_cpu=[513, 41],
    )
    rows = observe_schedule_batch(batch)
    assert [(row.produces_token, row.num_output_tokens) for row in rows] == [
        (True, 2),
        (True, 1),
    ]


def test_mixed_batch_counts_only_the_rows_sglang_consumes():
    running = FakeReq("old", output_ids=[5])
    batch = FakeScheduleBatch(
        reqs=[FakeReq("new", inflight_middle_chunks=1), running],
        forward_mode=FakeForwardMode("mixed"),
        seq_lens_cpu=[64, 201],
        extend_lens=[64, 1],
        decoding_reqs=[running],
    )
    record = SglStepTranslator().translate(
        step_index=0, virtual_time_ps=0, rows=observe_schedule_batch(batch)
    )
    assert record.num_sampled == 1
    assert record.sampled_request_ids == ["old"]
    assert sampled_request_ids(record) == {"old"}


def test_two_prefill_rows_need_the_identity_list_not_only_the_count():
    # One mid-prompt chunk and one prefill that completes its prompt: the
    # count alone is ambiguous because the scheduled decode set is empty, and
    # the shared completion rule refuses to guess.
    batch = FakeScheduleBatch(
        reqs=[FakeReq("chunked", inflight_middle_chunks=1), FakeReq("whole")],
        seq_lens_cpu=[600, 40],
        extend_lens=[600, 40],
    )
    record = SglStepTranslator().translate(
        step_index=0, virtual_time_ps=0, rows=observe_schedule_batch(batch)
    )
    assert record.num_sampled == 1
    assert record.sampled_request_ids == ["whole"]
    assert sampled_request_ids(record) == {"whole"}

    count_only = StepRecord(
        step_index=record.step_index,
        virtual_time_ps=record.virtual_time_ps,
        scheduled=record.scheduled,
        num_sampled=1,
    )
    with pytest.raises(ValueError, match="ambiguous"):
        sampled_request_ids(count_only)


def test_absent_identity_counts_every_scheduled_row_as_sampled():
    rows = [
        BatchRow(rid="chunked", is_decode=False, num_new_tokens=600,
                 context_length=600, produces_token=False),
        BatchRow(rid="whole", is_decode=False, num_new_tokens=40, context_length=40),
    ]
    record = SglStepTranslator(sample_identity=False).translate(
        step_index=0, virtual_time_ps=0, rows=rows
    )
    assert record.num_sampled is None
    assert record.sampled_request_ids is None
    payload = step_record_to_json(record)
    assert "num_sampled" not in payload
    assert "sampled_request_ids" not in payload
    # This is the defect the compatibility path preserves on purpose.
    assert sampled_request_ids(record) == {"chunked", "whole"}


def test_a_step_with_no_consuming_row_reports_zero_sampled():
    rows = [
        BatchRow(rid="chunked", is_decode=False, num_new_tokens=400,
                 context_length=400, produces_token=False)
    ]
    record = SglStepTranslator().translate(step_index=0, virtual_time_ps=0, rows=rows)
    assert record.num_sampled == 0
    assert record.sampled_request_ids == []
    assert sampled_request_ids(record) == set()


@pytest.mark.skipif(not SGLANG_INSTALLED, reason="requires the pinned SGLang install")
def test_pinned_sglang_classes_carry_every_transcribed_field():
    """The names this adapter reads with getattr exist on the pinned classes.

    Skipped wherever SGLang is absent, which includes CI. It is the structural
    half of the SGL-12 transcription: a rename upstream turns a getattr default
    into a silent wrong answer, and this test turns it into a failure.
    """

    import inspect

    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.sampling.sampling_params import SamplingParams

    from simllm.adapters.sglang.worker import SimModelRunnerStub, SimTpModelWorker

    def signature_shape(owner, name):
        return [
            (parameter.name, parameter.kind, parameter.default)
            for parameter in inspect.signature(getattr(owner, name)).parameters.values()
        ]

    assert signature_shape(SimTpModelWorker, "__init__") == signature_shape(
        TpModelWorker, "__init__"
    )
    assert signature_shape(
        SimTpModelWorker, "forward_batch_generation"
    ) == signature_shape(TpModelWorker, "forward_batch_generation")
    runner_parameters = inspect.signature(ModelRunner.__init__).parameters
    assert "draft_attention_backend" in runner_parameters
    stub_source = inspect.getsource(SimModelRunnerStub)
    for name in ("graph_memory_usage", "graph_time_usage", "weight_load_time"):
        assert f"self.{name}" in stub_source

    request_source = inspect.getsource(Req.__init__)
    for name in (
        "inflight_middle_chunks",
        "is_retracted",
        "origin_input_ids",
        "output_ids",
        "cached_tokens",
        "vocab_size",
        "eos_token_ids",
        "tokenizer",
    ):
        assert f"self.{name}" in request_source, name
    assert callable(Req.finished)
    assert "self.grammar" in inspect.getsource(Req)

    for name in (
        "reqs",
        "forward_mode",
        "seq_lens_cpu",
        "extend_lens",
        "decoding_reqs",
        "return_logprob",
        "device",
        "spec_algorithm",
    ):
        assert name in ScheduleBatch.__dataclass_fields__, name

    sampling_source = inspect.getsource(SamplingParams)
    for name in (
        "max_new_tokens",
        "min_new_tokens",
        "ignore_eos",
        "stop_token_ids",
        "stop_strs",
        "stop_regex_strs",
    ):
        assert f"    {name}:" in sampling_source or f"self.{name}" in sampling_source, name


# Geometry

def test_model_dims_from_sglang_shards_by_tp():
    class FakeHf:
        hidden_size = 4096
        intermediate_size = 14336
        num_attention_heads = 32
        num_key_value_heads = 8
        vocab_size = 128256

    class FakeDtype:
        itemsize = 2

    class FakeModelConfig:
        hf_text_config = FakeHf()
        num_hidden_layers = 32
        num_attention_heads = 32
        head_dim = 128
        vocab_size = 128256
        dtype = FakeDtype()

        @staticmethod
        def get_num_kv_heads(tp_size):
            return max(8 // tp_size, 1)

    dims = model_dims_from_sglang(FakeModelConfig(), tp_size=4)
    assert dims.num_layers == 32
    assert dims.num_heads == 8
    assert dims.num_kv_heads == 2
    assert dims.intermediate_size == 14336 // 4
    assert dims.head_size == 128
    assert dims.defaulted_fields == ()


def test_model_dims_from_sglang_defaults_loudly():
    dims = model_dims_from_sglang(object(), tp_size=1)
    assert dims.hidden_size == 4096
    assert "hidden_size" in dims.defaulted_fields
    assert "num_layers" in dims.defaulted_fields


def _model_config(**hf_overrides):
    fields = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "vocab_size": 49152,
    }
    fields.update(hf_overrides)
    hf = SimpleNamespace(**fields)
    config = SimpleNamespace(
        hf_text_config=hf,
        num_hidden_layers=24,
        num_attention_heads=hf.num_attention_heads,
        head_dim=hf.hidden_size // hf.num_attention_heads,
        vocab_size=hf.vocab_size,
        dtype=SimpleNamespace(itemsize=2),
    )
    config.get_num_kv_heads = lambda tp_size: max(hf.num_key_value_heads // tp_size, 1)
    return config


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        (
            {
                "model_type": "granitemoe",
                "architectures": ["GraniteMoeForCausalLM"],
                "intermediate_size": 512,
                "num_local_experts": 32,
                "num_experts_per_tok": 8,
            },
            (32, 8, 512, 32),
        ),
        (
            {
                "model_type": "mixtral",
                "architectures": ["MixtralForCausalLM"],
                "intermediate_size": 14336,
                "num_local_experts": 8,
                "num_experts_per_tok": 2,
            },
            (8, 2, 14336, 8),
        ),
        (
            {
                "model_type": "qwen3_moe",
                "architectures": ["Qwen3MoeForCausalLM"],
                "num_experts": 64,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 1408,
            },
            (64, 8, 1408, 64),
        ),
    ],
)
def test_model_dims_from_sglang_reads_supported_single_gpu_moe(fields, expected):
    dims = model_dims_from_sglang(_model_config(**fields))
    assert (
        dims.num_experts,
        dims.top_k,
        dims.moe_intermediate_size,
        dims.local_num_experts,
    ) == expected


def test_granite_moe_changes_active_and_resident_mlp_geometry():
    dense = model_dims_from_sglang(_model_config(intermediate_size=512))
    moe = model_dims_from_sglang(
        _model_config(
            model_type="granitemoe",
            architectures=["GraniteMoeForCausalLM"],
            intermediate_size=512,
            num_local_experts=32,
            num_experts_per_tok=8,
        )
    )
    assert moe.mlp_active_params == 8 * dense.mlp_active_params
    assert moe.mlp_resident_params == 32 * dense.mlp_resident_params


@pytest.mark.parametrize(
    "fields",
    [
        {
            "model_type": "custom_moe",
            "architectures": ["CustomMoeForCausalLM"],
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 512,
        },
        {
            "model_type": "qwen2_moe",
            "architectures": ["Qwen2MoeForCausalLM"],
            "num_experts": 60,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 1408,
            "shared_expert_intermediate_size": 5632,
        },
        {
            "model_type": "deepseek_v3",
            "architectures": ["DeepseekV3ForCausalLM"],
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
            "n_shared_experts": 1,
        },
        {
            "model_type": "dbrx",
            "architectures": ["DbrxForCausalLM"],
            "ffn_config": SimpleNamespace(
                moe_num_experts=16, moe_top_k=4, ffn_hidden_size=3584
            ),
        },
        {
            "model_type": "custom_moe",
            "architectures": ["CustomMoeForCausalLM"],
        },
        {
            "model_type": "quant_mixtral",
            "architectures": ["QuantMixtralForCausalLM"],
            "intermediate_size": 14336,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        },
    ],
)
def test_model_dims_from_sglang_rejects_unsupported_moe_instead_of_dense(fields):
    with pytest.raises(NotImplementedError, match="SGL-18"):
        model_dims_from_sglang(_model_config(**fields))


def test_model_dims_from_sglang_rejects_partial_or_disagreeing_moe_geometry():
    missing_top_k = _model_config(
        model_type="granitemoe",
        architectures=["GraniteMoeForCausalLM"],
        intermediate_size=512,
        num_local_experts=32,
    )
    with pytest.raises(ValueError, match="num_experts_per_tok"):
        model_dims_from_sglang(missing_top_k)

    disagreeing = _model_config(
        model_type="granitemoe",
        architectures=["GraniteMoeForCausalLM"],
        intermediate_size=512,
        num_local_experts=32,
        num_experts=16,
        num_experts_per_tok=8,
    )
    with pytest.raises(ValueError, match="disagree"):
        model_dims_from_sglang(disagreeing)

    invalid_top_k = _model_config(
        model_type="granitemoe",
        architectures=["GraniteMoeForCausalLM"],
        intermediate_size=512,
        num_local_experts=8,
        num_experts_per_tok=9,
    )
    with pytest.raises(ValueError, match="exceeds"):
        model_dims_from_sglang(invalid_top_k)

    noninteger_experts = _model_config(
        model_type="granitemoe",
        architectures=["GraniteMoeForCausalLM"],
        intermediate_size=512,
        num_local_experts=32.0,
        num_experts_per_tok=8,
    )
    with pytest.raises(ValueError, match="must be an integer"):
        model_dims_from_sglang(noninteger_experts)


def test_model_dims_from_sglang_rejects_conflicting_family_identity():
    config = _model_config(
        model_type="qwen2_moe",
        architectures=["GraniteMoeForCausalLM"],
        intermediate_size=512,
        num_local_experts=32,
        num_experts_per_tok=8,
    )
    with pytest.raises(ValueError, match="conflicting MoE families"):
        model_dims_from_sglang(config)


@pytest.mark.parametrize(
    "parallel",
    [
        {"tp_size": 2, "moe_tp_size": 2},
        {"moe_ep_size": 2},
        {"moe_dp_size": 2},
    ],
)
def test_model_dims_from_sglang_rejects_distributed_moe_first_slice(parallel):
    config = _model_config(
        model_type="granitemoe",
        architectures=["GraniteMoeForCausalLM"],
        intermediate_size=512,
        num_local_experts=32,
        num_experts_per_tok=8,
    )
    with pytest.raises(NotImplementedError, match="TP=EP=MoE-DP=1"):
        model_dims_from_sglang(config, **parallel)


def test_model_dims_from_sglang_rejects_redundant_expert_copies():
    config = _model_config(
        model_type="qwen3_moe",
        architectures=["Qwen3MoeForCausalLM"],
        num_experts=64,
        num_experts_per_tok=8,
        moe_intermediate_size=1408,
    )
    with pytest.raises(NotImplementedError, match="redundant expert"):
        model_dims_from_sglang(config, ep_num_redundant_experts=1)


def test_sglang_moe_tp_size_is_derived_from_parallel_state():
    assert _sglang_moe_parallel_sizes(8, 2, 2) == (2, 2, 2)
    with pytest.raises(ValueError, match="must be divisible"):
        _sglang_moe_parallel_sizes(6, 2, 2)


def test_dense_projection_does_not_validate_irrelevant_moe_parallel_sizes():
    dense = model_dims_from_sglang(
        _model_config(), tp_size=1, moe_ep_size=2, moe_dp_size=2
    )
    assert dense.num_experts == 0


def test_qwen35_dense_projection_ignores_inherited_moe_defaults():
    config = _model_config(
        model_type="qwen3_5_text",
        architectures=["Qwen3_5ForConditionalGeneration"],
        intermediate_size=17408,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=512,
    )

    dense = model_dims_from_sglang(config)

    assert dense.num_experts == 0
    assert dense.top_k == 0
    assert dense.moe_intermediate_size is None


def test_qwen35_dense_identity_conflict_still_rejects():
    config = _model_config(
        model_type="qwen3_5_text",
        architectures=["Qwen3MoeForCausalLM"],
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=512,
    )

    with pytest.raises(ValueError, match="conflicts with a routed MoE identity"):
        model_dims_from_sglang(config)


def test_model_dims_from_sglang_rejects_multimodal_moe_wrapper():
    text = _model_config(
        model_type="qwen3_moe",
        architectures=["Qwen3MoeForCausalLM"],
        num_experts=64,
        num_experts_per_tok=8,
        moe_intermediate_size=1408,
    ).hf_text_config
    outer = SimpleNamespace(
        architectures=["InternVLChatModel"],
        model_type="internvl_chat",
        llm_config=text,
    )
    config = SimpleNamespace(
        hf_text_config=text,
        hf_config=outer,
        is_multimodal=True,
        num_hidden_layers=24,
        num_attention_heads=16,
        head_dim=64,
        vocab_size=49152,
        dtype=SimpleNamespace(itemsize=2),
        get_num_kv_heads=lambda tp_size: max(4 // tp_size, 1),
    )
    with pytest.raises(NotImplementedError, match="multimodal MoE"):
        model_dims_from_sglang(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shared_expert_intermediate_size", 1024),
        ("decoder_sparse_step", 2),
        ("mlp_only_layers", [0]),
        ("kv_lora_rank", 512),
        ("use_mla", True),
        ("quantization", "fp8"),
        ("num_nextn_predict_layers", 1),
    ],
)
def test_model_dims_from_sglang_rejects_unrepresentable_moe_mechanisms(field, value):
    config = _model_config(
        model_type="qwen3_moe",
        architectures=["Qwen3MoeForCausalLM"],
        num_experts=64,
        num_experts_per_tok=8,
        moe_intermediate_size=1408,
        **{field: value},
    )
    with pytest.raises(NotImplementedError, match="SGL-18"):
        model_dims_from_sglang(config)


# Configuration

def test_config_from_env_reads_every_knob():
    config = SimWorkerConfig.from_env(
        {
            "SIMLLM_SGLANG_MODE": "paced",
            "SIMLLM_SGLANG_GPU": "H200",
            "SIMLLM_SGLANG_PEAK_FLOPS": "1e15",
            "SIMLLM_SGLANG_EFFICIENCY": "0.5",
            "SIMLLM_SGLANG_HOST_INIT_PS": "1234",
            "SIMLLM_SGLANG_TOKEN_ID": "99",
            "SIMLLM_SGLANG_STEP_RECORDS": "/tmp/steps.jsonl",
            "SIMLLM_SGLANG_COMMUNICATOR_TP_SIZE": "4",
            "SIMLLM_SGLANG_COMMUNICATOR_EVENTS": "/tmp/events.jsonl",
        }
    )
    assert config.mode == "paced"
    assert config.efficiency == 0.5
    assert config.host_initiation_ps == 1234
    assert config.token_id == 99
    assert config.communicator_tp_size == 4
    assert config.communicator_events_path == "/tmp/events.jsonl"
    gpu = config.gpu_spec()
    assert gpu.name == "h200"
    assert gpu.peak_flops == 1e15  # override applied
    assert gpu.mem_bandwidth == 4.8e12  # envelope default kept


def test_config_rejects_bad_values():
    with pytest.raises(ValueError, match="virtual or paced"):
        SimWorkerConfig.from_env({"SIMLLM_SGLANG_MODE": "fast"})
    with pytest.raises(ValueError, match="unknown SIMLLM_SGLANG_GPU"):
        SimWorkerConfig.from_env({"SIMLLM_SGLANG_GPU": "mi300"}).gpu_spec()
    with pytest.raises(ValueError, match="must be positive"):
        SimWorkerConfig.from_env({"SIMLLM_SGLANG_COMMUNICATOR_TP_SIZE": "0"})
    with pytest.raises(ValueError, match="must divide"):
        SimWorkerConfig.from_env({"SIMLLM_SGLANG_COMMUNICATOR_TP_SIZE": "3"})
    with pytest.raises(ValueError, match="requires"):
        SimWorkerConfig.from_env(
            {"SIMLLM_SGLANG_COMMUNICATOR_EVENTS": "/tmp/events.jsonl"}
        )


def test_sample_identity_defaults_on_and_needs_an_explicit_off_spelling():
    assert SimWorkerConfig.from_env({}).sample_identity is True
    assert SimWorkerConfig.from_env({"SIMLLM_SGLANG_SAMPLE_IDENTITY": ""}).sample_identity
    assert SimWorkerConfig.from_env({"SIMLLM_SGLANG_SAMPLE_IDENTITY": "1"}).sample_identity
    for spelling in ("0", "false", "No", " off "):
        config = SimWorkerConfig.from_env({"SIMLLM_SGLANG_SAMPLE_IDENTITY": spelling})
        assert config.sample_identity is False


def test_replay_run_path_defaults_to_the_fabricated_token_path(tmp_path):
    assert SimWorkerConfig.from_env({}).replay_run_path is None
    path = str(tmp_path / "joined-replay.json")
    assert SimWorkerConfig.from_env(
        {"SIMLLM_SGLANG_REPLAY_RUN": path}
    ).replay_run_path == path


def test_reset_configuration_drops_every_stale_hook():
    def first_sink(record):
        return None

    try:
        hooks = configure(step_sink=first_sink, config=SimWorkerConfig())
        assert hooks.step_sink is first_sink
        cleared = reset_configuration()
        assert cleared.step_sink is None
        assert cleared.config is None
        assert cleared.compute_provider is None
        assert cleared.gpu is None
        assert cleared.host_model is None
        # A second cell's configure must not observe the first cell's sink.
        second = configure(config=SimWorkerConfig(mode="paced"))
        assert second.step_sink is None
        assert second.config.mode == "paced"
    finally:
        reset_configuration()


# Pre-play replay serving

def joined_replay_path(tmp_path, *, trace_path=None, name="joined-replay.json"):
    from simllm.core import RequestBookkeeper

    run = join_preplay_arrivals(
        (RequestArrival(request_id="request-golden", arrived_at_ps=0),),
        GOLDEN_TRACE if trace_path is None else trace_path,
        RequestBookkeeper(),
    )
    return write_preplay_replay_run(run, tmp_path / name)


def joined_two_token_replay_path(tmp_path):
    """A joined run whose oracle is two tokens, 20 then 21."""

    from dataclasses import replace

    source = read_preplay_trace(GOLDEN_TRACE)
    request = source.requests[0]
    decode = replace(
        request.prefill_tokens[0],
        phase=ForwardPhase.DECODE,
        token_index=0,
        token_id=20,
    )
    request = replace(
        request,
        max_new_tokens=2,
        output_token_ids=(20, 21),
        decode_tokens=(decode,),
    )
    trace_path = write_preplay_trace(
        tmp_path / "two-token-trace.jsonl", source.provenance, (request,)
    )
    return joined_replay_path(
        tmp_path, trace_path=trace_path, name="joined-two-token-replay.json"
    )


def replay_sampling_params(output_length=1, **overrides):
    fields = {
        "max_new_tokens": output_length,
        "min_new_tokens": 0,
        "ignore_eos": False,
        "stop_token_ids": None,
        "stop_strs": None,
        "stop_regex_strs": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def replay_batch(*, chunked=False, params=None, **req_overrides):
    req = FakeReq(
        "request-golden",
        origin_input_ids=[10],
        sampling_params=replay_sampling_params() if params is None else params,
        vocab_size=1024,
        inflight_middle_chunks=1 if chunked else 0,
        **req_overrides,
    )
    return FakeScheduleBatch(reqs=[req], seq_lens_cpu=[1], extend_lens=[1])


def test_replay_serves_the_oracle_token_at_the_reported_output_index(tmp_path):
    source = SglReplayTokenSource.from_path(
        joined_replay_path(tmp_path), max_context_len=4096
    )
    batch = replay_batch()
    rows = observe_schedule_batch(batch)

    assert source.sample(batch, rows, fallback_token_id=512) == [20]
    snapshot = source.snapshot()
    assert snapshot.served_token_ids == (("request-golden", (20,)),)
    assert snapshot.completed_request_ids == ("request-golden",)


def test_replay_leaves_a_mid_prompt_chunk_on_the_fabricated_token(tmp_path):
    source = SglReplayTokenSource.from_path(
        joined_replay_path(tmp_path), max_context_len=4096
    )
    chunk = replay_batch(chunked=True)
    chunk_rows = observe_schedule_batch(chunk)
    assert chunk_rows[0].produces_token is False

    assert source.sample(chunk, chunk_rows, fallback_token_id=512) == [512]
    assert source.snapshot().served_token_ids == (("request-golden", ()),)
    assert source.snapshot().completed_request_ids == ()

    # The oracle index did not move, so the next producing row still serves
    # the request's first predefined token.
    final = replay_batch()
    assert source.sample(final, observe_schedule_batch(final), fallback_token_id=512) == [20]


def test_replay_validation_does_not_move_the_oracle(tmp_path):
    source = SglReplayTokenSource.from_path(
        joined_replay_path(tmp_path), max_context_len=4096
    )
    batch = replay_batch()
    rows = observe_schedule_batch(batch)
    source.validate_step(batch, rows)
    source.validate_step(batch, rows)
    assert source.snapshot().served_token_ids == (("request-golden", ()),)


def test_replay_pins_scheduler_visible_completion_to_the_oracle_length(tmp_path):
    source = SglReplayTokenSource.from_path(
        joined_replay_path(tmp_path), max_context_len=4096
    )
    batch = replay_batch(params=replay_sampling_params(output_length=8))
    with pytest.raises(RuntimeError, match="max_new_tokens=1"):
        source.sample(batch, observe_schedule_batch(batch), fallback_token_id=512)


def test_replay_refuses_an_exhausted_oracle(tmp_path):
    source = SglReplayTokenSource.from_path(
        joined_replay_path(tmp_path), max_context_len=4096
    )
    batch = replay_batch()
    assert source.sample(batch, observe_schedule_batch(batch), fallback_token_id=512) == [20]
    served = replay_batch(output_ids=[20])
    with pytest.raises(RuntimeError, match="exhausted its oracle"):
        source.sample(served, observe_schedule_batch(served), fallback_token_id=512)


def test_replay_refuses_an_output_index_the_scheduler_did_not_reach(tmp_path):
    source = SglReplayTokenSource.from_path(
        joined_replay_path(tmp_path), max_context_len=4096
    )
    ahead = replay_batch(output_ids=[20], params=replay_sampling_params(output_length=1))
    with pytest.raises(RuntimeError, match="reported output index"):
        source.validate_step(ahead, observe_schedule_batch(ahead))


def test_replay_refuses_an_unjoined_request(tmp_path):
    source = SglReplayTokenSource.from_path(
        joined_replay_path(tmp_path), max_context_len=4096
    )
    batch = replay_batch()
    batch.reqs[0].rid = "request-golden-deadbeef"
    with pytest.raises(RuntimeError, match="missing from the joined replay run"):
        source.validate_step(batch, observe_schedule_batch(batch))
    assert source.snapshot().served_token_ids == (("request-golden", ()),)


@pytest.mark.parametrize(
    ("params", "req_fields", "message"),
    [
        (
            replay_sampling_params(stop_strs=["stop"]),
            {},
            "stop strings",
        ),
        (
            replay_sampling_params(stop_regex_strs=["st.p"]),
            {},
            "stop strings",
        ),
        (
            replay_sampling_params(min_new_tokens=4),
            {},
            "beyond its oracle length",
        ),
        (
            replay_sampling_params(),
            {"grammar": object()},
            "structured output",
        ),
        (
            replay_sampling_params(),
            {"vocab_size": 8},
            "outside the",
        ),
        (
            None,
            {"sampling_params": None},
            "not a plain generation request",
        ),
    ],
)
def test_replay_refusal_boundaries(tmp_path, params, req_fields, message):
    source = SglReplayTokenSource.from_path(
        joined_replay_path(tmp_path), max_context_len=4096
    )
    batch = replay_batch(params=params)
    for name, value in req_fields.items():
        setattr(batch.reqs[0], name, value)
    with pytest.raises((RuntimeError, NotImplementedError), match=message):
        source.validate_step(batch, observe_schedule_batch(batch))


def test_replay_refuses_an_early_stop_token_but_not_a_final_one(tmp_path):
    source = SglReplayTokenSource.from_path(
        joined_replay_path(tmp_path), max_context_len=4096
    )
    final_stop = replay_batch()
    final_stop.reqs[0].eos_token_ids = {20}
    # The single oracle token is also the last one, so SGLang stops exactly at
    # the oracle length and the run is still exact.
    source.validate_step(final_stop, observe_schedule_batch(final_stop))

    two_token = SglReplayTokenSource.from_path(
        joined_two_token_replay_path(tmp_path), max_context_len=4096
    )
    batch = replay_batch(params=replay_sampling_params(output_length=2))
    batch.reqs[0].eos_token_ids = {20}
    with pytest.raises(RuntimeError, match="stop token before its oracle length"):
        two_token.validate_step(batch, observe_schedule_batch(batch))

    ignored = replay_batch(
        params=replay_sampling_params(output_length=2, ignore_eos=True)
    )
    ignored.reqs[0].eos_token_ids = {20}
    two_token.validate_step(ignored, observe_schedule_batch(ignored))


def test_replay_serves_a_multi_token_oracle_in_scheduler_order(tmp_path):
    source = SglReplayTokenSource.from_path(
        joined_two_token_replay_path(tmp_path), max_context_len=4096
    )
    params = replay_sampling_params(output_length=2)
    prefill = replay_batch(params=params)
    assert source.sample(prefill, observe_schedule_batch(prefill), fallback_token_id=512) == [20]

    decode_req = FakeReq(
        "request-golden",
        origin_input_ids=[10],
        output_ids=[20],
        sampling_params=params,
        vocab_size=1024,
    )
    decode = FakeScheduleBatch(
        reqs=[decode_req],
        forward_mode=FakeForwardMode("decode"),
        seq_lens_cpu=[2],
    )
    assert source.sample(decode, observe_schedule_batch(decode), fallback_token_id=512) == [21]
    snapshot = source.snapshot()
    assert snapshot.served_token_ids == (("request-golden", (20, 21)),)
    assert snapshot.completed_request_ids == ("request-golden",)


def test_replay_refuses_speculative_and_logprob_batches(tmp_path):
    source = SglReplayTokenSource.from_path(
        joined_replay_path(tmp_path), max_context_len=4096
    )
    logprob = replay_batch()
    logprob.return_logprob = True
    with pytest.raises(NotImplementedError, match="logprob"):
        source.validate_step(logprob, observe_schedule_batch(logprob))

    speculative = replay_batch()
    speculative.spec_algorithm = FakeSpecAlgorithm(none=False)
    with pytest.raises(NotImplementedError, match="speculative"):
        source.validate_step(speculative, observe_schedule_batch(speculative))


def test_replay_refuses_a_trace_that_does_not_fit_the_context(tmp_path):
    with pytest.raises(ValueError, match="beyond max_context_len"):
        SglReplayTokenSource.from_path(joined_replay_path(tmp_path), max_context_len=1)


def test_absent_replay_serves_the_fabricated_token_for_every_row(tmp_path):
    batch = replay_batch()
    rows = observe_schedule_batch(batch)
    assert sample_adapter_tokens(None, batch, rows, 512) == [512]

    source = SglReplayTokenSource.from_path(
        joined_replay_path(tmp_path), max_context_len=4096
    )
    assert sample_adapter_tokens(source, batch, rows, 512) == [20]
