"""SGLang adapter tests. Nothing here imports SGLang: the batch observation,
translation, geometry and config logic are plain Python that runs (and is
checked) without a GPU stack installed."""

import importlib.util
import sys
from dataclasses import dataclass, field

import pytest

from simllm.adapters.sglang import (
    BatchRow,
    SglStepTranslator,
    SimWorkerConfig,
    model_dims_from_sglang,
    observe_schedule_batch,
)
from simllm.adapters.sglang.plugin import ENABLE_ENV, enabled
from simllm.core import RequestPhase

SGLANG_INSTALLED = importlib.util.find_spec("sglang") is not None


# ScheduleBatch stubs: same attribute names as SGLang @ 8f2a3ad, nothing else


class FakeForwardMode:
    def __init__(self, mode: str) -> None:
        self._mode = mode

    def is_extend(self) -> bool:
        return self._mode in ("extend", "mixed")

    def is_decode(self) -> bool:
        return self._mode == "decode"

    def is_idle(self) -> bool:
        return self._mode == "idle"


@dataclass
class FakeReq:
    """Shaped like ``sglang.srt.managers.schedule_batch.Req``."""

    rid: str
    kv_committed_len: int = 0
    cached_tokens: int = 0
    origin_input_ids: list = field(default_factory=list)
    output_ids: list = field(default_factory=list)


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


# Import surface

@pytest.mark.skipif(SGLANG_INSTALLED, reason="the worker's guarded import pulls SGLang in")
def test_package_imports_without_sglang():
    assert "sglang" not in sys.modules
    import simllm.adapters.sglang.worker as worker_module

    assert worker_module.PINNED_SGLANG_COMMIT == "8f2a3ad"
    assert "sglang" not in sys.modules
    assert worker_module.sglang_is_available() is False


@pytest.mark.skipif(SGLANG_INSTALLED, reason="construction should succeed with SGLang")
def test_construction_without_sglang_raises_a_clear_error():
    from simllm.adapters.sglang import SimTpModelWorker

    with pytest.raises(ImportError) as excinfo:
        SimTpModelWorker(server_args=object(), gpu_id=0, ps=object(), nccl_port=0)
    message = str(excinfo.value)
    assert "8f2a3ad" in message
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
        }
    )
    assert config.mode == "paced"
    assert config.efficiency == 0.5
    assert config.host_initiation_ps == 1234
    assert config.token_id == 99
    gpu = config.gpu_spec()
    assert gpu.name == "h200"
    assert gpu.peak_flops == 1e15  # override applied
    assert gpu.mem_bandwidth == 4.8e12  # envelope default kept


def test_config_rejects_bad_values():
    with pytest.raises(ValueError, match="virtual or paced"):
        SimWorkerConfig.from_env({"SIMLLM_SGLANG_MODE": "fast"})
    with pytest.raises(ValueError, match="unknown SIMLLM_SGLANG_GPU"):
        SimWorkerConfig.from_env({"SIMLLM_SGLANG_GPU": "mi300"}).gpu_spec()
