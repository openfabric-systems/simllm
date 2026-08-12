import json
from dataclasses import dataclass

import pytest

from simllm.adapters.sglang.oracle import (
    ORACLE_ENABLE_ENV,
    ORACLE_LOG_ENV,
    ORACLE_OBSERVATION_SCHEMA,
    _initialize_cpu_host_cache,
    _observe_alloc_for_decode,
    _observe_alloc_for_extend,
    _observe_match_prefix,
    _observe_remove_event,
    _observe_retract_decode,
    _observe_routed_capture,
    mark_oracle_submission_group,
    oracle_enabled,
)
from simllm.adapters.sglang.plugin import ENABLE_ENV, register


@dataclass
class FakeKv:
    kv_allocated_len: int


@dataclass
class FakeReq:
    rid: str
    req_pool_idx: int
    kv: FakeKv
    num_matched_prefix_tokens: int = 0


class FakePool:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        row, column = key
        return self.rows[row][column]


class FakeReqToTokenPool:
    def __init__(self, rows):
        self.req_to_token = FakePool(rows)


@dataclass
class FakeBatch:
    reqs: list
    extend_lens: list
    req_to_token_pool: FakeReqToTokenPool
    forward_iter: int = 7


def sidecar_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def configure_sidecar(monkeypatch, tmp_path):
    path = tmp_path / "oracle.jsonl"
    monkeypatch.setenv(ORACLE_LOG_ENV, str(path))
    return path


def test_oracle_gate_is_independent_and_mutually_exclusive(monkeypatch):
    assert oracle_enabled({}) is False
    assert oracle_enabled({ORACLE_ENABLE_ENV: "1"}) is True

    monkeypatch.setenv(ORACLE_ENABLE_ENV, "1")
    monkeypatch.setenv(ENABLE_ENV, "1")
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        register()


def test_submission_group_marker_preserves_scheduler_boundary(monkeypatch, tmp_path):
    path = configure_sidecar(monkeypatch, tmp_path)

    mark_oracle_submission_group(3, ("r0", "r1"))

    assert sidecar_rows(path) == [
        {
            "group_index": 3,
            "kind": "submission-group-start",
            "request_ids": ["r0", "r1"],
            "schema": ORACLE_OBSERVATION_SCHEMA,
        }
    ]


def test_cpu_dispatch_capture_uses_unpinned_cpu_storage(monkeypatch, tmp_path):
    path = configure_sidecar(monkeypatch, tmp_path)
    monkeypatch.setenv("SGLANG_USE_CPU_ENGINE", "1")

    class Buffer:
        device = "cpu"
        shape = (65, 24, 8)

        @staticmethod
        def is_pinned():
            return False

    class Torch:
        int32 = object()

        @staticmethod
        def zeros(shape, *, dtype, device, pin_memory):
            assert shape == (65, 24, 8)
            assert dtype is Torch.int32
            assert device == "cpu"
            assert pin_memory is False
            return Buffer()

    class Cache:
        log_calls = 0

        def _log_allocation(self):
            self.log_calls += 1

    cache = Cache()
    _initialize_cpu_host_cache(
        lambda *_args: pytest.fail("upstream pinned allocation must not run"),
        cache,
        65,
        24,
        8,
        "routed_experts",
        _torch_module=Torch,
    )

    assert cache.buffer.device == "cpu"
    assert cache.log_calls == 1
    assert sidecar_rows(path) == [
        {
            "device": "cpu",
            "kind": "capture-storage-qualified",
            "name": "routed_experts",
            "pinned": False,
            "schema": ORACLE_OBSERVATION_SCHEMA,
            "shape": [65, 24, 8],
        }
    ]


def test_missing_granite_layer_ids_are_labeled_without_changing_experts(
    monkeypatch, tmp_path
):
    path = configure_sidecar(monkeypatch, tmp_path)

    class Capturer:
        num_layers = 2

    selected = object()
    calls = []

    def original(capturer, layer_id, topk_indices):
        calls.append((capturer, layer_id, topk_indices))
        return topk_indices

    capturer = Capturer()
    assert _observe_routed_capture(original, capturer, None, selected) is selected
    assert _observe_routed_capture(original, capturer, None, selected) is selected
    assert [call[1] for call in calls] == [0, 1]
    assert all(call[2] is selected for call in calls)
    assert sidecar_rows(path)[0] == {
        "kind": "dispatch-layer-qualified",
        "layer_count": 2,
        "mapping": "granite-model-order",
        "schema": ORACLE_OBSERVATION_SCHEMA,
        "selected_experts_unchanged": True,
    }


def test_allocation_hooks_record_exact_request_slots(monkeypatch, tmp_path):
    path = configure_sidecar(monkeypatch, tmp_path)
    reqs = [FakeReq("r0", 0, FakeKv(2)), FakeReq("r1", 1, FakeKv(1))]
    batch = FakeBatch(
        reqs=reqs,
        extend_lens=[2, 1],
        req_to_token_pool=FakeReqToTokenPool([[10, 11], [12]]),
    )

    extend_result = _observe_alloc_for_extend(
        lambda _batch: ([10, 11, 12], object(), object()), batch
    )
    decode_result = _observe_alloc_for_decode(
        lambda _batch, _tokens: [20, 21], batch, 1
    )

    assert extend_result[0] == [10, 11, 12]
    assert decode_result == [20, 21]
    values = sidecar_rows(path)
    assert [(row["request_id"], row["token_slot_ids"]) for row in values] == [
        ("r0", [10, 11]),
        ("r1", [12]),
        ("r0", [20]),
        ("r1", [21]),
    ]
    assert all(row["schema"] == ORACLE_OBSERVATION_SCHEMA for row in values)


def test_prefix_and_eviction_hooks_observe_framework_results(monkeypatch, tmp_path):
    path = configure_sidecar(monkeypatch, tmp_path)
    req = FakeReq("r0", 0, FakeKv(2), num_matched_prefix_tokens=2)
    tree_cache = type("Tree", (), {"req_to_token_pool": object()})()
    params = type("Params", (), {"req": req})()
    match = type(
        "Match",
        (),
        {"device_indices": [4, 5], "host_hit_length": 0},
    )()

    returned = _observe_match_prefix(
        lambda _cache, _params: match,
        tree_cache,
        params,
    )
    node = type("Node", (), {"value": [4, 5]})()
    marker = object()
    removed = _observe_remove_event(
        lambda _cache, _node: marker,
        tree_cache,
        node,
    )

    assert returned is match
    assert removed is marker
    values = sidecar_rows(path)
    assert values[0] == {
        "device_token_count": 2,
        "framework_step": None,
        "host_token_count": 0,
        "kind": "prefix-hit",
        "request_id": "r0",
        "schema": ORACLE_OBSERVATION_SCHEMA,
        "token_count": 2,
        "token_slot_ids": [4, 5],
    }
    assert values[1]["kind"] == "eviction"
    assert values[1]["token_slot_ids"] == [4, 5]


def test_retraction_records_preemption_and_released_slots(monkeypatch, tmp_path):
    path = configure_sidecar(monkeypatch, tmp_path)
    r0 = FakeReq("r0", 0, FakeKv(2))
    r1 = FakeReq("r1", 1, FakeKv(3))
    batch = FakeBatch(
        reqs=[r0, r1],
        extend_lens=[],
        req_to_token_pool=FakeReqToTokenPool([[1, 2], [7, 8, 9]]),
    )

    result = _observe_retract_decode(
        lambda _batch, _args: ([r1], 0.5, []), batch, object()
    )

    assert result[0] == [r1]
    values = sidecar_rows(path)
    assert [row["kind"] for row in values] == ["preemption", "release"]
    assert values[0]["request_id"] == "r1"
    assert values[0]["token_slot_ids"] == [7, 8, 9]
    assert values[0]["reason"] == "decode-pressure"
