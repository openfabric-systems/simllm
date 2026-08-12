import json
import sys
from types import SimpleNamespace

from simllm.adapters.vllm import oracle
from simllm.adapters.vllm.oracle import (
    ORACLE_ENABLE_ENV,
    ORACLE_LOG_ENV,
    ORACLE_OBSERVATION_SCHEMA,
    _observe_allocate_slots,
    _observe_eviction,
    _observe_manager_init,
    _observe_prefix_hit,
    _observe_request_finish,
    _observe_worker_load,
    mark_oracle_capture_start,
    mark_oracle_request_mapping,
    mark_oracle_submission_group,
    oracle_enabled,
)


class FakeBlocks:
    def __init__(self, *groups):
        self.groups = groups

    def get_block_ids(self):
        return self.groups


def configure_sidecar(monkeypatch, tmp_path):
    path = tmp_path / "vllm-oracle.jsonl"
    monkeypatch.setenv(ORACLE_LOG_ENV, str(path))
    return path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def manager(block_size=32, num_blocks=8):
    spec = SimpleNamespace(block_size=block_size)
    group = SimpleNamespace(kv_cache_spec=spec)
    config = SimpleNamespace(kv_cache_groups=[group], num_blocks=num_blocks)
    return SimpleNamespace(kv_cache_config=config)


def test_vllm_oracle_gate_and_request_markers(monkeypatch, tmp_path):
    path = configure_sidecar(monkeypatch, tmp_path)
    assert oracle_enabled({}) is False
    assert oracle_enabled({ORACLE_ENABLE_ENV: "1"}) is True

    mark_oracle_capture_start(("r0", "r1"))
    mark_oracle_submission_group(0, ("r0", "r1"))
    mark_oracle_request_mapping(0, {"0": "r0", "1": "r1"})

    values = rows(path)
    assert [value["kind"] for value in values] == [
        "capture-start",
        "submission-group-start",
        "request-mapping",
    ]
    assert values[-1]["mappings"] == [
        {"internal_request_id": "0", "request_id": "r0"},
        {"internal_request_id": "1", "request_id": "r1"},
    ]
    assert all(value["schema"] == ORACLE_OBSERVATION_SCHEMA for value in values)


def test_vllm_kv_hooks_project_actual_results(monkeypatch, tmp_path):
    path = configure_sidecar(monkeypatch, tmp_path)
    cache = manager()

    _observe_manager_init(lambda _manager: None, cache)
    request = SimpleNamespace(request_id="internal-0")
    hit = FakeBlocks([3, 4])
    returned_hit = _observe_prefix_hit(
        lambda _manager, _request: (hit, 64, 0), cache, request
    )
    allocated = FakeBlocks([7, 8])
    returned_allocated = _observe_allocate_slots(
        lambda _manager, _request, _tokens: allocated,
        cache,
        request,
        12,
    )

    assert returned_hit == (hit, 64, 0)
    assert returned_allocated is allocated
    values = rows(path)
    assert values[0]["token_capacity"] == 256
    assert values[1]["block_ids"] == [3, 4]
    assert values[1]["token_count"] == 64
    assert values[2]["block_ids"] == [7, 8]
    assert values[2]["token_count"] == 64


def test_vllm_eviction_hook_records_only_real_eviction(monkeypatch, tmp_path):
    path = configure_sidecar(monkeypatch, tmp_path)
    pool = SimpleNamespace(hash_block_size=32)
    block = SimpleNamespace(block_id=11)

    assert _observe_eviction(lambda _pool, _block: False, pool, block) is False
    assert not path.exists()
    assert _observe_eviction(lambda _pool, _block: True, pool, block) is True

    assert rows(path) == [
        {
            "block_ids": [11],
            "kind": "eviction",
            "reason": "prefix-cache-capacity",
            "request_id": None,
            "schema": ORACLE_OBSERVATION_SCHEMA,
            "token_count": 32,
        }
    ]


def test_vllm_final_counter_is_scheduler_owned(monkeypatch, tmp_path):
    path = configure_sidecar(monkeypatch, tmp_path)
    request = SimpleNamespace(request_id="internal-0", num_preemptions=3)

    returned = _observe_request_finish(
        lambda _scheduler, _request, *, delay_free_blocks: delay_free_blocks,
        SimpleNamespace(),
        request,
        delay_free_blocks=True,
    )

    assert returned is True
    assert rows(path) == [
        {
            "kind": "request-final-counters",
            "num_preemptions": 3,
            "request_id": "internal-0",
            "schema": ORACLE_OBSERVATION_SCHEMA,
        }
    ]


def test_vllm_worker_qualification_records_zero_cuda_allocation_delta(
    monkeypatch, tmp_path
):
    path = configure_sidecar(monkeypatch, tmp_path)
    fake_cuda = SimpleNamespace(
        is_available=lambda: False,
        memory_allocated=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(oracle, "_worker_qualified", False)
    model = SimpleNamespace(
        parameters=lambda: (SimpleNamespace(device="cpu"),)
    )
    runner = SimpleNamespace(get_model=lambda: model)
    worker = SimpleNamespace(model_runner=runner)

    returned = _observe_worker_load(lambda _worker: "loaded", worker)

    assert returned == "loaded"
    assert rows(path) == [
        {
            "cuda_available_after": False,
            "cuda_available_before": False,
            "cuda_memory_allocated_after": 0,
            "cuda_memory_allocated_before": 0,
            "kind": "worker-qualified",
            "model_class": "SimpleNamespace",
            "model_runner_class": "SimpleNamespace",
            "parameter_devices": ["cpu"],
            "schema": ORACLE_OBSERVATION_SCHEMA,
            "worker_class": "SimpleNamespace",
        }
    ]


def test_vllm_cpu_select_captures_returned_ids_without_replacement(monkeypatch):
    captured = []
    capturer = SimpleNamespace(capture=lambda layer_id, ids: captured.append((layer_id, ids)))
    selected = object()
    weights = object()
    monkeypatch.setattr(oracle, "_active_dispatch_capturer", capturer)
    token = oracle._active_cpu_layer_id.set(7)
    try:
        result = oracle._observe_cpu_select(
            lambda *_args, **_kwargs: (weights, selected), object()
        )
    finally:
        oracle._active_cpu_layer_id.reset(token)

    assert result == (weights, selected)
    assert captured == [(7, selected)]
