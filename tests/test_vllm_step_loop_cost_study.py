"""Deterministic checks of the attribution study's evidence reductions."""

import importlib.util
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_runner(directory):
    path = Path(__file__).parents[1] / "examples" / directory / "run_study.py"
    name = f"s5_test_{directory}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


oracle = load_runner("surrogate_conformance_v1")
study = load_runner("vllm_step_loop_cost_v1")


def test_nested_timer_removes_child_time_and_preserves_return():
    clock = iter([0, 10, 40, 100])
    timers = study.Timers(lambda: next(clock))
    assert timers.call("scheduler", lambda: timers.call("kv_allocate", lambda: 37)) == 37
    assert [(e["phase"], e["inclusive_ns"], e["exclusive_ns"]) for e in timers.events] == [
        ("kv_allocate", 30, 30), ("scheduler", 100, 70)]
    assert sum(e["exclusive_ns"] for e in timers.events) == 100
    assert timers.stack == []


def test_exception_unwinds_timer_and_restores_original_method():
    clock = iter([10, 30])
    timers = study.Timers(lambda: next(clock))

    def original():
        raise ValueError("test failure")

    obj = SimpleNamespace(work=original)
    with ExitStack() as stack:
        timers.wrap(stack, obj, "work", "executor")
        with pytest.raises(ValueError, match="test failure"):
            obj.work()
    assert obj.work is original
    assert timers.stack == []
    assert timers.events[0]["exclusive_ns"] == 20


def test_negative_exclusive_interval_is_fatal():
    clock = iter([0, 10, 100, 20])
    timers = study.Timers(lambda: next(clock))
    with pytest.raises(RuntimeError, match="negative exclusive"):
        timers.call("outer", lambda: timers.call("inner", lambda: None))


@pytest.mark.parametrize("values,passed", [
    ([100] * 7, True),
    ([95, 95, 95, 100, 105, 105, 105], True),
    ([94, 94, 94, 100, 106, 106, 106], False),
    ([100, 100, 100, 100, 112, 112, 112], False),
])
def test_stability_uses_both_three_run_medians(values, passed):
    assert study.stability(values)["passed"] is passed


@pytest.mark.parametrize("values", [[], [100] * 6, [0] + [100] * 6])
def test_stability_rejects_invalid_population(values):
    with pytest.raises(ValueError):
        study.stability(values)


def test_grid_preserves_entire_frozen_request_population():
    config = oracle.load_config()
    reference = oracle.wall_cell(config)
    grid = study.cells(oracle, config)
    assert len(grid) == 10
    assert grid[0].engine == reference.engine
    assert [(cell.engine["max_num_seqs"], cell.engine["budget"]) for cell in grid[1:]] == [
        (32, 256), (32, 512), (32, 1024), (64, 256), (64, 512), (64, 1024),
        (128, 256), (128, 512), (128, 1024)]
    for cell in grid:
        assert cell.requests is reference.requests or cell.requests == reference.requests
        assert study.canonical([vars(row) for row in cell.requests]) == study.canonical(
            [vars(row) for row in reference.requests])
        assert len(cell.requests) == 128
        assert sum(row.max_output_tokens for row in cell.requests) == 4096


def relation_grid():
    return [{"cell_id": f"n{cap}-b{budget}", "cap": cap, "budget": budget,
             "decode_at_cap_scheduler_ns": cap * 100,
             "kv_groups": [{"phase": "kv_allocate", "blocks": 1, "median_ns": 100},
                           {"phase": "kv_allocate", "blocks": 4, "median_ns": 250},
                           {"phase": "kv_free", "blocks": 6, "median_ns": 500}]}
            for cap in (32, 64, 128) for budget in (256, 512, 1024)]


def test_relations_reject_flat_scheduler_and_reversed_block_scaling():
    rows = relation_grid()
    assert all(row["status"] == "pass" for row in study.evaluate_relations(rows)
               if row["family"] == "R1")
    rows[3]["decode_at_cap_scheduler_ns"] = rows[0]["decode_at_cap_scheduler_ns"]
    rows[0]["kv_groups"][1]["median_ns"] = 50
    observations = study.evaluate_relations(rows)
    assert next(r for r in observations if r["instance"] == "b256:32-to-64")["status"] == "fail"
    assert next(r for r in observations if r["instance"] ==
                "n32-b256:kv_allocate:1-to-4")["status"] == "fail"


def test_missing_decode_and_constant_free_block_count_are_unscored():
    rows = relation_grid()
    rows[0]["decode_at_cap_scheduler_ns"] = None
    observed = study.evaluate_relations(rows)
    assert next(r for r in observed if r["instance"] == "b256:32-to-64")["status"] == "unevaluated"
    free = [r for r in observed if r["instance"].endswith("kv_free")]
    assert len(free) == 9
    assert all(r["status"] == "unevaluated" for r in free)


def test_evidence_is_append_only(tmp_path):
    path = tmp_path / "evidence.json"
    study.write_once(path, {"value": 1})
    with pytest.raises(FileExistsError):
        study.write_once(path, {"value": 2})
    assert path.read_bytes() == b'{"value":1}\n'


def test_profile_names_are_portable():
    assert study.portable_function("vllm/v1/engine/core.py", 584, "step") == (
        "vllm/v1/engine/core.py:584:step")
    assert study.portable_function("<frozen importlib._bootstrap>", 1, "load") == "builtin:load"
    assert study.portable_function("python3.10/threading.py", 1, "wait") == "threading.py:1:wait"
