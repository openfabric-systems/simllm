"""Deterministic checks of the attribution study's evidence reductions."""

import importlib.util
import json
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


def test_version_pin_accepts_only_the_exact_supplied_cpu_build():
    study.require_versions("0.27.1+cpu", "0.27.1")
    for distribution, module in (("0.27.1", "0.27.1"), ("0.27.2+cpu", "0.27.1"),
                                 ("0.27.1+cpu", "0.27.2")):
        with pytest.raises(RuntimeError, match="pin mismatch"):
            study.require_versions(distribution, module)


def test_scheduler_observer_preserves_native_throttle_argument():
    output = SimpleNamespace(num_scheduled_tokens={"r0": 1})
    seen = []

    def schedule(throttle_prefills=False):
        seen.append(throttle_prefills)
        return output

    manager = SimpleNamespace(allocate_slots=lambda: None, free=lambda: None)
    scheduler = SimpleNamespace(running=["r0"], schedule=schedule, kv_cache_manager=manager,
                                update_from_output=lambda: None)
    executor = SimpleNamespace(execute_model=lambda: None, sample_tokens=lambda: None)
    core = SimpleNamespace(scheduler=scheduler, model_executor=executor)
    engine = SimpleNamespace(engine_core=SimpleNamespace(engine_core=core),
                             output_processor=SimpleNamespace(process_outputs=lambda: None),
                             add_request=lambda: None,
                             step=lambda: scheduler.schedule(False))
    driver = SimpleNamespace(_observe_outputs=lambda: None)
    llm = SimpleNamespace(llm_engine=engine)
    timers = study.Timers()
    with ExitStack() as stack:
        study.install_timers(timers, stack, driver, llm, None)
        assert engine.step() is output
    assert seen == [False]
    assert scheduler.schedule is schedule
    event = next(e for e in timers.events if e["phase"] == "scheduler")
    assert event["running_before"] == event["scheduled"] == 1


def test_device_guard_accepts_explicit_none_and_rejects_device_work():
    assert study.device_is_absent(None, False)
    assert not study.device_is_absent("cuda:0", False)
    assert not study.device_is_absent("cpu", False)
    assert not study.device_is_absent(None, True)


def test_profile_names_distinguish_adapter_from_upstream_framework():
    assert study.portable_function("simllm/adapters/vllm/worker.py", 487, "execute_model") == (
        "simllm/adapters/vllm/worker.py:487:execute_model")


def controlled_snapshot():
    return {"affinity": list(range(24, 32)), "nproc": 8,
            "torch_intraop_threads": 1, "torch_interop_threads": 1,
            "thread_environment": dict.fromkeys(study.THREAD_ENV, "1")}


@pytest.mark.parametrize("field,value", [
    ("affinity", list(range(8))), ("affinity", list(range(24, 31))),
    ("nproc", 7), ("torch_intraop_threads", 2), ("torch_interop_threads", 2),
    ("thread_environment", dict.fromkeys(study.THREAD_ENV, "2")),
])
@pytest.mark.parametrize("boundary", ["before", "after"])
def test_host_guard_refuses_protocol_drift_at_either_boundary(field, value, boundary):
    snapshots = {"before": controlled_snapshot(), "after": controlled_snapshot()}
    assert study.host_guard(**snapshots)
    snapshots[boundary][field] = value
    assert not study.host_guard(**snapshots)


def test_host_inventory_retains_load_and_does_not_change_affinity(monkeypatch):
    calls = []

    def output(command, **kwargs):
        calls.append((command, kwargs))
        return "8\n" if command == ["nproc"] else "pid 123's current affinity mask: ff000000\n"

    monkeypatch.setattr(study.subprocess, "check_output", output)
    monkeypatch.setattr(study.os, "getloadavg", lambda: (11.0, 12.0, 13.0), raising=False)
    monkeypatch.setattr(study.os, "sched_getaffinity", lambda pid: set(range(24, 32)),
                        raising=False)
    monkeypatch.setattr(study.os, "getpid", lambda: 123)
    for name in study.THREAD_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("OMP_THREAD_LIMIT", "1")
    torch = SimpleNamespace(get_num_threads=lambda: 1, get_num_interop_threads=lambda: 1)
    snapshot = study.host_snapshot(torch)
    assert snapshot["load_average"] == [11, 12, 13]
    assert snapshot["taskset_output"].endswith("ff000000")
    assert calls[0][0] == ["taskset", "-p", "123"]
    assert "OMP_NUM_THREADS" not in calls[1][1]["env"]
    assert "OMP_THREAD_LIMIT" not in calls[1][1]["env"]
    assert study.os.environ["OMP_NUM_THREADS"] == "1"
    assert study.host_guard(snapshot, snapshot)


def test_text_digest_accepts_crlf_but_refuses_content_changes():
    frozen = b'{"value": 1}\n'
    converted = b'{"value": 1}\r\n'
    assert study.text_matches(converted, frozen)
    assert study.text_hashes(converted)["lf"] == study.text_hashes(frozen)["raw"]
    assert not study.text_matches(b'{"value": 2}\r\n', frozen)
    assert study.text_hashes(b'{"value": 2}\r\n')["lf"] != study.text_hashes(frozen)["raw"]


def test_publication_retains_void_record_without_rescoring_or_overwriting(tmp_path, monkeypatch):
    attempt = tmp_path / "controlled"
    prior = tmp_path / "prior.json"
    output = tmp_path / "published.json"
    previous = {"attempt": "prior", "status": "void", "fatal_violations": ["stability"]}
    study.write_once(prior, previous)
    prior_bytes = prior.read_bytes()
    result = {"attempt": "controlled", "status": "void", "behavioral_score": None,
              "fatal_violations": ["n32-b256:stability"], "plain_ns": [100] * 7}
    study.write_once(attempt / "results.json", result)
    (attempt / "profile").mkdir()
    (attempt / "profile" / "functions.prof").write_bytes(b"profile fixture")
    functions = [{"function": "schedule", "self_ns": 10, "cumulative_ns": 20}]
    monkeypatch.setattr(study, "profile_rows", lambda path: functions)
    study.publish(attempt, output, [prior])
    published = json.loads(output.read_bytes())
    assert published["status"] == "void" and published["behavioral_score"] is None
    assert published["plain_ns"] == result["plain_ns"]
    assert published["fatal_violations"] == result["fatal_violations"]
    retained = published["publication"]["retained_attempts"][0]
    assert retained["status"] == "void" and retained["artifact"] == "prior.json"
    assert retained["text_sha256"] == study.text_hashes(prior_bytes)
    assert prior.read_bytes() == prior_bytes
    assert (attempt / "results.json").read_bytes() == study.canonical(result)
    assert output.read_bytes().endswith(b"\n") and b"\r" not in output.read_bytes()
    with pytest.raises(FileExistsError):
        study.publish(attempt, output, [prior])


def test_controlled_publication_conserves_prior_workload_and_reports_fatal_drift():
    current = json.loads((study.STUDY / "controlled_results.json").read_bytes())
    prior_path = study.STUDY / "results.json"
    prior = json.loads(prior_path.read_bytes())
    assert current["expectation_commit"] == study.CONTROLLED_FREEZE
    host = current["host_control"]
    assert host["no_concurrent_local_suite"]
    assert host["passed"] == study.host_guard(host["before"], host["after"])
    assert current["provenance"]["workload_sha256"] == prior["provenance"]["workload_sha256"]
    assert len(current["cells"]) == len(prior["cells"]) == 10
    violations = []
    for cell, old in zip(current["cells"], prior["cells"]):
        for field in ("cell_id", "cap", "budget", "resolved", "step_count",
                      "records_sha256", "outputs_sha256"):
            assert cell[field] == old[field]
        for arm in ("plain", "timed"):
            assert cell["stability"][arm] == study.stability(cell[f"{arm}_ns"])
        assert cell["fatal_guards"]["stability"] == all(
            row["passed"] for row in cell["stability"].values())
        violations.extend(f"{cell['cell_id']}:{name}" for name, passed
                          in cell["fatal_guards"].items() if not passed)
        for group in cell["kv_groups"]:
            expected = group["median_ns"] / group["blocks"] if group["blocks"] else None
            assert group["median_ns_per_block"] == expected
    if not host["passed"]:
        violations.append("controlled_host")
    assert current["fatal_violations"] == violations
    assert current["status"] == ("void" if violations else "nonvoid")
    assert (current["behavioral_score"] is None) == bool(violations)
    retained = current["publication"]["retained_attempts"][0]
    assert retained["text_sha256"]["lf"] == study.text_hashes(prior_path.read_bytes())["lf"]
