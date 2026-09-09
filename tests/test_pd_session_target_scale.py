"""Adversarial evidence checks without importing or executing native vLLM."""

import json
import os
from copy import deepcopy
from itertools import pairwise
from types import SimpleNamespace

import pytest

from examples.pd_session_target_scale_v1 import checks, run_study
from examples.pd_session_target_scale_v1.native import ObservedClock
from simllm.core import DeclaredKvHandoffPolicy
from simllm.core.pd_session import DisaggregatedRequestTimeline

FROZEN = json.loads((run_study.HERE / "expectations.json").read_bytes())
SCALE = FROZEN["scales"][0]


def evidence():
    return checks.Evidence(FROZEN["evidence_domains"]["stages"])


@pytest.mark.parametrize("value", [True, False, 1.0, -1, None, "1"])
def test_integer_domain_excludes_coercions(value):
    assert not checks.integer(value)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1, 0, "1"])
def test_host_values_are_finite_and_positive(value):
    assert not checks.finite_positive(value)


@pytest.mark.parametrize("condition", [False, 1, None, "yes"])
def test_guard_requires_actual_boolean_true(condition):
    ledger = evidence()
    with pytest.raises(checks.GuardFailure, match="guard"):
        ledger.check("guard", condition)
    result = run_study.summary(ledger, FROZEN)
    assert result["verdict"] == "VOID"
    assert result["behavioral_score"] is None
    assert result["evidence"]["unscored_guards"] == [{"name": "guard", "passed": False}]


def test_duplicate_names_stages_and_unknown_stage_are_fatal():
    ledger = evidence()
    ledger.check("guard", True)
    with pytest.raises(checks.GuardFailure, match="duplicate"):
        ledger.check("guard", True, kind="oracles")
    ledger.finish("protocol")
    with pytest.raises(checks.GuardFailure):
        ledger.finish("protocol")
    with pytest.raises(checks.GuardFailure):
        evidence().finish("unfrozen-stage")
    with pytest.raises(checks.GuardFailure):
        evidence().check("bad-class", True, kind="score")


def test_missing_stage_voids_even_when_other_stages_are_declared_done():
    ledger = evidence()
    ledger.finished_stages = ledger.expected_stages - {"p16-d40:retention"}
    with pytest.raises(checks.GuardFailure, match="complete:stages"):
        run_study.completion(ledger, FROZEN)
    assert run_study.summary(ledger, FROZEN)["behavioral_score"] is None


def test_type_aware_equality_does_not_accept_bool_as_integer():
    with pytest.raises(checks.GuardFailure):
        evidence().equal("rank", True, 1)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":1}', b'{"x": 1}', b'{"x":NaN}', b'{'])
def test_noncanonical_receipt_is_rejected_without_rewriting(tmp_path, raw):
    path = tmp_path / "receipt.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        run_study.read(path)
    assert path.read_bytes() == raw


@pytest.mark.parametrize(("intervals", "union"), [
    ([], 0), ([(1, 5), (2, 3)], 4), ([(5, 9), (1, 5)], 8),
    ([(1, 5), (3, 8), (12, 15)], 10), ([(1, 5), (1, 5)], 4),
])
def test_service_union_is_distinct_from_additive_work(intervals, union):
    assert checks.interval_union(intervals) == union


def test_clock_observer_keeps_the_real_clock_as_authority():
    clock = ObservedClock()
    clock.advance_to(3)
    clock.advance_to(3)
    clock.advance_to(7)
    assert clock.now_ps == 7
    assert clock.advances == [{"before_ps": 0, "after_ps": 3},
                              {"before_ps": 3, "after_ps": 3},
                              {"before_ps": 3, "after_ps": 7}]


@pytest.fixture
def retained():
    engines, projected, construction = [], [], []
    for ordinal, role in enumerate(("prefill", "decode")):
        base, engine_id = 1000 * (ordinal + 1), f"simllm-{role}-0"
        workers = [{"object_id": base + 100 + local, "rank": local, "local_rank": local,
                    "config_id": base + 6, "device_absent": True, "model_runner_absent": True}
                   for local in range(8)]
        row = {
            "engine_id": engine_id, "role": role, "ordinal": 0,
            "engine_object_id": base + 1, "frontend_object_id": base + 2,
            "core_object_id": base + 3, "executor_object_id": base + 4,
            "frontend_executor_object_id": base + 4, "sink_object_id": base + 5,
            "clock_object_id": 1, "runtime_clock_object_id": 1,
            "config_object_id": base + 6, "frontend_config_object_id": base + 6,
            "world_size": 8, "executor_role": role,
            "connector_role": "kv_producer" if role == "prefill" else "kv_consumer",
            "workers": workers, "executor_workers": deepcopy(workers),
            "local_placement": [{"rank": local, "local_rank": local, "role": role,
                                 "gpu_uuid": f"sim-{role}-gpu-{local}", "tp_ranks": list(range(8))}
                                for local in range(8)],
        }
        engines.append(row)
        construction.append({"identity": deepcopy(row), "index": ordinal, "clock_ps": 0,
                             "all_observed_objects_alive": True, "current_rss_kib": 100,
                             "peak_rss_kib": 110, "elapsed_ns": ordinal + 1,
                             "native_constructor_seconds": 0.1})
        for local in range(8):
            rank = 8 * ordinal + local
            projected.append({"engine_id": engine_id, "role": role, "manifest_role": role,
                              "ordinal": 0, "worker_object_id": workers[local]["object_id"],
                              "local_rank": local, "manifest_local_rank": local, "global_rank": rank,
                              "gpu_id": f"sim-gpu-{rank:04d}", "gpu_uuid": f"sim-gpu-{rank:04d}",
                              "hostname": role + "-node-0", "gpu_node_id": role + "-node-0", "nic_node_id": role + "-node-0",
                              "nic_id": f"sim-nic-{rank:04d}", "nic_affine_rank": rank,
                              "tp_ranks": list(range(8 * ordinal, 8 * (ordinal + 1)))})
    return {"retention_before": engines, "retention_after": deepcopy(engines),
            "construction_rows": construction, "worker_manifest_projection": projected,
            "native_identity": {"clock_object_id": 1, "observer_weak_references_only": True,
                                "all_observed_objects_alive": True}}


def test_retention_requires_frontend_workers_and_read_only_global_join(retained):
    ledger = evidence()
    checks.check_retention(retained, SCALE, FROZEN, ledger)
    assert ledger.finished_stages == {"p1-d1:construction", "p1-d1:retention"}


@pytest.mark.parametrize(("mutation", "guard"), [
    ("missing-engine", "retention-order"), ("dead-object", "weak-observer"),
    ("wrong-clock", "shared-clock"), ("wrong-frontend", "frontend-executor"),
    ("worker-device", "worker-contract"), ("wrong-rank", "global-rank-domain"),
])
def test_retention_semantic_corruptions(retained, mutation, guard):
    if mutation == "missing-engine":
        retained["retention_before"].pop()
    elif mutation == "dead-object":
        retained["native_identity"]["all_observed_objects_alive"] = False
    elif mutation == "wrong-rank":
        retained["worker_manifest_projection"][0]["global_rank"] = True
    else:
        for row in (retained["retention_before"][0], retained["retention_after"][0]):
            if mutation == "wrong-clock":
                row["runtime_clock_object_id"] += 1
            elif mutation == "wrong-frontend":
                row["frontend_executor_object_id"] += 1
            else:
                row["workers"][0]["device_absent"] = False
                row["executor_workers"][0]["device_absent"] = False
    with pytest.raises(checks.GuardFailure, match=guard):
        checks.check_retention(retained, SCALE, FROZEN, evidence())


def build_execution(*, idle_during_ready_burst=False):
    """Serial cells and a common-arrival burst that keeps ready engines busy."""
    data = {"serial_cells": [], "baseline_controls": [], "unique_steps": [], "clock_advances": []}
    now, indices = 0, {"prefill": 0, "decode": 0}
    for spec in [*FROZEN["serial_cells"], FROZEN["burst"]]:
        cell_start, first_advance, first_step, rows = now, len(data["clock_advances"]), len(data["unique_steps"]), []
        accepted = next(row for row in FROZEN["baseline_control"]["accepted_cells"]
                        if row["prompt_tokens"] == spec["prompt_tokens"] and row["handoff_ps"] == spec["handoff_ps"])
        intervals = []
        is_burst = spec is FROZEN["burst"]
        serialize = not is_burst or idle_during_ready_burst
        for ordinal in range(80):
            request_id = f"{spec['id']}:request-{ordinal}"
            start = now if serialize else cell_start + ordinal * accepted["prefill_service_ps"]
            prefill_stop = start + accepted["prefill_service_ps"]
            handoff = DeclaredKvHandoffPolicy(spec["handoff_ps"]).schedule(
                submitted_at_ps=prefill_stop, request_id=request_id, kv_bytes=accepted["kv_bytes"])
            decode_start = (cell_start + 80 * accepted["prefill_service_ps"] + ordinal * 4 * accepted["tpot_ps"]
                            if not serialize else handoff.completed_at_ps)
            completions = tuple(decode_start + (token + 1) * accepted["tpot_ps"] for token in range(4))
            timeline = DisaggregatedRequestTimeline(
                request_id=request_id, admitted_at_ps=cell_start if spec is FROZEN["burst"] else start,
                prefill_eligible_at_ps=start, prefill_completed_at_ps=prefill_stop,
                handoff=handoff, decode_eligible_at_ps=decode_start, decode_token_completed_at_ps=completions)
            raw = timeline.to_json()
            record_indices = {}
            for role, boundaries in (("prefill", [start, prefill_stop]), ("decode", [decode_start, *completions])):
                record_indices[role + "_record_indices"] = []
                raw[role + "_engine_id"] = f"simllm-{role}-0"
                raw[role + "_internal_request_id"] = request_id + ":" + role
                raw[role + "_step_count"] = len(boundaries) - 1
                for visit, (lower, upper) in enumerate(pairwise(boundaries)):
                    index = indices[role]
                    record_indices[role + "_record_indices"].append(index)
                    data["unique_steps"].append({"engine_id": raw[role + "_engine_id"],
                        "record": {"schema": "atlahs-closed-loop-step-v1", "step_index": index,
                                   "virtual_time_ps": lower, "num_sampled": 1,
                                   "preempted_request_ids": [], "finished_request_ids": [],
                                   "scheduled": [{"request_id": raw[role + "_internal_request_id"],
                                                  "phase": "prefill" if role == "prefill" or visit == 0 else "decode",
                                                  "num_new_tokens": spec["prompt_tokens"] if role == "prefill" else 1,
                                                  "num_cached_tokens": spec["prompt_tokens"] if role == "decode" and visit == 0 else 0,
                                                  "context_length": spec["prompt_tokens"] + (visit + 1 if role == "decode" else 0)}]},
                        "result": {"step_index": index, "step_latency_ps": upper - lower, "completed_at_ps": upper,
                                   "request_metrics": [], "additive_visit_totals": None}})
                    intervals.append((lower, upper))
                    indices[role] += 1
            raw.update(bootstrap_token_id=512, decode_token_ids=[512] * 4,
                       kv_transfer_params={"schema": "simllm-pd-kv-params-v1", "do_remote_prefill": True,
                                           "do_remote_decode": False, "remote_engine_id": raw["prefill_engine_id"],
                                           "remote_request_id": request_id, "session_request_id": request_id,
                                           "remote_num_tokens": spec["prompt_tokens"], "bootstrap_token_id": 512,
                                           "worker_tensor_transfer": False, "timing_authority": "simllm-declared-kv-handoff-v1"})
            rows.append({"result": raw, **record_indices})
            now = completions[-1]
        cursor = cell_start
        for lower, upper in sorted(intervals):
            if cursor < lower:
                data["clock_advances"].append({"before_ps": cursor, "after_ps": lower})
            data["clock_advances"].append({"before_ps": lower, "after_ps": upper})
            cursor = upper
        cell = {"id": spec["id"], "requests": rows, "start_ps": cell_start, "end_ps": now,
                "wall_time_ns": 100, "wall_seconds": 100 / 1e9,
                "arrival_mode": "simultaneous-burst" if spec is FROZEN["burst"] else "serial-completion-admission",
                "offered_rate_requests_per_second": None,
                "engine_request_counts": {role: {f"simllm-{role}-0": 80} for role in indices},
                "step_identities": [[row["engine_id"], row["record"]["step_index"]]
                                    for row in data["unique_steps"][first_step:]],
                "clock_advance_indices": list(range(first_advance, len(data["clock_advances"])))}
        if spec is FROZEN["burst"]:
            data["burst_cell"] = cell
        else:
            data["serial_cells"].append(cell)
    data["unique_steps"].sort(key=lambda row: (0 if "prefill" in row["engine_id"] else 1, row["record"]["step_index"]))
    terminal = {row["result"][role + "_internal_request_id"]:
                row["result"]["prefill_completed_at_ps"] if role == "prefill"
                else row["result"]["decode_token_completed_at_ps"][-1]
                for cell in [*data["serial_cells"], data["burst_cell"]]
                for row in cell["requests"] for role in ("prefill", "decode")}
    prior_finished = {}
    for step in data["unique_steps"]:
        engine = step["engine_id"]
        step["record"]["finished_request_ids"] = prior_finished.get(engine, [])
        prior_finished[engine] = sorted(item["request_id"] for item in step["record"]["scheduled"]
                                        if terminal[item["request_id"]] == step["result"]["completed_at_ps"])
    burst = data["burst_cell"]
    for row in burst["requests"]:
        for role in ("prefill", "decode"):
            candidates = [step["record"]["step_index"] for step in data["unique_steps"]
                          if step["engine_id"] == row["result"][role + "_engine_id"]
                          and step["record"]["virtual_time_ps"] >= (
                              burst["start_ps"] if role == "prefill" else row["result"]["handoff"]["completed_at_ps"])]
            stop = row[role + "_record_indices"][-1] + 1
            row[role + "_record_indices"] = list(range(min(candidates), stop))
            row["result"][role + "_step_count"] = len(row[role + "_record_indices"])
    data["host_measurements"] = {"final_clock_ps": now}
    return data


@pytest.fixture
def execution():
    return build_execution()


def test_unique_step_partition_and_joint_relations(execution):
    ledger = evidence()
    requests, metrics = checks.check_cells(execution, SCALE, FROZEN, ledger)
    accounting = checks.check_accounting(execution, SCALE, FROZEN, requests, ledger)
    checks.check_relations(SCALE, FROZEN, metrics, execution["serial_cells"], ledger)
    assert accounting["unique_steps"] == 2000
    assert accounting["serial_handoff_ps"] == 80 * 600000000
    assert accounting["burst_wall_idle_ps"] == 0
    assert len(ledger.oracles) == 4
    assert len(ledger.relations) == 4
    assert accounting["final_clock_ps"] == sum(accounting[key] for key in (
        "unique_service_ps", "serial_handoff_ps", "burst_wall_idle_ps"))


@pytest.mark.parametrize(("mutation", "guard"), [
    ("foreign", "member"), ("duplicate", "unique-step-domain"), ("shift", ":result"),
    ("clock", ":clock:"), ("overlap", "cell-steps"), ("cell-clock", "cell-clock-indices"),
])
def test_unique_service_and_clock_corruptions(execution, mutation, guard):
    ledger = evidence()
    requests, _ = checks.check_cells(execution, SCALE, FROZEN, ledger)
    if mutation == "foreign":
        execution["unique_steps"][0]["record"]["scheduled"][0]["request_id"] = "foreign"
    elif mutation == "duplicate":
        execution["unique_steps"].append(deepcopy(execution["unique_steps"][0]))
    elif mutation == "shift":
        execution["unique_steps"][0]["result"]["completed_at_ps"] += 1
    elif mutation == "clock":
        execution["clock_advances"].pop(0)
    elif mutation == "overlap":
        cell = execution["serial_cells"][0]
        cell["step_identities"].append(cell["step_identities"][0])
    else:
        execution["serial_cells"][1]["clock_advance_indices"].pop(0)
    with pytest.raises(checks.GuardFailure, match=guard):
        checks.check_accounting(execution, SCALE, FROZEN, requests, ledger)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "invented-rate", "unknown-field"])
def test_closed_request_and_cell_population(execution, mutation):
    cell = execution["serial_cells"][0]
    if mutation == "missing":
        cell["requests"].pop()
    elif mutation == "extra":
        cell["requests"].append(deepcopy(cell["requests"][0]))
    elif mutation == "duplicate":
        cell["requests"][1] = deepcopy(cell["requests"][0])
    elif mutation == "invented-rate":
        cell["offered_rate_requests_per_second"] = 1.0
    else:
        cell["future"] = True
    with pytest.raises(checks.GuardFailure):
        checks.check_cells(execution, SCALE, FROZEN, evidence())


@pytest.mark.parametrize(("mutation", "guard"), [
    ("ghost-engine", "engine-domain"), ("swapped-cells", "cell-step-ownership|exact-slice"),
    ("extra-slice", "slice-domain"), ("wrong-count", "slice-count"),
    ("token-count", "token-contract"), ("wrong-phase", "token-contract"),
    ("foreign-host", "projection"), ("missing-hostname", "projection-fields"),
    ("missing-step-field", "record-fields"),
])
def test_reviewed_admission_counterexamples(execution, retained, mutation, guard):
    ledger = evidence()
    if mutation in ("foreign-host", "missing-hostname"):
        row = retained["worker_manifest_projection"][0]
        if mutation == "foreign-host":
            for key in ("hostname", "gpu_node_id", "nic_node_id"):
                row[key] = "foreign-node"
        else:
            row.pop("hostname")
        with pytest.raises(checks.GuardFailure, match=guard):
            checks.check_retention(retained, SCALE, FROZEN, ledger)
        return
    if mutation == "ghost-engine":
        ghost = deepcopy(execution["unique_steps"][0])
        ghost["engine_id"] = "foreign-engine"
        ghost["record"].update(scheduled=[], num_sampled=0)
        ghost["result"].update(step_latency_ps=0, completed_at_ps=ghost["record"]["virtual_time_ps"])
        execution["unique_steps"].append(ghost)
        execution["serial_cells"][0]["step_identities"].append(["foreign-engine", 0])
    elif mutation == "swapped-cells":
        first, second = execution["serial_cells"][:2]
        first["step_identities"], second["step_identities"] = second["step_identities"], first["step_identities"]
    elif mutation == "extra-slice":
        execution["serial_cells"][0]["requests"][0]["prefill_record_indices"].append(99999)
    elif mutation == "wrong-count":
        execution["serial_cells"][0]["requests"][0]["result"]["prefill_step_count"] += 1
    elif mutation == "token-count":
        execution["unique_steps"][0]["record"]["scheduled"][0]["num_new_tokens"] += 1
    elif mutation == "wrong-phase":
        execution["unique_steps"][0]["record"]["scheduled"][0]["phase"] = "decode"
    else:
        execution["unique_steps"][0]["record"].pop("num_sampled")
    requests, _ = checks.check_cells(execution, SCALE, FROZEN, ledger)
    with pytest.raises(checks.GuardFailure, match=guard):
        checks.check_accounting(execution, SCALE, FROZEN, requests, ledger)


def test_burst_cannot_idle_while_other_prefill_requests_are_ready():
    data = build_execution(idle_during_ready_burst=True)
    ledger = evidence()
    requests, _ = checks.check_cells(data, SCALE, FROZEN, ledger)
    with pytest.raises(checks.GuardFailure, match="burst-idle-no-ready-work"):
        checks.check_accounting(data, SCALE, FROZEN, requests, ledger)


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_coherently_shortened_shared_slice_is_rejected(execution, role):
    row = execution["burst_cell"]["requests"][-1]
    row[role + "_record_indices"].pop(0)
    row["result"][role + "_step_count"] -= 1
    ledger = evidence()
    requests, _ = checks.check_cells(execution, SCALE, FROZEN, ledger)
    with pytest.raises(checks.GuardFailure, match="exact-slice"):
        checks.check_accounting(execution, SCALE, FROZEN, requests, ledger)


@pytest.mark.parametrize(("mutation", "guard"), [
    ("empty-step", "justified-drain"), ("dropped-finish", "finished-lag"),
    ("duplicate-finish", "finished-domain"), ("premature-finish", "finished-lag"),
    ("request-metrics", "inactive-request-metrics"), ("visit-totals", "inactive-visit-totals"),
])
def test_completion_metadata_and_inactive_event_contract(execution, mutation, guard):
    if mutation == "empty-step":
        row = deepcopy(execution["unique_steps"][-1])
        row["record"].update(step_index=row["record"]["step_index"] + 1,
                             scheduled=[], num_sampled=0, finished_request_ids=[],
                             virtual_time_ps=execution["burst_cell"]["end_ps"])
        row["result"].update(step_index=row["record"]["step_index"], step_latency_ps=0)
        execution["unique_steps"].append(row)
        execution["burst_cell"]["step_identities"].append([row["engine_id"], row["record"]["step_index"]])
    elif mutation == "dropped-finish":
        execution["unique_steps"][1]["record"]["finished_request_ids"] = []
    elif mutation == "duplicate-finish":
        record = execution["unique_steps"][1]["record"]
        record["finished_request_ids"] *= 2
    elif mutation == "premature-finish":
        execution["unique_steps"][0]["record"]["finished_request_ids"] = execution["unique_steps"][1]["record"]["finished_request_ids"]
    elif mutation == "request-metrics":
        execution["unique_steps"][0]["result"]["request_metrics"] = [{"request_id": "fabricated"}]
    else:
        execution["unique_steps"][0]["result"]["additive_visit_totals"] = {}
    ledger = evidence()
    requests, _ = checks.check_cells(execution, SCALE, FROZEN, ledger)
    with pytest.raises(checks.GuardFailure, match=guard):
        checks.check_accounting(execution, SCALE, FROZEN, requests, ledger)


@pytest.mark.parametrize("mutation", ["authority", "pricing_arm", "queue", "visibility", "connector"])
def test_declared_handoff_rejects_false_authority_and_wait_splits(execution, mutation):
    raw = execution["serial_cells"][0]["requests"][0]["result"]
    if mutation in ("authority", "pricing_arm"):
        raw["handoff"][mutation] = "physical"
    elif mutation == "queue":
        raw["handoff"]["started_at_ps"] += 1
    elif mutation == "visibility":
        raw["handoff"]["finished_at_ps"] -= 1
    else:
        raw["kv_transfer_params"]["schema"] = "unknown"
    with pytest.raises(checks.GuardFailure, match="declared-handoff|connector"):
        checks.check_cells(execution, SCALE, FROZEN, evidence())


def test_ordinary_failure_retains_void_and_exception(tmp_path, monkeypatch):
    output = tmp_path / "new-run"
    monkeypatch.setattr(run_study, "protocol", lambda *args: (_ for _ in ()).throw(RuntimeError("injected")))
    assert run_study.execute(SimpleNamespace(output_root=output)) == 2
    retained = run_study.read(output / "summary.json")
    assert retained["verdict"] == "VOID" and retained["behavioral_score"] is None
    assert retained["failure"] == {"type": "RuntimeError", "reason": "injected"}
    assert "injected" in (output / "exception.txt").read_text()
    before = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_study.execute(SimpleNamespace(output_root=output))
    assert (output / "summary.json").read_bytes() == before


@pytest.fixture
def admitted(tmp_path, retained, execution):
    """A schema-complete artifact for testing admission, without native execution."""
    frozen = deepcopy(FROZEN)
    frozen["native_source_sha256"], frozen["current_source_sha256"] = {}, {}
    config = tmp_path / "config.json"
    config.write_bytes(b"{}")
    args = SimpleNamespace(native_python=tmp_path / "environment" / "python",
                           vllm_source=tmp_path / "vllm", output_root=tmp_path)
    path = tmp_path / SCALE["id"]
    path.mkdir()
    manifest_hashes = {}
    for name in ("placement.json", "fabric.json"):
        (path / name).write_bytes(b"{}")
        manifest_hashes[name] = run_study.sha(b"{}")
    pid = os.getpid() + 1000
    data = {**retained, **execution, "schema": "simllm-pd-session-target-scale-native-v1",
            "scale_id": SCALE["id"], "source_before": {}, "source_after": {}}
    data["native_identity"].update(
        pid=pid, interpreter=str(args.native_python), package=str(args.vllm_source),
        version=frozen["frontend"]["version"], config_sha256=run_study.sha(config.read_bytes()),
        config_path=str(config), freeze_sha256=run_study.sha((run_study.HERE / "expectations.json").read_bytes()),
        cuda_initialized_before=False, cuda_initialized_after=False, packet_backend_runs=0,
        native_origins={}, simllm_origins={}, simllm_source_before={}, simllm_source_after={},
        unfinished_requests_after=[False, False], collective_arms=[["intra-node-fixed-cost-v1", "lower"]])
    data["host_measurements"].update(
        current_rss_before_kib=100, peak_rss_before_kib=110, current_rss_constructed_kib=120,
        current_rss_after_requests_kib=120, peak_rss_after_requests_kib=130,
        construction_wall_time_ns=100, request_wall_time_ns=500, request_count=400)
    return data, frozen, args, config, {"pid": pid}, manifest_hashes


def test_complete_admission_without_gpu_or_packet_authority(admitted):
    data, frozen, args, config, monitor, manifests = admitted
    run_study.admit(data, SCALE, frozen, args, config, monitor, manifests, evidence())


@pytest.mark.parametrize(("mutation", "guard"), [
    ("missing-memory", "host-fields"), ("extra-identity", "identity-fields"),
    ("wrong-interpreter", "interpreter"), ("wrong-source", "source_before"),
    ("gpu", "gpu-after"), ("cap", "rss-cap"),
])
def test_native_admission_failures(admitted, mutation, guard):
    data, frozen, args, config, monitor, manifests = admitted
    if mutation == "missing-memory":
        data["host_measurements"].pop("current_rss_constructed_kib")
    elif mutation == "extra-identity":
        data["native_identity"]["unknown"] = True
    elif mutation == "wrong-interpreter":
        data["native_identity"]["interpreter"] = str(args.native_python.parent / "other-python")
    elif mutation == "wrong-source":
        data["source_before"]["extra.py"] = "0" * 64
    elif mutation == "gpu":
        data["native_identity"]["cuda_initialized_after"] = True
    else:
        data["host_measurements"]["current_rss_constructed_kib"] = frozen["construction"]["stop_rss_kib"]
    with pytest.raises(checks.GuardFailure, match=guard):
        run_study.admit(data, SCALE, frozen, args, config, monitor, manifests, evidence())
