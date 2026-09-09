"""Reject changed evidence while preserving source-bound model identities."""

import json
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
from test_flow_session import Peer, Stream

from examples.completion_boundary_v1.run_study import save_transcript
from examples.pd_session_target_scale_v1.checks import Evidence, GuardFailure
from examples.shared_kv_handoff_v1 import native, wire
from examples.shared_kv_handoff_v1.capture import CaptureClock, EvidenceJournal, ObservationOrder
from examples.shared_kv_handoff_v1.checks import native_schedule
from examples.shared_kv_handoff_v1.common import HERE, PUBLICATIONS, plain, read, write
from examples.shared_kv_handoff_v1.journal_checks import (
    check_journals,
    check_runtime_visibility,
    check_serialized,
    reconstruct,
)
from examples.shared_kv_handoff_v1.order_checks import check_order, packet_call
from simllm.backends import flow_session
from simllm.backends.flow_session import FlowSession, FlowSessionConfig
from simllm.core.engine_steps import EngineStepRuntime
from simllm.core.step import RequestPhase, ScheduledRequest, StepRecord, StepResult


def empty_journal_data():
    owners = ["simllm-prefill-0", "simllm-decode-0"]
    engine_keys, runtime_keys = ("records", "results", *PUBLICATIONS), ("events", "visits", "completed", "clock_advances")
    data = {"process_id": "fixture", "retained_before": [{"engine_id": owner} for owner in owners], "mode": "independent",
            "sinks": {owner: {name: [] for name in PUBLICATIONS} for owner in owners}, "steps": [],
            "projections": {name: [] for name in ("events", "visits", "completed")}, "clock_advances": [],
            "checkpoints": [], "serialized_observations": [], "final_clock_ps": 0}
    data["engine_evidence"] = [{"index": index, "engine_id": owner, "phase": "partial", "at_ps": 0,
        "starts": dict.fromkeys(engine_keys, 0), "stops": dict.fromkeys(engine_keys, 0),
        "values": {key: [] for key in engine_keys}} for index, owner in enumerate(owners)]
    data["runtime_evidence"] = [{"index": 0, "phase": "partial", "at_ps": 0,
        "starts": dict.fromkeys(runtime_keys, 0), "stops": dict.fromkeys(runtime_keys, 0), "values": {key: [] for key in runtime_keys}}]
    return data


def test_journal_capture_order_survives_sorted_json_roundtrip(tmp_path):
    data = empty_journal_data()
    write(tmp_path / "raw.json", data)
    retained = read(tmp_path / "raw.json")
    assert next(iter(retained["sinks"])) == "simllm-decode-0"
    check_journals(retained, Evidence([]))
    retained["engine_evidence"].reverse()
    for index, row in enumerate(retained["engine_evidence"]):
        row["index"] = index
    with pytest.raises(GuardFailure, match="engine-capture-domain"):
        check_journals(retained, Evidence([]))


@pytest.mark.parametrize("kind", ["changed-value", "missing-prefix", "duplicate-prefix", "foreign-owner", "backward-time", "changed-cursor"])
def test_full_journal_reconstruction_rejects_corruption(kind):
    expected = {"engine": {"records": [{"value": 1}, {"value": 2}]}}
    rows = [{"index": i, "engine_id": "engine", "phase": "retired", "at_ps": i,
             "starts": {"records": i}, "stops": {"records": i + 1}, "values": {"records": [expected["engine"]["records"][i]]}}
            for i in range(2)]
    reconstruct(rows, expected, "valid", Evidence([]), engine=True)
    changed = deepcopy(rows)
    if kind == "changed-value":
        changed[1]["values"]["records"][0]["value"] = 3
    elif kind == "missing-prefix":
        changed.pop()
    elif kind == "duplicate-prefix":
        changed[1]["values"]["records"] *= 2
    elif kind == "foreign-owner":
        changed[1]["engine_id"] = "foreign"
    elif kind == "backward-time":
        changed[1]["at_ps"] = -1
    else:
        changed[1]["starts"]["records"] = 0
    with pytest.raises(GuardFailure):
        reconstruct(changed, expected, "corrupt", Evidence([]), engine=True)


def test_serialized_method_identity_joins_inner_engine_separately_from_outer_wrapper():
    source = {"origin": "vllm/v1/engine/llm_engine.py", "sha256": "a" * 64}
    data = {"process_id": "fixture", "mode": "serialized", "cells": [], "serialized_observations": [],
        "retained_before": [{"engine_id": "engine", "frontend_object_id": 11}],
        "native_configurations": [{"engine_id": "engine", "llm_engine_object_id": 22}],
        "sources_before": {"native": {"vllm/v1/engine/llm_engine.py": source}},
        "serialized_bindings": [{"engine_id": "engine", "frontend_id": 22, "function_id": 33, "code_id": 44,
            "module": "vllm.v1.engine.llm_engine", "qualname": "LLMEngine.step", "source": source["origin"],
            "source_sha256": source["sha256"], "had_instance_binding": False, "restored": True}]}
    check_serialized(data, {}, {}, Evidence([]))
    data["serialized_bindings"][0]["frontend_id"] = 11
    with pytest.raises(GuardFailure, match="frontend"):
        check_serialized(data, {}, {}, Evidence([]))


@pytest.mark.parametrize("decode", [1, 2])
def test_native_schedule_uses_latest_admitted_shard_then_real_decode_exclusion(decode):
    frozen = json.loads((HERE / "expectations.json").read_bytes())
    spec = {"prompt_tokens": 8, "requests_per_batch": 2, "decode_engines": decode}
    cell = {"admission_ps": 0}
    flows = [{"source": i, "completion_time_ps": 123000000 if i < 8 else 124000000} for i in range(16)]
    flows[7]["completion_time_ps"] = 125000000
    rows = native_schedule(cell, spec, frozen, flows)
    assert rows[1]["decode_eligible_at_ps"] == 124000000
    assert rows[0]["decode_eligible_at_ps"] == (124000000 + 4 * 77952000 if decode == 1 else 125000000)
    assert all(row["ttft_ps"] == row["decode_eligible_at_ps"] + 77952000 for row in rows)


@pytest.mark.parametrize("fails", [False, True])
def test_serialized_observer_restores_actual_method_after_success_or_failure(tmp_path, monkeypatch, fails):
    from simllm.adapters.vllm import independent

    primary = RuntimeError("actual step failure")

    class Frontend:
        def step(self):
            if fails:
                raise primary
            return []

    frontend = Frontend()
    original = frontend.step
    engine = NS(engine_id="engine", llm=NS(llm_engine=frontend), executor=NS(step_records=[], step_results=[]))
    monkeypatch.setattr(native, "all_engines", lambda _: [engine])
    monkeypatch.setattr(independent, "native_engine_state", lambda _: {"actual": "state"})
    session = NS(clock=NS(now_ps=0))
    journal = NS(engine=lambda *args: None, runtime=lambda *args: None)
    rows, bindings = [], []

    def call():
        with native.observe_serialized(session, "serialized", tmp_path, rows, bindings, journal):
            assert "step" in vars(frontend)
            assert frontend.step() == []

    if fails:
        with pytest.raises(RuntimeError) as error:
            call()
        assert error.value is primary
    else:
        call()
    assert "step" not in vars(frontend)
    assert frontend.step.__func__ is original.__func__ and frontend.step.__self__ is original.__self__
    assert bindings[0]["restored"] and bindings[0]["frontend_id"] == id(frontend)
    assert [row["phase"] for row in rows] == ["before-step", "failed-step" if fails else "after-step"]


def test_retained_wire_replay_uses_complete_existing_codec_and_rejects_changed_frames(tmp_path, monkeypatch):
    frozen, deadline = (json.loads((HERE / name).read_bytes()) for name in ("expectations.json", "deadline-expectations.json"))
    spec = frozen["native_processes"][0]
    backend = frozen["backend"]

    class FullPeer(Peer):
        def reply(self, request):
            response = super().reply(request)
            if request["verb"] == "drain":
                response["sq_high_watermarks"] = [1] + [0] * 23
            return response

    peer = FullPeer()
    stream = Stream(peer)
    stream.io_deadline = lambda seconds: wire.nullcontext()
    monkeypatch.setattr(flow_session, "OwnedBinaryProcess", lambda *args, **kwargs: stream)
    config = FlowSessionConfig(backend["profile"], 24, spec["link_rate_bps"], backend["effective_hardware_sha256"], 9001,
                              **deadline["shared_flow_session"])
    client = FlowSession(config, ("scripted",), session_id="shared-kv:" + spec["id"])
    with client:
        client.inject(execution_id="shared-kv:" + spec["id"], operation_id="op", flow_id="flow", source=0,
                      destination=16, tag=6200, payload_bytes=49152, eligible_at_ps=20)
        peer.updates = [("completion", 200, [1], True, 4)]
        client.await_completion((1,))
    save_transcript(tmp_path / "wire", client.transcript, client.stderr)
    accepted = {key: value for key, value in peer.requests[1].items() if key not in ("schema", "verb", "policy_context_token")}
    accepted["request_id"] = "public"
    data = {"network_after": {"events": plain(client.events), "rows": plain(client.completion_rows),
                              "drain": plain(client._drain), "accepted": [accepted]}}
    wire.check_wire(tmp_path / "wire", data, spec, frozen, deadline, Evidence([]))
    response = tmp_path / "wire/00005-response.bin"
    response.write_bytes(response.read_bytes() + b"extra")
    with pytest.raises(GuardFailure, match="receipt"):
        wire.check_wire(tmp_path / "wire", data, spec, frozen, deadline, Evidence([]))


@pytest.fixture
def ordered_component(tmp_path, monkeypatch, request):
    """Capture real owner calls against a scripted peer, without a serving run."""
    frozen, deadline = (json.loads((HERE / name).read_bytes()) for name in ("expectations.json", "deadline-expectations.json"))
    spec = frozen["native_processes"][0]
    count = getattr(request, "param", 1)

    class FullPeer(Peer):
        def reply(self, request):
            response = super().reply(request)
            if request["verb"] == "drain":
                response["sq_high_watermarks"] = [1] * (8 * count) + [0] * (24 - 8 * count)
            return response

    class FullStream(Stream):
        pid = 12345

        def __init__(self, peer):
            super().__init__(peer)
            self._child = NS(process=NS(returncode=None))

        def io_deadline(self, seconds):
            return wire.nullcontext()

        def finish(self):
            result = super().finish()
            self._child.process.returncode = 0
            return result

    peer, order = FullPeer(), ObservationOrder(tmp_path)
    stream = FullStream(peer)
    monkeypatch.setattr(flow_session, "OwnedBinaryProcess", lambda *args, **kwargs: stream)
    clock = CaptureClock(order)
    runtime = EngineStepRuntime(clock)
    session, journal = NS(clock=clock, engine_runtime=runtime), EvidenceJournal(tmp_path)
    observations, checkpoints = [], []
    owner = native.recorded_owner(NS(output_root=tmp_path, htsim_rnic="scripted-no-executable"),
                                  spec, frozen, deadline, observations, order)
    engines = ("simllm-prefill-0", "simllm-prefill-1", "simllm-decode-0")
    owner.validate_engines(engines[:2], engines[2:], 8)
    owner.bind(clock, {name: tuple(range(8)) for name in engines})
    first, pending = len(order.rows), []

    def observe(engine, phase):
        row = {"engine_id": engine, "step_index": 0, "phase": phase, "at_ps": clock.now_ps}
        checkpoints.append(row)
        order.append("engine", len(checkpoints) - 1, clock.now_ps)
        journal.runtime(session, phase)

    for number in range(count):
        engine = engines[number]

        def publish(result, engine=engine, number=number):
            observe(engine, "before-retire")
            pending.append(owner.submit(request_id="public" + str(number), source_engine_id=engine,
                destination_engine_id=engines[-1], kv_bytes=393216, submitted_at_ps=clock.now_ps))
            observe(engine, "retired")

        observe(engine, "before-submit")
        runtime.submit(engine, StepRecord(0, 0, [ScheduledRequest("opaque", RequestPhase.PREFILL, 8)]),
                       StepResult(0, 10, 10), guard=lambda: None, publish=publish)
        observe(engine, "submitted")
    runtime.advance_to(10)
    runtime.complete_due()
    peer.updates = [("horizon", 10, [], False, 0)]
    assert owner.complete_due(tuple(pending)) == ()
    peer.updates = [("completion", 20000100, list(range(8 * number + 1, 8 * number + 9)), number == count - 1, 32)
                    for number in range(count)]
    assert owner.progress(tuple(pending), through_ps=None) == 20000100
    runtime.advance_to(20000100)
    joins = owner.complete_due(tuple(pending))
    stop = len(order.rows)
    journal.runtime(session, "partial")
    before_close = native.network_snapshot(owner)
    rows = [{"result": {"request_id": "public" + str(number), "prefill_engine_id": engines[number], "decode_engine_id": engines[-1],
                       "prefill_completed_at_ps": 10, "handoff": join.to_json()}} for number, join in enumerate(joins)]
    data = {"process_id": spec["id"], "mode": "independent", "identity": {"clock_object_id": id(clock)},
        "sinks": dict.fromkeys(engines), "checkpoints": checkpoints, "serialized_observations": [],
        "runtime_evidence": journal.runtime_rows, "clock_advances": plain(clock.advances),
        "projections": {"completed": [{"receipt": plain(receipt), "result": plain(result)} for receipt, result in runtime.results]},
        "cells": [{"requests": rows, "inputs": [{"request_id": "public" + str(number), "admitted_at_ps": 0} for number in range(count)],
            "observation_start": first, "observation_stop": stop, "start_ps": 0, "end_ps": clock.now_ps,
            "clock_advance_indices": [0, 1], "event_indices": list(range(5 * count)), "visit_indices": list(range(count)),
            "network_after": before_close}]}
    owner.close()
    runtime.close()
    data.update(network_after=native.network_snapshot(owner), network_observations=observations,
                observation_order=order.rows)
    data = json.loads(json.dumps(data, sort_keys=True))
    decoded = wire.check_wire(tmp_path / "network-transcript", data, spec, frozen, deadline, Evidence([]))
    check_order(data, spec, decoded, Evidence([]))
    check_runtime_visibility(data, Evidence([]))
    return data, spec, decoded


@pytest.mark.parametrize("kind", ["cell-prefix-deletion", "action", "time", "pending", "horizon", "early-join"])
def test_cross_clock_admission_rejects_coherent_packet_corruption(ordered_component, kind):
    data, spec, decoded = deepcopy(ordered_component)
    progress = next(row for row in data["network_observations"] if row["action"] == "progress")
    if kind == "cell-prefix-deletion":
        for key in ("accepted", "events", "rows", "joins"):
            data["cells"][0]["network_after"][key] = []
    elif kind == "action":
        progress["action"] = "invented"
    elif kind == "time":
        progress["at_ps"] -= 1
    elif kind == "pending":
        progress["snapshot"]["pending"] = []
    elif kind == "horizon":
        progress["call"]["arguments"]["through_ps"] = 20000100
        decoded["requests"][progress["call"]["transcript_start"] // 2]["through_ps"] = 20000100
    else:
        progress["snapshot"]["joins"] = data["network_after"]["joins"]
    with pytest.raises(GuardFailure):
        check_order(data, spec, decoded, Evidence([]))


def test_runtime_prefix_cannot_publish_all_results_at_first_callback(ordered_component):
    data, _, _ = deepcopy(ordered_component)
    final = data["runtime_evidence"][-1]["stops"]
    for index, row in enumerate(data["runtime_evidence"]):
        if index:
            row["starts"] = dict(final)
        row["stops"] = dict(final)
    with pytest.raises(GuardFailure, match="phase-exact-prefix"):
        check_runtime_visibility(data, Evidence([]))


def test_shared_clock_rejects_an_unmotivated_intermediate_advance(ordered_component):
    data, spec, decoded = deepcopy(ordered_component)
    position = next(index for index, row in enumerate(data["observation_order"]) if row["kind"] == "clock" and row["index"] == 1)
    original = data["clock_advances"][1]
    intermediate = (original["before_ps"] + original["after_ps"]) // 2
    data["clock_advances"][1:] = [{"before_ps": original["before_ps"], "after_ps": intermediate},
                                 {"before_ps": intermediate, "after_ps": original["after_ps"]}]
    data["observation_order"][position]["at_ps"] = intermediate
    data["observation_order"].insert(position + 1, {"kind": "clock", "index": 2, "at_ps": original["after_ps"]})
    for sequence, row in enumerate(data["observation_order"]):
        row["sequence"] = sequence
    for row in data["network_observations"]:
        if row["call"]["order_start"] > position:
            row["call"]["order_start"] += 1
            row["call"]["clock_before"]["advance_count"] += 1
            row["clock_after"]["advance_count"] += 1
    data["cells"][0]["observation_stop"] += 1
    data["cells"][0]["clock_advance_indices"].append(2)
    with pytest.raises(GuardFailure, match="next-causal-deadline"):
        check_order(data, spec, decoded, Evidence([]))


@pytest.mark.parametrize("ordered_component", [2], indirect=True)
def test_same_time_join_cannot_publish_before_other_native_callbacks_are_drained(ordered_component):
    data, spec, decoded = deepcopy(ordered_component)
    row = next(row for row in data["network_observations"] if row["action"] == "complete_due" and row["at_ps"] == 20000100)
    progress = data["network_observations"][row["index"] - 1]
    # Drop the second same-time callback and expose only the first ready join.
    row["transcript_stop"] = row["call"]["transcript_start"]
    row["snapshot"] = deepcopy(progress["snapshot"])
    row["snapshot"]["joins"] = data["network_after"]["joins"][:1]
    row["snapshot"]["pending"] = ["public1"]
    row["snapshot"]["staged"] = []
    row["result"] = row["result"][:1]
    state = {"now": 20000100, "clock": 2, "events": 10, "retired": 2}
    with pytest.raises(GuardFailure, match="drain-initial-pending"):
        packet_call(data, spec, decoded, row, row["call"]["order_start"], state, {}, None, [],
                    row["call"]["transcript_start"], Evidence([]))


@pytest.fixture
def serialized_component():
    """A synchronous prefill exposes one output only after its priced step."""
    source = {"origin": "vllm/v1/engine/llm_engine.py", "sha256": "a" * 64}
    params = {"schema": "simllm-pd-kv-params-v1", "session_request_id": "public"}
    raw = {"request_id": "public", "prefill_engine_id": "engine", "admitted_at_ps": 0,
           "handoff": {"completed_at_ps": 10}, "kv_transfer_params": params}
    cache = {"groups": [{}], "blocks": [[block, 0] for block in range(64)], "free_queue": list(range(1, 64))}
    before = {"requests": {"opaque": {"status": "WAITING", "num_computed_tokens": 0, "num_in_flight_tokens": 0,
        "output_token_ids": [], "all_token_ids": [7, 8], "stop_reason": None, "kv_transfer_params": params}},
        "visible": {"opaque": {"external_request_id": "public", "output_token_ids": [],
                              "is_prefilling": True, "sent_tokens_offset": 0}},
        "cache": deepcopy(cache), "queues": {"waiting": ["opaque"], "running": []}, "emitted": []}
    after = {"requests": {}, "visible": {}, "cache": deepcopy(cache), "queues": {"waiting": [], "running": []},
        "emitted": [{"request_id": "public", "finished": True, "token_ids": [512], "kv_transfer_params": params}]}
    data = {"process_id": "fixture", "mode": "serialized",
        "cells": [{"requests": [{"result": raw}], "inputs": [{"request_id": "public", "prompt_token_ids": [7, 8]}]}],
        "serialized_observations": [
            {"index": 0, "engine_id": "engine", "phase": "before-step", "at_ps": 0, "record_count": 0, "result_count": 0, "native": before},
            {"index": 1, "engine_id": "engine", "phase": "after-step", "at_ps": 10, "record_count": 1, "result_count": 1, "native": after}],
        "retained_before": [{"engine_id": "engine", "frontend_object_id": 11}],
        "native_configurations": [{"engine_id": "engine", "llm_engine_object_id": 22}],
        "sources_before": {"native": {"vllm/v1/engine/llm_engine.py": source}},
        "serialized_bindings": [{"engine_id": "engine", "frontend_id": 22, "function_id": 33, "code_id": 44,
            "module": "vllm.v1.engine.llm_engine", "qualname": "LLMEngine.step", "source": source["origin"],
            "source_sha256": source["sha256"], "had_instance_binding": False, "restored": True}]}
    steps = {("engine", 0): {"record": {"virtual_time_ps": 0, "scheduled": [{"request_id": "opaque"}]},
                              "result": {"completed_at_ps": 10}}}
    owners = {("engine", "opaque"): "public"}
    check_serialized(data, steps, owners, Evidence([]))
    return data, steps, owners


@pytest.mark.parametrize("kind", ["tokens", "computed", "status", "queue", "visible", "output"])
def test_serialized_admission_rejects_early_tokens_and_changed_native_state(serialized_component, kind):
    data, steps, owners = deepcopy(serialized_component)
    before = data["serialized_observations"][0]["native"]
    if kind == "tokens":
        before["requests"]["opaque"]["output_token_ids"] = [512] * 100
        before["requests"]["opaque"]["all_token_ids"] += [512] * 100
    elif kind == "computed":
        before["requests"]["opaque"]["num_computed_tokens"] = 1000
    elif kind == "status":
        before["requests"]["opaque"]["status"] = "RUNNING"
    elif kind == "queue":
        before["queues"] = {"waiting": [], "running": ["opaque"]}
    elif kind == "visible":
        before["visible"]["opaque"]["output_token_ids"] = [512]
    else:
        data["serialized_observations"][1]["native"]["emitted"][0]["token_ids"] = [512, 512]
    with pytest.raises(GuardFailure):
        check_serialized(data, steps, owners, Evidence([]))


def test_capture_source_admission_rejects_empty_helper_inventory(tmp_path):
    from examples.shared_kv_handoff_v1.common import validate_sources

    snapshot = {"packages": {}, "repository": str(tmp_path), "native": {}, "helpers": {},
                "origins": dict.fromkeys(("simllm", "simllm.core.step", "simllm.adapters.vllm.pd_session"))}
    with pytest.raises(ValueError, match="required native capture helper origins"):
        validate_sources(snapshot, tmp_path, {}, tmp_path, {"native_source_sha256": {}})
