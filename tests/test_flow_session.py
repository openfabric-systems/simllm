import copy
import json
import os
import struct
import sys
import time
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from simllm.backends import flow_session
from simllm.backends.flow_session import FlowSession, FlowSessionConfig, FlowSessionError

SCHEMA = "simllm-htsim-flow-session-v1"
CONFIG = FlowSessionConfig("rnic-nn", 2, 400_000_000_000, "a" * 64, 9001)


def encode(value):
    body = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode()
    return struct.pack(">I", len(body)) + body


def counters(posts):
    return {"legacy_aborts": 0, "legacy_ledger_constructed": 0,
            "legacy_mutations": 0, "legacy_posts": 0, "native_posts": posts,
            "native_session_constructed": 1}


class Peer:
    """Independent wire fixtures, with deliberately programmable corrupt responses."""

    def __init__(self):
        self.requests = []
        self.flows = {}
        self.done = {}
        self.updates = []
        self.mutate = lambda request, response: response
        self.now = 0
        self.boundary = 0
        self.prefix = None
        self.floor = None
        self.seen = set()

    def observation(self, sequence, time_ps):
        sent = self.flows[sequence]
        fields = {key: sent[key] for key in (
            "sequence", "execution_id", "operation_id", "flow_id", "source",
            "destination", "tag", "payload_bytes")}
        fields.update(native_flow_id=(sent["source"] << 32) | (sequence - 1),
                      wqe_id=sequence + 10, sq_id=1, sq_post_sequence=sequence,
                      cq_id=2, transport_kind="none", transport_object_id=0)
        events = []
        for kind in ("accepted", "queued", "started", "completed"):
            timestamp = time_ps if kind == "completed" else sent["eligible_at_ps"]
            if kind == "started":
                timestamp = min(timestamp + 1, time_ps)
            events.append({**fields, "kind": kind, "timestamp_ps": timestamp,
                           "policy_context_token": 9001,
                           "cq_post_sequence": sequence if kind == "completed" else None,
                           "cq_consume_sequence": sequence if kind == "completed" else None})
        row = {**fields, "sq_dispatch_sequence": sequence, "cq_post_sequence": sequence,
               "cq_consume_sequence": sequence, "start_time_ps": sent["eligible_at_ps"],
               "completion_time_ps": time_ps, "fct_ps": time_ps - sent["eligible_at_ps"],
               "completion_status": "success"}
        return events, row

    def reply(self, request):
        self.requests.append(copy.deepcopy(request))
        verb = request["verb"]
        common = {"schema": SCHEMA, "status": "ok", "verb": verb}
        if verb == "open":
            response = {**request, **common, "sequence": 0}
        elif verb in {"inject", "inject_at_boundary"}:
            self.flows[request["sequence"]] = request
            response = {**common, "accepted_sequence": request["sequence"],
                        "eligible_at_ps": request["eligible_at_ps"]}
            if verb == "inject_at_boundary":
                response["boundary_id"] = request["boundary_id"]
        elif verb == "await_completion":
            if request["until_quiescent"]:
                reason, at, sequences, quiescent, executed = (
                    "quiescence", self.now, [], True, 0)
            else:
                reason, at, sequences, quiescent, executed = self.updates.pop(0)
            self.now = at
            events, rows = [], []
            for sequence in self.flows:
                if sequence not in sequences and self.flows[sequence]["eligible_at_ps"] > at:
                    continue
                flow_events, row = self.observation(sequence, at)
                for event in flow_events:
                    key = (sequence, event["kind"])
                    if key in self.seen or (event["kind"] == "completed" and sequence not in sequences):
                        continue
                    events.append(event)
                    self.seen.add(key)
                if sequence in sequences:
                    rows.append(row)
                    self.done[sequence] = row
            events.sort(key=lambda event: (
                event["timestamp_ps"], event["sequence"],
                ("accepted", "queued", "started", "completed").index(event["kind"])))
            if reason == "completion":
                self.boundary += 1
                self.prefix = at - 1 if at else None
                self.floor = at
            elif reason == "horizon":
                self.prefix = self.floor = request["through_ps"]
            response = {
                **common, "reason": reason, "event_time_ps": at,
                "fully_processed_horizon_ps": self.prefix,
                "ordinary_injection_floor_ps": self.floor,
                "events_executed": executed, "events": events, "completion_rows": rows,
                "last_accepted_sequence": len(self.flows),
                "authority_counters": counters(sum(kind == "accepted" for _, kind in self.seen)),
                "quiescent": quiescent,
                "boundary_time_ps": at if reason == "completion" else None,
                "boundary_id": self.boundary if reason == "completion" else None,
            }
        elif verb == "drain":
            response = {
                **common, "authority_counters": counters(len(self.flows)),
                "completion_rows": [
                    {key: value for key, value in self.done[seq].items()
                     if key != "completion_status"} for seq in sorted(self.done)],
                "events": [], "last_accepted_sequence": len(self.flows),
                "quiescent": True,
                "sq_high_watermarks": [int(any(flow["source"] == endpoint
                                                for flow in self.flows.values()))
                                       for endpoint in range(2)],
            }
        elif verb == "close":
            response = {**common, "last_accepted_sequence": len(self.flows), "terminal": True}
        else:
            raise AssertionError(verb)
        return self.mutate(request, response)


class Stream:
    def __init__(self, peer):
        self.peer = peer
        self.pending = bytearray()
        self.aborted = False
        self.finished = False
        self.stderr = b"diagnostic stderr"

    def write(self, frame):
        assert len(frame) == 4 + struct.unpack(">I", frame[:4])[0]
        request = json.loads(frame[4:])
        response = self.peer.reply(request)
        self.pending.extend(response if isinstance(response, bytes) else encode(response))

    def read_exact(self, size):
        if len(self.pending) < size:
            raise EOFError("partial fixture frame")
        result = bytes(self.pending[:size])
        del self.pending[:size]
        return result

    def finish(self):
        assert not self.pending
        self.finished = True
        return 0

    def abort(self):
        self.aborted = True


@pytest.fixture
def fixture(monkeypatch):
    peer = Peer()
    stream = Stream(peer)
    calls = []

    def start(command, **kwargs):
        calls.append((command, kwargs))
        return stream

    monkeypatch.setattr(flow_session, "OwnedBinaryProcess", start)
    client = FlowSession(CONFIG, ("native", "--flow-session"), session_id="session-α")
    return client, peer, stream, calls


def inject(client, index=1, at=0, predecessors=()):
    return client.inject(execution_id="graph", operation_id=f"op-{index}",
                         flow_id=f"flow-{index}", source=(index - 1) % 2,
                         destination=index % 2, tag=index, payload_bytes=4096,
                         eligible_at_ps=at, predecessor_sequences=predecessors)


def test_owned_session_exact_continuation_and_full_evidence(fixture):
    client, peer, stream, calls = fixture
    peer.updates = [("completion", 20, [1], True, 4),
                    ("completion", 40, [2, 3], True, 6)]
    with client:
        assert inject(client) == 1
        update = client.await_completion()
        assert update.reason == "completion"
        assert client.boundary_time_ps == 20
        assert client.fully_processed_horizon_ps == 19
        assert client.ordinary_injection_floor_ps == 20
        assert update.events_executed == 4
        assert update.completion_rows[0]["start_time_ps"] == 0
        assert update.events[2]["timestamp_ps"] == 1
        assert update.completion_rows[0]["fct_ps"] == 20
        with pytest.raises(TypeError):
            update.completion_rows[0]["fct_ps"] = 0
        with pytest.raises(FrozenInstanceError):
            update.event_time_ps = 0
        assert inject(client, 2, 20, (1,)) == 2
        assert inject(client, 3, 20, (1,)) == 3
        update = client.await_completion((2, 3))
        assert len(update.completion_rows) == 2
        assert client.pending_sequences == ()
        assert client.completed_sequences == (1, 2, 3)
    assert len(calls) == 1
    assert calls[0][1]["timeout_s"] == 60
    assert stream.finished
    assert client.close().quiesced_at_ps == 40
    assert len(client.close().completion_rows) == 3
    assert client.stderr == b"diagnostic stderr"
    assert len(client.transcript) == 18
    assert peer.requests[3]["verb"] == "inject_at_boundary"
    assert peer.requests[4]["boundary_id"] == peer.requests[3]["boundary_id"]
    assert peer.requests[2]["max_time_ps"] == 10_000_000_000


def test_optional_exchange_deadline_wraps_complete_frames_and_finish(fixture, monkeypatch):
    client, peer, stream, calls = fixture
    client.config = replace(CONFIG, wall_timeout_s=1800, exchange_timeout_s=60)
    actions, active = [], []

    @contextmanager
    def budget(seconds):
        assert not active and seconds == 60
        active.append(seconds)
        actions.append(["begin"])
        try:
            yield stream
        finally:
            active.clear()
            actions[-1].append("end")

    monkeypatch.setattr(stream, "io_deadline", budget, raising=False)
    for name in ("write", "read_exact", "finish"):
        original = getattr(stream, name)

        def observe(*args, _name=name, _original=original):
            assert active == [60]
            actions[-1].append(_name)
            return _original(*args)

        monkeypatch.setattr(stream, name, observe)
    with client:
        pass
    assert calls[0][1]["timeout_s"] == 1800
    assert actions[:-1] == [["begin", "write", "read_exact", "read_exact", "end"]] * 4
    assert actions[-1] == ["begin", "finish", "end"]
    assert [row["verb"] for row in peer.requests] == ["open", "await_completion", "drain", "close"]


@pytest.mark.parametrize("invalid", [0, -1, True, float("nan"), float("inf"), "60"])
def test_optional_exchange_deadline_rejects_invalid_configuration(invalid):
    with pytest.raises(ValueError, match="exchange_timeout_s"):
        replace(CONFIG, exchange_timeout_s=invalid)


def test_optional_exchange_field_preserves_legacy_positional_configuration():
    config = FlowSessionConfig("rnic-nn", 2, 400_000_000_000, "a" * 64, 9001, 7, 15, 50, 100)
    assert (config.seed, config.wall_timeout_s, config.max_events, config.simulation_budget_ps,
            config.exchange_timeout_s) == (7, 15, 50, 100, None)


def test_zero_time_boundary_has_no_inclusive_prefix(fixture):
    client, peer, _, _ = fixture
    peer.updates = [("completion", 0, [1], True, 1)]
    with client:
        inject(client)
        update = client.await_completion()
        assert update.fully_processed_horizon_ps is None
        assert update.ordinary_injection_floor_ps == 0
        assert update.boundary_time_ps == 0


def test_horizon_yield_preserves_pending_flow_and_future_local_release(fixture):
    client, peer, _, _ = fixture
    peer.updates = [("horizon", 0, [], False, 0),
                    ("completion", 2000, [1, 2], True, 5)]
    with client:
        inject(client)
        update = client.await_completion(through_ps=999)
        assert update.reason == "horizon"
        assert client.event_time_ps == 0
        assert client.ordinary_injection_floor_ps == 999
        assert client.boundary_id is None
        inject(client, 2, 1000)
        assert peer.requests[-1]["verb"] == "inject"
        client.await_completion()


def test_explicit_subset_stops_on_any_target_and_returns_same_callback_peers(fixture):
    client, peer, _, _ = fixture
    peer.updates = [("completion", 20, [1, 2], False, 2),
                    ("completion", 40, [3], True, 2)]
    with client:
        for index in (1, 2, 3):
            inject(client, index)
        update = client.await_completion((2,))
        assert tuple(row["sequence"] for row in update.completion_rows) == (1, 2)
        assert client.pending_sequences == (3,)
        client.await_completion()


def test_open_and_close_empty_session_is_zero_event_quiescence(fixture):
    client, peer, stream, _ = fixture
    with client:
        assert client.open() is client
    assert stream.finished
    assert client.close().quiesced_at_ps == 0
    assert [request["verb"] for request in peer.requests] == [
        "open", "await_completion", "drain", "close"]


@pytest.mark.parametrize("change", [
    {"node_count": True}, {"node_count": 0}, {"node_count": 1 << 31},
    {"link_rate_bps": 0}, {"seed": -1}, {"seed": 1 << 64},
    {"policy_context_token": 0}, {"max_events": False}, {"max_events": 0},
    {"simulation_budget_ps": 0}, {"effective_hardware_sha256": "A" * 64},
    {"wall_timeout_s": float("nan")}, {"wall_timeout_s": True},
    {"profile": "rnic-nn-fluid"}, {"profile": "rnic-cn", "node_count": 3},
])
def test_config_rejects_invalid_values_without_child(change):
    with pytest.raises(ValueError):
        replace(CONFIG, **change)


def test_absolute_simulation_budget_includes_step_origin(fixture):
    _, peer, _, _ = fixture
    peer.updates = [("completion", 120, [1], True, 2)]
    client = FlowSession(replace(CONFIG, simulation_budget_ps=500), ("native",),
                         session_id="offset", time_origin_ps=100)
    with client:
        inject(client, at=100)
        client.await_completion()
    assert peer.requests[2]["max_time_ps"] == 600


@pytest.mark.parametrize("action", [
    lambda client: inject(client, at=True),
    lambda client: inject(client, at=-1),
    lambda client: inject(client, predecessors=(1,)),
    lambda client: client.await_completion(()),
    lambda client: client.await_completion((1, 1)),
    lambda client: client.await_completion((99,)),
    lambda client: client.await_completion(through_ps=True),
    lambda client: client.await_completion(through_ps=10_000_000_001),
])
def test_invalid_request_does_not_reach_native_or_consume_cursor(fixture, action):
    client, peer, stream, _ = fixture
    client.open()
    before = len(peer.requests)
    with pytest.raises(FlowSessionError):
        action(client)
    assert len(peer.requests) == before
    assert client.last_accepted_sequence == 0
    assert client.poisoned and stream.aborted


def test_ordinary_equal_time_injection_rejected_locally(fixture):
    client, peer, stream, _ = fixture
    peer.updates = [("completion", 20, [1], True, 1)]
    client.open()
    inject(client)
    client.await_completion()
    with pytest.raises(FlowSessionError, match="exclusion floor"):
        inject(client, 2, 20)
    assert client.last_accepted_sequence == 1
    assert client.boundary_time_ps == 20
    assert stream.aborted


@pytest.mark.parametrize("mutate", [
    lambda response: {**response, "unknown": 1},
    lambda response: {key: value for key, value in response.items() if key != "seed"},
    lambda response: {**response, "schema": "wrong"},
    lambda response: {**response, "sequence": False},
    lambda response: {**response, "node_count": True},
    lambda response: {**response, "status": "pending"},
    lambda response: {**response, "effective_hardware_sha256": "b" * 64},
    lambda response: struct.pack(">I", 0),
    lambda response: struct.pack(">I", 1_048_577),
    lambda response: b"\0\0\0\x05{}",
    lambda response: b"\0\0\0\x01\xff",
    lambda response: b"\0\0\0\x04NaN ",
    lambda response: b"\0\0\0\x05[1.0]",
    lambda response: struct.pack(">I", 13) + b'{"a":1,"a":2}',
    lambda response: struct.pack(">I", 3) + b" {}",
])
def test_bad_open_response_poison_and_reap(fixture, mutate):
    client, peer, stream, _ = fixture
    peer.mutate = lambda request, response: mutate(response)
    with pytest.raises(FlowSessionError):
        client.open()
    assert client.poisoned and stream.aborted
    with pytest.raises(FlowSessionError, match="terminal"):
        inject(client)


def change_row(response, key, value):
    response["completion_rows"][0][key] = value
    return response


@pytest.mark.parametrize("mutate", [
    lambda response: {**response, "last_accepted_sequence": 0},
    lambda response: {**response, "events_executed": True},
    lambda response: {**response, "events_executed": 1_000_001},
    lambda response: {**response, "fully_processed_horizon_ps": 20},
    lambda response: {**response, "ordinary_injection_floor_ps": 19},
    lambda response: {**response, "boundary_id": 0},
    lambda response: {**response, "boundary_id": True},
    lambda response: {**response, "quiescent": 1},
    lambda response: {**response, "reason": "horizon"},
    lambda response: {**response, "completion_rows": []},
    lambda response: {**response, "events": response["events"][1:]},
    lambda response: {**response, "events": response["events"] * 2},
    lambda response: {**response, "authority_counters": counters(0)},
    lambda response: change_row(response, "completion_status", "transport_error"),
    lambda response: change_row(response, "completion_status", "network_rejected"),
    lambda response: change_row(response, "flow_id", "other"),
    lambda response: change_row(response, "native_flow_id", 88),
    lambda response: change_row(response, "start_time_ps", 1),
    lambda response: change_row(response, "completion_time_ps", 21),
    lambda response: change_row(response, "fct_ps", 0),
    lambda response: change_row(response, "wqe_id", 99),
    lambda response: change_row(response, "cq_consume_sequence", 99),
    lambda response: change_row(response, "payload_bytes", True),
])
def test_bad_native_evidence_is_terminal_before_result(fixture, mutate):
    client, peer, stream, _ = fixture
    peer.updates = [("completion", 20, [1], True, 1)]
    client.open()
    inject(client)
    peer.mutate = lambda request, response: mutate(response)
    with pytest.raises(FlowSessionError):
        client.await_completion()
    assert client.poisoned and stream.aborted
    with pytest.raises(FlowSessionError):
        client.close()
    with pytest.raises(FlowSessionError):
        _ = client.completion_rows


def test_native_error_preserves_raw_frame_and_reaps(fixture):
    client, peer, stream, _ = fixture
    client.open()
    peer.mutate = lambda request, response: {
        "schema": SCHEMA, "status": "error", "verb": "error", "terminal": True,
        "authority_counters": counters(0),
        "error": {"code": "budget_exhausted", "message": "event limit reached"}}
    with pytest.raises(FlowSessionError, match="native budget_exhausted"):
        inject(client)
    assert b"budget_exhausted" in client.transcript[-1][1]
    assert client.last_accepted_sequence == 0
    assert stream.aborted


def test_context_cancellation_reaps_without_attempting_drain(fixture):
    client, peer, stream, _ = fixture
    with pytest.raises(KeyboardInterrupt), client:
        inject(client)
        raise KeyboardInterrupt
    assert stream.aborted
    assert [request["verb"] for request in peer.requests] == ["open", "inject"]


def test_drain_must_match_every_previously_observed_field(fixture):
    client, peer, stream, _ = fixture
    peer.updates = [("completion", 20, [1], True, 1)]
    client.open()
    inject(client)
    client.await_completion()

    def mutate(request, response):
        if request["verb"] == "drain":
            response["completion_rows"][0]["sq_dispatch_sequence"] = 0
        return response

    peer.mutate = mutate
    with pytest.raises(FlowSessionError, match="drain changed"):
        client.close()
    assert stream.aborted


def test_real_owned_stream_completes_a_framed_empty_session():
    code = r'''
import json, struct, sys
while True:
    header = sys.stdin.buffer.read(4)
    if not header:
        break
    request = json.loads(sys.stdin.buffer.read(struct.unpack(">I", header)[0]))
    verb = request["verb"]
    common = {"schema": "simllm-htsim-flow-session-v1", "status": "ok", "verb": verb}
    counters = {"legacy_aborts": 0, "legacy_ledger_constructed": 0,
                "legacy_mutations": 0, "legacy_posts": 0, "native_posts": 0,
                "native_session_constructed": 1}
    if verb == "open":
        response = {**request, **common, "sequence": 0}
    elif verb == "await_completion":
        response = {**common, "reason": "quiescence", "event_time_ps": 0,
                    "fully_processed_horizon_ps": None, "ordinary_injection_floor_ps": None,
                    "events_executed": 0, "events": [], "completion_rows": [],
                    "last_accepted_sequence": 0, "authority_counters": counters,
                    "quiescent": True, "boundary_time_ps": None, "boundary_id": None}
    elif verb == "drain":
        response = {**common, "authority_counters": counters, "completion_rows": [],
                    "events": [], "last_accepted_sequence": 0, "quiescent": True,
                    "sq_high_watermarks": [0, 0]}
    else:
        assert verb == "close"
        response = {**common, "terminal": True, "last_accepted_sequence": 0}
    body = json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
    sys.stdout.buffer.write(struct.pack(">I", len(body)) + body)
    sys.stdout.buffer.flush()
'''
    client = FlowSession(CONFIG, (sys.executable, "-c", code), session_id="framed")
    with client:
        assert client.pending_sequences == ()
    assert client.close().quiesced_at_ps == 0
    assert len(client.transcript) == 8


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux process-group evidence")
def test_framed_eof_kills_stubborn_descendant_after_leader_exits(tmp_path):
    marker = tmp_path / "descendant.pid"
    code = (
        "import os, signal, struct, sys, time\n"
        "from pathlib import Path\n"
        "header = sys.stdin.buffer.read(4)\n"
        "sys.stdin.buffer.read(struct.unpack('>I', header)[0])\n"
        f"marker = Path({str(marker)!r})\n"
        "if os.fork() == 0:\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "    for fd in (0, 1, 2): os.close(fd)\n"
        "    marker.write_text(str(os.getpid()), encoding='ascii')\n"
        "    time.sleep(30)\n"
        "else:\n"
        "    while not marker.exists() or not marker.read_text(): time.sleep(0.005)\n"
        "    os._exit(2)\n"
    )
    client = FlowSession(CONFIG, (sys.executable, "-c", code), session_id="early-exit")
    with pytest.raises(FlowSessionError, match="response early"):
        client.open()
    assert client.poisoned
    pid = int(marker.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            stat = (Path("/proc") / str(pid) / "stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            break
        if stat[stat.rfind(")") + 2:].split()[0] == "Z":
            break
        time.sleep(0.01)
    else:
        os.kill(pid, 9)
        pytest.fail("owned descendant survived terminal framed cleanup")


def test_callback_cannot_expose_a_target_after_its_completion_time(fixture):
    client, peer, _, _ = fixture
    peer.updates = [("completion", 20, [1], True, 2)]
    client.open()
    inject(client)

    def mutate(request, response):
        response.update(event_time_ps=21, boundary_time_ps=21,
                        ordinary_injection_floor_ps=21, fully_processed_horizon_ps=20)
        return response

    peer.mutate = mutate
    with pytest.raises(FlowSessionError, match="after its exact boundary"):
        client.await_completion()


def test_queue_aliases_cannot_join_distinct_flows(fixture):
    client, peer, _, _ = fixture
    peer.updates = [("completion", 20, [1, 3], False, 2)]
    client.open()
    for index in (1, 2, 3):
        inject(client, index)

    def mutate(request, response):
        for row in response["events"] + response["completion_rows"]:
            if row["sequence"] == 3:
                row["sq_post_sequence"] = 1
        return response

    peer.mutate = mutate
    with pytest.raises(FlowSessionError, match="queue sequence aliases"):
        client.await_completion()


def test_target_completion_wins_at_both_inclusive_limits(fixture):
    client, peer, _, _ = fixture
    client.config = replace(CONFIG, max_events=1, simulation_budget_ps=20)
    client.max_time_ps = 20
    peer.updates = [("completion", 20, [1], True, 1)]
    with client:
        inject(client)
        assert client.await_completion(through_ps=20).reason == "completion"


@pytest.mark.parametrize("case", ("header", "body"))
def test_partial_response_bytes_remain_in_failure_transcript(case):
    raw = b"\0\0" if case == "header" else b"\0\0\0\x05{}"
    code = (
        "import os, struct, sys\n"
        "header = sys.stdin.buffer.read(4)\n"
        "sys.stdin.buffer.read(struct.unpack('>I', header)[0])\n"
        f"os.write(1, {raw!r})\n"
    )
    client = FlowSession(CONFIG, (sys.executable, "-c", code), session_id="partial")
    with pytest.raises(FlowSessionError, match="response early"):
        client.open()
    assert client.transcript[-1] == ("response", raw)
