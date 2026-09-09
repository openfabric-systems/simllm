"""Read complete retained frames through the existing session codec."""

from __future__ import annotations

import json
import struct
from contextlib import nullcontext
from unittest.mock import patch

from simllm.backends import flow_session

from .common import plain, read, sha


class RetainedStream:
    """A read-only transcript cursor that starts no process and advances no time."""

    def __init__(self, frames):
        self.frames = frames
        self.cursor = 0
        self.response = b""
        self.finished = False
        self.aborted = False
        self.stderr = b""

    def io_deadline(self, _seconds):
        return nullcontext()

    def write(self, frame):
        if self.response or self.cursor + 1 >= len(self.frames):
            raise ValueError("replay frame cursor is incomplete")
        request, response = self.frames[self.cursor:self.cursor + 2]
        if request != ("request", frame) or response[0] != "response":
            raise ValueError("replayed request differs from the retained bytes")
        self.cursor += 2
        self.response = response[1]

    def read_exact(self, size):
        if len(self.response) < size:
            raise ValueError("retained response is incomplete")
        result, self.response = self.response[:size], self.response[size:]
        return result

    def finish(self):
        if self.response or self.cursor != len(self.frames) or self.finished or self.aborted:
            raise ValueError("retained transcript did not finish exactly once")
        self.finished = True
        return 0

    def abort(self):
        self.aborted = True


def decode_frame(raw):
    if len(raw) < 4 or struct.unpack(">I", raw[:4])[0] != len(raw) - 4:
        raise ValueError("retained frame size differs from its bytes")
    value = json.loads(raw[4:].decode("utf-8"), object_pairs_hook=flow_session._pairs,
                       parse_float=flow_session._reject_number, parse_constant=flow_session._reject_number)
    if raw[4:] != flow_session._canonical(value):
        raise ValueError("retained frame is not canonical JSON")
    return value


def check_wire(path, data, spec, frozen, deadline, evidence):
    label = spec["id"] + ":wire"
    if spec["kind"] == "compatibility":
        evidence.check(label + ":off", not path.exists())
        return {"requests": [], "responses": [], "prefixes": {}}
    index = read(path / "index.json")
    frames = []
    evidence.check(label + ":complete-pairs", len(index) >= 8 and len(index) % 2 == 0)
    for number, row in enumerate(index):
        direction = "request" if number % 2 == 0 else "response"
        name = f"{number:05d}-{direction}.bin"
        raw = (path / name).read_bytes()
        evidence.equal(label + ":receipt:" + str(number), row,
                       {"file": name, "direction": direction, "bytes": len(raw), "sha256": sha(path / name)})
        decode_frame(raw)
        frames.append((direction, raw))
    evidence.equal(label + ":file-domain", sorted(item.name for item in path.iterdir()),
                   sorted([row["file"] for row in index] + ["index.json", "stderr.bin"]))
    requests = [decode_frame(raw) for direction, raw in frames if direction == "request"]
    responses = [decode_frame(raw) for direction, raw in frames if direction == "response"]
    verbs = [row["verb"] for row in requests]
    evidence.equal(label + ":open-count", verbs.count("open"), 1)
    evidence.equal(label + ":terminal-verbs", verbs[-3:], ["await_completion", "drain", "close"])
    evidence.check(label + ":terminal-quiescence", requests[-3]["until_quiescent"] is True)
    evidence.check(label + ":no-intermediate-drain", all(verb in ("inject", "await_completion") for verb in verbs[1:-3]))
    evidence.check(label + ":no-intermediate-quiescence", all(
        row.get("until_quiescent") is False for row in requests[1:-3] if row["verb"] == "await_completion"))
    backend = frozen["backend"]
    config = flow_session.FlowSessionConfig(backend["profile"],
        8 * (spec["prefill_engines"] + spec["decode_engines"]), spec["link_rate_bps"],
        backend["effective_hardware_sha256"], backend["policy_context_token"], seed=backend["seed"],
        max_events=backend["max_events_per_call"], simulation_budget_ps=backend["simulation_budget_ps"],
        **deadline["shared_flow_session"])
    stream = RetainedStream(frames)
    session = flow_session.FlowSession(config, ("retained-frames-only",), session_id="shared-kv:" + spec["id"])
    prefixes = {0: {"events": [], "rows": [], "last_sequence": 0, "injection_floor_ps": None,
                    "pending_sequences": [], "drain": None}}

    def retain_prefix():
        prefixes[stream.cursor] = {"events": plain(session.events), "rows": plain(session.completion_rows),
            "last_sequence": session.last_accepted_sequence, "injection_floor_ps": session.ordinary_injection_floor_ps,
            "pending_sequences": list(session.pending_sequences), "drain": plain(session._drain)}

    with patch.object(flow_session, "OwnedBinaryProcess", return_value=stream) as constructor:
        session.open()
        retain_prefix()
        while stream.cursor < len(frames):
            request = decode_frame(frames[stream.cursor][1])
            if request["verb"] == "inject":
                fields = {key: request[key] for key in (
                    "execution_id", "operation_id", "flow_id", "source", "destination", "tag", "payload_bytes", "eligible_at_ps")}
                session.inject(**fields)
            elif request["verb"] == "await_completion":
                if request["until_quiescent"]:
                    session.close()
                else:
                    session.await_completion(request["completion_sequences"], through_ps=request["through_ps"])
            else:
                raise ValueError("unexpected request in retained shared session")
            retain_prefix()
        evidence.equal(label + ":one-codec-session", constructor.call_count, 1)
    evidence.check(label + ":terminal-replay", stream.finished and not stream.aborted and not session.poisoned)
    evidence.equal(label + ":full-events", plain(session.events), data["network_after"]["events"])
    evidence.equal(label + ":full-rows", plain(session.completion_rows), data["network_after"]["rows"])
    evidence.equal(label + ":full-drain", plain(session._drain), data["network_after"]["drain"])
    evidence.equal(label + ":all-injections", [{key: value for key, value in row.items()
                    if key not in ("schema", "verb", "policy_context_token")}
                   for row in requests if row["verb"] == "inject"], [
                       {key: value for key, value in row.items() if key != "request_id"}
                       for row in data["network_after"]["accepted"]])
    return {"requests": requests, "responses": responses, "prefixes": prefixes}
