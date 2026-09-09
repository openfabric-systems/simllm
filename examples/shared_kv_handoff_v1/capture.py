"""Append full completed evidence while native owners remain authoritative."""

from __future__ import annotations

from examples.pd_session_target_scale_v1.native import ObservedClock, append_progress
from examples.shared_kv_handoff_v1.common import PUBLICATIONS, plain, sink_publications, write


class ObservationOrder:
    """Record callback order without owning any simulated time or event."""

    def __init__(self, output):
        self.output, self.rows = output, []

    def append(self, kind, index, at_ps):
        row = {"sequence": len(self.rows), "kind": kind, "index": index, "at_ps": at_ps}
        self.rows.append(row)
        append_progress(self.output / "observation-order.jsonl", row)


class CaptureClock(ObservedClock):
    """Observe calls to the same clock used by the serving and engine owners."""

    def __init__(self, order):
        super().__init__()
        self.order = order

    def advance_to(self, at_ps):
        super().advance_to(at_ps)
        self.order.append("clock", len(self.advances) - 1, self.now_ps)


def clock_state(clock):
    if clock is None:
        return None
    reference = getattr(clock, "_engine_step_owner", None)
    runtime = None if reference is None else reference()
    return {"at_ps": clock.now_ps, "next_engine_completion_ps": clock.peek_next_time(),
            "engine_owner_present": runtime is not None,
            "advance_count": len(clock.advances), "event_count": 0 if runtime is None else len(runtime.events),
            "visit_count": 0 if runtime is None else len(runtime.visits),
            "result_count": 0 if runtime is None else len(runtime.results)}


class EvidenceJournal:
    def __init__(self, output):
        self.output = output
        self.engine_positions = {}
        self.runtime_positions = {name: 0 for name in ("events", "visits", "completed", "clock_advances")}
        self.engine_rows = []
        self.runtime_rows = []

    def engine(self, engine, phase, at_ps, publications=None):
        from simllm.core.step import step_record_to_json

        name = engine.engine_id
        previous = self.engine_positions.setdefault(name, {key: 0 for key in ("records", "results", *PUBLICATIONS)})
        publications = sink_publications(engine) if publications is None else publications
        values = {"records": [step_record_to_json(record) for record in engine.executor.step_records],
                  "results": plain(engine.executor.step_results), **publications}
        row = {"index": len(self.engine_rows), "engine_id": name, "phase": phase, "at_ps": at_ps,
               "starts": dict(previous), "stops": {key: len(value) for key, value in values.items()},
               "values": {key: value[previous[key]:] for key, value in values.items()}}
        if any(row["stops"][key] < previous[key] for key in previous):
            raise ValueError("engine evidence shrank before capture")
        append_progress(self.output / "engine-evidence.jsonl", row)
        self.engine_rows.append(row)
        previous.update(row["stops"])

    def runtime(self, session, phase):
        owner = session.engine_runtime
        values = {"events": [] if owner is None else plain(owner.events),
                  "visits": [] if owner is None else plain(owner.visits),
                  "completed": [] if owner is None else [
                      {"receipt": plain(receipt), "result": plain(result)} for receipt, result in owner.results],
                  "clock_advances": plain(session.clock.advances)}
        previous = self.runtime_positions
        row = {"index": len(self.runtime_rows), "phase": phase, "at_ps": session.clock.now_ps,
               "starts": dict(previous), "stops": {key: len(value) for key, value in values.items()},
               "values": {key: value[previous[key]:] for key, value in values.items()}}
        if any(row["stops"][key] < previous[key] for key in previous):
            raise ValueError("runtime evidence shrank before capture")
        append_progress(self.output / "runtime-evidence.jsonl", row)
        self.runtime_rows.append(row)
        previous.update(row["stops"])

    def partial(self, session, engines):
        """Retain completed prefixes and pending inputs before failure cleanup."""
        first, failures = None, []

        def attempt(surface, callback):
            nonlocal first
            try:
                callback()
            except BaseException as error:  # noqa: BLE001, capture every independent surface.
                first = first or error
                failures.append({"surface": surface, "type": type(error).__name__, "message": str(error)})

        for engine in engines:
            attempt("engine:" + engine.engine_id, lambda engine=engine: self.engine(engine, "partial", session.clock.now_ps))
        attempt("runtime", lambda: self.runtime(session, "partial"))
        attempt("partial-summary", lambda: write(self.output / "partial-evidence.json", {
            "at_ps": session.clock.now_ps, "engine_positions": self.engine_positions,
            "runtime_positions": self.runtime_positions,
            "runtime_failure": None if session.engine_runtime is None else session.engine_runtime.failure,
            "capture_failures": failures}))
        self.capture_failures = failures
        if first is not None:
            raise first
