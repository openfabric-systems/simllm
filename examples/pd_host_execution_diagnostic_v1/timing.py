"""Passive, main-thread phase counters with exact binding restoration."""

from __future__ import annotations

import cProfile
import functools
import importlib
import inspect
import pstats
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path


def callable_identity(value):
    function = getattr(value, "__func__", value)
    return (getattr(value, "__self__", None), function)


def source_identity(value):
    function = getattr(value, "__func__", value)
    code = getattr(function, "__code__", None)
    if code is None:
        raise TypeError("a phase probe requires a Python function")
    return {"module": function.__module__, "qualname": function.__qualname__,
            "filename": str(Path(inspect.getsourcefile(function)).resolve()),
            "first_line": code.co_firstlineno}


class PhaseCounters:
    """Observe selected bindings; never substitute a function result."""

    def __init__(self, *, wall_timer=time.perf_counter_ns, cpu_timer=time.thread_time_ns):
        self.wall_timer = wall_timer
        self.cpu_timer = cpu_timer
        self.thread = threading.get_ident()
        self.stack = []
        self.bindings = []
        self.rows = {}
        self.closed = False
        self.restored = []

    def bind(self, owner, attribute, phase):
        if self.closed or any(obj is owner and name == attribute
                              for obj, name, *_ in self.bindings):
            raise ValueError("probe binding is closed or duplicated")
        original = getattr(owner, attribute)
        if not callable(original):
            raise TypeError("probe target is not callable")
        present = attribute in vars(owner)
        stored = vars(owner).get(attribute)
        identity = source_identity(original)
        self.rows.setdefault(phase, {"calls": 0, "wall_ns": 0, "cpu_ns": 0,
                                    "child_wall_ns": 0, "child_cpu_ns": 0})

        @functools.wraps(original)
        def observe(*args, **kwargs):
            if self.closed or threading.get_ident() != self.thread:
                raise RuntimeError("probe called outside its active main-thread scope")
            frame = {"child_wall_ns": 0, "child_cpu_ns": 0}
            self.stack.append(frame)
            wall, cpu = self.wall_timer(), self.cpu_timer()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed_wall, elapsed_cpu = self.wall_timer() - wall, self.cpu_timer() - cpu
                if self.stack.pop() is not frame:
                    raise RuntimeError("phase counter stack changed")
                row = self.rows[phase]
                row["calls"] += 1
                row["wall_ns"] += elapsed_wall
                row["cpu_ns"] += elapsed_cpu
                row["child_wall_ns"] += frame["child_wall_ns"]
                row["child_cpu_ns"] += frame["child_cpu_ns"]
                if self.stack:
                    self.stack[-1]["child_wall_ns"] += elapsed_wall
                    self.stack[-1]["child_cpu_ns"] += elapsed_cpu

        setattr(owner, attribute, observe)
        self.bindings.append((owner, attribute, present, stored, original, observe,
                              phase, identity))

    def snapshot(self):
        if self.stack:
            raise RuntimeError("cannot snapshot an active phase counter")
        rows = deepcopy(self.rows)
        for row in rows.values():
            row["exclusive_wall_ns"] = row["wall_ns"] - row["child_wall_ns"]
            row["exclusive_cpu_ns"] = row["cpu_ns"] - row["child_cpu_ns"]
        return rows

    def check_bindings(self):
        if self.closed:
            raise RuntimeError("probe bindings are closed")
        for owner, name, _, _, _, wrapper, _, _ in self.bindings:
            if getattr(owner, name, None) is not wrapper:
                raise RuntimeError("active probe binding was replaced")

    def close(self):
        if self.closed:
            raise RuntimeError("probe bindings already restored")
        if self.stack:
            raise RuntimeError("cannot restore an active probe")
        findings = []
        for owner, name, present, stored, original, wrapper, phase, identity in reversed(self.bindings):
            if getattr(owner, name, None) is not wrapper:
                findings.append("active binding replaced: " + phase)
            restored = False
            try:
                if present:
                    setattr(owner, name, stored)
                elif name in vars(owner):
                    delattr(owner, name)
                actual_owner, actual_function = callable_identity(getattr(owner, name))
                original_owner, original_function = callable_identity(original)
                restored = (actual_owner is original_owner and actual_function is original_function
                            and (name in vars(owner)) is present
                            and (not present or vars(owner)[name] is stored))
            except Exception as error:  # noqa: BLE001
                findings.append("restoration failed: " + phase + ": " + str(error))
            self.restored.append({"phase": phase, "attribute": name,
                                  "original_own_attribute": present,
                                  "restored": restored, "source": identity, "owner_object_id": id(owner)})
            if not restored:
                findings.append("binding not restored: " + phase)
        self.closed = True
        if findings:
            raise RuntimeError("; ".join(findings))


def install(counters, session, entries):
    """Bind precisely the frozen aliases, including each native engine."""
    for entry in entries:
        if entry["binding"] == "native-instance":
            group, attribute = entry["target"].split(".")
            for engine in (*session.prefill_engines, *session.decode_engines):
                frontend = engine.llm.llm_engine
                owner = (frontend.output_processor if group == "output_processor"
                         else frontend.engine_core.engine_core.scheduler)
                counters.bind(owner, attribute, entry["phase"])
        else:
            module, target = entry["target"].split(":")
            owner = importlib.import_module(module)
            parts = target.split(".")
            for part in parts[:-1]:
                owner = getattr(owner, part)
            counters.bind(owner, parts[-1], entry["phase"])


def counter_delta(before, after):
    if set(before) != set(after):
        raise ValueError("counter phase domain changed")
    return {phase: {key: row[key] - before[phase][key] for key in row}
            for phase, row in after.items()}


def profile_rows(profile):
    """Retain every function and caller key, including built-in functions."""
    stats = pstats.Stats(profile)
    rows = []
    for key, (primitive, calls, self_seconds, cumulative_seconds, callers) in sorted(stats.stats.items()):
        rows.append({"function": list(key), "primitive_calls": primitive, "calls": calls,
                     "self_seconds": self_seconds, "cumulative_seconds": cumulative_seconds,
                     "callers": [{"function": list(caller), "counts": list(counts)
                                  if isinstance(counts, tuple) else [counts]}
                                 for caller, counts in sorted(callers.items())]})
    return rows


def profiled_call(call, *, profile_path=None):
    """Profile one bounded call and restore the original absent hook."""
    if sys.getprofile() is not None:
        raise RuntimeError("diagnostic requires an absent prior profile hook")
    if profile_path is None:
        return call(), None
    profile = cProfile.Profile()
    try:
        profile.enable()
        result = call()
    finally:
        profile.disable()
        profile.dump_stats(str(profile_path))
    if sys.getprofile() is not None:
        raise RuntimeError("profile hook was not restored")
    return result, profile_rows(profile)
