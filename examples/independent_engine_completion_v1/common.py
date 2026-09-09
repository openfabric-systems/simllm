"""Strict evidence encodings and explicit source and publication domains."""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import asdict
from enum import Enum
from pathlib import Path

from examples.pd_session_identity_v1.run_study import exact_json_bytes, sha, write
from examples.pd_session_target_scale_v1.checks import Evidence, GuardFailure
from simllm.core.step import step_record_from_json, step_record_to_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREEZE_COMMIT = "73270f6699440aa4a17ec7b2af5155698ab643b2"
PUBLICATIONS = (
    "outcomes", "locality_outcomes", "collective_timing_outcomes", "collective_floor_timing_outcomes",
    "dependency_cross_check_reports", "collective_registration_outcomes", "packet_breakdowns",
    "bottleneck_reports", "session_evidence", "peer_evidence",
)
SOURCES = (
    "simllm/adapters/vllm/executor.py", "simllm/adapters/vllm/pd_session.py",
    "simllm/adapters/vllm/pd_connector.py", "simllm/adapters/vllm/independent.py",
    "simllm/core/engine_steps.py", "simllm/core/value_snapshot.py", "simllm/core/clock.py",
    "simllm/core/step.py", "simllm/core/execution.py", "simllm/core/runtime.py",
    "simllm/core/completion.py", "simllm/backends/step_sink.py", "simllm/compute/provider.py",
    "simllm/backends/step_lowerer.py", "simllm/traffic/execution_goal.py",
    "examples/pd_session_identity_v1/run_study.py", "examples/pd_session_v1/run_study.py",
    "examples/pd_session_target_scale_v1/native.py", "examples/pd_session_target_scale_v1/checks.py",
    "examples/independent_engine_completion_v1/common.py", "examples/independent_engine_completion_v1/native.py",
    "examples/independent_engine_completion_v1/reference.py", "examples/independent_engine_completion_v1/checks.py",
    "examples/independent_engine_completion_v1/run_study.py",
)


def parse_value(raw, label):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise GuardFailure("duplicate JSON member: " + label)
            result[key] = value
        return result

    result = json.loads(raw, object_pairs_hook=unique)

    def finite(value):
        if isinstance(value, dict):
            return all(finite(child) for child in value.values())
        if isinstance(value, list):
            return all(finite(child) for child in value)
        return type(value) is not float or math.isfinite(value)

    if not finite(result):
        raise GuardFailure("nonfinite JSON: " + label)
    return result


def read(path):
    raw = path.read_bytes()
    value = parse_value(raw, path.name)
    if raw != exact_json_bytes(value):
        raise GuardFailure("noncanonical JSON: " + path.name)
    return value


def step_stream(path):
    raw = path.read_bytes()
    rows = [parse_value(line, path.name) for line in raw.splitlines()]
    encoded = b"".join((json.dumps(step_record_to_json(step_record_from_json(row))) + "\n").encode("ascii")
                       for row in rows)
    if raw != encoded:
        raise GuardFailure("native step writer encoding: " + path.name)
    return rows


def source_snapshot(frozen):
    result = {}
    for kind, names in (("repository", SOURCES), ("native", frozen["native_source_sha256"])):
        origins = {name: str(Path(importlib.import_module(
            name.removesuffix(".py").replace("/", ".")).__file__).resolve()) for name in names}
        result[kind] = {"origins": origins, "sha256": {name: sha(Path(path).read_bytes()) for name, path in origins.items()}}
    return result


def json_value(value):
    """Project dataclass tuple and string-enum members to their JSON wire form."""
    def encode(item):
        if isinstance(item, Enum):
            return item.value
        raise TypeError("unsupported native evidence value: " + type(item).__qualname__)

    return json.loads(json.dumps(value, default=encode, allow_nan=False))


def sink_publications(engine):
    return json_value({name: [asdict(row) for row in getattr(engine.step_sink, name)] for name in PUBLICATIONS})


__all__ = ["Evidence", "GuardFailure", "exact_json_bytes", "sha", "write"]
