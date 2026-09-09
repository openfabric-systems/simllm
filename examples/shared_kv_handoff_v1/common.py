"""Source-bound encodings and complete read-only native evidence."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path

from examples.independent_engine_completion_v1.common import PUBLICATIONS, sink_publications
from examples.pd_session_v1 import run_study as baseline
from examples.publication_snapshot_v1.common import git, packages, read, receipts, sha, write
from examples.snapshot_dispatch_v1.common import runtime_identity

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREEZE = "2ad72f72d947e636ea384bfa40f365da27e1ec5a"
DEADLINE_FREEZE = "0e0e71aa586b2ccd2d786534d6b686d888e300b0"
MANDATORY_HELPERS = {
    "examples.shared_kv_handoff_v1.common", "examples.shared_kv_handoff_v1.capture",
    "examples.pd_session_target_scale_v1.native", "examples.independent_engine_completion_v1.native",
    "examples.independent_engine_completion_v1.common", "examples.pd_session_v1.run_study",
    "examples.snapshot_dispatch_v1.common", "examples.publication_snapshot_v1.common",
    "examples.completion_boundary_v1.run_study",
}


def plain(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    return value


def source_snapshot(repository, frozen):
    native = {}
    for name in frozen["native_source_sha256"]:
        module = importlib.import_module(name.removesuffix(".py").replace("/", "."))
        path = Path(module.__file__).resolve()
        native[name] = {"origin": str(path), "sha256": sha(path)}
    origins, helpers = {}, {}
    for name, module in tuple(sys.modules.items()):
        if not getattr(module, "__file__", None):
            continue
        if name == "simllm" or name.startswith("simllm."):
            path = Path(module.__file__).resolve()
            origins[name] = {"origin": str(path), "sha256": sha(path)}
        elif name.startswith("examples."):
            path = Path(module.__file__).resolve()
            helpers[name] = {"origin": str(path), "sha256": sha(path)}
    entry = Path(sys.modules["__main__"].__file__).resolve()
    prompt = baseline.TRACE_PATH.resolve()
    return {"repository": str(repository), "commit": git(repository, "rev-parse", "HEAD"),
            "packages": packages(repository), "origins": origins, "helpers": helpers,
            "native": native, "runtime": runtime_identity(),
            "entry": {"origin": str(entry), "sha256": sha(entry)},
            "prompt": {"origin": str(prompt), "sha256": sha(prompt)}}


def validate_sources(snapshot, repository, manifest, vllm_root, frozen):
    """Reject an imported overlay before any native engine is constructed."""
    repository = repository.resolve()
    if snapshot["packages"] != manifest or snapshot["repository"] != str(repository):
        raise ValueError("selected repository manifest changed")
    if set(snapshot["native"]) != set(frozen["native_source_sha256"]):
        raise ValueError("native source inventory is incomplete")
    if not {"simllm", "simllm.core.step", "simllm.adapters.vllm.pd_session"} <= set(snapshot["origins"]):
        raise ValueError("required serving import origins are absent")
    if not MANDATORY_HELPERS <= set(snapshot["helpers"]):
        raise ValueError("required native capture helper origins are absent")
    for name, row in snapshot["origins"].items():
        origin = Path(row["origin"])
        relative = origin.relative_to(repository).as_posix()
        module = relative.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
        if module != name or manifest.get(relative) != row["sha256"] or sha(origin) != row["sha256"]:
            raise ValueError("SimLLM import differs from its selected source: " + name)
    for name, row in snapshot["native"].items():
        origin = (vllm_root.parent / name).resolve()
        if row != {"origin": str(origin), "sha256": frozen["native_source_sha256"][name]} or sha(origin) != row["sha256"]:
            raise ValueError("native import differs from its pinned source: " + name)
    for name, row in snapshot["helpers"].items():
        relative = Path(*name.split("."))
        relative = relative / "__init__.py" if Path(row["origin"]).name == "__init__.py" else relative.with_suffix(".py")
        allowed = next((root / relative for root in (repository, ROOT) if (root / relative).is_file()), None)
        if allowed is None or row != {"origin": str(allowed.resolve()), "sha256": sha(allowed)}:
            raise ValueError("study helper import differs from source precedence: " + name)
    if snapshot["entry"] != {"origin": str(HERE / "native.py"), "sha256": sha(HERE / "native.py")}:
        raise ValueError("native worker entry file changed")
    trace = repository / "examples/preplay_trace_v1/granite_length_cap.jsonl"
    if snapshot["prompt"] != {"origin": str(trace.resolve()), "sha256": sha(trace)}:
        raise ValueError("prompt fixture origin changed")
    prior = json.loads((repository / "examples/pd_session_v1/expectations.json").read_bytes())
    if snapshot["prompt"]["sha256"] != prior["frontend"]["fixture_sha256"]:
        raise ValueError("prompt fixture bytes changed")


def native_configuration(engine):
    config = engine.llm.llm_engine.vllm_config
    return {"engine_id": engine.engine_id, "llm_engine_object_id": id(engine.llm.llm_engine),
            "dtype": str(config.model_config.dtype),
            "max_model_len": config.model_config.max_model_len,
            "num_gpu_blocks_override": config.cache_config.num_gpu_blocks_override}


def network_snapshot(owner):
    if owner is None:
        return {"enabled": False}
    session = owner._session
    stream = None if session is None else session._stream
    return plain({
        "enabled": True, "owner_id": id(owner), "clock_id": None if owner._clock is None else id(owner._clock),
        "session_id": None if session is None else id(session),
        "child_pid": None if stream is None else stream.pid,
        "child_exit_code": None if stream is None else stream._child.process.returncode,
        "config": owner._config, "engine_ranks": owner._engine_ranks, "engine_roles": owner._engine_roles,
        "accepted": owner.accepted_shards, "events": owner.native_events, "rows": owner.native_rows,
        "joins": [join.to_json() for join in owner.joins],
        "pending": list(owner._pending), "staged": list(owner._staged),
        "poisoned": owner.poisoned, "closed": owner._closed, "drain": owner._drain,
        "last_sequence": None if session is None else session.last_accepted_sequence,
    })


__all__ = ["DEADLINE_FREEZE", "FREEZE", "HERE", "PUBLICATIONS", "ROOT", "git", "network_snapshot",
           "packages", "plain", "read", "receipts", "sha", "sink_publications", "source_snapshot", "write"]
