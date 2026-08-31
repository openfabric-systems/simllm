from __future__ import annotations

import builtins
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
STUDY = ROOT / "examples" / "surrogate_conformance_v1"
RUNNER_PATH = STUDY / "run_study.py"
RECORD_PATH = STUDY / "record.json"
RECORD_SHA256 = "bfd9c185a9d4d87b1daa6244933a9aeaf57b298547a0a5c80c694418b6a9556c"
SURROGATE_CELL_SHA256 = (
    "0b55ca42f27e646867b4bb14e81c92f7d608af1d30c4da0cd7f6fdd1fb0b5859"
)
SCHEDULER_SHA256 = (
    "c67bda2886b52865ddafabaae7d797c359e930752f374421a33e537d94a5f45a"
)


def _load_runner():
    name = "simllm_surrogate_conformance_run_study"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _record() -> dict[str, object]:
    return json.loads(RECORD_PATH.read_text(encoding="utf-8"))


def test_published_record_and_csv_are_byte_locked_and_lf_pinned() -> None:
    payload = RECORD_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == RECORD_SHA256
    assert (STUDY / "record.sha256").read_text(encoding="utf-8") == (
        f"{RECORD_SHA256}  record.json\n"
    )
    assert b"\r" not in payload
    csv_payload = (STUDY / "results.csv").read_bytes()
    assert b"\r" not in csv_payload
    assert len(csv_payload.splitlines()) == 111
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    for pattern in (
        "examples/surrogate_conformance_v1/*.csv text eol=lf",
        "examples/surrogate_conformance_v1/*.json text eol=lf",
        "examples/surrogate_conformance_v1/*.md text eol=lf",
        "examples/surrogate_conformance_v1/*.py text eol=lf",
        "tests/test_surrogate_conformance_study.py text eol=lf",
    ):
        assert pattern in attributes.splitlines()


def test_published_verdict_keeps_evidence_classes_separate() -> None:
    record = _record()
    assert record["verdict"] == {
        "certified": False,
        "statement": "14 frozen family rows missed and bound the surrogate envelope",
        "status": "NOT CERTIFIED",
        "void": False,
    }
    assert record["family_tallies"] == {
        "F1": {"failed": 0, "passed": 4, "total": 4},
        "F2": {"failed": 0, "passed": 8, "total": 8},
        "F3": {"failed": 4, "passed": 0, "total": 4},
        "F4": {"failed": 3, "passed": 0, "total": 3},
        "F5": {"failed": 0, "passed": 4, "total": 4},
        "F6": {"failed": 1, "passed": 2, "total": 3},
        "F7": {"failed": 5, "passed": 0, "total": 5},
        "W": {"failed": 1, "passed": 0, "total": 1},
    }
    assert record["guard_tally"] == {"failed": 0, "passed": 78, "total": 78}
    assert record["wall_time"]["surrogate_to_live_ratio"] == pytest.approx(
        0.41643053636053423
    )


def test_f1_surrogate_cell_replays_without_vllm_binaries(monkeypatch) -> None:
    runner = _load_runner()
    config = runner.load_config()
    cell = next(
        cell
        for cell in runner.frozen_cells(config)
        if cell.cell_id == "f1-budget16-seqs2"
    )
    original_import = builtins.__import__

    def reject_vllm_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm" or name.startswith("vllm."):
            raise AssertionError("surrogate-only replay imported vLLM")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_vllm_import)
    summary = runner.surrogate_summary(runner.run_surrogate_cell(cell, config))
    assert hashlib.sha256(runner._canonical_bytes(summary)).hexdigest() == (
        SURROGATE_CELL_SHA256
    )


def test_cache_enabled_f7_omits_only_the_non_authoritative_free_projection() -> None:
    runner = _load_runner()
    config = runner.load_config()
    witnessed = set(config["families"]["F7"]["witnessed_actions"])
    cells = {cell.cell_id: cell for cell in runner.frozen_cells(config)}

    assert runner._f7_scored_actions(cells["f3-blocks3-seqs2"], witnessed) == (
        witnessed
    )
    assert runner._f7_scored_actions(
        cells["f4-one-full-prefix-block"], witnessed
    ) == witnessed - {"free"}


@pytest.mark.parametrize("family", ("F1", "F2", "F3", "F4", "F5", "F6", "F7"))
def test_live_family_record_requires_the_qualified_pin(family: str) -> None:
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("live families require the qualified vLLM 0.27.1 environment")
    if version != "0.27.1":
        pytest.skip("live families require the qualified vLLM 0.27.1 environment")

    distribution = importlib.metadata.distribution("vllm")
    scheduler = Path(
        distribution.locate_file("vllm/v1/core/sched/scheduler.py")
    )
    if not scheduler.is_file():
        pytest.skip("qualified vLLM scheduler source is unavailable")
    assert hashlib.sha256(scheduler.read_bytes()).hexdigest() == SCHEDULER_SHA256
    assert _record()["family_tallies"][family]["total"] > 0
