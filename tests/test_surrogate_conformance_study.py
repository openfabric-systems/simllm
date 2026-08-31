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
RECORD_SHA256 = "f6f0687eff604c7c664318dcd75810452649bec1816fb6f7b7dec1ffb5426a8b"
RESULTS_CSV_SHA256 = (
    "214d236e9af5b6a2210df0bfbd906a3e57f0329f34892159b218e13b6f01777f"
)
SUPERSEDED_RECORD_SHA256 = (
    "bfd9c185a9d4d87b1daa6244933a9aeaf57b298547a0a5c80c694418b6a9556c"
)
FROZEN_INPUT_SHA256 = {
    "expectations.md": "b9087511cf482201b7d6a0b619f9454101de06bb22ec7df7c0531f2a75a0d4cc",
    "freeze_amendment.md": (
        "3df4f3319126381f4603b2ceef6d941da5f586764e604ba14543f44d30d7e813"
    ),
    "study_config.json": (
        "6c8271d8ef0061665cea5d5eddf72097e24408fe451531d10572fb1fd64be3d4"
    ),
}
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
    assert hashlib.sha256(csv_payload).hexdigest() == RESULTS_CSV_SHA256
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


def test_scoring_correction_keeps_every_frozen_input_byte_identical() -> None:
    assert {
        name: hashlib.sha256((STUDY / name).read_bytes()).hexdigest()
        for name in FROZEN_INPUT_SHA256
    } == FROZEN_INPUT_SHA256


def test_published_verdict_keeps_evidence_classes_separate() -> None:
    record = _record()
    assert record["verdict"] == {
        "certified": False,
        "statement": (
            "F1, F2, F3, F5, F7 pass their frozen rows; F4, F6, W prevent "
            "certification"
        ),
        "status": "NOT CERTIFIED",
        "void": False,
    }
    assert record["family_tallies"] == {
        "F1": {"failed": 0, "passed": 4, "total": 4},
        "F2": {"failed": 0, "passed": 8, "total": 8},
        "F3": {"failed": 0, "passed": 4, "total": 4},
        "F4": {"failed": 3, "passed": 0, "total": 3},
        "F5": {"failed": 0, "passed": 4, "total": 4},
        "F6": {"failed": 3, "passed": 0, "total": 3},
        "F7": {"failed": 0, "passed": 5, "total": 5},
        "W": {"failed": 1, "passed": 0, "total": 1},
    }
    assert record["certification_scope"] == {
        "failing_families": ["F4", "F6", "W"],
        "passing_families": ["F1", "F2", "F3", "F5", "F7"],
        "statement": (
            "only the complete frozen families listed as passing are qualified; "
            "the surrogate loop as a whole is not certified"
        ),
    }
    assert record["guard_tally"] == {"failed": 0, "passed": 78, "total": 78}
    assert record["wall_time"]["surrogate_to_live_ratio"] == pytest.approx(
        0.4386909057289039
    )


def test_postspecified_record_preserves_before_after_and_nonvacuous_control() -> None:
    record = _record()
    correction = record["scoring_correction"]
    assert correction["classification"] == "post-specified-scoring-correction"
    assert correction["superseded_attempt_id"] == "attempt-003"
    assert correction["superseded_record_sha256"] == SUPERSEDED_RECORD_SHA256
    assert correction["superseded_family_tallies"]["F3"] == {
        "failed": 4,
        "passed": 0,
        "total": 4,
    }
    assert correction["superseded_family_tallies"]["F6"] == {
        "failed": 1,
        "passed": 2,
        "total": 3,
    }
    assert correction["superseded_family_tallies"]["F7"] == {
        "failed": 5,
        "passed": 0,
        "total": 5,
    }

    guard = next(
        row
        for row in record["guards"]
        if row["cell_id"] == "end-to-end-mutation-controls"
    )
    assert guard["status"] == "PASS"
    assert guard["actual"]["kv_mutation"] == {
        "baseline_mismatch_count": 0,
        "baseline_status": "PASS",
        "mutant_mismatch_count": 1,
        "mutant_status": "FAIL",
        "pass_to_fail": True,
        "source_cell": "f3-blocks3-seqs2",
    }


def test_cache_enabled_f7_free_divergences_are_recorded_but_unscored() -> None:
    rows = {
        row["cell_id"]: row
        for row in _record()["checks"]
        if row["family"] == "F7"
    }
    assert all(row["status"] == "PASS" for row in rows.values())
    assert {
        cell_id: row["actual"]["free_projection_observation"]["mismatch_count"]
        for cell_id, row in rows.items()
        if row["actual"]["free_projection_observation"] is not None
    } == {
        "f4-one-full-prefix-block": 4,
        "f4-several-full-prefix-blocks": 6,
        "f4-zero-full-prefix-blocks": 4,
    }


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
