"""Regression checks for the frozen ideal host-step compatibility guard."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
CHECK_PATH = (
    REPOSITORY / "examples/host_step_cost_v1/check_ideal_compatibility.py"
)


def _check_module():
    spec = importlib.util.spec_from_file_location("host_step_cost_ideal", CHECK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_mission_run(tmp_path: Path, module: Any) -> tuple[Path, dict[str, Any]]:
    run_dir = tmp_path / "mission"
    cells = run_dir / "cells"
    names = [
        "a-ep4-400g",
        "a-ep8-100g",
        "a-ep8-200g",
        "a-ep8-400g",
        "b-ep8-400g",
    ]
    step_digests: dict[str, str] = {}
    for index, name in enumerate(names):
        cell_dir = cells / name
        cell_dir.mkdir(parents=True)
        _write_json(
            cell_dir / "cell.json",
            {
                "name": name,
                "timestamp_ps": index,
                "wall_seconds": 10.0 + index,
                "nested": {"wall_seconds": 100.0 + index},
            },
        )
        step_payload = f'{{"cell":"{name}","timestamp_ps":{index}}}\n'.encode()
        (cell_dir / "steps.jsonl").write_bytes(step_payload)
        step_digests[name] = hashlib.sha256(step_payload).hexdigest()

    relations = {f"E{index}": {"passed": True} for index in range(1, 14)}
    _write_json(
        run_dir / "summary.json",
        {
            "void": False,
            "violated_fatal_guards": [],
            "exact_oracle_relations": relations,
            "exact_oracle_passed": 13,
            "exact_oracle_total": 13,
            "behavioral_passed": 0,
            "behavioral_total": 4,
        },
    )
    digest, missing = module._canonical_cell_digest(module._cell_paths(run_dir))
    assert missing == []
    return run_dir, {
        "study": "end_to_end_replay_v1",
        "aggregate_cell_sha256": digest,
        "step_record_sha256": step_digests,
    }


def test_canonical_digest_removes_only_top_level_wall_seconds(tmp_path: Path):
    module = _check_module()
    path = tmp_path / "cell.json"
    _write_json(
        path,
        {
            "wall_seconds": 9.0,
            "nested": {"wall_seconds": 4.0},
            "timestamp_ps": 7,
        },
    )
    digest, missing = module._canonical_cell_digest({"cell": path})
    payload = (
        json.dumps(
            {"cell": {"nested": {"wall_seconds": 4.0}, "timestamp_ps": 7}},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()

    assert missing == []
    assert digest == hashlib.sha256(payload).hexdigest()


def test_off_guard_accepts_exact_nonvoid_mission_despite_behavioral_miss(
    tmp_path: Path,
):
    module = _check_module()
    run_dir, frozen = _make_mission_run(tmp_path, module)

    evaluation = module._evaluate(run_dir, frozen)
    result = module._result_document("1" * 40, evaluation)

    assert result["run_status"] == "nonvoid"
    assert result["evidence"] == {
        "class": "fatal_unscored_guard",
        "guard": module.GUARD_NAME,
        "held": True,
        "findings": [],
    }
    assert "behavioral" not in result


def test_off_guard_is_void_when_one_step_stream_changes(tmp_path: Path):
    module = _check_module()
    run_dir, frozen = _make_mission_run(tmp_path, module)
    changed = run_dir / "cells" / "a-ep8-400g" / "steps.jsonl"
    changed.write_text('{"changed":true}\n', encoding="utf-8")

    result = module._result_document("2" * 40, module._evaluate(run_dir, frozen))

    assert result["run_status"] == "void"
    assert result["evidence"]["held"] is False
    assert result["evidence"]["findings"] == [
        "a-ep8-400g steps.jsonl identity differs from the freeze"
    ]


def test_check_only_needs_no_mission_artifacts(
    tmp_path: Path, monkeypatch, capsys
):
    module = _check_module()
    result_path = tmp_path / "ideal.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(CHECK_PATH),
            "--run-dir",
            str(tmp_path / "does-not-exist"),
            "--result",
            str(result_path),
            "--check-only",
        ],
    )

    module.main()

    output = json.loads(capsys.readouterr().out)
    assert output["check_only"] is True
    assert output["fatal_unscored_guard"] == module.GUARD_NAME
    assert not result_path.exists()
