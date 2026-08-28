from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = REPOSITORY_ROOT / "offline/calibration/kernel_cycle_framework_target.py"


def _target():
    spec = importlib.util.spec_from_file_location("kernel_cycle_framework_target", TARGET_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell() -> dict:
    return {
        "cell_id": "vllm-decode-eager-te1-pi1-da1-ex1-b2-kv4-fresh-contiguous",
        "framework": "vllm",
        "kv_placement": "fresh-contiguous",
        "launch_mode": "eager",
        "model": {"name": "org/model", "revision": "pinned-revision"},
        "parallelism": {
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "data_parallel": 1,
            "expert_parallel": 1,
        },
        "pool": "decode",
        "replays": 64,
        "routing_evidence_required": False,
        "shape": {"batch_size": 2, "per_request_kv_lengths": [4, 4]},
    }


def test_cell_loader_pins_identity_and_rejects_unimplemented_parallelism(tmp_path: Path):
    target = _target()
    path = tmp_path / "cell.json"
    path.write_text(json.dumps(_cell()), encoding="utf-8", newline="\n")

    assert target.load_cell(path)["model"]["revision"] == "pinned-revision"

    changed = _cell()
    changed["parallelism"]["pipeline_parallel"] = 2
    path.write_text(json.dumps(changed), encoding="utf-8", newline="\n")
    with pytest.raises(RuntimeError, match="pipeline parallelism"):
        target.load_cell(path)


def test_prompt_rows_preserve_exact_decode_kv_lengths():
    target = _target()

    rows, output_tokens = target.prompt_rows(_cell())

    assert list(map(len, rows)) == [4, 4]
    assert rows[0] != rows[1]
    assert output_tokens == 2


def test_canonical_first_cell_fails_closed_on_unprovable_contracts():
    target = _target()
    first = _cell()
    first.update(
        {
            "cell_id": (
                "sglang-decode-cuda-graph-te1-pi1-da1-ex1-b1-kv1311-deliberately-fragmented"
            ),
            "framework": "sglang",
            "kv_placement": "deliberately-fragmented",
            "launch_mode": "cuda-graph",
            "routing_evidence_required": True,
            "shape": {"batch_size": 1, "per_request_kv_lengths": [1311]},
        }
    )

    gaps = target.capability_gaps(first)

    assert first["cell_id"] == (
        "sglang-decode-cuda-graph-te1-pi1-da1-ex1-b1-kv1311-deliberately-fragmented"
    )
    assert any("deliberately-fragmented" in gap for gap in gaps)
    assert any("routed-expert sidecar" in gap for gap in gaps)
    assert any("two-clean-run code-object" in gap for gap in gaps)
    with pytest.raises(RuntimeError, match="cannot be scored"):
        target.run_target(first, Path("unused.json"))
