from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.frontier_comparison_v1 import run_study

STUDY_DIR = REPOSITORY_ROOT / "examples" / "frontier_comparison_v1"
RESULT_PATH = STUDY_DIR / "results.json"
RESULT_SHA256 = "753cd31e99010c36d1314633e41b51161461bfd36f762278f1a622bed585a387"
ARTIFACT_SHA256 = {
    "RESULTS.md": "d599d7fd79a23074e65dde0654124f89dc7fd726230447e930c3900e976d21d9",
    "figures/frontier-comparison.pdf": (
        "60385e28e33dc677bfac6969963bd290d16faa91a6faee4e9f4b689a58f4573e"
    ),
    "figures/frontier-comparison.png": (
        "5079295158a3f76f95e63ebf27ca3ed45ff85fff8ed78659ac16aeff36e0180a"
    ),
    "results.csv": "d5ab69810da0dc94b25df41f26c29d0e73252b7cbe33dd5730727e0a3f936f16",
    "results.json": RESULT_SHA256,
}


def test_frontier_comparison_publication_bytes_are_locked() -> None:
    for relative, expected in ARTIFACT_SHA256.items():
        assert hashlib.sha256((STUDY_DIR / relative).read_bytes()).hexdigest() == (
            expected
        )


def test_frontier_comparison_frozen_inputs_are_unchanged() -> None:
    assert run_study.sha256_file(run_study.EXPECTATIONS_PATH) == (
        run_study.EXPECTATIONS_SHA256
    )
    rows, all_match = run_study._external_hashes()
    assert all_match is True
    assert len(rows) == 4


def test_published_frontier_comparison_record_is_locked() -> None:
    payload = RESULT_PATH.read_bytes()
    result = json.loads(payload)

    assert hashlib.sha256(payload).hexdigest() == RESULT_SHA256
    assert result["schema"] == run_study.RESULT_SCHEMA
    assert result["nonvoid"] is True
    assert all(guard["held"] for guard in result["fatal_guards"])
    assert result["pricing_subprocess_count"] == 0
    assert result["external_rows_entered_pricing"] is False
    assert result["acceptance"] == {
        "nonvoid": True,
        "wall_time_pass": True,
        "x1_pass": True,
        "x2_pass": True,
        "x3a_pass": True,
        "x3b_pass": True,
        "x3c_pass": True,
    }
    assert result["families"]["X3"]["X3b"]["passed"] == 10
    assert result["families"]["X3"]["X3c"]["passed"] == 9
    assert result["families"]["X3"]["X3c"]["rows"][-1]["miss_direction"] == (
        "below-0.6"
    )


def test_x1_and_x2_decode_direction_rerun_without_processes() -> None:
    inventory = run_study._load_json(run_study.INVENTORY_PATH)
    suite = run_study._load_json(run_study.SUITE_PATH)
    derivation = run_study.derive_work(inventory)

    with run_study.ProcessGuard() as guard:
        x1 = run_study._x1(
            derivation,
            suite["reference_model"]["parameter_count"],
        )
        x2 = run_study._x2(derivation)

    assert guard.attempts == []
    assert x1["passed"] == x1["denominator"] == 3
    assert x1["candidate_key"] == (
        "9981c2a9ebe89ed40002b9e1ce952ca1d742fa2b082e7f306a21a72d3966b9f9"
    )
    decode_direction = next(row for row in x2["rows"] if row["id"] == "X2a")
    assert decode_direction == {
        "id": "X2a",
        "passed": True,
        "predicted_ps": 5_379_515_733,
        "external_ps": 9_179_000_000,
    }
