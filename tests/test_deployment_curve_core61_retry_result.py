from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples/deployment_curve_v1"
RESULT_PATH = STUDY_DIR / "core61_depth_retry_result.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_published_depth_retry_is_the_exact_registered_pass():
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["status"] == "SCORED"
    assert result["task"] == "CORE-61"
    assert result["score"] == {
        "absolute_residual_percent": "3.355537",
        "held_out_tolerance_percent": "5",
        "linearity_verdict": "VALIDATED_LINEAR_DEPTH_SCALING",
        "measured_service_ps": 3_629_568_000,
        "prediction_within_tolerance": True,
        "preregistered_prediction_ps": 3_751_359_511,
        "signed_residual_percent": "-3.355537",
        "signed_residual_ps": -121_791_511,
    }
    assert result["registry"] == {
        "comp76": "UNCHANGED",
        "core61": "CLOSE_LINEARITY_VALIDATED",
        "core63": "NOT_REGISTERED",
    }


def test_published_measurement_retains_the_exact_scheduler_boundary():
    measurement = json.loads(RESULT_PATH.read_text(encoding="utf-8"))["measurement"]

    assert measurement["job_id"] == "200138"
    assert measurement["shape"] == {
        "batch_size": 32,
        "depth_layers": 8,
        "remote_kv_tokens_per_request": 2000,
    }
    assert measurement["boundary"] == {
        "basis": "exact-nvtx-runtime-correlation",
        "kernel_record_count": 236,
        "label": "execute_context_0(0)_generation_32(32)",
        "runtime_correlation_count": 36,
    }
    assert measurement["collective_service_ps"] == 0
    scheduler = measurement["scheduler_guard"]
    assert scheduler["is_decode"] is True
    assert scheduler["new_request_count"] == 0
    assert scheduler["num_requests"] == 32
    assert scheduler["total_num_scheduled_tokens"] == 32
    assert set(scheduler["cached_num_computed_tokens_by_request"].values()) == {
        2000
    }
    assert {source["name"] for source in measurement["sources"]} == {
        "alignment.json",
        "harness_sha256.txt",
        "job.log",
        "profile.json",
        "profile.nsys-rep",
        "profile.sqlite",
        "weight_files.txt",
    }


def test_published_result_closes_only_the_literal_depth_registry():
    core = (REPOSITORY_ROOT / "docs/modules/core.md").read_text(encoding="utf-8")
    compute = (REPOSITORY_ROOT / "docs/modules/compute.md").read_text(
        encoding="utf-8"
    )
    ledger = json.loads(
        (REPOSITORY_ROOT / "docs/task-ledger.json").read_text(encoding="utf-8")
    )

    assert "- CORE-61 (" not in core
    assert "so CORE-61 is\ncomplete" in core
    assert core.count("- CORE-63 (Precision; P0; M):") == 1
    assert "CORE-61" in ledger["closed"]
    assert "COMP-72 and COMP-78 remain\n  open on the 0-of-1,212 Granite" in compute
    assert "COMP-76 remains untouched" in (
        STUDY_DIR / "core61_depth_retry_result.md"
    ).read_text(encoding="utf-8")


def test_result_manifest_locks_the_published_pair():
    manifest_path = STUDY_DIR / "core61_depth_retry_result.sha256"
    rows = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        rows[name] = digest

    assert rows == {
        "core61_depth_retry_result.json": _sha256(RESULT_PATH),
        "core61_depth_retry_result.md": _sha256(
            STUDY_DIR / "core61_depth_retry_result.md"
        ),
    }
