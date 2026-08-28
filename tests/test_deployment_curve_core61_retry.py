from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples/deployment_curve_v1"
FREEZE_PATH = STUDY_DIR / "core61_depth_retry_expectations.json"


def _load(name: str):
    path = STUDY_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def analyzer():
    return _load("core61_depth_retry_analyze.py")


@pytest.fixture(scope="module")
def scorer():
    return _load("core61_depth_retry_score.py")


def _scheduler_marker() -> dict:
    computed = {f"decode_b32_c2000-{index:02d}": 2000 for index in range(32)}
    return {
        "cell": "decode_b32_c2000",
        "scheduler": {
            "is_decode": True,
            "num_requests": 32,
            "cached_num_computed_tokens_by_request": computed,
        },
    }


def _write_capture(run_dir: Path) -> None:
    profile = {
        "schema": "simllm-core61-depth8-retry-capture-v1",
        "model": "deepseek-ai/DeepSeek-V3",
        "revision": "e815299b0bcbac849fa540c768ef21845365c9eb",
        "model_config": {
            "config_sha256": (
                "cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9"
            )
        },
        "framework": {"name": "vllm", "version": "0.27.1+cu129"},
        "machine": {"slurm_job_id": "fixture-job"},
        "reduced_layers": 8,
        "startup_max_num_batched_tokens": 4096,
        "max_num_seqs": 32,
        "cases": [
            {
                "cell": "decode_b32_c2000",
                "pool": "decode",
                "batch_size": 32,
                "remote_kv_tokens_per_request": 2000,
                "decode_steps": 1,
                "started_epoch_ns": 1_000_100,
                "finished_epoch_ns": 1_001_000,
                "scheduler_marker": _scheduler_marker(),
            }
        ],
    }
    (run_dir / "profile.json").write_text(
        json.dumps(profile) + "\n", encoding="utf-8", newline="\n"
    )
    database = sqlite3.connect(run_dir / "profile.sqlite")
    database.executescript(
        """
        CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs INTEGER);
        INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (1000000);
        CREATE TABLE StringIds (id INTEGER, value TEXT);
        INSERT INTO StringIds VALUES (1, 'gemm_kernel');
        INSERT INTO StringIds VALUES (2, 'ncclKernel');
        CREATE TABLE NVTX_EVENTS (
            start INTEGER, end INTEGER, text TEXT, textId INTEGER
        );
        INSERT INTO NVTX_EVENTS VALUES (
            200, 800, 'execute_context_0(0)_generation_32(32)', NULL
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER, end INTEGER, correlationId INTEGER
        );
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (250, 260, 7);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (300, 310, 8);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER, demangledName INTEGER, correlationId INTEGER
        );
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (400, 430, 1, 7);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (440, 450, 2, 8);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (460, 480, 1, 8);
        """
    )
    database.commit()
    database.close()
    for name in (
        "profile.nsys-rep",
        "alignment.json",
        "harness_sha256.txt",
        "job.log",
    ):
        (run_dir / name).write_text(name + "\n", encoding="utf-8", newline="\n")
    (run_dir / "weight_files.txt").write_text("", encoding="utf-8", newline="\n")


def test_analyzer_extracts_only_the_exact_scheduler_guarded_boundary(
    analyzer, tmp_path: Path
):
    _write_capture(tmp_path)

    result = analyzer.analyze(tmp_path)

    assert result["status"] == "DIGEST_READY_MEASUREMENT"
    assert result["measured_service_ps"] == (30 + 20) * 1000
    assert result["collective_service_ps"] == 10 * 1000
    assert result["boundary"] == {
        "basis": "exact-nvtx-runtime-correlation",
        "label": "execute_context_0(0)_generation_32(32)",
        "runtime_correlation_count": 2,
        "kernel_record_count": 3,
    }
    assert len(result["sources"]) == 7


def test_analyzer_fails_closed_when_one_kv_length_changes(analyzer, tmp_path: Path):
    _write_capture(tmp_path)
    profile_path = tmp_path / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    computed = profile["cases"][0]["scheduler_marker"]["scheduler"][
        "cached_num_computed_tokens_by_request"
    ]
    computed["decode_b32_c2000-31"] = 1999
    profile_path.write_text(
        json.dumps(profile) + "\n", encoding="utf-8", newline="\n"
    )

    with pytest.raises(ValueError, match="not exact batch-32 KV-2000"):
        analyzer.analyze(tmp_path)


def _measurement(measured_service_ps: int) -> dict:
    return {
        "schema": "simllm-core61-depth8-retry-measurement-v1",
        "status": "DIGEST_READY_MEASUREMENT",
        "evidence_class": "MEASURED",
        "shape": {
            "depth_layers": 8,
            "batch_size": 32,
            "remote_kv_tokens_per_request": 2000,
        },
        "measured_service_ps": measured_service_ps,
    }


def test_scorer_keeps_the_frozen_sign_and_validates_a_close_measurement(scorer):
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))

    result = scorer.score(freeze, _measurement(3_800_000_000))

    score = result["score"]
    assert score["signed_residual_ps"] == 48_640_489
    assert score["prediction_within_tolerance"] is True
    assert score["linearity_verdict"] == "VALIDATED_LINEAR_DEPTH_SCALING"
    assert result["registry"]["core61"] == "CLOSE_LINEARITY_VALIDATED"
    assert result["registry"]["core63"] == "NOT_REGISTERED"
    assert result["signed_residual_ledger"][1]["owner"] == "TRAF-66"
    assert result["signed_residual_ledger"][1]["signed_ps"] is None


def test_scorer_publishes_a_negative_miss_without_moving_the_bar(scorer):
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))

    result = scorer.score(freeze, _measurement(3_000_000_000))

    score = result["score"]
    assert score["signed_residual_ps"] == -751_359_511
    assert score["signed_residual_percent"] == "-25.045317"
    assert score["prediction_within_tolerance"] is False
    assert score["linearity_verdict"] == "QUANTIFIED_DEPTH_NONLINEARITY"
    assert result["registry"]["core61"] == "OPEN_REFUTED"
    assert result["registry"]["core63"] == "REGISTER_IF_RESIDUAL_REMAINS"


def test_capture_harness_pins_offline_config_and_exact_alignment_route():
    capture = (STUDY_DIR / "core61_depth_retry_capture.py").read_text(encoding="utf-8")
    batch = (STUDY_DIR / "core61_depth_retry.sbatch").read_text(encoding="utf-8")
    hook = (
        STUDY_DIR / "core61_depth_retry_hook/sitecustomize.py"
    ).read_text(encoding="utf-8")

    assert "local_files_only=True" in capture
    assert "startup-max-num-batched-tokens" in capture
    assert "five alignment rounds" in capture
    assert "VLLM_ENABLE_V1_MULTIPROCESSING=0" in batch
    assert "HF_HUB_OFFLINE=1" in batch
    assert "core61_exact_decode" in hook
    assert "target_kv_tokens" in hook
    assert "typing_extensions" not in capture + hook
    assert "—" not in capture + batch + hook
