from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples" / "pd_session_batching_service_v1"
PRICING_SHA256 = "ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner():
    previous = sys.modules.get("service_model")
    sys.modules["service_model"] = _module(
        STUDY_DIR / "service_model.py",
        "vllm42_test_service_model",
    )
    try:
        return _module(STUDY_DIR / "run_study.py", "vllm42_test_runner")
    finally:
        if previous is None:
            del sys.modules["service_model"]
        else:
            sys.modules["service_model"] = previous


def _module_with_service_model(path: Path, name: str):
    previous = sys.modules.get("service_model")
    sys.modules["service_model"] = _module(
        STUDY_DIR / "service_model.py",
        f"{name}_service_model",
    )
    try:
        return _module(path, name)
    finally:
        if previous is None:
            del sys.modules["service_model"]
        else:
            sys.modules["service_model"] = previous


def _request(cell_key: tuple[int, int, int, int], index: int) -> dict:
    stable = "-".join(map(str, cell_key)) + f"-{index}"
    return {
        "request_id": f"request-{stable}",
        "expected_admitted_at_ps": index,
        "timeline": {
            "admitted_at_ps": index,
            "ttft_ps": 0,
            "decomposition": {
                "total_ps": 0,
                "prefill_queue_ps": 10,
                "decode_admission_wait_ps": 20,
            },
        },
        "prefill_internal_request_id": f"prefill-{stable}",
        "decode_internal_request_id": f"decode-{stable}",
        "decode_token_ids": [1, 2, 3, 4],
        "compute_pricing": {
            "prefill": None,
            "decode": {
                "record_sha256": PRICING_SHA256,
                "acceptance_status": "candidate",
                "calibration_claim": False,
            },
        },
    }


def _observation(split: str) -> tuple[dict, dict]:
    freeze = json.loads(
        (STUDY_DIR / "expectations.json").read_text(encoding="utf-8")
    )
    predictions = [
        row for row in freeze["prediction_bands"] if row["split"] == split
    ]
    cells = []
    for prediction in predictions:
        key = (
            *prediction["configuration"],
            prediction["offered_load_requests_per_second"],
        )
        cells.append(
            {
                "prefill_engines": key[0],
                "decode_engines": key[1],
                "prompt_tokens": key[2],
                "offered_load_requests_per_second": key[3],
                "requests": [_request(key, index) for index in range(64)],
                "maximum_prefill_batch_size": 1,
                "maximum_decode_batch_size": prediction[
                    "central_maximum_decode_batch_size"
                ],
                "amortized_batching_service_per_token_ps": prediction[
                    "predicted_batch_service_per_token_ps"
                ],
            }
        )
    return {"split": split, "cells": cells}, freeze


def test_split_selection_preserves_the_frozen_disclosure_counts() -> None:
    runner = _runner()

    non_held = sum(
        runner._cell_selected("non-held-out", prefill, decode, load)
        for prefill, decode in runner.POOL_RATIOS
        for _prompt in runner.PROMPT_LENGTHS
        for load in runner.OFFERED_LOADS
    )
    held = sum(
        runner._cell_selected("held-out", prefill, decode, load)
        for prefill, decode in runner.POOL_RATIOS
        for _prompt in runner.PROMPT_LENGTHS
        for load in runner.OFFERED_LOADS
    )

    assert non_held == 48
    assert held == 30


def test_exact_synthetic_non_held_out_projection_passes() -> None:
    runner = _runner()
    observation, freeze = _observation("non-held-out")

    analysis = runner.analyze_observation(observation, freeze)

    assert analysis["status"] == "PASS"
    assert analysis["fatal_guards"]["status"] == "HELD"
    assert analysis["service_band_summary"] == {
        "held": 48,
        "missed": 0,
        "evaluated": 48,
    }
    assert analysis["conservation"] == {
        "cells": 48,
        "admissions": 3072,
        "handoffs": 3072,
        "terminals": 3072,
        "terminal_decode_tokens": 12288,
        "maximum_ttft_residual_ps": 0,
    }


def test_service_miss_refutes_without_voiding() -> None:
    runner = _runner()
    observation, freeze = _observation("non-held-out")
    mutation = deepcopy(observation)
    mutation["cells"][0]["amortized_batching_service_per_token_ps"] = {
        "numerator": 1_000_000_000,
        "denominator": 1,
    }

    analysis = runner.analyze_observation(mutation, freeze)

    assert analysis["status"] == "REFUTED"
    assert analysis["fatal_guards"]["status"] == "HELD"
    assert analysis["service_band_summary"]["missed"] == 1


def test_conservation_failure_voids_the_split() -> None:
    runner = _runner()
    observation, freeze = _observation("non-held-out")
    mutation = deepcopy(observation)
    mutation["cells"][0]["requests"].pop()

    analysis = runner.analyze_observation(mutation, freeze)

    assert analysis["status"] == "VOID"
    assert analysis["fatal_guards"]["status"] == "VIOLATED"


def test_publisher_combines_only_released_unique_splits() -> None:
    publisher = _module(
        STUDY_DIR / "publish_results.py",
        "vllm42_test_publisher",
    )
    runner = _runner()
    publications = {}
    for split in ("non-held-out", "held-out"):
        observation, freeze = _observation(split)
        analysis = runner.analyze_observation(observation, freeze)
        provenance = {
            "freeze_commit": publisher.FREEZE_COMMIT,
            "expectations_sha256": publisher.EXPECTATIONS_SHA256,
            "split": split,
            "run_head": "a" * 40,
            "non_held_out_publication": (
                {"commit": "b" * 40, "sha256": "c" * 64}
                if split == "held-out"
                else None
            ),
        }
        raw = {
            "schema": publisher.SPLIT_RESULT_SCHEMA,
            "provenance": provenance,
            "observation": {
                **observation,
                "runtime": {"offline": True, "cluster_time": False},
            },
            "analysis": analysis,
        }
        publications[split] = publisher.publish_split(raw, "d" * 64)

    combined = publisher.combine_publications(
        publications["non-held-out"],
        publications["held-out"],
    )

    assert combined["status"] == "PASS"
    assert combined["service_band_summary"]["held"] == 78
    assert combined["closure"] == {"VLLM-42": "CLOSED", "VLLM-50": "UNUSED"}


def test_merge_accepts_one_complete_non_held_out_shard_set() -> None:
    merger = _module_with_service_model(
        STUDY_DIR / "merge_cells.py",
        "vllm42_test_merger",
    )
    observation, _freeze = _observation("non-held-out")
    provenance = {"freeze_commit": "f" * 40}
    runtime = {"python": "3.10.18", "vllm": "0.27.1"}
    documents = [
        {
            "schema": merger.CELL_RESULT_SCHEMA,
            "provenance": provenance,
            "runtime": runtime,
            "split": "non-held-out",
            "cell": cell,
            "onset_scored": False,
            "monotonic_direction_scored": False,
        }
        for cell in reversed(observation["cells"])
    ]

    merged = merger.merge_cell_documents("non-held-out", documents)

    assert merged["provenance"] == provenance
    assert len(merged["cells"]) == 48
    assert [tuple(cell[key] for key in (
        "prefill_engines",
        "decode_engines",
        "prompt_tokens",
        "offered_load_requests_per_second",
    )) for cell in merged["cells"]] == sorted(
        tuple(cell[key] for key in (
            "prefill_engines",
            "decode_engines",
            "prompt_tokens",
            "offered_load_requests_per_second",
        ))
        for cell in observation["cells"]
    )
