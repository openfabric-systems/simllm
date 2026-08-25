from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "offline/calibration/suites/qwen3.8-27b-text-v1/suite.json"
EXPECTATIONS = ROOT / "examples/model_extraction_qwen38_v1/expectations.json"
SUITE_SHA256 = "560aab048f7c9db463f53614178faded06a7d3b62b7e775f6943e1b52fbfe6e2"
WEIGHT_MANIFEST_SHA256 = (
    "72b5a8b6db0ad258d743ddbf3de4efda86b1ee894f08564f31044d921c17074c"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def test_qwen38_suite_identity_and_weight_metadata_are_frozen() -> None:
    raw = SUITE.read_bytes()
    suite = json.loads(raw)
    model = suite["reference_model"]
    shards = model["weight_shards"]

    assert hashlib.sha256(raw).hexdigest() == SUITE_SHA256
    assert model["name"] == "Qwen/Qwen3.8-27B"
    assert model["revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert model["parameter_count"] == 27_781_427_952
    assert model["weight_identity_source"] == "hugging-face-api-metadata"
    assert model["local_weight_byte_verification"] is False
    assert model["local_weight_verification_policy"] == (
        "intentionally-not-performed"
    )
    assert len(shards) == 18
    assert [shard["name"] for shard in shards] == sorted(
        shard["name"] for shard in shards
    )
    assert sum(shard["bytes"] for shard in shards) == 55_563_006_776
    assert hashlib.sha256(_canonical_bytes(shards)).hexdigest() == (
        WEIGHT_MANIFEST_SHA256
    )
    assert model["weight_sha256"] == WEIGHT_MANIFEST_SHA256


def test_qwen38_text_scope_and_hybrid_schedule_are_frozen() -> None:
    suite = _load(SUITE)
    model = suite["reference_model"]
    text = model["text_stack"]
    cells = suite["graph_cells"]
    schedule = text["layer_pattern"] * text["pattern_repetitions"]

    assert model["architecture"] == "Qwen3_5ForConditionalGeneration"
    assert model["model_type"] == "qwen3_5"
    assert text["model_type"] == "qwen3_5_text"
    assert suite["phase_scope"]["mode"] == "text-only"
    assert len(schedule) == model["geometry"]["layers"] == 64
    assert schedule.count("linear_attention") == 48
    assert schedule.count("full_attention") == 16
    assert len(cells) == len({cell["id"] for cell in cells}) == 15
    assert [cell["family"] for cell in cells].count("compute-prefill") == 5
    assert [cell["family"] for cell in cells].count("memory-decode") == 5
    assert [cell["family"] for cell in cells].count("dense-batch-decode") == 5


def test_qwen38_study_freezes_total_rejection_before_results() -> None:
    expectations = _load(EXPECTATIONS)
    contract = expectations["structure_contract"]
    relation_ids = {relation["id"] for relation in expectations["relations"]}

    assert expectations["working_tree_before_freeze"] == "clean"
    assert expectations["suite"]["sha256"] == SUITE_SHA256
    assert expectations["closure"]["closes"] == []
    assert expectations["closure"]["keeps_open"] == ["COMP-54", "COMP-62"]
    assert contract["required_outcome"] == (
        "reject-total-inventory-before-step-record-or-inventory-write"
    )
    assert contract["complete_inventory_count"] == 0
    assert relation_ids == {
        "R1-exact-framework-text-projection",
        "R2-hybrid-layer-scaling",
        "R3-shape-axis-sensitivity",
        "R4-total-rejection",
        "R5-rejection-byte-determinism",
        "R6-cross-framework-structural-agreement",
    }
    assert expectations["dropped_fatal_guards"] == [
        {
            "guard": "local-weight-shard-byte-and-hash-verification",
            "reason": (
                "maintainer-policy-forbids-model-weight-downloads-and-requires-"
                "api-metadata-identity"
            ),
        }
    ]
