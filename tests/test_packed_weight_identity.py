"""Check packed storage bounds through the existing metadata identity parser."""

import importlib.util
import json
from pathlib import Path

import pytest

from simllm.calibration.canonical import canonical_bytes, canonical_sha256
from simllm.calibration.extraction import (
    MXFP4_GROUP32_QUANTIZATION,
    ModelExtractionError,
    _expected_dims,
    load_extraction_suite,
)

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "examples/packed_weight_identity_v1"


@pytest.fixture
def study():
    spec = importlib.util.spec_from_file_location(
        "packed_identity_test_study", HERE / "run_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def suite(study):
    frozen = json.loads((HERE / "expectations.json").read_text())
    template = json.loads((ROOT / frozen["identity_template"]).read_bytes())
    return study.synthetic_suite(template, 512, 320, MXFP4_GROUP32_QUANTIZATION)


@pytest.mark.parametrize(
    "parameters,floor", [(31, 17), (32, 17), (33, 19), (512, 272), (1024, 544)]
)
def test_group_boundaries_accept_exact_floor_and_reject_one_byte_less(
    study, suite, parameters, floor
):
    assert study.independent_floor(parameters) == floor
    for extra in (0, 48):
        candidate = study.synthetic_suite(
            suite, parameters, floor + extra, MXFP4_GROUP32_QUANTIZATION
        )
        _, identity = load_extraction_suite(canonical_bytes(candidate))
        assert identity.weight_bytes == floor + extra
        assert identity.quantization == MXFP4_GROUP32_QUANTIZATION
    short = study.synthetic_suite(suite, parameters, floor - 1, MXFP4_GROUP32_QUANTIZATION)
    with pytest.raises(ModelExtractionError, match="parameter and scale payload"):
        load_extraction_suite(canonical_bytes(short))


def test_packed_metadata_does_not_enable_a_homogeneous_compute_model(suite):
    _, identity = load_extraction_suite(canonical_bytes(suite))
    with pytest.raises(ModelExtractionError, match="quantization does not match"):
        _expected_dims(identity)


@pytest.mark.parametrize(
    "quantization,floor",
    [("none", 1024), ("fp8-e4m3-block-128x128", 512), ("unrecognized-four-bit-label", 512)],
)
def test_legacy_formats_keep_their_existing_admission_bound(study, suite, quantization, floor):
    candidate = study.synthetic_suite(suite, 512, floor, quantization)
    _, identity = load_extraction_suite(canonical_bytes(candidate))
    assert identity.weight_bytes == floor
    short = study.synthetic_suite(suite, 512, floor - 1, quantization)
    with pytest.raises(ModelExtractionError):
        load_extraction_suite(canonical_bytes(short))


@pytest.mark.parametrize(
    "mutation", ["digest", "total", "boolean-count", "local-policy", "shard-name", "unsorted"]
)
def test_packed_admission_keeps_strict_manifest_guards(suite, mutation):
    model = suite["reference_model"]
    if mutation == "digest":
        model["weight_sha256"] = "0" * 64
    elif mutation == "total":
        model["weight_bytes"] += 1
    elif mutation == "boolean-count":
        model["parameter_count"] = True
    elif mutation == "local-policy":
        model["local_weight_byte_verification"] = True
    elif mutation == "shard-name":
        model["weight_shards"][0]["name"] = "weights.bin"
    else:
        model["weight_shards"] = [
            {"name": "model-00002-of-00002.safetensors", "sha256": "1" * 64, "bytes": 160},
            {"name": "model-00001-of-00002.safetensors", "sha256": "2" * 64, "bytes": 160},
        ]
        model["weight_sha256"] = canonical_sha256(model["weight_shards"])
    with pytest.raises(ModelExtractionError):
        load_extraction_suite(canonical_bytes(suite))
