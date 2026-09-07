"""The v2 pair shape is exact across phases, grouping and extraction versions."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from simllm.calibration import extraction
from simllm.calibration.model_inventory import (
    ATTENTION_PAIR_SHAPE_SCHEMA,
    ModelKernelInventory,
)
from simllm.compute import step_kernel, step_kernels, step_shape
from simllm.core import RequestPhase, ScheduledRequest, StepRecord


def _load_study(name):
    path = Path(__file__).resolve().parents[1] / "examples/frontier_comparison_v2" / f"{name}.py"
    module_name = f"pair_study_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


study = _load_study("run_study")
extraction_study = _load_study("extract_inventories")


def test_all_old_and_new_inventory_files_remain_canonical() -> None:
    for path in study.REGISTRY.glob("*.json"):
        inventory = ModelKernelInventory.from_obj(json.loads(study.lf_bytes(path)))
        assert study.digest_matches(path, path.stem)
        assert inventory.record.canonical == study.lf_bytes(path)


def test_successors_only_add_the_pair_shape_and_agree_for_every_case() -> None:
    new = {k: study.load_inventory(v) for k, v in study.INVENTORIES.items()}
    assert study.neutral(new["vllm"]) == study.neutral(new["sglang"])
    for framework, inventory in new.items():
        old = study.load_inventory(extraction_study.OLD_IDS[framework])
        value = inventory.to_obj()
        value["shape_schemas"] = old.to_obj()["shape_schemas"]
        value["kernel_families"] = old.to_obj()["kernel_families"]
        for current, previous in zip(value["cases"], old.to_obj()["cases"], strict=True):
            for p, q in zip(
                current["kernel_projections"], previous["kernel_projections"], strict=True
            ):
                if p["family_id"] == "attn_score":
                    assert p["shape_vector"]["values"][:2] == q["shape_vector"]["values"]
                    p["shape_vector"] = q["shape_vector"]
        assert value == old.to_obj()
        assert len(inventory.cases) == 15
        audit = study.audit_inventory(inventory, json.loads(study.lf_bytes(study.SUITE)))
        assert all(
            all(
                row[k]
                for k in (
                    "pair_axis_exact",
                    "coefficient_exact",
                    "family_flops_exact",
                    "family_bytes_exact",
                )
            )
            for row in audit
        )
        anchors = {r["case_id"]: r["attention_pairs"] for r in audit}
        assert anchors["cp-train-r4-t128"] == 2048
        assert anchors["db-train-b1-c2048"] == 2047
        assert anchors["md-train-b4-c128"] == 508


@pytest.mark.parametrize(
    ("shapes", "expected"),
    [
        ([(1, 1)], 0),
        ([(1, 2048)], 2047),
        ([(3, 3)], 4),
        ([(3, 7), (5, 12), (1, 128)], 190),
        ([(32, 32)] * 4, 2048),
        ([(128, 128)], 8192),
    ],
)
def test_pair_authority_rounds_each_sequence_and_conserves(shapes, expected) -> None:
    dims = study.dims_for(study.load_inventory(study.INVENTORIES["vllm"]))
    record = StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                str(i),
                RequestPhase.DECODE if n == 1 else RequestPhase.PREFILL,
                num_new_tokens=n,
                context_length=c,
            )
            for i, (n, c) in enumerate(shapes)
        ],
    )
    assert step_shape(record)[2] == expected
    families = step_kernels(dims, record, len(shapes))
    score = next(f for f in families if f.name == "attn_score")
    assert score.flops == expected * 2_097_152
    # KV sharing affects storage but cannot remove a query head's QK/PV products.
    mha = next(
        f
        for f in step_kernels(replace(dims, num_kv_heads=64), record, len(shapes))
        if f.name == "attn_score"
    )
    assert mha.flops == score.flops
    fused = step_kernel(dims, record, len(shapes))
    assert sum(f.flops for f in families) == fused.flops
    assert sum(f.bytes_moved for f in families) == fused.bytes_moved


@pytest.mark.parametrize("mutation", ["pairs", "flops", "unit", "axis-order"])
def test_v2_validation_rejects_contract_mutations(mutation) -> None:
    value = study.load_inventory(study.INVENTORIES["vllm"]).to_obj()
    score = next(
        p for p in value["cases"][0]["kernel_projections"] if p["family_id"] == "attn_score"
    )
    schema = next(
        s for s in value["shape_schemas"] if s["shape_schema_id"] == ATTENTION_PAIR_SHAPE_SCHEMA
    )
    if mutation == "pairs":
        score["shape_vector"]["values"][2] += 1
    elif mutation == "flops":
        score["aggregate_flops"] += 1
    elif mutation == "unit":
        schema["axes"][2]["unit"] = "tokens"
    else:
        schema["axes"][0], schema["axes"][1] = schema["axes"][1], schema["axes"][0]
    with pytest.raises(ValueError, match="attention"):
        ModelKernelInventory.from_obj(value)


@pytest.fixture
def inputs(tmp_path: Path):
    suite = json.loads(study.lf_bytes(study.SUITE))
    g = suite["reference_model"]["geometry"]
    config = {
        "num_hidden_layers": g["layers"],
        "hidden_size": g["hidden_size"],
        "intermediate_size": g["intermediate_size"],
        "num_attention_heads": g["num_heads"],
        "num_key_value_heads": g["num_kv_heads"],
        "head_dim": g["head_size"],
        "vocab_size": g["vocab_size"],
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
    }
    raw = study.json_bytes(config)
    suite["reference_model"]["config_sha256"] = hashlib.sha256(raw).hexdigest()
    checkpoint = tmp_path / suite["reference_model"]["revision"]
    checkpoint.mkdir()
    (checkpoint / "config.json").write_bytes(raw)
    return suite, config, checkpoint


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_existing_config_extractor_and_native_adapter_expose_both_versions(
    inputs, tmp_path, monkeypatch, framework
):
    suite, config, checkpoint = inputs
    old = study.load_inventory(extraction_study.OLD_IDS[framework])
    decl = next(d for d in suite["frameworks"] if d["id"] == framework)
    dims, projection = extraction_study.config_inputs(config, decl, old.framework)
    adapter = importlib.import_module(f"simllm.adapters.{framework}.extraction")
    monkeypatch.setattr(adapter, "_configuration", lambda _: (dims, old.framework, projection))
    generated = []
    for version in (1, 2, 2):
        inventory = adapter.extract(
            suite_raw=study.json_bytes(suite),
            checkpoint_root=checkpoint,
            step_records_path=tmp_path / "steps.jsonl",
            attention_shape_version=version,
        )
        generated.append(inventory)
    assert generated[0].cases == old.cases
    assert generated[1].cases == study.load_inventory(study.INVENTORIES[framework]).cases
    assert generated[1].record.canonical == generated[2].record.canonical


def test_extraction_rejects_nonconservation(inputs, tmp_path, monkeypatch):
    suite, config, checkpoint = inputs
    old = study.load_inventory(extraction_study.OLD_IDS["vllm"])
    dims, projection = extraction_study.config_inputs(config, suite["frameworks"][0], old.framework)
    original = extraction.step_kernels

    def broken(*args):
        families = original(*args)
        families[0] = replace(families[0], flops=families[0].flops + 1)
        return families

    monkeypatch.setattr(extraction, "step_kernels", broken)
    with pytest.raises(extraction.ModelExtractionError, match="family FLOPs are not exact"):
        extraction.extract_model_inventory(
            suite_raw=study.json_bytes(suite),
            framework=old.framework,
            checkpoint_root=checkpoint,
            framework_dims=dims,
            framework_projection=projection,
            step_records_path=tmp_path / "steps.jsonl",
            attention_shape_version=2,
        )


def test_zero_pair_coefficient_is_unidentifiable_and_v1_is_explicit():
    inventory = study.load_inventory(study.INVENTORIES["vllm"])
    score = next(p for p in inventory.cases[0].kernel_projections if p.family_id == "attn_score")
    zero = replace(
        score, aggregate_flops=0, shape_vector=replace(score.shape_vector, values=(1, 1, 0))
    )
    assert zero.attention_pairs == 0
    assert zero.attention_flops_per_pair is None
    with pytest.raises(ValueError, match="zero attention"):
        _ = replace(zero, aggregate_flops=1).attention_flops_per_pair
    old = study.load_inventory(extraction_study.OLD_IDS["vllm"])
    old_score = next(p for p in old.cases[0].kernel_projections if p.family_id == "attn_score")
    with pytest.raises(ValueError, match="v2"):
        _ = old_score.attention_pairs
    broken = copy.deepcopy(score.to_obj())
    broken["aggregate_flops"] += 1
    with pytest.raises(ValueError, match="exactly"):
        _ = type(score).from_obj(broken, "score").attention_flops_per_pair


def test_successor_work_and_x2_are_reachable_from_the_deployment_estimator():
    result = json.loads(study.lf_bytes(study.STUDY / "results.json"))
    inventory = study.load_inventory(study.INVENTORIES["vllm"])
    work = study.derive_work(inventory)
    assert work.decode_total_flops_per_batch_item == 72_877_867_008
    assert work.prefill_total_flops_per_request == 231_285_964_144_640
    assert study.derive_work(inventory, 2048).prefill_total_flops_per_request == 132_217_829_064_704
    pricer = study.reference_pricer()
    with pricer.ProcessGuard() as guard:
        x2 = pricer._x2(work)
        rows, behavioral = study.sweep(pricer, inventory)
    assert not guard.attempts
    assert x2 == result["x2"]
    assert x2["rows"][0]["predicted_ps"] == 5_379_515_733
    assert x2["rows"][1]["predicted_ps"] == 29_217_529_578
    assert [r["passed"] for r in x2["rows"]] == [True, True, True, False]
    assert rows == result["sweep"]
    assert len(rows) == 180 and all(behavioral.values())
    assert all(result["behavioral"].values())
    assert all(result["fatal_guards"].values())
    assert result["scores"] is not None
    assert result["inventory_sha256"] == study.INVENTORIES
    for p in (
        x2["matched_point"],
        *[p for arm in result["frontier"]["arms"].values() for p in arm["frontier"]],
    ):
        assert study.INVENTORIES["vllm"] in json.dumps(p["decode_stamp"])
        assert study.INVENTORIES["vllm"] in json.dumps(p["prefill_stamp"])
    old = json.loads(study.lf_bytes(study.ROOT / "examples/frontier_comparison_v1/results.json"))
    assert result["old_x2"] == old["families"]["X2"]


def test_fatal_input_failure_does_not_enter_pricing(monkeypatch):
    original = study.digest_matches
    monkeypatch.setattr(
        study,
        "digest_matches",
        lambda path, expected: (
            False if path.name == "expectations.md" else original(path, expected)
        ),
    )

    def forbidden():
        pytest.fail("fatal input guard must prevent pricing")

    monkeypatch.setattr(study, "reference_pricer", forbidden)
    result, _ = study.run_study()
    assert result["verdict"] == "VOID"
    assert result["scores"] is None
    assert "x2" not in result
    existing = json.loads(study.lf_bytes(study.STUDY / "results.json"))
    failed = study.finish({"no_pricing_subprocess": False}, existing)
    assert failed["scores"] is None and "x2" not in failed
    assert "x2" in failed["unscored_diagnostics"]


def test_digest_inputs_accept_windows_checkout_and_predecessors_remain_pinned(tmp_path):
    raw = b"first\nsecond\n"
    path = tmp_path / "input.md"
    expected = hashlib.sha256(raw).hexdigest()
    for content in (raw, raw.replace(b"\n", b"\r\n")):
        path.write_bytes(content)
        assert study.digest_matches(path, expected)
    manifest = json.loads(study.lf_bytes(study.STUDY / "predecessors.json"))
    assert study.digest_matches(study.STUDY / "predecessors.json", study.PREDECESSORS_SHA256)
    assert all(study.digest_matches(study.ROOT / name, digest) for name, digest in manifest.items())
    assert study.digest_matches(study.STUDY / "expectations.md", study.EXPECTATIONS_SHA256)
    encoded = study.json_bytes({"portable": "line\nend"})
    assert json.loads(encoded) == {"portable": "line\nend"}
    assert b"\r" not in encoded and encoded.endswith(b"\n")


def test_native_cli_exposes_explicit_successor_shape_option():
    from simllm.calibration.cli import build_parser

    args = build_parser().parse_args(
        [
            "extract",
            "--framework",
            "vllm",
            "--checkpoint-root",
            "checkpoint",
            "--step-records",
            "steps.jsonl",
            "--output-root",
            "objects",
            "--attention-shape-version",
            "2",
        ]
    )
    assert args.attention_shape_version == 2
