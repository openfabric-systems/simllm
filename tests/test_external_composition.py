from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError, asdict, replace
from fractions import Fraction
from pathlib import Path

import pytest

from examples.matched_seam_frontier_v1 import run_study as legacy
from simllm.calibration.canonical import canonical_loads
from simllm.calibration.external_composition import (
    EXTERNAL_ADJUSTMENT_IDS,
    ExternalServingComposition,
)
from simllm.calibration.external_db import ExternalOperationDatabase, default_artifact_dir
from simllm.calibration.record_types import RecordObject
from simllm.calibration.store import ObjectStore
from simllm.deploy import (
    EstimatorInputs,
    EvidenceClass,
    ExternalQwen32BDeploymentBinding,
    estimate_decode_step,
    estimate_prefill_request,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "examples/matched_seam_frontier_v1/study_config.json").read_text())
TABLE = json.loads(
    (ROOT / "examples/matched_seam_frontier_v1/external_adjustments.json").read_text()
)


@pytest.fixture(scope="module")
def database():
    return ExternalOperationDatabase.load(default_artifact_dir())


@pytest.fixture(scope="module")
def composition(database):
    return ExternalServingComposition.promote(database, TABLE)


@pytest.fixture
def binding(database, composition):
    return ExternalQwen32BDeploymentBinding(database, composition_record=composition)


def service(binding, oracle, phase):
    arguments = {name: oracle[name] for name in ("tensor_parallel", "batch_size", "isl", "prefix")}
    if binding.composition_record is None:
        arguments["latency_correction_scale"] = float.fromhex(
            CONFIG["composition"][f"{phase}_latency_correction_hex"]
        )
    if phase == "decode":
        arguments.update(osl=oracle["osl"], stride=oracle["stride"])
        return binding.decode_service(**arguments)
    return binding.prefill_service(**arguments)


def estimates(binding, *, batch=26, handoff_ps=0, prefix=500, stride=32):
    raw = next(
        row
        for row in legacy._csv_rows(legacy.DISAGG_PATH)
        if row["(d)tp"] == "4" and row["(d)bs"] == "26"
    )
    inventory = binding.database.manifest["source"]["model_config_sha256"]
    candidate = legacy._candidate(raw, row_number=6, disaggregated=True, inventory_sha256=inventory)
    work = legacy._model_work(inventory)
    envelope = legacy._envelope()
    prefill_value = binding.prefill_service(
        tensor_parallel=4, batch_size=1, isl=4000, prefix=prefix
    )
    decode_values = binding.decode_surface(
        tensor_parallel=4,
        batch_sizes=(16, 26, 56, 64),
        isl=4000,
        osl=500,
        prefix=prefix,
        stride=stride,
    )
    prefill = estimate_prefill_request(
        candidate,
        EstimatorInputs(
            model_work=work,
            envelopes={"h200": envelope},
            prefill_service=prefill_value.as_term(),
            handoff_ps=handoff_ps,
            handoff_source="declared test handoff",
        ),
    )
    decode = estimate_decode_step(
        candidate,
        batch,
        EstimatorInputs(
            model_work=work,
            envelopes={"h200": envelope},
            surfaces=tuple(value.as_batch_service_point() for value in decode_values),
            surface_evidence=EvidenceClass.MEASURED_EXTERNAL,
        ),
    )
    return candidate, prefill, decode, prefill_value


def test_record_is_deeply_immutable_and_content_addressed(composition, tmp_path):
    with pytest.raises(FrozenInstanceError):
        composition.record = None
    with pytest.raises(TypeError):
        composition.record.value["source_values_hex"]["prefill_latency_correction"] = "x"
    store = ObjectStore(tmp_path)
    store.write(composition.record)
    loaded = ExternalServingComposition.load(tmp_path, composition.record_sha256)
    assert loaded.record.canonical == composition.record.canonical
    assert loaded.record_sha256 == composition.record_sha256
    with pytest.raises(ValueError):
        ExternalServingComposition.load(tmp_path, "a" * 64)


@pytest.mark.parametrize("field", ["record_id", "schema", "canonical", "value"])
def test_forged_envelopes_reject(composition, field):
    replacements = {
        "record_id": "a" * 64,
        "schema": "wrong",
        "canonical": b"{}",
        "value": {"schema": "wrong"},
    }
    with pytest.raises(ValueError):
        ExternalServingComposition(replace(composition.record, **{field: replacements[field]}))


@pytest.mark.parametrize(
    "field",
    [
        "scope",
        "source_table_sha256",
        "operation_manifest_sha256",
        "source_values_hex",
        "runtime_objects_sha256",
        "overrides_hex",
        "parent_record_sha256",
        "intervention_reason",
    ],
)
def test_incomplete_or_wrong_record_fields_reject(composition, field):
    value = canonical_loads(composition.record.canonical)
    del value[field]
    with pytest.raises(ValueError):
        ExternalServingComposition(RecordObject.from_value(value))


@pytest.mark.parametrize("number", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_bandwidth_interventions_reject(composition, number):
    with pytest.raises(ValueError):
        composition.derive({"memory_bandwidth_empirical_scale": number}, reason="invalid control")


@pytest.mark.parametrize("name", EXTERNAL_ADJUSTMENT_IDS)
def test_derived_record_retains_source_and_parent(composition, name):
    row = next(row for row in TABLE["adjustments"] if row["id"] == name)
    before = composition.record.canonical
    removed = composition.derive({name: float(row["removal_value"])}, reason=f"remove {name}")
    assert removed.record_sha256 != composition.record_sha256
    assert removed.record.value["parent_record_sha256"] == composition.record_sha256
    assert (
        removed.record.value["source_values_hex"] == composition.record.value["source_values_hex"]
    )
    assert removed.number(name).hex() == float(row["removal_value"]).hex()
    assert composition.record.canonical == before
    restored = removed.derive({name: composition.number(name)}, reason="restore baseline")
    assert restored.record.canonical == before


def test_derived_record_cannot_claim_another_parent(composition):
    changed = composition.derive({"memory_empirical_constant_latency": 0.0}, reason="control")
    value = canonical_loads(changed.record.canonical)
    value["parent_record_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="parent"):
        ExternalServingComposition(RecordObject.from_value(value))


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_explicit_values_must_match_record(binding, phase):
    oracle = CONFIG["oracles"][phase][0]
    arguments = {name: oracle[name] for name in ("tensor_parallel", "batch_size", "isl", "prefix")}
    if phase == "decode":
        arguments.update(osl=oracle["osl"], stride=oracle["stride"])
    method = getattr(binding, f"{phase}_service")
    expected = method(**arguments)
    matching = binding.composition_record.number(f"{phase}_latency_correction")
    assert method(**arguments, latency_correction_scale=matching) == expected
    with pytest.raises(ValueError, match="conflicts"):
        method(**arguments, latency_correction_scale=1.0)
    with pytest.raises(ValueError, match="conflicts"):
        method(**arguments, latency_correction_scale=None)


@pytest.mark.parametrize("oracle", CONFIG["oracles"]["decode"])
def test_record_decode_reproduces_frozen_service_bits(database, binding, oracle):
    value = service(binding, oracle, "decode")
    old = service(ExternalQwen32BDeploymentBinding(database), oracle, "decode")
    assert value.service_ms_hex == old.service_ms_hex == oracle["expected_step_ms_hex"]
    assert value.total_ms_hex == old.total_ms_hex == oracle["expected_total_ms_hex"]
    assert value.entry_key_sha256 != old.entry_key_sha256
    assert binding.composition_record.record_sha256 in value.source
    assert "composition" not in old.source


@pytest.mark.parametrize("oracle", CONFIG["oracles"]["prefill"])
def test_record_prefill_reproduces_frozen_service_bits(database, binding, oracle):
    value = service(binding, oracle, "prefill")
    old = service(ExternalQwen32BDeploymentBinding(database), oracle, "prefill")
    assert value.service_ms_hex == old.service_ms_hex == oracle["expected_service_ms_hex"]
    assert value.total_ms_hex == old.total_ms_hex
    assert value.entry_key_sha256 != old.entry_key_sha256


@pytest.mark.parametrize("name", ["system_spec", "model_config", "manifest", "family_mapping"])
def test_mutated_database_rejects_before_cached_price(database, binding, monkeypatch, name):
    service(binding, CONFIG["oracles"]["prefill"][0], "prefill")
    mutated = copy.deepcopy(getattr(database, name))
    mutated["unexpected"] = "changed after caching"
    monkeypatch.setattr(database, name, mutated)
    with pytest.raises(ValueError, match="source or runtime state"):
        service(binding, CONFIG["oracles"]["prefill"][0], "prefill")


def test_mutated_cached_model_parameter_rejects(binding):
    service(binding, CONFIG["oracles"]["prefill"][0], "prefill")
    model = next(iter(binding._models.values()))
    model.memory_bandwidth_empirical_scale = 0.4
    with pytest.raises(ValueError, match="conflicts"):
        service(binding, CONFIG["oracles"]["prefill"][0], "prefill")


def test_source_swapped_before_promotion_rejects(database, monkeypatch):
    monkeypatch.setattr(database, "source", replace(database.source, system="gh200"))
    with pytest.raises(ValueError, match="imported manifest"):
        ExternalServingComposition.promote(database, TABLE)


@pytest.mark.parametrize(
    "name,value",
    [
        ("tensor_parallel", 4),
        ("kv_cache_quant_mode", "bfloat16"),
        ("fmha_quant_mode", "bfloat16"),
        ("communication_quant_mode", "bfloat16"),
    ],
)
def test_mutated_cached_model_shape_rejects(binding, name, value):
    service(binding, CONFIG["oracles"]["prefill"][0], "prefill")
    model = next(iter(binding._models.values()))
    assert getattr(model, name) != value
    setattr(model, name, value)
    with pytest.raises(ValueError, match="shape or quantization changed"):
        service(binding, CONFIG["oracles"]["prefill"][0], "prefill")


@pytest.mark.parametrize("phase,expected_delta", [("decode", 0.62532), ("prefill", 3.42474)])
def test_memory_constant_reaches_every_declared_visit(database, composition, phase, expected_delta):
    changed = composition.derive(
        {"memory_empirical_constant_latency": 0.0}, reason="constant control"
    )
    baseline = ExternalQwen32BDeploymentBinding(database, composition_record=composition)
    removed = ExternalQwen32BDeploymentBinding(database, composition_record=changed)
    for oracle in CONFIG["oracles"][phase]:
        delta = (
            service(baseline, oracle, phase).service_ms - service(removed, oracle, phase).service_ms
        )
        assert abs(delta - expected_delta) <= 1e-9


def test_aggregate_uses_only_the_declared_three_factors(database, composition):
    args = {
        "tensor_parallel": 4,
        "batch_size": 20,
        "isl": 4000,
        "osl": 500,
        "prefix": 500,
        "context_tokens": 4000,
    }
    old = ExternalQwen32BDeploymentBinding(database).aggregate_point(**args)
    new = ExternalQwen32BDeploymentBinding(
        database, composition_record=composition
    ).aggregate_point(**args)
    assert {
        key: value
        for key, value in new.as_dict().items()
        if key not in {"source", "entry_key_sha256"}
    } == {
        key: value
        for key, value in old.as_dict().items()
        if key not in {"source", "entry_key_sha256"}
    }
    changed = composition.derive(
        {
            "prefill_latency_correction": 1.0,
            "decode_latency_correction": 1.0,
            "autoscale_ttft_correction": 1.0,
            "prefill_rate_matching_degradation": 1.0,
            "decode_rate_matching_degradation": 1.0,
        },
        reason="inactive aggregate factors",
    )
    inactive = ExternalQwen32BDeploymentBinding(
        database, composition_record=changed
    ).aggregate_point(**args)
    assert inactive.ttft_ms.hex() == new.ttft_ms.hex()
    assert inactive.tpot_ms.hex() == new.tpot_ms.hex()
    assert (
        "composition-applied:memory_bandwidth_empirical_scale,memory_empirical_constant_latency,context_attention_extra_latency_correction"
        in new.source
    )
    with pytest.raises(ValueError, match="conflicts"):
        ExternalQwen32BDeploymentBinding(database, composition_record=composition).aggregate_point(
            **args, memory_empirical_constant_latency_s=0.0
        )


def test_record_absent_requires_original_explicit_service_value(database):
    old = ExternalQwen32BDeploymentBinding(database)
    with pytest.raises(TypeError, match="explicit value required"):
        old.prefill_service(tensor_parallel=4, batch_size=1, isl=4000, prefix=500)
    with pytest.raises(ValueError, match="explicitly selected"):
        old.autoscaled_ttft(None)


def test_capacity_preserves_exact_rates_and_stamps(binding):
    candidate, prefill, decode, _ = estimates(binding)
    result = binding.disaggregated_capacity(candidate, prefill, decode, 26, prefix=500, stride=32)
    pools = {pool.role: pool for pool in candidate.pools}
    p = Fraction(pools["prefill"].engines * 10**12, prefill.request_ps) * Fraction.from_float(0.9)
    d = Fraction(pools["decode"].engines * 26 * 10**12, 500 * decode.step_ps) * Fraction.from_float(
        0.92
    )
    assert result.prefill_requests_per_second == p
    assert result.decode_requests_per_second == d
    assert result.request_capacity == min(p, d)
    assert result.prefill.stamp is prefill.stamp
    assert result.decode.stamp is decode.stamp
    assert result.composition_sha256 == binding.composition_record.record_sha256


def test_handoff_changes_prefill_capacity_and_stays_outside_autoscale(binding):
    candidate, prefill, decode, value = estimates(binding)
    first = binding.disaggregated_capacity(candidate, prefill, decode, 26, prefix=500, stride=32)
    candidate, delayed, same_decode, same_value = estimates(binding, handoff_ps=100_000_000)
    second = binding.disaggregated_capacity(
        candidate, delayed, same_decode, 26, prefix=500, stride=32
    )
    assert second.prefill_requests_per_second < first.prefill_requests_per_second
    assert second.decode_requests_per_second == first.decode_requests_per_second
    assert binding.autoscaled_ttft(value).as_dict() == binding.autoscaled_ttft(same_value).as_dict()
    assert binding.autoscaled_ttft(value).latency_ms.hex() == (value.service_ms * 1.8).hex()


@pytest.mark.parametrize("batch", [16, 26, 32, 56, 64])
def test_capacity_joins_exact_and_interpolated_surface_keys(binding, batch):
    candidate, prefill, decode, _ = estimates(binding, batch=batch)
    result = binding.disaggregated_capacity(
        candidate, prefill, decode, batch, prefix=500, stride=32
    )
    assert result.decode.step_ps == decode.step_ps
    assert len(result.service_keys) == (3 if batch == 32 else 2)


@pytest.mark.parametrize(
    "alteration", ["duration", "stamp", "candidate", "prefix", "stride", "batch"]
)
def test_projection_rejects_inconsistent_source_or_shape(binding, alteration):
    candidate, prefill, decode, _ = estimates(binding)
    prefix, stride, batch = 500, 32, 26
    if alteration == "duration":
        kernel = replace(decode.kernel_floor, duration_ps=decode.step_ps + 1)
        stamp = replace(
            decode.stamp,
            terms=tuple(
                replace(term, estimate=kernel) if term.name == "kernel_floor" else term
                for term in decode.stamp.terms
            ),
        )
        decode = replace(
            decode,
            kernel_floor=kernel,
            batch_service=kernel,
            step_ps=kernel.duration_ps,
            analytical_step_ps=kernel.duration_ps,
            stamp=stamp,
        )
    elif alteration == "stamp":
        decode = replace(
            decode,
            stamp=replace(
                decode.stamp,
                terms=tuple(
                    replace(
                        term,
                        estimate=replace(term.estimate, duration_ps=term.estimate.duration_ps + 1),
                    )
                    if term.name == "kernel_floor"
                    else term
                    for term in decode.stamp.terms
                ),
            ),
        )
    elif alteration == "candidate":
        candidate = replace(candidate, candidate_id="other shape identity")
    elif alteration == "prefix":
        prefix = 499
    elif alteration == "stride":
        stride = 16
    else:
        batch = 27
    with pytest.raises(ValueError):
        binding.disaggregated_capacity(
            candidate, prefill, decode, batch, prefix=prefix, stride=stride
        )


def test_equal_duration_from_another_record_rejects(database, composition, binding):
    candidate, prefill, decode, value = estimates(binding)
    changed = composition.derive(
        {"prefill_rate_matching_degradation": 0.45},
        reason="different record, equal operation service",
    )
    other = ExternalQwen32BDeploymentBinding(database, composition_record=changed)
    _, _, _, equal_value = estimates(other)
    assert equal_value.service_ms.hex() == value.service_ms.hex()
    assert equal_value.entry_key_sha256 != value.entry_key_sha256
    with pytest.raises(ValueError):
        other.autoscaled_ttft(value)
    with pytest.raises(ValueError):
        other.disaggregated_capacity(candidate, prefill, decode, 26, prefix=500, stride=32)


def test_same_key_changed_autoscale_input_rejects(binding):
    _, _, _, value = estimates(binding)
    altered = replace(value, service_ms=value.service_ms + 0.001)
    with pytest.raises(ValueError, match="retained source"):
        binding.autoscaled_ttft(altered)


def test_all_historical_artifact_bytes_remain_unchanged():
    import hashlib

    freeze = json.loads((ROOT / "examples/external_composition_v1/expectations.json").read_text())
    for entry in freeze["preservation"]:
        assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]


def test_legacy_service_serialization_has_no_new_fields(database):
    value = service(
        ExternalQwen32BDeploymentBinding(database), CONFIG["oracles"]["decode"][0], "decode"
    )
    assert set(asdict(value)) == {
        "configuration_id",
        "phase",
        "tensor_parallel",
        "batch_size",
        "service_ms",
        "total_ms",
        "source",
        "entry_key_sha256",
        "evidence_class",
    }


def test_projection_cannot_replace_derived_factors_or_accounting(binding):
    candidate, prefill, decode, value = estimates(binding)
    autoscale = binding.autoscaled_ttft(value)
    capacity = binding.disaggregated_capacity(candidate, prefill, decode, 26, prefix=500, stride=32)
    for name, changed in (
        ("factor_hex", (2.0).hex()),
        ("composition_sha256", "a" * 64),
    ):
        with pytest.raises(TypeError):
            replace(autoscale, **{name: changed})
    for name, changed in (
        ("prefill_rate_hex", (1.0).hex()),
        ("decode_rate_hex", (1.0).hex()),
        ("prefill_engines", 999),
        ("decode_engines", 999),
        ("used_gpus", 1),
        ("output_tokens", 1),
        ("composition_sha256", "a" * 64),
    ):
        with pytest.raises(TypeError):
            replace(capacity, **{name: changed})


def test_projection_revalidates_replaced_record_or_selected_price(binding, composition):
    candidate, prefill, decode, value = estimates(binding)
    autoscale = binding.autoscaled_ttft(value)
    capacity = binding.disaggregated_capacity(candidate, prefill, decode, 26, prefix=500, stride=32)
    changed_record = composition.derive(
        {"autoscale_ttft_correction": 2.0}, reason="replacement identity control"
    )
    for result in (autoscale, capacity):
        with pytest.raises(ValueError, match="key disagrees"):
            replace(result, composition=changed_record)
    # Updating all redundant copies still cannot create an issued receipt.
    changed_ms = value.service_ms + 0.001
    altered = replace(
        value,
        service_ms=changed_ms,
        total_ms=changed_ms,
        source=value.source.replace(value.service_ms_hex, changed_ms.hex()),
    )
    with pytest.raises(TypeError, match="issued only"):
        replace(autoscale.selection, value=altered)
    with pytest.raises(TypeError, match="issued only"):
        type(autoscale.selection)(
            value=altered, configuration_items=autoscale.selection.configuration_items
        )
    with pytest.raises(FrozenInstanceError):
        autoscale.selection.value = altered


def test_capacity_revalidates_replaced_inputs_and_extra_selections(binding):
    candidate, prefill, decode, _ = estimates(binding)
    capacity = binding.disaggregated_capacity(candidate, prefill, decode, 26, prefix=500, stride=32)
    for values in (
        {"candidate": replace(candidate, candidate_id="replacement")},
        {"prefix": 499},
        {"stride": 16},
        {"batch_size": 27},
        {"selections": (*capacity.selections, capacity.selections[-1])},
        {"decode": replace(decode, step_ps=decode.step_ps + 1)},
    ):
        with pytest.raises(ValueError):
            replace(capacity, **values)


@pytest.mark.parametrize("failed_class", ["fatal", "exact", "behavioral"])
def test_study_failure_retains_evidence_and_cannot_close(failed_class):
    from examples.external_composition_v1.run_study import Evaluation

    evaluation = Evaluation()
    evaluation.guard("source", failed_class != "fatal", "synthetic reporting control")
    evaluation.exact(
        "known-reference", "hex", "0x1.0p0", "wrong" if failed_class == "exact" else "0x1.0p0"
    )
    evaluation.relation("scaling", "cell", 3.0 if failed_class == "behavioral" else 2.0, 2.0)
    result = evaluation.finish()
    assert result["state"] == ("VOID" if failed_class == "fatal" else "FAIL")
    assert (
        result["behavioral_score"] is None
        if failed_class == "fatal"
        else result["behavioral_score"] is not None
    )
    assert len(result["exact_oracles"]) == 1
    assert len(result["behavioral_relations"]) == 1
    assert len(result["fatal_guards"]) == 1


def test_study_capacity_setup_supplies_a_valid_exact_selection(binding):
    from examples.external_composition_v1.run_study import capacity_inputs

    p = service(binding, CONFIG["oracles"]["prefill"][1], "prefill")
    d = [
        service(binding, oracle, "decode")
        for oracle in sorted(CONFIG["oracles"]["decode"], key=lambda row: row["batch_size"])
        if oracle["id"] in {"decode-tp4-b16", "decode-tp4-b26"}
    ]
    candidate, prefill, decode = capacity_inputs(binding, p, d)
    result = binding.disaggregated_capacity(candidate, prefill, decode, 26, prefix=500, stride=32)
    assert result.decode.step_ps == d[1].service_ps
    assert result.service_keys == (p.entry_key_sha256, d[1].entry_key_sha256)


@pytest.mark.parametrize(
    "alteration",
    ["list-pid", "dict-pid", "missing-score", "false-exit", "missing-worker", "changed-source"],
)
def test_coordinator_retains_void_for_malformed_or_inconsistent_workers(alteration):
    from examples.external_composition_v1.run_study import Evaluation, aggregate_workers

    evaluation = Evaluation()
    evaluation.guard("synthetic-source", True)
    evaluation.exact("reference", "one", 1, 1)
    evaluation.relation("scaling", "one", 2.0, 2.0)
    first = evaluation.finish()
    values = [copy.deepcopy(first), copy.deepcopy(first)]
    metadata = [{"process_identity": 101}, {"process_identity": 102}]
    codes = [0, 0]
    assert (
        aggregate_workers(values, metadata, codes, equal=True, source_unchanged=True)["state"]
        == "PASS"
    )
    if alteration == "list-pid":
        metadata[0]["process_identity"] = []
    elif alteration == "dict-pid":
        metadata[0]["process_identity"] = {}
    elif alteration == "missing-score":
        del values[0]["behavioral_score"]
    elif alteration == "false-exit":
        codes[0] = 1
    elif alteration == "missing-worker":
        values[0] = None
    result = aggregate_workers(
        values, metadata, codes, equal=True, source_unchanged=alteration != "changed-source"
    )
    assert result["state"] == "VOID"
    assert result["behavioral_score"] is None
    assert result["fatal_findings"]


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("mode", ["omitted", "matching", "conflicting", "derived"])
def test_public_pass_phase_factor_belongs_to_the_record(database, composition, phase, mode):
    from simllm.calibration.external_db import ExternalQwen32BPassModel

    factor_name = f"{phase}_latency_correction"
    selected = (
        composition.derive({factor_name: 2.0}, reason="direct pass factor control")
        if mode == "derived"
        else composition
    )
    common = {
        "tensor_parallel": 4,
        "kv_cache_quant_mode": "fp8",
        "fmha_quant_mode": "fp8",
        "communication_quant_mode": "half",
    }
    model = ExternalQwen32BPassModel(database, **common, composition_record=selected)
    legacy_model = ExternalQwen32BPassModel(database, **common)
    kwargs = {"batch_size": 1, "isl": 4000}
    if phase == "prefill":
        kwargs["prefix"] = 500
        run, legacy_run = model.run_context, legacy_model.run_context
    else:
        kwargs.update(batch_size=26, osl=500, stride=32)
        run, legacy_run = model.run_generation, legacy_model.run_generation
    if mode == "matching":
        kwargs["latency_correction_scale"] = selected.number(factor_name)
    elif mode == "conflicting":
        kwargs["latency_correction_scale"] = 2.0
        with pytest.raises(ValueError, match="conflicts"):
            run(**kwargs)
        return
    actual = run(**kwargs)
    expected = legacy_run(**{**kwargs, "latency_correction_scale": selected.number(factor_name)})
    assert actual.total.latency_ms.hex() == expected.total.latency_ms.hex()
    assert selected.record_sha256 in actual.total.rule
    assert [entry.latency_ms.hex() for entry in actual.operations] == [
        entry.latency_ms.hex() for entry in expected.operations
    ]
    assert (
        legacy_run(**{**kwargs, "latency_correction_scale": 1.0}).total.latency_ms.hex()
        == legacy_run(
            **{key: value for key, value in kwargs.items() if key != "latency_correction_scale"}
        ).total.latency_ms.hex()
    )
