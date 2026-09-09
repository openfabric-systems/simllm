"""Retain prospective composition relations and unchanged historical oracles."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path

from examples.matched_seam_frontier_v1 import run_study as historical
from simllm.calibration.canonical import canonical_loads
from simllm.calibration.external_composition import ExternalServingComposition
from simllm.calibration.external_db import ExternalOperationDatabase, default_artifact_dir
from simllm.calibration.record_types import RecordObject
from simllm.deploy import (
    EstimatorInputs,
    EvidenceClass,
    ExternalQwen32BDeploymentBinding,
    candidate_to_json,
    estimate_decode_step,
    estimate_prefill_request,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FREEZE_COMMIT = "483b31a0d98ec1c4f5fd0668026055e84e363456"
BASELINE_RECORD = "6f1dccd1444fdd947b978551218b2a46b13842436222fe11df0b79f0aae438f3"
RECORD_ROOT = ROOT / "offline/calibration/external-compositions"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode()


def write_new(path, value):
    with path.open("xb") as stream:
        stream.write(encoded(value))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fraction_json(value):
    return {"numerator": value.numerator, "denominator": value.denominator, "decimal": float(value)}


def service_json(value):
    return {
        **asdict(value),
        "service_ms_hex": value.service_ms_hex,
        "total_ms_hex": value.total_ms_hex,
        "service_ps": value.service_ps,
    }


def legacy_service_reference(oracle, phase, external_source, correction_hex):
    """Project known oracles with the legacy recipe at authored_against 9335169."""

    config = {
        "id": oracle["id"],
        "phase": phase,
        **{name: oracle[name] for name in ("tensor_parallel", "batch_size", "isl", "prefix")},
        "latency_correction_scale_hex": correction_hex,
        "gemm_quant_mode": "fp8_block",
        "kv_cache_quant_mode": "fp8",
        "fmha_quant_mode": "fp8",
        "communication_quant_mode": "half",
    }
    if phase == "decode":
        config.update(osl=oracle["osl"], stride=oracle["stride"])
    payload = {"external_source": dict(external_source), "configuration": config}
    key = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
    ).hexdigest()
    service_hex = oracle.get("expected_step_ms_hex", oracle.get("expected_service_ms_hex"))
    total_hex = oracle.get("expected_total_ms_hex", service_hex)
    source = (
        f"external-operation-pass:slice-sha256:{external_source['data_slice_sha256']};"
        f"configuration:{oracle['id']};service-ms-hex:{service_hex};entry-key-sha256:{key}"
    )
    return {
        "configuration_id": oracle["id"],
        "phase": phase,
        "tensor_parallel": oracle["tensor_parallel"],
        "batch_size": oracle["batch_size"],
        "service_ms": float.fromhex(service_hex),
        "total_ms": float.fromhex(total_hex),
        "source": source,
        "entry_key_sha256": key,
        "evidence_class": "MEASURED-EXTERNAL",
        "service_ms_hex": service_hex,
        "total_ms_hex": total_hex,
        "service_ps": round(float.fromhex(service_hex) * 10**9),
    }


def price(binding, oracle, phase):
    arguments = {name: oracle[name] for name in ("tensor_parallel", "batch_size", "isl", "prefix")}
    if binding.composition_record is None:
        arguments["latency_correction_scale"] = 1.08 if phase == "decode" else 1.1
    if phase == "decode":
        arguments.update(osl=oracle["osl"], stride=oracle["stride"])
        return binding.decode_service(**arguments)
    return binding.prefill_service(**arguments)


def capacity_inputs(
    binding, prefill_value, decode_values, *, p_engines=1, d_engines=1, handoff_ps=0
):
    raw = next(
        row
        for row in historical._csv_rows(historical.DISAGG_PATH)
        if row["(d)tp"] == "4" and row["(d)bs"] == "26"
    )
    inventory = binding.database.manifest["source"]["model_config_sha256"]
    candidate = historical._candidate(
        raw, row_number=6, disaggregated=True, inventory_sha256=inventory
    )
    candidate = replace(
        candidate,
        pools=tuple(
            replace(pool, engines=p_engines if pool.role == "prefill" else d_engines)
            for pool in candidate.pools
        ),
    )
    common = {
        "model_work": historical._model_work(inventory),
        "envelopes": {"h200": historical._envelope()},
    }
    prefill = estimate_prefill_request(
        candidate,
        EstimatorInputs(
            **common,
            prefill_service=prefill_value.as_term(),
            handoff_ps=handoff_ps,
            handoff_source="declared composition-study handoff",
        ),
    )
    decode = estimate_decode_step(
        candidate,
        26,
        EstimatorInputs(
            **common,
            surfaces=tuple(value.as_batch_service_point() for value in decode_values),
            surface_evidence=EvidenceClass.MEASURED_EXTERNAL,
        ),
    )
    return candidate, prefill, decode


class Evaluation:
    """Separate fatal preconditions, exact references and prospective relations."""

    def __init__(self):
        self.freeze = read_json(HERE / "expectations.json")
        self.config = read_json(ROOT / self.freeze["service_config"]["path"])
        self.old = read_json(ROOT / self.freeze["comparison_source"]["path"])
        self.table = read_json(ROOT / self.freeze["adjustment_table"]["path"])
        self.data = {
            "schema": "simllm-external-composition-evaluation-v1",
            "task": "COMP-88",
            "records": {},
            "fatal_guards": [],
            "exact_oracles": [],
            "behavioral_relations": [],
            "baseline_services": [],
            "remove_one": [],
            "memory_cells": [],
            "capacity_cells": [],
            "autoscale_cells": [],
            "aggregate_cells": [],
        }

    def guard(self, identity, passed, detail=None):
        self.data["fatal_guards"].append({"id": identity, "passed": bool(passed), "detail": detail})

    def exact(self, family, identity, observed, expected):
        self.data["exact_oracles"].append(
            {
                "family": family,
                "id": identity,
                "observed": observed,
                "expected": expected,
                "passed": observed == expected,
            }
        )

    def relation(self, family, identity, observed, expected, *, tolerance=0.0, positive=False):
        residual = observed - expected
        self.data["behavioral_relations"].append(
            {
                "family": family,
                "id": identity,
                "observed": observed,
                "expected": expected,
                "absolute_tolerance": tolerance,
                "residual": residual,
                "requires_positive": positive,
                "passed": abs(residual) <= tolerance and (not positive or observed > 0),
            }
        )

    def reject(self, identity, action):
        try:
            action()
        except (ValueError, TypeError, FileNotFoundError) as error:
            self.guard(identity, True, type(error).__name__)
        else:
            self.guard(identity, False, "invalid input was accepted")

    def retain_record(self, record):
        self.data["records"][record.record_sha256] = canonical_loads(record.record.canonical)
        return record

    def derived(self, overrides, reason):
        return self.retain_record(self.baseline.derive(overrides, reason=reason))

    def binding(self, record):
        return ExternalQwen32BDeploymentBinding(self.database, composition_record=record)

    def preservation(self, stage):
        for row in self.freeze["preservation"]:
            self.guard(
                f"preserve-{stage}:{row['path']}", digest(ROOT / row["path"]) == row["sha256"]
            )

    def admission(self):
        self.database = ExternalOperationDatabase.load(default_artifact_dir())
        self.baseline = self.retain_record(
            ExternalServingComposition.load(RECORD_ROOT, BASELINE_RECORD)
        )
        promoted = ExternalServingComposition.promote(self.database, self.table)
        self.guard(
            "explicit-source-exact-object",
            promoted.record.canonical == self.baseline.record.canonical,
        )
        self.reject(
            "missing-object", lambda: ExternalServingComposition.load(RECORD_ROOT, "a" * 64)
        )
        for field, value in (
            ("record_id", "a" * 64),
            ("schema", "wrong"),
            ("canonical", b"{}"),
            ("value", {}),
        ):
            self.reject(
                f"forged-envelope:{field}",
                lambda field=field, value=value: ExternalServingComposition(
                    replace(self.baseline.record, **{field: value})
                ),
            )
        malformed = canonical_loads(self.baseline.record.canonical)
        del malformed["source_values_hex"]
        self.reject(
            "incomplete-record",
            lambda: ExternalServingComposition(RecordObject.from_value(malformed)),
        )
        self.reject(
            "illegal-physical-factor",
            lambda: self.baseline.derive(
                {"memory_bandwidth_empirical_scale": 0.0}, reason="invalid control"
            ),
        )
        source = self.database.source
        try:
            self.database.source = replace(source, backend="wrong")
            self.reject(
                "source-before-promotion",
                lambda: ExternalServingComposition.promote(self.database, self.table),
            )
        finally:
            self.database.source = source
        model = self.database.model_config
        try:
            self.database.model_config = {**model, "hidden_size": 1}
            self.reject("runtime-before-admission", lambda: self.binding(self.baseline))
        finally:
            self.database.model_config = model
        self.baseline_binding = self.binding(self.baseline)
        self.legacy_binding = ExternalQwen32BDeploymentBinding(self.database)
        oracle = self.config["oracles"]["prefill"][1]
        self.reject(
            "conflicting-explicit-phase",
            lambda: self.baseline_binding.prefill_service(
                tensor_parallel=4, batch_size=1, isl=4000, prefix=500, latency_correction_scale=1.0
            ),
        )
        self.reject("no-implicit-record", lambda: self.legacy_binding.autoscaled_ttft(None))
        cached = self.baseline_binding._model(4)
        for field, altered in (
            ("tensor_parallel", 2),
            ("kv_cache_quant_mode", "half"),
            ("fmha_quant_mode", "half"),
            ("communication_quant_mode", "fp8"),
            ("memory_bandwidth_empirical_scale", 0.4),
        ):
            original = getattr(cached, field)
            try:
                setattr(cached, field, altered)
                self.reject(
                    f"cached-identity:{field}",
                    lambda: price(self.baseline_binding, oracle, "prefill"),
                )
            finally:
                setattr(cached, field, original)
        original = self.database.system_spec["gpu"]["mem_bw_empirical_scaling_factor"]
        try:
            self.database.system_spec["gpu"]["mem_bw_empirical_scaling_factor"] = 0.4
            self.reject(
                "cached-source-state", lambda: price(self.baseline_binding, oracle, "prefill")
            )
        finally:
            self.database.system_spec["gpu"]["mem_bw_empirical_scaling_factor"] = original

    def direct_pass_controls(self):
        self.data["direct_pass_controls"] = []
        for phase in ("prefill", "decode"):
            method = "run_context" if phase == "prefill" else "run_generation"
            factor_name = f"{phase}_latency_correction"
            oracle = (
                self.config["oracles"]["prefill"][1]
                if phase == "prefill"
                else next(
                    row for row in self.config["oracles"]["decode"] if row["id"] == "decode-tp4-b26"
                )
            )
            arguments = {"batch_size": oracle["batch_size"], "isl": oracle["isl"]}
            if phase == "prefill":
                arguments["prefix"] = oracle["prefix"]
            else:
                arguments.update(osl=oracle["osl"], stride=oracle["stride"])
            run = getattr(self.baseline_binding._model(4), method)
            self.reject(
                f"direct-conflicting-phase:{phase}",
                lambda run=run, arguments=arguments: run(**arguments, latency_correction_scale=2.0),
            )
            omitted = run(**arguments)
            matching = run(**arguments, latency_correction_scale=self.baseline.number(factor_name))
            expected_hex = oracle.get(
                "expected_total_ms_hex", oracle.get("expected_service_ms_hex")
            )
            self.guard(
                f"direct-default-phase:{phase}", omitted.total.latency_ms.hex() == expected_hex
            )
            self.guard(
                f"direct-matching-phase:{phase}",
                omitted == matching and self.baseline.record_sha256 in omitted.total.rule,
            )
            derived = self.derived({factor_name: 2.0}, f"direct {phase} phase authority control")
            changed = getattr(self.binding(derived)._model(4), method)(**arguments)
            legacy_run = getattr(self.legacy_binding._model(4), method)
            expected = legacy_run(**arguments, latency_correction_scale=2.0)
            self.guard(
                f"direct-derived-phase:{phase}",
                changed.total.latency_ms.hex() == expected.total.latency_ms.hex()
                and derived.record_sha256 in changed.total.rule,
            )
            legacy_omitted = legacy_run(**arguments)
            legacy_matching = legacy_run(**arguments, latency_correction_scale=1.0)
            self.guard(f"direct-legacy-default:{phase}", legacy_omitted == legacy_matching)
            self.data["direct_pass_controls"].append(
                {
                    "phase": phase,
                    "configuration": arguments,
                    "omitted_total_ms_hex": omitted.total.latency_ms.hex(),
                    "matching_total_ms_hex": matching.total.latency_ms.hex(),
                    "derived_total_ms_hex": changed.total.latency_ms.hex(),
                    "legacy_derived_total_ms_hex": expected.total.latency_ms.hex(),
                    "legacy_omitted_total_ms_hex": legacy_omitted.total.latency_ms.hex(),
                    "record_sha256": self.baseline.record_sha256,
                    "derived_record_sha256": derived.record_sha256,
                }
            )

    def historical_oracles(self):
        self.services = {}
        self.old_services = {}
        for phase in ("decode", "prefill"):
            for oracle in self.config["oracles"][phase]:
                value = price(self.baseline_binding, oracle, phase)
                old_value = price(self.legacy_binding, oracle, phase)
                self.services[value.configuration_id] = value
                self.old_services[value.configuration_id] = old_value
                expected_service = oracle.get(
                    "expected_step_ms_hex", oracle.get("expected_service_ms_hex")
                )
                expected_total = oracle.get("expected_total_ms_hex", expected_service)
                self.exact(
                    "service",
                    value.configuration_id,
                    [value.service_ms_hex, value.total_ms_hex],
                    [expected_service, expected_total],
                )
                reference = legacy_service_reference(
                    oracle,
                    phase,
                    self.baseline.record.value["operation_source"],
                    self.config["composition"][f"{phase}_latency_correction_hex"],
                )
                self.guard(
                    f"absent-complete-serialization:{value.configuration_id}",
                    encoded(service_json(old_value)) == encoded(reference),
                    {"reference_recipe_commit": self.freeze["authored_against"]},
                )
                record_fields, old_fields = service_json(value), service_json(old_value)
                for key in ("source", "entry_key_sha256"):
                    del record_fields[key]
                    del old_fields[key]
                self.guard(f"absent-service:{value.configuration_id}", record_fields == old_fields)
                self.guard(
                    f"service-record-join:{value.configuration_id}",
                    self.baseline.record_sha256 in value.as_term().source
                    and (
                        phase == "prefill"
                        or value.as_batch_service_point().entry_key_sha256 == value.entry_key_sha256
                    ),
                )
                self.data["baseline_services"].append(
                    {
                        "configuration": oracle,
                        "record_enabled": service_json(value),
                        "record_absent": service_json(old_value),
                    }
                )
        quotients = self.quotients(self.services)
        prior = self.old["families"]["R"]["remove_one_sensitivity"][0]["rows"]
        for observed, expected in zip(quotients, prior, strict=True):
            self.exact("quotient", str(observed["row"]), observed, expected)
        for adjustment, expected in zip(
            self.table["adjustments"],
            self.old["families"]["R"]["remove_one_sensitivity"],
            strict=True,
        ):
            name = adjustment["id"]
            record = self.derived({name: float(adjustment["removal_value"])}, f"remove {name}")
            binding = self.binding(record)
            services = {
                oracle["id"]: price(binding, oracle, "decode")
                for oracle in self.config["oracles"]["decode"]
            }
            rows = self.quotients(services)
            for observed, reference in zip(rows, expected["rows"], strict=True):
                identity = f"{name}:{observed['row']}"
                if adjustment["family_r_reachable"]:
                    self.exact("active-remove-one", identity, observed, reference)
                else:
                    self.guard(f"inactive-remove-one:{identity}", observed == reference)
            # Retain historical disclosures unchanged, alongside the newly executed values.
            self.data["remove_one"].append(
                {
                    "record_sha256": record.record_sha256,
                    "adjustment_id": name,
                    "services": [service_json(value) for value in services.values()],
                    "quotients": rows,
                    "historical_disclosure": expected,
                }
            )
        corrected = self.services["prefill-tp4-b1"]
        raw_record = self.derived({"prefill_latency_correction": 1.0}, "raw prefill decomposition")
        raw = price(self.binding(raw_record), self.config["oracles"]["prefill"][1], "prefill")
        autoscaled = self.baseline_binding.autoscaled_ttft(corrected)
        published = self.old["families"]["D"]["published_ttft_ms"]
        components = {
            "raw_prefill_pass_ms": raw.service_ms,
            "corrected_prefill_service_ms": corrected.service_ms,
            "autoscaled_prefill_ms": autoscaled.latency_ms,
            "published_ttft_ms": published,
            "residual_from_raw_pass_ms": published - raw.service_ms,
            "attribution_ms": {
                "prefill_service_correction": corrected.service_ms - raw.service_ms,
                "autoscale_ttft_correction": autoscaled.latency_ms - corrected.service_ms,
                "publication_reconciliation": published - autoscaled.latency_ms,
            },
        }
        format_values = [
            raw.service_ms,
            components["residual_from_raw_pass_ms"],
            *components["attribution_ms"].values(),
        ]
        formatted = [
            {**row, "observed": f"{value:.15f}"}
            for row, value in zip(self.old["families"]["D"]["rows"], format_values, strict=True)
        ]
        self.exact(
            "first-token-decomposition",
            "all-numerical-and-formatted-components",
            {**components, "rows": formatted},
            {key: value for key, value in self.old["families"]["D"].items() if key != "status"},
        )
        self.data["first_token_decomposition"] = {
            **components,
            "rows": formatted,
            "raw_service": service_json(raw),
            "autoscale": autoscaled.as_dict(),
            "interpretation": "signed publication residual retained; not evidence of nearest three-decimal rounding",
        }

    def quotients(self, services):
        return [
            {"row": row, "quotient": fraction_json(value)}
            for row, value in historical._family_r_quotients(
                {key: value.service_ms for key, value in services.items()}
            )
        ]

    def memory(self):
        physics = self.freeze["physics"]
        layers, hidden, intermediate = (
            physics[name] for name in ("layers", "hidden_size", "intermediate_size")
        )
        bandwidth = physics["peak_memory_bytes_per_second"]
        peak_flops = physics["peak_fp8_flops_per_second"]
        scales = [float.fromhex(item) for item in self.freeze["memory_grid"]["bandwidth_scale_hex"]]
        constants = [
            float.fromhex(item) for item in self.freeze["memory_grid"]["constant_seconds_hex"]
        ]
        tolerance = float(self.freeze["memory_grid"]["absolute_rounding_allowance_ms"])
        bindings = {}
        for scale, constant in itertools.product(scales, constants):
            record = self.derived(
                {
                    "memory_bandwidth_empirical_scale": scale,
                    "memory_empirical_constant_latency": constant,
                },
                "prospective memory grid",
            )
            bindings[scale, constant] = self.binding(record)
        for phase in ("decode", "prefill"):
            for oracle in self.config["oracles"][phase]:
                identity, tp = oracle["id"], oracle["tensor_parallel"]
                tokens = oracle["batch_size"] * (
                    oracle["isl"] - oracle["prefix"] if phase == "prefill" else 1
                )
                weight_bytes = layers * 3 * hidden * intermediate / tp
                memory_floor = weight_bytes / bandwidth * 1000
                compute_floor = 2 * tokens * weight_bytes / peak_flops * 1000
                floor = max(memory_floor, compute_floor)
                ceiling = 2 * self.services[identity].service_ms
                main_bytes = tokens * (2 * hidden + layers * (16 * hidden + 6 * intermediate / tp))
                auxiliary = (
                    layers * 1.1 * (12 * (64 * 128 / tp) + 14 * (8 * 128 / tp))
                    if phase == "prefill"
                    else 0
                )
                phase_correction = 1.1 if phase == "prefill" else 1.08
                visits = 1 + 3 * layers + (12 * layers * 1.1 if phase == "prefill" else 0)
                expected_bandwidth = (
                    phase_correction
                    * (main_bytes + auxiliary)
                    / bandwidth
                    * (1 / scales[0] - 1 / scales[1])
                    * 1000
                )
                expected_constant = phase_correction * visits * (constants[1] - constants[0]) * 1000
                values = {}
                for (scale, constant), binding in bindings.items():
                    value = price(binding, oracle, phase)
                    values[scale, constant] = value.service_ms
                    cell_id = f"{identity}:{scale.hex()}:{constant.hex()}"
                    self.guard(
                        f"physical-memory-floor:{cell_id}",
                        value.service_ms >= floor,
                        {
                            "memory_floor_ms": memory_floor,
                            "compute_floor_ms": compute_floor,
                            "observed_ms": value.service_ms,
                        },
                    )
                    self.guard(
                        f"bounded-intervention-ceiling:{cell_id}",
                        value.service_ms <= ceiling,
                        {"ceiling_ms": ceiling, "observed_ms": value.service_ms},
                    )
                    self.data["memory_cells"].append(
                        {
                            "id": cell_id,
                            "configuration": oracle,
                            "record_sha256": binding.composition_record.record_sha256,
                            "bandwidth_scale_hex": scale.hex(),
                            "constant_seconds_hex": constant.hex(),
                            "service": service_json(value),
                            "memory_floor_ms": memory_floor,
                            "compute_floor_ms": compute_floor,
                            "ceiling_ms": ceiling,
                            "main_bytes": main_bytes,
                            "auxiliary_bytes": auxiliary,
                            "constant_visits": visits,
                        }
                    )
                for constant in constants:
                    observed = values[scales[0], constant] - values[scales[1], constant]
                    self.relation(
                        "memory-bandwidth",
                        f"{identity}:{constant.hex()}",
                        observed,
                        expected_bandwidth,
                        tolerance=tolerance,
                        positive=True,
                    )
                for scale in scales:
                    observed = values[scale, constants[1]] - values[scale, constants[0]]
                    self.relation(
                        "memory-constant",
                        f"{identity}:{scale.hex()}",
                        observed,
                        expected_constant,
                        tolerance=tolerance,
                        positive=True,
                    )

    def capacity(self):
        grid = self.freeze["capacity_grid"]
        rates_p = [float.fromhex(item) for item in grid["prefill_rate_hex"]]
        rates_d = [float.fromhex(item) for item in grid["decode_rate_hex"]]
        values = {}
        for p_rate, d_rate in itertools.product(rates_p, rates_d):
            record = self.derived(
                {
                    "prefill_rate_matching_degradation": p_rate,
                    "decode_rate_matching_degradation": d_rate,
                },
                "prospective pool capacity grid",
            )
            binding = self.binding(record)
            p_value = price(binding, self.config["oracles"]["prefill"][1], "prefill")
            decode_values = [
                price(binding, oracle, "decode")
                for oracle in sorted(
                    self.config["oracles"]["decode"], key=lambda row: row["batch_size"]
                )
                if oracle["id"] in {"decode-tp4-b16", "decode-tp4-b26"}
            ]
            d_value = decode_values[-1]
            for p_engines, d_engines in itertools.product(
                grid["prefill_engines"], grid["decode_engines"]
            ):
                candidate, prefill, decode = capacity_inputs(
                    binding, p_value, decode_values, p_engines=p_engines, d_engines=d_engines
                )
                result = binding.disaggregated_capacity(
                    candidate,
                    prefill,
                    decode,
                    grid["decode_batch"],
                    prefix=grid["prefix"],
                    stride=grid["stride"],
                )
                index = p_engines, d_engines, p_rate, d_rate
                values[index] = result
                identity = f"p{p_engines}-d{d_engines}-pr{p_rate.hex()}-dr{d_rate.hex()}"
                p = Fraction(p_engines * 10**12, prefill.request_ps) * Fraction.from_float(p_rate)
                d = Fraction(d_engines * 26 * 10**12, 500 * decode.step_ps) * Fraction.from_float(
                    d_rate
                )
                self.guard(
                    f"exact-capacity-arithmetic:{identity}",
                    result.prefill_requests_per_second == p
                    and result.decode_requests_per_second == d
                    and result.request_capacity == min(p, d),
                    {"prefill": fraction_json(p), "decode": fraction_json(d)},
                )
                self.guard(
                    f"capacity-original-stamps:{identity}",
                    result.prefill.stamp is prefill.stamp and result.decode.stamp is decode.stamp,
                )
                self.guard(
                    f"capacity-service-inactive-factors:{identity}",
                    p_value.service_ms_hex == self.services[p_value.configuration_id].service_ms_hex
                    and d_value.service_ms_hex
                    == self.services[d_value.configuration_id].service_ms_hex,
                )
                self.data["capacity_cells"].append(
                    {
                        "id": identity,
                        "configuration": candidate_to_json(candidate),
                        "result": result.as_dict(),
                        "prefill_service": service_json(p_value),
                        "decode_service": service_json(d_value),
                        "decode_surface": [service_json(item) for item in decode_values],
                    }
                )
                if index == (1, 1, rates_p[-1], rates_d[-1]):
                    self.projection_controls(
                        binding, candidate, prefill, decode, p_value, decode_values
                    )
        for axis, name in enumerate(
            ("prefill-engines", "decode-engines", "prefill-rate", "decode-rate")
        ):
            for index, before in values.items():
                domain = (grid["prefill_engines"], grid["decode_engines"], rates_p, rates_d)[axis]
                if index[axis] != domain[0]:
                    continue
                upper = list(index)
                upper[axis] = domain[1]
                after = values[tuple(upper)]
                before_pool = (
                    before.prefill_requests_per_second
                    if axis in (0, 2)
                    else before.decode_requests_per_second
                )
                after_pool = (
                    after.prefill_requests_per_second
                    if axis in (0, 2)
                    else after.decode_requests_per_second
                )
                ratio = after_pool / before_pool
                self.relation(name, repr(index), float(ratio), 2.0, positive=True)
                self.guard(f"exact-rational-doubling:{name}:{index}", ratio == 2)
                other_pool = (
                    before.decode_requests_per_second
                    if axis in (0, 2)
                    else before.prefill_requests_per_second
                )
                expected_delta = min(2 * before_pool, other_pool) - min(before_pool, other_pool)
                delta = after.request_capacity - before.request_capacity
                if expected_delta == 0:
                    self.guard(
                        f"inactive-pool-identity:{name}:{index}",
                        delta == 0,
                        {
                            "delta": fraction_json(delta),
                            "expected_delta": fraction_json(expected_delta),
                        },
                    )
                else:
                    self.relation(
                        "pool-limiter-clipping",
                        f"{name}:{index}",
                        float(delta),
                        float(expected_delta),
                        positive=True,
                    )
                self.guard(f"exact-rational-clipping:{name}:{index}", delta == expected_delta)
        self.guard(
            "both-pool-limiters-occur",
            {value.capacity_limiter for value in values.values()} == {"prefill", "decode"},
        )
        old_prefill = self.old_services["prefill-tp4-b1"]
        old_decode = [self.old_services[f"decode-tp4-b{batch}"] for batch in (16, 26)]
        _, p, d = capacity_inputs(self.legacy_binding, old_prefill, old_decode)
        selected = values[1, 1, rates_p[-1], rates_d[-1]]
        self.guard(
            "record-absent-capacity",
            selected.prefill_requests_per_second
            == Fraction(10**12, p.request_ps) * Fraction.from_float(0.9)
            and selected.decode_requests_per_second
            == Fraction(26 * 10**12, 500 * d.step_ps) * Fraction.from_float(0.92),
        )

    def projection_controls(self, binding, candidate, prefill, decode, value, decode_values):
        result = binding.disaggregated_capacity(
            candidate, prefill, decode, 26, prefix=500, stride=32
        )
        autoscale = binding.autoscaled_ttft(value)
        for name, altered in (
            ("batch_size", 27),
            ("prefix", 499),
            ("stride", 16),
            ("candidate", replace(candidate, candidate_id="wrong identity")),
        ):
            self.reject(
                f"projection-wrong-{name}",
                lambda name=name, altered=altered: replace(result, **{name: altered}),
            )
        for name, altered in (
            ("prefill_rate_hex", (1.0).hex()),
            ("decode_rate_hex", (1.0).hex()),
            ("used_gpus", 1),
            ("composition_sha256", "a" * 64),
        ):
            self.reject(
                f"projection-independent-{name}",
                lambda name=name, altered=altered: replace(result, **{name: altered}),
            )
        self.reject(
            "autoscale-independent-factor", lambda: replace(autoscale, factor_hex=(2.0).hex())
        )
        altered_ms = value.service_ms + 1.0
        altered_value = replace(
            value,
            service_ms=altered_ms,
            total_ms=altered_ms,
            source=value.source.replace(value.service_ms_hex, altered_ms.hex()),
        )
        self.reject(
            "coherent-selected-price-replacement",
            lambda: replace(autoscale.selection, value=altered_value),
        )
        self.reject(
            "coherent-binding-price-replacement", lambda: binding.autoscaled_ttft(altered_value)
        )
        other_record = self.derived({"autoscale_ttft_correction": 2.0}, "wrong record control")
        self.reject("projection-wrong-record", lambda: replace(result, composition=other_record))
        kernel = replace(decode.kernel_floor, duration_ps=decode.step_ps + 1)
        altered_stamp = replace(
            decode.stamp,
            terms=tuple(
                replace(term, estimate=kernel) if term.name == "kernel_floor" else term
                for term in decode.stamp.terms
            ),
        )
        altered_decode = replace(
            decode,
            kernel_floor=kernel,
            batch_service=kernel,
            step_ps=kernel.duration_ps,
            analytical_step_ps=kernel.duration_ps,
            stamp=altered_stamp,
        )
        self.reject(
            "coherent-estimator-duration-replacement",
            lambda: replace(result, decode=altered_decode),
        )
        _, delayed, same_decode = capacity_inputs(
            binding, value, decode_values, handoff_ps=100_000_000
        )
        with_handoff = binding.disaggregated_capacity(
            candidate, delayed, same_decode, 26, prefix=500, stride=32
        )
        self.guard(
            "handoff-remains-additive",
            delayed.request_ps == prefill.request_ps + 100_000_000
            and delayed.step_ps == prefill.step_ps
            and with_handoff.prefill_requests_per_second < result.prefill_requests_per_second,
        )
        self.guard(
            "handoff-outside-autoscale",
            binding.autoscaled_ttft(value).as_dict() == autoscale.as_dict(),
        )
        self.data["handoff_control"] = {
            "without": result.as_dict(),
            "with": with_handoff.as_dict(),
            "autoscale": autoscale.as_dict(),
        }

    def autoscale(self):
        grid = self.freeze["autoscale_grid"]
        values = {}
        for factor_hex in grid["scale_hex"]:
            factor = float.fromhex(factor_hex)
            record = self.derived(
                {"autoscale_ttft_correction": factor}, "prospective first-token heuristic grid"
            )
            binding = self.binding(record)
            for oracle in self.config["oracles"]["prefill"]:
                value = price(binding, oracle, "prefill")
                result = binding.autoscaled_ttft(value)
                values[oracle["tensor_parallel"], factor_hex] = result.latency_ms
                expected = value.service_ms * factor
                self.guard(
                    f"autoscale-binary-operator:{oracle['id']}:{factor_hex}",
                    result.latency_ms.hex() == expected.hex()
                    and result.as_term().duration_ps == round(expected * 10**9),
                )
                self.guard(
                    f"autoscale-isolated-identity:{oracle['id']}:{factor_hex}",
                    value.service_ms_hex == self.services[oracle["id"]].service_ms_hex,
                )
                self.data["autoscale_cells"].append(
                    {
                        "configuration": oracle,
                        "service": service_json(value),
                        "result": result.as_dict(),
                    }
                )
                if factor == 1.0:
                    self.guard(
                        f"autoscale-off-identity:{oracle['id']}",
                        result.latency_ms.hex() == value.service_ms_hex,
                    )
                else:
                    self.relation(
                        "autoscale-first-token",
                        f"{oracle['id']}:{factor_hex}",
                        result.latency_ms,
                        expected,
                        positive=True,
                    )
        for tp in grid["tensor_parallel"]:
            low, high = (values[tp, factor] for factor in grid["scale_hex"])
            self.relation(
                "autoscale-direction", f"tp{tp}", high - low, low * 1.8 - low, positive=True
            )
        legacy = self.old_services["prefill-tp4-b1"]
        self.guard(
            "record-absent-autoscale",
            (legacy.service_ms * 1.8).hex() == values[4, grid["scale_hex"][1]].hex(),
        )

    def aggregate(self):
        prior = read_json(ROOT / "examples/matched_seam_frontier_v1/agg_record.json")
        baseline_rows = prior["families"]["AR"]["baseline_projection"]
        inactive = self.derived(
            {name: 1.0 for name in self.freeze["aggregate_scope"]["inactive"]},
            "inactive aggregate serving factors",
        )
        inactive_binding = self.binding(inactive)

        def numerical(point):
            return {
                key: value
                for key, value in point.items()
                if key not in {"source", "entry_key_sha256"}
            }

        for row in baseline_rows:
            reference = row["point"]
            args = {
                name: reference[name]
                for name in (
                    "tensor_parallel",
                    "batch_size",
                    "isl",
                    "osl",
                    "prefix",
                    "context_tokens",
                )
            }
            old = self.legacy_binding.aggregate_point(**args).as_dict()
            current = self.baseline_binding.aggregate_point(**args).as_dict()
            unused = inactive_binding.aggregate_point(**args).as_dict()
            self.guard(f"aggregate-absent-complete-reference:{row['row']}", old == reference)
            self.guard(
                f"aggregate-baseline-numerical-identity:{row['row']}",
                numerical(old) == numerical(current),
            )
            self.guard(
                f"aggregate-five-inactive:{row['row']}", numerical(current) == numerical(unused)
            )
            self.data["aggregate_cells"].append(
                {
                    "configuration": args,
                    "record_absent": old,
                    "record_enabled": current,
                    "inactive_five": unused,
                }
            )
        # These are historical regression controls, not a new aggregate calibration.
        override_arguments = {
            "memory_bandwidth_empirical_scale": {"memory_bandwidth_empirical_scale": 1.0},
            "memory_empirical_constant_latency": {"memory_empirical_constant_latency_s": 0.0},
            "context_attention_extra_latency_correction": {
                "context_attention_extra_latency_correction": 1.0
            },
            "aggregate_ttft_queueing_heuristic": {"apply_ttft_queueing": False},
            "trtllm_tpot_mixed_step_reduction": {"tpot_mixed_step_reduction": 0},
        }
        self.data["aggregate_legacy_interventions"] = []
        for intervention in prior["families"]["ADJ"]["rows"]:
            name = intervention["adjustment_id"]
            if name not in override_arguments:
                continue
            for row in intervention["projection"]:
                reference = row["point"]
                args = {
                    key: reference[key]
                    for key in (
                        "tensor_parallel",
                        "batch_size",
                        "isl",
                        "osl",
                        "prefix",
                        "context_tokens",
                    )
                }
                current = self.legacy_binding.aggregate_point(
                    **args, **override_arguments[name]
                ).as_dict()
                self.guard(
                    f"aggregate-legacy-intervention:{name}:{row['row']}", current == reference
                )
                self.data["aggregate_legacy_interventions"].append(
                    {"adjustment": name, "configuration": args, "point": current}
                )
        first = self.data["aggregate_cells"][0]
        self.data["aggregate_active_controls"] = []
        for name in self.freeze["aggregate_scope"]["applied"]:
            removed = 0.0 if name == "memory_empirical_constant_latency" else 1.0
            record = self.derived({name: removed}, f"aggregate operation reachability {name}")
            point = self.binding(record).aggregate_point(**first["configuration"])
            expected = first["record_enabled"]
            self.guard(
                f"aggregate-active-reachability:{name}",
                point.ttft_ms < float.fromhex(expected["ttft_ms"])
                and point.tpot_ms < float.fromhex(expected["tpot_ms"]),
            )
            self.guard(
                f"aggregate-applied-subset:{name}",
                f"composition-applied:{','.join(self.freeze['aggregate_scope']['applied'])}"
                in point.source,
            )
            self.data["aggregate_active_controls"].append(
                {"record_sha256": record.record_sha256, "point": point.as_dict()}
            )

    def completeness(self):
        self.guard(
            "direct-pass-control-completeness",
            {row["phase"] for row in self.data.get("direct_pass_controls", [])}
            == {"prefill", "decode"}
            and len(self.data.get("direct_pass_controls", [])) == 2,
        )
        oracle_shapes = [
            row for phase in ("decode", "prefill") for row in self.config["oracles"][phase]
        ]
        expected_counts = self.freeze["historical_expected_classes"]
        exact_names = {
            "service": "service_oracle_rows",
            "quotient": "quotient_oracle_rows",
            "active-remove-one": "active_remove_one_oracle_rows",
            "first-token-decomposition": "complete_first_token_decompositions",
        }
        exact_rows = self.data["exact_oracles"]
        self.guard(
            "exact-class-completeness",
            {name: sum(row["family"] == name for row in exact_rows) for name in exact_names}
            == {name: expected_counts[count] for name, count in exact_names.items()}
            and {row["family"] for row in exact_rows} == set(exact_names),
        )
        self.guard(
            "unique-exact-row-identities",
            len({(row["family"], row["id"]) for row in exact_rows}) == len(exact_rows),
        )
        inactive_ids = {
            f"inactive-remove-one:{adjustment['id']}:{index}"
            for adjustment in self.table["adjustments"]
            if not adjustment["family_r_reachable"]
            for index in range(1, 11)
        }
        actual_inactive = [
            row["id"]
            for row in self.data["fatal_guards"]
            if row["id"].startswith("inactive-remove-one:")
        ]
        self.guard(
            "inactive-removal-completeness",
            set(actual_inactive) == inactive_ids
            and len(actual_inactive) == expected_counts["inactive_remove_one_identity_rows"],
        )
        memory = self.freeze["memory_grid"]
        expected_memory = {
            f"{shape['id']}:{scale}:{constant}"
            for shape in oracle_shapes
            for scale in memory["bandwidth_scale_hex"]
            for constant in memory["constant_seconds_hex"]
        }
        self.guard(
            "memory-grid-identities",
            {row["id"] for row in self.data["memory_cells"]} == expected_memory
            and len(self.data["memory_cells"]) == len(expected_memory),
        )
        grid = self.freeze["capacity_grid"]
        expected_capacity = {
            f"p{p}-d{d}-pr{pr}-dr{dr}"
            for p, d, pr, dr in itertools.product(
                grid["prefill_engines"],
                grid["decode_engines"],
                grid["prefill_rate_hex"],
                grid["decode_rate_hex"],
            )
        }
        self.guard(
            "capacity-grid-identities",
            {row["id"] for row in self.data["capacity_cells"]} == expected_capacity
            and len(self.data["capacity_cells"]) == len(expected_capacity),
        )
        autoscale = self.freeze["autoscale_grid"]
        expected_auto = set(itertools.product(autoscale["tensor_parallel"], autoscale["scale_hex"]))
        self.guard(
            "autoscale-grid-identities",
            {
                (row["configuration"]["tensor_parallel"], row["result"]["factor_hex"])
                for row in self.data["autoscale_cells"]
            }
            == expected_auto
            and len(self.data["autoscale_cells"]) == len(expected_auto),
        )
        expected_relations = set()
        for shape in oracle_shapes:
            expected_relations.update(
                ("memory-bandwidth", f"{shape['id']}:{constant}")
                for constant in memory["constant_seconds_hex"]
            )
            expected_relations.update(
                ("memory-constant", f"{shape['id']}:{scale}")
                for scale in memory["bandwidth_scale_hex"]
            )
        p_service = round(
            float.fromhex(self.config["oracles"]["prefill"][1]["expected_service_ms_hex"]) * 10**9
        )
        d_oracle = next(
            row for row in self.config["oracles"]["decode"] if row["id"] == "decode-tp4-b26"
        )
        d_service = round(float.fromhex(d_oracle["expected_step_ms_hex"]) * 10**9)
        domains = (
            grid["prefill_engines"],
            grid["decode_engines"],
            [float.fromhex(value) for value in grid["prefill_rate_hex"]],
            [float.fromhex(value) for value in grid["decode_rate_hex"]],
        )
        for axis, name in enumerate(
            ("prefill-engines", "decode-engines", "prefill-rate", "decode-rate")
        ):
            for index in itertools.product(*domains):
                if index[axis] != domains[axis][0]:
                    continue
                expected_relations.add((name, repr(index)))
                p = Fraction(index[0] * 10**12, p_service) * Fraction.from_float(index[2])
                d = Fraction(index[1] * 26 * 10**12, 500 * d_service) * Fraction.from_float(
                    index[3]
                )
                pool, other = (p, d) if axis in (0, 2) else (d, p)
                if min(2 * pool, other) != min(pool, other):
                    expected_relations.add(("pool-limiter-clipping", f"{name}:{index}"))
        for tp in autoscale["tensor_parallel"]:
            expected_relations.add(("autoscale-direction", f"tp{tp}"))
            expected_relations.update(
                ("autoscale-first-token", f"prefill-tp{tp}-b1:{factor}")
                for factor in autoscale["scale_hex"]
                if float.fromhex(factor) != 1.0
            )
        observed_relations = [
            (row["family"], row["id"]) for row in self.data["behavioral_relations"]
        ]
        self.guard(
            "behavioral-relation-identities",
            set(observed_relations) == expected_relations
            and len(observed_relations) == len(expected_relations),
        )
        guards = self.data["fatal_guards"]
        self.guard(
            "unique-fatal-guard-identities", len({row["id"] for row in guards}) == len(guards)
        )

    def run(self):
        import simllm.compute.provider as compute_provider
        from simllm.deploy import estimator

        calls = {"simllm_roofline": 0, "forbidden_import": 0, "packet_subprocess": 0}

        class ForbiddenRoofline:
            def __init__(self, *args, **kwargs):
                calls["simllm_roofline"] += 1
                raise AssertionError("SimLLM roofline is not an external measured service")

        def audit(event, arguments):
            if event == "import" and str(arguments[0]).split(".")[0] in {"torch", "cupy", "triton"}:
                calls["forbidden_import"] += 1
                raise AssertionError("GPU runtime imports are outside this study")
            if event == "subprocess.Popen":
                calls["packet_subprocess"] += 1
                raise AssertionError("the composition worker launches no backend process")

        sys.addaudithook(audit)
        compute_provider.RooflineProvider = ForbiddenRoofline
        estimator.RooflineProvider = ForbiddenRoofline
        self.preservation("before")
        for name in (
            "admission",
            "direct_pass_controls",
            "historical_oracles",
            "memory",
            "capacity",
            "autoscale",
            "aggregate",
        ):
            print(f"starting {name}", flush=True)
            getattr(self, name)()
            print(f"completed {name}", flush=True)
        self.preservation("after")
        self.guard("complete-service-oracles", len(self.data["baseline_services"]) == 13)
        self.guard("complete-memory-grid", len(self.data["memory_cells"]) == 52)
        self.guard(
            "complete-capacity-grid",
            len(self.data["capacity_cells"]) == self.freeze["capacity_grid"]["cell_count"],
        )
        self.guard(
            "complete-autoscale-grid",
            len(self.data["autoscale_cells"]) == self.freeze["autoscale_grid"]["cell_count"],
        )
        self.guard(
            "complete-removal-disclosures",
            len(self.data["remove_one"]) == 8
            and all(len(row["quotients"]) == 10 for row in self.data["remove_one"]),
        )
        self.guard(
            "no-GPU-work",
            calls["forbidden_import"] == 0
            and not any(name in sys.modules for name in ("torch", "cupy", "triton")),
        )
        self.guard("no-SimLLM-roofline", calls["simllm_roofline"] == 0)
        self.guard("no-packet-backend", calls["packet_subprocess"] == 0)
        self.completeness()
        self.data["runtime_calls"] = calls
        self.data["freeze"] = self.freeze

    def finish(self):
        fatal = [row["id"] for row in self.data["fatal_guards"] if not row["passed"]]
        exact = self.data["exact_oracles"]
        behavioral = self.data["behavioral_relations"]
        state = (
            "VOID"
            if fatal
            else "FAIL"
            if any(not row["passed"] for row in [*exact, *behavioral])
            else "PASS"
        )
        self.data.update(
            {
                "state": state,
                "fatal_findings": fatal,
                "behavioral_score": None
                if fatal
                else {
                    "passed": sum(row["passed"] for row in behavioral),
                    "instances": len(behavioral),
                },
                "exact_oracle_classes": {
                    name: sum(row["family"] == name for row in exact)
                    for name in sorted({row["family"] for row in exact})
                },
                "behavioral_families": sorted({row["family"] for row in behavioral}),
            }
        )
        return self.data


def worker(output):
    started = time.monotonic()
    evaluation = Evaluation()
    try:
        evaluation.run()
    except Exception as error:  # noqa: BLE001, preserve a void run on any worker failure
        evaluation.guard("worker-completed", False, type(error).__name__)
        traceback.print_exc()
    write_new(output, evaluation.finish())
    write_new(
        output.with_suffix(".process.json"),
        {
            "process_identity": os.getpid(),
            "elapsed_wall_seconds": time.monotonic() - started,
            "python": sys.version,
            "executable": sys.executable,
        },
    )
    return 0 if evaluation.data["state"] == "PASS" else 1


def git(*arguments):
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def valid_worker_summary(value):
    """Admit typed retained summaries before reading their aggregate fields."""

    if (
        not isinstance(value, dict)
        or type(value.get("state")) is not str
        or value["state"] not in {"PASS", "FAIL", "VOID"}
    ):
        return False
    for name in ("fatal_guards", "exact_oracles", "behavioral_relations"):
        rows = value.get(name)
        if not isinstance(rows, list) or any(
            not isinstance(row, dict)
            or type(row.get("passed")) is not bool
            or not isinstance(row.get("id"), str)
            or (name != "fatal_guards" and not isinstance(row.get("family"), str))
            for row in rows
        ):
            return False
    classes = value.get("exact_oracle_classes")
    if not isinstance(classes, dict) or any(
        type(count) is not int or count < 0 for count in classes.values()
    ):
        return False
    supplied_score = value.get("behavioral_score")
    if supplied_score is not None and (
        not isinstance(supplied_score, dict)
        or set(supplied_score) != {"passed", "instances"}
        or any(type(count) is not int or count < 0 for count in supplied_score.values())
    ):
        return False
    fatal = [row["id"] for row in value["fatal_guards"] if not row["passed"]]
    exact, behavioral = value["exact_oracles"], value["behavioral_relations"]
    state = (
        "VOID"
        if fatal
        else "FAIL"
        if any(not row["passed"] for row in [*exact, *behavioral])
        else "PASS"
    )
    score = (
        None
        if fatal
        else {"passed": sum(row["passed"] for row in behavioral), "instances": len(behavioral)}
    )
    return (
        value.get("state") == state
        and value.get("fatal_findings") == fatal
        and "behavioral_score" in value
        and value["behavioral_score"] == score
        and value.get("exact_oracle_classes")
        == {
            name: sum(row["family"] == name for row in exact)
            for name in sorted({row["family"] for row in exact})
        }
        and value.get("behavioral_families") == sorted({row["family"] for row in behavioral})
    )


def aggregate_workers(values, metadata, codes, *, equal, source_unchanged):
    """Convert malformed, incomplete or inconsistent process evidence to VOID."""

    findings = []
    valid = [valid_worker_summary(value) for value in values]
    identities = [value.get("process_identity") for value in metadata if isinstance(value, dict)]
    typed_ids = [value for value in identities if type(value) is int and value > 0]
    if not equal:
        findings.append("complete-evaluation-byte-identity")
    if len(metadata) != 2 or len(typed_ids) != 2 or len(set(typed_ids)) != 2:
        findings.append("two-fresh-worker-identities")
    if len(values) != 2 or len(codes) != 2:
        findings.append("two-complete-worker-results")
    for index, (value, accepted) in enumerate(zip(values, valid, strict=True)):
        if not accepted:
            findings.append(f"worker-summary-invalid:{index}")
            continue
        if value["state"] == "VOID":
            findings.append(f"worker-fatal-guard:{index}")
        expected_code = 0 if value["state"] == "PASS" else 1
        if index >= len(codes) or type(codes[index]) is not int or codes[index] != expected_code:
            findings.append(f"worker-process-state:{index}")
    if not source_unchanged:
        findings.append("source-changed-during-run")
    state = (
        "VOID"
        if findings
        else "PASS"
        if all(value["state"] == "PASS" for value in values)
        else "FAIL"
    )
    first = values[0] if valid and valid[0] else {}
    return {
        "state": state,
        "fatal_findings": findings,
        "worker_exit_codes": codes,
        "worker_states": [
            value.get("state", "MALFORMED") if isinstance(value, dict) else "MISSING"
            for value in values
        ],
        "deterministic_bytes_equal": equal,
        "fresh_worker_count": len(set(typed_ids)),
        "behavioral_score": None if findings else first["behavioral_score"],
        "exact_oracle_classes": first.get("exact_oracle_classes", {}),
        "behavioral_families": first.get("behavioral_families", []),
    }


def coordinator(output_root):
    output_root = output_root.resolve()
    if output_root == ROOT or ROOT in output_root.parents:
        raise ValueError("--output-root must be outside the repository")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("commit the implementation before running the prospective study")
    source_commit = git("rev-parse", "HEAD")
    git("merge-base", "--is-ancestor", FREEZE_COMMIT, source_commit)
    freeze_paths = [
        f"examples/external_composition_v1/{name}"
        for name in ("expectations.md", "expectations.json")
    ]
    if git("diff", "--name-only", FREEZE_COMMIT, "HEAD", "--", *freeze_paths):
        raise ValueError("frozen expectations changed after the expectations-only commit")
    output_root.mkdir(parents=True, exist_ok=False)
    write_new(
        output_root / "configuration.json",
        {
            "source_commit": source_commit,
            "expectations_commit": FREEZE_COMMIT,
            "expectations_sha256": {path: digest(ROOT / path) for path in freeze_paths},
            "baseline_record_sha256": BASELINE_RECORD,
            "workers": 2,
            "comparison": "complete deterministic evaluation bytes; process metadata outside evaluation",
        },
    )
    workers = []
    for index in range(2):
        output = output_root / f"evaluation-{index}.json"
        stdout = (output_root / f"worker-{index}.stdout").open("xb")
        stderr = (output_root / f"worker-{index}.stderr").open("xb")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "examples.external_composition_v1.run_study",
                "--worker-output",
                str(output),
            ],
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
        )
        workers.append((process, stdout, stderr, output))
    codes = []
    for process, stdout, stderr, output in workers:
        codes.append(process.wait())
        stdout.close()
        stderr.close()
    outputs = [row[3] for row in workers]

    def retained_json(path):
        try:
            value = read_json(path)
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    values = [retained_json(path) for path in outputs]
    equal = (
        all(path.exists() for path in outputs)
        and outputs[0].read_bytes() == outputs[1].read_bytes()
    )
    metadata = [retained_json(path.with_suffix(".process.json")) for path in outputs]
    aggregation = aggregate_workers(
        values,
        metadata,
        codes,
        equal=equal,
        source_unchanged=(
            git("rev-parse", "HEAD") == source_commit
            and not git("status", "--porcelain", "--untracked-files=normal")
        ),
    )
    summary = {
        "schema": "simllm-external-composition-summary-v1",
        **aggregation,
        "source_commit": source_commit,
        "expectations_commit": FREEZE_COMMIT,
        "evaluation_sha256": [digest(path) if path.exists() else None for path in outputs],
        "historical_runs_rescored": False,
        "GPU_calibration_changed": False,
    }
    write_new(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["state"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.output_root is None) == (args.worker_output is None):
        parser.error("provide --output-root")
    return worker(args.worker_output) if args.worker_output else coordinator(args.output_root)


if __name__ == "__main__":
    raise SystemExit(main())
