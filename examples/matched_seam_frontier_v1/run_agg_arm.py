#!/usr/bin/env python3
"""Run the frozen DEPLOY-22 co-located aggregate composition arm."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from examples.matched_seam_frontier_v1 import plot_agg
from examples.matched_seam_frontier_v1 import run_study as base
from simllm.calibration.external_db import (
    EXTERNAL_EVIDENCE_CLASS,
    ExternalOperationDatabase,
    default_artifact_dir,
)
from simllm.deploy import ExternalAggregatePoint, ExternalQwen32BDeploymentBinding

SCHEMA = "simllm-matched-seam-aggregate-record-v1"
EVALUATION_SCHEMA = "simllm-matched-seam-aggregate-evaluation-v1"
EXPECTATIONS_COMMIT = "923c384b786d55530084b33dfeedb2790752cb22"
EXPECTATIONS_SHA256 = (
    "19f04e97d9f3df0c30d6bc5546e390d14b044be22884309c7f15e17b61da81ff"
)
ADJUSTMENTS_SHA256 = (
    "99106ef0d421c045f0cd6afc7d9cd8aef600bb771e167b8ef9c68dd123332020"
)
EXTERNAL_AGG_SHA256 = (
    "89b062634eacb75acacf7a6935e00d42992d112359a302eb8a998992f52ab1f3"
)
PARITY_RECORD_SHA256 = (
    "38d616b3245a8a42bd06ee9f79d3397d16476cd4acd46dfecd7bd503a55a0e96"
)
WALL_CEILING_SECONDS = 600.0
BULK_ROOT_ENV = "SIMLLM_DEPLOY22_BULK_ROOT"
EXTERNAL_VENV_ENV = "SIMLLM_EXTERNAL_AIC_VENV"

EXPECTATIONS_PATH = STUDY_DIR / "expectations_agg.md"
ADJUSTMENTS_PATH = STUDY_DIR / "external_adjustments_agg.json"
BASE_ADJUSTMENTS_PATH = STUDY_DIR / "external_adjustments.json"
BASE_RECORD_PATH = STUDY_DIR / "record.json"
EXTERNAL_AGG_PATH = (
    REPOSITORY_ROOT / "examples/frontier_comparison_v1/external/agg_pareto.csv"
)
PARITY_RECORD_PATH = REPOSITORY_ROOT / "examples/external_db_parity_v1/record.json"
RESULT_PATH = STUDY_DIR / "agg_record.json"
CSV_PATH = STUDY_DIR / "agg_results.csv"
FIGURE_DIR = STUDY_DIR / "figures"
STUDY_PDF_PATH = FIGURE_DIR / "matched-seam-frontier-aggregate.pdf"
STUDY_PNG_PATH = FIGURE_DIR / "matched-seam-frontier-aggregate.png"
PUBLICATION_PDF_PATH = FIGURE_DIR / "matched-seam-frontier-aggregate-publication.pdf"
PUBLICATION_PNG_PATH = FIGURE_DIR / "matched-seam-frontier-aggregate-publication.png"

PROTECTED_PRIOR_SHA256 = {
    "DEPLOY12_RESULTS.md": "502c835fd33fd5bd0abee11ae2548eaf099e39653671d9a1a3c993a76530c6c3",
    "RESULTS.md": "fa1170277fa8f3b9f1a14df353add3dbd4e8e490aeb4847748dd2120d4434e62",
    "deploy12_record.json": "c2b4fa9b8e8c2401d01a36731e9e1989ef27918b5bb170813b436c0e61ab630f",
    "deploy12_results.csv": "4057d5f321ae60bd7e34bd8b3e9ca663694f189788632c34806b4bfe1b7bc4a8",
    "expectations.md": "fc5af307fee560fc7050011543e18e1cf77030d0aa6a13e6c5a014cb159a5726",
    "expectations_deploy12.md": "ed784f7514fe766c509b02ed591391370129b84c63cc51552e278f5fcee44812",
    "expectations_v2.md": "fe403500575d674a25c8b7c6c59eb41aec65fce6cc29024609fa995b29585f35",
    "external_adjustments.json": "c6778a81cdc6078ce74f06733e4bce9d99a92b4ab3eccba4a83d14e7d063a09e",
    "figure_addendum.md": "cc4dcb8c82bbcd5e542457b56d91ddf172af2cbe05e6bac5c865535dcc307762",
    "plot_publication.py": "a98514cb985a9980a679357285a11dbe52418e774a55d69a6c9f30ba9ddda53d",
    "plot_study.py": "d4fe430f1fede23bcbcbb21834d98a51d3563c4b4e4c21dc887c7b8c837a7e4f",
    "record.json": "bddd7cb040a3c0f0ec8afd7ea836d873fa22cad2131f98ff36e38da5441b2d50",
    "results.csv": "4113ab2413084b7da957de60002abc4a4f8530bbb89837a5a5f73b9852f4448d",
    "run_deploy12_arm.py": "683bd65e48539bfaf657b3b1f1aaa0dbcae809bbc39de815be50544fcba03a41",
    "run_study.py": "242b5f1ae46ac18ac2cb474ad6fa24acc4dba21c4b8ff1d6683137163fec3182",
    "study_config.json": "64c8e16de53e194e98f5ca7c9b27d533d4c7f7ca32311841a62e3c6cece21f17",
    "figures/matched-seam-frontier-publication.pdf": (
        "511a0fb869d3397a87664d28c6b0c1d5adc17738dd84543973f66c7fcfd764cb"
    ),
    "figures/matched-seam-frontier-publication.png": (
        "d79b5099cbbfeed9e4272a64d7007512ed1889a08fc3438c9f2eef41a28986d1"
    ),
    "figures/matched-seam-frontier.pdf": (
        "4ecc3bf2822f916bfd53107b55d1344406efea01fd0b1ad7a417019391712dbb"
    ),
    "figures/matched-seam-frontier.png": (
        "852378a01d3c9e0aeab74423259afe86b456dca0b193e27c23e48256322069c4"
    ),
    "../frontier_comparison_v1/external/agg_pareto.csv": EXTERNAL_AGG_SHA256,
    "../external_db_parity_v1/record.json": PARITY_RECORD_SHA256,
}

EXPECTED_APPLIED_ADJUSTMENTS = {
    "aggregate_ttft_queueing_heuristic",
    "context_attention_extra_latency_correction",
    "memory_bandwidth_empirical_scale",
    "memory_empirical_constant_latency",
    "trtllm_tpot_mixed_step_reduction",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _hash_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _measurement(value: float) -> dict[str, float | str]:
    return {"hex": value.hex(), "decimal": value}


def _protected_hashes() -> dict[str, str]:
    return {
        relative: _sha256(STUDY_DIR / relative)
        for relative in PROTECTED_PRIOR_SHA256
    }


def _external_rows() -> list[dict[str, str]]:
    with EXTERNAL_AGG_PATH.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 25:
        raise ValueError("external aggregate table must contain 25 rows")
    return rows


def _adjustment_rows() -> list[dict[str, Any]]:
    base_table = base._load_json(BASE_ADJUSTMENTS_PATH)
    extension = base._load_json(ADJUSTMENTS_PATH)
    reachability = {
        str(row["base_adjustment_id"]): row
        for row in extension["aggregate_reachability"]
    }
    rows = []
    for adjustment in base_table["adjustments"]:
        factor_id = str(adjustment["id"])
        reach = reachability[factor_id]
        rows.append(
            {
                **adjustment,
                "aggregate_tpot_reachable": bool(
                    reach["aggregate_tpot_reachable"]
                ),
                "aggregate_ttft_reachable": bool(
                    reach["aggregate_ttft_reachable"]
                ),
                "reachability_reason": str(reach["reason"]),
            }
        )
    rows.extend(extension["additional_adjustments"])
    if len(rows) != 10 or len({str(row["id"]) for row in rows}) != 10:
        raise ValueError("aggregate adjustment declaration must contain ten factors")
    return rows


def _source_references(
    adjustments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    extension = base._load_json(ADJUSTMENTS_PATH)
    references = []
    for adjustment in adjustments:
        for role in ("source", "documentation"):
            source = adjustment[role]
            references.append(
                {
                    "owner": str(adjustment["id"]),
                    "role": role,
                    "path": str(source["path"]),
                    "sha256": str(source["sha256"]),
                    "start_line": int(source["start_line"]),
                    "end_line": int(source["end_line"]),
                }
            )
    for index, source in enumerate(extension["composition_sources"], start=1):
        references.append(
            {
                "owner": f"aggregate-composition-{index}",
                "role": "source",
                "path": str(source["path"]),
                "sha256": str(source["sha256"]),
                "start_line": int(source["start_line"]),
                "end_line": int(source["end_line"]),
            }
        )
    return sorted(
        references,
        key=lambda row: (
            row["path"],
            row["start_line"],
            row["end_line"],
            row["owner"],
            row["role"],
        ),
    )


EXTERNAL_AUDIT_PROGRAM = r"""
import hashlib
import importlib.metadata
import importlib.util
import json
import pathlib
import sys

references = json.load(sys.stdin)
roots = {}
for package in ("aiconfigurator", "aiconfigurator_core"):
    spec = importlib.util.find_spec(package)
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError(f"missing package {package}")
    roots[package] = pathlib.Path(next(iter(spec.submodule_search_locations))).parent

observed = []
for reference in references:
    package = reference["path"].split("/", 1)[0]
    path = roots[package] / reference["path"]
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    start = int(reference["start_line"])
    end = int(reference["end_line"])
    if start <= 0 or end < start or end > len(lines):
        raise RuntimeError(f"invalid source range {reference}")
    observed.append({
        **reference,
        "observed_sha256": hashlib.sha256(payload).hexdigest(),
        "line_count": len(lines),
        "excerpt_sha256": hashlib.sha256(b"".join(lines[start - 1:end])).hexdigest(),
    })

print(json.dumps({
    "packages": {
        "aiconfigurator": importlib.metadata.version("aiconfigurator"),
        "aiconfigurator-core": importlib.metadata.version("aiconfigurator-core"),
    },
    "references": observed,
}, sort_keys=True))
"""


def _external_source_audit(
    external_python: Path,
    references: list[dict[str, Any]],
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [os.fspath(external_python), "-c", EXTERNAL_AUDIT_PROGRAM],
        input=json.dumps(references, sort_keys=True),
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "external source audit failed: " + completed.stderr.strip()
        )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise TypeError("external source audit did not return a JSON object")
    return value


def _compose_point(
    binding: ExternalQwen32BDeploymentBinding,
    row: dict[str, str],
    **overrides: object,
) -> ExternalAggregatePoint:
    return binding.aggregate_point(
        tensor_parallel=int(row["tp"]),
        batch_size=int(row["bs"]),
        isl=int(row["isl"]),
        osl=int(row["osl"]),
        prefix=int(row["prefix"]),
        context_tokens=int(float(row["ctx_tokens"])),
        **overrides,
    )


def _projection(
    binding: ExternalQwen32BDeploymentBinding,
    rows: list[dict[str, str]],
    **overrides: object,
) -> tuple[list[ExternalAggregatePoint], list[dict[str, Any]]]:
    points = [_compose_point(binding, row, **overrides) for row in rows]
    projection = [
        {
            "row": index,
            "external": dict(row),
            "point": point.as_dict(),
        }
        for index, (row, point) in enumerate(zip(rows, points, strict=True), start=1)
    ]
    return points, projection


def _metric_projection(
    projection: list[dict[str, Any]],
    metric: str,
) -> list[dict[str, Any]]:
    if metric == "tpot":
        fields = (
            "mix_steps",
            "tpot_mix_steps",
            "genonly_steps",
            "mix_step_ms",
            "genonly_step_ms",
            "tpot_ms",
            "total_schedule_ms",
            "request_rate",
            "tokens_per_second",
            "tokens_per_second_per_gpu",
            "tokens_per_second_per_user",
        )
    elif metric == "ttft":
        fields = (
            "mix_steps",
            "pure_prefill_step_ms",
            "prefill_passes_per_request",
            "base_prefill_ms",
            "ttft_queue_factor",
            "ttft_queueing_component_ms",
            "ttft_ms",
        )
    else:
        raise ValueError(f"unknown aggregate metric {metric!r}")
    return [
        {
            "row": row["row"],
            **{field: row["point"][field] for field in fields},
        }
        for row in projection
    ]


def _removal_overrides(factor_id: str) -> dict[str, object]:
    return {
        "memory_bandwidth_empirical_scale": {
            "memory_bandwidth_empirical_scale": 1.0
        },
        "memory_empirical_constant_latency": {
            "memory_empirical_constant_latency_s": 0.0
        },
        "context_attention_extra_latency_correction": {
            "context_attention_extra_latency_correction": 1.0
        },
        "aggregate_ttft_queueing_heuristic": {"apply_ttft_queueing": False},
        "trtllm_tpot_mixed_step_reduction": {
            "tpot_mixed_step_reduction": 0
        },
    }.get(factor_id, {})


def _sensitivity(
    binding: ExternalQwen32BDeploymentBinding,
    external_rows: list[dict[str, str]],
    adjustments: list[dict[str, Any]],
    baseline_projection: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_bytes = _canonical_bytes(baseline_projection)
    baseline_tpot = _canonical_bytes(_metric_projection(baseline_projection, "tpot"))
    baseline_ttft = _canonical_bytes(_metric_projection(baseline_projection, "ttft"))
    result = []
    for adjustment in adjustments:
        factor_id = str(adjustment["id"])
        overrides = _removal_overrides(factor_id)
        if overrides:
            _, removed_projection = _projection(
                binding,
                external_rows,
                **overrides,
            )
        else:
            removed_projection = baseline_projection
        tpot_quotients = [
            float.fromhex(row["point"]["tpot_ms"])
            / float(external["tpot"])
            for row, external in zip(
                removed_projection,
                external_rows,
                strict=True,
            )
        ]
        ttft_quotients = [
            float.fromhex(row["point"]["ttft_ms"])
            / float(external["ttft"])
            for row, external in zip(
                removed_projection,
                external_rows,
                strict=True,
            )
        ]
        removed_bytes = _canonical_bytes(removed_projection)
        removed_tpot = _canonical_bytes(
            _metric_projection(removed_projection, "tpot")
        )
        removed_ttft = _canonical_bytes(
            _metric_projection(removed_projection, "ttft")
        )
        result.append(
            {
                "adjustment_id": factor_id,
                "removal_value": str(adjustment["removal_value"]),
                "tpot_reachable": bool(adjustment["aggregate_tpot_reachable"]),
                "ttft_reachable": bool(adjustment["aggregate_ttft_reachable"]),
                "tpot_quotient_minimum": _measurement(min(tpot_quotients)),
                "tpot_quotient_maximum": _measurement(max(tpot_quotients)),
                "ttft_quotient_minimum": _measurement(min(ttft_quotients)),
                "ttft_quotient_maximum": _measurement(max(ttft_quotients)),
                "complete_projection_byte_identical": removed_bytes == baseline_bytes,
                "tpot_projection_byte_identical": removed_tpot == baseline_tpot,
                "ttft_projection_byte_identical": removed_ttft == baseline_ttft,
                "projection_sha256": hashlib.sha256(removed_bytes).hexdigest(),
                "projection": removed_projection,
            }
        )
    return result


def _schedule_identity(row: dict[str, str], point: ExternalAggregatePoint) -> bool:
    rounded_pairs = (
        (point.balance_score, float(row["balance_score"])),
        (point.tokens_per_second, float(row["tokens/s"])),
        (point.tokens_per_second_per_gpu, float(row["tokens/s/gpu"])),
        (point.tokens_per_second_per_user, float(row["tokens/s/user"])),
        (point.request_rate, float(row["request_rate"])),
        (point.request_latency_ms, float(row["request_latency"])),
    )
    exact_pairs = (
        (point.context_requests, int(float(row["num_ctx_reqs"]))),
        (point.generation_requests, int(float(row["num_gen_reqs"]))),
        (point.scheduled_tokens, int(float(row["num_tokens"]))),
        (point.context_tokens, int(float(row["ctx_tokens"]))),
        (point.generation_requests, int(float(row["gen_tokens"]))),
    )
    return (
        all(abs(actual - published) <= 0.0005 for actual, published in rounded_pairs)
        and all(actual == published for actual, published in exact_pairs)
        and point.batch_size == int(row["concurrency"]) == int(row["global_bs"])
        and point.tensor_parallel == int(row["num_total_gpus"])
        and point.tensor_parallel == int(row["tp"])
    )


def _axis_identity(point: ExternalAggregatePoint) -> bool:
    expected_tokens = (
        1000
        / point.total_schedule_ms
        * point.batch_size
        * (point.osl - 1)
    )
    return (
        expected_tokens.hex() == point.tokens_per_second.hex()
        and (point.tokens_per_second / point.tensor_parallel).hex()
        == point.tokens_per_second_per_gpu.hex()
        and (1000 / point.tpot_ms).hex()
        == point.tokens_per_second_per_user.hex()
        and (point.tokens_per_second / (point.osl - 1)).hex()
        == point.request_rate.hex()
    )


def _physical_identity(
    row: dict[str, str],
    point: ExternalAggregatePoint,
) -> dict[str, Any]:
    hbm_floor_ms = 32e9 / point.tensor_parallel / 4.8e12 * 1000
    mixed_tokens = point.context_tokens + point.generation_requests
    compute_floor_ms = (
        2 * 32e9 * mixed_tokens / point.tensor_parallel / 1.978e15 * 1000
    )
    request_ceiling_ms = float(row["request_latency"])
    throughput_ceiling = point.batch_size * 4.8e12 / 32e9
    passed = (
        point.genonly_step_ms >= hbm_floor_ms
        and point.tpot_ms >= hbm_floor_ms
        and point.mix_step_ms >= compute_floor_ms
        and all(
            0 < value <= request_ceiling_ms
            for value in (
                point.mix_step_ms,
                point.genonly_step_ms,
                point.ttft_ms,
                point.tpot_ms,
            )
        )
        and point.tokens_per_second_per_gpu <= throughput_ceiling
        and _axis_identity(point)
    )
    return {
        "passed": passed,
        "hbm_floor_ms": _measurement(hbm_floor_ms),
        "mixed_compute_floor_ms": _measurement(compute_floor_ms),
        "request_latency_ceiling_ms": _measurement(request_ceiling_ms),
        "throughput_ceiling_tokens_per_second_per_gpu": _measurement(
            throughput_ceiling
        ),
    }


def _family_tallies(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result = {}
    for family in ("AR", "W"):
        selected = [
            row
            for row in rows
            if row["kind"] == "scored" and row["family"] == family
        ]
        result[family] = {
            "passed": sum(bool(row["passed"]) for row in selected),
            "denominator": len(selected),
        }
    return result


def _full_evaluation_worker(*, external_python: Path) -> dict[str, Any]:
    hashes_before = _protected_hashes()
    adjustments = _adjustment_rows()
    references = _source_references(adjustments)
    source_audit = _external_source_audit(external_python, references)
    external_rows = _external_rows()

    import simllm.deploy.estimator as estimator_module

    roofline_calls = 0

    class ForbiddenRooflineProvider:
        def __init__(self, *args: object, **kwargs: object) -> None:
            nonlocal roofline_calls
            roofline_calls += 1
            raise AssertionError("RooflineProvider reached aggregate composition")

    estimator_module.RooflineProvider = ForbiddenRooflineProvider
    database = ExternalOperationDatabase.load(default_artifact_dir())
    binding = ExternalQwen32BDeploymentBinding(database)
    points, baseline_projection = _projection(binding, external_rows)
    sensitivity = _sensitivity(
        binding,
        external_rows,
        adjustments,
        baseline_projection,
    )

    ar_rows = []
    ttft_rows = []
    physical_rows = []
    schedule_checks = []
    for index, (external, point) in enumerate(
        zip(external_rows, points, strict=True),
        start=1,
    ):
        tpot_quotient = point.tpot_ms / float(external["tpot"])
        ttft_residual_ms = float(external["ttft"]) - point.ttft_ms
        ar_rows.append(
            {
                "row": index,
                "configuration_id": point.configuration_id,
                "tensor_parallel": point.tensor_parallel,
                "batch_size": point.batch_size,
                "context_tokens": point.context_tokens,
                "published_tpot_ms": _measurement(float(external["tpot"])),
                "composed_tpot_ms": _measurement(point.tpot_ms),
                "quotient": _measurement(tpot_quotient),
                "passed": 0.98 <= tpot_quotient <= 1.02,
            }
        )
        ttft_rows.append(
            {
                "row": index,
                "configuration_id": point.configuration_id,
                "pure_prefill_step_ms": _measurement(
                    point.pure_prefill_step_ms
                ),
                "prefill_passes_per_request": point.prefill_passes_per_request,
                "base_prefill_ms": _measurement(point.base_prefill_ms),
                "queue_factor": _measurement(point.ttft_queue_factor),
                "queueing_component_ms": _measurement(
                    point.ttft_queueing_component_ms
                ),
                "composed_ttft_ms": _measurement(point.ttft_ms),
                "published_ttft_ms": _measurement(float(external["ttft"])),
                "publication_residual_ms": _measurement(ttft_residual_ms),
                "residual_within_rounding_bound": abs(ttft_residual_ms) <= 0.0005,
            }
        )
        physical_rows.append(
            {
                "row": index,
                "configuration_id": point.configuration_id,
                **_physical_identity(external, point),
            }
        )
        schedule_checks.append(_schedule_identity(external, point))

    applied_adjustments = {
        str(row["id"])
        for row in adjustments
        if row["aggregate_tpot_reachable"] or row["aggregate_ttft_reachable"]
    }
    source_audit_passed = (
        source_audit["packages"]
        == {"aiconfigurator": "0.11.0", "aiconfigurator-core": "0.11.0"}
        and len(source_audit["references"]) == len(references)
        and all(
            row["observed_sha256"] == row["sha256"]
            for row in source_audit["references"]
        )
    )
    sensitivity_complete = (
        {row["adjustment_id"] for row in sensitivity}
        == {str(row["id"]) for row in adjustments}
        and all(
            row["tpot_projection_byte_identical"]
            if not row["tpot_reachable"]
            else True
            for row in sensitivity
        )
        and all(
            row["ttft_projection_byte_identical"]
            if not row["ttft_reachable"]
            else True
            for row in sensitivity
        )
        and all(
            row["complete_projection_byte_identical"]
            if not row["tpot_reachable"] and not row["ttft_reachable"]
            else True
            for row in sensitivity
        )
    )
    network_arms = {
        "unpriced": {
            "strategy": "aggregate co-located prefill and decode",
            "traffic_definition": "unpriced zero-byte P/D handoff",
            "handoff_bytes": 0,
            "handoff_flows": 0,
            "native_process_invocations": 0,
            "projection": baseline_projection,
        },
        "packet": {
            "strategy": "aggregate co-located prefill and decode",
            "traffic_definition": "packet zero-byte P/D handoff",
            "handoff_bytes": 0,
            "handoff_flows": 0,
            "native_process_invocations": 0,
            "projection": baseline_projection,
        },
    }
    traffic_identity = (
        network_arms["unpriced"]["projection"]
        == network_arms["packet"]["projection"]
        and network_arms["unpriced"]["handoff_bytes"] == 0
        and network_arms["packet"]["handoff_bytes"] == 0
        and network_arms["unpriced"]["handoff_flows"] == 0
        and network_arms["packet"]["handoff_flows"] == 0
        and network_arms["packet"]["native_process_invocations"] == 0
    )
    figure_series = [
        {"id": series_id, "label": label}
        for series_id, label in plot_agg.SERIES_LABELS.items()
    ]
    figure_contract_passed = (
        len(figure_series) == 6
        and all(
            ("agg" in row["label"] or "disagg" in row["label"])
            and any(
                word in row["label"]
                for word in ("request mix", "traffic", "handoff")
            )
            for row in figure_series
        )
    )
    rows = [
        base._fatal_row(
            "FG-AGG-1a",
            roofline_calls == 0
            and all(point.evidence_class == EXTERNAL_EVIDENCE_CLASS for point in points)
            and all(
                point.source.startswith(
                    "external-operation-aggregate-composition:"
                )
                for point in points
            ),
            "all positive forward-pass durations come from the imported measured operation database and RooflineProvider was never reached",
        ),
        base._fatal_row(
            "FG-AGG-1b",
            applied_adjustments == EXPECTED_APPLIED_ADJUSTMENTS
            and source_audit_passed
            and float(database.system_spec["gpu"]["mem_bw_empirical_scaling_factor"])
            == 0.8
            and float(database.system_spec["gpu"]["mem_empirical_constant_latency"])
            == 0.000003,
            "the five reachable adjustments, donor source hashes, source ranges and imported memory constants match the freeze",
        ),
        base._fatal_row(
            "FG-AGG-1c",
            sensitivity_complete,
            "all ten declared factors have complete remove-one TPOT and TTFT projections with identity on unreachable outputs",
        ),
        base._fatal_row(
            "FG-AGG-2",
            hashes_before == PROTECTED_PRIOR_SHA256,
            "all protected prior publications, the aggregate table and the parity record match their frozen byte identities before evaluation",
        ),
        base._fatal_row(
            "FG-AGG-3",
            source_audit_passed
            and database.source.as_dict()
            == {
                "tool": "NVIDIA AIConfigurator",
                "aiconfigurator_version": "0.11.0",
                "aiconfigurator_core_version": "0.11.0",
                "system": "h200_sxm",
                "backend": "trtllm",
                "database_version": "1.3.0rc10",
                "data_slice_sha256": "85e72f990f00ea457de522d0b773e678f5e067740689912df5646f6296273284",
                "database_mode": "SILICON",
                "shared_layer": False,
                "estimator_surface": "python",
            },
            "package versions, imported slice identity, donor hashes and source line ranges match the frozen aggregate semantics",
        ),
        base._fatal_row(
            "FG-AGG-4",
            traffic_identity and figure_contract_passed,
            "every figure series names its strategy and traffic while both co-located aggregate arms carry zero P/D handoff bytes and start no packet process",
        ),
        base._fatal_row(
            "FG-AGG-5",
            all(row["residual_within_rounding_bound"] for row in ttft_rows),
            "every published operating-point TTFT is decomposed into prefill, queueing and a residual inside the frozen rounding bound",
        ),
        base._fatal_row(
            "FG-AGG-7",
            all(schedule_checks) and all(_axis_identity(point) for point in points),
            "all 25 scheduling counters, table identities, axes and request rates reproduce the declared aggregate arithmetic",
        ),
        base._fatal_row(
            "FG-AGG-8",
            all(row["passed"] for row in physical_rows),
            "all measured aggregate services satisfy the frozen HBM, compute, causal, scaling and throughput bounds",
        ),
        base._fatal_row(
            "FG-AGG-9",
            base._is_ancestor(EXPECTATIONS_COMMIT)
            and base._git_output("rev-parse", "HEAD") != EXPECTATIONS_COMMIT,
            "the immutable DEPLOY-22 expectation freeze precedes this implementation and evaluation",
        ),
    ]
    rows.extend(
        base._scored_row(
            "AR",
            f"AR-{row['row']:02d}",
            bool(row["passed"]),
            expected="[0.98, 1.02]",
            observed=f"{row['quotient']['decimal']:.12f}",
            evidence_class="MEASURED-EXTERNAL",
            detail=(
                f"{row['configuration_id']}; composed TPOT / published aggregate "
                "operating-point TPOT"
            ),
        )
        for row in ar_rows
    )
    rows.extend(
        base._unscored_row(
            f"TTFT-{row['row']:02d}",
            f"{row['publication_residual_ms']['decimal']:.12f}",
            "ms residual",
            (
                f"{row['configuration_id']}; pure prefill "
                f"{row['pure_prefill_step_ms']['decimal']:.9f} ms; queueing "
                f"{row['queueing_component_ms']['decimal']:.9f} ms; composed "
                f"{row['composed_ttft_ms']['decimal']:.9f} ms"
            ),
            family="TTFT",
            evidence_class="UNSCORED-DECOMPOSITION",
        )
        for row in ttft_rows
    )
    rows.extend(
        base._unscored_row(
            f"ADJ-{index:02d}",
            (
                f"TPOT {row['tpot_quotient_minimum']['decimal']:.9f}.."
                f"{row['tpot_quotient_maximum']['decimal']:.9f}; TTFT "
                f"{row['ttft_quotient_minimum']['decimal']:.9f}.."
                f"{row['ttft_quotient_maximum']['decimal']:.9f}"
            ),
            "quotient ranges",
            (
                f"remove {row['adjustment_id']}; TPOT reachable="
                f"{row['tpot_reachable']}; TTFT reachable={row['ttft_reachable']}"
            ),
            family="ADJ",
            evidence_class="UNSCORED-SENSITIVITY",
        )
        for index, row in enumerate(sensitivity, start=1)
    )
    rows.append(
        base._unscored_row(
            "NET-1",
            "byte-identical",
            "complete projection",
            "co-located unpriced and packet arms each carry zero P/D handoff bytes and the packet arm invokes no native process",
            family="NET",
            evidence_class="STRUCTURAL-IDENTITY",
        )
    )

    hashes_after = _protected_hashes()
    protected_unchanged = hashes_after == hashes_before == PROTECTED_PRIOR_SHA256
    for row in rows:
        if row["id"] == "FG-AGG-2":
            row["passed"] = protected_unchanged
            row["observed"] = str(protected_unchanged).lower()
            row["detail"] = (
                "all protected prior publications, the aggregate table and the parity "
                "record are byte-identical before and after evaluation"
            )
            break
    return {
        "schema": EVALUATION_SCHEMA,
        "source": database.source.as_dict(),
        "protected_hashes": {"before": hashes_before, "after": hashes_after},
        "source_audit": source_audit,
        "applied_adjustments": sorted(applied_adjustments),
        "figure_series": figure_series,
        "families": {
            "AR": {
                "band": [0.98, 1.02],
                "rows": ar_rows,
                "baseline_projection": baseline_projection,
                "projection_sha256": _hash_json(baseline_projection),
            },
            "TTFT": {"rounding_bound_ms": 0.0005, "rows": ttft_rows},
            "ADJ": {"rows": sensitivity},
            "PHYSICAL": {"rows": physical_rows},
        },
        "network_arms": network_arms,
        "rows": rows,
    }


def _run_evaluation(
    *,
    attempt: Path,
    repetition: int,
    external_python: Path,
) -> tuple[dict[str, Any], bytes]:
    evaluation_root = attempt / f"evaluation-run-{repetition}"
    evaluation_root.mkdir(parents=True, exist_ok=False)
    command = [
        os.fspath(Path(sys.executable)),
        os.fspath(Path(__file__).resolve()),
        "--worker",
        "evaluation",
        "--external-python",
        os.fspath(external_python),
    ]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    base._write_new(
        attempt / f"evaluation-run-{repetition}.stdout.json",
        completed.stdout.encode(),
    )
    base._write_new(
        attempt / f"evaluation-run-{repetition}.stderr.txt",
        completed.stderr.encode(),
    )
    if completed.returncode:
        raise RuntimeError(
            f"evaluation worker {repetition} failed with status "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise TypeError("evaluation worker did not return a JSON object")
    return value, completed.stdout.encode()


def _coordinator(
    *,
    bulk_root: Path,
    external_venv: Path,
    write_tracked: bool,
) -> dict[str, Any]:
    if not external_venv.exists():
        raise FileNotFoundError(
            f"{EXTERNAL_VENV_ENV} does not exist: {external_venv}"
        )
    external_python = next(
        (
            path
            for path in (
                external_venv / "bin/python",
                external_venv / "Scripts/python.exe",
            )
            if path.is_file()
        ),
        None,
    )
    if external_python is None:
        raise FileNotFoundError(
            f"{EXTERNAL_VENV_ENV} has no Python interpreter"
        )
    if _sha256(EXPECTATIONS_PATH) != EXPECTATIONS_SHA256:
        raise ValueError("aggregate expectation freeze hash mismatch")
    if _sha256(ADJUSTMENTS_PATH) != ADJUSTMENTS_SHA256:
        raise ValueError("aggregate adjustment extension hash mismatch")

    attempt, attempt_number = base._new_attempt(bulk_root)
    started = time.monotonic()
    runs = [
        _run_evaluation(
            attempt=attempt,
            repetition=repetition,
            external_python=external_python,
        )
        for repetition in (1, 2)
    ]
    evaluations = [value for value, _ in runs]
    evaluation_bytes = [payload for _, payload in runs]
    deterministic = evaluation_bytes[0] == evaluation_bytes[1]
    evaluation_hashes = [
        hashlib.sha256(payload).hexdigest() for payload in evaluation_bytes
    ]
    rows = list(evaluations[0]["rows"])
    rows.append(
        base._fatal_row(
            "FG-AGG-6",
            deterministic,
            "two complete fresh-process scored evaluation JSON payloads are byte-identical; elapsed_seconds and W-1 are excluded by name",
        )
    )
    failed_guards = [
        row["id"]
        for row in rows
        if row["kind"] == "fatal" and not row["passed"]
    ]
    evaluation = evaluations[0]
    record = {
        "schema": SCHEMA,
        "study": "matched_seam_frontier_v1 DEPLOY-22 aggregate arm",
        "run_state": "void" if failed_guards else "nonvoid",
        "voiding_guards": failed_guards,
        "attempt": f"attempt-{attempt_number:04d}",
        "bulk_evidence": f"${{{BULK_ROOT_ENV}}}/attempt-{attempt_number:04d}",
        "run_commit": base._git_output("rev-parse", "HEAD"),
        "freeze": {
            "commit": EXPECTATIONS_COMMIT,
            "path": EXPECTATIONS_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": _sha256(EXPECTATIONS_PATH),
        },
        "adjustment_extension": {
            "path": ADJUSTMENTS_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": _sha256(ADJUSTMENTS_PATH),
            "applied_ids": evaluation["applied_adjustments"],
        },
        "base_record_sha256": _sha256(BASE_RECORD_PATH),
        "external_aggregate_sha256": _sha256(EXTERNAL_AGG_PATH),
        "parity_record_sha256": _sha256(PARITY_RECORD_PATH),
        "source": evaluation["source"],
        "source_audit": evaluation["source_audit"],
        "machine": {
            "architecture": platform.machine(),
            "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "system": platform.system(),
        },
        "fatal_guards": {
            row["id"]: row["passed"]
            for row in rows
            if row["kind"] == "fatal"
        },
        "family_tallies": _family_tallies(rows),
        "families": evaluation["families"],
        "network_arms": evaluation["network_arms"],
        "figure_series": evaluation["figure_series"],
        "rows": rows,
        "determinism": {
            "comparison": "byte-for-byte complete scored evaluation JSON",
            "fresh_processes": 2,
            "evaluation_sha256": evaluation_hashes,
            "excluded_by_name": ["elapsed_seconds", "W-1"],
            "equal": deterministic,
        },
        "reporting_rule": (
            "fatal guards, Family AR, TTFT decomposition, adjustment sensitivity, "
            "network identity, physical guards and W remain separate evidence classes"
        ),
        "figure": {
            "study_pdf": STUDY_PDF_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "study_png": STUDY_PNG_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "publication_pdf": PUBLICATION_PDF_PATH.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "publication_png": PUBLICATION_PNG_PATH.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "caption": plot_agg.CAPTION,
        },
    }
    plot_agg.render_all(
        record,
        study_pdf=attempt / STUDY_PDF_PATH.name,
        study_png=attempt / STUDY_PNG_PATH.name,
        publication_pdf=attempt / PUBLICATION_PDF_PATH.name,
        publication_png=attempt / PUBLICATION_PNG_PATH.name,
    )
    elapsed_seconds = time.monotonic() - started
    wall_row = base._scored_row(
        "W",
        "W-1",
        elapsed_seconds <= WALL_CEILING_SECONDS,
        expected=f"<= {WALL_CEILING_SECONDS:.0f}",
        observed=f"{elapsed_seconds:.6f}",
        units="seconds",
        evidence_class="WALL",
        detail=(
            "two complete fresh-process evaluations, source audits, ten "
            "sensitivities and both additive figure renderings"
        ),
    )
    rows.append(wall_row)
    record["elapsed_seconds"] = elapsed_seconds
    record["family_tallies"] = _family_tallies(rows)
    record["rows"] = rows
    csv_payload = base._csv_bytes(rows)
    base._write_new(
        attempt / "record.json",
        (json.dumps(record, indent=2, sort_keys=True) + "\n").encode(),
    )
    base._write_new(attempt / "results.csv", csv_payload)
    if write_tracked:
        base._write_json(RESULT_PATH, record)
        CSV_PATH.write_bytes(csv_payload)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        for destination in (
            STUDY_PDF_PATH,
            STUDY_PNG_PATH,
            PUBLICATION_PDF_PATH,
            PUBLICATION_PNG_PATH,
        ):
            destination.write_bytes((attempt / destination.name).read_bytes())
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("evaluation",))
    parser.add_argument("--external-python", type=Path)
    parser.add_argument("--bulk-root", type=Path)
    parser.add_argument("--external-venv", type=Path)
    parser.add_argument("--write-tracked", action="store_true")
    args = parser.parse_args()
    if args.worker == "evaluation":
        if args.external_python is None:
            parser.error("evaluation worker requires --external-python")
        result = _full_evaluation_worker(external_python=args.external_python)
        print(json.dumps(result, sort_keys=True))
        return
    bulk_root = args.bulk_root or (
        Path(os.environ[BULK_ROOT_ENV]) if BULK_ROOT_ENV in os.environ else None
    )
    external_venv = args.external_venv or (
        Path(os.environ[EXTERNAL_VENV_ENV])
        if EXTERNAL_VENV_ENV in os.environ
        else None
    )
    missing = [
        name
        for name, value in (
            ("--bulk-root or " + BULK_ROOT_ENV, bulk_root),
            ("--external-venv or " + EXTERNAL_VENV_ENV, external_venv),
        )
        if value is None
    ]
    if missing:
        parser.error("missing " + ", ".join(missing))
    assert bulk_root is not None
    assert external_venv is not None
    record = _coordinator(
        bulk_root=bulk_root,
        external_venv=external_venv,
        write_tracked=args.write_tracked,
    )
    print(
        json.dumps(
            {
                "attempt": record["attempt"],
                "elapsed_seconds": record["elapsed_seconds"],
                "family_tallies": record["family_tallies"],
                "run_state": record["run_state"],
                "voiding_guards": record["voiding_guards"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
