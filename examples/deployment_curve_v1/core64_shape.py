"""Frozen CORE-64 per-rank shape derivation and publication helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any

from core64_field_reader import ACCESS_SCHEMA

EXPECTATIONS_SCHEMA = "simllm-deployment-curve-core64-expectations-v1"
RESULT_SCHEMA = "simllm-deployment-curve-core64-shape-result-v1"
EXPECTED_ACCESS_COUNT = 3
EXPECTED_ACCESS_EVENT_COUNT = 6
STANDARD_CASE_ID = "sglang-decode-ep72-b32-c2000"
ROUTED_FAMILY = "moe_routed_experts"
MTP_FAMILY = "multi_token_prediction_head"


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def validate_expectations(expectations: Mapping[str, Any]) -> None:
    """Require the committed no-fit, no-score, null-direction freeze."""

    if expectations.get("schema") != EXPECTATIONS_SCHEMA:
        raise ValueError("CORE-64 expectations schema differs")
    if expectations.get("task") != "CORE-64":
        raise ValueError("CORE-64 task identity differs")
    if expectations.get("status") != "EXPECTATIONS_ONLY":
        raise ValueError("CORE-64 expectations must precede record access")
    shape = _mapping(
        "per_rank_shape_derivation",
        expectations.get("per_rank_shape_derivation"),
    )
    nodes = _integer("decode_nodes", shape.get("decode_nodes"), minimum=1)
    batch = _integer("batch_per_node", shape.get("batch_per_node"), minimum=1)
    dp = _integer(
        "attention_data_parallel_size",
        shape.get("attention_data_parallel_size"),
        minimum=1,
    )
    requests, remainder = divmod(nodes * batch, dp)
    if remainder or requests != 32 or shape.get("requests_per_rank") != requests:
        raise ValueError("frozen per-rank request arithmetic differs")
    kv = requests * _integer(
        "kv_tokens_per_request",
        shape.get("kv_tokens_per_request"),
        minimum=1,
    )
    if kv != 64_000 or shape.get("kv_tokens_per_rank") != kv:
        raise ValueError("frozen per-rank KV arithmetic differs")
    direction = expectations.get("expected_signed_direction")
    if direction != {
        "corrected_step": "no_change",
        "prediction": "no_change",
        "prediction_movement_tokens_per_second_per_node": "0",
        "signed_residual": "no_change",
        "signed_residual_movement_percentage_points": "0",
    }:
        raise ValueError("CORE-64 null direction differs")
    scope = _mapping("scope_locks", expectations.get("scope_locks"))
    if any(scope.values()):
        raise ValueError("every CORE-64 scope lock must remain false")
    component = _mapping(
        "component_classification",
        expectations.get("component_classification"),
    )
    families = component.get("logical_families")
    if not isinstance(families, list) or len(families) != 14:
        raise ValueError("CORE-64 must freeze all 14 DeepSeek logical families")
    identifiers = [row.get("family_id") for row in families]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("CORE-64 logical family identifiers must be unique")
    if identifiers[-1] != MTP_FAMILY or ROUTED_FAMILY not in identifiers:
        raise ValueError("CORE-64 routed or MTP family placement differs")


def validate_access_events(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate three contemporaneous, partial, held-out-free accesses."""

    if len(events) != EXPECTED_ACCESS_EVENT_COUNT:
        raise ValueError("CORE-64 access event count differs")
    completed = []
    for index, event in enumerate(events, start=1):
        if event.get("event_index") != index:
            raise ValueError("CORE-64 access event indices differ")
        if event.get("schema") != ACCESS_SCHEMA:
            raise ValueError("CORE-64 access schema differs")
        if event.get("held_out_mtp_value_accessed") is not False:
            raise ValueError("CORE-64 access reports held-out MTP exposure")
        if event.get("whole_file_streamed") is not False:
            raise ValueError("CORE-64 access reports a whole-file stream")
    for access_number in range(1, EXPECTED_ACCESS_COUNT + 1):
        begin = events[2 * (access_number - 1)]
        end = events[2 * (access_number - 1) + 1]
        expected_id = f"A{access_number:02d}"
        if begin.get("access_id") != expected_id or end.get("access_id") != expected_id:
            raise ValueError("CORE-64 access identifiers differ")
        if begin.get("event") != "BEGIN" or begin.get("bytes_accessed") != 0:
            raise ValueError("CORE-64 BEGIN event is not contemporaneous")
        if end.get("event") != "END" or end.get("status") != "PASS":
            raise ValueError("CORE-64 END event did not pass")
        consumed = _integer("bytes_accessed", end.get("bytes_accessed"), minimum=1)
        size = _integer(
            "record_size_bytes",
            end.get("record_size_bytes"),
            minimum=2,
        )
        if consumed >= size:
            raise ValueError("CORE-64 selector reached a whole-file stream")
        completed.append(dict(end))
    return {
        "access_count": EXPECTED_ACCESS_COUNT,
        "access_event_count": EXPECTED_ACCESS_EVENT_COUNT,
        "completed_accesses": completed,
        "forbidden_access_ledger": [],
        "held_out_mtp_numeric_values_accessed_or_compared": False,
        "whole_file_streams": 0,
    }


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _manifest_rows(path: Path, expected_columns: int = 2) -> list[tuple[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.rstrip("\n").split("  ", maxsplit=1)
            if len(fields) != expected_columns or not all(fields):
                raise ValueError(f"{path.name}:{line_number}: malformed lock row")
            rows.append((fields[0], fields[1]))
    return rows


def verify_preservation(
    expectations: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    """Check the inherited SHA-256 class plus every prior CORE-63 Git blob."""

    freeze = _mapping("preservation", expectations.get("preservation"))
    inherited_path = repository_root / PurePosixPath(
        str(freeze["inherited_manifest_path"])
    )
    inherited_payload = inherited_path.read_bytes()
    if hashlib.sha256(inherited_payload).hexdigest() != freeze.get(
        "inherited_manifest_sha256"
    ):
        raise ValueError("inherited preservation manifest identity differs")
    inherited_rows = _manifest_rows(inherited_path)
    if len(inherited_rows) != 93:
        raise ValueError("inherited preservation lock count differs")
    checked_paths: set[str] = set()
    for expected_sha256, relative in inherited_rows:
        target = repository_root / PurePosixPath(relative)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError(f"preservation SHA-256 mismatch: {relative}")
        checked_paths.add(relative)

    blob_path = repository_root / PurePosixPath(
        str(freeze["additional_git_blob_manifest_path"])
    )
    blob_rows = _manifest_rows(blob_path)
    if len(blob_rows) != freeze.get("additional_git_blob_lock_count"):
        raise ValueError("additional Git blob lock count differs")
    for expected_blob, relative in blob_rows:
        if relative in checked_paths:
            raise ValueError(f"duplicate preservation path: {relative}")
        target = repository_root / PurePosixPath(relative)
        if _git_blob_sha1(target.read_bytes()) != expected_blob:
            raise ValueError(f"preservation Git blob mismatch: {relative}")
        checked_paths.add(relative)
    expected_count = _integer(
        "expected_total_checked_count",
        freeze.get("expected_total_checked_count"),
        minimum=1,
    )
    if len(checked_paths) != expected_count:
        raise ValueError("total preservation lock count differs")
    return {
        "additional_git_blob_count": len(blob_rows),
        "checked_count": len(checked_paths),
        "hash_verification_decoded_artifact_values": False,
        "inherited_sha256_count": len(inherited_rows),
        "prior_artifacts_mutated": False,
    }


def _logical_family_ledger(
    expectations: Mapping[str, Any],
    standard_case: Mapping[str, Any],
) -> list[dict[str, Any]]:
    component = _mapping(
        "component_classification",
        expectations.get("component_classification"),
    )
    expected_rows = component.get("logical_families")
    if not isinstance(expected_rows, list):
        raise TypeError("logical family freeze must be an array")
    projections = standard_case.get("rank_invariant_family_projections")
    if not isinstance(projections, list):
        raise TypeError("rank-invariant family projection must be an array")
    observed = {}
    for item in projections:
        row = _mapping("rank-invariant family", item)
        family_id = row.get("family_id")
        vector = _mapping("shape_vector", row.get("shape_vector"))
        values = vector.get("values")
        if not isinstance(family_id, str) or not isinstance(values, list):
            raise TypeError("rank-invariant family shape differs")
        observed[family_id] = list(values)

    expected_invariant = {
        row["family_id"]
        for row in expected_rows
        if row["family_id"] not in {ROUTED_FAMILY, MTP_FAMILY}
    }
    if set(observed) != expected_invariant:
        raise ValueError("rank-invariant family set differs from the freeze")
    ledger = []
    for row in expected_rows:
        family_id = row["family_id"]
        expected_values = row["expected_shape_values"]
        if family_id == MTP_FAMILY:
            ledger.append(
                {
                    **row,
                    "observed_shape_values": None,
                    "shape_match": True,
                    "standard_decode_state": "absent_and_unread",
                }
            )
            continue
        observed_values = (
            [standard_case["new_tokens_per_rank"]]
            if family_id == ROUTED_FAMILY
            else observed[family_id]
        )
        if observed_values != expected_values:
            raise ValueError(f"deployment shape differs for {family_id}")
        ledger.append(
            {
                **row,
                "observed_shape_values": observed_values,
                "shape_match": True,
                "standard_decode_state": "present",
            }
        )
    return ledger


def build_result(
    expectations: Mapping[str, Any],
    inputs: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    preservation: Mapping[str, Any],
    *,
    base_commit: str,
    expectations_commit: str,
    runner_commit: str,
) -> dict[str, Any]:
    """Derive the frozen null movement from the selected standard fields."""

    validate_expectations(expectations)
    access = validate_access_events(events)
    if inputs.get("attention_parallelism") != "data-parallel":
        raise ValueError("EP72 attention parallelism differs from data-parallel")
    standard_case = _mapping("standard_case", inputs.get("standard_case"))
    if standard_case.get("case_id") != STANDARD_CASE_ID:
        raise ValueError("selected EP72 case is not standard decode")
    if standard_case.get("phase") != "decode":
        raise ValueError("selected EP72 case phase differs")
    if standard_case.get("new_tokens_per_rank") != 32:
        raise ValueError("selected EP72 request count differs")

    shape = _mapping(
        "per_rank_shape_derivation",
        expectations.get("per_rank_shape_derivation"),
    )
    requests = shape["decode_nodes"] * shape["batch_per_node"] // shape[
        "attention_data_parallel_size"
    ]
    kv_tokens = requests * shape["kv_tokens_per_request"]
    ledger = _logical_family_ledger(expectations, standard_case)

    core63 = _mapping("core63", inputs.get("core63"))
    calibration = _mapping("calibration_only", core63.get("calibration_only"))
    inherited = _mapping(
        "residency_corrected",
        calibration.get("residency_corrected"),
    )
    context = _mapping("calibration_context", expectations.get("calibration_context"))
    prediction = Decimal(str(inherited["prediction_tokens_per_second_per_node"]))
    frozen_prediction = Decimal(str(context["inherited_prediction_tokens_per_second_per_node"]))
    if prediction.quantize(Decimal("0.000001")) != frozen_prediction:
        raise ValueError("inherited CORE-63 prediction differs from the freeze")
    residual = Decimal(str(inherited["signed_residual_percent"]))
    frozen_residual = Decimal(str(context["inherited_signed_residual_percent"]))
    if residual.quantize(Decimal("0.000001")) != frozen_residual:
        raise ValueError("inherited CORE-63 residual differs from the freeze")
    if inherited.get("classification") != "UNDERCORRECTION":
        raise ValueError("inherited CORE-63 classification differs")
    anchor = Decimal(str(context["anchor_tokens_per_second_per_node"]))
    signed_difference = prediction - anchor

    derivation = _mapping("residency_derivation", core63.get("residency_derivation"))
    step = _mapping("step", derivation.get("step"))
    corrected_step = _mapping(
        "residency_corrected_ps",
        step.get("residency_corrected_ps"),
    )
    if corrected_step.get("published_ps_round_half_up") != context.get(
        "inherited_step_ps_round_half_up"
    ):
        raise ValueError("inherited CORE-63 step differs from the freeze")
    physical = _mapping(
        "family_decomposition",
        derivation.get("family_decomposition"),
    )
    physical_rows = physical.get("kernel_classification_ledger")
    if not isinstance(physical_rows, list) or not physical_rows:
        raise ValueError("inherited physical component ledger is empty")
    if physical.get("routed_kernel_row_count") != sum(
        row.get("family") == "routed_expert" for row in physical_rows
    ):
        raise ValueError("inherited routed physical row count differs")

    scope = _mapping("scope", core63.get("scope"))
    if scope != {
        "held_out_mtp_used_in_arithmetic_or_compared": False,
        "parameters_amended_or_refit": False,
        "scored_run_performed": False,
        "zero_free_or_fitted_constants": True,
    }:
        raise ValueError("inherited CORE-63 scope differs")
    if preservation.get("checked_count") != 134:
        raise ValueError("CORE-64 preservation class differs")

    return {
        "access": access,
        "base_commit": base_commit,
        "calibration_only": {
            "anchor_tokens_per_second_per_node": str(anchor),
            "classification": "UNDERCORRECTION",
            "corrected_step_ps_round_half_up": corrected_step[
                "published_ps_round_half_up"
            ],
            "final_prediction_tokens_per_second_per_node": f"{prediction:.6f}",
            "final_signed_residual_percent": f"{residual:.6f}",
            "prediction_movement_tokens_per_second_per_node": "0.000000",
            "signed_difference_from_anchor_tokens_per_second_per_node": (
                f"{signed_difference:.6f}"
            ),
            "signed_residual_movement_percentage_points": "0.000000",
        },
        "component_classification": {
            "fixed_term": expectations["component_classification"]["fixed_term"],
            "logical_family_ledger": ledger,
            "physical_kernel_rule": expectations["component_classification"][
                "physical_kernel_rule"
            ],
            "retained_physical_decomposition": dict(physical),
            "semantic_physical_binding_state": (
                "ABSENT_TOTAL_BINDING_NON_NUMERIC_FOR_NULL_SCALE"
            ),
        },
        "expectations_commit": expectations_commit,
        "per_rank_shape": {
            "attention_parallelism": inputs["attention_parallelism"],
            "batch_per_node": shape["batch_per_node"],
            "decode_nodes": shape["decode_nodes"],
            "global_requests": shape["decode_nodes"] * shape["batch_per_node"],
            "kv_tokens_per_rank": kv_tokens,
            "kv_tokens_per_request": shape["kv_tokens_per_request"],
            "requests_per_rank": requests,
            "requests_per_rank_formula": shape["requests_per_rank_formula"],
            "shape_mismatch_count": 0,
        },
        "preservation_lock": dict(preservation),
        "registry_disposition": {
            "core64": "REMAINS_OPEN_LITERAL_GAP_UNRESOLVED_BY_NULL_MOVEMENT",
            "exact_remainder_id": "CORE-65",
            "reserved_id_free_on_base_main": True,
        },
        "runner_commit": runner_commit,
        "schema": RESULT_SCHEMA,
        "scope": {
            "calibration_only": True,
            "decode_overlap_term_added": False,
            "held_out_mtp_used_in_arithmetic_or_compared": False,
            "model_weights_downloaded": False,
            "parameters_amended_or_refit": False,
            "scored_run_performed": False,
            "web_pages_fetched": False,
            "zero_free_or_fitted_constants": True,
        },
        "status": "PASS_NULL_SHAPE_MOVEMENT_EXACT_REMAINDER",
        "task": "CORE-64",
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render the null result with shape mismatches first."""

    shape = result["per_rank_shape"]
    calibration = result["calibration_only"]
    classification = result["component_classification"]
    families = classification["logical_family_ledger"]
    preservation = result["preservation_lock"]
    access = result["access"]
    return f"""# CORE-64 attention, MLA and shared-expert decode shape result

Status: **{result['status']}**. The preregistered component-backed shape test
produces an honest null movement and retains the full standard-decode
undercorrection.

## Enumerated shape mismatches

1. **MLA and attention: none.** The disclosed nine-node, eight-rank-per-node
   standard decode layout has attention DP72. Its 2,304 global requests divide
   to `9 * 256 / 72 = {shape['requests_per_rank']}` requests per rank. KV 2,000
   therefore produces {shape['kv_tokens_per_rank']:,} aggregate KV-token
   references per rank. This exactly matches the TP1 `b32/c2000` capture, so
   every MLA and attention scale remains one.
2. **Shared expert and dense feed-forward: none.** Each rank runs both paths
   for its own 32-request local stream. Decode dense DP1 is a singleton group;
   it does not replicate the node's 256 requests onto each rank. Both scales
   remain one.
3. **Router and LM head: none.** Both see the same 32 local tokens and retain
   scale one.
4. **Routed experts: no new CORE-64 mismatch.** CORE-63's exact `1/9`
   `fused_moe_kernel` residency scale remains unchanged.
5. **Fixed and other retained physical rows: none.** The 489 ps fixed term is
   kept once and every nonmatching noncollective row remains at scale one.
6. **Physical launch identity: unresolved attribution, not a numeric shape
   mismatch.** No total EP72 physical-kernel-to-logical-family binding exists.
   Because all non-routed logical scales are one, this cannot change the null
   arithmetic, but it prevents inventing a finer semantic timing attribution.
7. **MTP: absent and unread.** It is not part of standard decode.

These derivations use the [published deployment disclosure](../../docs/papers/deepseek-deployment-disclosures.md),
the [frozen SGLang DP72 and dense-DP1 arrangement](../sglang_pd_session_v1/expectations.md),
and the [framework-neutral DeepSeek family projection](../model_extraction_deepseek_v3_v1/RESULTS.md).

## Derived correction and signed movement

All {len(families) - 1} standard-decode logical families were enumerated; the
MTP family is separately absent. Shape mismatches: **0**. The inherited
CORE-63 step therefore stays **{calibration['corrected_step_ps_round_half_up']:,}
ps**.

```text
CORE-64 prediction movement = 0.000000 tokens/s/node
final standard-decode prediction = {calibration['final_prediction_tokens_per_second_per_node']} tokens/s/node
calibration anchor = {calibration['anchor_tokens_per_second_per_node']} tokens/s/node
signed difference = {calibration['signed_difference_from_anchor_tokens_per_second_per_node']} tokens/s/node
signed residual movement = 0.000000 percentage points
final signed residual = {calibration['final_signed_residual_percent']} percent
```

The result remains an **UNDERCORRECTION**. No constant was fitted and no
decode overlap term was introduced.

## Component locality and classification

MLA, compressed-KV read, dense early MLP, router, shared expert and LM head
all retain their exact rank-local standard-decode shape. The committed JSON
companion carries all 14 logical family rows and the complete inherited
physical classification ledger. Physical names are not guessed into semantic
families: `fused_moe_kernel` remains routed and every nonmatching row remains
retained.

## Access and preservation

All {access['access_count']} field-addressed accesses have contemporaneous
BEGIN and END events, and every completed byte count is below its source size.
The standard-case selector returned before the forbidden MTP case. Whole-file
streams: **0**. The forbidden-access ledger is exactly `[]`.

All {preservation['checked_count']} prior artifacts are byte-identical:
{preservation['inherited_sha256_count']} inherited SHA-256 locks plus
{preservation['additional_git_blob_count']} merged CORE-63 Git blob locks.
Hash verification decoded no artifact values.

## Registry disposition

CORE-64's literal gap-resolution clause is not satisfied by a zero movement,
so CORE-64 remains open. CORE-65 receives the exact remaining physical
attribution gap: **{calibration['signed_difference_from_anchor_tokens_per_second_per_node']}
tokens/s/node**, or **{calibration['final_signed_residual_percent']} percent**,
after the complete rank-local shape match. CORE-65 was free on base main.
"""


def write_new_json(path: Path, value: object) -> None:
    """Write one new JSON artifact with pinned POSIX newlines."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_new_text(path: Path, value: str) -> None:
    """Write one new text artifact with pinned POSIX newlines."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


__all__ = [
    "build_result",
    "render_markdown",
    "validate_access_events",
    "validate_expectations",
    "verify_preservation",
    "write_new_json",
    "write_new_text",
]
