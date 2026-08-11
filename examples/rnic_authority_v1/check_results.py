"""Validate the frozen CORE-21 and BACK-31 authority comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simllm.backends.rnic_records import (
    BypassArtifacts,
    canonical_bypass_parameters,
    compare_bypass_artifacts,
)
from simllm.core import (
    ControlMode,
    ControlWork,
    execution_graph_from_json,
    execution_graph_to_json,
    execution_result_from_json,
    step_record_from_json,
    step_record_to_json,
)

OBSERVATION_DIAGNOSTIC = "authority observations do not exist"
EXPECTATION_SCHEMA = "simllm-rnic-authority-expectations-v1"
OBSERVATION_SCHEMA = "simllm-rnic-authority-observations-v1"
RESULT_SCHEMA = "simllm-rnic-authority-results-v1"
NEGATIVE_SCHEMA = "simllm-rnic-authority-negative-evidence-v1"
SIMLLM_SOURCE_COMMIT = "90ada43070adb3b1e624b6819aff34d8620e8571"
HTSIM_SOURCE_COMMIT = "4885c647eecdfdf81479d1df052223c016ad086b"

TOP_KEY_ORDER = (
    "schema",
    "simllm_source_commit",
    "htsim_source_commit",
    "native_observations_sha256",
    "preflight",
    "cells",
)
TOP_KEYS = set(TOP_KEY_ORDER)
CELL_KEY_ORDER = (
    "link_rate_gbps",
    "doorbell_service_ps",
    "execution_graph",
    "step_record",
    "history_seed",
    "bypass",
    "structural",
    "transaction_failure",
)
CELL_KEYS = set(CELL_KEY_ORDER)
MODE_KEY_ORDER = (
    "hardware_mode",
    "authority",
    "execution_result",
    "runtime_report",
    "step_result",
    "request_summary",
    "wqe_count",
    "bypass_artifacts",
)
MODE_KEYS = set(MODE_KEY_ORDER)
PREFLIGHT_KEYS = {"profile", "manifest", "flow_count", "quiescent"}
HISTORY_KEYS = {
    "execution_graph",
    "step_record",
    "execution_result",
    "runtime_report",
    "step_result",
}
STEP_RESULT_KEYS = {
    "step_index",
    "step_latency_ps",
    "completed_at_ps",
    "request_metrics",
    "additive_visit_totals",
}
METRIC_KEYS = {
    "request_id",
    "phase",
    "token_index",
    "completed_at_ps",
    "latency_ps",
    "ttft_ps",
    "tpot_ps",
    "attribution",
    "additive_visit_totals",
}
ATTRIBUTION_KEYS = {
    "queue_ps",
    "kv_ps",
    "kernel_ps",
    "dma_ps",
    "collective_ps",
    "nic_ps",
    "control_ps",
}
ADDITIVE_KEYS = {"queue_wait_ps", "service_ps", "visibility_ps", "visit_count"}
REPORT_KEYS = {
    "execution_id",
    "authority",
    "operations",
    "visits",
    "wqes",
    "sum_visit_wait_ps",
    "critical_path_queue_ps",
    "realized_critical_path_operation_ids",
    "random_draw_count",
}
REPORT_OPERATION_KEYS = {
    "operation_id",
    "completed_at_ps",
    "critical_predecessor_id",
    "attribution",
}
VISIT_KEYS = {
    "execution_id",
    "operation_id",
    "stage",
    "resource_kind",
    "resource_id",
    "subject_object_id",
    "submitted_at_ps",
    "eligible_at_ps",
    "started_at_ps",
    "finished_at_ps",
    "completed_at_ps",
    "service_bytes",
}
WQE_KEYS = {
    "authority",
    "execution_id",
    "operation_id",
    "wqe_id",
    "native_wqe_id",
    "sq_id",
    "rq_id",
    "cq_id",
    "qp_id",
    "rnic_id",
    "source_rank",
    "destination_rank",
    "payload_bytes",
    "goal_tag",
    "extent_index",
    "sq_post_sequence",
    "cq_post_sequence",
    "submitted_at_ps",
    "eligible_at_ps",
    "started_at_ps",
    "finished_at_ps",
    "completed_at_ps",
    "channel_id",
    "nccl_command_id",
    "doorbell_started_at_ps",
    "doorbell_completed_at_ps",
    "network_eligible_at_ps",
    "network_started_at_ps",
    "network_finished_at_ps",
}
TRANSACTION_KEYS = {
    "exception_type",
    "exception_message",
    "before",
    "after",
    "retry",
    "runtime_last_report_is_none",
    "bypass_ledger_is_none",
    "authority",
}
COUNTER_KEYS = {"committed_transactions", "committed_wqes"}
BYPASS_ARTIFACT_KEYS = {
    "goal_text_hex",
    "goal_binary_hex",
    "topology_hex",
    "profile",
    "seed",
    "baseline_parameters",
    "completion_csv_hex",
    "canonical_completion_hex",
    "step_results_hex",
    "replay_summary_hex",
}
NEGATIVE_KEYS = {
    "schema",
    "htsim_source_commit",
    "cache_value",
    "negative_main_sha256",
    "positive_assets_before",
    "positive_assets_after",
    "positive_assets_preserved",
    "bypass_comparison",
    "negative_tier_a_target_absent",
    "producer_returncode",
    "producer_stderr",
    "checker_returncode",
    "checker_stderr",
    "forbidden_outputs_absent",
}
INPUT_ARTIFACT_NAMES = (
    "goal_text",
    "goal_binary",
    "topology",
    "profile",
    "seed",
    "baseline_parameters",
)
BEHAVIORAL_ARTIFACT_NAMES = (
    "completion_csv",
    "canonical_completion",
    "step_results",
    "replay_summary",
)
POSITIVE_ASSET_NAMES = {
    "tier_a_producer_sha256",
    "htsim_rnic_sha256",
    "txt2bin_sha256",
    "raw_observations_sha256",
    "native_observations_sha256",
    "bypass_goal_text_sha256",
    "bypass_goal_binary_sha256",
    "bypass_topology_sha256",
    "bypass_completion_csv_sha256",
    "bypass_canonical_completion_sha256",
    "bypass_step_results_sha256",
    "bypass_replay_summary_sha256",
}
FATAL_UNSCORED = (
    "d0_authority_identity",
    "identical_graph_and_step_input",
    "deployed_reducer_on_both_modes",
    "bypass_ledger_is_sole_bypass_authority",
    "native_session_is_sole_structural_authority",
    "transaction_failure_is_atomic",
    "transaction_retry_commits_once",
    "bypass_artifacts_preserved",
    "positive_assets_preserved",
    "negative_cache_is_off",
    "negative_main_binary_exists",
    "negative_tier_a_target_absent",
    "negative_producer_rejected_native_absence",
    "negative_checker_rejected_missing_observations",
    "negative_published_no_accepted_result",
)
EVALUATION_ORDER = (
    "raw_shape_and_types",
    "signed_authority_delta",
    "inverse_rate_delta",
    "fatal_exact_oracles_and_invariants",
    "negative_link_control",
)


class AuthorityCheckError(RuntimeError):
    """An observation or evidence row violates the frozen contract."""

    def __init__(self, message: str, *, evaluation_order: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.evaluation_order = evaluation_order


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityCheckError(message)


def _object(value: Any, name: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{name} must be an object")
    return value


def _array(value: Any, name: str) -> list[Any]:
    _require(type(value) is list, f"{name} must be an array")
    return value


def _integer(value: Any, name: str, *, nonnegative: bool = True) -> int:
    _require(type(value) is int, f"{name} must be an integer")
    if nonnegative:
        _require(value >= 0, f"{name} must be nonnegative")
    return value


def _boolean(value: Any, name: str) -> bool:
    _require(type(value) is bool, f"{name} must be a boolean")
    return value


def _text(value: Any, name: str) -> str:
    _require(type(value) is str and bool(value.strip()), f"{name} must be text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    _require(
        actual == expected,
        f"{name} keys differ: missing={sorted(expected - actual)}, "
        f"unexpected={sorted(actual - expected)}",
    )


def _fraction(value: Any, name: str, *, optional: bool = False) -> Fraction | None:
    if value is None and optional:
        return None
    row = _object(value, name)
    _exact_keys(row, {"numerator", "denominator"}, name)
    numerator = _integer(row["numerator"], f"{name}.numerator")
    denominator = _integer(row["denominator"], f"{name}.denominator")
    _require(denominator > 0, f"{name}.denominator must be positive")
    return Fraction(numerator, denominator)


def _hex_digest(value: Any, name: str) -> str:
    digest = _text(value, name)
    _require(
        len(digest) == 64
        and digest == digest.lower()
        and all(character in "0123456789abcdef" for character in digest),
        f"{name} must be a lower-case SHA-256 digest",
    )
    return digest


def _git_commit(value: Any, name: str) -> str:
    commit = _text(value, name)
    _require(
        len(commit) == 40
        and commit == commit.lower()
        and all(character in "0123456789abcdef" for character in commit),
        f"{name} must be a full lower-case Git commit",
    )
    return commit


def _hex_bytes(value: Any, name: str, *, allow_empty: bool = False) -> bytes:
    _require(type(value) is str, f"{name} must be hexadecimal text")
    _require(allow_empty or bool(value), f"{name} must not be empty")
    _require(value == value.lower() and len(value) % 2 == 0, f"{name} is not canonical hex")
    try:
        return bytes.fromhex(value)
    except ValueError as error:
        raise AuthorityCheckError(f"{name} is not hexadecimal") from error


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _pretty_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuthorityCheckError(f"cannot read {name} {path}: {error}") from error
    return _object(value, name)


def _validate_expectations(value: dict[str, Any]) -> None:
    _exact_keys(
        value,
        {
            "schema",
            "observation_schema",
            "result_schema",
            "simllm_base_commit",
            "htsim_source_commit",
            "fixed_graph",
            "history_seed",
            "sweep",
            "closed_forms",
            "signed_authority_relation",
            "inverse_rate_relation",
            "fatal_unscored",
            "negative_control",
            "builds",
            "producer_argument_names",
            "checker_argument_names",
            "raw_top_keys",
            "raw_cell_keys",
            "raw_mode_keys",
            "behavioral_family_instances",
        },
        "authority expectations",
    )
    expected_graph = {
        "execution_id": "core21-fixed-contended-graph",
        "step_index": 1,
        "released_at_ps": 17000,
        "source_rank": 0,
        "destination_ranks": [8, 16],
        "payload_bytes": 4096,
        "wqe_count": 2,
        "logical_queue": "core21-control",
        "operation_ids": ["core21-contended-send"],
        "completion_operation_ids": ["core21-contended-send"],
        "request_ids": ["core21-prefill", "core21-decode"],
    }
    expected_seed = {
        "first_release_ps": 7000,
        "duration_ps": 10000,
        "request_id": "core21-decode",
        "work_kind": "compute",
        "scored": False,
    }
    _require(value.get("schema") == EXPECTATION_SCHEMA, "expectation schema drifted")
    _require(value.get("observation_schema") == OBSERVATION_SCHEMA, "observation schema drifted")
    _require(value.get("result_schema") == RESULT_SCHEMA, "result schema drifted")
    _require(value.get("simllm_base_commit") == SIMLLM_SOURCE_COMMIT, "SimLLM anchor drifted")
    _require(value.get("htsim_source_commit") == HTSIM_SOURCE_COMMIT, "htsim anchor drifted")
    _require(value.get("fixed_graph") == expected_graph, "fixed graph registry drifted")
    _require(value.get("history_seed") == expected_seed, "history seed registry drifted")
    _require(
        value.get("sweep") == {"link_rate_gbps": [200, 400], "doorbell_service_ps": [0, 1000]},
        "authority sweep drifted",
    )
    _require(
        value.get("closed_forms")
        == {
            "wire_service_ps": "payload_bytes * 8 * 1000 / link_rate_gbps",
            "bypass_jct_ps": "2 * wire_service_ps",
            "structural_jct_ps": "doorbell_service_ps + 2 * wire_service_ps",
            "rate_200_minus_400_ps": 163840,
        },
        "closed-form registry drifted",
    )
    _require(
        value.get("signed_authority_relation")
        == {
            "orientation": "structural_minus_bypass",
            "doorbell_service_ps": 1000,
            "metrics": ["jct_ps", "prefill_ttft_ps", "decode_tpot_ps"],
            "direction": "positive",
            "lower_bound_ps": 1000,
            "upper_bound_ps": 1000,
            "instances": 6,
        },
        "signed authority relation drifted",
    )
    _require(
        value.get("inverse_rate_relation")
        == {
            "orientation": "rate_200_minus_rate_400",
            "metrics": ["jct_ps", "prefill_ttft_ps", "decode_tpot_ps"],
            "lower_bound_ps": 163840,
            "upper_bound_ps": 163840,
            "instances": 12,
        },
        "inverse-rate relation drifted",
    )
    _require(value.get("raw_top_keys") == list(TOP_KEY_ORDER), "raw top-key registry drifted")
    _require(value.get("raw_cell_keys") == list(CELL_KEY_ORDER), "raw cell-key registry drifted")
    _require(value.get("raw_mode_keys") == list(MODE_KEY_ORDER), "raw mode-key registry drifted")
    _require(
        value.get("behavioral_family_instances")
        == {"signed_authority_delta": 6, "inverse_rate_delta": 12},
        "behavioral family registry drifted",
    )
    _require(
        value.get("fatal_unscored") == list(FATAL_UNSCORED),
        "fatal-unscored registry drifted",
    )
    _require(
        value.get("negative_control")
        == {
            "profile": "rnic-nn",
            "expected_producer_exit": 2,
            "expected_checker_exit": 2,
            "producer_diagnostic": "composed preflight lacks structural native authority",
            "checker_diagnostic": OBSERVATION_DIAGNOSTIC,
            "forbidden_outputs": [
                "raw_observations.json",
                "raw_observations.json.tmp",
                "results.json",
                "results.json.tmp",
            ],
        },
        "negative-control registry drifted",
    )
    _require(
        value.get("builds")
        == {
            "positive": {
                "HTSIM_ENABLE_SIMLLM_RNIC": "ON",
                "HTSIM_CREATE_SOURCE_SYMLINKS": "OFF",
                "targets": ["htsim_rnic_tier_a", "htsim_rnic", "txt2bin"],
            },
            "negative": {
                "HTSIM_ENABLE_SIMLLM_RNIC": "OFF",
                "HTSIM_CREATE_SOURCE_SYMLINKS": "OFF",
                "targets": ["htsim_rnic", "txt2bin"],
                "absent_target": "htsim_rnic_tier_a",
            },
        },
        "build registry drifted",
    )
    _require(
        value.get("producer_argument_names")
        == [
            "--expectations",
            "--observations",
            "--tier-a-producer",
            "--htsim-rnic",
            "--txt2bin",
        ],
        "producer argument registry drifted",
    )
    _require(
        value.get("checker_argument_names")
        == ["--expectations", "--observations", "--negative-evidence", "--results"],
        "checker argument registry drifted",
    )


def _validate_additive_shape(value: Any, name: str) -> dict[str, int]:
    row = _object(value, name)
    _exact_keys(row, ADDITIVE_KEYS, name)
    return {key: _integer(row[key], f"{name}.{key}") for key in row}


def _validate_attribution_shape(value: Any, name: str) -> dict[str, int]:
    row = _object(value, name)
    _exact_keys(row, ATTRIBUTION_KEYS, name)
    return {key: _integer(row[key], f"{name}.{key}") for key in row}


def _parse_step_result_shape(
    value: Any, name: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    row = _object(value, name)
    _exact_keys(row, STEP_RESULT_KEYS, name)
    for key in ("step_index", "step_latency_ps", "completed_at_ps"):
        _integer(row[key], f"{name}.{key}")
    metrics: dict[str, dict[str, Any]] = {}
    for index, raw_metric in enumerate(_array(row["request_metrics"], f"{name}.request_metrics")):
        metric_name = f"{name}.request_metrics[{index}]"
        metric = _object(raw_metric, metric_name)
        _exact_keys(metric, METRIC_KEYS, metric_name)
        request_id = _text(metric["request_id"], f"{metric_name}.request_id")
        _text(metric["phase"], f"{metric_name}.phase")
        for key in ("token_index", "completed_at_ps", "latency_ps", "ttft_ps"):
            _integer(metric[key], f"{metric_name}.{key}")
        _fraction(metric["tpot_ps"], f"{metric_name}.tpot_ps", optional=True)
        _validate_attribution_shape(metric["attribution"], f"{metric_name}.attribution")
        _validate_additive_shape(
            metric["additive_visit_totals"], f"{metric_name}.additive_visit_totals"
        )
        _require(request_id not in metrics, f"{name} repeats request {request_id!r}")
        metrics[request_id] = metric
    _validate_additive_shape(row["additive_visit_totals"], f"{name}.additive_visit_totals")
    return row, metrics


def _validate_execution_result_shape(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    _exact_keys(
        row, {"schema", "execution_id", "completed_at_ps", "events", "quiesced_at_ps"}, name
    )
    _text(row["schema"], f"{name}.schema")
    _text(row["execution_id"], f"{name}.execution_id")
    _integer(row["completed_at_ps"], f"{name}.completed_at_ps")
    if row["quiesced_at_ps"] is not None:
        _integer(row["quiesced_at_ps"], f"{name}.quiesced_at_ps")
    event_keys = {
        "schema",
        "execution_id",
        "operation_id",
        "phase",
        "timestamp_ps",
        "resource",
        "completed_bytes",
        "subject_object_id",
    }
    for index, raw_event in enumerate(_array(row["events"], f"{name}.events")):
        event_name = f"{name}.events[{index}]"
        event = _object(raw_event, event_name)
        _exact_keys(event, event_keys, event_name)
        for key in ("schema", "execution_id", "operation_id", "phase"):
            _text(event[key], f"{event_name}.{key}")
        _integer(event["timestamp_ps"], f"{event_name}.timestamp_ps")
        if event["resource"] is not None:
            resource = _object(event["resource"], f"{event_name}.resource")
            _exact_keys(resource, {"kind", "resource_id"}, f"{event_name}.resource")
            _text(resource["kind"], f"{event_name}.resource.kind")
            _text(resource["resource_id"], f"{event_name}.resource.resource_id")
        if event["completed_bytes"] is not None:
            _integer(event["completed_bytes"], f"{event_name}.completed_bytes")
        _optional_text(event["subject_object_id"], f"{event_name}.subject_object_id")
    return row


def _validate_runtime_report_shape(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    _exact_keys(row, REPORT_KEYS, name)
    _text(row["execution_id"], f"{name}.execution_id")
    _text(row["authority"], f"{name}.authority")
    for index, raw_operation in enumerate(_array(row["operations"], f"{name}.operations")):
        item_name = f"{name}.operations[{index}]"
        item = _object(raw_operation, item_name)
        _exact_keys(item, REPORT_OPERATION_KEYS, item_name)
        _text(item["operation_id"], f"{item_name}.operation_id")
        _integer(item["completed_at_ps"], f"{item_name}.completed_at_ps")
        _optional_text(item["critical_predecessor_id"], f"{item_name}.critical_predecessor_id")
        _validate_attribution_shape(item["attribution"], f"{item_name}.attribution")
    for index, raw_visit in enumerate(_array(row["visits"], f"{name}.visits")):
        item_name = f"{name}.visits[{index}]"
        item = _object(raw_visit, item_name)
        _exact_keys(item, VISIT_KEYS, item_name)
        for key in ("execution_id", "operation_id", "stage", "resource_kind", "resource_id"):
            _text(item[key], f"{item_name}.{key}")
        _optional_text(item["subject_object_id"], f"{item_name}.subject_object_id")
        for key in (
            "submitted_at_ps",
            "eligible_at_ps",
            "started_at_ps",
            "finished_at_ps",
            "completed_at_ps",
            "service_bytes",
        ):
            _integer(item[key], f"{item_name}.{key}")
    optional_wqe_times = {
        "doorbell_started_at_ps",
        "doorbell_completed_at_ps",
        "network_eligible_at_ps",
        "network_started_at_ps",
        "network_finished_at_ps",
    }
    text_wqe_fields = {
        "authority",
        "execution_id",
        "operation_id",
        "wqe_id",
        "native_wqe_id",
        "sq_id",
        "rq_id",
        "cq_id",
        "qp_id",
        "rnic_id",
        "channel_id",
    }
    integer_wqe_fields = WQE_KEYS - text_wqe_fields - optional_wqe_times - {"nccl_command_id"}
    for index, raw_wqe in enumerate(_array(row["wqes"], f"{name}.wqes")):
        item_name = f"{name}.wqes[{index}]"
        item = _object(raw_wqe, item_name)
        _exact_keys(item, WQE_KEYS, item_name)
        for key in text_wqe_fields:
            _text(item[key], f"{item_name}.{key}")
        _optional_text(item["nccl_command_id"], f"{item_name}.nccl_command_id")
        for key in integer_wqe_fields:
            _integer(item[key], f"{item_name}.{key}")
        for key in optional_wqe_times:
            if item[key] is not None:
                _integer(item[key], f"{item_name}.{key}")
    for key in ("sum_visit_wait_ps", "critical_path_queue_ps", "random_draw_count"):
        _integer(row[key], f"{name}.{key}")
    for index, value_item in enumerate(
        _array(
            row["realized_critical_path_operation_ids"],
            f"{name}.realized_critical_path_operation_ids",
        )
    ):
        _text(value_item, f"{name}.realized_critical_path_operation_ids[{index}]")
    return row


def _artifact_shape(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    _exact_keys(row, BYPASS_ARTIFACT_KEYS, name)
    for key in (
        "goal_text_hex",
        "goal_binary_hex",
        "completion_csv_hex",
        "canonical_completion_hex",
        "step_results_hex",
        "replay_summary_hex",
    ):
        _hex_bytes(row[key], f"{name}.{key}")
    _hex_bytes(row["topology_hex"], f"{name}.topology_hex", allow_empty=True)
    _text(row["profile"], f"{name}.profile")
    _integer(row["seed"], f"{name}.seed")
    for index, raw_pair in enumerate(
        _array(row["baseline_parameters"], f"{name}.baseline_parameters")
    ):
        pair = _array(raw_pair, f"{name}.baseline_parameters[{index}]")
        _require(len(pair) == 2, f"{name}.baseline_parameters[{index}] must be a pair")
        _text(pair[0], f"{name}.baseline_parameters[{index}][0]")
        _text(pair[1], f"{name}.baseline_parameters[{index}][1]")
    return row


@dataclass(frozen=True)
class ModeView:
    raw: dict[str, Any]
    step_result: dict[str, Any]
    metrics: dict[str, dict[str, Any]]

    def scored_metrics(self) -> dict[str, Fraction]:
        prefill = self.metrics["core21-prefill"]
        decode = self.metrics["core21-decode"]
        tpot = _fraction(decode["tpot_ps"], "decode_tpot_ps")
        assert tpot is not None
        return {
            "jct_ps": Fraction(self.step_result["step_latency_ps"], 1),
            "prefill_ttft_ps": Fraction(prefill["ttft_ps"], 1),
            "decode_tpot_ps": tpot,
        }


@dataclass(frozen=True)
class CellView:
    raw: dict[str, Any]
    rate: int
    doorbell: int
    bypass: ModeView
    structural: ModeView


def _parse_mode_shape(value: Any, name: str) -> ModeView:
    row = _object(value, name)
    _exact_keys(row, MODE_KEYS, name)
    _text(row["hardware_mode"], f"{name}.hardware_mode")
    _text(row["authority"], f"{name}.authority")
    _integer(row["wqe_count"], f"{name}.wqe_count")
    _validate_execution_result_shape(row["execution_result"], f"{name}.execution_result")
    _validate_runtime_report_shape(row["runtime_report"], f"{name}.runtime_report")
    step_result, metrics = _parse_step_result_shape(row["step_result"], f"{name}.step_result")
    _require(
        set(metrics) == {"core21-prefill", "core21-decode"},
        f"{name}.step_result request inventory drifted",
    )
    summary = _object(row["request_summary"], f"{name}.request_summary")
    _exact_keys(summary, {"jct_ps", "prefill_ttft_ps", "decode_tpot_ps"}, f"{name}.request_summary")
    _integer(summary["jct_ps"], f"{name}.request_summary.jct_ps")
    _integer(summary["prefill_ttft_ps"], f"{name}.request_summary.prefill_ttft_ps")
    _fraction(summary["decode_tpot_ps"], f"{name}.request_summary.decode_tpot_ps")
    if row["bypass_artifacts"] is not None:
        _artifact_shape(row["bypass_artifacts"], f"{name}.bypass_artifacts")
    return ModeView(row, step_result, metrics)


def _parse_history_shape(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    _exact_keys(row, HISTORY_KEYS, name)
    _object(row["execution_graph"], f"{name}.execution_graph")
    _object(row["step_record"], f"{name}.step_record")
    _validate_execution_result_shape(row["execution_result"], f"{name}.execution_result")
    _validate_runtime_report_shape(row["runtime_report"], f"{name}.runtime_report")
    _parse_step_result_shape(row["step_result"], f"{name}.step_result")
    return row


def _parse_transaction_shape(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    _exact_keys(row, TRANSACTION_KEYS, name)
    _text(row["exception_type"], f"{name}.exception_type")
    _text(row["exception_message"], f"{name}.exception_message")
    for side in ("before", "after", "retry"):
        counters = _object(row[side], f"{name}.{side}")
        _exact_keys(counters, COUNTER_KEYS, f"{name}.{side}")
        for key in COUNTER_KEYS:
            _integer(counters[key], f"{name}.{side}.{key}")
    _boolean(row["runtime_last_report_is_none"], f"{name}.runtime_last_report_is_none")
    _boolean(row["bypass_ledger_is_none"], f"{name}.bypass_ledger_is_none")
    _text(row["authority"], f"{name}.authority")
    return row


def _parse_raw_shape(observations: dict[str, Any]) -> dict[tuple[int, int], CellView]:
    _exact_keys(observations, TOP_KEYS, "raw observations")
    for key in ("schema", "simllm_source_commit", "htsim_source_commit"):
        _text(observations[key], f"raw observations.{key}")
    _hex_digest(observations["native_observations_sha256"], "native_observations_sha256")
    preflight = _object(observations["preflight"], "preflight")
    _exact_keys(preflight, PREFLIGHT_KEYS, "preflight")
    _text(preflight["profile"], "preflight.profile")
    for index, line in enumerate(_array(preflight["manifest"], "preflight.manifest")):
        _text(line, f"preflight.manifest[{index}]")
    _integer(preflight["flow_count"], "preflight.flow_count")
    _boolean(preflight["quiescent"], "preflight.quiescent")
    cells: dict[tuple[int, int], CellView] = {}
    for index, raw_cell in enumerate(_array(observations["cells"], "cells")):
        name = f"cells[{index}]"
        cell = _object(raw_cell, name)
        _exact_keys(cell, CELL_KEYS, name)
        rate = _integer(cell["link_rate_gbps"], f"{name}.link_rate_gbps")
        doorbell = _integer(cell["doorbell_service_ps"], f"{name}.doorbell_service_ps")
        _object(cell["execution_graph"], f"{name}.execution_graph")
        _object(cell["step_record"], f"{name}.step_record")
        _parse_history_shape(cell["history_seed"], f"{name}.history_seed")
        bypass = _parse_mode_shape(cell["bypass"], f"{name}.bypass")
        structural = _parse_mode_shape(cell["structural"], f"{name}.structural")
        _parse_transaction_shape(cell["transaction_failure"], f"{name}.transaction_failure")
        key = (rate, doorbell)
        _require(key not in cells, f"authority matrix repeats {key}")
        cells[key] = CellView(cell, rate, doorbell, bypass, structural)
    _require(
        set(cells) == {(200, 0), (200, 1000), (400, 0), (400, 1000)},
        "authority matrix is incomplete",
    )
    return cells


def _json_fraction(value: Fraction) -> int | dict[str, int]:
    if value.denominator == 1:
        return value.numerator
    return {"numerator": value.numerator, "denominator": value.denominator}


def _family_report(name: str, rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    _require(len(rows) == expected, f"{name} evaluated {len(rows)} of {expected} instances")
    passed = sum(row["passed"] is True for row in rows)
    return {
        "expected_instances": expected,
        "evaluated_instances": len(rows),
        "passed_instances": passed,
        "genuine_risk_instances": len(rows),
        "genuine_risk_fraction": f"{len(rows)}/{expected}",
        "passed": passed == expected,
        "instances": rows,
    }


def _score_families(
    cells: dict[tuple[int, int], CellView], expectations: dict[str, Any]
) -> dict[str, Any]:
    signed = expectations["signed_authority_relation"]
    signed_rows: list[dict[str, Any]] = []
    for rate in (200, 400):
        cell = cells[(rate, signed["doorbell_service_ps"])]
        bypass_metrics = cell.bypass.scored_metrics()
        structural_metrics = cell.structural.scored_metrics()
        for metric in signed["metrics"]:
            delta = structural_metrics[metric] - bypass_metrics[metric]
            passed = delta > 0 and Fraction(signed["lower_bound_ps"], 1) <= delta <= Fraction(
                signed["upper_bound_ps"], 1
            )
            signed_rows.append(
                {
                    "instance": f"rate={rate},metric={metric}",
                    "metric": metric,
                    "link_rate_gbps": rate,
                    "observed_delta_ps": _json_fraction(delta),
                    "lower_bound_ps": signed["lower_bound_ps"],
                    "upper_bound_ps": signed["upper_bound_ps"],
                    "direction": signed["direction"],
                    "genuine_risk": True,
                    "passed": passed,
                }
            )
    inverse = expectations["inverse_rate_relation"]
    inverse_rows: list[dict[str, Any]] = []
    for mode_name in ("bypass", "structural"):
        for doorbell in (0, 1000):
            low = getattr(cells[(200, doorbell)], mode_name).scored_metrics()
            high = getattr(cells[(400, doorbell)], mode_name).scored_metrics()
            for metric in inverse["metrics"]:
                delta = low[metric] - high[metric]
                passed = (
                    Fraction(inverse["lower_bound_ps"], 1)
                    <= delta
                    <= Fraction(inverse["upper_bound_ps"], 1)
                )
                inverse_rows.append(
                    {
                        "instance": f"mode={mode_name},doorbell={doorbell},metric={metric}",
                        "mode": mode_name,
                        "doorbell_service_ps": doorbell,
                        "metric": metric,
                        "observed_delta_ps": _json_fraction(delta),
                        "lower_bound_ps": inverse["lower_bound_ps"],
                        "upper_bound_ps": inverse["upper_bound_ps"],
                        "genuine_risk": True,
                        "passed": passed,
                    }
                )
    counts = expectations["behavioral_family_instances"]
    families = {
        "signed_authority_delta": _family_report(
            "signed_authority_delta", signed_rows, counts["signed_authority_delta"]
        ),
        "inverse_rate_delta": _family_report(
            "inverse_rate_delta", inverse_rows, counts["inverse_rate_delta"]
        ),
    }
    misses = {
        name: [row["instance"] for row in report["instances"] if not row["passed"]]
        for name, report in families.items()
        if not report["passed"]
    }
    if misses:
        rendered = "; ".join(f"{name} misses={instances}" for name, instances in misses.items())
        raise AuthorityCheckError(rendered, evaluation_order=EVALUATION_ORDER[:3])
    return families


def _artifact_from_row(row: dict[str, Any]) -> BypassArtifacts:
    pairs = tuple((str(pair[0]), str(pair[1])) for pair in row["baseline_parameters"])
    return BypassArtifacts(
        goal_text=bytes.fromhex(row["goal_text_hex"]),
        goal_binary=bytes.fromhex(row["goal_binary_hex"]),
        topology=bytes.fromhex(row["topology_hex"]),
        profile=row["profile"],
        seed=row["seed"],
        baseline_parameters=pairs,
        completion_csv=bytes.fromhex(row["completion_csv_hex"]),
        canonical_completion=bytes.fromhex(row["canonical_completion_hex"]),
        step_results=bytes.fromhex(row["step_results_hex"]),
        replay_summary=bytes.fromhex(row["replay_summary_hex"]),
    )


def _completion_csv(execution_result: dict[str, Any]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "execution_id",
            "operation_id",
            "phase",
            "timestamp_ps",
            "subject_object_id",
            "completed_bytes",
        )
    )
    for event in execution_result["events"]:
        writer.writerow(
            (
                event["execution_id"],
                event["operation_id"],
                event["phase"],
                event["timestamp_ps"],
                event["subject_object_id"],
                event["completed_bytes"],
            )
        )
    return stream.getvalue().encode("utf-8")


def _validate_graph_and_seed(
    cells: dict[tuple[int, int], CellView], expectations: dict[str, Any]
) -> None:
    first = cells[(200, 0)].raw
    for cell in cells.values():
        _require(
            cell.raw["execution_graph"] == first["execution_graph"],
            "fixed graph bytes differ across cells",
        )
        _require(
            cell.raw["step_record"] == first["step_record"],
            "fixed StepRecord bytes differ across cells",
        )
        _require(
            cell.raw["history_seed"] == first["history_seed"],
            "history seed bytes differ across cells",
        )
    try:
        graph = execution_graph_from_json(first["execution_graph"])
        record = step_record_from_json(first["step_record"])
    except (TypeError, ValueError) as error:
        raise AuthorityCheckError(f"fixed graph or StepRecord is invalid: {error}") from error
    _require(
        execution_graph_to_json(graph) == first["execution_graph"], "fixed graph is not canonical"
    )
    _require(
        step_record_to_json(record) == first["step_record"], "fixed StepRecord is not canonical"
    )
    frozen = expectations["fixed_graph"]
    _require(graph.execution_id == frozen["execution_id"], "fixed execution ID drifted")
    _require(graph.step_index == frozen["step_index"], "fixed graph step drifted")
    _require(graph.released_at_ps == frozen["released_at_ps"], "fixed graph release drifted")
    _require(len(graph.operations) == 1, "fixed graph must contain one operation")
    operation = graph.operations[0]
    _require(operation.operation_id == frozen["operation_ids"][0], "fixed operation ID drifted")
    _require(operation.rank == frozen["source_rank"], "fixed source rank drifted")
    _require(operation.logical_queue == frozen["logical_queue"], "fixed logical queue drifted")
    _require(isinstance(operation.work, ControlWork), "fixed work is not control work")
    _require(
        operation.work.mode is ControlMode.SYNCHRONOUS, "fixed control work is not synchronous"
    )
    _require(
        list(operation.work.destination_ranks) == frozen["destination_ranks"],
        "fixed destinations drifted",
    )
    _require(operation.work.payload_bytes == frozen["payload_bytes"], "fixed payload drifted")
    _require(
        list(operation.correlation.request_ids) == frozen["request_ids"],
        "fixed request correlation drifted",
    )
    _require(
        list(graph.completion_operation_ids) == frozen["completion_operation_ids"],
        "completion endpoint drifted",
    )
    _require(
        record.step_index == graph.step_index and record.virtual_time_ps == graph.released_at_ps,
        "StepRecord disagrees with fixed graph",
    )
    scheduled = {item.request_id: item.phase.value for item in record.scheduled}
    _require(
        scheduled == {"core21-prefill": "prefill", "core21-decode": "decode"},
        "fixed scheduled requests drifted",
    )
    _require(
        record.num_sampled == 2 and set(record.sampled_request_ids or ()) == set(scheduled),
        "fixed sampled requests drifted",
    )

    history = first["history_seed"]
    try:
        seed_graph = execution_graph_from_json(history["execution_graph"])
        seed_record = step_record_from_json(history["step_record"])
        seed_result = execution_result_from_json(history["execution_result"])
    except (TypeError, ValueError) as error:
        raise AuthorityCheckError(f"history seed is invalid: {error}") from error
    seed_expectation = expectations["history_seed"]
    _require(
        seed_graph.released_at_ps == seed_expectation["first_release_ps"], "history release drifted"
    )
    _require(
        seed_result.completed_at_ps - seed_graph.released_at_ps == seed_expectation["duration_ps"],
        "history duration drifted",
    )
    _require(
        seed_record.virtual_time_ps == seed_graph.released_at_ps,
        "history StepRecord release drifted",
    )
    _require(
        len(seed_graph.operations) == 1 and not history["runtime_report"]["wqes"],
        "history seed is not compute-only",
    )
    _require(
        seed_expectation["request_id"] in seed_graph.operations[0].correlation.request_ids,
        "history request drifted",
    )
    seed_step, seed_metrics = _parse_step_result_shape(
        history["step_result"], "history_seed.step_result"
    )
    _require(
        seed_step["step_latency_ps"] == seed_expectation["duration_ps"],
        "history StepResult duration drifted",
    )
    _require(
        seed_step["completed_at_ps"] == seed_result.completed_at_ps == graph.released_at_ps,
        "history completion boundary drifted",
    )
    _require(
        set(seed_metrics) == {seed_expectation["request_id"]}, "history request metric drifted"
    )
    seed_metric = seed_metrics[seed_expectation["request_id"]]
    _require(
        seed_metric["token_index"] == 1
        and seed_metric["ttft_ps"] == seed_expectation["duration_ps"],
        "history first token drifted",
    )
    _require(seed_metric["tpot_ps"] is None, "history seed unexpectedly has TPOT")


def _validate_report_and_mode(cell: CellView, name: str, mode: ModeView, expected_jct: int) -> None:
    raw = mode.raw
    expected_authority = "AtlahsWqeLedger" if name == "bypass" else "SimllmNativeRnicSession"
    _require(raw["hardware_mode"] == name, f"{name} hardware mode drifted")
    _require(raw["authority"] == expected_authority, f"{name} authority drifted")
    _require(raw["wqe_count"] == 2, f"{name} WQE count drifted")
    try:
        result = execution_result_from_json(raw["execution_result"])
    except (TypeError, ValueError) as error:
        raise AuthorityCheckError(f"{name} ExecutionResult is invalid: {error}") from error
    graph = cell.raw["execution_graph"]
    release = graph["released_at_ps"]
    operation_id = graph["operations"][0]["operation_id"]
    _require(result.execution_id == graph["execution_id"], f"{name} ExecutionResult ID drifted")
    _require(
        result.completed_at_ps == release + expected_jct,
        f"{name} ExecutionResult boundary missed its oracle",
    )
    _require(
        result.quiesced_at_ps == result.completed_at_ps, f"{name} did not quiesce at completion"
    )
    report = raw["runtime_report"]
    _require(report["execution_id"] == result.execution_id, f"{name} RuntimeReport ID drifted")
    _require(report["authority"] == expected_authority, f"{name} RuntimeReport authority drifted")
    _require(len(report["operations"]) == 1, f"{name} RuntimeReport operation count drifted")
    report_operation = report["operations"][0]
    _require(report_operation["operation_id"] == operation_id, f"{name} report operation drifted")
    _require(
        report_operation["completed_at_ps"] == result.completed_at_ps,
        f"{name} report boundary drifted",
    )
    _require(
        sum(report_operation["attribution"].values()) == expected_jct,
        f"{name} report attribution does not conserve",
    )
    visits = report["visits"]
    for visit in visits:
        times = [
            visit[key]
            for key in (
                "submitted_at_ps",
                "eligible_at_ps",
                "started_at_ps",
                "finished_at_ps",
                "completed_at_ps",
            )
        ]
        _require(times == sorted(times), f"{name} visit timestamps are not monotonic")
        _require(
            visit["execution_id"] == result.execution_id and visit["operation_id"] == operation_id,
            f"{name} visit identity drifted",
        )
    _require(
        report["sum_visit_wait_ps"]
        == sum(visit["started_at_ps"] - visit["eligible_at_ps"] for visit in visits),
        f"{name} additive wait drifted",
    )
    _require(
        report["realized_critical_path_operation_ids"] == [operation_id],
        f"{name} critical path drifted",
    )
    _require(report["random_draw_count"] == 0, f"{name} random draw count drifted")
    wqes = report["wqes"]
    _require(len(wqes) == 2, f"{name} RuntimeReport WQE count drifted")
    _require(
        [wqe["destination_rank"] for wqe in wqes] == [8, 16],
        f"{name} destination projection drifted",
    )
    _require([wqe["extent_index"] for wqe in wqes] == [0, 1], f"{name} extent order drifted")
    for wqe in wqes:
        _require(wqe["authority"] == expected_authority, f"{name} WQE authority drifted")
        _require(
            wqe["execution_id"] == result.execution_id and wqe["operation_id"] == operation_id,
            f"{name} WQE identity drifted",
        )
        _require(wqe["payload_bytes"] == 4096, f"{name} WQE payload drifted")
        times = [
            wqe[key]
            for key in (
                "submitted_at_ps",
                "eligible_at_ps",
                "started_at_ps",
                "finished_at_ps",
                "completed_at_ps",
            )
        ]
        _require(times == sorted(times), f"{name} WQE timestamps are not monotonic")
    native_fields = (
        "doorbell_started_at_ps",
        "doorbell_completed_at_ps",
        "network_eligible_at_ps",
        "network_started_at_ps",
        "network_finished_at_ps",
    )
    if name == "bypass":
        _require(
            len(visits) == 4
            and {visit["stage"] for visit in visits} == {"coarse_runtime"}
            and all(all(wqe[key] is None for key in native_fields) for wqe in wqes),
            "bypass unexpectedly exposes native stages",
        )
    else:
        _require(
            len(visits) == 6
            and all(all(wqe[key] is not None for key in native_fields) for wqe in wqes),
            "structural native stages are absent",
        )
        _require(
            {visit["stage"] for visit in visits}
            == {"coarse_runtime", "native_doorbell", "native_network"},
            "structural native visits drifted",
        )

    step = mode.step_result
    _require(step["step_index"] == graph["step_index"], f"{name} StepResult index drifted")
    _require(step["step_latency_ps"] == expected_jct, f"{name} JCT missed its exact oracle")
    _require(
        step["completed_at_ps"] == result.completed_at_ps, f"{name} StepResult boundary drifted"
    )
    prefill = mode.metrics["core21-prefill"]
    decode = mode.metrics["core21-decode"]
    _require(
        prefill["phase"] == "prefill" and prefill["token_index"] == 1,
        f"{name} prefill metric identity drifted",
    )
    _require(
        decode["phase"] == "decode" and decode["token_index"] == 2,
        f"{name} decode metric identity drifted",
    )
    _require(prefill["ttft_ps"] == expected_jct, f"{name} prefill TTFT missed its exact oracle")
    _require(
        _fraction(decode["tpot_ps"], f"{name}.decode.tpot") == expected_jct,
        f"{name} decode TPOT missed its exact oracle",
    )
    _require(decode["ttft_ps"] == 10000, f"{name} decode history TTFT drifted")
    for metric in (prefill, decode):
        _require(
            metric["completed_at_ps"] == result.completed_at_ps,
            f"{name} request completion drifted",
        )
        _require(
            sum(metric["attribution"].values()) == metric["latency_ps"],
            f"{name} request attribution does not conserve",
        )
    summary = raw["request_summary"]
    _require(summary["jct_ps"] == step["step_latency_ps"], f"{name} request summary JCT drifted")
    _require(
        summary["prefill_ttft_ps"] == prefill["ttft_ps"], f"{name} request summary TTFT drifted"
    )
    _require(
        _fraction(summary["decode_tpot_ps"], f"{name}.summary.decode_tpot")
        == _fraction(decode["tpot_ps"], f"{name}.decode_tpot"),
        f"{name} request summary TPOT drifted",
    )
    additive = {
        "queue_wait_ps": sum(visit["started_at_ps"] - visit["eligible_at_ps"] for visit in visits),
        "service_ps": sum(visit["finished_at_ps"] - visit["started_at_ps"] for visit in visits),
        "visibility_ps": sum(
            visit["completed_at_ps"] - visit["finished_at_ps"] for visit in visits
        ),
        "visit_count": len(visits),
    }
    _require(
        step["additive_visit_totals"] == additive, f"{name} StepResult additive totals drifted"
    )


def _validate_artifact_provenance(cell: CellView) -> None:
    raw = cell.bypass.raw
    artifact_row = _object(raw["bypass_artifacts"], "bypass.bypass_artifacts")
    _require(
        cell.structural.raw["bypass_artifacts"] is None, "structural mode carries bypass artifacts"
    )
    expected_parameters = canonical_bypass_parameters(
        {
            "authority_mode": "bypass",
            "link_rate_gbps": cell.rate,
            "payload_bytes": 4096,
            "step_record_sha256": hashlib.sha256(
                _canonical_bytes(cell.raw["step_record"])
            ).hexdigest(),
            "wqe_count": 2,
        }
    )
    expected_pairs = [[name, value] for name, value in expected_parameters]
    _require(
        artifact_row["profile"] == "core21-fixed-contended-graph", "bypass artifact profile drifted"
    )
    _require(artifact_row["seed"] == 0, "bypass artifact seed drifted")
    _require(
        artifact_row["baseline_parameters"] == expected_pairs, "bypass artifact parameters drifted"
    )
    _require(
        bytes.fromhex(artifact_row["goal_text_hex"]) == _pretty_bytes(cell.raw["execution_graph"]),
        "bypass graph text bytes drifted",
    )
    _require(
        bytes.fromhex(artifact_row["goal_binary_hex"])
        == _canonical_bytes(cell.raw["execution_graph"]),
        "bypass graph binary bytes drifted",
    )
    _require(bytes.fromhex(artifact_row["topology_hex"]) == b"", "bypass topology must be empty")
    _require(
        bytes.fromhex(artifact_row["completion_csv_hex"])
        == _completion_csv(raw["execution_result"]),
        "bypass completion bytes drifted",
    )
    _require(
        bytes.fromhex(artifact_row["canonical_completion_hex"])
        == _canonical_bytes(raw["execution_result"]),
        "bypass canonical completion drifted",
    )
    _require(
        bytes.fromhex(artifact_row["step_results_hex"]) == _canonical_bytes(raw["step_result"]),
        "bypass StepResult bytes drifted",
    )
    _require(
        bytes.fromhex(artifact_row["replay_summary_hex"])
        == _canonical_bytes(raw["request_summary"]),
        "bypass request summary bytes drifted",
    )
    artifact = _artifact_from_row(artifact_row)
    comparison = compare_bypass_artifacts(artifact, artifact)
    _require(comparison.equivalent, "repository bypass comparator rejected accepted bytes")


def _validate_positive_fatal(
    cells: dict[tuple[int, int], CellView],
    observations: dict[str, Any],
    expectations: dict[str, Any],
) -> dict[str, bool]:
    _require(observations["schema"] == OBSERVATION_SCHEMA, "raw observation schema drifted")
    source_commit = _git_commit(observations["simllm_source_commit"], "raw simllm_source_commit")
    if source_commit != SIMLLM_SOURCE_COMMIT:
        ancestry = subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                SIMLLM_SOURCE_COMMIT,
                source_commit,
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        _require(
            ancestry.returncode == 0,
            "raw SimLLM source commit does not descend from the frozen anchor",
        )
    _require(observations["htsim_source_commit"] == HTSIM_SOURCE_COMMIT, "raw htsim anchor drifted")
    preflight = observations["preflight"]
    manifest = preflight["manifest"]
    _require(preflight["profile"] == "rnic-nn", "positive preflight profile drifted")
    _require(
        preflight["flow_count"] > 0 and preflight["quiescent"] is True,
        "positive preflight did not complete a quiescent flow",
    )
    _require(
        any(
            "hardware_mode=structural" in line
            and "wqe_authority=simllm-native-rnic-session" in line
            for line in manifest
        ),
        "positive preflight lacks structural native authority",
    )
    _require(
        any("physical_quiescence=verified" in line for line in manifest),
        "positive preflight lacks quiescence manifest",
    )
    _validate_graph_and_seed(cells, expectations)
    for key, cell in sorted(cells.items()):
        rate, doorbell = key
        wire = 4096 * 8 * 1000
        _require(wire % rate == 0, f"cell {key} has inexact wire service")
        bypass_jct = 2 * (wire // rate)
        structural_jct = doorbell + bypass_jct
        _validate_report_and_mode(cell, "bypass", cell.bypass, bypass_jct)
        _validate_report_and_mode(cell, "structural", cell.structural, structural_jct)
        if doorbell == 0:
            _require(
                cell.bypass.scored_metrics() == cell.structural.scored_metrics(),
                f"D-zero authority identity failed at {rate} Gbit/s",
            )
        transaction = cell.raw["transaction_failure"]
        _require(
            transaction["exception_type"] == "ValueError",
            f"cell {key} transaction exception type drifted",
        )
        _require(
            "consume every WQE" in transaction["exception_message"],
            f"cell {key} transaction diagnostic drifted",
        )
        _require(
            transaction["before"] == {"committed_transactions": 0, "committed_wqes": 0},
            f"cell {key} pre-failure counters drifted",
        )
        _require(
            transaction["after"] == transaction["before"],
            f"cell {key} failed transaction was not atomic",
        )
        _require(
            transaction["retry"] == {"committed_transactions": 1, "committed_wqes": 2},
            f"cell {key} successful retry counters drifted",
        )
        _require(
            transaction["runtime_last_report_is_none"] is True,
            f"cell {key} failed transaction installed a report",
        )
        _require(
            transaction["bypass_ledger_is_none"] is True,
            f"cell {key} structural runtime retained a bypass ledger",
        )
        _require(
            transaction["authority"] == "SimllmNativeRnicSession",
            f"cell {key} transaction authority drifted",
        )
        _validate_artifact_provenance(cell)
    return {
        "d0_authority_identity": True,
        "identical_graph_and_step_input": True,
        "deployed_reducer_on_both_modes": True,
        "bypass_ledger_is_sole_bypass_authority": True,
        "native_session_is_sole_structural_authority": True,
        "transaction_failure_is_atomic": True,
        "transaction_retry_commits_once": True,
    }


def _hash_map(value: Any, name: str) -> dict[str, str]:
    row = _object(value, name)
    _require(set(row) == POSITIVE_ASSET_NAMES, f"{name} inventory drifted")
    return {
        _text(key, f"{name} key"): _hex_digest(digest, f"{name}.{key}")
        for key, digest in row.items()
    }


def _comparison_pairs(value: Any, expected: tuple[str, ...], name: str) -> list[list[Any]]:
    result: list[list[Any]] = []
    for index, raw_pair in enumerate(_array(value, name)):
        pair = _array(raw_pair, f"{name}[{index}]")
        _require(len(pair) == 2, f"{name}[{index}] must be a pair")
        result.append(
            [_text(pair[0], f"{name}[{index}][0]"), _boolean(pair[1], f"{name}[{index}][1]")]
        )
    _require(tuple(pair[0] for pair in result) == expected, f"{name} inventory drifted")
    _require(all(pair[1] is True for pair in result), f"{name} contains a mismatch")
    return result


def _validate_negative(
    value: dict[str, Any],
    expectations: dict[str, Any],
    observations: dict[str, Any],
    raw_observations_sha256: str | None,
) -> dict[str, bool]:
    _exact_keys(value, NEGATIVE_KEYS, "negative evidence")
    _require(value["schema"] == NEGATIVE_SCHEMA, "negative evidence schema drifted")
    _require(value["htsim_source_commit"] == HTSIM_SOURCE_COMMIT, "negative htsim anchor drifted")
    _require(value["cache_value"] == "OFF", "negative cache did not disable the SimLLM link")
    _hex_digest(value["negative_main_sha256"], "negative_main_sha256")
    before = _hash_map(value["positive_assets_before"], "positive_assets_before")
    after = _hash_map(value["positive_assets_after"], "positive_assets_after")
    _require(
        before["native_observations_sha256"] == observations["native_observations_sha256"],
        "native observation digest disagrees with preserved assets",
    )
    if raw_observations_sha256 is not None:
        _require(
            before["raw_observations_sha256"] == raw_observations_sha256,
            "raw observation digest disagrees with preserved assets",
        )
    _require(
        before == after
        and _boolean(value["positive_assets_preserved"], "positive_assets_preserved") is True,
        "positive assets changed during the negative build",
    )
    comparison = _object(value["bypass_comparison"], "bypass_comparison")
    _exact_keys(
        comparison, {"input_matches", "behavioral_matches", "equivalent"}, "bypass_comparison"
    )
    _comparison_pairs(
        comparison["input_matches"], INPUT_ARTIFACT_NAMES, "bypass_comparison.input_matches"
    )
    _comparison_pairs(
        comparison["behavioral_matches"],
        BEHAVIORAL_ARTIFACT_NAMES,
        "bypass_comparison.behavioral_matches",
    )
    _require(
        _boolean(comparison["equivalent"], "bypass_comparison.equivalent") is True,
        "repository bypass comparison was not equivalent",
    )
    _require(
        _boolean(
            value["negative_tier_a_target_absent"],
            "negative_tier_a_target_absent",
        )
        is True,
        "negative Tier A target exists",
    )
    negative = expectations["negative_control"]
    _require(
        _integer(value["producer_returncode"], "producer_returncode")
        == negative["expected_producer_exit"],
        "negative producer return code drifted",
    )
    _require(
        _text(value["producer_stderr"], "producer_stderr").strip()
        == negative["producer_diagnostic"],
        "negative producer diagnostic drifted",
    )
    _require(
        _integer(value["checker_returncode"], "checker_returncode")
        == negative["expected_checker_exit"],
        "negative checker return code drifted",
    )
    _require(
        _text(value["checker_stderr"], "checker_stderr").strip() == negative["checker_diagnostic"],
        "negative checker diagnostic drifted",
    )
    forbidden = _object(value["forbidden_outputs_absent"], "forbidden_outputs_absent")
    _require(
        set(forbidden) == set(negative["forbidden_outputs"]), "forbidden output inventory drifted"
    )
    _require(
        all(
            _boolean(item, f"forbidden_outputs_absent.{name}") is True
            for name, item in forbidden.items()
        ),
        "negative run published a forbidden output",
    )
    return {
        "bypass_artifacts_preserved": True,
        "positive_assets_preserved": True,
        "negative_cache_is_off": True,
        "negative_main_binary_exists": True,
        "negative_tier_a_target_absent": True,
        "negative_producer_rejected_native_absence": True,
        "negative_checker_rejected_missing_observations": True,
        "negative_published_no_accepted_result": True,
    }


def check_observations(
    observations: dict[str, Any],
    expectations: dict[str, Any],
    negative_evidence: dict[str, Any],
    *,
    raw_observations_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate raw relations first, then fatal and link-negative evidence."""

    _validate_expectations(expectations)
    cells = _parse_raw_shape(observations)
    families = _score_families(cells, expectations)
    fatal = _validate_positive_fatal(cells, observations, expectations)
    fatal.update(
        _validate_negative(
            negative_evidence,
            expectations,
            observations,
            raw_observations_sha256,
        )
    )
    _require(set(fatal) == set(expectations["fatal_unscored"]), "fatal evidence inventory drifted")
    return {
        "schema": RESULT_SCHEMA,
        "simllm_source_commit": observations["simllm_source_commit"],
        "htsim_source_commit": observations["htsim_source_commit"],
        "passed": True,
        "evaluation_order": list(EVALUATION_ORDER),
        "entailment_analysis": {
            "raw_shape_and_types_only_before_scoring": True,
            "scored_before_per_mode_exact_oracles": True,
            "scored_sources": [
                "step_result.step_latency_ps",
                "step_result.request_metrics[core21-prefill].ttft_ps",
                "step_result.request_metrics[core21-decode].tpot_ps",
            ],
            "signed_instances_can_fail": 6,
            "inverse_rate_instances_can_fail": 12,
            "shared_raw_cells_are_not_independent_risks": True,
            "fatal_and_change_set_guards_are_unscored": True,
        },
        "behavioral_families": families,
        "fatal_unscored": fatal,
    }


def _validate_cli_paths(
    expectations: Path, observations: Path, negative: Path, results: Path
) -> None:
    _require(expectations.is_file(), "expectations path must name a file")
    resolved = [
        path.resolve(strict=False) for path in (expectations, observations, negative, results)
    ]
    _require(len(set(resolved)) == 4, "checker paths must be distinct")
    _require(results.name != "results.json.tmp", "results path must not be a temporary path")


def _atomic_publish(path: Path, value: dict[str, Any]) -> None:
    temporary = Path(f"{path}.tmp")
    _require(path.parent.is_dir(), "results parent directory does not exist")
    _require(
        not path.exists() and not temporary.exists(),
        "results or temporary publication already exists",
    )
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expectations", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--negative-evidence", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        _validate_cli_paths(
            arguments.expectations,
            arguments.observations,
            arguments.negative_evidence,
            arguments.results,
        )
        expectations = _load_json(arguments.expectations, "authority expectations")
        _validate_expectations(expectations)
    except AuthorityCheckError as error:
        print(str(error), file=sys.stderr)
        return 1
    if arguments.check_only:
        print("authority checker registry and paths validated; no output created")
        return 0
    if not arguments.observations.is_file():
        print(OBSERVATION_DIAGNOSTIC, file=sys.stderr)
        return 2
    try:
        observations = _load_json(arguments.observations, "authority observations")
        negative = _load_json(arguments.negative_evidence, "negative evidence")
        report = check_observations(
            observations,
            expectations,
            negative,
            raw_observations_sha256=hashlib.sha256(arguments.observations.read_bytes()).hexdigest(),
        )
        report["expectations_sha256"] = hashlib.sha256(
            arguments.expectations.read_bytes()
        ).hexdigest()
        _atomic_publish(arguments.results, report)
    except (AuthorityCheckError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    for name, family in report["behavioral_families"].items():
        print(f"{name} genuine-risk fraction: {family['genuine_risk_fraction']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["AuthorityCheckError", "check_observations", "main"]
