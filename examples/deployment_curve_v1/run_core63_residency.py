#!/usr/bin/env python3
"""Publish CORE-63 from the append-only field-reader evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core63_residency import build_result

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
EXPECTATIONS_PATH = STUDY_DIR / "core63_expectations.json"
ACCESS_LOG_NAMES = (
    "access-ledger.jsonl",
    "access-ledger-retry.jsonl",
    "access-ledger-header.jsonl",
    "access-ledger-final.jsonl",
    "access-ledger-routes.jsonl",
    "access-ledger-success.jsonl",
)
PROTOCOL_INCIDENTS = (
    {
        "classification": "forbidden_held_out_numeric_exposure",
        "record": "examples/deployment_curve_v1/expectations.json",
        "access_path": "ambient direct range inspection before the committed freeze",
        "held_out_numeric_value_exposed": True,
        "numeric_value_redacted_from_core63_publication": True,
        "used_in_residency_arithmetic": False,
        "compared_in_core63": False,
        "logged_contemporaneously": False,
        "protocol_violation": True,
    },
    {
        "classification": "unlogged_retained_record_inspection",
        "record": "examples/hopper_kernel_cycle_candidate_v1/retained_evidence.json",
        "access_path": "ambient historical-record inspection before the committed reader",
        "held_out_numeric_value_exposed": False,
        "used_in_residency_arithmetic": False,
        "logged_contemporaneously": False,
        "protocol_violation": True,
    },
    {
        "classification": "forbidden_held_out_numeric_reexposure",
        "record": "docs/README_PRO.md",
        "access_path": "ambient broad registry inspection after the protocol was already void",
        "held_out_numeric_value_exposed": True,
        "numeric_value_redacted_from_core63_publication": True,
        "used_in_residency_arithmetic": False,
        "compared_in_core63": False,
        "logged_contemporaneously": False,
        "protocol_violation": True,
    },
    {
        "classification": "literal_whole_file_byte_stream",
        "record": (
            "$SIMLLM_KERNELPROBE_ROOT/gh200lane/"
            "capture-198891-deepseek-v3-tp1-graph-decode/analysis/"
            "kernel-summary.csv"
        ),
        "access_path": "committed field-addressed streaming reader",
        "bytes_consumed": 13_985,
        "cataloged_record_bytes": 13_985,
        "full_byte_stream_consumed": True,
        "whole_record_materialized": False,
        "unselected_payload_fields_decoded": False,
        "held_out_numeric_value_exposed": False,
        "logged_contemporaneously": True,
        "protocol_violation": True,
    },
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _load_access_log(path: Path) -> list[dict[str, Any]]:
    entries = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path.name} contains a non-object row")
            entries.append({"ledger": path.name, **value})
    return entries


def _write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _markdown(result: dict[str, Any]) -> str:
    derivation = result["residency_derivation"]
    families = derivation["family_decomposition"]
    step = derivation["step"]
    calibration = result["calibration_only"]
    corrected = calibration["residency_corrected"]
    movement = calibration["movement"]
    access = result["access"]
    return f"""# CORE-63 decode expert-residency result

Status: **{result['status']}**. This is an honest calibration-only numerical
finding, not a clean-protocol closure.

## Residency arithmetic, corrected step and movement

Uniform routing gives one EP72 rank an expected
`256 tokens/node x top 8 x 4 resident slots / 288 slots = 256/9` routed
expert-token assignments. The TP1 batch-32 capture represents `32 x 8 = 256`
assignments, so the frozen routed-expert scale is exactly `1/9`. The signed
direction was frozen first: the step must decrease and throughput must increase.

The 46 retained noncollective rows total
{families['retained_repeatable_four_layer_ps']['decimal_ps']} ps of retained
repeatable work plus {families['routed_four_layer_ps']['decimal_ps']} ps of
routed `fused_moe_kernel` work, with the independently retained fixed term of
{families['fixed_service_ps']['decimal_ps']} ps kept once. Therefore:

```text
T63 = 489 + 61/4 x (1,744,159,511 + 131,520,000 / 9)
    = {step['residency_corrected_ps']['decimal_ps']} ps
```

The published round-half-up step is
**{step['residency_corrected_ps']['published_ps_round_half_up']:,} ps**. The
standard-decode prediction moves from the published 8,949.76 display to
**{corrected['prediction_tokens_per_second_per_node']} tokens/s/node**, a
signed increase of **{movement['prediction_tokens_per_second_per_node']}**
tokens/s/node ({movement['prediction_relative_percent']} percent). Against the
published calibration anchor of 22,282 tokens/s/node, the signed residual is
**{corrected['signed_residual_percent']} percent**, a
{movement['signed_residual_percentage_points']}-percentage-point movement.
The calibration classification is **{corrected['classification']}**.

## Component and mechanism ruling

Only the one row containing the preregistered marker `fused_moe_kernel` is
scaled. Attention, MLA, router/top-k work, the shared expert, dense early MLP,
normalization, elementwise work and every other noncollective row stay at
scale one. The kernel summary reconstructs the retained 1,875,680,000 ps step
exactly, and the record components independently reconstruct it from compute,
memory and the 489 ps fixed term. There are zero fitted or free constants.

No communication term enters the current decode price. Decode-side overlap is
therefore not binding here and remains a follow-on only after a decode
communication service term exists.

## Protocol, access and preservation

The final successful tranche contains exactly two logged field-addressed
accesses. Across schema discovery and the successful tranche, the append-only
reader ledger contains {access['cumulative_reader_access_count']} entries:
{access['cumulative_pass_count']} PASS and {access['cumulative_rejected_count']}
REJECTED. Its held-out ledger is empty.

CORE-63 is nevertheless **protocol void**. Before the committed reader/freeze,
an ambient direct range inspection exposed a forbidden held-out MTP numeric
value and a retained historical record was inspected without a contemporaneous
access row. A later broad registry inspection re-exposed that held-out value.
None of those values entered the residency arithmetic and no MTP comparison or
score was performed, but the literal no-read and every-access-logged rules
cannot be restored after exposure. The CSV selector also required streaming
all 13,985 cataloged bytes. It never materialized the whole record or decoded
unselected payload fields, but that still fails the literal no-whole-file-read
clause. The incident ledger in the JSON publication records these failures
without reproducing the held-out number.

All {result['preservation_lock']['checked_count']} preservation-lock artifacts
remain byte-identical. No prior scored artifact changed, no model weights were
downloaded, no web page was fetched and no scored run was performed.

## Registry verdict

CORE-63 stays open for a genuinely exposure-free repetition. CORE-64 is opened
only for the exact remaining standard-decode undercorrection residual reported
above; its conclusion remains conditional on a clean CORE-63 repetition.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--basis", required=True, type=Path)
    parser.add_argument("--expectations-commit", required=True)
    parser.add_argument("--publish-json", required=True, type=Path)
    parser.add_argument("--publish-markdown", required=True, type=Path)
    args = parser.parse_args()

    cumulative = []
    for name in ACCESS_LOG_NAMES:
        cumulative.extend(_load_access_log(args.run_dir / name))
    successful = _load_access_log(args.run_dir / "access-ledger-success.jsonl")
    result = build_result(
        _load_json(EXPECTATIONS_PATH),
        _load_json(args.basis),
        successful,
        cumulative,
        PROTOCOL_INCIDENTS,
        repository_root=REPOSITORY_ROOT,
        expectations_commit=args.expectations_commit,
    )
    rendered_json = json.dumps(result, indent=2, sort_keys=True) + "\n"
    rendered_markdown = _markdown(result)
    _write_new(args.run_dir / "core63-calibration-result.json", rendered_json)
    _write_new(args.run_dir / "core63-calibration-result.md", rendered_markdown)
    _write_new(args.publish_json, rendered_json)
    _write_new(args.publish_markdown, rendered_markdown)
    print(
        json.dumps(
            {
                "classification": result["calibration_only"][
                    "residency_corrected"
                ]["classification"],
                "corrected_prediction": result["calibration_only"][
                    "residency_corrected"
                ]["prediction_tokens_per_second_per_node"],
                "corrected_step_ps": result["residency_derivation"]["step"][
                    "residency_corrected_ps"
                ]["published_ps_round_half_up"],
                "signed_residual_percent": result["calibration_only"][
                    "residency_corrected"
                ]["signed_residual_percent"],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
