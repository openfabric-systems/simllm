#!/usr/bin/env python3
"""Resolve the literal CORE-63 and CORE-64 registry paragraphs once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core63_clean_field_reader import ACCESS_SCHEMA, read_registry_tail_entries
from core63_clean_residency import write_new_json, write_new_text

STUDY_DIR = Path(__file__).resolve().parent
EXPECTATIONS_PATH = STUDY_DIR / "core63_clean_registry_resolution_expectations.json"
ACCESS_LEDGER_PATH = (
    STUDY_DIR / "core63_clean_registry_resolution_access_ledger.jsonl"
)
RESULT_PATH = STUDY_DIR / "core63_clean_registry_resolution.json"
RESULT_MARKDOWN_PATH = STUDY_DIR / "core63_clean_registry_resolution.md"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path.name} contains a non-object row")
            values.append(value)
    return values


def _validate_access(events: list[dict[str, Any]]) -> None:
    if len(events) != 4:
        raise ValueError("registry resolution must contain four access events")
    for index, event in enumerate(events, start=1):
        if event.get("event_index") != index:
            raise ValueError("registry resolution event indices differ")
        if event.get("schema") != ACCESS_SCHEMA:
            raise ValueError("registry resolution access schema differs")
        if event.get("held_out_mtp_value_accessed") is not False:
            raise ValueError("registry resolution reports MTP exposure")
        if event.get("whole_file_streamed") is not False:
            raise ValueError("registry resolution reports a whole-file stream")
    for access_number in range(2):
        begin = events[2 * access_number]
        end = events[2 * access_number + 1]
        if begin.get("event") != "BEGIN" or begin.get("bytes_accessed") != 0:
            raise ValueError("registry resolution begin was not contemporaneous")
        if end.get("event") != "END" or end.get("status") != "PASS":
            raise ValueError("registry resolution access did not pass")
        consumed = end.get("bytes_accessed")
        size = end.get("record_size_bytes")
        unique = end.get("unique_bytes_accessed")
        if type(consumed) is not int or type(size) is not int or type(unique) is not int:
            raise TypeError("registry resolution byte accounting must use integers")
        if not 0 < unique <= consumed < size:
            raise ValueError("registry resolution did not remain partial")
        if end.get("access_pattern") != "reverse_nonterminal_task_paragraph":
            raise ValueError("registry resolution access pattern differs")


def _render(result: dict[str, Any]) -> str:
    return f"""# CORE-63 clean registry resolution

Status: **{result['status']}**. This addendum supersedes only the earlier
forward registry mentions in `core63_clean_calibration_result.json`.

## Literal CORE-63 registration

{result['entries']['CORE-63']}

## Literal CORE-64 registration

{result['entries']['CORE-64']}

## Locked clean finding

The clean reproduction remains
`{result['clean_reproduction_status']}`. Registry resolution is not an
arithmetic input. The forbidden-access ledger remains empty, whole-file
streams remain zero, and no held-out MTP value was accessed or compared.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expectations-commit", required=True)
    args = parser.parse_args()
    for path in (ACCESS_LEDGER_PATH, RESULT_PATH, RESULT_MARKDOWN_PATH):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    expectations = _load_json(EXPECTATIONS_PATH)
    if expectations.get("status") != "EXPECTATIONS_ONLY_REGISTRY_RESOLUTION":
        raise ValueError("registry resolution expectations status differs")
    entries = read_registry_tail_entries(ACCESS_LEDGER_PATH)
    events = _load_jsonl(ACCESS_LEDGER_PATH)
    _validate_access(events)
    if "CORE-63" not in entries["CORE-63"] or "CORE-64" not in entries["CORE-64"]:
        raise ValueError("registry resolution returned the wrong task entry")
    result = {
        "access": {
            "access_count": 2,
            "access_event_count": 4,
            "events": events,
            "forbidden_access_ledger": [],
            "held_out_mtp_numeric_values_accessed_or_compared": False,
            "whole_file_streams": 0,
        },
        "clean_reproduction_status": expectations["clean_reproduction"]["status"],
        "entries": entries,
        "expectations_commit": args.expectations_commit,
        "schema": "simllm-deployment-curve-core63-clean-registry-resolution-v1",
        "status": "PASS_LITERAL_REGISTRY_RESOLUTION",
        "task": "CORE-63",
    }
    write_new_json(RESULT_PATH, result)
    write_new_text(RESULT_MARKDOWN_PATH, _render(result))
    print(json.dumps({"entries": entries, "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
