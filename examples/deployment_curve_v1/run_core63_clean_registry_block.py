#!/usr/bin/env python3
"""Resolve complete literal CORE-63 and CORE-64 registry blocks once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core63_clean_field_reader import ACCESS_SCHEMA, read_registry_tail_blocks
from core63_clean_residency import write_new_json, write_new_text

STUDY_DIR = Path(__file__).resolve().parent
EXPECTATIONS_PATH = STUDY_DIR / "core63_clean_registry_block_expectations.json"
ACCESS_LEDGER_PATH = STUDY_DIR / "core63_clean_registry_block_access_ledger.jsonl"
RESULT_PATH = STUDY_DIR / "core63_clean_registry_block.json"
RESULT_MARKDOWN_PATH = STUDY_DIR / "core63_clean_registry_block.md"


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
        raise ValueError("registry block resolution requires four events")
    for index, event in enumerate(events, start=1):
        if event.get("event_index") != index:
            raise ValueError("registry block event indices differ")
        if event.get("schema") != ACCESS_SCHEMA:
            raise ValueError("registry block access schema differs")
        if event.get("held_out_mtp_value_accessed") is not False:
            raise ValueError("registry block access reports MTP exposure")
        if event.get("whole_file_streamed") is not False:
            raise ValueError("registry block access reports a whole-file stream")
    for access_number in range(2):
        begin = events[2 * access_number]
        end = events[2 * access_number + 1]
        if begin.get("event") != "BEGIN" or begin.get("bytes_accessed") != 0:
            raise ValueError("registry block begin was not contemporaneous")
        if end.get("event") != "END" or end.get("status") != "PASS":
            raise ValueError("registry block access did not pass")
        consumed = end.get("bytes_accessed")
        size = end.get("record_size_bytes")
        unique = end.get("unique_bytes_accessed")
        if type(consumed) is not int or type(size) is not int or type(unique) is not int:
            raise TypeError("registry block byte accounting must use integers")
        if not 0 < unique <= consumed < size:
            raise ValueError("registry block access did not remain partial")
        if end.get("access_pattern") != "reverse_nonterminal_task_block":
            raise ValueError("registry block access pattern differs")


def _render(result: dict[str, Any]) -> str:
    return f"""# CORE-63 clean literal registry blocks

Status: **{result['status']}**.

## CORE-63

{result['entries']['CORE-63']}

## CORE-64

{result['entries']['CORE-64']}

The blocks are registry-only evidence. The clean arithmetic remains unchanged,
the forbidden-access ledger remains empty, whole-file streams remain zero, and
no held-out MTP value was accessed or compared.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expectations-commit", required=True)
    args = parser.parse_args()
    for path in (ACCESS_LEDGER_PATH, RESULT_PATH, RESULT_MARKDOWN_PATH):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    expectations = _load_json(EXPECTATIONS_PATH)
    if expectations.get("status") != "EXPECTATIONS_ONLY_REGISTRY_BLOCK_RESOLUTION":
        raise ValueError("registry block expectations status differs")
    entries = read_registry_tail_blocks(ACCESS_LEDGER_PATH)
    events = _load_jsonl(ACCESS_LEDGER_PATH)
    _validate_access(events)
    result = {
        "access": {
            "access_count": 2,
            "access_event_count": 4,
            "events": events,
            "forbidden_access_ledger": [],
            "held_out_mtp_numeric_values_accessed_or_compared": False,
            "whole_file_streams": 0,
        },
        "entries": entries,
        "expectations_commit": args.expectations_commit,
        "schema": "simllm-deployment-curve-core63-clean-registry-block-v1",
        "status": "PASS_LITERAL_REGISTRY_BLOCK_RESOLUTION",
        "task": "CORE-63",
    }
    write_new_json(RESULT_PATH, result)
    write_new_text(RESULT_MARKDOWN_PATH, _render(result))
    print(json.dumps({"entries": entries, "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
