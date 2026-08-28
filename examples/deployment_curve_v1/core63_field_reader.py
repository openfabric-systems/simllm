"""Field-addressed readers for CORE-63's retained decode basis."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

ACCESS_SCHEMA = "simllm-deployment-curve-core63-access-v1"
STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
CANDIDATE_RECORD = (
    REPOSITORY_ROOT / "examples/hopper_kernel_cycle_candidate_v1/candidate-record.json"
)
CANDIDATE_LABEL = "examples/hopper_kernel_cycle_candidate_v1/candidate-record.json"
CANDIDATE_SELECTOR = "/entries[7]"
KERNEL_SUMMARY_RELATIVE = Path(
    "gh200lane/capture-198891-deepseek-v3-tp1-graph-decode/analysis/kernel-summary.csv"
)
KERNEL_SUMMARY_LABEL = f"$SIMLLM_KERNELPROBE_ROOT/{KERNEL_SUMMARY_RELATIVE.as_posix()}"
KERNEL_SUMMARY_SELECTOR = (
    "/rows[pool=decode,shape=32,device=0,is_collective=False]"
)
KERNEL_SUMMARY_SHA256_FROM_PUBLISHED_CATALOG = (
    "c4d8ece981478ce57ebca95f7f2f168865713b66e87ed21a1d3f76976e834b7c"
)
KERNEL_SUMMARY_BYTES_FROM_PUBLISHED_CATALOG = 13_985
EXPECTED_IMPLEMENTATION_ID = "deepseek-v3-reduced4-vllm-ep72-decode-b32-c2000"
EXPECTED_HEADER = (
    "pool",
    "shape",
    "device",
    "first_launch_order",
    "name",
    "is_collective",
    "collective_kind",
    "count_per_step",
    "median_duration_ns",
    "mean_duration_ns",
    "total_duration_per_step_ns",
    "share_of_step_compute",
    "graph_record_count",
    "record_count",
)
CAPTURED_KERNEL_FIELDS = (
    "first_launch_order",
    "name",
    "count_per_step",
    "median_duration_ns",
    "mean_duration_ns",
    "total_duration_per_step_ns",
    "share_of_step_compute",
    "graph_record_count",
    "record_count",
)


def _append_access(log_path: Path, entry: Mapping[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(entry), sort_keys=True) + "\n")


def _core61_reader() -> Any:
    path = STUDY_DIR / "core61_depth_field_reader.py"
    spec = importlib.util.spec_from_file_location("core63_core61_reader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the committed CORE-61 projection reader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_component_basis(stream: BinaryIO) -> tuple[dict[str, Any], int]:
    """Project only the retained measured entry's registered component fields."""

    value, consumed = _core61_reader().extract_depth_basis(stream)
    if value.get("implementation_id") != EXPECTED_IMPLEMENTATION_ID:
        raise ValueError("CORE-63 retained implementation identity differs")
    return value, consumed


def _csv_values(raw: bytes) -> list[str]:
    try:
        line = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("kernel summary is not UTF-8") from exc
    values = next(csv.reader([line]))
    if len(values) != len(EXPECTED_HEADER):
        raise ValueError("kernel summary row width differs")
    return values


def extract_standard_decode_kernels(
    stream: BinaryIO,
) -> tuple[list[dict[str, str]], int]:
    """Select only standard decode noncollective rows from a streamed CSV."""

    consumed = 0
    header_raw = stream.readline()
    consumed += len(header_raw)
    header = tuple(_csv_values(header_raw))
    if header != EXPECTED_HEADER:
        raise ValueError("kernel summary header differs from the frozen schema")
    selected: list[dict[str, str]] = []
    saw_selected_shape = False
    for raw in stream:
        consumed += len(raw)
        prefix = raw.split(b",", 3)
        if len(prefix) != 4:
            raise ValueError("kernel summary row lacks routing fields")
        pool, shape, device = prefix[:3]
        if (pool, shape, device) != (b"decode", b"32", b"0"):
            continue
        saw_selected_shape = True
        values = _csv_values(raw)
        row = dict(zip(EXPECTED_HEADER, values, strict=True))
        if row["is_collective"] not in {"True", "False"}:
            raise ValueError("kernel summary collective flag differs")
        if row["is_collective"] == "True":
            continue
        selected.append({name: row[name] for name in CAPTURED_KERNEL_FIELDS})
    if not saw_selected_shape or not selected:
        raise ValueError("standard decode noncollective kernel rows are missing")
    return selected, consumed


def read_component_basis(record_path: Path, access_log: Path) -> dict[str, Any]:
    """Read the allowlisted candidate entry and append one access row."""

    entry: dict[str, Any] = {
        "schema": ACCESS_SCHEMA,
        "classification": "retained_measured_component_decomposition",
        "record": CANDIDATE_LABEL,
        "selector": CANDIDATE_SELECTOR,
        "whole_record_loaded": False,
        "held_out_numeric_value_accessed": False,
        "unselected_values_decoded": False,
    }
    try:
        if record_path.resolve() != CANDIDATE_RECORD.resolve():
            raise ValueError("CORE-63 reader refuses every non-allowlisted JSON record")
        with record_path.open("rb", buffering=0) as stream:
            value, consumed = extract_component_basis(stream)
        entry.update({"bytes_consumed": consumed, "status": "PASS"})
        return value
    except Exception as exc:
        entry.update({"error": type(exc).__name__, "status": "REJECTED"})
        raise
    finally:
        _append_access(access_log, entry)


def read_standard_decode_kernels(
    record_path: Path,
    kernelprobe_root: Path,
    access_log: Path,
) -> list[dict[str, str]]:
    """Stream the sole allowlisted calibration kernel summary and log it."""

    entry: dict[str, Any] = {
        "schema": ACCESS_SCHEMA,
        "classification": "retained_standard_decode_kernel_decomposition",
        "record": KERNEL_SUMMARY_LABEL,
        "record_sha256_from_published_catalog": (
            KERNEL_SUMMARY_SHA256_FROM_PUBLISHED_CATALOG
        ),
        "selector": KERNEL_SUMMARY_SELECTOR,
        "fields": list(CAPTURED_KERNEL_FIELDS),
        "whole_record_loaded": False,
        "held_out_numeric_value_accessed": False,
        "unselected_payload_fields_decoded": False,
    }
    try:
        expected = kernelprobe_root.resolve() / KERNEL_SUMMARY_RELATIVE
        if record_path.resolve() != expected.resolve():
            raise ValueError("CORE-63 reader refuses every non-allowlisted CSV record")
        if record_path.stat().st_size != KERNEL_SUMMARY_BYTES_FROM_PUBLISHED_CATALOG:
            raise ValueError("kernel summary size differs from its published catalog")
        with record_path.open("rb", buffering=0) as stream:
            rows, consumed = extract_standard_decode_kernels(stream)
        entry.update(
            {
                "bytes_consumed": consumed,
                "selected_row_count": len(rows),
                "status": "PASS",
            }
        )
        return rows
    except Exception as exc:
        entry.update({"error": type(exc).__name__, "status": "REJECTED"})
        raise
    finally:
        _append_access(access_log, entry)


def read_core63_basis(
    *,
    candidate_record: Path,
    kernel_summary: Path,
    kernelprobe_root: Path,
    access_log: Path,
) -> dict[str, Any]:
    """Perform exactly the two preregistered calibration-only accesses."""

    return {
        "component_basis": read_component_basis(candidate_record, access_log),
        "kernel_rows": read_standard_decode_kernels(
            kernel_summary,
            kernelprobe_root,
            access_log,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--access-log", required=True, type=Path)
    parser.add_argument("--kernelprobe-root", type=Path)
    args = parser.parse_args()
    root = args.kernelprobe_root
    if root is None:
        configured = os.environ.get("SIMLLM_KERNELPROBE_ROOT")
        if not configured:
            raise ValueError("set SIMLLM_KERNELPROBE_ROOT or pass --kernelprobe-root")
        root = Path(configured)
    value = read_core63_basis(
        candidate_record=CANDIDATE_RECORD,
        kernel_summary=root / KERNEL_SUMMARY_RELATIVE,
        kernelprobe_root=root,
        access_log=args.access_log,
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
