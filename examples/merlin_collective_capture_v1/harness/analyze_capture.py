#!/usr/bin/env python3
"""Normalize fetched TRAF-77 captures and evaluate frozen directions offline.

This is the T2A analysis skeleton, not the T2B scored study runner. It reads
capture JSONL, NCCL logs and counter snapshots, emits one normalized row per
cell, checks FG-4 before exposing the remaining rows to directional evaluation,
and reports E1 through E4 without claiming a hardware result when inputs are
synthetic or incomplete.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).with_name("study_config.json")


@dataclass(frozen=True, order=True)
class CellIdentity:
    """The complete frozen coordinate of one normalized capture row."""

    width: int
    concentration: str
    operation: str
    payload_bytes: int

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> CellIdentity:
        return cls(
            width=int(row["width"]),
            concentration=str(row["concentration"]),
            operation=str(row["operation"]),
            payload_bytes=int(row["payload_bytes"]),
        )

    @property
    def cell_id(self) -> str:
        return (
            f"w{self.width}/{self.concentration}/"
            f"{self.operation}/{self.payload_bytes}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def counter_delta(before: int, after: int, bits: int = 64) -> int:
    """Return a monotonic counter delta, including one unsigned wrap."""

    if before < 0 or after < 0:
        raise ValueError("counter values must be nonnegative")
    limit = 1 << bits
    if before >= limit or after >= limit:
        raise ValueError(f"counter values do not fit in {bits} bits")
    return after - before if after >= before else limit - before + after


def diff_numeric_counters(
    before: Any,
    after: Any,
    prefix: str = "",
) -> list[dict[str, Any]]:
    """Flatten matching integer leaves into before, after and delta rows."""

    if isinstance(before, dict) and isinstance(after, dict):
        rows: list[dict[str, Any]] = []
        for name in sorted(set(before) & set(after)):
            child = f"{prefix}.{name}" if prefix else name
            rows.extend(diff_numeric_counters(before[name], after[name], child))
        return rows
    if isinstance(before, int) and not isinstance(before, bool) and isinstance(after, int):
        return [
            {
                "counter": prefix,
                "before": before,
                "after": after,
                "delta": counter_delta(before, after),
                "wrapped": after < before,
            }
        ]
    return []


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path.name}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"JSON row at {path.name}:{line_number} is not an object")
        rows.append(row)
    return rows


def _cell_counter_deltas(row: dict[str, Any]) -> dict[str, Any]:
    """Use local rank zero as the single node-level per-cell counter reader."""

    nodes: dict[str, Any] = {}
    rank_rows = sorted(
        row.get("rank_counters", []),
        key=lambda item: (int(item.get("local_rank", 0)), int(item["rank"])),
    )
    for rank_row in rank_rows:
        host = str(rank_row["host"])
        if host in nodes:
            continue
        ports = {}
        for port in rank_row["ports"]:
            ports[str(port["interface"])] = {
                "rx_delta_bytes": counter_delta(
                    int(port["before_rx_bytes"]), int(port["after_rx_bytes"])
                ),
                "tx_delta_bytes": counter_delta(
                    int(port["before_tx_bytes"]), int(port["after_tx_bytes"])
                ),
            }
        nodes[host] = {
            "reader_rank": int(rank_row["rank"]),
            "before_monotonic_raw_ns": int(rank_row["before_monotonic_raw_ns"]),
            "after_monotonic_raw_ns": int(rank_row["after_monotonic_raw_ns"]),
            "ports": ports,
        }
    return nodes


def _routing_observation(
    concentration: str,
    nodes: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Classify one counter window without pooling directional fractions."""

    rule = config["routing_proof"]
    observations = {}
    for node, node_row in sorted(nodes.items()):
        directional_traffic = {
            direction: {
                interface: int(values[f"{direction}_delta_bytes"])
                for interface, values in node_row["ports"].items()
                if interface in config["interfaces"]
            }
            for direction in ("tx", "rx")
        }
        total = sum(
            sum(traffic.values()) for traffic in directional_traffic.values()
        )
        enough_signal = total >= int(rule["minimum_total_delta_bytes"])
        directions = {}
        for direction, traffic in directional_traffic.items():
            direction_total = sum(traffic.values())
            fractions = {
                interface: value / direction_total if direction_total else 0.0
                for interface, value in sorted(traffic.items())
            }
            if concentration == "one-port":
                matches = fractions.get("hsn0", 0.0) >= float(
                    rule["one_port_primary_min_fraction"]
                )
            else:
                matches = all(
                    fractions.get(interface, 0.0) >= float(
                        rule["four_port_each_min_fraction"]
                    )
                    for interface in config["interfaces"]
                )
            directions[direction] = {
                "total_delta_bytes": direction_total,
                "fractions": fractions,
                "status": (
                    "insufficient-signal"
                    if not enough_signal
                    else ("proven" if matches else "contradicted")
                ),
            }
        direction_statuses = {row["status"] for row in directions.values()}
        if "contradicted" in direction_statuses:
            status = "contradicted"
        elif directions and direction_statuses == {"proven"}:
            status = "proven"
        else:
            status = "insufficient-signal"
        observations[node] = {
            "total_rx_plus_tx_delta_bytes": total,
            "directions": directions,
            "status": status,
        }
    statuses = {row["status"] for row in observations.values()}
    if "contradicted" in statuses:
        status = "contradicted"
    elif observations and statuses == {"proven"}:
        status = "proven"
    else:
        status = "insufficient-signal"
    return {"status": status, "nodes": observations}


def normalize_cell(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    identity = CellIdentity.from_row(row)
    durations = [int(sample["max_rank_duration_ns"]) for sample in row["samples"]]
    expected_repeats = int(config["measured_repeats"])
    if len(durations) != expected_repeats:
        raise ValueError(
            f"{identity.cell_id} has {len(durations)} repeats, expected {expected_repeats}"
        )

    chunk_count = int(row["chunk_count"])
    chunk_medians = []
    for chunk_index in range(chunk_count):
        per_repeat = []
        for sample in row["samples"]:
            per_rank = [
                int(rank["chunk_completions"][chunk_index]["elapsed_ns"])
                for rank in sample["ranks"]
            ]
            per_repeat.append(max(per_rank))
        chunk_medians.append(percentile(per_repeat, 0.5))
    chunk_spacings = [
        later - earlier
        for earlier, later in itertools.pairwise([0.0, *chunk_medians])
    ]

    port_count = 1 if identity.concentration == "one-port" else 4
    payload_floor_ns = math.ceil(
        identity.payload_bytes
        / (int(config["port_rate_bytes_per_second"]) * port_count)
        * 1_000_000_000
    )
    median_ns = percentile(durations, 0.5)
    per_cell_port_deltas = _cell_counter_deltas(row)
    return {
        "cell_id": identity.cell_id,
        **identity.to_dict(),
        "evidence_class": row["evidence_class"],
        "attempt_id": row["attempt_id"],
        "slurm_job_id": row["slurm_job_id"],
        "submitted_script": row["submitted_script"],
        "submitted_script_sha256": row["submitted_script_sha256"],
        "config_sha256": row["config_sha256"],
        "payload_semantics": row["payload_semantics"],
        "clock": row["clock"],
        "clock_epoch_scope": row["clock_epoch_scope"],
        "measured_repeats": int(row["measured_repeats"]),
        "excluded_warmups": int(row["excluded_warmups"]),
        "chunk_limit_bytes": int(row["chunk_limit_bytes"]),
        "chunk_count": chunk_count,
        "completion_ns": {
            "median": median_ns,
            "p95": percentile(durations, 0.95),
            "minimum": min(durations),
            "maximum": max(durations),
        },
        "chunk_completion_median_elapsed_ns": chunk_medians,
        "chunk_spacing_median_ns": (
            statistics.median(chunk_spacings) if chunk_spacings else None
        ),
        "chunk_completion_monotonic": all(
            later > earlier for earlier, later in itertools.pairwise(chunk_medians)
        ),
        "per_cell_port_deltas": per_cell_port_deltas,
        "per_cell_routing_observation": _routing_observation(
            identity.concentration, per_cell_port_deltas, config
        ),
        "max_rank_mismatches": int(row["max_rank_mismatches"]),
        "serialization_floor_ns": payload_floor_ns,
        "median_over_serialization_floor": median_ns / max(payload_floor_ns, 1),
        "fg4_anchor_ps": row.get("fg4_anchor_ps"),
        "fg4_anchor_ratio": row.get("fg4_anchor_ratio"),
        "fg4_anchor_held_in_lane": row.get("fg4_anchor_held"),
    }


def _load_counter_snapshots(attempt_dir: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    evidence_classes = set()
    for path in sorted((attempt_dir / "counter_snapshots").glob("*.jsonl")):
        for row in _read_json_lines(path):
            evidence_classes.add(str(row["evidence_class"]))
            key = (str(row["node"]), str(row["interface"]))
            grouped.setdefault(key, {})[str(row["tag"])] = row

    nodes: dict[str, Any] = {}
    for (node, interface), snapshots in sorted(grouped.items()):
        if "before" not in snapshots or "after" not in snapshots:
            continue
        before = snapshots["before"]
        after = snapshots["after"]
        sysfs_before = before["sources"]["sysfs_statistics"]
        sysfs_after = after["sources"]["sysfs_statistics"]
        nodes.setdefault(node, {})[interface] = {
            "rx_delta_bytes": counter_delta(
                int(sysfs_before["rx_bytes"]), int(sysfs_after["rx_bytes"])
            ),
            "tx_delta_bytes": counter_delta(
                int(sysfs_before["tx_bytes"]), int(sysfs_after["tx_bytes"])
            ),
            "all_numeric_deltas": diff_numeric_counters(
                before["sources"], after["sources"]
            ),
        }
    return {
        "evidence_classes": sorted(evidence_classes),
        "nodes": nodes,
    }


def _parse_nccl_logs(attempt_dir: Path) -> dict[str, Any]:
    lines = []
    paths = sorted(attempt_dir.glob("nccl_debug.*.log"))
    selection = attempt_dir / "nccl_selection.txt"
    if selection.is_file():
        paths.append(selection)
    for path in paths:
        lines.extend(path.read_text(encoding="utf-8", errors="replace").splitlines())
    text = "\n".join(lines)
    algorithms = sorted(
        set(
            re.findall(
                r"(?:Algo|algorithm)[ /:=]+([A-Za-z0-9_-]+)",
                text,
                re.IGNORECASE,
            )
        )
    )
    protocols = sorted(
        set(
            re.findall(
                r"(?:Proto|protocol)[ /:=]+([A-Za-z0-9_-]+)",
                text,
                re.IGNORECASE,
            )
        )
    )
    coll_channels = sorted(
        {int(value) for value in re.findall(r"(\d+) coll channels", text)}
    )
    return {
        "network_plugins": sorted(set(re.findall(r"Using network ([A-Za-z0-9_-]+)", text))),
        "algorithms": algorithms,
        "protocols": protocols,
        "collective_channel_counts": coll_channels,
        "selected_interfaces": sorted(set(re.findall(r"\bhsn[0-3]\b", text))),
        "socket_devices": sorted(
            {int(value) for value in re.findall(r"NET/Socket/(\d+)", text)}
        ),
        "gdr_states": sorted(
            {int(value) for value in re.findall(r"\bGDR (\d+)\b", text)}
        ),
        "selection_line_count": len(lines),
    }


def _routing_proof(
    concentration: str,
    counters: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    node_shape = {
        node: {"ports": ports}
        for node, ports in counters["nodes"].items()
    }
    observation = _routing_observation(concentration, node_shape, config)
    return {
        "concentration": concentration,
        "held": observation["status"] == "proven",
        **observation,
    }


def _fg4(normalized: dict[CellIdentity, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Recheck four-port anchor cells before any directional row is evaluated."""

    rows = []
    held = True
    low, high = (float(value) for value in config["anchor_factor_band"])
    for anchor in config["anchors_ps"]:
        identity = CellIdentity(
            width=int(anchor["width"]),
            concentration="four-port",
            operation=str(anchor["operation"]),
            payload_bytes=int(anchor["payload_bytes"]),
        )
        row = normalized.get(identity)
        if row is None:
            rows.append({"cell_id": identity.cell_id, "status": "missing"})
            held = False
            continue
        ratio = row["completion_ns"]["median"] * 1000.0 / int(anchor["completion_ps"])
        matched = low <= ratio <= high
        held = held and matched and row["max_rank_mismatches"] == 0
        rows.append(
            {
                "cell_id": identity.cell_id,
                "status": "held" if matched else "miss",
                "anchor_ps": int(anchor["completion_ps"]),
                "observed_median_ns": row["completion_ns"]["median"],
                "ratio": ratio,
                "factor_band": [low, high],
            }
        )
    return {"id": "FG-4", "held": held, "rows": rows}


def _outcome(identifier: str, rows: list[dict[str, Any]], complete: bool) -> dict[str, Any]:
    passed = complete and bool(rows) and all(bool(row["held"]) for row in rows)
    return {
        "id": identifier,
        "status": "PASS" if passed else ("REFUTED" if complete and rows else "unevaluated"),
        "complete": complete,
        "rows": rows,
    }


def evaluate_directions(
    normalized: dict[CellIdentity, dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    payloads = [int(value) for value in config["payload_bytes"]]
    operations = [str(value) for value in config["operations"]]
    large_min = int(config["large_payload_min_bytes"])

    e1_rows = []
    e1_complete = True
    for operation in operations:
        for payload in payloads:
            one = normalized.get(CellIdentity(8, "one-port", operation, payload))
            four = normalized.get(CellIdentity(8, "four-port", operation, payload))
            if one is None or four is None:
                e1_complete = False
                continue
            ratio = one["completion_ns"]["median"] / four["completion_ns"]["median"]
            floor = 2.0 if payload >= large_min else 1.0
            e1_rows.append(
                {
                    "operation": operation,
                    "payload_bytes": payload,
                    "one_over_four": ratio,
                    "required_minimum": floor,
                    "held": ratio >= floor,
                }
            )

    e2_rows = []
    e2_complete = True
    for concentration in config["concentrations"]:
        for payload in payloads:
            width2 = normalized.get(
                CellIdentity(2, str(concentration), "all_reduce", payload)
            )
            width8 = normalized.get(
                CellIdentity(8, str(concentration), "all_reduce", payload)
            )
            if width2 is None or width8 is None:
                e2_complete = False
                continue
            ratio = width8["completion_ns"]["median"] / width2["completion_ns"]["median"]
            e2_rows.append(
                {
                    "concentration": concentration,
                    "payload_bytes": payload,
                    "width8_over_width2": ratio,
                    "held": ratio > 1.0,
                }
            )

    e3_rows = []
    e3_complete = True
    for operation in operations:
        for payload in (value for value in payloads if value >= large_min):
            one = normalized.get(CellIdentity(8, "one-port", operation, payload))
            four = normalized.get(CellIdentity(8, "four-port", operation, payload))
            if one is None or four is None:
                e3_complete = False
                continue
            one_spacing = float(one["chunk_spacing_median_ns"])
            four_spacing = float(four["chunk_spacing_median_ns"])
            e3_rows.append(
                {
                    "operation": operation,
                    "payload_bytes": payload,
                    "one_port_spacing_ns": one_spacing,
                    "four_port_spacing_ns": four_spacing,
                    "spacing_ratio": one_spacing / four_spacing,
                    "one_port_completion_monotonic": one[
                        "chunk_completion_monotonic"
                    ],
                    "held": one["chunk_completion_monotonic"]
                    and one_spacing >= four_spacing,
                }
            )

    e4_rows = []
    e4_complete = True
    low, high = (float(value) for value in config["anchor_factor_band"])
    for anchor in config["anchors_ps"]:
        for concentration in config["concentrations"]:
            identity = CellIdentity(
                int(anchor["width"]),
                str(concentration),
                str(anchor["operation"]),
                int(anchor["payload_bytes"]),
            )
            row = normalized.get(identity)
            if row is None:
                e4_complete = False
                continue
            ratio = row["completion_ns"]["median"] * 1000.0 / int(
                anchor["completion_ps"]
            )
            e4_rows.append(
                {
                    "cell_id": identity.cell_id,
                    "observed_over_anchor": ratio,
                    "factor_band": [low, high],
                    "held": low <= ratio <= high,
                }
            )

    return [
        _outcome("E1", e1_rows, e1_complete),
        _outcome("E2", e2_rows, e2_complete),
        _outcome("E3", e3_rows, e3_complete),
        _outcome("E4", e4_rows, e4_complete),
    ]


def analyze(capture_root: Path) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    capture_paths = sorted(capture_root.rglob("capture.jsonl"))
    normalized_by_identity: dict[CellIdentity, list[dict[str, Any]]] = {}
    attempts = []
    evidence_classes = set()

    for capture_path in capture_paths:
        attempt_dir = capture_path.parent
        raw_rows = _read_json_lines(capture_path)
        if not raw_rows:
            continue
        attempt_classes = {str(row["evidence_class"]) for row in raw_rows}
        evidence_classes.update(attempt_classes)
        attempt_concentrations = {str(row["concentration"]) for row in raw_rows}
        if len(attempt_concentrations) != 1:
            raise ValueError(f"mixed concentrations in {attempt_dir.name}")
        counters = _load_counter_snapshots(attempt_dir)
        logs = _parse_nccl_logs(attempt_dir)
        concentration = next(iter(attempt_concentrations))
        routing_proof = _routing_proof(concentration, counters, config)
        attempt = {
            "attempt_id": str(raw_rows[0]["attempt_id"]),
            "directory_label": attempt_dir.name,
            "evidence_classes": sorted(attempt_classes),
            "raw_cell_count": len(raw_rows),
            "counter_snapshots": counters,
            "routing_proof": routing_proof,
            "nccl": logs,
        }
        attempts.append(attempt)
        for raw_row in raw_rows:
            identity = CellIdentity.from_row(raw_row)
            normalized = normalize_cell(raw_row, config)
            normalized["nccl_selection"] = {
                "scope": "attempt",
                **logs,
            }
            normalized["attempt_routing_proof"] = {
                "scope": "attempt",
                **routing_proof,
            }
            normalized_by_identity.setdefault(identity, []).append(normalized)

    unique_normalized = {
        identity: rows[0]
        for identity, rows in normalized_by_identity.items()
        if len(rows) == 1
    }
    duplicate_cells = sorted(
        identity.cell_id
        for identity, rows in normalized_by_identity.items()
        if len(rows) > 1
    )
    fg4 = _fg4(unique_normalized, config)
    directions = evaluate_directions(unique_normalized, config)
    expected = {
        CellIdentity(width, concentration, operation, payload).cell_id
        for width in config["widths"]
        for concentration in config["concentrations"]
        for operation in config["operations"]
        for payload in config["payload_bytes"]
    }
    observed = {identity.cell_id for identity in normalized_by_identity}
    fg2_held = bool(attempts) and all(
        attempt["routing_proof"]["held"] for attempt in attempts
    )

    return {
        "schema": "simllm-merlin-collective-normalized-evidence-v1",
        "study": config["study"],
        "analysis_scope": "T2A offline skeleton; T2B owns scored publication",
        "t2b_scored": False,
        "evidence_classes": sorted(evidence_classes),
        "input_root_label": capture_root.name,
        "coverage": {
            "expected_cells": len(expected),
            "observed_cells": len(observed),
            "missing_cell_ids": sorted(expected - observed),
            "unexpected_cell_ids": sorted(observed - expected),
            "duplicate_cell_ids": duplicate_cells,
            "complete": observed == expected and not duplicate_cells,
        },
        "fatal_guards": [
            {
                "id": "FG-2",
                "held": fg2_held,
                "detail": "routing concentration proven from fetched port counters",
            },
            fg4,
        ],
        "attempts": sorted(attempts, key=lambda row: row["attempt_id"]),
        "normalized_rows": [
            row
            for identity in sorted(normalized_by_identity)
            for row in sorted(
                normalized_by_identity[identity], key=lambda item: item["attempt_id"]
            )
        ],
        "directional_evaluations": directions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    result = analyze(args.capture_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
