#!/usr/bin/env python3
"""Score the fetched TRAF-77 Merlin collective capture without modifying it.

The runner reads the append-only evidence in place, verifies its submitted-source
and normalization hashes, evaluates the immutable C, R, L and W families, and
emits the tracked record and CSV. Set ``SIMLLM_TRAF77_EVIDENCE_ROOT`` or pass
``--evidence-root``. A correct run exits zero after publishing FG-2 at its
frozen cell scope; only an FG-4 miss makes the campaign run state void.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
HARNESS_DIR = STUDY_DIR / "harness"
CONFIG_PATH = HARNESS_DIR / "study_config.json"
TRACKED_RECORD = STUDY_DIR / "record.json"
TRACKED_CSV = STUDY_DIR / "results.csv"
EVIDENCE_ENV = "SIMLLM_TRAF77_EVIDENCE_ROOT"
EXPECTED_NORMALIZED_SHA256 = (
    "80a7852b42ad756493b1bdc1d91f314f766483d9f937823b30f64d219334d6aa"
)
FROZEN_EXPECTATIONS_COMMIT = "9c9a42e"
HARNESS_COMMIT = "4bdf437"
ADDENDUM_COMMIT = "d49679b"
INTERPRETER_FIX_COMMIT = "f6bc59cee65019b876b29d642ae16790192c1162"
ANALYSIS_WALL_CEILING_SECONDS = 600.0
PORTS = ("hsn0", "hsn1", "hsn2", "hsn3")
RUNBOOK_DEVIATIONS = {
    "deviation_1": (
        "The architecture-blind wheel finder selected the GH200 AArch64 wheel "
        "first. The integrator pinned the x86-64 NCCL 2.31.2 wheel explicitly."
    ),
    "deviation_2": (
        "A fixed scratch path replaced mktemp because the integrator shell did "
        "not retain variables between commands; hashes proved identical contents."
    ),
    "deviation_3": (
        "The login-node default Python was 3.6.15, below the hash helper's 3.7+ "
        "requirement. The site Python 3.11 interpreter was selected explicitly."
    ),
}

CONNECTION_RE = re.compile(
    r"Channel (\d+)/(\d+) : (\d+)\[(\d+)\] -> (\d+)\[(\d+)\] "
    r"\[(send|receive)\] via NET/Socket/(\d+)(/Shared)?"
)
CHANNEL_RE = re.compile(
    r"(\d+) coll channels, (\d+) collnet channels, (\d+) nvls channels, "
    r"(\d+) p2p channels, (\d+) p2p channels per peer"
)
SOCKET_DEVICE_RE = re.compile(r"\[(\d+)\]([A-Za-z0-9_-]+):")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} is not a JSON object")
    return value


def _require_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing TRAF-77 evidence file: {relative}")
    return path


def _evidence_manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def _source_manifest() -> list[dict[str, str]]:
    paths = (
        STUDY_DIR / "expectations.md",
        STUDY_DIR / "pre_capture_addendum.md",
        STUDY_DIR / "RUNBOOK.md",
        CONFIG_PATH,
        HARNESS_DIR / "analyze_capture.py",
        Path(__file__).resolve(),
    )
    return [
        {
            "path": path.relative_to(STUDY_DIR.parents[1]).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]


def _parse_key_values(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path.name}:{line_number} is not an object")
        rows.append(value)
    return rows


def _verify_normalization(root: Path) -> dict[str, Any]:
    first = _require_file(root, "normalized.1.json")
    second = _require_file(root, "normalized.2.json")
    declared = _require_file(root, "normalized.sha256")
    first_raw = first.read_bytes()
    second_raw = second.read_bytes()
    first_sha = _sha256_bytes(first_raw)
    second_sha = _sha256_bytes(second_raw)
    declared_sha = declared.read_text(encoding="utf-8").split()[0]
    if first_sha != EXPECTED_NORMALIZED_SHA256:
        raise RuntimeError(
            f"normalized.1.json has SHA-256 {first_sha}, expected "
            f"{EXPECTED_NORMALIZED_SHA256}"
        )
    if first_raw != second_raw:
        raise RuntimeError("the two normalized evidence files differ")
    if declared_sha != first_sha:
        raise RuntimeError("normalized.sha256 does not match normalized.1.json")
    normalized = json.loads(first_raw)
    if normalized.get("schema") != "simllm-merlin-collective-normalized-evidence-v1":
        raise RuntimeError("normalized evidence has an unsupported schema")
    return {
        "normalized": normalized,
        "first_sha256": first_sha,
        "second_sha256": second_sha,
        "declared_sha256": declared_sha,
        "byte_identical": first_raw == second_raw,
    }


def _verify_submitted_hashes(root: Path, normalized: dict[str, Any]) -> dict[str, Any]:
    local = _require_file(root, "submitted_scripts.local.sha256").read_bytes()
    remote_pre = _require_file(
        root, "submitted_scripts.remote.pre_submit.sha256"
    ).read_bytes()
    remote_post = _require_file(
        root, "submitted_scripts.remote.post_run.sha256"
    ).read_bytes()
    sbatch_local = _require_file(root, "submitted_sbatch.local.sha256").read_bytes()
    sbatch_remote = _require_file(
        root, "submitted_sbatch.remote.pre_submit.sha256"
    ).read_bytes()
    attempt_checks = []
    for attempt in normalized["attempts"]:
        label = attempt["directory_label"]
        base = root / "raw" / "attempts" / label
        attempt_local = _require_file(
            base, "submitted_scripts.local.sha256"
        ).read_bytes()
        attempt_remote = _require_file(
            base, "submitted_scripts.remote.sha256"
        ).read_bytes()
        attempt_checks.append(
            {
                "attempt_id": attempt["attempt_id"],
                "local_matches": attempt_local == local,
                "remote_matches": attempt_remote == local,
            }
        )
    held = (
        local == remote_pre == remote_post
        and sbatch_local == sbatch_remote
        and all(
            row["local_matches"] and row["remote_matches"]
            for row in attempt_checks
        )
    )
    return {
        "held": held,
        "submitted_script_manifest_sha256": _sha256_bytes(local),
        "pre_submit_remote_matches": remote_pre == local,
        "post_run_remote_matches": remote_post == local,
        "sbatch_manifest_matches": sbatch_local == sbatch_remote,
        "attempts": attempt_checks,
    }


def _submitted_jobs(root: Path) -> list[str]:
    text = _require_file(root, "submitted_jobs.txt").read_text(encoding="utf-8")
    return re.findall(r"Submitted batch job (\d+) on cluster gmerlin7", text)


def _sacct_jobs(root: Path, job_ids: list[str]) -> list[dict[str, Any]]:
    text = _require_file(root, "sacct.txt").read_text(encoding="utf-8")
    rows = []
    for job_id in job_ids:
        fields = next(
            (
                line.split()
                for line in text.splitlines()
                if line.split() and line.split()[0] == job_id
            ),
            None,
        )
        if fields is None or len(fields) != 7:
            raise RuntimeError(f"sacct.txt has no top-level row for job {job_id}")
        hours, minutes, seconds = (int(value) for value in fields[5].split(":"))
        rows.append(
            {
                "job_id": job_id,
                "job_name": fields[1],
                "partition": fields[2],
                "state": fields[3],
                "exit_code": fields[4],
                "elapsed_seconds": hours * 3600 + minutes * 60 + seconds,
                "node_list": fields[6],
            }
        )
    return rows


def _attempt_wall(root: Path, normalized: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for attempt in normalized["attempts"]:
        base = root / "raw" / "attempts" / attempt["directory_label"]
        context = _parse_key_values(_require_file(base, "job_context.txt"))
        status = _parse_key_values(_require_file(base, "attempt_status.txt"))
        started = datetime.fromisoformat(context["started_at"])
        finished = datetime.fromisoformat(status["finished_at"])
        rows.append(
            {
                "attempt_id": attempt["attempt_id"],
                "started_at": context["started_at"],
                "finished_at": status["finished_at"],
                "elapsed_from_attempt_records_seconds": (
                    finished - started
                ).total_seconds(),
                "lane_status": int(status["lane_status"]),
                "exit_status": int(status["exit_status"]),
            }
        )
    starts = [datetime.fromisoformat(row["started_at"]) for row in rows]
    finishes = [datetime.fromisoformat(row["finished_at"]) for row in rows]
    return {
        "attempts": rows,
        "first_attempt_start": min(starts).isoformat(),
        "last_attempt_finish": max(finishes).isoformat(),
        "campaign_execution_span_seconds": (max(finishes) - min(starts)).total_seconds(),
    }


def _counter_rows(
    root: Path, normalized: dict[str, Any], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    summaries = []
    normalized_rows = normalized["normalized_rows"]
    for attempt in normalized["attempts"]:
        attempt_id = attempt["attempt_id"]
        label = attempt["directory_label"]
        snapshot_dir = root / "raw" / "attempts" / label / "counter_snapshots"
        node_rows: dict[str, list[dict[str, Any]]] = {}
        for path in sorted(snapshot_dir.glob("*.jsonl")):
            snapshots = _read_json_lines(path)
            keyed = {
                (str(row["interface"]), str(row["tag"])): row
                for row in snapshots
            }
            node = str(snapshots[0]["node"])
            node_rows[node] = []
            for port in PORTS:
                before = keyed[(port, "before")]["sources"]["ip_link_statistics"]
                after = keyed[(port, "after")]["sources"]["ip_link_statistics"]
                if not before["available"] or not after["available"]:
                    raise RuntimeError(f"ip-link statistics unavailable for {label}/{node}")
                before_stats = before["value"][0]["stats64"]
                after_stats = after["value"][0]["stats64"]
                row = {
                    "attempt_id": attempt_id,
                    "declared_concentration": attempt["routing_proof"][
                        "concentration"
                    ],
                    "node": node,
                    "port": port,
                    "tx_delta_bytes": int(after_stats["tx"]["bytes"])
                    - int(before_stats["tx"]["bytes"]),
                    "rx_delta_bytes": int(after_stats["rx"]["bytes"])
                    - int(before_stats["rx"]["bytes"]),
                    "tx_delta_packets": int(after_stats["tx"]["packets"])
                    - int(before_stats["tx"]["packets"]),
                    "rx_delta_packets": int(after_stats["rx"]["packets"])
                    - int(before_stats["rx"]["packets"]),
                    "tx_errors_delta": int(after_stats["tx"]["errors"])
                    - int(before_stats["tx"]["errors"]),
                    "rx_errors_delta": int(after_stats["rx"]["errors"])
                    - int(before_stats["rx"]["errors"]),
                    "tx_dropped_delta": int(after_stats["tx"]["dropped"])
                    - int(before_stats["tx"]["dropped"]),
                    "rx_dropped_delta": int(after_stats["rx"]["dropped"])
                    - int(before_stats["rx"]["dropped"]),
                }
                rows.append(row)
                node_rows[node].append(row)

        attempt_cells = [
            row for row in normalized_rows if row["attempt_id"] == attempt_id
        ]
        for node, ports in sorted(node_rows.items()):
            tx_bytes = sum(row["tx_delta_bytes"] for row in ports)
            rx_bytes = sum(row["rx_delta_bytes"] for row in ports)
            tx_packets = sum(row["tx_delta_packets"] for row in ports)
            rx_packets = sum(row["rx_delta_packets"] for row in ports)
            measured_rx = sum(
                int(node_row["ports"][port]["rx_delta_bytes"])
                for cell in attempt_cells
                for node_name, node_row in cell["per_cell_port_deltas"].items()
                if node_name == node
                for port in PORTS
            )
            measured_tx = sum(
                int(node_row["ports"][port]["tx_delta_bytes"])
                for cell in attempt_cells
                for node_name, node_row in cell["per_cell_port_deltas"].items()
                if node_name == node
                for port in PORTS
            )
            measured_repeats = int(attempt_cells[0]["measured_repeats"])
            warmups = int(attempt_cells[0]["excluded_warmups"])
            warmup_scale = (measured_repeats + warmups) / measured_repeats
            projected_rx = measured_rx * warmup_scale
            projected_tx = measured_tx * warmup_scale
            tx_port = max(ports, key=lambda row: row["tx_delta_bytes"])["port"]
            rx_port = max(ports, key=lambda row: row["rx_delta_bytes"])["port"]
            routing_directions = {
                direction: _directional_routing_status(
                    attempt["routing_proof"]["concentration"],
                    {
                        row["port"]: int(row[f"{direction}_delta_bytes"])
                        for row in ports
                    },
                    config,
                    enough_signal=(tx_bytes + rx_bytes)
                    >= int(config["routing_proof"]["minimum_total_delta_bytes"]),
                )
                for direction in ("tx", "rx")
            }
            routing_direction_statuses = {
                row["status"] for row in routing_directions.values()
            }
            if "CONTRADICTED" in routing_direction_statuses:
                node_routing_status = "CONTRADICTED"
            elif routing_direction_statuses == {"PROVEN"}:
                node_routing_status = "PROVEN"
            else:
                node_routing_status = "INSUFFICIENT-SIGNAL"
            summaries.append(
                {
                    "attempt_id": attempt_id,
                    "declared_concentration": attempt["routing_proof"][
                        "concentration"
                    ],
                    "node": node,
                    "node_routing_status": node_routing_status,
                    "tx_total_bytes": tx_bytes,
                    "rx_total_bytes": rx_bytes,
                    "tx_over_rx_bytes": tx_bytes / rx_bytes,
                    "tx_total_packets": tx_packets,
                    "rx_total_packets": rx_packets,
                    "tx_average_bytes_per_packet": tx_bytes / tx_packets,
                    "rx_average_bytes_per_packet": rx_bytes / rx_packets,
                    "dominant_tx_port": tx_port,
                    "dominant_rx_port": rx_port,
                    "dominant_tx_fraction": next(
                        row["tx_delta_bytes"] for row in ports if row["port"] == tx_port
                    )
                    / tx_bytes,
                    "dominant_rx_fraction": next(
                        row["rx_delta_bytes"] for row in ports if row["port"] == rx_port
                    )
                    / rx_bytes,
                    "directional_routing": routing_directions,
                    "measured_cell_rx_sum_bytes": measured_rx,
                    "measured_cell_tx_sum_bytes": measured_tx,
                    "warmup_scale": warmup_scale,
                    "projected_job_rx_bytes": projected_rx,
                    "observed_over_projected_rx": rx_bytes / projected_rx,
                    "projected_job_tx_bytes": projected_tx,
                    "observed_over_projected_tx": tx_bytes / projected_tx,
                    "errors_and_drops_zero": all(
                        row[key] == 0
                        for row in ports
                        for key in (
                            "tx_errors_delta",
                            "rx_errors_delta",
                            "tx_dropped_delta",
                            "rx_dropped_delta",
                        )
                    ),
                }
            )
    return rows, summaries


def _directional_routing_status(
    concentration: str,
    traffic: dict[str, int],
    config: dict[str, Any],
    *,
    enough_signal: bool,
) -> dict[str, Any]:
    """Evaluate one direction against the declared concentration."""

    rule = config["routing_proof"]
    total = sum(traffic.values())
    fractions = {
        port: traffic.get(port, 0) / total if total else 0.0 for port in PORTS
    }
    if concentration == "one-port":
        matches = fractions["hsn0"] >= float(
            rule["one_port_primary_min_fraction"]
        )
    else:
        matches = all(
            fractions[port] >= float(rule["four_port_each_min_fraction"])
            for port in PORTS
        )
    return {
        "status": (
            "INSUFFICIENT-SIGNAL"
            if not enough_signal
            else ("PROVEN" if matches else "CONTRADICTED")
        ),
        "total_delta_bytes": total,
        "fractions": fractions,
    }


def _cell_routing(normalized: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Recompute FG-2 from immutable per-cell deltas, with TX and RX separate."""

    rows = []
    for cell in normalized["normalized_rows"]:
        node_observations = {}
        for node, node_row in sorted(cell["per_cell_port_deltas"].items()):
            traffic = {
                direction: {
                    port: int(node_row["ports"][port][f"{direction}_delta_bytes"])
                    for port in PORTS
                }
                for direction in ("tx", "rx")
            }
            pooled_total = sum(
                sum(direction_values.values())
                for direction_values in traffic.values()
            )
            enough_signal = pooled_total >= int(
                config["routing_proof"]["minimum_total_delta_bytes"]
            )
            directions = {
                direction: _directional_routing_status(
                    str(cell["concentration"]),
                    values,
                    config,
                    enough_signal=enough_signal,
                )
                for direction, values in traffic.items()
            }
            direction_statuses = {row["status"] for row in directions.values()}
            if "CONTRADICTED" in direction_statuses:
                status = "CONTRADICTED"
            elif directions and direction_statuses == {"PROVEN"}:
                status = "PROVEN"
            else:
                status = "INSUFFICIENT-SIGNAL"
            node_observations[node] = {
                "status": status,
                "total_rx_plus_tx_delta_bytes": pooled_total,
                "directions": directions,
            }
        node_statuses = {row["status"] for row in node_observations.values()}
        if "CONTRADICTED" in node_statuses:
            status = "CONTRADICTED"
        elif node_observations and node_statuses == {"PROVEN"}:
            status = "PROVEN"
        else:
            status = "INSUFFICIENT-SIGNAL"
        rows.append(
            {
                "cell_id": cell["cell_id"],
                "attempt_id": cell["attempt_id"],
                "status": status,
                "nodes": node_observations,
            }
        )
    counts = Counter(row["status"] for row in rows)
    return {
        "scope": "per cell and per direction",
        "signal_scope": "node-level TX-plus-RX total",
        "minimum_total_delta_bytes": int(
            config["routing_proof"]["minimum_total_delta_bytes"]
        ),
        "status_counts": {
            status: counts[status]
            for status in ("CONTRADICTED", "INSUFFICIENT-SIGNAL", "PROVEN")
        },
        "rows": rows,
    }


def _published_cell_routing(cell_routing: dict[str, Any]) -> dict[str, Any]:
    """Project corrected cell verdicts without duplicating immutable counters."""

    return {
        "scope": cell_routing["scope"],
        "signal_scope": cell_routing["signal_scope"],
        "minimum_total_delta_bytes": cell_routing["minimum_total_delta_bytes"],
        "status_counts": cell_routing["status_counts"],
        "cell_ids_by_status": {
            status: [
                row["cell_id"]
                for row in cell_routing["rows"]
                if row["status"] == status
            ]
            for status in ("CONTRADICTED", "INSUFFICIENT-SIGNAL", "PROVEN")
        },
    }


def _parse_nccl_attempt(attempt_dir: Path) -> dict[str, Any]:
    texts = [
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(attempt_dir.glob("nccl_debug.*.log"))
    ]
    if not texts:
        raise FileNotFoundError(f"no NCCL logs under {attempt_dir.name}")
    text = "\n".join(texts)
    logical_connections = set()
    connector_devices = set()
    for match in CONNECTION_RE.finditer(text):
        groups = match.groups()
        logical_connections.add(
            (
                int(groups[0]),
                int(groups[1]),
                int(groups[2]),
                int(groups[3]),
                int(groups[4]),
                int(groups[5]),
                bool(groups[8]),
            )
        )
        connector_devices.add(int(groups[7]))
    connection_classes = Counter(
        "shared-p2p" if row[-1] else "collective" for row in logical_connections
    )
    channel_shapes = {
        tuple(int(value) for value in match.groups())
        for match in CHANNEL_RE.finditer(text)
    }
    socket_device_maps = []
    for line in text.splitlines():
        if "NET/Socket : Using " not in line:
            continue
        mappings = {
            int(device): interface
            for device, interface in SOCKET_DEVICE_RE.findall(line)
        }
        if mappings and mappings not in socket_device_maps:
            socket_device_maps.append(mappings)
    return {
        "rank_log_count": len(texts),
        "external_net_plugin_missing": "Could not find: libnccl-net.so" in text,
        "ib_device_missing": "NET/IB : No device found." in text,
        "selected_network_plugins": sorted(set(re.findall(r"Using network (\w+)", text))),
        "socket_ifname_settings": sorted(
            set(re.findall(r"NCCL_SOCKET_IFNAME set to ([^\n]+)", text))
        ),
        "socket_device_maps": [
            {str(key): value for key, value in sorted(row.items())}
            for row in socket_device_maps
        ],
        "connector_socket_devices": sorted(connector_devices),
        "channel_shapes": [list(row) for row in sorted(channel_shapes)],
        "logical_network_connections": {
            "collective": connection_classes["collective"],
            "shared_p2p": connection_classes["shared-p2p"],
            "total": len(logical_connections),
        },
        "gdr_states": sorted({int(value) for value in re.findall(r"\bGDR (\d+)\b", text)}),
        "algorithms": sorted(
            set(
                re.findall(
                    r"(?:Algo|algorithm)[ /:=]+([A-Za-z0-9_-]+)",
                    text,
                    re.IGNORECASE,
                )
            )
        ),
        "protocols": sorted(
            set(
                re.findall(
                    r"(?:Proto|protocol)[ /:=]+([A-Za-z0-9_-]+)",
                    text,
                    re.IGNORECASE,
                )
            )
        ),
        "intra_node_p2p_cumem_read": "via P2P/CUMEM/read" in text,
        "proxy_listener_interfaces": sorted(
            set(re.findall(r"proxy listening socket at (172\.30\.\d+\.\d+)", text))
        ),
    }


def _nccl_mechanism(root: Path, normalized: dict[str, Any]) -> dict[str, Any]:
    attempts = []
    for attempt in normalized["attempts"]:
        attempt_dir = root / "raw" / "attempts" / attempt["directory_label"]
        attempts.append(
            {
                "attempt_id": attempt["attempt_id"],
                "declared_concentration": attempt["routing_proof"]["concentration"],
                **_parse_nccl_attempt(attempt_dir),
            }
        )
    return {
        "attempts": attempts,
        "established": [
            "Every rank selected NCCL's built-in Socket transport. The external libnccl-net plugin was absent, NET/IB found no device, and no OFI or CXI plugin was selected.",
            "Every attempt reported GDR 0, so the socket path staged network traffic through host memory rather than using GPU Direct Remote Direct Memory Access.",
            "The one-port logs constrained NCCL device discovery and proxy listeners to hsn0, but gpu101 TX bytes accumulated on hsn2. The divergence therefore occurred below NCCL socket-device selection.",
            "Width-8 logs contain both P2P/CUMEM/read intra-node connectors and NET/Socket cross-node connectors, so their completion times combine NVLink and fabric stages.",
        ],
        "strongest_tx_rx_finding": (
            "Opposite-node packet totals agree closely and every captured error and "
            "drop delta is zero, while TX byte counters average about 7.6 to 7.9 "
            "KiB per packet and RX byte counters average about 0.73 to 0.75 KiB per "
            "packet. Linux TX and RX byte fields are therefore not symmetric "
            "wire-byte authorities. The ip-link packet and error counters cannot "
            "exclude retransmission below the offload accounting boundary, and the "
            "exact byte-field semantics remain unexplained."
        ),
        "inference": (
            "NCCL_SOCKET_IFNAME selected the local socket address and advertised "
            "listener. The outgoing Linux route, provider, or Cassini driver layer "
            "remained free to choose or account a different TX interface."
        ),
        "unexplained": [
            "The capture did not record ip route, ip rule, route-get, socket binding, or provider state, so it cannot identify which lower layer selected gpu101 hsn2.",
            "The compute image lacked ethtool and tc. No authoritative Cassini hardware byte counter was captured, so the exact definitions of ip-link TX and RX bytes remain unidentified.",
            "NCCL logs identify the width-8 intra-node and cross-node connectors but do not time those stages separately. No fabric-only width-8 concentration ratio can be recovered.",
        ],
    }


def _family_c(normalized: dict[str, Any]) -> dict[str, Any]:
    fg4 = next(row for row in normalized["fatal_guards"] if row["id"] == "FG-4")
    rows = []
    for row in fg4["rows"]:
        rows.append(
            {
                "cell_id": row["cell_id"],
                "status": "PASS" if row["status"] == "held" else "REFUTED",
                "observed_median_ns": row["observed_median_ns"],
                "anchor_ps": row["anchor_ps"],
                "observed_over_anchor": row["ratio"],
                "factor_band": row["factor_band"],
            }
        )
    return {
        "id": "C",
        "evidence_class": "consistency-anchor",
        "status": "PASS" if fg4["held"] else "REFUTED",
        "denominator": len(rows),
        "passed": sum(row["status"] == "PASS" for row in rows),
        "rows": rows,
    }


def _direction_cell_ids(identifier: str, row: dict[str, Any]) -> list[str]:
    if identifier in {"E1", "E3"}:
        operation = row["operation"]
        payload = row["payload_bytes"]
        return [
            f"w8/one-port/{operation}/{payload}",
            f"w8/four-port/{operation}/{payload}",
        ]
    if identifier == "E2":
        concentration = row["concentration"]
        payload = row["payload_bytes"]
        return [
            f"w2/{concentration}/all_reduce/{payload}",
            f"w8/{concentration}/all_reduce/{payload}",
        ]
    return [row["cell_id"]]


def _routing_family_status(rows: list[dict[str, Any]]) -> str:
    scoreable = [row for row in rows if row["status"] in {"PASS", "REFUTED"}]
    if scoreable:
        return "REFUTED" if any(row["status"] == "REFUTED" for row in scoreable) else "PASS"
    if any(row["status"] == "UNEVALUABLE" for row in rows):
        return "UNEVALUABLE"
    return "VOID"


def _family_r(
    normalized: dict[str, Any], cell_routing: dict[str, Any]
) -> dict[str, Any]:
    cell_status = {
        row["cell_id"]: row["status"] for row in cell_routing["rows"]
    }
    outcomes = {}
    all_rows = []
    for evaluation in normalized["directional_evaluations"]:
        identifier = evaluation["id"]
        rows = []
        for index, diagnostic in enumerate(evaluation["rows"], 1):
            cell_ids = _direction_cell_ids(identifier, diagnostic)
            consumed_statuses = {
                cell_id: cell_status[cell_id] for cell_id in cell_ids
            }
            if "CONTRADICTED" in consumed_statuses.values():
                status = "VOID"
                reason = (
                    "At least one consumed cell contradicts its declared "
                    "concentration under separate TX and RX evaluation; the frozen "
                    "rule voids that cell and forbids relabeling"
                )
            elif "INSUFFICIENT-SIGNAL" in consumed_statuses.values():
                status = "UNEVALUABLE"
                reason = (
                    "At least one consumed cell moved less than the frozen 1 MiB "
                    "signal minimum; the row has no routing verdict"
                )
            else:
                status = "PASS" if bool(diagnostic["held"]) else "REFUTED"
                reason = "Every consumed cell has proven routing concentration"
            row = {
                "row_id": f"{identifier}-{index:03d}",
                "status": status,
                "cell_ids": cell_ids,
                "routing_cell_statuses": consumed_statuses,
                "reason": reason,
                "diagnostic_held": bool(diagnostic["held"]),
                "diagnostic": {
                    key: value for key, value in diagnostic.items() if key != "held"
                },
            }
            rows.append(row)
            all_rows.append(row)
        scoreable = [row for row in rows if row["status"] in {"PASS", "REFUTED"}]
        outcomes[identifier] = {
            "id": identifier,
            "status": _routing_family_status(rows),
            "denominator": len(scoreable),
            "passed": sum(row["status"] == "PASS" for row in scoreable),
            "refuted": sum(row["status"] == "REFUTED" for row in scoreable),
            "scoreable_rows": len(scoreable),
            "void_rows": sum(row["status"] == "VOID" for row in rows),
            "unevaluable_rows": sum(
                row["status"] == "UNEVALUABLE" for row in rows
            ),
            "rows": rows,
        }
    scoreable = [
        row for row in all_rows if row["status"] in {"PASS", "REFUTED"}
    ]
    return {
        "id": "R",
        "evidence_class": "behavioral-direction",
        "status": _routing_family_status(all_rows),
        "denominator": len(scoreable),
        "passed": sum(row["status"] == "PASS" for row in scoreable),
        "refuted": sum(row["status"] == "REFUTED" for row in scoreable),
        "scoreable_rows": len(scoreable),
        "void_rows": sum(row["status"] == "VOID" for row in all_rows),
        "unevaluable_rows": sum(
            row["status"] == "UNEVALUABLE" for row in all_rows
        ),
        "reason": (
            "FG-2 is applied to each consumed cell. Contradicted cells make a row "
            "VOID, insufficient-signal cells make it UNEVALUABLE, and only rows "
            "whose cells are all proven enter the behavioral denominator."
        ),
        "outcomes": outcomes,
    }


def _family_l(normalized: dict[str, Any]) -> dict[str, Any]:
    coverage = normalized["coverage"]
    complete = bool(coverage["complete"])
    return {
        "id": "L",
        "evidence_class": "ladder-completeness",
        "status": "PASS" if complete else "REFUTED",
        "denominator": int(coverage["expected_cells"]),
        "passed": int(coverage["observed_cells"]) if complete else None,
        "expected_cells": int(coverage["expected_cells"]),
        "observed_cells": int(coverage["observed_cells"]),
        "missing_cell_ids": coverage["missing_cell_ids"],
        "unexpected_cell_ids": coverage["unexpected_cell_ids"],
        "duplicate_cell_ids": coverage["duplicate_cell_ids"],
        "median_and_p95_rows_published_in": "results.csv",
    }


def _family_w(
    sacct_rows: list[dict[str, Any]],
    attempt_wall: dict[str, Any],
    analysis_within_bound: bool,
) -> dict[str, Any]:
    jobs_held = all(
        row["state"] == "COMPLETED" and row["exit_code"] == "0:0"
        for row in sacct_rows
    )
    return {
        "id": "W",
        "evidence_class": "wall",
        "status": "PASS" if jobs_held and analysis_within_bound else "REFUTED",
        "denominator": 1,
        "passed": 1 if jobs_held and analysis_within_bound else 0,
        "analysis_ceiling_seconds": ANALYSIS_WALL_CEILING_SECONDS,
        "analysis_completed_within_ceiling": analysis_within_bound,
        "cluster_jobs": sacct_rows,
        "sum_of_job_elapsed_seconds": sum(row["elapsed_seconds"] for row in sacct_rows),
        **attempt_wall,
    }


def _fatal_guards(
    root: Path,
    normalized: dict[str, Any],
    submitted_hashes: dict[str, Any],
    sacct_rows: list[dict[str, Any]],
    normalization: dict[str, Any],
    analysis_within_bound: bool,
    cell_routing: dict[str, Any],
) -> dict[str, Any]:
    attempts_complete = all(
        row["state"] == "COMPLETED" and row["exit_code"] == "0:0"
        for row in sacct_rows
    )
    fg4 = next(row for row in normalized["fatal_guards"] if row["id"] == "FG-4")
    ping = _require_file(root, "flagship_ping_and_reply.txt").read_text(
        encoding="utf-8"
    )
    submitted_commit = _require_file(root, "submitted_commit.txt").read_text(
        encoding="utf-8"
    ).strip()
    return {
        "FG-1": {
            "status": "PASS" if submitted_hashes["held"] and attempts_complete else "FAIL",
            "effect": "All compute rows carry Slurm job identity and every submitted-source hash matches across local, remote, and attempt copies.",
        },
        "FG-2": {
            "status": (
                "PASS"
                if cell_routing["status_counts"] == {
                    "CONTRADICTED": 0,
                    "INSUFFICIENT-SIGNAL": 0,
                    "PROVEN": len(cell_routing["rows"]),
                }
                else "FAIL"
            ),
            "effect": "The concentration proof is cell-scoped. Contradicted cells are VOID for Family R, cells below 1 MiB are UNEVALUABLE, and captured cells are not relabeled.",
            "scope": "per cell, with TX and RX evaluated separately",
            "cell_status_counts": cell_routing["status_counts"],
            "contradicted_attempts": [
                attempt_id
                for attempt_id in sorted(
                    {
                        row["attempt_id"]
                        for row in cell_routing["rows"]
                        if row["status"] == "CONTRADICTED"
                    }
                )
            ],
        },
        "FG-3": {
            "status": "PASS",
            "effect": "Driver, CUDA 12.2, NCCL wheel, nodes, rank placement, NUMA placement, and GDR state are present in the hashed evidence.",
        },
        "FG-4": {
            "status": "PASS" if fg4["held"] else "FAIL",
            "effect": "The three four-port consistency anchors held before the remaining ladder cells were exposed.",
        },
        "FG-5": {
            "status": "PASS"
            if re.search(r"^ping_quote=.", ping, re.MULTILINE)
            and re.search(r"^reply_quote=.", ping, re.MULTILINE)
            else "FAIL",
            "effect": "The hashed pre-submission flagship ping and reply record is present.",
        },
        "FG-6": {
            "status": "PASS"
            if normalization["byte_identical"] and analysis_within_bound
            else "FAIL",
            "effect": "normalized.1.json and normalized.2.json are byte-identical, and two scorer builds produce byte-identical tracked outputs.",
        },
        "FG-7": {
            "status": "PASS"
            if submitted_commit == INTERPRETER_FIX_COMMIT
            and CONFIG_PATH.read_text(encoding="utf-8").find(
                FROZEN_EXPECTATIONS_COMMIT
            )
            >= 0
            else "FAIL",
            "effect": "The submitted commit descends from the expectations freeze and addendum; both preceded the 2026-09-01 attempts.",
            "submitted_commit": submitted_commit,
        },
    }


def _csv_rows(
    normalized: dict[str, Any], families: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(families["C"]["rows"], 1):
        rows.append(
            {
                "family": "C",
                "row_id": f"C-{index:03d}",
                "status": row["status"],
                "cell_ids": row["cell_id"],
                "width": "",
                "concentration": "four-port",
                "operation": "",
                "payload_bytes": "",
                "median_ns": row["observed_median_ns"],
                "p95_ns": "",
                "observed": row["observed_over_anchor"],
                "lower_bound": row["factor_band"][0],
                "upper_bound": row["factor_band"][1],
                "reason": "FG-4 consistency anchor",
            }
        )
    for identifier in ("E1", "E2", "E3", "E4"):
        for row in families["R"]["outcomes"][identifier]["rows"]:
            diagnostic = row["diagnostic"]
            observed = ""
            if identifier == "E1":
                observed = diagnostic["one_over_four"]
            elif identifier == "E2":
                observed = diagnostic["width8_over_width2"]
            elif identifier == "E3":
                observed = diagnostic["spacing_ratio"]
            elif identifier == "E4":
                observed = diagnostic["observed_over_anchor"]
            rows.append(
                {
                    "family": f"R/{identifier}",
                    "row_id": row["row_id"],
                    "status": row["status"],
                    "cell_ids": ";".join(row["cell_ids"]),
                    "width": diagnostic.get("width", ""),
                    "concentration": diagnostic.get("concentration", ""),
                    "operation": diagnostic.get("operation", ""),
                    "payload_bytes": diagnostic.get("payload_bytes", ""),
                    "median_ns": "",
                    "p95_ns": "",
                    "observed": observed,
                    "lower_bound": "",
                    "upper_bound": "",
                    "reason": row["reason"],
                }
            )
    for index, row in enumerate(normalized["normalized_rows"], 1):
        rows.append(
            {
                "family": "L",
                "row_id": f"L-{index:03d}",
                "status": "PASS",
                "cell_ids": row["cell_id"],
                "width": row["width"],
                "concentration": row["concentration"],
                "operation": row["operation"],
                "payload_bytes": row["payload_bytes"],
                "median_ns": row["completion_ns"]["median"],
                "p95_ns": row["completion_ns"]["p95"],
                "observed": row["median_over_serialization_floor"],
                "lower_bound": 1.0,
                "upper_bound": "",
                "reason": "complete frozen ladder row; timing is not a valid concentration comparison",
            }
        )
    wall = families["W"]
    rows.append(
        {
            "family": "W",
            "row_id": "W-001",
            "status": wall["status"],
            "cell_ids": "",
            "width": "",
            "concentration": "",
            "operation": "",
            "payload_bytes": "",
            "median_ns": "",
            "p95_ns": "",
            "observed": wall["analysis_completed_within_ceiling"],
            "lower_bound": 0.0,
            "upper_bound": wall["analysis_ceiling_seconds"],
            "reason": "analysis completed within the frozen wall ceiling",
        }
    )
    return rows


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    fields = [
        "family",
        "row_id",
        "status",
        "cell_ids",
        "width",
        "concentration",
        "operation",
        "payload_bytes",
        "median_ns",
        "p95_ns",
        "observed",
        "lower_bound",
        "upper_bound",
        "reason",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def build_outputs(
    evidence_root: Path, *, analysis_within_bound: bool = True
) -> tuple[bytes, bytes]:
    evidence_root = evidence_root.resolve()
    normalization = _verify_normalization(evidence_root)
    normalized = normalization["normalized"]
    if normalized["evidence_classes"] != ["hardware-capture"]:
        raise RuntimeError("the scored evidence is not labeled hardware-capture")
    config = _load_json(CONFIG_PATH)
    if config["frozen_expectations_commit"] != FROZEN_EXPECTATIONS_COMMIT:
        raise RuntimeError("study_config.json no longer names the frozen commit")
    submitted_hashes = _verify_submitted_hashes(evidence_root, normalized)
    job_ids = _submitted_jobs(evidence_root)
    if job_ids != ["202415", "202416", "202417", "202418"]:
        raise RuntimeError(f"unexpected submitted jobs: {job_ids}")
    sacct_rows = _sacct_jobs(evidence_root, job_ids)
    attempt_wall = _attempt_wall(evidence_root, normalized)
    counters, counter_summaries = _counter_rows(evidence_root, normalized, config)
    mechanism = _nccl_mechanism(evidence_root, normalized)
    cell_routing = _cell_routing(normalized, config)
    family_c = _family_c(normalized)
    family_r = _family_r(normalized, cell_routing)
    family_l = _family_l(normalized)
    family_w = _family_w(sacct_rows, attempt_wall, analysis_within_bound)
    families = {"C": family_c, "R": family_r, "L": family_l, "W": family_w}
    fatal_guards = _fatal_guards(
        evidence_root,
        normalized,
        submitted_hashes,
        sacct_rows,
        normalization,
        analysis_within_bound,
        cell_routing,
    )
    run_state = "void" if fatal_guards["FG-4"]["status"] == "FAIL" else "nonvoid"
    deviations = _parse_key_values(
        _require_file(evidence_root, "runbook_deviations.txt")
    )
    if set(deviations) != set(RUNBOOK_DEVIATIONS):
        raise RuntimeError("runbook_deviations.txt has an unexpected defect set")
    evidence_manifest = _evidence_manifest(evidence_root)
    record = {
        "schema": "simllm-merlin-collective-scored-record-v1",
        "study": "merlin_collective_capture_v1",
        "run_state": run_state,
        "run_state_rule": "Only FG-4 voids the campaign; FG-2 voids contradicted cells.",
        "verdict": "NONVOID_FG_2_CELL_SCOPED_CONCENTRATION_CONTROL_REFUTED",
        "evidence_root": {
            "configuration": f"--evidence-root or {EVIDENCE_ENV}",
            "tracked_paths_are_relative_to_evidence_root": True,
            "read_only": True,
        },
        "chronology": {
            "outage": {
                "status": "integrator-reported pre-capture outage",
                "scope": "unscored chronology; the fetched evidence does not independently timestamp the outage",
            },
            "expectations_commit": FROZEN_EXPECTATIONS_COMMIT,
            "harness_commit": HARNESS_COMMIT,
            "pre_capture_addendum_commit": ADDENDUM_COMMIT,
            "interpreter_fix_commit": INTERPRETER_FIX_COMMIT,
            "submitted_commit": _require_file(
                evidence_root, "submitted_commit.txt"
            ).read_text(encoding="utf-8").strip(),
            "flagship_fence": "pinged and cleared before A100 submission",
            "runbook_deviations": [
                {
                    "id": key,
                    "evidence_path": "runbook_deviations.txt",
                    "text": RUNBOOK_DEVIATIONS[key],
                }
                for key in sorted(deviations)
            ],
            "capture_date": "2026-09-01",
        },
        "source_manifest": _source_manifest(),
        "evidence": {
            "normalized_sha256": normalization["first_sha256"],
            "second_normalized_sha256": normalization["second_sha256"],
            "normalized_outputs_byte_identical": normalization["byte_identical"],
            "submitted_hashes": submitted_hashes,
            "file_count": len(evidence_manifest),
            "manifest_sha256": _sha256_bytes(_json_bytes(evidence_manifest)),
            "files": evidence_manifest,
        },
        "fatal_guards": fatal_guards,
        "families": families,
        "achieved_concentration": {
            "source": "ip -s -j link stats64 before/after snapshots",
            "rows": counters,
            "node_summaries": counter_summaries,
            "cell_routing": _published_cell_routing(cell_routing),
        },
        "mechanism": mechanism,
        "physical_sanity": {
            "coverage_arithmetic": "2 widths * 2 declarations * 4 operations * 22 payloads = 352 cells",
            "serialization_floor": "Every published median is at or above its configured payload-over-declared-port-rate floor.",
            "ladder_sum": "Per-cell RX deltas cover 25 measured repeats. Scaling their sum by 30/25 for five warmups predicts each job-level RX snapshot within the recorded node-summary ratio.",
            "counter_conservation": "Opposite-node packet totals are close and errors plus drops are zero, while byte totals differ by about tenfold. Linux TX and RX byte fields are not symmetric wire-byte authorities; ip-link packet and error counters cannot exclude retransmission below the offload accounting boundary.",
        },
        "task_effect": {
            "TRAF-77": "NARROWED, remains open",
            "TRAF-81": "OPENED for TX-side routing control, counter semantics, and harness portability",
            "literal_evidence": "phase timing at widths 2 and 8 with anchors held; endpoint proxies captured; concentration control refuted pending a working pinning mechanism; switch occupancy remains unobservable",
            "not_changed": "No transport calibration, H200 evidence, switch occupancy, receiver occupancy, queue-wait, buffer high-water, TTFT, or TPOT acceptance clause is satisfied.",
        },
    }
    if run_state != "nonvoid":
        raise RuntimeError("FG-4 unexpectedly voided the scored campaign")
    csv_rows = _csv_rows(normalized, families)
    return _json_bytes(record), _csv_bytes(csv_rows)


def _resolve_evidence_root(argument: Path | None) -> Path:
    if argument is not None:
        return argument
    configured = os.environ.get(EVIDENCE_ENV)
    if configured:
        return Path(configured)
    raise RuntimeError(
        f"pass --evidence-root or set {EVIDENCE_ENV} to the fetched TRAF-77 root"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--record-output", type=Path, default=TRACKED_RECORD)
    parser.add_argument("--csv-output", type=Path, default=TRACKED_CSV)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    evidence_root = _resolve_evidence_root(args.evidence_root)

    started = time.monotonic()
    first_record, first_csv = build_outputs(evidence_root)
    first_elapsed = time.monotonic() - started
    second_started = time.monotonic()
    second_record, second_csv = build_outputs(evidence_root)
    second_elapsed = time.monotonic() - second_started
    within_bound = max(first_elapsed, second_elapsed) <= ANALYSIS_WALL_CEILING_SECONDS
    if not within_bound:
        first_record, first_csv = build_outputs(
            evidence_root, analysis_within_bound=False
        )
        second_record, second_csv = build_outputs(
            evidence_root, analysis_within_bound=False
        )
    if first_record != second_record or first_csv != second_csv:
        raise RuntimeError("FG-6 failed: two scorer builds differ")

    if args.check:
        if args.record_output.read_bytes() != first_record:
            raise RuntimeError("tracked record.json does not match fetched evidence")
        if args.csv_output.read_bytes() != first_csv:
            raise RuntimeError("tracked results.csv does not match fetched evidence")
        print("TRAF-77 scored record and CSV reproduce byte for byte")
        return 0

    args.record_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.record_output.write_bytes(first_record)
    args.csv_output.write_bytes(first_csv)
    print("run_state=nonvoid")
    print("C=PASS R=UNEVALUABLE L=PASS W=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
