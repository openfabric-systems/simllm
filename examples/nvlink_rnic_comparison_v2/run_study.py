#!/usr/bin/env python3
"""Run the frozen TRAF-72 corrected transport comparison."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import defaultdict, deque
from itertools import pairwise
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simllm.backends.htsim_nvlink import (
    NvlinkCandidateProfile,
    NvlinkDomainResult,
    NvlinkDomainService,
    NvlinkFlowPolicy,
    NvlinkPacket,
    NvlinkTransfer,
    load_nvlink_candidate_profile,
)

HERE = Path(__file__).resolve().parent
EXPECTATIONS_PATH = HERE / "expectations.json"
EXPECTATIONS_COMMIT = "8e69696ba22a600a9aefab21c9f5d93e3f977a77"
EXPECTATIONS_SHA256 = "9724d405c400d5e38582fd869f24866f31fc6e0907d4b1b558b620eb411324bb"
PINNED_HTSIM_COMMIT = "1dcbfec36a33753bf978cf6323bade1a6645fe4f"
BULK_ROOT_ENV = "SIMLLM_NVCOMPARE2_BULK_ROOT"
ADAPTER_ENV = "SIMLLM_NVCOMPARE2_RNIC_ADAPTER"
RESULT_SCHEMA = "simllm-nvlink-rnic-comparison-result-v2"
TRANSPORTS = ("nvlink-credit", "rnic-nn", "rnic-nn-fluid")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_expectations() -> dict[str, Any]:
    return json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))


def require_clean_authority(frozen: dict[str, Any]) -> dict[str, object]:
    if _sha256(EXPECTATIONS_PATH) != EXPECTATIONS_SHA256:
        raise SystemExit("the TRAF-72 expectations digest moved after the freeze")
    ancestor = _git("merge-base", "--is-ancestor", EXPECTATIONS_COMMIT, "HEAD")
    if ancestor.returncode:
        raise SystemExit("the TRAF-72 expectations commit is not an ancestor of HEAD")
    committed = _git(
        "show",
        f"{EXPECTATIONS_COMMIT}:examples/nvlink_rnic_comparison_v2/expectations.json",
    )
    if committed.returncode:
        raise SystemExit("the committed TRAF-72 expectations bytes are unavailable")
    if hashlib.sha256(committed.stdout.encode("utf-8")).hexdigest() != EXPECTATIONS_SHA256:
        raise SystemExit("the committed TRAF-72 expectations digest disagrees")
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status.returncode or status.stdout.strip():
        raise SystemExit("the TRAF-72 gated run requires a clean tracked worktree")
    tree = _git("ls-tree", "HEAD", "third_party/htsim")
    if tree.returncode or PINNED_HTSIM_COMMIT not in tree.stdout:
        raise SystemExit("the htsim submodule pin moved from the frozen commit")

    authority = frozen["source_authority"]
    for name in (
        "flow_expectations",
        "legacy_expectations",
        "legacy_result",
        "profile",
    ):
        path = ROOT / authority[f"{name}_path"]
        if _sha256(path) != authority[f"{name}_sha256"]:
            raise SystemExit(f"the frozen {name} authority moved")
    lock = frozen["preservation_lock"]
    if lock["artifact_count"] != len(lock["artifacts"]):
        raise SystemExit("the TRAF-71 preservation lock has the wrong size")
    for artifact in lock["artifacts"]:
        if _sha256(ROOT / artifact["path"]) != artifact["sha256"]:
            raise SystemExit(f"TRAF-71 preservation failure: {artifact['path']}")
    return {
        "expectations_commit": EXPECTATIONS_COMMIT,
        "expectations_sha256": EXPECTATIONS_SHA256,
        "htsim_commit": PINNED_HTSIM_COMMIT,
        "legacy_files_checked": len(lock["artifacts"]),
    }


def prepare_run_dir(path: Path) -> Path:
    raw_root = os.environ.get(BULK_ROOT_ENV)
    if not raw_root:
        raise SystemExit(f"{BULK_ROOT_ENV} must name the configured bulk-output root")
    root = Path(raw_root).resolve()
    candidate = path.resolve()
    if candidate == root or root not in candidate.parents:
        raise SystemExit(f"--run-dir must be a new child of {BULK_ROOT_ENV}")
    if candidate.exists():
        raise SystemExit("--run-dir must not exist")
    candidate.mkdir(parents=True)
    return candidate


def configured_adapter(explicit: Path | None) -> Path:
    configured = os.environ.get(ADAPTER_ENV)
    raw = explicit or (Path(configured) if configured else None)
    if raw is None:
        raise SystemExit(f"--rnic-adapter or {ADAPTER_ENV} must name the built adapter")
    path = raw.resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"TRAF-72 adapter is not executable: {path}")
    return path


def adapter_provenance(adapter: Path) -> dict[str, str]:
    completed = subprocess.run(
        [str(adapter), "--provenance"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise SystemExit(f"TRAF-72 adapter provenance failed: {completed.stderr.strip()}")
    values = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    expected = {
        "schema": "simllm-traf72-rnic-adapter-provenance-v1",
        "htsim_source_commit": PINNED_HTSIM_COMMIT,
        "packet_primitives": "RnicMaxMinAllocator+RnicPacketizedSlotCalendar",
        "fluid_primitive": "RnicFluidManifold",
        "mapping": "one-active-transfer-per-ordered-pair-class",
    }
    if values != expected:
        raise SystemExit(f"TRAF-72 adapter provenance disagrees: {values}")
    return {**values, "executable_sha256": _sha256(adapter)}


def _sample_releases(
    *,
    seed: int,
    degree: int,
    size_bytes: int,
    release_interval_ps: int,
    jitter_low_ps: int,
    jitter_high_ps: int,
    samples_per_sender: int,
) -> list[tuple[int, int, int]]:
    generator = random.Random(seed * 1_000_003 + degree * 10_007 + size_bytes)
    releases = []
    wave_release = 0
    for wave in range(samples_per_sender):
        if wave:
            wave_release += release_interval_ps + generator.randint(
                jitter_low_ps, jitter_high_ps
            )
        for source in range(degree):
            releases.append(
                (wave, source, wave_release + generator.randint(0, jitter_high_ps))
            )
    return releases


def release_lists(
    cell: dict[str, Any], workload: dict[str, Any]
) -> dict[str, list[tuple[int, int, int]]]:
    releases = {
        str(seed): _sample_releases(
            seed=seed,
            degree=cell["degree"],
            size_bytes=cell["size_bytes"],
            release_interval_ps=cell["release_interval_ps"],
            jitter_low_ps=cell["release_jitter_ps"][0],
            jitter_high_ps=cell["release_jitter_ps"][1],
            samples_per_sender=workload["samples_per_seed_per_sender"],
        )
        for seed in workload["seeds"]
    }
    if hashlib.sha256(_canonical(releases)).hexdigest() != cell[
        "release_schedule_sha256"
    ]:
        raise AssertionError("release schedule differs from its frozen digest")
    return releases


def _quantile(values: list[int], probability: float) -> int:
    if not values:
        raise ValueError("quantile needs at least one value")
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, int(probability * len(ordered) + 0.999999) - 1),
    )
    return ordered[index]


def _cdf_rows(samples: dict[int, list[int]]) -> list[dict[str, object]]:
    ordered = {seed: sorted(values) for seed, values in samples.items()}
    grid = sorted({value for values in ordered.values() for value in values})
    rows = []
    for value in grid:
        probabilities = [
            bisect.bisect_right(values, value) / len(values)
            for values in ordered.values()
        ]
        rows.append(
            {
                "fct_ps": value,
                "cdf_mean": sum(probabilities) / len(probabilities),
                "cdf_min": min(probabilities),
                "cdf_max": max(probabilities),
            }
        )
    return rows


def _cdf_valid(rows: list[dict[str, object]]) -> bool:
    tolerance = 1e-12
    if not rows or any(rows[-1][field] != 1 for field in ("cdf_min", "cdf_mean", "cdf_max")):
        return False
    for field in ("cdf_min", "cdf_mean", "cdf_max"):
        if any(left[field] > right[field] + tolerance for left, right in pairwise(rows)):
            return False
    return all(
        -tolerance
        <= row["cdf_min"]
        <= row["cdf_mean"] + tolerance
        <= row["cdf_max"] + 2 * tolerance
        <= 1 + 2 * tolerance
        for row in rows
    )


def _seed_tail(samples: dict[int, list[int]]) -> dict[str, object]:
    metrics = {
        "p50": [_quantile(values, 0.50) for values in samples.values()],
        "p99": [_quantile(values, 0.99) for values in samples.values()],
        "worst": [max(values) for values in samples.values()],
    }
    output: dict[str, object] = {}
    for name, values in metrics.items():
        output[f"{name}_seed_mean_ps"] = sum(values) / len(values)
        output[f"{name}_seed_min_ps"] = min(values)
        output[f"{name}_seed_max_ps"] = max(values)
    output["mean_fct_ps"] = sum(sum(values) for values in samples.values()) / sum(
        len(values) for values in samples.values()
    )
    output["sample_count"] = sum(len(values) for values in samples.values())
    return output


def _jain(values: list[float]) -> float:
    if not values:
        raise ValueError("Jain fairness needs at least one value")
    square_sum = sum(values) ** 2
    square_terms = len(values) * sum(value * value for value in values)
    return square_sum / square_terms


def _seed_fairness(
    rows_by_seed: dict[int, list[dict[str, object]]], size_bytes: int
) -> dict[str, object]:
    seed_values = []
    seed_worst = []
    for rows in rows_by_seed.values():
        by_wave: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            by_wave[int(row["wave"])].append(size_bytes / int(row["fct_ps"]))
        wave_values = [_jain(values) for values in by_wave.values()]
        seed_values.append(sum(wave_values) / len(wave_values))
        seed_worst.append(min(wave_values))
    return {
        "jain_seed_mean": sum(seed_values) / len(seed_values),
        "jain_seed_min": min(seed_values),
        "jain_seed_max": max(seed_values),
        "jain_worst_wave_seed_mean": sum(seed_worst) / len(seed_worst),
        "jain_worst_wave_seed_min": min(seed_worst),
        "jain_worst_wave_seed_max": max(seed_worst),
    }


def _serve_nvlink(
    profile: NvlinkCandidateProfile, transfers: list[NvlinkTransfer]
) -> NvlinkDomainResult:
    result = NvlinkDomainService(profile).serve(
        transfers,
        analytic_result=None,
        flow_policy=NvlinkFlowPolicy.RELEASE_AWARE_ROUND_ROBIN,
    )
    if not isinstance(result, NvlinkDomainResult):
        raise TypeError("the scored NVLink arm did not return a domain result")
    return result


def _flow_completions(
    result: NvlinkDomainResult, transfers: list[NvlinkTransfer]
) -> dict[str, int]:
    releases = {transfer.extent_id: transfer.released_at_ps for transfer in transfers}
    absolute: dict[str, int] = {}
    for packet in result.packets:
        if packet.delivered_at_ps is None:
            raise AssertionError("an NVLink packet has no delivery time")
        absolute[packet.extent_id] = max(
            absolute.get(packet.extent_id, 0), packet.delivered_at_ps
        )
    if absolute.keys() != releases.keys():
        raise AssertionError("the NVLink completion projection lost or added a flow")
    return {flow_id: absolute[flow_id] - releases[flow_id] for flow_id in releases}


def _serialize_ps(wire_bytes: int, bytes_per_second: int) -> int:
    return (wire_bytes * 1_000_000_000_000 + bytes_per_second - 1) // bytes_per_second


def _nvlink_diagnostics(
    packets: tuple[NvlinkPacket, ...], profile: NvlinkCandidateProfile
) -> dict[str, int]:
    credit_wait_ps = 0
    credit_wait_packets = 0
    rx_wait_ps = 0
    rx_wait_packets = 0
    wire_bytes = 0
    by_source: dict[int, list[NvlinkPacket]] = defaultdict(list)
    for packet in packets:
        if packet.tx_started_at_ps is None or packet.tx_finished_at_ps is None:
            raise AssertionError("an NVLink packet lacks TX timestamps")
        if packet.rx_started_at_ps is None:
            raise AssertionError("an NVLink packet lacks an RX start timestamp")
        by_source[packet.source].append(packet)
        wire_bytes += packet.wire_bytes
        wait = packet.rx_started_at_ps - packet.tx_finished_at_ps
        if wait > 0:
            rx_wait_ps += wait
            rx_wait_packets += 1
    for source_packets in by_source.values():
        ordered = sorted(
            source_packets,
            key=lambda packet: (
                packet.tx_started_at_ps,
                packet.tx_finished_at_ps,
                packet.destination,
                packet.extent_id,
                packet.sequence,
            ),
        )
        link_cursors: dict[tuple[int, int, int], int] = {}
        endpoint_cursor = 0
        credit_slots: dict[tuple[int, int], list[int]] = {}
        pair_visits: dict[tuple[int, int], int] = {}
        for packet in ordered:
            assert packet.tx_started_at_ps is not None
            assert packet.tx_finished_at_ps is not None
            assert packet.link_index is not None
            pair = (packet.source, packet.destination)
            slots = credit_slots.setdefault(
                pair, [0] * profile.tx.credits_per_destination
            )
            visit = pair_visits.get(pair, 0)
            slot_index = visit % profile.tx.credits_per_destination
            pair_visits[pair] = visit + 1
            link_key = (packet.source, packet.destination, packet.link_index)
            base = max(
                packet.released_at_ps,
                link_cursors.get(link_key, 0),
                endpoint_cursor,
            )
            wait = max(0, slots[slot_index] - base)
            if packet.tx_started_at_ps != max(base, slots[slot_index]):
                raise AssertionError("NVLink credit wait reconstruction disagrees")
            if wait:
                credit_wait_ps += wait
                credit_wait_packets += 1
            link_cursors[link_key] = packet.tx_finished_at_ps
            endpoint_cursor = packet.tx_started_at_ps + _serialize_ps(
                packet.wire_bytes,
                profile.tx.endpoint_egress_rate_bytes_per_second,
            )
            slots[slot_index] = (
                packet.tx_finished_at_ps + profile.rx.credit_return_latency_ps
            )
    return {
        "packet_count": len(packets),
        "wire_bytes": wire_bytes,
        "credit_wait_ps": credit_wait_ps,
        "credit_wait_packets": credit_wait_packets,
        "rx_wait_ps": rx_wait_ps,
        "rx_wait_packets": rx_wait_packets,
    }


def _rnic_command(
    adapter: Path,
    schedule_path: Path,
    completion_path: Path,
    manifest_path: Path,
    physical: dict[str, Any],
    node_count: int,
) -> list[str]:
    return [
        str(adapter),
        "--schedule-csv",
        str(schedule_path),
        "--completion-csv",
        str(completion_path),
        "--manifest-json",
        str(manifest_path),
        "--source-capacity-bps",
        str(physical["pair_raw_bytes_per_second"] * 8),
        "--destination-capacity-bps",
        str(physical["rx_ingress_bytes_per_second"] * 8),
        "--max-wire-bytes",
        str(physical["max_wire_bytes"]),
        "--header-bytes",
        str(physical["header_bytes"]),
        "--propagation-ps",
        str(physical["propagation_ps"]),
        "--nodes",
        str(node_count),
    ]


def _run_rnic(
    *,
    adapter: Path,
    cell_dir: Path,
    schedule_rows: list[dict[str, object]],
    physical: dict[str, Any],
    node_count: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    schedule_path = cell_dir / "schedule.csv"
    completion_path = cell_dir / "rnic-completions.csv"
    manifest_path = cell_dir / "rnic-manifest.json"
    _write_csv(
        schedule_path,
        [
            "numeric_id",
            "flow_id",
            "wave",
            "source",
            "destination",
            "payload_bytes",
            "released_at_ps",
        ],
        schedule_rows,
    )
    completed = subprocess.run(
        _rnic_command(
            adapter,
            schedule_path,
            completion_path,
            manifest_path,
            physical,
            node_count,
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"TRAF-72 adapter exited {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return _read_csv(completion_path), json.loads(
        manifest_path.read_text(encoding="utf-8")
    )


def _fluid_oracle(
    schedule_rows: list[dict[str, object]],
    source_capacity_bps: int,
    destination_capacity_bps: int,
) -> dict[str, int]:
    releases = sorted(
        schedule_rows,
        key=lambda row: (int(row["released_at_ps"]), int(row["numeric_id"])),
    )
    waiting: dict[int, deque[dict[str, object]]] = defaultdict(deque)
    active: dict[int, dict[str, object]] = {}
    completion: dict[str, int] = {}
    release_index = 0
    now = 0
    while len(completion) < len(releases):
        count = len(active)
        rate = min(source_capacity_bps, destination_capacity_bps // count) if count else 0
        next_completion = None
        if active:
            duration = min(
                (int(state["debt"]) + rate - 1) // rate for state in active.values()
            )
            next_completion = now + duration
        next_release = (
            int(releases[release_index]["released_at_ps"])
            if release_index < len(releases)
            else None
        )
        candidates = [value for value in (next_completion, next_release) if value is not None]
        if not candidates:
            raise AssertionError("fluid oracle has no next event")
        target = min(candidates)
        elapsed = target - now
        if elapsed and active:
            for state in active.values():
                state["debt"] = max(0, int(state["debt"]) - rate * elapsed)
        now = target
        completed_sources = []
        for source, state in active.items():
            if int(state["debt"]) == 0:
                row = state["row"]
                completion[str(row["flow_id"])] = now - int(row["released_at_ps"])
                completed_sources.append(source)
        for source in completed_sources:
            del active[source]
        while release_index < len(releases) and int(
            releases[release_index]["released_at_ps"]
        ) <= now:
            row = releases[release_index]
            waiting[int(row["source"])].append(row)
            release_index += 1
        for source, queue in waiting.items():
            if source in active or not queue:
                continue
            row = queue.popleft()
            active[source] = {
                "row": row,
                "debt": int(row["payload_bytes"]) * 8 * 1_000_000_000_000,
            }
    return completion


def _artifact(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _score_hypotheses(
    tail_rows: list[dict[str, object]],
    fairness_rows: list[dict[str, object]],
    frozen: dict[str, Any],
) -> list[dict[str, object]]:
    tail = {
        (row["transport"], row["degree"], row["size_bytes"]): row
        for row in tail_rows
    }
    fairness = {
        (row["transport"], row["degree"], row["size_bytes"]): row
        for row in fairness_rows
    }
    audit = frozen["mapping_audit"]
    legacy = audit["legacy_degree_3_512k_p50_ps"]["rnic_nn"]
    corrected = tail[("rnic-nn", 3, 524288)]["p50_seed_mean_ps"]
    nvlink = tail[("nvlink-credit", 3, 524288)]["p50_seed_mean_ps"]
    target = audit["normalized_queue_arithmetic"]["predicted_ratio_decimal"]
    h1_checks = [abs((legacy / corrected) / target - 1) <= 0.05, corrected <= nvlink]

    h2_checks = []
    for degree in frozen["workload"]["degrees"]:
        for size_bytes in frozen["workload"]["flow_sizes_bytes"]:
            fluid = tail[("rnic-nn-fluid", degree, size_bytes)]
            for metric in ("p50", "p99", "worst"):
                field = f"{metric}_seed_mean_ps"
                h2_checks.extend(
                    fluid[field] <= tail[(transport, degree, size_bytes)][field]
                    for transport in ("nvlink-credit", "rnic-nn")
                )
    small = (256, 1024, 4096)
    mesh = (4, 8, 16)
    h3_checks = []
    for size_bytes in small:
        for transport in ("rnic-nn", "rnic-nn-fluid"):
            for metric in ("p99", "worst"):
                field = f"{metric}_seed_mean_ps"
                advantages = []
                for degree in mesh:
                    reference = tail[(transport, degree, size_bytes)][field]
                    nv_value = tail[("nvlink-credit", degree, size_bytes)][field]
                    h3_checks.append(reference < nv_value)
                    advantages.append(nv_value / reference)
                h3_checks.append(all(left <= right for left, right in pairwise(advantages)))

    h4_checks = []
    for size_bytes in small:
        for transport in ("rnic-nn", "rnic-nn-fluid"):
            gaps = []
            for degree in mesh:
                reference = fairness[(transport, degree, size_bytes)]["jain_seed_mean"]
                nv_value = fairness[("nvlink-credit", degree, size_bytes)][
                    "jain_seed_mean"
                ]
                h4_checks.append(reference >= nv_value)
                gaps.append(reference - nv_value)
            h4_checks.append(all(left <= right for left, right in pairwise(gaps)))
    return [
        {
            "id": identifier,
            "passed_instances": sum(checks),
            "total_instances": len(checks),
            "verdict": "PASS" if all(checks) else "REFUTED",
        }
        for identifier, checks in (
            ("H1", h1_checks),
            ("H2", h2_checks),
            ("H3", h3_checks),
            ("H4", h4_checks),
        )
    ]


def run(run_dir: Path, adapter: Path) -> dict[str, object]:
    frozen = load_expectations()
    authority = require_clean_authority(frozen)
    provenance = adapter_provenance(adapter)
    physical = frozen["physical_constants"]
    workload = frozen["workload"]
    profile = load_nvlink_candidate_profile(
        ROOT / frozen["source_authority"]["profile_path"]
    )
    sample_rows: list[dict[str, object]] = []
    cdf_rows: list[dict[str, object]] = []
    tail_rows: list[dict[str, object]] = []
    fairness_rows: list[dict[str, object]] = []
    manifests: list[dict[str, Any]] = []
    maximum_fluid_oracle_error_ps = 0

    for cell in workload["cells"]:
        degree = int(cell["degree"])
        size_bytes = int(cell["size_bytes"])
        destination = int(cell["destination"])
        by_seed = release_lists(cell, workload)
        transport_samples: dict[str, dict[int, list[int]]] = {
            transport: {} for transport in TRANSPORTS
        }
        transport_rows: dict[str, dict[int, list[dict[str, object]]]] = {
            transport: {} for transport in TRANSPORTS
        }
        diagnostics = {
            "nvlink-credit": defaultdict(int),
            "rnic-nn": defaultdict(int),
            "rnic-nn-fluid": defaultdict(int),
        }
        for seed_text, release_spec in by_seed.items():
            seed = int(seed_text)
            flow_specs = [
                {
                    "numeric_id": index,
                    "flow_id": (
                        f"d{degree}-b{size_bytes}-s{seed}-w{wave}-src{source}"
                    ),
                    "wave": wave,
                    "source": source,
                    "destination": destination,
                    "payload_bytes": size_bytes,
                    "released_at_ps": release,
                }
                for index, (wave, source, release) in enumerate(release_spec, start=1)
            ]
            expected_flows = degree * workload["samples_per_seed_per_sender"]
            if len(flow_specs) != expected_flows:
                raise AssertionError("a frozen cell has the wrong flow count")
            expected_by_id = {str(row["flow_id"]): row for row in flow_specs}

            transfers = [
                NvlinkTransfer(
                    extent_id=str(row["flow_id"]),
                    source=int(row["source"]),
                    destination=int(row["destination"]),
                    payload_bytes=int(row["payload_bytes"]),
                    released_at_ps=int(row["released_at_ps"]),
                )
                for row in flow_specs
            ]
            nvlink = _serve_nvlink(profile, transfers)
            nvlink_fcts = _flow_completions(nvlink, transfers)
            nv_rows = []
            for row in flow_specs:
                projected = {
                    "transport": "nvlink-credit",
                    "degree": degree,
                    "size_bytes": size_bytes,
                    "seed": seed,
                    "flow_id": row["flow_id"],
                    "wave": row["wave"],
                    "source": row["source"],
                    "released_at_ps": row["released_at_ps"],
                    "admitted_at_ps": row["released_at_ps"],
                    "class_wait_ps": 0,
                    "fct_ps": nvlink_fcts[str(row["flow_id"])],
                }
                sample_rows.append(projected)
                nv_rows.append(projected)
            transport_rows["nvlink-credit"][seed] = nv_rows
            transport_samples["nvlink-credit"][seed] = list(nvlink_fcts.values())
            for key, value in _nvlink_diagnostics(nvlink.packets, profile).items():
                diagnostics["nvlink-credit"][key] += value

            cell_dir = run_dir / "cells" / f"d{degree}-b{size_bytes}-s{seed}"
            cell_dir.mkdir(parents=True)
            rnic_rows, manifest = _run_rnic(
                adapter=adapter,
                cell_dir=cell_dir,
                schedule_rows=flow_specs,
                physical=physical,
                node_count=destination + 1,
            )
            manifests.append(manifest)
            if (
                manifest["schema"] != "simllm-traf72-rnic-adapter-manifest-v1"
                or manifest["htsim_source_commit"] != PINNED_HTSIM_COMMIT
                or manifest["mapping"]
                != "one-active-transfer-per-ordered-pair-class"
                or manifest["source_capacity_bps"]
                != physical["pair_raw_bytes_per_second"] * 8
                or manifest["destination_capacity_bps"]
                != physical["rx_ingress_bytes_per_second"] * 8
                or manifest["max_wire_bytes"] != physical["max_wire_bytes"]
                or manifest["header_bytes"] != physical["header_bytes"]
                or manifest["propagation_ps"] != physical["propagation_ps"]
            ):
                raise AssertionError("the corrected RNIC physical mapping moved")
            for transport in ("rnic-nn", "rnic-nn-fluid"):
                rows = [row for row in rnic_rows if row["transport"] == transport]
                if {row["flow_id"] for row in rows} != expected_by_id.keys():
                    raise AssertionError(f"{transport} lost or added a flow")
                projected_rows = []
                for row in rows:
                    source = expected_by_id[row["flow_id"]]
                    if (
                        int(row["released_at_ps"]) != int(source["released_at_ps"])
                        or int(row["payload_bytes"]) != size_bytes
                        or int(row["wave"]) != int(source["wave"])
                    ):
                        raise AssertionError(f"{transport} changed a frozen flow")
                    projected = {
                        "transport": transport,
                        "degree": degree,
                        "size_bytes": size_bytes,
                        "seed": seed,
                        "flow_id": row["flow_id"],
                        "wave": int(row["wave"]),
                        "source": int(row["source"]),
                        "released_at_ps": int(row["released_at_ps"]),
                        "admitted_at_ps": int(row["admitted_at_ps"]),
                        "class_wait_ps": int(row["class_wait_ps"]),
                        "fct_ps": int(row["fct_ps"]),
                    }
                    sample_rows.append(projected)
                    projected_rows.append(projected)
                transport_rows[transport][seed] = projected_rows
                transport_samples[transport][seed] = [
                    int(row["fct_ps"]) for row in rows
                ]
                for key, value in manifest[transport].items():
                    diagnostics[transport][key] += int(value)

            expected_packets_per_flow = (size_bytes + 255) // 256
            expected_packets = expected_flows * expected_packets_per_flow
            expected_payload = expected_flows * size_bytes
            expected_wire = expected_payload + 16 * expected_packets
            packet_ledger = manifest["rnic-nn"]
            fluid_ledger = manifest["rnic-nn-fluid"]
            if (
                packet_ledger["flow_count"] != expected_flows
                or packet_ledger["payload_bytes"] != expected_payload
                or packet_ledger["packet_count"] != expected_packets
                or packet_ledger["wire_bytes"] != expected_wire
                or packet_ledger["active_pair_limit_violations"] != 0
                or fluid_ledger["flow_count"] != expected_flows
                or fluid_ledger["payload_bytes"] != expected_payload
                or fluid_ledger["packet_count"] != 0
                or fluid_ledger["wire_bytes"] != 0
                or fluid_ledger["active_pair_limit_violations"] != 0
                or manifest["ack_events"] != 0
                or manifest["reverse_control_bytes"] != 0
                or manifest["non_data_events"] != 0
            ):
                raise AssertionError("an RNIC byte or class ledger failed")

            oracle = _fluid_oracle(
                flow_specs,
                physical["pair_raw_bytes_per_second"] * 8,
                physical["rx_ingress_bytes_per_second"] * 8,
            )
            fluid_fcts = {
                row["flow_id"]: int(row["fct_ps"])
                for row in rnic_rows
                if row["transport"] == "rnic-nn-fluid"
            }
            maximum_fluid_oracle_error_ps = max(
                maximum_fluid_oracle_error_ps,
                max(abs(fluid_fcts[flow_id] - value) for flow_id, value in oracle.items()),
            )

        for transport in TRANSPORTS:
            curves = _cdf_rows(transport_samples[transport])
            if not _cdf_valid(curves):
                raise AssertionError(f"{transport} produced an invalid empirical CDF")
            cdf_rows.extend(
                {
                    "transport": transport,
                    "degree": degree,
                    "size_bytes": size_bytes,
                    **row,
                }
                for row in curves
            )
            tail_rows.append(
                {
                    "transport": transport,
                    "degree": degree,
                    "size_bytes": size_bytes,
                    **_seed_tail(transport_samples[transport]),
                    **dict(diagnostics[transport]),
                }
            )
            fairness_rows.append(
                {
                    "transport": transport,
                    "degree": degree,
                    "size_bytes": size_bytes,
                    **_seed_fairness(transport_rows[transport], size_bytes),
                }
            )

    if maximum_fluid_oracle_error_ps > 1:
        raise AssertionError(
            f"fluid oracle disagreement is {maximum_fluid_oracle_error_ps} ps"
        )
    if any(
        not 0 <= row["jain_seed_min"] <= row["jain_seed_max"] <= 1
        or (row["degree"] == 1 and row["jain_seed_mean"] != 1)
        for row in fairness_rows
    ):
        raise AssertionError("a fairness result is outside its physical range")

    _write_csv(
        run_dir / "fct-samples.csv",
        [
            "transport",
            "degree",
            "size_bytes",
            "seed",
            "flow_id",
            "wave",
            "source",
            "released_at_ps",
            "admitted_at_ps",
            "class_wait_ps",
            "fct_ps",
        ],
        sample_rows,
    )
    _write_csv(
        run_dir / "fct-cdf.csv",
        [
            "transport",
            "degree",
            "size_bytes",
            "fct_ps",
            "cdf_mean",
            "cdf_min",
            "cdf_max",
        ],
        cdf_rows,
    )
    tail_fields = sorted({key for row in tail_rows for key in row})
    fairness_fields = sorted({key for row in fairness_rows for key in row})
    _write_csv(run_dir / "tail-metrics.csv", tail_fields, tail_rows)
    _write_csv(run_dir / "fairness.csv", fairness_fields, fairness_rows)

    hypotheses = _score_hypotheses(tail_rows, fairness_rows, frozen)
    misses = [row["id"] for row in hypotheses if row["verdict"] == "REFUTED"]
    fatal_guards = [
        {"guard": guard, "verdict": "PASS"} for guard in frozen["fatal_guards"]
    ]
    result = {
        "schema": RESULT_SCHEMA,
        "task": "TRAF-72",
        "authority": authority,
        "adapter_provenance": provenance,
        "mapping_audit": frozen["mapping_audit"],
        "corrected_mapping": frozen["corrected_mapping"],
        "topology": frozen["topology"],
        "measurement_caveat": frozen["measurement_caveat"],
        "metrics": frozen["metrics"],
        "fatal_guard_verdict": "PASS",
        "fatal_guards": fatal_guards,
        "hypothesis_verdicts": hypotheses,
        "honest_refutations": misses,
        "maximum_fluid_oracle_error_ps": maximum_fluid_oracle_error_ps,
        "tail_metrics": tail_rows,
        "fairness_metrics": fairness_rows,
        "sample_count": len(sample_rows),
        "adapter_invocations": len(manifests),
        "study_verdict": "PASS" if not misses else "PASS_WITH_HONEST_REFUTATIONS",
    }
    _write_json(run_dir / "result.json", result)
    artifacts = [
        _artifact(path, run_dir)
        for path in sorted(candidate for candidate in run_dir.rglob("*") if candidate.is_file())
    ]
    _write_json(
        run_dir / "manifest.json",
        {
            "schema": "simllm-nvlink-rnic-comparison-run-manifest-v2",
            "artifacts": artifacts,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rnic-adapter", type=Path)
    arguments = parser.parse_args()
    run_dir = prepare_run_dir(arguments.run_dir)
    adapter = configured_adapter(arguments.rnic_adapter)
    result = run(run_dir, adapter)
    print(f"TRAF72_VERDICT={result['study_verdict']}")
    print(f"TRAF72_RUN_DIR={run_dir.as_posix()}")


if __name__ == "__main__":
    main()
