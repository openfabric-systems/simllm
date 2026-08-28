#!/usr/bin/env python3
"""Run the frozen TRAF-71 NVLink credit versus rnic-nn comparison."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from collections import defaultdict
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
EXPECTATIONS_COMMIT = "6224d90fea2eed788b8e6ba876787fe7f0e52319"
EXPECTATIONS_SHA256 = "4b60365d8251b5fd3c7627dbe38c66ad1fc1c096b21fdfada4fc744320a5bdfa"
BULK_ROOT_ENV = "SIMLLM_NVCOMPARE_BULK_ROOT"
ADAPTER_ENV = "SIMLLM_NVCOMPARE_RNIC_ADAPTER"
RESULT_SCHEMA = "simllm-nvlink-rnic-comparison-result-v1"
PINNED_HTSIM_COMMIT = "1dcbfec36a33753bf978cf6323bade1a6645fe4f"


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


def _inherited_preservation_artifacts(frozen: dict[str, Any]) -> list[dict[str, str]]:
    source_expectations = ROOT / frozen["source_authority"]["expectations_path"]
    source = json.loads(source_expectations.read_text(encoding="utf-8"))
    source_lock = source["preservation_lock"]
    inherited = json.loads(
        (ROOT / source_lock["inherited"]["path"]).read_text(encoding="utf-8")
    )["preservation_lock"]
    base = json.loads(
        (ROOT / inherited["inherited"]["path"]).read_text(encoding="utf-8")
    )["preservation_lock"]["artifacts"]
    return [*base, *inherited["additional_artifacts"], *source_lock["additional_artifacts"]]


def require_clean_authority(frozen: dict[str, Any]) -> dict[str, object]:
    if _sha256(EXPECTATIONS_PATH) != EXPECTATIONS_SHA256:
        raise SystemExit("the TRAF-71 expectations digest moved after the freeze")
    ancestor = _git("merge-base", "--is-ancestor", EXPECTATIONS_COMMIT, "HEAD")
    if ancestor.returncode:
        raise SystemExit("the TRAF-71 expectations commit is not an ancestor of HEAD")
    committed = _git(
        "show",
        f"{EXPECTATIONS_COMMIT}:examples/nvlink_rnic_comparison_v1/expectations.json",
    )
    if committed.returncode:
        raise SystemExit("the committed TRAF-71 expectations bytes are unavailable")
    if hashlib.sha256(committed.stdout.encode("utf-8")).hexdigest() != EXPECTATIONS_SHA256:
        raise SystemExit("the committed TRAF-71 expectations digest disagrees")
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status.returncode or status.stdout.strip():
        raise SystemExit("the TRAF-71 gated run requires a clean tracked worktree")

    tree = _git("ls-tree", "HEAD", frozen["htsim_authority"]["submodule_path"])
    if tree.returncode or PINNED_HTSIM_COMMIT not in tree.stdout:
        raise SystemExit("the htsim submodule pin moved from the frozen commit")

    source = frozen["source_authority"]
    for key in ("expectations", "result", "profile"):
        path = ROOT / source[f"{key}_path"]
        if _sha256(path) != source[f"{key}_sha256"]:
            raise SystemExit(f"the frozen source {key} moved")

    inherited = _inherited_preservation_artifacts(frozen)
    if len(inherited) != frozen["preservation_lock"]["inherited_expected_artifacts"]:
        raise SystemExit("the inherited preservation class has the wrong size")
    for artifact in inherited:
        if _sha256(ROOT / artifact["path"]) != artifact["sha256"]:
            raise SystemExit(f"inherited preservation failure: {artifact['path']}")
    direct = frozen["preservation_lock"]["direct_flow_dynamics_artifacts"]
    if len(direct) != frozen["preservation_lock"]["direct_count"]:
        raise SystemExit("the direct flow-dynamics preservation class has the wrong size")
    for artifact in direct:
        if _sha256(ROOT / artifact["path"]) != artifact["sha256"]:
            raise SystemExit(f"flow-dynamics preservation failure: {artifact['path']}")
    return {
        "expectations_commit": EXPECTATIONS_COMMIT,
        "expectations_sha256": EXPECTATIONS_SHA256,
        "htsim_commit": PINNED_HTSIM_COMMIT,
        "inherited_artifacts_checked": len(inherited),
        "flow_dynamics_files_checked": len(direct),
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
    raw = explicit or (Path(os.environ[ADAPTER_ENV]) if os.environ.get(ADAPTER_ENV) else None)
    if raw is None:
        raise SystemExit(f"--rnic-adapter or {ADAPTER_ENV} must name the built adapter")
    path = raw.resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"rnic-nn adapter is not executable: {path}")
    return path


def adapter_provenance(adapter: Path) -> dict[str, str]:
    completed = subprocess.run(
        [str(adapter), "--provenance"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise SystemExit(f"rnic-nn adapter provenance failed: {completed.stderr.strip()}")
    values = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    expected = {
        "schema": "simllm-traf71-rnic-adapter-provenance-v1",
        "htsim_source_commit": PINNED_HTSIM_COMMIT,
        "runtime_class": "RnicPacketizedManifoldRuntime",
        "transport": "rnic-nn",
        "ack_pacing": "absent",
    }
    if values != expected:
        raise SystemExit(f"rnic-nn adapter provenance disagrees: {values}")
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
            source_skew = generator.randint(0, jitter_high_ps)
            releases.append((wave, source, wave_release + source_skew))
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
    if (
        hashlib.sha256(_canonical(releases)).hexdigest()
        != cell["release_schedule_sha256"]
    ):
        raise AssertionError("release schedule differs from its frozen digest")
    return releases


def _quantile(values: list[int], probability: float) -> int:
    if not values:
        raise ValueError("quantile needs at least one value")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(probability * len(ordered) + 0.999999) - 1))
    return ordered[index]


def _cdf_rows(samples: dict[int, list[int]]) -> list[dict[str, object]]:
    ordered = {seed: sorted(values) for seed, values in samples.items()}
    grid = sorted({value for values in ordered.values() for value in values})
    rows = []
    for value in grid:
        probabilities = [
            bisect.bisect_right(values, value) / len(values) for values in ordered.values()
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
    if (
        not rows
        or rows[-1]["cdf_min"] != 1
        or rows[-1]["cdf_mean"] != 1
        or rows[-1]["cdf_max"] != 1
    ):
        return False
    for field in ("cdf_min", "cdf_mean", "cdf_max"):
        if any(
            left[field] > right[field] + tolerance
            for left, right in pairwise(rows)
        ):
            return False
    return all(
        -tolerance
        <= row["cdf_min"]
        <= row["cdf_mean"] + tolerance
        <= row["cdf_max"] + 2 * tolerance
        <= 1 + 2 * tolerance
        for row in rows
    )


def _seed_summary(samples: dict[int, list[int]]) -> dict[str, object]:
    p50 = [_quantile(values, 0.50) for values in samples.values()]
    p95 = [_quantile(values, 0.95) for values in samples.values()]
    p50_median = int(statistics.median(p50))
    width = max(p50) - min(p50)
    return {
        "sample_count": sum(len(values) for values in samples.values()),
        "p50_seed_min_ps": min(p50),
        "p50_seed_max_ps": max(p50),
        "p50_seed_mean_ps": sum(p50) / len(p50),
        "p50_seed_median_ps": p50_median,
        "p95_seed_mean_ps": sum(p95) / len(p95),
        "dispersion_width_ps": width,
        "dispersion_ratio": width / p50_median,
        "mean_fct_ps": sum(sum(values) for values in samples.values())
        / sum(len(values) for values in samples.values()),
    }


def _serve_nvlink(
    profile: NvlinkCandidateProfile,
    transfers: list[NvlinkTransfer],
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
    result: NvlinkDomainResult,
    transfers: list[NvlinkTransfer],
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
    packets: tuple[NvlinkPacket, ...],
    profile: NvlinkCandidateProfile,
) -> dict[str, int]:
    credit_wait_ps = 0
    credit_wait_packets = 0
    rx_wait_ps = 0
    rx_wait_packets = 0
    packet_count = 0
    wire_bytes = 0
    by_source: dict[int, list[NvlinkPacket]] = defaultdict(list)
    for packet in packets:
        if packet.tx_started_at_ps is None or packet.tx_finished_at_ps is None:
            raise AssertionError("an NVLink packet lacks TX timestamps")
        if packet.rx_started_at_ps is None:
            raise AssertionError("an NVLink packet lacks an RX start timestamp")
        by_source[packet.source].append(packet)
        packet_count += 1
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
            expected_start = max(base, slots[slot_index])
            if packet.tx_started_at_ps != expected_start:
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
        "packet_count": packet_count,
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
    rate_bps: int,
    physical: dict[str, Any],
) -> list[str]:
    return [
        str(adapter),
        "--schedule-csv",
        str(schedule_path),
        "--completion-csv",
        str(completion_path),
        "--manifest-json",
        str(manifest_path),
        "--capacity-bps",
        str(rate_bps),
        "--max-wire-bytes",
        str(physical["max_wire_bytes"]),
        "--header-bytes",
        str(physical["header_bytes"]),
        "--propagation-ps",
        str(physical["propagation_ps"]),
        "--nodes",
        "4",
    ]


def _run_rnic(
    *,
    adapter: Path,
    cell_dir: Path,
    schedule_rows: list[dict[str, object]],
    rate_bps: int,
    physical: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    schedule_path = cell_dir / "schedule.csv"
    completion_path = cell_dir / "rnic-completions.csv"
    manifest_path = cell_dir / "rnic-manifest.json"
    _write_csv(
        schedule_path,
        [
            "numeric_id",
            "flow_id",
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
            rate_bps,
            physical,
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"rnic-nn adapter exited {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return _read_csv(completion_path), json.loads(
        manifest_path.read_text(encoding="utf-8")
    )


def _add_diagnostics(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def _expected_direction_rows(
    summaries: list[dict[str, object]],
    rnic_semantics_pass: bool,
) -> list[dict[str, object]]:
    by_key = {
        (row["transport"], row["degree"], row["size_bytes"]): row
        for row in summaries
    }
    e1_instances = [
        by_key[("rnic-nn", degree, 256)]["p50_seed_mean_ps"]
        < by_key[("nvlink-credit", degree, 256)]["p50_seed_mean_ps"]
        for degree in (1, 2, 3)
    ]
    e2_instances = [
        by_key[(transport, degree, 524288)]["dispersion_ratio"]
        < by_key[(transport, degree, 1024)]["dispersion_ratio"]
        for transport in ("nvlink-credit", "rnic-nn")
        for degree in (1, 2, 3)
    ]
    large_cells = [
        by_key[("rnic-nn", degree, size_bytes)]["dispersion_ratio"]
        <= by_key[("nvlink-credit", degree, size_bytes)]["dispersion_ratio"]
        for degree in (1, 2, 3)
        for size_bytes in (65536, 262144, 524288)
    ]
    oddity_sizes = (1024, 4096, 16384, 65536, 262144, 524288)
    nvlink_oddity = [
        by_key[("nvlink-credit", 3, size_bytes)]["p50_seed_mean_ps"]
        < by_key[("nvlink-credit", 1, size_bytes)]["p50_seed_mean_ps"]
        for size_bytes in oddity_sizes
    ]
    nvlink_256_reversed = (
        by_key[("nvlink-credit", 3, 256)]["p50_seed_mean_ps"]
        >= by_key[("nvlink-credit", 1, 256)]["p50_seed_mean_ps"]
    )
    rnic_oddity = [
        by_key[("rnic-nn", 3, size_bytes)]["p50_seed_mean_ps"]
        < by_key[("rnic-nn", 1, size_bytes)]["p50_seed_mean_ps"]
        for size_bytes in oddity_sizes
    ]
    return [
        {
            "id": "E1",
            "passed_instances": sum(e1_instances),
            "total_instances": len(e1_instances),
            "verdict": "PASS" if all(e1_instances) else "REFUTED",
        },
        {
            "id": "E2",
            "passed_instances": sum(e2_instances),
            "total_instances": len(e2_instances),
            "verdict": "PASS" if all(e2_instances) else "REFUTED",
        },
        {
            "id": "E3",
            "passed_instances": sum(large_cells),
            "total_instances": len(large_cells),
            "required_passes": 7,
            "verdict": "PASS" if sum(large_cells) >= 7 else "REFUTED",
        },
        {
            "id": "E4",
            "passed_instances": sum(nvlink_oddity) + int(nvlink_256_reversed),
            "total_instances": len(nvlink_oddity) + 1,
            "verdict": (
                "PASS" if all(nvlink_oddity) and nvlink_256_reversed else "FATAL_REFUTED"
            ),
            "classification": "entailed preservation check",
        },
        {
            "id": "E5",
            "passed_instances": sum(rnic_oddity),
            "total_instances": len(rnic_oddity),
            "required_passes": 4,
            "verdict": "PASS" if sum(rnic_oddity) >= 4 else "REFUTED",
        },
        {
            "id": "E6",
            "passed_instances": int(rnic_semantics_pass),
            "total_instances": 1,
            "verdict": "PASS" if rnic_semantics_pass else "FATAL_REFUTED",
            "classification": "fatal source-semantics check",
        },
    ]


def _artifact(path: Path, run_dir: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def run(run_dir: Path, adapter: Path) -> dict[str, object]:
    frozen = load_expectations()
    authority = require_clean_authority(frozen)
    provenance = adapter_provenance(adapter)
    physical = frozen["physical_constants"]
    workload = frozen["workload"]
    profile = load_nvlink_candidate_profile(
        ROOT / frozen["source_authority"]["profile_path"]
    )
    rates = {
        row["degree"]: row["rnic_link_rate_bps"]
        for row in frozen["physical_mapping"]["degree_specific_rnic_rates"]
    }

    sample_rows: list[dict[str, object]] = []
    cdf_output_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    all_rnic_manifests: list[dict[str, Any]] = []
    for cell in workload["cells"]:
        degree = cell["degree"]
        size_bytes = cell["size_bytes"]
        by_seed = release_lists(cell, workload)
        transport_samples: dict[str, dict[int, list[int]]] = {
            "nvlink-credit": {},
            "rnic-nn": {},
        }
        diagnostics: dict[str, dict[str, int]] = {
            "nvlink-credit": {},
            "rnic-nn": {},
        }
        for seed_text, release_spec in by_seed.items():
            seed = int(seed_text)
            flow_specs = [
                {
                    "numeric_id": index,
                    "flow_id": f"d{degree}-b{size_bytes}-s{seed}-w{wave}-src{source}",
                    "source": source,
                    "destination": workload["destination"],
                    "payload_bytes": size_bytes,
                    "released_at_ps": release,
                }
                for index, (wave, source, release) in enumerate(release_spec, start=1)
            ]
            expected_flows = degree * workload["samples_per_seed_per_sender"]
            if len(flow_specs) != expected_flows:
                raise AssertionError("a frozen cell has the wrong flow count")

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
            transport_samples["nvlink-credit"][seed] = list(nvlink_fcts.values())
            _add_diagnostics(
                diagnostics["nvlink-credit"],
                _nvlink_diagnostics(nvlink.packets, profile),
            )
            for row in flow_specs:
                sample_rows.append(
                    {
                        "transport": "nvlink-credit",
                        "degree": degree,
                        "size_bytes": size_bytes,
                        "seed": seed,
                        "flow_id": row["flow_id"],
                        "source": row["source"],
                        "released_at_ps": row["released_at_ps"],
                        "fct_ps": nvlink_fcts[str(row["flow_id"])],
                    }
                )

            cell_dir = run_dir / "cells" / f"d{degree}-b{size_bytes}-s{seed}"
            cell_dir.mkdir(parents=True)
            rnic_rows, rnic_manifest = _run_rnic(
                adapter=adapter,
                cell_dir=cell_dir,
                schedule_rows=flow_specs,
                rate_bps=rates[degree],
                physical=physical,
            )
            all_rnic_manifests.append(rnic_manifest)
            if (
                rnic_manifest["schema"]
                != "simllm-traf71-rnic-adapter-manifest-v1"
                or rnic_manifest["capacity_bps"] != rates[degree]
                or rnic_manifest["max_wire_bytes"] != physical["max_wire_bytes"]
                or rnic_manifest["header_bytes"] != physical["header_bytes"]
                or rnic_manifest["propagation_ps"] != physical["propagation_ps"]
                or rnic_manifest["node_count"] != 4
            ):
                raise AssertionError("the rnic-nn physical mapping moved")
            expected_by_id = {str(row["flow_id"]): row for row in flow_specs}
            if {row["flow_id"] for row in rnic_rows} != expected_by_id.keys():
                raise AssertionError(
                    "the rnic-nn completion projection lost or added a flow"
                )
            expected_packets_per_flow = (size_bytes + 255) // 256
            expected_packets = expected_flows * expected_packets_per_flow
            expected_payload = expected_flows * size_bytes
            expected_wire = expected_payload + 16 * expected_packets
            if (
                rnic_manifest["flow_count"] != expected_flows
                or rnic_manifest["payload_bytes"] != expected_payload
                or rnic_manifest["packet_count"] != expected_packets
                or rnic_manifest["wire_bytes"] != expected_wire
                or rnic_manifest["completion_callbacks"] != expected_flows
                or rnic_manifest["data_events"] != 4 * expected_packets
            ):
                raise AssertionError("the rnic-nn physical ledger lost or added work")
            rnic_fcts = {row["flow_id"]: int(row["fct_ps"]) for row in rnic_rows}
            transport_samples["rnic-nn"][seed] = list(rnic_fcts.values())
            _add_diagnostics(
                diagnostics["rnic-nn"],
                {
                    "packet_count": int(rnic_manifest["packet_count"]),
                    "wire_bytes": int(rnic_manifest["wire_bytes"]),
                    "ack_events": int(rnic_manifest["ack_events"]),
                    "reverse_control_bytes": int(rnic_manifest["reverse_control_bytes"]),
                    "non_data_events": int(rnic_manifest["non_data_events"]),
                    "mapping_bias_packets": (
                        int(rnic_manifest["packet_count"])
                        if rates[degree] > physical["pair_raw_bytes_per_second"] * 8
                        else 0
                    ),
                },
            )
            for row in rnic_rows:
                source = expected_by_id[row["flow_id"]]
                if int(row["released_at_ps"]) != int(source["released_at_ps"]):
                    raise AssertionError("the rnic-nn adapter changed a release timestamp")
                if (
                    int(row["payload_bytes"]) != size_bytes
                    or int(row["packet_count"]) != expected_packets_per_flow
                    or int(row["wire_bytes"])
                    != size_bytes + 16 * expected_packets_per_flow
                ):
                    raise AssertionError("a rnic-nn flow changed packet geometry")
                sample_rows.append(
                    {
                        "transport": "rnic-nn",
                        "degree": degree,
                        "size_bytes": size_bytes,
                        "seed": seed,
                        "flow_id": row["flow_id"],
                        "source": row["source"],
                        "released_at_ps": row["released_at_ps"],
                        "fct_ps": row["fct_ps"],
                    }
                )

        for transport, samples in transport_samples.items():
            curves = _cdf_rows(samples)
            if not _cdf_valid(curves):
                raise AssertionError(f"{transport} produced an invalid empirical CDF")
            for row in curves:
                cdf_output_rows.append(
                    {
                        "transport": transport,
                        "degree": degree,
                        "size_bytes": size_bytes,
                        **row,
                    }
                )
            summary_rows.append(
                {
                    "transport": transport,
                    "degree": degree,
                    "size_bytes": size_bytes,
                    **_seed_summary(samples),
                    **diagnostics[transport],
                }
            )

    sample_path = run_dir / "fct-samples.csv"
    cdf_path = run_dir / "fct-cdf.csv"
    summary_path = run_dir / "cell-summary.csv"
    nvlink_source_samples_path = run_dir / "nvlink-source-fct-samples.csv"
    nvlink_source_cdf_path = run_dir / "nvlink-source-fct-cdf.csv"
    sample_fields = [
        "degree",
        "size_bytes",
        "seed",
        "flow_id",
        "source",
        "released_at_ps",
        "fct_ps",
    ]
    _write_csv(
        sample_path,
        [
            "transport",
            *sample_fields,
        ],
        sample_rows,
    )
    _write_csv(
        cdf_path,
        [
            "transport",
            "degree",
            "size_bytes",
            "fct_ps",
            "cdf_mean",
            "cdf_min",
            "cdf_max",
        ],
        cdf_output_rows,
    )
    summary_fields = sorted({key for row in summary_rows for key in row})
    _write_csv(summary_path, summary_fields, summary_rows)

    nvlink_source_samples = [
        {key: row[key] for key in sample_fields}
        for row in sample_rows
        if row["transport"] == "nvlink-credit"
    ]
    _write_csv(nvlink_source_samples_path, sample_fields, nvlink_source_samples)
    cdf_fields = [
        "degree",
        "size_bytes",
        "fct_ps",
        "cdf_mean",
        "cdf_min",
        "cdf_max",
    ]
    nvlink_source_cdf = [
        {key: row[key] for key in cdf_fields}
        for row in cdf_output_rows
        if row["transport"] == "nvlink-credit"
    ]
    _write_csv(nvlink_source_cdf_path, cdf_fields, nvlink_source_cdf)
    source_result = json.loads(
        (ROOT / frozen["source_authority"]["result_path"]).read_text(
            encoding="utf-8"
        )
    )
    source_raw = source_result["raw_evidence_sha256"]
    source_projection = {
        "fct-samples.csv": _sha256(nvlink_source_samples_path),
        "fct-cdf.csv": _sha256(nvlink_source_cdf_path),
    }
    if any(source_projection[name] != source_raw[name] for name in source_projection):
        raise AssertionError("the regenerated NVLink raw FCT evidence moved")

    rnic_semantics_pass = all(
        manifest["htsim_source_commit"] == PINNED_HTSIM_COMMIT
        and manifest["runtime_class"] == "RnicPacketizedManifoldRuntime"
        and manifest["ack_events"] == 0
        and manifest["reverse_control_bytes"] == 0
        and manifest["non_data_events"] == 0
        and manifest["pending_physical_work"] is False
        and manifest["max_wire_bytes"] == physical["max_wire_bytes"]
        and manifest["header_bytes"] == physical["header_bytes"]
        and manifest["propagation_ps"] == physical["propagation_ps"]
        for manifest in all_rnic_manifests
    )
    directions = _expected_direction_rows(summary_rows, rnic_semantics_pass)
    fatal_direction_failures = [
        row["id"] for row in directions if row["verdict"] == "FATAL_REFUTED"
    ]
    if fatal_direction_failures:
        raise AssertionError(f"fatal expected direction failed: {fatal_direction_failures}")
    scored_misses = [
        row["id"]
        for row in directions
        if row.get("classification") is None and row["verdict"] == "REFUTED"
    ]
    fatal_guards = [
        {"guard": guard, "verdict": "PASS"} for guard in frozen["fatal_guards"]
    ]
    result = {
        "schema": RESULT_SCHEMA,
        "task": "TRAF-71",
        "authority": authority,
        "adapter_provenance": provenance,
        "nvlink_source_raw_projection_sha256": source_projection,
        "fatal_guard_verdict": "PASS",
        "fatal_guards": fatal_guards,
        "physical_mapping": frozen["physical_mapping"],
        "metrics": frozen["metrics"],
        "expected_direction_verdicts": directions,
        "scored_misses": scored_misses,
        "cell_summaries": summary_rows,
        "sample_count": len(sample_rows),
        "rnic_adapter_invocations": len(all_rnic_manifests),
        "study_verdict": (
            "PASS" if not scored_misses else "PASS_WITH_HONEST_MISSES"
        ),
    }
    result_path = run_dir / "result.json"
    _write_json(result_path, result)
    artifacts = [
        _artifact(path, run_dir)
        for path in sorted(
            candidate for candidate in run_dir.rglob("*") if candidate.is_file()
        )
    ]
    _write_json(
        run_dir / "manifest.json",
        {
            "schema": "simllm-nvlink-rnic-comparison-run-manifest-v1",
            "artifacts": artifacts,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rnic-adapter", type=Path)
    arguments = parser.parse_args()
    run_dir = prepare_run_dir(arguments.run_dir)
    adapter = configured_adapter(arguments.rnic_adapter)
    result = run(run_dir, adapter)
    print(f"TRAF71_VERDICT={result['study_verdict']}")
    print(f"TRAF71_RUN_DIR={run_dir.as_posix()}")


if __name__ == "__main__":
    main()
