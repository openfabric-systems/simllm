#!/usr/bin/env python3
"""Run the frozen TRAF-69 NVLink flow-dynamics study."""

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
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simllm.backends.htsim_nvlink import (
    NvlinkDomainResult,
    NvlinkDomainService,
    NvlinkFlowPolicy,
    NvlinkOperation,
    NvlinkPacket,
    NvlinkTransfer,
    load_nvlink_candidate_profile,
    sha256_file,
)

HERE = Path(__file__).resolve().parent
EXPECTATIONS_PATH = HERE / "expectations.json"
EXPECTATIONS_COMMIT = "32a49805546bd038af5e49fd68b5d2ed0cea6174"
EXPECTATIONS_SHA256 = "6e6e8f0ed7c79572f1ef893f7f7869d8a4e854200bdee514b4338b87955e1261"
RESULT_SCHEMA = "simllm-nvlink-flow-dynamics-result-v1"
BULK_ROOT_ENV = "SIMLLM_NVFLOWS_BULK_ROOT"
STATIC_IDENTITY_SHA256 = "2f2af64619ed3c6341b209d877d9f1e6984a67e44b97b5eb176a157294a6c252"


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def load_expectations() -> dict[str, Any]:
    return json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))


def _preservation_artifacts(frozen: dict[str, Any]) -> list[dict[str, str]]:
    lock = frozen["preservation_lock"]
    inherited = json.loads((ROOT / lock["inherited"]["path"]).read_text(encoding="utf-8"))[
        "preservation_lock"
    ]
    base = json.loads((ROOT / inherited["inherited"]["path"]).read_text(encoding="utf-8"))[
        "preservation_lock"
    ]["artifacts"]
    return [*base, *inherited["additional_artifacts"], *lock["additional_artifacts"]]


def _static_identity(profile_path: Path) -> str:
    profile = load_nvlink_candidate_profile(profile_path)
    result = NvlinkDomainService(profile).serve(
        [
            NvlinkTransfer(
                extent_id="identity-write-a",
                source=0,
                destination=1,
                payload_bytes=769,
            ),
            NvlinkTransfer(
                extent_id="identity-write-b",
                source=0,
                destination=2,
                payload_bytes=513,
                released_at_ps=17000,
            ),
            NvlinkTransfer(
                extent_id="identity-read-c",
                source=3,
                destination=1,
                payload_bytes=1025,
                operation=NvlinkOperation.PEER_READ,
                released_at_ps=9000,
            ),
        ],
        analytic_result=None,
    )
    if not isinstance(result, NvlinkDomainResult):
        raise SystemExit("the default NVLink identity fixture did not use the scored domain")
    return hashlib.sha256(result.canonical_json_bytes()).hexdigest()


def require_clean_authority(frozen: dict[str, Any]) -> dict[str, object]:
    if sha256_file(EXPECTATIONS_PATH) != EXPECTATIONS_SHA256:
        raise SystemExit("the TRAF-69 expectations digest moved after its final freeze")
    ancestor = _git("merge-base", "--is-ancestor", EXPECTATIONS_COMMIT, "HEAD")
    if ancestor.returncode:
        raise SystemExit("the final TRAF-69 expectations commit is not an ancestor of HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status.returncode or status.stdout.strip():
        raise SystemExit("the TRAF-69 gated run requires a clean tracked worktree")
    committed = _git("show", f"{EXPECTATIONS_COMMIT}:examples/nvlink_flow_dynamics_v1/expectations.json")
    if committed.returncode:
        raise SystemExit("the final expectations bytes cannot be read from their commit")
    committed_sha256 = hashlib.sha256(committed.stdout.encode("utf-8")).hexdigest()
    if committed_sha256 != EXPECTATIONS_SHA256:
        raise SystemExit("the final expectations commit does not contain the frozen bytes")

    source = frozen["source_profile"]
    profile_path = ROOT / source["path"]
    if sha256_file(profile_path) != source["sha256"]:
        raise SystemExit("the scored TRAF-70 profile moved")
    profile = load_nvlink_candidate_profile(profile_path)
    if profile.status != source["status"]:
        raise SystemExit("the scored TRAF-70 profile status moved")
    if profile.score_publication is None:
        raise SystemExit("the scored TRAF-70 publication disappeared")
    if profile.score_publication.score_sha256 != source["score_sha256"]:
        raise SystemExit("the scored TRAF-70 score digest moved")
    if profile.score_publication.score_status != source["score_status"]:
        raise SystemExit("the scored TRAF-70 score status moved")
    if frozen["parameter_ledger"]["candidate_count"] != 10:
        raise SystemExit("the declared-candidate evidence count moved")
    if profile.score_publication.unchanged_parameter_count != 11:
        raise SystemExit("the scored unchanged-parameter count moved")

    artifacts = _preservation_artifacts(frozen)
    if len(artifacts) != frozen["preservation_lock"]["expected_total_artifacts"]:
        raise SystemExit("the TRAF-69 preservation class changed size")
    for artifact in artifacts:
        if sha256_file(ROOT / artifact["path"]) != artifact["sha256"]:
            raise SystemExit(f"preservation lock mismatch: {artifact['path']}")
    identity = _static_identity(profile_path)
    if identity != STATIC_IDENTITY_SHA256:
        raise SystemExit("the default NVLink flow-inactive canonical bytes moved")
    return {
        "expectations_commit": EXPECTATIONS_COMMIT,
        "expectations_sha256": EXPECTATIONS_SHA256,
        "profile_sha256": source["sha256"],
        "score_sha256": source["score_sha256"],
        "preservation_artifacts_checked": len(artifacts),
        "static_identity_sha256": identity,
    }


def prepare_run_dir(path: Path) -> Path:
    configured = os.environ.get(BULK_ROOT_ENV)
    if not configured:
        raise SystemExit(f"set {BULK_ROOT_ENV} to the external nvflows bulk root")
    bulk_root = Path(configured).resolve()
    resolved = path.resolve()
    if resolved.parent != bulk_root:
        raise SystemExit(f"--run-dir must be one new child of {bulk_root.as_posix()}")
    bulk_root.mkdir(parents=True, exist_ok=True)
    resolved.mkdir(parents=False, exist_ok=False)
    return resolved


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


def _serve(profile_path: Path, transfers: list[NvlinkTransfer]) -> NvlinkDomainResult:
    profile = load_nvlink_candidate_profile(profile_path)
    result = NvlinkDomainService(profile).serve(
        transfers,
        analytic_result=None,
        flow_policy=NvlinkFlowPolicy.RELEASE_AWARE_ROUND_ROBIN,
    )
    if not isinstance(result, NvlinkDomainResult):
        raise TypeError("the selected NVLink flow policy did not produce a domain result")
    return result


def _flow_completion_ps(
    result: NvlinkDomainResult, transfers: list[NvlinkTransfer]
) -> dict[str, int]:
    releases = {transfer.extent_id: transfer.released_at_ps for transfer in transfers}
    completions: dict[str, int] = {}
    for packet in result.packets:
        if packet.delivered_at_ps is None:
            raise AssertionError("a result packet has no delivery timestamp")
        completions[packet.extent_id] = max(
            completions.get(packet.extent_id, 0), packet.delivered_at_ps
        )
    if completions.keys() != releases.keys():
        raise AssertionError("the flow completion projection lost or added an extent")
    return {
        extent_id: completions[extent_id] - release
        for extent_id, release in releases.items()
    }


def _rate_rows(
    packets: tuple[NvlinkPacket, ...],
    flow_ids: list[str],
    *,
    bin_ps: int,
    origin_ps: int = 0,
    first_bin: int | None = None,
    last_bin: int | None = None,
) -> list[dict[str, object]]:
    payload_by_bin: dict[tuple[int, str], int] = defaultdict(int)
    observed_bins = []
    for packet in packets:
        if packet.delivered_at_ps is None:
            raise AssertionError("rate input packet has no delivery timestamp")
        index = (packet.delivered_at_ps - origin_ps) // bin_ps
        payload_by_bin[(index, packet.extent_id)] += packet.payload_bytes
        observed_bins.append(index)
    if not observed_bins:
        return []
    lower = min(observed_bins) if first_bin is None else first_bin
    upper = max(observed_bins) if last_bin is None else last_bin
    rows = []
    for index in range(lower, upper + 1):
        for flow_id in flow_ids:
            payload_bytes = payload_by_bin[(index, flow_id)]
            rows.append(
                {
                    "bin_index": index,
                    "bin_start_ps": origin_ps + index * bin_ps,
                    "bin_end_ps": origin_ps + (index + 1) * bin_ps,
                    "flow_id": flow_id,
                    "payload_bytes": payload_bytes,
                    "payload_gbps": payload_bytes * 1000 / bin_ps,
                }
            )
    return rows


def _steady_rows(
    rate_rows: list[dict[str, object]],
    transfers: list[NvlinkTransfer],
    completion_ps: dict[str, int],
    *,
    pair_payload_gbps: float,
    half_width_gbps: float,
    settle_ps: int,
) -> list[dict[str, object]]:
    releases = {transfer.extent_id: transfer.released_at_ps for transfer in transfers}
    ends = {
        transfer.extent_id: transfer.released_at_ps + completion_ps[transfer.extent_id]
        for transfer in transfers
    }
    transition_events = [*releases.values(), *ends.values()]
    scored = []
    for row in rate_rows:
        flow_id = str(row["flow_id"])
        start = int(row["bin_start_ps"])
        end = int(row["bin_end_ps"])
        if any(event - settle_ps < end and start < event + settle_ps for event in transition_events):
            continue
        active = [
            candidate
            for candidate in releases
            if releases[candidate] + settle_ps <= start and end + settle_ps <= ends[candidate]
        ]
        if flow_id not in active:
            continue
        expected = pair_payload_gbps / len(active)
        observed = float(row["payload_gbps"])
        scored.append(
            {
                **row,
                "active_flows": len(active),
                "expected_gbps": expected,
                "band_low_gbps": expected - half_width_gbps,
                "band_high_gbps": expected + half_width_gbps,
                "verdict": "PASS"
                if expected - half_width_gbps - 1e-12
                <= observed
                <= expected + half_width_gbps + 1e-12
                else "REFUTED",
            }
        )
    return scored


def _quantile(values: list[int], probability: float) -> int:
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


def _cdf_is_valid(rows: list[dict[str, object]]) -> bool:
    tolerance = 1e-12
    if not rows:
        return False
    for row in rows:
        if not (
            -tolerance
            <= row["cdf_min"]
            <= row["cdf_mean"] + tolerance
            <= row["cdf_max"] + 2 * tolerance
            <= 1 + 2 * tolerance
        ):
            return False
    for field in ("cdf_min", "cdf_mean", "cdf_max"):
        if any(
            left[field] > right[field] + tolerance
            for left, right in pairwise(rows)
        ):
            return False
    return True


def _run_overall(frozen: dict[str, Any], profile_path: Path) -> dict[str, object]:
    schedule = frozen["flow_schedule"]
    transfers = [
        NvlinkTransfer(
            extent_id=flow_id,
            source=0,
            destination=1,
            payload_bytes=target,
            released_at_ps=release,
        )
        for flow_id, release, target in zip(
            schedule["flow_ids"],
            schedule["release_ps"],
            schedule["target_bytes"],
            strict=True,
        )
    ]
    result = _serve(profile_path, transfers)
    completions = _flow_completion_ps(result, transfers)
    rate_rows = _rate_rows(
        result.packets,
        schedule["flow_ids"],
        bin_ps=schedule["raw_bin_ps"],
    )
    steady = _steady_rows(
        rate_rows,
        transfers,
        completions,
        pair_payload_gbps=frozen["packet_arithmetic"]["pair_payload_rate_gbps"],
        half_width_gbps=schedule["steady_bands"][0]["half_width_gbps"],
        settle_ps=schedule["raw_bin_ps"],
    )
    completion_order = sorted(
        completions,
        key=lambda flow_id: transfers[schedule["flow_ids"].index(flow_id)].released_at_ps
        + completions[flow_id],
    )
    return {
        "result": result,
        "transfers": transfers,
        "completion_ps": completions,
        "completion_order": completion_order,
        "reverse_target_verdict": "PASS" if completion_order[-1] == "flow-a" else "REFUTED",
        "rate_rows": rate_rows,
        "steady_rows": steady,
        "steady_verdict": (
            "PASS" if steady and all(row["verdict"] == "PASS" for row in steady) else "REFUTED"
        ),
    }


def _run_transitions(frozen: dict[str, Any], profile_path: Path) -> dict[str, object]:
    convergence = frozen["convergence_1_to_2"]
    join_ps = convergence["join_ps"]
    convergence_transfers = [
        NvlinkTransfer(
            extent_id="incumbent",
            source=0,
            destination=1,
            payload_bytes=convergence["incumbent_target_bytes"],
        ),
        NvlinkTransfer(
            extent_id="joiner",
            source=0,
            destination=1,
            payload_bytes=convergence["joiner_target_bytes"],
            released_at_ps=join_ps,
        ),
    ]
    convergence_result = _serve(profile_path, convergence_transfers)
    first_joiner = next(
        packet for packet in convergence_result.packets if packet.extent_id == "joiner"
    )
    if first_joiner.delivered_at_ps is None:
        raise AssertionError("the joiner has no first delivery")
    observed_open_ps = first_joiner.delivered_at_ps - join_ps
    convergence_rows = _rate_rows(
        convergence_result.packets,
        ["incumbent", "joiner"],
        bin_ps=convergence["raw_bin_ps"],
        origin_ps=join_ps,
        first_bin=-8,
        last_bin=80,
    )

    divergence = frozen["divergence_2_to_1"]
    divergence_transfers = [
        NvlinkTransfer(
            extent_id="remaining",
            source=0,
            destination=1,
            payload_bytes=divergence["remaining_target_bytes"],
        ),
        NvlinkTransfer(
            extent_id="departing",
            source=0,
            destination=1,
            payload_bytes=divergence["departing_target_bytes"],
        ),
    ]
    divergence_result = _serve(profile_path, divergence_transfers)
    departing_end = max(
        packet.delivered_at_ps or 0
        for packet in divergence_result.packets
        if packet.extent_id == "departing"
    )
    remaining_after = [
        packet.delivered_at_ps or 0
        for packet in divergence_result.packets
        if packet.extent_id == "remaining" and (packet.delivered_at_ps or 0) > departing_end
    ]
    observed_target_ps = remaining_after[4] - departing_end
    divergence_rows = _rate_rows(
        divergence_result.packets,
        ["remaining", "departing"],
        bin_ps=divergence["raw_bin_ps"],
        origin_ps=departing_end,
        first_bin=-80,
        last_bin=80,
    )
    return {
        "convergence": {
            "expected_open_ps": convergence["expected_open_ps"],
            "observed_open_ps": observed_open_ps,
            "residual_ps": observed_open_ps - convergence["expected_open_ps"],
            "verdict": "PASS" if observed_open_ps == convergence["expected_open_ps"] else "REFUTED",
            "rate_rows": convergence_rows,
        },
        "divergence": {
            "expected_time_to_target_ps": divergence["expected_time_to_target_ps"],
            "observed_time_to_target_ps": observed_target_ps,
            "residual_ps": observed_target_ps - divergence["expected_time_to_target_ps"],
            "departing_completion_ps": departing_end,
            "verdict": (
                "PASS"
                if observed_target_ps == divergence["expected_time_to_target_ps"]
                else "REFUTED"
            ),
            "rate_rows": divergence_rows,
        },
    }


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


def _run_fct_and_incast(
    frozen: dict[str, Any], profile_path: Path
) -> dict[str, object]:
    cdf = frozen["fct_cdf"]
    band_by_key = {
        (row["degree"], row["size_bytes"]): row for row in cdf["bands"]
    }
    sample_rows: list[dict[str, object]] = []
    cdf_output_rows: list[dict[str, object]] = []
    verdict_rows = []
    for degree in (1, 2, 3):
        for size_bytes in cdf["flow_sizes_bytes"]:
            band = band_by_key[(degree, size_bytes)]
            samples_by_seed: dict[int, list[int]] = {}
            for seed in cdf["seeds"]:
                release_spec = _sample_releases(
                    seed=seed,
                    degree=degree,
                    size_bytes=size_bytes,
                    release_interval_ps=band["release_interval_ps"],
                    jitter_low_ps=band["release_jitter_ps"][0],
                    jitter_high_ps=band["release_jitter_ps"][1],
                    samples_per_sender=cdf["samples_per_seed_per_sender"],
                )
                transfers = [
                    NvlinkTransfer(
                        extent_id=f"d{degree}-b{size_bytes}-s{seed}-w{wave}-src{source}",
                        source=source,
                        destination=3,
                        payload_bytes=size_bytes,
                        released_at_ps=release,
                    )
                    for wave, source, release in release_spec
                ]
                result = _serve(profile_path, transfers)
                completions = _flow_completion_ps(result, transfers)
                values = list(completions.values())
                samples_by_seed[seed] = values
                for transfer in transfers:
                    sample_rows.append(
                        {
                            "degree": degree,
                            "size_bytes": size_bytes,
                            "seed": seed,
                            "flow_id": transfer.extent_id,
                            "source": transfer.source,
                            "released_at_ps": transfer.released_at_ps,
                            "fct_ps": completions[transfer.extent_id],
                        }
                    )
            curves = _cdf_rows(samples_by_seed)
            for row in curves:
                cdf_output_rows.append(
                    {"degree": degree, "size_bytes": size_bytes, **row}
                )
            p50_values = [_quantile(values, 0.50) for values in samples_by_seed.values()]
            p95_values = [_quantile(values, 0.95) for values in samples_by_seed.values()]
            p50_mean = sum(p50_values) / len(p50_values)
            p95_mean = sum(p95_values) / len(p95_values)
            p50_pass = band["p50_band_ps"][0] <= p50_mean <= band["p50_band_ps"][1]
            p95_pass = band["p95_band_ps"][0] <= p95_mean <= band["p95_band_ps"][1]
            cdf_pass = _cdf_is_valid(curves)
            terminal_pass = curves[-1]["cdf_min"] == curves[-1]["cdf_max"] == 1
            verdict_rows.append(
                {
                    "degree": degree,
                    "size_bytes": size_bytes,
                    "sample_count": sum(len(values) for values in samples_by_seed.values()),
                    "p50_mean_ps": p50_mean,
                    "p50_seed_min_ps": min(p50_values),
                    "p50_seed_max_ps": max(p50_values),
                    "p50_band_ps": band["p50_band_ps"],
                    "p95_mean_ps": p95_mean,
                    "p95_seed_min_ps": min(p95_values),
                    "p95_seed_max_ps": max(p95_values),
                    "p95_band_ps": band["p95_band_ps"],
                    "monotone_cdf": cdf_pass,
                    "terminal_one": terminal_pass,
                    "verdict": (
                        "PASS" if p50_pass and p95_pass and cdf_pass and terminal_pass else "REFUTED"
                    ),
                }
            )

    incast_rows = []
    incast_rate_rows: dict[int, list[dict[str, object]]] = {}
    target = frozen["incast"]["schedule_target_bytes_per_flow"]
    for frozen_degree in frozen["incast"]["degrees"]:
        degree = frozen_degree["degree"]
        transfers = [
            NvlinkTransfer(
                extent_id=f"incast-degree-{degree}-source-{source}",
                source=source,
                destination=3,
                payload_bytes=target,
            )
            for source in range(degree)
        ]
        result = _serve(profile_path, transfers)
        observed_gbps = result.logical_bytes * 1000 / result.completion_time_ps
        ceiling_gbps = frozen_degree["payload_ceiling_gbps"]
        incast_rows.append(
            {
                "degree": degree,
                "simulated_payload_gbps": observed_gbps,
                "payload_ceiling_gbps": ceiling_gbps,
                "ceiling_fraction": observed_gbps / ceiling_gbps,
                "expected_binding_module": frozen_degree["expected_binding_module"],
                "completion_time_ps": result.completion_time_ps,
                "max_rx_buffer_occupancy_bytes": result.max_rx_buffer_occupancy_bytes,
                "verdict": "PASS" if observed_gbps <= ceiling_gbps else "REFUTED",
            }
        )
        incast_rate_rows[degree] = _rate_rows(
            result.packets,
            [transfer.extent_id for transfer in transfers],
            bin_ps=frozen["incast"]["raw_bin_ps"],
        )

    fanout = _serve(
        profile_path,
        [
            NvlinkTransfer(
                extent_id=f"fanout-{destination}",
                source=0,
                destination=destination,
                payload_bytes=target,
            )
            for destination in (1, 2, 3)
        ],
    )
    fanout_gbps = fanout.logical_bytes * 1000 / fanout.completion_time_ps
    published = frozen["incast"]["fanout_check"]["published_payload_gbps"]
    relative_error = abs(fanout_gbps - published) / published
    fanout_verdict = (
        "PASS"
        if relative_error <= frozen["incast"]["fanout_check"]["relative_error_limit"]
        else "REFUTED"
    )
    return {
        "sample_rows": sample_rows,
        "cdf_rows": cdf_output_rows,
        "verdict_rows": verdict_rows,
        "incast_rows": incast_rows,
        "incast_rate_rows": incast_rate_rows,
        "fanout": {
            "simulated_payload_gbps": fanout_gbps,
            "published_payload_gbps": published,
            "relative_error": relative_error,
            "relative_error_limit": frozen["incast"]["fanout_check"]["relative_error_limit"],
            "expected_verdict": frozen["incast"]["fanout_check"]["expected_verdict"],
            "verdict": fanout_verdict,
        },
    }


def _artifact(path: Path, run_dir: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run(run_dir: Path) -> dict[str, object]:
    frozen = load_expectations()
    authority = require_clean_authority(frozen)
    profile_path = ROOT / frozen["source_profile"]["path"]
    overall = _run_overall(frozen, profile_path)
    transitions = _run_transitions(frozen, profile_path)
    distributions = _run_fct_and_incast(frozen, profile_path)

    overall_path = run_dir / "overall-rate.csv"
    convergence_path = run_dir / "convergence-rate.csv"
    divergence_path = run_dir / "divergence-rate.csv"
    sample_path = run_dir / "fct-samples.csv"
    cdf_path = run_dir / "fct-cdf.csv"
    _write_csv(
        overall_path,
        ["bin_index", "bin_start_ps", "bin_end_ps", "flow_id", "payload_bytes", "payload_gbps"],
        overall["rate_rows"],
    )
    _write_csv(
        convergence_path,
        ["bin_index", "bin_start_ps", "bin_end_ps", "flow_id", "payload_bytes", "payload_gbps"],
        transitions["convergence"]["rate_rows"],
    )
    _write_csv(
        divergence_path,
        ["bin_index", "bin_start_ps", "bin_end_ps", "flow_id", "payload_bytes", "payload_gbps"],
        transitions["divergence"]["rate_rows"],
    )
    _write_csv(
        sample_path,
        ["degree", "size_bytes", "seed", "flow_id", "source", "released_at_ps", "fct_ps"],
        distributions["sample_rows"],
    )
    _write_csv(
        cdf_path,
        ["degree", "size_bytes", "fct_ps", "cdf_mean", "cdf_min", "cdf_max"],
        distributions["cdf_rows"],
    )
    incast_paths = []
    for degree, rows in distributions["incast_rate_rows"].items():
        path = run_dir / f"incast-degree-{degree}-rate.csv"
        _write_csv(
            path,
            ["bin_index", "bin_start_ps", "bin_end_ps", "flow_id", "payload_bytes", "payload_gbps"],
            rows,
        )
        incast_paths.append(path)

    fatal_guards = [
        {"guard": guard, "verdict": "PASS"} for guard in frozen["fatal_guards"]
    ]
    nonfatal_verdicts = [
        overall["reverse_target_verdict"],
        overall["steady_verdict"],
        transitions["convergence"]["verdict"],
        transitions["divergence"]["verdict"],
        *(row["verdict"] for row in distributions["verdict_rows"]),
        *(row["verdict"] for row in distributions["incast_rows"]),
    ]
    result = {
        "schema": RESULT_SCHEMA,
        "task": "TRAF-69",
        "authority": authority,
        "fatal_guard_verdict": "PASS",
        "fatal_guards": fatal_guards,
        "parameter_ledger": frozen["parameter_ledger"],
        "packet_arithmetic": frozen["packet_arithmetic"],
        "overall_schedule": {
            "completion_ps": overall["completion_ps"],
            "completion_order": overall["completion_order"],
            "reverse_target_verdict": overall["reverse_target_verdict"],
            "steady_rate_checks": len(overall["steady_rows"]),
            "steady_rate_failures": sum(
                row["verdict"] != "PASS" for row in overall["steady_rows"]
            ),
            "steady_verdict": overall["steady_verdict"],
        },
        "convergence_1_to_2": {
            key: value
            for key, value in transitions["convergence"].items()
            if key != "rate_rows"
        },
        "divergence_2_to_1": {
            key: value
            for key, value in transitions["divergence"].items()
            if key != "rate_rows"
        },
        "fct_cdf": {
            "seed_count": frozen["fct_cdf"]["seed_count"],
            "band": frozen["fct_cdf"]["band"],
            "verdicts": distributions["verdict_rows"],
        },
        "incast": distributions["incast_rows"],
        "fanout_separate_check": distributions["fanout"],
        "study_verdict": (
            "PASS_WITH_EXPECTED_FANOUT_REFUTATION"
            if all(verdict == "PASS" for verdict in nonfatal_verdicts)
            and distributions["fanout"]["verdict"] == distributions["fanout"]["expected_verdict"]
            else "REFUTED"
        ),
        "plot_contract": frozen["plot_contract"],
    }
    result_path = run_dir / "result.json"
    _write_json(result_path, result)
    artifacts = [
        overall_path,
        convergence_path,
        divergence_path,
        sample_path,
        cdf_path,
        *incast_paths,
        result_path,
    ]
    manifest = {
        "schema": "simllm-nvlink-flow-dynamics-run-manifest-v1",
        "artifacts": [_artifact(path, run_dir) for path in artifacts],
    }
    _write_json(run_dir / "manifest.json", manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run_dir = prepare_run_dir(arguments.run_dir)
    result = run(run_dir)
    print(f"TRAF69_VERDICT={result['study_verdict']}")
    print(f"TRAF69_RUN_DIR={run_dir.as_posix()}")


if __name__ == "__main__":
    main()
