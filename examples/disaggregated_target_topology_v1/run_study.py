"""Run the frozen PLACE-5 structural topology qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path, PurePath
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"
EXPECTATIONS_COMMIT = "241ba81d545ab0b38703efefb153819ac1cd3eb8"
RESULT_SCHEMA = "simllm-disaggregated-target-topology-result-v1"
RUN_ROOT_ENV = "SIMLLM_PLACE5_RUN_ROOT"


def render_cli_path(path: PurePath) -> str:
    """Render an external artifact path with POSIX separators."""

    return path.as_posix()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_clean_worktree() -> None:
    if _git_output("status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("the PLACE-5 run requires a clean tracked worktree")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTATIONS_COMMIT, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit("the PLACE-5 expectations commit is not an ancestor")


def _counter_is_one(values: list[int]) -> bool:
    return set(Counter(values).values()) == {1}


def _build_goal(rank_count: int):
    from simllm.goal import GoalTrace
    from simllm.traffic import ordered_pairwise_messages

    trace = GoalTrace(rank_count)
    ordered_pairwise_messages(
        trace,
        ranks=list(range(rank_count)),
        messages=[
            (
                f"endpoint-{source}",
                source,
                (source + 8) % rank_count,
                4_096,
            )
            for source in range(rank_count)
        ],
        tag=5_005,
        operation_id="place-5:goal-reachability",
    )
    return trace


def _run_cell(output_root: Path, frozen_cell: dict[str, Any]) -> dict[str, Any]:
    from simllm.placement import FabricTopologyManifest, disaggregated_manifests

    label = frozen_cell["label"]
    cell_root = output_root / label
    cell_root.mkdir(parents=True, exist_ok=False)
    enabled = disaggregated_manifests(
        prefill_nodes=frozen_cell["prefill_nodes"],
        decode_nodes=frozen_cell["decode_nodes"],
        render_physical_topology=True,
    )
    disabled = disaggregated_manifests(
        prefill_nodes=frozen_cell["prefill_nodes"],
        decode_nodes=frozen_cell["decode_nodes"],
        render_physical_topology=False,
    )

    enabled_placement_path = enabled.placement.save(
        cell_root / "placement-enabled.json"
    )
    disabled_placement_path = disabled.placement.save(
        cell_root / "placement-disabled.json"
    )
    fabric_path = enabled.fabric.save(cell_root / "fabric.json")
    loaded_fabric = FabricTopologyManifest.load(fabric_path)
    loaded_fabric.validate()
    round_trip_path = loaded_fabric.save(cell_root / "fabric-round-trip.json")

    trace = _build_goal(frozen_cell["rank_count"])
    goal_path = trace.write(cell_root / "reachability.goal")
    paths = enabled.fabric.resolve_goal_paths(trace)
    messages = trace.messages
    ranks = enabled.placement.ranks
    gpus = [gpu for node in enabled.fabric.nodes for gpu in node.gpus]
    nics = [nic for node in enabled.fabric.nodes for nic in node.nics]
    source_ranks = [message.source_rank for message in messages]
    destination_ranks = [message.destination_rank for message in messages]
    path_link_counts = [len(path) for path in paths]
    path_propagation_ps = [
        sum(link.propagation_delay_ps for link in path) for path in paths
    ]
    path_bottleneck_rates = [
        min(link.link_rate_bps for link in path) for path in paths
    ]
    reachable_from_zero = [0]
    for destination in range(1, len(ranks)):
        if enabled.fabric.path_between_ranks(0, destination):
            reachable_from_zero.append(destination)

    leaf_switches = [switch for switch in enabled.fabric.switches if switch.tier == 0]
    spine_switches = [switch for switch in enabled.fabric.switches if switch.tier == 1]
    endpoint_links = [
        link
        for link in enabled.fabric.links
        if link.endpoint_a.startswith("sim-nic-")
        or link.endpoint_b.startswith("sim-nic-")
    ]
    leaf_spine_links = [
        link for link in enabled.fabric.links if link not in endpoint_links
    ]
    enabled_placement_bytes = enabled_placement_path.read_bytes()
    disabled_placement_bytes = disabled_placement_path.read_bytes()
    fabric_bytes = fabric_path.read_bytes()
    round_trip_bytes = round_trip_path.read_bytes()

    return {
        "artifacts": {
            "fabric": render_cli_path(fabric_path.relative_to(output_root)),
            "fabric_round_trip": render_cli_path(
                round_trip_path.relative_to(output_root)
            ),
            "goal": render_cli_path(goal_path.relative_to(output_root)),
            "placement_disabled": render_cli_path(
                disabled_placement_path.relative_to(output_root)
            ),
            "placement_enabled": render_cli_path(
                enabled_placement_path.relative_to(output_root)
            ),
        },
        "counts": {
            "decode_ranks": sum(rank.pool_role == "decode" for rank in ranks),
            "endpoint_links": len(endpoint_links),
            "fabric_nodes": len(enabled.fabric.nodes),
            "gpu_count": len(gpus),
            "goal_messages": len(messages),
            "leaf_spine_links": len(leaf_spine_links),
            "leaf_switches": len(leaf_switches),
            "link_count": len(enabled.fabric.links),
            "nic_count": len(nics),
            "prefill_ranks": sum(rank.pool_role == "prefill" for rank in ranks),
            "rank_count": len(ranks),
            "spine_switches": len(spine_switches),
            "switch_count": len(enabled.fabric.switches),
            "switch_port_count": sum(
                len(switch.ports) for switch in enabled.fabric.switches
            ),
        },
        "fabric": {
            "all_link_delays_ps": sorted(
                {link.propagation_delay_ps for link in enabled.fabric.links}
            ),
            "all_link_rates_bps": sorted(
                {link.link_rate_bps for link in enabled.fabric.links}
            ),
            "evidence_class": enabled.fabric.evidence_class,
            "goal_rank_mapping": enabled.fabric.goal_rank_mapping,
            "physical_rendering_enabled": (
                enabled.fabric.physical_rendering_enabled
            ),
            "schema": enabled.fabric.schema,
            "switch_latency_ps": enabled.fabric.switch_latency_ps,
            "topology_name": enabled.fabric.topology_name,
        },
        "fabric_round_trip": {
            "bytes": len(fabric_bytes),
            "byte_identical": fabric_bytes == round_trip_bytes,
            "sha256": _sha256(fabric_path),
        },
        "goal": {
            "all_destinations_once": _counter_is_one(destination_ranks),
            "all_sources_once": _counter_is_one(source_ranks),
            "destination_ranks": sorted(destination_ranks),
            "path_bottleneck_rates_bps": sorted(set(path_bottleneck_rates)),
            "path_link_counts": sorted(set(path_link_counts)),
            "path_propagation_ps": sorted(set(path_propagation_ps)),
            "payload_bytes": sum(message.payload_bytes for message in messages),
            "resolved_path_count": len(paths),
            "source_ranks": sorted(source_ranks),
            "trace_num_ranks": trace.num_ranks,
        },
        "identity": {
            "gpu_ids_unique": len({gpu.gpu_id for gpu in gpus}) == len(gpus),
            "gpu_ranks": sorted(gpu.global_rank for gpu in gpus),
            "link_ids_unique": len({link.link_id for link in enabled.fabric.links})
            == len(enabled.fabric.links),
            "nic_affinity_ranks": sorted(nic.affine_gpu_rank for nic in nics),
            "nic_ids_unique": len({nic.nic_id for nic in nics}) == len(nics),
            "port_ids_unique": len(
                {
                    port.port_id
                    for switch in enabled.fabric.switches
                    for port in switch.ports
                }
            )
            == sum(len(switch.ports) for switch in enabled.fabric.switches),
            "rank_ids": [rank.global_rank for rank in ranks],
            "switch_ids_unique": len(
                {switch.switch_id for switch in enabled.fabric.switches}
            )
            == len(enabled.fabric.switches),
        },
        "placement_identity": {
            "disabled_bytes": len(disabled_placement_bytes),
            "disabled_sha256": _sha256(disabled_placement_path),
            "enabled_bytes": len(enabled_placement_bytes),
            "enabled_disabled_byte_identical": (
                enabled_placement_bytes == disabled_placement_bytes
            ),
            "enabled_sha256": _sha256(enabled_placement_path),
            "off_fabric_graph_empty": (
                not disabled.fabric.physical_rendering_enabled
                and not disabled.fabric.switches
                and not disabled.fabric.links
            ),
        },
        "reachability": {
            "all_endpoint_pairs_reachable": (
                len(reachable_from_zero) == len(ranks)
            ),
            "basis": (
                "all NICs share one undirected component reached from rank zero"
            ),
            "reachable_from_rank_zero": reachable_from_zero,
        },
    }


def analyze_observation(
    observation: dict[str, Any], frozen: dict[str, Any]
) -> dict[str, Any]:
    """Apply the frozen fatal guards without producing a behavioral score."""

    findings: list[str] = []
    observed_cells = observation.get("cells", {})
    expected_topology = frozen["topology"]
    expected_witness = frozen["goal_witness"]
    expected_records = frozen["baseline"]["placement_records"]

    if observation.get("expectations_commit") != EXPECTATIONS_COMMIT:
        findings.append("expectations commit identity")
    for expected in frozen["cells"]:
        label = expected["label"]
        cell = observed_cells.get(label)
        if cell is None:
            findings.append(f"{label}: missing cell")
            continue
        counts = cell.get("counts", {})
        for field in (
            "decode_ranks",
            "endpoint_links",
            "fabric_nodes",
            "gpu_count",
            "goal_messages",
            "leaf_spine_links",
            "leaf_switches",
            "link_count",
            "nic_count",
            "prefill_ranks",
            "rank_count",
            "spine_switches",
            "switch_count",
            "switch_port_count",
        ):
            if counts.get(field) != expected[field]:
                findings.append(f"{label}: {field}")

        dense_ranks = list(range(expected["rank_count"]))
        identity = cell.get("identity", {})
        for field in ("rank_ids", "gpu_ranks", "nic_affinity_ranks"):
            if identity.get(field) != dense_ranks:
                findings.append(f"{label}: {field}")
        for field in (
            "gpu_ids_unique",
            "link_ids_unique",
            "nic_ids_unique",
            "port_ids_unique",
            "switch_ids_unique",
        ):
            if identity.get(field) is not True:
                findings.append(f"{label}: {field}")

        fabric = cell.get("fabric", {})
        expected_fabric = {
            "all_link_delays_ps": [
                expected_topology["link_propagation_delay_ps"]
            ],
            "all_link_rates_bps": [expected_topology["link_rate_bps"]],
            "evidence_class": expected_topology["evidence_class"],
            "goal_rank_mapping": expected_topology["goal_rank_mapping"],
            "physical_rendering_enabled": True,
            "schema": "simllm-fabric-topology-v1",
            "switch_latency_ps": expected_topology["switch_latency_ps"],
            "topology_name": expected_topology["name"],
        }
        if fabric != expected_fabric:
            findings.append(f"{label}: declared fabric metadata")

        placement = cell.get("placement_identity", {})
        expected_record = expected_records[label]
        for prefix in ("disabled", "enabled"):
            if placement.get(f"{prefix}_bytes") != expected_record["bytes"]:
                findings.append(f"{label}: {prefix} placement bytes")
            if placement.get(f"{prefix}_sha256") != expected_record["sha256"]:
                findings.append(f"{label}: {prefix} placement digest")
        if placement.get("enabled_disabled_byte_identical") is not True:
            findings.append(f"{label}: placement mode identity")
        if placement.get("off_fabric_graph_empty") is not True:
            findings.append(f"{label}: disabled graph identity")
        if cell.get("fabric_round_trip", {}).get("byte_identical") is not True:
            findings.append(f"{label}: fabric round trip")

        goal = cell.get("goal", {})
        if goal.get("trace_num_ranks") != expected["rank_count"]:
            findings.append(f"{label}: GOAL rank count")
        if goal.get("source_ranks") != dense_ranks:
            findings.append(f"{label}: GOAL source set")
        if goal.get("destination_ranks") != dense_ranks:
            findings.append(f"{label}: GOAL destination set")
        if goal.get("all_sources_once") is not True:
            findings.append(f"{label}: GOAL source multiplicity")
        if goal.get("all_destinations_once") is not True:
            findings.append(f"{label}: GOAL destination multiplicity")
        if goal.get("payload_bytes") != (
            expected["goal_messages"] * expected_witness["payload_bytes"]
        ):
            findings.append(f"{label}: GOAL payload bytes")
        if goal.get("resolved_path_count") != expected["goal_messages"]:
            findings.append(f"{label}: GOAL resolved path count")
        if goal.get("path_link_counts") != [
            expected_witness["expected_path_links"]
        ]:
            findings.append(f"{label}: GOAL path link count")
        if goal.get("path_propagation_ps") != [
            expected_witness["expected_path_propagation_ps"]
        ]:
            findings.append(f"{label}: GOAL path propagation")
        if goal.get("path_bottleneck_rates_bps") != [
            expected_topology["link_rate_bps"]
        ]:
            findings.append(f"{label}: GOAL path bottleneck")

        reachability = cell.get("reachability", {})
        if reachability.get("all_endpoint_pairs_reachable") is not True:
            findings.append(f"{label}: endpoint reachability")
        if reachability.get("reachable_from_rank_zero") != dense_ranks:
            findings.append(f"{label}: connected-component membership")

    small = observed_cells.get("one_plus_one", {}).get("counts", {})
    target = observed_cells.get("target", {}).get("counts", {})
    for field in ("rank_count", "endpoint_links", "leaf_switches", "goal_messages"):
        if target.get(field) != 28 * small.get(field, -1):
            findings.append(f"scale relation: {field}")

    return {
        "evidence": {
            "fatal_guard_count": len(frozen["evidence"]["fatal_unscored"]),
            "scored_behavioral_families": 0,
        },
        "findings": findings,
        "status": "PASS" if not findings else "VOID",
    }


def run_study(output_root: Path) -> dict[str, Any]:
    frozen = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    observation = {
        "cells": {
            cell["label"]: _run_cell(output_root, cell)
            for cell in frozen["cells"]
        },
        "expectations_commit": EXPECTATIONS_COMMIT,
        "implementation_commit": _git_output("rev-parse", "HEAD"),
    }
    return {
        "analysis": analyze_observation(observation, frozen),
        "expectations_commit": EXPECTATIONS_COMMIT,
        "observation": observation,
        "schema": RESULT_SCHEMA,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=os.environ.get(RUN_ROOT_ENV),
        help=f"external artifact directory, or set {RUN_ROOT_ENV}",
    )
    args = parser.parse_args()
    if args.output_root is None:
        raise SystemExit(f"provide --output-root or set {RUN_ROOT_ENV}")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit("output root must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    _require_clean_worktree()
    result = run_study(output_root)
    _write_json(output_root / "results.json", result)
    print(json.dumps(result["analysis"], indent=2, sort_keys=True))
    if result["analysis"]["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
